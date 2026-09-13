"""Experimental, opt-in SteinerPy context selection (HKUDS/LightRAG#3866).

Records remain independently selectable. A selected entity and an incident
relation earn a forest connectivity bonus; absent endpoints earn no bonus.
The artificial root is an optimization device and never becomes context.
"""

from __future__ import annotations

import asyncio
import json
import math
from dataclasses import dataclass, replace
from functools import lru_cache
from time import perf_counter
from typing import Any

import networkx as nx

from lightrag.utils import logger, normalize_rerank_result, truncate_list_by_token_size


@dataclass(frozen=True)
class SelectionOptions:
    strategy: str = "steiner_soft"
    prize_source: str = "ordering"
    connectivity_bonus: float = 0.15
    max_candidates: int = 120
    multipliers: tuple[float, ...] = (0.5, 1.0, 2.0)

    def __post_init__(self) -> None:
        if self.strategy not in {
            "rank",
            "relevance",
            "soft_greedy",
            "steiner_soft",
            "steiner_hard",
        }:
            raise ValueError(f"Unknown context selection strategy: {self.strategy}")
        if self.prize_source not in {"ordering", "rerank_score"}:
            raise ValueError(f"Unknown prize source: {self.prize_source}")
        if not math.isfinite(self.connectivity_bonus) or self.connectivity_bonus < 0:
            raise ValueError("connectivity_bonus must be finite and non-negative")
        if not isinstance(self.max_candidates, int) or self.max_candidates < 1:
            raise ValueError("max_candidates must be a positive integer")
        if not self.multipliers or any(
            not math.isfinite(x) or x <= 0 for x in self.multipliers
        ):
            raise ValueError("multipliers must contain finite positive values")


def check_budget(used: tuple[int, int], budgets: tuple[int, int]) -> None:
    """Enforce the query-path invariant even under python -O."""
    if any(count > budget for count, budget in zip(used, budgets)):
        raise ValueError(
            f"Context selection exceeded token budgets: {used} > {budgets}"
        )


async def rerank_records(entities, relations, query, rerank_func):
    """Keep valid returned scores; exclude the unscored tail of capped providers.

    Scores travel separately from context records: they cost no prompt tokens
    and cannot overwrite source attribution. Require non-negative scores;
    callers using logits must explicitly configure their provider transform.
    """
    if not query or not callable(rerank_func):
        raise ValueError(
            "rerank_score requires a query and configured rerank_model_func"
        )
    documents = [f"{r['entity']}\n{r.get('description', '')}" for r in entities] + [
        f"{r['entity1']} -- {r['entity2']}\n{r.get('description', '')}"
        for r in relations
    ]
    if not documents:
        return entities, relations, [], 0.0
    started = perf_counter()
    results = await rerank_func(query=query, documents=documents, top_n=len(documents))
    elapsed = (perf_counter() - started) * 1000
    scores = {}
    if not isinstance(results, list):
        raise TypeError("KG reranker must return index/relevance_score results")
    for result in results:
        row, error = normalize_rerank_result(result, len(documents))
        if error or row["index"] in scores or row["relevance_score"] < 0:
            raise ValueError(f"Invalid or duplicate KG rerank score: {error or result}")
        scores[row["index"]] = row["relevance_score"]
    ne = len(entities)
    order_e = sorted((i for i in scores if i < ne), key=lambda i: (-scores[i], i))
    order_r = sorted((i for i in scores if i >= ne), key=lambda i: (-scores[i], i))
    return (
        [entities[i] for i in order_e],
        [relations[i - ne] for i in order_r],
        [scores[i] for i in order_e + order_r],
        elapsed,
    )


def record_graph(entities: list[dict], relations: list[dict]) -> nx.Graph:
    """Incidence graph over context records, preserving every relation row.

    Topology is undirected, matching LightRAG's retrieved KG. Ordered endpoint
    fields and descriptions remain untouched in the actual output records.
    A missing endpoint is never manufactured as a free bridge.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(len(entities) + len(relations)))
    entity_ids = {row["entity"]: i for i, row in enumerate(entities)}
    for j, row in enumerate(relations, len(entities)):
        for endpoint in (row["entity1"], row["entity2"]):
            if endpoint in entity_ids:
                graph.add_edge(entity_ids[endpoint], j)
    return graph


def utility(
    graph: nx.Graph, selected: set[int], prizes: list[float], beta: float
) -> float:
    """Relevance plus beta times the size of a maximum spanning forest."""
    relevance = sum(prizes[i] for i in selected)
    if beta == 0:
        return relevance
    components = nx.number_connected_components(graph.subgraph(selected))
    return relevance + beta * (len(selected) - components)


def pcst_proposal(
    graph: nx.Graph,
    prizes: list[float],
    costs: list[float],
    beta: float,
    multiplier: float,
    *,
    exact: bool = False,
    diagnostics: list[dict] | None = None,
) -> set[int]:
    """SteinerPy heuristic for utility(S) - multiplier * cost(S).

    Each record i becomes i_in -> i_out, with its token penalty on that arc.
    Only i_out has prize (relevance_i + beta). Root -> i_in costs beta;
    real incidence links cost zero. Thus a forest with k components pays
    beta*k and earns beta*|S|. Isolates earn exactly their relevance, with
    no connectivity penalty. Both endpoints' descriptions are paid once.

    This is a Lagrangian proposal, NOT a budget-feasible solution or an
    optimality certificate for the final selection.
    """
    try:
        from steinerpy import DirectedPrizeCollectingProblem
    except ImportError as exc:
        raise ImportError(
            "Install the optional dependency: pip install '.[steiner]'"
        ) from exc

    root = 2 * len(prizes)
    directed = nx.DiGraph()
    directed.add_node(root)
    node_prizes = {}
    for i in sorted(graph):
        directed.add_edge(2 * i, 2 * i + 1, weight=multiplier * costs[i])
        directed.add_edge(root, 2 * i, weight=beta)
        node_prizes[2 * i + 1] = prizes[i] + beta
    for u, v in graph.edges:
        directed.add_edge(2 * u + 1, 2 * v, weight=0.0)
        directed.add_edge(2 * v + 1, 2 * u, weight=0.0)
    solution = DirectedPrizeCollectingProblem(
        directed, node_prizes=node_prizes, root=root
    ).get_solution(exact=exact)
    if diagnostics is not None:
        gap = solution.gap
        # SteinerPy reports (primal - dual_bound) / max(1, abs(primal))
        # for its penalized minimization problem. This certifies only this
        # proposal, not token-budget repair or answer correctness.
        lower = solution.objective - gap * max(1.0, abs(solution.objective))
        diagnostics.append(
            {
                "multiplier": multiplier,
                "pcst_objective": solution.objective,
                "pcst_relative_gap": gap if math.isfinite(gap) else None,
                "lagrangian_upper_bound": sum(node_prizes.values()) - lower
                if math.isfinite(lower)
                else None,
            }
        )
    nodes = set(solution.selected_nodes)
    return {i for i in graph if 2 * i + 1 in nodes}


def _select_sync(
    entities: list[dict],
    relations: list[dict],
    tokenizer: Any,
    entity_budget: int,
    relation_budget: int,
    options: SelectionOptions,
    rerank_scores: list[float] | None = None,
) -> tuple[list[dict], list[dict], dict]:
    started = perf_counter()
    ne = len(entities)
    records = entities + relations
    budgets = (max(0, entity_budget), max(0, relation_budget))
    graph = record_graph(entities, relations)
    # The ordering proxy mixes vector retrieval and degree-based local edges.
    # It is an ordering control, not a calibrated query-relevance estimate.
    if options.prize_source == "rerank_score":
        if rerank_scores is None or len(rerank_scores) != len(records):
            raise ValueError("A rerank_score prize is required for every record")
        if any(not math.isfinite(x) or x < 0 for x in rerank_scores):
            raise ValueError("Rerank prizes must be finite and non-negative")
        prizes = list(rerank_scores)
    else:
        prizes = [
            1 / math.sqrt(i + 1)
            for rows in (entities, relations)
            for i in range(len(rows))
        ]
    rendered = [json.dumps(row, ensure_ascii=False) for row in records]
    costs = [len(tokenizer.encode(row + "\n")) for row in rendered]
    normalized = [cost / max(1, budgets[int(i >= ne)]) for i, cost in enumerate(costs)]
    # beta is a fraction of the largest prize, not an absolute provider score.
    # Raw rerank_score values (not ranks) remain the B-arm prizes.
    beta = (
        0.0
        if options.strategy in {"rank", "relevance"}
        else options.connectivity_bonus * max(prizes, default=0.0)
    )

    adjacency = {i: set(graph[i]) for i in graph}

    @lru_cache(maxsize=2048)
    def count_cached(selected: frozenset[int]) -> tuple[int, int]:
        # Count the exact downstream serialization, including separators.
        return tuple(
            len(
                tokenizer.encode(
                    "\n".join(
                        rendered[i] for i in sorted(selected) if int(i >= ne) == kind
                    )
                )
            )
            for kind in (0, 1)
        )

    def counts(selected: set[int]) -> tuple[int, int]:
        return count_cached(frozenset(selected))

    def fits(selected: set[int]) -> bool:
        return all(count <= budget for count, budget in zip(counts(selected), budgets))

    @lru_cache(maxsize=2048)
    def value_cached(selected: frozenset[int]) -> float:
        relevance = sum(prizes[i] for i in selected)
        if beta == 0:
            return relevance
        # Equivalent to utility(), without constructing NetworkX subgraph
        # views thousands of times during repair. Cache lives for one query.
        remaining = set(selected)
        components = 0
        while remaining:
            components += 1
            queue = [remaining.pop()]
            while queue:
                neighbors = adjacency[queue.pop()] & remaining
                remaining.difference_update(neighbors)
                queue.extend(neighbors)
        return relevance + beta * (len(selected) - components)

    def value(selected: set[int]) -> float:
        return value_cached(frozenset(selected))

    # Always retain the genuine LightRAG baseline as a feasible candidate.
    baseline: set[int] = set()
    for rows, budget, offset in (
        (entities, budgets[0], 0),
        (relations, budgets[1], ne),
    ):
        prefix = truncate_list_by_token_size(
            rows,
            key=lambda row: json.dumps(row, ensure_ascii=False),
            separator="\n",
            max_token_size=budget,
            tokenizer=tokenizer,
        )
        baseline.update(range(offset, offset + len(prefix)))

    fallback = options.strategy != "rank" and len(records) > options.max_candidates
    if options.strategy == "rank" or fallback:
        if fallback:
            logger.warning(
                "Context selection used same-prize rank fallback: candidate_limit"
            )
        used = counts(baseline)
        check_budget(used, budgets)
        return (
            [row for i, row in enumerate(entities) if i in baseline],
            [row for i, row in enumerate(relations, ne) if i in baseline],
            {
                "strategy": options.strategy,
                "executed_strategy": "rank",
                "fallback_reason": "candidate_limit" if fallback else None,
                "score_source": options.prize_source,
                "candidate_count": len(records),
                "selected_count": len(baseline),
                "entity_tokens": used[0],
                "relation_tokens": used[1],
                "solver_calls": 0,
                "solver_ms": 0.0,
                "worker_ms": (perf_counter() - started) * 1000,
            },
        )

    # Oversized records cannot be returned alone and must not provide free
    # connectivity to the proposal. Original positions still define prizes.
    eligible = {i for i in graph if budgets[int(i >= ne)] > 0 and fits({i})}
    candidate_graph = graph.subgraph(eligible).copy()
    bundles = [{i} for i in sorted(eligible)]
    seeds = [set(), baseline]
    solver_calls = 0
    solver_ms = 0.0
    proposals = []
    if options.strategy in {"steiner_soft", "steiner_hard"} and eligible:
        # Scale the grid to the observed relevance/token density. This is
        # fixed before evaluation and never uses answer labels.
        scale = sum(prizes[i] for i in eligible) / max(
            1e-12, sum(normalized[i] for i in eligible)
        )
        for multiplier in options.multipliers:
            t0 = perf_counter()
            proposed = pcst_proposal(
                candidate_graph,
                prizes,
                normalized,
                beta,
                scale * multiplier,
                diagnostics=proposals,
            )
            solver_ms += (perf_counter() - t0) * 1000
            solver_calls += 1
            seeds.append(proposed)
            bundles.extend(
                set(c) for c in nx.connected_components(graph.subgraph(proposed))
            )

    # Multiple multipliers often return the same proposal.
    seeds = [set(seed) for seed in dict.fromkeys(frozenset(seed) for seed in seeds)]
    bundles = [
        set(bundle) for bundle in dict.fromkeys(frozenset(bundle) for bundle in bundles)
    ]

    def repair(selected: set[int]) -> set[int]:
        selected = selected.copy()
        # Never assume additive token costs exactly equal BPE serialization.
        while selected and not fits(selected):
            current = value(selected)
            removed = min(
                selected,
                key=lambda i: (
                    (current - value(selected - {i})) / max(normalized[i], 1e-12),
                    -i,
                ),
            )
            selected.remove(removed)
        return selected

    def grow(selected: set[int]) -> set[int]:
        selected = repair(selected)
        remaining = bundles.copy()
        while remaining:
            current = value(selected)
            # Use additive costs for ranking ONLY. Admission uses exact tokens.
            ranked = sorted(
                remaining,
                key=lambda bundle: (
                    -(value(selected | bundle) - current)
                    / max(sum(normalized[i] for i in bundle - selected), 1e-12),
                    tuple(sorted(bundle)),
                ),
            )
            accepted = False
            used = counts(selected)
            remaining = []
            for bundle in ranked:
                if bundle <= selected:
                    continue
                trial = selected | bundle
                added = bundle - selected
                estimated_fits = all(
                    used[kind] + sum(costs[i] for i in added if int(i >= ne) == kind)
                    <= budgets[kind]
                    for kind in (0, 1)
                )
                if not accepted and estimated_fits and fits(trial):
                    selected = trial
                    accepted = True
                else:
                    remaining.append(bundle)
            if not accepted:
                break
        # One exact refill pass recovers singleton records screened out by the
        # additive estimate near a BPE boundary. Multi-record bundles remain
        # heuristic; feasibility of every accepted selection is checked.
        for i in sorted(
            eligible - selected,
            key=lambda i: (-prizes[i] / max(normalized[i], 1e-12), i),
        ):
            if fits(selected | {i}):
                selected.add(i)
        return selected

    # Full proposals preserve useful bundles; one greedy growth run avoids a
    # full O(n^2) refill for every Lagrangian seed on the per-query hot path.
    selections = [repair(seed) for seed in seeds] + [grow(set())]
    if options.strategy == "steiner_hard":
        # Diagnostic only: one connected component in the record graph.
        selections = [
            set(c)
            for selected in selections
            for c in nx.connected_components(graph.subgraph(selected))
            if fits(set(c))
        ] + [{i} for i in sorted(eligible)]
    selected = max(
        selections or [set()],
        key=lambda s: (value(s), -sum(counts(s)), tuple(-i for i in sorted(s))),
    )
    used = counts(selected)
    check_budget(used, budgets)
    diagnostics = {
        "strategy": options.strategy,
        "executed_strategy": options.strategy,
        "fallback_reason": None,
        "score_source": options.prize_source,
        "prize_max": max(prizes, default=0.0),
        "effective_connectivity_bonus": beta,
        "candidate_count": len(records),
        "selected_count": len(selected),
        "entity_tokens": used[0],
        "relation_tokens": used[1],
        "components": nx.number_connected_components(graph.subgraph(selected)),
        "utility": value(selected),
        "baseline_utility": value(baseline),
        "solver_calls": solver_calls,
        "solver_ms": solver_ms,
        "proposals": proposals,
        "worker_ms": (perf_counter() - started) * 1000,
    }
    return (
        [row for i, row in enumerate(entities) if i in selected],
        [row for j, row in enumerate(relations, ne) if j in selected],
        diagnostics,
    )


async def select_context(
    entities: list[dict],
    relations: list[dict],
    tokenizer: Any,
    entity_budget: int,
    relation_budget: int,
    config: dict,
    *,
    query: str | None = None,
    rerank_func: Any = None,
    enable_rerank: bool = True,
) -> tuple[list[dict], list[dict], dict]:
    """Run CPU work off the event loop; input size is capped, not wall time.

    SteinerPy's heuristic ignores the exact solver's time_limit argument.
    Do not advertise an asyncio timeout as cancellation of this worker.
    """
    options = SelectionOptions(**config)
    requested_prize_source = options.prize_source
    # Query-level opt-out disables KG and chunk reranking together. Preserve
    # the selector but explicitly use the ordering control, without a call.
    if not enable_rerank and options.prize_source == "rerank_score":
        options = replace(options, prize_source="ordering")
    original_count = len(entities) + len(relations)
    prizes = None
    kg_rerank_ms = 0.0
    if options.prize_source == "rerank_score":
        entities, relations, prizes, kg_rerank_ms = await rerank_records(
            entities, relations, query, rerank_func
        )
    selected_e, selected_r, diagnostics = await asyncio.to_thread(
        _select_sync,
        entities,
        relations,
        tokenizer,
        entity_budget,
        relation_budget,
        options,
        prizes,
    )
    diagnostics["kg_rerank_ms"] = kg_rerank_ms
    diagnostics["requested_prize_source"] = requested_prize_source
    diagnostics["rerank_disabled"] = not enable_rerank
    diagnostics["input_candidate_count"] = original_count
    diagnostics["unscored_candidate_count"] = (
        original_count - len(entities) - len(relations)
    )
    return selected_e, selected_r, diagnostics
