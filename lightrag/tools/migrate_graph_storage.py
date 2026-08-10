#!/usr/bin/env python3
"""Backend-to-backend graph storage migration (Phase 1).

Phase 1 migrates a graph from ``PGGraphStorage`` (Apache AGE) to
``PGTableGraphStorage`` through the public storage API only:
``get_all_nodes`` / ``get_all_edges`` to read, ``upsert_nodes_batch`` /
``upsert_edges_batch`` to write, ``remove_nodes`` / ``remove_edges`` to
compensate on failure. The first half of this module is pure,
side-effect-free building blocks — no I/O, no async, no database imports —
so every migration invariant can be unit-tested offline; the second half is
the async orchestration (preconditions, write pipeline, verification,
id-scoped compensation) driven entirely through the storage objects it is
handed, so it stays testable with fakes and importable without asyncpg.

Shared edge-dict shape (the "enumerated edge" currency of this module):
both backends' ``get_all_edges`` return each edge as a single flat dict of
its properties with the endpoint ids merged in under the ``source`` and
``target`` keys. The AGE backend enumerates the *directed* edge table and
documents that "if 2 directional edges exist between the same pair of nodes,
deduplication must be handled by the caller" — this module is that caller.

Ordering invariant carried over from ``pgtable_impl``: the target stores
each undirected edge once in canonical order ``src_id = min(a, b)``,
``tgt_id = max(a, b)`` computed with **Python** ``min``/``max`` (never SQL
``LEAST``/``GREATEST``: under non-C collations the two orderings diverge on
non-ASCII ids and would produce duplicate edges). ``canonicalize_edge`` is
the single place this tool decides canonical order.

Payload comparison is deliberately TYPE-STRICT (see ``payloads_equal``).
Residual risk accepted with that choice: if the agtype-to-JSONB round trip
ever drifts a numeric representation (e.g. ``1`` read back as ``1.0``),
type-strict verification will abort a migration that lost nothing. That
failure is visible, diagnosable at the source, and recoverable; the
opposite error — blessing a payload that silently changed type — is not.

Known caveat: a payload key literally named ``source`` or ``target`` is
already destroyed upstream — ``PGGraphStorage.get_all_edges`` overwrites
those keys with the endpoint ids when flattening each row — so by the time
these helpers run, the original value is unrecoverable and ``edge_properties``
strips the keys as endpoint identity. LightRAG's own edge schema uses
``source_id`` for chunk attribution, so this does not arise in practice.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

# Endpoint keys that get_all_edges merges into the property dict on both
# backends. They identify the edge; everything else IS the edge's payload.
_ENDPOINT_KEYS = ("source", "target")


def canonicalize_edge(source: str, target: str) -> tuple[str, str]:
    """Return the canonical (src, tgt) order for an undirected edge.

    Python ``min``/``max`` over code points, exactly as
    ``PGTableGraphStorage`` normalises writes (see its module docstring for
    why SQL ``LEAST``/``GREATEST`` must never be substituted). This is the
    single place the migration tool decides canonical order.
    """
    return min(source, target), max(source, target)


def edge_properties(edge: dict[str, Any]) -> dict[str, Any]:
    """The payload of an enumerated edge: everything except the endpoints.

    ``source``/``target`` are identity, not payload — the target backend
    re-derives them from canonical order, so they never participate in
    property comparison.
    """
    return {k: v for k, v in edge.items() if k not in _ENDPOINT_KEYS}


def payloads_equal(a: Any, b: Any) -> bool:
    """Type-strict structural equality for payload values.

    The single comparison rule shared by ``detect_divergent_reciprocals``
    and ``verify_source_side`` (node and edge payloads alike). Recurses into
    dicts and lists; every leaf must match in TYPE and value. Bare ``==``
    would coerce across ``bool``/``int``/``float`` (``True == 1``,
    ``1 == 1.0``), silently passing a payload whose type changed — and both
    agtype and JSON distinguish these types, so the change is observable
    downstream. ``type(a) is type(b)`` is required rather than isinstance
    checks precisely because ``bool`` subclasses ``int``.

    Under the fail-closed policy any observable difference must count as
    divergence: a false abort is resolvable by an operator at the source,
    a false pass is silent data loss. Consequences, all deliberate:

    - key order never matters (dict comparison is key-set + per-key);
    - a key present with ``None`` differs from an absent key;
    - ``"1"`` vs ``1``, ``True`` vs ``1``, ``1`` vs ``1.0`` all diverge;
    - ``NaN`` never equals anything, itself included, so byte-identical
      payloads containing ``NaN`` compare unequal — a false positive in the
      safe (abort) direction.

    Leaf ``==`` is reached ONLY for the exact JSON scalar types (str, int,
    float, bool, None). Tuples recurse like lists (positional containers,
    so hand-built input cannot smuggle a coercing leaf comparison past the
    type gate inside a tuple wrapper); any other type — sets, custom
    classes — compares unequal unconditionally, because ``==`` on an
    unknown type may coerce and this function must be unfoolable.
    JSON/agtype-parsed payloads only ever contain the safe set, so neither
    branch fires on real migration data.
    """
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(
            payloads_equal(value, b[key]) for key, value in a.items()
        )
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(payloads_equal(x, y) for x, y in zip(a, b))
    if type(a) in (str, int, float, bool, type(None)):
        return a == b
    # Unknown type: fail closed rather than trust its __eq__.
    return False


@dataclass(frozen=True)
class DivergentReciprocal:
    """A reciprocal directed pair whose payloads disagree.

    ``canonical_key`` is the (min, max) undirected key; ``forward`` holds
    the properties of the canonical-order row (canonical_key[0] ->
    canonical_key[1]) and ``backward`` the reverse row's.
    """

    canonical_key: tuple[str, str]
    forward: dict[str, Any]
    backward: dict[str, Any]


def detect_divergent_reciprocals(
    directed_edges: Iterable[dict[str, Any]],
) -> list[DivergentReciprocal]:
    """Detect reciprocal directed pairs (a->b AND b->a) whose payloads differ.

    Canonicalizing such a pair would keep exactly one payload and silently
    drop the other — the loss class this tool exists to prevent. Phase 1
    policy is fail-closed: this function only DETECTS; the caller aborts on a
    non-empty result.

    Property comparison rule: strip the ``source``/``target`` endpoint keys,
    then compare the remaining dicts with ``payloads_equal`` — type-strict
    structural equality. Every remaining key participates (the tool cannot
    know which keys a downstream consumer relies on, so no key is exempt),
    key order never matters, and no normalisation is applied: see the
    ``payloads_equal`` docstring for the full rule and its rationale.

    Self-loops (a->a) have a single orientation and can never diverge.
    Parallel rows for the SAME ordered pair are a different invariant
    violation — ``count_parallel_edges`` is the backstop for those; here the
    input is deterministically ordered first (``sort_directed_edges``) and
    the last row per ordered pair is compared, so even violating input
    yields the same verdict regardless of enumeration order.

    Results are sorted by canonical key so two runs over the same (unordered)
    enumeration report identically.
    """
    by_ordered_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for edge in sort_directed_edges(directed_edges):
        by_ordered_pair[(edge["source"], edge["target"])] = edge_properties(edge)

    divergent: list[DivergentReciprocal] = []
    for (src, tgt), forward_props in by_ordered_pair.items():
        if src >= tgt:
            # Visit each unordered pair once, from its canonical-order row;
            # self-loops (src == tgt) have no reverse orientation.
            continue
        reverse = by_ordered_pair.get((tgt, src))
        if reverse is not None and not payloads_equal(reverse, forward_props):
            divergent.append(
                DivergentReciprocal(
                    canonical_key=(src, tgt),
                    forward=forward_props,
                    backward=reverse,
                )
            )
    divergent.sort(key=lambda d: d.canonical_key)
    return divergent


def count_parallel_edges(
    directed_edges: Iterable[dict[str, Any]],
) -> dict[tuple[str, str], int]:
    """Count enumerated rows per ORDERED (source, target) pair.

    Backstop for the AGE invariant "at most one directed edge per ordered
    pair": a count above 1 means parallel edges survived the AGE enumeration
    query's SELECT DISTINCT (over source, target and properties), so their
    agtype representations differ — though the parsed Python payloads may
    still compare equal (e.g. a ``1`` vs ``1.0`` representation drift), so a
    violation is not proof of semantic divergence. Canonicalization would
    keep only one row either way; the caller fails closed on any count > 1.

    Keyed by the ordered pair, not the canonical key, because a legitimate
    reciprocal pair (a->b plus b->a) also maps two rows onto one canonical
    key — that case belongs to ``detect_divergent_reciprocals``, not here.
    """
    counts: dict[tuple[str, str], int] = {}
    for edge in directed_edges:
        key = (edge["source"], edge["target"])
        counts[key] = counts.get(key, 0) + 1
    return counts


def derive_written_node_ids(
    migrated_node_ids: Iterable[str],
    edges: Iterable[dict[str, Any]],
) -> set[str]:
    """The exact node-id set a target-side compensation must remove.

    ``written = migrated_node_ids UNION {every edge endpoint}``.

    ``PGTableGraphStorage.upsert_edges_batch`` returns ``None`` and silently
    auto-creates missing endpoints (a CTE inserting stub nodes with
    ``ON CONFLICT DO NOTHING``), so endpoints that were never in the node
    batch are written without being observable in any return value — they
    must be DERIVED here from the edge list itself.

    Precondition (empty-target invariant): this set equals "everything the
    migration wrote" ONLY if the target workspace/namespace slice was empty
    when the migration started. On a non-empty target, ``ON CONFLICT`` paths
    touch pre-existing rows this set cannot distinguish from migration
    writes, so an id-scoped compensation keyed on it would delete data the
    migration never created.
    """
    written = set(migrated_node_ids)
    for edge in edges:
        written.add(edge["source"])
        written.add(edge["target"])
    return written


def sort_directed_edges(
    directed_edges: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Deterministically order an enumerated directed edge list.

    The AGE enumeration query has no ORDER BY, so two reads of the same
    graph may enumerate in different orders; every downstream step must see
    a stable order for two runs to write and report identically. Sorted by
    (source, target) with a canonical-JSON rendering of the payload as
    tiebreaker, so the order is total even on invariant-violating input
    (parallel rows), which fails closed later with a deterministic message.

    The tiebreaker serialisation can raise ``TypeError`` on a nested payload
    dict with mixed-type keys and ``ValueError`` on a circular payload.
    Neither can arise from JSON-parsed AGE payloads (JSON object keys are
    always strings, parsed structures are acyclic); on hand-built input the
    exception is the correct fail-closed outcome.
    """
    return sorted(
        directed_edges,
        key=lambda edge: (
            edge["source"],
            edge["target"],
            json.dumps(
                edge_properties(edge), sort_keys=True, ensure_ascii=False, default=str
            ),
        ),
    )


@dataclass
class VerificationReport:
    """Source-driven post-migration comparison, one field per mismatch class.

    Verification is driven from the SOURCE side because the target has
    already canonicalized: a payload lost to reciprocal collapse or a
    dropped row is invisible when walking the target alone.
    """

    # Node-id set differences (source ids vs target ids).
    missing_node_ids: list[str] = field(default_factory=list)
    unexpected_node_ids: list[str] = field(default_factory=list)
    # Node ids present on both sides whose property dicts differ:
    # (node_id, source_properties, target_properties).
    node_property_mismatches: list[tuple[str, dict[str, Any], dict[str, Any]]] = field(
        default_factory=list
    )
    # Canonical-edge set differences.
    missing_edges: list[tuple[str, str]] = field(default_factory=list)
    unexpected_edges: list[tuple[str, str]] = field(default_factory=list)
    # Canonical keys present on both sides whose payloads differ:
    # (canonical_key, source_properties, target_properties).
    edge_property_mismatches: list[
        tuple[tuple[str, str], dict[str, Any], dict[str, Any]]
    ] = field(default_factory=list)
    # Divergent reciprocals found in the SOURCE enumeration itself. Non-empty
    # means the fail-closed precondition was violated upstream; the source
    # cannot be compared meaningfully until resolved.
    source_reciprocal_divergence: list[DivergentReciprocal] = field(
        default_factory=list
    )
    # Ordered pairs enumerated more than once in the SOURCE (parallel rows),
    # with their counts. Same reasoning as reciprocal divergence: a source
    # violating the at-most-one-row-per-ordered-pair invariant has no
    # well-defined expected target, and canonicalizing its edge map keeps an
    # arbitrary survivor — so verification must fail regardless of whether
    # the caller already ran the backstop.
    source_parallel_violations: dict[tuple[str, str], int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not (
            self.missing_node_ids
            or self.unexpected_node_ids
            or self.node_property_mismatches
            or self.missing_edges
            or self.unexpected_edges
            or self.edge_property_mismatches
            or self.source_reciprocal_divergence
            or self.source_parallel_violations
        )


def _node_payload(node: dict[str, Any]) -> dict[str, Any]:
    # Both backends' get_all_nodes merge the id into the dict under "id";
    # like edge endpoints, it is identity, not payload.
    return {k: v for k, v in node.items() if k != "id"}


def _canonical_edge_map(
    edges: Iterable[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        canonicalize_edge(edge["source"], edge["target"]): edge_properties(edge)
        for edge in edges
    }


def verify_source_side(
    source_nodes: list[dict[str, Any]],
    source_edges: list[dict[str, Any]],
    target_nodes: list[dict[str, Any]],
    target_edges: list[dict[str, Any]],
) -> VerificationReport:
    """Compare a migrated target against its source, source-driven.

    Inputs are the raw ``get_all_nodes`` / ``get_all_edges`` results of each
    backend (nodes carry ``id``; edges carry ``source``/``target`` merged
    over the payload; source edges may be AGE-shaped directed rows).

    Node ids are compared as sets, then per-id payloads with the shared
    type-strict ``payloads_equal`` rule; edges as canonical-key sets, then
    per-key payloads under the same rule. Both source-side preconditions —
    no divergent reciprocals, no parallel rows per ordered pair — are
    re-checked here and reported rather than assumed, because verification
    is the last line of defense: a source that fails its own precondition
    has no well-defined expected target, and it must not depend on the
    caller having run the backstops. The source enumeration is
    deterministically ordered first, so the whole report (not just ``ok``)
    is independent of SOURCE enumeration order. (Target order needs no such
    pinning: the PGTable primary key (workspace, namespace, src_id, tgt_id)
    guarantees at most one row per canonical key, so a real target cannot
    present the duplicate rows that would make last-write-wins order-
    sensitive.) ``unexpected_*`` entries are
    target rows with no source counterpart — under the empty-target
    invariant those can only be migration bugs (e.g. stub endpoints for
    edges the source never had).
    """
    report = VerificationReport()

    # AGE enumerates with no ORDER BY; pin the order so map construction
    # (last-write-wins on violating input) and every derived field are
    # reproducible across runs.
    source_edges = sort_directed_edges(source_edges)

    report.source_reciprocal_divergence = detect_divergent_reciprocals(source_edges)
    report.source_parallel_violations = {
        pair: count
        for pair, count in sorted(count_parallel_edges(source_edges).items())
        if count > 1
    }

    source_node_map = {node["id"]: _node_payload(node) for node in source_nodes}
    target_node_map = {node["id"]: _node_payload(node) for node in target_nodes}
    report.missing_node_ids = sorted(source_node_map.keys() - target_node_map.keys())
    report.unexpected_node_ids = sorted(target_node_map.keys() - source_node_map.keys())
    for node_id in sorted(source_node_map.keys() & target_node_map.keys()):
        if not payloads_equal(source_node_map[node_id], target_node_map[node_id]):
            report.node_property_mismatches.append(
                (node_id, source_node_map[node_id], target_node_map[node_id])
            )

    source_edge_map = _canonical_edge_map(source_edges)
    target_edge_map = _canonical_edge_map(target_edges)
    report.missing_edges = sorted(source_edge_map.keys() - target_edge_map.keys())
    report.unexpected_edges = sorted(target_edge_map.keys() - source_edge_map.keys())
    for key in sorted(source_edge_map.keys() & target_edge_map.keys()):
        if not payloads_equal(source_edge_map[key], target_edge_map[key]):
            report.edge_property_mismatches.append(
                (key, source_edge_map[key], target_edge_map[key])
            )

    return report


# ---------------------------------------------------------------------------
# Async orchestration — preconditions, write pipeline, compensation
# ---------------------------------------------------------------------------

# Phase 1 supports exactly one migration pair, keyed by class NAME rather
# than by imported class object so this module never has to import the DB
# backends (the offline CI environment does not install asyncpg).
ALLOWED_MIGRATION_PAIRS: frozenset[tuple[str, str]] = frozenset(
    {("PGGraphStorage", "PGTableGraphStorage")}
)

# Explicit extension points for Phase 2, deliberately empty: additional
# pairs register a per-backend precondition hook (extra checks to run
# before any write) and a compensation hook (how to undo that backend's
# writes) keyed by backend class name. Phase 1 hard-codes the PGTable
# behavior in migrate_graph instead of routing through these.
PRECONDITION_HOOKS: dict[str, Any] = {}
COMPENSATION_HOOKS: dict[str, Any] = {}


class MigrationError(RuntimeError):
    """Base class for migration failures.

    ``exit_code`` is the process exit status the CLI wrapper (wired in a
    later step) maps the failure to; it is part of the contract now so the
    fail-closed paths are pinned as non-zero by tests.
    """

    exit_code = 1


class MigrationPreconditionError(MigrationError):
    """A precondition (pair allowlist, empty target) failed before any write."""


class MigrationDataError(MigrationError):
    """The source enumeration violates a migration invariant (divergent
    reciprocals, parallel rows, node without entity_id). Raised before any
    write."""


class MigrationWriteError(MigrationError):
    """The write or verification phase failed after writes began.

    Carries the run report: ``report.compensated`` says whether the
    id-scoped compensation completed, and ``report.written_node_ids`` /
    ``report.written_edge_keys`` are exactly what an operator must remove
    by hand if it did not (compensation itself is non-atomic — there is no
    cross-backend transaction — so a crash mid-compensation can leave the
    target partially populated; the empty-target gate then rejects a rerun
    until the slice is cleaned up).
    """

    def __init__(self, message: str, report: MigrationRunReport):
        super().__init__(message)
        self.report = report


@dataclass
class MigrationRunReport:
    """What one migration run read, wrote, and concluded."""

    source_backend: str
    target_backend: str
    node_count: int = 0
    edge_count: int = 0
    # Everything this run wrote (or would have written when a write failed
    # partway): computed BEFORE the first write via derive_written_node_ids,
    # so compensation and manual cleanup always have the full superset.
    # Absent ids are safe to pass to remove_* (DELETE ... = ANY is a no-op
    # for them).
    written_node_ids: list[str] = field(default_factory=list)
    written_edge_keys: list[tuple[str, str]] = field(default_factory=list)
    verified: bool = False
    verification: VerificationReport | None = None
    compensated: bool = False
    compensation_error: str | None = None


def check_migration_pair(source: Any, target: Any) -> None:
    """Reject any (source, target) pair outside the Phase 1 allowlist.

    Runs before every other precondition and therefore before any read or
    write on either storage.
    """
    pair = (type(source).__name__, type(target).__name__)
    if pair not in ALLOWED_MIGRATION_PAIRS:
        allowed = ", ".join(f"{s} -> {t}" for s, t in sorted(ALLOWED_MIGRATION_PAIRS))
        raise MigrationPreconditionError(
            f"unsupported migration pair {pair[0]} -> {pair[1]}; "
            f"Phase 1 supports exactly: {allowed}"
        )


async def target_slice_is_empty(target: Any) -> bool:
    """Empty-target gate, via ``get_all_nodes``.

    NEVER ``is_empty()``: the BaseGraphStorage stub has a docstring-only
    body (returns None) and PGTableGraphStorage does not override it, so a
    gate built on it silently passes on a populated target — the exact
    fail-open this check exists to prevent.
    """
    return len(await target.get_all_nodes()) == 0


async def compensate_partial_migration(
    target: Any,
    written_node_ids: Iterable[str],
    written_edge_keys: Iterable[tuple[str, str]],
) -> None:
    """Remove exactly what this run wrote: edges first, then nodes.

    Edge-before-node order follows the FK direction (edges reference their
    endpoint rows). Both remove_* contracts are DELETE ... = ANY(ids), so
    ids that were never actually written — e.g. when the failure hit before
    the edge batch — are silent no-ops, which is why the full derived
    superset is always passed. Exact under the empty-target invariant only:
    on a target that was not empty at T0, these ids may name pre-existing
    rows the migration never created (see derive_written_node_ids).
    """
    await target.remove_edges(sorted(written_edge_keys))
    await target.remove_nodes(sorted(written_node_ids))


async def migrate_graph(
    source: Any, target: Any, *, force_empty_target: bool = False
) -> MigrationRunReport:
    """Migrate the whole graph from source to target via the storage API.

    Pipeline: pair allowlist -> empty-target gate -> enumerate + fail-closed
    source checks (parallel rows, divergent reciprocals, entity_id present)
    -> upsert nodes -> upsert edges -> source-driven verification. Any
    failure after writes begin triggers id-scoped compensation and raises
    MigrationWriteError carrying the run report.

    ``force_empty_target=True`` (CLI: --force-empty-target, wired in a later
    step) is the ONLY path that touches pre-existing target data: it drops
    the whole target slice instead of rejecting a non-empty target. It is
    never used for compensation.
    """
    check_migration_pair(source, target)

    if not await target_slice_is_empty(target):
        if not force_empty_target:
            raise MigrationPreconditionError(
                "target graph slice is not empty; refusing to write "
                "(pass force_empty_target to drop it instead)"
            )
        await target.drop()

    source_nodes = await source.get_all_nodes()
    source_edges = sort_directed_edges(await source.get_all_edges())

    violations = {
        pair: count
        for pair, count in count_parallel_edges(source_edges).items()
        if count > 1
    }
    if violations:
        raise MigrationDataError(
            f"source violates the one-directed-edge-per-ordered-pair "
            f"invariant; parallel rows for: {sorted(violations)}"
        )
    divergent = detect_divergent_reciprocals(source_edges)
    if divergent:
        raise MigrationDataError(
            "source has reciprocal directed edges with divergent properties; "
            "canonicalization would silently drop one payload per pair: "
            f"{[d.canonical_key for d in divergent]}"
        )

    node_items: list[tuple[str, dict[str, Any]]] = []
    for node in source_nodes:
        payload = _node_payload(node)
        if "entity_id" not in payload:
            raise MigrationDataError(
                f"source node {node.get('id')!r} has no entity_id property; "
                "the target backend requires it"
            )
        node_items.append((node["id"], payload))

    canonical_edges = _canonical_edge_map(source_edges)
    edge_items = [
        (src, tgt, props) for (src, tgt), props in sorted(canonical_edges.items())
    ]

    report = MigrationRunReport(
        source_backend=type(source).__name__,
        target_backend=type(target).__name__,
        node_count=len(node_items),
        edge_count=len(edge_items),
        # Derived before the first write so a failure at ANY later point
        # already has the complete compensation set, auto-created edge
        # endpoints included.
        written_node_ids=sorted(
            derive_written_node_ids(
                (node_id for node_id, _ in node_items), source_edges
            )
        ),
        written_edge_keys=sorted(canonical_edges),
    )

    try:
        await target.upsert_nodes_batch(node_items)
        await target.upsert_edges_batch(edge_items)
        report.verification = verify_source_side(
            source_nodes,
            source_edges,
            await target.get_all_nodes(),
            await target.get_all_edges(),
        )
        if not report.verification.ok:
            raise MigrationError("source-driven verification failed")
    except Exception as exc:
        try:
            await compensate_partial_migration(
                target, report.written_node_ids, report.written_edge_keys
            )
            report.compensated = True
        except Exception as compensation_exc:  # noqa: BLE001 — report, then surface the original failure
            report.compensation_error = repr(compensation_exc)
        raise MigrationWriteError(
            f"migration failed after writes began ({exc}); "
            + (
                "id-scoped compensation completed"
                if report.compensated
                else "COMPENSATION INCOMPLETE — remove the reported "
                "written_edge_keys then written_node_ids by hand"
            ),
            report,
        ) from exc

    report.verified = True
    return report
