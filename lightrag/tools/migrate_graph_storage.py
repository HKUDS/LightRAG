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

Usage (dry run is the default; no graph data is written without
--apply, though initializing the backends still creates their schema):

    python -m lightrag.tools.migrate_graph_storage
    python -m lightrag.tools.migrate_graph_storage --apply [--workspace WS]

Payload comparison is deliberately TYPE-STRICT (see ``payloads_equal``).
Residual risk accepted with that choice: if the agtype-to-JSONB round trip
ever drifts a numeric representation (e.g. ``1`` read back as ``1.0``),
type-strict verification will abort a migration that lost nothing. That
failure is visible, diagnosable at the source, and recoverable; the
opposite error — blessing a payload that silently changed type — is not.

Manual e2e checklist (not testable with fakes): real AGE builds the edge
``source``/``target`` ids via ``(agtype_access_operator(...))::text``;
whether that cast can carry agtype quoting into the id strings can only be
answered against a live AGE instance (cf. the lossy agtype-quote-strip
work in #3587, same territory). If it ever does, the failure mode here is
fail-closed — endpoint ids diverge, verification aborts and compensates —
not silent loss.

Known caveat: a payload key literally named ``source`` or ``target`` is
already destroyed upstream — ``PGGraphStorage.get_all_edges`` overwrites
those keys with the endpoint ids when flattening each row — so by the time
these helpers run, the original value is unrecoverable and ``edge_properties``
strips the keys as endpoint identity. LightRAG's own edge schema uses
``source_id`` for chunk attribution, so this does not arise in practice.
"""

from __future__ import annotations

import argparse
import asyncio
import configparser
import json
import os
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

    The single comparison rule used everywhere this module compares payloads
    — ``verify_source_side``, node and edge payloads alike. Recurses into
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
    if isinstance(a, float) and not isinstance(a, bool):
        # -0.0 == +0.0 is True, but they are observably different: json.dumps
        # renders them differently and math.copysign distinguishes them. Under
        # the fail-closed rule an observable difference must count.
        import math

        return a == b and math.copysign(1.0, a) == math.copysign(1.0, b)
    if type(a) in (str, int, float, bool, type(None)):
        return a == b
    # Unknown type: fail closed rather than trust its __eq__.
    return False


def validate_source_nodes(nodes: Iterable[dict[str, Any]]) -> None:
    """Reject malformed enumerated nodes BEFORE anything sorts or groups them.

    A real AGE graph can contain a vertex with no properties at all (``CREATE
    (n:base)``), which enumerates as ``{"id": None}``. Sorting or grouping that
    alongside real ids raises ``TypeError: '<' not supported between instances
    of 'NoneType' and 'str'`` — a crash instead of the refusal the operator
    needs, and it happens before the entity_id check can speak. So identity is
    validated first, and the error names the offending node and the type it
    actually had.

    ``id`` and ``entity_id`` must both be present, both be ``str``, and agree:
    the enumerator derives ``id`` from ``entity_id``, so a disagreement means
    the row is not what this tool assumes and is refused defensively.
    """
    for position, node in enumerate(nodes):
        node_id = node.get("id")
        entity_id = node.get("entity_id")
        if not isinstance(node_id, str):
            raise MigrationDataError(
                f"source node at position {position} has a non-string id "
                f"({type(node_id).__name__}: {node_id!r}); a vertex with no "
                "properties enumerates this way. Repair or remove it before "
                "migrating."
            )
        if not isinstance(entity_id, str):
            raise MigrationDataError(
                f"source node {node_id!r} has no usable entity_id "
                f"({type(entity_id).__name__}: {entity_id!r}); the target "
                "requires one on every node."
            )
        if node_id != entity_id:
            raise MigrationDataError(
                f"source node id {node_id!r} disagrees with its entity_id "
                f"{entity_id!r}; refusing rather than guessing which is the "
                "identity."
            )


def _is_non_text_safe(text: str) -> bool:
    """True when PostgreSQL ``text`` (and therefore jsonb) cannot hold ``text``.

    Exactly two cases, both of which a Python ``str`` and AGE's agtype accept
    but ``text`` does not: an embedded NUL, and an unpaired surrogate. The
    surrogate test is an explicit code-point range rather than a
    ``value.encode("utf-8")`` try/except because the range says what is
    rejected, while the encode also depends on which surrogate handler the
    driver happens to use.
    """
    return "\x00" in text or any(0xD800 <= ord(char) <= 0xDFFF for char in text)


def _unrepresentable_paths(value: Any, path: str = "") -> list[str]:
    """Paths of every value PostgreSQL jsonb cannot store faithfully.

    Two float families qualify, for the same underlying reason — the target
    cannot round-trip them, so the migration would fail or silently change a
    value AFTER the point of no return:

    - ``NaN`` / ``+-Infinity``: agtype represents them, jsonb rejects them
      outright ("invalid input syntax for type json");
    - ``-0.0``: jsonb accepts it but normalises the sign away
      (``'{"w": -0.0}'::jsonb`` stores ``{"w": 0.0}``). This tool's comparison
      rule deliberately treats signed zeros as different, so migrating one
      would write successfully and then fail its own verification.

    Two string families qualify for the same reason — jsonb is stored as
    PostgreSQL ``text``, which is strictly valid UTF-8 with no embedded NUL:

    - a ``\\x00`` anywhere in a string ("unsupported Unicode escape sequence:
      \\u0000 cannot be converted to text");
    - a lone surrogate (U+D800..U+DFFF), which Python tolerates in a ``str``
      but which has no UTF-8 encoding at all, so the driver cannot even send
      it. AGE stores both happily, so a source graph can carry them.

    Dict KEYS are checked with the same string rule: a key is serialized into
    the same jsonb document, so a NUL or lone surrogate in a key aborts the
    write exactly as one in a value does.
    """
    found: list[str] = []
    if isinstance(value, float) and not isinstance(value, bool):
        import math

        if (
            value != value
            or value in (float("inf"), float("-inf"))
            or (value == 0.0 and math.copysign(1.0, value) < 0)
        ):
            found.append(path or "<root>")
    elif isinstance(value, str):
        if _is_non_text_safe(value):
            found.append(path or "<root>")
    elif isinstance(value, dict):
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            if isinstance(key, str) and _is_non_text_safe(key):
                found.append(child)
            found.extend(_unrepresentable_paths(item, child))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.extend(_unrepresentable_paths(item, f"{path}[{index}]"))
    return found


def find_non_jsonb_representable(
    labelled_payloads: Iterable[tuple[str, dict[str, Any]]],
) -> list[tuple[str, list[str]]]:
    """Find payloads PostgreSQL ``jsonb`` cannot store faithfully.

    The general rule this enforces: if the target cannot round-trip a value,
    refuse BEFORE the destructive step. Nothing in the plan phase serializes to
    JSONB, so without this check such a payload passes the dry run and only
    fails at the write — which under ``--force-empty-target`` is after the
    pre-existing slice was dropped, leaving neither the old data nor the new.
    The tool already knows it is migrating agtype into jsonb, so a deterministic
    failure can be predicted instead of discovered.

    See ``_unrepresentable_paths`` for what qualifies and why.
    """
    offenders = []
    for label, payload in labelled_payloads:
        paths = _unrepresentable_paths(payload)
        if paths:
            offenders.append((label, sorted(paths)))
    return sorted(offenders)


def detect_reciprocal_pairs(
    directed_edges: Iterable[dict[str, Any]],
) -> list[tuple[str, str]]:
    """Canonical keys that have BOTH directions present, payloads irrelevant.

    Phase 1 refuses every reciprocal pair, not only the ones whose payloads
    disagree. Even an identical ``a->b`` / ``b->a`` pair is two relationship
    rows in AGE and one row in the target, and AGE computes degree by counting
    relationship rows — so collapsing them changes degree and traversal even
    though no property is lost. LightRAG's own write path matches edges
    undirected (``OPTIONAL MATCH (source)-[old:DIRECTED]-(target)``), so it does
    not produce reciprocals: finding one means the source violates the
    invariant this tool assumes, and refusing keeps ``verified`` meaning that
    node/edge cardinality and degree were preserved too.

    A payload-aware relaxation — rejecting only the reciprocal pairs whose two
    payloads disagree — is deliberately NOT provided. It would be wrong, not
    merely conservative: AGE computes degree by counting relationship rows, so
    collapsing even a byte-identical ``a->b`` / ``b->a`` pair drops the degree
    of both endpoints by one and changes what traversal returns, while losing
    no property. Phase 1 therefore fails closed on all of them, and an operator
    who knows the reciprocal is redundant deletes it at the source instead.
    """
    seen: dict[tuple[str, str], set[tuple[str, str]]] = {}
    for edge in directed_edges:
        src, tgt = edge["source"], edge["target"]
        if src == tgt:
            continue  # a self-loop has only one orientation
        seen.setdefault(canonicalize_edge(src, tgt), set()).add((src, tgt))
    return sorted(key for key, directions in seen.items() if len(directions) > 1)


def detect_duplicate_node_ids(nodes: Iterable[dict[str, Any]]) -> list[str]:
    """Node ids enumerated more than once, payloads irrelevant.

    Same reasoning as reciprocals: the target's primary key keeps one row per
    id, so two physical source vertices sharing an entity_id become one node
    and the graph's node cardinality (and any degree derived from the edges
    that pointed at each of them) changes. Refused regardless of whether the
    payloads match.

    As with ``detect_reciprocal_pairs``, a payload-aware relaxation that passed
    duplicates with identical payloads is deliberately NOT provided: the
    cardinality and degree change happens whether or not a property differs, so
    equal payloads make the collapse invisible rather than harmless.
    """
    counts: dict[str, int] = {}
    for node in nodes:
        node_id = node.get("id")
        if isinstance(node_id, str):
            counts[node_id] = counts.get(node_id, 0) + 1
    return sorted(node_id for node_id, count in counts.items() if count > 1)


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
    key — that case belongs to ``detect_reciprocal_pairs``, not here.
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
    source_reciprocal_pairs: list[tuple[str, str]] = field(default_factory=list)
    # Ordered pairs enumerated more than once in the SOURCE (parallel rows),
    # with their counts. Same reasoning as reciprocal divergence: a source
    # violating the at-most-one-row-per-ordered-pair invariant has no
    # well-defined expected target, and canonicalizing its edge map keeps an
    # arbitrary survivor — so verification must fail regardless of whether
    # the caller already ran the backstop.
    source_parallel_violations: dict[tuple[str, str], int] = field(default_factory=dict)
    # Node ids enumerated more than once in the SOURCE with differing
    # payloads. Same last-line-of-defense reasoning: the per-id maps below
    # collapse duplicates last-write-wins, so this is only detectable on the
    # raw enumeration — and without it a lost duplicate payload would verify
    # clean.
    source_duplicate_node_ids: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (
            self.missing_node_ids
            or self.unexpected_node_ids
            or self.node_property_mismatches
            or self.missing_edges
            or self.unexpected_edges
            or self.edge_property_mismatches
            or self.source_reciprocal_pairs
            or self.source_parallel_violations
            or self.source_duplicate_node_ids
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
    per-key payloads under the same rule. All three source-side
    preconditions — no divergent reciprocals, no parallel rows per ordered
    pair, no divergent duplicate node ids — are
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
    # reproducible across runs. Nodes get the same treatment as edges: on
    # duplicate-id input the per-id map below keeps the last payload, which
    # must not depend on enumeration order.
    source_edges = sort_directed_edges(source_edges)
    source_nodes = sorted(
        source_nodes,
        key=lambda node: (
            node["id"],
            json.dumps(
                _node_payload(node), sort_keys=True, ensure_ascii=False, default=str
            ),
        ),
    )

    # Same detectors the plan phase uses, so verification re-runs the source
    # checks rather than an older, weaker version of them: duplicates and
    # reciprocals count whether or not their payloads agree, because both
    # change cardinality on the way into the target.
    report.source_duplicate_node_ids = detect_duplicate_node_ids(source_nodes)
    report.source_reciprocal_pairs = detect_reciprocal_pairs(source_edges)
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
    # The workspace the backends actually resolved to, which is not
    # necessarily the one requested (POSTGRES_WORKSPACE outranks the
    # constructor argument). Reported so the slice a destructive run acted on
    # is never left implicit.
    resolved_workspace: str | None = None
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
    # Whether the target slice held pre-existing data at the gate, and
    # whether this run actually drop()ped it (apply + force_empty_target
    # only). Surfaced so a force-mode dry run cannot look like a no-risk
    # preview of a run that will destroy the slice.
    target_was_non_empty: bool = False
    dropped_target_slice: bool = False


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

    NEVER ``is_empty()``: graph storages do not have that method at all —
    the None-returning stub in ``lightrag/base.py`` belongs to
    ``BaseKVStorage``, a different branch of the hierarchy, and
    ``BaseGraphStorage`` defines nothing like it. A gate written against it
    would crash with AttributeError on a real backend (fail-closed by
    accident, and for the wrong reason); this gate asks a question the
    graph API actually answers.
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

    ``force_empty_target=True`` (CLI: --force-empty-target) is the ONLY path
    that touches pre-existing target data: it drops the whole target slice
    instead of rejecting a non-empty target. It is never used for
    compensation.
    """
    check_migration_pair(source, target)

    target_non_empty = not await target_slice_is_empty(target)
    if target_non_empty and not force_empty_target:
        raise MigrationPreconditionError(
            "target graph slice is not empty; refusing to write "
            "(pass force_empty_target to drop it instead)"
        )

    # Plan BEFORE dropping. _plan_migration is read-only and runs every
    # fail-closed source check, so ordering it first means a source that fails
    # one of them cannot cost the operator their existing target slice: the
    # single destructive action happens only once the migration is known to be
    # able to proceed.
    source_nodes, source_edges, node_items, edge_items, report = await _plan_migration(
        source, target
    )
    report.resolved_workspace = getattr(target, "workspace", None)
    report.target_was_non_empty = target_non_empty

    if target_non_empty:
        # PGTableGraphStorage.drop() reports failure by RETURNING
        # {"status": "error"} instead of raising, so an unchecked call would
        # record a slice as dropped while its rows are still there — and the
        # migration would then write into, and later compensate against, data
        # it does not own.
        drop_result = await target.drop()
        # Only an explicit success counts. The backend signals failure by
        # RETURNING {"status": "error"}, but treating "anything that is not
        # error" as success would also bless None, {"status": "failed"} or a
        # missing status — after which the run writes into, and compensates
        # against, rows it does not own. Phase 1 allow-lists exactly one target,
        # so its contract can be required exactly.
        if not (
            isinstance(drop_result, dict) and drop_result.get("status") == "success"
        ):
            raise MigrationPreconditionError(
                "failed to drop the target graph slice; expected "
                f'{{"status": "success"}} but got {drop_result!r}'
            )
        report.dropped_target_slice = True

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
    except BaseException as exc:
        # BaseException, not Exception: asyncio.CancelledError does NOT subclass
        # Exception, so Ctrl-C or a cancelled task partway through the writes
        # would otherwise leave the target populated with no compensation and no
        # report — the one failure mode the id-scoped undo exists to prevent.
        try:
            await compensate_partial_migration(
                target, report.written_node_ids, report.written_edge_keys
            )
            report.compensated = True
        except Exception as compensation_exc:  # noqa: BLE001 — report, then surface the original failure
            report.compensation_error = repr(compensation_exc)
        if isinstance(exc, asyncio.CancelledError):
            # Compensate, then let cancellation keep propagating as itself:
            # converting it would break the caller's cancellation semantics.
            raise
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


async def _plan_migration(
    source: Any, target: Any
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[tuple[str, dict[str, Any]]],
    list[tuple[str, str, dict[str, Any]]],
    MigrationRunReport,
]:
    """Enumerate the source, run every fail-closed check, build the write
    plan. Read-only; shared verbatim by migrate_graph and dry_run_migration
    so the dry run exercises exactly the checks the apply run would."""
    source_nodes = await source.get_all_nodes()
    # Identity FIRST: a vertex with no properties enumerates as {"id": None},
    # and sorting or grouping that alongside real ids raises TypeError before
    # any check can report anything useful.
    validate_source_nodes(source_nodes)

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
    # Reciprocals and duplicate ids are refused whether or not their payloads
    # agree. Collapsing them loses no property but does change node/edge
    # cardinality and degree, and `verified` is meant to mean the graph came
    # across — not merely its payload set.
    reciprocals = detect_reciprocal_pairs(source_edges)
    if reciprocals:
        raise MigrationDataError(
            "source has reciprocal directed edges (a->b and b->a); the target "
            "stores one row per canonical pair, so degree and traversal would "
            f"change even where payloads match: {reciprocals}"
        )
    duplicate_nodes = detect_duplicate_node_ids(source_nodes)
    if duplicate_nodes:
        raise MigrationDataError(
            "source enumerates the same node id more than once; the target "
            "keeps one row per id, so node cardinality would change even "
            f"where payloads match: {duplicate_nodes}"
        )
    non_jsonb = find_non_jsonb_representable(
        [(f"node {n['id']!r}", _node_payload(n)) for n in source_nodes]
        + [
            (
                f"edge {canonicalize_edge(e['source'], e['target'])!r}",
                edge_properties(e),
            )
            for e in source_edges
        ]
    )
    if non_jsonb:
        raise MigrationDataError(
            "source payloads contain values PostgreSQL jsonb cannot store "
            "faithfully (NaN, +-Infinity, -0.0 whose sign jsonb normalises "
            "away, or a string carrying a NUL or an unpaired surrogate, "
            "neither of which PostgreSQL text can hold); migrating them would "
            "fail after the point of no return: "
            f"{[(label, paths) for label, paths in non_jsonb]}"
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
    return source_nodes, source_edges, node_items, edge_items, report


async def dry_run_migration(
    source: Any, target: Any, *, force_empty_target: bool = False
) -> MigrationRunReport:
    """Read-only preflight: everything migrate_graph checks, zero writes.

    Runs the pair allowlist, the empty-target gate and every fail-closed
    source check, and returns the report the apply run would start from
    (counts and the would-be written id sets, ``verified`` False, no
    verification). Raises exactly the errors the apply run would raise at
    the same points. With ``force_empty_target`` a non-empty target is not
    rejected — but it is NOT dropped either; the drop happens only in the
    apply run.
    """
    check_migration_pair(source, target)
    target_non_empty = not await target_slice_is_empty(target)
    if target_non_empty and not force_empty_target:
        # Identical wording to migrate_graph: same code path, same claim.
        raise MigrationPreconditionError(
            "target graph slice is not empty; refusing to write "
            "(pass force_empty_target to drop it instead)"
        )
    _, _, _, _, report = await _plan_migration(source, target)
    # Reaching here with a non-empty target means force mode: the apply run
    # WILL drop the pre-existing slice. The report must say so — a
    # clean-looking force-mode preview of a destructive apply is exactly the
    # operator trap this field exists to prevent.
    report.resolved_workspace = getattr(target, "workspace", None)
    report.target_was_non_empty = target_non_empty
    return report


# ---------------------------------------------------------------------------
# CLI — dry-run by default, --apply to migrate
# ---------------------------------------------------------------------------


def _precheck_requested_workspace(requested: str) -> None:
    """Refuse an env-overridden workspace BEFORE anything is constructed.

    ``initialize()`` is not read-only: ``PGTableGraphStorage`` runs its DDL and
    then ``_normalize_legacy_edges``, which DELETEs and re-INSERTs edge rows.
    So a check that can only read the *resolved* workspace — which the backend
    settles during ``initialize()`` — necessarily runs after the wrong slice has
    already been touched. This mirrors the backend's documented priority
    (``POSTGRES_WORKSPACE`` env > ``config.ini`` ``[postgres] workspace`` >
    constructor argument > ``"default"``, pgtable_impl.py:531 resolving what
    ``PGPostgresDB.get_config`` settled) to catch the dangerous case before any
    connection is opened. ``_check_resolved_workspace`` stays as the backstop
    for whatever this cannot predict.

    ``config.ini`` matters as much as the env var and is easier to forget: it
    is not visible in the operator's shell, so a checked-in
    ``[postgres] workspace`` silently outranks ``--workspace`` with nothing on
    screen to suggest it. The error names which of the two sources spoke.
    """
    env_workspace = os.getenv("POSTGRES_WORKSPACE")
    if env_workspace is not None:
        # An env var set to "" is what get_config passes on, and an empty
        # workspace is falsy to the backend, so the constructor argument still
        # wins — and config.ini is never consulted. Mirror that exactly.
        effective, origin = env_workspace, "POSTGRES_WORKSPACE"
    else:
        # Same read as PGPostgresDB.get_config: cwd-relative, and a missing
        # file makes ConfigParser.read a silent no-op.
        config = configparser.ConfigParser()
        config.read("config.ini", "utf-8")
        effective = config.get("postgres", "workspace", fallback=None) or ""
        origin = "config.ini [postgres] workspace"
    if requested and effective and effective != requested:
        raise MigrationPreconditionError(
            f"requested workspace {requested!r} but {origin} is set to "
            f"{effective!r} and outranks it; refusing before opening any "
            "connection, because initializing a graph storage mutates the "
            "workspace it lands in"
        )


def _check_resolved_workspace(requested: str, source: Any, target: Any) -> None:
    """Fail closed when the backends did not land on the workspace we asked for.

    The graph backends resolve their workspace as ``POSTGRES_WORKSPACE`` env >
    ``config.ini`` ``[postgres] workspace`` > the value passed to the
    constructor > ``"default"`` (pgtable_impl.py:531 on top of
    ``PGPostgresDB.get_config``), so the environment or a checked-in config
    file silently outranks ``--workspace``. That matters
    because this tool has a destructive mode: without this check,
    ``--workspace staging --force-empty-target`` could drop a slice the
    operator never named. An empty request means the operator named nothing, so
    any resolution is legitimate.
    """
    resolved_source = getattr(source, "workspace", None)
    resolved_target = getattr(target, "workspace", None)
    if resolved_source != resolved_target:
        raise MigrationPreconditionError(
            "source and target resolved to different workspaces "
            f"({resolved_source!r} vs {resolved_target!r}); refusing to migrate"
        )
    if requested and resolved_target != requested:
        raise MigrationPreconditionError(
            f"requested workspace {requested!r} but the backends resolved to "
            f"{resolved_target!r} (POSTGRES_WORKSPACE overrides --workspace); "
            "refusing to operate on a workspace that was not named"
        )


async def _build_graph_storage(backend_name: str, workspace: str) -> Any:
    """Instantiate and initialize one graph storage via the factory.

    Imports are deliberately INSIDE the function (the rebuild_vdb pattern):
    the offline CI does not install asyncpg, and importing this module or
    its tests must never pull in a DB backend — only actually running the
    CLI may.
    """
    from lightrag.kg.factory import get_storage_class
    from lightrag.namespace import NameSpace
    from lightrag.utils import EmbeddingFunc

    async def _never_embed(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("graph migration never embeds")

    cls = get_storage_class(backend_name)
    storage = cls(
        namespace=NameSpace.GRAPH_STORE_CHUNK_ENTITY_RELATION,
        workspace=workspace,
        # Graph storages read their configuration via global_config.get(...)
        # (connection settings come from POSTGRES_* env vars); nothing is
        # required here for the graph-only paths this tool uses.
        global_config={},
        # Graph storages accept None at runtime (BaseVectorStorage validates
        # against None while the graph branch has no such guard — the
        # tolerance is deliberate), but the annotation is EmbeddingFunc and
        # does not promise it. Honor the declared contract the way
        # rebuild_vdb does when no embedding is available: manufacture a
        # stub whose func fails loudly, so a graph path that ever asks for
        # embeddings dies with an intentional message instead of an opaque
        # AttributeError. embedding_dim=1 and the omitted model_name are
        # safe here: the only embedding_func derivations in postgres_impl
        # (model_name suffix, embedding_dim) belong to PGVectorStorage;
        # PGGraphStorage never reads embedding_func.
        embedding_func=EmbeddingFunc(embedding_dim=1, func=_never_embed),
    )
    try:
        await storage.initialize()
    except BaseException:
        # Best-effort release of whatever initialize acquired before it
        # failed (e.g. a ClientManager reference, whose refcount would
        # otherwise stay stranded); the initialize failure is the error
        # worth surfacing, so a secondary finalize failure is swallowed.
        try:
            await storage.finalize()
        except Exception:
            pass
        raise
    return storage


def _report_dict(
    report: MigrationRunReport, mode: str, error: str | None = None
) -> dict[str, Any]:
    """stdout report shape, matching the existing tools' dict convention."""
    out: dict[str, Any] = {
        "mode": mode,
        "source_backend": report.source_backend,
        "target_backend": report.target_backend,
        "workspace": report.resolved_workspace,
        "nodes": report.node_count,
        "edges": report.edge_count,
        "verified": report.verified,
        "compensated": report.compensated,
        "target_non_empty": report.target_was_non_empty,
    }
    # Symmetric destructive-consequence fields: the force-mode dry run must
    # disclose that the apply run will drop the pre-existing slice, and the
    # apply run must disclose that it did.
    if mode == "dry-run":
        out["would_drop_target_slice"] = report.target_was_non_empty
    else:
        out["dropped_target_slice"] = report.dropped_target_slice
    if report.compensation_error is not None:
        out["compensation_error"] = report.compensation_error
        # The manual-cleanup contract: compensation did not finish, so the
        # operator needs the exact sets to remove by hand.
        out["written_node_ids"] = report.written_node_ids
        out["written_edge_keys"] = report.written_edge_keys
    else:
        out["written_node_count"] = len(report.written_node_ids)
        out["written_edge_count"] = len(report.written_edge_keys)
    if report.verification is not None and not report.verification.ok:
        out["verification_failed"] = True
    if error is not None:
        out["error"] = error
    return out


async def _async_main(args: argparse.Namespace) -> int:
    """Run the migration (or dry run) and return the process exit code."""
    mode = "apply" if args.apply else "dry-run"
    # Reject a disallowed pair before instantiating anything: building an
    # arbitrary backend could import drivers and open connections for a run
    # that is already doomed.
    pair = (args.source_backend, args.target_backend)
    if pair not in ALLOWED_MIGRATION_PAIRS:
        allowed = ", ".join(f"{s} -> {t}" for s, t in sorted(ALLOWED_MIGRATION_PAIRS))
        print(
            {
                "mode": mode,
                "error": f"unsupported migration pair {pair[0]} -> {pair[1]}; "
                f"Phase 1 supports exactly: {allowed}",
            }
        )
        return MigrationPreconditionError.exit_code

    source = None
    target = None
    try:
        try:
            # REQUIRED before any storage is initialized: the backends acquire
            # shared-storage locks in initialize(), which raise "Shared data not
            # initialized" without this. Every other tool in this directory does
            # the same (rebuild_vdb, migrate_llm_cache, clean_llm_query_cache,
            # source_conflict_repair); workers=1 because this is a standalone
            # single-process CLI, so the locks stay plain asyncio objects.
            from lightrag.kg.shared_storage import initialize_share_data

            initialize_share_data(workers=1)

            # Before ANY connection: initializing a graph storage mutates the
            # workspace it lands in (DDL plus _normalize_legacy_edges, which
            # deletes and re-inserts edge rows), so an env-overridden workspace
            # has to be refused here rather than after the fact.
            _precheck_requested_workspace(args.workspace)

            source = await _build_graph_storage(args.source_backend, args.workspace)
            target = await _build_graph_storage(args.target_backend, args.workspace)
            _check_resolved_workspace(args.workspace, source, target)
        except MigrationPreconditionError as exc:
            print({"mode": mode, "error": str(exc)})
            return exc.exit_code
        except Exception as exc:
            # Match rebuild_vdb: a storage build/initialize failure (DB
            # unreachable being the common real case) reports and exits
            # non-zero instead of escaping as a raw traceback.
            print({"mode": mode, "error": f"storage initialization failed: {exc}"})
            return 1
        if args.apply:
            report = await migrate_graph(
                source, target, force_empty_target=args.force_empty_target
            )
        else:
            report = await dry_run_migration(
                source, target, force_empty_target=args.force_empty_target
            )
        print(_report_dict(report, mode))
        return 0
    except MigrationWriteError as exc:
        print(_report_dict(exc.report, mode, error=str(exc)))
        return exc.exit_code
    except MigrationError as exc:
        print({"mode": mode, "error": str(exc)})
        return exc.exit_code
    # A non-MigrationError failure during the migration itself escapes as a
    # raw traceback — deliberate, matching kg_integrity_repair and
    # source_conflict_repair; the process still exits non-zero, and a
    # blanket catch here would only hide genuine bugs.
    finally:
        for storage in (source, target):
            if storage is not None:
                await storage.finalize()


def main(argv: list[str] | None = None) -> None:
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=".env", override=False)
    parser = argparse.ArgumentParser(
        description=(
            "Migrate a LightRAG graph between storage backends through the "
            "public storage API (Phase 1: PGGraphStorage -> "
            "PGTableGraphStorage). Stop every LightRAG writer first. "
            "Default is a dry run: no graph data is migrated, though "
            "initializing the backends does touch their storage."
        )
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform the migration (default: dry-run, which migrates nothing)",
    )
    parser.add_argument(
        "--workspace",
        default=os.getenv("WORKSPACE", ""),
        help="Workspace to migrate (default: WORKSPACE env / the default one)",
    )
    parser.add_argument(
        "--force-empty-target",
        action="store_true",
        help=(
            "DANGEROUS: drop() the whole non-empty target graph slice before "
            "migrating instead of refusing (default: refuse)"
        ),
    )
    parser.add_argument(
        "--source-backend",
        default="PGGraphStorage",
        help="Source graph storage class name (default: PGGraphStorage)",
    )
    parser.add_argument(
        "--target-backend",
        default="PGTableGraphStorage",
        help="Target graph storage class name (default: PGTableGraphStorage)",
    )
    args = parser.parse_args(argv)
    exit_code = asyncio.run(_async_main(args))
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
