"""Offline unit tests for the pure graph-migration helpers.

Covers the Phase 1 (AGE -> PGTable) migration invariants:
canonical ordering parity with PGTableGraphStorage (Python min/max, incl.
non-ASCII), fail-closed detection of divergent reciprocal directed edges,
the parallel-edge backstop, derived written-node sets for compensation, and
source-driven verification.
"""

from typing import Any

import pytest

from lightrag.tools.migrate_graph_storage import (
    DivergentReciprocal,
    VerificationReport,
    canonicalize_edge,
    count_parallel_edges,
    derive_written_node_ids,
    detect_divergent_reciprocals,
    edge_properties,
    payloads_equal,
    sort_directed_edges,
    verify_source_side,
)

pytestmark = pytest.mark.offline


def _edge(source, target, **props):
    """AGE-shaped enumerated edge: properties merged with source/target."""
    return {**props, "source": source, "target": target}


# ---------------------------------------------------------------------------
# canonicalize_edge — must match PGTable's Python min/max exactly
# ---------------------------------------------------------------------------


class TestCanonicalizeEdge:
    def test_orders_ascii(self):
        assert canonicalize_edge("b", "a") == ("a", "b")
        assert canonicalize_edge("a", "b") == ("a", "b")

    def test_matches_python_min_max(self):
        pairs = [
            ("alpha", "beta"),
            ("Zeta", "alpha"),  # case: uppercase sorts before lowercase
            ("node 1", "node-1"),
        ]
        for a, b in pairs:
            assert canonicalize_edge(a, b) == (min(a, b), max(a, b))
            assert canonicalize_edge(b, a) == (min(a, b), max(a, b))

    def test_non_ascii_code_point_order(self):
        # The collation trap: PostgreSQL LEAST/GREATEST under e.g. en_US.utf8
        # would order some of these pairs differently than Python code points.
        # The tool must reproduce Python min/max bit-for-bit.
        assert canonicalize_edge("한국", "일본") == (
            min("한국", "일본"),
            max("한국", "일본"),
        )
        assert canonicalize_edge("Über", "Uber") == ("Uber", "Über")
        assert canonicalize_edge("é", "z") == ("z", "é")  # é (U+00E9) > z (U+007A)

    def test_self_loop(self):
        assert canonicalize_edge("a", "a") == ("a", "a")

    def test_quote_bearing_ids(self):
        assert canonicalize_edge('He said "hi"', "O'Brien") == (
            'He said "hi"',
            "O'Brien",
        )


# ---------------------------------------------------------------------------
# payloads_equal — shared type-strict comparison rule
# ---------------------------------------------------------------------------


class TestPayloadsEqual:
    def test_type_strict_scalars(self):
        # dict == would pass all three of these; the shared rule must not.
        assert not payloads_equal(True, 1)
        assert not payloads_equal(1, 1.0)
        assert not payloads_equal("1", 1)
        assert payloads_equal(1, 1)
        assert payloads_equal(True, True)
        assert payloads_equal(None, None)

    def test_nested_structures(self):
        a = {"meta": {"weight": 1.0, "tags": ["x", "y"]}, "n": 2}
        b = {"n": 2, "meta": {"tags": ["x", "y"], "weight": 1.0}}
        assert payloads_equal(a, b)
        assert not payloads_equal(
            a, {"meta": {"weight": 1, "tags": ["x", "y"]}, "n": 2}
        )
        assert not payloads_equal({"tags": ["x"]}, {"tags": ["x", "y"]})

    def test_nan_never_equal(self):
        # Safe-direction false positive, documented in the module.
        assert not payloads_equal(float("nan"), float("nan"))

    def test_tuple_wrapper_cannot_defeat_type_gate(self):
        # Regression: tuples previously fell through to leaf ==, so a tuple
        # wrapper defeated the whole recursive type gate, nested dicts
        # included. Tuples must recurse like lists.
        assert not payloads_equal(({"x": True},), ({"x": 1},))
        assert not payloads_equal((True,), (1,))
        assert payloads_equal(({"x": 1},), ({"x": 1},))

    def test_tuple_is_not_list(self):
        assert not payloads_equal((1, 2), [1, 2])

    def test_unknown_types_fail_closed(self):
        # Non-JSON types never reach leaf ==: their __eq__ may coerce
        # ({True} == {1} is True), so they compare unequal unconditionally.
        assert not payloads_equal({True}, {1})
        assert not payloads_equal({1}, {1})


# ---------------------------------------------------------------------------
# detect_divergent_reciprocals — AC-c / AC-d
# ---------------------------------------------------------------------------


class TestDetectDivergentReciprocals:
    def test_divergent_reciprocal_detected(self):
        # AC-d: synthesized AGE-shaped directed reciprocal rows with divergent
        # properties. NetworkX cannot express this input (it enumerates each
        # undirected edge once), so the dicts are built directly.
        edges = [
            _edge("a", "b", weight=1.0, description="forward"),
            _edge("b", "a", weight=2.0, description="backward"),
        ]
        result = detect_divergent_reciprocals(edges)
        assert len(result) == 1
        found = result[0]
        assert isinstance(found, DivergentReciprocal)
        assert found.canonical_key == ("a", "b")
        assert found.forward == {"weight": 1.0, "description": "forward"}
        assert found.backward == {"weight": 2.0, "description": "backward"}

    def test_equal_reciprocals_not_flagged(self):
        edges = [
            _edge("a", "b", weight=1.0),
            _edge("b", "a", weight=1.0),
        ]
        assert detect_divergent_reciprocals(edges) == []

    def test_key_order_insensitive_comparison(self):
        # Same payload, different key insertion order: dict == ignores order,
        # so this must NOT be flagged as divergent.
        forward = {"source": "a", "target": "b", "x": 1, "y": 2}
        backward = {"y": 2, "x": 1, "source": "b", "target": "a"}
        assert detect_divergent_reciprocals([forward, backward]) == []

    def test_none_vs_absent_is_divergent(self):
        # Fail-closed rule: a key present with None is observably different
        # from a missing key; no normalisation is applied.
        edges = [
            _edge("a", "b", weight=None),
            _edge("b", "a"),
        ]
        assert len(detect_divergent_reciprocals(edges)) == 1

    def test_single_direction_never_flagged(self):
        assert detect_divergent_reciprocals([_edge("a", "b", w=1)]) == []

    def test_self_loop_never_flagged(self):
        assert detect_divergent_reciprocals([_edge("a", "a", w=1)]) == []

    def test_non_ascii_reciprocals(self):
        edges = [
            _edge("한국", "일본", relation="이웃"),
            _edge("일본", "한국", relation="neighbor"),
        ]
        result = detect_divergent_reciprocals(edges)
        assert len(result) == 1
        assert result[0].canonical_key == canonicalize_edge("한국", "일본")

    def test_bool_vs_int_is_divergent(self):
        edges = [_edge("a", "b", flag=True), _edge("b", "a", flag=1)]
        assert len(detect_divergent_reciprocals(edges)) == 1

    def test_int_vs_float_is_divergent(self):
        edges = [_edge("a", "b", weight=1), _edge("b", "a", weight=1.0)]
        assert len(detect_divergent_reciprocals(edges)) == 1

    def test_str_vs_int_is_divergent(self):
        edges = [_edge("a", "b", weight="1"), _edge("b", "a", weight=1)]
        assert len(detect_divergent_reciprocals(edges)) == 1

    def test_equal_nested_payloads_not_flagged(self):
        edges = [
            _edge("a", "b", meta={"w": 1.0, "tags": ["x"]}, n=2),
            _edge("b", "a", n=2, meta={"tags": ["x"], "w": 1.0}),
        ]
        assert detect_divergent_reciprocals(edges) == []

    def test_results_sorted_and_input_order_independent(self):
        edges = [
            _edge("c", "d", w=1),
            _edge("d", "c", w=2),
            _edge("a", "b", w=1),
            _edge("b", "a", w=2),
        ]
        forward_order = detect_divergent_reciprocals(edges)
        reverse_order = detect_divergent_reciprocals(list(reversed(edges)))
        assert forward_order == reverse_order
        assert [d.canonical_key for d in forward_order] == [("a", "b"), ("c", "d")]


# ---------------------------------------------------------------------------
# count_parallel_edges — backstop for the AGE ordered-pair invariant
# ---------------------------------------------------------------------------


class TestCountParallelEdges:
    def test_backstop_fires_on_parallel_rows(self):
        # Two rows for the SAME ordered pair with different payloads (identical
        # rows would already have collapsed under the AGE query's DISTINCT).
        edges = [
            _edge("a", "b", w=1),
            _edge("a", "b", w=2),
            _edge("x", "y", w=1),
        ]
        counts = count_parallel_edges(edges)
        assert counts[("a", "b")] == 2
        assert counts[("x", "y")] == 1
        violations = {k: c for k, c in counts.items() if c > 1}
        assert violations == {("a", "b"): 2}

    def test_reciprocal_pair_is_not_a_violation(self):
        # a->b plus b->a share a canonical key but are DISTINCT ordered pairs;
        # they belong to detect_divergent_reciprocals, not this backstop.
        edges = [_edge("a", "b", w=1), _edge("b", "a", w=2)]
        counts = count_parallel_edges(edges)
        assert counts == {("a", "b"): 1, ("b", "a"): 1}
        assert not any(c > 1 for c in counts.values())

    def test_parallel_self_loops(self):
        edges = [_edge("a", "a", w=1), _edge("a", "a", w=2)]
        assert count_parallel_edges(edges) == {("a", "a"): 2}


# ---------------------------------------------------------------------------
# derive_written_node_ids — compensation set under the empty-target invariant
# ---------------------------------------------------------------------------


class TestDeriveWrittenNodeIds:
    def test_includes_auto_created_endpoints(self):
        # upsert_edges_batch silently auto-creates missing endpoints; ids never
        # present in the migrated node list must still be in the written set.
        migrated = ["a", "b"]
        edges = [_edge("a", "b"), _edge("b", "ghost"), _edge("phantom", "phantom")]
        written = derive_written_node_ids(migrated, edges)
        assert written == {"a", "b", "ghost", "phantom"}

    def test_union_with_no_edges(self):
        assert derive_written_node_ids(["a"], []) == {"a"}

    def test_empty(self):
        assert derive_written_node_ids([], []) == set()


# ---------------------------------------------------------------------------
# sort_directed_edges — determinism (AC-c)
# ---------------------------------------------------------------------------


class TestSortDirectedEdges:
    def test_deterministic_across_enumeration_orders(self):
        edges = [
            _edge("b", "c", w=2),
            _edge("a", "b", w=1),
            _edge("a", "b", w=0),  # parallel row: tiebreaker keeps order total
            _edge("한국", "일본", w=3),
        ]
        import random

        shuffled = list(edges)
        random.Random(42).shuffle(shuffled)
        assert sort_directed_edges(edges) == sort_directed_edges(shuffled)

    def test_sorted_by_source_then_target(self):
        edges = [_edge("b", "a"), _edge("a", "z"), _edge("a", "b")]
        ordered = sort_directed_edges(edges)
        assert [(e["source"], e["target"]) for e in ordered] == [
            ("a", "b"),
            ("a", "z"),
            ("b", "a"),
        ]


# ---------------------------------------------------------------------------
# verify_source_side — source-driven comparison
# ---------------------------------------------------------------------------


class TestVerifySourceSide:
    def _clean_migration(self):
        # Source: AGE-shaped, one edge enumerated in both directions with
        # identical payloads (legal reciprocal), plus a directed-only edge.
        source_nodes: list[dict[str, Any]] = [
            {"id": "a", "entity_id": "a", "entity_type": "person"},
            {"id": "b", "entity_id": "b", "entity_type": "place"},
            {"id": "c", "entity_id": "c", "entity_type": "thing"},
        ]
        source_edges = [
            _edge("a", "b", weight=1.0),
            _edge("b", "a", weight=1.0),
            _edge("c", "a", weight=2.0),
        ]
        # Target: canonicalized, each undirected edge once.
        target_nodes = [dict(n) for n in source_nodes]
        target_edges = [
            _edge("a", "b", weight=1.0),
            _edge("a", "c", weight=2.0),
        ]
        return source_nodes, source_edges, target_nodes, target_edges

    def test_clean_migration_ok(self):
        report = verify_source_side(*self._clean_migration())
        assert isinstance(report, VerificationReport)
        assert report.ok

    def test_missing_node_and_edge(self):
        source_nodes, source_edges, target_nodes, target_edges = self._clean_migration()
        report = verify_source_side(
            source_nodes, source_edges, target_nodes[:-1], target_edges[:-1]
        )
        assert not report.ok
        assert report.missing_node_ids == ["c"]
        assert report.missing_edges == [("a", "c")]

    def test_unexpected_target_rows(self):
        source_nodes, source_edges, target_nodes, target_edges = self._clean_migration()
        target_nodes = target_nodes + [{"id": "stray", "entity_id": "stray"}]
        target_edges = target_edges + [_edge("a", "stray", w=9)]
        report = verify_source_side(
            source_nodes, source_edges, target_nodes, target_edges
        )
        assert not report.ok
        assert report.unexpected_node_ids == ["stray"]
        assert report.unexpected_edges == [("a", "stray")]

    def test_edge_property_mismatch(self):
        source_nodes, source_edges, target_nodes, target_edges = self._clean_migration()
        target_edges[0] = _edge("a", "b", weight=99.0)
        report = verify_source_side(
            source_nodes, source_edges, target_nodes, target_edges
        )
        assert not report.ok
        assert len(report.edge_property_mismatches) == 1
        key, src_props, tgt_props = report.edge_property_mismatches[0]
        assert key == ("a", "b")
        assert src_props == {"weight": 1.0}
        assert tgt_props == {"weight": 99.0}

    def test_node_property_mismatch(self):
        source_nodes, source_edges, target_nodes, target_edges = self._clean_migration()
        target_nodes[0] = {"id": "a", "entity_id": "a", "entity_type": "MUTATED"}
        report = verify_source_side(
            source_nodes, source_edges, target_nodes, target_edges
        )
        assert not report.ok
        assert report.node_property_mismatches == [
            (
                "a",
                {"entity_id": "a", "entity_type": "person"},
                {"entity_id": "a", "entity_type": "MUTATED"},
            )
        ]

    def test_parallel_source_rows_fail_verification_in_both_orders(self):
        # Regression: parallel source rows [a->b w=1, a->b w=2] against a
        # target holding only w=2 previously yielded ok=True in one
        # enumeration order and ok=False in the other. Verify must run the
        # backstop itself and fail closed in BOTH orders.
        source_nodes = [
            {"id": "a", "entity_id": "a"},
            {"id": "b", "entity_id": "b"},
        ]
        target_nodes = [dict(n) for n in source_nodes]
        target_edges = [_edge("a", "b", weight=2)]
        parallel = [_edge("a", "b", weight=1), _edge("a", "b", weight=2)]
        for source_edges in (parallel, list(reversed(parallel))):
            report = verify_source_side(
                source_nodes, source_edges, target_nodes, target_edges
            )
            assert not report.ok
            assert report.source_parallel_violations == {("a", "b"): 2}

    def test_report_deterministic_across_enumeration_orders(self):
        # The full report (every field, not just ok) must be independent of
        # the source enumeration order, including on invariant-violating
        # input where last-write-wins map construction could otherwise vary.
        source_nodes = [
            {"id": "a", "entity_id": "a"},
            {"id": "b", "entity_id": "b"},
            {"id": "c", "entity_id": "c"},
        ]
        source_edges = [
            _edge("a", "b", weight=1),
            _edge("a", "b", weight=2),  # parallel violation
            _edge("b", "c", weight=3),
            _edge("c", "b", weight=4),  # divergent reciprocal
        ]
        target_nodes = [dict(n) for n in source_nodes]
        target_edges = [_edge("a", "b", weight=2), _edge("b", "c", weight=3)]
        report_fwd = verify_source_side(
            source_nodes, source_edges, target_nodes, target_edges
        )
        report_rev = verify_source_side(
            list(reversed(source_nodes)),
            list(reversed(source_edges)),
            list(reversed(target_nodes)),
            list(reversed(target_edges)),
        )
        assert report_fwd == report_rev
        assert not report_fwd.ok

    def test_type_strict_node_property_mismatch(self):
        source_nodes, source_edges, target_nodes, target_edges = self._clean_migration()
        source_nodes[0]["flag"] = True
        target_nodes[0]["flag"] = 1  # dict == would coerce; must be flagged
        report = verify_source_side(
            source_nodes, source_edges, target_nodes, target_edges
        )
        assert not report.ok
        assert [m[0] for m in report.node_property_mismatches] == ["a"]

    def test_type_strict_edge_property_mismatch(self):
        source_nodes, source_edges, target_nodes, target_edges = self._clean_migration()
        target_edges[1] = _edge("a", "c", weight=2)  # source has 2.0
        report = verify_source_side(
            source_nodes, source_edges, target_nodes, target_edges
        )
        assert not report.ok
        assert [m[0] for m in report.edge_property_mismatches] == [("a", "c")]

    def test_source_reciprocal_divergence_reported(self):
        source_nodes, source_edges, target_nodes, target_edges = self._clean_migration()
        source_edges[1] = _edge("b", "a", weight=555.0)  # diverge the reciprocal
        report = verify_source_side(
            source_nodes, source_edges, target_nodes, target_edges
        )
        assert not report.ok
        assert len(report.source_reciprocal_divergence) == 1
        assert report.source_reciprocal_divergence[0].canonical_key == ("a", "b")


# ---------------------------------------------------------------------------
# End-to-end determinism of the pure pipeline (AC-c)
# ---------------------------------------------------------------------------


class TestDeterministicPipeline:
    def test_non_divergent_graph_identical_across_two_runs(self):
        # AC-c second half: with no divergent reciprocals (divergent input
        # fails closed instead of proceeding), two runs over differently
        # enumerated source reads produce identical canonical write sets.
        run_a = [
            _edge("b", "a", weight=1.0),
            _edge("a", "b", weight=1.0),
            _edge('He said "hi"', "O'Brien", rel="quote"),
            _edge("한국", "일본", rel="이웃"),
            _edge("loop", "loop", rel="self"),
        ]
        run_b = list(reversed(run_a))

        def pipeline(directed_edges):
            assert detect_divergent_reciprocals(directed_edges) == []
            assert not any(c > 1 for c in count_parallel_edges(directed_edges).values())
            ordered = sort_directed_edges(directed_edges)
            return [
                (
                    *canonicalize_edge(e["source"], e["target"]),
                    tuple(sorted(edge_properties(e).items())),
                )
                for e in ordered
            ]

        assert pipeline(run_a) == pipeline(run_b)

    def test_edge_properties_strips_only_endpoints(self):
        e = _edge("a", "b", source_id="chunk-1", weight=1)
        assert edge_properties(e) == {"source_id": "chunk-1", "weight": 1}
