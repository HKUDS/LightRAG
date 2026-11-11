from contextlib import asynccontextmanager
import json
import re
from types import SimpleNamespace

import pytest

import lightrag.kg.hologres.capabilities as hologres_capabilities
from lightrag.kg.hologres.capabilities import (
    CapabilityReport,
    HologresProbeError,
    HologresVersion,
    HologresVersionError,
    ProbeKind,
    ProbeResult,
    ProbeStatus,
    _probe_setup_reset_bindings,
    parse_hologres_version,
    prove_similarity_orientation,
    prove_stream_copy_capability,
    run_initial_isolated_probes,
    validate_hologres_version,
    validate_test_schema_name,
)
from lightrag.kg.hologres.client import HologresCapabilityError, HologresClient
from lightrag.kg.hologres.config import HologresConfig
from lightrag.kg.hologres.schema import _JSONB_COLUMNAR_POSTCONDITION


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Hologres 5.0.0", HologresVersion(5, 0, 0)),
        ("Hologres 5.1.2 (revision abc123)", HologresVersion(5, 1, 2)),
        ("PostgreSQL compatible; Hologres 6.0", HologresVersion(6, 0, 0)),
    ],
)
def test_version_parser_recognizes_hologres_versions(raw, expected):
    assert parse_hologres_version(raw) == expected
    assert validate_hologres_version(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "Hologres 4.9.9",
        "PostgreSQL 14.2",
        "5.0.0",
        "Hologres release-next",
        "",
    ],
)
def test_version_validation_fails_closed_for_old_or_unrecognized_servers(raw):
    with pytest.raises(HologresVersionError):
        validate_hologres_version(raw)


def test_capability_report_requires_a_passed_probe_to_enable_stream_copy():
    report = CapabilityReport(
        version=HologresVersion(5, 0, 0),
        results=(
            ProbeResult(
                kind=ProbeKind.STREAM_COPY,
                status=ProbeStatus.PASSED,
                blocking=False,
                detail_code="copy_round_trip_ok",
            ),
        ),
    )
    not_proven = CapabilityReport(
        version=HologresVersion(5, 0, 0),
        results=(
            ProbeResult(
                kind=ProbeKind.STREAM_COPY,
                status=ProbeStatus.NOT_RUN,
                blocking=False,
                detail_code="not_requested",
            ),
        ),
    )

    assert report.supports(ProbeKind.STREAM_COPY) is True
    assert not_proven.supports(ProbeKind.STREAM_COPY) is False


def test_capability_report_rejects_duplicate_probe_kinds():
    passed = ProbeResult(
        kind=ProbeKind.STREAM_COPY,
        status=ProbeStatus.PASSED,
        blocking=False,
        detail_code="copy_round_trip_ok",
    )
    failed = ProbeResult(
        kind=ProbeKind.STREAM_COPY,
        status=ProbeStatus.FAILED,
        blocking=False,
        detail_code="copy_round_trip_failed",
    )

    with pytest.raises(ValueError, match="duplicate"):
        CapabilityReport(
            version=HologresVersion(5, 0, 0), results=(passed, failed)
        )


@pytest.mark.parametrize(
    ("kind", "incorrect_blocking"),
    [(ProbeKind.HGRAPH, False), (ProbeKind.STREAM_COPY, True)],
)
def test_capability_report_rejects_incorrect_blocking_metadata(
    kind, incorrect_blocking
):
    result = ProbeResult(
        kind=kind,
        status=ProbeStatus.PASSED,
        blocking=incorrect_blocking,
        detail_code="probe_completed",
    )

    with pytest.raises(ValueError, match="blocking"):
        CapabilityReport(version=HologresVersion(5, 0, 0), results=(result,))


def test_probe_catalog_defines_all_later_blocking_and_optional_checks():
    assert set(ProbeKind) == {
        ProbeKind.SINGLE_AUTOCOMMIT_DDL,
        ProbeKind.ASYNCPG_SETUP_RESET_BINDINGS,
        ProbeKind.JSONB_ON_CONFLICT_ARRAYS_RECONNECT,
        ProbeKind.LOGICAL_PARTITION,
        ProbeKind.GRAPH_ADJACENCY_EXPLAIN,
        ProbeKind.HGRAPH,
        ProbeKind.STREAM_COPY,
        ProbeKind.JSONB_COLUMN_OPTIMIZATION,
        ProbeKind.AGE,
    }


def test_version_only_report_marks_every_blocking_probe_not_run():
    report = CapabilityReport(version=HologresVersion(5, 0, 0))

    failures = report.blocking_failures

    assert {result.kind for result in failures} == {
        ProbeKind.SINGLE_AUTOCOMMIT_DDL,
        ProbeKind.ASYNCPG_SETUP_RESET_BINDINGS,
        ProbeKind.JSONB_ON_CONFLICT_ARRAYS_RECONNECT,
        ProbeKind.LOGICAL_PARTITION,
        ProbeKind.GRAPH_ADJACENCY_EXPLAIN,
        ProbeKind.HGRAPH,
    }
    assert all(result.status is ProbeStatus.NOT_RUN for result in failures)
    assert all(result.detail_code == "probe_not_run" for result in failures)


def test_missing_blocking_probes_fail_closed_even_when_reported_probes_pass():
    report = CapabilityReport(
        version=HologresVersion(5, 0, 0),
        results=(
            ProbeResult(
                kind=ProbeKind.SINGLE_AUTOCOMMIT_DDL,
                status=ProbeStatus.PASSED,
                blocking=True,
                detail_code="autocommit_ddl_ok",
            ),
        ),
    )

    assert ProbeKind.SINGLE_AUTOCOMMIT_DDL not in {
        result.kind for result in report.blocking_failures
    }
    assert len(report.blocking_failures) == 5


@pytest.mark.parametrize(
    "schema",
    ["lightrag_test_a1b2c3", "lightrag_test_worker_2", "lightrag_test_abc_123"],
)
def test_live_probe_schema_guard_accepts_only_isolated_prefixed_schemas(schema):
    assert validate_test_schema_name(schema) == schema


@pytest.mark.parametrize(
    "schema",
    ["public", "lightrag", "lightrag_test_", "LightRAG_test_abc", "lightrag_test_x;drop"],
)
def test_live_probe_schema_guard_rejects_nonisolated_or_unsafe_schemas(schema):
    with pytest.raises(ValueError):
        validate_test_schema_name(schema)


def test_probe_results_reject_free_form_details_that_could_capture_secrets():
    with pytest.raises(ValueError):
        ProbeResult(
            kind=ProbeKind.HGRAPH,
            status=ProbeStatus.FAILED,
            blocking=True,
            detail_code="failed against password=secret",
        )


def test_probe_results_reject_unstructured_hgraph_evidence():
    with pytest.raises(ValueError, match="evidence"):
        ProbeResult(
            kind=ProbeKind.HGRAPH,
            status=ProbeStatus.FAILED,
            blocking=True,
            detail_code="hgraph_score_contract_mismatch",
            evidence="password=secret",
        )


@pytest.mark.parametrize(
    ("ordered_raw_scores", "vector_filter_used"),
    [
        (((True, 1.0),), True),
        (((1, "password=secret"),), True),
        (((1, float("nan")),), True),
        (((1, 1.0),), "yes"),
    ],
)
def test_hgraph_probe_evidence_rejects_untyped_or_nonfinite_values(
    ordered_raw_scores, vector_filter_used
):
    with pytest.raises(ValueError, match="evidence"):
        hologres_capabilities.HGraphProbeEvidence(
            ordered_raw_scores=ordered_raw_scores,
            vector_filter_used=vector_filter_used,
        )


class OrientationClient:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    async def fetch_value(self, sql, *values, descriptor):
        self.calls.append((descriptor, values))
        if self.error is not None:
            raise self.error
        return self.result


@pytest.mark.asyncio
async def test_orientation_probe_passes_when_similarity_outranks_distance():
    client = OrientationClient(result=2.0)

    await prove_similarity_orientation(client)

    (descriptor, values) = client.calls[0]
    assert descriptor == "probe.similarity.orientation"
    # Same vector as the query scores higher than the exact opposite, which
    # only holds when the function returns similarity rather than distance.
    assert values == (
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
    )


@pytest.mark.parametrize(
    "result",
    [0.0, -1.5, "0.5", None, True],
)
@pytest.mark.asyncio
async def test_orientation_probe_fails_closed_on_distance_or_unknown_semantics(
    result,
):
    client = OrientationClient(result=result)

    with pytest.raises(HologresCapabilityError, match="similarity"):
        await prove_similarity_orientation(client)


@pytest.mark.asyncio
async def test_orientation_probe_fails_closed_when_the_query_itself_fails():
    client = OrientationClient(error=RuntimeError("password=probe-secret"))

    with pytest.raises(HologresCapabilityError, match="orientation"):
        await prove_similarity_orientation(client)



class ExistingSchemaClient:
    def __init__(self):
        self.executed = []

    async def fetch_value(self, sql, *values, descriptor):
        if descriptor == "capability.version":
            return "Hologres 5.0.0"
        return True

    async def execute_one(self, sql, *values, descriptor, replay_safe=False):
        self.executed.append(sql)
        return "OK"


@pytest.mark.asyncio
async def test_isolated_probes_refuse_to_touch_a_preexisting_test_schema():
    client = ExistingSchemaClient()

    with pytest.raises(HologresProbeError):
        await run_initial_isolated_probes(client, "lightrag_test_existing")

    assert client.executed == []


class CreateFailureClient:
    def __init__(self):
        self.calls = []

    async def fetch_value(self, sql, *values, descriptor):
        self.calls.append(descriptor)
        if descriptor == "capability.version":
            return "Hologres 5.0.0"
        if descriptor == "probe.schema.preflight":
            return False
        raise AssertionError(f"Unexpected fetch: {descriptor}")

    async def execute_one(self, sql, *values, descriptor, replay_safe=False):
        self.calls.append(descriptor)
        if descriptor == "probe.schema.create":
            self.create_sql = sql
            raise RuntimeError("unknown create outcome")
        raise AssertionError(f"Unsafe call after CREATE failure: {descriptor}")


@pytest.mark.asyncio
async def test_create_failure_never_drops_or_continues_and_reports_possible_leak():
    client = CreateFailureClient()

    with pytest.raises(HologresProbeError) as exc_info:
        await run_initial_isolated_probes(client, "lightrag_test_create_race")

    assert client.calls == [
        "capability.version",
        "probe.schema.preflight",
        "probe.schema.create",
    ]
    assert "probe.schema.drop" not in client.calls
    assert exc_info.value.leaked_objects == ("lightrag_test_create_race",)


@pytest.mark.asyncio
async def test_probe_harness_generates_an_unpredictable_schema_by_default():
    client = CreateFailureClient()

    with pytest.raises(HologresProbeError) as exc_info:
        await run_initial_isolated_probes(client)

    generated_schema = exc_info.value.leaked_objects[0]
    assert generated_schema.startswith("lightrag_test_")
    assert len(generated_schema) >= len("lightrag_test_") + 24
    assert generated_schema in client.create_sql


_MATCH_OWNER = object()


class OwnedProbeClient:
    def __init__(self, *, ownership_results=(), fail_schema_drop=False):
        self.calls = []
        self.writes = []
        self.ownership_results = list(ownership_results)
        self.fail_schema_drop = fail_schema_drop
        self.owner_token = None
        self.schema = None
        self.reconnect_count = 0

    async def fetch_value(self, sql, *values, descriptor):
        self.calls.append(descriptor)
        if descriptor == "capability.version":
            return "Hologres 5.0.0"
        if descriptor == "probe.schema.preflight":
            self.schema = values[0]
            return False
        if descriptor == "probe.marker.verify":
            if self.ownership_results:
                result = self.ownership_results.pop(0)
                return self.owner_token if result is _MATCH_OWNER else result
            return self.owner_token
        if descriptor == "probe.reconnect.verify":
            return 1
        raise AssertionError(f"Unexpected fetch: {descriptor}")

    async def fetch_one(self, sql, *values, descriptor):
        self.calls.append(descriptor)
        return {"payload": {"step": 2}, "tags": ["beta", "gamma"]}

    async def execute_one(self, sql, *values, descriptor, replay_safe=False):
        self.calls.append(descriptor)
        self.writes.append((descriptor, sql, values, replay_safe))
        if descriptor == "probe.marker.insert":
            self.owner_token = values[0]
        if descriptor == "probe.schema.drop" and self.fail_schema_drop:
            raise RuntimeError("schema contains an unknown competing object")
        return "OK"

    async def _observe_setup_reset_for_probe(self, *, bound, marker):
        self.calls.append("probe.setup_reset")
        return SimpleNamespace(
            schema_name=self.schema,
            bound=bound,
            same_connection_reacquired=True,
            application_name="",
        )

    async def _reconnect_for_probe(self):
        self.calls.append("probe.reconnect")
        self.reconnect_count += 1


@pytest.mark.asyncio
async def test_marker_mismatch_never_runs_probes_or_cleanup():
    client = OwnedProbeClient(ownership_results=("wrong-owner-token",))

    with pytest.raises(HologresProbeError) as exc_info:
        await run_initial_isolated_probes(client, "lightrag_test_marker_mismatch")

    assert "probe.setup_reset" not in client.calls
    assert "probe.basic.create" not in client.calls
    assert not any(descriptor.endswith(".drop") for descriptor in client.calls)
    assert exc_info.value.leaked_objects == ("lightrag_test_marker_mismatch",)


@pytest.mark.asyncio
async def test_replaced_schema_stops_before_more_probes_or_cleanup():
    client = OwnedProbeClient(
        ownership_results=(_MATCH_OWNER, None, None),
    )

    with pytest.raises(HologresProbeError) as exc_info:
        await run_initial_isolated_probes(client, "lightrag_test_replaced_schema")

    assert client.calls.count("probe.setup_reset") == 1
    assert "probe.basic.create" not in client.calls
    assert not any(descriptor.endswith(".drop") for descriptor in client.calls)
    assert exc_info.value.leaked_objects == ("lightrag_test_replaced_schema",)


@pytest.mark.asyncio
async def test_unknown_objects_prevent_non_cascade_schema_cleanup_and_report_leak():
    schema = "lightrag_test_unknown_object"
    client = OwnedProbeClient(fail_schema_drop=True)

    with pytest.raises(HologresProbeError) as exc_info:
        await run_initial_isolated_probes(client, schema)

    writes_by_descriptor = {
        descriptor: (sql, values, replay_safe)
        for descriptor, sql, values, replay_safe in client.writes
    }
    assert "probe.basic.drop" in writes_by_descriptor
    assert "probe.marker.drop" in writes_by_descriptor
    assert "probe.schema.drop" in writes_by_descriptor
    assert all("CASCADE" not in sql.upper() for sql, _values, _safe in writes_by_descriptor.values())
    assert writes_by_descriptor["probe.basic.drop"][0].startswith(
        f'DROP TABLE IF EXISTS "{schema}"."lightrag_test_basic_'
    )
    assert writes_by_descriptor["probe.marker.drop"][0].startswith(
        f'DROP TABLE IF EXISTS "{schema}"."lightrag_test_owner_'
    )
    assert writes_by_descriptor["probe.schema.drop"][0] == f'DROP SCHEMA "{schema}"'
    assert exc_info.value.leaked_objects == (schema,)


@pytest.mark.asyncio
async def test_probe_ownership_marker_uses_unpredictable_name_and_token():
    client = OwnedProbeClient()

    await run_initial_isolated_probes(client, "lightrag_test_high_entropy")

    marker_create = next(
        sql
        for descriptor, sql, _values, _safe in client.writes
        if descriptor == "probe.marker.create"
    )
    marker_insert = next(
        write for write in client.writes if write[0] == "probe.marker.insert"
    )
    assert re.search(r'lightrag_test_owner_[0-9a-f]{32}"', marker_create)
    assert re.fullmatch(r"[0-9a-f]{32}", marker_insert[2][0])
    assert client.calls.count("probe.marker.verify") >= 3


class ResetProbeConnection:
    def __init__(self, backend_pid, schema):
        self.backend_pid = backend_pid
        self.schema = schema
        self.application_name = ""
        self.sql = []

    async def fetchrow(self, sql, *args, timeout=None):
        self.sql.append(sql)
        if "set_config" in sql:
            self.application_name = args[-1]
        if "current_schema" in sql:
            return {
                "schema_name": self.schema,
                "bound": args[0],
                "backend_pid": self.backend_pid,
            }
        return {
            "backend_pid": self.backend_pid,
            "application_name": self.application_name,
        }

    async def execute(self, sql, *args, timeout=None):
        self.sql.append(sql)
        if "application_name" in sql:
            self.application_name = "lightrag_test_probe"
        return "OK"

    async def fetchval(self, sql, *args, timeout=None):
        self.sql.append(sql)
        return self.application_name


class SequencedResetProbePool:
    def __init__(self, connections):
        self.connections = connections
        self.acquire_count = 0

    @asynccontextmanager
    async def acquire(self, *, timeout=None):
        index = min(self.acquire_count, len(self.connections) - 1)
        self.acquire_count += 1
        yield self.connections[index]


@pytest.mark.asyncio
async def test_reset_probe_cannot_pass_by_verifying_a_different_physical_connection():
    schema = "lightrag_test_reset_pid"
    seeded = ResetProbeConnection(101, schema)
    different = ResetProbeConnection(202, schema)
    pool = SequencedResetProbePool([seeded, seeded, different])
    client = HologresClient(
        HologresConfig(
            host="example.hologres.aliyuncs.com",
            port=80,
            user="test_user",
            password="secret",
            database="analytics",
            schema=schema,
            pool_max_size=2,
        ),
        pool=pool,
    )

    result = await _probe_setup_reset_bindings(client, schema)

    assert result.status is ProbeStatus.FAILED
    assert result.detail_code == "reset_not_observed"
    executed_sql = seeded.sql + different.sql
    assert any("set_config" in sql for sql in executed_sql)
    assert any("pg_backend_pid" in sql for sql in executed_sql)


class HGraphProbeClient:
    def __init__(self):
        self.events = []
        self.statements = {}

    async def execute_one(self, sql, *values, descriptor, replay_safe=False):
        self.events.append(descriptor)
        self.statements[descriptor] = (sql, values, replay_safe)
        return "OK"

    async def fetch_value(
        self,
        sql,
        *values,
        descriptor,
        operation_kind=None,
        replay_safe=None,
    ):
        self.events.append(descriptor)
        self.statements[descriptor] = (sql, values, replay_safe)
        if descriptor == "probe.hgraph.compact":
            assert operation_kind is hologres_capabilities.OperationKind.WRITE
            return "OK"
        if descriptor == "probe.hgraph.catalog":
            return (
                '{"embedding":{"algorithm":"HGraph",'
                '"distance_method":"Cosine"}}'
            )
        if descriptor == "probe.hgraph.delete.verify":
            return 0
        raise AssertionError(f"Unexpected value fetch: {descriptor}")

    async def fetch_all(self, sql, *values, descriptor):
        self.events.append(descriptor)
        self.statements[descriptor] = (sql, values, None)
        if descriptor == "probe.hgraph.query":
            return [
                {"id": 1, "raw_score": 1.0},
                {"id": 4, "raw_score": 1.0},
                {"id": 2, "raw_score": 0.0},
                {"id": 3, "raw_score": -1.0},
            ]
        if descriptor == "probe.hgraph.explain":
            return [{"QUERY PLAN": "Vector Filter: VectorCond => KNN"}]
        raise AssertionError(f"Unexpected row fetch: {descriptor}")

    async def fetch_one(self, sql, *values, descriptor):
        self.events.append(descriptor)
        self.statements[descriptor] = (sql, values, None)
        if descriptor == "probe.hgraph.update.verify":
            return {"embedding": [0.0, 0.0, 1.0]}
        raise AssertionError(f"Unexpected row fetch: {descriptor}")


@pytest.mark.asyncio
async def test_hgraph_probe_executes_disposable_index_crud_and_explain_contract():
    schema = "lightrag_test_hgraph_probe"
    table = "lightrag_test_hgraph_vectors"
    client = HGraphProbeClient()

    async def verify_ownership():
        client.events.append("probe.marker.verify")

    probe = getattr(hologres_capabilities, "_probe_hgraph")
    result = await probe(client, schema, table, verify_ownership)

    assert result.kind is ProbeKind.HGRAPH
    assert result.status is ProbeStatus.PASSED
    assert result.blocking is True
    assert result.detail_code == "hgraph_semantics_frozen"
    assert result.evidence.ordered_raw_scores == (
        (1, 1.0),
        (4, 1.0),
        (2, 0.0),
        (3, -1.0),
    )
    assert result.evidence.vector_filter_used is True
    assert client.events == [
        "probe.marker.verify",
        "probe.hgraph.create",
        "probe.marker.verify",
        "probe.hgraph.insert",
        "probe.marker.verify",
        "probe.hgraph.compact",
        "probe.marker.verify",
        "probe.hgraph.catalog",
        "probe.marker.verify",
        "probe.hgraph.query",
        "probe.marker.verify",
        "probe.hgraph.explain",
        "probe.marker.verify",
        "probe.hgraph.update",
        "probe.marker.verify",
        "probe.hgraph.update.verify",
        "probe.marker.verify",
        "probe.hgraph.delete",
        "probe.marker.verify",
        "probe.hgraph.delete.verify",
    ]

    create_sql, create_values, create_replay_safe = client.statements[
        "probe.hgraph.create"
    ]
    assert create_values == ()
    assert create_replay_safe is False
    assert "embedding float4[]" in create_sql
    assert "array_length(embedding, 1) = 3" in create_sql
    assert "vectors =" in create_sql
    assert '"algorithm":"HGraph"' in create_sql
    assert '"distance_method":"Cosine"' in create_sql
    assert "extra_columns" not in create_sql

    compact_sql, compact_values, _ = client.statements["probe.hgraph.compact"]
    assert compact_sql == "SELECT hologres.hg_full_compact_table($1, $2)"
    assert compact_values == (
        f"{schema}.{table}",
        "max_file_size_mb=4096",
    )

    catalog_sql, catalog_values, _ = client.statements["probe.hgraph.catalog"]
    assert "FROM hologres.hg_table_properties" in catalog_sql
    assert catalog_values == (schema, table, "vectors")

    query_sql, query_values, _ = client.statements["probe.hgraph.query"]
    assert "approx_cosine_distance(embedding, $1::float4[])" in query_sql
    assert "ORDER BY raw_score DESC, id ASC" in query_sql
    assert query_values == ([1.0, 0.0, 0.0],)

    explain_sql, explain_values, _ = client.statements["probe.hgraph.explain"]
    assert explain_sql.startswith("EXPLAIN SELECT")
    assert "approx_cosine_distance(embedding, $1::float4[])" in explain_sql
    assert explain_values == ([1.0, 0.0, 0.0],)

    for sql, _values, _replay_safe in client.statements.values():
        assert ";" not in sql


class ObservedFailureHGraphProbeClient(HGraphProbeClient):
    def __init__(self, stage):
        super().__init__()
        self.stage = stage

    async def fetch_all(self, sql, *values, descriptor):
        if descriptor == "probe.hgraph.query" and self.stage == "query":
            return [
                {"id": 3, "raw_score": 3.0},
                {"id": 2, "raw_score": 2.0},
                {"id": 1, "raw_score": 1.0},
                {"id": 4, "raw_score": 1.0},
            ]
        if descriptor == "probe.hgraph.query" and self.stage == "scores":
            return [
                {"id": 1, "raw_score": 0.5},
                {"id": 4, "raw_score": 0.5},
                {"id": 2, "raw_score": 0.0},
                {"id": 3, "raw_score": -0.5},
            ]
        if descriptor == "probe.hgraph.explain" and self.stage == "explain":
            return [{"QUERY PLAN": "Seq Scan on lightrag_test_hgraph_vectors"}]
        return await super().fetch_all(sql, *values, descriptor=descriptor)


@pytest.mark.asyncio
async def test_hgraph_probe_preserves_scores_when_query_semantics_mismatch():
    client = ObservedFailureHGraphProbeClient("query")

    async def verify_ownership():
        client.events.append("probe.marker.verify")

    result = await hologres_capabilities._probe_hgraph(
        client,
        "lightrag_test_hgraph_probe",
        "lightrag_test_hgraph_vectors",
        verify_ownership,
    )

    assert result.status is ProbeStatus.FAILED
    assert result.detail_code == "hgraph_query_mismatch"
    assert result.evidence == hologres_capabilities.HGraphProbeEvidence(
        ordered_raw_scores=(
            (3, 3.0),
            (2, 2.0),
            (1, 1.0),
            (4, 1.0),
        ),
        vector_filter_used=None,
    )


@pytest.mark.asyncio
async def test_hgraph_probe_fails_when_ordered_scores_break_the_frozen_contract():
    client = ObservedFailureHGraphProbeClient("scores")

    async def verify_ownership():
        client.events.append("probe.marker.verify")

    result = await hologres_capabilities._probe_hgraph(
        client,
        "lightrag_test_hgraph_probe",
        "lightrag_test_hgraph_vectors",
        verify_ownership,
    )

    assert result.status is ProbeStatus.FAILED
    assert result.detail_code == "hgraph_score_contract_mismatch"
    assert result.evidence == hologres_capabilities.HGraphProbeEvidence(
        ordered_raw_scores=(
            (1, 0.5),
            (4, 0.5),
            (2, 0.0),
            (3, -0.5),
        ),
        vector_filter_used=None,
    )
    assert "probe.hgraph.explain" not in client.events


@pytest.mark.asyncio
async def test_hgraph_probe_preserves_scores_when_vector_filter_is_not_used():
    client = ObservedFailureHGraphProbeClient("explain")

    async def verify_ownership():
        client.events.append("probe.marker.verify")

    result = await hologres_capabilities._probe_hgraph(
        client,
        "lightrag_test_hgraph_probe",
        "lightrag_test_hgraph_vectors",
        verify_ownership,
    )

    assert result.status is ProbeStatus.PASSED
    assert result.detail_code == "hgraph_semantics_frozen"
    assert result.evidence == hologres_capabilities.HGraphProbeEvidence(
        ordered_raw_scores=(
            (1, 1.0),
            (4, 1.0),
            (2, 0.0),
            (3, -1.0),
        ),
        vector_filter_used=False,
    )


class MalformedHGraphProbeClient(HGraphProbeClient):
    def __init__(self, stage):
        super().__init__()
        self.stage = stage

    async def fetch_value(
        self,
        sql,
        *values,
        descriptor,
        operation_kind=None,
        replay_safe=None,
    ):
        if descriptor == "probe.hgraph.catalog" and self.stage == "catalog":
            return "{"
        if descriptor == "probe.hgraph.delete.verify" and self.stage == "delete":
            return True
        return await super().fetch_value(
            sql,
            *values,
            descriptor=descriptor,
            operation_kind=operation_kind,
            replay_safe=replay_safe,
        )

    async def fetch_all(self, sql, *values, descriptor):
        if descriptor == "probe.hgraph.query" and self.stage == "query":
            return [
                {"id": 1, "raw_score": 1.0},
                {"id": 4},
                {"id": 2, "raw_score": 0.0},
                {"id": 3, "raw_score": -1.0},
            ]
        if descriptor == "probe.hgraph.explain" and self.stage == "explain":
            return None
        return await super().fetch_all(sql, *values, descriptor=descriptor)

    async def fetch_one(self, sql, *values, descriptor):
        if descriptor == "probe.hgraph.update.verify" and self.stage == "update":
            return {"embedding": None}
        return await super().fetch_one(sql, *values, descriptor=descriptor)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stage", "detail_code"),
    [
        ("catalog", "hgraph_catalog_mismatch"),
        ("query", "hgraph_query_mismatch"),
        ("update", "hgraph_update_mismatch"),
        ("delete", "hgraph_delete_mismatch"),
    ],
)
async def test_hgraph_probe_classifies_malformed_stage_results(
    stage, detail_code
):
    client = MalformedHGraphProbeClient(stage)

    async def verify_ownership():
        client.events.append("probe.marker.verify")

    result = await hologres_capabilities._probe_hgraph(
        client,
        "lightrag_test_hgraph_probe",
        "lightrag_test_hgraph_vectors",
        verify_ownership,
    )

    assert result == ProbeResult(
        kind=ProbeKind.HGRAPH,
        status=ProbeStatus.FAILED,
        blocking=True,
        detail_code=detail_code,
    )


class GraphProbeClient:
    def __init__(self):
        self.events = []
        self.statements = {}

    async def execute_one(self, sql, *values, descriptor, replay_safe=False):
        self.events.append(descriptor)
        self.statements[descriptor] = (sql, values, replay_safe)
        return "OK"

    async def fetch_value(self, sql, *values, descriptor):
        self.events.append(descriptor)
        self.statements[descriptor] = (sql, values, None)
        if descriptor == "probe.graph.collation":
            return "C"
        if descriptor == "probe.graph.orientation":
            return "row,column"
        if descriptor == "probe.graph.nodes.merge.verify":
            return '{"keep":"x","step":2}'
        if descriptor == "probe.graph.degree":
            return 3
        if descriptor == "probe.graph.pairs":
            return 2
        raise AssertionError(f"Unexpected value fetch: {descriptor}")

    async def fetch_all(self, sql, *values, descriptor):
        self.events.append(descriptor)
        self.statements[descriptor] = (sql, values, None)
        if descriptor == "probe.graph.nodes.order":
            return [{"id": "B"}, {"id": "_x"}, {"id": "a"}]
        if descriptor == "probe.graph.adjacency":
            return [{"nid": "B"}, {"nid": "_x"}]
        if descriptor == "probe.graph.explain":
            return [
                {"QUERY PLAN": "Seq Scan on lightrag_test_gedges"},
                {"QUERY PLAN": "Partition Filter: (workspace = 'w1')"},
            ]
        raise AssertionError(f"Unexpected row fetch: {descriptor}")


async def _run_graph_probe(client):
    async def verify_ownership():
        client.events.append("probe.marker.verify")

    return await hologres_capabilities._probe_graph_partition_adjacency(
        client,
        "lightrag_test_graph_probe",
        "lightrag_test_gnodes",
        "lightrag_test_gedges",
        verify_ownership,
    )


@pytest.mark.asyncio
async def test_graph_probe_freezes_partition_and_adjacency_contracts():
    client = GraphProbeClient()

    partition, adjacency = await _run_graph_probe(client)

    assert partition == ProbeResult(
        kind=ProbeKind.LOGICAL_PARTITION,
        status=ProbeStatus.PASSED,
        blocking=True,
        detail_code="logical_partition_semantics_frozen",
    )
    assert adjacency == ProbeResult(
        kind=ProbeKind.GRAPH_ADJACENCY_EXPLAIN,
        status=ProbeStatus.PASSED,
        blocking=True,
        detail_code="graph_adjacency_plan_partition_pruned",
    )
    assert client.events == [
        "probe.graph.collation",
        "probe.marker.verify",
        "probe.graph.nodes.create",
        "probe.marker.verify",
        "probe.graph.orientation",
        "probe.marker.verify",
        "probe.graph.edges.create",
        "probe.marker.verify",
        "probe.graph.nodes.insert",
        "probe.marker.verify",
        "probe.graph.nodes.merge",
        "probe.marker.verify",
        "probe.graph.nodes.merge.verify",
        "probe.marker.verify",
        "probe.graph.nodes.order",
        "probe.marker.verify",
        "probe.graph.edges.insert",
        "probe.marker.verify",
        "probe.graph.degree",
        "probe.marker.verify",
        "probe.graph.pairs",
        "probe.marker.verify",
        "probe.graph.adjacency",
        "probe.marker.verify",
        "probe.graph.explain",
    ]

    nodes_create_sql, _values, nodes_replay_safe = client.statements[
        "probe.graph.nodes.create"
    ]
    edges_create_sql, _values, edges_replay_safe = client.statements[
        "probe.graph.edges.create"
    ]
    assert nodes_replay_safe is False
    assert edges_replay_safe is False
    assert "LOGICAL PARTITION BY LIST (workspace)" in nodes_create_sql
    assert "LOGICAL PARTITION BY LIST (workspace)" in edges_create_sql
    assert "orientation = 'row,column'" in nodes_create_sql
    assert "orientation = 'row,column'" in edges_create_sql
    assert "COLLATE" not in nodes_create_sql
    assert "COLLATE" not in edges_create_sql

    for descriptor in (
        "probe.graph.nodes.insert",
        "probe.graph.nodes.merge",
        "probe.graph.edges.insert",
    ):
        assert client.statements[descriptor][2] is True

    merge_sql = client.statements["probe.graph.nodes.merge"][0]
    assert "current.properties || EXCLUDED.properties" in merge_sql

    pairs_sql = client.statements["probe.graph.pairs"][0]
    assert "generate_series(1, $5::int)" in pairs_sql
    assert "unnest($3::text[], $4" not in pairs_sql

    explain_sql = client.statements["probe.graph.explain"][0]
    assert explain_sql.startswith("EXPLAIN SELECT")

    for sql, _values, _replay_safe in client.statements.values():
        assert ";" not in sql


class MismatchGraphProbeClient(GraphProbeClient):
    def __init__(self, stage):
        super().__init__()
        self.stage = stage

    async def fetch_value(self, sql, *values, descriptor):
        if descriptor == "probe.graph.collation" and self.stage == "collation":
            self.events.append(descriptor)
            return "en_US.utf8"
        if descriptor == "probe.graph.orientation" and self.stage == "orientation":
            self.events.append(descriptor)
            return "row"
        if descriptor == "probe.graph.nodes.merge.verify" and self.stage == "merge":
            self.events.append(descriptor)
            return '{"step":2}'
        if descriptor == "probe.graph.degree" and self.stage == "degree":
            self.events.append(descriptor)
            return 2
        if descriptor == "probe.graph.pairs" and self.stage == "pairs":
            self.events.append(descriptor)
            return 1
        return await super().fetch_value(sql, *values, descriptor=descriptor)

    async def fetch_all(self, sql, *values, descriptor):
        if descriptor == "probe.graph.nodes.order" and self.stage == "order":
            self.events.append(descriptor)
            return [{"id": "a"}, {"id": "B"}, {"id": "_x"}]
        if descriptor == "probe.graph.adjacency" and self.stage == "adjacency":
            self.events.append(descriptor)
            return [{"nid": "B"}, {"nid": "_x"}, {"nid": "leaked"}]
        if descriptor == "probe.graph.explain" and self.stage == "explain":
            self.events.append(descriptor)
            return [{"QUERY PLAN": "Seq Scan on lightrag_test_gedges"}]
        return await super().fetch_all(sql, *values, descriptor=descriptor)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stage", "detail_code"),
    [
        ("collation", "graph_collation_mismatch"),
        ("orientation", "graph_orientation_mismatch"),
        ("merge", "graph_merge_semantics_mismatch"),
        ("order", "graph_order_mismatch"),
    ],
)
async def test_graph_probe_fails_partition_and_skips_adjacency_on_mismatch(
    stage, detail_code
):
    client = MismatchGraphProbeClient(stage)

    partition, adjacency = await _run_graph_probe(client)

    assert partition == ProbeResult(
        kind=ProbeKind.LOGICAL_PARTITION,
        status=ProbeStatus.FAILED,
        blocking=True,
        detail_code=detail_code,
    )
    assert adjacency == ProbeResult(
        kind=ProbeKind.GRAPH_ADJACENCY_EXPLAIN,
        status=ProbeStatus.NOT_RUN,
        blocking=True,
        detail_code="graph_tables_unavailable",
    )
    assert "probe.graph.edges.insert" not in client.events


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stage", "detail_code"),
    [
        ("degree", "graph_degree_mismatch"),
        ("pairs", "graph_pair_batch_mismatch"),
        ("adjacency", "graph_adjacency_mismatch"),
        ("explain", "graph_adjacency_plan_mismatch"),
    ],
)
async def test_graph_probe_keeps_partition_pass_when_adjacency_stage_fails(
    stage, detail_code
):
    client = MismatchGraphProbeClient(stage)

    partition, adjacency = await _run_graph_probe(client)

    assert partition.status is ProbeStatus.PASSED
    assert adjacency == ProbeResult(
        kind=ProbeKind.GRAPH_ADJACENCY_EXPLAIN,
        status=ProbeStatus.FAILED,
        blocking=True,
        detail_code=detail_code,
    )


class OwnershipLossGraphProbeClient(GraphProbeClient):
    async def execute_one(self, sql, *values, descriptor, replay_safe=False):
        if descriptor == "probe.graph.edges.insert":
            raise HologresProbeError(
                "ownership lost", leaked_objects=("lightrag_test_graph_probe",)
            )
        return await super().execute_one(
            sql, *values, descriptor=descriptor, replay_safe=replay_safe
        )


@pytest.mark.asyncio
async def test_graph_probe_reraises_ownership_errors_instead_of_classifying():
    client = OwnershipLossGraphProbeClient()

    with pytest.raises(HologresProbeError):
        await _run_graph_probe(client)


class JsonbColumnarProbeClient:
    def __init__(self):
        self.events = []
        self.statements = {}

    async def execute_one(self, sql, *values, descriptor, replay_safe=False):
        self.events.append(descriptor)
        self.statements[descriptor] = (sql, values, replay_safe)
        return "OK"

    async def fetch_value(self, sql, *values, descriptor):
        self.events.append(descriptor)
        self.statements[descriptor] = (sql, values, None)
        if descriptor in ("probe.jsonbcol.catalog", "probe.jsonbcol.catalog.replay"):
            return True
        if descriptor == "probe.jsonbcol.fetch":
            return '{"keep":"x","nested":{"a":[1,2]}}'
        raise AssertionError(f"Unexpected value fetch: {descriptor}")


async def _run_jsonb_columnar_probe(client):
    async def verify_ownership():
        client.events.append("probe.marker.verify")

    return await hologres_capabilities._probe_jsonb_column_optimization(
        client,
        "lightrag_test_jsonbcol_probe",
        "lightrag_test_jsonbcol",
        verify_ownership,
    )


@pytest.mark.asyncio
async def test_jsonb_columnar_probe_freezes_alter_catalog_and_roundtrip():
    client = JsonbColumnarProbeClient()

    result = await _run_jsonb_columnar_probe(client)

    assert result == ProbeResult(
        kind=ProbeKind.JSONB_COLUMN_OPTIMIZATION,
        status=ProbeStatus.PASSED,
        blocking=False,
        detail_code="jsonb_columnar_layout_frozen",
    )
    assert client.events == [
        "probe.marker.verify",
        "probe.jsonbcol.create",
        "probe.marker.verify",
        "probe.jsonbcol.alter",
        "probe.marker.verify",
        "probe.jsonbcol.catalog",
        "probe.marker.verify",
        "probe.jsonbcol.alter.replay",
        "probe.marker.verify",
        "probe.jsonbcol.catalog.replay",
        "probe.marker.verify",
        "probe.jsonbcol.upsert",
        "probe.marker.verify",
        "probe.jsonbcol.fetch",
    ]

    create_sql, _values, create_replay_safe = client.statements[
        "probe.jsonbcol.create"
    ]
    assert create_replay_safe is False
    assert "orientation = 'row,column'" in create_sql
    # The columnar property must go through the separate ALTER path; the
    # inline CREATE syntax was rejected by live Hologres.
    assert "enable_columnar_type" not in create_sql

    alter_sql, _values, alter_replay_safe = client.statements["probe.jsonbcol.alter"]
    replay_sql, _values, replay_replay_safe = client.statements[
        "probe.jsonbcol.alter.replay"
    ]
    assert alter_replay_safe is True
    assert replay_replay_safe is True
    assert alter_sql == replay_sql
    assert alter_sql == (
        'ALTER TABLE "lightrag_test_jsonbcol_probe"."lightrag_test_jsonbcol" '
        "ALTER COLUMN payload SET (enable_columnar_type = on)"
    )

    catalog_sql, catalog_values, _safe = client.statements["probe.jsonbcol.catalog"]
    assert catalog_sql == client.statements["probe.jsonbcol.catalog.replay"][0]
    assert catalog_sql == _JSONB_COLUMNAR_POSTCONDITION
    assert catalog_values == (
        "lightrag_test_jsonbcol_probe",
        "lightrag_test_jsonbcol",
        "payload",
    )

    assert client.statements["probe.jsonbcol.upsert"][2] is True
    for sql, _values, _replay_safe in client.statements.values():
        assert ";" not in sql


class MismatchJsonbColumnarProbeClient(JsonbColumnarProbeClient):
    def __init__(self, stage):
        super().__init__()
        self.stage = stage

    async def fetch_value(self, sql, *values, descriptor):
        if descriptor == "probe.jsonbcol.catalog" and self.stage == "catalog":
            self.events.append(descriptor)
            return False
        if descriptor == "probe.jsonbcol.catalog.replay" and self.stage == "replay":
            self.events.append(descriptor)
            return False
        if descriptor == "probe.jsonbcol.fetch" and self.stage == "roundtrip":
            self.events.append(descriptor)
            return '{"keep":"x"}'
        return await super().fetch_value(sql, *values, descriptor=descriptor)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stage", "detail_code"),
    [
        ("catalog", "jsonb_columnar_property_missing"),
        ("replay", "jsonb_columnar_replay_mismatch"),
        ("roundtrip", "jsonb_columnar_roundtrip_mismatch"),
    ],
)
async def test_jsonb_columnar_probe_classifies_stage_mismatches(stage, detail_code):
    client = MismatchJsonbColumnarProbeClient(stage)

    result = await _run_jsonb_columnar_probe(client)

    assert result == ProbeResult(
        kind=ProbeKind.JSONB_COLUMN_OPTIMIZATION,
        status=ProbeStatus.FAILED,
        blocking=False,
        detail_code=detail_code,
    )
    if stage == "catalog":
        assert "probe.jsonbcol.alter.replay" not in client.events
    if stage in ("catalog", "replay"):
        assert "probe.jsonbcol.upsert" not in client.events


class FailingJsonbColumnarProbeClient(JsonbColumnarProbeClient):
    async def execute_one(self, sql, *values, descriptor, replay_safe=False):
        if descriptor == "probe.jsonbcol.alter":
            raise RuntimeError("alter rejected")
        return await super().execute_one(
            sql, *values, descriptor=descriptor, replay_safe=replay_safe
        )


@pytest.mark.asyncio
async def test_jsonb_columnar_probe_classifies_execution_failures():
    client = FailingJsonbColumnarProbeClient()

    result = await _run_jsonb_columnar_probe(client)

    assert result == ProbeResult(
        kind=ProbeKind.JSONB_COLUMN_OPTIMIZATION,
        status=ProbeStatus.FAILED,
        blocking=False,
        detail_code="jsonb_columnar_probe_failed",
    )


class OwnershipLossJsonbColumnarProbeClient(JsonbColumnarProbeClient):
    async def execute_one(self, sql, *values, descriptor, replay_safe=False):
        if descriptor == "probe.jsonbcol.alter":
            raise HologresProbeError(
                "ownership lost", leaked_objects=("lightrag_test_jsonbcol_probe",)
            )
        return await super().execute_one(
            sql, *values, descriptor=descriptor, replay_safe=replay_safe
        )


@pytest.mark.asyncio
async def test_jsonb_columnar_probe_reraises_ownership_errors():
    client = OwnershipLossJsonbColumnarProbeClient()

    with pytest.raises(HologresProbeError):
        await _run_jsonb_columnar_probe(client)


class StreamCopyProbeClient:
    def __init__(self):
        self.events = []
        self.statements = {}
        self.copies = []

    async def execute_one(self, sql, *values, descriptor, replay_safe=False):
        self.events.append(descriptor)
        self.statements[descriptor] = (sql, values, replay_safe)
        return "OK"

    async def fetch_value(self, sql, *values, descriptor):
        self.events.append(descriptor)
        self.statements[descriptor] = (sql, values, None)
        if descriptor == "probe.streamcopy.check":
            return "after"
        if descriptor == "probe.streamcopy.count":
            return 2
        raise AssertionError(f"Unexpected value fetch: {descriptor}")

    async def _stream_copy_for_probe(
        self, schema, table, columns, records, *, descriptor
    ):
        self.events.append(descriptor)
        self.copies.append((schema, table, tuple(columns), tuple(records)))
        return "COPY 2"


async def _run_stream_copy_probe(client):
    async def verify_ownership():
        client.events.append("probe.marker.verify")

    return await hologres_capabilities._probe_stream_copy(
        client,
        "lightrag_test_streamcopy_probe",
        "lightrag_test_streamcopy",
        verify_ownership,
    )


@pytest.mark.asyncio
async def test_stream_copy_probe_freezes_conflict_update_contract():
    client = StreamCopyProbeClient()

    result = await _run_stream_copy_probe(client)

    assert result == ProbeResult(
        kind=ProbeKind.STREAM_COPY,
        status=ProbeStatus.PASSED,
        blocking=False,
        detail_code="stream_copy_conflict_update_frozen",
    )
    assert client.events == [
        "probe.marker.verify",
        "probe.streamcopy.create",
        "probe.marker.verify",
        "probe.streamcopy.seed",
        "probe.marker.verify",
        "probe.streamcopy.copy",
        "probe.marker.verify",
        "probe.streamcopy.check",
        "probe.marker.verify",
        "probe.streamcopy.count",
    ]

    create_sql, _values, create_replay_safe = client.statements[
        "probe.streamcopy.create"
    ]
    assert create_replay_safe is False
    assert create_sql == (
        'CREATE TABLE "lightrag_test_streamcopy_probe"."lightrag_test_streamcopy" '
        "(id text NOT NULL, val text NOT NULL, PRIMARY KEY (id)) "
        "WITH (orientation = 'row,column')"
    )

    _seed_sql, seed_values, seed_replay_safe = client.statements[
        "probe.streamcopy.seed"
    ]
    assert seed_replay_safe is False
    assert seed_values == ("conflict-key", "before")

    # The copy payload rewrites the seeded key: an update proves the stream
    # channel, a unique violation would be plain COPY semantics.
    assert client.copies == [
        (
            "lightrag_test_streamcopy_probe",
            "lightrag_test_streamcopy",
            ("id", "val"),
            (("conflict-key", "after"), ("fresh-key", "new")),
        )
    ]

    _check_sql, check_values, _safe = client.statements["probe.streamcopy.check"]
    assert check_values == ("conflict-key",)
    for sql, _values, _replay_safe in client.statements.values():
        assert ";" not in sql


class MismatchStreamCopyProbeClient(StreamCopyProbeClient):
    def __init__(self, stage):
        super().__init__()
        self.stage = stage

    async def fetch_value(self, sql, *values, descriptor):
        if descriptor == "probe.streamcopy.check" and self.stage == "conflict":
            self.events.append(descriptor)
            return "before"
        if descriptor == "probe.streamcopy.count" and self.stage == "count":
            self.events.append(descriptor)
            return 3
        return await super().fetch_value(sql, *values, descriptor=descriptor)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stage", "detail_code"),
    [
        ("conflict", "stream_copy_conflict_not_updated"),
        ("count", "stream_copy_row_count_mismatch"),
    ],
)
async def test_stream_copy_probe_classifies_stage_mismatches(stage, detail_code):
    client = MismatchStreamCopyProbeClient(stage)

    result = await _run_stream_copy_probe(client)

    assert result == ProbeResult(
        kind=ProbeKind.STREAM_COPY,
        status=ProbeStatus.FAILED,
        blocking=False,
        detail_code=detail_code,
    )
    if stage == "conflict":
        assert "probe.streamcopy.count" not in client.events


class FailingStreamCopyProbeClient(StreamCopyProbeClient):
    async def _stream_copy_for_probe(
        self, schema, table, columns, records, *, descriptor
    ):
        raise RuntimeError("stream copy rejected")


@pytest.mark.asyncio
async def test_stream_copy_probe_classifies_execution_failures():
    client = FailingStreamCopyProbeClient()

    result = await _run_stream_copy_probe(client)

    assert result == ProbeResult(
        kind=ProbeKind.STREAM_COPY,
        status=ProbeStatus.FAILED,
        blocking=False,
        detail_code="stream_copy_probe_failed",
    )


class OwnershipLossStreamCopyProbeClient(StreamCopyProbeClient):
    async def execute_one(self, sql, *values, descriptor, replay_safe=False):
        if descriptor == "probe.streamcopy.seed":
            raise HologresProbeError(
                "ownership lost",
                leaked_objects=("lightrag_test_streamcopy_probe",),
            )
        return await super().execute_one(
            sql, *values, descriptor=descriptor, replay_safe=replay_safe
        )


@pytest.mark.asyncio
async def test_stream_copy_probe_reraises_ownership_errors():
    client = OwnershipLossStreamCopyProbeClient()

    with pytest.raises(HologresProbeError):
        await _run_stream_copy_probe(client)


class ProductionStreamCopyClient:
    def __init__(self, *, enabled=True, capabilities=None):
        self.config = SimpleNamespace(
            stream_copy_enabled=enabled, schema="LightRAG"
        )
        self.capabilities = capabilities
        self.events = []
        self.statements = {}
        self.copies = []

    async def execute_one(self, sql, *values, descriptor, replay_safe=False):
        self.events.append(descriptor)
        self.statements[descriptor] = (sql, values, replay_safe)
        return "OK"

    async def fetch_value(self, sql, *values, descriptor):
        self.events.append(descriptor)
        self.statements[descriptor] = (sql, values, None)
        if descriptor == "capability.streamcopy.check":
            return "after"
        if descriptor == "capability.streamcopy.count":
            return 2
        raise AssertionError(f"Unexpected value fetch: {descriptor}")

    async def _stream_copy_for_probe(
        self, schema, table, columns, records, *, descriptor
    ):
        self.events.append(descriptor)
        self.copies.append((schema, table, tuple(columns), tuple(records)))
        return "COPY 2"


def _version_only_report():
    return CapabilityReport(version=HologresVersion(5, 0, 0), results=())


def _report_with_stream_copy(status, detail_code):
    return CapabilityReport(
        version=HologresVersion(5, 0, 0),
        results=(
            ProbeResult(
                kind=ProbeKind.STREAM_COPY,
                status=status,
                blocking=False,
                detail_code=detail_code,
            ),
        ),
    )


@pytest.mark.asyncio
async def test_stream_copy_proof_passes_reports_through_when_config_gate_is_off():
    client = ProductionStreamCopyClient(enabled=False)
    report = _version_only_report()

    assert await prove_stream_copy_capability(client, report) is report
    assert client.events == []


@pytest.mark.asyncio
async def test_stream_copy_proof_passes_reports_through_when_already_present():
    client = ProductionStreamCopyClient()
    report = _report_with_stream_copy(ProbeStatus.FAILED, "stream_copy_probe_failed")

    assert await prove_stream_copy_capability(client, report) is report
    assert client.events == []


@pytest.mark.asyncio
async def test_stream_copy_proof_reuses_the_result_cached_on_the_shared_client():
    cached_result = ProbeResult(
        kind=ProbeKind.STREAM_COPY,
        status=ProbeStatus.PASSED,
        blocking=False,
        detail_code="stream_copy_conflict_update_frozen",
    )
    client = ProductionStreamCopyClient(
        capabilities=CapabilityReport(
            version=HologresVersion(5, 0, 0), results=(cached_result,)
        )
    )

    enriched = await prove_stream_copy_capability(client, _version_only_report())

    assert enriched.results == (cached_result,)
    assert enriched.supports(ProbeKind.STREAM_COPY) is True
    assert client.events == []


@pytest.mark.asyncio
async def test_stream_copy_proof_probes_a_disposable_table_and_appends_the_result():
    client = ProductionStreamCopyClient()

    enriched = await prove_stream_copy_capability(client, _version_only_report())

    assert enriched.supports(ProbeKind.STREAM_COPY) is True
    (result,) = enriched.results
    assert result.detail_code == "stream_copy_conflict_update_frozen"
    assert client.events == [
        "capability.streamcopy.create",
        "capability.streamcopy.seed",
        "capability.streamcopy.copy",
        "capability.streamcopy.check",
        "capability.streamcopy.count",
        "capability.streamcopy.drop",
    ]

    create_sql, _values, create_replay_safe = client.statements[
        "capability.streamcopy.create"
    ]
    assert create_replay_safe is False
    match = re.fullmatch(
        r'CREATE TABLE "LightRAG"\."(lightrag_test_streamcopy_[0-9a-f]{32})" '
        r"\(id text NOT NULL, val text NOT NULL, PRIMARY KEY \(id\)\) "
        r"WITH \(orientation = 'row,column'\)",
        create_sql,
    )
    assert match is not None
    table = match.group(1)
    assert client.copies == [
        (
            "LightRAG",
            table,
            ("id", "val"),
            (("conflict-key", "after"), ("fresh-key", "new")),
        )
    ]

    drop_sql, _values, drop_replay_safe = client.statements[
        "capability.streamcopy.drop"
    ]
    assert drop_replay_safe is True
    assert drop_sql == f'DROP TABLE IF EXISTS "LightRAG"."{table}"'


class FailingProductionStreamCopyClient(ProductionStreamCopyClient):
    async def _stream_copy_for_probe(
        self, schema, table, columns, records, *, descriptor
    ):
        raise RuntimeError("stream copy rejected")


@pytest.mark.asyncio
async def test_stream_copy_proof_records_failures_and_still_drops_without_raising():
    client = FailingProductionStreamCopyClient()

    enriched = await prove_stream_copy_capability(client, _version_only_report())

    assert enriched.supports(ProbeKind.STREAM_COPY) is False
    (result,) = enriched.results
    assert result.status is ProbeStatus.FAILED
    assert result.detail_code == "stream_copy_probe_failed"
    assert client.events[-1] == "capability.streamcopy.drop"


_AGE_PROBE_PROPS = json.dumps(
    {
        "name": "alpha",
        "description": hologres_capabilities._AGE_PROBE_DESCRIPTION,
    },
    ensure_ascii=False,
)


class AgeProbeClient:
    def __init__(self):
        self.events = []
        self.statements = {}
        self.procedures = []

    async def fetch_value(self, sql, *values, descriptor):
        self.events.append(descriptor)
        self.statements[descriptor] = (sql, values, None)
        if descriptor.endswith(".extension"):
            return True
        if descriptor.endswith(".graph.check"):
            return True
        raise AssertionError(f"Unexpected value fetch: {descriptor}")

    async def call_age_procedure(
        self, procedure, *values, descriptor, replay_safe=False
    ):
        self.events.append(descriptor)
        self.procedures.append((procedure, values, replay_safe))
        return "CALL"

    async def fetch_all(
        self, sql, *values, descriptor, operation_kind=None, replay_safe=None
    ):
        self.events.append(descriptor)
        self.statements[descriptor] = (sql, values, replay_safe)
        if descriptor.endswith(".roundtrip") or descriptor.endswith(".bind"):
            return [(_AGE_PROBE_PROPS,)]
        if descriptor.endswith(".degree") or descriptor.endswith(".count"):
            return [("1",)]
        return []


@pytest.mark.asyncio
async def test_age_probe_freezes_graph_lifecycle_and_frozen_contract():
    client = AgeProbeClient()

    result = await hologres_capabilities.probe_age_graph_capability(client)

    assert result == ProbeResult(
        kind=ProbeKind.AGE,
        status=ProbeStatus.PASSED,
        blocking=False,
        detail_code="age_graph_semantics_frozen",
    )
    assert client.events == [
        "capability.age.extension",
        "capability.age.graph.create",
        "capability.age.vlabel",
        "capability.age.elabel",
        "capability.age.node.alpha",
        "capability.age.node.beta",
        "capability.age.edge",
        "capability.age.roundtrip",
        "capability.age.bind",
        "capability.age.degree",
        "capability.age.delete",
        "capability.age.count",
        "capability.age.graph.check",
        "capability.age.graph.drop",
    ]

    (create, vlabel, elabel, drop) = client.procedures
    graph = create[1][0]
    assert re.fullmatch(r"lightrag_test_age_[0-9a-f]{32}", graph)
    assert create == ("create_graph", (graph,), False)
    assert vlabel == ("create_vlabel", (graph, "Entity"), False)
    assert elabel == ("create_elabel", (graph, "DIRECTED"), False)
    assert drop == ("drop_graph", (graph, True), False)

    alpha_sql, _values, alpha_replay = client.statements["capability.age.node.alpha"]
    assert alpha_replay is False
    assert alpha_sql == (
        f"SELECT * FROM ag_catalog.cypher('{graph}', "
        "$lightrag_age$ CREATE (n:Entity {name: 'alpha', "
        "description: 'it\\'s a \"tricky\" \\\\ value; with 中文'}) "
        "$lightrag_age$) AS (result ag_catalog.agtype)"
    )
    edge_sql, _values, edge_replay = client.statements["capability.age.edge"]
    assert edge_replay is False
    assert "CREATE (a)-[:DIRECTED {weight: 2.5}]->(b)" in edge_sql
    assert "[r:" not in edge_sql

    bind_sql, bind_values, _safe = client.statements["capability.age.bind"]
    assert bind_sql.endswith(
        "$lightrag_age$, $1) AS (props ag_catalog.agtype)"
    )
    assert bind_values == ('{"name": "alpha"}',)

    delete_sql, _values, delete_replay = client.statements["capability.age.delete"]
    assert "DETACH DELETE n" in delete_sql
    assert delete_replay is True

    check_sql, check_values, _safe = client.statements["capability.age.graph.check"]
    assert check_values == (graph,)
    assert "pg_namespace" in check_sql


@pytest.mark.asyncio
async def test_age_probe_missing_extension_fails_closed_without_graph_creation():
    class MissingExtensionClient(AgeProbeClient):
        async def fetch_value(self, sql, *values, descriptor):
            self.events.append(descriptor)
            if descriptor.endswith(".extension"):
                return False
            raise AssertionError(f"Unexpected value fetch: {descriptor}")

    client = MissingExtensionClient()

    result = await hologres_capabilities.probe_age_graph_capability(client)

    assert result.status is ProbeStatus.FAILED
    assert result.detail_code == "age_extension_missing"
    assert client.procedures == []


class MismatchAgeProbeClient(AgeProbeClient):
    def __init__(self, stage):
        super().__init__()
        self.stage = stage

    async def fetch_all(
        self, sql, *values, descriptor, operation_kind=None, replay_safe=None
    ):
        if descriptor.endswith(".roundtrip") and self.stage == "roundtrip":
            self.events.append(descriptor)
            return [('{"name": "alpha", "description": "wrong"}',)]
        if descriptor.endswith(".bind") and self.stage == "bind":
            self.events.append(descriptor)
            return [('{"name": "alpha"}',)]
        if descriptor.endswith(".degree") and self.stage == "degree":
            self.events.append(descriptor)
            return [("2",)]
        if descriptor.endswith(".count") and self.stage == "count":
            self.events.append(descriptor)
            return [("2",)]
        return await super().fetch_all(
            sql,
            *values,
            descriptor=descriptor,
            operation_kind=operation_kind,
            replay_safe=replay_safe,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stage", "detail_code"),
    [
        ("roundtrip", "age_property_roundtrip_mismatch"),
        ("bind", "age_bind_parameter_mismatch"),
        ("degree", "age_undirected_degree_mismatch"),
        ("count", "age_delete_count_mismatch"),
    ],
)
async def test_age_probe_classifies_stage_mismatches_and_still_drops(
    stage, detail_code
):
    client = MismatchAgeProbeClient(stage)

    result = await hologres_capabilities.probe_age_graph_capability(client)

    assert result.status is ProbeStatus.FAILED
    assert result.detail_code == detail_code
    assert client.events[-1] == "capability.age.graph.drop"


@pytest.mark.asyncio
async def test_age_probe_classifies_execution_failures_and_still_drops():
    class FailingWriteClient(AgeProbeClient):
        async def fetch_all(
            self, sql, *values, descriptor, operation_kind=None, replay_safe=None
        ):
            if descriptor.endswith(".node.alpha"):
                self.events.append(descriptor)
                raise RuntimeError("cypher write rejected")
            return await super().fetch_all(
                sql,
                *values,
                descriptor=descriptor,
                operation_kind=operation_kind,
                replay_safe=replay_safe,
            )

    client = FailingWriteClient()

    result = await hologres_capabilities.probe_age_graph_capability(client)

    assert result.status is ProbeStatus.FAILED
    assert result.detail_code == "age_graph_probe_failed"
    assert client.events[-1] == "capability.age.graph.drop"


@pytest.mark.asyncio
async def test_age_probe_reports_leaked_graph_cleanup_with_a_distinct_code():
    class FailingDropClient(AgeProbeClient):
        async def call_age_procedure(
            self, procedure, *values, descriptor, replay_safe=False
        ):
            if procedure == "drop_graph":
                self.events.append(descriptor)
                raise RuntimeError("drop rejected")
            return await super().call_age_procedure(
                procedure, *values, descriptor=descriptor, replay_safe=replay_safe
            )

    client = FailingDropClient()

    result = await hologres_capabilities.probe_age_graph_capability(client)

    assert result.status is ProbeStatus.FAILED
    assert result.detail_code == "age_graph_probe_cleanup_failed"


@pytest.mark.asyncio
async def test_age_probe_drops_after_an_unknown_outcome_graph_create():
    class FailingCreateClient(AgeProbeClient):
        async def call_age_procedure(
            self, procedure, *values, descriptor, replay_safe=False
        ):
            if procedure == "create_graph":
                self.events.append(descriptor)
                raise RuntimeError("create outcome unknown")
            return await super().call_age_procedure(
                procedure, *values, descriptor=descriptor, replay_safe=replay_safe
            )

    client = FailingCreateClient()

    result = await hologres_capabilities.probe_age_graph_capability(client)

    assert result.status is ProbeStatus.FAILED
    assert result.detail_code == "age_graph_probe_failed"
    assert client.events[-2:] == [
        "capability.age.graph.check",
        "capability.age.graph.drop",
    ]


@pytest.mark.asyncio
async def test_age_probe_reraises_ownership_errors_with_the_graph_as_leaked():
    client = AgeProbeClient()
    calls = {"count": 0}

    async def verify_ownership():
        calls["count"] += 1
        if calls["count"] >= 3:
            raise HologresProbeError(
                "ownership lost", leaked_objects=("lightrag_test_probe",)
            )

    with pytest.raises(HologresProbeError) as exc_info:
        await hologres_capabilities._probe_age_graph(
            client, "lightrag_test_age_owned", verify_ownership, "probe.age"
        )

    assert exc_info.value.leaked_objects == (
        "lightrag_test_probe",
        "lightrag_test_age_owned",
    )


class CreateFailureProductionStreamCopyClient(ProductionStreamCopyClient):
    async def execute_one(self, sql, *values, descriptor, replay_safe=False):
        if descriptor == "capability.streamcopy.create":
            self.events.append(descriptor)
            raise RuntimeError("create outcome unknown")
        return await super().execute_one(
            sql, *values, descriptor=descriptor, replay_safe=replay_safe
        )


@pytest.mark.asyncio
async def test_stream_copy_proof_drops_best_effort_after_unknown_create_outcome():
    client = CreateFailureProductionStreamCopyClient()

    enriched = await prove_stream_copy_capability(client, _version_only_report())

    (result,) = enriched.results
    assert result.status is ProbeStatus.FAILED
    assert result.detail_code == "stream_copy_probe_failed"
    assert client.events == [
        "capability.streamcopy.create",
        "capability.streamcopy.drop",
    ]


class DropFailureProductionStreamCopyClient(ProductionStreamCopyClient):
    async def execute_one(self, sql, *values, descriptor, replay_safe=False):
        if descriptor == "capability.streamcopy.drop":
            self.events.append(descriptor)
            raise RuntimeError("drop rejected")
        return await super().execute_one(
            sql, *values, descriptor=descriptor, replay_safe=replay_safe
        )


@pytest.mark.asyncio
async def test_stream_copy_proof_reports_leaked_cleanup_with_a_distinct_code():
    client = DropFailureProductionStreamCopyClient()

    enriched = await prove_stream_copy_capability(client, _version_only_report())

    assert enriched.supports(ProbeKind.STREAM_COPY) is False
    (result,) = enriched.results
    assert result.status is ProbeStatus.FAILED
    assert result.detail_code == "stream_copy_probe_cleanup_failed"
    assert client.events[-1] == "capability.streamcopy.drop"


class SuccessfulProbeClient:
    def __init__(self):
        self.writes = []
        self.reconnect_count = 0
        self.owner_token = None

    async def fetch_value(self, sql, *values, descriptor):
        results = {
            "capability.version": "Hologres 5.0.0",
            "probe.schema.preflight": False,
            "probe.marker.verify": self.owner_token,
            "probe.reconnect.verify": 1,
            "probe.graph.collation": "C",
            "probe.graph.orientation": "row,column",
            "probe.graph.nodes.merge.verify": {"keep": "x", "step": 2},
            "probe.graph.degree": 3,
            "probe.graph.pairs": 2,
            "probe.jsonbcol.catalog": True,
            "probe.jsonbcol.catalog.replay": True,
            "probe.jsonbcol.fetch": '{"keep":"x","nested":{"a":[1,2]}}',
            "probe.streamcopy.check": "after",
            "probe.streamcopy.count": 2,
        }
        return results[descriptor]

    async def fetch_one(self, sql, *values, descriptor):
        return {"payload": {"step": 2}, "tags": ["beta", "gamma"]}

    async def fetch_all(self, sql, *values, descriptor):
        results = {
            "probe.graph.nodes.order": [{"id": "B"}, {"id": "_x"}, {"id": "a"}],
            "probe.graph.adjacency": [{"nid": "B"}, {"nid": "_x"}],
            "probe.graph.explain": [
                {"QUERY PLAN": "Partition Filter: (workspace = 'w1')"}
            ],
        }
        return results[descriptor]

    async def execute_one(self, sql, *values, descriptor, replay_safe=False):
        if descriptor == "probe.marker.insert":
            self.owner_token = values[0]
        self.writes.append((descriptor, replay_safe))
        return "OK"

    async def _stream_copy_for_probe(
        self, schema, table, columns, records, *, descriptor
    ):
        self.writes.append((descriptor, True))
        return "COPY 2"

    async def _observe_setup_reset_for_probe(self, *, bound, marker):
        return SimpleNamespace(
            schema_name="lightrag_test_new_probe",
            bound=bound,
            same_connection_reacquired=True,
            application_name="",
        )

    async def _reconnect_for_probe(self):
        self.reconnect_count += 1


@pytest.mark.asyncio
async def test_initial_probes_do_not_replay_nonidempotent_create_ddl():
    client = SuccessfulProbeClient()

    report = await run_initial_isolated_probes(client, "lightrag_test_new_probe")

    replay_safety = dict(client.writes)
    assert any(result.kind is ProbeKind.HGRAPH for result in report.results)
    failures = {result.kind: result for result in report.blocking_failures}
    assert set(failures) == {ProbeKind.HGRAPH}
    assert failures[ProbeKind.HGRAPH].status is ProbeStatus.FAILED
    assert failures[ProbeKind.HGRAPH].detail_code == "hgraph_probe_failed"
    passed = {result.kind: result for result in report.results}
    assert passed[ProbeKind.LOGICAL_PARTITION].status is ProbeStatus.PASSED
    assert passed[ProbeKind.GRAPH_ADJACENCY_EXPLAIN].status is ProbeStatus.PASSED
    assert passed[ProbeKind.JSONB_COLUMN_OPTIMIZATION].status is ProbeStatus.PASSED
    assert passed[ProbeKind.STREAM_COPY].status is ProbeStatus.PASSED
    assert replay_safety["probe.schema.create"] is False
    assert replay_safety["probe.marker.create"] is False
    assert replay_safety["probe.marker.insert"] is False
    assert replay_safety["probe.basic.create"] is False
    assert replay_safety["probe.hgraph.create"] is False
    assert replay_safety["probe.graph.nodes.create"] is False
    assert replay_safety["probe.graph.edges.create"] is False
    assert replay_safety["probe.jsonbcol.create"] is False
    assert replay_safety["probe.streamcopy.create"] is False
    assert replay_safety["probe.streamcopy.seed"] is False
    assert replay_safety["probe.hgraph.insert"] is True
    assert replay_safety["probe.graph.nodes.insert"] is True
    assert replay_safety["probe.graph.nodes.merge"] is True
    assert replay_safety["probe.graph.edges.insert"] is True
    assert replay_safety["probe.jsonbcol.alter"] is True
    assert replay_safety["probe.jsonbcol.alter.replay"] is True
    assert replay_safety["probe.jsonbcol.upsert"] is True
    assert replay_safety["probe.basic.drop"] is True
    assert replay_safety["probe.hgraph.drop"] is True
    assert replay_safety["probe.graph.edges.drop"] is True
    assert replay_safety["probe.graph.nodes.drop"] is True
    assert replay_safety["probe.jsonbcol.drop"] is True
    assert replay_safety["probe.streamcopy.drop"] is True
    assert replay_safety["probe.marker.drop"] is True
    assert replay_safety["probe.schema.drop"] is True
    assert client.reconnect_count == 1
