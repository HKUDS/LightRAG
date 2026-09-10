from contextlib import asynccontextmanager
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
    run_initial_isolated_probes,
    validate_hologres_version,
    validate_test_schema_name,
)
from lightrag.kg.hologres.client import HologresClient
from lightrag.kg.hologres.config import HologresConfig


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
        ProbeKind.FULL_TEXT_DDL,
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
        }
        return results[descriptor]

    async def fetch_one(self, sql, *values, descriptor):
        return {"payload": {"step": 2}, "tags": ["beta", "gamma"]}

    async def execute_one(self, sql, *values, descriptor, replay_safe=False):
        if descriptor == "probe.marker.insert":
            self.owner_token = values[0]
        self.writes.append((descriptor, replay_safe))
        return "OK"

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
    assert set(failures) == {
        ProbeKind.HGRAPH,
        ProbeKind.LOGICAL_PARTITION,
        ProbeKind.GRAPH_ADJACENCY_EXPLAIN,
    }
    assert failures[ProbeKind.HGRAPH].status is ProbeStatus.FAILED
    assert failures[ProbeKind.HGRAPH].detail_code == "hgraph_probe_failed"
    assert failures[ProbeKind.LOGICAL_PARTITION].status is ProbeStatus.NOT_RUN
    assert failures[ProbeKind.GRAPH_ADJACENCY_EXPLAIN].status is ProbeStatus.NOT_RUN
    assert replay_safety["probe.schema.create"] is False
    assert replay_safety["probe.marker.create"] is False
    assert replay_safety["probe.marker.insert"] is False
    assert replay_safety["probe.basic.create"] is False
    assert replay_safety["probe.hgraph.create"] is False
    assert replay_safety["probe.hgraph.insert"] is True
    assert replay_safety["probe.basic.drop"] is True
    assert replay_safety["probe.hgraph.drop"] is True
    assert replay_safety["probe.marker.drop"] is True
    assert replay_safety["probe.schema.drop"] is True
    assert client.reconnect_count == 1
