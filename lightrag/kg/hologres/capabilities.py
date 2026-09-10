"""Hologres version validation and isolated capability probes."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
import json
import math
import re
import uuid

from .client import (
    HologresClient,
    HologresSqlError,
    OperationKind,
    quote_qualified_identifier,
    validate_identifier,
)


class HologresVersionError(RuntimeError):
    """Raised when a server is not recognizably Hologres 5.0 or newer."""


class HologresProbeError(RuntimeError):
    """Raised when an isolated probe cannot prove ownership or clean up."""

    def __init__(self, message: str, *, leaked_objects: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.leaked_objects = leaked_objects


@dataclass(frozen=True, order=True)
class HologresVersion:
    major: int
    minor: int
    patch: int = 0


_VERSION_PATTERN = re.compile(
    r"\bHologres(?:\s+version)?(?:[\s/:-]+)v?"
    r"(?P<major>\d+)\.(?P<minor>\d+)(?:\.(?P<patch>\d+))?\b",
    re.IGNORECASE,
)
_TEST_SCHEMA_PATTERN = re.compile(r"^lightrag_test_[a-z0-9_]{3,48}$")
_DETAIL_CODE_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class ProbeKind(str, Enum):
    SINGLE_AUTOCOMMIT_DDL = "single_autocommit_ddl"
    ASYNCPG_SETUP_RESET_BINDINGS = "asyncpg_setup_reset_bindings"
    JSONB_ON_CONFLICT_ARRAYS_RECONNECT = "jsonb_on_conflict_arrays_reconnect"
    LOGICAL_PARTITION = "logical_partition"
    GRAPH_ADJACENCY_EXPLAIN = "graph_adjacency_explain"
    HGRAPH = "hgraph"
    STREAM_COPY = "stream_copy"
    FULL_TEXT_DDL = "full_text_ddl"
    JSONB_COLUMN_OPTIMIZATION = "jsonb_column_optimization"
    AGE = "age"


class ProbeStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    NOT_RUN = "not_run"


@dataclass(frozen=True)
class ProbeSpec:
    kind: ProbeKind
    blocking: bool
    disposable: bool


PROBE_SPECS = {
    ProbeKind.SINGLE_AUTOCOMMIT_DDL: ProbeSpec(
        ProbeKind.SINGLE_AUTOCOMMIT_DDL, blocking=True, disposable=True
    ),
    ProbeKind.ASYNCPG_SETUP_RESET_BINDINGS: ProbeSpec(
        ProbeKind.ASYNCPG_SETUP_RESET_BINDINGS, blocking=True, disposable=True
    ),
    ProbeKind.JSONB_ON_CONFLICT_ARRAYS_RECONNECT: ProbeSpec(
        ProbeKind.JSONB_ON_CONFLICT_ARRAYS_RECONNECT,
        blocking=True,
        disposable=True,
    ),
    ProbeKind.LOGICAL_PARTITION: ProbeSpec(
        ProbeKind.LOGICAL_PARTITION, blocking=True, disposable=True
    ),
    ProbeKind.GRAPH_ADJACENCY_EXPLAIN: ProbeSpec(
        ProbeKind.GRAPH_ADJACENCY_EXPLAIN, blocking=True, disposable=True
    ),
    ProbeKind.HGRAPH: ProbeSpec(
        ProbeKind.HGRAPH, blocking=True, disposable=True
    ),
    ProbeKind.STREAM_COPY: ProbeSpec(
        ProbeKind.STREAM_COPY, blocking=False, disposable=True
    ),
    ProbeKind.FULL_TEXT_DDL: ProbeSpec(
        ProbeKind.FULL_TEXT_DDL, blocking=False, disposable=True
    ),
    ProbeKind.JSONB_COLUMN_OPTIMIZATION: ProbeSpec(
        ProbeKind.JSONB_COLUMN_OPTIMIZATION, blocking=False, disposable=True
    ),
    ProbeKind.AGE: ProbeSpec(ProbeKind.AGE, blocking=False, disposable=True),
}


@dataclass(frozen=True)
class HGraphProbeEvidence:
    ordered_raw_scores: tuple[tuple[int, float], ...]
    vector_filter_used: bool | None

    def __post_init__(self) -> None:
        if not isinstance(self.ordered_raw_scores, tuple) or (
            self.vector_filter_used is not None
            and not isinstance(self.vector_filter_used, bool)
        ):
            raise ValueError("HGraph probe evidence is invalid")
        for observation in self.ordered_raw_scores:
            if not isinstance(observation, tuple) or len(observation) != 2:
                raise ValueError("HGraph probe evidence is invalid")
            identifier, raw_score = observation
            if (
                isinstance(identifier, bool)
                or not isinstance(identifier, int)
                or isinstance(raw_score, bool)
                or not isinstance(raw_score, (int, float))
                or not math.isfinite(raw_score)
            ):
                raise ValueError("HGraph probe evidence is invalid")


@dataclass(frozen=True)
class ProbeResult:
    kind: ProbeKind
    status: ProbeStatus
    blocking: bool
    detail_code: str
    evidence: HGraphProbeEvidence | None = None

    def __post_init__(self) -> None:
        if _DETAIL_CODE_PATTERN.fullmatch(self.detail_code) is None:
            raise ValueError("Probe detail_code must be a non-secret code")
        if self.evidence is not None and (
            self.kind is not ProbeKind.HGRAPH
            or not isinstance(self.evidence, HGraphProbeEvidence)
        ):
            raise ValueError("Probe evidence is invalid")


@dataclass(frozen=True)
class CapabilityReport:
    version: HologresVersion
    results: tuple[ProbeResult, ...] = ()

    def __post_init__(self) -> None:
        seen: set[ProbeKind] = set()
        for result in self.results:
            if result.kind in seen:
                raise ValueError("Capability report contains duplicate probe kinds")
            seen.add(result.kind)
            spec = PROBE_SPECS.get(result.kind)
            if spec is None or result.blocking is not spec.blocking:
                raise ValueError("Probe result blocking metadata is inconsistent")

    def supports(self, kind: ProbeKind) -> bool:
        return any(
            result.kind is kind and result.status is ProbeStatus.PASSED
            for result in self.results
        )

    @property
    def blocking_failures(self) -> tuple[ProbeResult, ...]:
        results_by_kind = {result.kind: result for result in self.results}
        failures: list[ProbeResult] = []
        for kind, spec in PROBE_SPECS.items():
            if not spec.blocking:
                continue
            result = results_by_kind.get(kind)
            if result is None:
                failures.append(_result(kind, ProbeStatus.NOT_RUN, "probe_not_run"))
            elif result.status is not ProbeStatus.PASSED:
                failures.append(result)
        return tuple(failures)


def parse_hologres_version(raw_version: str) -> HologresVersion:
    """Parse a recognizable Hologres version without accepting PostgreSQL alone."""

    if not isinstance(raw_version, str):
        raise HologresVersionError("Unrecognized Hologres server version")
    match = _VERSION_PATTERN.search(raw_version)
    if match is None:
        raise HologresVersionError("Unrecognized Hologres server version")
    return HologresVersion(
        major=int(match.group("major")),
        minor=int(match.group("minor")),
        patch=int(match.group("patch") or 0),
    )


def validate_hologres_version(raw_version: str) -> HologresVersion:
    """Require a recognizable Hologres 5.0+ server, failing closed."""

    version = parse_hologres_version(raw_version)
    if version < HologresVersion(5, 0, 0):
        raise HologresVersionError("Hologres 5.0 or newer is required")
    return version


def validate_test_schema_name(schema: str) -> str:
    """Guard all disposable live-probe DDL with an isolated schema prefix."""

    try:
        validate_identifier(schema)
    except HologresSqlError:
        raise ValueError(
            "Live probes require a unique lightrag_test_* schema"
        ) from None
    if _TEST_SCHEMA_PATTERN.fullmatch(schema) is None:
        raise ValueError("Live probes require a unique lightrag_test_* schema")
    return schema


def _result(
    kind: ProbeKind,
    status: ProbeStatus,
    detail_code: str,
    evidence: HGraphProbeEvidence | None = None,
) -> ProbeResult:
    return ProbeResult(
        kind=kind,
        status=status,
        blocking=PROBE_SPECS[kind].blocking,
        detail_code=detail_code,
        evidence=evidence,
    )


async def probe_production_capabilities(
    client: HologresClient,
) -> CapabilityReport:
    """Run only production-safe, read-only version validation."""

    raw_version = await client.fetch_value(
        "SELECT hg_version()", descriptor="capability.version"
    )
    version = validate_hologres_version(raw_version)
    return CapabilityReport(version=version)


async def _probe_setup_reset_bindings(
    client: HologresClient, schema: str
) -> ProbeResult:
    marker = "lightrag_test_probe"
    try:
        observation = await client._observe_setup_reset_for_probe(
            bound=37, marker=marker
        )
        if observation.schema_name != schema or observation.bound != 37:
            return _result(
                ProbeKind.ASYNCPG_SETUP_RESET_BINDINGS,
                ProbeStatus.FAILED,
                "setup_binding_mismatch",
            )
        if not observation.same_connection_reacquired:
            return _result(
                ProbeKind.ASYNCPG_SETUP_RESET_BINDINGS,
                ProbeStatus.FAILED,
                "reset_connection_not_reacquired",
            )
        if observation.application_name == marker:
            return _result(
                ProbeKind.ASYNCPG_SETUP_RESET_BINDINGS,
                ProbeStatus.FAILED,
                "reset_not_observed",
            )
        return _result(
            ProbeKind.ASYNCPG_SETUP_RESET_BINDINGS,
            ProbeStatus.PASSED,
            "setup_reset_bindings_ok",
        )
    except Exception:
        return _result(
            ProbeKind.ASYNCPG_SETUP_RESET_BINDINGS,
            ProbeStatus.FAILED,
            "setup_reset_bindings_failed",
        )


async def _probe_basic_types(
    client: HologresClient,
    schema: str,
    table: str,
    verify_ownership: Callable[[], Awaitable[None]],
) -> ProbeResult:
    qualified_table = quote_qualified_identifier(schema, table)
    try:
        await verify_ownership()
        await client.execute_one(
            f"CREATE TABLE {qualified_table} ("
            "id text PRIMARY KEY, payload jsonb NOT NULL, tags text[] NOT NULL)",
            descriptor="probe.basic.create",
            replay_safe=False,
        )
        upsert_sql = (
            f"INSERT INTO {qualified_table} (id, payload, tags) "
            "VALUES ($1, $2::jsonb, $3::text[]) "
            "ON CONFLICT (id) DO UPDATE SET "
            "payload = EXCLUDED.payload, tags = EXCLUDED.tags"
        )
        await verify_ownership()
        await client.execute_one(
            upsert_sql,
            "probe-row",
            '{"step": 1}',
            ["alpha"],
            descriptor="probe.basic.insert",
            replay_safe=True,
        )
        await verify_ownership()
        await client.execute_one(
            upsert_sql,
            "probe-row",
            '{"step": 2}',
            ["beta", "gamma"],
            descriptor="probe.basic.upsert",
            replay_safe=True,
        )
        await verify_ownership()
        row = await client.fetch_one(
            f"SELECT payload, tags FROM {qualified_table} WHERE id = $1",
            "probe-row",
            descriptor="probe.basic.fetch",
        )
        payload = row["payload"] if row is not None else None
        if isinstance(payload, str):
            payload = json.loads(payload)
        if (
            payload != {"step": 2}
            or row is None
            or list(row["tags"]) != ["beta", "gamma"]
        ):
            return _result(
                ProbeKind.JSONB_ON_CONFLICT_ARRAYS_RECONNECT,
                ProbeStatus.FAILED,
                "basic_type_mismatch",
            )
        await verify_ownership()
        await client._reconnect_for_probe()
        await verify_ownership()
        count = await client.fetch_value(
            f"SELECT count(*) FROM {qualified_table} WHERE id = $1",
            "probe-row",
            descriptor="probe.reconnect.verify",
        )
        if count != 1:
            return _result(
                ProbeKind.JSONB_ON_CONFLICT_ARRAYS_RECONNECT,
                ProbeStatus.FAILED,
                "reconnect_mismatch",
            )
        return _result(
            ProbeKind.JSONB_ON_CONFLICT_ARRAYS_RECONNECT,
            ProbeStatus.PASSED,
            "basic_types_reconnect_ok",
        )
    except HologresProbeError:
        raise
    except Exception:
        return _result(
            ProbeKind.JSONB_ON_CONFLICT_ARRAYS_RECONNECT,
            ProbeStatus.FAILED,
            "basic_types_reconnect_failed",
        )


async def _probe_hgraph(
    client: HologresClient,
    schema: str,
    table: str,
    verify_ownership: Callable[[], Awaitable[None]],
) -> ProbeResult:
    qualified_table = quote_qualified_identifier(schema, table)
    # extra_columns is deliberately absent so the probe DDL stays identical
    # to the shared vector table, whose text id column Hologres rejects there.
    vector_properties = json.dumps(
        {
            "embedding": {
                "algorithm": "HGraph",
                "distance_method": "Cosine",
                "builder_params": {
                    "max_degree": 64,
                    "ef_construction": 400,
                    "base_quantization_type": "fp32",
                    "precise_quantization_type": "fp32",
                    "use_reorder": True,
                },
            }
        },
        separators=(",", ":"),
    )
    create_sql = (
        f"CREATE TABLE {qualified_table} ("
        "id bigint PRIMARY KEY, "
        "embedding float4[] NOT NULL, "
        "CHECK (array_ndims(embedding) = 1 AND "
        "array_length(embedding, 1) = 3)"
        ") WITH (orientation = 'column', "
        f"vectors = '{vector_properties}')"
    )
    upsert_sql = (
        f"INSERT INTO {qualified_table} (id, embedding) VALUES "
        "($1, $2::float4[]), ($3, $4::float4[]), "
        "($5, $6::float4[]), ($7, $8::float4[]) "
        "ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding"
    )
    query_sql = (
        "SELECT id, approx_cosine_distance(embedding, $1::float4[]) "
        f"AS raw_score FROM {qualified_table} "
        "ORDER BY raw_score DESC, id ASC LIMIT 4"
    )

    try:
        await verify_ownership()
        await client.execute_one(
            create_sql,
            descriptor="probe.hgraph.create",
            replay_safe=False,
        )
        await verify_ownership()
        await client.execute_one(
            upsert_sql,
            1,
            [1.0, 0.0, 0.0],
            2,
            [0.0, 1.0, 0.0],
            3,
            [-1.0, 0.0, 0.0],
            4,
            [1.0, 0.0, 0.0],
            descriptor="probe.hgraph.insert",
            replay_safe=True,
        )
        await verify_ownership()
        await client.fetch_value(
            "SELECT hologres.hg_full_compact_table($1, $2)",
            f"{schema}.{table}",
            "max_file_size_mb=4096",
            descriptor="probe.hgraph.compact",
            operation_kind=OperationKind.WRITE,
            replay_safe=False,
        )
        await verify_ownership()
        raw_properties = await client.fetch_value(
            "SELECT property_value FROM hologres.hg_table_properties "
            "WHERE table_namespace = $1 AND table_name = $2 "
            "AND property_key = $3",
            schema,
            table,
            "vectors",
            descriptor="probe.hgraph.catalog",
        )
        if isinstance(raw_properties, str):
            try:
                raw_properties = json.loads(raw_properties)
            except (TypeError, ValueError, json.JSONDecodeError):
                return _result(
                    ProbeKind.HGRAPH,
                    ProbeStatus.FAILED,
                    "hgraph_catalog_mismatch",
                )
        if not isinstance(raw_properties, dict):
            return _result(
                ProbeKind.HGRAPH, ProbeStatus.FAILED, "hgraph_catalog_mismatch"
            )
        embedding_properties = raw_properties.get("embedding")
        if not isinstance(embedding_properties, dict) or (
            embedding_properties.get("algorithm") != "HGraph"
            or embedding_properties.get("distance_method") != "Cosine"
        ):
            return _result(
                ProbeKind.HGRAPH, ProbeStatus.FAILED, "hgraph_catalog_mismatch"
            )

        await verify_ownership()
        raw_rows = await client.fetch_all(
            query_sql,
            [1.0, 0.0, 0.0],
            descriptor="probe.hgraph.query",
        )
        try:
            rows = list(raw_rows)
        except (TypeError, ValueError):
            return _result(
                ProbeKind.HGRAPH, ProbeStatus.FAILED, "hgraph_query_mismatch"
            )
        if len(rows) != 4:
            return _result(
                ProbeKind.HGRAPH, ProbeStatus.FAILED, "hgraph_query_mismatch"
            )
        observed: list[tuple[int, float]] = []
        for row in rows:
            try:
                identifier = row["id"]
                raw_score = row["raw_score"]
            except (KeyError, TypeError, IndexError):
                return _result(
                    ProbeKind.HGRAPH, ProbeStatus.FAILED, "hgraph_query_mismatch"
                )
            if (
                isinstance(identifier, bool)
                or not isinstance(identifier, int)
                or isinstance(raw_score, bool)
                or not isinstance(raw_score, (int, float))
                or not math.isfinite(raw_score)
            ):
                return _result(
                    ProbeKind.HGRAPH, ProbeStatus.FAILED, "hgraph_query_mismatch"
                )
            observed.append((identifier, float(raw_score)))
        scores = dict(observed)
        if (
            [identifier for identifier, _score in observed] != [1, 4, 2, 3]
            or set(scores) != {1, 2, 3, 4}
            or scores[1] != scores[4]
            or not scores[1] > scores[2] > scores[3]
        ):
            return _result(
                ProbeKind.HGRAPH,
                ProbeStatus.FAILED,
                "hgraph_query_mismatch",
                HGraphProbeEvidence(
                    ordered_raw_scores=tuple(observed),
                    vector_filter_used=None,
                ),
            )
        # Freeze the score contract: approx_cosine_distance(stored, query)
        # under distance_method=Cosine returns cosine SIMILARITY (higher is
        # closer), proven here against known vectors, not inferred from docs.
        expected_scores = {1: 1.0, 4: 1.0, 2: 0.0, 3: -1.0}
        if any(
            abs(scores[identifier] - expected) > 1e-3
            for identifier, expected in expected_scores.items()
        ):
            return _result(
                ProbeKind.HGRAPH,
                ProbeStatus.FAILED,
                "hgraph_score_contract_mismatch",
                HGraphProbeEvidence(
                    ordered_raw_scores=tuple(observed),
                    vector_filter_used=None,
                ),
            )

        await verify_ownership()
        vector_filter_used = False
        try:
            raw_plan_rows = await client.fetch_all(
                f"EXPLAIN {query_sql}",
                [1.0, 0.0, 0.0],
                descriptor="probe.hgraph.explain",
            )
            plan_rows = list(raw_plan_rows)
            plan_text = "\n".join(
                str(value) for row in plan_rows for value in row.values()
            )
            vector_filter_used = "Vector Filter" in plan_text
        except Exception:
            pass

        await verify_ownership()
        await client.execute_one(
            f"INSERT INTO {qualified_table} (id, embedding) "
            "VALUES ($1, $2::float4[]) "
            "ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding",
            1,
            [0.0, 0.0, 1.0],
            descriptor="probe.hgraph.update",
            replay_safe=True,
        )
        await verify_ownership()
        updated = await client.fetch_one(
            f"SELECT embedding FROM {qualified_table} WHERE id = $1",
            1,
            descriptor="probe.hgraph.update.verify",
        )
        try:
            updated_embedding = (
                None if updated is None else list(updated["embedding"])
            )
        except (KeyError, TypeError, ValueError, IndexError):
            updated_embedding = None
        if updated_embedding != [0.0, 0.0, 1.0]:
            return _result(
                ProbeKind.HGRAPH, ProbeStatus.FAILED, "hgraph_update_mismatch"
            )

        await verify_ownership()
        await client.execute_one(
            f"DELETE FROM {qualified_table} WHERE id = $1",
            4,
            descriptor="probe.hgraph.delete",
            replay_safe=True,
        )
        await verify_ownership()
        remaining = await client.fetch_value(
            f"SELECT count(*) FROM {qualified_table} WHERE id = $1",
            4,
            descriptor="probe.hgraph.delete.verify",
        )
        if isinstance(remaining, bool) or remaining != 0:
            return _result(
                ProbeKind.HGRAPH, ProbeStatus.FAILED, "hgraph_delete_mismatch"
            )
        return _result(
            ProbeKind.HGRAPH,
            ProbeStatus.PASSED,
            "hgraph_semantics_frozen",
            HGraphProbeEvidence(
                ordered_raw_scores=tuple(observed),
                vector_filter_used=vector_filter_used,
            ),
        )
    except HologresProbeError:
        raise
    except Exception:
        return _result(
            ProbeKind.HGRAPH, ProbeStatus.FAILED, "hgraph_probe_failed"
        )


async def _probe_graph_partition_adjacency(
    client: HologresClient,
    schema: str,
    nodes_table: str,
    edges_table: str,
    verify_ownership: Callable[[], Awaitable[None]],
) -> tuple[ProbeResult, ProbeResult]:
    """Freeze the graph-table contract live: LOGICAL PARTITION + adjacency plan.

    The first result covers ProbeKind.LOGICAL_PARTITION (partitioned DDL is
    accepted, workspace isolation holds, jsonb ``||`` merge-upsert semantics,
    and bytewise "C" ordering that matches Python code-point comparison). The
    second covers ProbeKind.GRAPH_ADJACENCY_EXPLAIN (self-loop-counts-twice
    degree, generate_series pair-batch matching, undirected adjacency UNION,
    and an EXPLAIN proving the workspace partition filter prunes the scan).
    """

    qualified_nodes = quote_qualified_identifier(schema, nodes_table)
    qualified_edges = quote_qualified_identifier(schema, edges_table)
    adjacency_not_run = _result(
        ProbeKind.GRAPH_ADJACENCY_EXPLAIN,
        ProbeStatus.NOT_RUN,
        "graph_tables_unavailable",
    )

    def _partition_failed(detail_code: str) -> tuple[ProbeResult, ProbeResult]:
        return (
            _result(ProbeKind.LOGICAL_PARTITION, ProbeStatus.FAILED, detail_code),
            adjacency_not_run,
        )

    workspace = "lightrag_probe_w1"
    other_workspace = "lightrag_probe_w2"
    namespace = "graph"
    # Ids chosen so bytewise ("C") order differs from any case-insensitive or
    # locale order: 'B' (0x42) < '_x' (0x5F) < 'a' (0x61).
    ids = ["B", "_x", "a"]

    adjacency_sql = (
        f"SELECT tgt_id AS nid FROM {qualified_edges} "
        "WHERE workspace = $1 AND namespace = $2 AND src_id = $3 "
        "UNION "
        f"SELECT src_id AS nid FROM {qualified_edges} "
        "WHERE workspace = $1 AND namespace = $2 AND tgt_id = $3"
    )

    try:
        collation = await client.fetch_value(
            "SELECT datcollate FROM pg_database "
            "WHERE datname = current_database()",
            descriptor="probe.graph.collation",
        )
        if collation != "C":
            return _partition_failed("graph_collation_mismatch")

        await verify_ownership()
        await client.execute_one(
            f"CREATE TABLE {qualified_nodes} ("
            "workspace text NOT NULL, "
            "namespace text NOT NULL, "
            "id text NOT NULL, "
            "properties jsonb NOT NULL, "
            "updated_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP, "
            "PRIMARY KEY (workspace, namespace, id)"
            ") LOGICAL PARTITION BY LIST (workspace) "
            "WITH (orientation = 'row', distribution_key = 'namespace,id')",
            descriptor="probe.graph.nodes.create",
            replay_safe=False,
        )
        await verify_ownership()
        await client.execute_one(
            f"CREATE TABLE {qualified_edges} ("
            "workspace text NOT NULL, "
            "namespace text NOT NULL, "
            "src_id text NOT NULL, "
            "tgt_id text NOT NULL, "
            "properties jsonb NOT NULL, "
            "updated_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP, "
            "PRIMARY KEY (workspace, namespace, src_id, tgt_id)"
            ") LOGICAL PARTITION BY LIST (workspace) "
            "WITH (orientation = 'row', distribution_key = 'namespace,src_id')",
            descriptor="probe.graph.edges.create",
            replay_safe=False,
        )

        await verify_ownership()
        await client.execute_one(
            f"INSERT INTO {qualified_nodes} (workspace, namespace, id, properties) "
            "VALUES ($1, $2, $3, $4::jsonb), ($1, $2, $5, $6::jsonb), "
            "($1, $2, $7, $8::jsonb), ($9, $2, $3, $4::jsonb) "
            "ON CONFLICT (workspace, namespace, id) "
            "DO UPDATE SET properties = EXCLUDED.properties",
            workspace,
            namespace,
            ids[0],
            '{"keep":"x","step":1}',
            ids[1],
            '{"step":1}',
            ids[2],
            '{"step":1}',
            other_workspace,
            descriptor="probe.graph.nodes.insert",
            replay_safe=True,
        )

        # jsonb || merge-upsert: the omitted "keep" key must survive.
        await verify_ownership()
        await client.execute_one(
            f"INSERT INTO {qualified_nodes} AS current "
            "(workspace, namespace, id, properties) "
            "VALUES ($1, $2, $3, $4::jsonb) "
            "ON CONFLICT (workspace, namespace, id) "
            "DO UPDATE SET properties = current.properties "
            "|| EXCLUDED.properties",
            workspace,
            namespace,
            ids[0],
            '{"step":2}',
            descriptor="probe.graph.nodes.merge",
            replay_safe=True,
        )
        await verify_ownership()
        merged = await client.fetch_value(
            f"SELECT properties FROM {qualified_nodes} "
            "WHERE workspace = $1 AND namespace = $2 AND id = $3",
            workspace,
            namespace,
            ids[0],
            descriptor="probe.graph.nodes.merge.verify",
        )
        if isinstance(merged, str):
            try:
                merged = json.loads(merged)
            except (TypeError, ValueError, json.JSONDecodeError):
                return _partition_failed("graph_merge_semantics_mismatch")
        if merged != {"keep": "x", "step": 2}:
            return _partition_failed("graph_merge_semantics_mismatch")

        # Workspace isolation plus bytewise ordering in one query.
        await verify_ownership()
        ordered_rows = await client.fetch_all(
            f"SELECT id FROM {qualified_nodes} "
            "WHERE workspace = $1 AND namespace = $2 ORDER BY id",
            workspace,
            namespace,
            descriptor="probe.graph.nodes.order",
        )
        try:
            ordered_ids = [row["id"] for row in ordered_rows]
        except (KeyError, TypeError):
            return _partition_failed("graph_order_mismatch")
        if ordered_ids != sorted(ids):
            return _partition_failed("graph_order_mismatch")

        partition_result = _result(
            ProbeKind.LOGICAL_PARTITION,
            ProbeStatus.PASSED,
            "logical_partition_semantics_frozen",
        )
    except HologresProbeError:
        raise
    except Exception:
        return _partition_failed("graph_probe_failed")

    def _adjacency_failed(detail_code: str) -> tuple[ProbeResult, ProbeResult]:
        return (
            partition_result,
            _result(
                ProbeKind.GRAPH_ADJACENCY_EXPLAIN, ProbeStatus.FAILED, detail_code
            ),
        )

    try:
        await verify_ownership()
        await client.execute_one(
            f"INSERT INTO {qualified_edges} "
            "(workspace, namespace, src_id, tgt_id, properties) "
            "VALUES ($1, $2, $3, $4, $5::jsonb), ($1, $2, $4, $6, $5::jsonb), "
            "($1, $2, $3, $3, $5::jsonb), ($7, $2, $3, $4, $5::jsonb) "
            "ON CONFLICT (workspace, namespace, src_id, tgt_id) "
            "DO UPDATE SET properties = EXCLUDED.properties",
            workspace,
            namespace,
            ids[0],
            ids[1],
            '{"weight":1}',
            ids[2],
            other_workspace,
            descriptor="probe.graph.edges.insert",
            replay_safe=True,
        )

        # Degree of 'B' in w1: edge (B,_x) once + self-loop (B,B) twice = 3.
        await verify_ownership()
        degree = await client.fetch_value(
            "SELECT count(*) FROM ("
            f"SELECT src_id AS id FROM {qualified_edges} "
            "WHERE workspace = $1 AND namespace = $2 AND src_id = $3 "
            "UNION ALL "
            f"SELECT tgt_id AS id FROM {qualified_edges} "
            "WHERE workspace = $1 AND namespace = $2 AND tgt_id = $3"
            ") sub",
            workspace,
            namespace,
            ids[0],
            descriptor="probe.graph.degree",
        )
        if isinstance(degree, bool) or degree != 3:
            return _adjacency_failed("graph_degree_mismatch")

        # Pair-batch matching via generate_series array subscripts (the
        # live-proven substitute for multi-argument UNNEST).
        await verify_ownership()
        matched = await client.fetch_value(
            f"SELECT count(*) FROM {qualified_edges} e "
            "JOIN (SELECT ($3::text[])[g.idx] AS src, ($4::text[])[g.idx] AS tgt "
            "FROM generate_series(1, $5::int) AS g(idx)) p "
            "ON p.src = e.src_id AND p.tgt = e.tgt_id "
            "WHERE e.workspace = $1 AND e.namespace = $2",
            workspace,
            namespace,
            [ids[0], ids[1]],
            [ids[1], ids[2]],
            2,
            descriptor="probe.graph.pairs",
        )
        if isinstance(matched, bool) or matched != 2:
            return _adjacency_failed("graph_pair_batch_mismatch")

        # Undirected adjacency of 'B' in w1: {_x (outgoing), B (self-loop)}.
        # The w2 copy of edge (B, _x) must not leak in.
        await verify_ownership()
        neighbour_rows = await client.fetch_all(
            adjacency_sql,
            workspace,
            namespace,
            ids[0],
            descriptor="probe.graph.adjacency",
        )
        try:
            neighbours = {row["nid"] for row in neighbour_rows}
        except (KeyError, TypeError):
            return _adjacency_failed("graph_adjacency_mismatch")
        if neighbours != {ids[0], ids[1]}:
            return _adjacency_failed("graph_adjacency_mismatch")

        await verify_ownership()
        raw_plan_rows = await client.fetch_all(
            f"EXPLAIN {adjacency_sql}",
            workspace,
            namespace,
            ids[0],
            descriptor="probe.graph.explain",
        )
        plan_text = "\n".join(
            str(value) for row in raw_plan_rows for value in row.values()
        )
        if "Partition Filter" not in plan_text:
            return _adjacency_failed("graph_adjacency_plan_mismatch")

        return (
            partition_result,
            _result(
                ProbeKind.GRAPH_ADJACENCY_EXPLAIN,
                ProbeStatus.PASSED,
                "graph_adjacency_plan_partition_pruned",
            ),
        )
    except HologresProbeError:
        raise
    except Exception:
        return _adjacency_failed("graph_probe_failed")


async def _verify_probe_ownership(
    client: HologresClient,
    schema: str,
    marker_table: str,
    owner_token: str,
) -> None:
    qualified_marker = quote_qualified_identifier(schema, marker_table)
    try:
        observed_token = await client.fetch_value(
            f"SELECT ownership_token FROM {qualified_marker} "
            "WHERE ownership_token = $1",
            owner_token,
            descriptor="probe.marker.verify",
        )
    except Exception:
        raise HologresProbeError(
            "Unable to verify isolated Hologres probe schema ownership",
            leaked_objects=(schema,),
        ) from None
    if observed_token != owner_token:
        raise HologresProbeError(
            "Isolated Hologres probe schema ownership marker does not match",
            leaked_objects=(schema,),
        )


async def _cleanup_owned_probe_schema(
    client: HologresClient,
    schema: str,
    marker_table: str,
    owner_token: str,
    probe_tables: tuple[tuple[str, str], ...],
) -> None:
    try:
        for table, descriptor in probe_tables:
            await _verify_probe_ownership(client, schema, marker_table, owner_token)
            await client.execute_one(
                f"DROP TABLE IF EXISTS {quote_qualified_identifier(schema, table)}",
                descriptor=descriptor,
                replay_safe=True,
            )
        await _verify_probe_ownership(client, schema, marker_table, owner_token)
        await client.execute_one(
            "DROP TABLE IF EXISTS "
            f"{quote_qualified_identifier(schema, marker_table)}",
            descriptor="probe.marker.drop",
            replay_safe=True,
        )
        await client.execute_one(
            f"DROP SCHEMA {quote_qualified_identifier(schema)}",
            descriptor="probe.schema.drop",
            replay_safe=True,
        )
    except HologresProbeError:
        raise
    except Exception:
        raise HologresProbeError(
            "Isolated Hologres probe cleanup failed",
            leaked_objects=(schema,),
        ) from None


async def run_initial_isolated_probes(
    client: HologresClient, schema: str | None = None
) -> CapabilityReport:
    """Run initial probes only while this invocation can prove schema ownership."""

    guarded_schema = validate_test_schema_name(
        schema if schema is not None else f"lightrag_test_{uuid.uuid4().hex}"
    )
    version_report = await probe_production_capabilities(client)
    try:
        schema_exists = await client.fetch_value(
            "SELECT EXISTS (SELECT 1 FROM pg_namespace WHERE nspname = $1)",
            guarded_schema,
            descriptor="probe.schema.preflight",
        )
    except Exception:
        raise HologresProbeError(
            "Unable to verify isolated Hologres probe schema ownership"
        ) from None
    if schema_exists:
        raise HologresProbeError(
            "Refusing to use a pre-existing Hologres probe schema"
        )

    marker_table = f"lightrag_test_owner_{uuid.uuid4().hex}"
    owner_token = uuid.uuid4().hex
    table = f"lightrag_test_basic_{uuid.uuid4().hex}"
    hgraph_table = f"lightrag_test_hgraph_{uuid.uuid4().hex}"
    graph_nodes_table = f"lightrag_test_gnodes_{uuid.uuid4().hex}"
    graph_edges_table = f"lightrag_test_gedges_{uuid.uuid4().hex}"
    qualified_schema = quote_qualified_identifier(guarded_schema)
    qualified_marker = quote_qualified_identifier(guarded_schema, marker_table)
    try:
        await client.execute_one(
            f"CREATE SCHEMA {qualified_schema}",
            descriptor="probe.schema.create",
            replay_safe=False,
        )
    except Exception:
        raise HologresProbeError(
            "Isolated Hologres probe schema creation outcome is unknown",
            leaked_objects=(guarded_schema,),
        ) from None

    try:
        await client.execute_one(
            f"CREATE TABLE {qualified_marker} (ownership_token text PRIMARY KEY)",
            descriptor="probe.marker.create",
            replay_safe=False,
        )
        await client.execute_one(
            f"INSERT INTO {qualified_marker} (ownership_token) VALUES ($1)",
            owner_token,
            descriptor="probe.marker.insert",
            replay_safe=False,
        )
    except Exception:
        raise HologresProbeError(
            "Isolated Hologres probe ownership marker outcome is unknown",
            leaked_objects=(guarded_schema,),
        ) from None

    async def verify_ownership() -> None:
        await _verify_probe_ownership(
            client, guarded_schema, marker_table, owner_token
        )

    await verify_ownership()
    results = [
        _result(
            ProbeKind.SINGLE_AUTOCOMMIT_DDL,
            ProbeStatus.PASSED,
            "autocommit_ddl_ok",
        )
    ]
    try:
        results.append(await _probe_setup_reset_bindings(client, guarded_schema))
        await verify_ownership()
        results.append(
            await _probe_basic_types(
                client, guarded_schema, table, verify_ownership
            )
        )
        await verify_ownership()
        results.append(
            await _probe_hgraph(
                client, guarded_schema, hgraph_table, verify_ownership
            )
        )
        await verify_ownership()
        results.extend(
            await _probe_graph_partition_adjacency(
                client,
                guarded_schema,
                graph_nodes_table,
                graph_edges_table,
                verify_ownership,
            )
        )
    finally:
        await _cleanup_owned_probe_schema(
            client,
            guarded_schema,
            marker_table,
            owner_token,
            (
                (table, "probe.basic.drop"),
                (hgraph_table, "probe.hgraph.drop"),
                (graph_edges_table, "probe.graph.edges.drop"),
                (graph_nodes_table, "probe.graph.nodes.drop"),
            ),
        )

    return CapabilityReport(version=version_report.version, results=tuple(results))
