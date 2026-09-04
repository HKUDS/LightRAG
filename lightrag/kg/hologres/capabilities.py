"""Hologres version validation and isolated capability probes."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
import json
import re
import uuid

from .client import (
    HologresClient,
    HologresSqlError,
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
class ProbeResult:
    kind: ProbeKind
    status: ProbeStatus
    blocking: bool
    detail_code: str

    def __post_init__(self) -> None:
        if _DETAIL_CODE_PATTERN.fullmatch(self.detail_code) is None:
            raise ValueError("Probe detail_code must be a non-secret code")


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


def _result(kind: ProbeKind, status: ProbeStatus, detail_code: str) -> ProbeResult:
    return ProbeResult(
        kind=kind,
        status=status,
        blocking=PROBE_SPECS[kind].blocking,
        detail_code=detail_code,
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
    probe_tables: tuple[str, ...],
) -> None:
    try:
        for table in probe_tables:
            await _verify_probe_ownership(client, schema, marker_table, owner_token)
            await client.execute_one(
                f"DROP TABLE IF EXISTS {quote_qualified_identifier(schema, table)}",
                descriptor="probe.basic.drop",
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
    finally:
        await _cleanup_owned_probe_schema(
            client,
            guarded_schema,
            marker_table,
            owner_token,
            (table,),
        )

    return CapabilityReport(version=version_report.version, results=tuple(results))
