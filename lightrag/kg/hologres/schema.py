"""Resumable, autocommit-only schema management for the Hologres backend.

Every state mutation performed here is exactly one parameterized statement
issued through the restricted client, so an interrupted initializer never
leaves a partially visible change behind. Progress is recorded in a shared
control table (the "ledger") whose rows form a small state machine:

``prepared -> applying -> verifying -> applied`` with ``failed`` for refusals.

The ledger is claimed with a single compare-and-set upsert that also copies
the pre-claim state into ``previous_state``, which lets one statement report
both the new lease and the state it replaced. That is what makes crash
recovery decidable without reading and claiming in two racy steps.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
import hashlib
import re
from typing import Any
import uuid

from ...utils import logger
from .client import (
    HologresSqlError,
    OperationKind,
    _is_lexically_read_only,
    _scan_statement_tokens,
    _top_level_tokens,
    quote_qualified_identifier,
    validate_identifier,
    validate_single_statement,
)


LEDGER_TABLE_NAME = "lightrag_hologres_schema_ledger"

_MAX_ERROR_SUMMARY = 200
_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SUMMARY_ALLOWED = re.compile(r"[^A-Za-z0-9_.:| -]")
_ALLOWED_CREATE_OBJECTS = frozenset({"SCHEMA", "TABLE", "INDEX", "UNIQUE"})
_FORBIDDEN_ALTER_TOKENS = frozenset(
    {"ALTER", "DROP", "INHERIT", "OWNER", "RENAME", "SET", "TYPE"}
)
_OFFLINE_MIGRATION_HINT = (
    "requires an offline migration because automatic schema changes are "
    "restricted to additive, expand-only statements"
)

_LEDGER_COLUMNS = (
    "component, version, step, descriptor_name, digest, state, "
    "previous_state, owner_token, lease_expires_at, error_summary"
)


class SchemaState(str, Enum):
    """Ledger lifecycle states for one schema descriptor."""

    PREPARED = "prepared"
    APPLYING = "applying"
    VERIFYING = "verifying"
    APPLIED = "applied"
    FAILED = "failed"


_VALID_STATES = frozenset(state.value for state in SchemaState)
_RESUME_STATES = frozenset(
    {
        SchemaState.APPLYING.value,
        SchemaState.VERIFYING.value,
        SchemaState.FAILED.value,
    }
)


class HologresSchemaError(RuntimeError):
    """Base class for sanitized Hologres schema management failures."""


class HologresSchemaDefinitionError(HologresSchemaError):
    """Raised before any database work when a descriptor or plan is invalid."""


class HologresOfflineMigrationRequired(HologresSchemaDefinitionError):
    """Raised when a change cannot be applied by automatic migration."""


class HologresSchemaDriftError(HologresSchemaError):
    """Raised when a recorded descriptor digest no longer matches the plan."""


class HologresSchemaStateError(HologresSchemaError):
    """Raised when ledger state or catalog verification fails closed."""


class HologresSchemaBusyError(HologresSchemaError):
    """Raised when another initializer holds an unexpired ledger lease."""


class _MalformedCondition(Exception):
    """Internal signal for a condition query that did not return a boolean."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _validate_name(value: Any, label: str) -> str:
    if not isinstance(value, str) or _NAME.fullmatch(value) is None:
        raise HologresSchemaDefinitionError(f"Invalid schema descriptor {label}")
    return value


def _validate_ordinal(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise HologresSchemaDefinitionError(f"Invalid schema descriptor {label}")
    return value


def _validated_schema(schema: Any) -> str:
    try:
        return validate_identifier(schema)
    except HologresSqlError as error:
        raise HologresSchemaDefinitionError("Invalid schema identifier") from error


def _guarded_statement(sql: Any, label: str) -> str:
    try:
        return validate_single_statement(sql)
    except HologresSqlError as error:
        raise HologresSchemaDefinitionError(
            f"Invalid schema descriptor {label}: {error}"
        ) from error


def _validate_read_only(sql: Any, label: str) -> str:
    _guarded_statement(sql, label)
    try:
        read_only = _is_lexically_read_only(sql)
    except HologresSqlError as error:
        raise HologresSchemaDefinitionError(
            f"Invalid schema descriptor {label}: {error}"
        ) from error
    if not read_only:
        raise HologresSchemaDefinitionError(
            f"Schema descriptor {label} must be lexically read-only"
        )
    return sql


def _contains_sequence(tokens: Sequence[Any], expected: Sequence[str]) -> bool:
    limit = len(tokens) - len(expected)
    for start in range(limit + 1):
        if tuple(tokens[start : start + len(expected)]) == tuple(expected):
            return True
    return False


def _reject_offline() -> None:
    raise HologresOfflineMigrationRequired(
        f"Schema descriptor statement {_OFFLINE_MIGRATION_HINT}"
    )


def _validate_additive_statement(sql: str) -> None:
    """Reject anything that is not an additive, expand-only change.

    The classifier is lexical and deliberately conservative: only guarded
    ``CREATE ... IF NOT EXISTS``, ``ALTER TABLE ... ADD COLUMN IF NOT EXISTS``
    and ``COMMENT ON`` shapes are accepted. Destructive statements, renames,
    type changes and table rebuilds are refused with an offline-migration hint.
    """

    tokens = _top_level_tokens(_scan_statement_tokens(sql))
    if not tokens:
        _reject_offline()
    command = tokens[0]

    if command == "CREATE":
        target = tokens[1] if len(tokens) > 1 else None
        if target not in _ALLOWED_CREATE_OBJECTS:
            _reject_offline()
        if "SELECT" in tokens:
            _reject_offline()
        if not _contains_sequence(tokens, ("IF", "NOT", "EXISTS")):
            _reject_offline()
        return

    if command == "ALTER":
        if len(tokens) < 2 or tokens[1] != "TABLE":
            _reject_offline()
        if not _contains_sequence(tokens, ("ADD", "COLUMN", "IF", "NOT", "EXISTS")):
            _reject_offline()
        if any(token in _FORBIDDEN_ALTER_TOKENS for token in tokens[2:]):
            _reject_offline()
        return

    if command == "COMMENT":
        if len(tokens) < 2 or tokens[1] != "ON":
            _reject_offline()
        return

    _reject_offline()


def _summarize_failure(code: str, error: BaseException | None = None) -> str:
    """Return a bounded, sanitized ledger summary that never carries payloads."""

    text = code if error is None else f"{code}|{type(error).__name__}"
    return _SUMMARY_ALLOWED.sub("", text)[:_MAX_ERROR_SUMMARY]


def _identity_label(identity: tuple[Any, Any, Any, Any]) -> str:
    return f"{identity[0]}/{identity[1]}/{identity[2]}/{identity[3]}"


def _row_identity(row: Any) -> tuple[Any, Any, Any, Any]:
    return (
        row["component"],
        row["version"],
        row["step"],
        row["descriptor_name"],
    )


@dataclass(frozen=True)
class SchemaDescriptor:
    """One additive schema change with a stable identity and digest.

    A descriptor owns exactly one top-level statement, a required read-only
    catalog postcondition, an optional read-only precondition, and an explicit
    replay-safety decision used when resuming after an interrupted apply.
    """

    name: str
    component: str
    version: int
    step: int
    sql: str
    postcondition_sql: str
    postcondition_args: tuple[Any, ...] = ()
    precondition_sql: str | None = None
    precondition_args: tuple[Any, ...] = ()
    replay_safe: bool = False
    digest: str = field(init=False, default="")

    def __post_init__(self) -> None:
        _validate_name(self.name, "name")
        _validate_name(self.component, "component")
        _validate_ordinal(self.version, "version")
        _validate_ordinal(self.step, "step")
        if not isinstance(self.replay_safe, bool):
            raise HologresSchemaDefinitionError(
                "Schema descriptor replay_safe must be an explicit boolean"
            )

        _guarded_statement(self.sql, "statement")
        _validate_additive_statement(self.sql)

        if not isinstance(self.postcondition_sql, str) or not self.postcondition_sql:
            raise HologresSchemaDefinitionError(
                "Schema descriptor requires a catalog postcondition"
            )
        _validate_read_only(self.postcondition_sql, "postcondition")
        object.__setattr__(self, "postcondition_args", tuple(self.postcondition_args))

        if self.precondition_sql is not None:
            _validate_read_only(self.precondition_sql, "precondition")
        object.__setattr__(self, "precondition_args", tuple(self.precondition_args))

        object.__setattr__(self, "digest", self._compute_digest())

    def _compute_digest(self) -> str:
        payload = "\u001f".join(
            (
                "hologres-schema-v1",
                self.component,
                str(self.version),
                str(self.step),
                self.name,
                self.sql,
                self.postcondition_sql,
                repr(tuple(self.postcondition_args)),
                self.precondition_sql or "",
                repr(tuple(self.precondition_args)),
                "1" if self.replay_safe else "0",
            )
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def identity(self) -> tuple[str, int, int, str]:
        return (self.component, self.version, self.step, self.name)

    @property
    def label(self) -> str:
        return f"{self.component}/{self.version}/{self.step}/{self.name}"


@dataclass(frozen=True)
class SchemaApplication:
    """Outcome of reconciling one descriptor against the ledger."""

    component: str
    version: int
    step: int
    descriptor_name: str
    state: SchemaState
    executed_ddl: bool
    resumed: bool


def bootstrap_descriptors(schema: str) -> tuple[SchemaDescriptor, SchemaDescriptor]:
    """Return the narrowly scoped descriptors that create the ledger itself.

    These two statements resolve the bootstrap paradox: the ledger cannot
    record its own creation, so both are idempotent, applied one statement per
    driver call, and verified through the catalog instead of the ledger.
    """

    validated = _validated_schema(schema)
    qualified_schema = quote_qualified_identifier(validated)
    qualified_ledger = quote_qualified_identifier(validated, LEDGER_TABLE_NAME)

    namespace = SchemaDescriptor(
        name="namespace",
        component="bootstrap",
        version=1,
        step=1,
        sql=f"CREATE SCHEMA IF NOT EXISTS {qualified_schema}",
        postcondition_sql=(
            "SELECT EXISTS ("
            "SELECT 1 FROM pg_catalog.pg_namespace WHERE nspname = $1)"
        ),
        postcondition_args=(validated,),
        replay_safe=True,
    )
    ledger = SchemaDescriptor(
        name="ledger",
        component="bootstrap",
        version=1,
        step=2,
        sql=(
            f"CREATE TABLE IF NOT EXISTS {qualified_ledger} ("
            "component text NOT NULL, "
            "version integer NOT NULL, "
            "step integer NOT NULL, "
            "descriptor_name text NOT NULL, "
            "digest text NOT NULL, "
            "state text NOT NULL, "
            "previous_state text, "
            "owner_token text, "
            "lease_expires_at timestamptz, "
            "error_summary text, "
            "created_at timestamptz NOT NULL, "
            "updated_at timestamptz NOT NULL, "
            "PRIMARY KEY (component, version, step, descriptor_name)"
            ") WITH (orientation = 'row')"
        ),
        postcondition_sql=(
            "SELECT EXISTS ("
            "SELECT 1 FROM pg_catalog.pg_class c "
            "JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace "
            "WHERE n.nspname = $1 AND c.relname = $2 AND c.relkind = 'r')"
        ),
        postcondition_args=(validated, LEDGER_TABLE_NAME),
        replay_safe=True,
    )
    return namespace, ledger


def claim_statement(schema: str) -> str:
    """Return the single compare-and-set statement that claims a ledger row.

    ``previous_state`` is copied from the pre-claim row so the caller learns,
    from the same statement, whether it is starting fresh or resuming after an
    interrupted apply. The marker is sticky: re-claiming a row that is already
    parked in ``prepared`` keeps the earlier marker, so a second consecutive
    crash cannot erase the evidence that the statement was already attempted.

    Argument order:
    ``(component, version, step, descriptor_name, digest, owner, lease, now)``.
    """

    ledger = quote_qualified_identifier(_validated_schema(schema), LEDGER_TABLE_NAME)
    return (
        f"INSERT INTO {ledger} AS ledger ("
        "component, version, step, descriptor_name, digest, state, "
        "previous_state, owner_token, lease_expires_at, error_summary, "
        "created_at, updated_at) "
        "VALUES ($1, $2, $3, $4, $5, 'prepared', NULL, $6, $7, NULL, $8, $8) "
        "ON CONFLICT (component, version, step, descriptor_name) DO UPDATE SET "
        "state = 'prepared', "
        "previous_state = CASE "
        "WHEN ledger.state = 'prepared' THEN ledger.previous_state "
        "ELSE ledger.state END, "
        "owner_token = EXCLUDED.owner_token, "
        "lease_expires_at = EXCLUDED.lease_expires_at, "
        "error_summary = NULL, "
        "updated_at = EXCLUDED.updated_at "
        "WHERE ledger.digest = EXCLUDED.digest "
        "AND ledger.state <> 'applied' "
        "AND (ledger.owner_token = EXCLUDED.owner_token "
        "OR ledger.owner_token IS NULL "
        "OR ledger.lease_expires_at IS NULL "
        "OR ledger.lease_expires_at <= EXCLUDED.updated_at) "
        "RETURNING ledger.state, ledger.previous_state, ledger.owner_token"
    )


def transition_statement(schema: str) -> str:
    """Return the single guarded statement that advances one ledger row.

    The update only lands when the caller still owns an unexpired lease over a
    row in the expected state whose digest matches the descriptor, so a stolen
    or expired claim can never advance the state machine.

    Argument order:
    ``(state, error_summary, owner, lease, now, component, version, step,
    descriptor_name, digest, expected_owner, expected_state)``.
    """

    ledger = quote_qualified_identifier(_validated_schema(schema), LEDGER_TABLE_NAME)
    return (
        f"UPDATE {ledger} SET "
        "state = $1, "
        "previous_state = $12, "
        "error_summary = $2, "
        "owner_token = $3, "
        "lease_expires_at = $4, "
        "updated_at = $5 "
        "WHERE component = $6 "
        "AND version = $7 "
        "AND step = $8 "
        "AND descriptor_name = $9 "
        "AND digest = $10 "
        "AND owner_token = $11 "
        "AND state = $12 "
        "AND lease_expires_at > $5 "
        "RETURNING 1"
    )


def load_statement(schema: str) -> str:
    """Return the read-only statement that loads ledger rows for a plan.

    Argument order: ``(components,)``.
    """

    ledger = quote_qualified_identifier(_validated_schema(schema), LEDGER_TABLE_NAME)
    return (
        f"SELECT {_LEDGER_COLUMNS} FROM {ledger} "
        "WHERE component = ANY($1::text[]) "
        "ORDER BY component, version, step, descriptor_name"
    )


def inspect_statement(schema: str) -> str:
    """Return the read-only statement that reads one ledger row.

    Argument order: ``(component, version, step, descriptor_name)``.
    """

    ledger = quote_qualified_identifier(_validated_schema(schema), LEDGER_TABLE_NAME)
    return (
        f"SELECT {_LEDGER_COLUMNS} FROM {ledger} "
        "WHERE component = $1 AND version = $2 AND step = $3 "
        "AND descriptor_name = $4"
    )


class HologresSchemaManager:
    """Apply additive schema descriptors without any explicit transactions.

    The manager only ever calls ``execute_one``, ``fetch_one``, ``fetch_all``
    and ``fetch_value`` on the restricted client. Every ledger mutation is one
    autocommit statement whose predicates carry the expected state, the owning
    token, the descriptor digest and the lease deadline, so two initializers
    racing on the same descriptor cannot both advance it.
    """

    def __init__(
        self,
        client: Any,
        *,
        schema: str,
        owner_token: str | None = None,
        lease_seconds: float = 60.0,
        now_provider: Callable[[], datetime] | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
        busy_retries: int = 30,
        busy_delay: float = 0.5,
    ) -> None:
        self._client = client
        self._schema = _validated_schema(schema)
        if owner_token is not None and (
            not isinstance(owner_token, str) or not owner_token
        ):
            raise HologresSchemaDefinitionError("Invalid ledger owner token")
        self._owner_token = owner_token or uuid.uuid4().hex
        if (
            isinstance(lease_seconds, bool)
            or not isinstance(lease_seconds, (int, float))
            or lease_seconds <= 0
        ):
            raise HologresSchemaDefinitionError("Invalid ledger lease duration")
        self._lease = timedelta(seconds=float(lease_seconds))
        self._now: Callable[[], datetime] = now_provider or _utc_now
        self._sleep: Callable[[float], Awaitable[None]] = sleep or asyncio.sleep
        if isinstance(busy_retries, bool) or not isinstance(busy_retries, int):
            raise HologresSchemaDefinitionError("Invalid ledger busy retry count")
        self._busy_retries = max(0, busy_retries)
        self._busy_delay = float(busy_delay)

        self._bootstrap = bootstrap_descriptors(self._schema)
        self._claim_sql = claim_statement(self._schema)
        self._transition_sql = transition_statement(self._schema)
        self._load_sql = load_statement(self._schema)
        self._inspect_sql = inspect_statement(self._schema)

    @property
    def schema(self) -> str:
        return self._schema

    # -- public API --------------------------------------------------------

    async def bootstrap(self) -> None:
        """Create the schema and the ledger with idempotent single statements.

        The ledger cannot record its own creation, so both steps are verified
        through the catalog and fail closed when the object is not visible.
        """

        for descriptor in self._bootstrap:
            await self._client.execute_one(
                descriptor.sql,
                descriptor="schema.bootstrap.apply",
                replay_safe=descriptor.replay_safe,
            )
            try:
                verified = await self._condition(
                    descriptor.postcondition_sql,
                    descriptor.postcondition_args,
                    "schema.bootstrap.verify",
                )
            except _MalformedCondition as error:
                raise HologresSchemaStateError(
                    f"Hologres schema bootstrap step {descriptor.name} "
                    "verification did not return a boolean; failing closed"
                ) from error
            if not verified:
                raise HologresSchemaStateError(
                    f"Hologres schema bootstrap step {descriptor.name} is not "
                    "visible in the catalog; failing closed"
                )

    async def initialize(
        self, descriptors: Iterable[SchemaDescriptor]
    ) -> tuple[SchemaApplication, ...]:
        """Reconcile ``descriptors`` against the ledger in deterministic order."""

        plan = self._validate_plan(descriptors)
        await self.bootstrap()
        rows = await self._load_ledger(plan)
        recorded = self._classify_ledger(plan, rows)

        results: list[SchemaApplication] = []
        for descriptor in plan:
            results.append(
                await self._apply(descriptor, recorded.get(descriptor.identity))
            )
        return tuple(results)

    # -- planning ---------------------------------------------------------

    def _validate_plan(
        self, descriptors: Iterable[SchemaDescriptor]
    ) -> tuple[SchemaDescriptor, ...]:
        plan: list[SchemaDescriptor] = []
        seen: set[tuple[str, int, int, str]] = set()
        for descriptor in descriptors:
            if not isinstance(descriptor, SchemaDescriptor):
                raise HologresSchemaDefinitionError(
                    "Schema plans accept SchemaDescriptor entries only"
                )
            if descriptor.identity in seen:
                raise HologresSchemaDefinitionError(
                    f"Schema plan contains a duplicate identity: {descriptor.label}"
                )
            seen.add(descriptor.identity)
            plan.append(descriptor)
        return tuple(sorted(plan, key=lambda entry: entry.identity))

    async def _load_ledger(self, plan: Sequence[SchemaDescriptor]) -> Sequence[Any]:
        components = sorted({descriptor.component for descriptor in plan})
        if not components:
            return ()
        return await self._client.fetch_all(
            self._load_sql, components, descriptor="schema.ledger.load"
        )

    def _classify_ledger(
        self, plan: Sequence[SchemaDescriptor], rows: Sequence[Any]
    ) -> dict[tuple[str, int, int, str], dict[str, Any]]:
        by_identity = {descriptor.identity: descriptor for descriptor in plan}
        ceiling: dict[str, int] = {}
        for descriptor in plan:
            ceiling[descriptor.component] = max(
                ceiling.get(descriptor.component, 0), descriptor.version
            )

        recorded: dict[tuple[str, int, int, str], dict[str, Any]] = {}
        for row in rows:
            identity = _row_identity(row)
            label = _identity_label(identity)
            if row["state"] not in _VALID_STATES:
                raise HologresSchemaStateError(
                    f"Ledger row {label} carries an unknown state and will not "
                    "be skipped silently"
                )
            component_ceiling = ceiling.get(identity[0])
            if component_ceiling is not None and row["version"] > component_ceiling:
                raise HologresSchemaDefinitionError(
                    f"Ledger row {label} is newer than the requested plan; a "
                    "schema downgrade is refused"
                )
            descriptor = by_identity.get(identity)
            if descriptor is None:
                raise HologresSchemaStateError(
                    f"Ledger row {label} is unknown to the requested plan and "
                    "will not be skipped silently"
                )
            if row["digest"] != descriptor.digest:
                raise HologresSchemaDriftError(
                    f"Ledger row {label} records a different descriptor digest; "
                    "schema drift is refused before any DDL"
                )
            recorded[identity] = dict(row)
        return recorded

    # -- per-descriptor reconciliation ------------------------------------

    async def _apply(
        self, descriptor: SchemaDescriptor, row: dict[str, Any] | None
    ) -> SchemaApplication:
        if row is not None and row["state"] == SchemaState.APPLIED.value:
            await self._verify_applied(descriptor)
            return self._result(descriptor, SchemaState.APPLIED, False, False)

        outcome, claimed = await self._claim(descriptor)
        if outcome == "applied":
            await self._verify_applied(descriptor)
            return self._result(descriptor, SchemaState.APPLIED, False, False)

        resumed = claimed["previous_state"] in _RESUME_STATES

        # Postcondition first: a crash between a successful DDL and its
        # checkpoint leaves the object in place, so the next owner must not
        # replay the statement just because the ledger looks unfinished.
        if await self._guarded_condition(
            descriptor,
            SchemaState.PREPARED,
            descriptor.postcondition_sql,
            descriptor.postcondition_args,
            "schema.descriptor.postcondition",
            "postcondition",
        ):
            await self._transition(
                descriptor, SchemaState.PREPARED, SchemaState.VERIFYING
            )
            await self._transition(
                descriptor,
                SchemaState.VERIFYING,
                SchemaState.APPLIED,
                terminal=True,
            )
            logger.debug(
                "Hologres schema descriptor %s already satisfied; checkpointed "
                "without replaying DDL",
                descriptor.label,
            )
            return self._result(descriptor, SchemaState.APPLIED, False, resumed)

        if resumed and not descriptor.replay_safe:
            await self._fail(
                descriptor, SchemaState.PREPARED, "replay_not_permitted"
            )
            raise HologresSchemaStateError(
                f"Schema descriptor {descriptor.label} was interrupted and is "
                "not declared replay safe; refusing to replay it automatically"
            )

        if descriptor.precondition_sql is not None:
            satisfied = await self._guarded_condition(
                descriptor,
                SchemaState.PREPARED,
                descriptor.precondition_sql,
                descriptor.precondition_args,
                "schema.descriptor.precondition",
                "precondition",
            )
            if not satisfied:
                await self._fail(
                    descriptor, SchemaState.PREPARED, "precondition_unsatisfied"
                )
                raise HologresSchemaStateError(
                    f"Schema descriptor {descriptor.label} precondition is not "
                    "satisfied; failing closed"
                )

        await self._transition(descriptor, SchemaState.PREPARED, SchemaState.APPLYING)
        try:
            await self._client.execute_one(
                descriptor.sql,
                descriptor="schema.descriptor.apply",
                replay_safe=descriptor.replay_safe,
            )
        except Exception as error:
            summary = _summarize_failure("ddl_execution_failed", error)
            await self._fail(descriptor, SchemaState.APPLYING, summary)
            raise HologresSchemaStateError(
                f"Schema descriptor {descriptor.label} failed to apply "
                f"({summary})"
            ) from None

        await self._transition(descriptor, SchemaState.APPLYING, SchemaState.VERIFYING)
        if not await self._guarded_condition(
            descriptor,
            SchemaState.VERIFYING,
            descriptor.postcondition_sql,
            descriptor.postcondition_args,
            "schema.descriptor.postcondition",
            "postcondition",
        ):
            await self._fail(
                descriptor, SchemaState.VERIFYING, "postcondition_unsatisfied"
            )
            raise HologresSchemaStateError(
                f"Schema descriptor {descriptor.label} postcondition is not "
                "satisfied after applying it; refusing to mark it applied"
            )

        await self._transition(
            descriptor, SchemaState.VERIFYING, SchemaState.APPLIED, terminal=True
        )
        logger.debug(
            "Hologres schema descriptor %s applied and verified", descriptor.label
        )
        return self._result(descriptor, SchemaState.APPLIED, True, resumed)

    async def _claim(
        self, descriptor: SchemaDescriptor
    ) -> tuple[str, dict[str, Any]]:
        for attempt in range(self._busy_retries + 1):
            now = self._now()
            claimed = await self._client.fetch_one(
                self._claim_sql,
                descriptor.component,
                descriptor.version,
                descriptor.step,
                descriptor.name,
                descriptor.digest,
                self._owner_token,
                now + self._lease,
                now,
                descriptor="schema.ledger.claim",
                operation_kind=OperationKind.WRITE,
                replay_safe=True,
            )
            if claimed is not None:
                return "claimed", dict(claimed)

            existing = await self._client.fetch_one(
                self._inspect_sql,
                descriptor.component,
                descriptor.version,
                descriptor.step,
                descriptor.name,
                descriptor="schema.ledger.inspect",
            )
            if existing is None:
                raise HologresSchemaStateError(
                    f"Ledger row {descriptor.label} could not be claimed and is "
                    "no longer present; failing closed"
                )
            if existing["digest"] != descriptor.digest:
                raise HologresSchemaDriftError(
                    f"Ledger row {descriptor.label} records a different "
                    "descriptor digest; schema drift is refused"
                )
            state = existing["state"]
            if state not in _VALID_STATES:
                raise HologresSchemaStateError(
                    f"Ledger row {descriptor.label} carries an unknown state "
                    "and will not be skipped silently"
                )
            if state == SchemaState.APPLIED.value:
                return "applied", dict(existing)
            if attempt >= self._busy_retries:
                break
            await self._sleep(self._busy_delay)

        raise HologresSchemaBusyError(
            f"Ledger row {descriptor.label} is held by another initializer with "
            "an unexpired lease"
        )

    async def _transition(
        self,
        descriptor: SchemaDescriptor,
        expected_state: SchemaState,
        next_state: SchemaState,
        *,
        error_summary: str | None = None,
        terminal: bool = False,
        strict: bool = True,
    ) -> bool:
        now = self._now()
        updated = await self._client.fetch_value(
            self._transition_sql,
            next_state.value,
            error_summary,
            None if terminal else self._owner_token,
            None if terminal else now + self._lease,
            now,
            descriptor.component,
            descriptor.version,
            descriptor.step,
            descriptor.name,
            descriptor.digest,
            self._owner_token,
            expected_state.value,
            descriptor="schema.ledger.transition",
            operation_kind=OperationKind.WRITE,
            replay_safe=True,
        )
        if updated is None:
            if not strict:
                return False
            raise HologresSchemaStateError(
                f"Ledger transition for {descriptor.label} from "
                f"{expected_state.value} to {next_state.value} was refused "
                "because the owner lease is no longer held"
            )
        return True

    async def _fail(
        self,
        descriptor: SchemaDescriptor,
        current_state: SchemaState,
        summary: str,
    ) -> None:
        recorded = await self._transition(
            descriptor,
            current_state,
            SchemaState.FAILED,
            error_summary=_summarize_failure(summary),
            terminal=True,
            strict=False,
        )
        logger.debug(
            "Hologres schema descriptor %s recorded failure %s (persisted=%s)",
            descriptor.label,
            summary,
            recorded,
        )

    # -- condition helpers ------------------------------------------------

    async def _condition(
        self, sql: str, args: Sequence[Any], operation_descriptor: str
    ) -> bool:
        result = await self._client.fetch_value(
            sql, *args, descriptor=operation_descriptor
        )
        if not isinstance(result, bool):
            raise _MalformedCondition(operation_descriptor)
        return result

    async def _guarded_condition(
        self,
        descriptor: SchemaDescriptor,
        current_state: SchemaState,
        sql: str,
        args: Sequence[Any],
        operation_descriptor: str,
        label: str,
    ) -> bool:
        try:
            return await self._condition(sql, args, operation_descriptor)
        except _MalformedCondition as error:
            await self._fail(descriptor, current_state, f"{label}_malformed")
            raise HologresSchemaStateError(
                f"Schema descriptor {descriptor.label} {label} did not return a "
                "boolean; failing closed"
            ) from error

    async def _verify_applied(self, descriptor: SchemaDescriptor) -> None:
        try:
            verified = await self._condition(
                descriptor.postcondition_sql,
                descriptor.postcondition_args,
                "schema.descriptor.postcondition",
            )
        except _MalformedCondition as error:
            raise HologresSchemaStateError(
                f"Schema descriptor {descriptor.label} postcondition did not "
                "return a boolean while verifying an applied row"
            ) from error
        if not verified:
            raise HologresSchemaStateError(
                f"Schema descriptor {descriptor.label} is recorded as applied "
                "but its postcondition is not satisfied; failing closed"
            )

    def _result(
        self,
        descriptor: SchemaDescriptor,
        state: SchemaState,
        executed_ddl: bool,
        resumed: bool,
    ) -> SchemaApplication:
        return SchemaApplication(
            component=descriptor.component,
            version=descriptor.version,
            step=descriptor.step,
            descriptor_name=descriptor.name,
            state=state,
            executed_ddl=executed_ddl,
            resumed=resumed,
        )
