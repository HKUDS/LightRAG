import hashlib
import os
import uuid
from dataclasses import dataclass

import pytest

from lightrag.kg.hologres.capabilities import validate_test_schema_name
from lightrag.kg.hologres.client import (
    HologresClient,
    quote_qualified_identifier,
    validate_single_statement,
)
from lightrag.kg.hologres.config import HologresConfig


_REQUIRED_LIVE_ENVIRONMENT = (
    "HOLOGRES_HOST",
    "HOLOGRES_PORT",
    "HOLOGRES_USER",
    "HOLOGRES_PASSWORD",
    "HOLOGRES_DATABASE",
)


class HologresLiveCleanupError(RuntimeError):
    """Report every failed live-schema cleanup operation after all attempts."""

    def __init__(self, failures):
        self.failures = tuple(
            (
                stage,
                error if isinstance(error, str) else type(error).__name__,
            )
            for stage, error in failures
        )
        stages = ", ".join(stage for stage, _error_kind in self.failures)
        super().__init__(f"Hologres live schema cleanup failed: {stages}")


_RELATION_DROP_KINDS = {
    "v": "VIEW",
    "m": "MATERIALIZED VIEW",
    "f": "FOREIGN TABLE",
    "p": "TABLE",
    "r": "TABLE",
    "S": "SEQUENCE",
    "i": "INDEX",
    "I": "INDEX",
    "c": "TYPE",
}
_RELATION_ORDER = {
    "v": 0,
    "m": 1,
    "f": 2,
    "p": 3,
    "r": 4,
    "S": 5,
    "i": 6,
    "I": 6,
    "c": 7,
}
_ROUTINE_DROP_KINDS = {
    "f": "FUNCTION",
    "w": "FUNCTION",
    "p": "PROCEDURE",
    "a": "AGGREGATE",
}
_ROUTINE_ORDER = {"f": 0, "w": 0, "p": 1, "a": 2}
_TYPE_DROP_KINDS = {"e": "TYPE", "d": "DOMAIN", "r": "TYPE"}
_TYPE_ORDER = {"e": 0, "d": 1, "r": 2}
_MAX_FAILURE_STAGE_LENGTH = 96


@dataclass(frozen=True)
class _CleanupTarget:
    order_key: tuple
    failure_stage: str
    sql: str
    descriptor: str


def _safe_label_component(value):
    if not isinstance(value, str):
        return "unknown"
    label = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in value
    )
    return label[:48] or "unnamed"


def _object_failure_stage(category, object_kind, object_name, discriminator=None):
    object_label = f"{object_kind}:{_safe_label_component(object_name)}"
    if discriminator:
        fingerprint = hashlib.sha256(discriminator.encode("utf-8")).hexdigest()[:8]
        object_label = f"{object_label}#{fingerprint}"
    prefix = f"{category} drop ["
    available = _MAX_FAILURE_STAGE_LENGTH - len(prefix) - 1
    return f"{prefix}{object_label[:available]}]"


def _classification_stage(category, object_name):
    prefix = f"{category} classification ["
    available = _MAX_FAILURE_STAGE_LENGTH - len(prefix) - 1
    label = _safe_label_component(object_name)
    return f"{prefix}{label[:available]}]"


def _cleanup_target(order_key, failure_stage, sql, descriptor):
    validate_single_statement(sql)
    return _CleanupTarget(order_key, failure_stage, sql, descriptor)


async def _fetch_cleanup_catalog(client, schema, sql, descriptor, failure_stage, failures):
    try:
        return await client.fetch_all(
            sql,
            schema,
            descriptor=descriptor,
        )
    except Exception as error:
        failures.append((failure_stage, error))
        return ()


async def _drop_cleanup_targets(client, targets):
    pending = sorted(targets, key=lambda target: target.order_key)
    final_errors = {}

    while pending:
        failed = []
        made_progress = False
        for target in pending:
            try:
                await client.execute_one(
                    target.sql,
                    descriptor=target.descriptor,
                    replay_safe=True,
                )
            except Exception as error:
                final_errors[target] = error
                failed.append(target)
            else:
                final_errors.pop(target, None)
                made_progress = True

        pending = failed
        if not pending or not made_progress:
            break

    return [(target.failure_stage, final_errors[target]) for target in pending]


async def _cleanup_live_schema(client, schema):
    validated_schema = validate_test_schema_name(schema)
    failures = []

    foreign_keys = await _fetch_cleanup_catalog(
        client,
        validated_schema,
        "SELECT child.relname AS table_name, con.conname AS constraint_name "
        "FROM pg_catalog.pg_constraint con "
        "JOIN pg_catalog.pg_class child ON child.oid = con.conrelid "
        "JOIN pg_catalog.pg_namespace child_namespace "
        "ON child_namespace.oid = child.relnamespace "
        "JOIN pg_catalog.pg_class referenced ON referenced.oid = con.confrelid "
        "JOIN pg_catalog.pg_namespace referenced_namespace "
        "ON referenced_namespace.oid = referenced.relnamespace "
        "WHERE con.contype = 'f' "
        "AND child_namespace.nspname = $1 "
        "AND referenced_namespace.nspname = $1 "
        "ORDER BY child.relname, con.conname",
        "live.schema.cleanup.enumerate.foreign_keys",
        "foreign key enumeration",
        failures,
    )
    relations = await _fetch_cleanup_catalog(
        client,
        validated_schema,
        "SELECT c.relname, c.relkind::text AS relkind "
        "FROM pg_catalog.pg_class c "
        "JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace "
        "WHERE n.nspname = $1 "
        "AND c.relkind IN ('v', 'm', 'f', 'i', 'I', 'S', 'p', 'r', 'c') "
        "ORDER BY CASE c.relkind "
        "WHEN 'v' THEN 1 WHEN 'm' THEN 2 WHEN 'f' THEN 3 "
        "WHEN 'p' THEN 4 WHEN 'r' THEN 5 WHEN 'S' THEN 6 "
        "WHEN 'i' THEN 7 WHEN 'I' THEN 7 ELSE 8 END, c.relname",
        "live.schema.cleanup.enumerate.relations",
        "relation enumeration",
        failures,
    )
    routines = await _fetch_cleanup_catalog(
        client,
        validated_schema,
        "SELECT p.proname, p.prokind::text AS prokind, "
        "pg_catalog.pg_get_function_identity_arguments(p.oid) AS identity_args "
        "FROM pg_catalog.pg_proc p "
        "JOIN pg_catalog.pg_namespace n ON n.oid = p.pronamespace "
        "WHERE n.nspname = $1 "
        "AND p.prokind IN ('f', 'w', 'p', 'a') "
        "ORDER BY CASE p.prokind "
        "WHEN 'f' THEN 1 WHEN 'w' THEN 1 WHEN 'p' THEN 2 ELSE 3 END, "
        "p.proname, identity_args",
        "live.schema.cleanup.enumerate.routines",
        "routine enumeration",
        failures,
    )
    standalone_types = await _fetch_cleanup_catalog(
        client,
        validated_schema,
        "SELECT t.typname, t.typtype::text AS typtype "
        "FROM pg_catalog.pg_type t "
        "JOIN pg_catalog.pg_namespace n ON n.oid = t.typnamespace "
        "WHERE n.nspname = $1 "
        "AND t.typtype IN ('e', 'd', 'r') "
        "AND t.typrelid = 0 "
        "AND NOT (t.typelem <> 0 AND t.typarray = 0) "
        "ORDER BY CASE t.typtype "
        "WHEN 'e' THEN 1 WHEN 'd' THEN 2 ELSE 3 END, t.typname",
        "live.schema.cleanup.enumerate.types",
        "type enumeration",
        failures,
    )

    targets = []
    for foreign_key in foreign_keys:
        table_name = foreign_key["table_name"]
        constraint_name = foreign_key["constraint_name"]
        try:
            sql = (
                "ALTER TABLE IF EXISTS "
                f"{quote_qualified_identifier(validated_schema, table_name)} "
                "DROP CONSTRAINT IF EXISTS "
                f"{quote_qualified_identifier(constraint_name)}"
            )
            targets.append(
                _cleanup_target(
                    (0, table_name, constraint_name),
                    _object_failure_stage(
                        "foreign key",
                        "CONSTRAINT",
                        constraint_name,
                    ),
                    sql,
                    "live.schema.cleanup.foreign_key",
                )
            )
        except Exception as error:
            failures.append(
                (_classification_stage("foreign key", constraint_name), error)
            )

    for relation in relations:
        relation_name = relation["relname"]
        relation_kind_code = relation["relkind"]
        try:
            relation_kind = _RELATION_DROP_KINDS[relation_kind_code]
            sql = (
                f"DROP {relation_kind} IF EXISTS "
                f"{quote_qualified_identifier(validated_schema, relation_name)}"
            )
            targets.append(
                _cleanup_target(
                    (1, _RELATION_ORDER[relation_kind_code], relation_name),
                    _object_failure_stage(
                        "relation",
                        relation_kind,
                        relation_name,
                    ),
                    sql,
                    "live.schema.cleanup.relation",
                )
            )
        except Exception as error:
            failures.append(
                (_classification_stage("relation", relation_name), error)
            )

    for routine in routines:
        routine_name = routine["proname"]
        routine_kind_code = routine["prokind"]
        identity_args = routine["identity_args"]
        try:
            if not isinstance(identity_args, str):
                raise ValueError("routine identity arguments must be text")
            routine_kind = _ROUTINE_DROP_KINDS[routine_kind_code]
            drop_args = (
                "*"
                if routine_kind_code == "a" and identity_args == ""
                else identity_args
            )
            sql = (
                f"DROP {routine_kind} IF EXISTS "
                f"{quote_qualified_identifier(validated_schema, routine_name)}"
                f"({drop_args})"
            )
            targets.append(
                _cleanup_target(
                    (
                        2,
                        _ROUTINE_ORDER[routine_kind_code],
                        routine_name,
                        identity_args,
                    ),
                    _object_failure_stage(
                        "routine",
                        routine_kind,
                        routine_name,
                        identity_args,
                    ),
                    sql,
                    "live.schema.cleanup.routine",
                )
            )
        except Exception as error:
            failures.append((_classification_stage("routine", routine_name), error))

    for standalone_type in standalone_types:
        type_name = standalone_type["typname"]
        type_kind_code = standalone_type["typtype"]
        try:
            type_kind = _TYPE_DROP_KINDS[type_kind_code]
            sql = (
                f"DROP {type_kind} IF EXISTS "
                f"{quote_qualified_identifier(validated_schema, type_name)}"
            )
            targets.append(
                _cleanup_target(
                    (3, _TYPE_ORDER[type_kind_code], type_name),
                    _object_failure_stage("type", type_kind, type_name),
                    sql,
                    "live.schema.cleanup.type",
                )
            )
        except Exception as error:
            failures.append((_classification_stage("type", type_name), error))

    failures.extend(await _drop_cleanup_targets(client, targets))

    try:
        schema_drop = (
            "DROP SCHEMA IF EXISTS "
            f"{quote_qualified_identifier(validated_schema)}"
        )
        validate_single_statement(schema_drop)
        await client.execute_one(
            schema_drop,
            descriptor="live.schema.cleanup.schema",
            replay_safe=True,
        )
    except Exception as error:
        failures.append(("schema drop", error))

    if failures:
        raise HologresLiveCleanupError(failures) from None


async def _cleanup_live_schema_and_close(client, schema):
    cleanup_error = None
    try:
        await _cleanup_live_schema(client, schema)
    except BaseException as error:
        cleanup_error = error

    close_error = None
    try:
        await client.close()
    except BaseException as error:
        close_error = error

    if close_error is not None:
        if isinstance(cleanup_error, HologresLiveCleanupError):
            failures = (*cleanup_error.failures, ("client close", close_error))
        elif cleanup_error is not None:
            failures = (("schema cleanup", cleanup_error), ("client close", close_error))
        else:
            failures = (("client close", close_error),)
        raise HologresLiveCleanupError(failures) from None

    if cleanup_error is not None:
        raise cleanup_error from None


@pytest.fixture
async def hologres_live_client():
    if any(not os.environ.get(name) for name in _REQUIRED_LIVE_ENVIRONMENT):
        pytest.skip("Hologres live credentials are unavailable")

    schema = validate_test_schema_name(f"lightrag_test_{uuid.uuid4().hex}")
    config = HologresConfig.from_env({**os.environ, "HOLOGRES_SCHEMA": schema})
    client = HologresClient(config)
    opened = False
    try:
        await client.open()
        opened = True
        yield client, schema
    finally:
        if opened:
            await _cleanup_live_schema_and_close(client, schema)
        else:
            await client.close()
