"""Answer the statements of the PostgreSQL legacy vector migration in mock tests.

Tests that drive ``PGVectorStorage.setup_table`` through a hand-written
``db.query`` mock only care THAT a workspace was migrated, not how. Wrapping
their mock with ``answer_legacy_migration`` keeps them independent of the
statement sequence, which is pinned in
``tests/kg/postgres_impl/test_postgres_migration_restart.py``.
"""

from collections.abc import Awaitable, Callable
from typing import Any


def answer_legacy_migration(
    query: Callable[..., Awaitable[Any]],
    on_copy: Callable[[str, list[Any]], None],
) -> Callable[..., Awaitable[Any]]:
    """Wrap ``query`` so the migration sees one short, fully verified segment.

    ``on_copy(sql, params)`` runs when the copy statement is issued; use it to
    record the migration and to move the mock's row counts forward. Every other
    statement falls through to ``query`` unchanged.
    """

    async def wrapped(sql: str, params: Any = None, *args: Any, **kwargs: Any) -> Any:
        if "attname" in sql:
            return [{"attname": "id"}, {"attname": "workspace"}]
        if sql.startswith("WITH guard"):
            on_copy(sql, list(params or []))
            return {"batch_rows": 1, "last_id": "last-id", "active": True}
        if "AS cleared" in sql:
            return {"missing": 0, "cleared": 1}
        return await query(sql, params, *args, **kwargs)

    return wrapped
