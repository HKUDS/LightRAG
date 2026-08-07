"""Test get_all_update_flags_status return types in single process mode.

WHY: get_all_update_flags_status maps workspace namespaces to lists of boolean update flag statuses.
In single-process mode, returning raw MutableBoolean objects instead of Python bools breaks callers
expecting boolean flag values (e.g. JSON serialization, status checks).
"""

import pytest

from lightrag.kg.shared_storage import (
    clear_all_update_flags,
    finalize_share_data,
    get_all_update_flags_status,
    get_update_flag,
    initialize_share_data,
    set_all_update_flags,
)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_get_all_update_flags_status_returns_booleans():
    try:
        initialize_share_data()
        flag = await get_update_flag("test_ns", workspace="default")
        assert flag.value is False

        status_before = await get_all_update_flags_status(workspace="default")
        assert "default:test_ns" in status_before
        statuses = status_before["default:test_ns"]
        assert len(statuses) == 1
        # Must be actual Python bool, not MutableBoolean object
        assert type(statuses[0]) is bool
        assert statuses[0] is False

        await set_all_update_flags("test_ns", workspace="default")
        status_set = await get_all_update_flags_status(workspace="default")
        assert type(status_set["default:test_ns"][0]) is bool
        assert status_set["default:test_ns"][0] is True

        await clear_all_update_flags("test_ns", workspace="default")
        status_cleared = await get_all_update_flags_status(workspace="default")
        assert type(status_cleared["default:test_ns"][0]) is bool
        assert status_cleared["default:test_ns"][0] is False
    finally:
        finalize_share_data()
