"""The `recovery_required` fence must be VISIBLE without leaking its internals.

The raw fence record lives in ``_INTERNAL_PIPELINE_STATUS_FIELDS`` and is stripped
from every API response, and that is correct: it embeds an ``operation_record`` and
is written next to owner records carrying PIDs and reservation tokens, and a token
authorizes releasing a reservation.

But stripping it and nothing else left an operator with no read-only way to learn
that the workspace is fenced, why, or which documents to look at — just a 503 on
every write. The documentation told them to check
``GET /documents/pipeline_status``, where the field had been deleted.

So the endpoint reports a sanitized projection instead. These tests pin both
halves: the three safe fields are present and accurate, and none of the unsafe
ones come with them.
"""

from __future__ import annotations

import pytest

from lightrag.kg.shared_storage import (
    _INTERNAL_PIPELINE_STATUS_FIELDS,
    describe_recovery_fence,
)

pytestmark = pytest.mark.offline


def test_unfenced_pipeline_reports_false_not_null():
    """A dashboard should not have to special-case "no fence"."""
    view = describe_recovery_fence({"busy": False})

    assert view == {
        "recovery_required": False,
        "recovery_kind": None,
        "recovery_message": None,
    }


def test_unfenced_pipeline_with_an_empty_record_is_not_fenced():
    """A cleared fence is written as ``None``; an empty dict must read the same."""
    assert describe_recovery_fence({"recovery_required": None})[
        "recovery_required"
    ] is (False)
    assert describe_recovery_fence({"recovery_required": {}})["recovery_required"] is (
        False
    )


def test_stalled_drain_fence_surfaces_kind_and_blocker_sample():
    """The projection carries what an operator acts on: the coarse cause and the
    message, which for a stalled drain includes the bounded blocking doc ids."""
    view = describe_recovery_fence(
        {
            "recovery_required": {
                "kind": "manual_drain_stalled",
                "owner_key": "busy_owner",
                "operation_record": {"scope": "doc-a, doc-b"},
                "message": (
                    "manual retry drain stalled: 12 active document(s) have "
                    "blocked DRAIN_TO_IDLE for 3 consecutive rounds without "
                    "changing state (blocked doc id sample: doc-a, doc-b)."
                ),
            }
        }
    )

    assert view["recovery_required"] is True
    assert view["recovery_kind"] == "manual_drain_stalled"
    assert "blocked doc id sample: doc-a, doc-b" in view["recovery_message"]
    # The message explains what the 503 means, in the same words the refusal uses.
    assert "force-reset" in view["recovery_message"]


def test_dead_owner_fence_keeps_its_derived_wording():
    """A fence with no explicit message (the original dead-owner cause) still
    renders — it must not degrade to an empty string."""
    view = describe_recovery_fence(
        {
            "recovery_required": {
                "kind": "clear",
                "owner_key": "busy_owner",
                "operation_record": {"scope": "workspace"},
            }
        }
    )

    assert view["recovery_kind"] == "clear"
    assert "a worker died mid 'clear'" in view["recovery_message"]


def test_the_projection_leaks_no_credentials_or_process_identity():
    """Exactly three keys, and none of the record's credentials.

    The projection is a whitelist, so a future field added to the raw record does
    not ride along. What specifically must never appear is the reservation token
    (it authorizes RELEASING a reservation — a status page must not become a
    control surface) and the process identity. A target ``doc_id`` is a different
    matter: it is the actionable part and is already public across this API, so
    the dead-owner wording names it on purpose (pinned in
    ``tests/kg/test_reservation_dead_process_recovery.py``).
    """
    view = describe_recovery_fence(
        {
            "recovery_required": {
                "kind": "custom_chunks",
                "owner_key": "busy_owner",
                "operation_record": {"doc_id": "doc-x"},
                "message": "a worker died mid custom_chunks.",
                "owner_token": "tok-must-not-leak",
                "pid": 4242,
            }
        }
    )

    assert set(view) == {
        "recovery_required",
        "recovery_kind",
        "recovery_message",
    }
    rendered = repr(view)
    assert "tok-must-not-leak" not in rendered
    assert "4242" not in rendered
    assert "owner_key" not in rendered


def test_the_raw_record_is_still_internal():
    """Guards the reason this projection exists: the raw field stays stripped, so
    removing it from the internal list (and publishing it wholesale) is a
    deliberate change, not an accident."""
    assert "recovery_required" in _INTERNAL_PIPELINE_STATUS_FIELDS
