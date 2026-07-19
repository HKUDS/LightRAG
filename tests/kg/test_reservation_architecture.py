"""Architecture guards for centralized pipeline reservation coordination."""

from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIGHTRAG_ROOT = PROJECT_ROOT / "lightrag"
SHARED_STORAGE = LIGHTRAG_ROOT / "kg" / "shared_storage.py"


def _production_python_sources():
    for path in LIGHTRAG_ROOT.rglob("*.py"):
        if path != SHARED_STORAGE:
            yield path


@pytest.mark.offline
def test_dead_reservation_reconcile_is_shared_storage_private():
    offenders = [
        str(path.relative_to(PROJECT_ROOT))
        for path in _production_python_sources()
        if "reconcile_dead_pipeline_reservations" in path.read_text()
    ]
    assert offenders == []


@pytest.mark.offline
def test_reservation_owner_records_are_created_only_in_shared_storage():
    offenders = [
        str(path.relative_to(PROJECT_ROOT))
        for path in _production_python_sources()
        if "make_owner_record(" in path.read_text()
    ]
    assert offenders == []
