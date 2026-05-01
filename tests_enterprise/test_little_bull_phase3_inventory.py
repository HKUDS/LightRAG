from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _inventory_module() -> ModuleType:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "little_bull_phase3_inventory.py"
    )
    spec = importlib.util.spec_from_file_location(
        "little_bull_phase3_inventory", module_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_phase3_inventory_lists_expected_artifacts_without_delete_actions():
    inventory = _inventory_module()

    payload = inventory.expected_artifacts(
        "phase3-pilot-20260430130847",
        "/tmp/trag-lightrag-phase3/phase3-pilot-20260430130847",
    )

    assert payload["working_dir"].endswith("phase3-pilot-20260430130847")
    assert payload["postgres"]["workspace"] == "phase3-pilot-20260430130847"
    assert payload["qdrant"]["collections"] == [
        "lightrag_vdb_entities_phase3_fake_local_phase3_pilot_20260430130847_16d",
        "lightrag_vdb_relationships_phase3_fake_local_phase3_pilot_20260430130847_16d",
        "lightrag_vdb_chunks_phase3_fake_local_phase3_pilot_20260430130847_16d",
    ]
    assert payload["neo4j"]["fulltext_index"] == (
        "entity_id_fulltext_idx_phase3_pilot_20260430130847"
    )


def test_phase3_inventory_rejects_unsafe_workspace():
    inventory = _inventory_module()

    with pytest.raises(ValueError, match="Workspace"):
        inventory.expected_artifacts("../phase3")
