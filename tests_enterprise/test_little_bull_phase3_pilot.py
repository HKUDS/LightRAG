from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _pilot_module() -> ModuleType:
    module_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "little_bull_phase3_pilot.py"
    )
    spec = importlib.util.spec_from_file_location(
        "little_bull_phase3_pilot", module_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _base_env() -> dict[str, str]:
    return {
        "LITTLE_BULL_PHASE3_PILOT": "1",
        "POSTGRES_USER": "set",
        "POSTGRES_PASSWORD": "set",
        "POSTGRES_DATABASE": "set",
        "NEO4J_URI": "set",
        "NEO4J_USERNAME": "set",
        "NEO4J_PASSWORD": "set",
        "QDRANT_URL": "set",
    }


def test_phase3_pilot_storage_contract_matches_target_architecture():
    pilot = _pilot_module()

    assert pilot.STORAGE_CONTRACT == {
        "kv_storage": "PGKVStorage",
        "doc_status_storage": "PGDocStatusStorage",
        "graph_storage": "Neo4JStorage",
        "vector_storage": "QdrantVectorDBStorage",
    }


def test_phase3_pilot_requires_explicit_opt_in():
    pilot = _pilot_module()
    env = _base_env()
    env.pop("LITTLE_BULL_PHASE3_PILOT")

    with pytest.raises(pilot.PilotConfigError, match="LITTLE_BULL_PHASE3_PILOT=1"):
        pilot.build_config(["--workspace", "phase3-test"], env=env)


def test_phase3_pilot_reports_missing_env_names_without_values():
    pilot = _pilot_module()
    env = {
        "LITTLE_BULL_PHASE3_PILOT": "1",
        "POSTGRES_PASSWORD": "do-not-print-this",
    }

    with pytest.raises(pilot.PilotConfigError) as exc:
        pilot.build_config(["--workspace", "phase3-test"], env=env)

    message = str(exc.value)
    assert "POSTGRES_DATABASE" in message
    assert "NEO4J_PASSWORD" in message
    assert "do-not-print-this" not in message


def test_phase3_pilot_rejects_global_storage_workspace_overrides():
    pilot = _pilot_module()
    env = {**_base_env(), "QDRANT_WORKSPACE": "unsafe-global-workspace"}

    with pytest.raises(pilot.PilotConfigError, match="QDRANT_WORKSPACE"):
        pilot.build_config(["--workspace", "phase3-test"], env=env)


def test_phase3_pilot_can_allow_workspace_override_for_subprocess_experiments():
    pilot = _pilot_module()
    env = {**_base_env(), "QDRANT_WORKSPACE": "phase3-test"}

    config = pilot.build_config(
        ["--workspace", "phase3-test", "--allow-storage-workspace-env"],
        env=env,
    )

    assert config.workspace == "phase3-test"
    assert config.allow_storage_workspace_env is True


def test_phase3_pilot_has_explicit_diagnostic_overrides_for_local_risks():
    pilot = _pilot_module()
    env = {
        **_base_env(),
        "LITTLE_BULL_PHASE3_ALLOW_NEO4J_NO_AUTH": "1",
        "LITTLE_BULL_PHASE3_ALLOW_QDRANT_VERSION_MISMATCH": "1",
    }

    config = pilot.build_config(["--workspace", "phase3-test"], env=env)

    assert config.allow_neo4j_no_auth is True
    assert config.allow_qdrant_version_mismatch is True


def test_phase3_pilot_accepts_qdrant_versions_within_minor_delta():
    pilot = _pilot_module()

    assert pilot.qdrant_versions_are_compatible("1.15.1", "1.15.0") is True
    assert pilot.qdrant_versions_are_compatible("1.15.1", "1.16.3") is True
    assert pilot.qdrant_versions_are_compatible("1.15.1", "1.17.1") is False
    assert pilot.qdrant_versions_are_compatible("1.15.1", "2.0.0") is False


def test_phase3_pilot_workspace_must_be_storage_safe():
    pilot = _pilot_module()

    with pytest.raises(pilot.PilotConfigError, match="Workspace"):
        pilot.build_config(["--workspace", "../not-safe"], env=_base_env())
