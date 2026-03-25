from pathlib import Path

import pytest

from lightrag.prompt_version_store import PromptVersionStore


def test_initialize_registry_creates_localized_seed_versions(tmp_path: Path):
    store = PromptVersionStore(tmp_path, workspace="demo")

    registry = store.initialize(locale="zh")

    assert registry["workspace"] == "demo"
    assert registry["indexing"]["versions"]
    assert registry["retrieval"]["versions"]
    assert store.registry_path.exists()


def test_delete_inactive_version_keeps_lineage_readable(tmp_path: Path):
    store = PromptVersionStore(tmp_path, workspace="demo")
    store.initialize(locale="en")

    created = store.create_version(
        "retrieval",
        {
            "query": {"rag_response": "A {context_data}"},
            "keywords": {"keywords_extraction": "Q={query};E={examples}"},
        },
        "v1",
        "first",
        None,
    )
    copied = store.copy_version("retrieval", created["version_id"], "v2", "")

    store.delete_version("retrieval", created["version_id"])

    fetched = store.get_version("retrieval", copied["version_id"])
    assert fetched["source_version_id"] == created["version_id"]


def test_store_writes_registry_atomically(tmp_path: Path):
    store = PromptVersionStore(tmp_path, workspace="demo")

    store.initialize(locale="en")

    assert not list(tmp_path.rglob("*.tmp"))


def test_delete_active_version_is_rejected(tmp_path: Path):
    store = PromptVersionStore(tmp_path, workspace="demo")
    registry = store.initialize(locale="en")
    active_id = registry["retrieval"]["versions"][0]["version_id"]

    store.activate_version("retrieval", active_id)

    with pytest.raises(ValueError):
        store.delete_version("retrieval", active_id)
