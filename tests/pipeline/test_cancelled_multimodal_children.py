"""External cancellation drains analysis tasks before recovery starts."""

import asyncio
import json

import pytest

from .test_pipeline_cancellation import _build_rag, _shutdown_role_workers

pytestmark = pytest.mark.offline


async def test_external_cancel_drains_multimodal_tasks(tmp_path, monkeypatch):
    monkeypatch.setenv("MAX_EXTRACT_INPUT_TOKENS", "999999")
    prior_tasks = asyncio.all_tasks()
    entered = asyncio.Event()
    release = asyncio.Event()
    children = []
    writes = []

    async def extract(*args, **kwargs):
        children.append(asyncio.current_task())
        entered.set()
        await release.wait()
        writes.append("late analysis")
        return json.dumps({"name": "table", "description": "result"})

    blocks = tmp_path / "doc.blocks.jsonl"
    blocks.write_text(json.dumps({"type": "meta", "doc_id": "doc"}) + "\n")
    sidecar = tmp_path / "doc.tables.json"
    sidecar.write_text(
        json.dumps(
            {
                "tables": {
                    f"t{i}": {"format": "html", "content": "<table>x</table>"}
                    for i in range(3)
                }
            }
        )
    )
    rag = _build_rag(tmp_path)
    await rag.initialize_storages()
    extract_state = rag._role_llm_states["extract"]
    original_extract = extract_state.wrapped
    extract_state.wrapped = extract
    try:
        parent = asyncio.create_task(
            rag.analyze_multimodal(
                "doc", "doc.pdf", {"blocks_path": str(blocks)}, process_options="t"
            )
        )
        await asyncio.wait_for(entered.wait(), timeout=5)
        parent.cancel()
        with pytest.raises(asyncio.CancelledError):
            await parent
        assert all(task.done() for task in children)
        release.set()
        await asyncio.sleep(0)
        assert writes == []
    finally:
        release.set()
        remaining = asyncio.all_tasks() - prior_tasks
        for task in remaining:
            task.cancel()
        await asyncio.gather(*remaining, return_exceptions=True)
        extract_state.wrapped = original_extract
        await _shutdown_role_workers(rag)
        await rag.finalize_storages()
