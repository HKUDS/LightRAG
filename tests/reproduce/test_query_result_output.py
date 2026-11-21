"""A repeated query run must leave one readable result array.

Each Step_3 script is imported for real, so ``run_queries_and_save_to_json``
runs against its own globals; only the event loop and the RAG instance are
stand-ins. Importing is safe because both scripts keep their work under
``if __name__ == "__main__"``.
"""

import asyncio
import importlib.util
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

pytestmark = pytest.mark.offline

REPRODUCE_DIR = Path(__file__).resolve().parents[2] / "reproduce"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(
        f"reproduce_{Path(name).stem}", REPRODUCE_DIR / name
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _json_objects(text: str) -> list:
    """Decode the error log, which is a stream of JSON objects, not an array."""
    decoder, objects, index = json.JSONDecoder(), [], 0
    while index < len(text):
        if text[index].isspace():
            index += 1
            continue
        obj, index = decoder.raw_decode(text, index)
        objects.append(obj)
    return objects


@pytest.mark.parametrize("script", ["Step_3.py", "Step_3_openai_compatible.py"])
def test_repeated_runs_leave_valid_current_results(script, tmp_path, monkeypatch):
    module = _load_script(script)
    loop = asyncio.new_event_loop()
    monkeypatch.setattr(module, "always_get_an_event_loop", lambda: loop)
    output, errors = tmp_path / "results.json", tmp_path / "errors.json"
    answers = ["old answer", RuntimeError("old boom")]
    answers += ["new answer", RuntimeError("new boom")]
    rag = type("Rag", (), {"aquery": AsyncMock(side_effect=answers)})()
    try:
        for run in ["old", "new"]:
            module.run_queries_and_save_to_json(
                [f"{run} question", f"{run} failing"], rag, None, output, errors
            )
    finally:
        loop.close()

    # Results are overwritten: only the latest run, as one JSON array.
    assert json.loads(output.read_text(encoding="utf-8")) == [
        {"query": "new question", "result": "new answer"}
    ]
    # The error log still appends across runs.
    assert _json_objects(errors.read_text(encoding="utf-8")) == [
        {"query": "old failing", "error": "old boom"},
        {"query": "new failing", "error": "new boom"},
    ]
