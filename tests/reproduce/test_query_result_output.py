"""A repeated query run must leave one readable result array."""

import ast
import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

pytestmark = pytest.mark.offline


def test_repeated_runs_leave_valid_current_results(tmp_path):
    source = Path(__file__).resolve().parents[2] / "reproduce" / "Step_3.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    functions = [
        n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    loop = asyncio.new_event_loop()
    namespace = {"json": json, "always_get_an_event_loop": lambda: loop}
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"),
        namespace,
    )
    output, errors = tmp_path / "results.json", tmp_path / "errors.json"
    rag = type(
        "Rag", (), {"aquery": AsyncMock(side_effect=["old answer", "new answer"])}
    )()
    try:
        for query in ["old question", "new question"]:
            namespace["run_queries_and_save_to_json"](
                [query], rag, None, output, errors
            )
    finally:
        loop.close()
    assert json.loads(output.read_text(encoding="utf-8")) == [
        {"query": "new question", "result": "new answer"}
    ]
