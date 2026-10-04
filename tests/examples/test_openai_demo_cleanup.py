"""Exercise the official demos without executing their provider entry points."""

import ast
import os
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

pytestmark = pytest.mark.offline

DEMOS = ["lightrag_openai_demo.py", "lightrag_openai_compatible_demo.py"]


def load_function(demo, name, namespace):
    source = Path(__file__).resolve().parents[2] / "examples" / demo
    tree = ast.parse(source.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if getattr(n, "name", None) == name)
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return namespace[name]


@pytest.mark.parametrize("demo", DEMOS)
@pytest.mark.asyncio
async def test_initialization_failure_does_not_mask_diagnostic(
    demo, tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("OPENAI_API_KEY", "offline-placeholder")
    main = load_function(
        demo,
        "main",
        {
            "os": os,
            "WORKING_DIR": str(tmp_path),
            "initialize_rag": AsyncMock(
                side_effect=RuntimeError("storage unavailable")
            ),
        },
    )
    await main()
    assert "storage unavailable" in capsys.readouterr().out


@pytest.mark.parametrize("demo", DEMOS)
@pytest.mark.asyncio
async def test_initialized_rag_is_finalized(demo, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "offline-placeholder")
    (tmp_path / "book.txt").write_text("A short offline book.", encoding="utf-8")
    rag = Mock(
        embedding_func=AsyncMock(return_value=np.zeros((1, 2))),
        ainsert=AsyncMock(),
        aquery=AsyncMock(return_value="answer"),
        finalize_storages=AsyncMock(),
    )
    import inspect

    main = load_function(
        demo,
        "main",
        {
            "os": os,
            "inspect": inspect,
            "WORKING_DIR": str(tmp_path),
            "initialize_rag": AsyncMock(return_value=rag),
            "QueryParam": Mock(),
        },
    )
    await main()
    rag.finalize_storages.assert_awaited_once()
