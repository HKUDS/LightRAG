"""Exercise the demos' ``main`` cleanup without calling their providers."""

import inspect
import os
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

from tests.examples._demo_source import load_demo_definition

pytestmark = pytest.mark.offline

DEMOS = [
    "lightrag_openai_demo.py",
    "lightrag_openai_compatible_demo.py",
    "lightrag_ollama_demo.py",
    "unofficial-sample/lightrag_cloudflare_demo.py",
]


def _load_main(demo, working_dir, initialize_rag):
    return load_demo_definition(
        demo,
        "main",
        {
            "os": os,
            "inspect": inspect,
            "WORKING_DIR": str(working_dir),
            "initialize_rag": initialize_rag,
            "QueryParam": Mock(),
        },
    )


@pytest.mark.parametrize("demo", DEMOS)
async def test_initialization_failure_does_not_mask_diagnostic(
    demo, tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("OPENAI_API_KEY", "offline-placeholder")
    main = _load_main(
        demo,
        tmp_path,
        AsyncMock(side_effect=RuntimeError("storage unavailable")),
    )

    # Without ``rag = None`` before the ``try``, ``finally`` reads the
    # unassigned local and raises UnboundLocalError over the handled failure.
    await main()

    assert "storage unavailable" in capsys.readouterr().out


@pytest.mark.parametrize("demo", DEMOS)
async def test_initialized_rag_is_finalized(demo, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "offline-placeholder")
    (tmp_path / "book.txt").write_text("A short offline book.", encoding="utf-8")
    rag = Mock(
        embedding_func=AsyncMock(return_value=np.zeros((1, 2))),
        ainsert=AsyncMock(),
        aquery=AsyncMock(return_value="answer"),
        finalize_storages=AsyncMock(),
        llm_response_cache=Mock(index_done_callback=AsyncMock()),
    )
    main = _load_main(demo, tmp_path, AsyncMock(return_value=rag))

    await main()

    assert "An error occurred" not in capsys.readouterr().out
    rag.finalize_storages.assert_awaited_once()
