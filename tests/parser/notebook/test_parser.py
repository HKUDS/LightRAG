"""Regression tests for native ``.ipynb`` parsing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lightrag.parser.notebook.parser import NativeNotebookParser
from lightrag.sidecar import write_sidecar


def _write_notebook(path: Path, cells: list[dict]) -> None:
    path.write_text(
        json.dumps(
            {
                "cells": cells,
                "metadata": {"kernelspec": {"language": "python"}},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        ),
        encoding="utf-8",
    )


def test_notebook_cells_render_as_markdown_and_fenced_code(tmp_path: Path):
    source = tmp_path / "guide.ipynb"
    _write_notebook(
        source,
        [
            {"cell_type": "markdown", "source": ["# Guide\n", "Intro text."]},
            {
                "cell_type": "code",
                "source": ["value = 2\n", "print(value)"],
                "execution_count": 1,
                "outputs": [{"output_type": "stream", "text": "2\n"}],
            },
            {"cell_type": "raw", "source": "Raw context."},
        ],
    )

    markdown, counts = NativeNotebookParser._notebook_to_markdown(source)

    assert markdown == (
        "# Guide\nIntro text.\n\n"
        "```python\nvalue = 2\nprint(value)\n```\n\n"
        "Raw context."
    )
    assert counts == {
        "notebook_code_cells": 1,
        "notebook_markdown_cells": 1,
        "notebook_raw_cells": 1,
    }
    assert "execution_count" not in markdown
    assert "output_type" not in markdown


def test_notebook_extracts_content_without_execution_output(tmp_path: Path):
    source = tmp_path / "guide.ipynb"
    _write_notebook(
        source,
        [
            {"cell_type": "markdown", "source": "# Guide\n\nHello notebook."},
            {
                "cell_type": "code",
                "source": "answer = 42",
                "outputs": [{"output_type": "stream", "text": "secret output"}],
            },
        ],
    )

    blocks, warnings, metadata = NativeNotebookParser().extract(
        source, parsed_dir=tmp_path, asset_dir=tmp_path, base_name="guide"
    )

    rendered = "\n".join(block["content"] for block in blocks)
    assert "Hello notebook." in rendered
    assert "answer = 42" in rendered
    assert "secret output" not in rendered
    assert metadata["notebook_code_cells"] == 1
    assert metadata["notebook_markdown_cells"] == 1
    assert not warnings


def test_notebook_writes_ipynb_sidecar(tmp_path: Path):
    source = tmp_path / "guide.ipynb"
    _write_notebook(
        source,
        [
            {"cell_type": "markdown", "source": "# Guide\n\nHello notebook."},
            {"cell_type": "code", "source": "answer = 42"},
        ],
    )
    parser = NativeNotebookParser()
    parsed_dir = tmp_path / "guide.ipynb.parsed"
    asset_dir = parsed_dir / "guide.blocks.assets"
    parsed_dir.mkdir()
    asset_dir.mkdir()
    blocks, _, metadata = parser.extract(
        source, parsed_dir=parsed_dir, asset_dir=asset_dir, base_name="guide"
    )
    ir = parser.build_ir(
        blocks,
        document_name=source.name,
        asset_dir_name=asset_dir.name,
        metadata=metadata,
    )

    write_sidecar(
        ir,
        parsed_dir=parsed_dir,
        doc_id="doc-test",
        engine="native",
        clean_parsed_dir=False,
        block_drawing_path_style=parser.sidecar_path_style,
    )

    lines = [
        json.loads(line)
        for line in (parsed_dir / "guide.blocks.jsonl").read_text().splitlines()
    ]
    assert lines[0]["document_format"] == "ipynb"
    assert lines[0]["doc_title"] == "Guide"
    assert any("answer = 42" in line.get("content", "") for line in lines)


def test_notebook_code_fence_outgrows_backticks_in_source():
    rendered = NativeNotebookParser._fenced_code("value = ```example```", "python")

    assert rendered.startswith("````python\n")
    assert rendered.endswith("\n````")


@pytest.mark.parametrize(
    "cells, message",
    [
        ([{"cell_type": "code", "source": 42}], "must be a string or list of strings"),
        ([{"cell_type": "unknown", "source": "x"}], "unsupported cell type"),
    ],
)
def test_notebook_rejects_invalid_cell_shape(
    tmp_path: Path, cells: list[dict], message: str
):
    source = tmp_path / "invalid.ipynb"
    _write_notebook(source, cells)

    with pytest.raises(ValueError, match=message):
        NativeNotebookParser._notebook_to_markdown(source)
