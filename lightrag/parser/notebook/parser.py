"""Native Jupyter notebook parser built on the Markdown parsing pipeline."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from lightrag.constants import PARSER_ENGINE_NATIVE
from lightrag.parser.markdown.parser import NativeMarkdownParser
from lightrag.parser.native_base import NativeExtractRuntime
from lightrag.utils import logger


class NativeNotebookParser(NativeMarkdownParser):
    """Convert notebook text and code cells into Markdown before extraction."""

    engine_name = PARSER_ENGINE_NATIVE
    empty_content_label = "Notebook"

    def validate_source(self, source: Path, file_path: str) -> None:
        if not (
            source.exists() and source.is_file() and source.suffix.lower() == ".ipynb"
        ):
            raise ValueError(
                f"Native notebook parser does not support pending file: {file_path}"
            )

    def extract(
        self,
        source: Path,
        *,
        parsed_dir: Path,
        asset_dir: Path,
        base_name: str,
        runtime: NativeExtractRuntime | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        markdown, cell_counts = self._notebook_to_markdown(source)
        blocks, warnings, metadata = self._extract_text(markdown, bundle_root=None)
        metadata.update(cell_counts)

        ignored_params = (
            sorted(k for k, value in runtime.engine_params.items() if value)
            if runtime
            else []
        )
        if ignored_params:
            logger.warning(
                "[native_notebook] engine params %s are ignored for %s",
                ignored_params,
                source.name,
            )
            warnings["engine_params_ignored"] = 1

        return blocks, warnings, metadata

    @classmethod
    def _notebook_to_markdown(cls, source: Path) -> tuple[str, dict[str, int]]:
        try:
            notebook = json.loads(source.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid Jupyter notebook: {source.name}") from exc

        if not isinstance(notebook, dict):
            raise ValueError(
                f"Invalid Jupyter notebook: {source.name} must be an object"
            )
        cells = notebook.get("cells")
        if not isinstance(cells, list):
            raise ValueError(
                f"Invalid Jupyter notebook: {source.name} has no cells list"
            )

        language = cls._notebook_language(notebook)
        parts: list[str] = []
        counts = {
            "notebook_code_cells": 0,
            "notebook_markdown_cells": 0,
            "notebook_raw_cells": 0,
        }
        for index, cell in enumerate(cells):
            if not isinstance(cell, dict):
                raise ValueError(
                    f"Invalid Jupyter notebook: cell {index} in {source.name} is not an object"
                )
            cell_type = cell.get("cell_type")
            text = cls._cell_source(cell, index=index, source_name=source.name)
            if cell_type == "markdown":
                counts["notebook_markdown_cells"] += 1
                if text.strip():
                    parts.append(text.rstrip())
            elif cell_type == "code":
                counts["notebook_code_cells"] += 1
                if text.strip():
                    parts.append(cls._fenced_code(text, language))
            elif cell_type == "raw":
                counts["notebook_raw_cells"] += 1
                if text.strip():
                    parts.append(text.rstrip())
            else:
                raise ValueError(
                    f"Invalid Jupyter notebook: unsupported cell type {cell_type!r} "
                    f"at cell {index} in {source.name}"
                )

        return "\n\n".join(parts), counts

    @staticmethod
    def _cell_source(cell: dict[str, Any], *, index: int, source_name: str) -> str:
        value = cell.get("source", "")
        if isinstance(value, str):
            return value
        if isinstance(value, list) and all(isinstance(part, str) for part in value):
            return "".join(value)
        raise ValueError(
            f"Invalid Jupyter notebook: source for cell {index} in {source_name} "
            "must be a string or list of strings"
        )

    @staticmethod
    def _notebook_language(notebook: dict[str, Any]) -> str:
        metadata = notebook.get("metadata")
        if not isinstance(metadata, dict):
            return ""
        kernelspec = metadata.get("kernelspec")
        language_info = metadata.get("language_info")
        candidates = (
            kernelspec.get("language") if isinstance(kernelspec, dict) else None,
            language_info.get("name") if isinstance(language_info, dict) else None,
        )
        for candidate in candidates:
            if isinstance(candidate, str) and re.fullmatch(
                r"[A-Za-z0-9_+.-]+", candidate
            ):
                return candidate
        return ""

    @staticmethod
    def _fenced_code(code: str, language: str) -> str:
        runs = re.findall(r"`+", code)
        fence = "`" * max([3, *(len(run) + 1 for run in runs)])
        language_suffix = language if language else ""
        return f"{fence}{language_suffix}\n{code.rstrip()}\n{fence}"
