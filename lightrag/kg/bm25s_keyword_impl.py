"""BM25 keyword index over entity names, backed by the optional ``bm25s`` lib."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, final

from lightrag.base import BaseKeywordStorage
from lightrag.utils import TiktokenTokenizer, load_json, logger, write_json

_BM25S_WARNED = False


def _import_bm25s():
    """Lazy import so bm25s stays an optional dependency."""
    try:
        import bm25s

        return bm25s
    except ImportError:
        return None


@final
@dataclass
class Bm25KeywordStorage(BaseKeywordStorage):
    _names: list[str] = field(default_factory=list, init=False)
    _retriever: Any = field(default=None, init=False)
    _tokenizer: Any = field(default=None, init=False)
    _bm25s: Any = field(default=None, init=False)

    def __post_init__(self):
        working_dir = self.global_config["working_dir"]
        if self.workspace:
            workspace_dir = os.path.join(working_dir, self.workspace)
        else:
            workspace_dir = working_dir
        os.makedirs(workspace_dir, exist_ok=True)
        self._file_name = os.path.join(
            workspace_dir, f"keyword_store_{self.namespace}.json"
        )

    @property
    def available(self) -> bool:
        return self._bm25s is not None

    async def initialize(self):
        global _BM25S_WARNED
        self._bm25s = _import_bm25s()
        if self._bm25s is None:
            if not _BM25S_WARNED:
                logger.warning(
                    "bm25s is not installed; BM25 entity seeding is disabled. "
                    "Install it with: pip install bm25s"
                )
                _BM25S_WARNED = True
            return
        self._tokenizer = self.global_config.get("tokenizer") or TiktokenTokenizer()
        data = load_json(self._file_name)
        names = data.get("entity_names", []) if isinstance(data, dict) else []
        if names:
            self._names = names
            self._build_index()

    def _term_string(self, text: str) -> str:
        token_ids = self._tokenizer.encode(" " + text.strip().lower())
        return " ".join(str(t) for t in token_ids)

    def _build_index(self) -> None:
        corpus_terms = self._bm25s.tokenize(
            [self._term_string(name) for name in self._names],
            stopwords=None,
            stemmer=None,
            show_progress=False,
        )
        retriever = self._bm25s.BM25()
        retriever.index(corpus_terms, show_progress=False)
        self._retriever = retriever

    async def index_entities(self, names: list[str]) -> None:
        if not self.available:
            return
        seen: set[str] = set()
        unique_names = [n for n in names if n and not (n in seen or seen.add(n))]
        self._names = unique_names
        if unique_names:
            self._build_index()
        else:
            self._retriever = None
        write_json({"entity_names": self._names}, self._file_name)
        logger.info(
            f"[bm25] rebuilt entity keyword index: {len(self._names)} names "
            f"(workspace={self.workspace or '-'})"
        )

    async def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if not self.available or self._retriever is None or not self._names:
            return []
        try:
            query_terms = self._bm25s.tokenize(
                [self._term_string(query)],
                stopwords=None,
                stemmer=None,
                show_progress=False,
            )
            k = min(top_k, len(self._names))
            docs, scores = self._retriever.retrieve(
                query_terms, corpus=self._names, k=k, show_progress=False
            )
            results: list[tuple[str, float]] = []
            for name, score in zip(docs[0], scores[0]):
                if float(score) > 0.0:
                    results.append((str(name), float(score)))
            return results
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[bm25] search failed, returning no seeds: {e}")
            return []

    async def index_done_callback(self) -> None:
        return None

    async def drop(self) -> dict[str, str]:
        self._names = []
        self._retriever = None
        try:
            if os.path.exists(self._file_name):
                os.remove(self._file_name)
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
