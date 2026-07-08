# BM25 Entity Seeding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional BM25 keyword index over entity names as a parallel seed path beside vector search in LightRAG's local retrieval, fused via Reciprocal Rank Fusion (upstream RFC HKUDS/LightRAG#3198, MVP scope).

**Architecture:** A fifth pluggable storage role `KEYWORD_STORAGE` (interface `BaseKeywordStorage` in `lightrag/base.py`, default impl `Bm25KeywordStorage` in `lightrag/kg/bm25s_keyword_impl.py` backed by the optional `bm25s` library and the existing `TiktokenTokenizer`). The index is rebuilt from the graph's full entity-name set after each ingestion batch. At query time, `_get_node_data` optionally runs a BM25 search in parallel with the entity vector search and fuses both ranked lists with RRF before graph expansion. Feature is off by default; every failure path falls back to vector-only.

**Tech Stack:** Python 3.10+, asyncio, dataclasses, `bm25s` (optional extra, lazy import), tiktoken (already a dependency), pytest (mock-based).

**Spec:** `docs/superpowers/specs/2026-07-08-bm25-entity-seeding-design.md`

## Global Constraints

- Code, comments, and log messages in **English** (AGENTS.md).
- `bm25s` must NOT become a hard dependency: lazy import, optional extra, graceful degradation when missing.
- Default behavior with the feature disabled must be byte-for-byte identical to current behavior.
- A query must never fail because of this feature (catch → `logger.warning` → vector-only fallback).
- Tests are mock-based (no live services), live under `tests/` mirroring `lightrag/` layout, new dirs get an empty `__init__.py`.
- Run tests via `./scripts/test.sh <path>`; lint via `ruff check .`.
- Use `lightrag.utils.logger`, never `print`.
- All new async APIs follow existing storage patterns (`@dataclass`, `StorageNameSpace` base).

---

### Task 1: `BaseKeywordStorage` interface + namespace + registry

**Files:**
- Modify: `lightrag/base.py` (append new ABC after `BaseGraphStorage`, i.e. before `class DocStatus` around line 793)
- Modify: `lightrag/namespace.py` (add namespace constant)
- Modify: `lightrag/kg/__init__.py` (registry entries)
- Test: `tests/kg/test_keyword_storage_registry.py`

**Interfaces:**
- Consumes: `StorageNameSpace` (base.py:161) — fields `namespace, workspace, global_config`; abstract `index_done_callback`, `drop`.
- Produces (later tasks rely on these exact names):
  - `class BaseKeywordStorage(StorageNameSpace, ABC)` with
    `async def index_entities(self, names: list[str]) -> None`,
    `async def search(self, query: str, top_k: int) -> list[tuple[str, float]]`
  - `NameSpace.KEYWORD_STORE_ENTITY_NAMES = "entity_keywords"`
  - Registry key `"KEYWORD_STORAGE"`, impl name `"Bm25KeywordStorage"`, module `".kg.bm25s_keyword_impl"`.

- [ ] **Step 1: Write the failing test**

Create `tests/kg/test_keyword_storage_registry.py`:

```python
"""Registry and interface contract for the KEYWORD_STORAGE role."""

import inspect

from lightrag.kg import STORAGE_IMPLEMENTATIONS, STORAGE_ENV_REQUIREMENTS, STORAGES


def test_keyword_storage_role_registered():
    role = STORAGE_IMPLEMENTATIONS["KEYWORD_STORAGE"]
    assert "Bm25KeywordStorage" in role["implementations"]
    assert "index_entities" in role["required_methods"]
    assert "search" in role["required_methods"]


def test_bm25_keyword_storage_env_requirements_empty():
    assert STORAGE_ENV_REQUIREMENTS["Bm25KeywordStorage"] == []


def test_bm25_keyword_storage_module_mapped():
    assert STORAGES["Bm25KeywordStorage"] == ".kg.bm25s_keyword_impl"


def test_base_keyword_storage_interface():
    from lightrag.base import BaseKeywordStorage

    assert inspect.isabstract(BaseKeywordStorage)
    sig = inspect.signature(BaseKeywordStorage.search)
    assert list(sig.parameters) == ["self", "query", "top_k"]
    sig = inspect.signature(BaseKeywordStorage.index_entities)
    assert list(sig.parameters) == ["self", "names"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./scripts/test.sh tests/kg/test_keyword_storage_registry.py`
Expected: FAIL — `KeyError: 'KEYWORD_STORAGE'` (and ImportError for `BaseKeywordStorage`).

- [ ] **Step 3: Implement**

In `lightrag/namespace.py`, inside `class NameSpace` after `DOC_STATUS = "doc_status"`:

```python
    KEYWORD_STORE_ENTITY_NAMES = "entity_keywords"
```

In `lightrag/base.py`, insert after the end of `BaseGraphStorage` (immediately before `class DocStatus(str, Enum):`):

```python
@dataclass
class BaseKeywordStorage(StorageNameSpace, ABC):
    """Sparse keyword (BM25) index over entity names.

    Serves as a parallel seed path beside dense vector search: exact term
    matches on rare jargon survive even when embeddings fail. The index is
    derived data — it can always be rebuilt from the graph's entity-name
    set, so implementations favor idempotent full rebuilds over incremental
    updates.
    """

    @abstractmethod
    async def index_entities(self, names: list[str]) -> None:
        """(Re)build the index from the full entity-name corpus."""

    @abstractmethod
    async def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Return up to ``top_k`` ``(entity_name, score)`` pairs, best first.

        Must return an empty list (never raise) when the index is empty or
        unavailable.
        """
```

In `lightrag/kg/__init__.py`:

1. Add to `STORAGE_IMPLEMENTATIONS` (after the `"DOC_STATUS_STORAGE"` entry):

```python
    "KEYWORD_STORAGE": {
        "implementations": [
            "Bm25KeywordStorage",
        ],
        "required_methods": ["index_entities", "search"],
    },
```

2. Add to `STORAGE_ENV_REQUIREMENTS` (in the doc-status section, after `"MongoDocStatusStorage"`):

```python
    # Keyword (BM25) Storage Implementations
    "Bm25KeywordStorage": [],
```

3. Add to `STORAGES`:

```python
    "Bm25KeywordStorage": ".kg.bm25s_keyword_impl",
```

Note: `lightrag/kg/factory.py` needs NO change — non-default backends resolve
through the `STORAGES` registry dynamically. The module will exist after
Task 3; the registry test above does not import it.

- [ ] **Step 4: Run test to verify it passes**

Run: `./scripts/test.sh tests/kg/test_keyword_storage_registry.py`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add lightrag/base.py lightrag/namespace.py lightrag/kg/__init__.py tests/kg/test_keyword_storage_registry.py
git commit -m "feat: add BaseKeywordStorage interface and KEYWORD_STORAGE role"
```

---

### Task 2: RRF seed fusion helper

**Files:**
- Modify: `lightrag/operate.py` (add a module-level pure function; place it directly above `async def _get_node_data` at ~line 5144)
- Test: `tests/query/__init__.py` (new, empty), `tests/query/test_seed_fusion.py`

**Interfaces:**
- Consumes: nothing (pure function).
- Produces (Task 5 relies on this exact signature):

```python
def fuse_seed_rankings(
    vector_names: list[str],
    bm25_names: list[str],
    top_k: int,
    rrf_k: int = 60,
) -> list[tuple[str, str]]
```

Returns fused `[(entity_name, source)]`, best first, at most `top_k` items, `source ∈ {"vector", "bm25", "both"}`.

- [ ] **Step 1: Write the failing test**

Create `tests/query/__init__.py` (empty file) and `tests/query/test_seed_fusion.py`:

```python
"""RRF fusion of vector and BM25 seed rankings."""

from lightrag.operate import fuse_seed_rankings


def test_consensus_entity_ranks_first():
    # "shared" appears on both lists -> summed RRF contributions win.
    fused = fuse_seed_rankings(
        vector_names=["a", "shared", "b"],
        bm25_names=["shared", "c"],
        top_k=10,
    )
    names = [n for n, _ in fused]
    assert names[0] == "shared"
    assert dict(fused)["shared"] == "both"


def test_single_list_high_rank_survives():
    # Entity only on the BM25 list at rank 1 must beat deep vector tails.
    fused = fuse_seed_rankings(
        vector_names=[f"v{i}" for i in range(30)],
        bm25_names=["jargon"],
        top_k=5,
    )
    names = [n for n, _ in fused]
    assert "jargon" in names


def test_dedup_and_top_k_truncation():
    fused = fuse_seed_rankings(
        vector_names=["a", "b", "c"],
        bm25_names=["b", "d"],
        top_k=3,
    )
    names = [n for n, _ in fused]
    assert len(names) == 3
    assert len(set(names)) == 3


def test_empty_bm25_returns_vector_order():
    fused = fuse_seed_rankings(["a", "b"], [], top_k=10)
    assert [n for n, _ in fused] == ["a", "b"]
    assert all(src == "vector" for _, src in fused)


def test_empty_both_returns_empty():
    assert fuse_seed_rankings([], [], top_k=10) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./scripts/test.sh tests/query/test_seed_fusion.py`
Expected: FAIL — `ImportError: cannot import name 'fuse_seed_rankings'`.

- [ ] **Step 3: Implement**

In `lightrag/operate.py`, directly above `async def _get_node_data`:

```python
def fuse_seed_rankings(
    vector_names: list[str],
    bm25_names: list[str],
    top_k: int,
    rrf_k: int = 60,
) -> list[tuple[str, str]]:
    """Fuse two ranked entity-name lists with Reciprocal Rank Fusion.

    Score(name) = sum over lists of 1 / (rrf_k + rank). Dedup is inherent:
    a name on both lists sums both contributions, so consensus floats to
    the top while a high rank on a single list still survives. Truncation
    to ``top_k`` happens AFTER fusion — discarding tails beforehand would
    wrongly kill names ranked low on one list but high on the other.
    """
    scores: dict[str, float] = {}
    sources: dict[str, set[str]] = {}
    for source, names in (("vector", vector_names), ("bm25", bm25_names)):
        for rank, name in enumerate(names):
            scores[name] = scores.get(name, 0.0) + 1.0 / (rrf_k + rank + 1)
            sources.setdefault(name, set()).add(source)
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    result: list[tuple[str, str]] = []
    for name, _score in ordered[:top_k]:
        src = sources[name]
        result.append((name, "both" if len(src) == 2 else next(iter(src))))
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./scripts/test.sh tests/query/test_seed_fusion.py`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add lightrag/operate.py tests/query/__init__.py tests/query/test_seed_fusion.py
git commit -m "feat: add RRF fusion helper for hybrid seed rankings"
```

---

### Task 3: `Bm25KeywordStorage` implementation + optional dependency

**Files:**
- Create: `lightrag/kg/bm25s_keyword_impl.py`
- Modify: `pyproject.toml` (optional extra)
- Test: `tests/kg/bm25s_impl/__init__.py` (new, empty), `tests/kg/bm25s_impl/test_bm25_keyword_storage.py`

**Interfaces:**
- Consumes: `BaseKeywordStorage` (Task 1), `lightrag.utils.TiktokenTokenizer`, `lightrag.utils.logger`, `lightrag.utils.load_json` / `write_json`.
- Produces: `Bm25KeywordStorage` — constructed as
  `Bm25KeywordStorage(namespace=..., workspace=..., global_config={"working_dir": ...})`;
  `await initialize()` loads persisted names and builds the in-memory index;
  `await index_entities(names)` rebuilds + persists;
  `await search(query, top_k)` returns `[(entity_name, score)]`;
  class attribute/property `available: bool` is False when `bm25s` is not importable.

- [ ] **Step 1: Add the optional dependency**

In `pyproject.toml`, locate `[project.optional-dependencies]` and add a new
extra (keep existing extras untouched):

```toml
bm25 = [
    "bm25s>=0.2.0",
]
```

Also install it in the dev venv so tests can run: `uv pip install bm25s` (or
`pip install bm25s` in the active venv).

- [ ] **Step 2: Write the failing test**

Create `tests/kg/bm25s_impl/__init__.py` (empty) and
`tests/kg/bm25s_impl/test_bm25_keyword_storage.py`:

```python
"""Bm25KeywordStorage: build/search/persist/degrade."""

import pytest

pytest.importorskip("bm25s")

from lightrag.kg.bm25s_keyword_impl import Bm25KeywordStorage  # noqa: E402


def make_storage(tmp_path, workspace=""):
    return Bm25KeywordStorage(
        namespace="entity_keywords",
        workspace=workspace,
        global_config={"working_dir": str(tmp_path)},
    )


@pytest.mark.asyncio
async def test_index_and_exact_term_search(tmp_path):
    st = make_storage(tmp_path)
    await st.initialize()
    await st.index_entities(["NVLink", "PCIe", "Apple Inc.", "苹果公司"])
    hits = await st.search("what is the bandwidth of NVLink?", top_k=3)
    assert hits, "expected at least one hit"
    assert hits[0][0] == "NVLink"


@pytest.mark.asyncio
async def test_cjk_entity_search(tmp_path):
    st = make_storage(tmp_path)
    await st.initialize()
    await st.index_entities(["苹果公司", "富士康", "NVIDIA"])
    hits = await st.search("苹果公司的供应链", top_k=2)
    assert hits[0][0] == "苹果公司"


@pytest.mark.asyncio
async def test_empty_index_returns_empty(tmp_path):
    st = make_storage(tmp_path)
    await st.initialize()
    assert await st.search("anything", top_k=5) == []


@pytest.mark.asyncio
async def test_persistence_round_trip(tmp_path):
    st = make_storage(tmp_path)
    await st.initialize()
    await st.index_entities(["NVLink", "PCIe"])
    # New instance over the same directory must reload the corpus.
    st2 = make_storage(tmp_path)
    await st2.initialize()
    hits = await st2.search("NVLink", top_k=1)
    assert hits and hits[0][0] == "NVLink"


@pytest.mark.asyncio
async def test_workspace_isolation(tmp_path):
    a = make_storage(tmp_path, workspace="tenant_a")
    b = make_storage(tmp_path, workspace="tenant_b")
    await a.initialize()
    await b.initialize()
    await a.index_entities(["OnlyInA"])
    assert await b.search("OnlyInA", top_k=1) == []


@pytest.mark.asyncio
async def test_top_k_larger_than_corpus(tmp_path):
    st = make_storage(tmp_path)
    await st.initialize()
    await st.index_entities(["NVLink"])
    hits = await st.search("NVLink", top_k=50)
    assert len(hits) == 1


@pytest.mark.asyncio
async def test_rebuild_replaces_corpus(tmp_path):
    st = make_storage(tmp_path)
    await st.initialize()
    await st.index_entities(["OldEntity"])
    await st.index_entities(["NewEntity"])  # full rebuild semantics
    assert await st.search("OldEntity", top_k=5) == []
    hits = await st.search("NewEntity", top_k=5)
    assert hits and hits[0][0] == "NewEntity"


@pytest.mark.asyncio
async def test_drop_clears_everything(tmp_path):
    st = make_storage(tmp_path)
    await st.initialize()
    await st.index_entities(["NVLink"])
    await st.drop()
    assert await st.search("NVLink", top_k=5) == []


@pytest.mark.asyncio
async def test_unavailable_when_bm25s_missing(tmp_path, monkeypatch):
    import lightrag.kg.bm25s_keyword_impl as mod

    monkeypatch.setattr(mod, "_import_bm25s", lambda: None)
    st = make_storage(tmp_path)
    await st.initialize()
    assert st.available is False
    await st.index_entities(["NVLink"])  # must not raise
    assert await st.search("NVLink", top_k=5) == []
```

- [ ] **Step 3: Run test to verify it fails**

Run: `./scripts/test.sh tests/kg/bm25s_impl/test_bm25_keyword_storage.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'lightrag.kg.bm25s_keyword_impl'`.

- [ ] **Step 4: Implement**

Create `lightrag/kg/bm25s_keyword_impl.py`:

```python
"""BM25 keyword index over entity names, backed by the optional ``bm25s`` lib.

Terms are the string forms of TiktokenTokenizer BPE ids: the same encoder is
used for both the corpus and queries, which makes matching language-agnostic
(CJK included) without any bespoke tokenizer. The whole index is rebuilt on
every ``index_entities`` call — the corpus is short entity-name strings, so a
full rebuild is milliseconds and idempotently covers deletions.

Persistence: only the raw name list is persisted (JSON); the bm25s object is
rebuilt from it on ``initialize``. This avoids depending on bm25s'
serialization format. Single-process semantics (like NanoVectorDBStorage):
each process keeps its own in-memory index and reloads from the JSON file on
startup.
"""

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
        self._tokenizer = TiktokenTokenizer()
        data = load_json(self._file_name)
        names = data.get("entity_names", []) if isinstance(data, dict) else []
        if names:
            self._names = names
            self._build_index()

    def _term_string(self, text: str) -> str:
        """Encode text to BPE ids and join as a whitespace-separated string.

        The leading space normalizes tiktoken's boundary-sensitive merges for
        the first token; bm25s then tokenizes on whitespace with no stopwords
        or stemming (the "terms" are numeric id strings).
        """
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
        # Deduplicate while preserving order; drop empty names defensively.
        seen: set[str] = set()
        unique_names = [
            n for n in names if n and not (n in seen or seen.add(n))
        ]
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
        except Exception as e:  # noqa: BLE001 — this path must never raise
            logger.warning(f"[bm25] search failed, returning no seeds: {e}")
            return []

    async def index_done_callback(self) -> None:
        # index_entities persists synchronously; nothing to flush here.
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
```

Note for the implementer: check `lightrag/utils.py` for the actual
`load_json` return on a missing file (it returns `None`) — the
`isinstance(data, dict)` guard above covers that. If `write_json`'s
signature is `write_json(json_obj, file_name)` keep the call as written;
verify against an existing caller in `json_kv_impl.py` before running.

- [ ] **Step 5: Run test to verify it passes**

Run: `./scripts/test.sh tests/kg/bm25s_impl/test_bm25_keyword_storage.py`
Expected: 9 passed. If a bm25s API mismatch surfaces (e.g. `retrieve`
argument names), adapt the impl (not the tests) using bm25s' documented API.

- [ ] **Step 6: Commit**

```bash
git add lightrag/kg/bm25s_keyword_impl.py pyproject.toml tests/kg/bm25s_impl/
git commit -m "feat: add Bm25KeywordStorage backed by optional bm25s dependency"
```

---

### Task 4: Wire the storage into `LightRAG`

**Files:**
- Modify: `lightrag/lightrag.py`:
  - dataclass field near `doc_status_storage` (line ~284)
  - storage verification list (`storage_configs`, lines ~1007-1012)
  - instantiation after `self.doc_status` (line ~1190)
  - the storage list used by `initialize_storages` / `finalize_storages`
  - `_insert_done` rebuild hook
- Test: `tests/kg/test_keyword_storage_registry.py` (extend)

**Interfaces:**
- Consumes: `get_storage_class` (kg/factory.py), `NameSpace.KEYWORD_STORE_ENTITY_NAMES` (Task 1), `Bm25KeywordStorage` (Task 3), `BaseGraphStorage.get_all_labels()` (base.py:727).
- Produces: `self.entity_keywords: BaseKeywordStorage` attribute on `LightRAG`; dataclass field `keyword_storage: str = "Bm25KeywordStorage"`. Task 5/6 rely on `self.entity_keywords`.

- [ ] **Step 1: Extend the registry test with a resolution check**

Append to `tests/kg/test_keyword_storage_registry.py`:

```python
def test_factory_resolves_bm25_keyword_storage():
    pytest_bm25s = __import__("pytest").importorskip("bm25s")  # noqa: F841
    from lightrag.kg.factory import get_storage_class
    from lightrag.base import BaseKeywordStorage

    cls = get_storage_class("Bm25KeywordStorage")
    assert issubclass(cls, BaseKeywordStorage)
```

Run: `./scripts/test.sh tests/kg/test_keyword_storage_registry.py`
Expected: PASS already (registry + module exist since Task 3). This is a
regression pin, not a red test.

- [ ] **Step 2: Add the dataclass field**

In `lightrag/lightrag.py`, next to `doc_status_storage: str = field(default="JsonDocStatusStorage")` (line ~284), add:

```python
    keyword_storage: str = field(default="Bm25KeywordStorage")
    """Storage backend for the BM25 entity-name keyword index."""
```

- [ ] **Step 3: Add verification + instantiation**

In `__post_init__`, extend `storage_configs` (lines ~1007-1012):

```python
            ("KEYWORD_STORAGE", self.keyword_storage),
```

After the `self.doc_status` instantiation block (line ~1190), add:

```python
        # Keyword (BM25) storage for hybrid entity seeding
        keyword_storage_cls = get_storage_class(self.keyword_storage)
        self.entity_keywords: BaseKeywordStorage = keyword_storage_cls(  # type: ignore
            namespace=NameSpace.KEYWORD_STORE_ENTITY_NAMES,
            workspace=self.workspace,
            global_config=global_config,
        )
```

Add `BaseKeywordStorage` to the existing `from lightrag.base import (...)` block in `lightrag.py`.

- [ ] **Step 4: Register in the storage lifecycle**

Locate the storage list in `initialize_storages` (grep for `initialize_storages` in `lightrag/lightrag.py`; it iterates a tuple/list containing `self.full_docs, self.text_chunks, ... self.doc_status`). Append `self.entity_keywords` to that collection. Do the same in `finalize_storages` and in any `drop`-all flow that enumerates the same list (grep for `self.doc_status,` to find every enumeration site).

- [ ] **Step 5: Rebuild hook in `_insert_done`**

Locate `async def _insert_done` in `lightrag/lightrag.py`. At its end (after existing `index_done_callback` fan-out), add:

```python
        # Rebuild the BM25 entity-name index from the graph. Derived data:
        # failures must never block ingestion.
        try:
            if getattr(self.entity_keywords, "available", False):
                labels = await self.chunk_entity_relation_graph.get_all_labels()
                await self.entity_keywords.index_entities(labels)
        except Exception as e:
            logger.warning(f"[bm25] entity keyword index rebuild failed: {e}")
```

- [ ] **Step 6: Run the pinned tests + a smoke import**

Run: `./scripts/test.sh tests/kg/test_keyword_storage_registry.py tests/kg/bm25s_impl`
Expected: all pass.
Run: `python -c "import lightrag.lightrag"` (catches syntax/import errors).

- [ ] **Step 7: Commit**

```bash
git add lightrag/lightrag.py tests/kg/test_keyword_storage_registry.py
git commit -m "feat: wire keyword storage into LightRAG lifecycle and rebuild on insert"
```

---

### Task 5: Query-path integration (`QueryParam` flag + plumbing + fusion)

**Files:**
- Modify: `lightrag/base.py` (QueryParam field, after `include_references` line ~157)
- Modify: `lightrag/operate.py`:
  - `kg_query` signature (line ~3786) + its `_build_query_context` call
  - `_build_query_context` signature (line ~5024) + its `_perform_kg_search` call (line ~5047)
  - `_perform_kg_search` signature (line ~4315) + both `_get_node_data` call sites (lines ~4401, ~4420)
  - `_get_node_data` (line ~5144): BM25 branch + fusion
- Modify: `lightrag/lightrag.py`: every `kg_query(` call site — pass `keyword_storage=self.entity_keywords`
- Test: `tests/query/test_bm25_seeding_integration.py`

**Interfaces:**
- Consumes: `fuse_seed_rankings` (Task 2), `BaseKeywordStorage.search` (Task 1), `self.entity_keywords` (Task 4).
- Produces: `QueryParam.enable_bm25_seeding: bool` (env default `ENABLE_BM25_SEEDING`, false); every plumbed function gains keyword-only param `keyword_storage: "BaseKeywordStorage | None" = None`.

- [ ] **Step 1: Write the failing tests**

Create `tests/query/test_bm25_seeding_integration.py`:

```python
"""_get_node_data hybrid seeding: flag-off regression, fusion, fallback."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from lightrag.base import QueryParam
from lightrag.operate import _get_node_data


def make_graph(entity_names):
    graph = MagicMock()
    graph.get_nodes_batch = AsyncMock(
        return_value={
            name: {"entity_name": name, "description": f"desc of {name}"}
            for name in entity_names
        }
    )
    graph.node_degrees_batch = AsyncMock(
        return_value={name: 1 for name in entity_names}
    )
    graph.get_nodes_edges_batch = AsyncMock(
        return_value={name: [] for name in entity_names}
    )
    return graph


def make_vdb(entity_names):
    vdb = MagicMock()
    vdb.cosine_better_than_threshold = 0.2
    vdb.query = AsyncMock(
        return_value=[
            {"entity_name": name, "created_at": None} for name in entity_names
        ]
    )
    return vdb


def make_keyword_storage(hits):
    ks = MagicMock()
    ks.search = AsyncMock(return_value=hits)
    return ks


@pytest.mark.asyncio
async def test_flag_off_is_vector_only_and_never_touches_keyword_storage():
    graph = make_graph(["A", "B"])
    vdb = make_vdb(["A", "B"])
    ks = make_keyword_storage([("C", 9.0)])
    param = QueryParam(mode="local", top_k=10)
    param.enable_bm25_seeding = False

    node_datas, _ = await _get_node_data(
        "q", graph, vdb, param, keyword_storage=ks
    )
    ks.search.assert_not_awaited()
    assert [n["entity_name"] for n in node_datas] == ["A", "B"]


@pytest.mark.asyncio
async def test_bm25_only_entity_joins_seeds():
    graph = make_graph(["A", "B", "NVLink"])
    vdb = make_vdb(["A", "B"])
    ks = make_keyword_storage([("NVLink", 12.5)])
    param = QueryParam(mode="local", top_k=10)
    param.enable_bm25_seeding = True

    node_datas, _ = await _get_node_data(
        "NVLink bandwidth?", graph, vdb, param, keyword_storage=ks
    )
    names = [n["entity_name"] for n in node_datas]
    assert "NVLink" in names


@pytest.mark.asyncio
async def test_keyword_storage_exception_falls_back_to_vector_only():
    graph = make_graph(["A"])
    vdb = make_vdb(["A"])
    ks = MagicMock()
    ks.search = AsyncMock(side_effect=RuntimeError("index corrupted"))
    param = QueryParam(mode="local", top_k=10)
    param.enable_bm25_seeding = True

    node_datas, _ = await _get_node_data(
        "q", graph, vdb, param, keyword_storage=ks
    )
    assert [n["entity_name"] for n in node_datas] == ["A"]


@pytest.mark.asyncio
async def test_no_keyword_storage_behaves_as_before():
    graph = make_graph(["A"])
    vdb = make_vdb(["A"])
    param = QueryParam(mode="local", top_k=10)
    param.enable_bm25_seeding = True  # flag on but no storage wired

    node_datas, _ = await _get_node_data("q", graph, vdb, param)
    assert [n["entity_name"] for n in node_datas] == ["A"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./scripts/test.sh tests/query/test_bm25_seeding_integration.py`
Expected: FAIL — `_get_node_data` has no `keyword_storage` parameter.

- [ ] **Step 3: Add the QueryParam field**

In `lightrag/base.py`, after `include_references` (line ~157):

```python
    enable_bm25_seeding: bool = (
        os.getenv("ENABLE_BM25_SEEDING", "false").lower() == "true"
    )
    """Enable BM25 keyword search over entity names as a parallel seed path
    beside vector search (local/hybrid/mix modes). Requires the optional
    ``bm25s`` dependency; silently falls back to vector-only when the index
    is unavailable. Off by default.
    """
```

- [ ] **Step 4: Thread the parameter**

All new parameters are keyword-only with default `None`, appended after the
last existing parameter, so every existing caller stays valid.

`kg_query` (~3786): append `keyword_storage=None,` to the signature; inside
it, find the `_build_query_context(` call and pass
`keyword_storage=keyword_storage`.

`_build_query_context` (~5024): append `keyword_storage=None,`; pass through
in the `_perform_kg_search(` call (line ~5047).

`_perform_kg_search` (~4315): append `keyword_storage=None,`; pass
`keyword_storage=keyword_storage` at BOTH `_get_node_data(` call sites
(lines ~4401 and ~4420).

`_get_node_data` (~5144): append `keyword_storage=None,` to the signature.
Then replace the head of the function body — currently:

```python
    results = await entities_vdb.query(
        query, top_k=query_param.top_k, query_embedding=query_embedding
    )

    if not len(results):
        return [], []

    # Extract all entity IDs from your results list
    node_ids = [r["entity_name"] for r in results]
```

with:

```python
    results = await entities_vdb.query(
        query, top_k=query_param.top_k, query_embedding=query_embedding
    )

    # Optional BM25 seed path: exact term matches rescue jargon that dense
    # embeddings miss. Any failure falls back to vector-only silently.
    if keyword_storage is not None and query_param.enable_bm25_seeding:
        try:
            bm25_hits = await keyword_storage.search(
                query, top_k=query_param.top_k
            )
        except Exception as e:  # noqa: BLE001 — never fail the query
            logger.warning(
                f"[bm25] seed search failed, using vector-only seeds: {e}"
            )
            bm25_hits = []
        if bm25_hits:
            vector_names = [r["entity_name"] for r in results]
            fused = fuse_seed_rankings(
                vector_names,
                [name for name, _score in bm25_hits],
                top_k=query_param.top_k,
            )
            logger.info(
                "[bm25] fused seeds: "
                + ", ".join(f"{name}({src})" for name, src in fused)
            )
            by_name = {r["entity_name"]: r for r in results}
            results = [
                by_name.get(name, {"entity_name": name, "created_at": None})
                for name, _src in fused
            ]

    if not len(results):
        return [], []

    # Extract all entity IDs from your results list
    node_ids = [r["entity_name"] for r in results]
```

(BM25-only names get a minimal result record; downstream
`get_nodes_batch` fills in real node data, and genuinely missing nodes are
already dropped by the existing "Some nodes are missing" guard.)

- [ ] **Step 5: Pass the storage at the top**

In `lightrag/lightrag.py`, grep `kg_query(`; at every call site add
`keyword_storage=self.entity_keywords,` alongside the existing storage
arguments.

- [ ] **Step 6: Run tests**

Run: `./scripts/test.sh tests/query/test_bm25_seeding_integration.py tests/query/test_seed_fusion.py`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add lightrag/base.py lightrag/operate.py lightrag/lightrag.py tests/query/test_bm25_seeding_integration.py
git commit -m "feat: hybrid BM25+vector entity seeding with RRF fusion in local retrieval"
```

---

### Task 6: Server config + env template + docs

**Files:**
- Modify: `lightrag/api/config.py` (default + env read, near line ~68 and ~494)
- Modify: `lightrag/api/lightrag_server.py` (pass constructor arg, near line ~2046)
- Modify: `env.example` (document the two new env vars)
- Test: `tests/api/config/test_keyword_storage_config.py`

**Interfaces:**
- Consumes: `args`-style config plumbing as used by `doc_status_storage` (config.py:494, lightrag_server.py:2046).
- Produces: `LIGHTRAG_KEYWORD_STORAGE` env var (backend selection) and `ENABLE_BM25_SEEDING` env var (query default), both documented.

- [ ] **Step 1: Write the failing test**

Create `tests/api/config/test_keyword_storage_config.py`:

```python
"""Keyword-storage server config plumbing."""

import importlib


def test_keyword_storage_default(monkeypatch):
    monkeypatch.delenv("LIGHTRAG_KEYWORD_STORAGE", raising=False)
    import lightrag.api.config as config_mod

    importlib.reload(config_mod)
    assert config_mod.DefaultRAGStorageConfig.KEYWORD_STORAGE == "Bm25KeywordStorage"


def test_keyword_storage_env_override(monkeypatch):
    monkeypatch.setenv("LIGHTRAG_KEYWORD_STORAGE", "Bm25KeywordStorage")
    import lightrag.api.config as config_mod

    importlib.reload(config_mod)
    args = config_mod.parse_args()
    assert args.keyword_storage == "Bm25KeywordStorage"
```

Note: if `parse_args` requires CLI args in this repo's test conventions,
mirror how the sibling tests in `tests/api/config/` invoke it (check an
existing test file first and copy its invocation pattern).

- [ ] **Step 2: Run test to verify it fails**

Run: `./scripts/test.sh tests/api/config/test_keyword_storage_config.py`
Expected: FAIL — `DefaultRAGStorageConfig` has no `KEYWORD_STORAGE`.

- [ ] **Step 3: Implement**

`lightrag/api/config.py`: in `DefaultRAGStorageConfig` (line ~68) add:

```python
    KEYWORD_STORAGE = "Bm25KeywordStorage"
```

Next to the `doc_status_storage` env read (line ~494) add:

```python
    args.keyword_storage = get_env_value(
        "LIGHTRAG_KEYWORD_STORAGE", DefaultRAGStorageConfig.KEYWORD_STORAGE
    )
```

`lightrag/api/lightrag_server.py`: where the `LightRAG(...)` constructor
receives `doc_status_storage=args.doc_status_storage` (line ~2046 and the
second site ~2394), add:

```python
                        "keyword_storage": args.keyword_storage,
```

(match the surrounding dict/kwarg style at each site).

`env.example`: in the storage-selection section add:

```bash
### Keyword (BM25) storage for hybrid entity seeding (optional feature)
# LIGHTRAG_KEYWORD_STORAGE=Bm25KeywordStorage
### Enable BM25 seed path by default for queries (requires: pip install bm25s)
# ENABLE_BM25_SEEDING=false
```

- [ ] **Step 4: Run tests**

Run: `./scripts/test.sh tests/api/config/test_keyword_storage_config.py`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add lightrag/api/config.py lightrag/api/lightrag_server.py env.example tests/api/config/test_keyword_storage_config.py
git commit -m "feat: expose keyword storage and BM25 seeding via server config"
```

---

### Task 7: Full verification + lint

**Files:** none new.

- [ ] **Step 1: Full test suite**

Run: `./scripts/test.sh tests`
Expected: no regressions vs the pre-branch baseline (record pass/skip counts; compare with a `git stash`-free run on `main` if any failure looks pre-existing).

- [ ] **Step 2: Lint**

Run: `ruff check .`
Expected: clean (fix anything introduced by this branch).

- [ ] **Step 3: Manual smoke (flag-off default)**

Run: `python -c "from lightrag import LightRAG, QueryParam; print(QueryParam().enable_bm25_seeding)"`
Expected: `False`.

- [ ] **Step 4: Commit any fixes**

```bash
git add -A
git commit -m "chore: lint and test fixes for BM25 entity seeding"
```

---

## Self-Review Notes

- Spec coverage: interface+registry (T1), fusion (T2), impl+optional dep (T3), lifecycle+rebuild (T4), query integration+flag (T5), server config+env docs (T6), verification (T7). Spec's "issue comment" and "PR creation" are delivery steps outside the code plan.
- Deviation from spec (agreed with user): tokenizer reuses `TiktokenTokenizer` (no bespoke CJK tokenizer); `remove_entities` dropped in favor of idempotent full rebuild. Spec updated accordingly.
- Type consistency: `fuse_seed_rankings` (T2) consumed with same signature in T5; `search -> list[tuple[str, float]]` consumed as `[(name, score)]` in T5; `self.entity_keywords` produced in T4, consumed in T5.
- Known executor checkpoints (verify-before-use, called out inline): `write_json` argument order, bm25s `retrieve` API, `parse_args` invocation pattern in config tests, exact storage-list location in `initialize_storages`.
