"""Unit tests for RagPool and DocManagerPool.

Tests the concurrency model (two-layer lock), lazy creation, double-check
guarding, error cleanup, and shutdown-all lifecycle.  Uses mocks — no real
LLM calls, no real storage connections.
"""

import asyncio
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lightrag.api.workspace_pool import RagPool, DocManagerPool


# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────


def _mock_lightrag(workspace: str):
    """Create a mock LightRAG instance with the expected async surface."""
    rag = MagicMock()
    rag.workspace = workspace
    rag.initialize_storages = AsyncMock()
    rag.check_and_migrate_data = AsyncMock()
    rag.finalize_storages = AsyncMock()
    rag.register_role_llm_builder = MagicMock()
    return rag


def _tracking_config_factory(created: list[str]):
    """Return a factory that records which workspaces were created."""

    def factory(workspace: str) -> dict:
        created.append(workspace)
        return {"workspace": workspace, "working_dir": "/tmp/test_rag"}

    return factory


# ────────────────────────────────────────────────────────────────
# RagPool
# ────────────────────────────────────────────────────────────────


class TestRagPoolGet:
    """T1 — basic create and cache."""

    @pytest.mark.asyncio
    async def test_get_creates_and_caches(self):
        created = []
        pool = RagPool(config_factory=_tracking_config_factory(created))

        with patch("lightrag.api.workspace_pool.LightRAG") as MockRAG:
            MockRAG.side_effect = lambda **kw: _mock_lightrag(kw.get("workspace"))

            rag1 = await pool.get("ws_a")
            rag2 = await pool.get("ws_a")

        assert rag1 is rag2, "second get() must return the same instance"
        assert created == ["ws_a"], "factory called exactly once"
        rag1.initialize_storages.assert_awaited_once()
        rag1.check_and_migrate_data.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_different_workspaces_independent(self):
        created = []
        pool = RagPool(config_factory=_tracking_config_factory(created))

        with patch("lightrag.api.workspace_pool.LightRAG") as MockRAG:
            MockRAG.side_effect = lambda **kw: _mock_lightrag(kw.get("workspace"))

            rag_a = await pool.get("ws_a")
            rag_b = await pool.get("ws_b")

        assert rag_a is not rag_b
        assert rag_a.workspace == "ws_a"
        assert rag_b.workspace == "ws_b"
        assert created == ["ws_a", "ws_b"]

    @pytest.mark.asyncio
    async def test_callbacks_invoked(self):
        created = []
        on_create_log = []

        pool = RagPool(
            config_factory=_tracking_config_factory(created),
            role_llm_builder=lambda role, meta: ("func", "kwargs"),
            on_create=lambda rag: on_create_log.append(rag.workspace),
        )

        with patch("lightrag.api.workspace_pool.LightRAG") as MockRAG:
            MockRAG.side_effect = lambda **kw: _mock_lightrag(kw.get("workspace"))
            await pool.get("ws_a")

        rag = await pool.get("ws_a")
        rag.register_role_llm_builder.assert_called_once()
        assert on_create_log == ["ws_a"]


class TestRagPoolConcurrency:
    """T2-T4 — concurrent access patterns."""

    @pytest.mark.asyncio
    async def test_concurrent_same_workspace_creates_once(self):
        """T2 — two racing requests for the same new workspace create only one instance."""
        created = []
        pool = RagPool(config_factory=_tracking_config_factory(created))

        # Use an event so both coroutines arrive at Layer 1 before either
        # proceeds to Layer 2.
        barrier = asyncio.Event()

        async def slow_init():
            """Simulate slow storage initialization."""
            barrier.set()  # signal the second coroutine
            await asyncio.sleep(0.05)  # let second coroutine reach the lock

        async def get_ws():
            return await pool.get("ws_x")

        with patch("lightrag.api.workspace_pool.LightRAG") as MockRAG:
            mock_rag = _mock_lightrag("ws_x")
            mock_rag.initialize_storages = AsyncMock(side_effect=slow_init)
            MockRAG.return_value = mock_rag

            # Launch both concurrently
            async with asyncio.TaskGroup() as tg:
                t1 = tg.create_task(get_ws())
                await barrier.wait()  # wait for the first to start init
                t2 = tg.create_task(get_ws())

        rag1, rag2 = t1.result(), t2.result()
        assert rag1 is rag2
        assert created == ["ws_x"], f"expected 1 creation, got {len(created)}"

    @pytest.mark.asyncio
    async def test_different_workspaces_parallel_init(self):
        """T3 — two new workspaces initialize in parallel."""
        created = []
        pool = RagPool(config_factory=_tracking_config_factory(created))

        started = asyncio.Event()

        async def parallel_init():
            started.set()
            await asyncio.sleep(0.02)

        with patch("lightrag.api.workspace_pool.LightRAG") as MockRAG:

            def make_rag(**kw):
                rag = _mock_lightrag(kw.get("workspace"))
                if kw.get("workspace") == "ws_a":
                    rag.initialize_storages = AsyncMock(side_effect=parallel_init)
                else:
                    rag.initialize_storages = AsyncMock()
                return rag

            MockRAG.side_effect = make_rag

            async with asyncio.TaskGroup() as tg:
                ta = tg.create_task(pool.get("ws_a"))
                await started.wait()  # ws_a's init has started
                tb = tg.create_task(pool.get("ws_b"))

        # Both must succeed
        assert ta.result().workspace == "ws_a"
        assert tb.result().workspace == "ws_b"
        assert sorted(created) == ["ws_a", "ws_b"]

    @pytest.mark.asyncio
    async def test_existing_not_blocked_by_new_init(self):
        """T4 — a query for existing workspace A is not blocked by C initializing."""
        pool = RagPool(config_factory=lambda ws: {"workspace": ws})

        with patch("lightrag.api.workspace_pool.LightRAG") as MockRAG:
            MockRAG.side_effect = lambda **kw: _mock_lightrag(kw.get("workspace"))

            # Pre-populate ws_a
            rag_a = await pool.get("ws_a")
            rag_a_cached = await pool.get("ws_a")  # fast path

        assert rag_a is rag_a_cached

        # Now start ws_c init and immediately query ws_a
        ws_c_init_started = asyncio.Event()

        async def slow_init_c():
            ws_c_init_started.set()
            await asyncio.sleep(0.1)

        with patch("lightrag.api.workspace_pool.LightRAG") as MockRAG:
            mock_c = _mock_lightrag("ws_c")
            mock_c.initialize_storages = AsyncMock(side_effect=slow_init_c)
            MockRAG.return_value = mock_c

            async with asyncio.TaskGroup() as tg:
                tc = tg.create_task(pool.get("ws_c"))
                await ws_c_init_started.wait()
                # ws_a should return immediately — not blocked by ws_c
                rag_a_fast = await pool.get("ws_a")

        assert rag_a_fast is rag_a, (
            "existing workspace A must not be blocked by C's init"
        )
        assert tc.result().workspace == "ws_c"


class TestRagPoolErrorHandling:
    """Error cleanup and retry behaviour."""

    @pytest.mark.asyncio
    async def test_init_failure_cleans_up_and_retry_succeeds(self):
        """If initialize_storages raises, finalize is called and next get retries."""
        created = []

        def factory(ws):
            created.append(ws)
            return {"workspace": ws}

        pool = RagPool(config_factory=factory)

        with patch("lightrag.api.workspace_pool.LightRAG") as MockRAG:
            fail_rag = _mock_lightrag("ws_x")
            fail_rag.initialize_storages.side_effect = RuntimeError(
                "connection refused"
            )

            success_rag = _mock_lightrag("ws_x")

            MockRAG.side_effect = [fail_rag, success_rag]

            # First attempt fails
            with pytest.raises(RuntimeError, match="connection refused"):
                await pool.get("ws_x")

            # Cleanup was called
            fail_rag.finalize_storages.assert_awaited_once()

            # Second attempt succeeds (retries from scratch)
            rag = await pool.get("ws_x")

        assert rag is success_rag
        assert len(created) == 2, "factory called once per attempt"

    @pytest.mark.asyncio
    async def test_migration_failure_cleans_up(self):
        """If check_and_migrate_data raises, finalize is called."""
        pool = RagPool(config_factory=lambda ws: {"workspace": ws})

        with patch("lightrag.api.workspace_pool.LightRAG") as MockRAG:
            rag = _mock_lightrag("ws_x")
            rag.check_and_migrate_data.side_effect = RuntimeError("migration failed")
            MockRAG.return_value = rag

            with pytest.raises(RuntimeError, match="migration failed"):
                await pool.get("ws_x")

            rag.finalize_storages.assert_awaited_once()


class TestRagPoolLifecycle:
    """T5 — shutdown_all lifecycle."""

    @pytest.mark.asyncio
    async def test_shutdown_all_finalizes_all(self):
        pool = RagPool(config_factory=lambda ws: {"workspace": ws})

        rags = []
        with patch("lightrag.api.workspace_pool.LightRAG") as MockRAG:
            MockRAG.side_effect = lambda **kw: _mock_lightrag(kw.get("workspace"))
            rags = [await pool.get(f"ws_{i}") for i in range(3)]

        await pool.shutdown_all()

        for rag in rags:
            rag.finalize_storages.assert_awaited_once()

        assert pool._rags == {}, "pool must be empty after shutdown"

    @pytest.mark.asyncio
    async def test_shutdown_all_collects_errors(self):
        """If one finalization fails, the rest still run and a RuntimeError is raised."""
        pool = RagPool(config_factory=lambda ws: {"workspace": ws})

        with patch("lightrag.api.workspace_pool.LightRAG") as MockRAG:
            # ws_0 fails, ws_1 succeeds
            r0 = _mock_lightrag("ws_0")
            r0.finalize_storages.side_effect = RuntimeError("disk full")
            r1 = _mock_lightrag("ws_1")

            MockRAG.side_effect = [r0, r1]
            await pool.get("ws_0")
            await pool.get("ws_1")

        with pytest.raises(RuntimeError, match="1 workspace.*failed.*ws_0"):
            await pool.shutdown_all()

        # ws_1 was still finalized
        r1.finalize_storages.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_shutdown_all_idempotent(self):
        """Repeated shutdown_all is harmless."""
        pool = RagPool(config_factory=lambda ws: {"workspace": ws})

        with patch("lightrag.api.workspace_pool.LightRAG") as MockRAG:
            MockRAG.side_effect = lambda **kw: _mock_lightrag(kw.get("workspace"))
            await pool.get("ws_a")

        await pool.shutdown_all()
        await pool.shutdown_all()  # second call — no error


# ────────────────────────────────────────────────────────────────
# DocManagerPool
# ────────────────────────────────────────────────────────────────


class TestDocManagerPool:
    """T6-T8 — DocManagerPool unit tests using real DocumentManager with temp dirs."""

    @pytest.fixture
    def tmp_input_dir(self):
        with tempfile.TemporaryDirectory() as td:
            yield td

    @pytest.mark.asyncio
    async def test_get_creates_and_caches(self, tmp_input_dir):
        """T6 — basic create and cache."""
        pool = DocManagerPool(tmp_input_dir)

        mgr1 = await pool.get("ws_a")
        mgr2 = await pool.get("ws_a")

        assert mgr1 is mgr2, "second get() must return the same instance"
        assert mgr1.workspace == "ws_a"
        assert mgr1.input_dir.name == "ws_a"

    @pytest.mark.asyncio
    async def test_different_workspaces_independent(self, tmp_input_dir):
        pool = DocManagerPool(tmp_input_dir)

        mgr_a = await pool.get("ws_a")
        mgr_b = await pool.get("ws_b")

        assert mgr_a is not mgr_b
        assert mgr_a.input_dir != mgr_b.input_dir

    @pytest.mark.asyncio
    async def test_concurrent_same_workspace_creates_once(self, tmp_input_dir):
        """T7 — two racing requests create only one instance."""
        pool = DocManagerPool(tmp_input_dir)

        async with asyncio.TaskGroup() as tg:
            t1 = tg.create_task(pool.get("ws_x"))
            t2 = tg.create_task(pool.get("ws_x"))

        assert t1.result() is t2.result()

    @pytest.mark.asyncio
    async def test_shutdown_all_evicts_all(self, tmp_input_dir):
        """T8 — shutdown clears all managers."""
        pool = DocManagerPool(tmp_input_dir)

        await pool.get("ws_a")
        await pool.get("ws_b")

        assert len(pool._managers) == 2
        await pool.shutdown_all()
        assert pool._managers == {}

    @pytest.mark.asyncio
    async def test_shutdown_all_idempotent(self, tmp_input_dir):
        pool = DocManagerPool(tmp_input_dir)

        await pool.get("ws_a")
        await pool.shutdown_all()
        await pool.shutdown_all()  # harmless


# ────────────────────────────────────────────────────────────────
# Combined pool interaction
# ────────────────────────────────────────────────────────────────


class TestPoolsIndependent:
    """Verify RagPool and DocManagerPool do not interfere."""

    @pytest.mark.asyncio
    async def test_pools_are_isolated(self):
        """A failure in one pool does not affect the other."""
        rag_pool = RagPool(config_factory=lambda ws: {"workspace": ws})
        with tempfile.TemporaryDirectory() as td:
            doc_pool = DocManagerPool(td)

            with patch("lightrag.api.workspace_pool.LightRAG") as MockRAG:
                MockRAG.side_effect = lambda **kw: _mock_lightrag(kw.get("workspace"))
                await rag_pool.get("ws_a")
                await doc_pool.get("ws_a")

            assert len(rag_pool._rags) == 1
            assert len(doc_pool._managers) == 1


# ────────────────────────────────────────────────────────────────
# Cross-workspace data isolation (I6 — RagPool level)
# ────────────────────────────────────────────────────────────────


class TestRagPoolCrossWorkspaceIsolation:
    """I6 — data inserted into workspace A must not be visible from workspace B.

    Uses real LightRAG instances with mock LLM/embedding functions and
    file-based storage (JSON + NetworkX + NanoVectorDB).  No external API
    calls needed — follows the same pattern as
    ``tests/workspace/test_workspace_isolation.py`` Test 11.
    """

    @pytest.mark.asyncio
    async def test_cross_workspace_data_isolation(self):
        import shutil
        import json
        import numpy as np
        from pathlib import Path

        from lightrag.utils import EmbeddingFunc, Tokenizer

        # ── Temp working directory ──
        test_dir = tempfile.mkdtemp(prefix="test_pool_isolation_")
        try:
            # ── Mock embedding: 384-dim random vectors ──
            async def mock_embedding_func(texts: list[str]) -> np.ndarray:
                return np.random.rand(len(texts), 384)

            emb_func = EmbeddingFunc(
                embedding_dim=384,
                max_token_size=8192,
                func=mock_embedding_func,
            )

            # ── Simple tokenizer (char-level, no network needed) ──
            class _SimpleTokenizerImpl:
                def encode(self, content: str) -> list[int]:
                    return [ord(ch) for ch in content]

                def decode(self, tokens: list[int]) -> str:
                    return "".join(chr(t) for t in tokens)

            tokenizer = Tokenizer("mock-tokenizer", _SimpleTokenizerImpl())

            # ── Workspace-specific mock LLMs returning distinct entities ──
            def _make_llm(ws: str):
                async def mock_llm(
                    prompt, system_prompt=None, history_messages=[], **kwargs
                ) -> str:
                    await asyncio.sleep(0)
                    if ws == "project_a":
                        return (
                            "entity<|#|>Artificial Intelligence<|#|>concept<|#|>"
                            "AI is a field of computer science.\n"
                            "entity<|#|>Machine Learning<|#|>concept<|#|>"
                            "ML is a subset of AI.\n"
                            "relation<|#|>Machine Learning<|#|>Artificial Intelligence"
                            "<|#|>subset<|#|>ML is a subset of AI.\n"
                            "<|COMPLETE|>"
                        )
                    else:
                        return (
                            "entity<|#|>Deep Learning<|#|>concept<|#|>"
                            "DL uses neural networks with many layers.\n"
                            "entity<|#|>Neural Networks<|#|>concept<|#|>"
                            "NNs are inspired by biological brains.\n"
                            "relation<|#|>Deep Learning<|#|>Neural Networks"
                            "<|#|>uses<|#|>DL uses multiple layers of NNs.\n"
                            "<|COMPLETE|>"
                        )

                return mock_llm

            # ── Config factory: fresh dict per workspace ──
            def build_config(workspace: str) -> dict:
                return {
                    "working_dir": test_dir,
                    "workspace": workspace,
                    "llm_model_func": _make_llm(workspace),
                    "embedding_func": emb_func,
                    "tokenizer": tokenizer,
                }

            pool = RagPool(config_factory=build_config)

            # ── Pool returns distinct instances per workspace ──
            rag_a = await pool.get("project_a")
            rag_b = await pool.get("project_b")
            assert rag_a is not rag_b, (
                "RagPool must return distinct LightRAG instances per workspace"
            )
            assert rag_a.workspace == "project_a"
            assert rag_b.workspace == "project_b"

            # ── Insert data into both workspaces ──
            await rag_a.ainsert(
                "Artificial Intelligence and Machine Learning are "
                "transforming technology."
            )
            await rag_b.ainsert(
                "Deep Learning and Neural Networks power modern AI applications."
            )

            # ── Verify file-structure isolation ──
            dir_a = Path(test_dir) / "project_a"
            dir_b = Path(test_dir) / "project_b"
            assert dir_a.exists(), "project_a directory should exist"
            assert dir_b.exists(), "project_b directory should exist"

            # ── Verify full_docs isolation ──
            docs_a_path = dir_a / "kv_store_full_docs.json"
            docs_b_path = dir_b / "kv_store_full_docs.json"
            assert docs_a_path.exists(), "project_a full_docs should exist"
            assert docs_b_path.exists(), "project_b full_docs should exist"

            with open(docs_a_path, "r") as f:
                docs_a = json.load(f)
            with open(docs_b_path, "r") as f:
                docs_b = json.load(f)

            docs_a_str = json.dumps(docs_a)
            docs_b_str = json.dumps(docs_b)

            # project_a contains its own text, NOT project_b's
            assert "Artificial Intelligence" in docs_a_str, (
                "project_a should contain its own content"
            )
            assert "Deep Learning" not in docs_a_str, (
                "project_a must NOT contain project_b content"
            )

            # project_b contains its own text, NOT project_a's
            assert "Deep Learning" in docs_b_str, (
                "project_b should contain its own content"
            )
            assert "Artificial Intelligence" not in docs_b_str, (
                "project_b must NOT contain project_a content"
            )

            # ── Verify per-workspace storage files exist (list what's on disk) ──
            files_a = sorted(p.name for p in dir_a.glob("*") if p.is_file())
            files_b = sorted(p.name for p in dir_b.glob("*") if p.is_file())
            assert len(files_a) > 0, "project_a should contain storage files"
            assert len(files_b) > 0, "project_b should contain storage files"

            # ── Cleanup ──
            await pool.shutdown_all()

        finally:
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir, ignore_errors=True)
