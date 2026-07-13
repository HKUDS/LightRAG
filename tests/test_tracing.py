"""Tests for the Langfuse tracing module.

These are offline tests -- they verify tracing behavior without
requiring a Langfuse account or the langfuse package installed.
"""

from unittest.mock import MagicMock, patch

import pytest


# ============================================================================
# is_tracing_enabled() — env var and package detection
# ============================================================================


class TestIsTracingEnabled:
    """Tracing must be disabled when env vars are unset."""

    def test_disabled_when_no_env_vars(self, monkeypatch):
        """Returns False when API keys are not configured."""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")

        import lightrag.tracing as tracing_mod

        assert tracing_mod.is_tracing_enabled() is False

    def test_disabled_when_only_public_key(self, monkeypatch):
        """Returns False when only public key is set."""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")

        import lightrag.tracing as tracing_mod

        assert tracing_mod.is_tracing_enabled() is False

    def test_disabled_when_langfuse_not_installed(self, monkeypatch):
        """Returns False when langfuse package is missing."""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")

        import lightrag.tracing as tracing_mod

        with patch.dict("sys.modules", {"langfuse": None}):
            assert tracing_mod.is_tracing_enabled() is False

    def test_enabled_when_keys_set_and_langfuse_available(self, monkeypatch):
        """Returns True when both keys are set and langfuse is importable."""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")

        import lightrag.tracing as tracing_mod

        mock_langfuse = MagicMock()
        with patch.dict("sys.modules", {"langfuse": mock_langfuse}):
            assert tracing_mod.is_tracing_enabled() is True

    def test_reflects_env_changes(self, monkeypatch):
        """Result reflects env var changes (no permanent cache)."""
        import lightrag.tracing as tracing_mod

        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")
        assert tracing_mod.is_tracing_enabled() is False

        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
        mock_langfuse = MagicMock()
        with patch.dict("sys.modules", {"langfuse": mock_langfuse}):
            assert tracing_mod.is_tracing_enabled() is True


# ============================================================================
# LLM wrapper passthrough when tracing is disabled
# ============================================================================


class TestLLMWrapper:
    """LLM wrapper must pass through when tracing is disabled."""

    @pytest.fixture(autouse=True)
    def disable_tracing(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")

    def test_wrapper_returns_original_function_when_disabled(self):
        """create_traced_llm_wrapper returns the original function unchanged."""
        from lightrag.tracing import create_traced_llm_wrapper

        async def my_llm(prompt, **kwargs):
            return f"response to {prompt}"

        wrapped = create_traced_llm_wrapper(my_llm, model_name="test-model")
        assert wrapped is my_llm


# ============================================================================
# flush() / shutdown() — lifecycle no-ops when disabled
# ============================================================================


class TestLifecycle:
    """Lifecycle functions must be no-ops when tracing is disabled."""

    @pytest.fixture(autouse=True)
    def disable_tracing(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")

    def test_flush_noop_when_disabled(self):
        """flush() does nothing when tracing is disabled."""
        from lightrag.tracing import flush

        flush()  # Should not raise

    def test_shutdown_noop_when_disabled(self):
        """shutdown() does nothing when tracing is disabled."""
        from lightrag.tracing import shutdown

        shutdown()  # Should not raise


# ============================================================================
# report_token_usage() — token count forwarding to Langfuse
# ============================================================================


class TestReportTokenUsage:
    """report_token_usage forwards to update_current_generation."""

    def test_noop_when_disabled(self, monkeypatch):
        """No error when tracing is disabled."""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")

        from lightrag.tracing import report_token_usage

        report_token_usage({"input": 100, "output": 50})  # Should not raise

    def test_reports_usage_when_enabled(self, monkeypatch):
        """Forwards usage_details to client.update_current_generation."""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")

        mock_client = MagicMock()
        mock_langfuse_mod = MagicMock()
        mock_langfuse_mod.get_client = MagicMock(return_value=mock_client)

        with patch.dict("sys.modules", {"langfuse": mock_langfuse_mod}):
            from lightrag.tracing import report_token_usage

            report_token_usage({"input": 100, "output": 50})

        mock_client.update_current_generation.assert_called_once_with(
            usage_details={"input": 100, "output": 50}
        )

    def test_logs_on_error(self, monkeypatch):
        """Logs a warning instead of raising when Langfuse call fails."""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")

        mock_client = MagicMock()
        mock_client.update_current_generation.side_effect = RuntimeError("fail")
        mock_langfuse_mod = MagicMock()
        mock_langfuse_mod.get_client = MagicMock(return_value=mock_client)

        with patch.dict("sys.modules", {"langfuse": mock_langfuse_mod}):
            with patch("lightrag.tracing.logger") as mock_logger:
                from lightrag.tracing import report_token_usage

                report_token_usage({"input": 100, "output": 50})

            mock_logger.warning.assert_called()


# ============================================================================
# shutdown() — enabled path verifies client.shutdown() is called
# ============================================================================


class TestShutdownEnabled:
    """shutdown() calls client.shutdown() when tracing is enabled."""

    def test_calls_client_shutdown(self, monkeypatch):
        """Verifies client.shutdown() is invoked."""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")

        mock_client = MagicMock()
        mock_langfuse_mod = MagicMock()
        mock_langfuse_mod.get_client = MagicMock(return_value=mock_client)

        with patch.dict("sys.modules", {"langfuse": mock_langfuse_mod}):
            from lightrag.tracing import shutdown

            shutdown()

        mock_client.shutdown.assert_called_once()


# ============================================================================
# propagate_trace_attributes() — trace-level attribute propagation
# ============================================================================


class TestPropagateTraceAttributes:
    """propagate_trace_attributes wraps langfuse.propagate_attributes."""

    def test_noop_when_disabled(self, monkeypatch):
        """Context manager yields without error when tracing is disabled."""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")

        from lightrag.tracing import propagate_trace_attributes

        with propagate_trace_attributes(user_id="u1"):
            pass  # Should not raise


# ============================================================================
# create_traced_llm_wrapper() — observe() integration and model tagging
# ============================================================================


class TestTracedLLMWrapper:
    """LLM wrapper uses observe() functional form and sets model attribute."""

    @pytest.mark.asyncio
    async def test_wrapper_calls_function_and_returns_result(self, monkeypatch):
        """Wrapped function is called and its return value is passed through."""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")

        call_log = []

        async def my_llm(*args, **kwargs):
            call_log.append((args, kwargs))
            return "llm response"

        # Mock langfuse.observe to just return the function (passthrough)
        mock_langfuse_mod = MagicMock()
        mock_langfuse_mod.get_client = MagicMock(return_value=MagicMock())
        mock_langfuse_mod.observe = lambda func, **kw: func

        with patch.dict("sys.modules", {"langfuse": mock_langfuse_mod}):
            from lightrag.tracing import create_traced_llm_wrapper

            wrapped = create_traced_llm_wrapper(my_llm, model_name="gpt-4o")

        result = await wrapped("hello", system_prompt="be helpful")
        assert result == "llm response"
        assert len(call_log) == 1

    @pytest.mark.asyncio
    async def test_wrapper_sets_model_name(self, monkeypatch):
        """Wrapper calls update_current_generation(model=...) to tag the generation."""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")

        mock_client = MagicMock()

        async def my_llm(*args, **kwargs):
            return "ok"

        mock_langfuse_mod = MagicMock()
        mock_langfuse_mod.get_client = MagicMock(return_value=mock_client)
        mock_langfuse_mod.observe = lambda func, **kw: func

        with patch.dict("sys.modules", {"langfuse": mock_langfuse_mod}):
            from lightrag.tracing import create_traced_llm_wrapper

            wrapped = create_traced_llm_wrapper(my_llm, model_name="gpt-4o")

        await wrapped("test prompt")
        mock_client.update_current_generation.assert_called_with(model="gpt-4o")
