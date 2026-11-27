"""Shared fake lmdeploy for the lmdeploy binding tests.

The tests import only ``lightrag.llm.lmdeploy``; the real SDK is never needed.
This conftest must not import it at module level: pytest loads a directory's
conftest before ``tests/conftest.py::pytest_configure`` refuses
``pipmaster.install``, so the provider's install guard would run unrefused.
"""

import sys
from types import SimpleNamespace

import pytest


class FakeRequestError(RuntimeError):
    """Stand-in for ``lmdeploy.serve.core.exceptions.RequestError``."""

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(f"fake lmdeploy request error ({status_code})")


async def _default_generate(*_args, **_kwargs):
    yield SimpleNamespace(text="generated text", finish_reason="stop")


async def _instant_sleep(_seconds: float) -> None:
    return None


@pytest.fixture
def fake_lmdeploy(monkeypatch):
    """Install a fake ``lmdeploy`` (0.6.1 by default) and a cached pipeline.

    Returns a namespace: assign ``generate`` (an async generator function
    receiving ``Pipeline.generate``'s arguments) to script the engine, set
    ``module.version_info`` to emulate another release, and read
    ``gen_config`` for the kwargs ``GenerationConfig`` was built with.
    ``RequestError`` is the fake ``lmdeploy.serve.core.exceptions`` class,
    and the retry decorator's backoff does not wait.
    """
    fake = SimpleNamespace(
        generate=_default_generate,
        gen_config={},
        RequestError=FakeRequestError,
    )

    class FakeGenerationConfig:
        def __init__(self, **kwargs):
            fake.gen_config.update(kwargs)

    fake.module = SimpleNamespace(
        __version__="0.6.1",
        version_info=(0, 6, 1),
        GenerationConfig=FakeGenerationConfig,
    )
    monkeypatch.setitem(sys.modules, "lmdeploy", fake.module)
    # Separate sys.modules entries: _is_retryable_lmdeploy_error imports the
    # dotted path, which resolves each component from sys.modules.
    monkeypatch.setitem(sys.modules, "lmdeploy.serve", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "lmdeploy.serve.core", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "lmdeploy.serve.core.exceptions",
        SimpleNamespace(RequestError=FakeRequestError),
    )
    monkeypatch.setattr(
        "lightrag.llm.lmdeploy.initialize_lmdeploy_pipeline",
        lambda **_kwargs: SimpleNamespace(
            generate=lambda *args, **kwargs: fake.generate(*args, **kwargs)
        ),
    )
    from lightrag.llm.lmdeploy import lmdeploy_model_if_cache

    monkeypatch.setattr(lmdeploy_model_if_cache.retry, "sleep", _instant_sleep)
    return fake
