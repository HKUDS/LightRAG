"""Exercise settings configuration against the real LlamaIndex SDK."""

from copy import copy

import numpy as np
import pytest
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.settings import Settings

from lightrag.llm.llama_index_impl import configure_llama_index, llama_index_embed

pytestmark = pytest.mark.offline


def test_default_settings_returns_sdk_singleton():
    before = vars(Settings).copy()
    assert configure_llama_index() is Settings
    assert vars(Settings) == before


def test_explicit_settings_returns_sdk_singleton():
    before = vars(Settings).copy()
    assert configure_llama_index(Settings) is Settings
    assert vars(Settings) == before


@pytest.mark.asyncio
async def test_embedding_accepts_sdk_settings():
    before = vars(Settings).copy()
    model = MockEmbedding(embed_dim=3)
    actual = await llama_index_embed.func(
        ["alpha", "beta"], embed_model=model, settings=Settings
    )
    np.testing.assert_array_equal(actual, np.full((2, 3), 0.5))
    assert vars(Settings) == before


def test_provided_settings_are_not_copied_to_the_global_singleton():
    settings = copy(Settings)
    model = MockEmbedding(embed_dim=3)
    settings.embed_model = model
    before = vars(Settings).copy()
    assert configure_llama_index(settings) is settings
    assert settings.embed_model is model
    assert vars(Settings) == before


def test_unknown_setting_remains_a_warning(lightrag_log_records):
    settings = copy(Settings)
    assert configure_llama_index(settings, not_a_setting=1) is settings
    assert not hasattr(settings, "not_a_setting")
    assert any(
        "Unknown LlamaIndex setting: not_a_setting" in record.getMessage()
        for record in lightrag_log_records
    )
