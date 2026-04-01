"""Phase 1 tests for entity profile generation and persistence."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.constants import DEFAULT_ENTITY_PROFILE_FACETS
from lightrag.operate import _parse_entity_profile_generation_result
from lightrag.utils import EmbeddingFunc, Tokenizer, TokenizerInterface


class DummyTokenizer(TokenizerInterface):
    """Simple 1:1 tokenizer for deterministic offline tests."""

    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


@pytest.mark.offline
def test_parse_entity_profile_generation_result_prefers_longest_valid_profile():
    facet_catalog = DEFAULT_ENTITY_PROFILE_FACETS
    result = """profile<|#|>identity_definition<|#|>Identity / Definition<|#|>Short profile
profile<|#|>identity_definition<|#|>Identity / Definition<|#|>Alpha System is a research synthesis pipeline with a longer identity profile.
profile<|#|>unknown_facet<|#|>Unknown<|#|>Should be ignored
<|COMPLETE|>"""

    profiles = _parse_entity_profile_generation_result(
        result=result,
        entity_name="Alpha System",
        facet_catalog=facet_catalog,
        created_at=123,
    )

    assert len(profiles) == 1
    assert profiles[0]["facet_id"] == "identity_definition"
    assert profiles[0]["profile_text"].startswith("Alpha System is a research")
    assert profiles[0]["support_chunk_ids"] == []
    assert profiles[0]["grounding_status"] == "chunk_level"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_ainsert_persists_entity_profiles_with_fallback(tmp_path):
    tokenizer = Tokenizer("dummy-tokenizer", DummyTokenizer())

    async def mock_embedding_func(texts: list[str]) -> np.ndarray:
        await asyncio.sleep(0)
        return np.array(
            [
                [float(len(text)), float(index + 1), 1.0, 0.5]
                for index, text in enumerate(texts)
            ],
            dtype=np.float32,
        )

    async def mock_llm_func(
        prompt: str,
        system_prompt: str | None = None,
        history_messages=None,
        **kwargs,
    ) -> str:
        await asyncio.sleep(0)
        if system_prompt and "Knowledge Graph Profiling Specialist" in system_prompt:
            return """profile<|#|>identity_definition<|#|>Identity / Definition<|#|>Alpha System is a retrieval pipeline for research synthesis.
profile<|#|>role_function<|#|>Role / Function<|#|>Alpha System organizes evidence for downstream question answering.
<|COMPLETE|>"""

        return """entity<|#|>Alpha System<|#|>Method<|#|>Alpha System is a retrieval pipeline for research synthesis.
<|COMPLETE|>"""

    rag = LightRAG(
        working_dir=str(tmp_path / "rag_storage"),
        workspace="phase1_profiles",
        llm_model_func=mock_llm_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=4,
            max_token_size=8192,
            func=mock_embedding_func,
            model_name="phase1-test-embed",
        ),
        tokenizer=tokenizer,
        enable_entity_profiles=True,
        entity_extract_max_gleaning=0,
    )

    await rag.initialize_storages()
    try:
        await rag.ainsert(
            "Alpha System helps synthesize evidence for research questions."
        )

        profile_record = await rag.entity_profiles.get_by_id("Alpha System")
        assert profile_record is not None
        assert profile_record["entity_name"] == "Alpha System"
        assert profile_record["entity_type"] == "method"
        assert profile_record["base_description"].startswith(
            "Alpha System is a retrieval pipeline"
        )
        assert profile_record["facet_ids"] == [
            facet["facet_id"] for facet in DEFAULT_ENTITY_PROFILE_FACETS
        ]
        assert profile_record["count"] == len(DEFAULT_ENTITY_PROFILE_FACETS)
        assert len(profile_record["profiles"]) == len(DEFAULT_ENTITY_PROFILE_FACETS)

        grounding_statuses = {
            profile["facet_id"]: profile["grounding_status"]
            for profile in profile_record["profiles"]
        }
        assert grounding_statuses["identity_definition"] == "chunk_level"
        assert grounding_statuses["role_function"] == "chunk_level"
        assert grounding_statuses["attributes_composition"] == "fallback"
        assert grounding_statuses["state_behavior"] == "fallback"

        vector_records = await rag.entity_profiles_vdb.get_by_ids(
            profile_record["profile_ids"]
        )
        assert len([record for record in vector_records if record is not None]) == len(
            DEFAULT_ENTITY_PROFILE_FACETS
        )
        assert {
            record["facet_id"] for record in vector_records if record is not None
        } == {facet["facet_id"] for facet in DEFAULT_ENTITY_PROFILE_FACETS}
        assert all(
            record["entity_name"] == "Alpha System"
            for record in vector_records
            if record is not None
        )
    finally:
        await rag.finalize_storages()
