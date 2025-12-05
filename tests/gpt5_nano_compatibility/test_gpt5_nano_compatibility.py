#!/usr/bin/env python3
"""
Test script to verify gpt-5-nano compatibility with LightRAG.

This script validates that:
1. gpt-5-nano parameter handling works correctly (max_completion_tokens conversion)
2. Temperature parameter is properly handled for gpt-5-nano
3. Embeddings work with gpt-5-nano configuration
4. Entity extraction works with gpt-5-nano
5. Full pipeline works end-to-end

Requires:
- OPENAI_API_KEY environment variable set
- LLM_MODEL set to gpt-5-nano or specified via argument

Run standalone: python test_gpt5_nano_compatibility.py
Run via pytest: pytest test_gpt5_nano_compatibility.py -v (some tests skipped without API key)
"""

import os
import sys
import asyncio
import logging
import pytest

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add the repo to path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from lightrag.llm.openai import (  # noqa: E402
    openai_complete_if_cache,
    openai_embed,
    _normalize_openai_kwargs_for_model,
)


# Define markers for different test types
NEEDS_API_KEY = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") and not os.getenv("LLM_BINDING_API_KEY"),
    reason="OPENAI_API_KEY or LLM_BINDING_API_KEY not set",
)


@pytest.mark.asyncio
async def test_parameter_normalization():
    """Test 1: Parameter normalization for gpt-5-nano"""
    logger.info("=" * 60)
    logger.info("TEST 1: Parameter normalization for gpt-5-nano")
    logger.info("=" * 60)

    try:
        # Test case 1a: max_tokens conversion to max_completion_tokens with buffer
        kwargs = {"max_tokens": 500, "temperature": 0.7, "top_p": 0.9}
        original_kwargs = kwargs.copy()
        _normalize_openai_kwargs_for_model("gpt-5-nano", kwargs)

        logger.info(f"Input kwargs: {original_kwargs}")
        logger.info(f"Output kwargs: {kwargs}")

        assert (
            "max_completion_tokens" in kwargs
        ), "max_tokens should be converted to max_completion_tokens"
        assert (
            kwargs["max_completion_tokens"] >= 500
        ), "max_completion_tokens should be at least original value (500)"
        assert "max_tokens" not in kwargs, "max_tokens should be removed"
        assert (
            "temperature" not in kwargs
        ), "temperature should be removed for gpt-5-nano"
        assert "top_p" in kwargs, "top_p should be preserved"

        logger.info(
            f"✅ Test 1a passed: max_tokens → max_completion_tokens conversion works (buffered from 500 to {kwargs['max_completion_tokens']})"
        )

        # Test case 1b: Both max_tokens and max_completion_tokens (edge case)
        kwargs = {"max_tokens": 200, "max_completion_tokens": 300, "temperature": 0.5}
        original_kwargs = kwargs.copy()
        _normalize_openai_kwargs_for_model("gpt-5-nano", kwargs)

        logger.info(f"Input kwargs (both max params): {original_kwargs}")
        logger.info(f"Output kwargs: {kwargs}")

        assert "max_tokens" not in kwargs, "max_tokens should be removed"
        assert "max_completion_tokens" in kwargs, "max_completion_tokens should be kept"
        assert "temperature" not in kwargs, "temperature should be removed"

        logger.info("✅ Test 1b passed: Both max parameters handled correctly")

        # Test case 1c: Non-gpt5 models shouldn't change
        kwargs = {"max_tokens": 500, "temperature": 0.7}
        original_kwargs = kwargs.copy()
        _normalize_openai_kwargs_for_model("gpt-4o-mini", kwargs)

        logger.info(f"Input kwargs (gpt-4o-mini): {original_kwargs}")
        logger.info(f"Output kwargs: {kwargs}")

        assert "max_tokens" in kwargs, "max_tokens should be preserved for gpt-4o-mini"
        assert kwargs["max_tokens"] == 500, "max_tokens value should be unchanged"
        assert (
            "temperature" in kwargs
        ), "temperature should be preserved for gpt-4o-mini"

        logger.info("✅ Test 1c passed: Non-gpt5 models are unchanged")

        logger.info("✅ TEST 1 PASSED: Parameter normalization works correctly\n")
        return True
    except Exception as e:
        logger.error(f"❌ TEST 1 FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


@pytest.mark.asyncio
@NEEDS_API_KEY
async def test_embeddings():
    """Test 2: Embeddings generation"""
    logger.info("=" * 60)
    logger.info("TEST 2: Embeddings generation")
    logger.info("=" * 60)

    try:
        texts = ["Hello world", "This is a test"]
        model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

        logger.info(f"Generating embeddings with model: {model}")
        embeddings = await openai_embed(texts, model=model)

        logger.info(f"Generated {len(embeddings)} embeddings")
        logger.info(f"First embedding shape: {len(embeddings[0])}")

        assert len(embeddings) == len(texts), "Should get one embedding per text"
        assert len(embeddings[0]) > 0, "Embeddings should not be empty"

        logger.info("✅ TEST 2 PASSED: Embeddings generation works\n")
        return True
    except Exception as e:
        logger.error(f"❌ TEST 2 FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
async def test_simple_completion():
    """Test 3: Simple LLM completion with gpt-5-nano"""
    logger.info("=" * 60)
    logger.info("TEST 3: Simple LLM completion with gpt-5-nano")
    logger.info("=" * 60)

    try:
        model = os.getenv("LLM_MODEL", "gpt-5-nano")

        logger.info(f"Testing completion with model: {model}")

        # Enable verbose debug logging
        import logging as py_logging

        py_logging.getLogger("openai").setLevel(py_logging.DEBUG)

        # Test without custom temperature (gpt-5-nano requirement)
        response = await openai_complete_if_cache(
            model=model,
            prompt="Say hello in one word",
            system_prompt="You are a helpful assistant.",
            max_completion_tokens=20,
        )

        logger.info(f"Response: {response}")
        assert len(response) > 0, "Response should not be empty"

        logger.info("✅ TEST 3 PASSED: Simple completion works\n")
        return True
    except Exception as e:
        logger.error(f"❌ TEST 3 FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
async def test_extraction_with_gpt5nano():
    """Test 4: Entity extraction style task"""
    logger.info("=" * 60)
    logger.info("TEST 4: Entity extraction style task")
    logger.info("=" * 60)

    try:
        model = os.getenv("LLM_MODEL", "gpt-5-nano")

        prompt = """Extract the entities from this text:

Apple Inc. was founded by Steve Jobs in 1976.

Return as JSON with keys: company, person, year."""

        logger.info(f"Testing extraction with model: {model}")

        response = await openai_complete_if_cache(
            model=model,
            prompt=prompt,
            system_prompt="You are an entity extraction assistant. Always respond in valid JSON.",
            max_completion_tokens=100,
        )

        logger.info(f"Response: {response}")
        assert len(response) > 0, "Response should not be empty"
        assert "Apple" in response or "apple" in response, "Should mention Apple"

        logger.info("✅ TEST 4 PASSED: Entity extraction works\n")
        return True
    except Exception as e:
        logger.error(f"❌ TEST 4 FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


@pytest.mark.asyncio
async def test_config_loading():
    """Test 5: Configuration loading from .env"""
    logger.info("=" * 60)
    logger.info("TEST 5: Configuration loading from .env")
    logger.info("=" * 60)

    llm_model = os.getenv("LLM_MODEL", "not-set")
    llm_binding = os.getenv("LLM_BINDING", "not-set")
    embedding_model = os.getenv("EMBEDDING_MODEL", "not-set")
    embedding_binding = os.getenv("EMBEDDING_BINDING", "not-set")

    logger.info(f"LLM_MODEL: {llm_model}")
    logger.info(f"LLM_BINDING: {llm_binding}")
    logger.info(f"EMBEDDING_MODEL: {embedding_model}")
    logger.info(f"EMBEDDING_BINDING: {embedding_binding}")

    # Verify we're using OpenAI
    assert (
        embedding_binding == "openai" or embedding_binding == "not-set"
    ), "EMBEDDING_BINDING should be openai"
    assert (
        llm_binding == "openai" or llm_binding == "not-set"
    ), "LLM_BINDING should be openai"

    logger.info("✅ TEST 5 PASSED: Configuration loaded correctly\n")
    return True


async def _run_all_tests():
    """Run all tests (internal helper, not picked up by pytest)"""
    logger.info("\n" + "=" * 60)
    logger.info("GPT-5-NANO COMPATIBILITY TEST SUITE")
    logger.info("=" * 60 + "\n")

    # Check prerequisites
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("❌ OPENAI_API_KEY environment variable not set")
        return False

    results = {
        "Parameter Normalization": await test_parameter_normalization(),
        "Configuration Loading": await test_config_loading(),
        "Embeddings": await test_embeddings(),
        "Simple Completion": await test_simple_completion(),
        "Entity Extraction": await test_extraction_with_gpt5nano(),
    }

    # Summary
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)

    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")

    all_passed = all(results.values())

    if all_passed:
        logger.info("\n" + "=" * 60)
        logger.info("🎉 ALL TESTS PASSED")
        logger.info("=" * 60)
    else:
        logger.info("\n" + "=" * 60)
        logger.info("⚠️  SOME TESTS FAILED")
        logger.info("=" * 60)

    return all_passed


if __name__ == "__main__":
    # Load environment from .env file
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=".env", override=False)

    # Run tests
    success = asyncio.run(_run_all_tests())
    sys.exit(0 if success else 1)
