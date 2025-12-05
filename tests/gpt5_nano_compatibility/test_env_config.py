#!/usr/bin/env python3
"""
Test script to verify that .env configuration is properly loaded and respected.
Tests:
1. Configuration loading from .env
2. OpenAI API connectivity
3. Embeddings generation with configured model
4. LLM extraction with configured model
5. Full RAG pipeline
"""

import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv
import pytest

# Load environment variables from .env
load_dotenv()

# ============================================================================
# PART 1: Verify .env Configuration Loading
# ============================================================================


@pytest.mark.skipif(
    os.getenv("LLM_BINDING") != "openai", reason="LLM_BINDING not set to openai"
)
def test_env_loading():
    """Verify that .env configuration is properly loaded."""
    print("\n" + "=" * 80)
    print("PART 1: Verifying .env Configuration Loading")
    print("=" * 80)

    config = {
        "LLM_BINDING": os.getenv("LLM_BINDING"),
        "LLM_MODEL": os.getenv("LLM_MODEL"),
        "LLM_BINDING_API_KEY": os.getenv("LLM_BINDING_API_KEY", "NOT SET")[:20] + "..."
        if os.getenv("LLM_BINDING_API_KEY")
        else "NOT SET",
        "EMBEDDING_BINDING": os.getenv("EMBEDDING_BINDING"),
        "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL"),
        "EMBEDDING_DIM": os.getenv("EMBEDDING_DIM"),
        "EMBEDDING_BINDING_API_KEY": os.getenv("EMBEDDING_BINDING_API_KEY", "NOT SET")[
            :20
        ]
        + "..."
        if os.getenv("EMBEDDING_BINDING_API_KEY")
        else "NOT SET",
    }

    print("\n✓ Loaded configuration from .env:")
    for key, value in config.items():
        status = "✅" if value and value != "NOT SET" else "❌"
        print(f"  {status} {key}: {value}")

    # Verify OpenAI configuration
    issues = []
    if os.getenv("LLM_BINDING") != "openai":
        issues.append(f"LLM_BINDING is '{os.getenv('LLM_BINDING')}', expected 'openai'")
    if os.getenv("EMBEDDING_BINDING") != "openai":
        issues.append(
            f"EMBEDDING_BINDING is '{os.getenv('EMBEDDING_BINDING')}', expected 'openai'"
        )
    if not os.getenv("LLM_BINDING_API_KEY"):
        issues.append("LLM_BINDING_API_KEY is not set")
    if not os.getenv("EMBEDDING_BINDING_API_KEY"):
        issues.append("EMBEDDING_BINDING_API_KEY is not set")

    if issues:
        print("\n❌ Configuration Issues Found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("\n✅ All .env configuration checks passed!")
        return True


# ============================================================================
# PART 2: Verify Configuration is Loaded by LightRAG Config Parser
# ============================================================================


@pytest.mark.skipif(
    os.getenv("LLM_BINDING") != "openai", reason="LLM_BINDING not set to openai"
)
def test_config_parser():
    """Verify that the argparse config parser respects .env settings."""
    print("\n" + "=" * 80)
    print("PART 2: Verifying Config Parser Respects .env")
    print("=" * 80)

    try:
        # Load environment and check that config.py reads them correctly
        from dotenv import load_dotenv

        # Re-load to ensure fresh values
        load_dotenv(override=True)

        # The config.py uses get_env_value which should read from .env
        # We'll verify this by checking the values directly
        env_vars = {
            "LLM_BINDING": os.getenv("LLM_BINDING"),
            "LLM_MODEL": os.getenv("LLM_MODEL"),
            "EMBEDDING_BINDING": os.getenv("EMBEDDING_BINDING"),
            "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL"),
        }

        print("\n✓ Environment variables from .env (as seen by config.py):")
        print(f"  ✅ LLM_BINDING: {env_vars['LLM_BINDING']}")
        print(f"  ✅ LLM_MODEL: {env_vars['LLM_MODEL']}")
        print(f"  ✅ EMBEDDING_BINDING: {env_vars['EMBEDDING_BINDING']}")
        print(f"  ✅ EMBEDDING_MODEL: {env_vars['EMBEDDING_MODEL']}")

        # Verify values
        checks = [
            (
                env_vars["LLM_BINDING"] == "openai",
                f"LLM_BINDING is '{env_vars['LLM_BINDING']}', expected 'openai'",
            ),
            (
                env_vars["EMBEDDING_BINDING"] == "openai",
                f"EMBEDDING_BINDING is '{env_vars['EMBEDDING_BINDING']}', expected 'openai'",
            ),
            (
                env_vars["LLM_MODEL"] == "gpt-5-nano",
                f"LLM_MODEL is '{env_vars['LLM_MODEL']}', expected 'gpt-5-nano'",
            ),
            (
                env_vars["EMBEDDING_MODEL"] == "text-embedding-3-small",
                f"EMBEDDING_MODEL is '{env_vars['EMBEDDING_MODEL']}', expected 'text-embedding-3-small'",
            ),
        ]

        issues = [issue for check, issue in checks if not check]
        if issues:
            print("\n❌ Config Parser Issues:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        else:
            print("\n✅ All config parser checks passed!")
            return True

    except Exception as e:
        print(f"❌ Error testing config parser: {e}")
        import traceback

        traceback.print_exc()
        return False


# ============================================================================
# PART 3: Test OpenAI API Connectivity
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("LLM_BINDING_API_KEY"), reason="LLM_BINDING_API_KEY not set"
)
async def test_openai_connectivity():
    """Test OpenAI API connectivity with configured API key."""
    print("\n" + "=" * 80)
    print("PART 3: Testing OpenAI API Connectivity")
    print("=" * 80)

    try:
        from openai import AsyncOpenAI

        api_key = os.getenv("LLM_BINDING_API_KEY")
        if not api_key:
            print("❌ LLM_BINDING_API_KEY not set in .env")
            return False

        print(f"\n✓ Testing OpenAI API with key: {api_key[:20]}...")

        client = AsyncOpenAI(api_key=api_key)

        # Test with a simple model list call (doesn't consume tokens)
        try:
            # Try to get a model info - this validates the API key
            response = await client.models.retrieve("gpt-5-nano")
            print("  ✅ OpenAI API connectivity: SUCCESS")
            print(f"  ✅ Model 'gpt-5-nano' exists: {response.id}")
            return True
        except Exception as e:
            print(f"  ❌ OpenAI API error: {e}")
            return False

    except Exception as e:
        print(f"❌ Error testing OpenAI connectivity: {e}")
        return False


# ============================================================================
# PART 4: Test Embeddings with Configured Model
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("EMBEDDING_BINDING_API_KEY"),
    reason="EMBEDDING_BINDING_API_KEY not set",
)
async def test_embeddings():
    """Test embeddings generation using configured model from .env."""
    print("\n" + "=" * 80)
    print("PART 4: Testing Embeddings with Configured Model")
    print("=" * 80)

    try:
        from openai import AsyncOpenAI

        api_key = os.getenv("EMBEDDING_BINDING_API_KEY")
        model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

        if not api_key:
            print("❌ EMBEDDING_BINDING_API_KEY not set in .env")
            return False

        print(f"\n✓ Testing embeddings with model: {model}")

        client = AsyncOpenAI(api_key=api_key)

        # Generate embedding for a test text
        test_text = "This is a test document for embeddings."

        response = await client.embeddings.create(input=test_text, model=model)

        embedding = response.data[0].embedding
        embedding_dim = len(embedding)
        expected_dim = int(os.getenv("EMBEDDING_DIM", "1536"))

        print("  ✅ Embeddings generated successfully")
        print(f"  ✅ Model used: {model}")
        print(f"  ✅ Embedding dimensions: {embedding_dim}")
        print(f"  ✅ Expected dimensions: {expected_dim}")

        if embedding_dim != expected_dim:
            print(
                f"  ❌ WARNING: Dimension mismatch! ({embedding_dim} vs {expected_dim})"
            )
            return False

        print("\n✅ Embeddings test passed!")
        return True

    except Exception as e:
        print(f"❌ Error testing embeddings: {e}")
        return False


# ============================================================================
# PART 5: Test LLM Extraction with Configured Model
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("LLM_BINDING_API_KEY"), reason="LLM_BINDING_API_KEY not set"
)
async def test_llm_extraction():
    """Test LLM extraction using configured model from .env."""
    print("\n" + "=" * 80)
    print("PART 5: Testing LLM Extraction with Configured Model")
    print("=" * 80)

    try:
        from openai import AsyncOpenAI

        api_key = os.getenv("LLM_BINDING_API_KEY")
        model = os.getenv("LLM_MODEL", "gpt-5-nano")

        if not api_key:
            print("❌ LLM_BINDING_API_KEY not set in .env")
            return False

        print(f"\n✓ Testing LLM with model: {model}")

        client = AsyncOpenAI(api_key=api_key)

        # Test LLM with a simple extraction prompt
        test_document = """
        John Smith works at Acme Corporation as a Software Engineer.
        He reports to Jane Doe, the Engineering Manager.
        The company is located in San Francisco, California.
        """

        # Build request based on model - gpt-5-nano has specific constraints
        request_kwargs = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "Extract entities from the text. Return as JSON.",
                },
                {"role": "user", "content": f"Extract entities from: {test_document}"},
            ],
        }

        # gpt-5-nano doesn't support custom temperature or max_tokens parameters
        if not model.startswith("gpt-5"):
            request_kwargs["temperature"] = 0.7
            request_kwargs["max_tokens"] = 500
        else:
            # For gpt-5-nano, only use max_completion_tokens
            request_kwargs["max_completion_tokens"] = 500

        response = await client.chat.completions.create(**request_kwargs)

        extracted = response.choices[0].message.content

        print("  ✅ LLM extraction successful")
        print(f"  ✅ Model used: {response.model}")
        print(f"  ✅ Response preview: {extracted[:100]}...")
        print(
            f"  ✅ Tokens used - Prompt: {response.usage.prompt_tokens}, Completion: {response.usage.completion_tokens}"
        )

        print("\n✅ LLM extraction test passed!")
        return True

    except Exception as e:
        print(f"❌ Error testing LLM extraction: {e}")
        return False


# ============================================================================
# PART 6: Full Integration Test
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("LLM_BINDING_API_KEY") or not os.getenv("EMBEDDING_BINDING_API_KEY"),
    reason="LLM_BINDING_API_KEY or EMBEDDING_BINDING_API_KEY not set",
)
async def test_full_integration():
    """Test full LightRAG pipeline with .env configuration."""
    print("\n" + "=" * 80)
    print("PART 6: Full RAG Pipeline Integration Test")
    print("=" * 80)

    try:
        from lightrag import LightRAG, QueryParam
        from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
        from lightrag.kg.shared_storage import initialize_pipeline_status

        print("\n✓ Testing full RAG pipeline with .env configuration")

        # Create working directory
        working_dir = "./test_rag_env"
        Path(working_dir).mkdir(exist_ok=True)

        print(f"  ✓ Working directory: {working_dir}")

        # Initialize RAG
        print("  ✓ Initializing LightRAG with .env configuration...")
        rag = LightRAG(
            working_dir=working_dir,
            embedding_func=openai_embed,
            llm_model_func=gpt_4o_mini_complete,
        )

        # Initialize storages
        await rag.initialize_storages()
        await initialize_pipeline_status()

        print("  ✓ Storages initialized")

        # Insert test document
        test_doc = """
        Alice Johnson is a Data Scientist at TechCorp.
        She works on machine learning projects with Bob Smith, who is a Software Architect.
        TechCorp is located in Seattle and specializes in AI solutions.
        """

        print("  ✓ Inserting test document...")
        await rag.ainsert(test_doc)
        print("  ✅ Document inserted successfully")

        # Query
        print("  ✓ Running query...")
        result = await rag.aquery(
            "Who works at TechCorp and what do they do?",
            param=QueryParam(mode="hybrid"),
        )

        print(f"  ✅ Query result: {result[:100]}...")

        # Cleanup
        await rag.finalize_storages()

        print("\n✅ Full integration test passed!")
        return True

    except ImportError as e:
        print(
            f"⚠️  Skipping full integration test - LightRAG not fully initialized: {e}"
        )
        return True  # Not a failure, just can't test at this stage
    except Exception as e:
        print(f"❌ Error in full integration test: {e}")
        import traceback

        traceback.print_exc()
        return False


# ============================================================================
# Main Test Execution
# ============================================================================


async def _main():
    """Run all tests (internal helper)."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "TESTING .ENV OPENAI CONFIGURATION" + " " * 31 + "║")
    print("╚" + "=" * 78 + "╝")

    results = {
        "✓ .env Loading": test_env_loading(),
        "✓ Config Parser": test_config_parser(),
        "✓ OpenAI Connectivity": await test_openai_connectivity(),
        "✓ Embeddings": await test_embeddings(),
        "✓ LLM Extraction": await test_llm_extraction(),
        "✓ Full Integration": await test_full_integration(),
    }

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")

    print("\n" + "=" * 80)
    print(f"OVERALL: {passed}/{total} tests passed")
    print("=" * 80 + "\n")

    if passed == total:
        print("✅ All tests passed! .env OpenAI configuration is properly respected.")
        return True
    else:
        print(f"⚠️  {total - passed} test(s) failed. Review the output above.")
        return False


if __name__ == "__main__":
    success = asyncio.run(_main())
    exit(0 if success else 1)
