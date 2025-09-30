"""
LightRAG with LiteLLM Streaming Demo

This example demonstrates:
1. Streaming responses from LiteLLM
2. Using multiple providers (OpenAI, Anthropic, etc.)
3. Token usage tracking

Setup:
1. Install dependencies:
   pip install lightrag-hku litellm

2. Set your API keys:
   export OPENAI_API_KEY="sk-..."
   export ANTHROPIC_API_KEY="sk-ant-..."

3. Run:
   python examples/lightrag_litellm_streaming_demo.py
"""

import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.litellm import litellm_complete_if_cache, litellm_embed
from lightrag.utils import EmbeddingFunc, TokenTracker
from lightrag.kg.shared_storage import initialize_pipeline_status

WORKING_DIR = "./dickens_streaming"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], **kwargs
):
    """LLM function with streaming support."""
    return await litellm_complete_if_cache(
        model="gpt-4o-mini",  # Change to any LiteLLM-supported model
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def embedding_func(texts: list[str]):
    """Embedding function."""
    return await litellm_embed(
        texts=texts,
        model="text-embedding-3-small",
    )


async def initialize_rag():
    """Initialize LightRAG instance."""
    test_embedding = await embedding_func(["test"])
    embedding_dim = test_embedding.shape[1]

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=8192,
            func=embedding_func,
        ),
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


async def test_streaming_query(rag: LightRAG):
    """Test streaming query responses."""
    print("\n" + "=" * 60)
    print("STREAMING QUERY TEST")
    print("=" * 60)

    query = "What are the main characters in this story? Describe them briefly."

    # Non-streaming query
    print("\n1. Non-streaming response:")
    print("-" * 60)
    result = await rag.aquery(
        query,
        param=QueryParam(mode="hybrid", stream=False)
    )
    print(result)

    # Streaming query
    print("\n2. Streaming response:")
    print("-" * 60)
    result_stream = await rag.aquery(
        query,
        param=QueryParam(mode="hybrid", stream=True)
    )

    # Process stream
    async for chunk in result_stream:
        print(chunk, end="", flush=True)
    print("\n")


async def test_token_tracking(rag: LightRAG):
    """Test token usage tracking."""
    print("\n" + "=" * 60)
    print("TOKEN TRACKING TEST")
    print("=" * 60)

    tracker = TokenTracker()

    with tracker:
        # Insert document
        print("\nInserting document...")
        with open("./book.txt", "r", encoding="utf-8") as f:
            await rag.ainsert(f.read())

        # Query
        print("Querying...")
        await rag.aquery(
            "What are the top themes?",
            param=QueryParam(mode="hybrid")
        )

    usage = tracker.get_usage()
    print(f"\nToken Usage:")
    print(f"  Prompt tokens: {usage.get('prompt_tokens', 0)}")
    print(f"  Completion tokens: {usage.get('completion_tokens', 0)}")
    print(f"  Total tokens: {usage.get('total_tokens', 0)}")


async def test_multi_provider(rag: LightRAG):
    """Test querying with different providers."""
    print("\n" + "=" * 60)
    print("MULTI-PROVIDER TEST")
    print("=" * 60)

    query = "Summarize the story in one sentence."

    providers = [
        ("gpt-4o-mini", "OpenAI GPT-4o Mini"),
        ("anthropic/claude-3-haiku-20240307", "Anthropic Claude 3 Haiku"),
        ("cohere/command-r", "Cohere Command R"),
    ]

    for model, name in providers:
        try:
            print(f"\n{name} ({model}):")
            print("-" * 60)

            # Create custom LLM function for this provider
            async def custom_llm(prompt, system_prompt=None, history_messages=[], **kwargs):
                return await litellm_complete_if_cache(
                    model=model,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages,
                    **kwargs,
                )

            # Query with custom model
            result = await rag.aquery(
                query,
                param=QueryParam(mode="hybrid", model_func=custom_llm)
            )
            print(result)

        except Exception as e:
            print(f"Error with {name}: {e}")
            print("(Make sure you have the API key set for this provider)")


async def main():
    print("LightRAG with LiteLLM - Advanced Features Demo")
    print("=" * 60)

    try:
        # Initialize RAG
        print("\nInitializing LightRAG...")
        rag = await initialize_rag()

        # Check if document is already inserted
        if not os.path.exists(os.path.join(WORKING_DIR, "kv_store_full_docs.json")):
            print("Inserting document for the first time...")
            with open("./book.txt", "r", encoding="utf-8") as f:
                await rag.ainsert(f.read())
            print("Document inserted!\n")
        else:
            print("Using existing document index.\n")

        # Run tests
        await test_streaming_query(rag)
        await test_token_tracking(rag)

        # Only test multi-provider if user wants to
        print("\n" + "=" * 60)
        response = input("Test multiple providers? (requires multiple API keys) [y/N]: ")
        if response.lower() == 'y':
            await test_multi_provider(rag)

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if rag:
            await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(main())
    print("\nDone!")