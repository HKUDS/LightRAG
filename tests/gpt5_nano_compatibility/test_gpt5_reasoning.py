#!/usr/bin/env python3
"""
Test gpt-5-nano with different token limits and reasoning settings.

These are integration tests that require:
- LLM_BINDING_API_KEY or OPENAI_API_KEY environment variable
- Access to OpenAI API with gpt-5-nano model

Run standalone: python test_gpt5_reasoning.py
Run via pytest: pytest test_gpt5_reasoning.py -v (skipped without API key)
"""

import os
import asyncio
import pytest
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=False)


# Skip all tests if API key is not configured
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.getenv("LLM_BINDING_API_KEY") and not os.getenv("OPENAI_API_KEY"),
        reason="LLM_BINDING_API_KEY or OPENAI_API_KEY not set",
    ),
]


@pytest.fixture
def api_client():
    """Create OpenAI client for tests."""
    from openai import AsyncOpenAI

    api_key = os.getenv("LLM_BINDING_API_KEY") or os.getenv("OPENAI_API_KEY")
    return AsyncOpenAI(api_key=api_key)


async def test_with_more_tokens(api_client):
    """Test gpt-5-nano with 200 max_completion_tokens."""
    response = await api_client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello"},
        ],
        max_completion_tokens=200,
    )
    assert response.choices[0].message.content is not None
    print(f"Content: '{response.choices[0].message.content}'")
    print(f"Tokens used - Completion: {response.usage.completion_tokens}")


async def test_with_low_reasoning_effort(api_client):
    """Test gpt-5-nano with reasoning_effort='low'."""
    response = await api_client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello"},
        ],
        max_completion_tokens=50,
        reasoning_effort="low",
    )
    assert response.choices[0].message.content is not None
    print(f"Content: '{response.choices[0].message.content}'")
    print(f"Tokens used - Completion: {response.usage.completion_tokens}")


# Allow running as standalone script
if __name__ == "__main__":

    async def main():
        from openai import AsyncOpenAI

        api_key = os.getenv("LLM_BINDING_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("ERROR: LLM_BINDING_API_KEY or OPENAI_API_KEY not set")
            return

        client = AsyncOpenAI(api_key=api_key)

        print("Test 1: With 200 max_completion_tokens")
        await test_with_more_tokens(client)
        print()

        print("Test 2: With lower reasoning_effort='low'")
        await test_with_low_reasoning_effort(client)
        print()

    asyncio.run(main())
