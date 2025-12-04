#!/usr/bin/env python3
"""
Direct test of gpt-5-nano API behavior.

These are integration tests that require:
- LLM_BINDING_API_KEY or OPENAI_API_KEY environment variable
- Access to OpenAI API with gpt-5-nano model

Run standalone: python test_direct_gpt5nano.py
Run via pytest: pytest test_direct_gpt5nano.py -v (skipped without API key)
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
        reason="LLM_BINDING_API_KEY or OPENAI_API_KEY not set"
    )
]


@pytest.fixture
def api_client():
    """Create OpenAI client for tests."""
    from openai import AsyncOpenAI
    api_key = os.getenv("LLM_BINDING_API_KEY") or os.getenv("OPENAI_API_KEY")
    return AsyncOpenAI(api_key=api_key)


async def test_direct(api_client):
    """Test direct API call with gpt-5-nano."""
    response = await api_client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello"}
        ],
        max_completion_tokens=50
    )
    
    assert response.choices[0].message.content is not None
    print(f"Response content: {response.choices[0].message.content}")


# Allow running as standalone script
if __name__ == "__main__":
    async def main():
        from openai import AsyncOpenAI
        api_key = os.getenv("LLM_BINDING_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("ERROR: LLM_BINDING_API_KEY or OPENAI_API_KEY not set")
            return
        
        client = AsyncOpenAI(api_key=api_key)
        print("Testing direct API call with gpt-5-nano...")
        await test_direct(client)
    
    asyncio.run(main())
