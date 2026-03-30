"""
Integration test for entity extraction with JSON mode (GPTEntityExtractionFormat).

Verifies:
1. entity_types_guidance is correctly injected into the system prompt
2. LLM returns valid GPTEntityExtractionFormat JSON
3. JSON is correctly converted to delimiter-based format and parsed into entities/relations

Requires environment variables:
    LLM_MODEL          - model name (default: deepseek-chat)
    LLM_BINDING_API_KEY or OPENAI_API_KEY - API key
    LLM_BINDING_HOST   - base URL (default: https://api.deepseek.com)
"""

import os
import asyncio

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=False)

from lightrag.llm.openai import openai_complete_if_cache
from lightrag.operate import extract_entities
from lightrag.prompt import PROMPTS
from lightrag.utils import Tokenizer, TokenizerInterface


# ---------------------------------------------------------------------------
# Minimal tokenizer (UTF-8 byte level, no external deps)
# ---------------------------------------------------------------------------
class ByteTokenizer(TokenizerInterface):
    def encode(self, content: str):
        return list(content.encode("utf-8"))

    def decode(self, tokens):
        return bytes(tokens).decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# LLM function wrapping the real OpenAI-compatible API
# ---------------------------------------------------------------------------
async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        os.getenv("LLM_MODEL", "deepseek-chat"),
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("LLM_BINDING_API_KEY") or os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("LLM_BINDING_HOST", "https://api.deepseek.com"),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Intercept wrapper: captures every call's system_prompt without changing behavior
# ---------------------------------------------------------------------------
def intercept_llm_calls(original_func):
    captured = []

    async def wrapper(prompt, system_prompt=None, history_messages=[], **kwargs):
        captured.append({"system_prompt": system_prompt, "user_prompt": prompt})
        return await original_func(
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs,
        )

    return captured, wrapper


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------
async def run_test():
    print("=" * 70)
    print("Entity Extraction JSON Mode Integration Test")
    print("=" * 70)

    # Read test document
    doc_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MyTest.txt")
    with open(doc_path, "r", encoding="utf-8") as f:
        doc_text = f.read()

    chunks = {
        "chunk-001": {
            "tokens": len(doc_text),
            "content": doc_text,
            "full_doc_id": "doc-001",
            "chunk_order_index": 0,
            "file_path": doc_path,
        }
    }

    tokenizer = Tokenizer("byte", ByteTokenizer())
    captured_calls, wrapped_llm = intercept_llm_calls(llm_model_func)

    global_config = {
        "llm_model_func": wrapped_llm,
        "entity_extract_max_gleaning": 0,  # disable gleaning — minimise API calls
        "addon_params": {
            "language": "Chinese",
            # entity_types_guidance not set → must fall back to DEFAULT
        },
        "tokenizer": tokenizer,
        "max_extract_input_tokens": 32768,
        "llm_model_max_async": 1,
        "entity_extraction_use_json": True,
    }

    # ── Run extraction ──────────────────────────────────────────────────────
    print("\n[1/3] Calling extract_entities with entity_extraction_use_json=True ...")
    result = await extract_entities(chunks=chunks, global_config=global_config)
    # extract_entities returns a list of (nodes, edges) tuples, one per chunk
    maybe_nodes: dict = {}
    maybe_edges: dict = {}
    if result:
        for chunk_nodes, chunk_edges in result:
            maybe_nodes.update(chunk_nodes)
            maybe_edges.update(chunk_edges)

    # ── Check 1: entity_types_guidance present in system prompt ──────────────
    print("\n[2/3] Verifying entity_types_guidance in system prompt ...")
    assert captured_calls, "LLM was never called!"
    system_prompt_sent = captured_calls[0]["system_prompt"]

    # The DEFAULT_ENTITY_TYPES_GUIDANCE starts with "Use the following entity types"
    expected_fragment = PROMPTS["DEFAULT_ENTITY_TYPES_GUIDANCE"][:60]
    assert expected_fragment in system_prompt_sent, (
        f"FAIL: entity_types_guidance NOT found in system prompt!\n"
        f"Expected fragment: {expected_fragment!r}\n"
        f"System prompt snippet: {system_prompt_sent[:500]!r}"
    )
    print("  PASS: DEFAULT_ENTITY_TYPES_GUIDANCE correctly injected into system prompt.")

    print("\n--- System Prompt Sent to LLM ---")
    print(system_prompt_sent)

    # ── Check 2: entities and relations extracted ─────────────────────────────
    print("\n[3/3] Verifying extracted entities and relations ...")
    assert maybe_nodes, (
        "FAIL: No entities extracted — JSON parsing or conversion likely failed."
    )
    assert maybe_edges, (
        "FAIL: No relations extracted — JSON parsing or conversion likely failed."
    )
    print(f"  PASS: {len(maybe_nodes)} entities and {len(maybe_edges)} relations extracted.")

    # ── Print results for manual inspection ──────────────────────────────────
    print("\n--- Extracted Entities (first 20) ---")
    for name, data_list in list(maybe_nodes.items())[:20]:
        d = data_list[0]
        print(f"  [{d.get('entity_type', '?')}] {name}: {d.get('description', '')[:80]}")
    if len(maybe_nodes) > 20:
        print(f"  ... and {len(maybe_nodes) - 20} more")

    print("\n--- Extracted Relations (first 10) ---")
    for (src, tgt), data_list in list(maybe_edges.items())[:10]:
        d = data_list[0]
        print(f"  {src} -> {tgt}")
        print(f"    keywords: {d.get('keywords', '')}")
        print(f"    desc:     {d.get('description', '')[:80]}")
    if len(maybe_edges) > 10:
        print(f"  ... and {len(maybe_edges) - 10} more")

    print("\n" + "=" * 70)
    print("All assertions passed.")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_test())
