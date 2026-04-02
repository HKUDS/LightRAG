"""LightRAG + AG2 Multi-Agent Demo.

Demonstrates how AG2 agents can use LightRAG's knowledge graph retrieval
as a tool. Multiple specialized agents collaborate to answer complex
questions over indexed documents.

Architecture:
    User -> AG2 GroupChat (Researcher + Analyst + Writer) -> LightRAG queries
    - Researcher: uses LightRAG hybrid search to gather facts
    - Analyst: uses LightRAG naive (vector) search for complementary results
    - Writer: synthesizes findings into a final answer

Requires:
    pip install lightrag-hku "ag2[openai]>=0.11.4,<1.0"
    export OPENAI_API_KEY="..."

Usage:
    python examples/lightrag_ag2_multiagent_demo.py
"""

import asyncio
import json
import os
import shutil
import threading

from autogen import (
    AssistantAgent,
    GroupChat,
    GroupChatManager,
    LLMConfig,
    UserProxyAgent,
)

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

# --- Configuration ---

WORKING_DIR = "./ag2_demo_workdir"

SAMPLE_TEXT = """
Artificial intelligence has transformed multiple industries. Machine learning,
a subset of AI, enables systems to learn from data without explicit programming.
Deep learning, using neural networks with many layers, has achieved breakthroughs
in computer vision, natural language processing, and speech recognition.

Transformer architectures, introduced in the 2017 paper "Attention Is All You Need"
by Vaswani et al., revolutionized NLP. Models like GPT and BERT are built on
transformers. GPT (Generative Pre-trained Transformer) uses decoder-only architecture
for text generation, while BERT (Bidirectional Encoder Representations) uses
encoder-only architecture for understanding tasks.

Retrieval-Augmented Generation (RAG) combines the strengths of retrieval systems
and generative models. Instead of relying solely on parametric knowledge, RAG
systems retrieve relevant documents from a knowledge base and use them as context
for generation. This approach reduces hallucination and enables models to access
up-to-date information.

Knowledge graphs represent information as entities and relationships. When combined
with RAG, knowledge graphs enable structured reasoning over document collections.
LightRAG implements this approach with dual-level retrieval: local search focuses
on specific entities, while global search captures broader themes and relationships.
"""


# --- LightRAG Setup ---


async def setup_lightrag() -> LightRAG:
    """Initialize LightRAG and index sample documents."""
    if os.path.exists(WORKING_DIR):
        shutil.rmtree(WORKING_DIR)
    os.makedirs(WORKING_DIR, exist_ok=True)

    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
    )
    await rag.initialize_storages()
    await rag.ainsert(SAMPLE_TEXT)
    print("LightRAG initialized and documents indexed.\n")
    return rag


# --- Async Bridge ---
# AG2 runs tools in a background thread without an event loop.
# We maintain a dedicated event loop in a separate thread for LightRAG async calls.

_bg_loop: asyncio.AbstractEventLoop = None


def _start_background_loop(loop: asyncio.AbstractEventLoop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


def _run_async(coro):
    """Submit a coroutine to the background event loop and wait for the result."""
    future = asyncio.run_coroutine_threadsafe(coro, _bg_loop)
    return future.result(timeout=120)


# --- AG2 Agent Tools ---

# Global reference to LightRAG instance (set in main)
_rag_instance: LightRAG = None


def create_agents():
    """Create AG2 agents with LightRAG tools."""
    llm_config = LLMConfig(
        {
            "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            "api_key": os.environ["OPENAI_API_KEY"],
            "api_type": "openai",
        }
    )

    researcher = AssistantAgent(
        name="Researcher",
        system_message=(
            "You are a research specialist. Use the lightrag_query tool to search "
            "the knowledge base. Start with 'hybrid' mode for comprehensive results. "
            "If you need specific entity details, use 'local' mode. "
            "Present your findings as structured bullet points. "
            "Always call the tool -- do NOT answer from your own knowledge."
        ),
        llm_config=llm_config,
    )

    analyst = AssistantAgent(
        name="Analyst",
        system_message=(
            "You are a knowledge graph analyst. Your FIRST action MUST be calling "
            "the lightrag_query tool with mode='naive' to run a direct vector search. "
            "This gives different results from the Researcher's hybrid search. "
            "After receiving the naive search results, compare them with the "
            "Researcher's findings and highlight any additional insights. "
            "You MUST call the tool before writing any analysis."
        ),
        llm_config=llm_config,
    )

    writer = AssistantAgent(
        name="Writer",
        system_message=(
            "You are a technical writer. Synthesize the findings from the "
            "Researcher and Analyst into a clear, well-structured answer. "
            "Do NOT use the search tool -- work only with what the other agents "
            "have found. End your response with TERMINATE."
        ),
        llm_config=llm_config,
    )

    def is_termination(msg):
        return "TERMINATE" in (msg.get("content") or "")

    user_proxy = UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        code_execution_config=False,
        is_termination_msg=is_termination,
    )

    # --- Register LightRAG as a tool ---

    @user_proxy.register_for_execution()
    @researcher.register_for_llm(
        description=(
            "Query the LightRAG knowledge base. "
            "mode: 'naive' (simple vector), 'local' (entity-focused), "
            "'global' (theme/relationship-focused), 'hybrid' (combined). "
            "Returns retrieved context from indexed documents."
        )
    )
    @analyst.register_for_llm(
        description=(
            "Query the LightRAG knowledge base. "
            "mode: 'naive' (simple vector), 'local' (entity-focused), "
            "'global' (theme/relationship-focused), 'hybrid' (combined). "
            "Returns retrieved context from indexed documents."
        )
    )
    def lightrag_query(query: str, mode: str = "hybrid") -> str:
        """Query LightRAG synchronously (wraps async call)."""
        valid_modes = {"naive", "local", "global", "hybrid"}
        if mode not in valid_modes:
            return json.dumps(
                {"error": f"Invalid mode '{mode}'. Use one of: {valid_modes}"}
            )
        try:
            result = _run_async(
                _rag_instance.aquery(query, param=QueryParam(mode=mode))
            )
            return json.dumps({"mode": mode, "query": query, "result": result})
        except Exception as e:
            return json.dumps({"error": str(e)})

    return user_proxy, researcher, analyst, writer


def run_multiagent_query(user_proxy, researcher, analyst, writer, question: str):
    """Run a multi-agent GroupChat to answer a question using LightRAG."""
    # Enforce pipeline: Researcher -> Analyst -> Writer.
    # - Researcher and Analyst hand off to User for tool execution
    # - User returns results to the same agent or advances to the next
    # - Writer has no tool access, so it only follows Analyst
    allowed_transitions = {
        user_proxy: [researcher, analyst],
        researcher: [user_proxy, analyst],
        analyst: [user_proxy, writer],
        writer: [],
    }

    group_chat = GroupChat(
        agents=[user_proxy, researcher, analyst, writer],
        messages=[],
        max_round=12,
        allowed_or_disallowed_speaker_transitions=allowed_transitions,
        speaker_transitions_type="allowed",
    )

    manager = GroupChatManager(
        groupchat=group_chat,
        llm_config=LLMConfig(
            {
                "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                "api_key": os.environ["OPENAI_API_KEY"],
                "api_type": "openai",
            }
        ),
        is_termination_msg=lambda msg: "TERMINATE" in (msg.get("content") or ""),
    )

    print(f"Question: {question}\n{'=' * 60}\n")
    user_proxy.run(manager, message=question).process()
    print(f"\n{'=' * 60}")


# --- Main ---


def main():
    global _rag_instance, _bg_loop

    if not os.getenv("OPENAI_API_KEY"):
        print(
            "Error: OPENAI_API_KEY environment variable is not set.\n"
            "Set it by running: export OPENAI_API_KEY='your-openai-api-key'"
        )
        return

    # Start a background event loop for LightRAG async calls.
    # AG2 tools run in threads without an event loop, so we need a
    # persistent loop that can accept coroutines from any thread.
    _bg_loop = asyncio.new_event_loop()
    bg_thread = threading.Thread(
        target=_start_background_loop, args=(_bg_loop,), daemon=True
    )
    bg_thread.start()

    try:
        # Step 1: Set up LightRAG (async, runs on the background loop)
        _rag_instance = _run_async(setup_lightrag())

        # Step 2: Create AG2 agents with LightRAG tools
        user_proxy, researcher, analyst, writer = create_agents()

        # Step 3: Ask a complex question
        run_multiagent_query(
            user_proxy,
            researcher,
            analyst,
            writer,
            question=(
                "How do transformer architectures relate to RAG systems? "
                "What role do knowledge graphs play in improving retrieval quality?"
            ),
        )
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if _rag_instance:
            _run_async(_rag_instance.finalize_storages())
        _bg_loop.call_soon_threadsafe(_bg_loop.stop)
        bg_thread.join(timeout=5)
        shutil.rmtree(WORKING_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
    print("\nDone!")
