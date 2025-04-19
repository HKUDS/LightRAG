#!/usr/bin/env python
import os
import asyncio
import logging
import sys

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_embed, gpt_4o_mini_complete
from lightrag.kg.shared_storage import initialize_pipeline_status

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Set LightRAG logger to DEBUG level
lightrag_logger = logging.getLogger("lightrag")
lightrag_logger.setLevel(logging.DEBUG)

# Set API credentials
if "DEEPSEEK_API_KEY" not in os.environ:
    os.environ["DEEPSEEK_API_KEY"] = "YOUR DEEPSEEK API KEY FOR REASONING"
if "DEEPSEEK_API_BASE" not in os.environ:
    os.environ["DEEPSEEK_API_BASE"] = "https://api.deepseek.com/v1"
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = "YOUR OPENAI API KEY FOR EMBEDDING"

WORKING_DIR = "./YOUR WORKING DIRECTORY"

if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR)


async def initialize_rag():
    """Initialize LightRAG with the necessary configuration."""
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag


def main():
    # Initialize LightRAG
    rag = asyncio.run(initialize_rag())

    print("\n===== LIGHTRAG REASONING RE-RANKING DEMO =====")
    print("This demo shows the step-by-step reasoning process for re-ranking nodes")
    print(
        "You'll see: 1) Original node ranking, 2) Reasoning chain of thought, 3) Re-ranked nodes"
    )

    print("\n===== STANDARD RANKING (NO REASONING) =====")
    query = "Why does Scrooge manage to have a happy ending?"
    standard_result = rag.query(query, param=QueryParam(mode="local"))
    print(f"\n{standard_result}")

    print("\n===== WITH REASONING RE-RANKING =====")
    print("Now the same query but with reasoning-based re-ranking of nodes:")
    print(
        "Watch for the ORIGINAL NODE RANKING, CHAIN OF THOUGHT REASONING, and RE-RANKED NODE ORDER"
    )
    reasoning_result = rag.query(
        query,
        param=QueryParam(
            mode="local",
            use_reasoning_reranking=True,
            reasoning_model_name="deepseek_r1",
        ),
    )
    print("\n===== FINAL ANSWER WITH REASONING RE-RANKING =====")
    print(f"{reasoning_result}")

    print("\n===== HYBRID MODE WITH REASONING RE-RANKING =====")
    complex_query = "How does Scrooge make a lot of money in the end of the story?"
    print("Using a different query in hybrid mode with reasoning re-ranking:")
    hybrid_result = rag.query(
        complex_query,
        param=QueryParam(
            mode="hybrid",
            use_reasoning_reranking=True,
            reasoning_model_name="deepseek_r1",
        ),
    )
    print("\n===== FINAL ANSWER =====")
    print(f"{hybrid_result}")


if __name__ == "__main__":
    main()
