import os
import json
import asyncio
import logging
from neo4j import AsyncGraphDatabase
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.utils import setup_logger


# Neo4j configuration (set as environment variables)
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "your_password"
os.environ["NEO4J_DATABASE"] = "neo4j"


# ---------------------------------------------------------------------------
# Monkey-patch the ASYNC Neo4j driver to log all Cypher queries
# LightRAG uses AsyncGraphDatabase, not the sync GraphDatabase.
# ---------------------------------------------------------------------------
_original_async_driver = AsyncGraphDatabase.driver


def _logging_async_driver(*args, **kwargs):
    """Wrap the real async driver so every session.run() prints its Cypher."""
    driver = _original_async_driver(*args, **kwargs)

    _original_session = driver.session

    def _patched_session(**sess_kwargs):
        session = _original_session(**sess_kwargs)

        _original_run = session.run

        async def _logging_run(query, parameters=None, **run_kwargs):
            print(f"\n{'=' * 60}")
            print(f"--- Cypher Query ---")
            print(query)

            # Merge both sources: the positional `parameters` dict
            # AND keyword args (LightRAG passes params as kwargs like
            # session.run(query, pairs=pairs, entity_id=node_id))
            all_params = {}
            if parameters:
                all_params.update(parameters)
            if run_kwargs:
                all_params.update(run_kwargs)

            if all_params:
                print(f"--- Parameters ---")
                for k, v in all_params.items():
                    try:
                        val_str = json.dumps(v, ensure_ascii=False, indent=2)
                    except (TypeError, ValueError):
                        val_str = str(v)
                    # Truncate very long values for readability
                    if len(val_str) > 1000:
                        val_str = val_str[:1000] + "\n  ... (truncated)"
                    print(f"  ${k}: {val_str}")
            print(f"{'=' * 60}\n")
            return await _original_run(query, parameters, **run_kwargs)

        session.run = _logging_run
        return session

    driver.session = _patched_session
    return driver


AsyncGraphDatabase.driver = _logging_async_driver


# ---------------------------------------------------------------------------
# Also allow the neo4j driver's own debug logs (optional, very verbose)
# ---------------------------------------------------------------------------
# logging.getLogger("neo4j").setLevel(logging.DEBUG)

# Configure LightRAG logger
setup_logger("lightrag", level="DEBUG")  # Set to DEBUG for more verbose logs


async def main():
    # Initialize LightRAG with Neo4j
    rag = LightRAG(
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
        graph_storage="Neo4JStorage",
    )
    await rag.initialize_storages()

    # Ask a question (this will trigger Cypher queries against Neo4j)
    question = "what is footwork, how it works"

    # All modes except "naive" and "bypass" use Neo4j graph queries:
    #   "local"  - entity-focused (low-level keywords → nodes → connected edges)
    #   "global" - relationship-focused (high-level keywords → edges → connected nodes)
    #   "hybrid" - local + global combined
    #   "mix"    - hybrid + text chunk vector search (default)
    answer = await rag.aquery(
        question,
        param=QueryParam(mode="global"),  # Forces Neo4j graph retrieval
    )

    print(f"\n{'#' * 60}")
    print(f"Answer: {answer}")
    print(f"{'#' * 60}")

    await rag.finalize_storages()


asyncio.run(main())

# Restore original driver
AsyncGraphDatabase.driver = _original_async_driver
