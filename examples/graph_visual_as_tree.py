import json
import os
import asyncio
import re

import sys
from typing import Dict, Any

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
import numpy as np
from lightrag.kg.shared_storage import initialize_pipeline_status


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


print("Current working directory:", os.getcwd())
WORKING_DIR = "./dickens"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        "solar-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("UPSTAGE_API_KEY"),
        base_url="https://api.upstage.ai/v1/solar",
        **kwargs,
    )


async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embed(
        texts,
        model="solar-embedding-1-large-query",
        api_key=os.getenv("UPSTAGE_API_KEY"),
        base_url="https://api.upstage.ai/v1/solar",
    )


async def get_embedding_dim():
    test_text = ["This is a test sentence."]
    embedding = await embedding_func(test_text)
    embedding_dim = embedding.shape[1]
    return embedding_dim


# function test
async def test_funcs():
    result = await llm_model_func("How are you?")
    print("llm_model_func: ", result)

    result = await embedding_func(["How are you?"])
    print("embedding_func: ", result)


# asyncio.run(test_funcs())


async def initialize_rag():
    embedding_dimension = await get_embedding_dim()
    print(f"Detected embedding dimension: {embedding_dimension}")

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dimension,
            max_token_size=8192,
            func=embedding_func,
        ),
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag

def extract_json_from_response(response: str) -> Dict[str, Any]:
    """Extract valid JSON content from response"""
    # Try direct parsing
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Try removing Markdown code blocks
    cleaned = re.sub(r'```(json)?|```', '', response).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try extracting content between first {...}
    match = re.search(r'\{[\s\S]*\}', cleaned)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Final attempt: remove all possible non-JSON content
    lines = []
    in_json = False
    for line in cleaned.split('\n'):
        if line.strip().startswith('{') or in_json:
            in_json = True
            lines.append(line)
        if line.strip().endswith('}'):
            break
    final_attempt = '\n'.join(lines)
    try:
        return json.loads(final_attempt)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to extract valid JSON from response: {e}\nOriginal response:\n{response}")

async def main():
    try:
        rag = await initialize_rag()

        # Read and insert text
        with open("./pythonBasic.txt", "r", encoding="utf-8") as f:
            await rag.ainsert(f.read())

        custom_prompt = """
        You are a strict JSON generator. You must output only JSON content in the following format, without any additional explanations, comments, or Markdown markers:

        {{
          "id": "Knowledge point name",
          "entity_type": "Type (e.g., concept/technique/method)",
          "description": "Detailed explanation of the knowledge point (merge related sentences, keep core ideas)",
          "source_id": "Source markers (use <SEP> to separate multiple sources)",
          "style": {{"fill": "color code"}},
          "children": [list of sub-knowledge points]
        }}

        Notes:
        1. Do not include any non-JSON content
        2. Do not use Markdown code blocks
        3. Start directly with {{ and end with }}
        4. If there are no sub-knowledge points, assign children field as empty list []
        """

        response = await rag.aquery(
            "Generate a tree diagram from the text I provided, and output it strictly in the above JSON format (overview and preface sections of the textbook do not need to generate knowledge points, no need to generate labels)",
            param=QueryParam(mode="hybrid"),
            system_prompt=custom_prompt
        )

        print("Raw response:\n", response)

        try:
            data = extract_json_from_response(response)
            with open("tree_graph.json", "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print("JSON file saved successfully!")
        except ValueError as e:
            print(f"JSON processing failed: {e}")
            with open("failed_response.txt", "w", encoding="utf-8") as f:
                f.write(response)

    except Exception as e:
        print(f"Program encountered an error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())