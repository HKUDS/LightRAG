import json
import os
import re

import numpy as np

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc, always_get_an_event_loop


## For Upstage API
# please check if embedding_dim=4096 in lightrag.py and llm.py in lightrag direcotry
async def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs) -> str:
    if history_messages is None:
        history_messages = []
    return await openai_complete_if_cache(
        'solar-mini',
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv('UPSTAGE_API_KEY'),
        base_url='https://api.upstage.ai/v1/solar',
        **kwargs,
    )


async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embed(
        texts,
        model='solar-embedding-1-large-query',
        api_key=os.getenv('UPSTAGE_API_KEY'),
        base_url='https://api.upstage.ai/v1/solar',
    )


## /For Upstage API


def extract_queries(file_path):
    with open(file_path) as f:
        data = f.read()

    data = data.replace('**', '')

    queries = re.findall(r'- Question \d+: (.+)', data)

    return queries


async def process_query(query_text, rag_instance, query_param):
    try:
        result = await rag_instance.aquery(query_text, param=query_param)
        return {'query': query_text, 'result': result}, None
    except Exception as e:
        return None, {'query': query_text, 'error': str(e)}


def run_queries_and_save_to_json(queries, rag_instance, query_param, output_file, error_file):
    loop = always_get_an_event_loop()

    with (
        open(output_file, 'a', encoding='utf-8') as result_file,
        open(error_file, 'a', encoding='utf-8') as err_file,
    ):
        result_file.write('[\n')
        first_entry = True

        for query_text in queries:
            result, error = loop.run_until_complete(process_query(query_text, rag_instance, query_param))

            if result:
                if not first_entry:
                    result_file.write(',\n')
                json.dump(result, result_file, ensure_ascii=False, indent=4)
                first_entry = False
            elif error:
                json.dump(error, err_file, ensure_ascii=False, indent=4)
                err_file.write('\n')

        result_file.write('\n]')


if __name__ == '__main__':
    cls = 'mix'
    mode = 'hybrid'
    WORKING_DIR = f'../{cls}'

    rag = LightRAG(working_dir=WORKING_DIR)
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(embedding_dim=4096, func=embedding_func),
    )
    query_param = QueryParam(mode=mode)

    base_dir = '../datasets/questions'
    queries = extract_queries(f'{base_dir}/{cls}_questions.txt')
    run_queries_and_save_to_json(queries, rag, query_param, f'{base_dir}/result.json', f'{base_dir}/errors.json')
