import json
import re

from lightrag import LightRAG, QueryParam
from lightrag.utils import always_get_an_event_loop, logger


def extract_queries(file_path):
    try:
        logger.info(f'Reading queries from {file_path}')
        with open(file_path, encoding='utf-8') as f:
            data = f.read()
    except FileNotFoundError:
        logger.error(f'File not found: {file_path}')
        return []
    except OSError as exc:
        logger.error(f'Error reading file {file_path}: {exc}')
        return []

    data = data.replace('**', '')
    queries = re.findall(r'- Question \d+: (.+)', data)
    if not queries:
        logger.warning(f'No queries found in {file_path}; unexpected format?')

    logger.info(f'Extracted {len(queries)} queries')
    return queries


async def process_query(query_text, rag_instance, query_param):
    try:
        logger.debug(f'Processing query: {query_text[:100]}...')
        result = await rag_instance.aquery(query_text, param=query_param)
        return {'query': query_text, 'result': result}, None
    except Exception as e:
        logger.error(f'Error processing query: {e}', exc_info=True)
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
            try:
                result, error = loop.run_until_complete(process_query(query_text, rag_instance, query_param))
            except RuntimeError as e:
                if 'attached to a different loop' in str(e):
                    logger.error(f'Event loop mismatch while processing query: {e}')
                    error = {'query': query_text, 'error': f'Event loop error: {e}'}
                    result = None
                else:
                    raise
            except Exception as e:
                logger.error(f'Unexpected error running query: {e}', exc_info=True)
                error = {'query': query_text, 'error': f'Unexpected error: {e}'}
                result = None

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
    cls = 'agriculture'
    mode = 'hybrid'
    WORKING_DIR = f'../{cls}'

    rag = LightRAG(working_dir=WORKING_DIR)
    query_param = QueryParam(mode=mode)

    queries = extract_queries(f'../datasets/questions/{cls}_questions.txt')
    run_queries_and_save_to_json(queries, rag, query_param, f'{cls}_result.json', f'{cls}_errors.json')
