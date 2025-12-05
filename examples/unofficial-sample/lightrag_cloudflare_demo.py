import asyncio
import inspect
import logging
import logging.config
import os

import numpy as np
import requests
from dotenv import load_dotenv

from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug

"""This code is a modified version of lightrag_openai_demo.py"""

# ideally, as always, env!
load_dotenv(dotenv_path='.env', override=False)


"""    ----========= IMPORTANT CHANGE THIS! =========----    """
cloudflare_api_key = 'YOUR_API_KEY'
account_id = 'YOUR_ACCOUNT ID'  # This is unique to your Cloudflare account

# Authomatically changes
api_base_url = f'https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/'


# choose an embedding model
EMBEDDING_MODEL = '@cf/baai/bge-m3'
# choose a generative model
LLM_MODEL = '@cf/meta/llama-3.2-3b-instruct'

WORKING_DIR = '../dickens'  # you can change output as desired


# Cloudflare init
class CloudflareWorker:
    def __init__(
        self,
        cloudflare_api_key: str,
        api_base_url: str,
        llm_model_name: str,
        embedding_model_name: str,
        max_tokens: int = 4080,
        max_response_tokens: int = 4080,
    ):
        self.cloudflare_api_key = cloudflare_api_key
        self.api_base_url = api_base_url
        self.llm_model_name = llm_model_name
        self.embedding_model_name = embedding_model_name
        self.max_tokens = max_tokens
        self.max_response_tokens = max_response_tokens

    async def _send_request(self, model_name: str, input_: dict, debug_log: str):
        headers = {'Authorization': f'Bearer {self.cloudflare_api_key}'}

        print(f"""
        data sent to Cloudflare
        ~~~~~~~~~~~
        {debug_log}
        """)

        try:
            response_raw = requests.post(f'{self.api_base_url}{model_name}', headers=headers, json=input_).json()
            print(f"""
        Cloudflare worker responded with:
        ~~~~~~~~~~~
        {response_raw!s}
            """)
            result = response_raw.get('result', {})

            if 'data' in result:  # Embedding case
                return np.array(result['data'])

            if 'response' in result:  # LLM response
                return result['response']

            raise ValueError('Unexpected Cloudflare response format')

        except Exception as e:
            print(f"""
            Cloudflare API returned:
            ~~~~~~~~~
            Error: {e}
            """)
            input('Press Enter to continue...')
            return None

    async def query(self, prompt, system_prompt: str = '', **kwargs) -> str:
        # since no caching is used and we don't want to mess with everything lightrag, pop the kwarg it is
        kwargs.pop('hashing_kv', None)

        message = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': prompt},
        ]

        input_ = {
            'messages': message,
            'max_tokens': self.max_tokens,
            'response_token_limit': self.max_response_tokens,
        }

        return await self._send_request(
            self.llm_model_name,
            input_,
            debug_log=f'\n- model used {self.llm_model_name}\n- system prompt: {system_prompt}\n- query: {prompt}',
        )

    async def embedding_chunk(self, texts: list[str]) -> np.ndarray:
        print(f"""
        TEXT inputted
        ~~~~~
        {texts}
        """)

        input_ = {
            'text': texts,
            'max_tokens': self.max_tokens,
            'response_token_limit': self.max_response_tokens,
        }

        return await self._send_request(
            self.embedding_model_name,
            input_,
            debug_log=f'\n-llm model name {self.embedding_model_name}\n- texts: {texts}',
        )


def configure_logging():
    """Configure logging for the application"""

    # Reset any existing handlers to ensure clean configuration
    for logger_name in ['uvicorn', 'uvicorn.access', 'uvicorn.error', 'lightrag']:
        logger_instance = logging.getLogger(logger_name)
        logger_instance.handlers = []
        logger_instance.filters = []

    # Get log directory path from environment variable or use current directory
    log_dir = os.getenv('LOG_DIR', os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, 'lightrag_cloudflare_worker_demo.log'))

    print(f'\nLightRAG compatible demo log file: {log_file_path}\n')
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    # Get log file max size and backup count from environment variables
    log_max_bytes = int(os.getenv('LOG_MAX_BYTES', 10485760))  # Default 10MB
    log_backup_count = int(os.getenv('LOG_BACKUP_COUNT', 5))  # Default 5 backups

    logging.config.dictConfig(
        {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'default': {
                    'format': '%(levelname)s: %(message)s',
                },
                'detailed': {
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                },
            },
            'handlers': {
                'console': {
                    'formatter': 'default',
                    'class': 'logging.StreamHandler',
                    'stream': 'ext://sys.stderr',
                },
                'file': {
                    'formatter': 'detailed',
                    'class': 'logging.handlers.RotatingFileHandler',
                    'filename': log_file_path,
                    'maxBytes': log_max_bytes,
                    'backupCount': log_backup_count,
                    'encoding': 'utf-8',
                },
            },
            'loggers': {
                'lightrag': {
                    'handlers': ['console', 'file'],
                    'level': 'INFO',
                    'propagate': False,
                },
            },
        }
    )

    # Set the logger level to INFO
    logger.setLevel(logging.INFO)
    # Enable verbose debug if needed
    set_verbose_debug(os.getenv('VERBOSE_DEBUG', 'false').lower() == 'true')


if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def initialize_rag():
    cloudflare_worker = CloudflareWorker(
        cloudflare_api_key=cloudflare_api_key,
        api_base_url=api_base_url,
        embedding_model_name=EMBEDDING_MODEL,
        llm_model_name=LLM_MODEL,
    )

    rag = LightRAG(
        working_dir=WORKING_DIR,
        max_parallel_insert=2,
        llm_model_func=cloudflare_worker.query,
        llm_model_name=os.getenv('LLM_MODEL', LLM_MODEL),
        summary_max_tokens=4080,
        embedding_func=EmbeddingFunc(
            embedding_dim=int(os.getenv('EMBEDDING_DIM', '1024')),
            max_token_size=int(os.getenv('MAX_EMBED_TOKENS', '2048')),
            func=lambda texts: cloudflare_worker.embedding_chunk(
                texts,
            ),
        ),
    )

    await rag.initialize_storages()  # Auto-initializes pipeline_status
    return rag


async def print_stream(stream):
    async for chunk in stream:
        print(chunk, end='', flush=True)


async def main():
    try:
        # Clear old data files
        files_to_delete = [
            'graph_chunk_entity_relation.graphml',
            'kv_store_doc_status.json',
            'kv_store_full_docs.json',
            'kv_store_text_chunks.json',
            'vdb_chunks.json',
            'vdb_entities.json',
            'vdb_relationships.json',
        ]

        for file in files_to_delete:
            file_path = os.path.join(WORKING_DIR, file)
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f'Deleting old file:: {file_path}')

        # Initialize RAG instance
        rag = await initialize_rag()

        # Test embedding function
        test_text = ['This is a test string for embedding.']
        embedding = await rag.embedding_func(test_text)
        embedding_dim = embedding.shape[1]
        print('\n=======================')
        print('Test embedding function')
        print('========================')
        print(f'Test dict: {test_text}')
        print(f'Detected embedding dimension: {embedding_dim}\n\n')

        # Locate the location of what is needed to be added to the knowledge
        # Can add several simultaneously by modifying code
        with open('./book.txt', encoding='utf-8') as f:
            await rag.ainsert(f.read())

        # Perform naive search
        print('\n=====================')
        print('Query mode: naive')
        print('=====================')
        resp = await rag.aquery(
            'What are the top themes in this story?',
            param=QueryParam(mode='naive', stream=True),
        )
        if inspect.isasyncgen(resp):
            await print_stream(resp)
        else:
            print(resp)

        # Perform local search
        print('\n=====================')
        print('Query mode: local')
        print('=====================')
        resp = await rag.aquery(
            'What are the top themes in this story?',
            param=QueryParam(mode='local', stream=True),
        )
        if inspect.isasyncgen(resp):
            await print_stream(resp)
        else:
            print(resp)

        # Perform global search
        print('\n=====================')
        print('Query mode: global')
        print('=====================')
        resp = await rag.aquery(
            'What are the top themes in this story?',
            param=QueryParam(mode='global', stream=True),
        )
        if inspect.isasyncgen(resp):
            await print_stream(resp)
        else:
            print(resp)

        # Perform hybrid search
        print('\n=====================')
        print('Query mode: hybrid')
        print('=====================')
        resp = await rag.aquery(
            'What are the top themes in this story?',
            param=QueryParam(mode='hybrid', stream=True),
        )
        if inspect.isasyncgen(resp):
            await print_stream(resp)
        else:
            print(resp)

        """ FOR TESTING (if you want to test straight away, after building. Uncomment this part"""

        """
        print("\n" + "=" * 60)
        print("AI ASSISTANT READY!")
        print("Ask questions about (your uploaded) regulations")
        print("Type 'quit' to exit")
        print("=" * 60)

        while True:
            question = input("\n🔥 Your question: ")

            if question.lower() in ['quit', 'exit', 'bye']:
                break

            print("\nThinking...")
            response = await rag.aquery(question, param=QueryParam(mode="hybrid"))
            print(f"\nAnswer: {response}")

        """

    except Exception as e:
        print(f'An error occurred: {e}')
    finally:
        if rag:
            await rag.llm_response_cache.index_done_callback()
            await rag.finalize_storages()


if __name__ == '__main__':
    # Configure logging before running the main function
    configure_logging()
    asyncio.run(main())
    print('\nDone!')
