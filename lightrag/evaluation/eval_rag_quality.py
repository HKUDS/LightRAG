#!/usr/bin/env python3
"""
RAGAS Evaluation Script for LightRAG System

Evaluates RAG response quality using RAGAS metrics:
- Faithfulness: Is the answer factually accurate based on context?
- Answer Relevance: Is the answer relevant to the question?
- Context Recall: Is all relevant information retrieved?
- Context Precision: Is retrieved context clean without noise?

Usage:
    # Use defaults (sample_dataset.json, http://localhost:9621)
    python lightrag/evaluation/eval_rag_quality.py

    # Specify custom dataset
    python lightrag/evaluation/eval_rag_quality.py --dataset my_test.json
    python lightrag/evaluation/eval_rag_quality.py -d my_test.json

    # Specify custom RAG endpoint
    python lightrag/evaluation/eval_rag_quality.py --ragendpoint http://my-server.com:9621
    python lightrag/evaluation/eval_rag_quality.py -r http://my-server.com:9621

    # Specify both
    python lightrag/evaluation/eval_rag_quality.py -d my_test.json -r http://localhost:9621

    # Get help
    python lightrag/evaluation/eval_rag_quality.py --help

Results are saved to: lightrag/evaluation/results/
    - results_YYYYMMDD_HHMMSS.csv   (CSV export for analysis)
    - results_YYYYMMDD_HHMMSS.json  (Full results with details)

Technical Notes:
    - Uses stable RAGAS API (LangchainLLMWrapper) for maximum compatibility
    - Supports custom OpenAI-compatible endpoints via EVAL_LLM_BINDING_HOST
    - Enables bypass_n mode for endpoints that don't support 'n' parameter
    - Deprecation warnings are suppressed for cleaner output
"""

import argparse
import asyncio
import csv
import json
import math
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

import httpx
from dotenv import load_dotenv

from lightrag.utils import logger

# Suppress LangchainLLMWrapper deprecation warning
# We use LangchainLLMWrapper for stability and compatibility with all RAGAS versions
warnings.filterwarnings(
    'ignore',
    message='.*LangchainLLMWrapper is deprecated.*',
    category=DeprecationWarning,
)

# Suppress token usage warning for custom OpenAI-compatible endpoints
# Custom endpoints (vLLM, SGLang, etc.) often don't return usage information
# This is non-critical as token tracking is not required for RAGAS evaluation
warnings.filterwarnings(
    'ignore',
    message='.*Unexpected type for token usage.*',
    category=UserWarning,
)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path='.env', override=False)

# Placeholder annotations for optional dependencies
ChatOpenAI: Any = None
OpenAIEmbeddings: Any = None
LangchainLLMWrapper: Any = None
AnswerRelevancy: Any = None
ContextPrecision: Any = None
ContextRecall: Any = None
Faithfulness: Any = None
Dataset: Any = None
evaluate: Any = None
tqdm: Any = None

# Conditional imports - will raise ImportError if dependencies not installed
try:
    from datasets import Dataset
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )
    from tqdm.auto import tqdm

    RAGAS_AVAILABLE = True

except ImportError:
    RAGAS_AVAILABLE = False
    Dataset = None
    evaluate = None
    LangchainLLMWrapper = None
    ChatOpenAI = None
    OpenAIEmbeddings = None
    AnswerRelevancy = None
    ContextPrecision = None
    ContextRecall = None
    Faithfulness = None
    tqdm = None


CONNECT_TIMEOUT_SECONDS = 180.0
READ_TIMEOUT_SECONDS = 300.0
TOTAL_TIMEOUT_SECONDS = 180.0


def _is_nan(value: Any) -> bool:
    """Return True when value is a float NaN."""
    return isinstance(value, float) and math.isnan(value)


class RAGEvaluator:
    """Evaluate RAG system quality using RAGAS metrics"""

    def __init__(
        self,
        test_dataset_path: str | Path | None = None,
        rag_api_url: str | None = None,
        query_mode: str = 'mix',
        debug_mode: bool = False,
    ):
        """
        Initialize evaluator with test dataset

        Args:
            test_dataset_path: Path to test dataset JSON file
            rag_api_url: Base URL of LightRAG API (e.g., http://localhost:9621)
                        If None, will try to read from environment or use default
            query_mode: Query mode for retrieval (local, global, hybrid, mix, naive)
            debug_mode: Enable verbose logging of retrieved contexts

        Environment Variables:
            EVAL_LLM_MODEL: LLM model for evaluation (default: gpt-4o-mini)
            EVAL_EMBEDDING_MODEL: Embedding model for evaluation (default: text-embedding-3-small)
            EVAL_LLM_BINDING_API_KEY: API key for LLM (fallback to OPENAI_API_KEY)
            EVAL_LLM_BINDING_HOST: Custom endpoint URL for LLM (optional)
            EVAL_EMBEDDING_BINDING_API_KEY: API key for embeddings (fallback: EVAL_LLM_BINDING_API_KEY -> OPENAI_API_KEY)
            EVAL_EMBEDDING_BINDING_HOST: Custom endpoint URL for embeddings (fallback: EVAL_LLM_BINDING_HOST)

        Raises:
            ImportError: If ragas or datasets packages are not installed
            EnvironmentError: If EVAL_LLM_BINDING_API_KEY and OPENAI_API_KEY are both not set
        """
        # Validate RAGAS dependencies are installed
        if not RAGAS_AVAILABLE:
            raise ImportError('RAGAS dependencies not installed. Install with: pip install ragas datasets')

        # Configure evaluation LLM (for RAGAS scoring)
        eval_llm_api_key = os.getenv('EVAL_LLM_BINDING_API_KEY') or os.getenv('OPENAI_API_KEY')
        if not eval_llm_api_key:
            raise OSError(
                'EVAL_LLM_BINDING_API_KEY or OPENAI_API_KEY is required for evaluation. '
                'Set EVAL_LLM_BINDING_API_KEY to use a custom API key, '
                'or ensure OPENAI_API_KEY is set.'
            )

        eval_model = os.getenv('EVAL_LLM_MODEL', 'gpt-4o-mini')
        eval_llm_base_url = os.getenv('EVAL_LLM_BINDING_HOST')

        # Configure evaluation embeddings (for RAGAS scoring)
        # Fallback chain: EVAL_EMBEDDING_BINDING_API_KEY -> EVAL_LLM_BINDING_API_KEY -> OPENAI_API_KEY
        eval_embedding_api_key = (
            os.getenv('EVAL_EMBEDDING_BINDING_API_KEY')
            or os.getenv('EVAL_LLM_BINDING_API_KEY')
            or os.getenv('OPENAI_API_KEY')
        )
        eval_embedding_model = os.getenv('EVAL_EMBEDDING_MODEL', 'text-embedding-3-large')
        # Fallback chain: EVAL_EMBEDDING_BINDING_HOST -> EVAL_LLM_BINDING_HOST -> None
        eval_embedding_base_url = os.getenv('EVAL_EMBEDDING_BINDING_HOST') or os.getenv('EVAL_LLM_BINDING_HOST')

        # Create LLM and Embeddings instances for RAGAS
        llm_kwargs = {
            'model': eval_model,
            'api_key': eval_llm_api_key,
            'max_retries': int(os.getenv('EVAL_LLM_MAX_RETRIES', '5')),
            'request_timeout': int(os.getenv('EVAL_LLM_TIMEOUT', '180')),
        }
        embedding_kwargs = {
            'model': eval_embedding_model,
            'api_key': eval_embedding_api_key,
        }

        if eval_llm_base_url:
            llm_kwargs['base_url'] = eval_llm_base_url

        if eval_embedding_base_url:
            embedding_kwargs['base_url'] = eval_embedding_base_url

        # Create base LangChain LLM
        base_llm = ChatOpenAI(**llm_kwargs)
        self.eval_embeddings = OpenAIEmbeddings(**embedding_kwargs)

        # Wrap LLM with LangchainLLMWrapper and enable bypass_n mode for custom endpoints
        # This ensures compatibility with endpoints that don't support the 'n' parameter
        # by generating multiple outputs through repeated prompts instead of using 'n' parameter
        try:
            self.eval_llm = LangchainLLMWrapper(
                langchain_llm=base_llm,
                bypass_n=True,  # Enable bypass_n to avoid passing 'n' to OpenAI API
            )
            logger.debug('Successfully configured bypass_n mode for LLM wrapper')
        except Exception as e:
            logger.warning(
                'Could not configure LangchainLLMWrapper with bypass_n: %s. '
                'Using base LLM directly, which may cause warnings with custom endpoints.',
                e,
            )
            self.eval_llm = base_llm

        if test_dataset_path is None:
            test_dataset_path = Path(__file__).parent / 'sample_dataset.json'

        if rag_api_url is None:
            rag_api_url = os.getenv('LIGHTRAG_API_URL', 'http://localhost:9621')

        self.test_dataset_path = Path(test_dataset_path)
        self.rag_api_url = rag_api_url.rstrip('/')
        self.query_mode = query_mode
        self.debug_mode = debug_mode
        self.results_dir = Path(__file__).parent / 'results'
        self.results_dir.mkdir(exist_ok=True)

        # Load test dataset
        self.test_cases = self._load_test_dataset()

        # Store configuration values for display
        self.eval_model = eval_model
        self.eval_embedding_model = eval_embedding_model
        self.eval_llm_base_url = eval_llm_base_url
        self.eval_embedding_base_url = eval_embedding_base_url
        self.eval_max_retries = llm_kwargs['max_retries']
        self.eval_timeout = llm_kwargs['request_timeout']

        # Display configuration
        self._display_configuration()

    def _display_configuration(self):
        """Display all evaluation configuration settings"""
        logger.info('Evaluation Models:')
        logger.info('  • LLM Model:            %s', self.eval_model)
        logger.info('  • Embedding Model:      %s', self.eval_embedding_model)

        # Display LLM endpoint
        if self.eval_llm_base_url:
            logger.info('  • LLM Endpoint:         %s', self.eval_llm_base_url)
            logger.info('  • Bypass N-Parameter:   Enabled (use LangchainLLMWrapper for compatibility)')
        else:
            logger.info('  • LLM Endpoint:         OpenAI Official API')

        # Display Embedding endpoint (only if different from LLM)
        if self.eval_embedding_base_url:
            if self.eval_embedding_base_url != self.eval_llm_base_url:
                logger.info('  • Embedding Endpoint:   %s', self.eval_embedding_base_url)
            # If same as LLM endpoint, no need to display separately
        elif not self.eval_llm_base_url:
            # Both using OpenAI - already displayed above
            pass
        else:
            # LLM uses custom endpoint, but embeddings use OpenAI
            logger.info('  • Embedding Endpoint:   OpenAI Official API')

        logger.info('Concurrency & Rate Limiting:')
        query_top_k = int(os.getenv('EVAL_QUERY_TOP_K', '10'))
        logger.info('  • Query Top-K:          %s Entities/Relations', query_top_k)
        logger.info('  • LLM Max Retries:      %s', self.eval_max_retries)
        logger.info('  • LLM Timeout:          %s seconds', self.eval_timeout)

        logger.info('Test Configuration:')
        logger.info('  • Total Test Cases:     %s', len(self.test_cases))
        logger.info('  • Test Dataset:         %s', self.test_dataset_path.name)
        logger.info('  • LightRAG API:         %s', self.rag_api_url)
        logger.info('  • Query Mode:           %s', self.query_mode)
        logger.info('  • Results Directory:    %s', self.results_dir.name)

    def _load_test_dataset(self) -> list[dict[str, str]]:
        """Load test cases from JSON file"""
        if not self.test_dataset_path.exists():
            raise FileNotFoundError(f'Test dataset not found: {self.test_dataset_path}')

        with open(self.test_dataset_path) as f:
            data = json.load(f)

        return data.get('test_cases', [])

    async def generate_rag_response(
        self,
        question: str,
        client: httpx.AsyncClient,
        test_case: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Generate RAG response by calling LightRAG API.

        Args:
            question: The user query.
            client: Shared httpx AsyncClient for connection pooling.
            test_case: Optional test case dict with keyword overrides (hl_keywords, ll_keywords).

        Returns:
            Dictionary with 'answer' and 'contexts' keys.
            'contexts' is a list of strings (one per retrieved document).

        Raises:
            Exception: If LightRAG API is unavailable.
        """
        try:
            # Build payload with tunable parameters via environment variables
            # These allow testing different retrieval configurations without code changes
            payload = {
                'query': question,
                'mode': self.query_mode,
                'include_references': True,
                'include_chunk_content': True,  # Request chunk content in references
                'response_type': 'Single Paragraph',
                # Retrieval tuning parameters (override via EVAL_* env vars)
                'top_k': int(os.getenv('EVAL_QUERY_TOP_K', '10')),
                'chunk_top_k': int(os.getenv('EVAL_CHUNK_TOP_K', '10')),
                'max_total_tokens': int(os.getenv('EVAL_MAX_TOTAL_TOKENS', '30000')),
                'enable_rerank': os.getenv('EVAL_ENABLE_RERANK', 'true').lower() == 'true',
            }

            # Optional keyword overrides from test dataset - bypasses LLM keyword extraction
            # Useful for testing if poor keyword extraction is causing retrieval failures
            if test_case:
                if test_case.get('hl_keywords'):
                    payload['hl_keywords'] = test_case['hl_keywords']
                    if self.debug_mode:
                        logger.info('[DEBUG] Using HL keywords override: %s', test_case['hl_keywords'])
                if test_case.get('ll_keywords'):
                    payload['ll_keywords'] = test_case['ll_keywords']
                    if self.debug_mode:
                        logger.info('[DEBUG] Using LL keywords override: %s', test_case['ll_keywords'])

            # Get API key from environment for authentication
            api_key = os.getenv('LIGHTRAG_API_KEY')

            # Prepare headers with optional authentication
            headers = {}
            if api_key:
                headers['X-API-Key'] = api_key

            # Single optimized API call - gets both answer AND chunk content
            response = await client.post(
                f'{self.rag_api_url}/query',
                json=payload,
                headers=headers if headers else None,
            )
            response.raise_for_status()
            result = response.json()

            answer = result.get('response', 'No response generated')
            references = result.get('references', [])

            # DEBUG: Inspect the API response
            logger.debug('🔍 References Count: %s', len(references))
            if references:
                first_ref = references[0]
                logger.debug('🔍 First Reference Keys: %s', list(first_ref.keys()))
                if 'content' in first_ref:
                    content_preview = first_ref['content']
                    if isinstance(content_preview, list) and content_preview:
                        logger.debug(
                            '🔍 Content Preview (first chunk): %s...',
                            content_preview[0][:100],
                        )
                    elif isinstance(content_preview, str):
                        logger.debug('🔍 Content Preview: %s...', content_preview[:100])

            # Extract chunk content from enriched references
            # Note: content is now a list of chunks per reference (one file may have multiple chunks)
            contexts = []
            for ref in references:
                content = ref.get('content', [])
                if isinstance(content, list):
                    # Flatten the list: each chunk becomes a separate context
                    contexts.extend(content)
                elif isinstance(content, str):
                    # Backward compatibility: if content is still a string (shouldn't happen)
                    contexts.append(content)

            # Debug logging for troubleshooting retrieval issues
            if self.debug_mode:
                logger.info('[DEBUG] Query: %s', question[:100])
                logger.info('[DEBUG] Retrieved %d context chunks', len(contexts))
                if not contexts:
                    logger.warning('[DEBUG] ⚠️ NO CONTEXTS RETRIEVED! Check keyword extraction and KB coverage.')
                    logger.info('[DEBUG] Answer preview: %s', answer[:200] if answer else 'No answer')
                else:
                    for i, ctx in enumerate(contexts[:3]):
                        ctx_preview = ctx[:300] if isinstance(ctx, str) else str(ctx)[:300]
                        logger.info('[DEBUG] Context %d: %s...', i + 1, ctx_preview)
                    if len(contexts) > 3:
                        logger.info('[DEBUG] ... and %d more contexts', len(contexts) - 3)

            return {
                'answer': answer,
                'contexts': contexts,  # List of strings from actual retrieved chunks
            }

        except httpx.ConnectError as e:
            raise Exception(
                f'❌ Cannot connect to LightRAG API at {self.rag_api_url}\n'
                f'   Make sure LightRAG server is running:\n'
                f'   python -m lightrag.api.lightrag_server\n'
                f'   Error: {e!s}'
            ) from e
        except httpx.HTTPStatusError as e:
            raise Exception(f'LightRAG API error {e.response.status_code}: {e.response.text}') from e
        except httpx.ReadTimeout as e:
            raise Exception(
                f'Request timeout after waiting for response\n   Question: {question[:100]}...\n   Error: {e!s}'
            ) from e
        except Exception as e:
            raise Exception(f'Error calling LightRAG API: {type(e).__name__}: {e!s}') from e

    async def evaluate_single_case(
        self,
        idx: int,
        test_case: dict[str, str],
        rag_semaphore: asyncio.Semaphore,
        eval_semaphore: asyncio.Semaphore,
        client: httpx.AsyncClient,
        progress_counter: dict[str, int],
        position_pool: asyncio.Queue,
        pbar_creation_lock: asyncio.Lock,
    ) -> dict[str, Any]:
        """
        Evaluate a single test case with two-stage pipeline concurrency control

        Args:
            idx: Test case index (1-based)
            test_case: Test case dictionary with question and ground_truth
            rag_semaphore: Semaphore to control overall concurrency (covers entire function)
            eval_semaphore: Semaphore to control RAGAS evaluation concurrency (Stage 2)
            client: Shared httpx AsyncClient for connection pooling
            progress_counter: Shared dictionary for progress tracking
            position_pool: Queue of available tqdm position indices
            pbar_creation_lock: Lock to serialize tqdm creation and prevent race conditions

        Returns:
            Evaluation result dictionary
        """
        # rag_semaphore controls the entire evaluation process to prevent
        # all RAG responses from being generated at once when eval is slow
        async with rag_semaphore:
            question = test_case['question']
            ground_truth = test_case['ground_truth']

            # Stage 1: Generate RAG response
            try:
                rag_response = await self.generate_rag_response(question=question, client=client, test_case=test_case)
            except Exception as e:
                logger.error('Error generating response for test %s: %s', idx, str(e))
                progress_counter['completed'] += 1
                return {
                    'test_number': idx,
                    'question': question,
                    'error': str(e),
                    'metrics': {},
                    'ragas_score': 0,
                    'timestamp': datetime.now().isoformat(),
                }

            # *** CRITICAL FIX: Use actual retrieved contexts, NOT ground_truth ***
            retrieved_contexts = rag_response['contexts']

            # Prepare dataset for RAGAS evaluation with CORRECT contexts
            eval_dataset = Dataset.from_dict(
                {
                    'question': [question],
                    'answer': [rag_response['answer']],
                    'contexts': [retrieved_contexts],
                    'ground_truth': [ground_truth],
                }
            )

            # Stage 2: Run RAGAS evaluation (controlled by eval_semaphore)
            # IMPORTANT: Create fresh metric instances for each evaluation to avoid
            # concurrent state conflicts when multiple tasks run in parallel
            async with eval_semaphore:
                pbar = None
                position = None
                try:
                    # Acquire a position from the pool for this tqdm progress bar
                    position = await position_pool.get()

                    # Serialize tqdm creation to prevent race conditions
                    # Multiple tasks creating tqdm simultaneously can cause display conflicts
                    async with pbar_creation_lock:
                        # Create tqdm progress bar with assigned position to avoid overlapping
                        # leave=False ensures the progress bar is cleared after completion,
                        # preventing accumulation of completed bars and allowing position reuse
                        pbar = tqdm(
                            total=4,
                            desc=f'Eval-{idx:02d}',
                            position=position,
                            leave=False,
                        )
                        # Give tqdm time to initialize and claim its screen position
                        await asyncio.sleep(0.05)

                    eval_results = evaluate(
                        dataset=eval_dataset,
                        metrics=[
                            Faithfulness(),
                            AnswerRelevancy(),
                            ContextRecall(),
                            ContextPrecision(),
                        ],
                        llm=self.eval_llm,
                        embeddings=self.eval_embeddings,
                        _pbar=pbar,
                    )

                    # Convert to DataFrame (RAGAS v0.3+ API)
                    df = eval_results.to_pandas()

                    # Extract scores from first row
                    scores_row = df.iloc[0]

                    # Extract scores (RAGAS v0.3+ uses .to_pandas())
                    result = {
                        'test_number': idx,
                        'question': question,
                        'answer': rag_response['answer'][:200] + '...'
                        if len(rag_response['answer']) > 200
                        else rag_response['answer'],
                        'ground_truth': ground_truth[:200] + '...' if len(ground_truth) > 200 else ground_truth,
                        'project': test_case.get('project', 'unknown'),
                        'metrics': {
                            'faithfulness': float(scores_row.get('faithfulness', 0)),
                            'answer_relevance': float(scores_row.get('answer_relevancy', 0)),
                            'context_recall': float(scores_row.get('context_recall', 0)),
                            'context_precision': float(scores_row.get('context_precision', 0)),
                        },
                        'timestamp': datetime.now().isoformat(),
                    }

                    # Calculate RAGAS score (average of all metrics, excluding NaN values)
                    metrics = result['metrics']
                    valid_metrics = [v for v in metrics.values() if not _is_nan(v)]
                    ragas_score = sum(valid_metrics) / len(valid_metrics) if valid_metrics else 0
                    result['ragas_score'] = round(ragas_score, 4)

                    # Update progress counter
                    progress_counter['completed'] += 1

                    return result

                except Exception as e:
                    logger.error('Error evaluating test %s: %s', idx, str(e))
                    progress_counter['completed'] += 1
                    return {
                        'test_number': idx,
                        'question': question,
                        'error': str(e),
                        'metrics': {},
                        'ragas_score': 0,
                        'timestamp': datetime.now().isoformat(),
                    }
                finally:
                    # Force close progress bar to ensure completion
                    if pbar is not None:
                        pbar.close()
                    # Release the position back to the pool for reuse
                    if position is not None:
                        await position_pool.put(position)

    async def evaluate_responses(self) -> list[dict[str, Any]]:
        """
        Evaluate all test cases in parallel with two-stage pipeline and return metrics

        Returns:
            List of evaluation results with metrics
        """
        # Get evaluation concurrency from environment (default to 2 for parallel evaluation)
        max_async = int(os.getenv('EVAL_MAX_CONCURRENT', '2'))

        logger.info('%s', '=' * 70)
        logger.info('🚀 Starting RAGAS Evaluation of LightRAG System')
        logger.info('🔧 RAGAS Evaluation (Stage 2): %s concurrent', max_async)
        logger.info('%s', '=' * 70)

        # Create two-stage pipeline semaphores
        # Stage 1: RAG generation - allow x2 concurrency to keep evaluation fed
        rag_semaphore = asyncio.Semaphore(max_async * 2)
        # Stage 2: RAGAS evaluation - primary bottleneck
        eval_semaphore = asyncio.Semaphore(max_async)

        # Create progress counter (shared across all tasks)
        progress_counter = {'completed': 0}

        # Create position pool for tqdm progress bars
        # Positions range from 0 to max_async-1, ensuring no overlapping displays
        position_pool = asyncio.Queue()
        for i in range(max_async):
            await position_pool.put(i)

        # Create lock to serialize tqdm creation and prevent race conditions
        # This ensures progress bars are created one at a time, avoiding display conflicts
        pbar_creation_lock = asyncio.Lock()

        # Create shared HTTP client with connection pooling and proper timeouts
        # Timeout: 3 minutes for connect, 5 minutes for read (LLM can be slow)
        timeout = httpx.Timeout(
            TOTAL_TIMEOUT_SECONDS,
            connect=CONNECT_TIMEOUT_SECONDS,
            read=READ_TIMEOUT_SECONDS,
        )
        limits = httpx.Limits(
            max_connections=(max_async + 1) * 2,  # Allow buffer for RAG stage
            max_keepalive_connections=max_async + 1,
        )

        async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
            # Create tasks for all test cases
            tasks = [
                self.evaluate_single_case(
                    idx,
                    test_case,
                    rag_semaphore,
                    eval_semaphore,
                    client,
                    progress_counter,
                    position_pool,
                    pbar_creation_lock,
                )
                for idx, test_case in enumerate(self.test_cases, 1)
            ]

            # Run all evaluations in parallel (limited by two-stage semaphores)
            results = await asyncio.gather(*tasks)

        return list(results)

    def _export_to_csv(self, results: list[dict[str, Any]]) -> Path:
        """
        Export evaluation results to CSV file

        Args:
            results: List of evaluation results

        Returns:
            Path to the CSV file

        CSV Format:
            - question: The test question
            - project: Project context
            - faithfulness: Faithfulness score (0-1)
            - answer_relevance: Answer relevance score (0-1)
            - context_recall: Context recall score (0-1)
            - context_precision: Context precision score (0-1)
            - ragas_score: Overall RAGAS score (0-1)
            - timestamp: When evaluation was run
        """
        csv_path = self.results_dir / f'results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'test_number',
                'question',
                'project',
                'faithfulness',
                'answer_relevance',
                'context_recall',
                'context_precision',
                'ragas_score',
                'status',
                'timestamp',
            ]

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for idx, result in enumerate(results, 1):
                metrics = result.get('metrics', {})
                writer.writerow(
                    {
                        'test_number': idx,
                        'question': result.get('question', ''),
                        'project': result.get('project', 'unknown'),
                        'faithfulness': f'{metrics.get("faithfulness", 0):.4f}',
                        'answer_relevance': f'{metrics.get("answer_relevance", 0):.4f}',
                        'context_recall': f'{metrics.get("context_recall", 0):.4f}',
                        'context_precision': f'{metrics.get("context_precision", 0):.4f}',
                        'ragas_score': f'{result.get("ragas_score", 0):.4f}',
                        'status': 'success' if metrics else 'error',
                        'timestamp': result.get('timestamp', ''),
                    }
                )

        return csv_path

    def _format_metric(self, value: float, width: int = 6) -> str:
        """
        Format a metric value for display, handling NaN gracefully

        Args:
            value: The metric value to format
            width: The width of the formatted string

        Returns:
            Formatted string (e.g., "0.8523" or "  N/A ")
        """
        if _is_nan(value):
            return 'N/A'.center(width)
        return f'{value:.4f}'.rjust(width)

    def _display_results_table(self, results: list[dict[str, Any]]):
        """
        Display evaluation results in a formatted table

        Args:
            results: List of evaluation results
        """
        logger.info('')
        logger.info('%s', '=' * 115)
        logger.info('📊 EVALUATION RESULTS SUMMARY')
        logger.info('%s', '=' * 115)

        # Table header
        logger.info(
            '%-4s | %-50s | %6s | %7s | %6s | %7s | %6s | %6s',
            '#',
            'Question',
            'Faith',
            'AnswRel',
            'CtxRec',
            'CtxPrec',
            'RAGAS',
            'Status',
        )
        logger.info('%s', '-' * 115)

        # Table rows
        for result in results:
            test_num = result.get('test_number', 0)
            question = result.get('question', '')
            # Truncate question to 50 chars
            question_display = (question[:47] + '...') if len(question) > 50 else question

            metrics = result.get('metrics', {})
            if metrics:
                # Success case - format each metric, handling NaN values
                faith = metrics.get('faithfulness', 0)
                ans_rel = metrics.get('answer_relevance', 0)
                ctx_rec = metrics.get('context_recall', 0)
                ctx_prec = metrics.get('context_precision', 0)
                ragas = result.get('ragas_score', 0)
                status = '✓'

                logger.info(
                    '%-4d | %-50s | %s | %s | %s | %s | %s | %6s',
                    test_num,
                    question_display,
                    self._format_metric(faith, 6),
                    self._format_metric(ans_rel, 7),
                    self._format_metric(ctx_rec, 6),
                    self._format_metric(ctx_prec, 7),
                    self._format_metric(ragas, 6),
                    status,
                )
            else:
                # Error case
                error = result.get('error', 'Unknown error')
                error_display = (error[:20] + '...') if len(error) > 23 else error
                logger.info(
                    '%-4d | %-50s | %6s | %7s | %6s | %7s | %6s | ✗ %s',
                    test_num,
                    question_display,
                    'N/A',
                    'N/A',
                    'N/A',
                    'N/A',
                    'N/A',
                    error_display,
                )

        logger.info('%s', '=' * 115)

    def _calculate_benchmark_stats(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Calculate benchmark statistics from evaluation results

        Args:
            results: List of evaluation results

        Returns:
            Dictionary with benchmark statistics
        """
        # Filter out results with errors
        valid_results = [r for r in results if r.get('metrics')]
        total_tests = len(results)
        successful_tests = len(valid_results)
        failed_tests = total_tests - successful_tests

        if not valid_results:
            return {
                'total_tests': total_tests,
                'successful_tests': 0,
                'failed_tests': failed_tests,
                'success_rate': 0.0,
            }

        # Calculate averages for each metric (handling NaN values correctly)
        # Track both sum and count for each metric to handle NaN values properly
        metrics_data = {
            'faithfulness': {'sum': 0.0, 'count': 0},
            'answer_relevance': {'sum': 0.0, 'count': 0},
            'context_recall': {'sum': 0.0, 'count': 0},
            'context_precision': {'sum': 0.0, 'count': 0},
            'ragas_score': {'sum': 0.0, 'count': 0},
        }

        for result in valid_results:
            metrics = result.get('metrics', {})

            # For each metric, sum non-NaN values and count them
            faithfulness = metrics.get('faithfulness', 0)
            if not _is_nan(faithfulness):
                metrics_data['faithfulness']['sum'] += faithfulness
                metrics_data['faithfulness']['count'] += 1

            answer_relevance = metrics.get('answer_relevance', 0)
            if not _is_nan(answer_relevance):
                metrics_data['answer_relevance']['sum'] += answer_relevance
                metrics_data['answer_relevance']['count'] += 1

            context_recall = metrics.get('context_recall', 0)
            if not _is_nan(context_recall):
                metrics_data['context_recall']['sum'] += context_recall
                metrics_data['context_recall']['count'] += 1

            context_precision = metrics.get('context_precision', 0)
            if not _is_nan(context_precision):
                metrics_data['context_precision']['sum'] += context_precision
                metrics_data['context_precision']['count'] += 1

            ragas_score = result.get('ragas_score', 0)
            if not _is_nan(ragas_score):
                metrics_data['ragas_score']['sum'] += ragas_score
                metrics_data['ragas_score']['count'] += 1

        # Calculate averages using actual counts for each metric
        avg_metrics = {}
        for metric_name, data in metrics_data.items():
            if data['count'] > 0:
                avg_val = data['sum'] / data['count']
                avg_metrics[metric_name] = round(avg_val, 4) if not _is_nan(avg_val) else 0.0
            else:
                avg_metrics[metric_name] = 0.0

        # Find min and max RAGAS scores (filter out NaN)
        ragas_scores = []
        for r in valid_results:
            score = r.get('ragas_score', 0)
            if _is_nan(score):
                continue  # Skip NaN values
            ragas_scores.append(score)

        min_score = min(ragas_scores) if ragas_scores else 0
        max_score = max(ragas_scores) if ragas_scores else 0

        return {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'failed_tests': failed_tests,
            'success_rate': round(successful_tests / total_tests * 100, 2),
            'average_metrics': avg_metrics,
            'min_ragas_score': round(min_score, 4),
            'max_ragas_score': round(max_score, 4),
        }

    async def run(self) -> dict[str, Any]:
        """Run complete evaluation pipeline"""

        start_time = time.time()

        # Evaluate responses
        results = await self.evaluate_responses()

        elapsed_time = time.time() - start_time

        # Calculate benchmark statistics
        benchmark_stats = self._calculate_benchmark_stats(results)

        # Save results
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': len(results),
            'elapsed_time_seconds': round(elapsed_time, 2),
            'benchmark_stats': benchmark_stats,
            'results': results,
        }

        # Display results table
        self._display_results_table(results)

        # Save JSON results
        json_path = self.results_dir / f'results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)

        # Export to CSV
        csv_path = self._export_to_csv(results)

        # Print summary
        logger.info('')
        logger.info('%s', '=' * 70)
        logger.info('📊 EVALUATION COMPLETE')
        logger.info('%s', '=' * 70)
        logger.info('Total Tests:    %s', len(results))
        logger.info('Successful:     %s', benchmark_stats['successful_tests'])
        logger.info('Failed:         %s', benchmark_stats['failed_tests'])
        logger.info('Success Rate:   %.2f%%', benchmark_stats['success_rate'])
        logger.info('Elapsed Time:   %.2f seconds', elapsed_time)
        logger.info('Avg Time/Test:  %.2f seconds', elapsed_time / len(results))

        # Print benchmark metrics
        logger.info('')
        logger.info('%s', '=' * 70)
        logger.info('📈 BENCHMARK RESULTS (Average)')
        logger.info('%s', '=' * 70)
        avg = benchmark_stats['average_metrics']
        logger.info('Average Faithfulness:      %.4f', avg['faithfulness'])
        logger.info('Average Answer Relevance:  %.4f', avg['answer_relevance'])
        logger.info('Average Context Recall:    %.4f', avg['context_recall'])
        logger.info('Average Context Precision: %.4f', avg['context_precision'])
        logger.info('Average RAGAS Score:       %.4f', avg['ragas_score'])
        logger.info('%s', '-' * 70)
        logger.info(
            'Min RAGAS Score:           %.4f',
            benchmark_stats['min_ragas_score'],
        )
        logger.info(
            'Max RAGAS Score:           %.4f',
            benchmark_stats['max_ragas_score'],
        )

        logger.info('')
        logger.info('%s', '=' * 70)
        logger.info('📁 GENERATED FILES')
        logger.info('%s', '=' * 70)
        logger.info('Results Dir:    %s', self.results_dir.absolute())
        logger.info('   • CSV:  %s', csv_path.name)
        logger.info('   • JSON: %s', json_path.name)
        logger.info('%s', '=' * 70)

        return summary


# Available query modes for multi-mode comparison
QUERY_MODES = ['local', 'global', 'hybrid', 'mix', 'naive']


def generate_mode_comparison(
    all_results: dict[str, dict[str, Any]],
    results_dir: Path,
) -> Path:
    """
    Generate a comparison report showing best mode per question.

    Args:
        all_results: Dict mapping mode name to evaluation summary.
        results_dir: Directory to save comparison report.

    Returns:
        Path to the generated comparison CSV file.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    comparison_path = results_dir / f'mode_comparison_{timestamp}.csv'

    # Get the test questions from the first mode's results
    first_mode = next(iter(all_results.keys()))
    questions = [r['question'] for r in all_results[first_mode]['results']]
    num_questions = len(questions)

    with open(comparison_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header: question, best_mode, best_ragas, then each mode's score
        header = ['question', 'best_mode', 'best_ragas', *QUERY_MODES]
        writer.writerow(header)

        # For each question, find best mode
        for q_idx in range(num_questions):
            question = questions[q_idx][:50] + '...'  # Truncate for readability
            scores = {}
            for mode in QUERY_MODES:
                if mode in all_results:
                    result = all_results[mode]['results'][q_idx]
                    scores[mode] = result.get('ragas_score', 0)
                else:
                    scores[mode] = None

            # Find best mode (ignore None values)
            valid_scores = {m: s for m, s in scores.items() if s is not None}
            if valid_scores:
                best_mode, best_score = max(valid_scores.items(), key=lambda item: item[1])
            else:
                best_mode = 'N/A'
                best_score = 0

            row = [question, best_mode, f'{best_score:.4f}']
            for mode in QUERY_MODES:
                if scores[mode] is not None:
                    row.append(f'{scores[mode]:.4f}')
                else:
                    row.append('N/A')
            writer.writerow(row)

        # Add summary row with averages
        avg_row = ['AVERAGE', '', '']
        best_avg = 0
        best_avg_mode = ''
        for mode in QUERY_MODES:
            if mode in all_results:
                avg = all_results[mode]['benchmark_stats']['average_metrics']['ragas_score']
                avg_row.append(f'{avg:.4f}')
                if avg > best_avg:
                    best_avg = avg
                    best_avg_mode = mode
            else:
                avg_row.append('N/A')
        avg_row[1] = best_avg_mode
        avg_row[2] = f'{best_avg:.4f}'
        writer.writerow(avg_row)

    logger.info('')
    logger.info('%s', '=' * 70)
    logger.info('📊 MODE COMPARISON SUMMARY')
    logger.info('%s', '=' * 70)
    logger.info('Best overall mode: %s (avg RAGAS: %.4f)', best_avg_mode, best_avg)
    logger.info('')
    for mode in QUERY_MODES:
        if mode in all_results:
            avg = all_results[mode]['benchmark_stats']['average_metrics']['ragas_score']
            marker = '⭐' if mode == best_avg_mode else '  '
            logger.info('%s %-8s: %.4f', marker, mode, avg)
    logger.info('')
    logger.info('Comparison CSV: %s', comparison_path.name)
    logger.info('%s', '=' * 70)

    return comparison_path


async def main():
    """
    Main entry point for RAGAS evaluation

    Command-line arguments:
        --dataset, -d: Path to test dataset JSON file (default: sample_dataset.json)
        --ragendpoint, -r: LightRAG API endpoint URL (default: http://localhost:9621 or $LIGHTRAG_API_URL)

    Usage:
        python lightrag/evaluation/eval_rag_quality.py
        python lightrag/evaluation/eval_rag_quality.py --dataset my_test.json
        python lightrag/evaluation/eval_rag_quality.py -d my_test.json -r http://localhost:9621
    """
    try:
        # Parse command-line arguments
        parser = argparse.ArgumentParser(
            description='RAGAS Evaluation Script for LightRAG System',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Use defaults
  python lightrag/evaluation/eval_rag_quality.py

  # Specify custom dataset
  python lightrag/evaluation/eval_rag_quality.py --dataset my_test.json

  # Specify custom RAG endpoint
  python lightrag/evaluation/eval_rag_quality.py --ragendpoint http://my-server.com:9621

  # Specify both
  python lightrag/evaluation/eval_rag_quality.py -d my_test.json -r http://localhost:9621

  # Run with debug logging
  python lightrag/evaluation/eval_rag_quality.py --debug

  # Run multi-mode comparison (evaluates all modes and generates comparison)
  python lightrag/evaluation/eval_rag_quality.py --compare-modes

  # Tune retrieval parameters via environment variables
  EVAL_QUERY_TOP_K=15 EVAL_CHUNK_TOP_K=20 python lightrag/evaluation/eval_rag_quality.py

Environment Variables (for parameter tuning):
  EVAL_QUERY_TOP_K     Number of entities/relations to retrieve (default: 10)
  EVAL_CHUNK_TOP_K     Number of text chunks to retrieve (default: 10)
  EVAL_MAX_TOTAL_TOKENS  Maximum tokens for context (default: 30000)
  EVAL_ENABLE_RERANK   Enable reranking (default: true)
  LIGHTRAG_API_KEY     API key for LightRAG authentication (optional)
            """,
        )

        parser.add_argument(
            '--dataset',
            '-d',
            type=str,
            default=None,
            help='Path to test dataset JSON file (default: sample_dataset.json in evaluation directory)',
        )

        parser.add_argument(
            '--ragendpoint',
            '-r',
            type=str,
            default=None,
            help='LightRAG API endpoint URL (default: http://localhost:9621 or $LIGHTRAG_API_URL environment variable)',
        )

        parser.add_argument(
            '--mode',
            '-m',
            type=str,
            default='mix',
            choices=['local', 'global', 'hybrid', 'mix', 'naive'],
            help="Query mode for retrieval (default: mix). 'local' for entity-specific questions, 'mix' for comprehensive retrieval.",
        )

        parser.add_argument(
            '--debug',
            '-v',
            action='store_true',
            help='Enable verbose debug logging of retrieved contexts (useful for diagnosing retrieval issues)',
        )

        parser.add_argument(
            '--compare-modes',
            action='store_true',
            help='Run evaluation across ALL query modes and generate a comparison report. Ignores --mode.',
        )

        args = parser.parse_args()

        logger.info('%s', '=' * 70)
        logger.info('🔍 RAGAS Evaluation - Using Real LightRAG API')
        logger.info('%s', '=' * 70)

        if args.compare_modes:
            # Multi-mode comparison: run evaluation for each query mode
            logger.info('🔄 Running multi-mode comparison across: %s', ', '.join(QUERY_MODES))
            logger.info('%s', '=' * 70)

            all_results: dict[str, dict[str, Any]] = {}
            results_dir = None

            for mode in QUERY_MODES:
                logger.info('')
                logger.info('%s', '=' * 70)
                logger.info('📊 Evaluating mode: %s', mode.upper())
                logger.info('%s', '=' * 70)

                evaluator = RAGEvaluator(
                    test_dataset_path=args.dataset,
                    rag_api_url=args.ragendpoint,
                    query_mode=mode,
                    debug_mode=args.debug,
                )
                summary = await evaluator.run()
                all_results[mode] = summary

                # Capture the results directory from first evaluator
                if results_dir is None:
                    results_dir = evaluator.results_dir

            # Generate comparison report
            if results_dir:
                generate_mode_comparison(all_results, results_dir)
        else:
            # Single mode evaluation
            evaluator = RAGEvaluator(
                test_dataset_path=args.dataset,
                rag_api_url=args.ragendpoint,
                query_mode=args.mode,
                debug_mode=args.debug,
            )
            await evaluator.run()
    except Exception as e:
        logger.exception('❌ Error: %s', e)
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
