#!/usr/bin/env python3
"""
E2E RAGAS Test Harness for LightRAG

Complete end-to-end testing pipeline:
1. Download arXiv papers (reproducible test data)
2. Clear existing data (optional)
3. Ingest papers into LightRAG
4. Wait for processing
5. Generate Q&A dataset
6. Run RAGAS evaluation
7. Optional: A/B comparison

Usage:
    # Full E2E test
    python lightrag/evaluation/e2e_test_harness.py

    # A/B comparison (with/without orphan connections)
    python lightrag/evaluation/e2e_test_harness.py --ab-test

    # Skip download if papers exist
    python lightrag/evaluation/e2e_test_harness.py --skip-download

    # Use existing dataset
    python lightrag/evaluation/e2e_test_harness.py --dataset existing_dataset.json
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load environment variables
load_dotenv(dotenv_path='.env', override=False)

# Configuration
DEFAULT_RAG_URL = 'http://localhost:9622'
DEFAULT_PAPERS = ['2312.10997', '2404.10981', '2005.11401']
POLL_INTERVAL_SECONDS = 10
MAX_WAIT_SECONDS = 600  # 10 minutes max wait for processing


class E2ETestHarness:
    """End-to-end test harness for LightRAG RAGAS evaluation."""

    def __init__(
        self,
        rag_url: str | None = None,
        paper_ids: list[str] | None = None,
        questions_per_paper: int = 5,
        skip_download: bool = False,
        skip_ingest: bool = False,
        dataset_path: str | None = None,
        output_dir: str | None = None,
    ):
        self.rag_url = (rag_url or os.getenv('LIGHTRAG_API_URL', DEFAULT_RAG_URL)).rstrip('/')
        self.paper_ids = paper_ids or DEFAULT_PAPERS
        self.questions_per_paper = questions_per_paper
        self.skip_download = skip_download
        self.skip_ingest = skip_ingest
        self.dataset_path = Path(dataset_path) if dataset_path else None

        # Determine directories
        self.eval_dir = Path(__file__).parent
        self.papers_dir = self.eval_dir / 'papers'
        self.results_dir = Path(output_dir) if output_dir else self.eval_dir / 'results'
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # API key for LightRAG
        self.api_key = os.getenv('LIGHTRAG_API_KEY')

    async def check_lightrag_health(self) -> bool:
        """Check if LightRAG API is accessible."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f'{self.rag_url}/health')
                response.raise_for_status()
                print(f'[OK] LightRAG API accessible at {self.rag_url}')
                return True
        except Exception as e:
            print(f'[ERROR] Cannot connect to LightRAG API: {e}')
            return False

    async def download_papers(self) -> list[str]:
        """Download arXiv papers."""
        if self.skip_download:
            print('[SKIP] Paper download (--skip-download)')
            # Check existing papers
            existing = [
                str(self.papers_dir / f'{pid}.pdf')
                for pid in self.paper_ids
                if (self.papers_dir / f'{pid}.pdf').exists()
            ]
            print(f'[INFO] Found {len(existing)} existing papers')
            return existing

        print('\n' + '=' * 60)
        print('STEP 1: Download arXiv Papers')
        print('=' * 60)

        from lightrag.evaluation.download_arxiv import download_papers

        results = await download_papers(self.paper_ids, self.papers_dir)
        return [r['path'] for r in results if r['status'] in ('downloaded', 'exists')]

    async def clear_existing_data(self) -> bool:
        """Clear existing documents in LightRAG (optional)."""
        print('\n[INFO] Clearing existing data...')
        try:
            headers = {'X-API-Key': self.api_key} if self.api_key else {}
            async with httpx.AsyncClient(timeout=60.0) as client:
                # Get current documents
                response = await client.get(
                    f'{self.rag_url}/documents',
                    headers=headers,
                )
                response.raise_for_status()
                docs = response.json()

                # Clear all documents
                statuses = docs.get('statuses', {})
                all_docs = []
                for status_docs in statuses.values():
                    all_docs.extend(status_docs)

                if all_docs:
                    print(f'[INFO] Clearing {len(all_docs)} existing documents...')
                    for doc in all_docs:
                        doc_id = doc.get('id')
                        if doc_id:
                            await client.delete(
                                f'{self.rag_url}/documents/{doc_id}',
                                headers=headers,
                            )
                    print('[OK] Cleared existing documents')
                else:
                    print('[OK] No existing documents to clear')

                return True
        except Exception as e:
            print(f'[WARN] Could not clear data: {e}')
            return False

    async def ingest_papers(self, paper_paths: list[str]) -> bool:
        """Ingest papers into LightRAG."""
        if self.skip_ingest:
            print('[SKIP] Paper ingestion (--skip-ingest)')
            return True

        print('\n' + '=' * 60)
        print('STEP 2: Ingest Papers into LightRAG')
        print('=' * 60)

        headers = {'X-API-Key': self.api_key} if self.api_key else {}

        async with httpx.AsyncClient(timeout=300.0) as client:
            for paper_path in paper_paths:
                path = Path(paper_path)
                if not path.exists():
                    print(f'[WARN] Paper not found: {paper_path}')
                    continue

                print(f'[UPLOAD] {path.name}')

                try:
                    with open(path, 'rb') as f:
                        files = {'file': (path.name, f, 'application/pdf')}
                        response = await client.post(
                            f'{self.rag_url}/documents/upload',
                            files=files,
                            headers=headers,
                        )
                        response.raise_for_status()
                        result = response.json()
                        print(f'  [OK] Uploaded: {result}')
                except Exception as e:
                    print(f'  [ERROR] Upload failed: {e}')

        return True

    async def wait_for_processing(self) -> bool:
        """Wait for all documents to finish processing."""
        print('\n' + '=' * 60)
        print('STEP 3: Wait for Document Processing')
        print('=' * 60)

        headers = {'X-API-Key': self.api_key} if self.api_key else {}
        start_time = time.time()

        async with httpx.AsyncClient(timeout=30.0) as client:
            while time.time() - start_time < MAX_WAIT_SECONDS:
                try:
                    response = await client.get(
                        f'{self.rag_url}/documents',
                        headers=headers,
                    )
                    response.raise_for_status()
                    docs = response.json()

                    statuses = docs.get('statuses', {})
                    # API returns lowercase status keys
                    processing = len(statuses.get('processing', []))
                    pending = len(statuses.get('pending', []))
                    completed = len(statuses.get('processed', []))  # Note: "processed" not "completed"
                    failed = len(statuses.get('failed', []))

                    elapsed = int(time.time() - start_time)
                    print(
                        f'  [{elapsed}s] Processing: {processing}, Pending: {pending}, Completed: {completed}, Failed: {failed}'
                    )

                    if processing == 0 and pending == 0:
                        print('[OK] All documents processed')
                        return True

                except Exception as e:
                    print(f'  [WARN] Status check failed: {e}')

                await asyncio.sleep(POLL_INTERVAL_SECONDS)

        print('[ERROR] Timeout waiting for document processing')
        return False

    async def generate_dataset(self) -> Path:
        """Generate Q&A dataset from ingested papers."""
        if self.dataset_path and self.dataset_path.exists():
            print(f'[SKIP] Using existing dataset: {self.dataset_path}')
            return self.dataset_path

        print('\n' + '=' * 60)
        print('STEP 4: Generate Q&A Dataset')
        print('=' * 60)

        from lightrag.evaluation.generate_arxiv_dataset import generate_dataset

        output_path = self.eval_dir / f'arxiv_dataset_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'

        await generate_dataset(
            paper_ids=self.paper_ids,
            questions_per_paper=self.questions_per_paper,
            rag_url=self.rag_url,
            output_path=output_path,
        )

        return output_path

    async def run_ragas_evaluation(self, dataset_path: Path) -> dict:
        """Run RAGAS evaluation."""
        print('\n' + '=' * 60)
        print('STEP 5: Run RAGAS Evaluation')
        print('=' * 60)

        from lightrag.evaluation.eval_rag_quality import RAGEvaluator

        evaluator = RAGEvaluator(
            test_dataset_path=str(dataset_path),
            rag_api_url=self.rag_url,
        )

        results = await evaluator.run()
        return results

    async def run_full_pipeline(self) -> dict:
        """Run the complete E2E test pipeline."""
        print('=' * 70)
        print('E2E RAGAS TEST HARNESS FOR LIGHTRAG')
        print('=' * 70)
        print(f'RAG URL:    {self.rag_url}')
        print(f'Papers:     {", ".join(self.paper_ids)}')
        print(f'Questions:  {self.questions_per_paper} per paper')
        print(f'Results:    {self.results_dir}')
        print('=' * 70)

        start_time = time.time()

        # Check LightRAG is accessible
        if not await self.check_lightrag_health():
            return {'error': 'LightRAG API not accessible'}

        # Step 1: Download papers
        paper_paths = await self.download_papers()
        if not paper_paths:
            return {'error': 'No papers to process'}

        # Step 2: Ingest papers
        if not await self.ingest_papers(paper_paths):
            return {'error': 'Paper ingestion failed'}

        # Step 3: Wait for processing
        if not self.skip_ingest and not await self.wait_for_processing():
            return {'error': 'Document processing timeout'}

        # Step 4: Generate dataset
        dataset_path = await self.generate_dataset()

        # Step 5: Run RAGAS evaluation
        results = await self.run_ragas_evaluation(dataset_path)

        elapsed_time = time.time() - start_time

        # Save summary
        summary = {
            'pipeline_completed_at': datetime.now().isoformat(),
            'total_elapsed_seconds': round(elapsed_time, 2),
            'papers': self.paper_ids,
            'dataset_path': str(dataset_path),
            'ragas_results': results.get('benchmark_stats', {}),
        }

        summary_path = self.results_dir / f'e2e_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print('\n' + '=' * 70)
        print('E2E PIPELINE COMPLETE')
        print('=' * 70)
        print(f'Total time: {elapsed_time:.1f} seconds')
        print(f'Summary saved: {summary_path}')
        print('=' * 70)

        return summary


async def run_ab_test(
    harness_config: dict,
    clear_between_runs: bool = True,
) -> dict:
    """
    Run A/B test comparing with/without orphan connections.

    Args:
        harness_config: Configuration for E2ETestHarness
        clear_between_runs: Clear data between A and B runs

    Returns:
        A/B comparison results
    """
    print('=' * 70)
    print('A/B TEST: WITH vs WITHOUT ORPHAN CONNECTIONS')
    print('=' * 70)

    results = {}

    # Test A: WITHOUT orphan connections
    print('\n[A] Running WITHOUT orphan connections...')
    os.environ['AUTO_CONNECT_ORPHANS'] = 'false'

    harness_a = E2ETestHarness(**harness_config)
    results['without_orphans'] = await harness_a.run_full_pipeline()

    # Clear for next run
    if clear_between_runs:
        await harness_a.clear_existing_data()

    # Test B: WITH orphan connections
    print('\n[B] Running WITH orphan connections...')
    os.environ['AUTO_CONNECT_ORPHANS'] = 'true'

    # Force re-ingest for test B
    harness_config_b = harness_config.copy()
    harness_config_b['skip_download'] = True  # Papers already downloaded
    harness_config_b['skip_ingest'] = False  # Need to re-ingest

    harness_b = E2ETestHarness(**harness_config_b)
    results['with_orphans'] = await harness_b.run_full_pipeline()

    # Compare results
    print('\n' + '=' * 70)
    print('A/B COMPARISON')
    print('=' * 70)

    a_stats = results['without_orphans'].get('ragas_results', {}).get('average_metrics', {})
    b_stats = results['with_orphans'].get('ragas_results', {}).get('average_metrics', {})

    comparison = {
        'timestamp': datetime.now().isoformat(),
        'without_orphans': a_stats,
        'with_orphans': b_stats,
        'improvement': {},
    }

    for metric in ['faithfulness', 'answer_relevance', 'context_recall', 'context_precision', 'ragas_score']:
        a_val = a_stats.get(metric, 0)
        b_val = b_stats.get(metric, 0)
        diff = b_val - a_val
        pct = (diff / a_val * 100) if a_val > 0 else 0

        comparison['improvement'][metric] = {
            'absolute': round(diff, 4),
            'percent': round(pct, 2),
        }

        status = 'UP' if diff > 0 else ('DOWN' if diff < 0 else '~')
        print(f'  {metric:<20} A: {a_val:.4f}  B: {b_val:.4f}  [{status}] {pct:+.1f}%')

    # Verdict
    ragas_improvement = comparison['improvement'].get('ragas_score', {}).get('percent', 0)
    if ragas_improvement > 5:
        verdict = 'ORPHAN CONNECTIONS IMPROVE QUALITY'
    elif ragas_improvement < -5:
        verdict = 'ORPHAN CONNECTIONS DEGRADE QUALITY'
    else:
        verdict = 'NO SIGNIFICANT DIFFERENCE'

    comparison['verdict'] = verdict
    print(f'\nVERDICT: {verdict}')

    # Save comparison
    comp_path = harness_a.results_dir / f'ab_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(comp_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f'\nComparison saved: {comp_path}')

    return comparison


async def main():
    parser = argparse.ArgumentParser(
        description='E2E RAGAS Test Harness for LightRAG',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full E2E test
  python lightrag/evaluation/e2e_test_harness.py

  # A/B test (with/without orphan connections)
  python lightrag/evaluation/e2e_test_harness.py --ab-test

  # Skip paper download
  python lightrag/evaluation/e2e_test_harness.py --skip-download

  # Use existing dataset
  python lightrag/evaluation/e2e_test_harness.py --dataset arxiv_dataset.json
        """,
    )

    parser.add_argument(
        '--rag-url',
        '-r',
        type=str,
        default=None,
        help=f'LightRAG API URL (default: {DEFAULT_RAG_URL})',
    )

    parser.add_argument(
        '--papers',
        '-p',
        type=str,
        default=None,
        help='Comma-separated arXiv paper IDs',
    )

    parser.add_argument(
        '--questions',
        '-q',
        type=int,
        default=5,
        help='Questions per paper (default: 5)',
    )

    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip paper download (use existing)',
    )

    parser.add_argument(
        '--skip-ingest',
        action='store_true',
        help='Skip paper ingestion (use existing data)',
    )

    parser.add_argument(
        '--dataset',
        '-d',
        type=str,
        default=None,
        help='Path to existing Q&A dataset (skip generation)',
    )

    parser.add_argument(
        '--output-dir',
        '-o',
        type=str,
        default=None,
        help='Output directory for results',
    )

    parser.add_argument(
        '--ab-test',
        action='store_true',
        help='Run A/B test comparing with/without orphan connections',
    )

    args = parser.parse_args()

    # Parse paper IDs
    paper_ids = None
    if args.papers:
        paper_ids = [p.strip() for p in args.papers.split(',')]

    harness_config = {
        'rag_url': args.rag_url,
        'paper_ids': paper_ids,
        'questions_per_paper': args.questions,
        'skip_download': args.skip_download,
        'skip_ingest': args.skip_ingest,
        'dataset_path': args.dataset,
        'output_dir': args.output_dir,
    }

    if args.ab_test:
        await run_ab_test(harness_config)
    else:
        harness = E2ETestHarness(**harness_config)
        await harness.run_full_pipeline()


if __name__ == '__main__':
    asyncio.run(main())
