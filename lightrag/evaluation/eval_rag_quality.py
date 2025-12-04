#!/usr/bin/env python3
"""
RAGAS Evaluation Script for Portfolio RAG System

Evaluates RAG response quality using RAGAS metrics:
- Faithfulness: Is the answer factually accurate based on context?
- Answer Relevance: Is the answer relevant to the question?
- Context Recall: Is all relevant information retrieved?
- Context Precision: Is retrieved context clean without noise?

Usage:
    python lightrag/evaluation/eval_rag_quality.py
    python lightrag/evaluation/eval_rag_quality.py http://localhost:9621
    python lightrag/evaluation/eval_rag_quality.py http://your-rag-server.com:9621

Results are saved to: lightrag/evaluation/results/
    - results_YYYYMMDD_HHMMSS.csv   (CSV export for analysis)
    - results_YYYYMMDD_HHMMSS.json  (Full results with details)
"""

import asyncio
import csv
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import httpx
from dotenv import load_dotenv
from lightrag.utils import logger

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load .env from project root
project_root = Path(__file__).parent.parent.parent
load_dotenv(project_root / ".env")

# Conditional imports - will raise ImportError if dependencies not installed
try:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    Dataset = None
    evaluate = None


CONNECT_TIMEOUT_SECONDS = 180.0
READ_TIMEOUT_SECONDS = 300.0
TOTAL_TIMEOUT_SECONDS = 180.0


def _is_nan(value: Any) -> bool:
    """Return True when value is a float NaN."""
    return isinstance(value, float) and math.isnan(value)


class RAGEvaluator:
    """Evaluate RAG system quality using RAGAS metrics"""

    def __init__(self, test_dataset_path: str = None, rag_api_url: str = None):
        """
        Initialize evaluator with test dataset

        Args:
            test_dataset_path: Path to test dataset JSON file
            rag_api_url: Base URL of LightRAG API (e.g., http://localhost:9621)
                        If None, will try to read from environment or use default

        Raises:
            ImportError: If ragas or datasets packages are not installed
            ValueError: If LLM_BINDING is not set to 'openai'
            EnvironmentError: If LLM_BINDING_API_KEY is not set
        """
        # Validate RAGAS dependencies are installed
        if not RAGAS_AVAILABLE:
            raise ImportError(
                "RAGAS dependencies not installed. "
                "Install with: pip install ragas datasets"
            )

        # Validate LLM_BINDING is set to openai (required for RAGAS)
        llm_binding = os.getenv("LLM_BINDING", "").lower()
        if llm_binding != "openai":
            raise ValueError(
                f"LLM_BINDING must be set to 'openai' for RAGAS evaluation. "
                f"Current value: '{llm_binding or '(not set)'}'"
            )

        # Validate LLM_BINDING_API_KEY exists
        llm_binding_key = os.getenv("LLM_BINDING_API_KEY")
        if not llm_binding_key:
            raise EnvironmentError(
                "LLM_BINDING_API_KEY environment variable is not set. "
                "This is required for RAGAS evaluation."
            )

        # Set OPENAI_API_KEY from LLM_BINDING_API_KEY for RAGAS
        os.environ["OPENAI_API_KEY"] = llm_binding_key
        logger.info("✅ LLM_BINDING: openai")

        if test_dataset_path is None:
            test_dataset_path = Path(__file__).parent / "sample_dataset.json"

        if rag_api_url is None:
            rag_api_url = os.getenv("LIGHTRAG_API_URL", "http://localhost:9621")

        self.test_dataset_path = Path(test_dataset_path)
        self.rag_api_url = rag_api_url.rstrip("/")
        self.results_dir = Path(__file__).parent / "results"
        self.results_dir.mkdir(exist_ok=True)

        # Load test dataset
        self.test_cases = self._load_test_dataset()

    def _load_test_dataset(self) -> List[Dict[str, str]]:
        """Load test cases from JSON file"""
        if not self.test_dataset_path.exists():
            raise FileNotFoundError(f"Test dataset not found: {self.test_dataset_path}")

        with open(self.test_dataset_path) as f:
            data = json.load(f)

        return data.get("test_cases", [])

    async def generate_rag_response(
        self,
        question: str,
        client: httpx.AsyncClient,
    ) -> Dict[str, Any]:
        """
        Generate RAG response by calling LightRAG API.

        Args:
            question: The user query.
            client: Shared httpx AsyncClient for connection pooling.

        Returns:
            Dictionary with 'answer' and 'contexts' keys.
            'contexts' is a list of strings (one per retrieved document).

        Raises:
            Exception: If LightRAG API is unavailable.
        """
        try:
            payload = {
                "query": question,
                "mode": "mix",
                "include_references": True,
                "include_chunk_content": True,  # NEW: Request chunk content in references
                "response_type": "Multiple Paragraphs",
                "top_k": 10,
            }

            # Single optimized API call - gets both answer AND chunk content
            response = await client.post(
                f"{self.rag_api_url}/query",
                json=payload,
            )
            response.raise_for_status()
            result = response.json()

            answer = result.get("response", "No response generated")
            references = result.get("references", [])

            # DEBUG: Inspect the API response
            logger.debug("🔍 References Count: %s", len(references))
            if references:
                first_ref = references[0]
                logger.debug("🔍 First Reference Keys: %s", list(first_ref.keys()))
                if "content" in first_ref:
                    content_preview = first_ref["content"]
                    if isinstance(content_preview, list) and content_preview:
                        logger.debug(
                            "🔍 Content Preview (first chunk): %s...",
                            content_preview[0][:100],
                        )
                    elif isinstance(content_preview, str):
                        logger.debug("🔍 Content Preview: %s...", content_preview[:100])

            # Extract chunk content from enriched references
            # Note: content is now a list of chunks per reference (one file may have multiple chunks)
            contexts = []
            for ref in references:
                content = ref.get("content", [])
                if isinstance(content, list):
                    # Flatten the list: each chunk becomes a separate context
                    contexts.extend(content)
                elif isinstance(content, str):
                    # Backward compatibility: if content is still a string (shouldn't happen)
                    contexts.append(content)

            return {
                "answer": answer,
                "contexts": contexts,  # List of strings from actual retrieved chunks
            }

        except httpx.ConnectError as e:
            raise Exception(
                f"❌ Cannot connect to LightRAG API at {self.rag_api_url}\n"
                f"   Make sure LightRAG server is running:\n"
                f"   python -m lightrag.api.lightrag_server\n"
                f"   Error: {str(e)}"
            )
        except httpx.HTTPStatusError as e:
            raise Exception(
                f"LightRAG API error {e.response.status_code}: {e.response.text}"
            )
        except httpx.ReadTimeout as e:
            raise Exception(
                f"Request timeout after waiting for response\n"
                f"   Question: {question[:100]}...\n"
                f"   Error: {str(e)}"
            )
        except Exception as e:
            raise Exception(f"Error calling LightRAG API: {type(e).__name__}: {str(e)}")

    async def evaluate_single_case(
        self,
        idx: int,
        test_case: Dict[str, str],
        semaphore: asyncio.Semaphore,
        client: httpx.AsyncClient,
    ) -> Dict[str, Any]:
        """
        Evaluate a single test case with concurrency control

        Args:
            idx: Test case index (1-based)
            test_case: Test case dictionary with question and ground_truth
            semaphore: Semaphore to control concurrency
            client: Shared httpx AsyncClient for connection pooling

        Returns:
            Evaluation result dictionary
        """
        total_cases = len(self.test_cases)

        async with semaphore:
            question = test_case["question"]
            ground_truth = test_case["ground_truth"]

            logger.info("[%s/%s] Evaluating: %s...", idx, total_cases, question[:60])

            # Generate RAG response by calling actual LightRAG API
            rag_response = await self.generate_rag_response(
                question=question, client=client
            )

            # *** CRITICAL FIX: Use actual retrieved contexts, NOT ground_truth ***
            retrieved_contexts = rag_response["contexts"]

            # DEBUG: Print what was actually retrieved
            logger.debug("📝 Retrieved %s contexts", len(retrieved_contexts))
            if retrieved_contexts:
                logger.debug(
                    "📄 First context preview: %s...", retrieved_contexts[0][:100]
                )
            else:
                logger.warning("⚠️  No contexts retrieved!")

            # Prepare dataset for RAGAS evaluation with CORRECT contexts
            eval_dataset = Dataset.from_dict(
                {
                    "question": [question],
                    "answer": [rag_response["answer"]],
                    "contexts": [retrieved_contexts],
                    "ground_truth": [ground_truth],
                }
            )

            # Run RAGAS evaluation
            try:
                eval_results = evaluate(
                    dataset=eval_dataset,
                    metrics=[
                        faithfulness,
                        answer_relevancy,
                        context_recall,
                        context_precision,
                    ],
                )

                # Convert to DataFrame (RAGAS v0.3+ API)
                df = eval_results.to_pandas()

                # Extract scores from first row
                scores_row = df.iloc[0]

                # Extract scores (RAGAS v0.3+ uses .to_pandas())
                result = {
                    "question": question,
                    "answer": rag_response["answer"][:200] + "..."
                    if len(rag_response["answer"]) > 200
                    else rag_response["answer"],
                    "ground_truth": ground_truth[:200] + "..."
                    if len(ground_truth) > 200
                    else ground_truth,
                    "project": test_case.get("project_context", "unknown"),
                    "metrics": {
                        "faithfulness": float(scores_row.get("faithfulness", 0)),
                        "answer_relevance": float(
                            scores_row.get("answer_relevancy", 0)
                        ),
                        "context_recall": float(scores_row.get("context_recall", 0)),
                        "context_precision": float(
                            scores_row.get("context_precision", 0)
                        ),
                    },
                    "timestamp": datetime.now().isoformat(),
                }

                # Calculate RAGAS score (average of all metrics)
                metrics = result["metrics"]
                ragas_score = sum(metrics.values()) / len(metrics) if metrics else 0
                result["ragas_score"] = round(ragas_score, 4)

                logger.info("✅ Faithfulness: %.4f", metrics["faithfulness"])
                logger.info("✅ Answer Relevance: %.4f", metrics["answer_relevance"])
                logger.info("✅ Context Recall: %.4f", metrics["context_recall"])
                logger.info("✅ Context Precision: %.4f", metrics["context_precision"])
                logger.info("📊 RAGAS Score: %.4f", result["ragas_score"])

                return result

            except Exception as e:
                logger.exception("❌ Error evaluating: %s", e)
                return {
                    "question": question,
                    "error": str(e),
                    "metrics": {},
                    "ragas_score": 0,
                    "timestamp": datetime.now().isoformat(),
                }

    async def evaluate_responses(self) -> List[Dict[str, Any]]:
        """
        Evaluate all test cases in parallel and return metrics

        Returns:
            List of evaluation results with metrics
        """
        # Get MAX_ASYNC from environment (default to 4 if not set)
        max_async = int(os.getenv("MAX_ASYNC", "4"))

        logger.info("")
        logger.info("%s", "=" * 70)
        logger.info("🚀 Starting RAGAS Evaluation of Portfolio RAG System")
        logger.info("🔧 Parallel evaluations: %s", max_async)
        logger.info("%s", "=" * 70)

        # Create semaphore to limit concurrent evaluations
        semaphore = asyncio.Semaphore(max_async)

        # Create shared HTTP client with connection pooling and proper timeouts
        # Timeout: 3 minutes for connect, 5 minutes for read (LLM can be slow)
        timeout = httpx.Timeout(
            TOTAL_TIMEOUT_SECONDS,
            connect=CONNECT_TIMEOUT_SECONDS,
            read=READ_TIMEOUT_SECONDS,
        )
        limits = httpx.Limits(
            max_connections=max_async * 2,  # Allow some buffer
            max_keepalive_connections=max_async,
        )

        async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
            # Create tasks for all test cases
            tasks = [
                self.evaluate_single_case(idx, test_case, semaphore, client)
                for idx, test_case in enumerate(self.test_cases, 1)
            ]

            # Run all evaluations in parallel (limited by semaphore)
            results = await asyncio.gather(*tasks)

        return list(results)

    def _export_to_csv(self, results: List[Dict[str, Any]]) -> Path:
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
        csv_path = (
            self.results_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "test_number",
                "question",
                "project",
                "faithfulness",
                "answer_relevance",
                "context_recall",
                "context_precision",
                "ragas_score",
                "status",
                "timestamp",
            ]

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for idx, result in enumerate(results, 1):
                metrics = result.get("metrics", {})
                writer.writerow(
                    {
                        "test_number": idx,
                        "question": result.get("question", ""),
                        "project": result.get("project", "unknown"),
                        "faithfulness": f"{metrics.get('faithfulness', 0):.4f}",
                        "answer_relevance": f"{metrics.get('answer_relevance', 0):.4f}",
                        "context_recall": f"{metrics.get('context_recall', 0):.4f}",
                        "context_precision": f"{metrics.get('context_precision', 0):.4f}",
                        "ragas_score": f"{result.get('ragas_score', 0):.4f}",
                        "status": "success" if metrics else "error",
                        "timestamp": result.get("timestamp", ""),
                    }
                )

        return csv_path

    def _calculate_benchmark_stats(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate benchmark statistics from evaluation results

        Args:
            results: List of evaluation results

        Returns:
            Dictionary with benchmark statistics
        """
        # Filter out results with errors
        valid_results = [r for r in results if r.get("metrics")]
        total_tests = len(results)
        successful_tests = len(valid_results)
        failed_tests = total_tests - successful_tests

        if not valid_results:
            return {
                "total_tests": total_tests,
                "successful_tests": 0,
                "failed_tests": failed_tests,
                "success_rate": 0.0,
            }

        # Calculate averages for each metric (handling NaN values)
        metrics_sum = {
            "faithfulness": 0.0,
            "answer_relevance": 0.0,
            "context_recall": 0.0,
            "context_precision": 0.0,
            "ragas_score": 0.0,
        }

        for result in valid_results:
            metrics = result.get("metrics", {})
            # Skip NaN values when summing
            faithfulness = metrics.get("faithfulness", 0)
            if not _is_nan(faithfulness):
                metrics_sum["faithfulness"] += faithfulness

            answer_relevance = metrics.get("answer_relevance", 0)
            if not _is_nan(answer_relevance):
                metrics_sum["answer_relevance"] += answer_relevance

            context_recall = metrics.get("context_recall", 0)
            if not _is_nan(context_recall):
                metrics_sum["context_recall"] += context_recall

            context_precision = metrics.get("context_precision", 0)
            if not _is_nan(context_precision):
                metrics_sum["context_precision"] += context_precision

            ragas_score = result.get("ragas_score", 0)
            if not _is_nan(ragas_score):
                metrics_sum["ragas_score"] += ragas_score

        # Calculate averages
        n = len(valid_results)
        avg_metrics = {}
        for k, v in metrics_sum.items():
            avg_val = v / n if n > 0 else 0
            # Handle NaN in average
            avg_metrics[k] = round(avg_val, 4) if not _is_nan(avg_val) else 0.0

        # Find min and max RAGAS scores (filter out NaN)
        ragas_scores = []
        for r in valid_results:
            score = r.get("ragas_score", 0)
            if _is_nan(score):
                continue  # Skip NaN values
            ragas_scores.append(score)

        min_score = min(ragas_scores) if ragas_scores else 0
        max_score = max(ragas_scores) if ragas_scores else 0

        return {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "failed_tests": failed_tests,
            "success_rate": round(successful_tests / total_tests * 100, 2),
            "average_metrics": avg_metrics,
            "min_ragas_score": round(min_score, 4),
            "max_ragas_score": round(max_score, 4),
        }

    async def run(self) -> Dict[str, Any]:
        """Run complete evaluation pipeline"""

        start_time = time.time()

        # Evaluate responses
        results = await self.evaluate_responses()

        elapsed_time = time.time() - start_time

        # Calculate benchmark statistics
        benchmark_stats = self._calculate_benchmark_stats(results)

        # Save results
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": len(results),
            "elapsed_time_seconds": round(elapsed_time, 2),
            "benchmark_stats": benchmark_stats,
            "results": results,
        }

        # Save JSON results
        json_path = (
            self.results_dir
            / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("✅ JSON results saved to: %s", json_path)

        # Export to CSV
        csv_path = self._export_to_csv(results)
        logger.info("✅ CSV results saved to: %s", csv_path)

        # Print summary
        logger.info("")
        logger.info("%s", "=" * 70)
        logger.info("📊 EVALUATION COMPLETE")
        logger.info("%s", "=" * 70)
        logger.info("Total Tests:    %s", len(results))
        logger.info("Successful:     %s", benchmark_stats["successful_tests"])
        logger.info("Failed:         %s", benchmark_stats["failed_tests"])
        logger.info("Success Rate:   %.2f%%", benchmark_stats["success_rate"])
        logger.info("Elapsed Time:   %.2f seconds", elapsed_time)
        logger.info("Avg Time/Test:  %.2f seconds", elapsed_time / len(results))

        # Print benchmark metrics
        logger.info("")
        logger.info("%s", "=" * 70)
        logger.info("📈 BENCHMARK RESULTS (Average)")
        logger.info("%s", "=" * 70)
        avg = benchmark_stats["average_metrics"]
        logger.info("Average Faithfulness:      %.4f", avg["faithfulness"])
        logger.info("Average Answer Relevance:  %.4f", avg["answer_relevance"])
        logger.info("Average Context Recall:    %.4f", avg["context_recall"])
        logger.info("Average Context Precision: %.4f", avg["context_precision"])
        logger.info("Average RAGAS Score:       %.4f", avg["ragas_score"])
        logger.info("")
        logger.info(
            "Min RAGAS Score:           %.4f",
            benchmark_stats["min_ragas_score"],
        )
        logger.info(
            "Max RAGAS Score:           %.4f",
            benchmark_stats["max_ragas_score"],
        )

        logger.info("")
        logger.info("%s", "=" * 70)
        logger.info("📁 GENERATED FILES")
        logger.info("%s", "=" * 70)
        logger.info("Results Dir:    %s", self.results_dir.absolute())
        logger.info("   • CSV:  %s", csv_path.name)
        logger.info("   • JSON: %s", json_path.name)
        logger.info("%s", "=" * 70)

        return summary


async def main():
    """
    Main entry point for RAGAS evaluation

    Usage:
        python lightrag/evaluation/eval_rag_quality.py
        python lightrag/evaluation/eval_rag_quality.py http://localhost:9621
        python lightrag/evaluation/eval_rag_quality.py http://your-server.com:9621
    """
    try:
        # Get RAG API URL from command line or environment
        rag_api_url = None
        if len(sys.argv) > 1:
            rag_api_url = sys.argv[1]

        logger.info("")
        logger.info("%s", "=" * 70)
        logger.info("🔍 RAGAS Evaluation - Using Real LightRAG API")
        logger.info("%s", "=" * 70)
        if rag_api_url:
            logger.info("📡 RAG API URL: %s", rag_api_url)
        else:
            logger.info("📡 RAG API URL: http://localhost:9621 (default)")
        logger.info("%s", "=" * 70)

        evaluator = RAGEvaluator(rag_api_url=rag_api_url)
        await evaluator.run()
    except Exception as e:
        logger.exception("❌ Error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
