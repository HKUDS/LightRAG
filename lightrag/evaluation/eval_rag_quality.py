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
    python lightrag/evaluation/eval_rag_quality.py http://localhost:8000
    python lightrag/evaluation/eval_rag_quality.py http://your-rag-server.com:8000

Results are saved to: lightrag/evaluation/results/
    - results_YYYYMMDD_HHMMSS.csv   (CSV export for analysis)
    - results_YYYYMMDD_HHMMSS.json  (Full results with details)
"""

import json
import asyncio
import time
import csv
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List
import sys
import httpx
import os
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load .env from project root
project_root = Path(__file__).parent.parent.parent
load_dotenv(project_root / ".env")

# Setup OpenAI API key (required for RAGAS evaluation)
# Use LLM_BINDING_API_KEY if OPENAI_API_KEY is not set
if "OPENAI_API_KEY" not in os.environ:
    if "LLM_BINDING_API_KEY" in os.environ:
        os.environ["OPENAI_API_KEY"] = os.environ["LLM_BINDING_API_KEY"]
    else:
        os.environ["OPENAI_API_KEY"] = input("Enter your OpenAI API key: ")

try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
    )
    from datasets import Dataset
except ImportError as e:
    print(f"❌ RAGAS import error: {e}")
    print("   Install with: pip install ragas datasets")
    sys.exit(1)


class RAGEvaluator:
    """Evaluate RAG system quality using RAGAS metrics"""

    def __init__(self, test_dataset_path: str = None, rag_api_url: str = None):
        """
        Initialize evaluator with test dataset

        Args:
            test_dataset_path: Path to test dataset JSON file
            rag_api_url: Base URL of LightRAG API (e.g., http://localhost:8000)
                        If None, will try to read from environment or use default
        """
        if test_dataset_path is None:
            test_dataset_path = Path(__file__).parent / "test_dataset.json"

        if rag_api_url is None:
            rag_api_url = os.getenv("LIGHTRAG_API_URL", "http://localhost:8000")

        self.test_dataset_path = Path(test_dataset_path)
        self.rag_api_url = rag_api_url.rstrip("/")  # Remove trailing slash
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
    ) -> Dict[str, Any]:
        """
        Generate RAG response by calling LightRAG API.

        Args:
            question: The user query.

        Returns:
            Dictionary with 'answer' and 'contexts' keys.
            'contexts' is a list of strings (one per retrieved document).

        Raises:
            Exception: If LightRAG API is unavailable.
        """
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                payload = {
                    "query": question,
                    "mode": "mix",
                    "include_references": True,
                    "response_type": "Multiple Paragraphs",
                    "top_k": 10,
                }

                response = await client.post(
                    f"{self.rag_api_url}/query",
                    json=payload,
                )
                response.raise_for_status()  # Better error handling
                result = response.json()

                # Extract text content from each reference document
                references = result.get("references", [])
                contexts = [
                    ref.get("text", "") for ref in references if ref.get("text")
                ]

                return {
                    "answer": result.get("response", "No response generated"),
                    "contexts": contexts,  # List of strings, not JSON dump
                }

        except httpx.ConnectError:
            raise Exception(
                f"❌ Cannot connect to LightRAG API at {self.rag_api_url}\n"
                f"   Make sure LightRAG server is running:\n"
                f"   python -m lightrag.api.lightrag_server"
            )
        except httpx.HTTPStatusError as e:
            raise Exception(
                f"LightRAG API error {e.response.status_code}: {e.response.text}"
            )
        except Exception as e:
            raise Exception(f"Error calling LightRAG API: {str(e)}")

    async def evaluate_responses(self) -> List[Dict[str, Any]]:
        """
        Evaluate all test cases and return metrics

        Returns:
            List of evaluation results with metrics
        """
        print("\n" + "=" * 70)
        print("🚀 Starting RAGAS Evaluation of Portfolio RAG System")
        print("=" * 70 + "\n")

        results = []

        for idx, test_case in enumerate(self.test_cases, 1):
            question = test_case["question"]
            ground_truth = test_case["ground_truth"]

            print(f"[{idx}/{len(self.test_cases)}] Evaluating: {question[:60]}...")

            # Generate RAG response by calling actual LightRAG API
            rag_response = await self.generate_rag_response(question=question)

            # *** CRITICAL FIX: Use actual retrieved contexts, NOT ground_truth ***
            retrieved_contexts = rag_response["contexts"]

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

                results.append(result)

                # Print metrics
                print(f"   ✅ Faithfulness:      {metrics['faithfulness']:.4f}")
                print(f"   ✅ Answer Relevance:  {metrics['answer_relevance']:.4f}")
                print(f"   ✅ Context Recall:    {metrics['context_recall']:.4f}")
                print(f"   ✅ Context Precision: {metrics['context_precision']:.4f}")
                print(f"   📊 RAGAS Score:       {result['ragas_score']:.4f}\n")

            except Exception as e:
                import traceback
                print(f"   ❌ Error evaluating: {str(e)}")
                print(f"   🔍 Full traceback:\n{traceback.format_exc()}\n")
                result = {
                    "question": question,
                    "error": str(e),
                    "metrics": {},
                    "ragas_score": 0,
                    "timestamp": datetime.now().isoformat()
                }
                results.append(result)

        return results

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
        csv_path = self.results_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

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
                writer.writerow({
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
                })

        return csv_path

    async def run(self) -> Dict[str, Any]:
        """Run complete evaluation pipeline"""

        start_time = time.time()

        # Evaluate responses
        results = await self.evaluate_responses()

        elapsed_time = time.time() - start_time

        # Save results
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": len(results),
            "elapsed_time_seconds": round(elapsed_time, 2),
            "results": results
        }

        # Save JSON results
        json_path = self.results_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"✅ JSON results saved to: {json_path}")

        # Export to CSV
        csv_path = self._export_to_csv(results)
        print(f"✅ CSV results saved to: {csv_path}")

        # Print summary
        print("\n" + "="*70)
        print("📊 EVALUATION COMPLETE")
        print("="*70)
        print(f"Total Tests:    {len(results)}")
        print(f"Elapsed Time:   {elapsed_time:.2f} seconds")
        print(f"Results Dir:    {self.results_dir.absolute()}")
        print("\n📁 Generated Files:")
        print(f"   • CSV:  {csv_path.name}")
        print(f"   • JSON: {json_path.name}")
        print("="*70 + "\n")

        return summary


async def main():
    """
    Main entry point for RAGAS evaluation

    Usage:
        python lightrag/evaluation/eval_rag_quality.py
        python lightrag/evaluation/eval_rag_quality.py http://localhost:8000
        python lightrag/evaluation/eval_rag_quality.py http://your-server.com:8000
    """
    try:
        # Get RAG API URL from command line or environment
        rag_api_url = None
        if len(sys.argv) > 1:
            rag_api_url = sys.argv[1]

        print("\n" + "="*70)
        print("🔍 RAGAS Evaluation - Using Real LightRAG API")
        print("="*70)
        if rag_api_url:
            print(f"📡 RAG API URL: {rag_api_url}")
        else:
            print(f"📡 RAG API URL: http://localhost:8000 (default)")
        print("="*70 + "\n")

        evaluator = RAGEvaluator(rag_api_url=rag_api_url)
        await evaluator.run()
    except Exception as e:
        print(f"\n❌ Error: {str(e)}\n")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
