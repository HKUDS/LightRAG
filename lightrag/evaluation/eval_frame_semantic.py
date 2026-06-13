#!/usr/bin/env python3
"""
Đánh giá chất lượng trích xuất ngữ nghĩa frame-semantic-transformer.

Bao gồm hai phần:
  A) Đánh giá nội bộ (không cần server) — kiểm tra chất lượng frame detection,
     entity/relation extraction trực tiếp trên văn bản mẫu.
  B) Đánh giá end-to-end RAGAS (cần LightRAG API server đang chạy) — giống
     eval_rag_quality.py nhưng ghi lại thêm thống kê frame-specific và có thể
     so sánh với kết quả LLM-based trước đó.

Cách chạy:
    # Chỉ đánh giá nội bộ frame extraction (không cần server):
    python lightrag/evaluation/eval_frame_semantic.py --internal-only

    # Đánh giá end-to-end (cần server đang chạy):
    python lightrag/evaluation/eval_frame_semantic.py

    # So sánh với kết quả LLM cũ (truyền file JSON từ eval_rag_quality.py):
    python lightrag/evaluation/eval_frame_semantic.py --compare path/to/llm_results.json

    # Dùng dataset riêng:
    python lightrag/evaluation/eval_frame_semantic.py -d my_dataset.json

Kết quả lưu tại: lightrag/evaluation/results/frame_semantic_YYYYMMDD_HHMMSS.json
"""

from __future__ import annotations

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
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import httpx
from dotenv import load_dotenv

warnings.filterwarnings("ignore", message=".*LangchainLLMWrapper is deprecated.*")
warnings.filterwarnings("ignore", message=".*Unexpected type for token usage.*")

load_dotenv(dotenv_path=".env", override=False)

# ──────────────────────────────────────────────────────────────────────────────
# Imports có điều kiện
# ──────────────────────────────────────────────────────────────────────────────

try:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
    from ragas.llms import LangchainLLMWrapper
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

try:
    from frame_semantic_transformer import FrameSemanticTransformer
    FRAME_TRANSFORMER_AVAILABLE = True
except Exception:
    FRAME_TRANSFORMER_AVAILABLE = False

try:
    from tqdm.auto import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from lightrag.utils import logger

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = EVAL_DIR / "sample_dataset.json"
DEFAULT_DOCS_DIR = EVAL_DIR / "sample_documents"
RESULTS_DIR = EVAL_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CONNECT_TIMEOUT = 180.0
READ_TIMEOUT = 300.0


# ──────────────────────────────────────────────────────────────────────────────
# Phần A: Đánh giá nội bộ frame extraction
# ──────────────────────────────────────────────────────────────────────────────

# Văn bản mẫu cố định để kiểm tra ổn định của frame detection
_SAMPLE_TEXTS = [
    {
        "id": "commerce_1",
        "text": "Apple CEO Tim Cook announced the new iPhone 16 at the company's annual event in Cupertino.",
        "expected_frames": ["Statement", "Announcing"],
        "expected_entities": ["Tim Cook", "iPhone 16", "Cupertino"],
        "description": "Câu thông báo sự kiện — kỳ vọng frame Statement/Announcing",
    },
    {
        "id": "commerce_2",
        "text": "The woman sold the car to the man for five thousand dollars.",
        "expected_frames": ["Commerce_sell"],
        "expected_entities": ["The woman", "the car", "the man"],
        "description": "Câu giao dịch thương mại — kỳ vọng frame Commerce_sell",
    },
    {
        "id": "motion_1",
        "text": "The athlete ran from the stadium to the finish line at record speed.",
        "expected_frames": ["Self_motion", "Motion"],
        "expected_entities": ["The athlete", "the stadium", "the finish line"],
        "description": "Câu chuyển động — kỳ vọng frame Self_motion/Motion",
    },
    {
        "id": "cause_effect",
        "text": "The heavy rain caused severe flooding in the coastal villages.",
        "expected_frames": ["Causation"],
        "expected_entities": ["The heavy rain", "severe flooding", "the coastal villages"],
        "description": "Câu nhân quả — kỳ vọng frame Causation",
    },
    {
        "id": "cognition_1",
        "text": "Scientists believe that climate change will significantly impact global agriculture.",
        "expected_frames": ["Cogitation", "Opinion"],
        "expected_entities": ["Scientists", "climate change", "global agriculture"],
        "description": "Câu nhận thức/ý kiến — kỳ vọng frame Cogitation/Opinion",
    },
]


class InternalFrameEvaluator:
    """Đánh giá chất lượng frame detection không cần LightRAG server."""

    def __init__(self):
        if not FRAME_TRANSFORMER_AVAILABLE:
            raise ImportError(
                "frame-semantic-transformer chưa được cài.\n"
                "Cài đặt: pip install frame-semantic-transformer"
            )
        logger.info("Đang load FrameSemanticTransformer model...")
        t0 = time.time()
        self.transformer = FrameSemanticTransformer()
        logger.info("Model loaded trong %.1f giây.", time.time() - t0)

    def evaluate_single_text(self, sample: dict) -> dict:
        """Chạy frame detection trên một văn bản mẫu, trả về kết quả chi tiết."""
        text = sample["text"]
        t0 = time.time()
        result = self.transformer.detect_frames(text)
        elapsed = time.time() - t0

        detected_frames = [f.name for f in result.frames]
        detected_entities = list({
            fe.text.strip()
            for f in result.frames
            for fe in f.frame_elements
            if fe.text.strip()
        })
        detected_fe_roles = list({
            fe.name
            for f in result.frames
            for fe in f.frame_elements
        })

        # Tính precision/recall đơn giản dựa trên danh sách kỳ vọng
        expected_frames = set(sample.get("expected_frames", []))
        expected_entities_lower = {e.lower() for e in sample.get("expected_entities", [])}
        detected_frames_set = set(detected_frames)
        detected_entities_lower = {e.lower() for e in detected_entities}

        frame_hits = len(expected_frames & detected_frames_set)
        frame_recall = frame_hits / len(expected_frames) if expected_frames else 1.0
        frame_precision = frame_hits / len(detected_frames_set) if detected_frames_set else 0.0

        entity_hits = sum(
            1 for ee in expected_entities_lower
            if any(ee in de or de in ee for de in detected_entities_lower)
        )
        entity_recall = entity_hits / len(expected_entities_lower) if expected_entities_lower else 1.0

        total_frame_elements = sum(len(f.frame_elements) for f in result.frames)

        return {
            "id": sample["id"],
            "description": sample["description"],
            "text": text,
            "elapsed_seconds": round(elapsed, 3),
            "detected_frames": detected_frames,
            "detected_entities": detected_entities,
            "detected_fe_roles": detected_fe_roles,
            "total_frames": len(detected_frames),
            "total_entities": len(detected_entities),
            "total_frame_elements": total_frame_elements,
            "frame_recall": round(frame_recall, 4),
            "frame_precision": round(frame_precision, 4),
            "entity_recall": round(entity_recall, 4),
            "expected_frames": list(expected_frames),
            "expected_entities": sample.get("expected_entities", []),
        }

    def run(self, extra_texts: list[dict] | None = None) -> dict:
        """Chạy toàn bộ đánh giá nội bộ, trả về báo cáo tổng hợp."""
        samples = _SAMPLE_TEXTS + (extra_texts or [])
        logger.info("=" * 65)
        logger.info("PHAN A: DANH GIA NOI BO FRAME EXTRACTION (%d mau)", len(samples))
        logger.info("=" * 65)

        results = []
        for sample in samples:
            logger.info("  [%s] %s", sample["id"], sample["description"])
            r = self.evaluate_single_text(sample)
            results.append(r)
            logger.info(
                "    Frames: %s | Entities: %d | FEs: %d | %.3fs",
                r["detected_frames"],
                r["total_entities"],
                r["total_frame_elements"],
                r["elapsed_seconds"],
            )
            logger.info(
                "    Frame recall=%.2f  precision=%.2f  entity recall=%.2f",
                r["frame_recall"],
                r["frame_precision"],
                r["entity_recall"],
            )

        # Tổng hợp
        avg_frame_recall = sum(r["frame_recall"] for r in results) / len(results)
        avg_frame_precision = sum(r["frame_precision"] for r in results) / len(results)
        avg_entity_recall = sum(r["entity_recall"] for r in results) / len(results)
        avg_elapsed = sum(r["elapsed_seconds"] for r in results) / len(results)
        total_frames = sum(r["total_frames"] for r in results)
        total_entities = sum(r["total_entities"] for r in results)

        summary = {
            "total_samples": len(results),
            "avg_frame_recall": round(avg_frame_recall, 4),
            "avg_frame_precision": round(avg_frame_precision, 4),
            "avg_entity_recall": round(avg_entity_recall, 4),
            "avg_elapsed_seconds": round(avg_elapsed, 3),
            "total_frames_detected": total_frames,
            "total_entities_detected": total_entities,
        }

        logger.info("")
        logger.info("TOM TAT PHAN A:")
        logger.info("  Avg Frame Recall:    %.4f", avg_frame_recall)
        logger.info("  Avg Frame Precision: %.4f", avg_frame_precision)
        logger.info("  Avg Entity Recall:   %.4f", avg_entity_recall)
        logger.info("  Avg Time/Sample:     %.3f giay", avg_elapsed)
        logger.info("  Tong frames phat hien: %d", total_frames)
        logger.info("  Tong entities phat hien: %d", total_entities)
        logger.info("=" * 65)

        return {"summary": summary, "details": results}


# ──────────────────────────────────────────────────────────────────────────────
# Phần B: Đánh giá end-to-end RAGAS (frame-semantic pipeline)
# ──────────────────────────────────────────────────────────────────────────────

def _is_nan(v: Any) -> bool:
    return isinstance(v, float) and math.isnan(v)


class FrameSemanticRAGEvaluator:
    """
    Đánh giá chất lượng RAG end-to-end với frame-semantic extraction,
    dùng RAGAS metrics (Faithfulness, AnswerRelevance, ContextRecall, ContextPrecision).
    """

    def __init__(self, test_dataset_path: str | None = None, rag_api_url: str | None = None):
        if not RAGAS_AVAILABLE:
            raise ImportError(
                "RAGAS chưa được cài. Cài đặt: pip install ragas datasets langchain-openai"
            )

        # Cấu hình LLM và embedding cho RAGAS scoring
        api_key = (
            os.getenv("EVAL_LLM_BINDING_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        if not api_key:
            raise EnvironmentError(
                "Cần EVAL_LLM_BINDING_API_KEY hoặc OPENAI_API_KEY để chạy RAGAS."
            )

        model = os.getenv("EVAL_LLM_MODEL", "gpt-4o-mini")
        base_url = os.getenv("EVAL_LLM_BINDING_HOST")
        emb_model = os.getenv("EVAL_EMBEDDING_MODEL", "text-embedding-3-large")
        emb_key = (
            os.getenv("EVAL_EMBEDDING_BINDING_API_KEY")
            or api_key
        )
        emb_base_url = os.getenv("EVAL_EMBEDDING_BINDING_HOST") or base_url

        llm_kwargs: dict = {"model": model, "api_key": api_key, "max_retries": 5}
        if base_url:
            llm_kwargs["base_url"] = base_url
        emb_kwargs: dict = {"model": emb_model, "api_key": emb_key}
        if emb_base_url:
            emb_kwargs["base_url"] = emb_base_url

        base_llm = ChatOpenAI(**llm_kwargs)
        try:
            self.eval_llm = LangchainLLMWrapper(langchain_llm=base_llm, bypass_n=True)
        except Exception:
            self.eval_llm = base_llm
        self.eval_embeddings = OpenAIEmbeddings(**emb_kwargs)

        if test_dataset_path is None:
            test_dataset_path = DEFAULT_DATASET
        if rag_api_url is None:
            rag_api_url = os.getenv("LIGHTRAG_API_URL", "http://localhost:9621")

        self.dataset_path = Path(test_dataset_path)
        self.api_url = rag_api_url.rstrip("/")
        self.test_cases = self._load_dataset()

        logger.info("RAGAS endpoint: %s  |  dataset: %s  |  %d cases",
                    self.api_url, self.dataset_path.name, len(self.test_cases))

    def _load_dataset(self) -> list[dict]:
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset không tồn tại: {self.dataset_path}")
        with open(self.dataset_path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("test_cases", [])

    async def _query_api(self, question: str, client: httpx.AsyncClient) -> dict:
        """Gọi LightRAG API và lấy answer + retrieved contexts."""
        payload = {
            "query": question,
            "mode": os.getenv("EVAL_QUERY_MODE", "hybrid"),
            "include_references": True,
            "include_chunk_content": True,
            "response_type": "Multiple Paragraphs",
            "top_k": int(os.getenv("EVAL_QUERY_TOP_K", "10")),
        }
        headers = {}
        api_key = os.getenv("LIGHTRAG_API_KEY")
        if api_key:
            headers["X-API-Key"] = api_key

        resp = await client.post(f"{self.api_url}/query", json=payload, headers=headers)
        resp.raise_for_status()
        result = resp.json()

        answer = result.get("response", "")
        contexts: list[str] = []
        for ref in result.get("references", []):
            content = ref.get("content", [])
            if isinstance(content, list):
                contexts.extend(content)
            elif isinstance(content, str):
                contexts.append(content)

        return {"answer": answer, "contexts": contexts}

    async def _eval_one(
        self,
        idx: int,
        case: dict,
        semaphore: asyncio.Semaphore,
        client: httpx.AsyncClient,
    ) -> dict:
        async with semaphore:
            question = case["question"]
            ground_truth = case["ground_truth"]

            try:
                rag = await self._query_api(question, client)
            except Exception as exc:
                logger.error("Loi query [%d]: %s", idx, exc)
                return {
                    "test_number": idx,
                    "question": question,
                    "error": str(exc),
                    "metrics": {},
                    "ragas_score": 0.0,
                    "timestamp": datetime.now().isoformat(),
                }

            eval_ds = Dataset.from_dict({
                "question": [question],
                "answer": [rag["answer"]],
                "contexts": [rag["contexts"]],
                "ground_truth": [ground_truth],
            })

            try:
                eval_result = evaluate(
                    dataset=eval_ds,
                    metrics=[
                        Faithfulness(),
                        AnswerRelevancy(),
                        ContextRecall(),
                        ContextPrecision(),
                    ],
                    llm=self.eval_llm,
                    embeddings=self.eval_embeddings,
                )
                df = eval_result.to_pandas()
                row = df.iloc[0]
                metrics = {
                    "faithfulness": float(row.get("faithfulness", 0)),
                    "answer_relevance": float(row.get("answer_relevancy", 0)),
                    "context_recall": float(row.get("context_recall", 0)),
                    "context_precision": float(row.get("context_precision", 0)),
                }
                valid = [v for v in metrics.values() if not _is_nan(v)]
                ragas_score = sum(valid) / len(valid) if valid else 0.0
                return {
                    "test_number": idx,
                    "question": question,
                    "answer": rag["answer"][:300],
                    "ground_truth": ground_truth[:300],
                    "project": case.get("project", ""),
                    "contexts_count": len(rag["contexts"]),
                    "metrics": metrics,
                    "ragas_score": round(ragas_score, 4),
                    "timestamp": datetime.now().isoformat(),
                }
            except Exception as exc:
                logger.error("Loi RAGAS [%d]: %s", idx, exc)
                return {
                    "test_number": idx,
                    "question": question,
                    "error": str(exc),
                    "metrics": {},
                    "ragas_score": 0.0,
                    "timestamp": datetime.now().isoformat(),
                }

    async def run(self) -> dict:
        """Chạy đánh giá RAGAS cho toàn bộ test cases."""
        max_concurrent = int(os.getenv("EVAL_MAX_CONCURRENT", "2"))
        semaphore = asyncio.Semaphore(max_concurrent)
        timeout = httpx.Timeout(CONNECT_TIMEOUT, read=READ_TIMEOUT)
        limits = httpx.Limits(
            max_connections=(max_concurrent + 1) * 2,
            max_keepalive_connections=max_concurrent + 1,
        )

        logger.info("=" * 65)
        logger.info("PHAN B: RAGAS END-TO-END (%d cases, mode=%s)",
                    len(self.test_cases), os.getenv("EVAL_QUERY_MODE", "hybrid"))
        logger.info("=" * 65)

        t0 = time.time()
        async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
            tasks = [
                self._eval_one(idx, case, semaphore, client)
                for idx, case in enumerate(self.test_cases, 1)
            ]
            results = list(await asyncio.gather(*tasks))
        elapsed = time.time() - t0

        return self._summarize(results, elapsed)

    def _summarize(self, results: list[dict], elapsed: float) -> dict:
        valid = [r for r in results if r.get("metrics")]
        metric_names = ["faithfulness", "answer_relevance", "context_recall", "context_precision"]
        avg: dict[str, float] = {}
        for m in metric_names:
            vals = [r["metrics"].get(m, 0) for r in valid if not _is_nan(r["metrics"].get(m, 0))]
            avg[m] = round(sum(vals) / len(vals), 4) if vals else 0.0
        scores = [r["ragas_score"] for r in valid if not _is_nan(r["ragas_score"])]
        avg["ragas_score"] = round(sum(scores) / len(scores), 4) if scores else 0.0

        logger.info("")
        logger.info("TOM TAT PHAN B (RAGAS):")
        logger.info("  Faithfulness:      %.4f", avg["faithfulness"])
        logger.info("  Answer Relevance:  %.4f", avg["answer_relevance"])
        logger.info("  Context Recall:    %.4f", avg["context_recall"])
        logger.info("  Context Precision: %.4f", avg["context_precision"])
        logger.info("  RAGAS Score:       %.4f", avg["ragas_score"])
        logger.info("  Thoi gian:         %.1f giay  (%d/%d thanh cong)",
                    elapsed, len(valid), len(results))
        logger.info("=" * 65)

        return {
            "elapsed_seconds": round(elapsed, 2),
            "total_tests": len(results),
            "successful_tests": len(valid),
            "failed_tests": len(results) - len(valid),
            "average_metrics": avg,
            "results": results,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Đánh giá trích xuất keyword cho 3 chế độ (không cần server)
# ──────────────────────────────────────────────────────────────────────────────

class ThreeModeKeywordEvaluator:
    """
    Đánh giá chất lượng keyword extraction cho 3 chế độ:
      - none    : LLM cho tất cả (baseline)
      - hl_only : Frame-semantic cho high-level, LLM cho low-level
      - full    : Frame-semantic cho cả hai

    Phần này chỉ kiểm tra keyword extraction (không cần LightRAG server).
    Chỉ frame-semantic modes có thể chạy mà không cần LLM API key.
    """

    _QUERY_SAMPLES = [
        {
            "id": "trade_query",
            "text": "How does international trade influence global economic stability?",
            "expected_hl": ["International trade", "Global economic stability"],
            "expected_ll": ["Trade agreements", "Tariffs", "Currency exchange"],
        },
        {
            "id": "deforestation_query",
            "text": "What are the environmental consequences of deforestation on biodiversity?",
            "expected_hl": ["Environmental consequences", "Deforestation", "Biodiversity loss"],
            "expected_ll": ["Species extinction", "Habitat destruction"],
        },
        {
            "id": "education_query",
            "text": "What is the role of education in reducing poverty?",
            "expected_hl": ["Education", "Poverty reduction"],
            "expected_ll": ["School access", "Literacy rates", "Income inequality"],
        },
    ]

    def __init__(self, modes: list[str] | None = None):
        self.modes = modes or ["full", "hl_only"]  # "none" needs LLM API key
        if "full" in self.modes or "hl_only" in self.modes:
            if not FRAME_TRANSFORMER_AVAILABLE:
                raise ImportError("frame-semantic-transformer chưa được cài.")
            logger.info("Đang load FrameSemanticTransformer...")
            t0 = time.time()
            from frame_semantic_transformer import FrameSemanticTransformer
            self._fst = FrameSemanticTransformer()
            logger.info("Model loaded trong %.1f giây.", time.time() - t0)

    def _extract_with_frame(self, text: str) -> tuple[list[str], list[str]]:
        result = self._fst.detect_frames(text)
        hl = list({f.name for f in result.frames if f.name})
        ll = list({fe.text.strip() for f in result.frames for fe in f.frame_elements if fe.text.strip()})
        return hl, ll

    def _recall(self, expected: list[str], detected: list[str]) -> float:
        exp_lower = {e.lower() for e in expected}
        det_lower = {d.lower() for d in detected}
        if not exp_lower:
            return 1.0
        hits = sum(1 for e in exp_lower if any(e in d or d in e for d in det_lower))
        return hits / len(exp_lower)

    def evaluate_mode(self, mode: str) -> dict:
        """Đánh giá keyword extraction cho một chế độ trên tập mẫu."""
        results = []
        for sample in self._QUERY_SAMPLES:
            t0 = time.time()
            if mode in ("full", "hl_only"):
                hl, ll = self._extract_with_frame(sample["text"])
            else:
                # mode "none" requires LLM — skip with empty result
                hl, ll = [], []
                logger.warning("Mode 'none' cần LLM API key — bỏ qua mẫu %s", sample["id"])

            elapsed = time.time() - t0
            hl_recall = self._recall(sample["expected_hl"], hl)
            ll_recall = self._recall(sample["expected_ll"], ll)

            results.append({
                "id": sample["id"],
                "text": sample["text"],
                "mode": mode,
                "hl_keywords": hl,
                "ll_keywords": ll,
                "hl_recall": round(hl_recall, 4),
                "ll_recall": round(ll_recall, 4),
                "elapsed_seconds": round(elapsed, 3),
            })

        avg_hl_recall = sum(r["hl_recall"] for r in results) / len(results)
        avg_ll_recall = sum(r["ll_recall"] for r in results) / len(results)
        avg_elapsed = sum(r["elapsed_seconds"] for r in results) / len(results)

        return {
            "mode": mode,
            "avg_hl_recall": round(avg_hl_recall, 4),
            "avg_ll_recall": round(avg_ll_recall, 4),
            "avg_elapsed_seconds": round(avg_elapsed, 3),
            "details": results,
        }

    def run(self) -> dict:
        """Chạy đánh giá cho tất cả các chế độ và in bảng tổng hợp."""
        mode_results = {}
        logger.info("=" * 70)
        logger.info("DANH GIA KEYWORD EXTRACTION — 3 CHE DO")
        logger.info("=" * 70)

        for mode in self.modes:
            logger.info("\n[CHE DO: %s]", mode.upper())
            r = self.evaluate_mode(mode)
            mode_results[mode] = r
            logger.info("  HL Recall: %.4f  |  LL Recall: %.4f  |  Toc do: %.3fs",
                        r["avg_hl_recall"], r["avg_ll_recall"], r["avg_elapsed_seconds"])

        logger.info("")
        logger.info("%-12s  %12s  %12s  %12s", "Che do", "HL Recall", "LL Recall", "Toc do (s)")
        logger.info("-" * 55)
        for mode, r in mode_results.items():
            logger.info("%-12s  %12.4f  %12.4f  %12.3f",
                        mode, r["avg_hl_recall"], r["avg_ll_recall"], r["avg_elapsed_seconds"])
        logger.info("=" * 70)

        return {
            "timestamp": datetime.now().isoformat(),
            "modes_evaluated": self.modes,
            "results_by_mode": mode_results,
        }


# ──────────────────────────────────────────────────────────────────────────────
# So sánh kết quả 3 chế độ RAGAS end-to-end
# ──────────────────────────────────────────────────────────────────────────────

def compare_three_modes(results_by_mode: dict[str, dict]) -> dict:
    """
    So sánh kết quả RAGAS của 3 chế độ extraction.

    Args:
        results_by_mode: {"none": ragas_summary, "hl_only": ragas_summary, "full": ragas_summary}
            Mỗi ragas_summary là output của FrameSemanticRAGEvaluator.run()

    Returns:
        Dict chứa bảng so sánh đầy đủ
    """
    metric_names = ["faithfulness", "answer_relevance", "context_recall", "context_precision", "ragas_score"]
    mode_avgs: dict[str, dict] = {}
    for mode, summary in results_by_mode.items():
        mode_avgs[mode] = summary.get("average_metrics", {})

    comparison: dict[str, dict] = {}
    for m in metric_names:
        vals = {mode: mode_avgs[mode].get(m, 0.0) for mode in results_by_mode}
        best_mode = max(vals, key=lambda k: vals[k])
        comparison[m] = {
            **{mode: round(v, 4) for mode, v in vals.items()},
            "best_mode": best_mode,
        }

    # In bảng so sánh
    logger.info("")
    logger.info("=" * 75)
    logger.info("SO SANH 3 CHE DO RAGAS")
    logger.info("=" * 75)
    modes = list(results_by_mode.keys())
    header = f"{'Metric':<22}  " + "  ".join(f"{m:>12}" for m in modes) + "  Best"
    logger.info(header)
    logger.info("-" * 75)
    for metric, c in comparison.items():
        row = f"{metric:<22}  " + "  ".join(f"{c.get(m, 0):>12.4f}" for m in modes)
        row += f"  {c['best_mode']}"
        logger.info(row)
    logger.info("=" * 75)

    # Đếm số chỉ tiêu mỗi chế độ thắng
    wins = {m: 0 for m in modes}
    for c in comparison.values():
        wins[c["best_mode"]] += 1
    logger.info("So chi tieu thang: %s", " | ".join(f"{m}={w}" for m, w in wins.items()))

    return {
        "metrics_comparison": comparison,
        "wins_per_mode": wins,
        "modes": modes,
    }


# ──────────────────────────────────────────────────────────────────────────────
# So sánh kết quả frame-semantic vs LLM-based (binary, giữ nguyên cho backward compat)
# ──────────────────────────────────────────────────────────────────────────────

def compare_results(frame_summary: dict, llm_results_path: str) -> dict:
    """
    So sánh kết quả RAGAS của frame-semantic extraction với kết quả LLM-based
    được lưu trước đó (từ eval_rag_quality.py).

    Args:
        frame_summary: Kết quả RAGAS của frame-semantic (phần B)
        llm_results_path: Đường dẫn file JSON kết quả của eval_rag_quality.py

    Returns:
        Dict chứa bảng so sánh
    """
    llm_path = Path(llm_results_path)
    if not llm_path.exists():
        raise FileNotFoundError(f"File kết quả LLM không tồn tại: {llm_path}")

    with open(llm_path, encoding="utf-8") as f:
        llm_data = json.load(f)

    # Lấy average_metrics từ file LLM (eval_rag_quality.py lưu trong benchmark_stats)
    llm_avg = (
        llm_data.get("benchmark_stats", {}).get("average_metrics", {})
        or llm_data.get("average_metrics", {})
    )
    frame_avg = frame_summary.get("average_metrics", {})

    metric_names = ["faithfulness", "answer_relevance", "context_recall", "context_precision", "ragas_score"]
    comparison: dict[str, dict] = {}
    for m in metric_names:
        llm_val = llm_avg.get(m, 0.0)
        frame_val = frame_avg.get(m, 0.0)
        delta = frame_val - llm_val
        comparison[m] = {
            "llm_based": llm_val,
            "frame_semantic": frame_val,
            "delta": round(delta, 4),
            "change_pct": round(delta / llm_val * 100, 2) if llm_val != 0 else 0.0,
            "winner": "frame_semantic" if delta > 0 else ("llm_based" if delta < 0 else "tie"),
        }

    logger.info("")
    logger.info("=" * 65)
    logger.info("SO SANH: FRAME-SEMANTIC vs LLM-BASED")
    logger.info("%-22s  %10s  %10s  %8s  %s",
                "Metric", "LLM-based", "Frame-Sem", "Delta", "Thang")
    logger.info("-" * 65)
    for m, c in comparison.items():
        winner_str = "Frame +" if c["winner"] == "frame_semantic" else (
            "LLM  +" if c["winner"] == "llm_based" else "Hoa  "
        )
        logger.info("%-22s  %10.4f  %10.4f  %+8.4f  %s",
                    m, c["llm_based"], c["frame_semantic"], c["delta"], winner_str)
    logger.info("=" * 65)

    frame_wins = sum(1 for c in comparison.values() if c["winner"] == "frame_semantic")
    llm_wins = sum(1 for c in comparison.values() if c["winner"] == "llm_based")
    logger.info("Ket qua: Frame-semantic thang %d/%d chi tieu | LLM thang %d/%d",
                frame_wins, len(comparison), llm_wins, len(comparison))

    return {
        "comparison": comparison,
        "frame_wins": frame_wins,
        "llm_wins": llm_wins,
        "ties": len(comparison) - frame_wins - llm_wins,
        "llm_results_file": str(llm_path),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Lưu kết quả
# ──────────────────────────────────────────────────────────────────────────────

def _save_results(payload: dict, prefix: str = "frame_semantic") -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"{prefix}_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info("Ket qua da luu: %s", path)
    return path


def _save_csv(ragas_results: list[dict], prefix: str = "frame_semantic") -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"{prefix}_{ts}.csv"
    fieldnames = [
        "test_number", "question", "project",
        "faithfulness", "answer_relevance", "context_recall", "context_precision",
        "ragas_score", "contexts_count", "status", "timestamp",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in ragas_results:
            metrics = r.get("metrics", {})
            writer.writerow({
                "test_number": r.get("test_number", ""),
                "question": r.get("question", ""),
                "project": r.get("project", ""),
                "faithfulness": f"{metrics.get('faithfulness', 0):.4f}" if metrics else "ERR",
                "answer_relevance": f"{metrics.get('answer_relevance', 0):.4f}" if metrics else "ERR",
                "context_recall": f"{metrics.get('context_recall', 0):.4f}" if metrics else "ERR",
                "context_precision": f"{metrics.get('context_precision', 0):.4f}" if metrics else "ERR",
                "ragas_score": f"{r.get('ragas_score', 0):.4f}",
                "contexts_count": r.get("contexts_count", 0),
                "status": "ok" if metrics else "error",
                "timestamp": r.get("timestamp", ""),
            })
    logger.info("CSV da luu: %s", path)
    return path


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Danh gia frame-semantic extraction trong LightRAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Vi du:
  # Chi phan A (frame extraction, khong can server):
  python lightrag/evaluation/eval_frame_semantic.py --internal-only

  # Ca hai phan (can server dang chay tai port 9621):
  python lightrag/evaluation/eval_frame_semantic.py

  # So sanh voi ket qua LLM cu:
  python lightrag/evaluation/eval_frame_semantic.py \\
      --compare lightrag/evaluation/results/results_20250518_120000.json

  # Dung dataset rieng va endpoint khac:
  python lightrag/evaluation/eval_frame_semantic.py \\
      -d my_data.json -r http://my-server:9621
        """,
    )
    p.add_argument("--internal-only", action="store_true",
                   help="Chi chay phan A (frame extraction), bo qua RAGAS end-to-end")
    p.add_argument("--ragas-only", action="store_true",
                   help="Chi chay phan B (RAGAS), bo qua phan A")
    p.add_argument("--compare", "-c", type=str, default=None,
                   metavar="LLM_RESULTS_JSON",
                   help="Duong dan file JSON tu eval_rag_quality.py de so sanh")
    p.add_argument("--dataset", "-d", type=str, default=None,
                   help="Duong dan file dataset JSON (mac dinh: sample_dataset.json)")
    p.add_argument("--ragendpoint", "-r", type=str, default=None,
                   help="URL LightRAG API (mac dinh: http://localhost:9621)")
    return p.parse_args()


async def main():
    args = parse_args()

    full_report: dict = {
        "timestamp": datetime.now().isoformat(),
        "extraction_mode": "frame_semantic_transformer",
        "internal_eval": None,
        "ragas_eval": None,
        "comparison": None,
    }

    # ── Phần A: đánh giá nội bộ ──────────────────────────────────────────────
    if not args.ragas_only:
        try:
            internal_eval = InternalFrameEvaluator()
            full_report["internal_eval"] = internal_eval.run()
        except ImportError as exc:
            logger.error("Bo qua phan A: %s", exc)

    # ── Phần B: RAGAS end-to-end ─────────────────────────────────────────────
    if not args.internal_only:
        try:
            ragas_eval = FrameSemanticRAGEvaluator(
                test_dataset_path=args.dataset,
                rag_api_url=args.ragendpoint,
            )
            ragas_summary = await ragas_eval.run()
            full_report["ragas_eval"] = ragas_summary

            # Lưu CSV riêng
            _save_csv(ragas_summary["results"])

            # ── Phần C: So sánh (nếu có) ─────────────────────────────────────
            if args.compare:
                try:
                    cmp = compare_results(ragas_summary, args.compare)
                    full_report["comparison"] = cmp
                except (FileNotFoundError, KeyError) as exc:
                    logger.error("Loi so sanh: %s", exc)

        except ImportError as exc:
            logger.error("Bo qua phan B: %s", exc)
        except Exception as exc:
            logger.exception("Loi RAGAS: %s", exc)

    # ── Lưu báo cáo tổng hợp ─────────────────────────────────────────────────
    report_path = _save_results(full_report)
    logger.info("")
    logger.info("BAO CAO TONG HOP: %s", report_path)


if __name__ == "__main__":
    asyncio.run(main())
