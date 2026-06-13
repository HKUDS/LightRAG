#!/usr/bin/env python3
"""
Script chạy đánh giá SO SÁNH đầy đủ giữa:
  - LLM-based extraction  (phiên bản gốc LightRAG)
  - Frame-semantic extraction (phiên bản đã tích hợp frame-semantic-transformer)

Kịch bản chạy:
  1. Đánh giá nội bộ frame extraction (Phần A — không cần server).
  2. Đánh giá RAGAS end-to-end của hệ thống HIỆN TẠI (frame-semantic).
  3. (Tuỳ chọn) Nếu đã có kết quả LLM-based trước đó, tạo bảng so sánh.
  4. (Tuỳ chọn) Nếu truyền --run-llm-baseline, script khởi động một LightRAG
     với LLM extraction rồi tự so sánh (cần cài phiên bản gốc hoặc dùng
     cờ LIGHTRAG_USE_LLM_EXTRACTION=1 để tạm thời bật lại LLM extraction).

Yêu cầu:
  - LightRAG API server đang chạy (cho phần RAGAS)
  - EVAL_LLM_BINDING_API_KEY hoặc OPENAI_API_KEY (cho RAGAS scoring)
  - frame-semantic-transformer đã cài

Cách chạy:
    # Chạy nhanh — chỉ phan A:
    python lightrag/evaluation/run_comparison.py --quick

    # Chạy đầy đủ (cần server):
    python lightrag/evaluation/run_comparison.py

    # So sánh với kết quả LLM có sẵn:
    python lightrag/evaluation/run_comparison.py \\
        --llm-results lightrag/evaluation/results/results_20250518.json

    # Chỉ định endpoint và dataset riêng:
    python lightrag/evaluation/run_comparison.py \\
        -r http://localhost:9621 -d my_dataset.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
from lightrag.utils import logger

load_dotenv(dotenv_path=".env", override=False)

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Bảng kết quả cuối
# ──────────────────────────────────────────────────────────────────────────────

def _print_banner(title: str, width: int = 70):
    logger.info("=" * width)
    logger.info(title.center(width))
    logger.info("=" * width)


def print_final_report(full: dict):
    """In bảng tổng hợp cuối cùng ra console."""
    _print_banner("BAO CAO SO SANH CUOI CUNG")

    # Phần A
    internal = full.get("internal_eval")
    if internal:
        summ = internal.get("summary", {})
        logger.info("")
        logger.info("[PHAN A] FRAME EXTRACTION NOI BO")
        logger.info("  Frame Recall:         %.4f", summ.get("avg_frame_recall", 0))
        logger.info("  Frame Precision:      %.4f", summ.get("avg_frame_precision", 0))
        logger.info("  Entity Recall:        %.4f", summ.get("avg_entity_recall", 0))
        logger.info("  Toc do trung binh:    %.3f giay/mau", summ.get("avg_elapsed_seconds", 0))
        logger.info("  Tong frames phat hien: %d", summ.get("total_frames_detected", 0))
        logger.info("  Tong entities:         %d", summ.get("total_entities_detected", 0))

    # Phần B
    ragas = full.get("ragas_eval")
    if ragas:
        avg = ragas.get("average_metrics", {})
        logger.info("")
        logger.info("[PHAN B] RAGAS END-TO-END (FRAME-SEMANTIC)")
        logger.info("  Faithfulness:      %.4f", avg.get("faithfulness", 0))
        logger.info("  Answer Relevance:  %.4f", avg.get("answer_relevance", 0))
        logger.info("  Context Recall:    %.4f", avg.get("context_recall", 0))
        logger.info("  Context Precision: %.4f", avg.get("context_precision", 0))
        logger.info("  RAGAS Score:       %.4f", avg.get("ragas_score", 0))
        logger.info("  Thanh cong:        %d/%d",
                    ragas.get("successful_tests", 0), ragas.get("total_tests", 0))
        logger.info("  Thoi gian tong:    %.1f giay", ragas.get("elapsed_seconds", 0))

    # Phần so sánh
    cmp = full.get("comparison")
    if cmp:
        comparison = cmp.get("comparison", {})
        logger.info("")
        logger.info("[SO SANH] FRAME-SEMANTIC vs LLM-BASED")
        logger.info("  %-22s  %10s  %10s  %8s  %s",
                    "Metric", "LLM-based", "Frame-Sem", "Delta", "Thang")
        logger.info("  " + "-" * 63)
        for metric, c in comparison.items():
            winner_label = (
                "Frame +" if c["winner"] == "frame_semantic"
                else ("LLM  +" if c["winner"] == "llm_based" else "Hoa  ")
            )
            logger.info(
                "  %-22s  %10.4f  %10.4f  %+8.4f  %s",
                metric, c["llm_based"], c["frame_semantic"], c["delta"], winner_label,
            )
        logger.info("")
        logger.info("  Ket luan: Frame-semantic thang %d chi tieu | LLM thang %d chi tieu | Hoa %d",
                    cmp["frame_wins"], cmp["llm_wins"], cmp["ties"])

        # Đánh giá tổng thể
        if cmp["frame_wins"] > cmp["llm_wins"]:
            logger.info("  => Frame-semantic extraction TOT HON tong the.")
        elif cmp["llm_wins"] > cmp["frame_wins"]:
            logger.info("  => LLM-based extraction TOT HON tong the.")
        else:
            logger.info("  => Hai phuong phap TUONG DUONG nhau.")

    logger.info("")
    logger.info("=" * 70)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Chay danh gia so sanh Frame-Semantic vs LLM-based extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Vi du:

  Kiem tra nhanh chi frame extraction (khong can server):
    python lightrag/evaluation/run_comparison.py --quick

  Danh gia day du (server tai localhost:9621):
    python lightrag/evaluation/run_comparison.py

  So sanh voi ket qua LLM cu (tu eval_rag_quality.py):
    python lightrag/evaluation/run_comparison.py \\
        --llm-results lightrag/evaluation/results/results_20250518.json

  Tuy chinh endpoint va dataset:
    python lightrag/evaluation/run_comparison.py \\
        -r http://my-server:9621 -d my_dataset.json
        """,
    )
    p.add_argument("--quick", action="store_true",
                   help="Chi chay phan A (frame extraction) — khong can server, khong can API key")
    p.add_argument("--llm-results", "-l", type=str, default=None,
                   metavar="JSON_FILE",
                   help="File JSON ket qua tu eval_rag_quality.py de so sanh")
    p.add_argument("--dataset", "-d", type=str, default=None,
                   help="Path file dataset JSON (mac dinh: sample_dataset.json)")
    p.add_argument("--ragendpoint", "-r", type=str, default=None,
                   help="LightRAG API URL (mac dinh: http://localhost:9621)")
    p.add_argument("--no-internal", action="store_true",
                   help="Bo qua phan A (frame extraction noi bo)")
    p.add_argument("--output", "-o", type=str, default=None,
                   help="Duong dan luu file JSON ket qua (mac dinh: tu dong theo timestamp)")
    return p.parse_args()


async def main():
    args = parse_args()

    t_start = time.time()
    _print_banner("LIGHTRAG — DANH GIA FRAME-SEMANTIC EXTRACTION")
    logger.info("Bat dau: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("Mode:    %s", "quick (chi phan A)" if args.quick else "day du (A + B)")
    if args.llm_results:
        logger.info("So sanh: %s", args.llm_results)
    logger.info("")

    full_report: dict = {
        "timestamp": datetime.now().isoformat(),
        "extraction_mode": "frame_semantic_transformer",
        "run_mode": "quick" if args.quick else "full",
        "llm_results_file": args.llm_results,
        "internal_eval": None,
        "ragas_eval": None,
        "comparison": None,
    }

    # ── Phần A: Frame extraction nội bộ ──────────────────────────────────────
    if not args.no_internal:
        try:
            from lightrag.evaluation.eval_frame_semantic import InternalFrameEvaluator
            evaluator_a = InternalFrameEvaluator()
            full_report["internal_eval"] = evaluator_a.run()
        except ImportError as exc:
            logger.warning("Bo qua phan A — thieu thu vien: %s", exc)
        except Exception as exc:
            logger.error("Loi phan A: %s", exc)

    # ── Phần B: RAGAS end-to-end ─────────────────────────────────────────────
    if not args.quick:
        try:
            from lightrag.evaluation.eval_frame_semantic import (
                FrameSemanticRAGEvaluator,
                compare_results,
                _save_csv,
            )

            evaluator_b = FrameSemanticRAGEvaluator(
                test_dataset_path=args.dataset,
                rag_api_url=args.ragendpoint,
            )
            ragas_summary = await evaluator_b.run()
            full_report["ragas_eval"] = ragas_summary

            # Lưu CSV chi tiết
            _save_csv(ragas_summary["results"])

            # ── Phần C: So sánh nếu có file LLM cũ ──────────────────────────
            if args.llm_results:
                try:
                    cmp = compare_results(ragas_summary, args.llm_results)
                    full_report["comparison"] = cmp
                except Exception as exc:
                    logger.error("Loi so sanh: %s", exc)

        except ImportError as exc:
            logger.warning("Bo qua phan B — thieu thu vien: %s", exc)
        except Exception as exc:
            logger.exception("Loi phan B: %s", exc)

    # ── Lưu + in báo cáo cuối ─────────────────────────────────────────────────
    elapsed = time.time() - t_start
    full_report["total_elapsed_seconds"] = round(elapsed, 2)

    # Đường dẫn lưu
    if args.output:
        out_path = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = RESULTS_DIR / f"comparison_{ts}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, ensure_ascii=False, indent=2)

    print_final_report(full_report)

    logger.info("Bao cao day du: %s", out_path.absolute())
    logger.info("Tong thoi gian: %.1f giay", elapsed)


if __name__ == "__main__":
    asyncio.run(main())
