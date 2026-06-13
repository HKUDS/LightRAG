#!/usr/bin/env python3
"""
Chạy đánh giá SO SÁNH BA CHẾ ĐỘ trích xuất LightRAG:

  Mode "none"    — Baseline: LLM cho mọi thứ (giống LightRAG gốc)
  Mode "hl_only" — Case 1:  Frame-semantic → HL keyword; LLM → LL keyword và entities
  Mode "full"    — Case 2:  Frame-semantic → cả HL và LL (mặc định)

Ghi chú quan trọng:
  - Pha de-duplicate / summarise LUÔN dùng LLM ở cả 3 chế độ.
  - Chỉ pha trích xuất keywords (online) và entities (offline indexing) thay đổi.
  - Muốn so sánh end-to-end chính xác, cần build index riêng cho từng chế độ.

Các bước script thực hiện:
  A) Đánh giá keyword extraction nội bộ (frame modes, không cần server hay API key).
  B) Nếu truyền --ragas-files, tổng hợp và so sánh 3 file JSON kết quả RAGAS
     (thu thập bằng cách chạy thủ công server 3 lần rồi eval với eval_frame_semantic.py).
  C) In bảng so sánh tổng hợp.

Cách sử dụng nhanh (chỉ keyword eval, không cần server):
    python lightrag/evaluation/run_three_mode_eval.py --keyword-only

Cách sử dụng đầy đủ (cần 3 file JSON RAGAS kết quả):
    python lightrag/evaluation/run_three_mode_eval.py \\
        --ragas-files none=results/ragas_none.json \\
                      hl_only=results/ragas_hlonly.json \\
                      full=results/ragas_full.json

Xem FRAME_SEMANTIC_README.md để biết hướng dẫn build index 3 chế độ.
"""

from __future__ import annotations

import argparse
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

ALL_MODES = ["none", "hl_only", "full"]


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="So sanh 3 che do extraction cua LightRAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Vi du:

  Kiem tra keyword extraction (khong can server):
    python lightrag/evaluation/run_three_mode_eval.py --keyword-only

  So sanh ket qua RAGAS (da co san 3 file):
    python lightrag/evaluation/run_three_mode_eval.py \\
        --ragas-files none=results/r_none.json hl_only=results/r_hl.json full=results/r_full.json

  Chi danh gia che do "full" va "hl_only" (bo qua "none"):
    python lightrag/evaluation/run_three_mode_eval.py \\
        --modes full hl_only --keyword-only
        """,
    )
    p.add_argument(
        "--keyword-only", action="store_true",
        help="Chi chay phan A (keyword extraction noi bo, khong can server)",
    )
    p.add_argument(
        "--modes", nargs="+", choices=ALL_MODES, default=["hl_only", "full"],
        metavar="MODE",
        help="Cac che do can danh gia: none, hl_only, full (mac dinh: hl_only full)",
    )
    p.add_argument(
        "--ragas-files", nargs="+", default=None,
        metavar="MODE=FILE",
        help="File JSON ket qua RAGAS cho tung che do, vi du: none=results/r.json",
    )
    p.add_argument(
        "--output", "-o", type=str, default=None,
        help="File JSON ket qua tong hop (mac dinh: tu dong theo timestamp)",
    )
    return p.parse_args()


def parse_ragas_files(ragas_files: list[str]) -> dict[str, Path]:
    """Parse danh sach 'mode=file' thanh dict."""
    result: dict[str, Path] = {}
    for item in ragas_files:
        if "=" not in item:
            logger.error("Dinh dang sai (can 'mode=file'): %s", item)
            continue
        mode, file_path = item.split("=", 1)
        mode = mode.strip()
        if mode not in ALL_MODES:
            logger.warning("Mode khong hop le: %s (bo qua)", mode)
            continue
        path = Path(file_path.strip())
        if not path.exists():
            logger.error("File khong ton tai: %s", path)
            continue
        result[mode] = path
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    t_start = time.time()

    logger.info("=" * 70)
    logger.info("LIGHTRAG — DANH GIA SO SANH 3 CHE DO EXTRACTION".center(70))
    logger.info("=" * 70)
    logger.info("Bat dau: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("Modes: %s", args.modes)
    logger.info("")

    full_report: dict = {
        "timestamp": datetime.now().isoformat(),
        "modes_evaluated": args.modes,
        "keyword_eval": None,
        "ragas_comparison": None,
    }

    # ── Phần A: Keyword extraction nội bộ ────────────────────────────────────
    try:
        from lightrag.evaluation.eval_frame_semantic import ThreeModeKeywordEvaluator

        # Loại bỏ "none" nếu không có LLM key (frame-only modes don't need API)
        keyword_modes = [m for m in args.modes if m != "none"]
        if keyword_modes:
            kw_evaluator = ThreeModeKeywordEvaluator(modes=keyword_modes)
            keyword_report = kw_evaluator.run()
            full_report["keyword_eval"] = keyword_report
        else:
            logger.warning("Khong co mode frame-semantic de danh gia keyword extraction.")
    except ImportError as exc:
        logger.warning("Bo qua phan A — thieu thu vien: %s", exc)
    except Exception as exc:
        logger.exception("Loi phan A: %s", exc)

    # ── Phần B: So sánh RAGAS (từ file JSON) ─────────────────────────────────
    if not args.keyword_only and args.ragas_files:
        ragas_paths = parse_ragas_files(args.ragas_files)
        if ragas_paths:
            try:
                from lightrag.evaluation.eval_frame_semantic import compare_three_modes

                # Load từng file JSON
                ragas_results_by_mode: dict[str, dict] = {}
                for mode, path in ragas_paths.items():
                    logger.info("Loading RAGAS result [%s]: %s", mode, path)
                    with open(path, encoding="utf-8") as f:
                        data = json.load(f)
                    # Có thể là file từ eval_frame_semantic.py (ragas_eval) hoặc trực tiếp
                    ragas_results_by_mode[mode] = (
                        data.get("ragas_eval") or
                        data.get("benchmark_stats") or
                        data
                    )

                cmp = compare_three_modes(ragas_results_by_mode)
                full_report["ragas_comparison"] = cmp
            except Exception as exc:
                logger.exception("Loi so sanh RAGAS: %s", exc)
    elif not args.keyword_only:
        logger.info("")
        logger.info("TIP: Truyen --ragas-files de so sanh RAGAS end-to-end.")
        logger.info("     Xem FRAME_SEMANTIC_README.md de biet cach thu thap ket qua.")

    # ── Lưu báo cáo ───────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    full_report["total_elapsed_seconds"] = round(elapsed, 2)

    if args.output:
        out_path = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = RESULTS_DIR / f"three_mode_eval_{ts}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, ensure_ascii=False, indent=2)

    logger.info("")
    logger.info("=" * 70)
    logger.info("KET QUA DA LUU: %s", out_path.absolute())
    logger.info("TONG THOI GIAN: %.1f giay", elapsed)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
