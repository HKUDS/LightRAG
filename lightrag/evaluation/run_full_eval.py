#!/usr/bin/env python3
"""
Script chạy đánh giá ĐẦY ĐỦ trên bộ dữ liệu chuẩn.

Thực hiện tuần tự 4 bước:
  1. Kiểm tra môi trường  — frame-semantic-transformer, RAGAS, kết nối server
  2. Frame extraction nội bộ  — đánh giá chất lượng detect frames + entities trên
                                 văn bản mẫu tích hợp sẵn (không cần server)
  3. Index bộ dữ liệu  — đẩy toàn bộ tài liệu từ thư mục --docs-dir vào LightRAG
                          qua API (bỏ qua nếu --skip-index)
  4. RAGAS end-to-end  — đánh giá pipeline RAG trên --dataset bằng 4 metrics RAGAS,
                          lưu kết quả CSV + JSON

Cách chạy:
    # Chạy đầy đủ trên bộ dữ liệu mặc định (sample_dataset.json):
    python lightrag/evaluation/run_full_eval.py

    # Dùng dataset và thư mục tài liệu riêng:
    python lightrag/evaluation/run_full_eval.py \\
        --dataset path/to/my_test.json \\
        --docs-dir path/to/my_documents/

    # Bỏ qua bước index (tài liệu đã được insert sẵn):
    python lightrag/evaluation/run_full_eval.py --skip-index

    # So sánh với kết quả LLM-based:
    python lightrag/evaluation/run_full_eval.py \\
        --compare path/to/llm_results.json

    # Chỉ chạy bước 2 (frame extraction, không cần server):
    python lightrag/evaluation/run_full_eval.py --internal-only

Biến môi trường cần thiết:
    OPENAI_API_KEY (hoặc EVAL_LLM_BINDING_API_KEY) — cho RAGAS scoring
    LIGHTRAG_API_URL — URL server (mặc định http://localhost:9621)
    LIGHTRAG_API_KEY — nếu server có xác thực

Kết quả lưu tại: lightrag/evaluation/results/full_eval_YYYYMMDD_HHMMSS.json
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

load_dotenv(dotenv_path=".env", override=False)

from lightrag.utils import logger

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DEFAULT_DATASET = EVAL_DIR / "sample_dataset.json"
DEFAULT_DOCS_DIR = EVAL_DIR / "sample_documents"
DEFAULT_API_URL = os.getenv("LIGHTRAG_API_URL", "http://localhost:9621")


# ──────────────────────────────────────────────────────────────────────────────
# Bước 1: Kiểm tra môi trường
# ──────────────────────────────────────────────────────────────────────────────

def check_environment(api_url: str, skip_index: bool, internal_only: bool) -> dict:
    """Kiểm tra tất cả điều kiện cần thiết trước khi chạy evaluate."""
    checks: dict[str, dict] = {}

    # frame-semantic-transformer — chỉ bắt buộc khi mode != "none"
    import os as _os
    _frame_mode = _os.getenv("LIGHTRAG_FRAME_EXTRACTION_MODE", "full").lower().strip()
    try:
        from frame_semantic_transformer import FrameSemanticTransformer  # noqa: F401
        checks["frame_semantic"] = {"ok": True, "msg": "frame-semantic-transformer: OK"}
    except Exception as _e:
        if _frame_mode == "none":
            checks["frame_semantic"] = {
                "ok": True,
                "msg": f"frame-semantic-transformer: SKIP (mode=none, khong can)",
            }
        else:
            checks["frame_semantic"] = {
                "ok": False,
                "msg": (
                    f"frame-semantic-transformer khong import duoc (mode={_frame_mode}): {_e}. "
                    "Chay: pip install 'transformers<4.40.0' frame-semantic-transformer --no-deps"
                ),
            }

    # RAGAS
    if not internal_only:
        try:
            import ragas  # noqa: F401
            from datasets import Dataset  # noqa: F401
            checks["ragas"] = {"ok": True, "msg": "ragas + datasets: OK"}
        except ImportError:
            checks["ragas"] = {
                "ok": False,
                "msg": "ragas chua duoc cai. Chay: pip install ragas datasets langchain-openai",
            }

        # LLM API key
        api_key = os.getenv("EVAL_LLM_BINDING_API_KEY") or os.getenv("OPENAI_API_KEY")
        if api_key:
            checks["api_key"] = {"ok": True, "msg": f"API key: {'*' * 8}{api_key[-4:]}"}
        else:
            checks["api_key"] = {
                "ok": False,
                "msg": "Thieu EVAL_LLM_BINDING_API_KEY hoac OPENAI_API_KEY",
            }

        # Kết nối server
        try:
            import httpx
            resp = httpx.get(f"{api_url}/health", timeout=5)
            if resp.status_code == 200:
                checks["server"] = {"ok": True, "msg": f"LightRAG server: OK ({api_url})"}
            else:
                checks["server"] = {
                    "ok": False,
                    "msg": f"Server tra ve {resp.status_code}. Kiem tra server tai {api_url}",
                }
        except Exception as exc:
            checks["server"] = {
                "ok": False,
                "msg": f"Khong the ket noi server {api_url}: {exc}",
            }

    # In kết quả kiểm tra
    logger.info("=" * 65)
    logger.info("BUOC 1: KIEM TRA MOI TRUONG")
    logger.info("=" * 65)
    all_ok = True
    for name, result in checks.items():
        status = "OK " if result["ok"] else "LOI"
        logger.info("  [%s] %s", status, result["msg"])
        if not result["ok"]:
            all_ok = False

    if not all_ok:
        logger.error("")
        logger.error("Co loi moi truong. Kiem tra va cai dat cac goi con thieu truoc khi chay lai.")
        sys.exit(1)

    logger.info("  Tat ca dieu kien: OK")
    return checks


# ──────────────────────────────────────────────────────────────────────────────
# Bước 3: Index tài liệu
# ──────────────────────────────────────────────────────────────────────────────

async def _wait_for_indexing_complete(api_url: str, headers: dict, timeout: int = 600) -> bool:
    """Poll /documents endpoint cho đến khi không còn document nào pending/processing.

    Returns:
        True nếu tất cả docs đã processed thành công.
        False nếu có lỗi (tất cả failed) hoặc timeout.
    """
    import httpx

    logger.info("  Dang doi indexing hoan tat (polling /documents)...")
    start = time.time()
    last_statuses: dict = {}

    async with httpx.AsyncClient(timeout=httpx.Timeout(15, connect=5), headers=headers) as client:
        while time.time() - start < timeout:
            await asyncio.sleep(5)
            try:
                resp = await client.get(f"{api_url}/documents")
                resp.raise_for_status()
                data = resp.json()

                # Hỗ trợ nhiều format: list hoặc dict với key "statuses"/"documents"/"data"
                raw = data.get("statuses", data) if isinstance(data, dict) else data
                if isinstance(raw, dict) and any(
                    isinstance(v, list) for v in raw.values()
                ):
                    # Format: {"statuses": {"failed": [...], "processed": [...]}}
                    statuses: dict = {k: len(v) for k, v in raw.items() if isinstance(v, list)}
                elif isinstance(raw, dict):
                    # Format tổng hợp: {"pending": n, "processing": n, ...}
                    statuses = raw
                else:
                    # Format list
                    statuses = {}
                    for d in raw:
                        s = d.get("status", "unknown")
                        statuses[s] = statuses.get(s, 0) + 1

                if statuses != last_statuses:
                    logger.info("  [%.0fs] Trang thai: %s", time.time() - start, statuses)
                    last_statuses = statuses

                pending = statuses.get("pending", 0) + statuses.get("processing", 0)
                failed = statuses.get("failed", 0)
                processed = statuses.get("processed", statuses.get("success", 0))

                if pending == 0 and statuses:
                    elapsed = time.time() - start
                    if failed > 0 and processed == 0:
                        logger.error(
                            "  TAT CA %d DOCUMENTS BI FAILED! RAGAS se tra ve 0. "
                            "Kiem tra EMBEDDING_DIM trong .env va xoa storage roi thu lai.",
                            failed,
                        )
                        return False
                    if failed > 0:
                        logger.warning(
                            "  %d documents failed, %d thanh cong. RAGAS co the bi anh huong.",
                            failed, processed,
                        )
                    logger.info("  Indexing hoan tat sau %.0f giay.", elapsed)
                    return True

            except Exception as exc:
                logger.warning("  Loi poll /documents: %s — thu lai...", exc)

    logger.warning("  Timeout doi indexing (%ds). Tiep tuc evaluate.", timeout)
    return False


async def index_documents(docs_dir: Path, api_url: str) -> dict:
    """
    Đẩy toàn bộ file .txt/.md trong docs_dir vào LightRAG qua API,
    sau đó đợi server xử lý xong trước khi trả về.
    """
    import httpx

    logger.info("=" * 65)
    logger.info("BUOC 3: INDEX TAI LIEU tu %s", docs_dir)
    logger.info("=" * 65)

    files = list(docs_dir.glob("*.txt")) + list(docs_dir.glob("*.md"))
    files = [f for f in files if f.name.lower() != "readme.md"]

    if not files:
        logger.warning("Khong tim thay file .txt hoac .md trong %s", docs_dir)
        return {"indexed": 0, "skipped": 0, "errors": 0}

    api_key = os.getenv("LIGHTRAG_API_KEY")
    headers: dict = {}
    if api_key:
        headers["X-API-Key"] = api_key

    indexed = 0
    skipped = 0
    errors = 0
    total_chars = 0
    t0 = time.time()

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(300, connect=30),
        headers=headers,
    ) as client:
        for fpath in sorted(files):
            text = fpath.read_text(encoding="utf-8")
            if not text.strip():
                skipped += 1
                continue
            try:
                resp = await client.post(
                    f"{api_url}/documents/text",
                    json={"text": text, "file_path": str(fpath.name)},
                )
                resp.raise_for_status()
                indexed += 1
                total_chars += len(text)
                logger.info("  [OK] %s (%d ky tu)", fpath.name, len(text))
            except Exception as exc:
                errors += 1
                logger.error("  [LOI] %s: %s", fpath.name, exc)

    submit_elapsed = time.time() - t0
    logger.info(
        "  Da gui: %d files | %d bo qua | %d loi | %.1f giay",
        indexed, skipped, errors, submit_elapsed,
    )

    # Đợi server xử lý xong hoàn toàn trước khi cho RAGAS chạy
    indexing_ok = True
    if indexed > 0:
        indexing_ok = await _wait_for_indexing_complete(api_url, headers)

    elapsed = time.time() - t0
    logger.info("  Tong thoi gian indexing: %.1f giay | %d ky tu", elapsed, total_chars)
    return {
        "indexed": indexed,
        "skipped": skipped,
        "errors": errors,
        "total_chars": total_chars,
        "elapsed_seconds": round(elapsed, 2),
        "indexing_ok": indexing_ok,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Lưu & In báo cáo
# ──────────────────────────────────────────────────────────────────────────────

def _print_separator(width: int = 65):
    logger.info("=" * width)


def print_full_report(report: dict):
    """In bảng tổng hợp kết quả đánh giá."""
    _print_separator()
    logger.info("  BAO CAO DANH GIA DAY DU".center(65))
    _print_separator()
    logger.info("  Thoi gian chay: %s", report.get("timestamp", ""))
    logger.info("  Phuong phap:    frame-semantic-transformer (T5/FrameNet)")

    # Bước 2 — Frame extraction nội bộ
    internal = report.get("internal_eval")
    if internal and internal.get("summary"):
        s = internal["summary"]
        logger.info("")
        logger.info("  [A] FRAME EXTRACTION NOI BO")
        logger.info("      Frame Recall:         %.4f", s.get("avg_frame_recall", 0))
        logger.info("      Frame Precision:      %.4f", s.get("avg_frame_precision", 0))
        logger.info("      Entity Recall:        %.4f", s.get("avg_entity_recall", 0))
        logger.info("      Toc do (trung binh):  %.3f giay/mau", s.get("avg_elapsed_seconds", 0))
        logger.info("      Tong frames:          %d", s.get("total_frames_detected", 0))
        logger.info("      Tong entities:        %d", s.get("total_entities_detected", 0))

    # Bước 4 — RAGAS
    ragas_eval = report.get("ragas_eval")
    if ragas_eval:
        avg = ragas_eval.get("average_metrics", {})
        logger.info("")
        logger.info("  [B] RAGAS END-TO-END")
        logger.info("      Faithfulness:      %.4f", avg.get("faithfulness", 0))
        logger.info("      Answer Relevance:  %.4f", avg.get("answer_relevance", 0))
        logger.info("      Context Recall:    %.4f", avg.get("context_recall", 0))
        logger.info("      Context Precision: %.4f", avg.get("context_precision", 0))
        logger.info("      RAGAS Score:       %.4f", avg.get("ragas_score", 0))
        logger.info(
            "      Thanh cong: %d/%d  |  Thoi gian: %.1f giay",
            ragas_eval.get("successful_tests", 0),
            ragas_eval.get("total_tests", 0),
            ragas_eval.get("elapsed_seconds", 0),
        )

    # So sánh
    cmp = report.get("comparison")
    if cmp:
        comparison = cmp.get("comparison", {})
        logger.info("")
        logger.info("  [C] SO SANH: FRAME-SEMANTIC vs LLM-BASED")
        logger.info("      %-22s  %9s  %9s  %8s  %s",
                    "Metric", "LLM", "Frame", "Delta", "Thang")
        logger.info("      " + "-" * 58)
        for m, c in comparison.items():
            lbl = "Frame+" if c["winner"] == "frame_semantic" else (
                "LLM+  " if c["winner"] == "llm_based" else "Hoa   "
            )
            logger.info(
                "      %-22s  %9.4f  %9.4f  %+8.4f  %s",
                m, c["llm_based"], c["frame_semantic"], c["delta"], lbl,
            )
        logger.info("      Frame-semantic thang: %d/%d | LLM thang: %d/%d",
                    cmp["frame_wins"], len(comparison),
                    cmp["llm_wins"], len(comparison))

    _print_separator()


def save_report(report: dict, output_path: Path | None = None) -> Path:
    if output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = RESULTS_DIR / f"full_eval_{ts}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info("Bao cao JSON: %s", output_path.absolute())
    return output_path


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Chay danh gia day du pipeline LightRAG voi frame-semantic extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Vi du:
  # Chay day du voi du lieu mac dinh:
  python lightrag/evaluation/run_full_eval.py

  # Dung dataset va thu muc tai lieu rieng:
  python lightrag/evaluation/run_full_eval.py \\
      --dataset my_test.json --docs-dir my_docs/

  # Bo qua buoc index (da index truoc):
  python lightrag/evaluation/run_full_eval.py --skip-index

  # Chi kiem tra frame extraction (khong can server):
  python lightrag/evaluation/run_full_eval.py --internal-only

  # So sanh voi ket qua LLM:
  python lightrag/evaluation/run_full_eval.py \\
      --compare lightrag/evaluation/results/results_20250518.json \\
      --skip-index
        """,
    )
    p.add_argument("--dataset", "-d", type=str, default=None,
                   help=f"File dataset JSON (mac dinh: {DEFAULT_DATASET.name})")
    p.add_argument("--docs-dir", type=str, default=None,
                   help=f"Thu muc tai lieu de index (mac dinh: {DEFAULT_DOCS_DIR.name})")
    p.add_argument("--ragendpoint", "-r", type=str, default=None,
                   help=f"LightRAG API URL (mac dinh: {DEFAULT_API_URL})")
    p.add_argument("--skip-index", action="store_true",
                   help="Bo qua buoc 3 (index tai lieu) — dung khi da insert roi")
    p.add_argument("--internal-only", action="store_true",
                   help="Chi chay buoc 2 (frame extraction noi bo), khong can server")
    p.add_argument("--compare", "-c", type=str, default=None,
                   metavar="LLM_JSON",
                   help="File JSON ket qua LLM-based de tao bang so sanh")
    p.add_argument("--output", "-o", type=str, default=None,
                   help="Duong dan luu file JSON bao cao (mac dinh: auto theo timestamp)")
    p.add_argument("--query-mode", type=str, default="hybrid",
                   choices=["local", "global", "hybrid", "mix", "naive"],
                   help="Mode truy van cho RAGAS (mac dinh: hybrid)")
    return p.parse_args()


async def main():
    args = parse_args()

    api_url = args.ragendpoint or DEFAULT_API_URL
    dataset_path = Path(args.dataset) if args.dataset else DEFAULT_DATASET
    docs_dir = Path(args.docs_dir) if args.docs_dir else DEFAULT_DOCS_DIR

    # Đặt mode truy vấn
    os.environ["EVAL_QUERY_MODE"] = args.query_mode

    t_global = time.time()

    report: dict = {
        "timestamp": datetime.now().isoformat(),
        "extraction_mode": "frame_semantic_transformer",
        "api_url": api_url,
        "dataset": str(dataset_path),
        "docs_dir": str(docs_dir),
        "query_mode": args.query_mode,
        "env_check": None,
        "index_stats": None,
        "internal_eval": None,
        "ragas_eval": None,
        "comparison": None,
    }

    # ── Bước 1: Kiểm tra môi trường ──────────────────────────────────────────
    report["env_check"] = check_environment(api_url, args.skip_index, args.internal_only)

    # ── Bước 2: Frame extraction nội bộ ──────────────────────────────────────
    logger.info("")
    _print_separator = lambda: logger.info("=" * 65)  # noqa: E731
    _print_separator()
    logger.info("BUOC 2: FRAME EXTRACTION NOI BO")
    _print_separator()
    try:
        from lightrag.evaluation.eval_frame_semantic import InternalFrameEvaluator
        internal_eval = InternalFrameEvaluator()
        report["internal_eval"] = internal_eval.run()
    except ImportError as exc:
        logger.error("Bo qua buoc 2: %s", exc)
    except Exception as exc:
        logger.exception("Loi buoc 2: %s", exc)

    if args.internal_only:
        report["total_elapsed_seconds"] = round(time.time() - t_global, 2)
        output_path = save_report(report, Path(args.output) if args.output else None)
        print_full_report(report)
        logger.info("Hoan tat (internal-only). Bao cao: %s", output_path)
        return

    # ── Bước 3: Index tài liệu ───────────────────────────────────────────────
    if not args.skip_index:
        logger.info("")
        try:
            report["index_stats"] = await index_documents(docs_dir, api_url)
            if not report["index_stats"].get("indexing_ok", True):
                logger.error("")
                logger.error("=" * 65)
                logger.error("  LOI INDEXING: Tat ca documents deu FAILED.")
                logger.error("  Nguyen nhan pho bien: EMBEDDING_DIM sai trong .env")
                logger.error("  Kiem tra: EMBEDDING_MODEL va EMBEDDING_DIM phai khop nhau")
                logger.error("    text-embedding-3-small  -> EMBEDDING_DIM=1536")
                logger.error("    text-embedding-3-large  -> EMBEDDING_DIM=3072")
                logger.error("    text-embedding-ada-002  -> EMBEDDING_DIM=1536")
                logger.error("  Sau do: xoa storage, restart server, chay lai.")
                logger.error("=" * 65)
                logger.error("")
                sys.exit(2)
        except Exception as exc:
            logger.error("Loi buoc 3 (index): %s", exc)
            report["index_stats"] = {"error": str(exc)}
    else:
        logger.info("")
        logger.info("BUOC 3: BO QUA (--skip-index)")

    # ── Bước 4: RAGAS end-to-end ─────────────────────────────────────────────
    logger.info("")
    try:
        from lightrag.evaluation.eval_frame_semantic import (
            FrameSemanticRAGEvaluator,
            compare_results,
            _save_csv,
        )

        ragas_eval = FrameSemanticRAGEvaluator(
            test_dataset_path=str(dataset_path),
            rag_api_url=api_url,
        )
        ragas_summary = await ragas_eval.run()
        report["ragas_eval"] = ragas_summary

        # Lưu CSV chi tiết
        _save_csv(ragas_summary["results"], prefix="full_eval")

        # ── Bước 5: So sánh (nếu có) ─────────────────────────────────────────
        if args.compare:
            logger.info("")
            _print_separator()
            logger.info("BUOC 5: SO SANH VOI KET QUA LLM-BASED")
            _print_separator()
            try:
                cmp = compare_results(ragas_summary, args.compare)
                report["comparison"] = cmp
            except (FileNotFoundError, KeyError) as exc:
                logger.error("Loi so sanh: %s", exc)

    except ImportError as exc:
        logger.error("Bo qua buoc 4 (RAGAS): %s", exc)
    except Exception as exc:
        logger.exception("Loi buoc 4 (RAGAS): %s", exc)

    # ── Lưu & In báo cáo ─────────────────────────────────────────────────────
    report["total_elapsed_seconds"] = round(time.time() - t_global, 2)
    output_path = save_report(report, Path(args.output) if args.output else None)
    print_full_report(report)
    logger.info("Tong thoi gian: %.1f giay", report["total_elapsed_seconds"])
    logger.info("Bao cao day du: %s", output_path.absolute())


if __name__ == "__main__":
    asyncio.run(main())
