#!/usr/bin/env python3
"""
Post-fix evaluation: compare none vs llm_frames+FAGE on sample dataset.

Spawns two isolated subprocesses (one per mode) so the module-level
LIGHTRAG_FRAME_EXTRACTION_MODE constant is set correctly before any
LightRAG imports occur.

Usage:
    python eval_postfix.py

Estimated cost: ~$0.05-0.10 total
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

STORAGE_NONE = Path("eval_tmp_none")
STORAGE_LLM = Path("eval_tmp_llm")
WORKER = Path(__file__).parent / "eval_mode_worker.py"


def run_mode(mode: str, storage_dir: Path, out_json: Path) -> dict:
    """Run one eval mode in a subprocess and return its result dict."""
    print(f"\n{'='*60}", flush=True)
    print(f"MODE: {mode.upper()}", flush=True)
    print(f"{'='*60}", flush=True)

    if storage_dir.exists():
        shutil.rmtree(storage_dir)
    storage_dir.mkdir(parents=True)

    cmd = [
        sys.executable, str(WORKER),
        "--mode", mode,
        "--storage", str(storage_dir),
        "--output", str(out_json),
    ]
    proc = subprocess.run(cmd, capture_output=False)  # stream stdout live

    if proc.returncode != 0:
        print(f"[{mode}] subprocess returned exit code {proc.returncode}", flush=True)

    if out_json.exists():
        return json.loads(out_json.read_text(encoding="utf-8"))
    return {"mode": mode, "error": "no output file"}


def main():
    print("Post-fix FSRAG Evaluation (subprocess isolation)", flush=True)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    out_none = Path("_eval_none_tmp.json")
    out_llm = Path("_eval_llm_tmp.json")

    results = {}
    results["none"] = run_mode("none", STORAGE_NONE, out_none)
    results["llm_frames"] = run_mode("llm_frames", STORAGE_LLM, out_llm)

    # ── summary table ────────────────────────────────────────────────────────
    print(f"\n{'='*62}", flush=True)
    print("RESULTS SUMMARY", flush=True)
    print(f"{'='*62}", flush=True)
    print(f"{'Metric':<32} {'none':>12} {'llm_frames':>14}", flush=True)
    print("-" * 60, flush=True)

    def _fmt(val):
        if isinstance(val, float):
            return f"{val:.4f}"
        if val is None:
            return "N/A"
        return str(val)

    for key in ["faithfulness", "answer_relevancy", "context_recall", "context_precision", "composite"]:
        n_val = results["none"].get("ragas", {}).get(key)
        l_val = results["llm_frames"].get("ragas", {}).get(key)
        winner = ""
        if isinstance(n_val, float) and isinstance(l_val, float):
            winner = " ← none" if n_val > l_val else " ← llm_frames" if l_val > n_val else " (tie)"
        print(f"  {key:<30} {_fmt(n_val):>12} {_fmt(l_val):>14}{winner}", flush=True)

    print("-" * 60, flush=True)
    ng = results["none"].get("graph", {})
    lg = results["llm_frames"].get("graph", {})
    print(f"  {'Graph nodes':<30} {ng.get('nodes', 'N/A'):>12} {lg.get('nodes', 'N/A'):>14}", flush=True)
    print(f"  {'Graph edges':<30} {ng.get('edges', 'N/A'):>12} {lg.get('edges', 'N/A'):>14}", flush=True)
    print(f"  {'Index time (s)':<30} {results['none'].get('index_time_s', 'N/A'):>12} {results['llm_frames'].get('index_time_s', 'N/A'):>14}", flush=True)
    print(f"  {'Query time (s)':<30} {results['none'].get('query_time_s', 'N/A'):>12} {results['llm_frames'].get('query_time_s', 'N/A'):>14}", flush=True)

    # Save combined results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(f"eval_results_postfix_{ts}.json")
    out_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\nFull results: {out_path}", flush=True)

    # Cleanup
    for f in [out_none, out_llm]:
        if f.exists():
            f.unlink()
    for d in [STORAGE_NONE, STORAGE_LLM]:
        if d.exists():
            shutil.rmtree(d)
    print("Cleanup done.", flush=True)


if __name__ == "__main__":
    main()
