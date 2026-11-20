"""Write amplification and backfill throughput.

Rules:
- The write-amplification comparison measures per-batch upsert+flush p95 of
  the branch against a pristine
  baseline checkout of ``baseline_ref``, running the SAME self-contained
  driver with each checkout's own interpreter on the same corpus and PG
  load. The baseline lives in a temporary detached git worktree under the
  system temp dir and is ALWAYS removed afterwards, even on failure.
- GIN-vs-vector index sizes come from per-index ``pg_relation_size`` on the
  ingested chunks table, classified by the production index-name suffixes.
- The backfill measurement seeds NULL-tsv rows for the run's workspace,
  triggers the production
  backfill scheduling path, and polls the pending count to zero under a
  bounded deadline; throughput is report-only.
- Every subprocess and poll has an explicit timeout; nothing waits forever.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from lightrag import LightRAG
from lightrag.evaluation.hybrid_bench.rag_setup import BenchConfig

_DRIVER_PATH = Path(__file__).resolve().parent / "write_amp_driver.py"
_UV_SYNC_TIMEOUT = 1800
_DRIVER_TIMEOUT = 2000


@dataclass(frozen=True, slots=True)
class WriteAmpResult:
    branch_p95: float
    baseline_p95: float | None
    upsert_increase: float | None
    gin_bytes: int | None
    vector_bytes: int | None
    gin_vector_ratio: float | None
    baseline_ref: str
    corpus_sha256: str


def _run_driver(
    interpreter: str,
    cwd: Path,
    corpus_path: Path,
    workspace: str,
    dim: int,
    batch_size: int,
    out_json: Path,
) -> dict:
    # The driver runs with cwd=temp dir: a relative --dataset-dir corpus
    # path must be resolved against the CALLER's cwd before the subprocess.
    completed = subprocess.run(
        [
            interpreter,
            str(_DRIVER_PATH),
            str(Path(corpus_path).resolve()),
            workspace,
            str(dim),
            str(batch_size),
            str(out_json),
        ],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=_DRIVER_TIMEOUT,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"write-amp driver failed (rc={completed.returncode}) under "
            f"{interpreter}:\n{completed.stderr[-4000:]}"
        )
    return json.loads(out_json.read_text(encoding="utf-8"))


def measure_write_amp(
    config: BenchConfig,
    corpus_path: Path,
    repo_root: Path,
    baseline_ref: str,
    batch_size: int = 32,
    include_baseline: bool = True,
) -> WriteAmpResult:
    """Run the driver on the branch and, optionally, on a detached baseline
    worktree that is removed in ``finally``."""
    suffix = uuid4().hex[:8]
    with tempfile.TemporaryDirectory(prefix="hbench-writeamp-") as tmp:
        tmp_dir = Path(tmp)
        branch_out = tmp_dir / "branch.json"
        branch = _run_driver(
            sys.executable,
            tmp_dir,
            corpus_path,
            f"wab_{suffix}",
            config.embedding_dim,
            batch_size,
            branch_out,
        )
        baseline = None
        if include_baseline:
            worktree = tmp_dir / "baseline"
            subprocess.run(
                [
                    "git",
                    "worktree",
                    "add",
                    "--detach",
                    str(worktree),
                    baseline_ref,
                ],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                timeout=300,
                check=True,
            )
            try:
                subprocess.run(
                    ["uv", "sync", "--extra", "test", "--extra", "offline"],
                    cwd=str(worktree),
                    capture_output=True,
                    text=True,
                    timeout=_UV_SYNC_TIMEOUT,
                    check=True,
                )
                baseline_out = tmp_dir / "baseline.json"
                baseline = _run_driver(
                    str(worktree / ".venv" / "bin" / "python"),
                    tmp_dir,
                    corpus_path,
                    f"wab_base_{suffix}",
                    config.embedding_dim,
                    batch_size,
                    baseline_out,
                )
            finally:
                subprocess.run(
                    ["git", "worktree", "remove", "--force", str(worktree)],
                    cwd=str(repo_root),
                    capture_output=True,
                    text=True,
                    timeout=300,
                    check=False,
                )
                subprocess.run(
                    ["git", "worktree", "prune"],
                    cwd=str(repo_root),
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )
    increase = None
    if baseline is not None and baseline["p95_seconds"] > 0:
        increase = (branch["p95_seconds"] - baseline["p95_seconds"]) / baseline[
            "p95_seconds"
        ]
    return WriteAmpResult(
        branch_p95=branch["p95_seconds"],
        baseline_p95=baseline["p95_seconds"] if baseline else None,
        upsert_increase=increase,
        gin_bytes=None,
        vector_bytes=None,
        gin_vector_ratio=None,
        baseline_ref=baseline_ref,
        corpus_sha256=branch["corpus_sha256"],
    )


async def measure_index_sizes(rag: LightRAG) -> tuple[int | None, int | None]:
    """GIN and vector index bytes on the chunks table, by production name
    suffixes; None when an index is absent."""
    db = rag.chunks_vdb.db
    table = rag.chunks_vdb.table_name
    rows = await db.query(
        "SELECT c.relname AS index_name, pg_relation_size(c.oid) AS bytes "
        "FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid "
        "WHERE i.indrelid = to_regclass($1)",
        [table],
        multirows=True,
    )
    gin_bytes = None
    vector_bytes = None
    for row in rows or []:
        name = row["index_name"]
        if name.endswith("content_tsv_gin"):
            gin_bytes = row["bytes"]
        elif name.endswith("_cosine"):
            vector_bytes = row["bytes"]
    return gin_bytes, vector_bytes


async def measure_backfill(rag: LightRAG, deadline_seconds: float = 600.0) -> float:
    """Backfill: NULL the workspace's tsv rows, drive the production backfill
    scheduling path, poll to zero, return chunks per minute. Report-only.

    Uses the storage's private readiness/cancel hooks because the production
    backfill has no public trigger; the measured path is the real one.
    """
    vdb = rag.chunks_vdb
    db = vdb.db
    # Table names are the storage's own uppercase constants, interpolated
    # UNQUOTED exactly like the production SQL templates so PG folds them to
    # the lowercase physical names; quoting would preserve the uppercase.
    table = vdb.table_name
    await db.execute(
        f"UPDATE {table} SET content_tsv = NULL WHERE workspace = $1",
        {"workspace": vdb.workspace},
    )
    pending = await db.query(
        f"SELECT COUNT(*) AS n FROM {table} "
        "WHERE workspace = $1 AND content_tsv IS NULL",
        [vdb.workspace],
    )
    total = pending["n"]
    if total == 0:
        raise ValueError("backfill measurement needs seeded rows, found none")
    start = time.perf_counter()
    await vdb._ensure_lexical_readiness()
    try:
        while True:
            row = await db.query(
                f"SELECT COUNT(*) AS n FROM {table} "
                "WHERE workspace = $1 AND content_tsv IS NULL",
                [vdb.workspace],
            )
            if row["n"] == 0:
                break
            if time.perf_counter() - start > deadline_seconds:
                raise TimeoutError(
                    f"backfill did not finish within {deadline_seconds}s; "
                    f"{row['n']} rows still pending"
                )
            await asyncio.sleep(0.05)
    finally:
        await vdb._cancel_lexical_backfill()
    elapsed_minutes = (time.perf_counter() - start) / 60.0
    return total / elapsed_minutes
