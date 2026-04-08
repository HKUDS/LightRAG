import json
import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx


REPO_ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = Path(os.getenv("BENCH_PDF_PATH", "/root/autodl-tmp/Attention Is All You Need.pdf"))
RUNS_ROOT = REPO_ROOT / "reproduce_outputs" / "fullsvc_attention_runs"

LLM_BASE_URL = os.getenv("BENCH_LLM_BASE_URL", "https://api.moonshot.cn/v1")
LLM_MODEL = os.getenv("BENCH_LLM_MODEL", "kimi-k2-0905-preview")
LLM_API_KEY = os.getenv("BENCH_LLM_API_KEY") or os.getenv("LLM_BINDING_API_KEY") or ""
EMBEDDING_BASE_URL = os.getenv(
    "BENCH_EMBEDDING_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"
)
EMBEDDING_MODEL = os.getenv("BENCH_EMBEDDING_MODEL", "doubao-embedding-text-240715")
EMBEDDING_DIM = os.getenv("BENCH_EMBEDDING_DIM", "2560")
EMBEDDING_API_KEY = os.getenv("BENCH_EMBEDDING_API_KEY") or os.getenv(
    "EMBEDDING_BINDING_API_KEY"
) or ""

VARIANTS = [
    {
        "name": "schema_no_type_guidance",
        "repo_dir": os.getenv("BENCH_SCHEMA_NO_TYPE_REPO_DIR", "/root/autodl-tmp/root_fullsvc_schema_no_type"),
        "port": int(os.getenv("BENCH_SCHEMA_NO_TYPE_PORT", "9860")),
    },
    {
        "name": "schema_with_type_guidance",
        "repo_dir": os.getenv(
            "BENCH_SCHEMA_WITH_TYPE_REPO_DIR", "/root/autodl-tmp/root_fullsvc_schema_with_type"
        ),
        "port": int(os.getenv("BENCH_SCHEMA_WITH_TYPE_PORT", "9861")),
    },
    {
        "name": "json_object_no_type_guidance",
        "repo_dir": os.getenv("BENCH_JSON_NO_TYPE_REPO_DIR", "/root/autodl-tmp/root_fullsvc_json_no_type"),
        "port": int(os.getenv("BENCH_JSON_NO_TYPE_PORT", "9862")),
    },
    {
        "name": "json_object_with_type_guidance",
        "repo_dir": os.getenv("BENCH_JSON_WITH_TYPE_REPO_DIR", "/root/autodl-tmp/root_fullsvc_json_with_type"),
        "port": int(os.getenv("BENCH_JSON_WITH_TYPE_PORT", "9863")),
    },
]


def write_env(run_dir: Path, port: int) -> None:
    if not LLM_API_KEY:
        raise RuntimeError("Missing BENCH_LLM_API_KEY/LLM_BINDING_API_KEY.")
    if not EMBEDDING_API_KEY:
        raise RuntimeError("Missing BENCH_EMBEDDING_API_KEY/EMBEDDING_BINDING_API_KEY.")

    lines = [
        "HOST=127.0.0.1",
        f"PORT={port}",
        f"INPUT_DIR={run_dir / 'inputs'}",
        f"WORKING_DIR={run_dir / 'rag_storage'}",
        "SUMMARY_LANGUAGE=English",
        "ENABLE_LLM_CACHE=true",
        "ENABLE_LLM_CACHE_FOR_EXTRACT=true",
        "LLM_BINDING=openai",
        f"LLM_BINDING_HOST={LLM_BASE_URL}",
        f'LLM_BINDING_API_KEY="{LLM_API_KEY}"',
        f"LLM_MODEL={LLM_MODEL}",
        "LLM_TIMEOUT=600",
        "TIMEOUT=600",
        "EMBEDDING_BINDING=openai",
        f"EMBEDDING_BINDING_HOST={EMBEDDING_BASE_URL}",
        f'EMBEDDING_BINDING_API_KEY="{EMBEDDING_API_KEY}"',
        f"EMBEDDING_MODEL={EMBEDDING_MODEL}",
        f"EMBEDDING_DIM={EMBEDDING_DIM}",
        "EMBEDDING_TOKEN_LIMIT=4096",
        "EMBEDDING_SEND_DIM=false",
        "ENTITY_EXTRACTION_USE_JSON=true",
        "MAX_PARALLEL_INSERT=1",
        "MAX_ASYNC=4",
        "CHUNK_SIZE=450",
        "CHUNK_OVERLAP_SIZE=50",
        "",
    ]
    (run_dir / ".env").write_text("\n".join(lines), encoding="utf-8")


def wait_health(port: int, timeout_s: int = 120) -> None:
    deadline = time.time() + timeout_s
    with httpx.Client(timeout=5) as client:
        while time.time() < deadline:
            try:
                resp = client.get(f"http://127.0.0.1:{port}/health")
                if resp.status_code == 200 and resp.json().get("status") == "healthy":
                    return
            except Exception:
                pass
            time.sleep(2)
    raise RuntimeError(f"port {port} did not become healthy")


def upload_and_wait(port: int, pdf_path: Path, timeout_s: int = 3600) -> dict:
    with httpx.Client(timeout=60) as client:
        with open(pdf_path, "rb") as fh:
            resp = client.post(
                f"http://127.0.0.1:{port}/documents/upload",
                files={"file": (pdf_path.name, fh, "application/pdf")},
            )
        resp.raise_for_status()

        deadline = time.time() + timeout_s
        while time.time() < deadline:
            status_resp = client.post(
                f"http://127.0.0.1:{port}/documents/paginated",
                json={
                    "page": 1,
                    "page_size": 100,
                    "sort_field": "updated_at",
                    "sort_direction": "desc",
                },
            )
            status_resp.raise_for_status()
            docs = status_resp.json().get("documents", [])
            match = next((d for d in docs if d.get("file_path") == pdf_path.name), None)
            if match:
                status = match.get("status")
                if status == "processed":
                    return match
                if status == "failed":
                    raise RuntimeError(match.get("error_msg", "document failed"))
            time.sleep(5)

    raise TimeoutError(f"processing timeout on port {port}")


def parse_graphml(path: Path) -> tuple[int, int]:
    tree = ET.parse(path)
    root = tree.getroot()
    ns = {"g": "http://graphml.graphdrawing.org/xmlns"}
    node_count = len(root.findall(".//g:node", ns))
    edge_count = len(root.findall(".//g:edge", ns))
    return node_count, edge_count


def safe_parse_graphml(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    try:
        return parse_graphml(path)
    except Exception:
        return 0, 0


def safe_load_chunk_stats(path: Path) -> dict:
    if not path.exists():
        return {
            "actual_chunk_count": 0,
            "chunks_are_contiguous": False,
            "max_chunk_tokens": 0,
        }
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        items = list(data.values())
        order = sorted(item["chunk_order_index"] for item in items)
        return {
            "actual_chunk_count": len(items),
            "chunks_are_contiguous": order == list(range(len(items))),
            "max_chunk_tokens": max(item["tokens"] for item in items) if items else 0,
        }
    except Exception:
        return {
            "actual_chunk_count": 0,
            "chunks_are_contiguous": False,
            "max_chunk_tokens": 0,
        }


def analyze_logs(log_text: str) -> dict:
    chunk_matches = re.findall(r"Chunk \d+ of \d+ extracted (\d+) Ent \+ (\d+) Rel", log_text)
    raw_entities = sum(int(ent) for ent, _ in chunk_matches)
    raw_relations = sum(int(rel) for _, rel in chunk_matches)
    return {
        "chunk_log_count": len(chunk_matches),
        "raw_entity_mentions": raw_entities,
        "raw_relation_mentions": raw_relations,
        "raw_total_volume": raw_entities + raw_relations,
        "schema_parse_failed": len(re.findall(r"schema parse failed", log_text, flags=re.I)),
        "schema_parse_failed_salvageable": len(
            re.findall(r"diagnostic json_object probe recovered parseable output", log_text, flags=re.I)
        ),
        "malformed_json_repaired": len(re.findall(r"malformed_json_repaired", log_text)),
        "failed_json_parse": len(re.findall(r"Failed to parse JSON extraction result", log_text)),
    }


def stop_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def run_variant(variant: dict) -> dict:
    run_dir = RUNS_ROOT / variant["name"]
    if run_dir.exists():
        subprocess.run(["rm", "-rf", str(run_dir)], check=True)
    (run_dir / "inputs").mkdir(parents=True)
    (run_dir / "rag_storage").mkdir()
    (run_dir / "upload_source").mkdir()
    write_env(run_dir, variant["port"])

    target_pdf = run_dir / "upload_source" / PDF_PATH.name
    target_pdf.write_bytes(PDF_PATH.read_bytes())

    env = os.environ.copy()
    env["PYTHONPATH"] = variant["repo_dir"]
    env["PYTHONUNBUFFERED"] = "1"
    service_log = run_dir / "service.log"

    proc = subprocess.Popen(
        [sys.executable, "-m", "lightrag.api.lightrag_server"],
        cwd=run_dir,
        env=env,
        stdout=open(service_log, "w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
    )

    started = time.time()
    status = "success"
    error_message = None
    doc_meta = {}
    try:
        wait_health(variant["port"])
        doc_meta = upload_and_wait(variant["port"], target_pdf)
    except Exception as exc:
        status = "failed"
        error_message = str(exc)
    finally:
        stop_process(proc)

    graph_path = run_dir / "rag_storage" / "graph_chunk_entity_relation.graphml"
    entity_count, relation_count = safe_parse_graphml(graph_path)
    chunk_stats = safe_load_chunk_stats(run_dir / "rag_storage" / "kv_store_text_chunks.json")
    log_path = run_dir / "lightrag.log"
    log_text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    log_stats = analyze_logs(log_text)

    return {
        "variant": variant["name"],
        "status": status,
        "error": error_message,
        "processing_time_s": doc_meta.get("metadata", {}).get("processing_end_time", 0)
        - doc_meta.get("metadata", {}).get("processing_start_time", 0),
        "elapsed_s": round(time.time() - started, 4),
        "doc_chunks_count": doc_meta.get("chunks_count"),
        "graph_entity_count": entity_count,
        "graph_relation_count": relation_count,
        "graph_total_volume": entity_count + relation_count,
        **chunk_stats,
        "log_stats": log_stats,
        "run_dir": str(run_dir),
    }


def main() -> None:
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    results = []
    for variant in VARIANTS:
        print(f"[RUN] {variant['name']}", flush=True)
        result = run_variant(variant)
        results.append(result)
        print(
            f"      status={result['status']} chunks={result['actual_chunk_count']} contiguous={result['chunks_are_contiguous']} "
            f"raw={result['log_stats']['raw_total_volume']} graph={result['graph_total_volume']} "
            f"schema_fail={result['log_stats']['schema_parse_failed']} "
            f"repair={result['log_stats']['malformed_json_repaired']}",
            flush=True,
        )

    out_path = RUNS_ROOT / "full_service_attention_results.json"
    out_path.write_text(json.dumps({"results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] {out_path}", flush=True)


if __name__ == "__main__":
    main()
