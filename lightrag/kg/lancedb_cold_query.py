"""Cold-query job: exact top-k over a tagged Lance snapshot through DuckDB.

Caller rules:
- Offline JOB: no registry role, no write path, no embedding or LLM call; the
  query vector is supplied.
- ``tag`` must be the table's latest version. DuckDB's core lance extension
  (1.4.4/2.0.1) reads only the latest version and has no tag argument, so a
  stale tag is refused rather than answered from newer rows. Publish immutable
  snapshots to their own prefix; do not export during a cold query.
- Exact scan (no ANN): cosine (1 - similarity, default) or l2; smaller is
  closer. Cosine matches the hot metric used by the recall comparison.
- Project only ``id`` and ``embedding``: ``lance_vector_search`` projects every
  column and panics on a constant-encoded column, so top-k is ordered in SQL
  over the two columns the answer needs. A fragment with exactly one row still
  panics on DuckDB 1.4.4 (2.0.1 reads it), so export with a batch size that
  leaves no single-row fragment.
- Install the extension once (``duckdb -c "INSTALL lance FROM core"``); the job runs ``LOAD lance``.
  S3/OSS needs a TYPE LANCE secret from ``storage_options_from_env`` (bucket-prefixed s3. endpoint).
- Missing table/tag, stale tag, wrong dimension or non-finite values raise
  ``ColdQueryError`` before any hit.

CLI: ``python -m lightrag.kg.lancedb_cold_query --dest DEST --tag TAG
--query-vector '[0.1, 0.2]'``; stdout is one JSON object with ``hits`` and
``latency_ms``, and failures exit non-zero with the reason on stderr.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import lancedb
import pyarrow as pa
from lancedb.table import Table

from lightrag.kg.lancedb_impl import storage_options_from_env

_TABLE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_METRIC_FUNCTIONS = {"cosine": "array_cosine_distance", "l2": "array_distance"}


class ColdQueryError(RuntimeError):
    """A cold query was refused or could not be executed."""


class ColdHit(NamedTuple):
    id: str
    distance: float


@dataclass(frozen=True)
class ColdQueryConfig:
    dest: str
    tag: str
    table: str = "chunks"
    metric: str = "cosine"
    duckdb_bin: str | None = None
    timeout_s: float = 120.0
    storage_options: Mapping[str, str] | None = None
    duckdb_setup: str = ""


def _check_config(config: ColdQueryConfig) -> None:
    if not config.dest.strip() or not config.tag.strip():
        raise ColdQueryError("dest and tag must be non-empty")
    if not _TABLE_NAME.fullmatch(config.table):
        raise ColdQueryError(f"invalid table name {config.table!r}")
    if config.metric not in _METRIC_FUNCTIONS or config.timeout_s <= 0:
        raise ColdQueryError(f"unsupported metric {config.metric!r} or timeout_s")


def _open_table(config: ColdQueryConfig) -> Table:
    try:
        options = dict(config.storage_options or {})
        return lancedb.connect(config.dest, storage_options=options).open_table(config.table)
    except Exception as exc:
        raise ColdQueryError(f"cannot open table {config.table!r} at {config.dest!r}: {exc}") from exc


def _latest_tag_version(table: Table, config: ColdQueryConfig) -> int:
    tags = table.tags.list()
    if config.tag not in tags:
        raise ColdQueryError(f"tag {config.tag!r} not found on table {config.table!r}; available: {sorted(tags)}")
    tagged = int(table.tags.get_version(config.tag))
    if tagged != int(table.version):
        raise ColdQueryError(
            f"tag {config.tag!r} is behind the latest version ({tagged} != "
            f"{table.version}); DuckDB's lance extension cannot pin a tag, so "
            "publish each snapshot to its own prefix"
        )
    return tagged


def _embedding_dim(table: Table, config: ColdQueryConfig) -> int:
    try:
        field = table.schema.field("embedding")
    except KeyError as exc:
        raise ColdQueryError(f"table {config.table!r} has no embedding column") from exc
    if not pa.types.is_fixed_size_list(field.type):
        raise ColdQueryError(f"embedding column is {field.type}, expected a fixed-size list")
    return int(field.type.list_size)


def _query_values(query_vector: Sequence[float], dim: int) -> list[float]:
    try:
        values = [float(value) for value in query_vector]
    except (TypeError, ValueError) as exc:
        raise ColdQueryError("query vector must be a sequence of numbers") from exc
    if len(values) != dim:
        raise ColdQueryError(f"query vector has dimension {len(values)}, table embedding dimension is {dim}")
    if not all(math.isfinite(value) for value in values):
        raise ColdQueryError("query vector contains a non-finite value")
    return values


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _s3_setup(options: Mapping[str, str], dest: str) -> str:
    key = options.get("aws_access_key_id") or options.get("access_key_id")
    secret = options.get("aws_secret_access_key") or options.get("access_key_secret")
    if not key or not secret:
        return ""
    bucket = dest.split("://", 1)[-1].split("/", 1)[0]
    raw = (options.get("aws_endpoint") or options.get("endpoint") or "").strip()
    scheme = "http://" if raw.startswith("http://") else "https://"
    host = raw.removeprefix("https://").removeprefix("http://")
    if options.get("access_key_id") and host:
        # The bundled lance-io uses a custom endpoint host as-is: OSS needs the s3. host with the bucket prefixed.
        host = host if host.startswith("s3.") else f"s3.{host}"
        host = host if host.startswith(f"{bucket}.") else f"{bucket}.{host}"
    token = options.get("aws_session_token") or options.get("security_token")
    parts = ["PROVIDER config", f"SCOPE {_sql_literal('s3://' + bucket + '/')}", f"ACCESS_KEY_ID {_sql_literal(key)}",
             f"SECRET_ACCESS_KEY {_sql_literal(secret)}", f"REGION {_sql_literal(options.get('aws_region') or options.get('region') or 'us-east-1')}",
             "VIRTUAL_HOSTED_STYLE_REQUEST 'true'"]
    if token:
        parts.append(f"SESSION_TOKEN {_sql_literal(token)}")
    if host:
        parts.append(f"ENDPOINT {_sql_literal(scheme + host)}")
    return ("INSTALL httpfs; LOAD httpfs; CREATE OR REPLACE SECRET "
            "(TYPE LANCE, " + ", ".join(parts) + ");")


def _build_sql(
    config: ColdQueryConfig, values: list[float], dim: int, top_k: int, setup: str
) -> str:
    # DuckDB httpfs reads OSS via the S3 protocol: scan the oss:// lake under its s3:// alias.
    path = f"{config.dest.replace('oss://', 's3://', 1).rstrip('/')}/{config.table}.lance"
    vector = "[" + ", ".join(repr(value) for value in values) + f"]::FLOAT[{dim}]"
    statements = ["LOAD lance;"]
    statements.extend(s for s in (setup, config.duckdb_setup.strip()) if s)
    statements.append(
        f"SELECT id, {_METRIC_FUNCTIONS[config.metric]}(embedding, {vector}) AS distance\n"
        f"FROM __lance_scan({_sql_literal(path)})\n"
        "WHERE embedding IS NOT NULL\nORDER BY distance ASC, id ASC\n"
        f"LIMIT {top_k};"
    )
    return "\n".join(statements) + "\n"


def _duckdb_binary(config: ColdQueryConfig) -> str:
    binary = config.duckdb_bin or os.getenv("DUCKDB_BIN") or shutil.which("duckdb")
    if not binary:
        raise ColdQueryError(
            "duckdb CLI not found; install it with INSTALL lance FROM core or set DUCKDB_BIN"
        )
    return binary


def _query_rows(binary: str, sql: str, timeout_s: float) -> list[dict]:
    try:
        proc = subprocess.run(
            [binary, "-json"],
            input=sql,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ColdQueryError(f"duckdb timed out after {timeout_s}s") from exc
    except OSError as exc:
        raise ColdQueryError(f"cannot run duckdb: {exc}") from exc
    output = proc.stdout.strip()
    if proc.returncode != 0 or not output:
        detail = re.sub(r"(?i)\b(ACCESS_KEY_ID|SECRET_ACCESS_KEY|SESSION_TOKEN|KEY_ID|SECRET)\s+'[^']*'", r"\1 '***'", (proc.stderr or output).strip()[-2000:])
        raise ColdQueryError(f"duckdb failed (exit {proc.returncode}): {detail}")
    # duckdb -json emits one JSON doc per statement (CREATE SECRET -> [{"Success":true}])
    # and may pretty-print rows across lines; decode every doc, keep the last (SELECT rows).
    decoder = json.JSONDecoder()
    rows, idx = None, 0
    while idx < len(output):
        try:
            rows, idx = decoder.raw_decode(output, idx)
        except json.JSONDecodeError:
            idx += 1
    if not isinstance(rows, list):
        raise ColdQueryError("duckdb did not return a row list")
    return rows


def cold_search(
    config: ColdQueryConfig, query_vector: Sequence[float], top_k: int
) -> list[ColdHit]:
    """Return the exact top-k hits, smallest distance first.

    The module docstring holds the caller rules (latest-version tag, one writer
    per destination, projection limits). Every refusal or failed execution
    raises ``ColdQueryError``; a partial ranking is never returned.
    """
    _check_config(config)
    if top_k < 1:
        raise ColdQueryError("top_k must be >= 1")
    table = _open_table(config)
    _latest_tag_version(table, config)
    dim = _embedding_dim(table, config)
    values = _query_values(query_vector, dim)
    setup = _s3_setup(config.storage_options or {}, config.dest) if config.dest.startswith(("s3://", "oss://")) else ""
    rows = _query_rows(
        _duckdb_binary(config), _build_sql(config, values, dim, top_k, setup), config.timeout_s
    )
    hits: list[ColdHit] = []
    for row in rows:
        try:
            hit_id, distance = str(row["id"]), float(row["distance"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ColdQueryError(f"duckdb row is missing id/distance: {row!r}") from exc
        if not math.isfinite(distance):
            raise ColdQueryError(f"non-finite distance for id {hit_id!r}")
        hits.append(ColdHit(hit_id, distance))
    return hits


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an exact top-k cold query over a tagged Lance snapshot."
    )
    parser.add_argument("--dest", required=True, help="local directory, oss:// or s3:// URI")
    parser.add_argument("--tag", required=True, help="tag at the table's latest version")
    parser.add_argument("--table", default="chunks")
    parser.add_argument("--query-vector", help="JSON list, e.g. '[0.1, 0.2]'")
    parser.add_argument("--query-vector-file", help="file containing a JSON list")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--metric", choices=sorted(_METRIC_FUNCTIONS), default="cosine")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--duckdb-bin")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    try:
        if (args.query_vector is None) == (args.query_vector_file is None):
            raise ColdQueryError("pass exactly one query vector source")
        raw = args.query_vector
        if raw is None:
            try:
                raw = Path(args.query_vector_file).read_text(encoding="utf-8")
            except OSError as exc:
                raise ColdQueryError(f"cannot read query vector file: {exc}") from exc
        try:
            vector = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ColdQueryError(f"query vector is not valid JSON: {exc}") from exc
        if not isinstance(vector, list) or not vector:
            raise ColdQueryError("query vector must be a non-empty JSON list")
        options = storage_options_from_env(args.dest) if args.dest.startswith(("s3://", "oss://")) else None
        config = ColdQueryConfig(
            args.dest, args.tag, args.table, args.metric,
            args.duckdb_bin, args.timeout, options,
        )
        started = time.perf_counter()
        hits = cold_search(config, vector, args.top_k)
        latency_ms = (time.perf_counter() - started) * 1000
    except ColdQueryError as exc:
        sys.stderr.write(f"cold-query error: {exc}\n")
        raise SystemExit(1) from exc
    result = {"dest": args.dest, "table": args.table, "tag": args.tag, "metric": args.metric,
              "top_k": args.top_k, "latency_ms": round(latency_ms, 3),
              "hits": [hit._asdict() for hit in hits]}
    sys.stdout.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()