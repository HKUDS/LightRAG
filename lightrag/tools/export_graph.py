#!/usr/bin/env python3
"""Stream a workspace's knowledge graph to files, without loading it whole.

Reads the graph storage only, through the bounded-iteration API every
built-in graph backend implements (``iter_labels`` / ``get_nodes_batch`` /
``iter_edges``), and writes each batch as soon as it is read:

- ``jsonl`` (default): ``graph.jsonl``, one header line, then one object per
  node and per edge with every stored property. The format meant for re-import.
- ``csv``: ``nodes.csv`` and ``edges.csv``, the LightRAG fields as columns.
- ``graphml``: ``graph.graphml``, for Gephi, yEd and NetworkX.

Rules a caller must know:

- A backend without ``iter_labels`` / ``iter_edges`` is refused
  (``StorageCapabilityError``); there is no fallback to ``get_all_nodes``,
  which would load the graph into memory.
- Labels are spooled to a private temporary SQLite file before any node is
  read, so the label cursor is closed when ``get_nodes_batch`` asks for a
  connection (a pool of one would otherwise time out). The same file answers
  the dangling-endpoint check for edges.
- Nodes and edges are read in two passes with no snapshot across them. Stop
  writers for a consistent export; otherwise the summary reports the nodes
  that vanished between the passes and the edges whose endpoint was not
  exported.
- Output files are written under a ``.partial`` name and renamed only when
  the whole export succeeded; an existing output is refused unless
  ``--overwrite`` is given.

See ``lightrag/tools/README_EXPORT_GRAPH.md`` for formats and examples.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sqlite3
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Iterable, TextIO
from xml.sax.saxutils import escape, quoteattr

FORMAT_NAME = "lightrag-graph"
FORMAT_VERSION = 1
DEFAULT_BATCH_SIZE = 500
DEFAULT_OUTPUT_DIR = "./graph_export"
MAX_REPORTED_EXAMPLES = 20

FORMATS = ("jsonl", "csv", "graphml")
OUTPUT_FILES = {
    "jsonl": ("graph.jsonl",),
    "csv": ("nodes.csv", "edges.csv"),
    "graphml": ("graph.graphml",),
}

# The fixed LightRAG schema used as CSV columns and GraphML keys. Any other
# property goes, JSON-encoded, into the ``extra`` column / key.
NODE_FIELDS = (
    "entity_id",
    "entity_type",
    "description",
    "source_id",
    "file_path",
    "created_at",
)
EDGE_FIELDS = (
    "source",
    "target",
    "weight",
    "description",
    "keywords",
    "source_id",
    "file_path",
    "created_at",
)


class ExportError(RuntimeError):
    """A refusal the operator can act on; reported without a traceback."""


@dataclass
class ExportReport:
    graph_storage: str
    workspace: str
    format: str
    files: list[str] = field(default_factory=list)
    nodes: int = 0
    edges: int = 0
    duplicate_labels: int = 0
    duplicate_edge_pairs: int = 0
    vanished_nodes: list[str] = field(default_factory=list)
    dangling_edges: list[tuple[str, str]] = field(default_factory=list)
    dangling_edge_count: int = 0
    vanished_node_count: int = 0
    replaced_xml_chars: int = 0

    @property
    def consistent(self) -> bool:
        return not (self.vanished_node_count or self.dangling_edge_count)

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "graph_storage": self.graph_storage,
            "workspace": self.workspace,
            "format": self.format,
            "files": self.files,
            "nodes": self.nodes,
            "edges": self.edges,
            "consistent": self.consistent,
        }
        if self.duplicate_labels:
            out["duplicate_labels_skipped"] = self.duplicate_labels
        if self.duplicate_edge_pairs:
            out["duplicate_edge_pairs_kept"] = self.duplicate_edge_pairs
        if self.vanished_node_count:
            out["vanished_nodes"] = self.vanished_node_count
            out["vanished_node_examples"] = self.vanished_nodes
        if self.dangling_edge_count:
            out["dangling_edges"] = self.dangling_edge_count
            out["dangling_edge_examples"] = [list(e) for e in self.dangling_edges]
        if self.replaced_xml_chars:
            out["graphml_replaced_xml_chars"] = self.replaced_xml_chars
        return out


# --------------------------------------------------------------------------
# Label spool: drains iter_labels, then replays labels and answers membership.
# --------------------------------------------------------------------------


class LabelSpool:
    """Private on-disk set of labels, in first-seen order.

    SQLite rather than a flat file because the edge pass needs a membership
    test per endpoint, and a Python set of every label is exactly the memory
    growth this tool exists to avoid.
    """

    def __init__(self) -> None:
        self._dir = tempfile.mkdtemp(prefix="lightrag-export-")
        self._path = os.path.join(self._dir, "labels.sqlite")
        self._db = sqlite3.connect(self._path)
        self._db.execute(
            "CREATE TABLE labels (seq INTEGER PRIMARY KEY, label TEXT UNIQUE NOT NULL,"
            " exported INTEGER NOT NULL DEFAULT 0)"
        )
        self._db.execute(
            "CREATE TABLE pairs (a TEXT NOT NULL, b TEXT NOT NULL, PRIMARY KEY (a, b))"
            " WITHOUT ROWID"
        )

    def add(self, labels: Iterable[str]) -> int:
        """Insert labels, returning how many were duplicates."""
        labels = list(labels)
        before = self._db.total_changes
        self._db.executemany(
            "INSERT OR IGNORE INTO labels (label) VALUES (?)", ((x,) for x in labels)
        )
        return len(labels) - (self._db.total_changes - before)

    def batches(self, batch_size: int) -> Iterable[list[str]]:
        last = 0
        while True:
            rows = self._db.execute(
                "SELECT seq, label FROM labels WHERE seq > ? ORDER BY seq LIMIT ?",
                (last, batch_size),
            ).fetchall()
            if not rows:
                return
            last = rows[-1][0]
            yield [label for _, label in rows]

    def mark_exported(self, labels: Iterable[str]) -> None:
        self._db.executemany(
            "UPDATE labels SET exported = 1 WHERE label = ?", ((x,) for x in labels)
        )

    def exported_subset(self, labels: Iterable[str]) -> set[str]:
        wanted = list(set(labels))
        found: set[str] = set()
        # SQLite's default host-parameter limit is 999 on old builds.
        for i in range(0, len(wanted), 900):
            chunk = wanted[i : i + 900]
            marks = ",".join("?" * len(chunk))
            found.update(
                row[0]
                for row in self._db.execute(
                    f"SELECT label FROM labels WHERE exported = 1 AND label IN ({marks})",
                    chunk,
                )
            )
        return found

    def add_pairs(self, edges: Iterable[dict]) -> int:
        """Record each edge's undirected pair, returning how many were seen before.

        Graphs are undirected: a second row for the same pair (``A->B`` beside
        ``B->A``, which a directed edge table such as Apache AGE's can hold) is
        written as stored, and only counted here.
        """
        pairs = [
            (min(s, t), max(s, t))
            for s, t in ((str(e["source"]), str(e["target"])) for e in edges)
        ]
        before = self._db.total_changes
        self._db.executemany("INSERT OR IGNORE INTO pairs (a, b) VALUES (?, ?)", pairs)
        return len(pairs) - (self._db.total_changes - before)

    def close(self) -> None:
        self._db.close()
        for name in os.listdir(self._dir):
            os.remove(os.path.join(self._dir, name))
        os.rmdir(self._dir)


# --------------------------------------------------------------------------
# Value helpers shared by the CSV and GraphML writers.
# --------------------------------------------------------------------------


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value) if isinstance(value, float) else str(value)
    return json.dumps(value, ensure_ascii=False, default=str)


def split_node(label: str, props: dict[str, Any]) -> tuple[dict[str, Any], dict]:
    """Fixed node fields and the remaining properties.

    ``entity_id`` is the label; a stored ``entity_id`` property that differs
    from it is kept in ``extra`` rather than dropped.
    """
    props = dict(props)
    fixed: dict[str, Any] = {"entity_id": label}
    if props.get("entity_id", label) == label:
        props.pop("entity_id", None)
    for name in NODE_FIELDS[1:]:
        if name in props:
            fixed[name] = props.pop(name)
    return fixed, props


def split_edge(edge: dict[str, Any]) -> tuple[dict[str, Any], dict]:
    props = dict(edge)
    fixed = {name: props.pop(name) for name in EDGE_FIELDS if name in props}
    return fixed, props


def _json_line(obj: dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str) + "\n"


# --------------------------------------------------------------------------
# Writers. Each takes open .partial streams and is fed batch by batch.
# --------------------------------------------------------------------------


class JsonlWriter:
    def __init__(self, streams: list[TextIO], header: dict[str, Any]) -> None:
        (self._out,) = streams
        self._out.write(_json_line(header))

    def nodes(self, rows: list[tuple[str, dict]]) -> None:
        self._out.writelines(
            _json_line({"type": "node", "id": label, "properties": props})
            for label, props in rows
        )

    def edges(self, rows: list[dict]) -> None:
        for edge in rows:
            props = dict(edge)
            source, target = props.pop("source"), props.pop("target")
            self._out.write(
                _json_line(
                    {
                        "type": "edge",
                        "source": source,
                        "target": target,
                        "properties": props,
                    }
                )
            )

    def close(self) -> None:
        pass


class CsvWriter:
    def __init__(self, streams: list[TextIO], header: dict[str, Any]) -> None:
        nodes_out, edges_out = streams
        self._nodes = csv.writer(nodes_out)
        self._edges = csv.writer(edges_out)
        self._nodes.writerow([*NODE_FIELDS, "extra"])
        self._edges.writerow([*EDGE_FIELDS, "extra"])

    @staticmethod
    def _row(fields: tuple[str, ...], fixed: dict, extra: dict) -> list[str]:
        return [_cell(fixed.get(name)) for name in fields] + [
            json.dumps(extra, ensure_ascii=False, default=str) if extra else ""
        ]

    def nodes(self, rows: list[tuple[str, dict]]) -> None:
        self._nodes.writerows(
            self._row(NODE_FIELDS, *split_node(label, props)) for label, props in rows
        )

    def edges(self, rows: list[dict]) -> None:
        self._edges.writerows(self._row(EDGE_FIELDS, *split_edge(e)) for e in rows)

    def close(self) -> None:
        pass


def _xml_safe(text: str) -> tuple[str, int]:
    """Replace the code points XML 1.0 cannot carry, counting them."""
    replaced = 0
    out = []
    for ch in text:
        cp = ord(ch)
        if (
            cp in (0x9, 0xA, 0xD)
            or 0x20 <= cp <= 0xD7FF
            or 0xE000 <= cp <= 0xFFFD
            or 0x10000 <= cp <= 0x10FFFF
        ):
            out.append(ch)
        else:
            out.append("\ufffd")
            replaced += 1
    return ("".join(out), replaced) if replaced else (text, 0)


class GraphmlWriter:
    """One-pass GraphML: keys are the fixed schema plus ``extra``."""

    def __init__(self, streams: list[TextIO], header: dict[str, Any]) -> None:
        (self._out,) = streams
        self.replaced = 0
        out = self._out
        out.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        out.write('<graphml xmlns="http://graphml.graphdrawing.org/xmlns">\n')
        for name in NODE_FIELDS[1:]:
            out.write(
                f'  <key id="n_{name}" for="node" attr.name="{name}" attr.type="string"/>\n'
            )
        out.write(
            '  <key id="n_extra" for="node" attr.name="extra" attr.type="string"/>\n'
        )
        for name in EDGE_FIELDS[2:]:
            kind = "double" if name == "weight" else "string"
            out.write(
                f'  <key id="e_{name}" for="edge" attr.name="{name}" attr.type="{kind}"/>\n'
            )
        out.write(
            '  <key id="e_extra" for="edge" attr.name="extra" attr.type="string"/>\n'
        )
        desc = escape(json.dumps(header, ensure_ascii=False, default=str))
        out.write(
            f'  <graph id="G" edgedefault="undirected">\n    <desc>{desc}</desc>\n'
        )

    def _text(self, value: str) -> str:
        safe, n = _xml_safe(value)
        self.replaced += n
        return safe

    def _data(self, prefix: str, fixed: dict, extra: dict, names) -> str:
        parts = []
        for name in names:
            if name not in fixed or fixed[name] is None:
                continue
            value = fixed[name]
            if name == "weight":
                try:
                    value = repr(float(value))
                except (TypeError, ValueError):
                    extra = {**extra, "weight": value}
                    continue
            parts.append(
                f'<data key="{prefix}_{name}">{escape(self._text(_cell(value)))}</data>'
            )
        if extra:
            blob = json.dumps(extra, ensure_ascii=False, default=str)
            parts.append(
                f'<data key="{prefix}_extra">{escape(self._text(blob))}</data>'
            )
        return "".join(parts)

    def nodes(self, rows: list[tuple[str, dict]]) -> None:
        for label, props in rows:
            fixed, extra = split_node(label, props)
            data = self._data("n", fixed, extra, NODE_FIELDS[1:])
            node_id = quoteattr(self._text(label))
            self._out.write(f"    <node id={node_id}>{data}</node>\n")

    def edges(self, rows: list[dict]) -> None:
        for edge in rows:
            fixed, extra = split_edge(edge)
            data = self._data("e", fixed, extra, EDGE_FIELDS[2:])
            src = quoteattr(self._text(str(fixed["source"])))
            tgt = quoteattr(self._text(str(fixed["target"])))
            self._out.write(f"    <edge source={src} target={tgt}>{data}</edge>\n")

    def close(self) -> None:
        self._out.write("  </graph>\n</graphml>\n")


WRITERS = {"jsonl": JsonlWriter, "csv": CsvWriter, "graphml": GraphmlWriter}


# --------------------------------------------------------------------------
# The export itself, driven only through the storage object it is handed.
# --------------------------------------------------------------------------


async def _check_capability(batches: AsyncIterator[Any]) -> Any:
    """First batch of an iterator, turning a missing capability into a refusal."""
    from lightrag.exceptions import StorageCapabilityError

    try:
        return await anext(batches)
    except StopAsyncIteration:
        return None
    except StorageCapabilityError as exc:
        raise ExportError(
            f"{exc}; this tool needs iter_labels() and iter_edges() and does not "
            "fall back to loading the whole graph"
        ) from exc


async def _drain(batches: AsyncIterator[Any]):
    first = await _check_capability(batches)
    if first is None:
        return
    yield first
    async for batch in batches:
        yield batch


def _output_paths(output_dir: str, fmt: str) -> list[str]:
    return [os.path.join(output_dir, name) for name in OUTPUT_FILES[fmt]]


async def export_graph(
    storage: Any,
    output_dir: str,
    *,
    fmt: str = "jsonl",
    batch_size: int = DEFAULT_BATCH_SIZE,
    overwrite: bool = False,
    graph_storage_name: str | None = None,
) -> ExportReport:
    """Export ``storage`` to ``output_dir`` and return what was written.

    Raises :class:`ExportError` for a refusal (unsupported backend, existing
    output without ``overwrite``); on any failure the ``.partial`` files are
    removed and no final file is created or replaced.
    """
    if fmt not in WRITERS:
        raise ExportError(f"unknown format {fmt!r}; choose one of {', '.join(FORMATS)}")
    if batch_size < 1:
        raise ExportError("batch size must be >= 1")
    finals = _output_paths(output_dir, fmt)
    existing = [p for p in finals if os.path.exists(p)]
    if existing and not overwrite:
        raise ExportError(
            f"output already exists: {', '.join(existing)}; pass --overwrite to replace it"
        )
    os.makedirs(output_dir, exist_ok=True)

    report = ExportReport(
        graph_storage=graph_storage_name or type(storage).__name__,
        workspace=getattr(storage, "workspace", "") or "",
        format=fmt,
    )
    header = {
        "type": "header",
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "graph_storage": report.graph_storage,
        "workspace": report.workspace,
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    partials = [p + ".partial" for p in finals]
    streams: list[TextIO] = []
    spool = LabelSpool()
    try:
        # Phase 1: drain labels, closing the backend cursor before node reads.
        async for labels in _drain(storage.iter_labels(batch_size)):
            report.duplicate_labels += spool.add(labels)

        newline = "" if fmt == "csv" else None
        streams = [open(p, "w", encoding="utf-8", newline=newline) for p in partials]
        writer = WRITERS[fmt](streams, header)

        # Phase 2: node details, one get_nodes_batch per spooled batch.
        for labels in spool.batches(batch_size):
            got = await storage.get_nodes_batch(labels)
            rows = [(label, got[label]) for label in labels if label in got]
            for label in labels:
                if label not in got:
                    report.vanished_node_count += 1
                    if len(report.vanished_nodes) < MAX_REPORTED_EXAMPLES:
                        report.vanished_nodes.append(label)
            writer.nodes(rows)
            spool.mark_exported(label for label, _ in rows)
            report.nodes += len(rows)

        # Phase 3: edges, complete rows, checked against the exported nodes.
        async for edges in _drain(storage.iter_edges(batch_size)):
            writer.edges(edges)
            report.edges += len(edges)
            report.duplicate_edge_pairs += spool.add_pairs(edges)
            endpoints = {e["source"] for e in edges} | {e["target"] for e in edges}
            missing = endpoints - spool.exported_subset(endpoints)
            if missing:
                for edge in edges:
                    if edge["source"] in missing or edge["target"] in missing:
                        report.dangling_edge_count += 1
                        if len(report.dangling_edges) < MAX_REPORTED_EXAMPLES:
                            report.dangling_edges.append(
                                (edge["source"], edge["target"])
                            )

        writer.close()
        report.replaced_xml_chars = getattr(writer, "replaced", 0)
        for stream in streams:
            stream.close()
        for partial, final in zip(partials, finals):
            os.replace(partial, final)
        report.files = finals
        return report
    finally:
        for stream in streams:
            stream.close()
        for partial in partials:
            if os.path.exists(partial):
                os.remove(partial)
        spool.close()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


async def _build_graph_storage(backend_name: str, workspace: str) -> Any:
    """Instantiate and initialize the configured graph storage.

    Imports stay inside the function so importing this module never pulls in
    a database driver.
    """
    from lightrag.constants import DEFAULT_WORKING_DIR
    from lightrag.kg import verify_storage_implementation
    from lightrag.kg.factory import get_storage_class
    from lightrag.namespace import NameSpace
    from lightrag.utils import EmbeddingFunc

    verify_storage_implementation("GRAPH_STORAGE", backend_name)

    async def _never_embed(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("graph export never embeds")

    cls = get_storage_class(backend_name)
    storage = cls(
        namespace=NameSpace.GRAPH_STORE_CHUNK_ENTITY_RELATION,
        workspace=workspace,
        global_config={
            "working_dir": os.getenv("WORKING_DIR", DEFAULT_WORKING_DIR),
            "graph_storage": backend_name,
            "workspace": workspace,
        },
        embedding_func=EmbeddingFunc(embedding_dim=1, func=_never_embed),
    )
    try:
        await storage.initialize()
    except BaseException:
        try:
            await storage.finalize()
        except Exception:
            pass
        raise
    return storage


def _workspace_argument(value: str) -> str:
    from lightrag.utils import validate_workspace

    try:
        return validate_workspace(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


async def _async_main(args: argparse.Namespace) -> int:
    from lightrag.kg.shared_storage import initialize_share_data
    from lightrag.utils import normalize_server_workspace

    backend = args.graph_storage or os.getenv(
        "LIGHTRAG_GRAPH_STORAGE", "NetworkXStorage"
    )
    workspace = (
        args.workspace
        if args.workspace is not None
        else normalize_server_workspace(os.getenv("WORKSPACE", "")) or ""
    )
    # Refuse before connecting: an existing output would fail the run anyway.
    existing = [p for p in _output_paths(args.output, args.format) if os.path.exists(p)]
    if existing and not args.overwrite:
        print(
            json.dumps(
                {
                    "error": f"output already exists: {', '.join(existing)}; "
                    "pass --overwrite to replace it"
                }
            )
        )
        return 2

    initialize_share_data(workers=1)
    try:
        storage = await _build_graph_storage(backend, workspace)
    except Exception as exc:
        print(json.dumps({"error": f"graph storage initialization failed: {exc}"}))
        return 1
    try:
        report = await export_graph(
            storage,
            args.output,
            fmt=args.format,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
            graph_storage_name=backend,
        )
    except ExportError as exc:
        print(json.dumps({"error": str(exc)}))
        return 2
    finally:
        await storage.finalize()
    print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    if not report.consistent:
        print(
            "warning: the graph changed during the export (see vanished_nodes / "
            "dangling_edges); stop writers and re-run for a consistent copy"
        )
    return 0


def main(argv: list[str] | None = None) -> None:
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(
        description=(
            "Export a workspace's knowledge graph (graph storage only) to "
            "jsonl, csv or graphml, streaming batch by batch."
        )
    )
    parser.add_argument(
        "--format",
        choices=FORMATS,
        default="jsonl",
        help="Output format (default: jsonl)",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--workspace",
        default=None,
        type=_workspace_argument,
        help="Exact workspace name; use an empty string for the default workspace. "
        "If omitted, WORKSPACE is normalized as by the server.",
    )
    parser.add_argument(
        "--graph-storage",
        default=None,
        help="Graph storage class (default: LIGHTRAG_GRAPH_STORAGE, else NetworkXStorage)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Labels / edges per storage call (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing output files"
    )
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    load_dotenv(dotenv_path=".env", override=False)
    exit_code = asyncio.run(_async_main(args))
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
