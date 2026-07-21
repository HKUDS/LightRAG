"""Storage-neutral graph projection for API consumers.

Graph backends expose a common ``get_all_nodes``/``get_all_edges`` contract,
but their raw dictionaries differ.  The projector normalizes those records,
resolves source document IDs through the text-chunk KV storage when possible,
and applies deterministic pruning/pagination before returning the public API
schema.
"""

from __future__ import annotations

import hashlib
import base64
import json
import re
from collections import defaultdict, deque
from datetime import UTC, datetime
from typing import Any

from lightrag.api.assets import WorkspaceAssetCatalog
from lightrag.constants import GRAPH_FIELD_SEP


def _split_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        values = [str(item).strip() for item in value]
    else:
        values = [part.strip() for part in str(value).split(GRAPH_FIELD_SEP)]
    return [value for value in values if value]


def _keywords(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value or "").replace("，", ",").split(",") if item.strip()]


def _entity_id(raw: dict[str, Any]) -> str:
    return str(raw.get("id") or raw.get("entity_id") or raw.get("name") or raw.get("label") or "").strip()


def _edge_endpoints(raw: dict[str, Any]) -> tuple[str, str]:
    source = raw.get("source") or raw.get("src_id") or raw.get("source_id") or raw.get("from")
    target = raw.get("target") or raw.get("tgt_id") or raw.get("target_id") or raw.get("to")
    return str(source or "").strip(), str(target or "").strip()


_DOC_ARTIFACT_ID_RE = re.compile(r"^(doc-[0-9a-f]{32})(?:-|$)")


def _canonical_document_id(value: str) -> str:
    match = _DOC_ARTIFACT_ID_RE.match(value)
    return match.group(1) if match else value


def _relation_id(raw: dict[str, Any], source: str, target: str) -> str:
    supplied = raw.get("id") or raw.get("relation_id") or raw.get("edge_id")
    if supplied:
        return str(supplied)
    digest = hashlib.sha256(f"{source}\0{target}\0{raw.get('description', '')}".encode()).hexdigest()[:20]
    return f"rel_{digest}"


def _node_score(node: dict[str, Any], strategy: str) -> tuple[Any, ...]:
    degree = int(node.get("metadata", {}).get("degree", 0) or 0)
    if strategy == "community":
        community = str(node.get("metadata", {}).get("community", ""))
        return (community, -degree, str(node.get("id", "")))
    if strategy == "degree":
        return (-degree, str(node.get("id", "")))
    # importance: degree first, then explicit importance/weight, then name.
    importance = float(node.get("metadata", {}).get("importance", 0) or 0)
    return (-(degree + importance), -degree, str(node.get("id", "")))


def _decode_projection_cursor(value: str | None) -> tuple[int, int]:
    if not value:
        return 0, 0
    try:
        padded = value + "=" * (-len(value) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode()).decode())
        if not isinstance(payload, dict):
            raise ValueError
        return max(int(payload.get("node_offset", 0)), 0), max(int(payload.get("edge_offset", 0)), 0)
    except (ValueError, TypeError, KeyError, UnicodeError, json.JSONDecodeError, base64.binascii.Error) as exc:
        raise ValueError("cursor must be a valid opaque graph cursor") from exc


def _encode_projection_cursor(node_offset: int, edge_offset: int) -> str:
    payload = json.dumps(
        {"node_offset": max(node_offset, 0), "edge_offset": max(edge_offset, 0)},
        separators=(",", ":"),
    ).encode()
    return base64.urlsafe_b64encode(payload).decode().rstrip("=")


async def _source_documents(rag: Any, raw: dict[str, Any]) -> list[str]:
    direct: list[str] = []
    chunk_ids: list[str] = []
    for value in _split_values(raw.get("source_document_ids") or raw.get("full_doc_id")):
        if value.startswith("doc-"):
            direct.append(_canonical_document_id(value))
        else:
            chunk_ids.append(value)
    for value in _split_values(raw.get("source_id")):
        if value.startswith("doc-"):
            direct.append(_canonical_document_id(value))
        else:
            chunk_ids.append(value)
    if chunk_ids:
        storage = getattr(rag, "text_chunks", None)
        getter = getattr(storage, "get_by_ids", None)
        if callable(getter):
            try:
                chunks = await getter(list(dict.fromkeys(chunk_ids)))
                for chunk in chunks or []:
                    if isinstance(chunk, dict) and chunk.get("full_doc_id"):
                        direct.append(str(chunk["full_doc_id"]))
            except Exception:
                # Projection remains useful even when an optional KV backend
                # is temporarily unavailable; raw source IDs are still safe.
                pass
    return list(dict.fromkeys(item for item in direct if item))


async def build_graph_projection(
    rag: Any,
    asset_catalog: WorkspaceAssetCatalog,
    workspace_id: str,
    *,
    view: str = "pruned",
    max_nodes: int = 300,
    max_edges: int = 800,
    strategy: str = "importance",
    seed: str = "",
    max_depth: int = 2,
    include_media: bool = True,
    cursor: str | None = None,
) -> dict[str, Any]:
    raw_nodes = await rag.chunk_entity_relation_graph.get_all_nodes()
    raw_edges = await rag.chunk_entity_relation_graph.get_all_edges()
    nodes_by_id: dict[str, dict[str, Any]] = {}
    for raw in raw_nodes or []:
        if not isinstance(raw, dict):
            continue
        node_id = _entity_id(raw)
        if node_id:
            nodes_by_id.setdefault(node_id, raw)

    normalized_edges: list[dict[str, Any]] = []
    for raw in raw_edges or []:
        if not isinstance(raw, dict):
            continue
        source, target = _edge_endpoints(raw)
        if source and target and source in nodes_by_id and target in nodes_by_id:
            normalized_edges.append({"raw": raw, "source": source, "target": target})

    adjacency: dict[str, set[str]] = defaultdict(set)
    degree: dict[str, int] = defaultdict(int)
    for edge in normalized_edges:
        source, target = edge["source"], edge["target"]
        adjacency[source].add(target)
        adjacency[target].add(source)
        degree[source] += 1
        degree[target] += 1

    # Derive a deterministic community label for backends that do not expose
    # one.  This is intentionally lightweight (connected components), keeps
    # the endpoint storage-neutral, and lets ``strategy=community`` provide a
    # stable grouping without requiring a graph-specific algorithm.
    communities: dict[str, int] = {}
    community_index = 0
    for node_id in sorted(nodes_by_id):
        if node_id in communities:
            continue
        queue = deque([node_id])
        communities[node_id] = community_index
        while queue:
            current = queue.popleft()
            for neighbour in adjacency.get(current, ()):
                if neighbour not in communities:
                    communities[neighbour] = community_index
                    queue.append(neighbour)
        community_index += 1

    selected_ids = set(nodes_by_id)
    if seed:
        seed_id = next(
            (
                item
                for item, raw in nodes_by_id.items()
                if item == seed
                or item.lower() == seed.lower()
                or str(raw.get("name") or raw.get("entity_name") or "").lower() == seed.lower()
                or seed in {str(label) for label in raw.get("labels") or []}
            ),
            None,
        )
        if seed_id is None:
            selected_ids = set()
        else:
            selected_ids = {seed_id}
            queue = deque([(seed_id, 0)])
            while queue:
                current, depth = queue.popleft()
                if depth >= max_depth:
                    continue
                for neighbour in sorted(adjacency.get(current, ())):
                    if neighbour not in selected_ids:
                        selected_ids.add(neighbour)
                        queue.append((neighbour, depth + 1))

    normalized_nodes: list[dict[str, Any]] = []
    for node_id in selected_ids:
        raw = nodes_by_id[node_id]
        source_documents = await _source_documents(rag, raw)
        metadata = dict(raw.get("metadata") or {}) if isinstance(raw.get("metadata"), dict) else {}
        metadata["degree"] = degree.get(node_id, 0)
        if raw.get("community") is not None:
            metadata.setdefault("community", raw.get("community"))
        else:
            metadata.setdefault("community", communities.get(node_id))
        node_assets = asset_catalog.for_documents(set(source_documents)) if include_media else []
        display = dict(raw.get("display") or {"kind": "entity", "title": node_id})
        if node_assets:
            display.setdefault("asset_ids", [asset.asset_id for asset in node_assets])
        node: dict[str, Any] = {
            "id": node_id,
            "name": str(raw.get("name") or raw.get("entity_name") or node_id),
            "entity_type": str(raw.get("entity_type") or raw.get("type") or "entity"),
            "labels": list(raw.get("labels") or [node_id]),
            "description": str(raw.get("description") or ""),
            "source_document_ids": source_documents,
            "display": display,
            "metadata": metadata,
        }
        if include_media:
            node["media"] = [
                {
                    "asset_id": asset.asset_id,
                    "kind": "image",
                    "mime_type": asset.mime_type,
                    "page": asset.page,
                    "caption": asset.caption,
                }
                for asset in node_assets
            ]
        normalized_nodes.append(node)

    if view == "full":
        normalized_nodes.sort(key=lambda item: str(item.get("id", "")))
    else:
        normalized_nodes.sort(key=lambda item: _node_score(item, strategy))
    node_offset, edge_offset = _decode_projection_cursor(cursor)
    page_nodes = normalized_nodes[node_offset : node_offset + max_nodes]
    page_ids = {node["id"] for node in page_nodes}
    total_projectable_edges = sum(
        1
        for edge in normalized_edges
        if edge["source"] in selected_ids and edge["target"] in selected_ids
    )

    normalized_edges_out: list[dict[str, Any]] = []
    for edge in normalized_edges:
        if edge["source"] not in page_ids or edge["target"] not in page_ids:
            continue
        raw = edge["raw"]
        source, target = edge["source"], edge["target"]
        source_documents = await _source_documents(rag, raw)
        normalized_edges_out.append(
            {
                "id": _relation_id(raw, source, target),
                "source": source,
                "target": target,
                "relation": str(raw.get("relation") or raw.get("label") or "related_to"),
                "description": str(raw.get("description") or ""),
                "keywords": _keywords(raw.get("keywords")),
                "weight": float(raw.get("weight", 1.0) or 1.0),
                "source_document_ids": source_documents,
            }
        )
    if view == "full":
        normalized_edges_out.sort(key=lambda item: item["id"])
    else:
        normalized_edges_out.sort(key=lambda item: (-item["weight"], item["id"]))
    returned_edges = normalized_edges_out[edge_offset : edge_offset + max_edges]

    has_more_nodes = node_offset + len(page_nodes) < len(normalized_nodes)
    has_more_edges = edge_offset + len(returned_edges) < len(normalized_edges_out)
    if has_more_edges:
        next_cursor = _encode_projection_cursor(node_offset, edge_offset + max_edges)
    elif has_more_nodes:
        next_cursor = _encode_projection_cursor(node_offset + max_nodes, 0)
    else:
        next_cursor = None
    return {
        "workspace_id": workspace_id,
        "view": view,
        "revision": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "nodes": page_nodes,
        "edges": returned_edges,
        "page": {
            "next_cursor": next_cursor,
            "truncated": bool(next_cursor or has_more_edges),
        },
        "stats": {
            "total_nodes": len(normalized_nodes),
            "total_edges": total_projectable_edges,
            "returned_nodes": len(page_nodes),
            "returned_edges": len(returned_edges),
        },
    }


__all__ = ["build_graph_projection"]
