"""Read-only offline vector inventory; see docs/design/VectorCensus.md."""

from __future__ import annotations

from typing import Any

from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.exceptions import VectorSpaceMismatchError
from lightrag.kg import file_fingerprint
from lightrag.utils import (
    compute_mdhash_id,
    has_chunk_tracking_row,
    make_relation_chunk_key,
    make_relation_vdb_ids,
)

_LIMIT = 20
_BUFFERS = (
    "_pending_upserts",
    "_pending_deletes",
    "_unsaved_upserts",
    "_unsaved_deletes",
)


def _unavailable(reason: str, status: str = "unavailable") -> dict[str, Any]:
    return {"status": status, "reason": reason}


def _pending(storage) -> bool:
    return any(getattr(storage, name) for name in _BUFFERS) or storage._client_dirty


async def _nano_inventory(storage):
    if storage is None:
        return None, None, _unavailable("storage_not_supplied")
    if getattr(storage, "persists_vectors", True) is False:
        return None, None, _unavailable("vectors_not_persisted", "not_applicable")
    # Subclasses may change visibility: only this implementation is certified.
    if (
        type(storage).__name__ != "NanoVectorDBStorage"
        or type(storage).__module__ != "lightrag.kg.nano_vector_db_impl"
    ):
        return (
            None,
            None,
            _unavailable(
                f"complete_population_not_verified_for_{type(storage).__name__}"
            ),
        )
    try:
        if _pending(storage):
            return None, None, _unavailable("pending_changes")
        before = storage._stat_fingerprint()
        if before is file_fingerprint.UNREADABLE or any(v is None for v in before):
            return None, None, _unavailable("namespace_missing_or_unreadable")
        snapshot = await storage.client_storage
        # No await while retaining the live storage reference.
        rows = snapshot["data"]
        if not isinstance(rows, list):
            return None, None, _unavailable("malformed_inventory")
        ids = set()
        for row in rows:
            value = row.get("__id__") if isinstance(row, dict) else None
            if not isinstance(value, str) or not value or value in ids:
                return None, None, _unavailable("missing_or_duplicate_vector_id")
            ids.add(value)
        if snapshot["matrix"].shape[0] != len(rows):
            return None, None, _unavailable("metadata_vector_count_mismatch")
        if storage._loaded_fingerprint != before or _pending(storage):
            return None, None, _unavailable("snapshot_changed", "inconclusive")
        return ids, before, None
    except VectorSpaceMismatchError as exc:
        return None, None, _unavailable(str(exc), "incompatible")
    except Exception as exc:
        return (
            None,
            None,
            _unavailable(f"inventory_read_failed: {type(exc).__name__}: {exc}"),
        )


async def _rows(storage, keys: list[str], batch_size: int) -> dict[str, dict]:
    result = {}
    for start in range(0, len(keys), batch_size):
        batch = keys[start : start + batch_size]
        rows = await storage.get_by_ids(batch)
        if len(rows) != len(batch):
            raise RuntimeError("Incomplete source/provenance batch")
        for key, row in zip(batch, rows):
            if row is not None and not isinstance(row, dict):
                raise RuntimeError("Malformed source/provenance row")
            result[key] = row
    return result


async def audit_vector_census(
    nodes,
    edges,
    entities_vdb,
    relationships_vdb,
    *,
    chunks_vdb=None,
    text_chunks=None,
    entity_chunks=None,
    relation_chunks=None,
    batch_size: int = 500,
    incompatible=None,
) -> dict[str, Any]:
    """Census an idle offline workspace, without flushing or repairing it.

    Only completed Nano inventories support coverage/reverse conclusions.
    Other backends report unavailable, never an inferred zero. Call with one
    shared graph enumeration; full materialization and fingerprint limits are
    documented in docs/design/VectorCensus.md. All writers must be stopped.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    incompatible = incompatible or {}
    stores = {
        "entities": entities_vdb,
        "relationships": relationships_vdb,
        "chunks": chunks_vdb,
    }
    inventories, fingerprints, report = {}, {}, {}
    for label, storage in stores.items():
        if label in incompatible:
            report[label] = _unavailable(incompatible[label], "incompatible")
            continue
        ids, fingerprint, failure = await _nano_inventory(storage)
        if failure:
            report[label] = failure
        else:
            inventories[label], fingerprints[label] = ids, fingerprint

    entity_records = {}
    relation_records = {}
    for node in nodes:
        name = node.get("entity_id") or node.get("id")
        if not isinstance(name, str) or not name.strip():
            report["entities"] = _unavailable("malformed_graph_source")
            continue
        entity_records[name] = node
    for edge in edges:
        src, tgt = edge.get("source"), edge.get("target")
        if not all(isinstance(v, str) and v.strip() for v in (src, tgt)):
            report["relationships"] = _unavailable("malformed_graph_source")
            continue
        relation_records.setdefault(tuple(sorted((src, tgt))), edge)
    items = {
        "entities": [
            (name, (compute_mdhash_id(name, prefix="ent-"),)) for name in entity_records
        ],
        "relationships": [
            (pair, tuple(make_relation_vdb_ids(*pair))) for pair in relation_records
        ],
    }
    chunk_rows = None
    try:
        if text_chunks is None:
            raise ValueError("text_chunks_not_supplied")
        from lightrag.tools.rebuild_vdb import enumerate_kv_keys

        keys = await enumerate_kv_keys(text_chunks)
        if any(not isinstance(key, str) or not key for key in keys) or len(
            set(keys)
        ) != len(keys):
            raise ValueError("Invalid or duplicate source chunk IDs")
        chunk_rows = await _rows(text_chunks, keys, batch_size)
        if any(row is None for row in chunk_rows.values()):
            raise RuntimeError("Source chunk disappeared during census")
        items["chunks"] = [(key, (key,)) for key in keys]
    except Exception as exc:
        report.setdefault("chunks", _unavailable(f"source_inventory_failed: {exc}"))

    missing_by_label = {}
    for label, source in items.items():
        if label in report or label not in inventories:
            continue
        ids = inventories[label]
        source_ids = {value for _, candidates in source for value in candidates}
        missing = [
            key for key, candidates in source if not ids.intersection(candidates)
        ]
        matched = ids & source_ids
        reverse = ids - source_ids
        report[label] = {
            "status": "complete",
            "source_total": len(source),
            "vector_total": len(ids),
            "matched_physical": len(matched),
            "missing": len(missing),
            "reverse_excess": len(reverse),
            "legacy_duplicates": sum(
                max(0, len(ids.intersection(candidates)) - 1)
                for _, candidates in source
            ),
            "missing_examples": sorted(missing)[:_LIMIT],
            "reverse_examples": sorted(reverse)[:_LIMIT],
        }
        missing_by_label[label] = missing

    # Attribute only source-present missing vectors. A reverse ID proves no doc ownership.
    for label, missing in missing_by_label.items():
        target = report[label]
        docs, unknown = {}, 0
        try:
            if chunk_rows is None:
                raise RuntimeError("complete chunk provenance unavailable")
            tracking = None
            if label == "entities" and entity_chunks is not None:
                tracking = await _rows(entity_chunks, missing, batch_size)
            elif label == "relationships" and relation_chunks is not None:
                tracking = await _rows(
                    relation_chunks,
                    [make_relation_chunk_key(*pair) for pair in missing],
                    batch_size,
                )
            for key in missing:
                if label == "chunks":
                    sources = [key]
                else:
                    record = (
                        entity_records[key]
                        if label == "entities"
                        else relation_records[key]
                    )
                    tracking_key = (
                        key if label == "entities" else make_relation_chunk_key(*key)
                    )
                    tracked = (
                        tracking.get(tracking_key) if tracking is not None else None
                    )
                    if has_chunk_tracking_row(tracked):
                        sources = tracked["chunk_ids"]
                    elif tracked is not None:
                        raise RuntimeError("malformed tracking row")
                    else:
                        sources = (record.get("source_id") or "").split(GRAPH_FIELD_SEP)
                owners = {
                    chunk_rows[c].get("full_doc_id") for c in sources if c in chunk_rows
                }
                owners.discard(None)
                owners.discard("")
                if not owners:
                    unknown += 1
                for owner in owners:
                    docs[owner] = docs.get(owner, 0) + 1
            target["missing_by_document"] = docs
            target["unattributed_missing"] = unknown
        except Exception as exc:
            target["document_attribution"] = _unavailable(str(exc))

    # Bracket the whole census, including source and provenance reads.
    for label, before in fingerprints.items():
        if report.get(label, {}).get("status") != "complete":
            continue
        try:
            after = stores[label]._stat_fingerprint()
            changed = (
                after is file_fingerprint.UNREADABLE
                or after != before
                or _pending(stores[label])
            )
        except Exception:
            changed = True
        if changed:
            report[label] = _unavailable(
                "snapshot_changed_or_unreadable", "inconclusive"
            )
    complete = [r for r in report.values() if r["status"] == "complete"]
    return {
        "targets": report,
        "complete": all(
            r["status"] in ("complete", "not_applicable") for r in report.values()
        ),
        "coverage_gaps": any(r["missing"] or r["reverse_excess"] for r in complete),
        "interpretation": "Coverage gaps may be accepted lag/residue, not proof of data loss. Recheck offline before a separately confirmed rebuild.",
    }


def print_vector_census(census: dict[str, Any]) -> None:
    print("Vector census (offline, no repair):")
    for label, row in census["targets"].items():
        if row["status"] == "complete":
            print(
                f"  {label}: source={row['source_total']}, vectors={row['vector_total']}, "
                f"missing={row['missing']}, reverse={row['reverse_excess']}, legacy duplicates={row['legacy_duplicates']}"
            )
        else:
            print(f"  {label}: {row['status']} ({row['reason']})")
    print(census["interpretation"])
    print(
        "Repair, if needed: existing rebuild tool targets; this census performs no writes."
    )
