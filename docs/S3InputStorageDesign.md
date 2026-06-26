# Design Proposal: Backing the Entire INPUT_DIR with S3 (for Distributed Deployment)

> Status: Draft / Pending review
> Related: PR #3331 (Upload sidecar image assets to S3), Issue #3315
> Applies to: based on `main` @ `40a62c9`
> Chinese version: `docs/S3InputStorageDesign-zh.md`

## 1. Background and Motivation

LightRAG's `INPUT_DIR` (default `./inputs`) is today a **server-local directory** that carries three completely different responsibilities:

1. **User-uploaded source files**: `/documents/upload` streams bytes to `input_dir/<filename>`.
2. **Parser-engine intermediates / caches**: external parser raw bundles `*.mineru_raw/`, `*.docling_raw/`, and native's `*.native_raw/`.
3. **Post-parse sidecar artifacts and archived sources**: `__parsed__/<name>.parsed/` (`*.blocks.jsonl`, `*.drawings.json`, `*.tables.json`, `*.equations.json`, `*.blocks.assets/`) plus the source file moved under `__parsed__/` once processing completes.

This "everything lands on local disk" design has three fundamental problems:

- **It blocks distributed / multi-replica deployment.** Multiple API / worker replicas each have their own local disk. An upload received by replica A, or a sidecar written by A, is invisible to replica B. `/scan`, dedup (by filename / content hash), deletion, and query-time traceability all become inconsistent across replicas.
- **Query results cannot trace back to the original file / images for clients.** Sidecars record image assets as `file://` local paths (see `sidecar_uri_for` in `utils_pipeline.py`); a browser cannot reach the server filesystem. This is exactly the pain point of Issue #3315.
- **Data lifecycle is coupled to container lifecycle.** Rebuilding a container/Pod loses data, so a PVC is required (see `k8s-deploy/`), which is operationally expensive and hard to scale horizontally.

### 1.1 Why PR #3331 Is Not Enough

PR #3331's approach: after the sidecar is written to disk (Stage 1b), it additionally uploads the **already-materialized images** in `*.blocks.assets/` to S3, and writes the remote URL into the `blocks.jsonl` placeholder and the `remote_url` field of `drawings.json`.

It solves the narrow "show images in the frontend" problem, but **does not change the nature of INPUT_DIR**:

- Uploaded source files still live only on local disk;
- Parser intermediates (mineru_raw / docling_raw) still live only on local disk;
- The sidecar `blocks.jsonl` / per-modality JSON still live only on local disk, and VLM still hard-depends on "reading images from local disk" (the PR comment states explicitly: `local path is always preserved so VLM analysis reads from disk`);
- `/scan`, dedup, deletion, and archiving are still entirely based on local `iterdir/glob/unlink/rmtree`.

Therefore it cannot support distributed deployment, nor "any replica can trace back any document's source and assets." The goal of this proposal is to **abstract the entire INPUT_DIR into a pluggable object-storage backend**, where local disk is just one implementation and S3 is the distribution-oriented one.

## 2. Goals and Non-Goals

**Goals**

- G1: All data carried by `INPUT_DIR` (source files, intermediates, sidecar artifacts, archived sources) is read/written through a unified **storage abstraction layer**, whose backend can switch between "local filesystem" and "S3-compatible object storage."
- G2: Zero behavior change, zero config by default — with S3 disabled, behavior is byte-for-byte identical to today (backward compatible).
- G3: Under multiple replicas, any replica can read uploads / sidecars / archives written by any other; dedup, scan, deletion, and query-time traceability are all consistent.
- G4: Query results can securely trace back the source and image assets to the client (presigned URL or server-side proxy), without exposing server-local files to clients.
- G5: Provide a migration path from an existing local deployment to S3, and keep historical `file://` URIs already written to `full_docs.sidecar_location` resolvable.

**Non-Goals**

- No changes to the four **structured-storage** backends KV / Vector / Graph / DocStatus (they already have their own pluggable system, see `kg/`). This proposal targets only **file / blob storage**.
- S3 is not mandatory; the local backend remains the default and the single-node recommendation.
- This proposal does not implement object-storage-side lifecycle / versioning / encryption policy (left to S3's own configuration).

## 3. Current-State Inventory: Local Filesystem Touchpoints

The list below enumerates every touchpoint that must be taken over by the abstraction layer (each verified against source), grouped by responsibility. This is the "work list" for the refactor.

### 3.1 Config and Path Resolution
- `lightrag/api/config.py:274-277,487-488` — `--input-dir` argument, `abspath` resolution.
- `lightrag/utils_pipeline.py:540-542` — `configured_input_dir()`, reads the `INPUT_DIR` env var.
- `lightrag/utils_pipeline.py:634-676` — `input_dir_path()` / `parsed_dir()` / `parsed_artifact_dir_for()`, path assembly.
- `lightrag/api/routers/document_routes.py:977-998` — `DocumentManager.__init__` joins `base_input_dir / workspace` and `mkdir`s it.

### 3.2 Upload to Disk
- `document_routes.py:118-161` — `sanitize_filename` (path-traversal check).
- `document_routes.py:2692,2715-2729` — target path computation + duplicate-name precheck (`exists` / `iterdir` in `find_existing_file_by_file_path`).
- `document_routes.py:2736-2769` — `aiofiles` streaming write + `unlink` on size overflow.
- `document_routes.py:94-95,1746-1750` — `__tmp__` temp-file prefix and cleanup after enqueue.

### 3.3 Scan / Discovery
- `document_routes.py:1028-1055` — `scan_directory_for_new_files`, `glob("*{ext}")` per engine suffix.
- `document_routes.py:1421-1431` — `find_existing_file_by_file_path`, `iterdir` to find same-name files.

### 3.4 Parse Pipeline (read source + write artifacts + cache)
- `parser/external/_base.py:116-163` — `source.is_file()` check; `raw_dir_for_parsed_dir` resolves the raw-bundle dir; `mkdir` + `clear_dir_contents`; `download_into` hands bytes to the external HTTP service; `build_ir`; `write_sidecar`.
- `sidecar/writer.py:60-112` — `write_sidecar` clears/creates `parsed_dir` (`exists`/`rmtree`/`mkdir`).
- `sidecar/writer.py:330-367` — `write_text` writes `blocks.jsonl` and per-modality JSON.
- `sidecar/writer.py:395-460` — `_materialize_assets`: `mkdir` / `copyfile` / `write_bytes` to materialize assets.
- `s3_uploader.py` (added by PR #3331) — Stage 1b asset upload.

### 3.5 Sidecar URI / Reading Artifacts
- `utils_pipeline.py:690-751` — `sidecar_uri_for` / `resolve_sidecar_uri` / `sidecar_blocks_path` / `sidecar_modality_path` / `sidecar_assets_dir_for_uri`. **This is the key extension point** — the URI scheme already reserves `s3://` (see the comment at `utils_pipeline.py:683-687`; `resolve_sidecar_uri` returns `None` for non-`file://`).
- `utils_pipeline.py:786-816` — `load_lightrag_document_content`, `open`s `blocks.jsonl` to merge body text.

### 3.6 Serving Assets / Images to Clients
- There is currently **no** real "file download" endpoint (`document_routes.py` has no `FileResponse`). Sidecars hold `file://` URIs that clients cannot consume → Issue #3315. PR #3331 works around this with "upload to S3 + remote URL."

### 3.7 Deletion / Clear
- `document_routes.py:1479-1558` — `delete_file_variants_by_file_path`: `iterdir` + `unlink` (source) + `rmtree` (`.parsed` / `*_raw` dirs).
- `document_routes.py:3207-3210` — `/documents/clear` `unlink`s each entry of `input_dir.glob("*")`.

### 3.8 Path References in Archiving and Storage
- `utils.py:741-787` — `get_unique_filename_in_parsed` / `move_file_to_parsed_dir` (`rename`).
- `utils_pipeline.py:759-778` — `archive_source_after_full_docs_sync`.
- `pipeline.py:594,806-837` — writes `file_path` into doc_status / full_docs.
- `parser/external/_base.py:165-179` — writes `sidecar_location` (`file://` URI) into full_docs.

## 4. Core Design: File Storage Abstraction Layer (FileStore)

Introduce an abstraction isomorphic to `kg/`'s `BaseKVStorage` — `BaseFileStorage` (working name `FileStore`) — that upgrades "INPUT_DIR" from a `Path` into a **logical namespace + backend implementation**.

### 4.1 Abstract Interface

```python
# lightrag/filestore/base.py
class BaseFileStore(ABC):
    """Object-storage abstraction addressed by 'key' (a relative logical path, '/'-separated).

    Example keys: 'report.pdf'
                  '__parsed__/report.pdf.parsed/report.blocks.jsonl'
                  '__parsed__/report.pdf.parsed/report.blocks.assets/img-001.png'
    The workspace is prefixed internally by the implementation (same as kg);
    callers only use relative keys.
    """

    # --- byte-level ---
    async def put_bytes(self, key: str, data: bytes, *, content_type: str | None = None) -> None: ...
    async def put_stream(self, key: str, fileobj) -> None: ...        # streaming / multipart upload
    async def get_bytes(self, key: str) -> bytes: ...
    async def open_stream(self, key: str): ...                         # returns an async-readable stream
    async def exists(self, key: str) -> bool: ...
    async def size(self, key: str) -> int | None: ...

    # --- list / delete ---
    async def list(self, prefix: str) -> list[str]: ...                # replaces glob/iterdir
    async def delete(self, key: str) -> None: ...
    async def delete_prefix(self, prefix: str) -> None: ...            # replaces rmtree

    # --- move / copy (S3 has no atomic rename -> copy+delete) ---
    async def copy(self, src_key: str, dst_key: str) -> None: ...
    async def move(self, src_key: str, dst_key: str) -> None: ...      # used for archiving

    # --- interop with tools that "require a real file path" (key) ---
    @asynccontextmanager
    async def materialize(self, key: str) -> AsyncIterator[Path]:
        """Download the object to a local temp file, yield a local Path, clean up on exit.
        The local backend yields the real path directly (zero copy)."""

    @asynccontextmanager
    async def staging_dir(self, prefix: str) -> AsyncIterator[Path]:
        """Provide a local temp dir to write a batch of artifacts; on exit, bulk-put the
        dir's files back to the store (the local backend writes in place). Used by
        write_sidecar / external parser bundles."""

    # --- traceability to clients ---
    async def public_url(self, key: str, *, expires: int = 3600) -> str:
        """S3: return a presigned GET URL; local: return a server-proxy endpoint URL."""

    # --- URI conversion (bridges the existing sidecar_location) ---
    def uri_for(self, key: str) -> str: ...        # local->file://  s3->s3://bucket/...
    def key_for_uri(self, uri: str) -> str | None: ...
```

### 4.2 The Two Implementations

- **`LocalFileStore`** (default): `materialize` is zero-copy (returns the real path), `staging_dir` writes in place, `list` = `glob`, `move` = `rename`, `public_url` returns a server-proxy endpoint. In other words, it gathers all of today's local logic, with unchanged behavior.
- **`S3FileStore`**: based on `boto3` (reusing the `s3-upload` optional dependency already added by PR #3331). `put_stream` uses `upload_fileobj` (auto multipart), `materialize`/`staging_dir` bridge via a local temp dir, `list` uses `list_objects_v2` (paginated), `move` = copy+delete, `public_url` uses `generate_presigned_url`. The workspace and `__parsed__` both manifest as key prefixes.

### 4.3 Factory and Config

Modeled on `kg/factory.py::get_storage_class()`, add `filestore/factory.py`:

```
INPUT_STORAGE=local            # default; or s3
```

`S3FileStore` reuses a set of consistently-prefixed env vars (converging with PR #3331's naming):
```
INPUT_STORAGE=s3
S3_ENDPOINT=...           S3_BUCKET=...
S3_ACCESS_KEY=...         S3_SECRET_KEY=...        # fall back to AWS_*
S3_REGION=us-east-1       S3_PREFIX=lightrag/
S3_PUBLIC_URL_PREFIX=...  # optional CDN
S3_PRESIGN_EXPIRE=3600    # presigned URL lifetime
S3_VERIFY_TLS=true        # note: PR #3331 hardcodes verify=False; this design defaults to true and is configurable
```

### 4.4 Single Injection Point

`DocumentManager` no longer holds `input_dir: Path`; instead it holds `store: BaseFileStore` and a logical prefix. All `Path`-assembly functions in `utils_pipeline` (`parsed_artifact_dir_for`, etc.) return a **key** rather than a `Path`. The `LightRAG` instance builds the store from the factory during `initialize_storages()` (alongside the existing storage initialization).

### 4.5 VLM Image Reads: Staging Local Copy + materialize Cache Semantics

This deserves its own section, because multimodal analysis (VLM) is the only consumer in INPUT_DIR that **hard-depends on a "real local file path,"** and it is precisely why PR #3331 didn't dare touch it and could only "upload an extra copy."

**Current state (strictly reads local disk).** Layer 2's `analyze_multimodal` fetches each drawing's image like this (`pipeline.py:3591-3654`):

```python
def _resolve_image_path(path_str, sidecar_dir) -> Path | None:
    candidate = Path(path_str)
    if not candidate.is_absolute():
        candidate = sidecar_dir / path_str       # drawings.json's path is relative to the sidecar dir
    if candidate.exists() and candidate.is_file():
        return candidate
    return None
...
raw = candidate.read_bytes()                      # reads local disk bytes directly
img_payload = {"base64": base64.b64encode(raw).decode("ascii"), ...}
```

That is, `drawings.json.path` → join into a local path → `exists()/is_file()` → `read_bytes()` → base64 → feed VLM. The whole path requires that image file to truly exist on the **local filesystem**. This is the origin of PR #3331's comment `local path is always preserved so VLM analysis reads from disk` — it uploads an extra copy to S3 solely for the frontend, while VLM still reads local.

**The problem.** Once INPUT_DIR is backed by S3, the authoritative copy of `*.blocks.assets/img-001.png` is in S3 and may not exist on local disk. If the replica running VLM is not the one that parsed the document (retry, later re-analysis, or parse and analysis scheduled onto different workers), `candidate.exists()` is simply `False` → skip or failure.

**Design: one `materialize` call + one local cache layer that collapses both cases.** Change the "join local path + read_bytes" above to:

```python
async with store.materialize(asset_key) as local_path:
    raw = local_path.read_bytes()
    ...  # size checks, _VLM_RASTER_EXTS, max_image_bytes, base64, cache key — all unchanged
```

- **`LocalFileStore.materialize`** → zero-copy, returns the real path directly (behavior equals today).
- **`S3FileStore.materialize`** → checks a **local cache dir** first:
  - **Hot path (within the same parse batch)**: during parse Layer 1, `write_sidecar` produces `blocks.jsonl` and the images under `*.blocks.assets/` inside `staging_dir()`'s local temp dir, then flushes them back to S3; VLM Layer 2 runs immediately after on the **same worker, same document**, so that batch of local copies (= the cache) is still present → cache hit, near-zero cost, performance equal to today. This is the overwhelmingly common case.
  - **Cold path (cross-replica / later re-analysis)**: local cache miss → download from S3 into the cache dir and return the local path, cleaned up afterward per the rules below. Only this case pays one download.

Thus "staging local copy" and "cross-replica materialize" are no longer two separate code paths, but two outcomes — **cache hit / miss** — of the same `materialize` call.

**Impact on `drawings.json`.** `path` no longer stores a "permanent local absolute path"; it stores a **store-relative key** (a filename relative to the sidecar, almost identical to today's relative path). VLM resolves it as: `full_docs.sidecar_location` URI → store + base key → joined asset key → `materialize`. This **lifts PR #3331's constraint of "must permanently preserve a local path for VLM disk reads"**; the URL handed to the frontend is also produced uniformly by `store.public_url(key)` (S3 = presigned URL), so the separate `remote_url` field is no longer maintained.

**Two engineering caveats.**

1. **Cleanup (borrowed vs owned)**: `materialize` is a context manager. On exit, delete only "the temp file you downloaded on this cold path"; a "borrowed" result pointing at the staging cache / a real local path **must not be deleted**. The implementation must distinguish owned vs borrowed returns to avoid deleting a cache entry still in use by other steps in the same batch.
2. **Concurrency**: the same image may be `materialize`d concurrently. Cache writes use `<key>.tmp` + atomic rename (or a per-key lock) to avoid reading a half-written file. This is the same class of problem as the external parser's `*_raw/` cache-hit check (`is_bundle_valid`) and can reuse the same cache-dir and atomic-write conventions.

## 5. Per-Touchpoint Refactor Mapping

| Area | Today | After |
|---|---|---|
| Upload to disk | `aiofiles.open(path,'wb')` streaming | `store.put_stream(key, upload.file)`; on overflow, abort before/during upload and `store.delete(key)` |
| Temp upload | `__tmp__`-prefixed file | `__tmp__`-prefixed key, `store.delete` after enqueue |
| Duplicate precheck | `path.exists()` / `iterdir` | `store.exists(key)` / `store.list(prefix)` (still prefer content-hash dedup, minimize reliance on listing) |
| Scan | `glob("*{ext}")` | `store.list("")` filtered by suffix; at scale, treat doc_status as authoritative and scan as compensation |
| Read source for parser | pass a local `Path` | `async with store.materialize(key) as p:` then hand to `download_into(p)` |
| Parse intermediate `*_raw/` | local `mkdir`+write | build inside `store.staging_dir()` then flush back; cache hits judged via `store.exists`/`list` |
| Write sidecar | `write_text`/`copyfile` to `parsed_dir` | `write_sidecar` produces inside `staging_dir`, bulk-flushed on exit; or `put_bytes` directly |
| Asset materialization | `_materialize_assets` to disk | same as above; PR #3331's "extra upload" is naturally superseded — the asset is already in the store |
| VLM image read | reads local disk (`read_bytes`) | `async with store.materialize(asset_key)`: hot path hits the staging local cache, cold path downloads (see §4.5) |
| Sidecar URI | `file://abs/...` | `store.uri_for(key)`: local→`file://`, s3→`s3://bucket/prefix/...` |
| Read blocks.jsonl | `open(path)` | `store.get_bytes(key)` / `open_stream`; `load_lightrag_document_content` reads via URI→key→store |
| Traceability to client | none / file:// | new `GET /documents/file?uri=...`: s3 backend 302s to a presigned URL, local backend validates then `FileResponse` |
| Deletion | `unlink` / `rmtree` | `store.delete` / `store.delete_prefix(parsed_dir_key)` |
| Clear | `glob("*")`+`unlink` | `store.delete_prefix(workspace_prefix)` |
| Archive | `rename` into `__parsed__/` | `store.move(src_key, parsed_key)` |

Key point: **the `sidecar_location` URI remains the sole handle between a document and its artifacts**. We only extend the URI scheme from a single `file://` to `file://` + `s3://`, routing to the corresponding store by scheme at resolution time. `resolve_sidecar_uri` today returns `None` for `s3://` — that is exactly the resolver to fill in.

## 6. Key Technical Questions and Decisions

**Q1 How do external parsers (mineru/docling) get the source file?**
They are HTTP adapters; `download_into` needs local bytes. Decision: use `store.materialize(key)` to download to a local temp file and hand it to the adapter. We do not introduce the complexity of "pass a presigned URL directly to the third party" (the third party may not be able to reach our S3).

**Q2 Must VLM read local disk?** See §4.5.
In short: VLM switches to `async with store.materialize(asset_key) as p:` to get a local path, with the rest of the logic unchanged. Within the same parse batch it hits the local cache written by `staging_dir` (zero extra download); only cross-replica / later re-analysis actually downloads from S3. This **lifts** PR #3331's constraint of "must permanently preserve a local path for VLM."

**Q3 Presigned or proxy for client traceability?**
Both are supported, decided by the backend: S3 uses a presigned URL (302 redirect, offloads bandwidth, supports expiry), local uses a server-side `FileResponse` proxy. Unified through `store.public_url(key)`, transparent to callers. **Recommended**: the S3 backend defaults to presigned, with `S3_PUBLIC_URL_PREFIX` configurable to go through a CDN.

**Q4 Distributed correctness / atomicity.**
- A single-object S3 PUT is atomic, naturally fitting "visible only after upload completes." The two-phase `__tmp__` prefix + rename (copy+delete) after enqueue still holds.
- No directory concept: all "directory" operations become prefix semantics; `delete_prefix` must paginate.
- Cross-replica mutual exclusion is still handled by the existing `shared_storage` (Redis namespace locks; see the pipeline concurrency contract in AGENTS.md), **not** by the filesystem.
- Dedup treats doc_status (content_hash / canonical basename) as authoritative, with `store.list` only as compensation, to avoid a large-bucket S3 list becoming a hotspot.

**Q5 Where do intermediate caches (`*_raw/`) go?**
Under a cache prefix in the store (S3), so any replica can reuse the expensive mineru/docling parse results. Hit detection uses `store.exists`/manifest comparison (reusing the existing `is_bundle_valid`). An `S3_CACHE_RAW=true|false` can let users trade off "save parsing" vs "save storage."

**Q6 Consistency / list latency.**
S3 is now strongly consistent for reads and writes (including list-after-write), so no extra handling is needed; non-AWS-compatible implementations (MinIO, etc.) satisfy this too. We still hold to "doc_status authoritative, list compensatory" to reduce reliance on list.

## 7. Compatibility and Migration

- **Zero change by default**: `INPUT_STORAGE` unset or `local` → `LocalFileStore`, byte-for-byte equivalent to today.
- **Historical `file://` URIs**: `resolve_sidecar_uri` keeps resolving `file://`; new writes produce `s3://` or `file://` per the current backend. During the mixed-existence period, routing is by scheme and they do not interfere.
- **Migration tool**: add `python -m lightrag.tools.migrate_input_to_s3` to fully upload the existing `INPUT_DIR` to S3 and rewrite the `file://` `sidecar_location` in doc_status/full_docs to `s3://` (dry-run + verification).
- **K8s**: under the S3 backend the inputs PVC can be dropped (see `k8s-deploy/`), and replicas can exceed 1.

## 8. Phased Implementation Plan

1. **Phase 0 — Land the abstraction (no behavior change)**: add `lightrag/filestore/{base,local,factory}.py`; switch `utils_pipeline` path functions to key semantics; inject `store` into `DocumentManager`. Use `LocalFileStore` throughout, pass the existing tests, **default behavior unchanged**.
2. **Phase 1 — Refactor write/read touchpoints**: upload, scan, parse source reads, `write_sidecar`, `load_lightrag_document_content`, deletion, archiving all go through the store. Still the local backend; regression tests.
3. **Phase 2 — S3 backend**: implement `S3FileStore` (reusing `boto3`), `materialize`/`staging_dir`/`list`/`move`/presigned. Add MinIO integration tests (mock + `requires_db` marker).
4. **Phase 3 — Client traceability endpoint**: `GET /documents/file` (presigned 302 / proxy FileResponse); the WebUI uses it to replace `file://`, naturally compatible with and superseding PR #3331's `remote_url`.
5. **Phase 4 — Migration tool + docs + drop K8s PVC**: migration script, `env.example`, `docs/` (including this file's Chinese version), `k8s-deploy` values.

Each phase is independently mergeable and revertible; the default config is always local.

## 9. Relationship to PR #3331

PR #3331 is a local solution for "upload image assets to S3"; this proposal is its **superset**: when `INPUT_STORAGE=s3`, assets are already written in S3, no "extra upload" needed, and the `remote_url` field is produced uniformly by `store.public_url`. Recommendations:

- If #3331 already has dependent users, keep a version of it as an "assets-only egress" transitional option under `INPUT_STORAGE=local`, documented as superseded by this abstraction;
- Converge env var naming (`LIGHTRAG_ASSET_UPLOAD_*` → `S3_*`);
- Fix the `verify=False` TLS hazard (verify by default, `S3_VERIFY_TLS=false` on demand).

## 10. Testing Strategy

- Abstraction contract tests: run the same set of contract cases (put/get/list/move/delete_prefix/materialize/staging/public_url) against both `LocalFileStore` and `S3FileStore` (MinIO/mock).
- Touchpoint regression: upload dedup, scan classification, delete variants, clear, archiving, sidecar read/write, cross-replica traceability.
- Backward compatibility: historical `file://` URI resolution, byte-for-byte equivalence of the default local path.
- Follow AGENTS.md: mock external services, mark integration tests with `integration`/`requires_db`, and add a regression test for each refactored touchpoint.
