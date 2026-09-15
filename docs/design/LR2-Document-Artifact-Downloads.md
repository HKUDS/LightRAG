# RFC: Document Artifact Downloads

> https://github.com/HKUDS/LightRAG/issues/3585 ("RFC:
>Document Artifact Downloads")

## Why basename lookup is invalid

`doc_status.file_path` is a canonical, hint-stripped logical basename. It is not a physical locator.

The logical filename is unique in the current single-workspace contract, and `doc_id` is derived from that canonical filename. Therefore a canonical archive target occupied while a replacement document is being archived cannot belong to another currently valid document. It may instead be:

- a source left behind after a deleted document's incomplete filesystem cleanup;
- a scan-time alias or duplicate archived without becoming the existing document's source;
- a post-parse content duplicate;
- a historical or manually placed file.

Such a file is an orphan candidate, not an alternative source for the current `doc_id`. The server may move it aside only after a strict storage read proves that no valid document currently names that exact location. A storage error is not proof of absence and fails closed.

Scanning `__parsed__`, stripping `_001`, and returning an arbitrary match remains forbidden. Filesystem iteration order never defines document ownership.

## Artifact identity fields

The model uses four fields:

| Field | Storage | Normative meaning |
| --- | --- | --- |
| `file_path` | `doc_status` and `full_docs` | Stable canonical logical filename with parser hints removed. Used for scheduling, deduplication, display, citations, and ZIP entry names. Never interpreted as a physical path. |
| `metadata.source_location` | `doc_status.metadata` | Exact current location of the managed source file, stored as a POSIX path relative to the current workspace input root. |
| `metadata.source_archive_location` | `doc_status.metadata` | Exact canonical archive target determined and persisted before the managed move. It is a recovery anchor, not a directory-search hint. |
| `sidecar_location` | `full_docs` | Existing exact locator of the parser-produced `*.parsed/` directory and the only authority for the parsed artifact. |

Before archival:

```text
file_path                         = report.pdf
metadata.source_location         = report.[mineru-iet].pdf
metadata.source_archive_location = __parsed__/report.[mineru-iet].pdf
full_docs.sidecar_location       = file:///.../__parsed__/report.pdf.parsed/
```

After archival:

```text
file_path                         = report.pdf
metadata.source_location         = __parsed__/report.[mineru-iet].pdf
metadata.source_archive_location = __parsed__/report.[mineru-iet].pdf
full_docs.sidecar_location       = file:///.../__parsed__/report.pdf.parsed/
```

`source_location`, `source_archive_location`, move/recovery state, deletion journals, raw-cache locators, and internal `file://` URIs are protected implementation metadata. Ordinary document-list, pagination, tracking, logging, and audit responses must filter them. Only the artifact APIs expose designed public filenames, kinds, states, and URLs.

### Deprecate document-level `source_file`

Once `source_location` exists from enqueue time and `source_archive_location` is persisted before archival, document-level `source_file` / legacy `source_file_name` is redundant:

- parser resolution uses the exact current `source_location`;
- parser choice/options come from persisted `parse_engine`, `process_options`, and `chunk_options`;
- scan identity compares the incoming workspace-relative location with `source_location`;
- source ZIP entries use canonical `file_path`; the ZIP response `Content-Disposition` uses the restart-stable `{track_id}.{artifact_kind}.zip` export filename;
- exact deletion uses persisted locators and a deletion journal.

New documents stop writing document-level `source_file`. Fields with the same name in multimodal image payloads or parser-local variables are unrelated and remain unchanged.

If a future requirement needs the exact client-supplied filename, add an explicit `original_filename`; do not overload either locator. Version 1 uses canonical `file_path` as the public source filename.

## Source-location lifecycle and orphan rotation

1. Managed upload/scan enqueue persists the exact current `source_location` and the exact canonical `source_archive_location`.
2. Both keys are reserved long-lived metadata preserved through every status transition and retry reset.
3. After successful parsing and `full_docs` synchronization, the pipeline owner compares the two exact paths while it still owns the workspace pipeline reservation.
4. If current and target are equal and the target is the expected regular file, archival is already committed.
5. If the current source exists and the target does not, atomically move the source to the target.
6. If both exist, strictly confirm that the target is not referenced by any valid document. Only then move that orphan to an available numbered backup and move the new source into the unchanged canonical target.
7. A numbered orphan backup is not a managed document artifact and is never returned by the artifact API.
8. After the source reaches the canonical target, commit `source_location = source_archive_location` using an owner-fenced targeted metadata update.
9. A scan-time alias/already-processed duplicate that is merely archived must not overwrite the existing document's locator.
10. Every state is idempotently recoverable from the two persisted exact locations; recovery never searches suffix variants.
11. Steps 5 and 6 acquire the per-`(doc_id, artifact_kind="source")` artifact-export lock (§Per-artifact keyed lock) for the duration of the move, so a concurrent source-artifact export build can never observe a half-moved file. This lock has no other effect on this lifecycle and is unrelated to the workspace pipeline reservation in step 3.

Recovery table:

| Current location | Archive target | Meaning and action |
| --- | --- | --- |
| exists | absent | move not completed; archive to the exact target |
| absent | exists | move completed; commit `source_location` |
| exists | exists | prove target is orphan, rotate it, then move current source |
| absent | absent | artifact missing; fail closed and require repair |

A storage error during orphan proof, move-state read, or locator commit is an explicit recoverable inconsistency. It is never converted into a directory scan or basename guess.

## Exact artifact deletion

When `DELETE /documents/{doc_id}` is invoked with `delete_file=true`, its filesystem-cleanup branch must capture and clean exact artifact locators before removing the records that contain them:

1. under the destructive pipeline reservation, strictly read `source_location`, `source_archive_location`, `sidecar_location`, and any parser raw-cache locators;
2. persist a bounded deletion journal containing the exact internal targets;
3. remove only those exact validated files/directories;
4. verify that the targets are absent;
5. delete `doc_status` / `full_docs` and associated data;
6. clear the deletion journal only after completion.

Steps 3–4 run per artifact kind (`source`, `parsed`) under the corresponding per-`(doc_id, artifact_kind)` artifact-export lock (§Per-artifact keyed lock): the lock for a kind is acquired before removing that kind's files and released once its targets are confirmed absent. This keeps a concurrent export build for the *same* document from reading a file this deletion is about to remove, while never blocking an export for any other document, or a different kind of the same document once its own removal has completed. Before contending for that lock, deletion atomically installs a per-key deletion fence in the export JobStore and requests cancellation of any `inflight` build on that key (§Per-artifact keyed lock). The fence rejects new export admission until the journaled deletion completes or yields to its retry path, closing the cancellation-to-lock-acquisition window in which a new builder could otherwise repeatedly get ahead of deletion.

If deletion cannot acquire the `(doc_id, artifact_kind)` lock within its own bounded budget, it abandons this attempt with the deletion journal and the deletion fence both intact and returns a retryable error. It never proceeds with filesystem removal without the lock, and it never preempts the current holder. The fence keeps new exports out, so the retry converges once that holder finishes or its process exits.

A partial failure preserves enough journal state for an idempotent retry. The `delete_file=true` branch must not call a basename variant sweep. The default `DELETE /documents` record-clear operation does not clear the workspace artifact root; any future filesystem-clear option must be separately explicit and run under the destructive reservation.

Successfully published export ZIPs are immutable snapshots created while their authoritative inputs existed. Neither `delete_file=true` nor the default record-only deletion removes or revokes those snapshots: a previously issued `track_id` remains downloadable until the independent export-cache retention or capacity policy reclaims its ZIP, subject to authorization being checked again on every download. The deletion journal therefore contains authoritative source, sidecar, and parser raw-cache locators only; it never needs a persisted `doc_id`-to-`track_id` index.

### Retention of the archived source is deliberate, not incomplete cleanup

Exact locator cleanup above describes **how** a filesystem artifact is removed once removal is requested. It does not make removal the default, and the current retention behavior is a safety property that must be preserved:

- `DELETE /documents/{doc_id}` defaults to `delete_file=false`. It deletes the storage records (`doc_status`, `full_docs`, chunks, KG contributions) and leaves the archived source file together with its sibling `.parsed/`, `.mineru_raw/`, and `.docling_raw/` directories in `/__parsed__/` untouched.
- Only `delete_file=true` removes those filesystem artifacts. The journaled exact-locator procedure above governs that branch alone; the default branch has no filesystem targets to journal.
- `DELETE /documents` (clear) currently deletes only top-level files in the configured input directory and preserves subdirectories, so `__parsed__` survives a full clear as well. If this RFC ever lets clear reach the artifact root, that must stay an explicitly requested destructive action — never an implied side effect of clearing records.

Two reasons the archive is kept:

1. **Safety.** Ingestion moves the source out of the input directory, so the copy under `__parsed__/` is the only remaining copy of the user's original file. Removing a document from the index must not silently destroy it. Deletion is reversible by default; irreversibility is opt-in.
2. **Diagnostics.** The preserved source and the parser output next to it stay inspectable after the document has left the index, which is how parse and extraction defects are reproduced.

Re-ingesting a deleted document therefore needs no export, no re-upload, and no API that reads `__parsed__`: move the archived file back to its parent input directory and rescan.

```text
mv /__parsed__/report.[mineru-iet].pdf /
POST /documents/scan
```

The scan re-derives `doc_id` from the canonical filename, re-persists `source_location` and `source_archive_location`, reuses the existing `report.pdf.parsed/` directory in place, and re-archives the source on completion. `__parsed__` is skipped by scan discovery, so nothing is re-ingested until an operator moves a file out of it. This recovery path is a local filesystem action and does not widen the non-goal *"Exposing arbitrary files below `INPUT_DIR` or `__parsed__`"*.

Consequence for the orphan rules in §**Why basename lookup is invalid** and §**Source-location lifecycle and orphan rotation**: a source retained by a default (`delete_file=false`) deletion is an *expected, correct* occupant of a canonical archive target, not evidence of incomplete cleanup. It is exactly the case step 6 rotates aside after a strict storage read proves no valid document currently names that location — a further reason basename lookup cannot define document ownership.

## Artifact availability

```http
GET /documents/{doc_id}/artifacts
```

This returns public kinds, availability/export capability, public filenames, and export-request URLs. It never returns relative internal locators, absolute paths, raw `file://` URIs, move state, or cache paths.

Required permission: `documents.artifacts.read`.

## Asynchronous ZIP export API

Source and parsed artifacts use the same three-stage export workflow.

### 1. Request an export

```http
POST /documents/{doc_id}/artifacts/{artifact_kind}/exports
```

Initial kinds:

| Kind | Authoritative input | Permission | ZIP result |
| --- | --- | --- | --- |
| `source` | `doc_status.metadata.source_location` | `documents.artifacts.source.download` | `{track_id}.source.zip` containing one entry named after the document's canonical `file_path` |
| `parsed` | `full_docs.sidecar_location` | `documents.artifacts.parsed.download` | `{track_id}.parsed.zip` containing a top-level `.parsed/` directory |

The endpoint validates authorization before document lookup, then either joins an in-flight export job for the same `(doc_id, artifact_kind)` or creates a new bounded export job (§Export jobs and independent build scheduling), and returns `202 Accepted` with that job's `track_id`:

```json
{
  "track_id": "artifact-...",
  "status": "queued",
  "status_url": "/documents/artifact-exports/source/artifact-...",
  "download_url": "/documents/artifact-exports/source/artifact-.../download"
}
```

A request that joins an in-flight job may receive `status: "running"` instead of `"queued"` if the join happens after the build has already started; either way the returned `track_id` is shared with every other concurrent requester for the same artifact.

### 2. Query export status

```http
GET /documents/artifact-exports/{artifact_kind}/{track_id}
```

States:

```text
queued
running
ready
failed
cancelled
```

`queued` and `running` come from the transient build JobStore. Terminal state comes from the kind-specific download directory: `{track_id}.zip`, `{track_id}.failed`, or `{track_id}.cancelled`. A successfully published ZIP is therefore still `ready` after an API worker or the whole service restarts. Once a terminal file is reclaimed, no tombstone remains and the track ID becomes indistinguishable from one that never existed.

Every status read first performs bounded reconciliation for that exact track ID. A terminal file outranks a stale active record: the handler owner-safely clears leftover `jobs`/`inflight`/counter/reservation bookkeeping and then returns the file state. With no terminal file, a valid leased job returns `queued` or `running`; an orphaned `.pending` or expired owner is recovered to `.failed` before the response. Multiple mutually exclusive terminal files return `409`. Only the absence of an active job, `.pending`, and every terminal file returns `404`.

Public status fields may include `track_id`, kind, public filename, timestamps, compressed/uncompressed sizes, file count, status, and sanitized error code/message. Internal locators, cache paths, owner tokens, PIDs, and credentials are never returned.

### 3. Download a ready ZIP

```http
GET /documents/artifact-exports/{artifact_kind}/{track_id}/download
```

| State/condition | HTTP |
| --- | --- |
| `ready` and cached ZIP present | `200` |
| `queued` or `running` | `409` with current state and `Retry-After` |
| `failed` | `409` |
| `cancelled` | `409` |
| no current build or terminal file | `404` |
| serving concurrency full | `429` with `Retry-After` |
| cache/download-lease provider unavailable | `503` |

The explicit `artifact_kind` path segment lets the route declare and enforce the kind-specific permission before track/document existence lookup. After authorization, an active record must match the path kind; terminal lookup is confined to that kind's cache directory, and a marker whose embedded kind disagrees is a `409` inconsistency. A track ID is not a credential.

## Export jobs and independent build scheduling

> Renamed from "Export jobs, mailbox, and pipeline scheduling": building is no longer owned by, or scheduled through, the document-ingestion pipeline. Concurrent downloads of the same or different artifacts need no pipeline coordination at all.

The export design separates four responsibilities:

```text
ArtifactExportJobStore  = transient queued/running build ownership and reservations
Per-artifact keyed lock = source/sidecar read-versus-mutate exclusion
Terminal cache files    = durable ready/failed/cancelled status
Download lease gate     = transient cross-worker serving admission and cache-reclamation fence
```

There is no mailbox and no pipeline-owned build step. A request either joins an in-flight build or starts a new one in its own bounded task, independent of `pipeline_status` and `pipeline_ingress`; the document-ingestion pipeline never learns that an export happened, and an export never sets `pipeline_status.busy`. Only the build is ephemeral. A published terminal cache file survives API-worker and whole-service restarts and is sufficient for status and download without a document or JobStore lookup.

### Job store

`ArtifactExportJobStore` is not a new subsystem with its own lock and multiprocess plumbing. It is a shared, workspace-scoped dict obtained via `get_namespace_data("artifact_export_jobs", workspace=...)` and mutated only under `get_namespace_lock("artifact_export_jobs", workspace=...)` — the exact mechanism `pipeline_status` already uses, pointed at a new namespace name instead of a new class hierarchy. This gets single-process vs. Manager-backed multi-worker support and per-workspace isolation from existing infrastructure; no Hub, explicit `BaseProxy`, or second parallel shared-state mechanism is required.

An `initialize_artifact_export_status(workspace=None)` bootstrap, called at the same lifespan point as `initialize_pipeline_status()`, seeds one namespace with:

```text
jobs: {track_id -> queued/running build record}
inflight: {(doc_id, artifact_kind) -> track_id}
deletion_fences: {(doc_id, artifact_kind) -> deletion_operation_id}
running_builds: int
reserved_cache_count: int
reserved_cache_bytes: int
```

`jobs`, `inflight`, and `deletion_fences` are `manager.dict()` objects in multiprocess mode and plain dicts otherwise. A value read from a `manager.dict()` is a disconnected copy, so every mutation replaces the whole value; nested record keys are never changed in place. Admission, owner/version checks, cancellation intent, deletion fencing, `inflight`, concurrency, and cache-reservation accounting are updated together under the namespace lock.

One active record exists per `track_id`:

```text
track_id, doc_id, artifact_kind, owner_token
status: queued | running
version, created_at, updated_at
lease_expires_at, cancellation_requested
reserved_cache_count, reserved_cache_bytes
```

Both `queued` and `running` carry an owner and a renewable lease. Creation uses the repository's committed-background-child handshake: the request returns only after the child has taken ownership, while a failure or cancellation between record creation and child takeover runs owner-checked compensation that clears the job, matching `inflight`, counters, reservation, and `.pending` file. This prevents a worker death before the first builder instruction from leaving an immortal queued job.

The JobStore never stores `ready`, `failed`, or `cancelled`. Those states are terminal cache files. On success, failure, or cancellation the owner publishes the corresponding terminal file first and then, in one JobStore critical section, removes the matching job and `inflight`, decrements `running_builds`, and releases or converts the cache reservation. If the process dies after terminal publication but before bookkeeping cleanup, a later read or maintenance pass treats the terminal file as authoritative and performs the same idempotent cleanup rather than overwriting success with a lease-expired failure.

### Per-artifact keyed lock and deletion fence

A separate lock protects the one thing shared with ingestion: the source/sidecar file for one `doc_id`. It is `get_storage_keyed_lock(keys=[f"{doc_id}:{artifact_kind}"], namespace="artifact_export")`, reusing the existing storage keyed-lock primitive with a dedicated namespace. A builder holds it from strict locator resolution through reading the last input byte. Source archival and exact deletion take the same source key; the parser takes the parsed key before its first sidecar mutation and holds it until the directory has reached a complete persisted outcome or the parse has failed. Acquisition of that lock is bounded by the caller, not by the primitive: `_KeyedLeaseLock.acquire()` polls the holder table with exponential backoff and never times out on its own. A builder MUST therefore wrap acquisition in a deadline derived from its remaining `DEFAULT_MAX_DOWNLOAD_PREPARE_SECONDS` budget and fail the job with a sanitized `artifact_busy` error when that deadline elapses. This is safe to cancel: `_KeyedLockContext.__aenter__` rolls back its reference count, its per-process gate, and any partially acquired key under `asyncio.shield` on `CancelledError`, and the multiprocess lease is installed only after the last cancellation point — a cancelled acquisition leaves no holder record behind.

The parser must acquire the `(doc_id, "parsed")` key before clearing an existing sidecar directory, not merely when it begins writing. Supported parsers can clear `*.parsed/` before an external call and repopulate it later, so this whole attempt is one mutation interval. During a first parse no export can pass availability admission because `sidecar_location` is not published yet.

#### The build lease and the artifact lock are independent

They are separate mechanisms with no implication between them. A reaped build lease means only that the job's *visible status* has been resolved to a terminal marker; it NEVER means the artifact keyed lock is free. No code path — deletion, source archival, a new export, or crash reconciliation — may bypass or preempt the artifact lock on the strength of a lease expiry, a `.failed` marker, or an absent job record. Exclusion on the files themselves is established solely by holding the lock.

Reclamation of the keyed lock is deliberately **dead-only**: a holder whose owner process is confirmed dead is reclaimed atomically by the next `try_acquire`, but a live-though-slow owner is never preempted and needs no fencing token. This is not a gap to be closed with a TTL. Preempting a live holder would authorize deleting or rewriting a file while a reader still holds it open — precisely what this lock exists to prevent.

#### Releasing the lock is gated on the executor, not on the awaiting coroutine

The build's traversal and compression run in a dedicated executor, and cancelling an `await` does not interrupt a thread already running there. The artifact keyed lock MUST therefore be released only once the executor task is confirmed finished — never merely because the awaiting coroutine was cancelled, which would hand the file to a waiting deletion while the worker thread is still reading it.

On deadline or cancellation the builder sets the shared cancellation flag, then performs a bounded, shielded join on the executor future within `DEFAULT_ARTIFACT_EXPORT_EXECUTOR_JOIN_GRACE_SECONDS` before unwinding the lock context.

If that grace elapses with the thread still running — for example blocked in an uninterruptible read on a stalled filesystem — the builder MUST NOT release the lock. It marks the key poisoned, emits an audit event, and leaves the lock held. A held lock blocks further exports, archival, and deletion for that one `(doc_id, artifact_kind)` and nothing else, which is the safe failure mode; releasing it while a live reader holds the descriptor is not. Dead-only reclamation still bounds the damage: when the process actually exits, the holder table reclaims the key.

#### Deletion versus an in-flight build

Deletion resolves a conflict without allowing a new builder to slip between cancellation and lock acquisition:

1. under the JobStore lock, install `deletion_fences[(doc_id, artifact_kind)] = deletion_operation_id` and set `cancellation_requested=True` on a matching active build;
2. reject every new export admission on a fenced key;
3. let the builder observe cancellation at directory-entry and input-chunk boundaries, publish `.cancelled`, and owner-finalize its JobStore reservation;
4. acquire the artifact keyed lock and perform the exact journaled filesystem deletion;
5. clear the fence only after the deletion completes, or after its durable journal has handed responsibility to the retry path.

The deletion caller never impersonates the builder or uses its private owner token. A slow single-file source export remains promptly cancellable because cancellation is checked between byte chunks, not only between ZIP entries. Successfully published `.zip` snapshots are outside this protocol and remain untouched.

### Availability and half-processed content

"Confirm the artifact is available" in request-handling step 1 below is deliberately a cheap, advisory existence check, not a race-free guarantee — its only job is to avoid creating a build for content that was never produced:

- `source`: available whenever `doc_status.metadata.source_location` is persisted and resolves to an existing regular file. This is true from shortly after enqueue onward, because the RFC's own model guarantees the source bytes never change in place — only their location does, which the per-artifact lock already governs (§Source-location lifecycle and orphan rotation, step 11). `doc_status`'s processing stage is irrelevant here: no operation ever rewrites the source's bytes in place, so there is no in-progress state that could produce a torn or empty read.
- `parsed`: available whenever `full_docs.sidecar_location` is persisted for that `doc_id` **and** `doc_status.status == PROCESSED`. Unlike `source`, `doc_status` is part of this predicate on purpose: because a parse attempt can clear an existing sidecar directory before repopulating it (§Per-artifact keyed lock), `sidecar_location` being persisted is *not* by itself proof that the directory currently holds valid content — the metadata is unchanged across a retry even while the directory underneath is briefly empty or, if that retry fails, potentially left incomplete. Requiring `PROCESSED` excludes every one of those in-progress and failed-mid-retry cases at the cheap, advisory-check stage, before a build is ever created.

  This is a deliberate trade-off: a document that reaches `FAILED` *after* a prior successful parse (e.g. a later extraction step fails) has an intact, complete `*.parsed/` directory that this rule makes undownloadable via the export API until the document is reprocessed to `PROCESSED` again — weakening, for this one path, the kind of parse/extraction-defect diagnosis the retention rationale in §Retention of the archived source is deliberate relies on for deleted documents. The alternative (excluding only the transient `PARSING`/`PROCESSING` states, not `FAILED`) does not fully close the gap it looks like it closes: it would still admit a `FAILED` document whose last retry's clear-then-write left the directory incomplete, since nothing in that weaker rule forces `sidecar_location` to be invalidated when a retry fails. `PROCESSED`-only avoids depending on the pipeline's failure path getting that invalidation right, at the cost of the diagnostic case above. If that trade is wrong for a given deployment, the fix is on the pipeline side — never leave `sidecar_location` pointing at a directory a failed retry has touched — not a weaker availability predicate here.

  Before a document's first successful parse, neither `sidecar_location` nor `PROCESSED` holds, so the check alone already returns "unavailable" (`404`) — there is nothing yet to protect, so there is no race to resolve.

The advisory check above only decides whether it is worth creating a build at all — it is checked once, at request time, and can go stale immediately after. The actual guarantee against downloading half-processed content is the per-artifact lock (§Per-artifact keyed lock), enforced at build time, after admission, and unconditionally on both sides. If a retry starts rewriting a document's `*.parsed/` directory at the moment a client requests the `parsed` artifact, or at the moment an admitted build starts reading it, exactly one of two things happens:

1. the build's lock acquisition (§Build task, step 2) happens first, and it reads the complete pre-retry output; or
2. the retry's lock acquisition happens first, and the build blocks until the rewrite finishes, then reads the complete post-retry output.

There is no interleaving between these two outcomes — never a torn, half-written read, regardless of what the advisory check observed a moment earlier. A build that cannot acquire the lock within its own `DEFAULT_MAX_DOWNLOAD_PREPARE_SECONDS` budget fails the job with a sanitized `artifact_busy` error rather than blocking indefinitely; the client may simply retry the export request.

A concurrent delete racing an export request the other way is likewise self-healing without extra mechanism: if `adelete_by_doc_id` has already removed a document's files and its records by the time a build re-resolves the locator (§Build task, step 3), that read fails cleanly (locator or record not found) and the job transitions to `failed` — the same handled-failure path as any other storage error, not a special case.

### Request handling

`POST /documents/{doc_id}/artifacts/{artifact_kind}/exports`:

1. authorize, then confirm the artifact is available per §Availability and half-processed content (read-only; no lock needed);
2. under the JobStore lock, reject a deletion-fenced key with `409`; if `(doc_id, artifact_kind)` is already `inflight`, return `202` with its existing `track_id`;
3. otherwise perform one bounded cache-reclamation pass; if build concurrency, cache count, or cache bytes still cannot be reserved, return `429` immediately (`503` if the shared store or cache provider is unavailable) and create no job;
4. mint a high-entropy `track_id`, derive its kind-specific shard, atomically create the queued leased job, matching `inflight`, concurrency count, and one-count/`MAX_DOWNLOAD_SIZE` reservation;
5. start the committed builder child and return `202` only after takeover succeeds.

Single-flight lasts only while a build is queued or running. A later export request after publication creates a new snapshot and track ID; no persistent `doc_id`-to-completed-export index exists.

### Build task

1. CAS-transition the leased record to `running` and renew its build lease from an event-loop heartbeat;
2. create the exclusive `{track_id}.pending` output and acquire the per-artifact keyed lock within the overall preparation deadline;
3. strictly re-resolve and open the locator from storage — never from request-time state — using root containment, `lstat`/no-follow, and descriptor-level type validation;
4. run recursive enumeration, reads, and ZIP compression in a dedicated executor, enforcing every limit and checking a thread-safe cancellation/deadline signal at every directory entry and input chunk;
5. release the artifact lock once the last source byte has been consumed **and** the executor task is confirmed finished (§Per-artifact keyed lock — a cancelled `await` alone is never sufficient);
6. finish the ZIP, refresh its controlled mtime to the publication time, fsync it, atomically rename `.pending` to `.zip`, and then fsync the containing directory as required;
7. owner-finalize all transient JobStore bookkeeping in one critical section.

Before publishing ANY terminal file, the builder re-validates its ownership (owner token plus record version) in one JobStore critical section. If its record is gone or is now owned by another token, the build has already been reaped: the builder removes its `.pending` file and exits silently, publishing nothing. Terminal publication is authorized by live ownership, never by having done the work. This removes reaper-versus-woken-builder as a source of conflicting terminal files, leaving the `409` repair path for genuinely unexplained inconsistency only.

On handled failure or cancellation, remove the partial ZIP instead of renaming its potentially large bytes. Write a bounded, sanitized JSON record to a separate temporary file, fsync it, and atomically rename it to `.failed` or `.cancelled`, then perform the same owner-finalization. The marker includes only `track_id`, `artifact_kind`, terminal status/time, and a bounded public error code/message; it contains no `doc_id`, principal, locator, cache path, credential, or owner token.

### Crash recovery

An unhandled worker crash stops the build heartbeat. Any later status read, export request, startup reconciliation, or maintenance trigger checks the filesystem and JobStore together:

| Files | Job/lease | Meaning and action |
| --- | --- | --- |
| `.zip` | present or absent | Publication succeeded; return `ready` and idempotently clear stale transient bookkeeping. |
| `.failed` / `.cancelled` | present or absent | Terminal marker is authoritative; return it and clear stale transient bookkeeping. |
| `.pending` | valid queued/running owner | Build is live; leave it. |
| `.pending` | no job or expired owner | Remove partial bytes, publish a sanitized `builder_lease_expired` `.failed` marker, and release stale bookkeeping. The artifact lock is not implied to be free. |
| no file | expired queued/running owner | Publish a sanitized failure marker and release stale bookkeeping. The artifact lock is not implied to be free. |
| multiple mutually exclusive terminal files | any | Return `409`, audit the inconsistency, and run bounded repair; never guess a downloadable object. |

None of these recovery actions release, steal, or bypass the per-artifact keyed lock. Reconciliation resolves *job records and cache files* only; file-level exclusion remains governed entirely by the lock and its dead-only reclamation, so a reconciliation pass never has to decide whether a build is still reading a file.

A single API worker restart does not imply that the shared JobStore is empty and must not reclaim another worker's live `.pending`. When the whole shared control plane restarts, all old build owners are gone: startup reconciliation converts orphaned `.pending` files into `.failed` markers but preserves every `.zip`, `.failed`, and `.cancelled`. There is no periodic per-worker watchdog; reconciliation stays event-driven and lease-based.

## Resource limits

All values are byte counts or positive integers. Invalid/non-positive configured values fail startup; normal authorized mode does not silently become unlimited.

### Environment variables

| Environment variable | Default | Meaning |
| --- | ---: | --- |
| `MAX_DOWNLOAD_SIZE` | `3 * MAX_UPLOAD_SIZE`, or 300 MiB when upload is unlimited | Maximum completed ZIP bytes |
| `DOWNLOAD_CACHE_TTL_SECONDS` | `43200` (12 hours) | Retention age for `.zip`, `.failed`, and `.cancelled` terminal files |
| `MAX_DOWNLOAD_CACHE_COUNT` | `10000` | Maximum active reservations plus terminal cache entries |
| `MAX_DOWNLOAD_CACHE_BYTES` | `20 * MAX_DOWNLOAD_SIZE` | Maximum reserved or physical cache bytes |
| `MAX_DOWNLOAD_BUILD_CONCURRENCY` | `5` | Concurrent ZIP builders across all workers |
| `MAX_DOWNLOAD_SERVE_CONCURRENCY` | `5` | Concurrent ZIP responses across all workers |

These are deployment policy or capacity values. Disk size, inode budget, CPU count, and network capacity vary by deployment, so the cache and concurrency bounds are configurable but never accept zero, negative, malformed, or unlimited values. `MAX_DOWNLOAD_SIZE` remains tied by default to the upload ceiling; raising it also raises the derived cache-byte default unless the operator sets `MAX_DOWNLOAD_CACHE_BYTES` explicitly.

### Internal constants (`lightrag/constants.py`)

| Constant | Default | Meaning |
| --- | ---: | --- |
| `DEFAULT_MAX_DOWNLOAD_UNCOMPRESSED_SIZE` | `2147483648` (2 GiB) | Maximum sum of regular-file bytes before compression |
| `DEFAULT_MAX_DOWNLOAD_FILE_COUNT` | `10000` | Maximum regular files in one export |
| `DEFAULT_MAX_DOWNLOAD_DIRECTORY_DEPTH` | `16` | Maximum parsed directory depth |
| `DEFAULT_MAX_DOWNLOAD_PREPARE_SECONDS` | `600` | Maximum validation/compression time |
| `DEFAULT_ARTIFACT_EXPORT_BUILD_LEASE_SECONDS` | `60` | Queued/running owner-lease duration |
| `DEFAULT_ARTIFACT_EXPORT_EXECUTOR_JOIN_GRACE_SECONDS` | `30` | Bounded wait for the build executor thread to confirm exit after cancellation, before the artifact lock may be released |
| `DEFAULT_ARTIFACT_EXPORT_STATUS_MAX_BYTES` | `4096` | Maximum serialized `.failed` or `.cancelled` marker size |
| `DEFAULT_ARTIFACT_EXPORT_SHARD_HEX_CHARS` | `2` | Number of high-entropy track-ID hex characters used for cache sharding |
| `DEFAULT_ARTIFACT_CACHE_MAINTENANCE_BATCH` | `256` | Maximum filesystem entries examined by one event-triggered maintenance pass |

These constants are protocol safety bounds or internal implementation geometry, not storage-capacity policy. The build lease is renewed by an event-loop heartbeat; moving synchronous traversal and compression into the dedicated executor keeps that heartbeat schedulable, but does not make it unconditionally responsive — a long event-loop stall (a slow synchronous Manager RPC, a long GC pause) can still let the lease lapse while the build is healthy. That is tolerable precisely because a lapsed lease resolves only the job's visible status and grants no one access to the files (§Per-artifact keyed lock), and because a woken builder revalidates ownership before publishing anything. The preparation deadline covers lock acquisition, validation, traversal, reads, compression, fsync, and terminal publication.

(`MAX_DOWNLOAD_JOBS_PER_CYCLE` is removed outright, not merely relocated: there is no pipeline export cycle left to bound.)

A 2 GiB uncompressed limit is a safety ceiling, not a promise that a 2 GiB image artifact is downloadable. Image-heavy inputs compress poorly and will normally hit `MAX_DOWNLOAD_SIZE` first.

One active reservation counts as one entry and `MAX_DOWNLOAD_SIZE` bytes; its `.pending` file is covered by that reservation rather than counted a second time. Terminal publication converts the reservation to the terminal file's actual byte size. Published `.zip`, `.failed`, and `.cancelled` files, plus internal reclaim-pending files that have not yet been physically removed, count by their actual size. Shared counters may accelerate admission, but the filesystem is authoritative and startup/maintenance reconciliation repairs conservative counter drift.

After a whole-control-plane restart, one cache-bootstrap owner performs a complete kind/shard census in bounded batches before new build reservations are accepted. Existing terminal files remain queryable and downloadable during that census; only new export admission returns `503` with `Retry-After` while authoritative count/byte totals are unknown. A single worker restart reuses the already-verified shared accounting state and does not start a competing census.

Before creating a reservation, perform one bounded reclamation pass. Reclaim terminal files whose controlled publication mtime is older than `DOWNLOAD_CACHE_TTL_SECONDS`, then evict the oldest remaining terminal files if capacity still requires it. Never evict `.pending` or a ZIP with a live download lease. If the hard count or byte bound still cannot be satisfied, return `429` immediately; no job is created or queued for capacity.

## ZIP construction and cache publication

The cache is accessed through a small artifact-cache backend interface. Version 1 uses the single host's persistent local directory `/artifact_exports`, not OS temporary storage. Internal paths are derived only from a validated high-entropy server-generated track ID and an allowlisted kind. The shard is the first `DEFAULT_ARTIFACT_EXPORT_SHARD_HEX_CHARS` characters of the track ID's random component, not its common textual prefix:

```text
/artifact_exports/{artifact_kind}/{shard}/{track_id}.pending
/artifact_exports/{artifact_kind}/{shard}/{track_id}.zip
/artifact_exports/{artifact_kind}/{shard}/{track_id}.failed
/artifact_exports/{artifact_kind}/{shard}/{track_id}.cancelled
```

`.pending` contains the partial ZIP. `.failed` and `.cancelled` are bounded JSON terminal markers. An implementation may use an unaddressable temporary suffix while atomically publishing a marker and a `.reclaiming` suffix while physically deleting a terminal file; neither is a public state or downloadable object.

Build rules:

1. resolve and validate the exact persisted input locator while holding the per-`(doc_id, artifact_kind)` lock;
2. establish allowed-root containment and open each component/file with descriptor-level no-follow semantics; a pre-open `stat` alone is insufficient;
3. recursively enumerate with bounded memory, accepting only regular files and ordinary directories;
4. reject symlinks, FIFO/socket/device entries, absolute/`..`/NUL entry names, duplicate entry names, and entries escaping the exact root;
5. enforce uncompressed bytes, compressed bytes, file count, depth, cancellation, and the preparation deadline while reading chunks, not only from initial metadata;
6. write `.pending` with `ZIP_DEFLATED` at fixed compression level 6 and ZIP64 support in the dedicated executor;
7. stop and remove `.pending` once any bound is exceeded;
8. after the last input byte, and once the executor task is confirmed finished (§Per-artifact keyed lock), release the artifact lock, complete the ZIP central directory, set the controlled publication mtime, and fsync the file;
9. atomically rename `.pending` to `.zip` and fsync the containing directory as required; only this suffix is downloadable;
10. clear transient JobStore ownership and reservations only after `.zip` exists with its verified size.

The artifact-export lock is released once the last source byte has been read **and** the executor task is confirmed finished (§Per-artifact keyed lock — on cancellation, release waits on the bounded executor join); finishing, syncing, and renaming the cache file require no source lock. The public download filename is `{track_id}.{artifact_kind}.zip`; ZIP entries continue to use the canonical `file_path` and parsed top-level directory rules.

An interrupted `.pending` ZIP is never downloadable. Startup preserves published terminal files and reconciles only orphaned active/temporary files as described in §Crash recovery. A restarted client may continue using a successfully published track ID without resubmitting the export.

A future S3-backed cache may replace the local backend without changing routes or public suffix states. It must provide the equivalent of conditional exclusive `.pending` creation and atomic terminal publication; S3 Lifecycle may reclaim terminal objects as a physical-cleanup backstop but must not run earlier than the application retention policy. Once an object is absent, status and download return `404` because version 1 deliberately keeps no expiration tombstone.

## Event-driven cache reclamation, eviction, and active-download leases

Version 1 creates no periodic ZIP-cleanup task per API worker. Cleanup is a bounded, idempotent maintenance step triggered by existing artifact-cache activity:

- server startup;
- export request and cache-capacity reservation;
- successful or failed ZIP build completion;
- export-status query;
- download admission before creating a lease;
- download-lease release;
- a capacity-reservation failure before the caller retries or fails.

Any worker may initiate a maintenance pass. The JobStore lock serializes active build reconciliation and cache reservations; an independent workspace-scoped `artifact_download_leases` namespace serializes serving admission and terminal-file reclamation. Neither is a pipeline lock. Maintenance is cursor-based across kind/shard directories and examines at most `DEFAULT_ARTIFACT_CACHE_MAINTENANCE_BATCH` entries per trigger, so a cache containing many thousands of files does not turn one request into an unbounded directory walk.

Two independent lease concepts cover different lifecycle phases and live in different namespaces:

| | Build lease | Download lease |
| --- | --- | --- |
| Fields | `owner_token`, `lease_expires_at` | `lease_id`, `track_id`, owner PID/process identity, heartbeat time |
| Held while | status is `queued` or `running` | a client is reading a published `.zip` |
| Purpose | detect a builder that died mid-build, so the job reaps to `failed` without a watchdog process | serving-concurrency accounting; block physical deletion while a client reads the ZIP |
| Storage | `artifact_export_jobs` | `artifact_download_leases` |
| Cardinality | exactly one owner per active job | zero or more per ZIP, one per concurrent downloader |

A completed build has no JobStore record. A ZIP may have multiple download leases, but those leases never recreate a build job or a `doc_id` mapping.

Serving concurrency is enforced across all workers with the download-lease gate. A provider failure fails closed. Under that gate, admission verifies the kind-specific `.zip` is present, is younger than its retention TTL, is not being reclaimed, and serving capacity is available; it then creates a unique lease before opening the ZIP. The route does not look up the document or build JobStore. Response completion, cancellation, or disconnect releases only that request's lease in a cancellation-safe callback/finally. PID/process-identity and heartbeat checks reclaim leases after worker death without reclaiming a live request merely because a PID was reused.

TTL and capacity reclamation acquire the same gate, revalidate that the selected terminal file has no live lease, atomically rename it to an internal `.reclaiming` name, and then delete it. Renaming immediately makes it non-downloadable even if physical deletion fails; the object continues counting against hard byte/count bounds until deletion succeeds. A download admitted before the TTL boundary may finish. Releasing the last lease triggers another bounded reclamation pass.

The controlled terminal-file mtime is the publication timestamp and defines both TTL age and oldest-first capacity selection; filename order and filesystem iteration order do not. `.failed` and `.cancelled` markers follow the same 12-hour/default capacity policy as `.zip`. No expired/evicted tombstone is retained, so successful reclamation makes status and download return `404`.

If no artifact-cache activity occurs after a TTL boundary, an old terminal file may remain physically present until the next trigger, but download admission itself is a trigger and refuses a new lease once the controlled mtime has crossed the TTL. Exact-to-the-second physical deletion is not part of the contract; hard count/byte admission bounds remain mandatory.

## HTTP and audit semantics

- `401`: missing or invalid credentials.
- `403`: authenticated principal lacks the required permission.
- `404`: an authorized caller requested an unavailable document/artifact, or no active build or terminal cache file exists for the track ID (including after TTL/capacity reclamation).
- `409`: locator, filesystem type, path, conflicting terminal files, or build-state inconsistency; queued/running download with `Retry-After`; failed or cancelled export download.
- `413`: configured input/output/file-count/depth limit exceeded when detected synchronously; asynchronous jobs expose the equivalent sanitized limit error in `failed` status.
- `429`: build, serving, cache-count, cache-byte, or shared-store capacity temporarily unavailable, with `Retry-After` where retry timing is known.
- `503`: authorization, job, cache, or shared-control-plane provider unavailable.

Audit events cover export request, claim, success/failure/cancellation, stale-build recovery, TTL/capacity reclamation, deletion-fence admission denial, download allow/deny/start/finish/cancel, byte counts, principal, permission, `doc_id` when the build still has it, `track_id`, and artifact kind. They never contain credentials, internal locators, absolute paths, cache paths, ZIP contents, owner tokens, or unsanitized exceptions. A post-restart download may have no `doc_id`; that absence never causes a reverse lookup or weakens the kind-specific permission check.
