# Runtime Configuration Propagation Contract

Read this before changing runtime publication or application of `addon_params`,
role LLM configuration, role queue limits, the global concurrency gate, runtime
configuration fields in `/health`, or the corresponding shared-storage
namespaces. Summary in `AGENTS.md`.

## Purpose and scope

LightRAG permits useful configuration changes inside a running Python process,
but a Gunicorn deployment has one `LightRAG` instance per worker. Updating one
instance therefore does not update its peers. This contract defines how an
explicit SDK publication converges those workers without persisting deployment
configuration, copying secrets into shared memory, or adding a Manager call to
every LLM invocation.

This work lands in phases:

1. workspace-scoped `addon_params` publication and application;
2. process-wide role LLM publication and batch application;
3. dynamic `max_async` propagation into the global concurrency gate.

The propagation layer and SDK publishing methods land before any HTTP control
plane. REST authorization and WebUI permissions are separate work and are not
licensed by this contract. Environment variables and constructor arguments
remain the startup source of truth; runtime publications are intentionally lost
on a full server restart.

## Terms

- **Publisher**: the `LightRAG` instance that explicitly commits a new shared
  configuration version.
- **Desired version**: the highest committed version in a shared namespace.
- **Applied version**: the highest version an individual instance has installed
  successfully in full.
- **Eligible boundary**: a point before new work captures or consumes runtime
  configuration. Checks occur before an enqueue snapshot, at the start of each
  document, at request/SDK operation entry, and before a role wrapper admits a
  new LLM call.
- **Residue**: an observable intermediate state left by a failure or crash in a
  sequence that has no cross-object transaction.

An operation already in progress is never rewritten underneath itself. A new
version governs work admitted after that version has been applied locally.

## Namespace topology

There are two independent shared namespaces and two independent version
counters. A change in one family must not wake or rebuild the other family.

| Family | Namespace | Workspace argument | Contents |
| --- | --- | --- | --- |
| Addon parameters | `runtime_addon_params` | `rag.workspace` | `language`, content-addressed entity prompt profile, `entity_types_guidance`, and `chunker` |
| Role LLM configuration | `runtime_llm_config` | `""` | complete non-secret role configuration, including `max_async` |

`runtime_addon_params` is instance/workspace scoped. `runtime_llm_config` is
process-wide, as are the existing `concurrency_leases` and `queue_stats`
namespaces. The process-wide half does not depend on server-side multi-workspace
ownership.

Each namespace uses flat Manager-dictionary entries:

```text
head_version                         -> non-negative integer
payload:<version>                    -> complete immutable payload for version
applied:<instance_id>                -> per-instance application report
```

`instance_id` is unique per `LightRAG` instance and is reported together with
the PID. It cannot be only the PID: tests construct two real instances in one
process, and future server layouts may do the same.

Version `0` means that no runtime payload has been published since shared-data
initialization. Each instance then uses its startup configuration. The first
publication is version `1`. Timestamps and PIDs are diagnostic fields only and
never participate in ordering.

### Addon payload

The addon payload is a full normalized snapshot, not a patch:

```text
{
  addon_params: {
    language,
    entity_types_guidance,
    chunker
  },
  entity_type_prompt: null | {
    sha256,
    content_utf8,
    path_hint
  }
}
```

Only the fields governed by this contract are publishable. Unknown
`addon_params` remain local and cause explicit publication to fail rather than
being silently copied or discarded.

`entity_type_prompt_file` is expanded by the publisher from one read of the
file. The SHA-256 digest is computed from those exact bytes, and those same
bytes are decoded and validated before publication. `content_utf8`, not the
path, is authoritative. `path_hint` exists only for diagnostics and is never
opened by applying workers. A later edit at the same path therefore cannot
split workers between two prompt profiles. A malformed payload or digest
mismatch is an application failure; an on-disk mismatch is impossible because
workers do not re-read the hint.

Workers validate the published content and install the resulting resolved
profile as part of the same local batch as
`_replace_addon_params(..., mark_dirty=True)`. The live mapping, derived
language, chunker normalization, and entity-extraction prompt cache either all
advance to the new version or all remain on the previous version.

### Role LLM payload

The role payload is also a complete snapshot. It contains every registered role
and only these publishable fields:

```text
binding, model, host, max_async, timeout, model_kwargs
```

Callables and credentials are not serializable runtime configuration.
`model_func`, `api_key`, authentication options, and provider credential
objects are never published. Each worker resolves the requested binding using
its locally operator-provisioned credentials and registered production
builder. A `publish=True` update that depends on a caller-supplied callable is
rejected before the shared commit; the existing local-only update path remains
available.

A role update may name one role, but the committed payload is a complete desired
snapshot. Under the namespace lock, the publisher merges that change into the
latest committed payload, not a potentially stale local copy. This prevents a
concurrent update to a different role from being overwritten.

## Secret exclusion

Shared runtime namespaces are non-secret control data. Publication validates
the complete candidate recursively before the first shared write:

- keys matching `_RoleLLMMixin._SECRET_MARKERS` are rejected;
- values must belong to the explicitly supported, serializable schema;
- documented numeric token-count keys such as `chunk_token_size` and
  `chunk_overlap_token_size` require explicit safe-key exceptions; there is no
  generic exception for keys merely containing `token`;
- validation failure leaves `head_version` and every payload entry unchanged.

Secret-looking fields are rejected rather than scrubbed. Silent scrubbing could
publish a configuration different from the one the caller requested. Tests
walk every published key recursively with the production secret predicate,
mirroring `test_get_llm_role_config_has_no_secret_escape_hatch`.

## Publication protocol

Publication is serialized with `get_namespace_lock()` for the corresponding
namespace and scope. The commit protocol is:

1. normalize and validate the complete candidate locally;
2. acquire the namespace lock;
3. read `head_version` and choose `next_version = head_version + 1`;
4. for a role patch, merge it into the payload named by the locked head;
5. write the complete candidate to `payload:<next_version>`;
6. commit by assigning `head_version = next_version`;
7. call `set_all_update_flags()` for that namespace before releasing the lock;
8. return the committed version and allow the publisher to apply through the
   same worker path as every peer.

The payload-first ordering is the governing invariant: a worker may never see a
head version whose complete payload has not already been published. A payload
whose version is above the head is inert staging residue.

Two publishers cannot choose the same committed version because version
selection happens under the namespace lock. Workers never apply a version less
than or equal to their applied version. If versions are published faster than a
worker checks, it may skip intermediate versions and apply the current head;
it must never move backwards.

The shared assignment at step 6 is the commit point. Cancellation is deferred
from the first shared write through notification. A failure before the commit
reports no publication. A notification or cleanup failure after the commit must
not be reported as though the publication did not happen: the committed
version is returned or carried in an explicit commit-aware outcome, and the
degraded notification is logged.

Old versioned payloads may be pruned only when doing so cannot make the current
head unreadable. A reader that loses a superseded payload race keeps its current
configuration, re-reads the head, and retries; it never substitutes a lower
version.

## Worker refresh and staleness

Every multi-worker `LightRAG` instance registers its own update flag for both
families it consumes and stores local applied versions. Checking is
boundary-driven and locally debounced:

- an active instance reads its Manager-backed flags at most once every 250 ms;
- a true flag triggers a `head_version` read and, when newer, a payload read;
- regardless of the flag, an active instance audits `head_version` at least
  once per second, recovering a notification failure;
- an idle instance performs no polling and checks before its next eligible
  operation.

The documented worst-case staleness window is therefore one second plus the
time to the next eligible boundary. In the normal notified path it is 250 ms
plus the time to that boundary. Local monotonic-clock checks happen freely, but
the debounce permits no more than four flag-read cycles per second per active
instance. The version audit, rather than the flag alone, is what makes a missed
notification self-healing.

Only the instance's own flag is cleared, and only after successful application
or after confirming that the local applied version already equals the head.
One worker must never call `clear_all_update_flags()` to acknowledge work for
its peers. An apply failure leaves that worker's flag armed for retry.

### Apply boundaries

- `addon_params` refresh runs before an enqueue resolves and persists
  `chunk_options`, before a query/SDK operation captures global configuration,
  and before a pipeline worker starts the next document.
- Role refresh runs before a role wrapper admits a new LLM call. Calls already
  submitted retain their old wrapper and finish under the old configuration.
- `/health` forces a refresh attempt before reporting this instance, without
  claiming that a failed peer has converged.

Documents freeze `chunk_options` at enqueue. A document enqueued before a
publication keeps that snapshot even if it is processed afterward; only
subsequent enqueues receive the new chunker defaults. A document already being
processed is not restarted. Other addon values become visible at the next
document boundary.

## Local atomic application and rollback

Reading a new head does not make it applied. Each worker first validates and
builds a complete local candidate without mutating live state.

For addon parameters, prompt content validation, normalization, replacement,
chunk-size overlay, and derived-cache refresh form one batch. For role LLM
configuration, every role in the payload is resolved and wrapped into a staged
snapshot before any live role reference is swapped. If any binding is
unresolvable, prompt profile is invalid, builder raises, or wrapper construction
fails, the worker:

1. retains its previous configuration in full;
2. retains its previous applied version;
3. logs the family, target version, failing role/field, and a sanitized
   actionable error;
4. reports the failed/retrying state for observability;
5. leaves its update flag armed and retries at the next eligible check.

The final live-state swap contains no `await`, so other coroutines in that
event loop observe either the old complete snapshot or the new complete
snapshot. Successful application updates the per-instance report only after
the swap.

Retired role wrappers drain through the existing cleanup path. Failure to clean
up a retired wrapper does not roll back a successfully installed version: no
new call can acquire that wrapper, the failure is logged, and finalization or a
later cleanup attempt retires it.

After initialization, assigning `rag.role_llm_configs = ...` directly must
raise a clear error directing the caller to the supported update method. It
must not remain a silent mutation of a dataclass field that leaves live wrappers
unchanged. Constructor-time `role_llm_configs` behavior is unchanged.

## Dynamic `max_async` and the global gate

The shared role payload carries `max_async`, but the concurrency hot path never
reads it from a Manager proxy. Applying a role version updates a worker-local
cached limit and generation. `get_global_concurrency_limit()`,
`is_global_concurrency_limited()`, and `_acquire_global_slot()` continue to read
only worker-local memory.

Role wrappers check the local generation before admission. A wrapper created
while its group was unlimited must begin using the gate after an unlimited to
limited transition; rebuilding the `LightRAG` instance is not required.
Increases must raise observed cross-worker concurrency, while decreases stop
new admissions above the new cap.

A lower limit does not cancel calls or leases already admitted. During the
bounded convergence window, workers may temporarily enforce different limits,
and existing holders may keep global usage above the new value. Once every
worker applies, no new call is admitted above the new cap; ordinary completion
and the existing lease reaper drain the excess.

## Cache behavior

Role model changes do not invalidate or delete LLM cache rows. The existing
cache identity partitions different model configurations. Retired key spaces
are not reported to the maintenance GC: doing so would make GC correctness
depend on propagation and rollback state. Old rows remain ordinary unreachable
cache data under their existing identities.

## Observability

Each namespace's `applied:<instance_id>` report contains, at minimum:

```text
instance_id, pid, applied_version, target_version, state, last_seen
```

`state` distinguishes `applied` from `retrying`; error details exposed through
health are bounded and sanitized. Dead reports expire from the global view
after the documented worker-report TTL. Expiry hides dead bookkeeping; it does
not change the desired version.

`/health` reports each family's desired version and the applied version/state
of every live reporting instance. It never presents the serving worker's local
version as cluster-wide truth. During convergence, the differing versions stay
visible.

`get_llm_queue_status()` includes the desired role version and the applied role
version for each worker entry. `/health` returns that queue-status view without
reinterpreting it, so both surfaces agree after convergence and expose the same
partial state while a publication is in flight.

## Single-worker and SDK compatibility

When `workers == 1`, existing operations take no propagation path:

- no Manager is created or accessed;
- no runtime namespace, update flag, applied report, or polling task is added;
- `rag.addon_params[...] = ...` remains the same direct observable-dict
  operation;
- local role update behavior remains unchanged unless the caller explicitly
  uses the new publishing option;
- explicit publishing applies locally without IPC.

Tests assert both observable behavior and zero calls into the new shared
namespace/flag helpers. The propagation feature may not tax the SDK or
single-worker server for a capability that only multiple workers need.

## Accepted residues and recovery

| Residue | Why it is acceptable | Recovery |
| --- | --- | --- |
| `payload:N` exists while `head_version < N` | Payload-first publication can crash before commit; the staged value is inert. | The next publisher overwrites/reuses the uncommitted version under the lock, or shutdown clears the Manager. |
| `head_version` advanced but some flags were not set | The committed payload is complete; notification is an accelerator, not truth. | The one-second version audit discovers the head and applies it. |
| Some workers applied the head while another still reports an older version | This is the bounded convergence state the protocol makes observable. | The lagging worker applies at its next eligible check. |
| One worker cannot apply a validly published payload | Keeping the old full snapshot is safer than a partial rebuild. Failure is loud and visible. | Its flag remains armed; it retries on every eligible check after the debounce. Operator fixes its local binding/file-independent validation problem if retries continue. |
| The publisher's `addon_params` was changed locally but validation or publication failed before commit | Direct SDK mutation is intentionally local and predates propagation; claiming peers changed would be false. | The caller corrects the value and republishes, or restores the local mapping. |
| A reader targeted a payload pruned after it read an older head | It has not modified live state and retains a complete older version. | It re-reads the current head and retries, skipping superseded versions. |
| A dead instance leaves an update-flag handle or applied report | The handle carries no configuration and cannot make another worker move backward. | Reports expire by TTL; handles disappear when the Manager is finalized/restarted. |
| A retired role wrapper fails cleanup after the new wrapper is installed | New calls use only the new wrapper; the old queue is no longer reachable for admission. | Existing cleanup/finalization retries or terminates it, with a warning. |
| Global usage temporarily exceeds a newly lowered limit | Already-admitted work is not cancelled, and workers converge within the documented bound. | Completions and lease expiry drain usage; new admissions use the lower local cache after apply. |
| An old document uses pre-publication `chunk_options` | This is the required enqueue-time reproducibility guarantee, not divergence. | No repair: only documents enqueued after publication use new defaults. |
| Old LLM cache identities remain after a model change | Identity partitioning prevents reuse under the new configuration. | No propagation action and no GC report; normal cache lifecycle applies. |
| A full server restart loses all runtime versions | Runtime state is intentionally not deployment configuration. | Every worker starts again from operator-managed environment and constructor values at version `0`. |

No failure is swallowed. Logs and observability must distinguish an uncommitted
attempt, a committed-but-partially-notified version, and a worker-local apply
failure.

## Test contract

Every propagation test uses `initialize_share_data(2)` and two real `LightRAG`
instances connected to that Manager. A plain dictionary, permissive proxy,
stand-in client constructor, or fake object that accepts arbitrary arguments
cannot prove ordering, serialization, rollback, or cross-instance application.

The phased test matrix is:

- shared-storage tests: concurrent publishers, monotonic ordering, flag
  isolation, missed-notification audit, stale report handling, and the
  `workers == 1` no-new-IPC path;
- addon tests: publish/apply for every governed field, content-addressed prompt
  identity, invalid-profile rollback, and `_replace_addon_params(...,
  mark_dirty=True)` cache refresh;
- chunker tests: documents enqueued before publication retain frozen
  `chunk_options`, while later documents use the new defaults;
- role tests: real production resolution, complete batch swap, builder-failure
  rollback, direct-assignment rejection, secret exclusion, cache identity, and
  queue/health version agreement;
- gate tests: no Manager read on acquisition, increased and decreased achieved
  concurrency, transient over-cap drain, and wrappers created before an
  unlimited to limited transition.

Fault injection is allowed only at an explicit production failure seam and may
not replace the Manager, `LightRAG` instances, payload schema, or constructor
signature being tested. Each regression test is first demonstrated failing
against the broken behavior and then restored.

English and Chinese user documentation change together with the implementation.
In particular, the runtime-mutation sections of
`docs/FileProcessingPipeline.md` and `docs/FileProcessingPipeline-zh.md` must
explain server propagation while preserving the enqueue-time snapshot rule.
