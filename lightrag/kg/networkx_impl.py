import asyncio
import os
from collections import deque
from dataclasses import dataclass
from operator import itemgetter
from typing import final

from lightrag.exceptions import CommitBookkeepingError
from lightrag.file_atomic import atomic_write, reap_orphan_tmp_files
from lightrag.types import KnowledgeGraph, KnowledgeGraphNode, KnowledgeGraphEdge
from lightrag.utils import (
    commit_in_storage_io,
    log_without_raising,
    logger,
    validate_workspace,
    validate_xml_attributes,
)
from lightrag.base import BaseGraphStorage
import networkx as nx
from . import file_fingerprint
from .shared_storage import (
    get_namespace_lock,
    get_update_flag,
    set_all_update_flags,
)

from dotenv import load_dotenv

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)


@final
@dataclass
class NetworkXStorage(BaseGraphStorage):
    """File-backed knowledge-graph storage built on ``networkx.Graph``.

    Storage model:
        A single ``networkx.Graph`` instance lives in process memory; its
        full state is serialized to one GraphML file at
        ``working_dir/[workspace/]graph_<namespace>.graphml``. That GraphML
        file is the **only** cross-process synchronization surface — there
        is no shared memory, no message bus, and no network channel
        between processes. Cross-process visibility is mediated by (a) an
        atomic file write at commit time and (b) a per-namespace
        ``storage_updated`` flag distributed through
        ``lightrag.kg.shared_storage``.

    Concurrency invariants (the code in this file is correct *only* while
    all three hold):
        1. **Single writer per workspace.** The document pipeline's
           ``busy`` / ``destructive_busy`` flags (see
           ``docs/design/PipelineConcurrencyContract.md``) guarantee at most
           one process
           performs ``upsert_*`` / ``delete_*`` / ``remove_*`` /
           ``index_done_callback`` at any time. Every other process is
           read-only.
        2. **Eventual consistency is sufficient.** Read-only processes
           only need to observe the writer's data *after* the writer's
           ``index_done_callback`` completes. Reads landing in the gap
           between a writer's in-memory mutation and its commit may
           legitimately return the pre-update snapshot.
        3. **networkx operations are fully synchronous.** Under a
           single-threaded asyncio event loop, ``graph.add_node`` /
           ``graph.remove_node`` / ``graph.degree`` / etc. cannot be
           preempted by another coroutine, which gives them implicit
           mutual exclusion over ``self._graph``. This is why the methods
           below don't have to hold ``_storage_lock`` while calling into
           ``graph``. The one place that is NOT on the event loop is the
           GraphML serialization, which runs in the storage-IO pool — see
           *Commit gate* below for what re-establishes the exclusion there.

    Commit gate:
        ``index_done_callback`` hands ``self._graph`` to a worker thread,
        which iterates ``graph._node`` / ``graph._adj`` for the length of
        the write. Invariant (3) does not cover that: a coroutine on the
        loop could mutate the graph mid-iteration, tearing the snapshot or
        raising ``RuntimeError: dictionary changed size during iteration``
        from inside the writer.

        ``_commit_gate`` is an ``asyncio.Event``, set except while this
        process is serializing. The committer clears it inside
        ``_storage_lock`` and restores it in a ``finally`` (every path,
        cancellation included — leaking a cleared gate once deadlocks every
        later graph operation in this workspace). ``_get_graph`` waits on it.

        **Holding ``_storage_lock`` across the write is NOT sufficient on
        its own**, which is why the gate exists as well. Releasing a
        ``NamespaceLock`` runs the release on a fresh task and awaits a
        shield, so ``__aexit__`` always suspends at least once. A mutator
        that has read ``self._graph`` and is suspended in that release is
        past the lock and would resume — and mutate — while the worker
        thread is iterating. The gate is therefore checked AFTER
        ``_get_graph``'s lock block, i.e. after the last suspension point:
        from the ``is_set()`` check through the caller's synchronous
        ``graph.add_node()`` there is no ``await``, so no commit can start
        in the middle of a mutation.

        The gate sits at ``_get_graph``, the single choke point every read
        and write goes through, rather than in the seven mutators. Reads are
        gated too. That is deliberate and free: a reader already has to pass
        ``_storage_lock``, which the committer holds throughout, so the gate
        adds no wait it did not already have — and a ``for_write=True``
        variant would buy nothing while forking the semantics of the one
        choke point.

    Cross-process sync protocol (two channels — see issue #3854):
        The fence rests on **two** independent tests, OR-ed at every point
        that decides whether this process holds a current snapshot. They are
        not redundant: their blind spots do not overlap.

        * **Authoritative channel — the file fingerprint.**
          ``(st_mtime_ns, st_size)`` of the GraphML file, compared against
          what this process recorded when it last loaded or wrote it
          (``_loaded_fingerprint``). It is *state*, not an event: nothing
          consumes it, nothing can lose it, and it matches the storage model
          above, where the file is the only cross-process synchronization
          surface. Blind spot: two commits landing inside one filesystem
          timestamp tick with an identical file size.
        * **Accelerator channel — the ``storage_updated`` flag.**
          Distributed through ``lightrag.kg.shared_storage``. Read first,
          because a ``True`` value already answers the question. It is an
          *event*: the reader consumes it, and ``set_all_update_flags``
          publishes it with one Manager RPC per process, so it can be lost —
          for one process, for several, or for all. Blind spot: exactly that
          loss. It covers the fingerprint's tick collision, because a
          collision needs a healthy, fast-committing system.

        Writer side (``index_done_callback``):
            1. ``write_nx_graph`` atomically writes the GraphML file
               (``atomic_write`` lays a tmp file beside the target and
               renames it into place — readers either see the previous
               file in full or the new file in full, never a torn write).
            2. Record the fingerprint of the file just written, so this
               process does not later mistake its own commit for a peer's.
               A local ``stat``, done before step 3 because it cannot fail
               with the manager.
            3. ``set_all_update_flags`` flips every process's
               ``storage_updated`` flag (including the writer's own), then
               reset the writer's own flag to ``False``.
        Reader side (any method that goes through ``_get_graph``):
            1. Inside ``_storage_lock``, test the flag; if it is ``False``,
               test the fingerprint.
            2. On either, **fully reload** ``self._graph`` from disk via
               ``load_nx_graph``. networkx GraphML has no incremental sync
               API, so the entire file is re-parsed.
            3. One reload, one post-condition, whichever channel fired:
               record the new fingerprint, clear the flag, and clear any
               pending recovery reload (``_reload_locked`` -- do not
               open-code any of the three).
        Writer side, before saving (``index_done_callback``'s first block):
            The same two tests. A writer holding a snapshot the file has
            moved past **declines** (returns ``False``) instead of writing:
            its save serializes the whole graph, so proceeding would
            overwrite the peer commit it never saw.

            A declined commit must reach its caller as a failure, because
            declining DISCARDS this process's pending mutation. All three
            callers do that:

            * ``utils_graph._commit_graph_or_raise`` raises on the ``False``
              (``adelete_by_entity``, ``_edit_entity_impl``,
              ``_merge_entities_impl``);
            * ``LightRAG._flush_storages``'s ``_flush_one`` turns it into
              ``IndexFlushError`` (pipeline paths);
            * ``utils_graph._persist_graph_updates`` raises
              ``_declined_commit_error`` for the graph store — and only for
              it, since the vector stores carry no such fence
              (``aedit_relation`` / ``acreate_entity`` /
              ``acreate_relation``).

            Without the second one the fence would only swap which side loses
            data — the peer's commit preserved, this document marked
            PROCESSED with its graph writes dropped and nothing to recover
            them from. As a failure it heals instead: the document goes
            FAILED and its reprocessing re-extracts and re-writes the work.

            The third was the last one to get there: it discarded the return
            value until ``dac05a0``, which closed it as part of the #3838
            ordering rework. Before that a decline through the admin create
            and edit paths was silent — the graph mutation discarded, the
            same operation's vector and tracking rows durable, and the caller
            told it succeeded.

        Ordering rule that keeps the fingerprint honest: it is sampled
        **before** the file is read, never after. A fingerprint sampled after
        the parse can belong to a newer file than the one now in memory, and
        recording it would suppress the reload that newer file needs — the
        one way this fence could *introduce* a lost write. Sampling early can
        only cost a redundant reload, which is harmless.

        A sample that fails outright (``UNREADABLE``) is not a third option
        for a reload: ``_reload_locked`` refuses to load at all, because it
        could neither trust nor record what it read. See its docstring for
        both halves of that, and ``file_fingerprint.adopted`` for the residue
        that remains where a ``None`` fingerprint is still recorded.

        Single-process mode skips the fingerprint test entirely
        (``file_fingerprint.fence_enabled()``): there is no peer that could have
        committed, so a divergent file means an external edit, and reloading
        for it would discard this process's own uncommitted mutations.

        Both sites test one more thing first, and it is not a channel: see
        *Recovery reload* below for the process-local condition that says
        "my own memory is unpersisted" rather than "a peer committed".

        Accepted residues (see *Consistency without transactions* in
        ``AGENTS.md`` — a documented residue is a decision, an undocumented
        one is a defect):
            * Two commits inside one filesystem timestamp tick, with an
              identical file size, **and** the notification lost for this
              process: this process does not observe the second one.
              Recovery: the next commit by any process both flips the flag
              and changes the fingerprint. Timestamp granularity is
              kernel-dependent (~10 µs on Linux ≥ 6.13 multigrain
              timestamps, 1–4 ms on older kernels, coarser on ext3 / HFS+ /
              FAT), while the deployment shape this fence exists for — a
              peer becoming the next writer through a later request —
              separates the two commits by at least a request round trip.
            * A ``stat`` this process cannot perform (permissions, EIO):
              the fingerprint channel reports "no change" and the fence
              degrades to the flag alone, i.e. to the behaviour that
              predates it. Failing towards a reload instead would install an
              empty graph from a file it cannot read, and the next commit
              would serialize that over the real one.
            * The *same* failure inside a **reload** is refused rather
              than absorbed: ``_reload_locked`` raises without loading. Both
              of the alternatives lose data. Loading blind installs an empty
              graph (``load_nx_graph`` gates on ``os.path.exists``, false for
              any stat failure) that the next commit writes over the real
              file; loading and recording ``adopted(UNREADABLE)`` — i.e.
              ``None``, against which every state is a divergence — makes the
              NEXT ``_get_graph`` reload again and discard whatever was
              mutated in between, after which the commit SUCCEEDS without it
              and the document is marked PROCESSED. Refusing costs the batch,
              which the FAILED path reprocesses. See ``_reload_locked``.
            * What remains of that failure is the writer adopting its OWN
              commit or drop through ``_record_fingerprint`` (and
              ``initialize``'s first load), which records ``None``. That
              costs one redundant reload of a graph that already equals the
              file, plus — if the ``stat`` recovers in time for the
              divergence branch — one ``_missed_notification_reloads``
              increment for a peer commit that never happened. Bounded,
              self-healing on the next readable ``stat``, and biased towards
              over- rather than under-reporting. What is NOT optional is
              that the redundant reload land BEFORE the next mutation, which
              is why ``_get_graph`` treats a ``None`` fingerprint as
              unresolved: a readable sample settles it through the divergence
              test, and one that is ``UNREADABLE`` -- reporting "no change",
              so no channel would act on it -- makes the call REFUSE instead
              of serving a snapshot a mutation would then land on, whose
              reload would arrive mid-batch and drop it. See
              ``file_fingerprint.adopted`` for why the obvious fix to the
              ``None`` itself is recorded there as an option rather than
              taken.

    Recovery reload (process-local, NOT a third fence channel):
        ``_recovery_reload_pending`` is a plain ``bool`` on the instance, and
        the rule for it is **armed when the reload becomes owed, cleared only
        by one that completes** -- never "armed when a reload fails".

        Owed is the earlier moment and the safe one: it is the branch entry,
        before the sample is classified and before anything is loaded.
        ``_reload_locked``'s third post-condition clears it, so the happy path
        needs no bookkeeping, and *everything* that can go wrong in between
        leaves the record standing -- including
        ``_count_unannounced_peer_commit_locked``, which reads
        ``storage_updated.value`` over the Manager and can raise before any
        reload is attempted. Two earlier attempts at this were both too late:
        arming only in the failed-SAVE handler left the other four sites
        unarmed, and moving it into ``_reload_locked``'s failure paths still
        missed everything that raises before the call. Both had the same
        consequence -- an unreadable ``stat`` blinds the channels, and the
        next commit publishes work already reported as failed over a peer's
        durable commit.

        What the state means, whichever site armed it: this process holds a
        graph that does not match the file, and the operation that wanted it
        failed, so what the graph holds belongs to work already reported as
        failed. It is tested first at both sites
        the two channels are tested at: ``_get_graph`` discards the divergent
        graph before serving it, and ``index_done_callback`` declines rather
        than publishing mutations that belong to a batch already reported as
        failed. ``_reload_locked`` clears it, as its third post-condition.

        It states a different fact from either channel, and the distinction is
        the point. The channels answer *"did a peer commit?"* — the file has
        moved on and this process's snapshot is behind it. This answers
        *"do I still owe myself a reload?"* — whatever the file did, this
        process failed to converge on it and cannot say what its own graph
        represents. Same remedy, and it outranks both channels because a
        reload discharges all three while only this one is certain.

        Armed after a failed save, that means "my memory is unpersisted": the
        file has not moved and it is memory that is wrong. Armed after a
        failed reload at either channel, it means the opposite -- the file
        moved and this process could not follow. The remedy does not care,
        and neither does the danger: in both, the graph holds work already
        reported as failed, and publishing it is what must not happen.

        Why not ``storage_updated``, which carried this before:
            * **Accuracy.** That flag means "a peer committed". Nothing here
              committed, and no peer is involved. Every log line downstream of
              it said "modifications by another process" for an event that had
              none.
            * **Clean evidence.** ``_missed_notification_reloads`` counts lost
              notifications, and it is the instrument #3854 leaves behind to
              decide whether the writer-side ``os.utime`` monotonicity option
              is ever needed. Arming the *file* channel for recovery — which
              the previous code did, by invalidating ``_loaded_fingerprint`` —
              made a failed flag write show up as a lost notification that
              never happened. A process-local test cannot overcount that way.

              It can *under*count, though, and that is what
              ``_count_unannounced_peer_commit_locked`` exists for: this test
              wins over both channels, and one reload discharges all of them,
              so a peer commit that arrived unannounced *while* recovery was
              pending would be handled correctly and never counted. Both
              recovery branches therefore classify the peer channel before
              they reload -- and so does the reload that would otherwise arm
              the flag, the failed-save handler's own recovery reload.
              Overcounting and undercounting are both defects in
              an instrument a later decision rests on, which is why that one
              helper is now the single increment site for every branch that
              counts — and why it deduplicates by ``(st_mtime_ns, st_size)``:
              a reload that raises leaves the fingerprint and this flag
              untouched, so without that the same peer commit is re-counted
              by every later call, without bound.
            * **Arming cannot fail.** In multiprocess mode
              ``storage_updated`` is a ``Manager().Value`` proxy, so arming it
              was an RPC to the very process whose outage may be why the
              reload just failed. That needed its own failure path, its own
              best-effort log, and an ordering argument against the
              fingerprint half. An attribute write needs none of them, and
              covers single-process mode too — where the file channel is
              gated off entirely, so the flag was the only thing arming
              recovery at all.

              **Arming**, precisely — not the recovery path, which still
              reads ``storage_updated.value`` to classify and writes it in
              ``_reload_locked``, both Manager RPCs. A manager still down
              when recovery is attempted raises out of ``_get_graph`` /
              ``index_done_callback`` with the flag **still armed**, which is
              the outcome to want: the divergence stays visible and the next
              call retries. What the change buys is that *recording* the
              divergence no longer depends on the thing that may have caused
              it — the old code could lose the fact itself, and then nothing
              later would retry.

        The fingerprint is deliberately *not* invalidated when this is armed:
        the save failed, so the file is untouched and the recorded fingerprint
        still describes it correctly. Saying otherwise would be a second lie
        in the opposite direction. Untouched *by this process*, that is --
        ``index_done_callback`` releases the lock between its fence block and
        its save, so the recovery reload in its failure handler classifies the
        peer channel before it adopts the file, like the branches above.

        ``drop`` clears it: the mutation it protects is destroyed with
        everything else, and memory matches the file again. That clear is
        load-bearing, not cosmetic — the flag is sticky and is tested by
        ``index_done_callback``, so leaving it set would make the first commit
        after a clear decline and discard fresh work.

    Lock scope:
        ``_storage_lock`` is a per-``(namespace, workspace)`` keyed lock
        spanning both intra-process coroutines and inter-process workers.
        It wraps only the *reload* and *commit* critical sections, not
        every ``graph.xxx`` call. Operating on ``graph`` outside the lock
        is safe *because of invariant (3)* plus the commit gate, which
        covers the one case invariant (3) does not: the serialization
        running in a worker thread. If invariant (3) is broken further —
        ``graph.xxx`` itself moved to a thread pool, or networkx swapped
        for an async graph library — the gate is no longer enough either
        and the lock scope must be widened to cover the mutation/read
        itself.

    Implementation differences from ``NanoVectorDBStorage`` (same design,
    different surface):
        * **The fence is shared, its verdict is not.** All three test the
          file fingerprint as well as the flag, through the same
          ``lightrag.kg.file_fingerprint`` helpers. What differs is what
          happens once a peer commit is detected on a WRITE path: this class
          **declines** the save (it has no way to keep the mutation), while
          the vector backends reload the peer snapshot and replay their
          pending buffer and redo logs on top (issue #3688) — so they lose
          nothing and report nothing. The flush-failure propagation a decline
          requires (``LightRAG._flush_storages``) is therefore this class's
          concern alone.
        * No ``client_storage`` property — there is no equivalent live
          reference being exposed to callers, so NanoVectorDB's
          "do-not-retain-across-await" caveat does not apply here.
        * ``write_nx_graph`` passes the tmp path directly to
          ``nx.write_graphml``, so the writer needs no equivalent of
          NanoVectorDB's "temporarily reassign ``storage_file``" trick.
        * Mutation surface is finer-grained (``upsert_node`` /
          ``upsert_edge`` / ``upsert_nodes_batch`` /
          ``upsert_edges_batch`` / ``delete_node`` / ``remove_nodes`` /
          ``remove_edges``); each goes through ``_get_graph`` once and
          then operates synchronously on ``self._graph``.

    Attribute validation (why this backend validates, and why the rule is
    narrower than the caller contract):
        The ``upsert_*`` methods reject an attribute name or value XML
        cannot encode (``validate_xml_attributes``) **before** touching
        ``self._graph``. This backend needs the guard more than the
        others because of the shape above, not because its callers are
        less trustworthy: the mutation happens in memory, the
        serialization that would reject the value happens later in
        ``index_done_callback``, and nothing rolls the mutation back. One
        unencodable value therefore stops *all* persistence for the life
        of the process -- ``write_nx_graph`` serializes the whole graph,
        so every later flush by any caller re-hits the same failure while
        reads keep succeeding. Validating first converts that into a
        failed single write. See GHSA-c922-pw4m-4wcv.

        The rule is exactly "can GraphML encode this", not the portable
        contract in ``BaseGraphStorage.upsert_node``. ``NaN``, ``inf``
        and an integer past int64 are all refused by that contract (the
        Neo4j driver cannot pack them) but round-trip through GraphML
        unchanged -- so a workspace can already hold one, and every
        rewrite path spreads a fetched object's stored attributes back
        into the upsert payload. Enforcing the portable rule here would
        make those objects permanently unmodifiable; enforcing it where
        caller input enters (``utils_graph``) costs nothing. The portable
        bounds are not this backend's to police.

        Names get the same XML rule and nothing more. GraphML writes them
        into the XML ``attr.name`` field, so an unencodable *name* breaks
        the write exactly like an unencodable value -- but ``a.b``,
        ``$set``, ``display-name`` and ``has space`` all round-trip, so
        refusing those would strand a node whose stored names predate
        this validation while preventing nothing. Rules about names a
        backend *interprets* belong to the backends that interpret them
        (MongoDB's ``$set`` paths).

        The XML name rule is safe to apply to a rewrite payload for the
        reason a portable rule would not be: a name it rejects can never
        have been persisted here, because the write that would have
        stored it failed.

        The batch variants validate the entire batch before applying any
        of it: rejecting halfway would leave the earlier items in the
        in-memory graph, which is exactly the partial-mutation state the
        guard exists to prevent.

        All four methods validate *after* ``_get_graph()`` -- their only
        await -- and then mutate with nothing awaited in between. That
        ordering is what makes the guard airtight rather than advisory:
        by invariant (3) above a synchronous run cannot be preempted, so a
        caller that retains the mapping it passed in has no window in which
        to add a value after the check and before ``add_node`` /
        ``add_edge`` consumes it.

    Commit granularity — a commit publishes the whole namespace:
        ``index_done_callback`` serializes ``self._graph`` in full and
        renames the result over the GraphML file. There is no scoped or
        transactional commit, and none is planned (issue #3838 rejects one
        for the file-backed storages explicitly). Two consequences the
        callers have to live with:

        * **Any writer's flush durably publishes every other writer's
          pending in-memory mutation in this namespace**, including partial
          state its author had not finished. A caller that stages its own
          writes to guarantee an ordering guarantees it only for its own
          objects; a co-tenant's half-applied sequence rides along.
        * The pipeline tolerates this, because ``doc_status`` plus
          idempotent FAILED reprocessing rewrites whatever a restart finds
          half-applied. The admin flows below have no equivalent, which is
          what makes their residue worth stating rather than assuming away.

    Scope decision (issue #3838):
        This backend — like ``JsonKVStorage`` / ``JsonDocStatusStorage`` /
        ``NanoVectorDBStorage`` / ``FaissVectorDBStorage`` — is supported for
        **small-scale testing and validation only**. Production deployments
        run a server-backed graph store. Write throughput of the whole-file
        commit path is therefore not a consideration, and no change to this
        file may be justified by — or blocked on — it.

    Non-pipeline write paths:
        The pipeline's ``busy`` gate serializes mutation calls reached
        through the document ingestion and purge flows. The following
        entry points are **not** serialized by the pipeline gate:
            * ``drop`` — gated by the API layer (the ``/documents/clear``
              endpoint takes the pipeline busy reservation before invoking
              it).
            * ``delete_node`` / ``remove_nodes`` / ``remove_edges`` /
              ``upsert_node`` / ``upsert_edge`` when invoked from the
              ``utils_graph.py`` admin flows (``adelete_by_entity`` /
              ``adelete_by_relation`` / the create, edit and merge paths).

        For the admin flows, invariant 1 is **not** met and is not going to
        be. Stating it accurately, because the previous wording ("must be
        guarded externally", plus a claim that the flows are not exposed in
        the WebUI) was false on both counts — the WebUI calls
        ``/graph/entity/edit``:

        * Admin-vs-pipeline **is** guarded: every graph mutation endpoint
          calls ``check_pipeline_busy_or_raise`` and returns 409 while the
          pipeline is busy. That check is a snapshot, so a narrow race with
          the underlying write remains; closing it would mean holding
          ``busy`` across every UI edit, which is deliberately rejected.
        * Admin-vs-admin is **not** guarded. Two concurrent
          ``/graph/*`` calls for different keys both pass the busy check and
          proceed under different per-entity keyed locks. A workspace-wide
          admin lock was specified (issue #3838 R3) and then dropped: with
          the reload fence of issue #3854 in place the losing writer of an
          overlapping pair *declines*, so the caller gets a loud 500 and
          retries rather than losing the write silently; and the residue the
          lock would have prevented is reachable with no concurrency at all,
          so serializing admin writes would not have changed what a crash can
          leave behind.

        **Accepted residue.** A hard process exit, or an ill-timed co-tenant
        flush, can leave a chunk-tracking row whose graph object never became
        durable. It is harmless to queries; it cannot be inherited as
        evidence by a later object, because the explicit creation paths reset
        attribution (issue #3838 R1); and it is repairable offline with the
        chunk-tracking rebuild tool (R4). The mirror state — a graph object
        durable without its tracking row — is the forbidden one, and the
        creation paths in ``utils_graph`` keep it out of reach by writing and
        committing the tracking row **before** calling ``upsert_node`` /
        ``upsert_edge`` at all. Ordering only the flushes would not be enough
        here: an uncommitted ``upsert_node`` still sits in the process-wide
        in-memory graph, where the next flush by any co-tenant publishes it.

        The edit paths reach the same guarantee by a different route, because
        their delta can also REMOVE ids and a narrowed row landing ahead of the
        graph write is itself the over-deleting state: ``aedit_relation`` and
        ``_edit_entity_impl``'s non-rename branch stage the row as
        grow-then-shrink -- superset row, graph write, final row -- so the
        durable row is never a strict subset of the durable evidence.

        **Requirement on new callers.** Any new caller of these mutators must
        make the tracking row durable BEFORE the mutation call, not merely
        before the flush, and must not be invoked concurrently with another
        admin writer on a file-backed workspace.
    """

    def _node_context(self, node_id: str) -> str:
        """Error-message prefix identifying a node write."""
        return f"[{self.workspace}] node `{node_id}`"

    def _edge_context(self, source_node_id: str, target_node_id: str) -> str:
        """Error-message prefix identifying an edge write."""
        return f"[{self.workspace}] edge `{source_node_id}`~`{target_node_id}`"

    @staticmethod
    def load_nx_graph(file_name) -> nx.Graph:
        if os.path.exists(file_name):
            return nx.read_graphml(file_name)
        return None

    @staticmethod
    def write_nx_graph(graph: nx.Graph, file_name, workspace="_"):
        logger.info(
            f"[{workspace}] Writing graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        )
        atomic_write(
            file_name,
            lambda tmp: nx.write_graphml(graph, tmp),
            workspace,
        )

    def __post_init__(self):
        # Reject path traversal before using workspace in a file path
        validate_workspace(self.workspace)
        working_dir = self.global_config["working_dir"]
        if self.workspace:
            # Include workspace in the file path for data isolation
            workspace_dir = os.path.join(working_dir, self.workspace)
        else:
            # Default behavior when workspace is empty
            workspace_dir = working_dir
            self.workspace = ""

        os.makedirs(workspace_dir, exist_ok=True)
        self._graphml_xml_file = os.path.join(
            workspace_dir, f"graph_{self.namespace}.graphml"
        )
        self._storage_lock = None
        self.storage_updated = None
        self._graph = None
        # ``(st_mtime_ns, st_size)`` of the GraphML file this process last
        # loaded or wrote -- the authoritative half of the cross-process
        # fence. ``None`` means "no file" (or a stat this process could not
        # perform). See *Cross-process sync protocol* in the class docstring.
        self._loaded_fingerprint = None
        # A reload this process owes ITSELF: armed at the moment the reload
        # becomes owed, cleared only by one that completes. Process-local on
        # purpose -- it says "my memory diverged from the file", which no peer
        # can observe and none needs to. See *Recovery reload* in the class
        # docstring.
        self._recovery_reload_pending = False
        # The on-disk state the file channel has already counted as a lost
        # notification. A reload that fails leaves the fingerprint and the
        # recovery flag untouched, so the same peer commit is re-detected by
        # every later call; without this it would also be re-counted, without
        # bound. See _count_unannounced_peer_commit_locked.
        self._counted_peer_fingerprint = None
        # How many times the file channel caught a commit the flag channel
        # never announced. Reported in the warning that records each one, so a
        # deployment can tell whether the lost-notification window in #3854
        # actually occurs rather than only being reachable on paper.
        self._missed_notification_reloads = 0
        # Created lazily by _gate(): asyncio.Event binds to a loop on first
        # wait, and instances are constructed outside any running loop.
        self._commit_gate = None
        self._commit_gate_loop = None

        reap_orphan_tmp_files(self._graphml_xml_file, workspace=self.workspace or "_")

        # Load initial graph. The fingerprint is sampled BEFORE the read --
        # see the ordering rule in *Cross-process sync protocol*.
        fingerprint = self._stat_fingerprint()
        preloaded_graph = NetworkXStorage.load_nx_graph(self._graphml_xml_file)
        if preloaded_graph is not None:
            logger.info(
                f"[{self.workspace}] Loaded graph from {self._graphml_xml_file} with {preloaded_graph.number_of_nodes()} nodes, {preloaded_graph.number_of_edges()} edges"
            )
        else:
            logger.info(
                f"[{self.workspace}] Created new empty graph file: {self._graphml_xml_file}"
            )
        self._graph = preloaded_graph or nx.Graph()
        self._adopt_fingerprint(fingerprint)

    async def initialize(self):
        """Initialize storage data"""
        # Get the update flag for cross-process update notification
        self.storage_updated = await get_update_flag(
            self.namespace, workspace=self.workspace
        )
        # Get the storage lock for use in other methods
        self._storage_lock = get_namespace_lock(
            self.namespace, workspace=self.workspace
        )

    def _gate(self) -> asyncio.Event:
        """The commit gate: set except while this process is serializing.

        See *Commit gate* in the class docstring. Lazily created so the
        ``asyncio.Event`` binds to whichever loop first waits on it, which is
        never the loop-less ``__post_init__``.

        Rebuilt when the running loop changes. An ``asyncio.Event`` binds to a
        loop the first time a ``wait()`` actually suspends on it, and from then
        on ``wait()`` from any other loop raises "is bound to a different event
        loop". A single instance CAN legitimately outlive its loop: the
        synchronous wrappers let calls through on a fresh loop once the original
        ``owning_loop`` has closed (see ``_run_sync`` — the common
        ``rag = asyncio.run(initialize_rag())`` shape), and a gate bound during
        some earlier commit would then break the next one.

        Rebinding is safe rather than a race: a commit in flight holds
        ``_storage_lock`` and runs on the loop being replaced, so reaching here
        on a different loop means no commit of ours is outstanding and an open
        gate is the correct state. Lazily rebuilt here, not eagerly in
        ``initialize()``, because that is not where the loop change shows up.
        """
        loop = asyncio.get_running_loop()
        if self._commit_gate is None or self._commit_gate_loop is not loop:
            self._commit_gate = asyncio.Event()
            self._commit_gate.set()
            self._commit_gate_loop = loop
        return self._commit_gate

    def _stat_fingerprint(self) -> file_fingerprint.Fingerprint | object:
        """Sample the GraphML file's identity. See ``kg.file_fingerprint``."""
        return file_fingerprint.sample(
            (self._graphml_xml_file,), workspace=self.workspace
        )

    def _adopt_fingerprint(
        self, fingerprint: file_fingerprint.Fingerprint | object
    ) -> None:
        """Record ``fingerprint`` as the file this process now holds."""
        adopted = file_fingerprint.adopted(fingerprint)
        self._loaded_fingerprint = adopted
        # The dedupe marker's job ends here -- but ONLY if a concrete state
        # was recorded. It exists to stop a detection being re-counted while
        # the reload that should discharge it keeps failing, and a landed
        # reload normally ends that: ``_loaded_fingerprint`` IS this state
        # from here, so any later divergence is genuinely new. Keeping it
        # past that point would suppress a state that RECURS -- a peer drop,
        # a notified recreation, then a second drop whose notification is
        # lost, all sharing the "absent" fingerprint, which is a real second
        # loss and not the same-tick collision residue.
        #
        # ``adopted(UNREADABLE)`` is ``None``, which is not a state: it means
        # "nothing recorded", and ``peer_commit_detected`` reports a change
        # against it for ANY state. Clearing on that would forget which
        # commit was already counted and count the same one again on the next
        # call. The post-drop fingerprint is ``(None,)`` -- a real, concrete
        # state -- so a drop still clears.
        if adopted is not None:
            self._counted_peer_fingerprint = None

    def _peer_commit_detected(
        self, sampled: file_fingerprint.Fingerprint | object | None = None
    ) -> bool:
        """Whether the file on disk differs from the one this process loaded.

        The fence's authoritative test — the one a failed notification cannot
        disable. ``False`` in single-process mode and on an unreadable
        ``stat``; see ``kg.file_fingerprint`` for both.

        ``sampled`` is for the caller that will also count and reload from one
        observation -- it decides from that same sample rather than taking a
        third ``stat`` that could fail on its own. See
        ``file_fingerprint.divergence_detected``.
        """
        if sampled is None:
            return file_fingerprint.peer_commit_detected(
                (self._graphml_xml_file,),
                self._loaded_fingerprint,
                workspace=self.workspace,
            )
        return file_fingerprint.divergence_detected(
            sampled,
            self._loaded_fingerprint,
            paths=(self._graphml_xml_file,),
            workspace=self.workspace,
        )

    def _count_unannounced_peer_commit_locked(
        self, sampled: file_fingerprint.Fingerprint | object
    ) -> bool:
        """Count a peer commit no notification announced. True if it counted.

        Precondition: the caller holds ``_storage_lock``, and has NOT reloaded
        yet -- a reload adopts the file, after which the question cannot be
        asked any more. The caller logs; this decides and counts, so
        ``_missed_notification_reloads`` has exactly one increment site.

        ``_missed_notification_reloads`` is the evidence the writer-side
        ``os.utime`` decision waits on (see ``kg.file_fingerprint``), so it has
        to count *unannounced on-disk states*, not detections of them -- one
        per distinct state this process found, however many detections or
        reload attempts that took. (Per state, not per commit: this is a state
        channel, so peer commits that batch into one observation are one
        increment. ``kg.file_fingerprint`` records that as one of the two
        by-design blind spots.) State and detection are not the same thing,
        in both directions:

        * **Undercount.** The recovery flag wins over both channel tests and
          one reload discharges all of them, so a peer commit that arrives
          while recovery is pending would be handled and never counted. That
          is why the recovery branches -- and the failed-save handler's own
          recovery reload -- call this at all.
        * **Overcount.** A reload that raises leaves ``_loaded_fingerprint``
          and ``_recovery_reload_pending`` exactly as they were, so the same
          peer commit is re-detected by every later call -- and, counted at
          detection, re-counted every time, without bound. Persistent
          unreadability is not exotic here: a failed reload is what arms
          recovery in the first place. ``_counted_peer_fingerprint`` plus
          ``file_fingerprint.counts_as_a_new_lost_notification`` (shared with
          the vector backends, which have the same counter and had the same
          defect) makes the count once-per-state instead of once-per-attempt.
          See that function for what it does and does not distinguish.

        ``sampled`` is the caller's own sample, and the caller MUST hand the
        same one to ``_reload_locked``. Sampling independently here would make
        the count and the adoption describe two different observations, and a
        transient ``UNREADABLE`` on this one alone would then be uncountable
        forever: this helper would skip the increment while the reload's own
        successful stat adopted the peer state, erasing the divergence that
        would have let a later call count it. Sharing one sample keeps the two
        outcomes tied -- either the event is counted, or no fingerprint is
        adopted that could suppress counting it next time. The vector backends
        share their pre-read sample for exactly this reason.

        Self-contained otherwise: it re-tests the flag and re-applies the full
        divergence decision even where the caller's branch condition already
        established them, so no site can count by satisfying only half the
        condition. It re-*observes* nothing, though -- the divergence test runs
        against ``sampled`` too. The flag re-read is safe to repeat because it
        is not an observation of the file: a flag that turned True in between
        means the notification arrived after all, and declining to count that
        is correct.
        """
        if self.storage_updated.value:
            return False
        if not self._peer_commit_detected(sampled):
            return False
        if not file_fingerprint.counts_as_a_new_lost_notification(
            sampled, self._counted_peer_fingerprint
        ):
            logger.debug(
                f"[{self.workspace}] The peer commit to {self._graphml_xml_file} "
                "is the one already counted, or its stat failed; this is a "
                "retry of a reload that did not land, not a second lost "
                "notification."
            )
            return False
        self._counted_peer_fingerprint = file_fingerprint.adopted(sampled)
        self._missed_notification_reloads += 1
        return True

    def _reload_locked(
        self, fingerprint: file_fingerprint.Fingerprint | object | None = None
    ) -> None:
        """Reload ``self._graph`` from disk and satisfy BOTH fence channels.

        Precondition: the caller holds ``_storage_lock``.

        ``fingerprint`` is for the callers that already sampled in order to
        count -- they pass theirs in so the count and the adoption describe
        one observation; see ``_count_unannounced_peer_commit_locked``.
        Omitted, this samples for itself. ``None`` is unambiguous as "not
        given": ``_stat_fingerprint`` returns a tuple or ``UNREADABLE``, never
        ``None``.

        One reload, one post-condition, whichever condition fired: record the
        new fingerprint, clear the flag, **and** clear the pending recovery
        reload. Open-coding any of the three is the way this fence rots -- a
        missed fingerprint or a missed flag clear costs a full GraphML
        re-parse on the very next call, and a missed recovery clear is worse:
        it is sticky, so it would make every later commit decline forever.

        And one on FAILURE: whatever went wrong, the pending recovery reload
        is armed before the exception leaves. So this method's own invariant
        is total -- after it returns the graph matches the file, after it
        raises a reload is owed and recorded. That is a backstop, not the
        mechanism: callers arm at the branch entry instead, because the
        classification between there and here can raise on its own. See
        *Recovery reload*.

        **Raises** ``OSError`` without loading anything when the sample is
        ``UNREADABLE`` -- given or taken here. A reload has to record what it
        read, and an unsampled file cannot be recorded; the branch below says
        what each half of that costs. Every caller's ``except`` already treats
        a failed reload as "the divergence stands, retry next call", so this
        needs no new handling -- but it does mean no reload path can adopt
        ``UNREADABLE``. The only remaining producers of a ``None``
        fingerprint are ``_record_fingerprint``'s adoptions of this process's
        OWN commit or drop, and ``initialize``'s first load. What that costs
        is one redundant reload -- and it discards nothing only because
        ``_get_graph`` settles the ``None`` before it serves the graph:
        through the divergence test if it can sample, by REFUSING if it
        cannot. Deferred instead, that same reload lands mid-batch and drops
        what was applied in the meantime; see that branch.

        Synchronous on purpose: it adds no suspension point inside
        ``_get_graph``'s lock body, which the *Commit gate* reasoning depends
        on.
        """
        try:
            self._reload_or_arm_locked(fingerprint)
        except BaseException:
            # ARM, then re-raise. This is the whole reason the reload is not
            # open-coded at its five call sites: a reload that fails leaves
            # this process holding a graph that does not match the file, and
            # the operation that asked for it fails -- so the mutations in it
            # belong to a batch that has just been reported as failed. Exactly
            # the state _recovery_reload_pending exists for, and until this
            # was the single choke point only the failed-SAVE path armed it.
            #
            # What the four other sites left open: the flag and the
            # fingerprint stay as they were, which keeps the divergence
            # visible only while the file can be sampled. Let the stat keep
            # failing and both channels report "no change" (the documented
            # degrade to the flag alone), so the next commit passes the fence
            # and serializes that graph -- overwriting the peer's durable
            # commit AND publishing work already reported as FAILED. Armed,
            # the same commit declines instead and the next successful reload
            # discards it.
            #
            # BaseException, not Exception: a CancelledError delivered inside
            # the load leaves the same divergence, and losing the record of it
            # because the caller went away is how it becomes permanent.
            self._recovery_reload_pending = True
            raise

    def _reload_or_arm_locked(
        self, fingerprint: file_fingerprint.Fingerprint | object | None
    ) -> None:
        """``_reload_locked``'s body. Call THAT, never this -- it arms.

        Split out only so the arming wrapper has one expression to guard;
        every post-condition and every refusal documented on
        ``_reload_locked`` lives here.
        """
        # Sampled before the read, never after. See the ordering rule in
        # *Cross-process sync protocol*.
        if fingerprint is None:
            fingerprint = self._stat_fingerprint()
        if fingerprint is file_fingerprint.UNREADABLE:
            # REFUSE, do not load. A reload out of an unreadable sample is
            # unsafe twice over, and both ways cost data:
            #
            # * ``load_nx_graph`` gates on ``os.path.exists``, which reports
            #   False for *any* stat failure -- the same failure that produced
            #   UNREADABLE. So the load returns None, ``or nx.Graph()`` makes
            #   that an EMPTY graph, and while the stat keeps failing the fence
            #   cannot object (``divergence_detected`` returns False for
            #   UNREADABLE): the next commit serializes the empty graph over
            #   the real file. That is the outcome *Cross-process sync
            #   protocol* names as the one thing this fence must never do.
            # * Even with the file readable and only its stat failing,
            #   ``adopted(UNREADABLE)`` records ``None``, against which every
            #   state is a divergence -- so the NEXT ``_get_graph`` reloads
            #   again, and that second reload discards whatever this process
            #   mutated in between. The commit after it then succeeds without
            #   those mutations and their document is marked PROCESSED: a
            #   silent partial loss. See ``file_fingerprint.adopted``.
            #
            # Raising instead costs work, never data: the flag and the
            # fingerprint are left exactly as they were, so the divergence
            # stays visible and the next call retries the reload; the
            # recovery reload is ARMED by _reload_locked's wrapper, so the
            # divergence survives even a stat that never recovers; and the
            # operation fails, which takes its batch through the FAILED path
            # and reprocesses it. Same shape as a manager outage during
            # recovery.
            raise OSError(
                f"[{self.workspace}] Refusing to reload "
                f"{self._graphml_xml_file}: its identity could not be sampled, "
                "so a load could neither be trusted to have read the current "
                "file nor be recorded as one. Retrying on the next call."
            )
        self._graph = (
            NetworkXStorage.load_nx_graph(self._graphml_xml_file) or nx.Graph()
        )
        self._adopt_fingerprint(fingerprint)
        self.storage_updated.value = False
        # Cleared LAST, and only once the load above has actually returned:
        # a load that raises leaves it armed, which is exactly what keeps the
        # divergence visible to the next call.
        self._recovery_reload_pending = False

    def _record_fingerprint(self) -> None:
        """Adopt the file currently on disk without reloading from it.

        For the writer: after its own commit the in-memory graph already *is*
        the file's content, so the fingerprint is recorded and no reload is
        needed.
        """
        self._adopt_fingerprint(self._stat_fingerprint())

    async def _get_graph(self):
        """Return the live ``networkx.Graph``, reloading from disk if needed.

        This is the **single entry point** every public method funnels
        through to obtain ``self._graph``. It is also the **only place
        readers transition to a fresher on-disk snapshot**: when another
        process has committed (via ``index_done_callback``), the next call
        here rebuilds ``self._graph`` by re-parsing the entire GraphML file.
        networkx has no incremental sync API — the reload is
        unconditionally a full file reload.

        Two tests decide that, and both must stay — see *Cross-process sync
        protocol* in the class docstring for why neither is redundant: this
        process's ``storage_updated`` flag, and the GraphML file's
        ``(st_mtime_ns, st_size)`` against the fingerprint recorded when this
        process last loaded or wrote it.

        Two conditions that are not channels are tested alongside them: the
        process-local recovery reload FIRST (see *Recovery reload*), and an
        unresolved fingerprint — a recorded ``None`` — LAST. The file channel
        settles that one itself whenever it can sample: any concrete state is
        a divergence against ``None``, so it reloads. What the last test
        covers is the case where it cannot, since an unreadable ``stat``
        reports "no change" and would leave the question open across a
        mutation. Each branch below says why.

        Under the *Single writer* invariant (see class docstring), neither
        branch fires in the writer process: the writer resets its own flag
        and adopts its own file's fingerprint at the end of every
        ``index_done_callback``. Both exist for readers — and for the process
        that becomes the *next* writer, which the invariant does not require
        to be the same one.

        ``_storage_lock`` is held during the check-and-reload to (a)
        serialize concurrent reload attempts by sibling coroutines in
        the same process and (b) interlock with ``index_done_callback``
        so a reader cannot observe a partially-saved file.
        """
        async with self._storage_lock:
            # The process-local recovery reload is tested FIRST, before either
            # cross-process channel. It is free, it cannot be wrong, and it
            # answers a different question than they do: not "did a peer
            # commit?" but "is my own memory unpersisted?". Testing it here
            # also keeps the two channels' log lines describing only what they
            # name -- see *Recovery reload* in the class docstring. What that
            # precedence must NOT do is hide a peer commit from the counter,
            # which is why the branch below classifies before it reloads.
            if self._recovery_reload_pending:
                # Classify the peer channel BEFORE reloading. One reload
                # discharges both conditions, but _reload_locked adopts the
                # file, so afterwards nothing can tell that a peer commit had
                # also arrived unannounced -- and that is a number this fence
                # is measured by.
                # One sample, shared by the counting and the reload -- see
                # _count_unannounced_peer_commit_locked for why they must not
                # be two independent observations.
                sampled = self._stat_fingerprint()
                if self._count_unannounced_peer_commit_locked(sampled):
                    logger.warning(
                        f"[{self.workspace}] Process {os.getpid()}: the file on "
                        f"disk ({self._graphml_xml_file}) is not the one this "
                        "process loaded and no reload notification arrived for "
                        "it, so a notification was lost. Recovered through the "
                        "file channel, folded into the recovery reload below "
                        f"(occurrence #{self._missed_notification_reloads} in "
                        "this process)."
                    )
                logger.warning(
                    f"[{self.workspace}] Process {os.getpid()} reloading graph "
                    f"{self._graphml_xml_file}: a reload this process owed "
                    "itself failed earlier -- after a failed save, or at "
                    "either fence channel -- so its graph does not match the "
                    "file and holds mutations from an operation already "
                    "reported as failed. Discarding them now."
                )
                self._reload_locked(sampled)
            # Flag next -- it is the accelerator channel and a True value
            # already answers the question. The fingerprint test in the elif is
            # the authoritative one: it is what makes a lost notification
            # recoverable. See *Cross-process sync protocol*.
            elif self.storage_updated.value:
                # OWED from here on, so record it before anything can go wrong
                # on the way to discharging it -- see *Recovery reload*. A
                # completed _reload_locked clears it again.
                self._recovery_reload_pending = True
                logger.info(
                    f"[{self.workspace}] Process {os.getpid()} reloading graph {self._graphml_xml_file} due to modifications by another process"
                )
                # One sample for the log line and the reload alike. This
                # branch counts nothing (a notification DID arrive, so nothing
                # was lost), but the two must still agree on what they saw:
                # sampling twice let the log claim the file was unchanged while
                # the reload refused an unreadable one.
                sampled = self._stat_fingerprint()
                if sampled is file_fingerprint.UNREADABLE:
                    logger.debug(
                        f"[{self.workspace}] Reload notification for "
                        f"{self._graphml_xml_file}, whose identity could not be "
                        "sampled; the reload below refuses rather than load "
                        "blind"
                    )
                elif file_fingerprint.fence_enabled() and not (
                    self._peer_commit_detected(sampled)
                ):
                    # Notified about the file this process already holds: a
                    # self-notification, or a peer commit that a sibling
                    # coroutine reloaded before this call got the lock. The
                    # reload below is honoured anyway rather than skipped --
                    # skipping it would change which uncommitted in-memory
                    # mutations survive a notification, which is not this
                    # fence's business.
                    logger.debug(
                        f"[{self.workspace}] Reload notification for "
                        f"{self._graphml_xml_file} names the snapshot already "
                        "loaded; reloading anyway"
                    )
                self._reload_locked(sampled)
            elif file_fingerprint.fence_enabled():
                # ONE observation for everything the file channel does on this
                # call: the decision, the count, the adoption, and the
                # unresolved-fingerprint test below. That is the rule
                # file_fingerprint.divergence_detected states, and it is
                # satisfied here rather than argued around: this branch used to
                # sample once in its condition and again for the count, which
                # held only because _storage_lock is cross-process and there is
                # no await between the two -- an unwritten invariant propping up
                # a contract that says three-from-one, with no exceptions.
                #
                # In single-process mode the fence is off entirely (a divergent
                # file is an external edit, not a peer commit), so no stat is
                # taken at all -- the same as before.
                sampled = self._stat_fingerprint()
                if self._peer_commit_detected(sampled):
                    # Owed from here on; recorded before the classification,
                    # which reads storage_updated.value over the Manager and
                    # can raise on its own. See *Recovery reload*.
                    self._recovery_reload_pending = True
                    if self._count_unannounced_peer_commit_locked(sampled):
                        logger.warning(
                            f"[{self.workspace}] Process {os.getpid()} reloading "
                            f"graph {self._graphml_xml_file}: the file on disk is "
                            "not the one this process loaded and no reload "
                            "notification arrived for it, so a notification was "
                            "lost. Recovered through the file channel (occurrence "
                            f"#{self._missed_notification_reloads} in this "
                            "process)."
                        )
                    self._reload_locked(sampled)
                elif self._loaded_fingerprint is None:
                    # Owed from here on, as above -- and here the reload always
                    # raises, so the record is all that survives the call.
                    self._recovery_reload_pending = True
                    # An UNRESOLVED fingerprint with no way to resolve it: the
                    # sample must be ``UNREADABLE`` to get here, because a
                    # concrete one is a divergence against ``None`` and was
                    # handled above. So this always REFUSES -- ``_reload_locked``
                    # raises on that sample, and it is called rather than
                    # open-coding the raise so the refusal has one message.
                    #
                    # Why refusing is the answer. ``None`` is recorded only by
                    # an adoption whose ``stat`` failed --
                    # ``_record_fingerprint`` after this process's own commit
                    # or drop, or ``initialize``'s first load (no reload path
                    # can record it, since they refuse the same sample). The
                    # graph therefore matches the file, and the redundant
                    # reload that settles the ``None`` is harmless -- but only
                    # if it happens BEFORE the next mutation, and the
                    # divergence test cannot make it happen while the ``stat``
                    # keeps failing (``UNREADABLE`` reports "no change").
                    # Serving the graph here is what opened that window:
                    # a mutation landed on it and the reload was deferred to
                    # whichever later call first managed to ``stat`` --
                    # mid-batch, discarding that mutation, after which the
                    # commit SUCCEEDS without it and its document is marked
                    # PROCESSED.
                    #
                    # The two remedies that do not refuse are both worse.
                    # Re-sampling for a second chance breaks the
                    # one-observation rule this branch exists inside, for a
                    # ``stat`` that failed microseconds ago. Adopting the
                    # fresh ``stat`` without reloading would take a peer's
                    # commit for this process's own -- it cannot tell them
                    # apart, that is what ``None`` means -- and overwrite it on
                    # the next save: a DURABLE write lost, silently, which is
                    # the defect this whole fence exists for.
                    logger.warning(
                        f"[{self.workspace}] Process {os.getpid()} refusing to "
                        f"serve graph {self._graphml_xml_file}: this process "
                        "could not record the identity of the file it last "
                        "adopted, and cannot sample it now either, so it can "
                        "neither tell that file from a peer's later commit nor "
                        "settle the question before a mutation lands on this "
                        "snapshot."
                    )
                    self._reload_locked(sampled)

            graph = self._graph

        # The gate is checked HERE, after the lock block, and that placement is
        # the whole point -- see *Commit gate* in the class docstring. Holding
        # _storage_lock in the committer does not exclude a caller that is
        # already past this method's lock body: releasing a NamespaceLock runs
        # the release on a fresh task and awaits a shield, so __aexit__ ALWAYS
        # suspends at least once. Such a caller would resume and mutate
        # self._graph while the committer's worker thread iterates graph._node /
        # graph._adj.
        #
        # After this check there is no suspension point left: is_set() does not
        # await, so the caller's synchronous graph.add_node() / remove_node()
        # cannot have a commit start in the middle of it.
        #
        # A loop, not one wait(): a woken waiter's wait() returns True even if a
        # second committer has cleared the gate again in the meantime.
        while not self._gate().is_set():
            await self._gate().wait()
        return graph

    async def has_node(self, node_id: str) -> bool:
        graph = await self._get_graph()
        return graph.has_node(node_id)

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        graph = await self._get_graph()
        return graph.has_edge(source_node_id, target_node_id)

    async def get_node(self, node_id: str) -> dict[str, str] | None:
        graph = await self._get_graph()
        node = graph.nodes.get(node_id)
        # Shallow-copy so callers cannot mutate the live NetworkX attr dict
        # (same class as JsonKV/JsonDocStatus copy-on-read). get_all_nodes
        # already copies; get_node/get_edge must match.
        return dict(node) if node is not None else None

    async def node_degree(self, node_id: str) -> int:
        graph = await self._get_graph()
        if graph.has_node(node_id):
            return graph.degree(node_id)
        return 0

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        graph = await self._get_graph()
        src_degree = graph.degree(src_id) if graph.has_node(src_id) else 0
        tgt_degree = graph.degree(tgt_id) if graph.has_node(tgt_id) else 0
        return src_degree + tgt_degree

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> dict[str, str] | None:
        graph = await self._get_graph()
        edge = graph.edges.get((source_node_id, target_node_id))
        return dict(edge) if edge is not None else None

    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        graph = await self._get_graph()
        if graph.has_node(source_node_id):
            return list(graph.edges(source_node_id))
        return None

    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        """Insert or update a single node; persistence is deferred.

        Persistence:
            Changes are in-memory only; cross-process visibility requires
            a subsequent ``index_done_callback``. In ``lightrag.py`` this
            is handled by ``_insert_done()`` at the end of the document
            batch. Callers outside the pipeline must persist explicitly.

        Correctness relies on the class docstring *Lock scope* invariant
        (synchronous networkx ops + single-writer pipeline gate).

        Validates before mutating: see *Attribute validation* in the class
        docstring.
        """
        graph = await self._get_graph()
        # Validate *after* the only await, so the check and the mutation are one
        # synchronous block -- see *Attribute validation* in the class docstring.
        validate_xml_attributes(node_data, context=self._node_context(node_id))
        graph.add_node(node_id, **node_data)

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ) -> None:
        """Insert or update a single edge; persistence is deferred.

        Persistence:
            Changes are in-memory only; cross-process visibility requires
            a subsequent ``index_done_callback``. Callers outside the
            pipeline must persist explicitly.

        Correctness relies on the class docstring *Lock scope* invariant.

        Validates before mutating: see *Attribute validation* in the class
        docstring.
        """
        graph = await self._get_graph()
        # See upsert_node: checked after the await, mutated with none in between.
        validate_xml_attributes(
            edge_data, context=self._edge_context(source_node_id, target_node_id)
        )
        graph.add_edge(source_node_id, target_node_id, **edge_data)

    async def upsert_nodes_batch(self, nodes: list[tuple[str, dict[str, str]]]) -> None:
        """Batch insert/update multiple nodes in a single call.

        Much faster than calling upsert_node() in a loop for large imports
        because it avoids per-call async event loop overhead.

        Persistence:
            Changes are in-memory only; cross-process visibility requires
            a subsequent ``index_done_callback``. Callers outside the
            pipeline must persist explicitly.

        Args:
            nodes: List of (node_id, node_data) tuples.
        """
        graph = await self._get_graph()
        # Validate the whole batch before applying any of it: a rejection halfway
        # through the apply loop would leave the earlier nodes in the in-memory
        # graph, which is the partial-mutation state this exists to prevent. Both
        # loops run after the only await, with none in between.
        for node_id, node_data in nodes:
            validate_xml_attributes(node_data, context=self._node_context(node_id))
        for node_id, node_data in nodes:
            graph.add_node(node_id, **node_data)

    async def has_nodes_batch(self, node_ids: list[str]) -> set[str]:
        """Check existence of multiple nodes in a single call.

        Returns:
            Set of node_ids that exist in the graph.
        """
        graph = await self._get_graph()
        return {nid for nid in node_ids if graph.has_node(nid)}

    async def upsert_edges_batch(
        self, edges: list[tuple[str, str, dict[str, str]]]
    ) -> None:
        """Batch insert/update multiple edges in a single call.

        Persistence:
            Changes are in-memory only; cross-process visibility requires
            a subsequent ``index_done_callback``. Callers outside the
            pipeline must persist explicitly.

        Args:
            edges: List of (source_id, target_id, edge_data) tuples.
        """
        graph = await self._get_graph()
        # Whole batch first, after the only await -- see upsert_nodes_batch.
        for src, tgt, edge_data in edges:
            validate_xml_attributes(edge_data, context=self._edge_context(src, tgt))
        for src, tgt, edge_data in edges:
            graph.add_edge(src, tgt, **edge_data)

    async def delete_node(self, node_id: str) -> None:
        """Remove a single node from the graph; persistence is deferred.

        Persistence:
            Changes are in-memory only; cross-process visibility requires
            a subsequent ``index_done_callback``. Callers outside the
            pipeline must persist explicitly.

        Pipeline-gating depends on the caller: invocations from the
        document purge flow are serialized by ``pipeline busy``;
        invocations from ``utils_graph.py`` admin flows are **not** —
        see class docstring *Non-pipeline write paths*.
        """
        graph = await self._get_graph()
        if graph.has_node(node_id):
            graph.remove_node(node_id)
            logger.debug(f"[{self.workspace}] Node {node_id} deleted from the graph")
        else:
            logger.warning(
                f"[{self.workspace}] Node {node_id} not found in the graph for deletion"
            )

    async def remove_nodes(self, nodes: list[str]):
        """Delete multiple nodes from the graph.

        Persistence:
            Changes are in-memory only; cross-process visibility requires
            a subsequent ``index_done_callback``. Callers outside the
            pipeline must persist explicitly.

        Pipeline-gating depends on the caller — see ``delete_node`` and
        class docstring *Non-pipeline write paths*.

        Args:
            nodes: List of node IDs to be deleted
        """
        graph = await self._get_graph()
        for node in nodes:
            if graph.has_node(node):
                graph.remove_node(node)

    async def remove_edges(self, edges: list[tuple[str, str]]):
        """Delete multiple edges from the graph.

        Persistence:
            Changes are in-memory only; cross-process visibility requires
            a subsequent ``index_done_callback``. Callers outside the
            pipeline must persist explicitly.

        Pipeline-gating depends on the caller — see ``delete_node`` and
        class docstring *Non-pipeline write paths*.

        Args:
            edges: List of edges to be deleted, each edge is a (source, target) tuple
        """
        graph = await self._get_graph()
        for source, target in edges:
            if graph.has_edge(source, target):
                graph.remove_edge(source, target)

    async def get_all_labels(self) -> list[str]:
        """
        Get all node labels(entity names) in the graph
        Returns:
            [label1, label2, ...]  # Alphabetically sorted label list
        """
        graph = await self._get_graph()
        labels = set()
        for node in graph.nodes():
            labels.add(str(node))  # Add node id as a label

        # Return sorted list
        return sorted(list(labels))

    async def iter_labels(self, batch_size: int):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        graph = await self._get_graph()
        batch: list[str] = []
        for node in graph.nodes():
            batch.append(str(node))
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    async def get_popular_labels(self, limit: int = 300) -> list[str]:
        """
        Get popular labels(entity names) by node degree (most connected entities)

        Args:
            limit: Maximum number of labels to return

        Returns:
            List of labels sorted by degree (highest first), ties broken on the
            label ascending
        """
        graph = await self._get_graph()

        # Degree descending, then label ascending. The tie-break is not
        # cosmetic: `sorted(..., key=degree, reverse=True)` is stable, so ties
        # used to come back in node INSERTION order, and when more labels share
        # the cutoff degree than fit in `limit` that decided which ones the
        # caller never sees — a graph that happened to insert "Zeta" before
        # "Alpha" returned Zeta and dropped Alpha. Every other backend orders
        # ties by label (SQL `ORDER BY degree DESC, label ASC` / COLLATE "C",
        # Cypher `ORDER BY degree DESC, label ASC`), and this is the default
        # backend the contract in BaseGraphStorage points at. Comparing on
        # str() gives the same code-point order as COLLATE "C".
        degrees = dict(graph.degree())
        sorted_nodes = sorted(
            degrees.items(), key=lambda item: (-item[1], str(item[0]))
        )

        # Return top labels limited by the specified limit
        popular_labels = [str(node) for node, _ in sorted_nodes[:limit]]

        logger.debug(
            f"[{self.workspace}] Retrieved {len(popular_labels)} popular labels (limit: {limit})"
        )

        return popular_labels

    async def search_labels(self, query: str, limit: int = 50) -> list[str]:
        """
        Search labels(entity names) with fuzzy matching

        Args:
            query: Search query string
            limit: Maximum number of results to return

        Returns:
            List of matching labels sorted by relevance
        """
        graph = await self._get_graph()
        query_lower = query.lower().strip()

        if not query_lower:
            return []

        # Collect matching nodes with relevance scores
        matches = []
        for node in graph.nodes():
            node_str = str(node)
            node_lower = node_str.lower()

            # Skip if no match
            if query_lower not in node_lower:
                continue

            # Calculate relevance score
            # Exact match gets highest score
            if node_lower == query_lower:
                score = 1000
            # Prefix match gets high score
            elif node_lower.startswith(query_lower):
                score = 500
            # Contains match gets base score, with bonus for shorter strings
            else:
                # Shorter strings with matches are more relevant
                score = 100 - len(node_str)
                # Bonus for word boundary matches
                if f" {query_lower}" in node_lower or f"_{query_lower}" in node_lower:
                    score += 50

            matches.append((node_str, score))

        # Sort by relevance score (desc) then alphabetically
        matches.sort(key=lambda x: (-x[1], x[0]))

        # Return top matches limited by the specified limit
        search_results = [match[0] for match in matches[:limit]]

        logger.debug(
            f"[{self.workspace}] Search query '{query}' returned {len(search_results)} results (limit: {limit})"
        )

        return search_results

    async def get_knowledge_graph(
        self,
        node_label: str,
        max_depth: int = 3,
        max_nodes: int = None,
    ) -> KnowledgeGraph:
        """
        Retrieve a connected subgraph of nodes where the label includes the specified `node_label`.

        Args:
            node_label: Label of the starting node，* means all nodes
            max_depth: Maximum depth of the subgraph, Defaults to 3
            max_nodes: Maxiumu nodes to return by BFS, Defaults to 1000

        Returns:
            KnowledgeGraph object containing nodes and edges, with an is_truncated flag
            indicating whether the graph was truncated due to max_nodes limit
        """
        # Get max_nodes from global_config if not provided
        if max_nodes is None:
            max_nodes = self.global_config.get("max_graph_nodes", 1000)
        else:
            # Limit max_nodes to not exceed global_config max_graph_nodes
            max_nodes = min(max_nodes, self.global_config.get("max_graph_nodes", 1000))

        graph = await self._get_graph()

        result = KnowledgeGraph()

        # Handle special case for "*" label
        if node_label == "*":
            # Get degrees of all nodes
            degrees = dict(graph.degree())
            # Degree descending, then label ascending — same contract as
            # get_popular_labels / BaseGraphStorage. Stable degree-only sort
            # kept insertion order on ties, so max_nodes truncation dropped
            # different isolates depending on insert order.
            #
            # Two stable passes rather than one `(-degree, label)` tuple key:
            # this ranks EVERY node in the graph, and building a tuple per node
            # costs about 3x the sort (measured 56ms -> 189ms at 500k nodes,
            # against 74ms for the two passes). The label pass runs FIRST and
            # the degree pass second — `list.sort` is stable, so equal degrees
            # keep the label order established by the first pass. Swapping them
            # silently restores the insertion-order bug.
            sorted_nodes = sorted(degrees.items(), key=lambda item: str(item[0]))
            sorted_nodes.sort(key=itemgetter(1), reverse=True)

            # Check if graph is truncated
            if len(sorted_nodes) > max_nodes:
                result.is_truncated = True
                logger.info(
                    f"[{self.workspace}] Graph truncated: {len(sorted_nodes)} nodes found, limited to {max_nodes}"
                )

            limited_nodes = [node for node, _ in sorted_nodes[:max_nodes]]
            # Create subgraph with the highest degree nodes
            subgraph = graph.subgraph(limited_nodes)
        else:
            # Check if node exists
            if node_label not in graph:
                logger.warning(
                    f"[{self.workspace}] Node {node_label} not found in the graph"
                )
                return KnowledgeGraph()  # Return empty graph

            # Use modified BFS to get nodes, prioritizing high-degree nodes at the same depth
            bfs_nodes = []
            visited = set()
            # Store (node, depth, degree) in the queue
            queue = deque([(node_label, 0, graph.degree(node_label))])

            # Flag to track if there are unexplored neighbors due to depth limit
            has_unexplored_neighbors = False
            has_unprocessed_level_nodes = False

            # Modified breadth-first search with degree-based prioritization
            while queue and len(bfs_nodes) < max_nodes:
                # Get the current depth from the first node in queue
                current_depth = queue[0][1]

                # Collect all nodes at the current depth
                current_level_nodes = []
                while queue and queue[0][1] == current_depth:
                    current_level_nodes.append(queue.popleft())

                # Degree descending, then label ascending — matches '*' mode
                # and get_popular_labels. Degree-only reverse sort is stable and
                # kept neighbor insertion order on ties at the max_nodes cutoff.
                # Plain tuple key here, unlike '*' mode: this sorts one depth
                # level, not the whole graph, so the tuple allocation does not
                # pay for the two-pass idiom's dependence on sort stability.
                current_level_nodes.sort(key=lambda x: (-x[2], str(x[0])))

                # Process all nodes at current depth in order of degree
                for idx, (current_node, depth, degree) in enumerate(
                    current_level_nodes
                ):
                    if current_node not in visited:
                        visited.add(current_node)
                        bfs_nodes.append(current_node)

                        # Only explore neighbors if we haven't reached max_depth
                        if depth < max_depth:
                            # Add neighbor nodes to queue with incremented depth
                            neighbors = list(graph.neighbors(current_node))
                            # Filter out already visited neighbors
                            unvisited_neighbors = [
                                n for n in neighbors if n not in visited
                            ]
                            # Add neighbors to the queue with their degrees
                            for neighbor in unvisited_neighbors:
                                neighbor_degree = graph.degree(neighbor)
                                queue.append((neighbor, depth + 1, neighbor_degree))
                        else:
                            # Check if there are unexplored neighbors (skipped due to depth limit)
                            neighbors = list(graph.neighbors(current_node))
                            unvisited_neighbors = [
                                n for n in neighbors if n not in visited
                            ]
                            if unvisited_neighbors:
                                has_unexplored_neighbors = True

                    # Check if we've reached max_nodes
                    if len(bfs_nodes) >= max_nodes:
                        if any(
                            n not in visited
                            for n, _, _ in current_level_nodes[idx + 1 :]
                        ):
                            has_unprocessed_level_nodes = True
                        break

            # Check if graph is truncated - either due to max_nodes limit or depth limit
            has_unvisited_in_queue = any(n not in visited for n, _, _ in queue)
            has_max_nodes_truncation = len(bfs_nodes) >= max_nodes and (
                has_unvisited_in_queue
                or has_unprocessed_level_nodes
                or has_unexplored_neighbors
            )
            if has_max_nodes_truncation or has_unexplored_neighbors:
                if has_max_nodes_truncation:
                    result.is_truncated = True
                    logger.info(
                        f"[{self.workspace}] Graph truncated: max_nodes limit {max_nodes} reached"
                    )
                else:
                    logger.info(
                        f"[{self.workspace}] Graph truncated: found {len(bfs_nodes)} nodes within max_depth {max_depth}"
                    )
            # Create subgraph with BFS discovered nodes
            subgraph = graph.subgraph(bfs_nodes)

        # Add nodes to result
        seen_nodes = set()
        seen_edges = set()
        for node in subgraph.nodes():
            if str(node) in seen_nodes:
                continue

            node_data = dict(subgraph.nodes[node])
            # Get entity_type as labels
            labels = []
            if "entity_type" in node_data:
                if isinstance(node_data["entity_type"], list):
                    labels.extend(node_data["entity_type"])
                else:
                    labels.append(node_data["entity_type"])

            # Create node with properties
            node_properties = {k: v for k, v in node_data.items()}

            result.nodes.append(
                KnowledgeGraphNode(
                    id=str(node), labels=[str(node)], properties=node_properties
                )
            )
            seen_nodes.add(str(node))

        # Add edges to result
        for edge in subgraph.edges():
            source, target = edge
            # Esure unique edge_id for undirect graph
            if str(source) > str(target):
                source, target = target, source
            edge_id = f"{source}-{target}"
            if edge_id in seen_edges:
                continue

            edge_data = dict(subgraph.edges[edge])

            # Create edge with complete information
            result.edges.append(
                KnowledgeGraphEdge(
                    id=edge_id,
                    type="DIRECTED",
                    source=str(source),
                    target=str(target),
                    properties=edge_data,
                )
            )
            seen_edges.add(edge_id)

        logger.info(
            f"[{self.workspace}] Subgraph query successful | Node count: {len(result.nodes)} | Edge count: {len(result.edges)}"
        )
        return result

    async def get_all_nodes(self) -> list[dict]:
        """Get all nodes in the graph.

        Returns:
            A list of all nodes, where each node is a dictionary of its properties
        """
        graph = await self._get_graph()
        all_nodes = []
        for node_id, node_data in graph.nodes(data=True):
            node_data_with_id = node_data.copy()
            node_data_with_id["id"] = node_id
            all_nodes.append(node_data_with_id)
        return all_nodes

    async def get_all_edges(self) -> list[dict]:
        """Get all edges in the graph.

        Returns:
            A list of all edges, where each edge is a dictionary of its properties
        """
        graph = await self._get_graph()
        all_edges = []
        for u, v, edge_data in graph.edges(data=True):
            edge_data_with_nodes = edge_data.copy()
            edge_data_with_nodes["source"] = u
            edge_data_with_nodes["target"] = v
            all_edges.append(edge_data_with_nodes)
        return all_edges

    async def iter_edges(self, batch_size: int):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        graph = await self._get_graph()
        batch: list[dict] = []
        for source, target, edge_data in graph.edges(data=True):
            edge = edge_data.copy()
            edge["source"] = source
            edge["target"] = target
            batch.append(edge)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    async def index_done_callback(self) -> bool:
        """Commit in-memory graph to disk and notify other processes.

        This is the writer's **commit point** in the cross-process sync
        protocol (see class docstring). Two effects, in order:
            1. ``write_nx_graph`` atomically writes the GraphML file
               (``atomic_write`` swaps a tmp file into place).
            2. ``set_all_update_flags`` flips every registered process's
               ``storage_updated`` flag, then we immediately reset our
               own flag to ``False`` so the writer does not self-reload
               on the next call to ``_get_graph``.

        Two-block structure (intentional, do not collapse):
            * **First ``async with``** — the decline path: this process is
              about to serialize its whole graph over a file that has moved
              on. Two tests, per *Cross-process sync protocol*.

              The ``storage_updated.value`` half is permanently ``False`` in
              the writer under the single-writer pipeline contract (class
              docstring, invariant 1), so it is defensive scaffolding for a
              future relaxation of that invariant.

              The fingerprint half is **not** — it is live in production.
              Invariant 1 gives one writer *at a time*, not always the same
              process, so a worker whose notification was lost can become the
              next legitimate writer holding a stale snapshot. That is the
              lost write in issue #3854, and this is where it is refused.
            * **Second ``async with``** — the actual save + notify.
        """
        async with self._storage_lock:
            # Same three tests, same order and same meaning as _get_graph.
            # The process-local one first: a reload this process owed itself
            # failed, so self._graph carries mutations that belong to an
            # operation already reported as failed. Saving would publish them
            # under a later document's commit, while their own documents are
            # marked FAILED and reprocessed from scratch -- duplicating the
            # work at best, and overwriting a peer's durable commit at worst.
            # Declining discards them, which is what the failed batch's
            # reprocessing expects.
            if self._recovery_reload_pending:
                # Same precedence, same blind spot, same fix as in _get_graph.
                # One sample, shared by the counting and the reload -- see
                # _count_unannounced_peer_commit_locked for why they must not
                # be two independent observations.
                sampled = self._stat_fingerprint()
                if self._count_unannounced_peer_commit_locked(sampled):
                    logger.warning(
                        f"[{self.workspace}] Declining to save graph "
                        f"{self._graphml_xml_file}: the file on disk is also not "
                        "the one this process loaded and no reload notification "
                        "arrived for it, so a notification was lost (occurrence "
                        f"#{self._missed_notification_reloads} in this process)."
                    )
                logger.warning(
                    f"[{self.workspace}] Declining to save graph "
                    f"{self._graphml_xml_file}: a reload this process owed "
                    "itself failed earlier -- after a failed save, or at "
                    "either fence channel -- so its graph holds mutations "
                    "from an operation already reported as failed. Discarding "
                    "them and reporting the commit as declined."
                )
                self._reload_locked(sampled)
                return False
            # Then both fence channels. A writer holding a snapshot the file
            # has moved past must DECLINE: write_nx_graph serializes the WHOLE
            # graph, so saving would overwrite the peer commit it never saw.
            #
            # No unresolved-fingerprint test here, unlike _get_graph: with a
            # `None` fingerprint and a still-failing stat all three tests
            # below report "nothing to do" and the save proceeds, which is
            # correct. What it serializes is this process's own commit plus
            # its own mutations, so saving loses nothing; and the fence being
            # blind to a peer while the stat fails is the documented
            # "degrades to the flag alone" residue, not something this branch
            # could improve on. _get_graph needs the test because a reload
            # there DISCARDS, and only its timing decides what.
            if self.storage_updated.value:
                # Owed from here on -- see *Recovery reload*.
                self._recovery_reload_pending = True
                # Storage was updated by another process, reload data instead of saving
                logger.info(
                    f"[{self.workspace}] Graph was updated by another process, reloading..."
                )
                self._reload_locked()
                return False  # Return error
            if file_fingerprint.fence_enabled():
                # ONE observation for the decision, the count and the
                # adoption, same rule and same reason as in _get_graph: a
                # contract that says three-from-one should not have an
                # exception resting on the unwritten fact that _storage_lock
                # is cross-process. In single-process mode no stat is taken.
                sampled = self._stat_fingerprint()
                if self._peer_commit_detected(sampled):
                    # Owed from here on; recorded before the classification,
                    # which can raise on its own Manager read. See *Recovery
                    # reload*.
                    self._recovery_reload_pending = True
                    # The lost-notification case this fence exists for. Before
                    # it, this branch was unreachable without a flag and the
                    # save went ahead, silently replacing the peer's commit
                    # with this process's stale snapshot. Declining loses THIS
                    # mutation instead, and loudly: _commit_graph_or_raise
                    # turns the False into an error for the caller.
                    if self._count_unannounced_peer_commit_locked(sampled):
                        logger.warning(
                            f"[{self.workspace}] Declining to save graph "
                            f"{self._graphml_xml_file}: the file on disk is not "
                            "the one this process loaded and no reload "
                            "notification arrived for it, so saving would "
                            "overwrite another process's commit. Reloading and "
                            "reporting the commit as declined (occurrence "
                            f"#{self._missed_notification_reloads} in this "
                            "process)."
                        )
                    self._reload_locked(sampled)
                    return False

        # Acquire lock and perform persistence
        async with self._storage_lock:
            # Close the commit gate for the duration of the serialization: the
            # worker thread iterates graph._node / graph._adj, and a concurrent
            # add_node/remove_node would tear the snapshot (or raise
            # "dictionary changed size during iteration" from inside the writer).
            # The lock alone is not enough -- see _get_graph.
            gate = self._gate()
            gate.clear()
            try:

                async def _committed() -> None:
                    # Runs inside the same uncancellable region as the write,
                    # and only if the write landed. Inlined after the offload it
                    # would be skippable by a cancel, leaving the new GraphML
                    # published while every other process keeps reading the
                    # previous one until some later commit happens to notify it.
                    # Adopt the file this process just published, BEFORE the
                    # fallible notification below. A local stat, so it cannot
                    # fail with the manager -- and doing it first means a failed
                    # notification does not additionally leave this process
                    # treating its own commit as a peer's, which would cost a
                    # full GraphML re-parse on the next call for nothing.
                    self._record_fingerprint()
                    await set_all_update_flags(self.namespace, workspace=self.workspace)
                    # Reset own update flag to avoid self-reloading. Inside the
                    # same publication step on purpose: in multiprocess mode this
                    # flag is a `Manager().Value` proxy, so the assignment is
                    # another RPC to the very process whose outage makes the call
                    # above fail. Separating them would let the identical failure
                    # take a different path one line later.
                    self.storage_updated.value = False

                try:
                    # Save data to disk, off the event loop. The name is resolved
                    # at call time on purpose: test_networkx_index_done.py
                    # monkeypatches write_nx_graph, and hoisting the reference
                    # would leave that test green while testing nothing.
                    await commit_in_storage_io(
                        lambda: NetworkXStorage.write_nx_graph(
                            self._graph, self._graphml_xml_file, self.workspace
                        ),
                        _committed,
                    )
                except CommitBookkeepingError as e:
                    # The write already landed (`on_committed` runs only after
                    # `fn` succeeded), so what failed is the publication of that
                    # write, not the write: other workers keep reading the
                    # previous snapshot until the next commit anywhere flips
                    # their flags, and this process may redundantly reload the
                    # file it just wrote. Both are visibility effects, not a lost
                    # write. Letting this reach the handler below would report a
                    # durable mutation as a failed one, and every caller inherits
                    # that lie -- the deletion paths in utils_graph skip the
                    # tracking retirement they still owe (leaving a vanished
                    # object's authoritative rows behind), and _insert_done marks
                    # a document FAILED whose graph writes are on disk.
                    log_without_raising(
                        logger.error,
                        f"[{self.workspace}] Graph saved to "
                        f"{self._graphml_xml_file}, but publishing that write "
                        f"failed: {e.__cause__}. The notification flips "
                        "one flag per process, so an unknown remainder of them "
                        "keeps reading the previous snapshot until the next "
                        "commit notifies them; this process may also reload "
                        "the file it just wrote.",
                    )
                return True  # Return success
            except Exception as e:
                # Only a genuine write failure reaches here. A failure of the
                # publication hook is caught above as CommitBookkeepingError and
                # does not reach this handler: the file is already on disk by the
                # time that hook runs, so the recovery reload below -- and the
                # "the write did not land" reasoning it rests on -- would both be
                # wrong for it.
                #
                # Raise (do NOT swallow + return False): _insert_done's
                # _flush_one only detects failures via exceptions, so a
                # swallowed graph-save error would let the document be marked
                # PROCESSED with the graph changes unpersisted. Surfacing it
                # aligns this backend with the others (faiss/nano raise too).
                logger.error(f"[{self.workspace}] Error saving graph: {e}")
                # Restore the process view from the file before re-raising,
                # symmetrically with the declined-commit branch above. The write
                # did not land, so self._graph now claims a state the file does
                # not have -- and nothing would ever repair it: a failed write
                # never reaches _committed, so storage_updated stays False and
                # _get_graph's reload branch never fires again. A caller must
                # not be told an object is absent while it is still on disk:
                # utils_graph's deletion retry reads that as a durable removal
                # and sweeps the object's authoritative tracking row, leaving a
                # live node on disk with no provenance. It also stops the next
                # successful commit from publishing mutations that belong to
                # the failed batch, whose documents are marked FAILED and
                # reprocessed from scratch.
                #
                # Safe here: _storage_lock is held and the commit gate is still
                # closed (the finally below reopens it), so no other coroutine
                # can be reading or mutating self._graph.
                try:
                    # Classify the peer channel before reloading, exactly as
                    # the two recovery branches do and for the same reason:
                    # _reload_locked adopts the file, after which the question
                    # cannot be asked any more, so a lost notification would
                    # go uncounted -- the undercount that made
                    # _count_unannounced_peer_commit_locked the single
                    # increment site in the first place.
                    #
                    # A divergence here can only be a peer commit, never this
                    # process's own half-written file: atomic_write leaves the
                    # destination untouched when the write raises, and
                    # commit_in_storage_io persists nothing in that case. What
                    # it would be is a peer commit landing in the window
                    # between the fence block above and this one, where the
                    # lock is released. That window is live, not scaffolding:
                    # invariant 1 is not met for the utils_graph admin flows
                    # (see *Non-pipeline write paths*), where two concurrent
                    # /graph/* calls on different workers both commit.
                    #
                    # Owed from the moment the save failed, so recorded here
                    # rather than in the handler below -- the classification
                    # reads storage_updated.value over the Manager and can
                    # raise before any reload is attempted, and a manager
                    # outage is a plausible reason the save failed in the first
                    # place. A completed reload clears it; nothing else does.
                    self._recovery_reload_pending = True
                    # One sample, shared by the counting and the reload -- see
                    # _count_unannounced_peer_commit_locked for why they must
                    # not be two independent observations. Nothing is adopted
                    # before the count, so the event stays countable.
                    sampled = self._stat_fingerprint()
                    if self._count_unannounced_peer_commit_locked(sampled):
                        logger.warning(
                            f"[{self.workspace}] The save failed and "
                            f"{self._graphml_xml_file} is not the one this "
                            "process loaded either, and no reload notification "
                            "arrived for it, so a notification was lost "
                            f"(occurrence #{self._missed_notification_reloads} "
                            "in this process)."
                        )
                    self._reload_locked(sampled)
                except Exception as reload_error:
                    # Report, never mask: the save error is what the caller
                    # must see, and a failed reload leaves the divergence in
                    # place, so it has to be visible in the log on its own.
                    #
                    # The recovery reload is already armed -- above, before
                    # the attempt, not here after it. What it buys is that
                    # every later public graph operation enters through
                    # _get_graph, which discards this view before trusting it,
                    # and index_done_callback declines rather than publishing
                    # it. Without that, a deletion retry could mistake the
                    # unpersisted mutation for durable state and sweep the live
                    # object's tracking row.
                    #
                    # A plain attribute write, deliberately: what happened here
                    # is process-local ("my memory does not match the file"),
                    # not a peer commit, so neither cross-process channel says
                    # it. Arming storage_updated instead -- as this did before
                    # -- was an RPC to the very manager whose outage may be why
                    # the reload just failed, and it needed a whole failure
                    # path of its own. See *Recovery reload* in the class
                    # docstring for the rest of the reasoning, including why
                    # the fingerprint is NOT invalidated here: the save failed,
                    # so the file is untouched and the recorded fingerprint
                    # still describes it correctly.
                    logger.error(
                        f"[{self.workspace}] Failed to restore the in-memory "
                        f"graph after a failed save; it may not match "
                        f"{self._graphml_xml_file}: {reload_error}"
                    )
                raise
            finally:
                # Every path, including CancelledError. Leaking a cleared gate
                # once deadlocks every later graph operation in this workspace.
                gate.set()

        return True

    async def drop(self) -> dict[str, str]:
        """Drop all graph data from storage and reinitialize the graph.

        This method will:
        1. Remove the graph storage file if it exists
        2. Reset the graph to an empty ``nx.Graph()``
        3. Update flags to notify other processes
        4. Changes are persisted to disk immediately

        Caller contract:
            ``drop`` is destructive and **not** serialized by this storage
            class. The caller must hold the pipeline ``busy`` reservation
            (the ``/documents/clear`` endpoint does this) before invoking
            it — running ``drop`` concurrently with an active document
            pipeline will tear down storage out from under the writer and
            silently lose data. See class docstring,
            *Non-pipeline write paths*.

        Returns:
            dict[str, str]: Operation status and message
            - On success: {"status": "success", "message": "data dropped"}
            - On destructive failure: {"status": "error", "message": "<error details>"}

            A peer notification failure after the file deletion is logged but
            does not change the successful status of the completed drop. No
            step after the deletion — notification, writer-flag reset, or the
            success log — can turn it into an error response.
            This status confirms durable deletion, not convergence of all worker
            snapshots. A worker that missed the notification may later write its
            stale graph back. Stop workspace writes and restart affected workers
            before resuming; if stale data has already been written, clear again.

        Cancellation:
            Before submission, cancellation leaves storage unchanged. Once file
            deletion is submitted, the storage lock stays held until deletion
            and its notification/reset/logging hook finish, then caller
            cancellation propagates. Notification errors are still logged.
        """

        def _delete_file() -> None:
            if os.path.exists(self._graphml_xml_file):
                os.remove(self._graphml_xml_file)

        async def _committed() -> None:
            self._graph = nx.Graph()
            # The file is gone and this process's graph is empty to match, so
            # adopt that state instead of letting the next _get_graph read the
            # removal as a peer commit and reload for it.
            self._record_fingerprint()
            # A recovery reload armed by an earlier failed save is satisfied by
            # the drop: the mutation it was protecting has just been destroyed
            # along with everything else, and memory matches the file again.
            # Clearing it is not cosmetic -- the flag is sticky and is tested
            # by index_done_callback, so leaving it set would make the first
            # commit after a clear DECLINE and discard fresh work.
            self._recovery_reload_pending = False
            # Keep publication under the storage lock. Once deletion starts,
            # commit_in_storage_io defers caller cancellation through this hook
            # so it cannot release readers before the notification attempt.
            try:
                await set_all_update_flags(self.namespace, workspace=self.workspace)
            except Exception as notification_error:
                # Notification can fail partway through the registered flags.
                # A missed worker may later become the writer and persist its
                # stale graph, resurrecting deleted data. A notification from
                # that writer would spread the stale state, not repair it.
                log_without_raising(
                    logger.error,
                    f"[{self.workspace}] Dropped graph file:{self._graphml_xml_file}, "
                    "but failed while notifying all processes; some processes may "
                    "not reload and may restore deleted data if they later write. "
                    "Stop workspace writes and restart all affected workers before "
                    f"resuming: {notification_error}",
                )
            # The local graph is already empty, even after partial notification.
            # A broken shared-state manager can fail this reset independently;
            # report it without misclassifying the durable deletion as failed.
            try:
                self.storage_updated.value = False
            except Exception as reset_error:
                log_without_raising(
                    logger.error,
                    f"[{self.workspace}] Dropped graph file:{self._graphml_xml_file}, "
                    f"but failed to reset the writer reload flag: {reset_error}",
                )
            # Log inside the cancellation-protected hook: the caller may receive
            # CancelledError after it completes instead of a success response.
            # Routed through log_without_raising like every other log call in
            # this hook: a broken log sink cannot unmake the removal, so it
            # must not surface as a failed drop. See that helper for why the
            # failure is swallowed rather than re-reported.
            log_without_raising(
                logger.info,
                f"[{self.workspace}] Process {os.getpid()} drop graph file:{self._graphml_xml_file}",
            )

        try:
            async with self._storage_lock:
                await commit_in_storage_io(_delete_file, _committed)
        except CommitBookkeepingError as e:
            # The file is already gone; only the post-removal bookkeeping failed.
            # Every step of `_committed` guards itself, so nothing raises this
            # today -- it is the standing answer for a future step that forgets
            # to, because "error" for a completed destruction is precisely the
            # misreport those guards exist to prevent.
            log_without_raising(
                logger.error,
                f"[{self.workspace}] Dropped graph file:{self._graphml_xml_file}, "
                f"but its post-removal bookkeeping failed: {e.__cause__}",
            )
        except Exception as e:
            log_without_raising(
                logger.error,
                f"[{self.workspace}] Error dropping graph file:{self._graphml_xml_file}: {e}",
            )
            return {"status": "error", "message": str(e)}

        return {"status": "success", "message": "data dropped"}
