#!/usr/bin/env python3
"""
Offline tool that clears every data storage of ONE LightRAG workspace.

The server refuses to start a workspace whose vector index is empty or
unreadable while its sources still hold data. ``lightrag-rebuild-vdb`` is the
recovery when the data matters; when it does not -- a test corpus, a workspace
being decommissioned -- re-embedding everything just to be allowed to delete
it is wasted time and money. This tool is the other way out: it drops the
same storages the WebUI *Clear* button drops (``/documents/clear``), through
the same ordering, without a running server.

What it does, in order:

1. shows what is about to be deleted: document counts per status, the ten
   most recently updated documents, whether the text chunk store, the
   knowledge graph and each vector index hold anything (all through reads
   that fail loudly), the recorded embedding baselines and the input files.
   A value that cannot be read is shown as UNREADABLE, never as zero;
2. asks the operator to type ``Delete All``;
3. drops the eleven data storages, then the workspace's configuration
   records (only when EVERY drop succeeded), then the top-level input files.

Two kinds of failure are told apart, and the rule is the backend's KIND:

* A **server backend** (PostgreSQL, Redis, Mongo, Milvus, Qdrant, Neo4j,
  OpenSearch, ...) that cannot be opened or read REFUSES the run before
  anything is dropped. Clearing the storages that did answer would leave the
  unreachable one populated -- a partial clear nobody asked for.
* A **file-backed storage** (JSON, NetworkX, Nano, Faiss) whose file is
  corrupt is DATA the operator is about to delete. Its failure is shown, its
  summary line reads UNREADABLE, and its ``drop()`` is still attempted.

What it never touches: the LLM response cache (clear it later through the
WebUI, once the server is up again), the ``__parsed__`` directory, and any
other workspace.

Usage:
    lightrag-clear-storage
    python -m lightrag.tools.clear_storage

Reads the same ``.env`` / environment the server reads (``LIGHTRAG_*_STORAGE``,
``WORKSPACE``, ``WORKING_DIR``, ``INPUT_DIR``, ``EMBEDDING_MODEL`` /
``EMBEDDING_DIM``, backend connection settings). See
``lightrag/tools/README_CLEAR_STORAGE.md``.
"""

import asyncio
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Tuple

from dotenv import load_dotenv

# Add project root to path for imports
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from lightrag.base import DocStatus
from lightrag.config_store import (
    EMBEDDING_TARGETS,
    EmbeddingBaseline,
    create_configuration_storage,
    delete_workspace_configuration,
    describe_configuration_container,
    embedding_baseline_key,
    read_config_row_strict,
    resolve_config_dir,
    resolve_configuration_storage,
)
from lightrag.constants import (
    DEFAULT_COSINE_THRESHOLD,
    DEFAULT_EMBEDDING_BATCH_NUM,
)
from lightrag.exceptions import (
    ConfigurationStorageError,
    CorruptStorageSnapshotError,
    StorageCapabilityError,
    VectorSpaceMismatchError,
    WorkingDirectoryInUseError,
)
from lightrag.kg import STORAGE_ENV_REQUIREMENTS
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.kg.working_dir_lock import (
    acquire_working_dir_lock,
    release_working_dir_lock,
    uses_working_dir,
)
from lightrag.namespace import NameSpace
from lightrag.vector_space_gate import (
    chunk_source_is_populated,
    graph_has_edges,
    graph_has_nodes,
)
from lightrag.utils import (
    EmbeddingFunc,
    get_env_value,
    logger,
    setup_logger,
    validate_workspace,
    warn_about_workspace_overrides,
)

# NOTE: .env loading and logger setup are deferred to main() so that importing
# this module as a library has no side effects on the caller's environment.

CONFIRMATION_PHRASE = "Delete All"
RECENT_DOCS_SHOWN = 10

# Labels in the order ``/documents/clear`` drops them. The LLM response cache
# is deliberately absent: see the module docstring.
DATA_STORAGE_LABELS: Tuple[str, ...] = (
    "text_chunks",
    "full_docs",
    "full_entities",
    "full_relations",
    "entity_chunks",
    "relation_chunks",
    "entities_vdb",
    "relationships_vdb",
    "chunks_vdb",
    "chunk_entity_relation_graph",
    "doc_status",
)

VECTOR_LABELS: Tuple[str, ...] = ("entities_vdb", "relationships_vdb", "chunks_vdb")

# Which configured backend (``resolve_storage_names`` key) each label uses.
STORAGE_KIND_OF_LABEL: Dict[str, str] = {
    **{label: "kv" for label in DATA_STORAGE_LABELS[:6]},
    **{label: "vector" for label in VECTOR_LABELS},
    "chunk_entity_relation_graph": "graph",
    "doc_status": "doc_status",
}

# ANSI color codes for terminal output
BOLD_CYAN = "\033[1;36m"
BOLD_RED = "\033[1;31m"
BOLD_GREEN = "\033[1;32m"
RESET = "\033[0m"


class RemoteBackendUnavailableError(RuntimeError):
    """A server backend could not be opened or read, so the run is refused.

    Raised before anything is dropped. The alternative -- clearing the
    storages that did answer -- leaves the unreachable one populated, and a
    workspace half cleared is worse than one not cleared at all.
    """


def is_server_backed(storage_name: str) -> bool:
    """Whether a backend is reached over a connection rather than a file.

    Decided from ``STORAGE_ENV_REQUIREMENTS``: a backend that needs a
    connection setting is a server, one that needs none is file-backed. A
    name the table does not know is treated as a server, the conservative
    reading -- refusing a run is recoverable, a partial clear is not.
    """
    requirements = STORAGE_ENV_REQUIREMENTS.get(storage_name)
    if requirements is None:
        return True
    return bool(requirements)


def classify_drop_result(result: Any) -> str | None:
    """Return ``None`` when a ``drop()`` result reports success, else why not.

    Mirrors the classification in ``/documents/clear``
    (``lightrag/api/routers/document_routes.py``): a ``BaseException`` object
    -- not merely ``Exception`` -- is a failure, because ``asyncio.gather(...,
    return_exceptions=True)`` hands a cancelled drop back as a
    ``CancelledError`` OBJECT and a cancelled drop is a drop that did not
    happen; and a dict whose ``status`` is not ``"success"`` is a backend
    reporting a non-raising failure. Either read as success would let the
    configuration records be deleted over surviving data, the one residue
    *Workspace drop* in ``docs/design/ConfigurationStorage.md`` never accepts.
    """
    if isinstance(result, BaseException):
        # ``repr``: a CancelledError stringifies to the empty string.
        return f"{result!r}"
    if isinstance(result, dict) and result.get("status") != "success":
        return str(result.get("message", "unknown error"))
    return None


@dataclass(frozen=True)
class Unreadable:
    """A summary value that could not be read. Rendered as such, never as 0."""

    reason: str


@dataclass
class DropOutcome:
    """What the drop step did, per storage label."""

    succeeded: List[str] = field(default_factory=list)
    failed: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def all_succeeded(self) -> bool:
        return not self.failed


class ClearTool:
    """Interactive CLI for the offline workspace clear."""

    def __init__(self):
        self.storages: Dict[str, Any] = {}
        self.configuration_storage = None
        self._holds_working_dir = False
        self.config_dir = ""
        self.input_dir = ""
        self.working_dir = ""
        self.workspace = ""
        self.global_config: Dict[str, Any] = {}
        self.storage_names: Dict[str, str] = {}
        # Vector targets whose ``initialize()`` raised one of the two typed
        # refusals, label -> diagnostic. Dropped like every other storage;
        # nothing is backed up, because the operator is about to type the
        # phrase that deletes it.
        self.refused_vdbs: Dict[str, str] = {}
        # File-backed storages whose construction or ``initialize()`` failed
        # for any other reason, label -> diagnostic. Their data is unreadable,
        # which is not a reason to keep it; ``drop()`` is still attempted. A
        # storage that could not even be constructed is ``None`` in
        # ``self.storages`` and is reported as a failed drop.
        self.unavailable: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Configuration / setup
    # ------------------------------------------------------------------

    def server_args(self):
        """The server's own parsed configuration, or ``None`` without the api.

        Workspace, directories and backend names are taken from here and
        never from the raw environment, because the server does not use the
        raw environment either: its parser sanitizes ``WORKSPACE`` (every
        character but ``[A-Za-z0-9_]`` becomes ``_``, so ``customer-prod``
        stores under ``customer_prod``) and makes the directories absolute.
        A tool reading ``WORKSPACE`` raw would summarize and drop a
        workspace the server never wrote to, and could destroy one created
        through the library under the raw name. The tool refuses every
        command-line argument (see ``main``), so what the parser sees is
        exactly the environment the server would see.
        """
        try:
            from lightrag.api.config import global_args
        except ImportError as e:
            print(f"\n✗ Could not import the LightRAG API package: {e}")
            print(
                '  This tool needs the api extra: pip install "lightrag-hku[api]".\n'
                "  Without it neither the workspace nor the embedding model and "
                "dimension the server uses can be resolved."
            )
            return None
        return global_args

    def resolve_storage_names(self, args) -> Dict[str, str]:
        return {
            "graph": args.graph_storage,
            "vector": args.vector_storage,
            "kv": args.kv_storage,
            "doc_status": args.doc_status_storage,
            # Resolved exactly as the server resolves it: the records this
            # tool deletes last live in the container the server reads.
            "config": resolve_configuration_storage(
                args.config_storage, kv_storage=args.kv_storage
            ),
        }

    def resolve_input_dir(self, base_input_dir: str) -> str:
        """The directory the server uploads THIS workspace's files to.

        Mirrors ``DocumentManager.__init__``: a named workspace keeps its
        uploads under ``INPUT_DIR/<workspace>``, the default workspace under
        ``INPUT_DIR`` itself. Resolving the base directory for a named
        workspace would delete the default workspace's files and leave this
        one's in place to be re-enqueued.
        """
        base = os.path.abspath(base_input_dir)
        if self.workspace:
            validate_workspace(self.workspace)
            return os.path.join(base, self.workspace)
        return base

    def storage_name_of(self, label: str) -> str:
        """The configured backend name behind a data storage label."""
        return self.storage_names[STORAGE_KIND_OF_LABEL[label]]

    def check_env_vars(self, storage_name: str) -> None:
        """Warn about missing env vars (initialization is the real validation)."""
        required_vars = STORAGE_ENV_REQUIREMENTS.get(storage_name, [])
        missing_vars = [var for var in required_vars if var not in os.environ]
        if missing_vars:
            print(
                f"⚠️  Warning: {storage_name} normally requires: "
                f"{', '.join(missing_vars)} (may be provided via config.ini)"
            )

    def build_embedding_func(self) -> EmbeddingFunc | None:
        """Build the embedding function through the server's own factory.

        The tool never embeds, but the function's ``model_name`` and
        ``embedding_dim`` decide WHICH container Qdrant, PostgreSQL and
        Milvus open -- they put both in the container name -- and the server
        derives an omitted ``EMBEDDING_DIM`` from the provider's default. A
        stub that guessed a dimension would open, and drop, a container the
        server never wrote to, then delete the configuration records while
        the real vectors survived. So the factory is the only source, as in
        ``lightrag-rebuild-vdb``, and unlike that tool there is no check-only
        fallback: returns ``None`` when the api extra is unavailable, and the
        caller refuses the run.
        """
        try:
            from lightrag.api.config import global_args
            from lightrag.api.lightrag_server import (
                create_embedding_function_from_args,
            )
        except ImportError as e:
            print(f"\n✗ Could not import the LightRAG API package: {e}")
            print(
                '  This tool needs the api extra: pip install "lightrag-hku[api]".\n'
                "  Without it the embedding model and dimension the server uses "
                "cannot be resolved, and a guessed dimension would clear the "
                "wrong vector container."
            )
            return None
        embedding_func = create_embedding_function_from_args(global_args)
        print(
            f"- Embedding: binding={global_args.embedding_binding} "
            f"model={embedding_func.model_name} dim={embedding_func.embedding_dim}"
        )
        return embedding_func

    def build_global_config(self, embedding_func: EmbeddingFunc) -> Dict[str, Any]:
        return {
            "working_dir": self.working_dir,
            # Backend selection, mirroring LightRAG._build_global_config: PG
            # storages derive enable_vector from global_config["vector_storage"].
            "kv_storage": self.storage_names["kv"],
            "vector_storage": self.storage_names["vector"],
            "graph_storage": self.storage_names["graph"],
            "doc_status_storage": self.storage_names["doc_status"],
            "config_dir": self.config_dir,
            "embedding_batch_num": get_env_value(
                "EMBEDDING_BATCH_NUM", DEFAULT_EMBEDDING_BATCH_NUM, int
            ),
            "vector_db_storage_cls_kwargs": {
                "cosine_better_than_threshold": get_env_value(
                    "COSINE_THRESHOLD", DEFAULT_COSINE_THRESHOLD, float
                )
            },
            "embedding_func": embedding_func,
        }

    def build_storage(self, label: str, embedding_func: EmbeddingFunc):
        """Instantiate one of the eleven data storages ``/documents/clear`` drops.

        Namespaces and ``meta_fields`` match ``LightRAG.__post_init__`` so
        every backend resolves the same container the server writes to.
        """
        from lightrag.kg.factory import get_storage_class

        cls = get_storage_class(self.storage_name_of(label))
        common = {
            "workspace": self.workspace,
            "global_config": self.global_config,
            "embedding_func": embedding_func,
        }
        vector_meta = {
            "entities_vdb": (
                NameSpace.VECTOR_STORE_ENTITIES,
                {"entity_name", "source_id", "content", "file_path"},
            ),
            "relationships_vdb": (
                NameSpace.VECTOR_STORE_RELATIONSHIPS,
                {"src_id", "tgt_id", "source_id", "content", "file_path"},
            ),
            "chunks_vdb": (
                NameSpace.VECTOR_STORE_CHUNKS,
                {"full_doc_id", "content", "file_path"},
            ),
        }
        kv_namespaces = {
            "text_chunks": NameSpace.KV_STORE_TEXT_CHUNKS,
            "full_docs": NameSpace.KV_STORE_FULL_DOCS,
            "full_entities": NameSpace.KV_STORE_FULL_ENTITIES,
            "full_relations": NameSpace.KV_STORE_FULL_RELATIONS,
            "entity_chunks": NameSpace.KV_STORE_ENTITY_CHUNKS,
            "relation_chunks": NameSpace.KV_STORE_RELATION_CHUNKS,
        }
        if label in vector_meta:
            namespace, meta_fields = vector_meta[label]
            return cls(namespace=namespace, meta_fields=meta_fields, **common)
        if label in kv_namespaces:
            return cls(namespace=kv_namespaces[label], **common)
        if label == "chunk_entity_relation_graph":
            return cls(namespace=NameSpace.GRAPH_STORE_CHUNK_ENTITY_RELATION, **common)
        if label == "doc_status":
            return cls(
                namespace=NameSpace.DOC_STATUS,
                workspace=self.workspace,
                global_config=self.global_config,
                embedding_func=None,
            )
        raise ValueError(f"unknown storage label {label!r}")

    async def _open_data_storage(self, label: str, embedding_func: EmbeddingFunc):
        """Construct and initialize one data storage, applying the kind rule.

        Returns the instance, or ``None`` when a file-backed storage could
        not even be constructed. Raises ``RemoteBackendUnavailableError``
        for a server backend that failed for any reason but the two typed
        vector refusals, which are data-level on every backend and leave
        ``drop()`` servable.
        """
        storage_name = self.storage_name_of(label)
        storage = None
        try:
            storage = self.build_storage(label, embedding_func)
            # The server-identical path, one-time migrations included: on a
            # named-container backend whose legacy (unsuffixed) container
            # exists while the current one does not, this copies the legacy
            # vectors into the current container -- before the summary, so
            # before the phrase. Accepted, as ``lightrag-rebuild-vdb`` accepts
            # it: the migration moves data the clear is about to drop into
            # the container the clear drops, loses nothing, and leaves the
            # state a server start would have produced; a cancelled run keeps
            # that state. No backend offers a non-mutating attach, and
            # dropping without attaching is not servable on every backend.
            await storage.initialize()
        except (VectorSpaceMismatchError, CorruptStorageSnapshotError) as e:
            self.refused_vdbs[label] = str(e)
            print(f"⚠️  {label} refused to attach: {e}")
        except Exception as e:
            if is_server_backed(storage_name):
                raise RemoteBackendUnavailableError(
                    f"{label} ({storage_name}) could not be opened: {e}"
                ) from e
            self.unavailable[label] = f"{type(e).__name__}: {e}"
            print(
                f"⚠️  {label} ({storage_name}) could not be opened, its data is "
                f"unreadable and will be dropped anyway: {e}"
            )
        return storage

    async def setup_storages(self) -> bool:
        """Instantiate and initialize every storage. Returns False on failure.

        The configuration storage takes the server-identical path and any
        failure aborts: a clear that could not delete its configuration
        records last is not a clean clear. The data storages follow the
        kind rule in the module docstring, applied by ``_open_data_storage``.
        """
        args = self.server_args()
        if args is None:
            return False
        self.storage_names = self.resolve_storage_names(args)
        self.workspace = args.workspace or ""
        self.working_dir = args.working_dir
        self.config_dir = resolve_config_dir(args.config_dir, args.working_dir)
        self.input_dir = self.resolve_input_dir(args.input_dir)

        # Claim the configuration directory FIRST, for the reason
        # ``lightrag-rebuild-vdb`` does: a server on the same directory keeps
        # its own copy of a file-backed configuration namespace and would
        # republish the records this tool deletes.
        self._holds_working_dir = uses_working_dir(self.storage_names["config"])
        if self._holds_working_dir:
            try:
                acquire_working_dir_lock(self.config_dir)
            except WorkingDirectoryInUseError as e:
                self._holds_working_dir = False
                print(f"\n✗ {e}")
                return False

        print("\nChecking configuration...")
        for storage_name in set(self.storage_names.values()):
            self.check_env_vars(storage_name)

        embedding_func = self.build_embedding_func()
        if embedding_func is None:
            return False
        self.global_config = self.build_global_config(embedding_func)

        print("\nInitializing storages...")
        try:
            from lightrag.kg.factory import get_storage_class

            self.configuration_storage = create_configuration_storage(
                get_storage_class(self.storage_names["config"]),
                global_config=self.global_config,
                embedding_func=embedding_func,
            )
            await self.configuration_storage.initialize()
        except Exception as e:
            print(
                f"✗ The configuration storage could not be opened, so the "
                f"workspace records could not be deleted after a clear: {e}"
            )
            self._print_env_requirements()
            return False

        try:
            for label in DATA_STORAGE_LABELS:
                self.storages[label] = await self._open_data_storage(
                    label, embedding_func
                )
        except RemoteBackendUnavailableError as e:
            print(
                f"\n✗ {e}\n  A server backend that cannot be reached refuses the "
                f"run: clearing the others would leave this one populated."
            )
            self._print_env_requirements()
            return False

        print(f"- Graph Storage:      {self.storage_names['graph']}")
        print(f"- Vector Storage:     {self.storage_names['vector']}")
        print(f"- KV Storage:         {self.storage_names['kv']}")
        print(f"- Doc Status Storage: {self.storage_names['doc_status']}")
        print(
            f"- Configuration:      "
            f"{describe_configuration_container(self.storage_names['config'], self.config_dir)}"
        )
        print(
            f"- Workspace:          {self.workspace if self.workspace else '(default)'}"
        )
        print(f"- Working Dir:        {self.global_config['working_dir']}")
        print(f"- Input Dir:          {self.input_dir}")
        print("- Connection Status:  ✓ Success")
        return True

    def _print_env_requirements(self) -> None:
        for storage_name in set(self.storage_names.values()):
            required = STORAGE_ENV_REQUIREMENTS.get(storage_name, [])
            if required:
                print(f"  {storage_name} requires: {', '.join(required)}")

    # ------------------------------------------------------------------
    # Pre-delete summary
    # ------------------------------------------------------------------

    def input_files(self) -> List[str]:
        """Top-level files of the input directory, the ones a clear deletes.

        Subdirectories (``__parsed__`` among them) are never listed: the
        endpoint preserves them, and so does this tool. A missing input
        directory is simply empty.
        """
        if not os.path.isdir(self.input_dir):
            return []
        return sorted(
            entry.path for entry in os.scandir(self.input_dir) if entry.is_file()
        )

    async def _read(self, label: str, read: Callable[[], Awaitable[Any]]) -> Any:
        """Run one summary read against the storage behind ``label``.

        A storage that never opened answers ``Unreadable`` without being
        asked -- a JSON storage whose load failed still holds an EMPTY
        shared dict, and asking it would render a corrupt file as zero rows.
        A read that raises is ``Unreadable`` when the backend is file-backed
        or the storage merely lacks the capability; a server backend that
        cannot answer refuses the run (``RemoteBackendUnavailableError``),
        since it will not serve the drop either.
        """
        if label in self.unavailable:
            return Unreadable(f"storage did not open ({self.unavailable[label]})")
        try:
            return await read()
        except (StorageCapabilityError, ValueError) as e:
            return Unreadable(f"{type(e).__name__}: {e}")
        except Exception as e:
            if is_server_backed(self.storage_name_of(label)):
                raise RemoteBackendUnavailableError(
                    f"{label} ({self.storage_name_of(label)}) could not be read: {e}"
                ) from e
            return Unreadable(f"{type(e).__name__}: {e}")

    async def _read_baselines(self) -> Dict[str, Any]:
        """The recorded baselines, a parse failure per target as ``Unreadable``.

        Transport is separated from parsing on purpose: a row that cannot be
        FETCHED means the configuration storage is not serving, and the
        records could not be deleted after the clear either, so that refuses
        the run. A row that was fetched but does not PARSE is exactly what
        the clear removes -- ``delete_workspace_configuration`` deletes by
        key without reading the value -- so it is shown and not obeyed.
        """
        out: Dict[str, Any] = {}
        for target in EMBEDDING_TARGETS:
            key = embedding_baseline_key(self.workspace, target)
            row = await read_config_row_strict(self.configuration_storage, key)
            if row is None:
                out[target] = None
                continue
            try:
                out[target] = EmbeddingBaseline.from_row(row, key=key)
            except ConfigurationStorageError as e:
                out[target] = Unreadable(str(e))
        return out

    async def collect_summary(self) -> Dict[str, Any]:
        """Read everything the operator sees before confirming.

        Every value is either read or reported ``Unreadable``, never guessed:
        counts come from ``count_docs_by_statuses(strict=True)``, not the
        best-effort ``get_status_counts()``, and a storage that did not open
        is not asked. Raises ``RemoteBackendUnavailableError`` when a server
        backend cannot answer -- see ``_read``.
        """
        doc_status = self.storages["doc_status"]

        async def count(status: DocStatus) -> int:
            return await doc_status.count_docs_by_statuses([status], strict=True)

        counts: Dict[str, Any] = {}
        for status in DocStatus:
            counts[status.value] = await self._read(
                "doc_status", lambda status=status: count(status)
            )

        # The total is the sum of the STRICT counts above, never the page's
        # own total: ``get_docs_paginated`` is a listing read, and on Redis and
        # OpenSearch it catches its backend errors and returns ``([], 0)``.
        strict_total: Any = (
            Unreadable("one or more status counts could not be read")
            if any(isinstance(c, Unreadable) for c in counts.values())
            else sum(counts.values())
        )

        async def page():
            rows, _page_total = await doc_status.get_docs_paginated(
                page=1,
                page_size=RECENT_DOCS_SHOWN,
                sort_field="updated_at",
                sort_direction="desc",
            )
            # An empty page under a strict count that says documents exist is
            # the swallowed failure showing through: the backend was reachable
            # for the counts a moment ago and is not now. Raising here hands
            # it to ``_read``, which refuses the run on a server backend --
            # its drop would fail after the healthy siblings were dropped.
            if not rows and isinstance(strict_total, int) and strict_total > 0:
                raise RuntimeError(
                    f"get_docs_paginated returned no rows while the strict "
                    f"counts report {strict_total} document(s); the listing "
                    f"read failed silently"
                )
            return rows

        recent = await self._read("doc_status", page)
        total = strict_total

        async def text_chunks_state() -> Any:
            # The same strict existence read the startup gate uses to CLAIM a
            # chunk baseline, not ``BaseKVStorage.is_empty()``: the KV
            # backends' ``is_empty()`` catch their transport errors and answer
            # "empty", which here would show a Redis or Mongo outage as an
            # empty store and let the confirmation drop every healthy sibling
            # around it -- the partial clear the kind rule refuses. The
            # strict read raises instead, and ``_read`` applies the rule.
            populated = await chunk_source_is_populated(
                self.storages["text_chunks"], strict=True
            )
            if populated is None:
                return Unreadable(
                    "the backend cannot enumerate rows and its is_empty() "
                    "cannot tell an outage from an empty store"
                )
            return "has data" if populated else "EMPTY"

        async def graph_state() -> Any:
            # The knowledge graph is the most expensive thing this run drops
            # and the one a rebuild treats as authoritative, so the operator
            # sees it before typing the phrase. Both probes are the startup
            # gate's own, and both RAISE on a backend failure rather than
            # answering "empty" -- so ``_read`` applies the kind rule to a
            # Neo4j outage exactly as it does to a Redis one.
            graph = self.storages["chunk_entity_relation_graph"]
            # ``get_popular_labels`` is abstract on ``BaseGraphStorage``, so
            # every backend answers it; a failure reaches ``_read``.
            if not await graph_has_nodes(graph):
                return "EMPTY"
            try:
                has_edges = await graph_has_edges(graph)
            except StorageCapabilityError:
                # ``iter_edges`` is fail-closed on a backend that never
                # implemented it. The entity count already proves the graph
                # holds data, which is what the confirmation turns on, so the
                # relation half is simply not reported.
                return "has entities"
            return "has entities and relations" if has_edges else "has entities"

        vectors: Dict[str, Any] = {}
        for label in VECTOR_LABELS:
            vdb = self.storages[label]
            if label in self.refused_vdbs:
                vectors[label] = "refused to attach (will be dropped)"
                continue
            if vdb is not None and not getattr(vdb, "persists_vectors", True):
                vectors[label] = "no-op backend (nothing stored)"
                continue

            async def is_empty(vdb=vdb) -> str:
                return "EMPTY" if await vdb.is_empty() else "has vectors"

            vectors[label] = await self._read(label, is_empty)

        try:
            baselines = await self._read_baselines()
        except ConfigurationStorageError as e:
            raise RemoteBackendUnavailableError(
                f"the configuration storage could not be read: {e}"
            ) from e

        return {
            "counts": counts,
            "total_docs": total,
            "recent": recent,
            "text_chunks": await self._read("text_chunks", text_chunks_state),
            "graph": await self._read("chunk_entity_relation_graph", graph_state),
            "vectors": vectors,
            "baselines": baselines,
            "input_files": self.input_files(),
            # A backend-specific ``*_WORKSPACE`` variable outranks WORKSPACE
            # inside the storage constructor, and nothing above it sees that.
            # Some backends write the effective name back onto ``workspace``
            # (PostgreSQL, MongoDB, Milvus); Qdrant is the only one that
            # publishes it separately, as ``effective_workspace``. Redis folds
            # it straight into ``final_namespace`` and exposes no name at all,
            # so a REDIS_WORKSPACE override is invisible HERE -- which is why
            # the override list read from the environment is the authoritative
            # signal and these resolved names are only a bonus.
            "workspace_overrides": warn_about_workspace_overrides(),
            "resolved_workspaces": sorted(
                {
                    str(
                        getattr(storage, "effective_workspace", None)
                        or getattr(storage, "workspace", "")
                        or ""
                    )
                    for storage in self.storages.values()
                    if storage is not None
                }
            ),
        }

    def unreadable_items(self, summary: Dict[str, Any]) -> List[str]:
        """Names of everything the operator is about to delete unread.

        The storages that never opened come FIRST and are listed here even
        though they have no summary line of their own: their one-line warning
        is printed at the top of a summary dozens of lines long, and this list
        is what sits directly above the prompt. A storage whose whole file
        could not be parsed is the most important entry in it, not an
        exception to it.
        """
        items: List[str] = [f"{label} (did not open)" for label in self.unavailable]

        def name(label: str, item: str) -> None:
            # A storage that did not open is already named above, and every
            # value derived from it is Unreadable for that one reason. Naming
            # both would pad the list with restatements of its first entry.
            if label not in self.unavailable:
                items.append(item)

        for status_value, count in summary["counts"].items():
            if isinstance(count, Unreadable):
                name("doc_status", f"documents in status {status_value}")
        if isinstance(summary["recent"], Unreadable):
            name("doc_status", "most recently updated documents")
        if isinstance(summary["text_chunks"], Unreadable):
            name("text_chunks", "text_chunks state")
        if isinstance(summary["graph"], Unreadable):
            name("chunk_entity_relation_graph", "knowledge graph state")
        for label, state in summary["vectors"].items():
            if isinstance(state, Unreadable):
                name(label, f"{label} state")
        for target, baseline in summary["baselines"].items():
            if isinstance(baseline, Unreadable):
                items.append(f"{target} embedding baseline")
        return items

    def print_summary(self, summary: Dict[str, Any]) -> None:
        def show(value: Any) -> str:
            if isinstance(value, Unreadable):
                return f"{BOLD_RED}UNREADABLE{RESET} ({value.reason})"
            return str(value)

        print("\n" + "=" * 60)
        print(f"{BOLD_CYAN}What will be deleted{RESET}")
        print("=" * 60)

        overrides = summary["workspace_overrides"]
        if overrides:
            print(
                f"\n⚠️  Workspace override(s) in effect: {', '.join(overrides)}. "
                f"The storages they govern do NOT live under "
                f"WORKSPACE={self.workspace or '(default)'}; check the values "
                f"before confirming."
            )
        resolved = summary["resolved_workspaces"]
        shown = [ws if ws else "(default)" for ws in resolved]
        if len(resolved) > 1 or (resolved and resolved[0] != self.workspace):
            print(
                f"\n⚠️  Storages resolved to workspace(s) {', '.join(shown)} "
                f"(WORKSPACE={self.workspace or '(default)'})"
            )
        for label, reason in self.unavailable.items():
            print(f"\n⚠️  {label} did not open: {reason}")

        print(f"\nDocuments by status ({show(summary['total_docs'])} total):")
        for status_value, count in summary["counts"].items():
            print(f"    {status_value:14s} {show(count)}")

        recent = summary["recent"]
        print(f"\nMost recently updated documents (up to {RECENT_DOCS_SHOWN}):")
        if isinstance(recent, Unreadable):
            print(f"    {show(recent)}")
        elif not recent:
            print("    (none)")
        else:
            for doc_id, doc in recent:
                status = getattr(doc, "status", "")
                status_value = getattr(status, "value", status)
                updated_at = getattr(doc, "updated_at", "") or ""
                file_path = getattr(doc, "file_path", "") or ""
                print(
                    f"    {updated_at:20s} {status_value:11s} {file_path}  [{doc_id}]"
                )

        print(f"\nText chunks: {show(summary['text_chunks'])}")
        print(f"Knowledge graph: {show(summary['graph'])}")

        print("\nVector storages:")
        for label, state in summary["vectors"].items():
            print(f"    {label:18s} {show(state)}")

        print("\nRecorded embedding baselines:")
        for target in EMBEDDING_TARGETS:
            baseline = summary["baselines"].get(target)
            if baseline is None:
                print(f"    {target:14s} (none recorded)")
            elif isinstance(baseline, Unreadable):
                print(f"    {target:14s} {show(baseline)}")
            else:
                print(
                    f"    {target:14s} model={baseline.model!r} dim={baseline.dim} "
                    f"origin={baseline.origin}"
                )

        files = summary["input_files"]
        print(f"\nInput files to delete (top level of {self.input_dir}): {len(files)}")
        for path in files[:RECENT_DOCS_SHOWN]:
            print(f"    {os.path.basename(path)}")
        if len(files) > RECENT_DOCS_SHOWN:
            print(f"    ... and {len(files) - RECENT_DOCS_SHOWN} more")

        print(
            "\nPreserved: the LLM response cache (clear it later from the "
            "WebUI), the __parsed__ directory, every other workspace."
        )

        unreadable = self.unreadable_items(summary)
        if unreadable:
            print(
                f"\n{BOLD_RED}⚠️  {len(unreadable)} item(s) above could not be "
                f"read: {'; '.join(unreadable)}.{RESET}\n  What they hold is "
                f"unknown, and the clear deletes it anyway if you confirm."
            )

    # ------------------------------------------------------------------
    # Confirmation
    # ------------------------------------------------------------------

    def confirm_server_stopped(self) -> bool:
        confirm = (
            input("\nHas the LightRAG Server been shut down? (yes/no): ")
            .strip()
            .lower()
        )
        if confirm != "yes":
            print("\n✓ Operation cancelled - please shut down the server first")
            return False
        return True

    def confirm_delete_all(self) -> bool:
        """Only the exact phrase proceeds; anything else cancels untouched."""
        print(
            f"\n{BOLD_RED}This deletes every storage listed above and cannot be "
            f"undone.{RESET}"
        )
        answer = input(f'Type "{CONFIRMATION_PHRASE}" to confirm: ').strip()
        if answer != CONFIRMATION_PHRASE:
            print("\n✓ Operation cancelled - no workspace data was deleted")
            return False
        return True

    # ------------------------------------------------------------------
    # The clear
    # ------------------------------------------------------------------

    async def drop_all(self) -> DropOutcome:
        """Drop the eleven data storages, exactly as ``/documents/clear`` does.

        Concurrent, ``return_exceptions=True``, classified by
        :func:`classify_drop_result`. Every storage is attempted even when an
        earlier one fails: leaving a storage undropped because a sibling
        failed only leaves more for the re-run. A storage that could not be
        constructed has nothing to call and is a failed drop outright.
        """
        outcome = DropOutcome()
        # ``is not None``, not truthiness: no storage class defines
        # ``__bool__``/``__len__`` today, but an empty one that did would
        # silently drop out of this list and be reported as never
        # constructed -- keeping the configuration records over data the
        # run never touched.
        droppable = [
            label
            for label in DATA_STORAGE_LABELS
            if self.storages.get(label) is not None
        ]
        for label in DATA_STORAGE_LABELS:
            if label not in droppable:
                reason = (
                    f"not constructed ({self.unavailable.get(label, 'unknown')}); "
                    f"remove its files under {self.global_config.get('working_dir')} "
                    f"by hand"
                )
                outcome.failed.append((label, reason))
                logger.error(f"Cannot drop {label}: {reason}")
                print(f"  ✗ {label}: {reason}")
        results = await asyncio.gather(
            *(self.storages[label].drop() for label in droppable),
            return_exceptions=True,
        )
        for label, result in zip(droppable, results):
            failure = classify_drop_result(result)
            storage_name = type(self.storages[label]).__name__
            if failure is None:
                outcome.succeeded.append(label)
                logger.info(f"Dropped {label} ({storage_name})")
                print(f"  ✓ {label}")
            else:
                outcome.failed.append((label, failure))
                logger.error(f"Error dropping {label} ({storage_name}): {failure}")
                print(f"  ✗ {label}: {failure}")
        return outcome

    async def delete_configuration(self, outcome: DropOutcome) -> bool:
        """Delete the workspace's configuration records, data first.

        Only when EVERY drop succeeded: records gone with data still present
        lets the next startup adopt a wrong baseline over it, the residue
        *Workspace drop* in ``docs/design/ConfigurationStorage.md`` never
        accepts. A partial drop keeps all three records and says so. Returns
        False when the records should have gone and did not.
        """
        if not outcome.all_succeeded:
            kept = ", ".join(label for label, _ in outcome.failed)
            print(
                f"\n⚠️  Kept the workspace configuration records: the drop of "
                f"{kept} failed. Re-run this tool once the cause is fixed."
            )
            return True
        try:
            await delete_workspace_configuration(
                self.configuration_storage, self.workspace
            )
        except Exception as e:
            print(f"\n✗ Could not delete the workspace configuration records: {e}")
            return False
        print("\n  ✓ workspace configuration records deleted")
        return True

    def delete_input_files(self) -> Tuple[int, int]:
        """Delete the top-level input files; returns ``(deleted, failed)``.

        Unconditional, as in the endpoint: a later ``/documents/scan`` would
        otherwise re-enqueue them against the emptied storages.
        """
        deleted = failed = 0
        for path in self.input_files():
            try:
                os.remove(path)
                deleted += 1
            except Exception as e:
                logger.error(f"Error deleting input file {path}: {e}")
                failed += 1
        return deleted, failed

    async def clear(self) -> bool:
        """Run the clear after confirmation. Returns True only when everything landed."""
        print("\nDropping storages...")
        outcome = await self.drop_all()
        success = outcome.all_succeeded
        if not await self.delete_configuration(outcome):
            success = False

        if not outcome.succeeded:
            print(
                f"\n{BOLD_RED}✗ Every storage drop failed; input files were "
                f"left in place.{RESET}"
            )
            return False

        deleted, failed = self.delete_input_files()
        print(
            f"\n  Input files deleted: {deleted}"
            + (f", failed: {failed}" if failed else "")
        )
        if failed:
            success = False

        if success:
            print(
                f"\n{BOLD_GREEN}✓ Workspace cleared.{RESET} The server can be started now."
            )
        else:
            print(
                f"\n{BOLD_RED}✗ The clear finished with errors{RESET} "
                f"({len(outcome.succeeded)} of {len(DATA_STORAGE_LABELS)} storages "
                f"dropped). Fix the cause and re-run this tool."
            )
        return success

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def print_header(self):
        print("\n" + "=" * 60)
        print(f"{BOLD_CYAN}LightRAG Offline Storage Clear Tool{RESET}")
        print("=" * 60)
        print("\nDrops every data storage of ONE workspace, like the WebUI")
        print("'Clear' button. The LLM response cache is never touched.")
        print("\n" + "=" * 60)
        print(f"{BOLD_RED}⚠️  IMPORTANT: STOP THE LIGHTRAG SERVER FIRST{RESET}")
        print("=" * 60)
        print("\nRunning this while the server (or any other writer) is active")
        print("tears storage down under a live pipeline and loses data.")
        print("\nOpening the storages runs the same one-time migrations a server")
        print("start would (e.g. moving legacy vectors into the current container)")
        print("before anything is shown or asked; that moves data, never deletes it.")

    async def run(self) -> bool:
        """Returns True on a clean clear or a deliberate cancellation."""
        try:
            initialize_share_data(workers=1)

            self.print_header()
            if not self.confirm_server_stopped():
                return True

            if not await self.setup_storages():
                return False

            print("\nReading what the workspace holds...")
            try:
                summary = await self.collect_summary()
            except RemoteBackendUnavailableError as e:
                print(
                    f"\n✗ {e}\n  A server backend that cannot answer will not "
                    f"serve the drop either, so nothing was deleted."
                )
                return False
            self.print_summary(summary)

            if not self.confirm_delete_all():
                return True

            return await self.clear()
        except KeyboardInterrupt:
            print("\n\n✗ Interrupted by user")
            return False
        except Exception as e:
            print(f"\n✗ Clear tool failed: {e}")
            import traceback

            traceback.print_exc()
            return False
        finally:
            for storage in [*self.storages.values(), self.configuration_storage]:
                if storage is not None:
                    try:
                        await storage.finalize()
                    except Exception:
                        pass
            try:
                finalize_share_data()
            except Exception:
                pass
            # Last, after every storage that writes under it is down.
            if self._holds_working_dir:
                self._holds_working_dir = False
                release_working_dir_lock(self.config_dir)


async def async_main() -> bool:
    tool = ClearTool()
    return await tool.run()


USAGE = """usage: lightrag-clear-storage

Takes no options. Configuration comes from .env / the environment exactly as
the server reads it: WORKSPACE, WORKING_DIR, INPUT_DIR, LIGHTRAG_*_STORAGE,
EMBEDDING_* and the backend connection settings.

Server flags such as --workspace or --input-dir are refused rather than
ignored: the embedding function is built through the server's own argument
parser, which WOULD honor them, while the workspace and directories this tool
clears come from the environment -- so a flag could select one embedding
configuration and clear another workspace. See README_CLEAR_STORAGE.md.
"""


def main():
    """Synchronous entry point. Exits non-zero on any failure or partial clear."""
    if len(sys.argv) > 1:
        wants_help = sys.argv[1] in ("-h", "--help")
        print(USAGE, file=sys.stdout if wants_help else sys.stderr)
        if not wants_help:
            print(
                f"error: unrecognized arguments: {' '.join(sys.argv[1:])}",
                file=sys.stderr,
            )
        raise SystemExit(0 if wants_help else 2)
    load_dotenv(dotenv_path=".env", override=False)
    setup_logger("lightrag", level="INFO")
    success = asyncio.run(async_main())
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
