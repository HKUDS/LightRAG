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
   most recently updated documents, the number of text chunks, whether each
   vector index is empty, the recorded embedding baselines and the input
   files -- every read fail-loud, so a backend that cannot answer aborts the
   run instead of rendering as "nothing here";
2. asks the operator to type ``Delete All``;
3. drops the eleven data storages, then the workspace's configuration
   records (only when EVERY drop succeeded), then the top-level input files.

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
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv

# Add project root to path for imports
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from lightrag.base import DocStatus
from lightrag.config_store import (
    EMBEDDING_TARGETS,
    create_configuration_storage,
    delete_workspace_configuration,
    describe_configuration_container,
    read_embedding_baselines,
    resolve_config_dir,
    resolve_configuration_storage,
)
from lightrag.constants import (
    DEFAULT_COSINE_THRESHOLD,
    DEFAULT_EMBEDDING_BATCH_NUM,
    DEFAULT_INPUT_DIR,
    DEFAULT_WORKING_DIR,
)
from lightrag.exceptions import (
    CorruptStorageSnapshotError,
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
from lightrag.tools.rebuild_vdb import enumerate_kv_keys
from lightrag.utils import (
    EmbeddingFunc,
    get_env_value,
    logger,
    setup_logger,
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

# ANSI color codes for terminal output
BOLD_CYAN = "\033[1;36m"
BOLD_RED = "\033[1;31m"
BOLD_GREEN = "\033[1;32m"
RESET = "\033[0m"


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
        self.workspace = ""
        self.global_config: Dict[str, Any] = {}
        self.storage_names: Dict[str, str] = {}
        # Vector targets whose ``initialize()`` refused, label -> diagnostic.
        # They are dropped like every other storage; nothing is backed up,
        # because the operator is about to type the phrase that deletes it.
        self.refused_vdbs: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Configuration / setup
    # ------------------------------------------------------------------

    def resolve_storage_names(self) -> Dict[str, str]:
        kv = os.getenv("LIGHTRAG_KV_STORAGE", "JsonKVStorage")
        return {
            "graph": os.getenv("LIGHTRAG_GRAPH_STORAGE", "NetworkXStorage"),
            "vector": os.getenv("LIGHTRAG_VECTOR_STORAGE", "NanoVectorDBStorage"),
            "kv": kv,
            "doc_status": os.getenv(
                "LIGHTRAG_DOC_STATUS_STORAGE", "JsonDocStatusStorage"
            ),
            # Resolved exactly as the server resolves it: the records this
            # tool deletes last live in the container the server reads.
            "config": resolve_configuration_storage(
                os.getenv("LIGHTRAG_CONFIG_STORAGE", ""), kv_storage=kv
            ),
        }

    def resolve_config_dir(self) -> str:
        return resolve_config_dir(
            os.getenv("LIGHTRAG_CONFIG_DIR", ""),
            os.getenv("WORKING_DIR", DEFAULT_WORKING_DIR),
        )

    def resolve_input_dir(self) -> str:
        return os.path.abspath(get_env_value("INPUT_DIR", DEFAULT_INPUT_DIR))

    def check_env_vars(self, storage_name: str) -> None:
        """Warn about missing env vars (initialization is the real validation)."""
        required_vars = STORAGE_ENV_REQUIREMENTS.get(storage_name, [])
        missing_vars = [var for var in required_vars if var not in os.environ]
        if missing_vars:
            print(
                f"⚠️  Warning: {storage_name} normally requires: "
                f"{', '.join(missing_vars)} (may be provided via config.ini)"
            )

    def build_embedding_stub(self) -> EmbeddingFunc:
        """An embedding function that names the server's model but never embeds.

        Nothing here computes a vector, so the api extra is not needed. The
        model name and dimension still matter: Qdrant and PostgreSQL derive
        the collection / table name from them, so an unnamed stub would open
        -- and clear -- the wrong container.
        """

        async def _no_embedding(*_args, **_kwargs):
            raise RuntimeError(
                "lightrag-clear-storage never embeds; a storage asked it to"
            )

        return EmbeddingFunc(
            embedding_dim=get_env_value("EMBEDDING_DIM", 1024, int),
            func=_no_embedding,
            model_name=get_env_value("EMBEDDING_MODEL", None, special_none=True),
        )

    def build_global_config(self, embedding_func: EmbeddingFunc) -> Dict[str, Any]:
        return {
            "working_dir": os.getenv("WORKING_DIR", DEFAULT_WORKING_DIR),
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

    def build_storages(self, embedding_func: EmbeddingFunc) -> Dict[str, Any]:
        """Instantiate the eleven data storages ``/documents/clear`` drops.

        Namespaces and ``meta_fields`` match ``LightRAG.__post_init__`` so
        every backend resolves the same container the server writes to.
        """
        from lightrag.kg.factory import get_storage_class

        graph_cls = get_storage_class(self.storage_names["graph"])
        vector_cls = get_storage_class(self.storage_names["vector"])
        kv_cls = get_storage_class(self.storage_names["kv"])
        doc_status_cls = get_storage_class(self.storage_names["doc_status"])

        def kv(namespace: str):
            return kv_cls(
                namespace=namespace,
                workspace=self.workspace,
                global_config=self.global_config,
                embedding_func=embedding_func,
            )

        def vdb(namespace: str, meta_fields: set):
            return vector_cls(
                namespace=namespace,
                workspace=self.workspace,
                global_config=self.global_config,
                embedding_func=embedding_func,
                meta_fields=meta_fields,
            )

        return {
            "text_chunks": kv(NameSpace.KV_STORE_TEXT_CHUNKS),
            "full_docs": kv(NameSpace.KV_STORE_FULL_DOCS),
            "full_entities": kv(NameSpace.KV_STORE_FULL_ENTITIES),
            "full_relations": kv(NameSpace.KV_STORE_FULL_RELATIONS),
            "entity_chunks": kv(NameSpace.KV_STORE_ENTITY_CHUNKS),
            "relation_chunks": kv(NameSpace.KV_STORE_RELATION_CHUNKS),
            "entities_vdb": vdb(
                NameSpace.VECTOR_STORE_ENTITIES,
                {"entity_name", "source_id", "content", "file_path"},
            ),
            "relationships_vdb": vdb(
                NameSpace.VECTOR_STORE_RELATIONSHIPS,
                {"src_id", "tgt_id", "source_id", "content", "file_path"},
            ),
            "chunks_vdb": vdb(
                NameSpace.VECTOR_STORE_CHUNKS,
                {"full_doc_id", "content", "file_path"},
            ),
            "chunk_entity_relation_graph": graph_cls(
                namespace=NameSpace.GRAPH_STORE_CHUNK_ENTITY_RELATION,
                workspace=self.workspace,
                global_config=self.global_config,
                embedding_func=embedding_func,
            ),
            "doc_status": doc_status_cls(
                namespace=NameSpace.DOC_STATUS,
                workspace=self.workspace,
                global_config=self.global_config,
                embedding_func=None,
            ),
        }

    async def setup_storages(self) -> bool:
        """Instantiate and initialize every storage. Returns False on failure.

        The configuration storage, the KV stores, the graph and doc_status
        take the server-identical path and any failure aborts: a clear that
        could not delete its configuration records last is not a clean clear.
        The three vector targets are initialized one at a time, and ONLY the
        typed embedding-space refusal and the typed corrupt-snapshot error
        are tolerated -- they are the states this tool exists to clear, and
        ``drop()`` is servable while refused. Anything else aborts.
        """
        from lightrag.kg.factory import get_storage_class

        self.storage_names = self.resolve_storage_names()
        self.config_dir = self.resolve_config_dir()
        self.input_dir = self.resolve_input_dir()
        self.workspace = os.getenv("WORKSPACE", "")

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

        embedding_func = self.build_embedding_stub()
        self.global_config = self.build_global_config(embedding_func)
        self.storages = self.build_storages(embedding_func)
        self.configuration_storage = create_configuration_storage(
            get_storage_class(self.storage_names["config"]),
            global_config=self.global_config,
            embedding_func=embedding_func,
        )

        print("\nInitializing storages...")
        try:
            await self.configuration_storage.initialize()
            for label in DATA_STORAGE_LABELS:
                if label in VECTOR_LABELS:
                    continue
                await self.storages[label].initialize()
            for label in VECTOR_LABELS:
                try:
                    await self.storages[label].initialize()
                except (VectorSpaceMismatchError, CorruptStorageSnapshotError) as e:
                    self.refused_vdbs[label] = str(e)
                    print(f"⚠️  {label} refused to attach: {e}")
        except Exception as e:
            print(f"✗ Storage initialization failed: {e}")
            for storage_name in set(self.storage_names.values()):
                required = STORAGE_ENV_REQUIREMENTS.get(storage_name, [])
                if required:
                    print(f"  {storage_name} requires: {', '.join(required)}")
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

    async def collect_summary(self) -> Dict[str, Any]:
        """Read everything the operator sees before confirming.

        Every read here is fail-loud on purpose: a swallowed error that
        renders as zero documents is exactly what makes an operator delete
        the wrong workspace. So counts come from
        ``count_docs_by_statuses(strict=True)``, never the best-effort
        ``get_status_counts()``, and a raise anywhere aborts the run with
        nothing touched.
        """
        doc_status = self.storages["doc_status"]
        counts: Dict[str, int] = {}
        for status in DocStatus:
            counts[status.value] = await doc_status.count_docs_by_statuses(
                [status], strict=True
            )
        recent, total = await doc_status.get_docs_paginated(
            page=1,
            page_size=RECENT_DOCS_SHOWN,
            sort_field="updated_at",
            sort_direction="desc",
        )
        chunk_count = len(await enumerate_kv_keys(self.storages["text_chunks"]))

        vectors: Dict[str, str] = {}
        for label in VECTOR_LABELS:
            vdb = self.storages[label]
            if label in self.refused_vdbs:
                vectors[label] = "refused to attach (will be dropped)"
            elif not getattr(vdb, "persists_vectors", True):
                vectors[label] = "no-op backend (nothing stored)"
            elif await vdb.is_empty():
                vectors[label] = "EMPTY"
            else:
                vectors[label] = "has vectors"

        baselines = await read_embedding_baselines(
            self.configuration_storage, self.workspace
        )
        return {
            "counts": counts,
            "total_docs": total,
            "recent": recent,
            "chunk_count": chunk_count,
            "vectors": vectors,
            "baselines": baselines,
            "input_files": self.input_files(),
            # A backend-specific ``*_WORKSPACE`` variable outranks WORKSPACE
            # inside the storage constructor, and nothing above it sees that.
            # Some backends write the effective name back onto ``workspace``
            # (PostgreSQL, MongoDB, Milvus); others keep it elsewhere (Redis,
            # Qdrant), so the override list from the environment is the
            # authoritative signal and the resolved names are a bonus.
            "workspace_overrides": warn_about_workspace_overrides(),
            "resolved_workspaces": sorted(
                {
                    str(
                        getattr(storage, "effective_workspace", None)
                        or getattr(storage, "workspace", "")
                        or ""
                    )
                    for storage in self.storages.values()
                }
            ),
        }

    def print_summary(self, summary: Dict[str, Any]) -> None:
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

        print(f"\nDocuments by status ({summary['total_docs']} total):")
        for status_value, count in summary["counts"].items():
            print(f"    {status_value:14s} {count}")

        recent = summary["recent"]
        print(f"\nMost recently updated documents (up to {RECENT_DOCS_SHOWN}):")
        if not recent:
            print("    (none)")
        for doc_id, doc in recent:
            status = getattr(doc, "status", "")
            status_value = getattr(status, "value", status)
            updated_at = getattr(doc, "updated_at", "") or ""
            file_path = getattr(doc, "file_path", "") or ""
            print(f"    {updated_at:20s} {status_value:11s} {file_path}  [{doc_id}]")

        print(f"\nText chunks: {summary['chunk_count']}")

        print("\nVector storages:")
        for label, state in summary["vectors"].items():
            print(f"    {label:18s} {state}")

        print("\nRecorded embedding baselines:")
        for target in EMBEDDING_TARGETS:
            baseline = summary["baselines"].get(target)
            if baseline is None:
                print(f"    {target:14s} (none recorded)")
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
            print("\n✓ Operation cancelled - nothing was deleted")
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
        failed only leaves more for the re-run.
        """
        labels = list(DATA_STORAGE_LABELS)
        results = await asyncio.gather(
            *(self.storages[label].drop() for label in labels),
            return_exceptions=True,
        )
        outcome = DropOutcome()
        for label, result in zip(labels, results):
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
            except Exception as e:
                print(
                    f"\n✗ Could not read the workspace, so nothing was deleted: {e}\n"
                    f"  A count that cannot be read must not be shown as zero."
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


def main():
    """Synchronous entry point. Exits non-zero on any failure or partial clear."""
    load_dotenv(dotenv_path=".env", override=False)
    setup_logger("lightrag", level="INFO")
    success = asyncio.run(async_main())
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
