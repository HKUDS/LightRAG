import os
from collections import deque
from dataclasses import dataclass
from operator import itemgetter
from typing import final

from lightrag.file_atomic import atomic_write, reap_orphan_tmp_files
from lightrag.types import KnowledgeGraph, KnowledgeGraphNode, KnowledgeGraphEdge
from lightrag.utils import (
    logger,
    validate_xml_attributes,
    validate_workspace,
)
from lightrag.base import BaseGraphStorage
import networkx as nx
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
           ``busy`` / ``destructive_busy`` flags (see ``AGENTS.md``
           *Pipeline concurrency contract*) guarantee at most one process
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
           ``graph``.

    Cross-process sync protocol (identical in shape to
    ``NanoVectorDBStorage`` — see that class's docstring for the canonical
    description):
        Writer side (``index_done_callback``):
            1. ``write_nx_graph`` atomically writes the GraphML file
               (``atomic_write`` lays a tmp file beside the target and
               renames it into place — readers either see the previous
               file in full or the new file in full, never a torn write).
            2. ``set_all_update_flags`` flips every process's
               ``storage_updated`` flag (including the writer's own).
            3. Immediately reset the writer's own flag to ``False`` so
               the next call to ``_get_graph`` does not trigger a
               self-reload of the data this process just wrote.
        Reader side (any method that goes through ``_get_graph``):
            1. Inside ``_storage_lock``, observe
               ``storage_updated.value is True``.
            2. **Fully reload** ``self._graph`` from disk via
               ``load_nx_graph``. networkx GraphML has no incremental
               sync API, so the entire file is re-parsed.
            3. Reset the reader's own flag.

    Lock scope:
        ``_storage_lock`` is a per-``(namespace, workspace)`` keyed lock
        spanning both intra-process coroutines and inter-process workers.
        It wraps only the *reload* and *commit* critical sections, not
        every ``graph.xxx`` call. Operating on ``graph`` outside the lock
        is safe today *because of invariant (3)* — if either premise is
        ever broken (e.g. ``graph.xxx`` is moved to a thread pool, or
        networkx is swapped for an async graph library), the lock scope
        must be widened to cover the mutation/read itself.

    Implementation differences from ``NanoVectorDBStorage`` (same design,
    different surface):
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

    Non-pipeline write paths:
        The pipeline's ``busy`` gate serializes mutation calls reached
        through the document ingestion and purge flows. The following
        entry points are **not** serialized by the pipeline gate and
        must be guarded externally:
            * ``drop`` — currently gated by the API layer (the
              ``/documents/clear`` endpoint takes the pipeline busy
              reservation before invoking it).
            * ``delete_node`` / ``remove_nodes`` / ``remove_edges`` /
              ``upsert_node`` / ``upsert_edge`` when invoked from
              ``utils_graph.py`` admin flows (``adelete_by_entity`` /
              ``adelete_by_relation`` / entity-edit flows). These flows
              are currently not exposed in the WebUI; any future caller
              must arrange single-writer serialization the same way the
              pipeline does.
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

        reap_orphan_tmp_files(self._graphml_xml_file, workspace=self.workspace or "_")

        # Load initial graph
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

    async def _get_graph(self):
        """Return the live ``networkx.Graph``, reloading from disk if needed.

        This is the **single entry point** every public method funnels
        through to obtain ``self._graph``. It is also the **only place
        readers transition to a fresher on-disk snapshot**: when another
        process has committed (via ``index_done_callback``) and flipped
        this process's ``storage_updated`` flag, the next call here
        rebuilds ``self._graph`` by re-parsing the entire GraphML file.
        networkx has no incremental sync API — the reload is
        unconditionally a full file reload.

        Under the *Single writer* invariant (see class docstring), the
        reload branch never fires in the writer process: the writer
        resets its own flag at the end of every ``index_done_callback``.
        The branch exists for readers.

        ``_storage_lock`` is held during the check-and-reload to (a)
        serialize concurrent reload attempts by sibling coroutines in
        the same process and (b) interlock with ``index_done_callback``
        so a reader cannot observe a partially-saved file.
        """
        async with self._storage_lock:
            # Check if data needs to be reloaded
            if self.storage_updated.value:
                logger.info(
                    f"[{self.workspace}] Process {os.getpid()} reloading graph {self._graphml_xml_file} due to modifications by another process"
                )
                # Reload data
                self._graph = (
                    NetworkXStorage.load_nx_graph(self._graphml_xml_file) or nx.Graph()
                )
                # Reset update flag
                self.storage_updated.value = False

            return self._graph

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
            * **First ``async with``** — early-return path for a
              hypothetical second writer. Under the current single-writer
              pipeline contract (class docstring, invariant 1) the
              ``storage_updated.value`` check is permanently ``False`` in
              the writer, so this branch is **dead code in production**.
              It is kept as defensive scaffolding for any future
              relaxation of the single-writer invariant; removing it
              would silently re-enable lost-write bugs the moment a
              second writer is introduced.
            * **Second ``async with``** — the actual save + notify.
        """
        async with self._storage_lock:
            # Check if storage was updated by another process
            if self.storage_updated.value:
                # Storage was updated by another process, reload data instead of saving
                logger.info(
                    f"[{self.workspace}] Graph was updated by another process, reloading..."
                )
                self._graph = (
                    NetworkXStorage.load_nx_graph(self._graphml_xml_file) or nx.Graph()
                )
                # Reset update flag
                self.storage_updated.value = False
                return False  # Return error

        # Acquire lock and perform persistence
        async with self._storage_lock:
            try:
                # Save data to disk
                NetworkXStorage.write_nx_graph(
                    self._graph, self._graphml_xml_file, self.workspace
                )
                # Notify other processes that data has been updated
                await set_all_update_flags(self.namespace, workspace=self.workspace)
                # Reset own update flag to avoid self-reloading
                self.storage_updated.value = False
                return True  # Return success
            except Exception as e:
                # Raise (do NOT swallow + return False): _insert_done's
                # _flush_one only detects failures via exceptions, so a
                # swallowed graph-save error would let the document be marked
                # PROCESSED with the graph changes unpersisted. Surfacing it
                # aligns this backend with the others (faiss/nano raise too).
                logger.error(f"[{self.workspace}] Error saving graph: {e}")
                raise

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
            - On failure: {"status": "error", "message": "<error details>"}
        """
        try:
            async with self._storage_lock:
                # delete _client_file_name
                if os.path.exists(self._graphml_xml_file):
                    os.remove(self._graphml_xml_file)
                self._graph = nx.Graph()
                # Notify other processes that data has been updated
                await set_all_update_flags(self.namespace, workspace=self.workspace)
                # Reset own update flag to avoid self-reloading
                self.storage_updated.value = False
                logger.info(
                    f"[{self.workspace}] Process {os.getpid()} drop graph file:{self._graphml_xml_file}"
                )
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error dropping graph file:{self._graphml_xml_file}: {e}"
            )
            return {"status": "error", "message": str(e)}
