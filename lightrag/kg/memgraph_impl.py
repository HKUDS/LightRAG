import os
import asyncio
import random
from dataclasses import dataclass
from typing import final, List
import configparser

from ..utils import logger
from ..base import BaseGraphStorage
from ..types import KnowledgeGraph, KnowledgeGraphNode, KnowledgeGraphEdge
from ..constants import GRAPH_FIELD_SEP
from ..kg.shared_storage import get_data_init_lock, get_graph_db_lock
import pipmaster as pm

# Asegurarse de que neo4j está instalado
if not pm.is_installed("neo4j"):
    pm.install("neo4j")
from neo4j import (
    AsyncGraphDatabase,
    AsyncManagedTransaction,
)
from neo4j.exceptions import TransientError, ResultFailedError

from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv(dotenv_path=".env", override=False)

MAX_GRAPH_NODES = int(os.getenv("MAX_GRAPH_NODES", 1000))

config = configparser.ConfigParser()
config.read("config.ini", "utf-8")


@final
@dataclass
class MemgraphStorage(BaseGraphStorage):
    # ... (El método __init__ y otros métodos iniciales permanecen sin cambios) ...
    def __init__(self, namespace, global_config, embedding_func, workspace=None):
        memgraph_workspace = os.environ.get("MEMGRAPH_WORKSPACE")
        if memgraph_workspace and memgraph_workspace.strip():
            workspace = memgraph_workspace

        if not workspace or not str(workspace).strip():
            workspace = "base"

        super().__init__(
            namespace=namespace,
            workspace=workspace,
            global_config=global_config,
            embedding_func=embedding_func,
        )
        self._driver = None

    def _get_workspace_label(self) -> str:
        return self.workspace

    async def initialize(self):
        # ... (Este método permanece sin cambios) ...
        async with get_data_init_lock():
            URI = os.environ.get(
                "MEMGRAPH_URI",
                config.get("memgraph", "uri", fallback="bolt://localhost:7687"),
            )
            USERNAME = os.environ.get(
                "MEMGRAPH_USERNAME", config.get("memgraph", "username", fallback="")
            )
            PASSWORD = os.environ.get(
                "MEMGRAPH_PASSWORD", config.get("memgraph", "password", fallback="")
            )
            DATABASE = os.environ.get(
                "MEMGRAPH_DATABASE",
                config.get("memgraph", "database", fallback="memgraph"),
            )

            self._driver = AsyncGraphDatabase.driver(
                URI,
                auth=(USERNAME, PASSWORD),
            )
            self._DATABASE = DATABASE
            try:
                async with self._driver.session(database=DATABASE) as session:
                    try:
                        workspace_label = self._get_workspace_label()
                        await session.run(
                            f"""CREATE INDEX ON :{workspace_label}(entity_id)"""
                        )
                        logger.info(
                            f"[{self.workspace}] Created index on :{workspace_label}(entity_id) in Memgraph."
                        )
                    except Exception as e:
                        logger.warning(
                            f"[{self.workspace}] Index creation on :{workspace_label}(entity_id) may have failed or already exists: {e}"
                        )
                    await session.run("RETURN 1")
                    logger.info(f"[{self.workspace}] Connected to Memgraph at {URI}")
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Failed to connect to Memgraph at {URI}: {e}"
                )
                raise

    # --- INICIO DE LA MODIFICACIÓN 1: Nueva función para chequear caminos (esencial para DAG) ---
    async def check_path_exists(self, source_node_id: str, target_node_id: str) -> bool:
        """
        Checks if a directed path exists from the source node to the target node.
        """
        if self._driver is None:
            raise RuntimeError("Memgraph driver is not initialized.")
        
        async with self._driver.session(database=self._DATABASE, default_access_mode="READ") as session:
            try:
                workspace_label = self._get_workspace_label()
                query = f"""
                MATCH (source:`{workspace_label}` {{entity_id: $source_id}}), (target:`{workspace_label}` {{entity_id: $target_id}})
                RETURN EXISTS( (source)-[*]->(target) ) AS path_exists
                """
                result = await session.run(query, source_id=source_node_id, target_id=target_node_id)
                record = await result.single()
                await result.consume()
                return record["path_exists"] if record else False
            except Exception as e:
                logger.error(f"[{self.workspace}] Error checking path existence: {e}")
                return False # Asumir que no existe en caso de error para ser conservador
    # --- FIN DE LA MODIFICACIÓN 1 ---

    # --- INICIO DE LA MODIFICACIÓN 2: Lógica de validación de ciclos en `upsert_edge` ---
    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ) -> None:
        """
        Upsert a DIRECTED edge and its properties, but only if it does not create a cycle.
        """
        if self._driver is None:
            raise RuntimeError("Memgraph driver is not initialized.")

        # VALIDACIÓN DE CICLO: Comprobar si ya existe un camino desde el target al source.
        # Si existe, añadir un camino de source a target crearía un ciclo.
        if await self.check_path_exists(target_node_id, source_node_id):
            logger.warning(
                f"[{self.workspace}] Ignorando la relación de '{source_node_id}' a '{target_node_id}' para prevenir un ciclo."
            )
            return # Detener la ejecución si se detecta un posible ciclo

        edge_properties = edge_data
        # Extraer el tipo de relación de edge_data, que viene de nuestro nuevo prompt
        relationship_type = edge_properties.get("type", "RELATED_TO").upper().replace(" ", "_")


        max_retries = 100
        initial_wait_time = 0.2
        backoff_factor = 1.1
        jitter_factor = 0.1

        for attempt in range(max_retries):
            try:
                logger.debug(
                    f"[{self.workspace}] Attempting edge upsert, attempt {attempt + 1}/{max_retries}"
                )
                async with self._driver.session(database=self._DATABASE) as session:

                    async def execute_upsert(tx: AsyncManagedTransaction):
                        workspace_label = self._get_workspace_label()
                        # Consulta Cypher modificada para crear una relación DIRIGIDA con un tipo dinámico.
                        query = f"""
                        MATCH (source:`{workspace_label}` {{entity_id: $source_entity_id}})
                        MATCH (target:`{workspace_label}` {{entity_id: $target_entity_id}})
                        MERGE (source)-[r:`{relationship_type}`]->(target)
                        SET r += $properties
                        """
                        result = await tx.run(
                            query,
                            source_entity_id=source_node_id,
                            target_entity_id=target_node_id,
                            properties=edge_properties,
                        )
                        await result.consume()

                    await session.execute_write(execute_upsert)
                    break 
            except (TransientError, ResultFailedError) as e:
                # ... (lógica de reintentos sin cambios) ...
                root_cause = e
                while hasattr(root_cause, "__cause__") and root_cause.__cause__:
                    root_cause = root_cause.__cause__
                is_transient = (
                    isinstance(root_cause, TransientError)
                    or isinstance(e, TransientError)
                    or "TransientError" in str(e)
                    or "Cannot resolve conflicting transactions" in str(e)
                )
                if is_transient:
                    if attempt < max_retries - 1:
                        jitter = random.uniform(0, jitter_factor) * initial_wait_time
                        wait_time = (
                            initial_wait_time * (backoff_factor**attempt) + jitter
                        )
                        logger.warning(
                            f"[{self.workspace}] Edge upsert failed. Attempt #{attempt + 1} retrying in {wait_time:.3f} seconds... Error: {str(e)}"
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(
                            f"[{self.workspace}] Memgraph transient error during edge upsert after {max_retries} retries: {str(e)}"
                        )
                        raise
                else:
                    logger.error(
                        f"[{self.workspace}] Non-transient error during edge upsert: {str(e)}"
                    )
                    raise
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Unexpected error during edge upsert: {str(e)}"
                )
                raise
    # --- FIN DE LA MODIFICACIÓN 2 ---

    # --- INICIO DE LA MODIFICACIÓN 3: Ajustes menores en otras funciones para direccionalidad ---
    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        # Modificar para buscar relaciones dirigidas
        # ...
        query = (
            f"MATCH (a:`{workspace_label}` {{entity_id: $source_entity_id}})-[r]->(b:`{workspace_label}` {{entity_id: $target_entity_id}}) "
            "RETURN COUNT(r) > 0 AS edgeExists"
        )
        # ... (el resto de la función sigue igual)
        if self._driver is None:
            raise RuntimeError(
                "Memgraph driver is not initialized. Call 'await initialize()' first."
            )
        async with self._driver.session(
            database=self._DATABASE, default_access_mode="READ"
        ) as session:
            try:
                workspace_label = self._get_workspace_label()
                result = await session.run(
                    query,
                    source_entity_id=source_node_id,
                    target_entity_id=target_node_id,
                )
                single_result = await result.single()
                await result.consume()
                return (
                    single_result["edgeExists"] if single_result is not None else False
                )
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Error checking edge existence between {source_node_id} and {target_node_id}: {str(e)}"
                )
                await result.consume()
                raise
    
    async def get_node_edges(self, source_node_id: str) -> List[tuple[str, str, str]] | None:
        # Modificar para devolver también el tipo de relación y manejar direccionalidad
        # ...
        query = f"""MATCH (n:`{workspace_label}` {{entity_id: $entity_id}})-[r]->(connected:`{workspace_label}`)
                     RETURN type(r) as rel_type, connected.entity_id as target_id
                     UNION
                     MATCH (connected:`{workspace_label}`)-[r]->(n:`{workspace_label}` {{entity_id: $entity_id}})
                     RETURN type(r) as rel_type, connected.entity_id as source_id
                  """
        # ... (el resto de la función necesitaría ser adaptada para manejar esta nueva consulta)
        if self._driver is None:
            raise RuntimeError(
                "Memgraph driver is not initialized. Call 'await initialize()' first."
            )
        try:
            async with self._driver.session(
                database=self._DATABASE, default_access_mode="READ"
            ) as session:
                try:
                    results = await session.run(query, entity_id=source_node_id)
                    edges = []
                    async for record in results:
                        if record.get("target_id"):
                            edges.append((source_node_id, record["target_id"], record["rel_type"]))
                        elif record.get("source_id"):
                            edges.append((record["source_id"], source_node_id, record["rel_type"]))
                    await results.consume()
                    return edges
                except Exception as e:
                    logger.error(
                        f"[{self.workspace}] Error getting edges for node {source_node_id}: {str(e)}"
                    )
                    await results.consume()
                    raise
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error in get_node_edges for {source_node_id}: {str(e)}"
            )
            raise


    # ... (El resto del archivo, como upsert_node, delete_node, etc., puede permanecer sin cambios significativos por ahora) ...
    # ... (Copiar y pegar el resto de las funciones originales desde `upsert_node` hasta el final) ...
    # upsert_node, delete_node, remove_nodes, remove_edges, drop, edge_degree, get_nodes_by_chunk_ids, etc.
    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        if self._driver is None:
            raise RuntimeError(
                "Memgraph driver is not initialized. Call 'await initialize()' first."
            )
        properties = node_data
        entity_type = properties["entity_type"]
        if "entity_id" not in properties:
            raise ValueError(
                "Memgraph: node properties must contain an 'entity_id' field"
            )
        max_retries = 100
        initial_wait_time = 0.2
        backoff_factor = 1.1
        jitter_factor = 0.1
        for attempt in range(max_retries):
            try:
                logger.debug(
                    f"[{self.workspace}] Attempting node upsert, attempt {attempt + 1}/{max_retries}"
                )
                async with self._driver.session(database=self._DATABASE) as session:
                    workspace_label = self._get_workspace_label()
                    async def execute_upsert(tx: AsyncManagedTransaction):
                        query = f"""
                        MERGE (n:`{workspace_label}` {{entity_id: $entity_id}})
                        SET n += $properties
                        SET n:`{entity_type}`
                        """
                        result = await tx.run(
                            query, entity_id=node_id, properties=properties
                        )
                        await result.consume()
                    await session.execute_write(execute_upsert)
                    break
            except (TransientError, ResultFailedError) as e:
                root_cause = e
                while hasattr(root_cause, "__cause__") and root_cause.__cause__:
                    root_cause = root_cause.__cause__
                is_transient = (
                    isinstance(root_cause, TransientError)
                    or isinstance(e, TransientError)
                    or "TransientError" in str(e)
                    or "Cannot resolve conflicting transactions" in str(e)
                )
                if is_transient:
                    if attempt < max_retries - 1:
                        jitter = random.uniform(0, jitter_factor) * initial_wait_time
                        wait_time = (
                            initial_wait_time * (backoff_factor**attempt) + jitter
                        )
                        logger.warning(
                            f"[{self.workspace}] Node upsert failed. Attempt #{attempt + 1} retrying in {wait_time:.3f} seconds... Error: {str(e)}"
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(
                            f"[{self.workspace}] Memgraph transient error during node upsert after {max_retries} retries: {str(e)}"
                        )
                        raise
                else:
                    logger.error(
                        f"[{self.workspace}] Non-transient error during node upsert: {str(e)}"
                    )
                    raise
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Unexpected error during node upsert: {str(e)}"
                )
                raise
