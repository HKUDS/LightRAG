import pipmaster as pm
from llama_index.core.llms import (
    ChatMessage,
    MessageRole,
    ChatResponse,
)
from typing import List, Optional, Tuple, Dict, Any
from lightrag.utils import logger
import re # Importamos re para el parseo

# --- INICIO DE LA MODIFICACIÓN 1: Añadir networkx ---
# Asegurarse de que networkx está instalado y añadirlo al principio del archivo
if not pm.is_installed("networkx"):
    pm.install("networkx")
import networkx as nx
# --- FIN DE LA MODIFICACIÓN 1 ---

# Install required dependencies
if not pm.is_installed("llama-index"):
    pm.install("llama-index")

from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.settings import Settings as LlamaIndexSettings
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from lightrag.utils import (
    wrap_embedding_func_with_attrs,
)
from lightrag.exceptions import (
    APIConnectionError,
    RateLimitError,
    APITimeoutError,
)
import numpy as np


def configure_llama_index(settings: LlamaIndexSettings = None, **kwargs):
    """
    Configure LlamaIndex settings.

    Args:
        settings: LlamaIndex Settings instance. If None, uses default settings.
        **kwargs: Additional settings to override/configure
    """
    if settings is None:
        settings = LlamaIndexSettings()

    # Update settings with any provided kwargs
    for key, value in kwargs.items():
        if hasattr(settings, key):
            setattr(settings, key, value)
        else:
            logger.warning(f"Unknown LlamaIndex setting: {key}")

    # Set as global settings
    LlamaIndexSettings.set_global(settings)
    return settings


def format_chat_messages(messages):
    """Format chat messages into LlamaIndex format."""
    formatted_messages = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            formatted_messages.append(
                ChatMessage(role=MessageRole.SYSTEM, content=content)
            )
        elif role == "assistant":
            formatted_messages.append(
                ChatMessage(role=MessageRole.ASSISTANT, content=content)
            )
        elif role == "user":
            formatted_messages.append(
                ChatMessage(role=MessageRole.USER, content=content)
            )
        else:
            logger.warning(f"Unknown role {role}, treating as user message")
            formatted_messages.append(
                ChatMessage(role=MessageRole.USER, content=content)
            )

    return formatted_messages


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError)
    ),
)
async def llama_index_complete_if_cache(
    model: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: List[dict] = [],
    enable_cot: bool = False,
    chat_kwargs={},
) -> str:
    """Complete the prompt using LlamaIndex."""
    if enable_cot:
        logger.debug(
            "enable_cot=True is not supported for LlamaIndex implementation and will be ignored."
        )
    try:
        # Format messages for chat
        formatted_messages = []

        # Add system message if provided
        if system_prompt:
            formatted_messages.append(
                ChatMessage(role=MessageRole.SYSTEM, content=system_prompt)
            )

        # Add history messages
        for msg in history_messages:
            formatted_messages.append(
                ChatMessage(
                    role=MessageRole.USER
                    if msg["role"] == "user"
                    else MessageRole.ASSISTANT,
                    content=msg["content"],
                )
            )

        # Add current prompt
        formatted_messages.append(ChatMessage(role=MessageRole.USER, content=prompt))

        response: ChatResponse = await model.achat(
            messages=formatted_messages, **chat_kwargs
        )

        # In newer versions, the response is in message.content
        content = response.message.content
        return content

    except Exception as e:
        logger.error(f"Error in llama_index_complete_if_cache: {str(e)}")
        raise


async def llama_index_complete(
    prompt,
    system_prompt=None,
    history_messages=None,
    enable_cot: bool = False,
    keyword_extraction=False,
    settings: LlamaIndexSettings = None,
    **kwargs,
) -> str:
    """
    Main completion function for LlamaIndex

    Args:
        prompt: Input prompt
        system_prompt: Optional system prompt
        history_messages: Optional chat history
        keyword_extraction: Whether to extract keywords from response
        settings: Optional LlamaIndex settings
        **kwargs: Additional arguments
    """
    if history_messages is None:
        history_messages = []

    kwargs.pop("keyword_extraction", None)
    result = await llama_index_complete_if_cache(
        kwargs.get("llm_instance"),
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        enable_cot=enable_cot,
        **kwargs,
    )
    return result


@wrap_embedding_func_with_attrs(embedding_dim=1536)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError)
    ),
)
async def llama_index_embed(
    texts: list[str],
    embed_model: BaseEmbedding = None,
    settings: LlamaIndexSettings = None,
    **kwargs,
) -> np.ndarray:
    """
    Generate embeddings using LlamaIndex

    Args:
        texts: List of texts to embed
        embed_model: LlamaIndex embedding model
        settings: Optional LlamaIndex settings
        **kwargs: Additional arguments
    """
    if settings:
        configure_llama_index(settings)

    if embed_model is None:
        raise ValueError("embed_model must be provided")

    # Use _get_text_embeddings for batch processing
    embeddings = embed_model._get_text_embeddings(texts)
    return np.array(embeddings)


# --- INICIO DE LA MODIFICACIÓN 2: Nueva función para parsear y validar datos para el DAG ---
def parse_and_validate_graph_data(
    llm_output: str,
    existing_graph: nx.DiGraph,
    tuple_delimiter: str = "<|>",
    record_delimiter: str = "##",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Parses the raw text output from the LLM and validates new relationships to maintain a DAG.

    Args:
        llm_output (str): The raw string output from the entity extraction LLM call.
        existing_graph (nx.DiGraph): The current state of the knowledge graph.
        tuple_delimiter (str): The delimiter used for fields within an entity/relationship tuple.
        record_delimiter (str): The delimiter used to separate records.

    Returns:
        Tuple[List[Dict], List[Dict]]: A tuple containing two lists:
                                       - A list of valid entities to be added/updated.
                                       - A list of valid relationships (edges) to be added.
    """
    valid_entities = []
    valid_relationships = []

    # Crear una copia temporal del grafo para probar la adición de aristas
    temp_graph = existing_graph.copy()

    records = llm_output.strip().split(record_delimiter)
    for record in records:
        record = record.strip()
        if not record or not record.startswith("("):
            continue
        
        # Eliminar paréntesis y parsear los campos
        content = record[1:-1]
        fields = content.split(tuple_delimiter)

        try:
            record_type = fields[0]
            if record_type == "entity" and len(fields) == 4:
                entity = {
                    "name": fields[1],
                    "type": fields[2],
                    "description": fields[3],
                }
                valid_entities.append(entity)
                # Añadir nodo a nuestro grafo temporal para la validación de relaciones
                temp_graph.add_node(fields[1])

            elif record_type == "relationship" and len(fields) == 5:
                source, target, rel_type, desc = fields[1], fields[2], fields[3], fields[4]
                
                # --- Lógica de validación de Ciclos ---
                # Comprobar si la adición de esta arista crearía un ciclo en el grafo temporal
                if not nx.has_path(temp_graph, target, source):
                    relationship = {
                        "source": source,
                        "target": target,
                        "type": rel_type,
                        "description": desc,
                    }
                    valid_relationships.append(relationship)
                    # Añadir la arista validada al grafo temporal para las siguientes comprobaciones
                    temp_graph.add_edge(source, target)
                else:
                    logger.warning(
                        f"Ignorando relación de '{source}' a '{target}' para prevenir un ciclo."
                    )
        except IndexError:
            logger.warning(f"Error al parsear el registro: {record}")
            continue

    return valid_entities, valid_relationships
# --- FIN DE LA MODIFICACIÓN 2 ---
