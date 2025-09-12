import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_embed
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status

# --- INICIO DE LA MODIFICACIÓN 1: Añadir la librería de grafos ---
import networkx as nx # Esencial para manejar la estructura del grafo
# --- FIN DE LA MODIFICACIÓN 1 ---

# WorkingDir
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKING_DIR = os.path.join(ROOT_DIR, "myEpidemiologyKG") # Cambiado para reflejar el nuevo propósito
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)
print(f"WorkingDir: {WORKING_DIR}")

# Configuraciones de Redis, Neo4j y Milvus (sin cambios)
os.environ["REDIS_URI"] = "redis://localhost:6379"
os.environ["NEO4J_URI"] = "neo4j://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "12345678"
os.environ["MILVUS_URI"] = "http://localhost:19530"
os.environ["MILVUS_USER"] = "root"
os.environ["MILVUS_PASSWORD"] = "Milvus"
os.environ["MILVUS_DB_NAME"] = "lightrag_epi"


async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    # ... (sin cambios)
    return await openai_complete_if_cache(
        "deepseek-chat",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key="",
        base_url="",
        **kwargs,
    )


embedding_func = EmbeddingFunc(
    # ... (sin cambios)
    embedding_dim=768,
    max_token_size=512,
    func=lambda texts: ollama_embed(
        texts, embed_model="shaw/dmeta-embedding-zh", host="http://117.50.173.35:11434"
    ),
)


async def initialize_rag():
    # ... (sin cambios)
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        summary_max_tokens=10000,
        embedding_func=embedding_func,
        chunk_token_size=512,
        chunk_overlap_token_size=256,
        kv_storage="RedisKVStorage",
        graph_storage="Neo4JStorage",
        vector_storage="MilvusVectorDBStorage",
        doc_status_storage="RedisKVStorage",
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag

# --- INICIO DE LA MODIFICACIÓN 2: Nueva función de consulta causal (el "Retriever" del DAG) ---
async def rag_dag_query(rag_instance: LightRAG, query: str, start_node: str):
    """
    Esta función actúa como el retriever para el DAG.
    Interpreta la pregunta y recorre el grafo para encontrar causas o efectos.
    """
    print(f"\n--- Ejecutando consulta causal para: '{query}' ---")
    query_lower = query.lower()
    
    context_nodes = []
    direction = "desconocida"

    # 1. Interpretar la intención de la consulta y recorrer el grafo
    if any(keyword in query_lower for keyword in ["consecuencias", "efectos", "causa", "provoca", "lleva a"]):
        direction = "Consecuencias (hacia adelante)"
        # NOTA: Debes implementar `get_downstream_nodes` en `memgraph_impl.py`
        # Esta función debe seguir las flechas -> desde el start_node
        context_nodes = await rag_instance.graph_storage.get_downstream_nodes(start_node)
    elif any(keyword in query_lower for keyword in ["causado por", "origen", "factor de riesgo"]):
        direction = "Causas (hacia atrás)"
        # NOTA: Debes implementar `get_upstream_nodes` en `memgraph_impl.py`
        # Esta función debe seguir las flechas -> hacia el start_node
        context_nodes = await rag_instance.graph_storage.get_upstream_nodes(start_node)
    else:
        print("La consulta no parece causal. Usando recuperación local estándar.")
        return rag_instance.query(query, param=QueryParam(mode="local"))

    if not context_nodes:
        return f"No se encontraron {direction} para '{start_node}' en el grafo de conocimiento."

    # 2. Construir un contexto claro para el LLM final
    context_str = f"Se encontró la siguiente cadena causal para '{start_node}' en dirección '{direction}':\n"
    for node in context_nodes:
        # Suponiendo que cada `node` es una tupla (source, target, relationship_type)
        source, target, rel_type = node
        context_str += f"- La entidad '{source}' tiene una relación de '{rel_type}' con la entidad '{target}'.\n"
    
    final_prompt = f"Basado en el siguiente contexto causal, responde la pregunta del usuario de forma clara y concisa.\n\nContexto:\n{context_str}\n\nPregunta: {query}"
    
    # 3. Generar la respuesta final
    final_answer = await llm_model_func(final_prompt, system_prompt="Eres un asistente experto en epidemiología.")
    return final_answer
# --- FIN DE LA MODIFICACIÓN 2 ---

# --- INICIO DE LA MODIFICACIÓN 3: Nueva función `main` asíncrona y enfocada en el caso de uso ---
async def main():
    # Inicializar la instancia de RAG
    rag = await initialize_rag()
    
    # Texto de ejemplo de epidemiología para insertar en el grafo
    # (Reemplaza "book.txt" con un archivo que contenga textos médicos o epidemiológicos)
    epidemiology_text = """
    A cohort study published in the Lancet found that long-term exposure to PM2.5 air pollution is a significant risk factor for the development of hypertension. 
    The mechanism is believed to involve systemic inflammation, which is a known predictor for cardiovascular diseases. 
    Hypertension, in turn, is a primary cause of ischemic stroke. Vaping has also been linked to systemic inflammation.
    """
    print("\n--- Insertando texto de epidemiología en el grafo ---")
    rag.insert(epidemiology_text)
    
    # Esperar un momento para que la indexación asíncrona progrese
    await asyncio.sleep(10)

    # Realizar consultas causales utilizando nuestra nueva función
    respuesta1 = await rag_dag_query(rag, query="¿Cuáles son las consecuencias de la hipertensión?", start_node="Hypertension")
    print("\n[Respuesta 1]:")
    print(respuesta1)
    
    respuesta2 = await rag_dag_query(rag, query="¿Qué es un factor de riesgo para la hipertensión?", start_node="Hypertension")
    print("\n[Respuesta 2]:")
    print(respuesta2)

    respuesta3 = await rag_dag_query(rag, query="¿A qué lleva la exposición a PM2.5?", start_node="PM2.5 Exposure")
    print("\n[Respuesta 3]:")
    print(respuesta3)


if __name__ == "__main__":
    asyncio.run(main())
# --- FIN DE LA MODIFICACIÓN 3 ---
