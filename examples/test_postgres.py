import os
import asyncio
from lightrag.kg.postgres_impl import PGGraphStorage
from lightrag.llm.ollama import ollama_embedding
from lightrag.utils import EmbeddingFunc

#########
# Uncomment the below two lines if running in a jupyter notebook to handle the async nature of rag.insert()
# import nest_asyncio
# nest_asyncio.apply()
#########

WORKING_DIR = "./local_neo4jWorkDir"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# AGE
os.environ["AGE_GRAPH_NAME"] = "dickens"

os.environ["POSTGRES_HOST"] = "localhost"
os.environ["POSTGRES_PORT"] = "15432"
os.environ["POSTGRES_USER"] = "rag"
os.environ["POSTGRES_PASSWORD"] = "rag"
os.environ["POSTGRES_DATABASE"] = "rag"


async def main():
    graph_db = PGGraphStorage(
        namespace="dickens",
        embedding_func=EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=8192,
            func=lambda texts: ollama_embedding(
                texts, embed_model="bge-m3", host="http://localhost:11434"
            ),
        ),
        global_config={},
    )
    await graph_db.initialize()
    labels = await graph_db.get_all_labels()
    print("all labels", labels)

    res = await graph_db.get_knowledge_graph("FEZZIWIG")
    print("knowledge graphs", res)

    await graph_db.finalize()


if __name__ == "__main__":
    asyncio.run(main())
