import os
import logging
import numpy as np

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from openai import AzureOpenAI
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from lightrag.kg.faiss_impl import FaissVectorDBStorage

# Configure Logging
logging.basicConfig(level=logging.INFO)

# Load environment variables from .env file
load_dotenv()
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

async def llm_model_func(
    prompt,
    system_prompt=None,
    history_messages=[],
    keyword_extraction=False,
    **kwargs
) -> str:
    
    # Create a client for AzureOpenAI
    client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )

    # Build the messages list for the conversation
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    # Call the LLM
    chat_completion = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=messages,
        temperature=kwargs.get("temperature", 0),
        top_p=kwargs.get("top_p", 1),
        n=kwargs.get("n", 1),
    )

    return chat_completion.choices[0].message.content


async def embedding_func(texts: list[str]) -> np.ndarray:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings

def main():
    
    WORKING_DIR = "./dickens"

    # Initialize LightRAG with the LLM model function and embedding function
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=384,
            max_token_size=8192,
            func=embedding_func,
        ),
        vector_storage="FaissVectorDBStorage",
        vector_db_storage_cls_kwargs={
            "cosine_better_than_threshold": 0.3  # Your desired threshold
        }
    )

    # Insert the custom chunks into LightRAG
    book1 = open("./book_1.txt", encoding="utf-8")
    book2 = open("./book_2.txt", encoding="utf-8")

    rag.insert([book1.read(), book2.read()])

    query_text = "What are the main themes?"

    print("Result (Naive):")
    print(rag.query(query_text, param=QueryParam(mode="naive")))

    print("\nResult (Local):")
    print(rag.query(query_text, param=QueryParam(mode="local")))

    print("\nResult (Global):")
    print(rag.query(query_text, param=QueryParam(mode="global")))

    print("\nResult (Hybrid):")
    print(rag.query(query_text, param=QueryParam(mode="hybrid")))


if __name__ == "__main__":
    main()