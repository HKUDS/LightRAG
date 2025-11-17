==========controled ingestion in batches=============
import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
import numpy as np
from dotenv import load_dotenv
import logging
from openai import AzureOpenAI
import time
start_time = time.time() 
logging.basicConfig(level=logging.INFO)

load_dotenv()

AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
AZURE_EMBEDDING_API_VERSION = os.getenv("AZURE_EMBEDDING_API_VERSION")

WORKING_DIR = "C:\\Users\\user\\testfolder" 

# The code below removes the working_dir and creates a new one!
# if os.path.exists(WORKING_DIR):
#     import shutil

#     shutil.rmtree(WORKING_DIR)

# os.mkdir(WORKING_DIR)


async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    chat_completion = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,  # model = "deployment_name".
        messages=messages,
        temperature=kwargs.get("temperature", 0),
        top_p=kwargs.get("top_p", 1),
        n=kwargs.get("n", 1),
    )
    return chat_completion.choices[0].message.content


async def embedding_func(texts: list[str]) -> np.ndarray:
    client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_EMBEDDING_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )
    embedding = client.embeddings.create(model=AZURE_EMBEDDING_DEPLOYMENT, input=texts)

    embeddings = [item.embedding for item in embedding.data]
    return np.array(embeddings)


async def test_funcs():
    result = await llm_model_func("How are you?")
    print("Response from llm_model_func: ", result)

    result = await embedding_func(["How are you?"])
    print("Result from embedding_func: ", result.shape)
    print("Embedding dimension: ", result.shape[1])


asyncio.run(test_funcs())

embedding_dimension = 1536

rag = LightRAG(
    working_dir=WORKING_DIR,
    addon_params={"insert_batch_size": 3},
    llm_model_func=llm_model_func,
    embedding_func=EmbeddingFunc(
        embedding_dim=embedding_dimension,
        max_token_size=8192,
        func=embedding_func,
    ),
)

folder_path = 'C:/Users/example/test/LightRAG/my_docs' # With os, this specification of the documents folder is not a problem.

def normalize_path(path):
    # Normalize the path
    normalized_path = os.path.normpath(path)
    # Replace backslashes with forward slashes
    return normalized_path.replace('\\', '/')

# Output file where we store filenames
output_file = 'processed.txt'
# The maximum number of files to process
batch_files = 5 

# Function to include document in vector-store
def process_doc(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return content

#     logging.info(f"Adding document: {file_path}")
#     rag.insert(content)

# Function to read existing filenames from the output file
def read_existing_files(output_file):
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            return set(line.strip() for line in f.readlines())
    return set()

input_docs = []
# Check if the folder exists
if os.path.exists(folder_path) and os.path.isdir(folder_path):
    # Read existing filenames from the output file
    existing_files = read_existing_files(output_file)
    # Open the output


==========Querier==============================================

import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
import numpy as np
from dotenv import load_dotenv
import logging
from openai import AzureOpenAI
import time
from multiprocessing import freeze_support

# Configure logging
logging.basicConfig(level=logging.INFO)

def main():
    start_time = time.time()
    load_dotenv()

    # Version of the script
    VERSION = "0.2"

    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
    AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

    AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
    AZURE_EMBEDDING_API_VERSION = os.getenv("AZURE_EMBEDDING_API_VERSION")

    WORKING_DIR = "C:/Users/user/testfolder" 

    async def llm_model_func(
        prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
    ) -> str:
        client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
        )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history_messages:
            messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})

        chat_completion = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=messages,
            temperature=kwargs.get("temperature", 0.5),
            max_tokens=kwargs.get("max_tokens", 1500)
        )

        return chat_completion.choices[0].message.content

    async def embedding_func(texts: list[str]) -> np.ndarray:
        client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_EMBEDDING_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
        )
        embedding = client.embeddings.create(model=AZURE_EMBEDDING_DEPLOYMENT, input=texts)

        embeddings = [item.embedding for item in embedding.data]
        return np.array(embeddings)

    # async def test_funcs():
    #     try:
    #         result_llm = await llm_model_func("How are you?", system_prompt="Act as a friendly assistant.")
    #         print("Response from llm_model_func: ", result_llm)

    #         result_embedding = await embedding_func(["How are you?"])
    #         print("Result from embedding_func: ", result_embedding.shape)
    #         print("Embedding dimension: ", result_embedding.shape[1])
    #     except Exception as e:
    #         print(f"An error occurred in test_funcs: {e}")

    # asyncio.run(test_funcs())

    embedding_dimension = 1536

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dimension,
            max_token_size=8192,
            func=embedding_func,
        ),
    )

    def query_rag(question):
        """
        Execute a query in hybrid mode.
        """
        try:
            start_time = time.time()
            response = rag.query(
                question,
                param=QueryParam(mode="hybrid")  # Set hybrid mode
            )
            if response is None or not response.strip():
                print("No relevant answer found. Check if the database is correctly populated.")
                return

            print("\n--- Answer ---")
            print(response)
            print("\n")
            duration = time.time() - start_time  
            print(f"The answer took {duration} seconds.")      
        except Exception as e:
            print(f"An error occurred while executing the query: {e}")

    print("LightRAG - Interactive Questioning")
    print(f"Version: {VERSION}")
    print("Type 'exit' to terminate the program.\n")

    while True:
        try:
            question = input("Ask your question: ")

            if question.lower() in ["exit", "quit"]:
                print("Program terminated.")
                break
            query_rag(question)    
        except:
            print("Error answering the question")

if __name__ == '__main__':
   
