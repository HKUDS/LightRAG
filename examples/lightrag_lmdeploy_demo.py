import os

from lightrag import LightRAG, QueryParam
from lightrag.llm import lmdeploy_model_if_cache, hf_embedding
from lightrag.utils import EmbeddingFunc
from transformers import AutoModel, AutoTokenizer

WORKING_DIR = "./dickens"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def lmdeploy_model_complete(
    prompt=None, system_prompt=None, history_messages=[], **kwargs
) -> str:
    model_name = kwargs["hashing_kv"].global_config["llm_model_name"]
    return await lmdeploy_model_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        ## please specify chat_template if your local path does not follow original HF file name,
        ## or model_name is a pytorch model on huggingface.co,
        ## you can refer to https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/model.py
        ## for a list of chat_template available in lmdeploy.
        chat_template="llama3",
        # model_format ='awq', # if you are using awq quantization model.
        # quant_policy=8, # if you want to use online kv cache, 4=kv int4, 8=kv int8.
        **kwargs,
    )


rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=lmdeploy_model_complete,
    llm_model_name="meta-llama/Llama-3.1-8B-Instruct",  # please use definite path for local model
    embedding_func=EmbeddingFunc(
        embedding_dim=384,
        max_token_size=5000,
        func=lambda texts: hf_embedding(
            texts,
            tokenizer=AutoTokenizer.from_pretrained(
                "sentence-transformers/all-MiniLM-L6-v2"
            ),
            embed_model=AutoModel.from_pretrained(
                "sentence-transformers/all-MiniLM-L6-v2"
            ),
        ),
    ),
)


with open("./book.txt", "r", encoding="utf-8") as f:
    rag.insert(f.read())

# Perform naive search
print(
    rag.query("What are the top themes in this story?", param=QueryParam(mode="naive"))
)

# Perform local search
print(
    rag.query("What are the top themes in this story?", param=QueryParam(mode="local"))
)

# Perform global search
print(
    rag.query("What are the top themes in this story?", param=QueryParam(mode="global"))
)

# Perform hybrid search
print(
    rag.query("What are the top themes in this story?", param=QueryParam(mode="hybrid"))
)
