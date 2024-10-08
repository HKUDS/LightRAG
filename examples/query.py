import os
import sys

from lightrag import LightRAG, QueryParam

os.environ["OPENAI_API_KEY"] = ""

WORKING_DIR = ""

rag = LightRAG(working_dir=WORKING_DIR)

mode = 'global'
query_param = QueryParam(mode=mode)

result = rag.query("", param=query_param)
print(result)