import os
import sys
sys.path.append('xxx/xxx/LightRAG')

from lightrag import LightRAG, QueryParam

os.environ["OPENAI_API_KEY"] = ""

WORKING_DIR = ""

rag = LightRAG(working_dir=WORKING_DIR)

mode = 'global'
query_param = QueryParam(mode=mode)

result, _ = rag.query("", param=query_param)
print(result)