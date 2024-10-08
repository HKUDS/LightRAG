import os
import sys

from lightrag import LightRAG

os.environ["OPENAI_API_KEY"] = ""

WORKING_DIR = ""

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = LightRAG(working_dir=WORKING_DIR)

with open('./text.txt', 'r') as f:
    text = f.read()

rag.insert(text)