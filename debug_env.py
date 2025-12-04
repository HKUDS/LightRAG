import os
from dotenv import load_dotenv
from lightrag.utils import get_env_value

print(f"Before load_dotenv: EMBEDDING_BINDING={os.getenv('EMBEDDING_BINDING')}")
load_dotenv(dotenv_path=".env", override=False)
print(f"After load_dotenv: EMBEDDING_BINDING={os.getenv('EMBEDDING_BINDING')}")

val = get_env_value("EMBEDDING_BINDING", "ollama")
print(f"get_env_value: {val}")
