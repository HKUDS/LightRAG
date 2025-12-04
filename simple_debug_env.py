import os
from dotenv import load_dotenv

print(f"Before load_dotenv: EMBEDDING_BINDING={os.getenv('EMBEDDING_BINDING')}")
load_dotenv(dotenv_path=".env", override=False)
print(f"After load_dotenv: EMBEDDING_BINDING={os.getenv('EMBEDDING_BINDING')}")
