import lightrag
from lightrag import QueryParam
import inspect

print("_____________")

for name, param in inspect.signature(QueryParam.__init__).parameters.items():
    print(f"{name}: default={param.default}, kind={param.kind}")

print("_____________")


print("LightRAG is loaded from:", lightrag.__file__)

print("_____________")

print(QueryParam.__annotations__)