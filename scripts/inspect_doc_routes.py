from fastapi import FastAPI
from lightrag.api.routers.document_routes import create_document_routes
from lightrag.lightrag import LightRAG
from unittest.mock import AsyncMock

# Create a simple app with document routes
app = FastAPI()

class DummyRagManager:
    def __init__(self, mapping):
        self.mapping = mapping
    async def get_rag_instance(self, tenant_id, kb_id, user_id=None):
        return self.mapping.get(tenant_id)

mock_rag_instances = {"tenant-a": AsyncMock(spec=LightRAG), "tenant-b": AsyncMock(spec=LightRAG)}

dummy_rag = AsyncMock(spec=LightRAG)
dummy_doc_manager = AsyncMock()
rag_manager = DummyRagManager(mock_rag_instances)

doc_router = create_document_routes(dummy_rag, dummy_doc_manager, rag_manager=rag_manager)
app.include_router(doc_router, prefix="/api")

# Print route info
for route in app.routes:
    if hasattr(route, 'methods') and 'POST' in route.methods and '/api/documents/text' in route.path:
        print('Path:', route.path)
        print('Name:', route.name)
        print('Endpoint signature:', route.dependant)
        print('Endpoint params:', [p.name for p in route.dependant.path_params + route.dependant.query_params + route.dependant.header_params + route.dependant.cookie_params + route.dependant.body_params + route.dependant.dependency_params])

print('\nAll routes:')
for r in app.routes:
    print(r.path)
