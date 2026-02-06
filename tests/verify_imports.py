import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_imports():
    print("Testing imports...")
    try:
        print("Importing lightrag_server...")
        from lightrag.api import lightrag_server
        print("SUCCESS: lightrag_server imported.")
        
        print("Importing rag_manager...")
        from lightrag.api.rag_manager import rag_manager
        print("SUCCESS: rag_manager imported.")
        
        print("Importing routers...")
        from lightrag.api.routers import tenant_document_routes
        from lightrag.api.routers import tenant_query_routes
        from lightrag.api.routers import chat_routes
        from lightrag.api.routers import tenant_auth_routes
        from lightrag.api.routers import tenant_graph_routes
        print("SUCCESS: All routers imported.")
        
    except ImportError as e:
        print(f"FAILURE: ImportError: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"FAILURE: Exception: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_imports()
