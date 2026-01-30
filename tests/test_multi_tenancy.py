import sys
import os

# Patch sys.argv BEFORE importing lightrag
sys.argv = ["lightrag-server", "--working-dir", "./test_rag_data", "--llm-binding", "lollms", "--embedding-binding", "lollms"] 

# Set test environment vars BEFORE importing modules that use them
os.environ["LIGHTRAG_DB_PATH"] = "test_lightrag.db"
os.environ["LIGHTRAG_ADMIN_PASSWORD"] = "admin"

# Ensure we can import the app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from fastapi.testclient import TestClient

from lightrag.api.lightrag_server import create_app
from lightrag.api.config import global_args
# Use explicit alias to avoid namespace collisions
import lightrag.api.db as lightrag_db

@pytest.fixture(scope="module")
def client():
    # Helper to clean DB before test
    if os.path.exists("test_lightrag.db"):
        os.remove("test_lightrag.db")
    
    # Initialize DB (creates default org/admin)
    print(f"DEBUG: Initializing DB using {lightrag_db}")
    lightrag_db.init_db()
    
    # Manually create Second Org for testing
    with lightrag_db.get_db_cursor() as cur:
        # Check if exists first to avoid sqlite syntax issues if OR IGNORE not supported (it is standard though)
        cur.execute("INSERT OR IGNORE INTO organizations (id, name) VALUES (?, ?)", ("org_b", "Organization B"))
    
    # Create App
    app = create_app(global_args)
    
    with TestClient(app) as test_client:
        yield test_client
    
    # Cleanup
    if os.path.exists("test_lightrag.db"):
        os.remove("test_lightrag.db")

def test_auth_and_isolation(client):
    # 1. Register User A (Org Default)
    # Note: endpoint is /register
    # We need to register via API or DB? 
    # tenant_auth_routes has /register
    
    # Register User A
    resp = client.post("/register", json={
        "username": "user_a",
        "password": "password_a",
        "org_id": "org_default"
    })
    assert resp.status_code == 200
    token_a = resp.json()["access_token"]
    
    # Register User B (Org B)
    resp = client.post("/register", json={
        "username": "user_b", 
        "password": "password_b",
        "org_id": "org_b"
    })
    assert resp.status_code == 200
    token_b = resp.json()["access_token"]
    
    # 2. Verify Session Isolation via Chat Routes
    headers_a = {"Authorization": f"Bearer {token_a}"}
    headers_b = {"Authorization": f"Bearer {token_b}"}
    
    # User A creates a chat
    resp = client.post("/chats", json={"title": "Chat A"}, headers=headers_a)
    assert resp.status_code == 200
    chat_id_a = resp.json()["id"]
    
    # User B creates a chat
    resp = client.post("/chats", json={"title": "Chat B"}, headers=headers_b)
    assert resp.status_code == 200
    chat_id_b = resp.json()["id"]
    
    # 3. User A lists chats -> Should see Chat A, NOT Chat B
    resp = client.get("/chats", headers=headers_a)
    assert resp.status_code == 200
    chats_a = resp.json()
    ids_a = [c["id"] for c in chats_a]
    assert chat_id_a in ids_a
    assert chat_id_b not in ids_a
    
    # 4. User B lists chats -> Should see Chat B, NOT Chat A
    resp = client.get("/chats", headers=headers_b)
    assert resp.status_code == 200
    chats_b = resp.json()
    ids_b = [c["id"] for c in chats_b]
    assert chat_id_b in ids_b
    assert chat_id_a not in ids_b
    
    # 5. Access Control: User B tries to fetch User A's chat messages
    resp = client.get(f"/chats/{chat_id_a}/messages", headers=headers_b)
    assert resp.status_code == 404 # Should be Not Found (or Forbidden)
    
    print("\nSUCCESS: Multi-tenancy Isolation Verified!")

if __name__ == "__main__":
    # Allow running directly without pytest
    # But need to mock client fixture manually or simplified
    print("Please run with: pytest tests/test_multi_tenancy.py")
