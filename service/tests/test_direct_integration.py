import sys
import os
import asyncio
import uuid
from unittest.mock import MagicMock, AsyncMock

# Set SQLite for testing
db_path = "./test.db"
if os.path.exists(db_path):
    os.remove(db_path)
os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "service"))

# Mocking removed to allow real imports

# Now import the modified router
from lightrag.api.routers.query_routes import create_query_routes, QueryRequest

# Import service DB to check if records are created
from app.core.database import SessionLocal, engine, Base
from app.models.models import ChatMessage, ChatSession

# Create tables
Base.metadata.create_all(bind=engine)

async def test_direct_integration():
    print("Testing Direct Integration...")
    
    # Mock RAG instance
    mock_rag = MagicMock()
    mock_rag.aquery_llm = AsyncMock(return_value={
        "llm_response": {"content": "This is a mocked response."},
        "data": {"references": [{"reference_id": "1", "file_path": "doc1.txt"}]}
    })
    
    # Create router (this registers the endpoints but we'll call the function directly for testing)
    # We need to access the function decorated by @router.post("/query")
    # Since we can't easily get the route function from the router object without starting FastAPI,
    # we will inspect the router.routes
    
    create_query_routes(mock_rag)
    
    from lightrag.api.routers.query_routes import router
    
    # Find the query_text function
    query_route = next(r for r in router.routes if r.path == "/query")
    query_func = query_route.endpoint
    
    # Prepare Request
    request = QueryRequest(
        query="Test Query Direct",
        mode="hybrid",
        session_id=None # Should create new session
    )
    
    # Call the endpoint function directly
    print("Calling query_text...")
    response = await query_func(request)
    print("Response received:", response)
    
    # Verify DB
    db = SessionLocal()
    messages = db.query(ChatMessage).all()
    print(f"Total messages: {len(messages)}")
    for msg in messages:
        print(f"Msg: {msg.content} ({msg.role}) at {msg.created_at}")
    
    last_message = db.query(ChatMessage).filter(ChatMessage.role == "assistant").order_by(ChatMessage.created_at.desc()).first()
    
    if last_message:
        print(f"Last Assistant Message: {last_message.content}")
        assert last_message.content == "This is a mocked response."
        assert last_message.role == "assistant"
        
        # Check user message
        user_msg = db.query(ChatMessage).filter(ChatMessage.session_id == last_message.session_id, ChatMessage.role == "user").first()
        assert user_msg.content == "Test Query Direct"
        print("Verification Successful: History logged to DB.")
    else:
        print("Verification Failed: No message found in DB.")
    
    db.close()

async def test_stream_integration():
    print("\nTesting Stream Integration...")
    
    # Mock RAG instance for streaming
    mock_rag = MagicMock()
    
    # Mock response iterator
    async def response_iterator():
        yield "Chunk 1 "
        yield "Chunk 2"
    
    mock_rag.aquery_llm = AsyncMock(return_value={
        "llm_response": {
            "is_streaming": True,
            "response_iterator": response_iterator()
        },
        "data": {"references": [{"reference_id": "2", "file_path": "doc2.txt"}]}
    })
    
    from lightrag.api.routers.query_routes import router
    router.routes = [] # Clear existing routes to avoid conflict
    create_query_routes(mock_rag)
    
    # Find the query_text_stream function
    stream_route = next(r for r in router.routes if r.path == "/query/stream")
    stream_func = stream_route.endpoint
    
    # Prepare Request
    request = QueryRequest(
        query="Test Stream Direct",
        mode="hybrid",
        stream=True,
        session_id=None
    )
    
    # Call the endpoint
    print("Calling query_text_stream...")
    response = await stream_func(request)
    
    # Consume the stream
    content = ""
    async for chunk in response.body_iterator:
        print(f"Chunk received: {chunk}")
        content += chunk
        
    print("Stream finished.")
    
    # Verify DB
    db = SessionLocal()
    # Check for the new message
    # We expect "Chunk 1 Chunk 2" as content
    # Note: The chunk in body_iterator is NDJSON string, e.g. '{"response": "Chunk 1 "}\n'
    # But the DB should contain the parsed content.
    
    last_message = db.query(ChatMessage).filter(ChatMessage.role == "assistant", ChatMessage.content == "Chunk 1 Chunk 2").first()
    
    if last_message:
        print(f"Stream Message in DB: {last_message.content}")
        assert last_message.content == "Chunk 1 Chunk 2"
        print("Verification Successful: Stream History logged to DB.")
    else:
        print("Verification Failed: Stream message not found in DB.")
        # Print all to debug
        messages = db.query(ChatMessage).all()
        for msg in messages:
            print(f"Msg: {msg.content} ({msg.role})")
            
    db.close()

if __name__ == "__main__":
    asyncio.run(test_direct_integration())
    asyncio.run(test_stream_integration())
