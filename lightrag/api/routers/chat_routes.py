from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from lightrag import LightRAG
from lightrag.api.utils_api import get_combined_auth_dependency
from datetime import datetime

chatRoute = APIRouter(
    prefix="/chat",
    tags=["chat"],
)

class SeedChatRequest(BaseModel):
    text: str = Field(..., description="The chat-style free text to process")

def create_chat_route(rag: LightRAG, api_key: str = None):
    """
    Augments the chatRoute with the provided RAG instance and authentication dependency.
    """
    # Create combined auth dependency for chat routes
    combined_auth = get_combined_auth_dependency(api_key)

    @chatRoute.post("/seed_chat", dependencies=[Depends(combined_auth)])
    async def seed_chat(request: SeedChatRequest, background_tasks: BackgroundTasks):
        """
        Endpoint to seed the knowledge graph using chat-style free text.
        For now, it processes the text and adds it to the RAG system.
        """
        try:
            # Remove leading and trailing whitespaces from the text
            id = f"user-root-{datetime.now().isoformat()}" 
            processed_text = request.text.strip()

            await rag.ainsert(processed_text,ids=id,file_paths=f"chat-{id}")

            return {"status": "success", "message": "Text successfully received and processing started."}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing text: {str(e)}")

    return chatRoute