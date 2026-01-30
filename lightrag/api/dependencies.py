from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from typing import Annotated

from lightrag import LightRAG
from .secure_auth import secure_auth_handler
from .rag_manager import rag_manager

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

async def get_current_user_token(token: Annotated[str, Depends(oauth2_scheme)]):
    return secure_auth_handler.validate_token(token)

async def get_current_user(token_data: dict = Depends(get_current_user_token)):
    # In a real app we might fetch from DB to ensure user is still valid/active
    # For speed, we trust the JWT claims
    return token_data

async def get_current_rag(current_user: dict = Depends(get_current_user)) -> LightRAG:
    """
    Dependency to get the LightRAG instance for the current user's organization.
    """
    org_id = current_user.get("org_id", "default")
    if not org_id:
        # Fallback for legacy or admin-global? 
        # For strict multi-tenancy, every user MUST have an org_id.
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, 
            detail="User does not belong to an organization"
        )
    
    return await rag_manager.get_rag(workspace=org_id)
