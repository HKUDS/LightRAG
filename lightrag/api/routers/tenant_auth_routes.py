from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from .. import db
from ..secure_auth import secure_auth_handler

router = APIRouter(tags=["auth"])

class LoginRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    username: str
    password: str
    org_id: str = "org_default" # Default to default org for now

class TokenResponse(BaseModel):
    access_token: str
    token_type: str

@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    user = secure_auth_handler.authenticate_user(request.username, request.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = secure_auth_handler.create_token(
        username=user["username"],
        user_id=user["id"],
        org_id=user["org_id"],
        role=user["role"]
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/register", response_model=TokenResponse)
async def register(request: RegisterRequest):
    # Check if user exists
    if db.get_user_by_username(request.username):
        raise HTTPException(status_code=400, detail="Username already registered")
        
    organization = db.get_organization(request.org_id)
    if not organization:
         # Auto-create organization for multi-tenancy demo/bootstrap
         db.create_organization(request.org_id, f"Organization {request.org_id}")
         # raise HTTPException(status_code=400, detail="Organization does not exist")

    user = db.create_user(request.username, request.password, request.org_id)
    if not user:
         raise HTTPException(status_code=400, detail="User creation failed (username likely taken)")

    # Auto-login
    access_token = secure_auth_handler.create_token(
        username=request.username,
        user_id=user["id"],
        org_id=request.org_id,
        role="user"
    )
    
    return {"access_token": access_token, "token_type": "bearer"}
