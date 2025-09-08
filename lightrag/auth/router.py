from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from typing import Any

from .models.user import UserCreate, UserResponse
from .service import AuthService, oauth2_scheme

router = APIRouter(prefix="/auth", tags=["authentication"])

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserCreate) -> Any:
    """Register a new user."""
    try:
        user = AuthService.create_user(user_data)
        return user
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()) -> dict:
    """OAuth2 compatible token login, get an access token for future requests."""
    return await AuthService.login_for_access_token(form_data.username, form_data.password)

@router.get("/me", response_model=UserResponse)
async def read_users_me(current_user: UserResponse = Depends(AuthService.get_current_user)) -> Any:
    """Get current user information."""
    return current_user

@router.post("/logout")
async def logout(token: str = Depends(oauth2_scheme)) -> dict:
    """Logout user (invalidate token)."""
    # In a real application, you would add the token to a blacklist
    return {"message": "Successfully logged out"}
