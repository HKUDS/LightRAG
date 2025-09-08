from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field
from uuid import UUID, uuid4

class UserBase(BaseModel):
    """Base user model containing common fields."""
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    is_active: bool = True
    is_superuser: bool = False

class UserCreate(UserBase):
    """Model for creating a new user (includes password)."""
    password: str = Field(..., min_length=8)

class UserUpdate(BaseModel):
    """Model for updating user information."""
    email: Optional[EmailStr] = None
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    password: Optional[str] = Field(None, min_length=8)
    is_active: Optional[bool] = None
    is_superuser: Optional[bool] = None

class UserInDB(UserBase):
    """Database model for user."""
    id: UUID = Field(default_factory=uuid4)
    hashed_password: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        from_attributes = True
        json_encoders = {
            UUID: lambda v: str(v),
            datetime: lambda v: v.isoformat()
        }

class UserResponse(UserBase):
    """Response model for user data (excludes sensitive information)."""
    id: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
        json_encoders = {
            UUID: lambda v: str(v),
            datetime: lambda v: v.isoformat()
        }
