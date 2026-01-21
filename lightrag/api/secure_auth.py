from datetime import datetime, timedelta
from typing import Optional

import jwt
from dotenv import load_dotenv
from fastapi import HTTPException, status
from pydantic import BaseModel

from .config import global_args
from . import db

# user the .env that is inside the current folder
load_dotenv(dotenv_path=".env", override=False)

class TokenPayload(BaseModel):
    sub: str  # Username
    user_id: str # User ID
    org_id: str # Organization ID (Workspace)
    exp: datetime  # Expiration time
    role: str = "user"  # User role
    metadata: dict = {}

class SecureAuthHandler:
    def __init__(self):
        self.secret = global_args.token_secret
        self.algorithm = global_args.jwt_algorithm
        self.expire_hours = global_args.token_expire_hours
        self.guest_expire_hours = global_args.guest_token_expire_hours
        
    def authenticate_user(self, username, password):
        # 1. Try DB
        user = db.get_user_by_username(username)
        if user:
            if db.verify_password(password, user["password_hash"]):
                return user
        return None

    def create_token(
        self,
        username: str,
        user_id: str,
        org_id: str,
        role: str = "user",
        custom_expire_hours: int = None,
        metadata: dict = None,
    ) -> str:
        """
        Create JWT token for multi-tenant auth
        """
        if custom_expire_hours is None:
            if role == "guest":
                expire_hours = self.guest_expire_hours
            else:
                expire_hours = self.expire_hours
        else:
            expire_hours = custom_expire_hours

        expire = datetime.utcnow() + timedelta(hours=expire_hours)

        payload = TokenPayload(
            sub=username, 
            user_id=user_id,
            org_id=org_id,
            exp=expire, 
            role=role, 
            metadata=metadata or {}
        )

        return jwt.encode(payload.dict(), self.secret, algorithm=self.algorithm)

    def validate_token(self, token: str) -> dict:
        try:
            payload = jwt.decode(token, self.secret, algorithms=[self.algorithm])
            expire_timestamp = payload["exp"]
            expire_time = datetime.utcfromtimestamp(expire_timestamp)

            if datetime.utcnow() > expire_time:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired"
                )

            return {
                "username": payload["sub"],
                "user_id": payload.get("user_id"),
                "org_id": payload.get("org_id"),
                "role": payload.get("role", "user"),
                "metadata": payload.get("metadata", {}),
                "exp": expire_time,
            }
        except jwt.PyJWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
            )

secure_auth_handler = SecureAuthHandler()
