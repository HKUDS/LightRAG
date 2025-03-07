import os
from datetime import datetime, timedelta
import jwt
from fastapi import HTTPException, status
from pydantic import BaseModel


class TokenPayload(BaseModel):
    sub: str
    exp: datetime


class AuthHandler:
    def __init__(self):
        self.secret = os.getenv("TOKEN_SECRET", "4f85ds4f56dsf46")
        self.algorithm = "HS256"
        self.expire_hours = int(os.getenv("TOKEN_EXPIRE_HOURS", 4))

    def create_token(self, username: str) -> str:
        expire = datetime.utcnow() + timedelta(hours=self.expire_hours)
        payload = TokenPayload(sub=username, exp=expire)
        return jwt.encode(payload.dict(), self.secret, algorithm=self.algorithm)

    def validate_token(self, token: str) -> str:
        try:
            payload = jwt.decode(token, self.secret, algorithms=[self.algorithm])
            expire_timestamp = payload["exp"]
            expire_time = datetime.utcfromtimestamp(expire_timestamp)

            if datetime.utcnow() > expire_time:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired"
                )
            return payload["sub"]
        except jwt.PyJWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
            )


auth_handler = AuthHandler()
