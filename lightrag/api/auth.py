from datetime import datetime, timedelta, timezone

import jwt
from dotenv import load_dotenv
from fastapi import HTTPException, status
from pydantic import BaseModel

from .config import global_args

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path='.env', override=False)


class TokenPayload(BaseModel):
    sub: str  # Username
    exp: datetime  # Expiration time
    role: str = 'user'  # User role, default is regular user
    metadata: dict = {}  # Additional metadata


class AuthHandler:
    def __init__(self):
        self.secret = global_args.token_secret
        self.algorithm = global_args.jwt_algorithm
        self.expire_hours = global_args.token_expire_hours
        self.guest_expire_hours = global_args.guest_token_expire_hours
        self.accounts = {}
        auth_accounts = global_args.auth_accounts
        if auth_accounts:
            for account in auth_accounts.split(','):
                username, password = account.split(':', 1)
                self.accounts[username] = password

    def create_token(
        self,
        username: str,
        role: str = 'user',
        custom_expire_hours: int | None = None,
        metadata: dict | None = None,
    ) -> str:
        """
        Create JWT token

        Args:
            username: Username
            role: User role, default is "user", guest is "guest"
            custom_expire_hours: Custom expiration time (hours), if None use default value
            metadata: Additional metadata

        Returns:
            str: Encoded JWT token
        """
        # Choose default expiration time based on role
        if custom_expire_hours is None:
            expire_hours = self.guest_expire_hours if role == 'guest' else self.expire_hours
        else:
            expire_hours = custom_expire_hours

        expire = datetime.now(timezone.utc) + timedelta(hours=expire_hours)

        # Create payload
        payload = TokenPayload(sub=username, exp=expire, role=role, metadata=metadata or {})

        return jwt.encode(payload.model_dump(), self.secret, algorithm=self.algorithm)

    def validate_token(self, token: str) -> dict:
        """
        Validate JWT token

        Args:
            token: JWT token

        Returns:
            dict: Dictionary containing user information

        Raises:
            HTTPException: If token is invalid or expired
        """
        try:
            payload = jwt.decode(token, self.secret, algorithms=[self.algorithm])
            expire_timestamp = payload['exp']
            expire_time = datetime.fromtimestamp(expire_timestamp, timezone.utc)

            if datetime.now(timezone.utc) > expire_time:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Token expired')

            # Return complete payload instead of just username
            return {
                'username': payload['sub'],
                'role': payload.get('role', 'user'),
                'metadata': payload.get('metadata', {}),
                'exp': expire_time,
            }
        except jwt.PyJWTError as e:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Invalid token') from e


auth_handler = AuthHandler()