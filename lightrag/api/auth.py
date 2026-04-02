from datetime import datetime, timedelta
import secrets

import jwt
from dotenv import load_dotenv
from fastapi import HTTPException, status
from pydantic import BaseModel

from ..utils import logger
from .config import global_args
from .passwords import verify_password

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)


class TokenPayload(BaseModel):
    sub: str  # Username
    exp: datetime  # Expiration time
    role: str = "user"  # User role, default is regular user
    metadata: dict = {}  # Additional metadata


class AuthHandler:
    def __init__(self):
        auth_accounts = global_args.auth_accounts
        self.secret = global_args.token_secret
        if not self.secret:
            if auth_accounts:
                raise ValueError(
                    "TOKEN_SECRET must be explicitly set to a non-default value when AUTH_ACCOUNTS is configured."
                )
            self.secret = secrets.token_urlsafe(32)
            logger.info(
                "TOKEN_SECRET not set; generated an ephemeral secret for guest tokens because AUTH_ACCOUNTS is not configured."
            )
        self.algorithm = global_args.jwt_algorithm
        self.expire_hours = global_args.token_expire_hours
        self.guest_expire_hours = global_args.guest_token_expire_hours
        self.accounts = {}
        invalid_accounts = []
        if auth_accounts:
            for account in auth_accounts.split(","):
                try:
                    username, password = account.split(":", 1)
                    if not username or not password:
                        raise ValueError
                    self.accounts[username] = password
                except ValueError:
                    invalid_accounts.append(account)
        if invalid_accounts:
            invalid_entries = ", ".join(invalid_accounts)
            logger.error(f"Invalid account format in AUTH_ACCOUNTS: {invalid_entries}")
            raise ValueError(
                "AUTH_ACCOUNTS must use comma-separated user:password pairs."
            )

    def verify_password(self, username: str, plain_password: str) -> bool:
        """
        Verify password for a user. Supports explicit bcrypt values and plaintext.

        Args:
            username: Username to verify
            plain_password: Plaintext password to check

        Returns:
            bool: True if password is correct, False otherwise
        """
        if username not in self.accounts:
            return False

        stored_password = self.accounts[username]
        return verify_password(plain_password, stored_password)

    def create_token(
        self,
        username: str,
        role: str = "user",
        custom_expire_hours: int = None,
        metadata: dict = None,
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
            if role == "guest":
                expire_hours = self.guest_expire_hours
            else:
                expire_hours = self.expire_hours
        else:
            expire_hours = custom_expire_hours

        expire = datetime.utcnow() + timedelta(hours=expire_hours)

        # Create payload
        payload = TokenPayload(
            sub=username, exp=expire, role=role, metadata=metadata or {}
        )

        return jwt.encode(payload.dict(), self.secret, algorithm=self.algorithm)

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
            expire_timestamp = payload["exp"]
            expire_time = datetime.utcfromtimestamp(expire_timestamp)

            if datetime.utcnow() > expire_time:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired"
                )

            # Return complete payload instead of just username
            return {
                "username": payload["sub"],
                "role": payload.get("role", "user"),
                "metadata": payload.get("metadata", {}),
                "exp": expire_time,
            }
        except jwt.PyJWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
            )


auth_handler = AuthHandler()
