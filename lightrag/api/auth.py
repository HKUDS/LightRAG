# auth.py
from datetime import datetime, timedelta, timezone
import jwt
from dotenv import load_dotenv
from fastapi import HTTPException, status
from pydantic import BaseModel
from typing import Optional, Dict, Tuple

from .config import global_args
from .database import SessionLocal

load_dotenv(dotenv_path=".env", override=False)

from sqlalchemy import select
from sqlalchemy.orm import Session
from .models import User

def get_user_id_by_username(db: Session, username: str) -> str | None:
    return db.execute(
        select(User.id).where(User.username == username)
    ).scalar_one_or_none()

class TokenPayload(BaseModel):
    sub: str                 # Username
    exp: datetime            # Expiration time (UTC)
    role: str = "user"       # Role
    uid: Optional[str] = None  # <-- DB user_id
    metadata: dict = {}        # Additional metadata

class AuthHandler:
    def __init__(self):
        self.secret = global_args.token_secret
        self.algorithm = global_args.jwt_algorithm
        self.expire_hours = global_args.token_expire_hours
        self.guest_expire_hours = global_args.guest_token_expire_hours

        # env-based fallback accounts (optional)
        self.accounts: Dict[str, str] = {}
        auth_accounts = global_args.auth_accounts
        if auth_accounts:
            for account in auth_accounts.split(","):
                parts = account.split(":", 2)  # username:password[:user_id]  (user_id ignored now)
                username = parts[0]
                password = parts[1] if len(parts) > 1 else ""
                self.accounts[username] = password

        # tiny in-memory cache for username -> (user_id, expires_at)
        self._uid_cache: Dict[str, Tuple[str, float]] = {}
        self._uid_ttl_seconds = 300  # 5 minutes

    def _now_utc(self) -> datetime:
        return datetime.now(timezone.utc)

    def _cache_get_uid(self, username: str) -> Optional[str]:
        rec = self._uid_cache.get(username)
        if not rec:
            return None
        uid, exp_ts = rec
        if self._now_utc().timestamp() > exp_ts:
            self._uid_cache.pop(username, None)
            return None
        return uid

    def _cache_put_uid(self, username: str, uid: str) -> None:
        self._uid_cache[username] = (uid, self._now_utc().timestamp() + self._uid_ttl_seconds)

    def _lookup_user_id(self, username: str) -> Optional[str]:
        # 1) try cache
        cached = self._cache_get_uid(username)
        if cached:
            return cached

        # 2) DB lookup
        with SessionLocal() as db:
            uid = get_user_id_by_username(db, username)

        if uid:
            self._cache_put_uid(username, uid)
        return uid

    def create_token(
        self,
        username: str,
        role: str = "user",
        custom_expire_hours: Optional[int] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        if custom_expire_hours is None:
            expire_hours = self.guest_expire_hours if role == "guest" else self.expire_hours
        else:
            expire_hours = custom_expire_hours

        expire = self._now_utc() + timedelta(hours=expire_hours)

        # Pull user_id from DB (guest users may not exist—allow None if you want)
        uid = self._lookup_user_id(username)
        if role != "guest" and not uid:
            # For non-guest flows, enforce that the user actually exists in DB
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unknown user"
            )

        payload = TokenPayload(
            sub=username,
            exp=expire,
            role=role,
            uid=uid,                     # <-- embed user_id
            metadata=metadata or {},
        )

        # PyJWT accepts aware datetimes for 'exp'
        return jwt.encode(payload.dict(), self.secret, algorithm=self.algorithm)

    def validate_token(self, token: str) -> dict:
        try:
            payload = jwt.decode(token, self.secret, algorithms=[self.algorithm])
            # 'exp' is validated by PyJWT; if you want manual check, you can still read it:
            exp_ts = payload.get("exp")
            if exp_ts is None:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

            return {
                "username": payload.get("sub"),
                "user_id": payload.get("uid"),  # <-- expose user_id
                "role": payload.get("role", "user"),
                "metadata": payload.get("metadata", {}),
                "exp": datetime.fromtimestamp(exp_ts, tz=timezone.utc),
            }
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
        except jwt.PyJWTError:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

auth_handler = AuthHandler()
