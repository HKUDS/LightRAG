"""Optional JWT auth for FrameRAG API.

Enabled only when FRAMERAG_AUTH_ACCOUNTS env var is set.
Format: "user1:pass1,user2:pass2"

If not set, all endpoints are open (dev mode).
"""
from __future__ import annotations

import os
import time
from typing import Optional

from lightrag.utils import logger

try:
    import jwt
    _JWT_AVAILABLE = True
except ImportError:
    _JWT_AVAILABLE = False

try:
    from fastapi import HTTPException, Security, status
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False


_SECRET     = os.environ.get("FRAMERAG_TOKEN_SECRET", "framerag-dev-secret-change-me")
_ALGORITHM  = "HS256"
_EXPIRE_HRS = int(os.environ.get("FRAMERAG_TOKEN_EXPIRE_HOURS", "24"))

_bearer = HTTPBearer(auto_error=False) if _FASTAPI_AVAILABLE else None


def _parse_accounts() -> dict[str, str]:
    raw = os.environ.get("FRAMERAG_AUTH_ACCOUNTS", "")
    accounts: dict[str, str] = {}
    if not raw.strip():
        return accounts
    for pair in raw.split(","):
        pair = pair.strip()
        if ":" in pair:
            user, pwd = pair.split(":", 1)
            accounts[user.strip()] = pwd.strip()
    return accounts


_ACCOUNTS: dict[str, str] = _parse_accounts()
AUTH_ENABLED: bool = bool(_ACCOUNTS)


def create_token(username: str) -> str:
    if not _JWT_AVAILABLE:
        raise RuntimeError("PyJWT not installed: pip install PyJWT")
    payload = {
        "sub": username,
        "iat": int(time.time()),
        "exp": int(time.time()) + _EXPIRE_HRS * 3600,
    }
    return jwt.encode(payload, _SECRET, algorithm=_ALGORITHM)


def verify_token(token: str) -> str:
    """Verify JWT and return username. Raises HTTPException on failure."""
    if not _JWT_AVAILABLE:
        raise RuntimeError("PyJWT not installed")
    try:
        payload = jwt.decode(token, _SECRET, algorithms=[_ALGORITHM])
        return payload["sub"]
    except jwt.ExpiredSignatureError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token")


def verify_credentials(username: str, password: str) -> bool:
    return _ACCOUNTS.get(username) == password


async def require_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(_bearer),
) -> Optional[str]:
    """FastAPI dependency — returns username if auth enabled, None if open."""
    if not AUTH_ENABLED:
        return None
    if credentials is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing Bearer token")
    return verify_token(credentials.credentials)
