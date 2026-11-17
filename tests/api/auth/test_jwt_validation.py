"""Offline guards that forged or tampered JWTs are rejected after a PyJWT upgrade."""

import base64
import importlib
import json
import sys
from types import SimpleNamespace

import jwt
import pytest
from fastapi import HTTPException

_SECRET = "test-jwt-secret-that-is-at-least-32-bytes"


@pytest.fixture
def handler(monkeypatch):
    original_argv = sys.argv[:]
    sys.argv = [sys.argv[0]]
    try:
        config = importlib.import_module("lightrag.api.config")
    finally:
        sys.argv = original_argv

    monkeypatch.setattr(
        config,
        "global_args",
        SimpleNamespace(
            token_secret=_SECRET,
            jwt_algorithm="HS256",
            token_expire_hours=48,
            guest_token_expire_hours=24,
            auth_accounts="admin:admin_pass",
        ),
    )
    sys.modules.pop("lightrag.api.auth", None)
    auth = importlib.import_module("lightrag.api.auth")
    yield auth.AuthHandler()
    sys.modules.pop("lightrag.api.auth", None)


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _assert_rejected(handler, token: str) -> None:
    with pytest.raises(HTTPException) as exc_info:
        handler.validate_token(token)
    assert exc_info.value.status_code == 401


@pytest.mark.offline
def test_valid_token_round_trips(handler):
    token = handler.create_token("alice", role="user")

    info = handler.validate_token(token)

    assert info["username"] == "alice"
    assert info["role"] == "user"


@pytest.mark.offline
def test_tampered_signature_is_rejected(handler):
    header, payload, signature = handler.create_token("alice").split(".")
    flipped = ("A" if signature[0] != "A" else "B") + signature[1:]

    _assert_rejected(handler, ".".join([header, payload, flipped]))


@pytest.mark.offline
def test_token_signed_with_other_secret_is_rejected(handler):
    forged = jwt.encode(
        {"sub": "alice", "role": "user", "exp": 4102444800},
        "some-other-secret-that-is-32-bytes-long",
        algorithm="HS256",
    )

    _assert_rejected(handler, forged)


@pytest.mark.offline
def test_unsigned_alg_none_token_is_rejected(handler):
    header = _b64(json.dumps({"alg": "none", "typ": "JWT"}).encode())
    payload = _b64(
        json.dumps({"sub": "admin", "role": "user", "exp": 4102444800}).encode()
    )

    _assert_rejected(handler, f"{header}.{payload}.")


@pytest.mark.offline
def test_swapped_payload_with_old_signature_is_rejected(handler):
    header, _, signature = handler.create_token("alice").split(".")
    forged_payload = _b64(
        json.dumps({"sub": "admin", "role": "user", "exp": 4102444800}).encode()
    )

    _assert_rejected(handler, ".".join([header, forged_payload, signature]))
