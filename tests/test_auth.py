import importlib
import sys
from types import SimpleNamespace

import bcrypt
import pytest

from lightrag.api.passwords import BCRYPT_PASSWORD_PREFIX, hash_password
from lightrag.tools.hash_password import main as hash_password_main


def import_real_api_module(module_name: str):
    sys.modules.pop(module_name, None)

    package_name, _, child_name = module_name.rpartition(".")
    package = sys.modules.get(package_name)
    if package is not None and hasattr(package, child_name):
        delattr(package, child_name)

    return importlib.import_module(module_name)


@pytest.fixture
def auth_module(monkeypatch):
    config = import_real_api_module("lightrag.api.config")

    mock_global_args = SimpleNamespace(
        token_secret="lightrag-jwt-default-secret-key!",
        jwt_algorithm="HS256",
        token_expire_hours=48,
        guest_token_expire_hours=24,
        auth_accounts="admin:admin_pass",
    )

    monkeypatch.setattr(config, "global_args", mock_global_args)

    module = import_real_api_module("lightrag.api.auth")
    module = importlib.reload(module)
    yield module
    sys.modules.pop("lightrag.api.auth", None)


def build_bcrypt_value(password: str) -> str:
    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    return f"{BCRYPT_PASSWORD_PREFIX}{hashed}"


def test_verify_plaintext_password(auth_module):
    handler = auth_module.AuthHandler()
    handler.accounts = {"admin": "admin_pass"}

    assert handler.verify_password("admin", "admin_pass")
    assert not handler.verify_password("admin", "wrong_pass")


def test_verify_prefixed_bcrypt_password(auth_module):
    handler = auth_module.AuthHandler()
    handler.accounts = {"user": build_bcrypt_value("user_pass")}

    assert handler.verify_password("user", "user_pass")
    assert not handler.verify_password("user", "wrong_pass")


def test_plaintext_password_with_bcrypt_prefix_stays_plaintext(auth_module):
    handler = auth_module.AuthHandler()
    handler.accounts = {"user": "$2b$not-a-real-hash"}

    assert handler.verify_password("user", "$2b$not-a-real-hash")
    assert not handler.verify_password("user", "anything-else")


def test_invalid_auth_accounts_raises(monkeypatch):
    config = import_real_api_module("lightrag.api.config")

    mock_global_args = SimpleNamespace(
        token_secret="lightrag-jwt-default-secret-key!",
        jwt_algorithm="HS256",
        token_expire_hours=48,
        guest_token_expire_hours=24,
        auth_accounts="admin",
    )

    monkeypatch.setattr(config, "global_args", mock_global_args)

    with pytest.raises(ValueError, match="AUTH_ACCOUNTS must use"):
        import_real_api_module("lightrag.api.auth")

    sys.modules.pop("lightrag.api.auth", None)


def test_hash_password_returns_prefixed_value(auth_module):
    hashed = hash_password("new_password")

    assert hashed.startswith(BCRYPT_PASSWORD_PREFIX)
    raw_hash = hashed[len(BCRYPT_PASSWORD_PREFIX) :]
    assert bcrypt.checkpw("new_password".encode("utf-8"), raw_hash.encode("utf-8"))


def test_hash_password_cli_outputs_auth_accounts_entry(capsys):
    exit_code = hash_password_main(["--username", "admin", "secret"])

    assert exit_code == 0
    output = capsys.readouterr().out.strip()
    username, hashed = output.split(":", 1)
    assert username == "admin"
    assert hashed.startswith(BCRYPT_PASSWORD_PREFIX)
    raw_hash = hashed[len(BCRYPT_PASSWORD_PREFIX) :]
    assert bcrypt.checkpw("secret".encode("utf-8"), raw_hash.encode("utf-8"))
