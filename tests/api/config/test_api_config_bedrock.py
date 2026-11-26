import sys

import pytest

from lightrag.api.config import parse_args


pytestmark = pytest.mark.offline


def _clear_bedrock_auth_env(monkeypatch):
    for key in (
        "LLM_BINDING",
        "EMBEDDING_BINDING",
        "QUERY_LLM_BINDING",
        "QUERY_LLM_BINDING_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_BEARER_TOKEN_BEDROCK",
        "AWS_ROLE_ARN",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
        "QUERY_AWS_ACCESS_KEY_ID",
        "QUERY_AWS_SECRET_ACCESS_KEY",
    ):
        monkeypatch.delenv(key, raising=False)


def test_bedrock_env_binding_alias_is_normalized(monkeypatch):
    _clear_bedrock_auth_env(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    monkeypatch.setenv("LLM_BINDING", "aws_bedrock")
    monkeypatch.setenv("EMBEDDING_BINDING", "aws_bedrock")
    # Also the case that a bearer token alone is accepted.
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "absk-test")

    args = parse_args()

    assert args.llm_binding == "bedrock"
    assert args.embedding_binding == "bedrock"


def test_bedrock_cli_binding_alias_is_not_supported(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["lightrag-server", "--llm-binding", "aws_bedrock"]
    )

    with pytest.raises(SystemExit):
        parse_args()


def test_bedrock_role_env_binding_alias_is_normalized(monkeypatch):
    _clear_bedrock_auth_env(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    monkeypatch.setenv("QUERY_LLM_BINDING", "aws_bedrock")
    monkeypatch.setenv("QUERY_LLM_MODEL", "us.amazon.nova-lite-v1:0")
    monkeypatch.setenv("QUERY_AWS_REGION", "us-west-2")
    monkeypatch.setenv("QUERY_AWS_ACCESS_KEY_ID", "akid")
    monkeypatch.setenv("QUERY_AWS_SECRET_ACCESS_KEY", "secret")

    args = parse_args()

    assert args.query_llm_binding == "bedrock"
    assert args.query_aws_region == "us-west-2"


def test_bedrock_role_api_key_is_rejected(monkeypatch):
    _clear_bedrock_auth_env(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    monkeypatch.setenv("QUERY_LLM_BINDING", "bedrock")
    monkeypatch.setenv("QUERY_LLM_MODEL", "us.amazon.nova-lite-v1:0")
    monkeypatch.setenv("QUERY_LLM_BINDING_API_KEY", "absk-role")

    with pytest.raises(SystemExit, match="does not support QUERY_LLM_BINDING_API_KEY"):
        parse_args()


def test_bedrock_binding_accepts_default_aws_credential_provider_chain(monkeypatch):
    _clear_bedrock_auth_env(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    monkeypatch.setenv("LLM_BINDING", "bedrock")
    monkeypatch.setenv("EMBEDDING_BINDING", "ollama")
    monkeypatch.setenv(
        "AWS_ROLE_ARN", "arn:aws:iam::123456789012:role/lightrag-bedrock"
    )
    monkeypatch.setenv(
        "AWS_WEB_IDENTITY_TOKEN_FILE",
        "/var/run/secrets/eks.amazonaws.com/serviceaccount/token",
    )

    args = parse_args()

    assert args.llm_binding == "bedrock"


def test_bedrock_role_accepts_default_aws_credential_provider_chain(monkeypatch):
    _clear_bedrock_auth_env(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    monkeypatch.setenv("QUERY_LLM_BINDING", "bedrock")
    monkeypatch.setenv("QUERY_LLM_MODEL", "us.amazon.nova-lite-v1:0")
    monkeypatch.setenv(
        "AWS_ROLE_ARN", "arn:aws:iam::123456789012:role/lightrag-bedrock"
    )
    monkeypatch.setenv(
        "AWS_WEB_IDENTITY_TOKEN_FILE",
        "/var/run/secrets/eks.amazonaws.com/serviceaccount/token",
    )

    args = parse_args()

    assert args.query_llm_binding == "bedrock"


# A bearer token does not excuse a half-configured SigV4 pair.
@pytest.mark.parametrize("bearer_token", [None, "absk-test"])
def test_bedrock_binding_rejects_partial_sigv4_pair(monkeypatch, bearer_token):
    _clear_bedrock_auth_env(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    monkeypatch.setenv("LLM_BINDING", "bedrock")
    monkeypatch.setenv("EMBEDDING_BINDING", "ollama")
    if bearer_token:
        monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", bearer_token)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "akid")

    with pytest.raises(ValueError, match="Bedrock LLM binding requires"):
        parse_args()


def test_bedrock_binding_accepts_sigv4_pair(monkeypatch):
    _clear_bedrock_auth_env(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    monkeypatch.setenv("LLM_BINDING", "bedrock")
    monkeypatch.setenv("EMBEDDING_BINDING", "ollama")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "akid")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")

    args = parse_args()

    assert args.llm_binding == "bedrock"


@pytest.mark.parametrize("bearer_token", [None, "absk-test"])
def test_bedrock_role_requires_complete_role_sigv4_pair(monkeypatch, bearer_token):
    _clear_bedrock_auth_env(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    monkeypatch.setenv("QUERY_LLM_BINDING", "bedrock")
    monkeypatch.setenv("QUERY_LLM_MODEL", "us.amazon.nova-lite-v1:0")
    if bearer_token:
        monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", bearer_token)
    monkeypatch.setenv("QUERY_AWS_ACCESS_KEY_ID", "akid")

    with pytest.raises(ValueError, match="Bedrock role 'query' requires"):
        parse_args()
