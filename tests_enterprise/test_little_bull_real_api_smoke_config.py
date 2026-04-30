from __future__ import annotations

from tests_enterprise.test_little_bull_real_api_smoke import (
    NO_CONTEXT_FALLBACK,
    _assert_smoke_response_proves_llm_call,
    _expects_hosted_private_exception,
    _expects_private_runtime,
    _query_payload,
)


def test_real_api_smoke_defaults_to_hosted_openrouter_profile(monkeypatch):
    monkeypatch.delenv("LITTLE_BULL_E2E_CONFIDENTIALITY", raising=False)
    monkeypatch.delenv("LITTLE_BULL_E2E_MODEL_PROFILE", raising=False)

    payload = _query_payload("default")

    assert payload["confidentiality"] == "normal"
    assert payload["model_profile"] == "equilibrado"
    assert _expects_private_runtime(payload) is False


def test_real_api_smoke_can_be_explicitly_switched_to_private_profile(monkeypatch):
    monkeypatch.setenv("LITTLE_BULL_E2E_CONFIDENTIALITY", "privado")
    monkeypatch.setenv("LITTLE_BULL_E2E_MODEL_PROFILE", "privado")

    payload = _query_payload("default")

    assert payload["confidentiality"] == "privado"
    assert payload["model_profile"] == "privado"
    assert _expects_private_runtime(payload) is True


def test_real_api_smoke_can_expect_hosted_private_exception(monkeypatch):
    monkeypatch.setenv("LITTLE_BULL_E2E_CONFIDENTIALITY", "privado")
    monkeypatch.setenv("LITTLE_BULL_E2E_MODEL_PROFILE", "equilibrado")
    monkeypatch.setenv("LITTLE_BULL_E2E_HOSTED_PRIVATE_EXCEPTION", "1")

    payload = _query_payload("default")

    assert _expects_hosted_private_exception(payload) is True
    assert _expects_private_runtime(payload) is False


def test_real_api_smoke_rejects_no_context_fallback_response():
    try:
        _assert_smoke_response_proves_llm_call(NO_CONTEXT_FALLBACK)
    except AssertionError as exc:
        assert "no-context fallback" in str(exc)
    else:
        raise AssertionError("Expected the real API smoke to reject fallback responses")
