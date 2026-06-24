"""LM Studio bindings via its OpenAI-compatible local API."""

import os
from collections.abc import AsyncIterator
from typing import Literal, Union

import httpx
import numpy as np

from lightrag.exceptions import APIConnectionError
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import logger, wrap_embedding_func_with_attrs

_DEFAULT_HOST = "http://localhost:1234/v1"
_DEFAULT_API_KEY = "lm-studio"
_ANY_AVAILABLE_ALIASES = frozenset(
    {"", "any-available", "any_available", "auto", "*", "any"}
)
_LLM_MODEL_TYPES = frozenset({"llm", "vlm"})
_EMBEDDING_MODEL_TYPES = frozenset({"embeddings", "embedding"})
_MODEL_RESOLUTION_CACHE: dict[tuple[str, str], str] = {}
_EMBEDDING_DIM_CACHE: dict[str, int] = {}


def is_any_available_model(model: str | None) -> bool:
    """Return True when the model id should be resolved from LM Studio."""
    if model is None:
        return True
    return model.strip().lower() in _ANY_AVAILABLE_ALIASES


def is_auto_embedding_dim(embedding_dim: int | str | None) -> bool:
    """Return True when embedding dimension should be probed from LM Studio."""
    if embedding_dim is None:
        return True
    if isinstance(embedding_dim, str):
        return is_any_available_model(embedding_dim)
    return False


def clear_lmstudio_model_cache() -> None:
    """Clear cached any-available model resolutions (mainly for tests)."""
    _MODEL_RESOLUTION_CACHE.clear()
    _EMBEDDING_DIM_CACHE.clear()


def _resolve_host(base_url: str | None) -> str:
    if base_url:
        return base_url
    return os.getenv("LLM_BINDING_HOST", _DEFAULT_HOST)


def _resolve_api_key(api_key: str | None) -> str:
    if api_key:
        return api_key
    for env_var in (
        "LMSTUDIO_API_KEY",
        "EMBEDDING_BINDING_API_KEY",
        "LLM_BINDING_API_KEY",
        "OPENAI_API_KEY",
    ):
        value = os.getenv(env_var)
        if value:
            return value
    return _DEFAULT_API_KEY


def _v0_api_base(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        return normalized[:-3] + "/api/v0"
    return f"{normalized}/api/v0"


def _pick_model_from_v0_entries(
    entries: list[dict],
    purpose: Literal["llm", "embedding"],
) -> str | None:
    allowed_types = _LLM_MODEL_TYPES if purpose == "llm" else _EMBEDDING_MODEL_TYPES

    for entry in entries:
        entry_type = str(entry.get("type", "")).lower()
        if entry_type not in allowed_types:
            continue
        loaded_instances = entry.get("loaded_instances") or []
        if loaded_instances:
            instance_id = loaded_instances[0].get("id")
            if instance_id:
                return str(instance_id)

    for entry in entries:
        entry_type = str(entry.get("type", "")).lower()
        if entry_type not in allowed_types:
            continue
        for field in ("id", "key", "selected_variant"):
            candidate = entry.get(field)
            if candidate:
                return str(candidate)
    return None


async def _list_v0_models(base_url: str, api_key: str) -> list[dict]:
    url = f"{_v0_api_base(base_url)}/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        payload = response.json()

    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return payload["data"]
    if isinstance(payload, list):
        return payload
    return []


async def _resolve_via_v1_models(
    base_url: str,
    api_key: str,
    purpose: Literal["llm", "embedding"],
) -> str:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    try:
        page = await client.models.list()
        model_ids = [model.id for model in page.data if model.id]
    finally:
        await client.close()

    if not model_ids:
        raise APIConnectionError(f"LM Studio returned no models from {base_url}/models")

    if purpose == "embedding":
        for model_id in model_ids:
            if "embed" in model_id.lower():
                return model_id

    return model_ids[0]


async def resolve_lmstudio_model(
    model: str | None,
    base_url: str | None = None,
    api_key: str | None = None,
    purpose: Literal["llm", "embedding"] = "llm",
) -> str:
    """Resolve a concrete LM Studio model id, including any-available aliases."""
    if not is_any_available_model(model):
        return str(model).strip()

    resolved_host = _resolve_host(base_url)
    resolved_api_key = _resolve_api_key(api_key)
    cache_key = (resolved_host, purpose)
    if cache_key in _MODEL_RESOLUTION_CACHE:
        return _MODEL_RESOLUTION_CACHE[cache_key]

    resolved_model: str | None = None
    try:
        v0_entries = await _list_v0_models(resolved_host, resolved_api_key)
        resolved_model = _pick_model_from_v0_entries(v0_entries, purpose)
    except (httpx.HTTPError, ValueError, TypeError) as exc:
        logger.debug(
            "LM Studio v0 model listing failed for %s (%s); falling back to /v1/models",
            resolved_host,
            exc,
        )

    if resolved_model is None:
        resolved_model = await _resolve_via_v1_models(
            resolved_host, resolved_api_key, purpose
        )

    _MODEL_RESOLUTION_CACHE[cache_key] = resolved_model
    logger.info(
        "LM Studio resolved %s model '%s' to '%s' at %s",
        purpose,
        model or "<blank>",
        resolved_model,
        resolved_host,
    )
    return resolved_model


def _normalize_lmstudio_response_format(kwargs: dict) -> None:
    """Translate OpenAI response_format values for LM Studio's stricter API."""
    response_format = kwargs.get("response_format")
    if response_format is None:
        return
    if not isinstance(response_format, dict):
        return

    response_type = response_format.get("type")
    if response_type == "json_object":
        kwargs["response_format"] = {"type": "text"}
        logger.debug(
            "LM Studio does not support response_format json_object; using type=text"
        )
        return

    if response_type not in ("json_schema", "text"):
        logger.warning(
            "LM Studio unsupported response_format type '%s'; using type=text",
            response_type,
        )
        kwargs["response_format"] = {"type": "text"}


async def probe_lmstudio_embedding_dim(
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> tuple[int, str]:
    """Probe LM Studio once for embedding dimension and resolved model id."""
    resolved_host = _resolve_host(base_url)
    resolved_api_key = _resolve_api_key(api_key)
    if resolved_host in _EMBEDDING_DIM_CACHE:
        return _EMBEDDING_DIM_CACHE[resolved_host]

    resolved_model = await resolve_lmstudio_model(
        model,
        base_url=resolved_host,
        api_key=resolved_api_key,
        purpose="embedding",
    )
    vectors = await openai_embed.func(
        ["dimension probe"],
        model=resolved_model,
        base_url=resolved_host,
        api_key=resolved_api_key,
        embedding_dim=None,
    )
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise APIConnectionError(
            f"LM Studio embedding probe at {resolved_host} returned invalid shape "
            f"{vectors.shape} for model '{resolved_model}'"
        )

    embedding_dim = int(vectors.shape[1])
    _EMBEDDING_DIM_CACHE[resolved_host] = embedding_dim
    logger.info(
        "LM Studio probed embedding dimension %d for model '%s' at %s",
        embedding_dim,
        resolved_model,
        resolved_host,
    )
    return embedding_dim, resolved_model


async def lmstudio_complete_if_cache(
    model,
    prompt,
    system_prompt=None,
    history_messages=[],
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs,
) -> Union[str, AsyncIterator[str]]:
    resolved_host = _resolve_host(base_url)
    resolved_api_key = _resolve_api_key(api_key)
    resolved_model = await resolve_lmstudio_model(
        model,
        base_url=resolved_host,
        api_key=resolved_api_key,
        purpose="llm",
    )
    _normalize_lmstudio_response_format(kwargs)
    return await openai_complete_if_cache(
        resolved_model,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=resolved_api_key,
        base_url=resolved_host,
        **kwargs,
    )


async def lmstudio_model_complete(
    prompt,
    system_prompt=None,
    history_messages=None,
    enable_cot: bool = False,
    keyword_extraction=False,
    entity_extraction=False,
    **kwargs,
) -> Union[str, AsyncIterator[str]]:
    if history_messages is None:
        history_messages = []
    if keyword_extraction:
        kwargs.setdefault("keyword_extraction", True)
    if entity_extraction:
        kwargs.setdefault("entity_extraction", True)
    model_name = kwargs["hashing_kv"].global_config["llm_model_name"]
    return await lmstudio_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        enable_cot=enable_cot,
        **kwargs,
    )


@wrap_embedding_func_with_attrs(
    embedding_dim=1536,
    max_token_size=8192,
    model_name="any-available",
    supports_asymmetric=True,
)
async def lmstudio_embed(
    texts: list[str],
    model: str | None = "any-available",
    base_url: str | None = None,
    api_key: str | None = None,
    embedding_dim: int | None = None,
    max_token_size: int | None = None,
    context: str = "document",
    query_prefix: str | None = None,
    document_prefix: str | None = None,
    **kwargs,
) -> np.ndarray:
    resolved_host = _resolve_host(base_url)
    resolved_api_key = _resolve_api_key(api_key)
    resolved_model = await resolve_lmstudio_model(
        model,
        base_url=resolved_host,
        api_key=resolved_api_key,
        purpose="embedding",
    )
    return await openai_embed.func(
        texts,
        model=resolved_model,
        base_url=resolved_host,
        api_key=resolved_api_key,
        embedding_dim=embedding_dim,
        max_token_size=max_token_size,
        context=context,
        query_prefix=query_prefix,
        document_prefix=document_prefix,
        **kwargs,
    )
