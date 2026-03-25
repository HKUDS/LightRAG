from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from lightrag.constants import DEFAULT_ENTITY_TYPES, DEFAULT_SUMMARY_LANGUAGE
from lightrag.prompt import (
    PROMPT_FAMILY_NAMES,
    get_default_prompt_config,
    validate_prompt_config,
)

INDEXING_PROMPT_VERSION_FAMILIES = {"shared", "entity_extraction", "summary"}
RETRIEVAL_PROMPT_VERSION_FAMILIES = {"query", "keywords"}
INDEXING_PROMPT_VERSION_EXTRA_FIELDS = {"entity_types", "summary_language"}

PROMPT_VERSION_GROUPS: dict[str, dict[str, set[str]]] = {
    "indexing": {
        "families": INDEXING_PROMPT_VERSION_FAMILIES,
        "extra_fields": INDEXING_PROMPT_VERSION_EXTRA_FIELDS,
    },
    "retrieval": {
        "families": RETRIEVAL_PROMPT_VERSION_FAMILIES,
        "extra_fields": set(),
    },
}

_LOCALIZED_GROUP_SUMMARY_LANGUAGE = {
    "en": DEFAULT_SUMMARY_LANGUAGE,
    "zh": "Chinese",
}

_LOCALIZED_GROUP_COMMENT = {
    "en": "English initial version",
    "zh": "中文初始版本",
}


def _get_group(group_type: str) -> dict[str, set[str]]:
    group = PROMPT_VERSION_GROUPS.get(group_type)
    if group is None:
        raise ValueError(f"Unknown prompt version group '{group_type}'")
    return group


def _validate_indexing_extras(payload: dict[str, Any]) -> None:
    if "entity_types" in payload:
        entity_types = payload["entity_types"]
        if not isinstance(entity_types, list) or not entity_types:
            raise ValueError("Indexing payload 'entity_types' must be a non-empty list[str]")
        if not all(isinstance(item, str) and item.strip() for item in entity_types):
            raise ValueError(
                "Indexing payload 'entity_types' must be a non-empty list[str]"
            )

    if "summary_language" in payload:
        summary_language = payload["summary_language"]
        if not isinstance(summary_language, str) or not summary_language.strip():
            raise ValueError(
                "Indexing payload 'summary_language' must be a non-empty string"
            )


def validate_prompt_group_payload(group_type: str, payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise ValueError("Prompt version payload must be a dict")

    group = _get_group(group_type)
    allowed_keys = group["families"] | group["extra_fields"]
    unknown_keys = sorted(set(payload) - allowed_keys)
    if unknown_keys:
        raise ValueError(
            f"Prompt version payload for '{group_type}' contains unknown keys: "
            + ", ".join(unknown_keys)
        )

    prompt_subset = project_group_prompt_config(group_type, payload)
    if prompt_subset:
        validate_prompt_config(prompt_subset, allowed_families=group["families"])

    if group_type == "indexing":
        _validate_indexing_extras(payload)


def project_group_prompt_config(group_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    families = _get_group(group_type)["families"]
    return {family: deepcopy(payload[family]) for family in families if family in payload}


def _localize_prompt_config(locale: str) -> dict[str, Any]:
    if locale not in _LOCALIZED_GROUP_SUMMARY_LANGUAGE:
        raise ValueError(f"Unsupported prompt seed locale '{locale}'")

    prompt_config = get_default_prompt_config()
    if locale == "en":
        return prompt_config

    localized = deepcopy(prompt_config)
    localized["query"]["rag_response"] = (
        "请使用中文回答，并保持引用结构清晰。\n\n"
        + localized["query"]["rag_response"]
    )
    localized["query"]["naive_rag_response"] = (
        "请使用中文回答，并保持引用结构清晰。\n\n"
        + localized["query"]["naive_rag_response"]
    )
    localized["keywords"]["keywords_extraction"] = (
        "请优先以中文理解并提取关键词。\n\n"
        + localized["keywords"]["keywords_extraction"]
    )
    localized["entity_extraction"]["system_prompt"] = (
        "请在保证结构化输出格式不变的前提下，尽量使用中文完成实体与关系抽取。\n\n"
        + localized["entity_extraction"]["system_prompt"]
    )
    localized["entity_extraction"]["user_prompt"] = (
        "请使用中文输出提取结果。\n\n"
        + localized["entity_extraction"]["user_prompt"]
    )
    localized["entity_extraction"]["continue_prompt"] = (
        "请继续使用中文补全缺失的抽取结果。\n\n"
        + localized["entity_extraction"]["continue_prompt"]
    )
    localized["summary"]["summarize_entity_descriptions"] = (
        "请使用中文整理并总结以下描述。\n\n"
        + localized["summary"]["summarize_entity_descriptions"]
    )
    return localized


def _build_seed_record(
    locale: str, group_type: str, payload: dict[str, Any]
) -> dict[str, Any]:
    timestamp = datetime.now(timezone.utc).isoformat()
    return {
        "version_id": f"{group_type}-{locale}-default-v1",
        "group_type": group_type,
        "version_name": f"{group_type}-{locale}-default",
        "version_number": 1,
        "comment": _LOCALIZED_GROUP_COMMENT[locale],
        "source_version_id": None,
        "created_at": timestamp,
        "payload": payload,
    }


def build_localized_seed_versions(locale: str) -> dict[str, dict[str, Any]]:
    prompt_config = _localize_prompt_config(locale)
    summary_language = _LOCALIZED_GROUP_SUMMARY_LANGUAGE[locale]

    indexing_payload = project_group_prompt_config(
        "indexing",
        prompt_config,
    )
    indexing_payload["entity_types"] = deepcopy(DEFAULT_ENTITY_TYPES)
    indexing_payload["summary_language"] = summary_language

    retrieval_payload = project_group_prompt_config("retrieval", prompt_config)

    seeds = {
        "indexing": _build_seed_record(locale, "indexing", indexing_payload),
        "retrieval": _build_seed_record(locale, "retrieval", retrieval_payload),
    }

    for group_name, seed in seeds.items():
        projected = project_group_prompt_config(group_name, seed["payload"])
        validate_prompt_config(
            projected,
            allowed_families=PROMPT_VERSION_GROUPS[group_name]["families"],
        )
        if group_name == "indexing":
            _validate_indexing_extras(seed["payload"])

    unexpected_families = set(prompt_config) - PROMPT_FAMILY_NAMES
    if unexpected_families:
        raise ValueError(
            "Prompt seed config contains unknown families: "
            + ", ".join(sorted(unexpected_families))
        )

    return seeds
