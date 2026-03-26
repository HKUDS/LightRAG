from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import re
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
    "zh": "中文",
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


def _normalize_entity_types_value(entity_types: Any) -> list[str]:
    if isinstance(entity_types, str):
        parsed = [
            item.strip()
            for item in re.split(r"[,，\n]+", entity_types)
            if item.strip()
        ]
        if not parsed:
            raise ValueError(
                "Indexing payload 'entity_types' must be a non-empty list[str]"
            )
        return parsed

    if not isinstance(entity_types, list) or not entity_types:
        raise ValueError("Indexing payload 'entity_types' must be a non-empty list[str]")
    if not all(isinstance(item, str) and item.strip() for item in entity_types):
        raise ValueError("Indexing payload 'entity_types' must be a non-empty list[str]")
    return [item.strip() for item in entity_types]


def normalize_prompt_group_payload(group_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Prompt version payload must be a dict")

    _get_group(group_type)
    normalized_payload = deepcopy(payload)

    if group_type == "indexing" and "entity_types" in normalized_payload:
        normalized_payload["entity_types"] = _normalize_entity_types_value(
            normalized_payload["entity_types"]
        )

    if group_type == "indexing" and "summary_language" in normalized_payload:
        summary_language = normalized_payload["summary_language"]
        if isinstance(summary_language, str):
            normalized_payload["summary_language"] = summary_language.strip()

    return normalized_payload


def _validate_indexing_extras(payload: dict[str, Any]) -> None:
    if "entity_types" in payload:
        _normalize_entity_types_value(payload["entity_types"])

    if "summary_language" in payload:
        summary_language = payload["summary_language"]
        if not isinstance(summary_language, str) or not summary_language.strip():
            raise ValueError(
                "Indexing payload 'summary_language' must be a non-empty string"
            )


def validate_prompt_group_payload(group_type: str, payload: dict[str, Any]) -> None:
    normalized_payload = normalize_prompt_group_payload(group_type, payload)
    group = _get_group(group_type)
    allowed_keys = group["families"] | group["extra_fields"]
    unknown_keys = sorted(set(normalized_payload) - allowed_keys)
    if unknown_keys:
        raise ValueError(
            f"Prompt version payload for '{group_type}' contains unknown keys: "
            + ", ".join(unknown_keys)
        )

    prompt_subset = project_group_prompt_config(group_type, normalized_payload)
    if prompt_subset:
        validate_prompt_config(prompt_subset, allowed_families=group["families"])

    if group_type == "indexing":
        _validate_indexing_extras(normalized_payload)


def project_group_prompt_config(group_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    families = _get_group(group_type)["families"]
    return {family: deepcopy(payload[family]) for family in families if family in payload}


def _localize_prompt_config(locale: str) -> dict[str, Any]:
    if locale not in _LOCALIZED_GROUP_SUMMARY_LANGUAGE:
        raise ValueError(f"Unsupported prompt seed locale '{locale}'")

    prompt_config = get_default_prompt_config()
    if locale == "en":
        return prompt_config

    return {
        "shared": {
            "tuple_delimiter": prompt_config["shared"]["tuple_delimiter"],
            "completion_delimiter": prompt_config["shared"]["completion_delimiter"],
        },
        "query": {
            "rag_response": """---角色---

你是一名知识库问答助手。你的任务是严格基于给定上下文回答用户问题，不要引入上下文之外的事实。

---目标---

请根据提供的知识图谱信息与文档片段，输出清晰、准确、可引用的答案。

---要求---

1. 只能使用给定上下文中的信息作答。
2. 如果上下文不足以支持结论，请明确说明信息不足，不要猜测。
3. 答案的组织形式应符合：{response_type}。
4. 如果存在额外输出要求，请遵循：{user_prompt}。
5. 如上下文中存在参考资料，请在答案末尾保留参考信息。

---上下文---

{context_data}
""",
            "naive_rag_response": """---角色---

你是一名文档问答助手。你的任务是严格基于给定文档片段回答问题，不要补充外部知识。

---目标---

请根据提供的文档内容生成清晰、准确、可引用的回答。

---要求---

1. 只能使用给定内容中的信息作答。
2. 如果信息不足，请明确说明，而不是猜测。
3. 答案的组织形式应符合：{response_type}。
4. 如果存在额外输出要求，请遵循：{user_prompt}。

---内容---

{content_data}
""",
            "kg_query_context": """知识图谱实体信息：

```json
{entities_str}
```

知识图谱关系信息：

```json
{relations_str}
```

文档片段（每条都包含 reference_id，可与参考文档列表对应）：

```json
{text_chunks_str}
```

参考文档列表：

```
{reference_list_str}
```
""",
            "naive_query_context": """文档片段（每条都包含 reference_id，可与参考文档列表对应）：

```json
{text_chunks_str}
```

参考文档列表：

```
{reference_list_str}
```
""",
        },
        "keywords": {
            "keywords_extraction": """---角色---

你是一名 RAG 查询关键词提取助手，负责从用户问题中识别高层关键词与底层关键词，以便进行更有效的检索。

---任务---

请分析下面的问题，并输出 JSON：

- `high_level_keywords`: 更偏主题、概念、方向性的关键词
- `low_level_keywords`: 更偏实体、细节、术语、具体对象的关键词

---要求---

1. 输出必须是合法 JSON。
2. 不要输出任何额外说明。
3. 输出语言优先使用：{language}。

---示例---
{examples}

---用户问题---
{query}

---输出---
""",
            "keywords_extraction_examples": [
                """示例 1：

问题："人工智能如何影响制造业自动化？"

输出：
{
  "high_level_keywords": ["人工智能", "制造业自动化", "产业影响"],
  "low_level_keywords": ["工业机器人", "视觉检测", "预测性维护", "产线调度"]
}""",
                """示例 2：

问题："有哪些因素会影响新能源车的续航表现？"

输出：
{
  "high_level_keywords": ["新能源车", "续航表现", "影响因素"],
  "low_level_keywords": ["电池容量", "气温", "驾驶习惯", "能量回收", "轮胎阻力"]
}""",
            ],
        },
        "entity_extraction": {
            "system_prompt": """---角色---
你是一名知识图谱专家，负责从输入文本中抽取实体与关系。

---要求---
1. 每个实体输出 4 个字段，并使用 `{tuple_delimiter}` 分隔：
   `entity{tuple_delimiter}实体名{tuple_delimiter}实体类型{tuple_delimiter}实体描述`
2. 每个关系输出 5 个字段，并使用 `{tuple_delimiter}` 分隔：
   `relation{tuple_delimiter}源实体{tuple_delimiter}目标实体{tuple_delimiter}关系关键词{tuple_delimiter}关系描述`
3. 实体类型应尽量限定在 `{entity_types}` 中；若都不适用，使用 `Other`。
4. 输出语言使用 `{language}`，专有名词可保留原文。
5. 所有结果输出完成后，最后一行必须是 `{completion_delimiter}`。

---示例---
{examples}
""",
            "user_prompt": """---任务---
请从下面的文本中抽取实体与关系。

---补充要求---
1. 输入文本：{input_text}
2. 输出语言：{language}
3. 实体类型优先使用：{entity_types}
4. 输出完成后请以 `{completion_delimiter}` 结束。

---输出---
""",
            "continue_prompt": """---任务---
请基于上一轮结果继续补充遗漏或格式不完整的实体与关系。

---要求---
1. 继续使用 `{tuple_delimiter}` 作为字段分隔符。
2. 所有补充输出完成后，以 `{completion_delimiter}` 结束。
3. 输出语言保持为 `{language}`。
""",
            "examples": [
                """<实体类型>
[{entity_types}]

<输入文本>
```
张三在北京创立了星河科技，公司专注于智能制造软件。
```

<输出>
entity{tuple_delimiter}张三{tuple_delimiter}Person{tuple_delimiter}张三是星河科技的创始人。
entity{tuple_delimiter}北京{tuple_delimiter}Location{tuple_delimiter}北京是张三创立公司的城市。
entity{tuple_delimiter}星河科技{tuple_delimiter}Organization{tuple_delimiter}星河科技是一家专注于智能制造软件的公司。
relation{tuple_delimiter}张三{tuple_delimiter}星河科技{tuple_delimiter}创立, 创始人{tuple_delimiter}张三创立了星河科技。
relation{tuple_delimiter}星河科技{tuple_delimiter}北京{tuple_delimiter}所在地点{tuple_delimiter}星河科技在北京创立。
{completion_delimiter}""",
                """<实体类型>
[{entity_types}]

<输入文本>
```
华南理工大学与云启实验室联合发布了新的机器人控制框架。
```

<输出>
entity{tuple_delimiter}华南理工大学{tuple_delimiter}Organization{tuple_delimiter}华南理工大学是参与联合发布的机构之一。
entity{tuple_delimiter}云启实验室{tuple_delimiter}Organization{tuple_delimiter}云启实验室是参与联合发布的实验室。
entity{tuple_delimiter}机器人控制框架{tuple_delimiter}Method{tuple_delimiter}机器人控制框架是一项新发布的技术成果。
relation{tuple_delimiter}华南理工大学{tuple_delimiter}云启实验室{tuple_delimiter}合作发布{tuple_delimiter}两家机构联合发布了新的机器人控制框架。
relation{tuple_delimiter}华南理工大学{tuple_delimiter}机器人控制框架{tuple_delimiter}发布成果{tuple_delimiter}华南理工大学参与发布了该框架。
relation{tuple_delimiter}云启实验室{tuple_delimiter}机器人控制框架{tuple_delimiter}发布成果{tuple_delimiter}云启实验室参与发布了该框架。
{completion_delimiter}""",
            ],
        },
        "summary": {
            "summarize_entity_descriptions": """---角色---
你是一名知识图谱整理助手，负责将同一实体或关系的多条描述合并为一段完整摘要。

---任务---
请根据下面的描述列表，输出一段连贯、客观、信息完整的总结。

---要求---
1. 输入描述列表为：{description_list}
2. 需要在总结开头明确说明对象类型：{description_type}
3. 需要明确对象名称：{description_name}
4. 总结长度尽量控制在 {summary_length} 以内。
5. 输出语言为 {language}。

---输出---
""",
        },
    }


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
