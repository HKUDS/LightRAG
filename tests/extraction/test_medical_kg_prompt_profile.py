import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from lightrag.prompt import load_entity_extraction_prompt_profile
from lightrag.utils import EmbeddingFunc


PROFILE_PATH = Path("prompts/entity_type/医学实体类型提示词.yml")
FORBIDDEN_ENTITY_TYPES = {"Dosage", "TimeCourse", "Biomarker", "Other"}
ALLOWED_ENTITY_TYPES = {
    "Disease",
    "Pathogen",
    "Symptom",
    "Complication",
    "Population",
    "RiskFactor",
    "Drug",
    "Vaccine",
    "TreatmentRegimen",
    "DiagnosticTest",
    "DiagnosticCriterion",
    "PublicHealthMeasure",
    "Guideline",
    "Recommendation",
    "ClinicalDepartment",
    "Anatomy",
}
RELATION_KEYWORDS = {
    "病原导致",
    "病原分型",
    "临床表现",
    "症状归类",
    "并发风险",
    "高危因素",
    "推荐治疗",
    "剂量用法",
    "诊断依据",
    "检测方法",
    "重症判定",
    "预防措施",
    "指南建议",
    "适用于",
    "属于",
}


@pytest.fixture(scope="module")
def medical_profile() -> dict:
    return load_entity_extraction_prompt_profile(PROFILE_PATH)


def test_medical_kg_prompt_profile_loads_guidance_as_normal_block(
    medical_profile: dict,
) -> None:
    guidance = medical_profile["entity_types_guidance"]

    assert not guidance.lstrip().startswith("entity_types_guidance:")
    assert "只抽取原文明确支持" in guidance
    assert "同义词、简称和全称" in guidance
    assert "Dosage" in guidance
    assert "TimeCourse" in guidance
    assert "Biomarker" in guidance
    assert "Other" in guidance

    for entity_type in FORBIDDEN_ENTITY_TYPES:
        assert f"{entity_type}:" not in guidance
    for entity_type in ALLOWED_ENTITY_TYPES:
        assert entity_type in guidance
    for keyword in RELATION_KEYWORDS:
        assert keyword in guidance


def test_medical_kg_prompt_profile_json_examples_are_valid_and_do_not_promote_values(
    medical_profile: dict,
) -> None:
    examples = medical_profile["entity_extraction_json_examples"]
    assert examples

    for example in examples:
        payload = json.loads(example)
        entities = payload["entities"]
        relationships = payload["relationships"]

        assert all(entity["type"] not in FORBIDDEN_ENTITY_TYPES for entity in entities)
        assert all(entity["name"] != "75 mg" for entity in entities)
        assert any(
            relationship["source"] == "奥司他韦"
            and relationship["keywords"] in {"剂量用法", "推荐治疗"}
            and "75 mg" in relationship["description"]
            for relationship in relationships
        )


def test_medical_kg_prompt_profile_text_example_keeps_dosage_in_relation_description(
    medical_profile: dict,
) -> None:
    examples = medical_profile["entity_extraction_examples"]
    assert examples

    combined = "\n".join(examples)
    assert "relation{tuple_delimiter}" in combined
    assert "75 mg" in combined
    assert "entity{tuple_delimiter}75 mg" not in combined
    assert (
        "relation{tuple_delimiter}奥司他韦{tuple_delimiter}儿童"
        in combined
        or "relation{tuple_delimiter}奥司他韦{tuple_delimiter}流行性感冒"
        in combined
    )


def test_clinical_guideline_profile_auto_uses_medical_prompt(tmp_path, monkeypatch) -> None:
    from lightrag import LightRAG

    async def _embed(texts: list[str]):
        return [[0.0, 0.0, 0.0] for _ in texts]

    monkeypatch.delenv("ENTITY_TYPE_PROMPT_FILE", raising=False)

    rag = LightRAG(
        working_dir=str(tmp_path / "rag-medical-profile-prompt"),
        llm_model_func=AsyncMock(),
        embedding_func=EmbeddingFunc(embedding_dim=3, func=_embed),
        entity_extraction_use_json=True,
        addon_params={"medical_kg_profile": "clinical_guideline_zh"},
    )

    global_config = rag._build_global_config()
    prompt_profile = global_config["_entity_extraction_prompt_profile"]

    assert (
        global_config["addon_params"]["entity_type_prompt_file"]
        == "医学实体类型提示词.yml"
    )
    assert "只抽取原文明确支持" in prompt_profile["entity_types_guidance"]
    assert "面向中文临床指南" in prompt_profile["entity_types_guidance"]
    assert prompt_profile["entity_extraction_json_examples"]


def test_clinical_guideline_profile_uses_bundled_prompt_without_prompt_dir(
    tmp_path, monkeypatch
) -> None:
    from lightrag import LightRAG

    async def _embed(texts: list[str]):
        return [[0.0, 0.0, 0.0] for _ in texts]

    monkeypatch.delenv("PROMPT_DIR", raising=False)
    monkeypatch.delenv("ENTITY_TYPE_PROMPT_FILE", raising=False)

    rag = LightRAG(
        working_dir=str(tmp_path / "rag-medical-profile-default-prompt-dir"),
        llm_model_func=AsyncMock(),
        embedding_func=EmbeddingFunc(embedding_dim=3, func=_embed),
        entity_extraction_use_json=True,
        addon_params={"medical_kg_profile": "clinical_guideline_zh"},
    )

    global_config = rag._build_global_config()
    prompt_profile = global_config["_entity_extraction_prompt_profile"]

    assert (
        global_config["addon_params"]["entity_type_prompt_file"]
        == "医学实体类型提示词.yml"
    )
    assert "面向中文临床指南" in prompt_profile["entity_types_guidance"]


def test_clinical_guideline_profile_uses_bundled_prompt_outside_repo_cwd(
    tmp_path, monkeypatch
) -> None:
    from lightrag import LightRAG

    async def _embed(texts: list[str]):
        return [[0.0, 0.0, 0.0] for _ in texts]

    monkeypatch.delenv("PROMPT_DIR", raising=False)
    monkeypatch.delenv("ENTITY_TYPE_PROMPT_FILE", raising=False)
    monkeypatch.chdir(tmp_path)

    rag = LightRAG(
        working_dir=str(tmp_path / "rag-medical-profile-package-prompt"),
        llm_model_func=AsyncMock(),
        embedding_func=EmbeddingFunc(embedding_dim=3, func=_embed),
        entity_extraction_use_json=True,
        addon_params={"medical_kg_profile": "clinical_guideline_zh"},
    )

    prompt_profile = rag._build_global_config()["_entity_extraction_prompt_profile"]

    assert "面向中文临床指南" in prompt_profile["entity_types_guidance"]


def test_clinical_guideline_profile_keeps_explicit_prompt_file(monkeypatch) -> None:
    from lightrag.addon_params import default_addon_params, normalize_addon_params

    monkeypatch.setenv("MEDICAL_KG_PROFILE", "clinical_guideline_zh")
    monkeypatch.setenv("ENTITY_TYPE_PROMPT_FILE", "custom.yml")

    default_params = default_addon_params()
    env_backfilled_params = normalize_addon_params({})
    explicit_params = normalize_addon_params(
        {
            "medical_kg_profile": "clinical_guideline_zh",
            "entity_type_prompt_file": "custom.yml",
        }
    )

    assert default_params["entity_type_prompt_file"] == "custom.yml"
    assert env_backfilled_params["entity_type_prompt_file"] == "custom.yml"
    assert explicit_params["entity_type_prompt_file"] == "custom.yml"
