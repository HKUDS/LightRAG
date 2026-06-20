from pathlib import Path

from lightrag.kb_iteration.agent_pipeline import _stage_prompt
from lightrag.kb_iteration.review_loop import LLMReviewLoopConfig, _system_prompt


def test_llm_review_prompts_exist_and_keep_safety_rules():
    prompt_dir = Path("lightrag/kb_iteration/prompts")
    expected = [
        "planner_zh.md",
        "reviewer_zh.md",
        "judge_zh.md",
        "patch_generator_zh.md",
    ]

    for filename in expected:
        text = (prompt_dir / filename).read_text(encoding="utf-8")
        assert "LLM" in text
        assert "LLM 输出不是医学证据" in text
        assert "source_id" in text
        assert "requires_approval" in text

    reviewer_text = (prompt_dir / "reviewer_zh.md").read_text(encoding="utf-8")
    judge_text = (prompt_dir / "judge_zh.md").read_text(encoding="utf-8")
    patch_text = (prompt_dir / "patch_generator_zh.md").read_text(encoding="utf-8")
    assert "source_id、file_path 和 chunk" in reviewer_text
    assert "source_id、file_path 和 chunk" in judge_text
    assert "source_id、file_path 和 chunk" in patch_text
    assert "不能自动应用" in patch_text
    assert "不输出 patch" in patch_text


def test_medical_agent_stage_prompt_includes_relation_schema():
    prompt = _stage_prompt("propose", "clinical_guideline_zh")

    assert "Medical Relationship Schema v1" in prompt
    assert "has_manifestation" in prompt
    assert "recommended_for" in prompt
    assert "推荐治疗" in prompt
    assert "诊断依据" in prompt


def test_non_medical_agent_stage_prompt_does_not_include_relation_schema():
    prompt = _stage_prompt("propose", None)

    assert "Medical Relationship Schema v1" not in prompt


def test_medical_review_loop_system_prompt_includes_relation_schema():
    prompt = _system_prompt(LLMReviewLoopConfig(), "medical_kg")

    assert "Medical Relationship Schema v1" in prompt
    assert "supports_or_refutes" in prompt
    assert "contraindicated_for" in prompt
