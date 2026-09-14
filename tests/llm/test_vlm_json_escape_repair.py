"""Regression tests: LaTeX escape damage repair for VLM JSON responses.

Models writing LaTeX inside JSON strings routinely under-escape
backslashes: ``"\\frac"`` is *valid* JSON meaning form feed + ``rac``, so
``json_repair.loads`` silently decodes it and the LaTeX command is
destroyed (``$\\frac{...}$`` -> ``$\\x0crac{...}$``). The damage surface
is exactly the five decodable escape letters b/f/n/r/t — json_repair
preserves invalid escapes like ``\\alpha`` verbatim.

``repair_vlm_json_escape_damage`` restores the two zero-risk cases
(form feed / backspace + letter: no legitimate use in LLM prose), and restores
whitespace-class cases only inside explicit dollar math. Outside math,
tab/CR/newline remain ambiguous legitimate whitespace and are only logged.
"""

import json_repair
import logging

import pytest

from lightrag.utils import repair_vlm_json_escape_damage


@pytest.fixture
def _propagate_lightrag_logger(monkeypatch):
    """``lightrag.utils.logger`` sets ``propagate = False``; restore
    propagation locally so ``caplog`` can capture WARNING records."""
    monkeypatch.setattr(logging.getLogger("lightrag"), "propagate", True)


@pytest.mark.offline
def test_formfeed_followed_by_letter_restores_backslash_f():
    assert repair_vlm_json_escape_damage("$\x0crac{610}{C}$") == r"$\frac{610}{C}$"


@pytest.mark.offline
def test_backspace_followed_by_letter_restores_backslash_b():
    assert (
        repair_vlm_json_escape_damage("$\x08eta + \x08ar{x}$") == r"$\beta + \bar{x}$"
    )


@pytest.mark.offline
def test_isolated_control_chars_left_for_sanitization():
    """Form feed / backspace NOT followed by a letter are junk, not LaTeX —
    leave them untouched so downstream sanitization drops them."""
    text = "before\x0c after\x08."
    assert repair_vlm_json_escape_damage(text) == text


@pytest.mark.offline
def test_clean_text_is_unchanged_and_idempotent():
    """Correctly double-escaped LaTeX decodes to real backslash sequences;
    repair must not touch them, and repairing twice equals repairing once."""
    clean = r"$\frac{a}{b}$ and \beta with plain text"
    assert repair_vlm_json_escape_damage(clean) == clean
    damaged = "$\x0crac{a}{b}$"
    once = repair_vlm_json_escape_damage(damaged)
    assert repair_vlm_json_escape_damage(once) == once


@pytest.mark.parametrize(
    ("damaged", "expected"),
    [
        ("domain is $\tau^2$", r"domain is $\tau^2$"),
        ("$a \times b$", r"$a \times b$"),
        ("$$\nabla f = 0$$", "$$\\nabla f = 0$$"),
        ("$\rho + \right)$", r"$\rho + \right)$"),
    ],
)
@pytest.mark.offline
def test_whitespace_class_damage_inside_dollar_math_is_repaired(
    damaged, expected, caplog, _propagate_lightrag_logger
):
    with caplog.at_level(logging.WARNING, logger="lightrag"):
        result = repair_vlm_json_escape_damage(damaged, context="table/t1")
    assert result == expected
    assert any(
        "Repaired whitespace-class LaTeX escape damage inside dollar math"
        in rec.message
        and "table/t1" in rec.getMessage()
        for rec in caplog.records
    )
    assert not any("not auto-repaired" in rec.message for rec in caplog.records)


@pytest.mark.offline
def test_whitespace_class_damage_outside_math_is_logged_not_rewritten(
    caplog, _propagate_lightrag_logger
):
    """Outside math, tab + residue stays ambiguous and is not rewritten."""
    damaged = "label:\tau"
    with caplog.at_level(logging.WARNING, logger="lightrag"):
        result = repair_vlm_json_escape_damage(damaged, context="table/t1")
    assert result == damaged
    assert any(
        "whitespace-class LaTeX escape damage" in rec.message
        and "table/t1" in rec.getMessage()
        for rec in caplog.records
    )


@pytest.mark.offline
def test_unpaired_or_escaped_dollars_do_not_enable_whitespace_repair():
    assert repair_vlm_json_escape_damage("price $5 then\tau") == "price $5 then\tau"
    damaged = r"escaped \$" + "\tau" + "$"
    assert repair_vlm_json_escape_damage(damaged) == damaged


@pytest.mark.offline
def test_math_whitespace_repair_is_idempotent_and_handles_multiple_spans():
    damaged = "first $\tau^2$, second $$a \times b$$"
    expected = r"first $\tau^2$, second $$a \times b$$"
    once = repair_vlm_json_escape_damage(damaged)
    assert once == expected
    assert repair_vlm_json_escape_damage(once) == expected


@pytest.mark.offline
def test_legitimate_whitespace_is_not_flagged(caplog, _propagate_lightrag_logger):
    """Whitespace followed by ordinary words (no whitelist residue with a
    word boundary) must not trigger the detection warning."""
    legit = "col1\tauthor list\nablation studies follow\nexists in the table"
    with caplog.at_level(logging.WARNING, logger="lightrag"):
        result = repair_vlm_json_escape_damage(legit)
    assert result == legit
    assert not caplog.records


@pytest.mark.offline
def test_mixed_escaping_real_world_response():
    """Pin the real-world shape that crashed ingestion: within ONE response
    the model double-escaped ``\\times``/``\\text`` but single-escaped
    ``\\frac``. After json_repair + repair, the formula is whole again."""
    raw_response = (
        '{"name": "成本对比", '
        '"description": "GraphRAG消耗$\\frac{610 \\\\times 1,000}{C_{\\\\text{max}}}$次调用"}'
    )
    parsed = json_repair.loads(raw_response)
    assert isinstance(parsed, dict)
    description = parsed["description"]
    assert isinstance(description, str)
    assert "\x0c" in description  # damage confirmed pre-repair

    repaired = repair_vlm_json_escape_damage(description)
    assert "$\\frac{610 \\times 1,000}{C_{\\text{max}}}$" in repaired
    assert "\x0c" not in repaired


@pytest.mark.offline
def test_prompts_require_double_escaped_backslashes():
    """All three modality prompts must instruct double-escaping; the
    equation prompt had this rule first, image/table were aligned to it."""
    from lightrag.prompt_multimodal import MULTIMODAL_PROMPTS

    for key in ("image_analysis", "table_analysis", "equation_analysis"):
        template = MULTIMODAL_PROMPTS[key]
        assert "escape backslashes" in template or "double-escaped" in template, (
            f"{key}: missing backslash escaping rule in OUTPUT RULES"
        )


@pytest.mark.offline
def test_extraction_system_prompt_requires_double_escaped_backslashes():
    """The JSON entity-extraction system prompt governs both the initial
    round and the gleaning round (same system prompt is passed to the
    gleaning call), so the escaping rule lives there — the user prompts
    deliberately carry no copy."""
    from lightrag.prompt import PROMPTS

    template = PROMPTS["entity_extraction_json_system_prompt"]
    assert "escape backslashes" in template and "double-escaped" in template, (
        "entity_extraction_json_system_prompt: missing backslash escaping "
        "rule in JSON Contract"
    )


@pytest.mark.offline
@pytest.mark.asyncio
async def test_extraction_json_result_repairs_latex_escape_damage():
    """Wiring regression for the extraction side: a raw LLM response with
    single-escaped LaTeX must yield entity/relation descriptions carrying
    the intact command — covers initial extraction, gleaning, and rebuild,
    which all parse through _process_json_extraction_result."""
    from lightrag.operate import _process_json_extraction_result

    raw_response = (
        '{"entities": [{"name": "LightRAG", "type": "Other", '
        '"description": "成本为 $\\frac{610}{C}$，领域为 $\\tau^2$"}], '
        '"relationships": [{"source": "LightRAG", "target": "GraphRAG", '
        '"keywords": "cost", '
        '"description": "比较 $\\frac{a}{b}$、$a \\times b$ 与 $\\\\beta$"}]}'
    )

    nodes, edges = await _process_json_extraction_result(
        raw_response, chunk_key="chunk-test", timestamp=0
    )

    (entity_list,) = [nodes[k] for k in nodes if k == "LightRAG"]
    assert "\\frac{610}{C}" in entity_list[0]["description"]
    assert "\\tau^2" in entity_list[0]["description"]
    assert "\x0c" not in entity_list[0]["description"]
    assert "\t" not in entity_list[0]["description"]

    (edge_list,) = list(edges.values())
    assert "\\frac{a}{b}" in edge_list[0]["description"]
    assert "\\times" in edge_list[0]["description"]
    assert "\\beta" in edge_list[0]["description"]
    assert "\x0c" not in edge_list[0]["description"]
