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


@pytest.mark.parametrize(
    ("damaged", "expected"),
    [
        # "\b" refuses a boundary before a word character, but "_", "{" and a
        # digit are the most common characters to follow a LaTeX command.
        ("$\tau_i$", r"$\tau_i$"),
        ("$\rho_{ij}$", r"$\rho_{ij}$"),
        ("$\theta_0$", r"$\theta_0$"),
        ("$\times2$", r"$\times2$"),
        ("$$\nabla_x f$$", "$$\\nabla_x f$$"),
    ],
)
@pytest.mark.offline
def test_in_math_repair_is_not_blocked_by_a_word_character(
    damaged, expected, caplog, _propagate_lightrag_logger
):
    """Inside a confirmed math span the residue whitelist matches on "not a
    letter", not on a word boundary -- otherwise the most common LaTeX
    continuations stay damaged AND unwarned, because the prose pattern that
    drives the tail warning refuses them too, so the loss is silent."""
    with caplog.at_level(logging.WARNING, logger="lightrag"):
        result = repair_vlm_json_escape_damage(damaged, context="table/t1")
    assert result == expected
    assert not any("not auto-repaired" in rec.message for rec in caplog.records)


@pytest.mark.offline
def test_currency_amount_does_not_consume_a_later_math_span():
    """A lone "$" price must not pair with the opening delimiter of a real
    formula: the "$" before the damage is preceded by a space and therefore
    cannot close an inline span."""
    assert repair_vlm_json_escape_damage("Cost is $5. The domain is $\tau^2$ end.") == (
        "Cost is $5. The domain is " + r"$\tau^2$" + " end."
    )
    # Two amounts must not form a span of their own around prose.
    between_amounts = "$5\rangle and $10"
    assert repair_vlm_json_escape_damage(between_amounts) == between_amounts
    # A range of amounts followed by real math: only the math is repaired.
    assert repair_vlm_json_escape_damage("$5-$10 range $\tau$") == (
        "$5-$10 range " + r"$\tau$"
    )


@pytest.mark.offline
def test_intact_math_does_not_block_a_following_damaged_span():
    """Pairing stays left-to-right and greedy for well-formed spans:
    "$10 \times 5$" is math, not a price, and consuming it must leave the
    next span pairable."""
    assert repair_vlm_json_escape_damage("$10 \times 5$ then $\tau$") == (
        r"$10 \times 5$ then $\tau$"
    )


@pytest.mark.offline
def test_inline_span_closes_against_the_first_dollar_of_a_display_pair():
    """An inline opener followed by a stray "$$" must still close: the first
    of the two dollars is a legal inline closer (non-space on its left, no
    digit on its right)."""
    assert repair_vlm_json_escape_damage("a $\tau$$ b") == "a " + r"$\tau$" + "$ b"


@pytest.mark.offline
def test_prose_between_two_spans_is_never_rewritten():
    """The scanner must not reach over a span boundary. A padded "$ x $" is
    not math (Pandoc requires a non-space after the opener), and treating it
    as a half-open span rewrote the ordinary prose that followed it."""
    damaged = "$ x $ then\text column $y$"
    assert repair_vlm_json_escape_damage(damaged) == damaged
    # Same shape without spaces around the delimiters: Chinese prose between
    # two real spans must survive too.
    cjk = "成本为$a$，领域\tau为$b$"
    assert repair_vlm_json_escape_damage(cjk) == cjk


@pytest.mark.offline
def test_currency_dollar_is_rejected_as_an_opener():
    """A "$" that cannot close against the very next delimiter is skipped as
    an ordinary character, not treated as an opener and not allowed to end
    the scan -- otherwise a price consumes half of the following display
    delimiter and every later repair is lost."""
    assert repair_vlm_json_escape_damage("Cost $5. Formula $$\tau^2$$") == (
        "Cost $5. Formula " + r"$$\tau^2$$"
    )
    # The span must also stay narrow: prose between the price and the real
    # formula is not math and must not be rewritten along with it.
    assert repair_vlm_json_escape_damage("Cost $5 and\text is $\tau$") == (
        "Cost $5 and\text is " + r"$\tau$"
    )


@pytest.mark.offline
def test_markdown_code_is_not_scanned_for_math():
    """Code quotes dollars for its own reasons -- two shell variable
    expansions pair exactly as neatly as a formula does -- so a fenced block
    or an inline code span is copied through verbatim."""
    fenced = '```sh\necho "$HOME"\n\text=1\necho "$PATH"\n```'
    assert repair_vlm_json_escape_damage(fenced) == fenced
    inline = 'inline `echo "$HOME"; \text=1; echo "$PATH"` done'
    assert repair_vlm_json_escape_damage(inline) == inline
    # An unclosed fence protects the rest of the text (CommonMark reads it
    # the same way); an unclosed single backtick protects nothing, so one
    # stray backtick cannot suppress every later repair.
    unclosed_fence = "text\n```sh\n$A \text $B\n"
    assert repair_vlm_json_escape_damage(unclosed_fence) == unclosed_fence
    assert repair_vlm_json_escape_damage("stray ` tick then $\tau$ ok") == (
        "stray ` tick then " + r"$\tau$" + " ok"
    )


@pytest.mark.offline
def test_math_after_a_code_region_is_still_repaired():
    """Protecting code must not end the scan: spans are repaired in every
    region between code, not just before the first one."""
    damaged = '```sh\necho "$HOME"\n```\nthen $\tau$ here'
    assert repair_vlm_json_escape_damage(damaged) == (
        '```sh\necho "$HOME"\n```\nthen ' + r"$\tau$" + " here"
    )


@pytest.mark.offline
def test_unmarked_code_is_indistinguishable_from_math():
    """Stability test for the one accepted rewrite: shell written as plain
    prose, with no fence and no backticks, pairs its two variable expansions
    and the text between them is rewritten. No delimiter rule can tell this
    from "$x ... $y$"; marking code as code is the remedy. Pinned so the
    limit is a decision rather than a surprise."""
    assert repair_vlm_json_escape_damage('echo "$HOME"; \text=1; echo "$PATH"') == (
        'echo "$HOME"; ' + r"\text" + '=1; echo "$PATH"'
    )


@pytest.mark.offline
def test_unclosed_display_delimiter_is_skipped_whole():
    """A "$$" with no display closer must be skipped as one token. Skipping
    only its first dollar leaves the second one free to open an inline span
    against the next single "$", which rewrites the prose in between -- the
    same corruption as pairing across a span boundary, entered through a
    malformed delimiter."""
    damaged = "Unclosed $$x and\text$ suffix"
    assert repair_vlm_json_escape_damage(damaged) == damaged
    # An inline opener is still reached after the failed display delimiter.
    assert repair_vlm_json_escape_damage("$$ broken and $\tau$ here") == (
        "$$ broken and " + r"$\tau$" + " here"
    )


@pytest.mark.offline
def test_display_math_with_newlines_is_repaired():
    """Display math is routinely formatted across lines; the "$$" branch must
    keep repairing it even though a newline is not a valid inline opener."""
    assert (
        repair_vlm_json_escape_damage("$$\n\tau^2\n$$") == "$$\n" + r"\tau^2" + "\n$$"
    )
    assert repair_vlm_json_escape_damage("$$\n\nabla f$$") == "$$\n" + r"\nabla f$$"


@pytest.mark.offline
def test_padded_inline_span_is_not_math(caplog, _propagate_lightrag_logger):
    """Documented residue: "$ x $" has whitespace after the opener, so like
    Pandoc the scanner does not read it as math. The damage inside is warned
    about, never rewritten -- a warned miss beats a silent prose rewrite."""
    damaged = "$ \tau $"
    with caplog.at_level(logging.WARNING, logger="lightrag"):
        assert repair_vlm_json_escape_damage(damaged) == damaged
    assert any("not auto-repaired" in rec.message for rec in caplog.records)


@pytest.mark.offline
def test_stray_dollar_against_math_without_spaces_is_a_warned_miss(
    caplog, _propagate_lightrag_logger
):
    """Stability test for the other documented residue: with no spaces to
    separate them (Chinese text), a stray dollar pairs with the formula's
    opener and the repair is missed. Pinned so a future pairing change has
    to face the trade-off rather than silently flip to rewriting prose.
    English prose survives the same shape, because the space in front of the
    formula stops the price from closing against its opener."""
    damaged = "价格$5，公式$\tau$为"
    with caplog.at_level(logging.WARNING, logger="lightrag"):
        assert repair_vlm_json_escape_damage(damaged) == damaged
    assert any("not auto-repaired" in rec.message for rec in caplog.records)


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
