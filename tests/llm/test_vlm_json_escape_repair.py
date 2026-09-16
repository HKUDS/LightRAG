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

from lightrag.utils import (
    _WS_LATEX_MATH_PATTERN,
    _WS_LATEX_SUSPECT_PATTERN,
    repair_vlm_json_escape_damage,
)


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
    "damaged",
    [
        # CJK ideographs: the shape Chinese corpora actually produce, where a
        # model writes an inline formula into prose without $ delimiters.
        "阈值\tau为0.5",
        "数据\times倍",
        "算子\nabla作用于 f",
        "密度\rho的分布",
        # Non-ASCII that is not CJK: a Greek letter is a word character too.
        "Δ\theta角",
    ],
)
@pytest.mark.offline
def test_damage_followed_by_a_non_ascii_character_is_reported(
    damaged, caplog, _propagate_lightrag_logger
):
    """A word boundary alone hid every one of these: Python's ``re`` counts a
    CJK ideograph as a word character, so no boundary exists between the
    residue and what follows it. Outside math nothing is rewritten, which
    makes the WARNING the only trace the command was ever there."""
    with caplog.at_level(logging.WARNING, logger="lightrag"):
        result = repair_vlm_json_escape_damage(damaged, context="table/t1")
    assert result == damaged
    assert any(
        "not auto-repaired" in rec.message and "table/t1" in rec.getMessage()
        for rec in caplog.records
    )


@pytest.mark.parametrize(
    "damaged",
    [
        "阈值\tau。",  # ideographic full stop
        "参数\tilde，如下",  # full-width comma
        "the \tau value",  # space
        "label:\tau",  # end of string
        "$5 costs \rho.",  # ASCII punctuation
    ],
)
@pytest.mark.offline
def test_word_boundary_shapes_are_still_reported(
    damaged, caplog, _propagate_lightrag_logger
):
    r"""Widening the guard must not cost the cases ``\b`` already caught."""
    with caplog.at_level(logging.WARNING, logger="lightrag"):
        result = repair_vlm_json_escape_damage(damaged, context="table/t1")
    assert result == damaged
    assert any("not auto-repaired" in rec.message for rec in caplog.records)


@pytest.mark.parametrize(
    "legit",
    [
        "col1\tauthor list",  # residue starts an English word
        "col\text_id header",  # ASCII underscore follows
        "the \tau2 value",  # ASCII digit follows
        "row\rightmost column",
    ],
)
@pytest.mark.offline
def test_ascii_word_characters_after_the_residue_stay_silent(
    legit, caplog, _propagate_lightrag_logger
):
    """The accepted gap: in tab-separated data these are plausible values, so
    only NON-ASCII is admitted as the alternative to a word boundary."""
    with caplog.at_level(logging.WARNING, logger="lightrag"):
        result = repair_vlm_json_escape_damage(legit)
    assert result == legit
    assert not caplog.records


@pytest.mark.offline
def test_in_math_pattern_stays_a_superset_of_the_prose_pattern():
    """The contract's reason a repaired span never re-triggers the prose
    warning. Widening the prose guard is only safe while this holds."""
    probes = [
        f"{residue}{tail}"
        for residue, tails in (
            ("\t", ("au", "heta", "imes", "ext", "ilde", "herefore", "riangle")),
            ("\r", ("ho", "ight", "angle", "ceil")),
            ("\n", ("abla", "otin")),
        )
        for tail in tails
    ]
    followers = ["", " ", ".", "_", "2", "x", "为", "。", "Δ"]
    for probe in probes:
        for follower in followers:
            text = f"prefix{probe}{follower}"
            if _WS_LATEX_SUSPECT_PATTERN.search(text):
                assert _WS_LATEX_MATH_PATTERN.search(text), text


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
def test_fence_like_line_inside_a_block_is_not_a_closer():
    """A closing fence may carry only whitespace after it -- an info string
    is the opening fence's privilege. Accepting one on the closer lets a
    fence-like line inside the block end the protected region early and the
    rest of the code gets scanned as math."""
    damaged = '```sh\n```not a closer\necho "$HOME"\n\text=1\necho "$PATH"\n```'
    assert repair_vlm_json_escape_damage(damaged) == damaged
    # The block still ends where it really ends.
    assert repair_vlm_json_escape_damage(damaged + "\nafter $\tau$") == (
        damaged + "\nafter " + r"$\tau$"
    )


@pytest.mark.parametrize(
    ("damaged", "expected"),
    [
        ("$\nabla f$", "$" + r"\nabla" + " f$"),
        ("$x \notin A$", "$x " + r"\notin" + " A$"),
        ("$\rho + 1$", "$" + r"\rho" + " + 1$"),
        ("$x \right)$", "$x " + r"\right" + ")$"),
    ],
)
@pytest.mark.offline
def test_line_break_commands_survive_the_inline_veto(damaged, expected):
    """Every "\r" and "\n" residue is the damage AND a line break at once: a
    decoded "\nabla" IS a newline followed by "abla", a decoded "\rho" IS a
    carriage return followed by "ho". A blanket line-break veto on inline
    spans rejects exactly the damage the gate exists to let through, so the
    exemption covers the whole residue whitelist, not one line-ending."""
    assert repair_vlm_json_escape_damage(damaged) == expected


@pytest.mark.parametrize("line_break", ["\n", "\r", "\r\n"], ids=["lf", "cr", "crlf"])
@pytest.mark.offline
def test_inline_veto_covers_every_line_ending(line_break):
    """The veto tests a character class, not one example. Looking only at
    "\n" let the same text pass or fail on line ending alone -- CR-separated
    shell was rewritten while its LF twin was correctly left alone."""
    code = f"A=$X{line_break}\text=1{line_break}B=$Y"
    assert repair_vlm_json_escape_damage(code) == code
    prose = f"see $a +{line_break}\tau$ end"
    assert repair_vlm_json_escape_damage(prose) == prose
    # Display math stays exempt from the line-break limit on every ending.
    assert repair_vlm_json_escape_damage(f"$$a +{line_break}\tau$$") == (
        f"$$a +{line_break}" + r"\tau$$"
    )


@pytest.mark.offline
def test_fence_indentation_is_spaces_only():
    """CommonMark allows at most three leading SPACES before a fence; a tab
    advances to the fourth column and is code content. Accepting one lets a
    tab-indented fence-like line close a real block early, and the code
    after it is then scanned as math -- a rewrite, not a miss."""
    damaged = "```sh\n\t```\nA=$X; \text=1; B=$Y\n```"
    assert repair_vlm_json_escape_damage(damaged) == damaged
    # Three spaces are still legal indentation for a real fence.
    indented = "   ```sh\n   A=$X; \text=1; B=$Y\n   ```"
    assert repair_vlm_json_escape_damage(indented) == indented


@pytest.mark.offline
def test_backtick_fence_info_string_may_not_carry_a_backtick():
    """CommonMark forbids a backtick inside a backtick fence's info string,
    to keep it unambiguous with an inline span; a tilde fence allows it.
    Reading such a line as an opener leaves the block unclosed, so the
    protection runs to the end of the text and costs every repair after it.
    """
    assert repair_vlm_json_escape_damage("```foo`bar\nafter $\tau$") == (
        "```foo`bar\nafter " + r"$\tau$"
    )
    # A real opener with no closer does protect the rest -- that IS
    # CommonMark -- and a tilde fence may carry the backtick.
    for protected in ("```foo\nafter $\tau$", "~~~foo`bar\nafter $\tau$"):
        assert repair_vlm_json_escape_damage(protected) == protected


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
@pytest.mark.offline
def test_fence_regions_behave_the_same_on_both_line_endings(newline):
    """MULTILINE "$" matches before the "\n", with a CRLF's "\r" still ahead
    of it, so a closer suffix of only spaces and tabs never matches on
    Windows line endings. The block then looks unclosed and protects the
    whole rest of the text, costing every repair after it."""
    block = newline.join(["```sh", "a", "```"])
    assert repair_vlm_json_escape_damage(block + newline + "after $\tau$") == (
        block + newline + "after " + r"$\tau$"
    )
    # The block itself stays protected on both line endings.
    code = newline.join(["```sh", 'echo "$HOME"', "\text=1", 'echo "$PATH"', "```"])
    assert repair_vlm_json_escape_damage(code) == code


@pytest.mark.offline
def test_closing_fence_may_be_longer_than_the_opener():
    """CommonMark lets the closing fence exceed the opening one. Requiring
    an exact length match leaves the block unterminated, which protects the
    whole rest of the text and loses every repair after it."""
    fenced = '````sh\necho "$A"\n\text=1\necho "$B"\n`````'
    assert repair_vlm_json_escape_damage(fenced + "\nthen $\tau$") == (
        fenced + "\nthen " + r"$\tau$"
    )


@pytest.mark.offline
def test_inline_span_closer_is_bounded_on_both_sides():
    """The closing backtick run must match the opening one exactly, so it is
    bounded on both sides. Guarding only its right let the backreference land
    inside a longer run within the span, ending the protected region early
    and exposing the rest of the code to the scanner."""
    damaged = 'see `A ``B "$HOME" \text=1 "$PATH"` end'
    assert repair_vlm_json_escape_damage(damaged) == damaged


@pytest.mark.offline
def test_math_after_a_code_region_is_still_repaired():
    """Protecting code must not end the scan: spans are repaired in every
    region between code, not just before the first one."""
    damaged = '```sh\necho "$HOME"\n```\nthen $\tau$ here'
    assert repair_vlm_json_escape_damage(damaged) == (
        '```sh\necho "$HOME"\n```\nthen ' + r"$\tau$" + " here"
    )


@pytest.mark.parametrize(
    "damaged",
    [
        # Unmarked: no fence, no backticks -- nothing identifies it as code,
        # so only the span's own content can.
        'echo "$HOME"; \text=1; echo "$PATH"',
        # Marked, but reached through delimiters the region pass gets wrong:
        # a stray backtick before a fence, and a four-space indented block.
        'stray ` tick\n```sh\nrun `cmd` now\necho "$HOME"\n\text=1\necho "$PATH"\n```',
        'Example:\n\n    echo "$HOME"\n    \text=1\n    echo "$PATH"\n\ndone',
        # Unquoted, but spanning lines -- no inline formula does.
        "A=$X\n\text=1\nB=$Y",
    ],
)
@pytest.mark.offline
def test_code_content_is_not_repaired_however_it_is_reached(damaged):
    """Pairing says WHERE a span is; the content gate says whether it is math.

    Dollars pair in code for reasons of their own (two "$VAR" expansions
    satisfy every delimiter rule), so enumerating the Markdown constructs
    that contain code is a blacklist that never closes -- each of these
    inputs reaches the scanner through a different one. A body carrying a
    double quote or a backtick, or an inline span that spans lines, is code
    whichever construct delivered it.
    """
    assert repair_vlm_json_escape_damage(damaged) == damaged


@pytest.mark.parametrize("stray", ["`", "\\`"], ids=["bare", "escaped"])
@pytest.mark.offline
def test_a_stray_backtick_cannot_steal_a_fence_opener(stray):
    """CommonMark settles block structure before inline spans, so an inline
    span can never cross a fence boundary. One alternation cannot say that --
    the inline branch wins by position, so a backtick anywhere earlier paired
    with one inside the block, swallowed the opening fence, and handed the
    rest of the code to the scanner. Parametrized over an escaped backtick
    too, to record that the escape is not the mechanism: a bare one does it.
    """
    damaged = (
        f"A stray {stray} tick.\n\n```sh\necho `date` ; A=$X; \text=1; B=$Y\n```\n"
    )
    assert repair_vlm_json_escape_damage(damaged) == damaged


@pytest.mark.parametrize(
    "blank",
    ["\n\n", "\r\n\r\n", "\n   \n"],
    ids=["lf", "crlf", "spaces-only"],
)
@pytest.mark.offline
def test_a_stray_backtick_cannot_cross_a_blank_line(blank):
    """The other block boundary. A code span is an inline inside ONE leaf
    block, so backtick runs in two paragraphs cannot pair -- and letting them
    pair had the same cost as letting them cross a fence: the stray one ate
    the real span's opener, leaving that span's closer unmatched and its code
    exposed. CRLF is parametrized because a blank line written "\\n[ \\t]*\\n"
    misses it, the same trap a closing fence fell into once.
    """
    damaged = f"A stray ` tick.{blank}Run `A=$X; \text=1; B=$Y` now."
    assert repair_vlm_json_escape_damage(damaged) == damaged


@pytest.mark.offline
def test_an_inline_span_may_still_cross_a_single_line_break():
    """The paragraph rule is a blank line, not any line break -- CommonMark
    lets a code span run across an ordinary one, and this is what keeps
    wrapped shell inside a real span protected."""
    damaged = "Run `A=$X;\n\text=1; B=$Y` now."
    assert repair_vlm_json_escape_damage(damaged) == damaged


@pytest.mark.offline
def test_paragraphs_no_longer_disappear_between_two_stray_backticks():
    """Miss side of the same rule: two stray backticks paragraphs apart used
    to swallow everything between them, real math included."""
    damaged = "A stray ` tick.\n\nvalue $\tau$ here\n\nanother ` tick."
    assert repair_vlm_json_escape_damage(damaged) == (
        "A stray ` tick.\n\nvalue $" + r"\tau" + "$ here\n\nanother ` tick."
    )


@pytest.mark.offline
def test_an_escaped_backtick_does_not_open_an_inline_span():
    """``\\` `` is a literal backtick. Treating it as an opener closed the span
    on the NEXT run -- the real code span's opener -- leaving that span's own
    closer as an unclosed single backtick, which protects nothing.
    """
    damaged = "Type \\`x then `A=$X; \text=1; B=$Y`"
    assert repair_vlm_json_escape_damage(damaged) == damaged


@pytest.mark.offline
def test_a_doubled_backslash_still_leaves_a_real_opener():
    """Parity, not a bare lookbehind: an even run of backslashes is a literal
    backslash and the backtick after it opens a span for real."""
    damaged = "Path C:\\\\ then `A=$X; \text=1; B=$Y`"
    assert repair_vlm_json_escape_damage(damaged) == damaged


@pytest.mark.offline
def test_an_escaped_backtick_does_not_close_an_inline_span_either():
    """Stability test for the half of the escape rule deliberately NOT
    applied. CommonMark gives backslash escapes no effect inside a code span,
    so ``\\` `` closes it. Honouring the escape there would leave the span
    unclosed and hand its code to the scanner -- protection turned into a
    rewrite, which is the direction this whole path is organized against.
    """
    damaged = "`A=$X; \text=1; B=$Y\\`"
    assert repair_vlm_json_escape_damage(damaged) == damaged


@pytest.mark.offline
def test_escaped_backticks_no_longer_shield_prose_between_them():
    """The other side of the opener rule: two escaped backticks are literal
    text, not a code span, so math between them is reached and repaired."""
    assert repair_vlm_json_escape_damage("Use \\`foo, then $\tau$ and \\`bar") == (
        "Use \\`foo, then $" + r"\tau" + "$ and \\`bar"
    )


@pytest.mark.offline
def test_unquoted_single_line_shell_is_still_rewritten():
    """Stability test for what the gate does NOT cover: one line, no quotes,
    no backticks -- nothing separates it from "$x ... $y$". Pinned so the
    remaining exposure is a known boundary rather than a surprise."""
    assert repair_vlm_json_escape_damage("A=$X; \text=1; B=$Y") == (
        "A=$X; " + r"\text" + "=1; B=$Y"
    )


@pytest.mark.parametrize(
    "damaged",
    [
        '$\text{"x"} + \tau$',  # a double quote inside real math
        "see $a +\n\tau$ end",  # an inline span across a line break
    ],
)
@pytest.mark.offline
def test_accepted_cost_of_the_content_gate(damaged):
    """The gate's price, pinned: real math carrying a quote, and inline math
    written across a line break, are no longer repaired. Display math is
    exempt from the line-break limit, so a multi-line "$$...$$" still is."""
    assert repair_vlm_json_escape_damage(damaged) == damaged
    assert repair_vlm_json_escape_damage("$$a +\n\tau$$") == "$$a +\n" + r"\tau$$"


@pytest.mark.offline
def test_unmatched_backtick_run_does_not_open_a_span():
    """An opening run is bounded on both sides like a closing one. Without a
    left guard the regex restarts inside a longer unmatched run and swallows
    the math after it, costing a repair CommonMark never protected."""
    assert repair_vlm_json_escape_damage("x `` unmatched $\tau$ then ` end") == (
        "x `` unmatched " + r"$\tau$" + " then ` end"
    )


@pytest.mark.offline
def test_repair_log_names_what_it_rewrote(caplog, _propagate_lightrag_logger):
    """A count alone cannot be checked against a corpus. Every rewritten span
    reaches the log, so a wrong rewrite is visible rather than silent."""
    with caplog.at_level(logging.WARNING, logger="lightrag"):
        repair_vlm_json_escape_damage("domain is $\tau^2$", context="table/t1")
    assert any(
        "inside dollar math" in rec.message and r"\\tau^2" in rec.getMessage()
        for rec in caplog.records
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
