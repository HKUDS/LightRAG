"""Tests for the space that removing inner quotes leaves in an entity name.

``normalize_extracted_info`` removes the space between Chinese characters, and
between Chinese and Latin text, before it removes the quotes next to Chinese
characters. In ``Apple "苹果"`` the space sits next to the quote, so no rule
matched it: the quoted spelling became ``Apple 苹果`` and the unquoted one
``Apple苹果``, two names for one entity. The space rules now run once more on
the final single-line result.

A change to what a name normalizes to may only merge names that used to
differ, never split names that used to be equal (see "Entity name normalization
changes" in docs/design/PurgeRecoveryContract.md).
``_name_before_the_second_pass`` is the baseline for that.
"""

from __future__ import annotations

import itertools
import re

import pytest

from lightrag.utils import normalize_entity_name, sanitize_text_for_encoding

pytestmark = pytest.mark.offline


@pytest.mark.parametrize(
    "spelling, expected",
    [
        ("Apple 苹果", "Apple苹果"),
        ('Apple "苹果"', "Apple苹果"),
        ("Apple '苹果'", "Apple苹果"),
        ("Apple “苹果”", "Apple苹果"),
        ("Apple ‘苹果’", "Apple苹果"),
        ('"苹果" Apple', "苹果Apple"),
        ("'苹果' Apple", "苹果Apple"),
        ("“苹果” Apple", "苹果Apple"),
        ("苹果 公司", "苹果公司"),
        ('"苹果" 公司', "苹果公司"),
        ('苹果 "公司"', "苹果公司"),
        ("‘苹果’ 公司", "苹果公司"),
    ],
)
def test_a_quoted_spelling_lands_on_the_unquoted_name(spelling, expected):
    name = normalize_entity_name(spelling)

    assert name == expected
    assert normalize_entity_name(name) == name


def test_a_space_next_to_a_quote_that_stays_is_kept():
    # No Chinese character touches these quotes, so they are not removed and
    # the space next to them is still not one the space rules match.
    assert normalize_entity_name('Apple "Inc" 苹果') == 'Apple "Inc" 苹果'
    assert normalize_entity_name("Apple Inc") == "Apple Inc"


def _name_before_the_second_pass(raw: str) -> str:
    """``normalize_entity_name`` as it was before the second pass. Frozen.

    A copy of the single-line path of ``normalize_extracted_info`` with
    ``remove_inner_quotes=True``, taken at 75d6a0c. It is the "old result" of
    the merge-only rule, so it must not follow later changes to the function.
    """
    name = sanitize_text_for_encoding(raw)
    if not name:
        return ""

    name = re.sub(r"</p\s*>|<p\s*/?>", "", name, flags=re.IGNORECASE)
    name = re.sub(r"</br\s*>|<br\s*/?>", "", name, flags=re.IGNORECASE)
    name = name.translate(
        str.maketrans(
            "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ",
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
        )
    )
    name = name.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    name = name.replace("－", "-")
    name = name.replace("＋", "+")
    name = name.replace("／", "/")
    name = name.replace("＊", "*")
    name = name.replace("（", "(").replace("）", ")")
    name = name.replace("—", "-").replace("－", "-")
    name = name.replace("　", " ")

    name = re.sub(r"(?<=[\u4e00-\u9fa5])\s+(?=[\u4e00-\u9fa5])", "", name)
    name = re.sub(
        r"(?<=[\u4e00-\u9fa5])\s+(?=[a-zA-Z0-9\(\)\[\]@#$%!&\*\-=+_])", "", name
    )
    name = re.sub(
        r"(?<=[a-zA-Z0-9\(\)\[\]@#$%!&\*\-=+_])\s+(?=[\u4e00-\u9fa5])", "", name
    )

    if len(name) >= 2:
        if name.startswith('"') and name.endswith('"'):
            inner_content = name[1:-1]
            if '"' not in inner_content:
                name = inner_content
        if name.startswith("'") and name.endswith("'"):
            inner_content = name[1:-1]
            if "'" not in inner_content:
                name = inner_content
        if name.startswith("“") and name.endswith("”"):
            inner_content = name[1:-1]
            if "“" not in inner_content and "”" not in inner_content:
                name = inner_content
        if name.startswith("‘") and name.endswith("’"):
            inner_content = name[1:-1]
            if "‘" not in inner_content and "’" not in inner_content:
                name = inner_content
        if name.startswith("《") and name.endswith("》"):
            inner_content = name[1:-1]
            if "《" not in inner_content and "》" not in inner_content:
                name = inner_content

    name = name.replace("“", "").replace("”", "").replace("‘", "").replace("’", "")
    name = re.sub(r"['\"]+(?=[\u4e00-\u9fa5])", "", name)
    name = re.sub(r"(?<=[\u4e00-\u9fa5])['\"]+", "", name)
    name = name.replace("\u00a0", " ")
    name = re.sub(r"(?<=[^\d])\u202F", " ", name)

    name = re.sub(r"[ \t\n\r\x85\u2028\u2029]+", " ", name)
    name = name.strip()

    if len(name) < 3 and re.match(r"^[0-9]+$", name):
        return ""
    if len(name) < 6 and "." in name and all(c.isdigit() or c == "." for c in name):
        return ""
    return name


# Latin, digit and Chinese text, a few things the normalizer rewrites, and
# every quote it removes. Four pieces reach both sides of the bug: 'A "苹' has
# the space before the quote, '苹" A' has it after.
_PIECES = ["A", "1", "苹", " ", "(", ".", "<br>", "\n", *"\"'“”‘’《》"]

# Every kind of whitespace the rules treat differently: the second pass matches
# any \s, the collapse only spaces, tabs and line breaks, U+00A0 becomes a
# space, U+202F only after a non-digit, U+3000 before the first pass.
_WHITESPACE_PIECES = ["A", "1", "苹", " ", "\t", "\u00a0", "\u202f", "\u3000", '"', "“"]


def _generated_names(pieces):
    for length in range(1, 5):
        for combination in itertools.product(pieces, repeat=length):
            yield "".join(combination)


@pytest.mark.parametrize(
    "pieces", [_PIECES, _WHITESPACE_PIECES], ids=["quotes", "whitespace"]
)
def test_the_second_pass_only_merges_names(pieces):
    new_name_by_old_name: dict[str, str] = {}
    for raw in _generated_names(pieces):
        old_name = _name_before_the_second_pass(raw)
        new_name = normalize_entity_name(raw)

        # Two spellings that shared a name must still share one.
        assert new_name_by_old_name.setdefault(old_name, new_name) == new_name, raw

    # ...and the pass did merge some: the check above is not vacuous.
    assert len(set(new_name_by_old_name.values())) < len(new_name_by_old_name)
