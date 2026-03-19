"""Heuristic token estimator – see docx-extraction-guide-zh.md Appendix."""

from __future__ import annotations

import re

_CJK_RE = re.compile(
    r"[\u4e00-\u9fff\u3400-\u4dbf\u2e80-\u2eff\u3000-\u303f"
    r"\uff00-\uffef\u2000-\u206f]"
)
_JSON_STRUCT_RE = re.compile(r'[{}\[\],:"\']')


def estimate_tokens(text: str) -> int:
    """Return a rough token count without calling a real tokenizer.

    Rules (from the guide):
    - CJK chars  ≈ 0.75 token each
    - JSON / HTML structural chars ≈ 1 token each
    - Other chars ≈ 0.4 token each
    - +5 % buffer + 10 fixed offset
    """
    if not text:
        return 0

    cjk_count = len(_CJK_RE.findall(text))
    json_count = len(_JSON_STRUCT_RE.findall(text))
    other_count = len(text) - cjk_count - json_count

    raw = cjk_count * 0.75 + json_count * 1.0 + other_count * 0.4
    return int(raw * 1.05 + 10)
