"""Deterministic smoke infrastructure: hash embeddings and a corpus-aware LLM.

SYNTHETIC BY DESIGN. These fakes prove harness mechanics, never retrieval
quality; any report produced with them must carry the synthetic label. The
hash embedding is the default-off test's bag-of-tokens projection; the fake LLM answers the
three prompt families the pipeline and query paths send it, and records every
call so a test can assert which paths fired.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from functools import lru_cache

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")
_INPUT_TEXT_RE = re.compile(
    r"---Input Text---\s*```\s*\n(.*?)\n```", re.DOTALL
)
_QUERY_RE = re.compile(r"User Query: (.+)")


@lru_cache(maxsize=16384)
def _token_vector(token: str, dim: int) -> tuple[float, ...]:
    # One sha256 block covers 32 dims; larger dims concatenate numbered
    # blocks. Must stay identical to write_amp_driver._token_vector.
    values: list[float] = []
    block = 0
    while len(values) < dim:
        digest = hashlib.sha256(f"{block}:{token}".encode("utf-8")).digest()
        values.extend(b / 127.5 - 1.0 for b in digest)
        block += 1
    return tuple(values[:dim])


def hash_embed_texts(texts: list[str], dim: int) -> np.ndarray:
    """Bag-of-tokens sha256 projection, L2-normalized, float32."""
    rows = []
    for text in texts:
        normalized = unicodedata.normalize("NFKC", text).lower()
        vec = np.zeros(dim, dtype=np.float64)
        for token in _TOKEN_RE.findall(normalized):
            vec += np.asarray(_token_vector(token, dim), dtype=np.float64)
        norm = np.linalg.norm(vec)
        rows.append(vec / norm if norm > 0 else vec)
    return np.asarray(rows, dtype=np.float32)


class FakeLLM:
    """Corpus-aware stand-in for the answer/extraction/keyword LLM.

    Rules:
    - Prompt families are recognized by CONTENT, not by the deprecated
      entity_extraction/keyword_extraction flags: the pipeline's extraction
      call no longer forwards them to the model function.
    - Extraction prompts are answered from ``entity_specs``, keyed by the
      exact passage text; an unrecognized input text raises, so a chunking
      change cannot silently produce an empty KG. A continue-extraction
      prompt without an input-text section gets the completion marker only.
    - Keyword prompts return the whole query as both keyword lists, which
      keeps entity seeding deterministic without a real keyword model.
    - Any other prompt gets a canned synthetic answer.
    - Every call is recorded in ``calls`` with its family.
    """

    def __init__(self, entity_specs: dict[str, list[dict[str, str]]]):
        self.entity_specs = entity_specs
        self.calls: list[str] = []

    async def __call__(
        self,
        prompt,
        system_prompt=None,
        history_messages=[],
        keyword_extraction=False,
        entity_extraction=False,
        **kwargs,
    ) -> str:
        text = str(prompt)
        is_extraction = entity_extraction or "Extract entities" in text
        is_keywords = keyword_extraction or "User Query:" in text
        if is_keywords and not is_extraction:
            self.calls.append("keywords")
            match = _QUERY_RE.search(text)
            query = match.group(1).strip() if match else ""
            return json.dumps(
                {
                    "high_level_keywords": [query] if query else [],
                    "low_level_keywords": [query] if query else [],
                }
            )
        if is_extraction:
            self.calls.append("extraction")
            match = _INPUT_TEXT_RE.search(text)
            if match is None:
                return "<|COMPLETE|>"
            input_text = match.group(1)
            specs = self.entity_specs.get(input_text)
            if specs is None:
                raise AssertionError(
                    "FakeLLM got an unrecognized extraction input text: "
                    f"{input_text[:80]!r}"
                )
            rows = [
                "entity<|#|>{entity_name}<|#|>{entity_type}<|#|>{description}".format(
                    **spec
                )
                for spec in specs
            ]
            return "\n".join(rows + ["<|COMPLETE|>"])
        self.calls.append("answer")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        return f"[synthetic answer {digest}]"