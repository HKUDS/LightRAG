"""The final space pass of an entity name must run after the length cap.

The extraction parsers cut a name to ``DEFAULT_ENTITY_NAME_MAX_LENGTH``
characters (and a byte limit) after normalizing it. The final space pass of
``normalize_extracted_info`` shortens a name, so running it before the cut gave
two spellings different keys where the cut of the previous rules gave them one:
with ``p = "A" * 255``, ``p + " B"`` and ``p + ' "苹果"'`` were both stored as
``p + " "``, and the second one became ``p + "苹"``. A rebuild then matched the
stored node with only one of its records and dropped the facts of the other.
The parsers now cap first and run the pass on the capped key.

The merge-only rule is in "Entity name normalization changes",
docs/design/PurgeRecoveryContract.md.
"""

from __future__ import annotations

import itertools
import json

import pytest

from lightrag.constants import DEFAULT_ENTITY_NAME_MAX_LENGTH
from lightrag.operate import (
    _finalize_extracted_identifier,
    _process_extraction_result,
    _process_json_extraction_result,
    _truncate_entity_identifier,
)
from lightrag.utils import normalize_entity_name
from tests.utils.test_normalize_entity_name_quoted_chinese import (
    _name_before_the_second_pass,
)

pytestmark = pytest.mark.offline

_PREFIX = "A" * (DEFAULT_ENTITY_NAME_MAX_LENGTH - 1)


def _stored_key_before_the_second_pass(raw: str) -> str:
    return _truncate_entity_identifier(
        _name_before_the_second_pass(raw),
        DEFAULT_ENTITY_NAME_MAX_LENGTH,
        "chunk-1",
        "Entity name",
    )


def _stored_key(raw: str) -> str:
    return _finalize_extracted_identifier(
        normalize_entity_name(raw, final_space_pass=False), "chunk-1", "Entity name"
    )


async def _text_path(entities=(), relations=()):
    records = [
        f"entity<|#|>{name}<|#|>organization<|#|>A company." for name in entities
    ]
    records += [
        f"relation<|#|>{source}<|#|>{target}<|#|>supplies<|#|>One supplies the other."
        for source, target in relations
    ]
    return await _process_extraction_result(
        "\n".join(records) + "\n<|COMPLETE|>", "chunk-1", 1, "doc.txt"
    )


async def _json_path(entities=(), relations=()):
    payload = {
        "entities": [
            {"name": name, "type": "organization", "description": "A company."}
            for name in entities
        ],
        "relationships": [
            {
                "source": source,
                "target": target,
                "keywords": "supplies",
                "description": "One supplies the other.",
            }
            for source, target in relations
        ],
    }
    return await _process_json_extraction_result(
        json.dumps(payload, ensure_ascii=False), "chunk-1", 1, "doc.json"
    )


_PATHS = pytest.mark.parametrize(
    "parse", [_text_path, _json_path], ids=["text", "json"]
)


@pytest.mark.asyncio
@_PATHS
async def test_spellings_cut_to_one_key_still_share_it(parse):
    spellings = [_PREFIX + " B", _PREFIX + ' "苹果"']
    old_key = _stored_key_before_the_second_pass(spellings[0])
    assert {_stored_key_before_the_second_pass(raw) for raw in spellings} == {old_key}

    nodes, _edges = await parse(entities=spellings)

    assert list(nodes) == [old_key]
    assert len(nodes[old_key]) == 2


@pytest.mark.asyncio
@_PATHS
async def test_a_name_within_the_cap_still_gets_the_second_pass(parse):
    nodes, _edges = await parse(entities=['Apple "苹果"', "Apple 苹果"])

    assert list(nodes) == ["Apple苹果"]


@pytest.mark.asyncio
@_PATHS
async def test_endpoints_the_second_pass_makes_equal_are_a_self_loop(parse):
    _nodes, edges = await parse(relations=[('A "苹果"', "A苹果"), ('A "苹果"', "B")])

    assert list(edges) == [("A苹果", "B")]


# Prefixes that put the cut in front of, inside and behind a short tail, by the
# character limit (Latin) and by the byte limit (3-byte Chinese characters).
_PREFIXES = [
    *(
        "A" * n
        for n in range(
            DEFAULT_ENTITY_NAME_MAX_LENGTH - 5, DEFAULT_ENTITY_NAME_MAX_LENGTH + 1
        )
    ),
    *("苹" * n for n in range(167, 172)),
]
_TAIL_PIECES = ["A", "苹", " ", '"', "“"]


def test_the_capped_key_only_merges_names():
    new_key_by_old_key: dict[str, str] = {}
    for prefix in _PREFIXES:
        for length in range(1, 4):
            for pieces in itertools.product(_TAIL_PIECES, repeat=length):
                raw = prefix + "".join(pieces)
                old_key = _stored_key_before_the_second_pass(raw)
                new_key = _stored_key(raw)

                # Two spellings that shared a key must still share one.
                assert new_key_by_old_key.setdefault(old_key, new_key) == new_key, raw

    # ...and the pass did merge some: the check above is not vacuous.
    assert len(set(new_key_by_old_key.values())) < len(new_key_by_old_key)
