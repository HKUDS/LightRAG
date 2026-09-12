"""Stage 2's output order must reach stage 3 intact.

``_apply_token_truncation`` returns two views of the same selection: the
``*_context`` lists that go into the prompt, and the ``filtered_*`` lists of
original records that go into chunk selection. It used to build the second pair
by filtering ``final_entities`` / ``final_relations`` (stage-1 retrieval order)
through a membership test, so the ``filtered_*`` order was stage-1's, not its
own.

That was invisible while stage 2 only did prefix truncation -- a prefix of
stage-1 order filtered by membership reproduces stage-1 order exactly -- but
stage 3 treats the order as an importance ranking in two load-bearing places:
``_find_related_text_unit_from_*`` attributes a shared chunk to the
earlier-positioned record, and ``pick_by_weighted_polling`` allocates a chunk
quota that decreases with list position. A stage-2 selector that reorders (a
reranker, say) would have reached the prompt but not the chunk budget.

These tests pin the invariant by monkeypatching the truncation step into a
selector that reorders, which is the only way to observe it.
"""

import pytest

from lightrag.base import QueryParam
from lightrag.operate import _apply_token_truncation

pytestmark = pytest.mark.offline


GLOBAL_CONFIG = {"tokenizer": object()}


def _entity(name: str, **extra) -> dict:
    record = {
        "entity_name": name,
        "entity_type": "person",
        "description": f"description of {name}",
        "file_path": f"{name}.txt",
    }
    record.update(extra)
    return record


def _relation(src: str, tgt: str, **extra) -> dict:
    record = {
        "src_id": src,
        "tgt_id": tgt,
        "description": f"{src} -> {tgt}",
        "file_path": f"{src}-{tgt}.txt",
    }
    record.update(extra)
    return record


def _search_result(entities: list[dict], relations: list[dict]) -> dict:
    return {"final_entities": entities, "final_relations": relations}


def _install_selector(monkeypatch, selector):
    """Replace the truncation step with ``selector(list_data) -> list_data``."""

    async def fake_atruncate_list_by_token_size(list_data, **kwargs):
        return selector(list_data)

    monkeypatch.setattr(
        "lightrag.operate.atruncate_list_by_token_size",
        fake_atruncate_list_by_token_size,
    )


def _names(entities: list[dict]) -> list[str]:
    return [e["entity_name"] for e in entities]


def _pairs(relations: list[dict]) -> list[tuple[str, str]]:
    return [(r["src_id"], r["tgt_id"]) for r in relations]


async def test_reordering_selector_propagates_into_filtered_lists(monkeypatch):
    """A stage-2 reorder must show up in ``filtered_*``, not just the prompt."""
    _install_selector(monkeypatch, lambda list_data: list(reversed(list_data)))

    result = await _apply_token_truncation(
        _search_result(
            [_entity("alice"), _entity("bob"), _entity("carol")],
            [_relation("alice", "bob"), _relation("bob", "carol")],
        ),
        QueryParam(),
        GLOBAL_CONFIG,
    )

    assert [e["entity"] for e in result["entities_context"]] == [
        "carol",
        "bob",
        "alice",
    ]
    assert _names(result["filtered_entities"]) == ["carol", "bob", "alice"]

    assert [(r["entity1"], r["entity2"]) for r in result["relations_context"]] == [
        ("bob", "carol"),
        ("alice", "bob"),
    ]
    assert _pairs(result["filtered_relations"]) == [
        ("bob", "carol"),
        ("alice", "bob"),
    ]


async def test_reordering_selector_that_also_drops_records(monkeypatch):
    """Reorder plus selection: order follows stage 2, membership follows it too."""
    _install_selector(monkeypatch, lambda list_data: [list_data[2], list_data[0]])

    result = await _apply_token_truncation(
        _search_result(
            [_entity("alice"), _entity("bob"), _entity("carol")],
            [
                _relation("alice", "bob"),
                _relation("bob", "carol"),
                _relation("carol", "dave"),
            ],
        ),
        QueryParam(),
        GLOBAL_CONFIG,
    )

    assert _names(result["filtered_entities"]) == ["carol", "alice"]
    assert _pairs(result["filtered_relations"]) == [
        ("carol", "dave"),
        ("alice", "bob"),
    ]
    assert list(result["entity_id_to_original"]) == ["carol", "alice"]
    assert list(result["relation_id_to_original"]) == [
        ("carol", "dave"),
        ("alice", "bob"),
    ]


async def test_prefix_truncation_is_unchanged(monkeypatch):
    """The shipped default path -- a prefix -- keeps stage-1 order, as before."""
    _install_selector(monkeypatch, lambda list_data: list_data[:2])

    result = await _apply_token_truncation(
        _search_result(
            [_entity("alice"), _entity("bob"), _entity("carol")],
            [
                _relation("alice", "bob"),
                _relation("bob", "carol"),
                _relation("carol", "dave"),
            ],
        ),
        QueryParam(),
        GLOBAL_CONFIG,
    )

    assert _names(result["filtered_entities"]) == ["alice", "bob"]
    assert _pairs(result["filtered_relations"]) == [
        ("alice", "bob"),
        ("bob", "carol"),
    ]


async def test_non_prefix_subset_without_reorder(monkeypatch):
    """A packing selector that only drops records still resolves correctly."""
    _install_selector(monkeypatch, lambda list_data: [list_data[0], list_data[2]])

    result = await _apply_token_truncation(
        _search_result(
            [_entity("alice"), _entity("bob"), _entity("carol")],
            [],
        ),
        QueryParam(),
        GLOBAL_CONFIG,
    )

    assert _names(result["filtered_entities"]) == ["alice", "carol"]
    assert result["filtered_relations"] == []
    assert result["relations_context"] == []


async def test_duplicate_names_keep_the_first_retrieved_record(monkeypatch):
    """Dedup resolves to the first occurrence in stage-1 order, as it always did.

    Stage 2 decides *order*; it does not get to promote a later duplicate of a
    name over the record that was actually retrieved first.
    """
    _install_selector(monkeypatch, lambda list_data: list(reversed(list_data)))

    result = await _apply_token_truncation(
        _search_result(
            [
                _entity("alice", description="first alice"),
                _entity("bob"),
                _entity("alice", description="second alice"),
            ],
            [
                _relation("alice", "bob", description="first edge"),
                _relation("alice", "bob", description="second edge"),
            ],
        ),
        QueryParam(),
        GLOBAL_CONFIG,
    )

    assert _names(result["filtered_entities"]) == ["alice", "bob"]
    assert result["filtered_entities"][0]["description"] == "first alice"
    assert result["entity_id_to_original"]["alice"]["description"] == "first alice"

    assert _pairs(result["filtered_relations"]) == [("alice", "bob")]
    assert result["filtered_relations"][0]["description"] == "first edge"


async def test_src_tgt_relation_format_survives_reordering(monkeypatch):
    """Relations carrying ``src_tgt`` instead of ``src_id``/``tgt_id`` resolve too."""
    _install_selector(monkeypatch, lambda list_data: list(reversed(list_data)))

    result = await _apply_token_truncation(
        _search_result(
            [],
            [
                {"src_tgt": ("alice", "bob"), "description": "a-b"},
                {"src_tgt": ("bob", "carol"), "description": "b-c"},
            ],
        ),
        QueryParam(),
        GLOBAL_CONFIG,
    )

    assert [r["src_tgt"] for r in result["filtered_relations"]] == [
        ("bob", "carol"),
        ("alice", "bob"),
    ]
    assert result["filtered_entities"] == []
    assert result["entities_context"] == []


async def test_missing_tokenizer_short_circuits(monkeypatch):
    """No tokenizer: nothing is truncated and the originals pass straight through."""
    entities = [_entity("alice")]
    relations = [_relation("alice", "bob")]

    result = await _apply_token_truncation(
        _search_result(entities, relations),
        QueryParam(),
        {},
    )

    assert result["entities_context"] == []
    assert result["relations_context"] == []
    assert result["filtered_entities"] is entities
    assert result["filtered_relations"] is relations
