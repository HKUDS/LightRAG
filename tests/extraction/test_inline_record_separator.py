"""Records that an LLM puts on one line, separated by the tuple delimiter
instead of a newline, must parse the same as one record per line."""

import pytest

from lightrag.operate import _process_extraction_result

D = "<|#|>"

ALICE = f"entity{D}Alice{D}person{D}Alice is an engineer."
BOB = f"entity{D}Bob{D}person{D}Bob is a manager."
CAROL = f"entity{D}Carol{D}person{D}Carol is a designer."
REL = f"relation{D}Alice{D}Bob{D}reports to{D}Alice reports to Bob."


async def _parse(text: str):
    return await _process_extraction_result(text, "chunk-1", 0)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_entities_on_one_line_are_all_kept():
    multi_line_nodes, multi_line_edges = await _parse(
        "\n".join([ALICE, BOB, CAROL, REL]) + "\n<|COMPLETE|>"
    )
    one_line_nodes, one_line_edges = await _parse(
        D.join([ALICE, BOB, CAROL, REL]) + "\n<|COMPLETE|>"
    )

    assert set(multi_line_nodes) == {"Alice", "Bob", "Carol"}
    assert set(one_line_nodes) == set(multi_line_nodes)
    assert set(one_line_edges) == set(multi_line_edges) == {("Alice", "Bob")}
    assert (
        one_line_nodes["Bob"][0]["description"]
        == multi_line_nodes["Bob"][0]["description"]
    )


@pytest.mark.offline
@pytest.mark.asyncio
async def test_entity_named_like_a_record_keyword_on_one_line():
    relational = f"entity{D}relational database{D}concept{D}A kind of database."
    nodes, _ = await _parse(D.join([ALICE, relational]) + "\n<|COMPLETE|>")

    assert set(nodes) == {"Alice", "relational database"}


@pytest.mark.offline
@pytest.mark.asyncio
async def test_parenthesised_first_record_still_parses():
    nodes, edges = await _parse(f"(entity{D}Carol{D}person{D}A designer.\n<|COMPLETE|>")

    assert set(nodes) == {"Carol"}
    assert not edges
