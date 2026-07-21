"""Regression: delimiter repair must not forge separators inside free text."""

import pytest

from lightrag.operate import (
    _handle_single_entity_extraction,
    _handle_single_relationship_extraction,
)
from lightrag.utils import fix_tuple_delimiter_corruption, split_string_by_multi_markers

pytestmark = pytest.mark.offline

_DELIM = "<|#|>"
_CORE = "#"


def test_spaced_angle_pipe_in_description_does_not_split_entity():
    """Prose that mentions shell ``<|>`` must stay one description field."""
    description = "Documents the shell redirect form a <|> b used in pipelines"
    record = f"entity{_DELIM}ShellRedirect{_DELIM}concept{_DELIM}{description}"
    fixed = fix_tuple_delimiter_corruption(record, _CORE, _DELIM)
    attrs = split_string_by_multi_markers(fixed, [_DELIM])
    assert len(attrs) == 4
    entity = _handle_single_entity_extraction(attrs, "chunk-1", 1, "docs.md")
    assert entity is not None
    assert entity["entity_name"] == "ShellRedirect"
    assert "<|>" in entity["description"]


def test_spaced_double_pipe_in_description_does_not_split_entity():
    """Prose/code that mentions ``<||`` must not become an extra delimiter."""
    description = "Pseudocode guard: if (x <|| y) then skip"
    record = f"entity{_DELIM}Guard{_DELIM}concept{_DELIM}{description}"
    fixed = fix_tuple_delimiter_corruption(record, _CORE, _DELIM)
    attrs = split_string_by_multi_markers(fixed, [_DELIM])
    assert len(attrs) == 4
    entity = _handle_single_entity_extraction(attrs, "chunk-2", 1, "docs.md")
    assert entity is not None
    assert "<||" in entity["description"]


def test_spaced_angle_pipe_in_relationship_does_not_split_relation():
    """Same guard for the 5-field relationship record shape (description field)."""
    description = "Explains the shell redirect form a <|> b shared by both tools"
    record = (
        f"relation{_DELIM}ToolA{_DELIM}ToolB{_DELIM}pipeline,redirect{_DELIM}"
        f"{description}"
    )
    fixed = fix_tuple_delimiter_corruption(record, _CORE, _DELIM)
    attrs = split_string_by_multi_markers(fixed, [_DELIM])
    assert len(attrs) == 5
    relation = _handle_single_relationship_extraction(attrs, "chunk-4", 1, "docs.md")
    assert relation is not None
    assert relation["src_id"] == "ToolA"
    assert relation["tgt_id"] == "ToolB"
    assert "<|>" in relation["description"]


def test_glued_missing_core_delimiter_in_relationship_still_repaired():
    """Glued missing-core separators still repair the 5-field relationship shape."""
    broken = "relation<|>ToolA<|>ToolB<|>pipeline<|>Kept relation description"
    fixed = fix_tuple_delimiter_corruption(broken, _CORE, _DELIM)
    attrs = split_string_by_multi_markers(fixed, [_DELIM])
    assert attrs == [
        "relation",
        "ToolA",
        "ToolB",
        "pipeline",
        "Kept relation description",
    ]
    relation = _handle_single_relationship_extraction(attrs, "chunk-5", 1, "docs.md")
    assert relation is not None
    assert relation["description"] == "Kept relation description"


def test_glued_missing_core_delimiter_still_repaired():
    """LLM field separators with a missing core (entity<|>name) still repair."""
    broken = "entity<|>ShellRedirect<|>concept<|>Kept description"
    fixed = fix_tuple_delimiter_corruption(broken, _CORE, _DELIM)
    attrs = split_string_by_multi_markers(fixed, [_DELIM])
    assert attrs == [
        "entity",
        "ShellRedirect",
        "concept",
        "Kept description",
    ]
    entity = _handle_single_entity_extraction(attrs, "chunk-3", 1, "docs.md")
    assert entity is not None
    assert entity["description"] == "Kept description"
