"""`validate_graph_attributes` -- the shared definition of a storable attribute.

The rules are the *intersection* of what the registered GRAPH_STORAGE backends
can carry, not what any one of them happens to tolerate. That matters because
the backends disagree loudly on a violation: the same non-scalar is refused by
the Neo4j driver, stored verbatim by MongoDB and by PostgreSQL's ``jsonb``
column, and fatal to GraphML serialization. Anything that is legal here must be
legal everywhere, and the two easy-to-miss cases are pinned below: ``None``
(deletes the property on the Cypher backends, hard error on NetworkX) and a
string holding a code point XML cannot encode.

The behavioural regression tests for the vulnerability this validator closes
(GHSA-c922-pw4m-4wcv) are in tests/api/routes/test_graph_attribute_validation.py.
"""

from __future__ import annotations

import pytest

from lightrag.utils import validate_graph_attributes

pytestmark = pytest.mark.offline


class TestStorableValues:
    @pytest.mark.parametrize("value", ["text", "", 1, 0, -3, 2.5, True, False])
    def test_scalars_are_accepted(self, value):
        validate_graph_attributes({"attr": value}, context="entity 'X'")

    @pytest.mark.parametrize(
        "text",
        [
            # Tab, newline and carriage return are the only C0 controls XML
            # admits -- and descriptions legitimately carry all three.
            "first line\nsecond line",
            "col\tcol",
            "line\r\nline",
            "中文描述",
            "emoji \U0001f600",
            "DEL \x7f and NEL \x85 are valid XML 1.0 characters",
        ],
    )
    def test_xml_compatible_strings_are_accepted(self, text):
        validate_graph_attributes({"description": text}, context="entity 'X'")


class TestUnstorableValues:
    @pytest.mark.parametrize(
        "value, expected",
        [
            (None, "got NoneType"),
            ({"a": 1}, "got dict"),
            (["a"], "got list"),
            (("a",), "got tuple"),
            (b"bytes", "got bytes"),
            (set(), "got set"),
        ],
    )
    def test_non_scalars_are_rejected(self, value, expected):
        with pytest.raises(ValueError, match=expected):
            validate_graph_attributes({"attr": value}, context="entity 'X'")

    @pytest.mark.parametrize(
        "value",
        [2**63, -(2**63) - 1, 10**400, -(10**400)],
    )
    def test_integers_outside_int64_are_rejected(self, value):
        """networkx writes them and jsonb stores them; Neo4j cannot pack them.

        Verified against the installed driver: neo4j's packstream Packer raises
        ``OverflowError`` outside ``[-2**63, 2**63)``. GraphML also mislabels
        such a value, declaring ``attr.type="long"`` (``xsd:long`` is int64). So
        it is outside the intersection this validator defines even though two of
        the three backends accept it.
        """
        with pytest.raises(ValueError, match="must be a 64-bit integer"):
            validate_graph_attributes({"created_at": value}, context="entity 'X'")

    @pytest.mark.parametrize("value", [2**63 - 1, -(2**63), 0, 1786431045])
    def test_integers_inside_int64_are_accepted(self, value):
        validate_graph_attributes({"created_at": value}, context="entity 'X'")

    @pytest.mark.parametrize(
        "value", [float("nan"), float("inf"), float("-inf"), float("1e999")]
    )
    def test_non_finite_floats_are_rejected(self, value):
        """They survive GraphML but ``json.dumps`` emits bare ``NaN``/``Infinity``.

        That is not valid JSON, so the ``jsonb`` column PGTableGraphStorage
        writes to rejects it -- the value is storable on one backend and not on
        another, which is exactly what this validator exists to prevent.
        """
        with pytest.raises(ValueError, match="must be a finite number"):
            validate_graph_attributes({"weight": value}, context="relation 'A~B'")

    @pytest.mark.parametrize(
        "char, code",
        [
            ("\x00", r"U\+0000"),
            ("\x0b", r"U\+000B"),
            ("\x1f", r"U\+001F"),
            ("￾", r"U\+FFFE"),
            ("\ud800", r"U\+D800"),
        ],
    )
    def test_xml_incompatible_strings_are_rejected(self, char, code):
        with pytest.raises(ValueError, match=code):
            validate_graph_attributes(
                {"description": f"a{char}b"}, context="entity 'X'"
            )

    def test_the_first_offender_is_reported(self):
        with pytest.raises(ValueError, match="attribute 'bad'"):
            validate_graph_attributes({"good": "ok", "bad": None}, context="entity 'X'")


class TestAttributeNames:
    @pytest.mark.parametrize("key", ["entity_id", "_private", "created_at", "a9", "A"])
    def test_identifiers_are_accepted(self, key):
        validate_graph_attributes({key: "x"}, context="entity 'X'")

    @pytest.mark.parametrize(
        "key",
        [
            # In a MongoDB ``$set`` a dot is a *path* separator, so this would
            # rewrite an element of the chunk-attribution array rather than
            # create a field named "source_ids.0".
            "source_ids.0",
            "a.b",
            # A leading ``$`` is read as an update operator.
            "$set",
            "$where",
            "has space",
            "9leading_digit",
            "dash-ed",
            "",
        ],
    )
    def test_unsafe_names_are_rejected(self, key):
        with pytest.raises(ValueError, match="invalid attribute name"):
            validate_graph_attributes({key: "x"}, context="entity 'X'")

    @pytest.mark.parametrize("key", [1, None, ("a",)])
    def test_non_string_names_are_rejected(self, key):
        with pytest.raises(ValueError, match="invalid attribute name"):
            validate_graph_attributes({key: "x"}, context="entity 'X'")


def test_context_appears_in_the_message():
    with pytest.raises(ValueError, match="entity 'Tesla'"):
        validate_graph_attributes({"attr": None}, context="entity 'Tesla'")


def test_empty_mapping_is_accepted():
    validate_graph_attributes({}, context="entity 'X'")
