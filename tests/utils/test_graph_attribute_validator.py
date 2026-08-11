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

import enum
import sys
from decimal import Decimal

import numpy as np
import pytest

from lightrag.utils import (
    validate_graph_attributes,
    validate_xml_attributes,
    xml_attribute_value_rejection,
)

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
    def test_ordinary_names_are_accepted(self, key):
        validate_graph_attributes({key: "x"}, context="entity 'X'")

    @pytest.mark.parametrize(
        "key",
        ["display-name", "has space", "9leading_digit", "MiXeD Case!", "名前"],
    )
    def test_unusual_but_stored_names_are_accepted(self, key):
        """The rule is the interpretation hazard, not "must be an identifier".

        These carry no hazard on any backend, and the manual edit API accepted
        them before its field allowlist landed -- so the document backends that
        persisted them hold such names today. Every rewrite path (entity edit,
        rename, merge, extraction rebuild) spreads a fetched object's stored
        attributes back into the upsert payload, so refusing them would make
        those objects permanently unmodifiable.
        """
        validate_graph_attributes({key: "x"}, context="entity 'X'")

    @pytest.mark.parametrize(
        "key",
        [
            # In a MongoDB ``$set`` a dot is a *path* separator, so this would
            # rewrite an element of the chunk-attribution array rather than
            # create a field named "source_ids.0".
            "source_ids.0",
            "a.b",
            ".leading",
            "trailing.",
        ],
    )
    def test_names_read_as_a_field_path_are_rejected(self, key):
        with pytest.raises(ValueError, match="read as a field path"):
            validate_graph_attributes({key: "x"}, context="entity 'X'")

    @pytest.mark.parametrize("key", ["$set", "$where", "$"])
    def test_names_read_as_an_update_operator_are_rejected(self, key):
        with pytest.raises(ValueError, match="read as\\s+an update operator"):
            validate_graph_attributes({key: "x"}, context="entity 'X'")

    def test_a_dollar_sign_inside_the_name_is_accepted(self):
        """Only the *leading* ``$`` is an operator; mid-name it is just a char."""
        validate_graph_attributes({"cost$usd": "x"}, context="entity 'X'")

    @pytest.mark.parametrize("key", [1, None, ("a",)])
    def test_non_string_names_are_rejected(self, key):
        """Caught by the character rule first, which names the actual type."""
        with pytest.raises(ValueError, match="must be a string, got"):
            validate_graph_attributes({key: "x"}, context="entity 'X'")

    def test_an_empty_name_is_rejected(self):
        """XML would encode it; it is the interpretation rule that refuses it."""
        with pytest.raises(ValueError, match="must be a non-empty string"):
            validate_graph_attributes({"": "x"}, context="entity 'X'")


def test_context_appears_in_the_message():
    with pytest.raises(ValueError, match="entity 'Tesla'"):
        validate_graph_attributes({"attr": None}, context="entity 'Tesla'")


def test_empty_mapping_is_accepted():
    validate_graph_attributes({}, context="entity 'X'")


class TestTheTwoValueRulesDiffer:
    """``xml_...`` is what a store cannot serialize; ``graph_...`` is portability.

    Keeping them apart is what lets a storage backend guard itself without
    stranding data: a workspace can already hold a value the portable rule
    refuses, and every rewrite path spreads a fetched object's stored attributes
    back into the upsert payload.
    """

    @pytest.mark.parametrize(
        "value",
        [float("nan"), float("inf"), float("-inf"), 2**63, -(2**63) - 1, 10**400],
    )
    def test_unportable_but_encodable_values_split_the_two_rules(self, value):
        # XML can carry it -- GraphML round-trips all of these unchanged.
        validate_xml_attributes({"legacy": value}, context="entity 'X'")
        # The portable contract cannot: the Neo4j driver refuses to pack them.
        with pytest.raises(ValueError, match="attribute 'legacy'"):
            validate_graph_attributes({"legacy": value}, context="entity 'X'")

    @pytest.mark.parametrize(
        "value", [None, {"a": 1}, ["a"], b"bytes", "a\x0bb", "a\x00b"]
    )
    def test_unencodable_values_are_refused_by_both(self, value):
        with pytest.raises(ValueError, match="attribute 'attr'"):
            validate_xml_attributes({"attr": value}, context="entity 'X'")
        with pytest.raises(ValueError, match="attribute 'attr'"):
            validate_graph_attributes({"attr": value}, context="entity 'X'")

    @pytest.mark.parametrize("value", ["text", 1, 2.5, True, "multi\nline"])
    def test_ordinary_values_are_accepted_by_both(self, value):
        validate_xml_attributes({"attr": value}, context="entity 'X'")
        validate_graph_attributes({"attr": value}, context="entity 'X'")

    @pytest.mark.parametrize("name", ["a.b", "$set", "", "source_ids.0"])
    def test_interpreted_names_split_the_two_rules(self, name):
        """GraphML stores them; MongoDB would read them as a path or operator."""
        validate_xml_attributes({name: "x"}, context="entity 'X'")
        with pytest.raises(ValueError, match="invalid attribute name"):
            validate_graph_attributes({name: "x"}, context="entity 'X'")

    @pytest.mark.parametrize("name", ["display-name", "has space", "名前", "cost$usd"])
    def test_merely_unusual_names_are_accepted_by_both(self, name):
        """Neither backend interprets these, and both can store them."""
        validate_xml_attributes({name: "x"}, context="entity 'X'")
        validate_graph_attributes({name: "x"}, context="entity 'X'")

    @pytest.mark.parametrize(
        "name", ["bad\x0bname", "bad\x00name", "bad\ud800name", "bad\ufffename"]
    )
    def test_unencodable_names_are_refused_by_both(self, name):
        """The portable rule is a superset: XML cannot encode it, so no backend can.

        A name MongoDB stores happily but GraphML cannot serialize is still
        outside the intersection, so caller input must not carry one even on a
        Mongo deployment.
        """
        with pytest.raises(ValueError, match="attribute name"):
            validate_xml_attributes({name: "x"}, context="entity 'X'")
        with pytest.raises(ValueError, match="attribute name"):
            validate_graph_attributes({name: "x"}, context="entity 'X'")


@pytest.mark.skipif(
    not hasattr(sys, "get_int_max_str_digits"),
    reason="int_max_str_digits arrived in 3.11 (backported to 3.10.7)",
)
class TestIntegerStringificationLimit:
    """ "Serializable" is not the same as "is a scalar".

    GraphML stores values as text, and CPython refuses to stringify an integer
    with more digits than ``sys.get_int_max_str_digits()``. So an int past that
    limit is unencodable even though its *type* is fine -- both rules must refuse
    it, unlike the merely-unportable big ints only the portable rule refuses.
    """

    def test_an_int_too_long_to_stringify_is_refused_by_both(self):
        oversized = 10 ** (sys.get_int_max_str_digits() + 100)

        assert "more digits" in (xml_attribute_value_rejection(oversized) or "")
        with pytest.raises(ValueError, match="more digits than Python will render"):
            validate_xml_attributes({"n": oversized}, context="entity 'X'")
        with pytest.raises(ValueError):
            validate_graph_attributes({"n": oversized}, context="entity 'X'")

    def test_the_check_tracks_the_runtime_limit(self):
        """The limit is settable, so the rule attempts the conversion rather
        than comparing against a hardcoded threshold."""
        original = sys.get_int_max_str_digits()
        value = 10**700  # 701 digits: over a 640-digit limit, under the default
        try:
            sys.set_int_max_str_digits(640)
            assert xml_attribute_value_rejection(value) is not None
            sys.set_int_max_str_digits(original)
            assert xml_attribute_value_rejection(value) is None
        finally:
            sys.set_int_max_str_digits(original)

    def test_an_int_just_under_the_default_limit_is_accepted(self):
        assert (
            xml_attribute_value_rejection(10 ** (sys.get_int_max_str_digits() - 10))
            is None
        )


# `enum.StrEnum` is 3.11+, and this package declares
# `requires-python = ">=3.10"`. At module level that would fail
# collection and silently disable every test in the file, and CI only
# runs 3.12/3.14 so it would not be caught here. `(str, Enum)` has the
# property these tests need on every supported version: an exact type
# that is not `str`, with `isinstance(x, str)` true.
class _Colour(str, enum.Enum):
    RED = "red"


class TestExactTypeResolution:
    """A subclass of a supported scalar is not a supported scalar.

    networkx's GraphML writer looks the value's type up in a dict, so
    ``isinstance`` is the wrong test -- an ``enum.StrEnum`` passes it and still
    breaks the write. Both rules refuse subclasses, since neither backend can
    take one: the portable rule delegates to the XML rule for exactly this.
    """

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(_Colour.RED, id="str-enum"),
            pytest.param(np.str_("x"), id="numpy-str"),
        ],
    )
    def test_subclasses_are_refused_by_both(self, value):
        with pytest.raises(ValueError, match="GraphML resolves value types exactly"):
            validate_xml_attributes({"v": value}, context="entity 'X'")
        with pytest.raises(ValueError, match="GraphML resolves value types exactly"):
            validate_graph_attributes({"v": value}, context="entity 'X'")

    @pytest.mark.parametrize("value", ["text", 5, 1.5, True, False, 0, ""])
    def test_the_four_builtin_scalars_are_accepted(self, value):
        validate_xml_attributes({"v": value}, context="entity 'X'")
        validate_graph_attributes({"v": value}, context="entity 'X'")

    def test_a_non_scalar_keeps_the_plain_message(self):
        """`Decimal` is not a subclass of any supported scalar, so the subclass
        remark would be noise -- it gets the ordinary type message."""
        assert "string, number or boolean" in (
            xml_attribute_value_rejection(Decimal("1.5")) or ""
        )

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(np.float64(1.5), id="np-float64"),
            pytest.param(np.float32(1.5), id="np-float32"),
            pytest.param(np.int64(5), id="np-int64"),
            pytest.param(np.uint32(5), id="np-uint32"),
        ],
    )
    def test_numpy_scalars_split_the_two_rules(self, value):
        """GraphML writes them (reloading as plain float/int); nothing else takes them.

        The XML rule's type set comes from networkx's own ``construct_types``
        table, so it accepts exactly what the writer accepts. The portable rule
        does not: ``json.dumps`` (PGTableGraphStorage's jsonb column) and ``bson``
        (MongoGraphStorage) both refuse ``np.float32`` / ``np.int64`` /
        ``np.uint32`` outright, so a numpy value that persists fine on NetworkX
        cannot cross to another backend.
        """
        validate_xml_attributes({"v": value}, context="entity 'X'")
        with pytest.raises(ValueError, match="portable across graph backends"):
            validate_graph_attributes({"v": value}, context="entity 'X'")

    @pytest.mark.parametrize(
        "value",
        [
            # An ``np.floating`` that networkx does *not* list -- which is why the
            # rule reads the table instead of testing ``isinstance(np.floating)``.
            pytest.param(np.longdouble(1.5), id="np-longdouble"),
            pytest.param(np.str_("x"), id="np-str"),
            pytest.param(np.bool_(True), id="np-bool"),
        ],
    )
    def test_numpy_types_networkx_cannot_write_are_refused_by_both(self, value):
        with pytest.raises(ValueError, match="attribute 'v'"):
            validate_xml_attributes({"v": value}, context="entity 'X'")
        with pytest.raises(ValueError, match="attribute 'v'"):
            validate_graph_attributes({"v": value}, context="entity 'X'")

    def test_the_encodable_set_is_sourced_from_networkx(self):
        """Every refused type must be one the write would have failed on.

        That claim is only true by construction if the set *is* networkx's table,
        so pin the sourcing rather than a hardcoded list.
        """
        from networkx.readwrite.graphml import GraphML

        from lightrag.utils import _GRAPHML_ENCODABLE_TYPES

        probe = GraphML()
        probe.construct_types()
        assert frozenset(probe.xml_type) <= _GRAPHML_ENCODABLE_TYPES
