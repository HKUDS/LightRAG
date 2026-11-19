"""Regression tests for the bounded ``__cause__`` unwrap in
``flush_may_have_lost_reference``.

``tests/pipeline/test_insert_done_error_handling.py`` already pins the
single-hop case (one ``IndexFlushError`` wrapping either a
``ReferencesIntactFlushError`` or a plain error). The function's own
docstring makes a second, separate promise that nothing exercises: the
unwrap is "bounded rather than recursive" over at most 8 hops precisely
because the chain is attacker-shaped data from a storage backend, and "a
self-referential chain must not turn a diagnostic into a hang". Both the
multi-hop chain and the self-referential cycle are real inputs a backend
wrapping its own flush error more than once -- or a buggy ``__cause__``
assignment -- can produce, and neither is covered anywhere.
"""

import pytest

from lightrag.exceptions import (
    IndexFlushError,
    ReferencesIntactFlushError,
    flush_may_have_lost_reference,
)

pytestmark = pytest.mark.offline


def _chain(*causes: BaseException) -> BaseException:
    """Build a ``__cause__`` chain: causes[0] -> causes[1] -> ... -> causes[-1]."""
    for earlier, later in zip(causes, causes[1:]):
        earlier.__cause__ = later
    return causes[0]


def test_multi_hop_chain_within_bound_finds_the_intact_cause():
    """Several IndexFlushError hops are all unwrapped, not just the first."""
    intact = ReferencesIntactFlushError("bulk retried and landed")
    outer = _chain(
        IndexFlushError("A", "ns", RuntimeError("x")),
        IndexFlushError("B", "ns", RuntimeError("x")),
        IndexFlushError("C", "ns", RuntimeError("x")),
        intact,
    )
    assert flush_may_have_lost_reference(outer) is False


def test_multi_hop_chain_within_bound_finds_a_real_loss():
    """A multi-hop chain bottoming out in an ordinary error still answers True."""
    outer = _chain(
        IndexFlushError("A", "ns", RuntimeError("x")),
        IndexFlushError("B", "ns", RuntimeError("x")),
        IndexFlushError("C", "ns", RuntimeError("x")),
        RuntimeError("permanently rejected"),
    )
    assert flush_may_have_lost_reference(outer) is True


def test_chain_deeper_than_the_bound_is_not_unwrapped_past_it():
    """Documented bound: at most 8 hops are followed.

    Nine ``IndexFlushError`` hops (error0..error8) wrap a tenth,
    ``ReferencesIntactFlushError`` (error9), one hop beyond what 8
    reassignments can reach from error0. The predicate must fall back to
    the safe "assume a loss" answer instead of unwrapping indefinitely to
    find it.
    """
    intact = ReferencesIntactFlushError("findable only past the bound")
    hops = [IndexFlushError(f"hop{i}", "ns", RuntimeError("x")) for i in range(9)]
    outer = _chain(*hops, intact)
    assert flush_may_have_lost_reference(outer) is True


def test_chain_exactly_at_the_bound_is_still_unwrapped():
    """The boundary case next to the one above: 8 hops does reach the cause."""
    intact = ReferencesIntactFlushError("findable exactly at the bound")
    hops = [IndexFlushError(f"hop{i}", "ns", RuntimeError("x")) for i in range(8)]
    outer = _chain(*hops, intact)
    assert flush_may_have_lost_reference(outer) is False


def test_self_referential_cause_terminates_instead_of_hanging():
    """A cycle (error.__cause__ is error) must not hang the predicate."""
    cyclic = IndexFlushError("Self", "ns", RuntimeError("x"))
    cyclic.__cause__ = cyclic
    assert flush_may_have_lost_reference(cyclic) is True


def test_mutual_cycle_of_two_wrappers_terminates():
    """A longer cycle (A -> B -> A -> ...) is equally safe, not just a 1-cycle."""
    a = IndexFlushError("A", "ns", RuntimeError("x"))
    b = IndexFlushError("B", "ns", RuntimeError("x"))
    a.__cause__ = b
    b.__cause__ = a
    assert flush_may_have_lost_reference(a) is True
