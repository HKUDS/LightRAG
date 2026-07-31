"""GHSA-32jh-39m7-8x84: persisted ``sentence_split_regex`` must never be used.

Rejecting the field on ``/documents/text`` closes the ingress for NEW requests,
but ``chunk_options`` is snapshotted into ``full_docs`` at *enqueue* time —
before chunking runs. A build that still accepted the field therefore persisted
the attacker's pattern to storage, and the worker that froze on it left the
document in ``PROCESSING``, which is an auto-resume status. Without a scrub at
the processing trust boundary, upgrading would reload the stored pattern and
freeze the worker again on every restart: a boot loop no restart clears.

These tests pin the scrub itself (``apply_trusted_sentence_split_regex``) and
the trust-boundary property it exists to guarantee.
"""

import pytest

from lightrag.constants import DEFAULT_SENTENCE_SPLIT_REGEX
from lightrag.utils_pipeline import apply_trusted_sentence_split_regex

pytestmark = pytest.mark.offline

# The advisory's payload: 40 `a`s partition 2**39 ways and the trailing anchor
# forces the engine to reject every one.
REDOS = "(a+)+$"


def _addon(regex: str | None) -> dict:
    sv: dict = {"buffer_size": 1}
    if regex is not None:
        sv["sentence_split_regex"] = regex
    return {"chunker": {"semantic_vector": sv}}


def test_poisoned_snapshot_regex_is_discarded():
    # The exact upgrade scenario: a snapshot written by a pre-fix build.
    poisoned = {"buffer_size": 1, "sentence_split_regex": REDOS}
    out = apply_trusted_sentence_split_regex(poisoned, _addon(r"(?<=[.?!])\s+"))
    assert out["sentence_split_regex"] == r"(?<=[.?!])\s+"
    assert REDOS not in out.values()
    # Unrelated snapshot params still win from the snapshot.
    assert out["buffer_size"] == 1


def test_scrub_does_not_mutate_the_snapshot():
    # The caller passes the persisted dict; mutating it in place would write the
    # scrubbed value back into the doc's stored options on the next persist.
    poisoned = {"sentence_split_regex": REDOS}
    apply_trusted_sentence_split_regex(poisoned, _addon(None))
    assert poisoned == {"sentence_split_regex": REDOS}


def test_live_env_regex_wins_over_snapshot_even_when_benign():
    # Not a "reject bad patterns" filter — the snapshot value is unconditionally
    # replaced, so provenance (API vs SDK) never has to be reconstructed.
    snapshot = {"sentence_split_regex": r"(?<=[.?!])\s+"}
    out = apply_trusted_sentence_split_regex(snapshot, _addon(r"(?<=[。！？])"))
    assert out["sentence_split_regex"] == r"(?<=[。！？])"


def test_missing_live_value_omits_the_key_entirely():
    # With nothing configured the key must be absent, so the chunker applies its
    # own default — never the persisted pattern.
    out = apply_trusted_sentence_split_regex({"sentence_split_regex": REDOS}, _addon(None))
    assert "sentence_split_regex" not in out


def test_empty_addon_params_falls_back_to_env_default():
    # A LightRAG instance whose addon_params were never populated must still get
    # the trusted default rather than the snapshot value.
    for addon in ({}, None, {"chunker": {}}):
        out = apply_trusted_sentence_split_regex({"sentence_split_regex": REDOS}, addon)
        assert out.get("sentence_split_regex") == DEFAULT_SENTENCE_SPLIT_REGEX


def test_scrub_is_applied_by_the_v_dispatch():
    # Guard the wiring, not just the helper: if the V branch stops calling the
    # scrub, the helper tests above would all still pass.
    import inspect

    from lightrag import pipeline

    src = inspect.getsource(pipeline)
    v_branch = src.split('elif strategy == "V":', 1)[1].split("else:", 1)[0]
    assert "apply_trusted_sentence_split_regex" in v_branch
    # And the scrubbed dict is what reaches the chunker.
    assert v_branch.index("apply_trusted_sentence_split_regex") < v_branch.index(
        "chunking_by_semantic_vector"
    )
