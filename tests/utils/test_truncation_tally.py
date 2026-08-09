"""``TokenLimitTruncationTally`` — the aggregation behind the truncation report.

The tally exists so a document that truncates on every chunk produces ONE
status line and ONE ``doc_status.metadata`` record instead of N of each. These
tests pin the three properties that make that record trustworthy: events are
counted per response while subjects are de-duplicated, the sample list is
bounded independently of document size, and a clean run yields an EMPTY
metadata fragment (which is what clears a previous attempt's summary, since
the key is deliberately absent from the carry-over whitelist).
"""

import pytest

from lightrag.utils import (
    LLM_TRUNCATION_METADATA_KEY,
    TRUNCATION_METADATA_SAMPLE_LIMIT,
    TokenLimitTruncationTally,
)

pytestmark = pytest.mark.offline


def test_a_clean_run_records_nothing():
    tally = TokenLimitTruncationTally()

    assert not tally
    assert tally.events == 0
    assert tally.affected == 0
    assert tally.as_metadata() is None
    # An empty fragment is load-bearing: it leaves the key out of the
    # transition payload, which CLEARS a previous attempt's summary.
    assert tally.as_metadata_extra() == {}


def test_only_the_first_event_is_announced():
    tally = TokenLimitTruncationTally()

    assert tally.record("initial", "chunk-001") is True
    assert tally.record("initial", "chunk-002") is False
    assert tally.record("gleaning", "chunk-001") is False


def test_events_count_responses_while_subjects_deduplicate():
    """A chunk truncated at both stages is 2 events but 1 affected chunk."""
    tally = TokenLimitTruncationTally()
    tally.record("initial", "chunk-001")
    tally.record("gleaning", "chunk-001")
    tally.record("initial", "chunk-002")

    assert tally.events == 3
    assert tally.affected == 2
    assert tally.as_metadata()["stages"] == {"initial": 2, "gleaning": 1}
    assert tally.stage_breakdown() == "initial: 2, gleaning: 1"


def test_absorb_folds_a_stage_tally_into_the_document_tally():
    extraction = TokenLimitTruncationTally()
    extraction.record("initial", "chunk-001")
    extraction.record("gleaning", "chunk-001")
    merge = TokenLimitTruncationTally()
    merge.record("summary", "Entity:ALICE")

    document = TokenLimitTruncationTally()
    document.absorb(extraction)
    document.absorb(merge)

    assert document.events == 3
    assert document.affected == 2
    assert document.as_metadata()["stages"] == {
        "initial": 1,
        "gleaning": 1,
        "summary": 1,
    }
    assert document.as_metadata()["samples"] == ["chunk-001", "Entity:ALICE"]


def test_absorb_deduplicates_subjects_across_stages():
    document = TokenLimitTruncationTally()
    document.record("initial", "chunk-001")

    other = TokenLimitTruncationTally()
    other.record("summary", "chunk-001")
    document.absorb(other)

    assert document.events == 2
    assert document.affected == 1


def test_absorb_tolerates_an_absent_tally():
    document = TokenLimitTruncationTally()
    document.absorb(None)
    document.absorb(TokenLimitTruncationTally())

    assert not document


def test_the_metadata_sample_is_bounded_by_document_size():
    """doc_status rows are serialized into every listing response, so the
    sample must not grow with the number of truncated chunks."""
    tally = TokenLimitTruncationTally()
    total = TRUNCATION_METADATA_SAMPLE_LIMIT + 7
    for index in range(total):
        tally.record("initial", f"chunk-{index:03d}")

    payload = tally.as_metadata()

    assert payload["events"] == total
    assert payload["affected"] == total
    assert len(payload["samples"]) == TRUNCATION_METADATA_SAMPLE_LIMIT
    # First-seen order, so the sample points at the earliest chunks.
    assert payload["samples"][0] == "chunk-000"
    assert payload["samples_omitted"] == 7


def test_no_omitted_counter_when_every_subject_fits():
    tally = TokenLimitTruncationTally()
    tally.record("initial", "chunk-001")

    assert "samples_omitted" not in tally.as_metadata()


def test_metadata_extra_is_keyed_for_the_doc_status_transition():
    tally = TokenLimitTruncationTally()
    tally.record("initial", "chunk-001")

    extra = tally.as_metadata_extra()

    assert set(extra) == {LLM_TRUNCATION_METADATA_KEY}
    assert extra[LLM_TRUNCATION_METADATA_KEY]["events"] == 1
