"""``PIPELINE_REQUIRE_STRICT_STORAGE_READS`` and the startup capability warning.

``/health`` already publishes which strict capabilities a doc_status backend has
(LR2 Phase 6 item 2), but a status page only helps someone who already suspects a
problem. Every one of these capabilities fails CLOSED when absent — admission
answers 503, the conflict endpoints answer 501, a scan keeps re-examining a stale
stub — and from the outside each looks like a broken database rather than a
backend that was never able to do it. The startup log is where an operator finds
out before the first 503; ``require=True`` is for a deployment that would rather
not run at all.

Pinned here as well: the warning names the operator-facing CONSEQUENCE, not just
the capability key (a line saying "strict_active_count: False" tells whoever reads
the log nothing about why uploads started failing), and a failure of the probe
ITSELF never blocks startup — refusing to boot over a broken diagnostic would be
worse than the degradation it was looking for.
"""

from __future__ import annotations

import logging

import pytest

from lightrag.base import DocStatusStorage
from lightrag.exceptions import StorageCapabilityError
from lightrag.kg.json_doc_status_impl import JsonDocStatusStorage
from lightrag.utils_pipeline import enforce_strict_storage_capabilities

pytestmark = pytest.mark.offline


def _minimal_backend():
    """A third-party doc_status that implements only the abstract methods and so
    keeps every fail-closed base default."""

    class _MinimalDocStatus(DocStatusStorage):
        supports_strict_point_reads = False

        async def get_docs_by_statuses_page(self, *args, **kwargs): ...
        async def get_docs_by_ids(self, *args, **kwargs): ...
        async def resolve_doc_source_strict(self, *args, **kwargs): ...
        async def get_full_docs_by_ids(self, *args, **kwargs): ...

    # The probe only reads the class; clearing the ABC guard beats stubbing the
    # data-plane methods this test never touches.
    _MinimalDocStatus.__abstractmethods__ = frozenset()
    return _MinimalDocStatus.__new__(_MinimalDocStatus)


def test_a_first_party_backend_warns_about_nothing(caplog):
    with caplog.at_level(logging.WARNING, logger="lightrag"):
        capabilities = enforce_strict_storage_capabilities(
            JsonDocStatusStorage.__new__(JsonDocStatusStorage)
        )

    assert all(capabilities.values())
    assert caplog.records == []


def test_gaps_are_warned_with_their_operator_facing_consequence(caplog):
    # lightrag's logger does not propagate, so caplog needs it turned on.
    logger = logging.getLogger("lightrag")
    previous = logger.propagate
    logger.propagate = True
    try:
        with caplog.at_level(logging.WARNING, logger="lightrag"):
            enforce_strict_storage_capabilities(_minimal_backend())
    finally:
        logger.propagate = previous

    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    # Every genuinely-missable capability is named...
    for capability in (
        "strict_active_count",
        "source_conflict_listing",
        "source_conflict_repair",
        "strict_point_reads",
    ):
        assert capability in message
    # ...with what it costs, not just its key.
    assert "503" in message and "501" in message
    assert "PIPELINE_REQUIRE_STRICT_STORAGE_READS=true" in message


def test_require_turns_the_gaps_into_a_startup_failure():
    with pytest.raises(StorageCapabilityError) as excinfo:
        enforce_strict_storage_capabilities(_minimal_backend(), require=True)

    assert "PIPELINE_REQUIRE_STRICT_STORAGE_READS" in str(excinfo.value)
    assert "strict_active_count" in str(excinfo.value)


def test_require_does_not_fail_a_backend_that_has_everything():
    enforce_strict_storage_capabilities(
        JsonDocStatusStorage.__new__(JsonDocStatusStorage), require=True
    )


def test_a_broken_probe_never_blocks_startup(monkeypatch):
    """Refusing to boot because the diagnostic broke would be worse than the
    degradation the diagnostic was looking for."""
    import lightrag.utils_pipeline as utils_pipeline

    def _boom(_doc_status):
        raise RuntimeError("probe exploded")

    monkeypatch.setattr(utils_pipeline, "describe_doc_status_capabilities", _boom)

    assert utils_pipeline.enforce_strict_storage_capabilities({}, require=True) == {}


def test_the_paging_capabilities_are_not_part_of_the_gate():
    """They are ``@abstractmethod``: a backend without them cannot be
    instantiated, which is stronger than an opt-in check. Including them here
    would imply the knob can be turned off for them."""
    from lightrag.utils_pipeline import _CAPABILITY_CONSEQUENCES

    assert "scheduling_pages" not in _CAPABILITY_CONSEQUENCES
    assert "typed_source_resolution" not in _CAPABILITY_CONSEQUENCES

    for name in ("get_docs_by_statuses_page", "resolve_doc_source_strict"):
        assert getattr(DocStatusStorage, name).__isabstractmethod__
