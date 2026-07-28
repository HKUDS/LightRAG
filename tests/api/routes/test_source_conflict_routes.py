"""Operator source-conflict list / repair endpoints (LR2 Phase 4-f, §5.5).

A scan that meets two primary documents claiming one canonical source refuses to
pick a winner: it leaves the file and both rows alone and records the conflict.
These endpoints are the operator's way out — enumerate the conflicts, then
settle one by naming the document that keeps the source, guarded by a
compare-and-set token so a concurrent change cannot be overwritten.

The storage semantics (fingerprints, transactions, demotion) are covered per
backend under ``tests/kg/``; what is pinned here is the HTTP contract: the
status code each storage outcome maps to, the opaque cursor round-trip, the
CAS-token requirement, and the audit trail.
"""

from __future__ import annotations

import hashlib
import importlib
import sys
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_document_routes = importlib.import_module("lightrag.api.routers.document_routes")
sys.argv = _original_argv

from lightrag.base import (  # noqa: E402
    CURSOR_END,
    CURSOR_START,
    CursorAfter,
    SourceConflictPage,
    SourceConflictRepairResult,
    SourceConflictSummary,
)
from lightrag.exceptions import (  # noqa: E402
    SourceConflictRepairCASError,
    StorageCapabilityError,
    StorageControlPlaneError,
)

create_document_routes = _document_routes.create_document_routes

pytestmark = pytest.mark.offline

_HEADERS = {"X-API-Key": "test-key"}


def _fingerprint(doc_ids: list[str]) -> str:
    return hashlib.sha256("\x00".join(sorted(doc_ids)).encode("utf-8")).hexdigest()[:32]


@pytest.fixture(autouse=True)
def _shared_storage():
    """The repair COMMIT takes the canonical-key + enqueue-serialize locks, which
    live in shared storage. Without this the file only passed when a sibling test
    module happened to initialise it first — running it alone raised
    "Shared-Data is not initialized" from inside the endpoint and showed up as a
    500."""
    from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data

    initialize_share_data()
    yield
    finalize_share_data()


class _ConflictDocStatus:
    """doc_status double whose conflict surface mirrors the real backends.

    ``groups`` maps a canonical source key to its primary candidates; a repair
    demotes losers by removing them from the group, exactly as marking
    ``metadata.is_duplicate=true`` removes them from the primary candidate set.
    """

    _SAMPLE_CAP = 2

    def __init__(
        self,
        groups: dict[str, list[str]] | None = None,
        *,
        content_hashes: dict[str, str] | None = None,
        source_of: dict[str, str] | None = None,
    ):
        self.groups = groups or {}
        # doc_id → content_hash, and doc_id → the canonical source it claims.
        # The commit refuses a primary whose content already lives under ANOTHER
        # source, so both are needed to exercise (and to stay out of) that check.
        self.content_hashes = content_hashes or {}
        self.source_of = source_of or {}
        self.list_calls: list[tuple[int, object]] = []
        self.repair_calls: list[dict] = []
        self.list_error: Exception | None = None
        self.repair_error: Exception | None = None

    def _row(self, doc_id: str) -> SimpleNamespace:
        source = self.source_of.get(doc_id) or next(
            (key for key, ids in self.groups.items() if doc_id in ids), ""
        )
        return SimpleNamespace(
            content_hash=self.content_hashes.get(doc_id, ""),
            file_path=source,
            metadata={},
        )

    async def get_full_docs_by_ids(self, doc_ids, *, strict=False):
        known = set(self.content_hashes) | set(self.source_of)
        for ids in self.groups.values():
            known.update(ids)
        return {doc_id: self._row(doc_id) for doc_id in doc_ids if doc_id in known}

    async def get_doc_by_content_hash(self, content_hash, *, exclude_doc_id=None):
        """Earliest (by id, deterministically) OTHER holder of the hash — the
        exclusions the base contract requires, on a double small enough to reason
        about."""
        holders = sorted(
            doc_id
            for doc_id, value in self.content_hashes.items()
            if value == content_hash and doc_id != exclude_doc_id
        )
        if not holders:
            return None
        return holders[0], self._row(holders[0])

    async def list_source_conflicts_page(self, *, limit, position=CURSOR_START):
        self.list_calls.append((limit, position))
        if self.list_error is not None:
            raise self.list_error
        keys = sorted(k for k, v in self.groups.items() if len(v) >= 2)
        if isinstance(position, CursorAfter):
            keys = [k for k in keys if k > position.opaque]
        page_keys = keys[:limit]
        conflicts = tuple(
            SourceConflictSummary(
                canonical_source_key=key,
                candidate_count=len(self.groups[key]),
                sample_doc_ids=tuple(sorted(self.groups[key])[: self._SAMPLE_CAP]),
            )
            for key in page_keys
        )
        next_position = (
            CursorAfter(page_keys[-1]) if len(page_keys) == limit else CURSOR_END
        )
        return SourceConflictPage(conflicts=conflicts, next_position=next_position)

    async def resolve_doc_source_strict(self, canonical_source_key):
        """The same typed resolution the real backends serve, so the post-commit
        verification runs against a real answer rather than being stubbed out."""
        from lightrag.base import SourceAbsent, SourceConflict, SourceUnique

        candidates = sorted(self.groups.get(canonical_source_key, []))
        if not candidates:
            return SourceAbsent()
        if len(candidates) == 1:
            return SourceUnique(doc_id=candidates[0], doc=None)
        return SourceConflict(
            candidate_count=len(candidates),
            sample_doc_ids=tuple(candidates[: self._SAMPLE_CAP]),
        )

    async def repair_source_conflict(
        self,
        canonical_source_key,
        *,
        primary_doc_id,
        expected_candidate_count,
        expected_candidate_fingerprint,
        dry_run=True,
    ):
        self.repair_calls.append(
            {
                "canonical_source_key": canonical_source_key,
                "primary_doc_id": primary_doc_id,
                "expected_candidate_count": expected_candidate_count,
                "expected_candidate_fingerprint": expected_candidate_fingerprint,
                "dry_run": dry_run,
            }
        )
        if self.repair_error is not None:
            raise self.repair_error
        candidates = sorted(self.groups.get(canonical_source_key, []))
        count = len(candidates)
        fingerprint = _fingerprint(candidates)
        if primary_doc_id not in candidates:
            raise ValueError(
                f"primary_doc_id {primary_doc_id!r} is not a current primary "
                f"candidate for {canonical_source_key!r}"
            )
        demoted = [d for d in candidates if d != primary_doc_id]
        if not dry_run:
            if (
                count != expected_candidate_count
                or fingerprint != expected_candidate_fingerprint
            ):
                raise SourceConflictRepairCASError(
                    f"source-conflict repair CAS failed for {canonical_source_key!r}"
                )
            self.groups[canonical_source_key] = [primary_doc_id]
        return SourceConflictRepairResult(
            canonical_source_key=canonical_source_key,
            primary_doc_id=primary_doc_id,
            candidate_count=count,
            fingerprint=fingerprint,
            demoted_sample_doc_ids=tuple(demoted[: self._SAMPLE_CAP]),
            committed=not dry_run,
        )


class _FullDocs:
    """full_docs double for the contentless-primary refusal.

    ``strict`` mirrors ``supports_strict_point_reads``: only a backend that HAS
    strict point reads may have an absence trusted, so the doubles cover both.
    """

    def __init__(
        self,
        contents: dict[str, dict] | None = None,
        *,
        strict: bool = True,
        read_error: Exception | None = None,
    ):
        self.contents = contents if contents is not None else {}
        self.supports_strict_point_reads = strict
        self.read_error = read_error
        self.reads: list[str] = []

    async def get_by_id_strict(self, doc_id: str):
        self.reads.append(doc_id)
        if self.read_error is not None:
            raise self.read_error
        return self.contents.get(doc_id)


def _client(
    doc_status: _ConflictDocStatus, full_docs: _FullDocs | None = None
) -> TestClient:
    app = FastAPI()
    app.include_router(
        create_document_routes(
            SimpleNamespace(
                doc_status=doc_status,
                full_docs=full_docs if full_docs is not None else _ContentEverywhere(),
                workspace="conflict-test",
            ),
            SimpleNamespace(),
            api_key="test-key",
        )
    )
    return TestClient(app)


class _ContentEverywhere:
    """Default for tests not about the content check: every document has content."""

    supports_strict_point_reads = True

    async def get_by_id_strict(self, doc_id: str):
        return {"content": f"body of {doc_id}"}


# --------------------------------------------------------------------------- #
# listing
# --------------------------------------------------------------------------- #


def test_listing_projects_bounded_samples():
    storage = _ConflictDocStatus(
        {"a.pdf": ["doc-1", "doc-2", "doc-3"], "solo.pdf": ["doc-9"]}
    )
    response = _client(storage).get("/documents/source_conflicts", headers=_HEADERS)

    assert response.status_code == 200
    body = response.json()
    assert body["next_cursor"] is None  # exhausted
    assert len(body["conflicts"]) == 1  # the single-candidate key is not a conflict
    conflict = body["conflicts"][0]
    assert conflict["canonical_source_key"] == "a.pdf"
    assert conflict["candidate_count"] == 3
    # Sample is capped: three candidates, two reported.
    assert conflict["sample_doc_ids"] == ["doc-1", "doc-2"]


def test_listing_cursor_round_trips_without_leaking_backend_tokens():
    storage = _ConflictDocStatus(
        {"a.pdf": ["doc-1", "doc-2"], "b.pdf": ["doc-3", "doc-4"]}
    )
    client = _client(storage)

    first = client.get(
        "/documents/source_conflicts", headers=_HEADERS, params={"limit": 1}
    ).json()
    assert [c["canonical_source_key"] for c in first["conflicts"]] == ["a.pdf"]
    cursor = first["next_cursor"]
    # The backend's own token must not appear verbatim in the client cursor.
    assert cursor is not None and "a.pdf" not in cursor

    second = client.get(
        "/documents/source_conflicts",
        headers=_HEADERS,
        params={"limit": 1, "cursor": cursor},
    ).json()
    assert [c["canonical_source_key"] for c in second["conflicts"]] == ["b.pdf"]
    # The endpoint handed the decoded backend token back to storage.
    assert storage.list_calls[-1][1].opaque == "a.pdf"


def test_malformed_cursor_is_a_client_error():
    """A garbled cursor must not be reported as an unavailable service: the
    envelope is validated at the endpoint, before storage sees it."""
    storage = _ConflictDocStatus({"a.pdf": ["doc-1", "doc-2"]})
    response = _client(storage).get(
        "/documents/source_conflicts",
        headers=_HEADERS,
        params={"cursor": "not-base64!!"},
    )

    assert response.status_code == 400
    assert "cursor" in response.json()["detail"].lower()
    assert storage.list_calls == []  # never reached storage


def test_listing_limit_is_bounded():
    storage = _ConflictDocStatus({})
    client = _client(storage)

    assert (
        client.get(
            "/documents/source_conflicts", headers=_HEADERS, params={"limit": 0}
        ).status_code
        == 422
    )
    assert (
        client.get(
            "/documents/source_conflicts", headers=_HEADERS, params={"limit": 10_000}
        ).status_code
        == 422
    )


def test_listing_maps_capability_and_storage_failures():
    storage = _ConflictDocStatus({})
    client = _client(storage)

    storage.list_error = StorageCapabilityError("no strict source resolution")
    assert (
        client.get("/documents/source_conflicts", headers=_HEADERS).status_code == 501
    )

    storage.list_error = StorageControlPlaneError("index rebuilding")
    response = client.get("/documents/source_conflicts", headers=_HEADERS)
    assert response.status_code == 503
    # The storage message never reaches the client verbatim.
    assert "rebuilding" not in response.json()["detail"]


# --------------------------------------------------------------------------- #
# repair
# --------------------------------------------------------------------------- #


def test_dry_run_reports_the_cas_token_without_mutating():
    storage = _ConflictDocStatus({"a.pdf": ["doc-1", "doc-2"]})
    response = _client(storage).post(
        "/documents/source_conflicts/repair",
        headers=_HEADERS,
        json={"canonical_source_key": "a.pdf", "primary_doc_id": "doc-2"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["committed"] is False
    assert body["candidate_count"] == 2
    assert body["fingerprint"] == _fingerprint(["doc-1", "doc-2"])
    assert body["demoted_sample_doc_ids"] == ["doc-1"]
    assert storage.groups["a.pdf"] == ["doc-1", "doc-2"]  # untouched
    assert storage.repair_calls[-1]["dry_run"] is True


def test_commit_requires_the_cas_tokens():
    """Without the echoed tokens there is nothing to compare against, so the
    commit is refused before any backend sees a placeholder expectation."""
    storage = _ConflictDocStatus({"a.pdf": ["doc-1", "doc-2"]})
    response = _client(storage).post(
        "/documents/source_conflicts/repair",
        headers=_HEADERS,
        json={
            "canonical_source_key": "a.pdf",
            "primary_doc_id": "doc-2",
            "dry_run": False,
        },
    )

    assert response.status_code == 422
    assert storage.repair_calls == []


def test_commit_demotes_and_then_resolves_uniquely():
    storage = _ConflictDocStatus({"a.pdf": ["doc-1", "doc-2"]})
    client = _client(storage)

    dry = client.post(
        "/documents/source_conflicts/repair",
        headers=_HEADERS,
        json={"canonical_source_key": "a.pdf", "primary_doc_id": "doc-2"},
    ).json()
    commit = client.post(
        "/documents/source_conflicts/repair",
        headers=_HEADERS,
        json={
            "canonical_source_key": "a.pdf",
            "primary_doc_id": "doc-2",
            "expected_candidate_count": dry["candidate_count"],
            "expected_candidate_fingerprint": dry["fingerprint"],
            "dry_run": False,
        },
    )

    assert commit.status_code == 200
    assert commit.json()["committed"] is True
    assert storage.groups["a.pdf"] == ["doc-2"]
    # The listing no longer reports it.
    listed = client.get("/documents/source_conflicts", headers=_HEADERS).json()
    assert listed["conflicts"] == []


def test_stale_cas_token_is_a_conflict_not_a_service_error():
    """The CAS failure subclasses StorageControlPlaneError; it must be caught
    ahead of its parent so a moved candidate set reads as 409 (retry the
    dry-run) rather than 503 (storage unavailable)."""
    storage = _ConflictDocStatus({"a.pdf": ["doc-1", "doc-2"]})
    response = _client(storage).post(
        "/documents/source_conflicts/repair",
        headers=_HEADERS,
        json={
            "canonical_source_key": "a.pdf",
            "primary_doc_id": "doc-2",
            "expected_candidate_count": 7,
            "expected_candidate_fingerprint": "stale",
            "dry_run": False,
        },
    )

    assert response.status_code == 409
    assert "dry-run" in response.json()["detail"]


def test_unknown_primary_is_refused():
    storage = _ConflictDocStatus({"a.pdf": ["doc-1", "doc-2"]})
    response = _client(storage).post(
        "/documents/source_conflicts/repair",
        headers=_HEADERS,
        json={"canonical_source_key": "a.pdf", "primary_doc_id": "doc-typo"},
    )

    assert response.status_code == 409
    assert "doc-typo" in response.json()["detail"]


def test_repair_maps_capability_and_storage_failures():
    storage = _ConflictDocStatus({"a.pdf": ["doc-1", "doc-2"]})
    client = _client(storage)
    payload = {"canonical_source_key": "a.pdf", "primary_doc_id": "doc-2"}

    storage.repair_error = StorageCapabilityError("cannot repair")
    assert (
        client.post(
            "/documents/source_conflicts/repair", headers=_HEADERS, json=payload
        ).status_code
        == 501
    )

    storage.repair_error = StorageControlPlaneError("lock unavailable")
    response = client.post(
        "/documents/source_conflicts/repair", headers=_HEADERS, json=payload
    )
    assert response.status_code == 503
    assert "lock unavailable" not in response.json()["detail"]


def test_every_repair_is_audited(monkeypatch):
    """A commit is an operator action on someone else's data: dry-runs land at
    INFO, commits and refusals at WARNING, with the identifiers sanitized so a
    crafted key cannot forge extra log lines."""
    infos: list[str] = []
    warnings: list[str] = []
    monkeypatch.setattr(_document_routes.logger, "info", infos.append)
    monkeypatch.setattr(_document_routes.logger, "warning", warnings.append)

    storage = _ConflictDocStatus({"a\nb.pdf": ["doc-1", "doc-2"]})
    client = _client(storage)

    dry = client.post(
        "/documents/source_conflicts/repair",
        headers=_HEADERS,
        json={"canonical_source_key": "a\nb.pdf", "primary_doc_id": "doc-2"},
    ).json()
    audit = [line for line in infos if "source-conflict repair" in line]
    assert len(audit) == 1
    assert "dry-run" in audit[0]
    assert "\n" not in audit[0]  # newline in the key neutralized

    client.post(
        "/documents/source_conflicts/repair",
        headers=_HEADERS,
        json={
            "canonical_source_key": "a\nb.pdf",
            "primary_doc_id": "doc-2",
            "expected_candidate_count": dry["candidate_count"],
            "expected_candidate_fingerprint": dry["fingerprint"],
            "dry_run": False,
        },
    )
    commits = [
        line
        for line in warnings
        if "source-conflict repair" in line and "COMMIT" in line
    ]
    assert len(commits) == 1
    assert "REFUSED" not in commits[0]
    assert "doc-2" in commits[0]

    # A refusal is audited too — a failed attempt is part of the trail.
    client.post(
        "/documents/source_conflicts/repair",
        headers=_HEADERS,
        json={"canonical_source_key": "a\nb.pdf", "primary_doc_id": "doc-1"},
    )
    assert any("REFUSED" in line for line in warnings)


def test_endpoints_require_authentication():
    """Both endpoints are behind ``combined_auth``, and a refused caller must not
    reach the storage.

    The refusal CODE is deliberately not pinned: the shared dependency answers
    401 ("please login") when password auth is configured and 403 ("API Key
    required") when only an API key is, so hard-coding either makes this test
    pass or fail on whether the developer happens to have ``AUTH_ACCOUNTS`` in a
    local ``.env`` — it passed locally and failed in CI for exactly that reason.
    Which code the dependency picks is its own tests' business; what belongs here
    is that these two routes are gated and the repair never ran.
    """
    storage = _ConflictDocStatus({"a.pdf": ["doc-1", "doc-2"]})
    client = _client(storage)

    listing = client.get("/documents/source_conflicts")
    repair = client.post(
        "/documents/source_conflicts/repair",
        json={"canonical_source_key": "a.pdf", "primary_doc_id": "doc-2"},
    )

    assert listing.status_code in (401, 403), listing.status_code
    assert repair.status_code in (401, 403), repair.status_code
    assert storage.repair_calls == []


def test_a_commit_that_leaves_the_key_unsettled_is_not_reported_as_success(
    monkeypatch,
):
    """``committed=True`` only claims the named demotions landed. The repair locks
    exclude a concurrent ENQUEUE and nothing else — a concurrent delete of the
    kept primary, a processing-stage duplicate marking, or a scan stale-stub
    deletion all mutate the candidate set without them — so the end state is
    verified and reported, never inferred.

    Fix-proof: the endpoint used to return 200 "committed" for exactly this.
    """
    doc_status = _ConflictDocStatus({"a.pdf": ["doc-1", "doc-2"]})
    client = _client(doc_status)

    dry = client.post(
        "/documents/source_conflicts/repair",
        json={"canonical_source_key": "a.pdf", "primary_doc_id": "doc-2"},
        headers=_HEADERS,
    )
    assert dry.status_code == 200

    original_repair = doc_status.repair_source_conflict

    async def _repair_then_lose_the_primary(*args, **kwargs):
        result = await original_repair(*args, **kwargs)
        if not kwargs.get("dry_run", True):
            # A concurrent delete removes the primary we just kept.
            doc_status.groups["a.pdf"] = []
        return result

    doc_status.repair_source_conflict = _repair_then_lose_the_primary

    body = dry.json()
    commit = client.post(
        "/documents/source_conflicts/repair",
        json={
            "canonical_source_key": "a.pdf",
            "primary_doc_id": "doc-2",
            "expected_candidate_count": body["candidate_count"],
            "expected_candidate_fingerprint": body["fingerprint"],
            "dry_run": False,
        },
        headers=_HEADERS,
    )
    assert commit.status_code != 200


def test_a_primary_with_no_content_is_refused():
    """A row with no ``full_docs`` content is an unprocessable stub: it can never
    own the source, and scan classification deletes exactly such rows on sight
    (STALE_STUB fires only when the content is CONFIRMED absent), which would
    delete the primary and leave the demoted rows pointing at a missing id — with
    no way back, since a repair only demotes.

    Fix-proof: the repair used to accept it and report success. Refusing here is
    what removes the interaction at its root, instead of locking scan
    classification against the repair.
    """
    doc_status = _ConflictDocStatus({"a.pdf": ["doc-1", "doc-2"]})
    full_docs = _FullDocs({"doc-1": {"content": "real body"}})  # doc-2 is a stub
    client = _client(doc_status, full_docs)

    dry = client.post(
        "/documents/source_conflicts/repair",
        headers=_HEADERS,
        json={"canonical_source_key": "a.pdf", "primary_doc_id": "doc-2"},
    ).json()

    commit = client.post(
        "/documents/source_conflicts/repair",
        headers=_HEADERS,
        json={
            "canonical_source_key": "a.pdf",
            "primary_doc_id": "doc-2",
            "expected_candidate_count": dry["candidate_count"],
            "expected_candidate_fingerprint": dry["fingerprint"],
            "dry_run": False,
        },
    )

    assert commit.status_code == 409
    assert "no full_docs content" in commit.json()["detail"]
    # ... and only for THIS reason (the sibling refusal has its own sentence).
    assert "same content" not in commit.json()["detail"]
    # Nothing demoted: both rows still claim the source.
    assert doc_status.groups["a.pdf"] == ["doc-1", "doc-2"]
    assert full_docs.reads == ["doc-2"]  # only the chosen primary is read


def test_a_primary_with_content_is_accepted():
    """The complement — the check must not get in the way of a real repair."""
    doc_status = _ConflictDocStatus({"a.pdf": ["doc-1", "doc-2"]})
    full_docs = _FullDocs({"doc-1": {"content": "x"}, "doc-2": {"content": "y"}})
    client = _client(doc_status, full_docs)

    dry = client.post(
        "/documents/source_conflicts/repair",
        headers=_HEADERS,
        json={"canonical_source_key": "a.pdf", "primary_doc_id": "doc-2"},
    ).json()
    commit = client.post(
        "/documents/source_conflicts/repair",
        headers=_HEADERS,
        json={
            "canonical_source_key": "a.pdf",
            "primary_doc_id": "doc-2",
            "expected_candidate_count": dry["candidate_count"],
            "expected_candidate_fingerprint": dry["fingerprint"],
            "dry_run": False,
        },
    )
    assert commit.status_code == 200
    assert doc_status.groups["a.pdf"] == ["doc-2"]


def test_a_failed_content_read_is_a_503_not_a_committed_demotion():
    """Fix-proof: a strict read that RAISED used to warn and commit anyway. That
    spends the one guarantee the check exists for — the demotions cannot be undone
    (a repair only demotes, and a key left with no candidate cannot be repaired
    again), so committing on an unverified primary trades an irreversible loss for
    a retry the operator can simply make later. 503, and nothing demoted."""
    doc_status = _ConflictDocStatus({"a.pdf": ["doc-1", "doc-2"]})
    full_docs = _FullDocs(read_error=RuntimeError("connection reset"))
    client = _client(doc_status, full_docs)

    dry = client.post(
        "/documents/source_conflicts/repair",
        headers=_HEADERS,
        json={"canonical_source_key": "a.pdf", "primary_doc_id": "doc-2"},
    ).json()
    commit = client.post(
        "/documents/source_conflicts/repair",
        headers=_HEADERS,
        json={
            "canonical_source_key": "a.pdf",
            "primary_doc_id": "doc-2",
            "expected_candidate_count": dry["candidate_count"],
            "expected_candidate_fingerprint": dry["fingerprint"],
            "dry_run": False,
        },
    )

    assert commit.status_code == 503
    assert doc_status.groups["a.pdf"] == ["doc-1", "doc-2"]  # nothing demoted
    # Only the operator's own dry-run ran; the commit never reached the backend.
    assert [call["dry_run"] for call in doc_status.repair_calls] == [True]


def test_a_backend_that_cannot_prove_content_is_a_503_not_a_commit():
    """Fix-proof: a backend with no strict point reads used to warn and commit.
    The old reasoning — such a backend's scan cannot delete the stub either, so
    the interaction cannot arise — covers only one of the two harms: a stub can
    never be processed, so it would own the canonical source forever, whatever the
    scan does. Unverified is not verified, and the demotions are irreversible."""
    doc_status = _ConflictDocStatus({"a.pdf": ["doc-1", "doc-2"]})
    full_docs = _FullDocs({}, strict=False)  # nothing known, nothing provable
    client = _client(doc_status, full_docs)

    dry = client.post(
        "/documents/source_conflicts/repair",
        headers=_HEADERS,
        json={"canonical_source_key": "a.pdf", "primary_doc_id": "doc-2"},
    ).json()
    commit = client.post(
        "/documents/source_conflicts/repair",
        headers=_HEADERS,
        json={
            "canonical_source_key": "a.pdf",
            "primary_doc_id": "doc-2",
            "expected_candidate_count": dry["candidate_count"],
            "expected_candidate_fingerprint": dry["fingerprint"],
            "dry_run": False,
        },
    )
    assert commit.status_code == 503
    assert full_docs.reads == []  # the capability is missing; no read to attempt
    assert doc_status.groups["a.pdf"] == ["doc-1", "doc-2"]  # nothing demoted


def test_a_primary_whose_content_lives_under_another_source_is_refused():
    """The processing stage marks a content duplicate FAILED and deletes its
    content, and it holds no source-key lock — so a primary that ALREADY shares
    its content hash with a document under a different source cannot keep this
    one: the key would end up with no primary, which no repair can settle (the
    endpoint only accepts a current primary candidate).

    Refusing here is the same move the contentless-stub check makes, for the same
    reason: it removes the interaction before the irreversible demotions, where a
    lock could not (the marking may land the moment the lock is released).
    """
    doc_status = _ConflictDocStatus(
        {"a.pdf": ["doc-1", "doc-2"]},
        content_hashes={"doc-2": "hash-x", "doc-elsewhere": "hash-x"},
        source_of={"doc-elsewhere": "other.pdf"},
    )
    client = _client(doc_status, _FullDocs({"doc-2": {"content": "body"}}))

    dry = client.post(
        "/documents/source_conflicts/repair",
        headers=_HEADERS,
        json={"canonical_source_key": "a.pdf", "primary_doc_id": "doc-2"},
    ).json()
    commit = client.post(
        "/documents/source_conflicts/repair",
        headers=_HEADERS,
        json={
            "canonical_source_key": "a.pdf",
            "primary_doc_id": "doc-2",
            "expected_candidate_count": dry["candidate_count"],
            "expected_candidate_fingerprint": dry["fingerprint"],
            "dry_run": False,
        },
    )

    assert commit.status_code == 409
    assert doc_status.groups["a.pdf"] == ["doc-1", "doc-2"]  # nothing demoted
    # Fix-proof: the detail is built here (the exception text is not guaranteed
    # client-safe), and both reasons rendered the SAME sentence — so a document
    # whose content was fine was told it had none. The reason travels with the
    # exception now.
    detail = commit.json()["detail"]
    assert "no full_docs content" not in detail
    assert "doc-elsewhere" in detail and "same content" in detail


def test_a_content_twin_among_the_candidates_is_not_a_reason_to_refuse():
    """The complement, and the check's documented boundary: two candidates for the
    SAME key holding identical content is the ordinary conflict to settle (a
    custom-ID insert over a scanned file), not a reason to refuse. Only a holder
    under a DIFFERENT source dooms the primary."""
    doc_status = _ConflictDocStatus(
        {"a.pdf": ["doc-1", "doc-2"]},
        content_hashes={"doc-1": "hash-x", "doc-2": "hash-x"},
    )
    client = _client(doc_status, _FullDocs({"doc-2": {"content": "body"}}))

    dry = client.post(
        "/documents/source_conflicts/repair",
        headers=_HEADERS,
        json={"canonical_source_key": "a.pdf", "primary_doc_id": "doc-2"},
    ).json()
    commit = client.post(
        "/documents/source_conflicts/repair",
        headers=_HEADERS,
        json={
            "canonical_source_key": "a.pdf",
            "primary_doc_id": "doc-2",
            "expected_candidate_count": dry["candidate_count"],
            "expected_candidate_fingerprint": dry["fingerprint"],
            "dry_run": False,
        },
    )

    assert commit.status_code == 200
    assert doc_status.groups["a.pdf"] == ["doc-2"]
