"""Raw request-body ceilings (GHSA-r8jh-295g-vv42).

The ceiling used to live inside the admission middleware and therefore covered
the three ingestion routes admission cares about — so an operator could set it,
watch ``/documents/text`` reject an oversized body in milliseconds, and still
have ``/api/chat`` accept the same body and stall the process on it.

These tests drive the raw ASGI callable rather than a TestClient because the
property that matters is *whether ``receive()`` was called at all*, which a
client-level test cannot observe.
"""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace
from uuid import uuid4

import pytest

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_body_limit_mod = importlib.import_module("lightrag.api.body_limit_middleware")
_admission_mod = importlib.import_module("lightrag.api.admission_middleware")
_shared_storage = importlib.import_module("lightrag.kg.shared_storage")
_utils_api = importlib.import_module("lightrag.api.utils_api")
sys.argv = _original_argv

BodyLimitMiddleware = _body_limit_mod.BodyLimitMiddleware
resolve_body_limits = _body_limit_mod.resolve_body_limits

pytestmark = pytest.mark.offline

DEFAULT_LIMIT = 1024 * 1024
INGEST_LIMIT = 50 * 1024 * 1024
UPLOAD_LIMIT = 101 * 1024 * 1024


def _scope(path="/api/chat", method="POST", headers=None):
    return {
        "type": "http",
        "method": method,
        "path": path,
        "headers": headers or [],
    }


class _Recorder:
    """Supplies receive/send and records what the middleware did."""

    def __init__(self, chunks: list[bytes] | None = None):
        self._chunks = list(chunks or [])
        self.receives = 0
        self.delivered = 0
        self.status = None
        self.body = b""

    async def receive(self):
        self.receives += 1
        if not self._chunks:
            return {"type": "http.disconnect"}
        chunk = self._chunks.pop(0)
        self.delivered += 1
        return {
            "type": "http.request",
            "body": chunk,
            "more_body": bool(self._chunks),
        }

    async def send(self, message):
        if message["type"] == "http.response.start":
            self.status = message["status"]
        elif message["type"] == "http.response.body":
            self.body += message.get("body", b"")


class _BodyReader:
    """Downstream app that drains the whole body, like multipart parsing does."""

    def __init__(self):
        self.received = b""
        self.completed = False
        self.calls = 0

    async def __call__(self, scope, receive, send):
        self.calls += 1
        while True:
            message = await receive()
            if message["type"] != "http.request":
                break
            self.received += message.get("body", b"")
            if not message.get("more_body"):
                break
        self.completed = True
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"{}"})


def _mw(app, **kwargs):
    limits = {
        "default_limit": DEFAULT_LIMIT,
        "ingest_limit": INGEST_LIMIT,
        "upload_limit": UPLOAD_LIMIT,
    }
    limits.update(kwargs)
    return BodyLimitMiddleware(app, **limits)


# --------------------------------------------------------------------------- #
# Enforcement
# --------------------------------------------------------------------------- #


async def test_oversized_declared_length_is_refused_before_the_body():
    """The honest-client shortcut: Content-Length alone is enough to say no."""
    app = _BodyReader()
    recorder = _Recorder([b"x" * 100])

    await _mw(app, default_limit=50)(
        _scope(headers=[(b"content-length", b"100")]),
        recorder.receive,
        recorder.send,
    )

    assert recorder.status == 413
    assert recorder.receives == 0
    assert app.completed is False


async def test_understated_length_is_still_cut_off_mid_stream():
    """Content-Length is a hint, not the protection.

    A body that keeps coming is stopped by the counting wrapper, and nothing was
    buffered to find that out.
    """
    app = _BodyReader()
    recorder = _Recorder([b"x" * 100, b"y" * 100, b"z" * 100])

    await _mw(app, default_limit=150)(
        _scope(headers=[(b"content-length", b"10")]),
        recorder.receive,
        recorder.send,
    )

    assert recorder.status == 413
    # First chunk through, second tripped the limit: the app never saw a complete
    # body and the stream was not drained past the ceiling.
    assert app.completed is False
    assert recorder.delivered == 2


async def test_body_within_the_limit_streams_through_untouched():
    app = _BodyReader()
    recorder = _Recorder([b"a" * 40, b"b" * 40])

    await _mw(app, default_limit=100)(_scope(), recorder.receive, recorder.send)

    assert recorder.status == 200
    assert app.completed is True
    assert app.received == b"a" * 40 + b"b" * 40


async def test_zero_limit_leaves_receive_unwrapped():
    app = _BodyReader()
    recorder = _Recorder([b"x" * 10_000])

    await _mw(app, default_limit=0)(_scope(), recorder.receive, recorder.send)

    assert recorder.status == 200
    assert len(app.received) == 10_000


async def test_non_http_scopes_pass_through():
    app = _BodyReader()
    recorder = _Recorder([b"x"])

    await _mw(app)({"type": "lifespan"}, recorder.receive, recorder.send)

    assert app.calls == 1


# --------------------------------------------------------------------------- #
# Per-route tiers — the whole point of the change
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "path,accepted,refused",
    [
        # An ordinary route gets the tight default. This is the route the
        # advisory used: unauthenticated under the shipped whitelist.
        ("/api/chat", DEFAULT_LIMIT, DEFAULT_LIMIT + 1),
        ("/query", DEFAULT_LIMIT, DEFAULT_LIMIT + 1),
        # Text ingestion gets a far more generous one: pasting a document or
        # batching several is not a chat turn, and a batch insert has no
        # equivalent on the upload route.
        ("/documents/text", DEFAULT_LIMIT + 1, INGEST_LIMIT + 1),
        ("/documents/texts", INGEST_LIMIT, INGEST_LIMIT + 1),
        # Upload derives from MAX_UPLOAD_SIZE plus multipart overhead.
        ("/documents/upload", INGEST_LIMIT + 1, UPLOAD_LIMIT + 1),
    ],
)
async def test_each_route_tier_is_applied(path, accepted, refused):
    for declared, expected in ((accepted, 200), (refused, 413)):
        app = _BodyReader()
        recorder = _Recorder([b""])
        await _mw(app)(
            _scope(path=path, headers=[(b"content-length", str(declared).encode())]),
            recorder.receive,
            recorder.send,
        )
        assert recorder.status == expected, (path, declared)


async def test_api_prefix_is_stripped_before_matching():
    """Under a mount prefix the ingestion tier must still be recognised."""
    app = _BodyReader()
    recorder = _Recorder([b""])

    await _mw(app, api_prefix="/site01")(
        {
            "type": "http",
            "method": "POST",
            "path": "/site01/documents/texts",
            "root_path": "/site01",
            "headers": [(b"content-length", str(DEFAULT_LIMIT + 1).encode())],
        },
        recorder.receive,
        recorder.send,
    )

    # Would be a 413 if the prefix leaked into the match and the route fell back
    # to the ordinary tier.
    assert recorder.status == 200


async def test_a_limit_applies_to_methods_other_than_post():
    app = _BodyReader()
    recorder = _Recorder([b"x" * 200])

    await _mw(app, default_limit=50)(
        _scope(method="PUT"), recorder.receive, recorder.send
    )

    assert recorder.status == 413


# --------------------------------------------------------------------------- #
# Interaction with admission: the ticket must survive a mid-body 413
# --------------------------------------------------------------------------- #


class _CountingDocStatus:
    def __init__(self, active: int):
        self.active = active

    async def count_docs_by_statuses(self, statuses, *, strict=True):
        return self.active


async def _rag(*, capacity: int, active: int = 0):
    workspace = f"bl-{uuid4().hex[:8]}"
    _shared_storage.initialize_share_data()
    await _shared_storage.initialize_pipeline_status(workspace=workspace)
    return SimpleNamespace(
        workspace=workspace,
        doc_status=_CountingDocStatus(active),
        max_pending_documents=capacity,
    )


async def _tokens(rag):
    status = await _shared_storage.get_namespace_data(
        "pipeline_status", workspace=rag.workspace
    )
    return dict(status.get("pending_enqueue_tokens") or {})


@pytest.fixture
def _open_auth(monkeypatch):
    monkeypatch.setattr(_utils_api, "auth_configured", False)


async def test_mid_body_413_still_releases_the_admission_reservation(_open_auth):
    """Stacked exactly as production stacks them: body limit outside admission.

    BodyLimitExceeded is raised from a receive() wrapper installed by the outer
    middleware and travels up through admission's finally block, which is what
    returns the reservation. If admission ever stops wrapping its downstream call
    in try/finally, this leaks a capacity slot per oversized upload.
    """
    rag = await _rag(capacity=10)
    app = _BodyReader()
    admission = _admission_mod.AdmissionMiddleware(app, rag_getter=lambda: rag)
    stacked = _mw(admission, upload_limit=50)
    recorder = _Recorder([b"x" * 200])

    await stacked(_scope(path="/documents/upload"), recorder.receive, recorder.send)

    assert recorder.status == 413
    assert await _tokens(rag) == {}


async def test_oversized_declared_length_never_reaches_admission(_open_auth):
    """No capacity slot is spent on a request that was refused on its size."""
    rag = await _rag(capacity=10)
    app = _BodyReader()
    admission = _admission_mod.AdmissionMiddleware(app, rag_getter=lambda: rag)
    stacked = _mw(admission, upload_limit=50)
    recorder = _Recorder([b"x" * 200])

    await stacked(
        _scope(path="/documents/upload", headers=[(b"content-length", b"200")]),
        recorder.receive,
        recorder.send,
    )

    assert recorder.status == 413
    assert recorder.receives == 0
    assert app.calls == 0
    assert await _tokens(rag) == {}


# --------------------------------------------------------------------------- #
# Configuration resolution
# --------------------------------------------------------------------------- #


def _args(**kwargs):
    base = {
        "max_request_body_bytes": DEFAULT_LIMIT,
        "max_request_body_bytes_explicit": False,
        "max_upload_size": 104857600,
    }
    base.update(kwargs)
    return SimpleNamespace(**base)


def test_defaults_produce_the_three_tiers():
    limits = resolve_body_limits(_args())
    assert limits == {
        "default_limit": DEFAULT_LIMIT,
        "ingest_limit": INGEST_LIMIT,
        "upload_limit": 104857600 + 1024 * 1024,
    }


def test_an_explicit_value_governs_every_non_upload_route():
    """Operator intent wins, ingestion included — otherwise the knob would be
    unable to tighten the routes it names."""
    limits = resolve_body_limits(
        _args(max_request_body_bytes=4096, max_request_body_bytes_explicit=True)
    )
    assert limits["default_limit"] == 4096
    assert limits["ingest_limit"] == 4096
    assert limits["upload_limit"] == 104857600 + 1024 * 1024


def test_an_explicit_value_equal_to_the_default_still_governs_ingestion():
    """The one case a value comparison cannot see.

    Before the tiers existed, ``MAX_REQUEST_BODY_BYTES=N`` applied to
    ``/documents/text`` and ``/documents/texts`` as well. Deciding "was this
    configured?" by testing ``value == DEFAULT_MAX_REQUEST_BODY_BYTES`` reads an
    operator who deliberately set exactly 1 MiB as having set nothing, and hands
    those two routes the 50 MiB built-in tier — silently relaxing a configured
    ceiling 50-fold on upgrade.
    """
    limits = resolve_body_limits(
        _args(
            max_request_body_bytes=DEFAULT_LIMIT,
            max_request_body_bytes_explicit=True,
        )
    )
    assert limits["default_limit"] == DEFAULT_LIMIT
    assert limits["ingest_limit"] == DEFAULT_LIMIT


def test_missing_provenance_is_treated_as_configured():
    """Hand-built args without the flag must not get the wider tier.

    A 413 on a large ingest is visible and recoverable; a silently widened
    ceiling is the failure this parameter exists to prevent, so the unknown case
    resolves to the tighter reading.
    """
    args = SimpleNamespace(
        max_request_body_bytes=DEFAULT_LIMIT, max_upload_size=104857600
    )
    assert resolve_body_limits(args)["ingest_limit"] == DEFAULT_LIMIT


def test_zero_disables_every_ceiling_including_the_derived_upload_one():
    assert resolve_body_limits(_args(max_request_body_bytes=0)) is None


@pytest.mark.parametrize("value", [None, 0, -1])
def test_unlimited_upload_size_leaves_the_upload_route_uncapped(value):
    """``MAX_UPLOAD_SIZE=None`` is documented as "unlimited"; there is then no
    size to derive a body ceiling from."""
    limits = resolve_body_limits(_args(max_upload_size=value))
    assert limits["upload_limit"] == 0
    assert limits["default_limit"] == DEFAULT_LIMIT
