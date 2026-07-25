"""Pre-body admission middleware (LR2 Phase 5-b, §9.3).

A FastAPI handler runs only after its ``UploadFile`` / Pydantic parameters are
parsed, i.e. after the body has been read. The middleware moves the decision
ahead of the first ``receive()``, which is what these tests pin:

* a refused request never has ``receive()`` called — so no body is transferred
  and (with ``Expect: 100-continue``) the server never invites the client to
  send one;
* the reservation is atomic, not a stateless guess: capacity 1 with many
  concurrent requests admits exactly one;
* an unauthenticated caller is refused before reserving AND before the body;
* the reservation travels to the route (adopt) instead of being taken twice, and
  is released by whichever layer still owns it.
"""

from __future__ import annotations

import asyncio
import importlib
import sys
from types import SimpleNamespace
from uuid import uuid4

import pytest

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_document_routes = importlib.import_module("lightrag.api.routers.document_routes")
_shared_storage = importlib.import_module("lightrag.kg.shared_storage")
_admission = importlib.import_module("lightrag.api.admission")
_middleware_mod = importlib.import_module("lightrag.api.admission_middleware")
_utils_api = importlib.import_module("lightrag.api.utils_api")
sys.argv = _original_argv

AdmissionMiddleware = _middleware_mod.AdmissionMiddleware

pytestmark = pytest.mark.offline


class _CountingDocStatus:
    def __init__(self, active: int):
        self.active = active

    async def count_docs_by_statuses(self, statuses, *, strict=True):
        return self.active


async def _rag(*, capacity: int, active: int = 0):
    workspace = f"mw-{uuid4().hex[:8]}"
    _shared_storage.initialize_share_data()
    await _shared_storage.initialize_pipeline_status(workspace=workspace)
    return SimpleNamespace(
        workspace=workspace,
        doc_status=_CountingDocStatus(active),
        max_pending_documents=capacity,
    )


class _Downstream:
    """Records whether the app was reached, whether the body was read, and what
    ticket (if any) the route would have adopted."""

    def __init__(self, *, adopt: bool = True, fail: bool = False):
        self.calls = 0
        self.body_reads = 0
        self.tickets: list = []
        self._adopt = adopt
        self._fail = fail

    async def __call__(self, scope, receive, send):
        self.calls += 1
        ticket = (scope.get("state") or {}).get(_admission.ADMISSION_STATE_KEY)
        self.tickets.append(ticket)
        if self._adopt and ticket is not None:
            ticket.adopted = True
        message = await receive()
        self.body_reads += 1
        assert message["type"] == "http.request"
        if self._fail:
            raise RuntimeError("route blew up after adopting")
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": b"{}"})


def _scope(path="/documents/upload", method="POST", headers=None):
    return {
        "type": "http",
        "method": method,
        "path": path,
        "headers": headers or [],
    }


class _Recorder:
    """Captures the ASGI response and counts ``receive()`` calls."""

    def __init__(self):
        self.messages: list[dict] = []
        self.receives = 0

    async def receive(self):
        self.receives += 1
        return {"type": "http.request", "body": b"x", "more_body": False}

    async def send(self, message):
        self.messages.append(message)

    @property
    def status(self):
        for message in self.messages:
            if message["type"] == "http.response.start":
                return message["status"]
        return None

    @property
    def headers(self) -> dict[str, str]:
        for message in self.messages:
            if message["type"] == "http.response.start":
                return {
                    k.decode("latin-1"): v.decode("latin-1")
                    for k, v in message["headers"]
                }
        return {}

    def body(self) -> bytes:
        return b"".join(
            m.get("body", b"")
            for m in self.messages
            if m["type"] == "http.response.body"
        )


def _mw(rag, downstream, **kwargs):
    return AdmissionMiddleware(downstream, rag_getter=lambda: rag, **kwargs)


async def _tokens(rag) -> dict:
    status = await _shared_storage.get_namespace_data(
        "pipeline_status", workspace=rag.workspace
    )
    return dict(status.get("pending_enqueue_tokens", {}))


@pytest.fixture(autouse=True)
def _open_auth(monkeypatch):
    """Default to the fully-open auth profile; auth tests opt back in."""
    monkeypatch.setattr(_utils_api, "auth_configured", False)


async def test_refused_request_never_reads_the_body():
    """The whole point: over capacity, ``receive()`` is not called, so the body
    is not transferred and no 100-continue is emitted."""
    rag = await _rag(capacity=1, active=1)
    downstream = _Downstream()
    recorder = _Recorder()

    await _mw(rag, downstream)(_scope(), recorder.receive, recorder.send)

    assert recorder.status == 429
    assert recorder.receives == 0
    assert downstream.calls == 0
    assert recorder.headers["retry-after"]
    assert b"capacity" in recorder.body()
    assert await _tokens(rag) == {}


async def test_capacity_one_admits_exactly_one_of_many_concurrent_requests():
    """Not a stateless pre-check: with capacity 1 and 50 simultaneous uploads,
    one request reserves and reads a body; the rest are refused before theirs."""
    rag = await _rag(capacity=1, active=0)
    downstream = _Downstream()
    middleware = _mw(rag, downstream)

    async def _one():
        recorder = _Recorder()
        await middleware(_scope(), recorder.receive, recorder.send)
        return recorder

    recorders = await asyncio.gather(*[_one() for _ in range(50)])

    admitted = [r for r in recorders if r.status == 200]
    refused = [r for r in recorders if r.status == 429]
    assert len(admitted) == 1
    assert len(refused) == 49
    assert downstream.body_reads == 1
    assert sum(r.receives for r in refused) == 0


async def test_unauthenticated_request_is_refused_before_reserving():
    rag = await _rag(capacity=10)
    downstream = _Downstream()
    recorder = _Recorder()

    await _mw(rag, downstream, api_key="secret")(
        _scope(), recorder.receive, recorder.send
    )

    assert recorder.status == 401
    assert recorder.receives == 0
    assert downstream.calls == 0
    # Nothing reserved: an anonymous caller cannot consume capacity.
    assert await _tokens(rag) == {}


async def test_valid_api_key_is_admitted():
    rag = await _rag(capacity=10)
    downstream = _Downstream()
    recorder = _Recorder()

    await _mw(rag, downstream, api_key="secret")(
        _scope(headers=[(b"x-api-key", b"secret")]),
        recorder.receive,
        recorder.send,
    )

    assert recorder.status == 200
    assert downstream.calls == 1


async def test_whitelisted_path_skips_pre_auth_like_the_route_does(monkeypatch):
    """An operator who whitelists the ingestion paths must not start getting
    401s from the middleware — the route itself waves those through."""
    monkeypatch.setattr(_utils_api, "whitelist_patterns", [("", True)])  # "/*"
    rag = await _rag(capacity=10)
    downstream = _Downstream()
    recorder = _Recorder()

    await _mw(rag, downstream, api_key="secret")(
        _scope(), recorder.receive, recorder.send
    )

    assert recorder.status == 200


async def test_ticket_is_published_and_adopted_exactly_once():
    rag = await _rag(capacity=10)
    downstream = _Downstream(adopt=True)
    recorder = _Recorder()

    await _mw(rag, downstream)(_scope(), recorder.receive, recorder.send)

    ticket = downstream.tickets[0]
    assert ticket is not None and ticket.adopted is True
    # Adopted → the middleware left it in place for the background task.
    assert list((await _tokens(rag)).keys()) == [ticket.token]


async def test_unadopted_ticket_is_released_by_the_middleware():
    """The route never adopted (rejected before adoption, or no middleware-aware
    route ran), so the reservation must not outlive the request."""
    rag = await _rag(capacity=10)
    downstream = _Downstream(adopt=False)
    recorder = _Recorder()

    await _mw(rag, downstream)(_scope(), recorder.receive, recorder.send)

    assert recorder.status == 200
    assert await _tokens(rag) == {}


async def test_route_exception_after_adoption_leaves_ownership_downstream():
    """Once adopted, the endpoint/background task owns the release; the
    middleware must not release it a second time (nor swallow the error)."""
    rag = await _rag(capacity=10)
    downstream = _Downstream(adopt=True, fail=True)
    recorder = _Recorder()

    with pytest.raises(RuntimeError, match="blew up"):
        await _mw(rag, downstream)(_scope(), recorder.receive, recorder.send)

    ticket = downstream.tickets[0]
    assert list((await _tokens(rag)).keys()) == [ticket.token]


async def test_fenced_pipeline_is_409_before_the_body():
    rag = await _rag(capacity=10)
    status = await _shared_storage.get_namespace_data(
        "pipeline_status", workspace=rag.workspace
    )
    status["manual_freeze_requested"] = True
    downstream = _Downstream()
    recorder = _Recorder()

    await _mw(rag, downstream)(_scope(), recorder.receive, recorder.send)

    assert recorder.status == 409
    assert recorder.receives == 0
    assert downstream.calls == 0


@pytest.mark.parametrize(
    "scope_kwargs",
    [
        {"path": "/documents/scan"},  # no body, exempt from capacity by design
        {"path": "/query"},
        {"path": "/documents/upload", "method": "GET"},
    ],
)
async def test_non_ingestion_requests_pass_through_untouched(scope_kwargs):
    rag = await _rag(capacity=1, active=99)
    downstream = _Downstream()
    recorder = _Recorder()

    await _mw(rag, downstream)(_scope(**scope_kwargs), recorder.receive, recorder.send)

    assert downstream.calls == 1
    assert downstream.tickets == [None]  # no ticket published


async def test_api_prefix_is_stripped_before_matching():
    """The middleware runs outside the path-normalizing middleware, so it sees
    the mounted path."""
    rag = await _rag(capacity=1, active=1)
    downstream = _Downstream()
    recorder = _Recorder()

    await _mw(rag, downstream, api_prefix="/lightrag")(
        _scope(path="/lightrag/documents/upload"), recorder.receive, recorder.send
    )

    assert recorder.status == 429
    assert downstream.calls == 0


async def test_disabled_capacity_is_a_pure_passthrough():
    rag = await _rag(capacity=0, active=10_000)
    downstream = _Downstream()
    recorder = _Recorder()

    await _mw(rag, downstream)(_scope(), recorder.receive, recorder.send)

    assert downstream.calls == 1
    assert downstream.tickets == [None]
    assert await _tokens(rag) == {}


async def test_unresolvable_rag_degrades_to_the_route_reservation():
    downstream = _Downstream()
    recorder = _Recorder()

    def _boom():
        raise RuntimeError("rag not built yet")

    middleware = AdmissionMiddleware(downstream, rag_getter=_boom)
    await middleware(_scope(), recorder.receive, recorder.send)

    # Passthrough, NOT a 500: the endpoint's own reservation still enforces the
    # same capacity (just after the body has been read).
    assert downstream.calls == 1
    assert downstream.tickets == [None]


# --------------------------------------------------------------------------- #
# end to end through the real route stack
# --------------------------------------------------------------------------- #


async def test_route_adopts_the_middleware_reservation_instead_of_taking_a_second():
    """One request must hold ONE reservation. The route adopts the middleware's
    ticket; if it took its own the same upload would be charged twice, and a
    capacity-1 workspace would refuse itself."""
    rag = await _rag(capacity=1, active=0)
    seen: dict = {}

    async def _fake_index(rag_arg, file_path, track_id=None, admission_token=None):
        seen["token"] = admission_token
        seen["tokens_during"] = await _tokens(rag)

    reserved_tokens: list[str] = []
    original_reserve = _document_routes._reserve_enqueue_slot

    async def _tracking_reserve(rag_arg, token, **kwargs):
        reserved_tokens.append(token)
        return await original_reserve(rag_arg, token, **kwargs)

    from io import BytesIO

    doc_manager = _document_routes.DocumentManager(
        str(__import__("tempfile").mkdtemp())
    )
    rag.doc_status.resolve_doc_source_strict = None  # not used by upload
    rag.doc_status.get_doc_by_file_basename = lambda *a, **k: None

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(_document_routes, "pipeline_index_file", _fake_index)
        monkey.setattr(_document_routes, "_reserve_enqueue_slot", _tracking_reserve)
        monkey.setattr(
            _document_routes,
            "global_args",
            SimpleNamespace(max_upload_size=None),
        )
        monkey.setattr(
            _document_routes,
            "get_existing_doc_by_file_path_candidates",
            _no_existing_doc,
        )
        router = _document_routes.create_document_routes(rag, doc_manager)
        upload_endpoint = [
            route.endpoint
            for route in router.routes
            if getattr(route, "name", "") == "upload_to_input_dir"
        ][-1]

        ticket = _admission.AdmissionTicket(token="mw-token", weight=1)
        await _document_routes._reserve_enqueue_slot(rag, ticket.token, weight=1)
        request = SimpleNamespace(
            state=SimpleNamespace(**{_admission.ADMISSION_STATE_KEY: ticket})
        )

        upload_file = _document_routes.UploadFile(
            filename="adopted.txt", file=BytesIO(b"hello")
        )
        response = await upload_endpoint(set(), upload_file, request)
        assert response.status == "success"
    finally:
        monkey.undo()

    # The route reserved nothing of its own; the middleware's token is the one
    # the enqueue was told about, and it was still held while indexing ran.
    assert ticket.adopted is True
    assert reserved_tokens == ["mw-token"]
    assert seen["token"] == "mw-token"
    assert list(seen["tokens_during"].keys()) == ["mw-token"]
    # ...and released once the background work finished.
    assert await _tokens(rag) == {}


async def _no_existing_doc(*args, **kwargs):
    return None


# --------------------------------------------------------------------------- #
# request body ceiling (LR2 §9.4)
# --------------------------------------------------------------------------- #


class _StreamingRecorder(_Recorder):
    """Feeds the body in chunks so the counting wrapper is exercised, and reports
    how many chunks the app actually got."""

    def __init__(self, chunks: list[bytes], declared_length: bytes | None = None):
        super().__init__()
        self._chunks = list(chunks)
        self.declared_length = declared_length
        self.delivered = 0

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


class _BodyReader:
    """Downstream app that drains the whole body, like multipart parsing does."""

    def __init__(self):
        self.received = b""
        self.completed = False

    async def __call__(self, scope, receive, send):
        while True:
            message = await receive()
            if message["type"] != "http.request":
                break
            self.received += message.get("body", b"")
            if not message.get("more_body"):
                break
        self.completed = True
        await send(
            {"type": "http.response.start", "status": 200, "headers": []},
        )
        await send({"type": "http.response.body", "body": b"{}"})


async def test_oversized_declared_length_is_refused_before_the_body():
    """The honest-client shortcut: Content-Length alone is enough to say no, and
    no capacity slot is spent on the way."""
    rag = await _rag(capacity=10)
    app = _BodyReader()
    recorder = _StreamingRecorder([b"x" * 100])

    middleware = _mw(rag, app, max_body_bytes=50)
    await middleware(
        _scope(headers=[(b"content-length", b"100")]),
        recorder.receive,
        recorder.send,
    )

    assert recorder.status == 413
    assert recorder.receives == 0
    assert app.completed is False
    assert await _tokens(rag) == {}


async def test_understated_length_is_still_cut_off_mid_stream():
    """Content-Length is a hint, not the protection: a body that keeps coming is
    stopped by the counting wrapper, and nothing was buffered to find out."""
    rag = await _rag(capacity=10)
    app = _BodyReader()
    # Declares 10 bytes, sends 300 in three chunks.
    recorder = _StreamingRecorder(
        [b"x" * 100, b"y" * 100, b"z" * 100], declared_length=b"10"
    )

    middleware = _mw(rag, app, max_body_bytes=150)
    await middleware(
        _scope(headers=[(b"content-length", b"10")]),
        recorder.receive,
        recorder.send,
    )

    assert recorder.status == 413
    # The first chunk passed through, the second tripped the limit: the app never
    # saw a complete body, and the stream was not drained past the ceiling.
    assert app.completed is False
    assert recorder.delivered == 2


async def test_body_within_the_limit_streams_through_untouched():
    rag = await _rag(capacity=10)
    app = _BodyReader()
    recorder = _StreamingRecorder([b"a" * 40, b"b" * 40])

    middleware = _mw(rag, app, max_body_bytes=100)
    await middleware(_scope(), recorder.receive, recorder.send)

    assert recorder.status == 200
    assert app.completed is True
    assert app.received == b"a" * 40 + b"b" * 40


async def test_body_limit_works_with_admission_disabled():
    """§9.4: each branch honours its own switch, so a deployment may enable the
    byte ceiling without any capacity limit."""
    rag = await _rag(capacity=0)
    app = _BodyReader()
    recorder = _StreamingRecorder([b"x" * 200])

    middleware = _mw(rag, app, max_body_bytes=50)
    await middleware(_scope(), recorder.receive, recorder.send)

    assert recorder.status == 413
    assert await _tokens(rag) == {}


async def test_body_limit_releases_an_unadopted_reservation():
    """A request killed by the ceiling must not leave capacity charged."""
    rag = await _rag(capacity=10)
    app = _BodyReader()
    recorder = _StreamingRecorder([b"x" * 200])

    middleware = _mw(rag, app, max_body_bytes=50)
    await middleware(_scope(), recorder.receive, recorder.send)

    assert recorder.status == 413
    assert await _tokens(rag) == {}


async def test_no_limit_configured_leaves_receive_unwrapped():
    rag = await _rag(capacity=10)
    app = _BodyReader()
    recorder = _StreamingRecorder([b"x" * 10_000])

    await _mw(rag, app)(_scope(), recorder.receive, recorder.send)

    assert recorder.status == 200
    assert len(app.received) == 10_000
