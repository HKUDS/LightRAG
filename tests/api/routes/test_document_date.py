"""REST document-date validation and forwarding coverage."""

from __future__ import annotations

import importlib
import inspect
import sys
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from lightrag.chunker import chunking_by_token_size

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_dr = importlib.import_module("lightrag.api.routers.document_routes")
sys.argv = _original_argv

DocumentManager = _dr.DocumentManager
InsertTextRequest = _dr.InsertTextRequest
InsertTextsRequest = _dr.InsertTextsRequest
create_document_routes = _dr.create_document_routes
pipeline_index_file = _dr.pipeline_index_file
pipeline_index_texts = _dr.pipeline_index_texts

pytestmark = pytest.mark.offline

_HEADERS = {"X-API-Key": "test-key"}


class _DocStatus:
    async def get_doc_by_file_basename(self, _basename):
        return None


class _Rag:
    workspace = "document-date-api"
    addon_params: dict = {}
    chunking_func = chunking_by_token_size

    def __init__(self):
        self.doc_status = _DocStatus()


class _PipelineRag(_Rag):
    def __init__(self):
        super().__init__()
        self.enqueued: list[tuple[tuple, dict]] = []
        self.process_calls = 0
        self.errors: list[tuple[list[dict], str | None]] = []

    async def apipeline_enqueue_documents(self, *args, **kwargs):
        self.enqueued.append((args, kwargs))
        return kwargs.get("track_id") or "generated-track"

    async def apipeline_process_enqueue_documents(self):
        self.process_calls += 1

    async def apipeline_enqueue_error_documents(self, error_files, track_id=None):
        self.errors.append((error_files, track_id))


def _patch_ingress_guards(monkeypatch) -> None:
    async def _noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr(_dr, "_reserve_enqueue_slot", _noop)
    monkeypatch.setattr(_dr, "_release_enqueue_slot", _noop)
    monkeypatch.setattr(_dr, "_reweight_enqueue_slot", _noop)


def _build_client(monkeypatch, tmp_path):
    _patch_ingress_guards(monkeypatch)
    monkeypatch.setattr(
        _dr,
        "global_args",
        SimpleNamespace(max_upload_size=None, max_texts_per_request=None),
    )

    captured: dict[str, list] = {"texts": [], "uploads": []}

    async def _index_texts(
        rag,
        texts,
        file_sources=None,
        track_id=None,
        chunking=None,
        resolved_chunking=None,
        admission_token=None,
        document_dates=None,
    ):
        captured["texts"].append(
            {
                "texts": texts,
                "file_sources": file_sources,
                "document_dates": document_dates,
            }
        )

    async def _index_file(
        rag,
        file_path,
        track_id=None,
        admission_token=None,
        document_date=None,
    ):
        captured["uploads"].append(
            {"file_path": file_path, "document_date": document_date}
        )

    monkeypatch.setattr(_dr, "pipeline_index_texts", _index_texts)
    monkeypatch.setattr(_dr, "pipeline_index_file", _index_file)

    app = FastAPI()
    app.state.background_tasks = set()
    app.include_router(
        create_document_routes(
            _Rag(),
            DocumentManager(str(tmp_path)),
            api_key="test-key",
        )
    )
    return TestClient(app), captured


@pytest.mark.parametrize(
    "document_date",
    [
        "0000",
        "2018-0",
        "2018-00",
        "2018-13",
        "2018-00-00",
        "2018-00-01",
        "2018-10-00",
        "2019-02-29",
        "2018-02-30",
        " ",
        " 2018",
        "2018 ",
        "2018/10/01",
        # Date-times and EDTF extensions remain outside the phase-one contract.
        "2018-10-01T00:00:00Z",
        pytest.param("2018?", id="edtf-uncertain"),
        pytest.param("2018~", id="edtf-approximate"),
        pytest.param("2018%", id="edtf-uncertain-and-approximate"),
        pytest.param("201X", id="edtf-unspecified-year-digit"),
        pytest.param("2018/2020", id="edtf-interval"),
        pytest.param("[2018,2019]", id="edtf-set"),
    ],
)
def test_text_request_rejects_invalid_document_date(document_date):
    with pytest.raises(ValidationError, match="YYYY, YYYY-MM, or YYYY-MM-DD"):
        InsertTextRequest(
            text="historical facts",
            file_source="organization.txt",
            document_date=document_date,
        )


@pytest.mark.parametrize(
    "document_date", [None, "", "2018", "2018-10", "2018-10-01", "2024-02-29"]
)
def test_text_request_accepts_document_date_or_sentinel(document_date):
    request = InsertTextRequest(
        text="historical facts",
        file_source="organization.txt",
        document_date=document_date,
    )
    assert request.document_date == document_date


def test_batch_dates_must_align_with_texts():
    with pytest.raises(ValidationError, match="same number of items as texts"):
        InsertTextsRequest(
            texts=["one", "two"],
            file_sources=["one.txt", "two.txt"],
            document_dates=["2018-10-01"],
        )


def test_pipeline_index_texts_keeps_legacy_positional_parameter_order():
    parameters = list(inspect.signature(pipeline_index_texts).parameters)
    assert parameters[:7] == [
        "rag",
        "texts",
        "file_sources",
        "track_id",
        "chunking",
        "resolved_chunking",
        "admission_token",
    ]
    assert parameters[7] == "document_dates"


@pytest.mark.asyncio
async def test_pipeline_index_texts_forwards_aligned_dates_to_enqueue():
    rag = _PipelineRag()

    await pipeline_index_texts(
        rag,
        ["facts from 2018", "undated facts", "explicitly undated"],
        file_sources=["organization-2018.txt", "organization.txt", "undated.txt"],
        track_id="track-texts",
        document_dates=["2018", None, ""],
    )

    assert len(rag.enqueued) == 1
    args, kwargs = rag.enqueued[0]
    assert args == ()
    assert kwargs["input"] == ["facts from 2018", "undated facts", "explicitly undated"]
    assert kwargs["file_paths"] == [
        "organization-2018.txt",
        "organization.txt",
        "undated.txt",
    ]
    assert kwargs["document_dates"] == ["2018", None, ""]
    assert rag.process_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("document_date", ["2018-10", ""])
async def test_pipeline_index_file_forwards_date_through_deferred_parse(
    monkeypatch, tmp_path, document_date
):
    monkeypatch.delenv("LIGHTRAG_PARSER", raising=False)
    file_path = tmp_path / "organization-2018.txt"
    file_path.write_text("historical facts", encoding="utf-8")
    rag = _PipelineRag()

    await pipeline_index_file(
        rag,
        file_path,
        track_id="track-upload",
        document_date=document_date,
    )

    assert len(rag.enqueued) == 1
    args, kwargs = rag.enqueued[0]
    assert args == ("",)
    assert kwargs["file_paths"] == str(file_path)
    assert kwargs["document_dates"] == [document_date]
    assert kwargs["docs_format"] == "pending_parse"
    assert rag.process_calls == 1
    assert rag.errors == []


@pytest.mark.parametrize(
    "path,payload",
    [
        (
            "/documents/text",
            {
                "text": "historical facts",
                "file_source": "organization.txt",
                "document_date": "2018/10/01",
            },
        ),
        (
            "/documents/texts",
            {
                "texts": ["valid historical facts", "invalid historical facts"],
                "file_sources": ["valid.txt", "invalid.txt"],
                "document_dates": ["2018", "2018-02-30"],
            },
        ),
    ],
)
def test_text_endpoints_return_422_for_invalid_dates(
    monkeypatch, tmp_path, path, payload
):
    client, captured = _build_client(monkeypatch, tmp_path)
    response = client.post(path, headers=_HEADERS, json=payload)

    assert response.status_code == 422
    assert "YYYY, YYYY-MM, or YYYY-MM-DD" in str(response.json()["detail"])
    assert captured["texts"] == []


@pytest.mark.parametrize("document_date", ["2018", ""])
def test_text_endpoints_forward_dates_one_to_one(monkeypatch, tmp_path, document_date):
    client, captured = _build_client(monkeypatch, tmp_path)

    single = client.post(
        "/documents/text",
        headers=_HEADERS,
        json={
            "text": "facts from 2018",
            "file_source": "organization-2018.txt",
            "document_date": document_date,
        },
    )
    batch = client.post(
        "/documents/texts",
        headers=_HEADERS,
        json={
            "texts": ["facts from 2019", "undated facts", "explicitly undated"],
            "file_sources": [
                "organization-2019.txt",
                "organization.txt",
                "undated.txt",
            ],
            "document_dates": ["2019-10", None, ""],
        },
    )

    assert single.status_code == 200
    assert batch.status_code == 200
    assert captured["texts"][0]["document_dates"] == [document_date]
    assert captured["texts"][1]["document_dates"] == ["2019-10", None, ""]


@pytest.mark.parametrize("explicit_null", [False, True])
@pytest.mark.parametrize(
    "path,payload,date_field",
    [
        (
            "/documents/text",
            {"text": "undated facts", "file_source": "organization.txt"},
            "document_date",
        ),
        (
            "/documents/texts",
            {"texts": ["undated facts"], "file_sources": ["organization.txt"]},
            "document_dates",
        ),
    ],
)
def test_omitted_or_null_text_dates_keep_the_legacy_none_path(
    monkeypatch, tmp_path, path, payload, date_field, explicit_null
):
    client, captured = _build_client(monkeypatch, tmp_path)
    payload = dict(payload)
    if explicit_null:
        payload[date_field] = None

    response = client.post(
        path,
        headers=_HEADERS,
        json=payload,
    )

    assert response.status_code == 200
    assert captured["texts"][0]["document_dates"] is None


@pytest.mark.parametrize("document_date", ["2018-10", ""])
def test_upload_forwards_valid_form_date(monkeypatch, tmp_path, document_date):
    client, captured = _build_client(monkeypatch, tmp_path)

    response = client.post(
        "/documents/upload",
        headers=_HEADERS,
        files={"file": ("organization.txt", b"historical facts", "text/plain")},
        data={"document_date": document_date},
    )

    assert response.status_code == 200
    assert captured["uploads"][0]["document_date"] == document_date


@pytest.mark.parametrize("document_date", ["2018-02-30", "2018?", " "])
def test_upload_returns_422_before_writing_for_invalid_form_date(
    monkeypatch, tmp_path, document_date
):
    client, captured = _build_client(monkeypatch, tmp_path)

    response = client.post(
        "/documents/upload",
        headers=_HEADERS,
        files={"file": ("organization.txt", b"historical facts", "text/plain")},
        data={"document_date": document_date},
    )

    assert response.status_code == 422
    assert "YYYY, YYYY-MM, or YYYY-MM-DD" in response.json()["detail"]
    assert captured["uploads"] == []
    assert not (tmp_path / "organization.txt").exists()


def test_upload_omitted_date_keeps_the_legacy_none_path(monkeypatch, tmp_path):
    client, captured = _build_client(monkeypatch, tmp_path)

    response = client.post(
        "/documents/upload",
        headers=_HEADERS,
        files={"file": ("organization.txt", b"historical facts", "text/plain")},
    )

    assert response.status_code == 200
    assert captured["uploads"][0]["document_date"] is None
