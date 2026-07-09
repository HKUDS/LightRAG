"""Tests for PaddleOCR-VL official async API client."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import pytest

from lightrag.parser.external.paddleocr_vl.client import PaddleOCRVLRawClient

DEFAULT_PAYLOAD = {
    "useDocOrientationClassify": False,
    "useDocUnwarping": False,
    "useLayoutDetection": True,
    "useChartRecognition": True,
    "useSealRecognition": True,
    "useOcrForImageBlock": False,
    "layoutNms": True,
    "layoutShapeMode": "auto",
    "promptLabel": "ocr",
    "formatBlockContent": False,
    "repetitionPenalty": 1,
    "temperature": 0,
    "topP": 1,
    "minPixels": 147384,
    "maxPixels": 2822400,
    "mergeLayoutBlocks": True,
    "markdownIgnoreLabels": [
        "header",
        "header_image",
        "footer",
        "footer_image",
        "number",
        "footnote",
        "aside_text",
    ],
    "prettifyMarkdown": True,
    "showFormulaNumber": False,
    "restructurePages": True,
    "mergeTables": True,
    "relevelTitles": True,
    "visualize": False,
}


class _Response:
    def __init__(
        self,
        *,
        status_code: int = 200,
        payload: Any | None = None,
        text: str | None = None,
        content: bytes | None = None,
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = (
            text
            if text is not None
            else (json.dumps(payload) if payload is not None else "")
        )
        self.content = content if content is not None else self.text.encode("utf-8")

    def json(self) -> Any:
        if self._payload is not None:
            return self._payload
        return json.loads(self.text)


class _Recorder:
    def __init__(self) -> None:
        self.post_calls: list[dict[str, Any]] = []
        self.get_calls: list[dict[str, Any]] = []


_CURRENT: dict[str, _Recorder] = {}


class _FakeAsyncClient:
    def __init__(self, *_: Any, **__: Any) -> None:
        pass

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        pass

    async def post(self, url: str, **kwargs: Any) -> _Response:
        recorder = _CURRENT["recorder"]
        if url.endswith("/layout-parsing"):
            recorder.post_calls.append({**kwargs, "url": url})
            return _Response(
                payload={
                    "logId": "local-log-1",
                    "errorCode": 0,
                    "errorMsg": "Success",
                    "result": {
                        "layoutParsingResults": [
                            {
                                "prunedResult": {"parsing_res_list": []},
                                "markdown": {
                                    "text": "# local",
                                    "images": {
                                        "imgs/fig.jpg": base64.b64encode(
                                            b"local-fig"
                                        ).decode("ascii")
                                    },
                                },
                                "outputImages": {
                                    "layout_det_res": base64.b64encode(
                                        b"local-layout"
                                    ).decode("ascii")
                                },
                            }
                        ],
                        "dataInfo": {"pageCount": 1},
                    },
                }
            )
        files = kwargs.get("files")
        if files and "file" in files:
            payload = files["file"]
            filename = None
            ctype = None
            if isinstance(payload, tuple):
                filename, payload, ctype = payload
            if hasattr(payload, "read"):
                payload = payload.read()
            files = {"file": (filename, payload, ctype)}
        recorder.post_calls.append({**kwargs, "url": url, "files": files})
        return _Response(payload={"data": {"jobId": "job-123"}})

    async def get(self, url: str, **kwargs: Any) -> _Response:
        recorder = _CURRENT["recorder"]
        recorder.get_calls.append({"url": url, **kwargs})
        if url.endswith("/job-123"):
            return _Response(
                payload={
                    "data": {
                        "state": "done",
                        "extractProgress": {"extractedPages": 1},
                        "resultUrl": {"jsonUrl": "http://files.test/result.jsonl"},
                    }
                }
            )
        if url == "http://files.test/result.jsonl":
            return _Response(
                text=json.dumps(
                    {
                        "result": {
                            "layoutParsingResults": [
                                {
                                    "prunedResult": {"parsing_res_list": []},
                                    "markdown": {
                                        "text": "# hello",
                                        "images": {
                                            "imgs/fig.jpg": "http://files.test/fig.jpg"
                                        },
                                    },
                                    "outputImages": {
                                        "layout_det_res": "http://files.test/layout.jpg"
                                    },
                                }
                            ]
                        }
                    }
                )
            )
        if url in {"http://files.test/fig.jpg", "http://files.test/layout.jpg"}:
            return _Response(content=b"\xff\xd8fake")
        raise AssertionError(f"unexpected GET {url}")


def _install_httpx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "lightrag.parser.external.paddleocr_vl.client.httpx.AsyncClient",
        _FakeAsyncClient,
    )
    monkeypatch.setattr(
        "lightrag.parser.external.paddleocr_vl.client.httpx.Timeout",
        lambda *_, **__: None,
    )


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PADDLEOCR_VL_API_MODE", "official")
    monkeypatch.setenv("PADDLEOCR_VL_API_TOKEN", "token-1")
    monkeypatch.setenv("PADDLEOCR_VL_ENDPOINT", "http://paddle.test/api/v2/ocr/jobs")
    for name in (
        "PADDLEOCR_VL_MODEL",
        "PADDLEOCR_VL_USE_DOC_ORIENTATION_CLASSIFY",
        "PADDLEOCR_VL_USE_DOC_UNWARPING",
        "PADDLEOCR_VL_USE_CHART_RECOGNITION",
    ):
        monkeypatch.delenv(name, raising=False)


async def test_client_submits_polls_downloads_result_and_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "demo.pdf"
    source.write_bytes(b"%PDF fake")
    recorder = _Recorder()
    _CURRENT["recorder"] = recorder
    _install_httpx(monkeypatch)

    raw_dir = tmp_path / "demo.paddleocr_vl_raw"
    await PaddleOCRVLRawClient().download_into(
        raw_dir, source, upload_name="canonical.pdf"
    )

    assert recorder.post_calls[0]["url"] == "http://paddle.test/api/v2/ocr/jobs"
    assert recorder.post_calls[0]["headers"]["Authorization"] == "bearer token-1"
    assert recorder.post_calls[0]["data"]["model"] == "PaddleOCR-VL-1.6"
    assert (
        json.loads(recorder.post_calls[0]["data"]["optionalPayload"]) == DEFAULT_PAYLOAD
    )
    assert recorder.post_calls[0]["files"]["file"] == (
        "canonical.pdf",
        b"%PDF fake",
        "application/octet-stream",
    )

    assert (raw_dir / "content_list.json").is_file()
    assert (raw_dir / "imgs" / "fig.jpg").read_bytes() == b"\xff\xd8fake"
    assert (
        raw_dir / "outputImages" / "layout_det_res_0.jpg"
    ).read_bytes() == b"\xff\xd8fake"
    assert (raw_dir / "_manifest.json").is_file()


async def test_client_sends_documented_optional_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "demo.pdf"
    source.write_bytes(b"%PDF fake")
    recorder = _Recorder()
    _CURRENT["recorder"] = recorder
    _install_httpx(monkeypatch)
    monkeypatch.setenv("PADDLEOCR_VL_USE_LAYOUT_DETECTION", "false")
    monkeypatch.setenv("PADDLEOCR_VL_LAYOUT_THRESHOLD", "0.42")
    monkeypatch.setenv("PADDLEOCR_VL_LAYOUT_NMS", "true")
    monkeypatch.setenv("PADDLEOCR_VL_LAYOUT_UNCLIP_RATIO", "[1.0, 1.2]")
    monkeypatch.setenv("PADDLEOCR_VL_LAYOUT_MERGE_BBOXES_MODE", "union")
    monkeypatch.setenv("PADDLEOCR_VL_LAYOUT_SHAPE_MODE", "poly")
    monkeypatch.setenv("PADDLEOCR_VL_PROMPT_LABEL", "chart")
    monkeypatch.setenv("PADDLEOCR_VL_REPETITION_PENALTY", "1.1")
    monkeypatch.setenv("PADDLEOCR_VL_TEMPERATURE", "0.2")
    monkeypatch.setenv("PADDLEOCR_VL_TOP_P", "0.9")
    monkeypatch.setenv("PADDLEOCR_VL_MIN_PIXELS", "1024")
    monkeypatch.setenv("PADDLEOCR_VL_MAX_PIXELS", "4096")
    monkeypatch.setenv("PADDLEOCR_VL_SHOW_FORMULA_NUMBER", "true")
    monkeypatch.setenv("PADDLEOCR_VL_RESTRUCTURE_PAGES", "true")
    monkeypatch.setenv("PADDLEOCR_VL_MERGE_TABLES", "false")
    monkeypatch.setenv("PADDLEOCR_VL_RELEVEL_TITLES", "false")
    monkeypatch.setenv("PADDLEOCR_VL_PRETTIFY_MARKDOWN", "true")
    monkeypatch.setenv("PADDLEOCR_VL_VISUALIZE", "false")

    await PaddleOCRVLRawClient().download_into(
        tmp_path / "demo.paddleocr_vl_raw", source
    )

    assert json.loads(recorder.post_calls[0]["data"]["optionalPayload"]) == {
        **DEFAULT_PAYLOAD,
        # Env-set overrides
        "useLayoutDetection": False,
        "layoutThreshold": 0.42,
        "layoutUnclipRatio": [1.0, 1.2],
        "layoutMergeBboxesMode": "union",
        "layoutShapeMode": "poly",
        "promptLabel": "chart",
        "repetitionPenalty": 1.1,
        "temperature": 0.2,
        "topP": 0.9,
        "minPixels": 1024,
        "maxPixels": 4096,
        "showFormulaNumber": True,
        "mergeTables": False,
        "relevelTitles": False,
        "prettifyMarkdown": True,
        "visualize": False,
    }


async def test_client_applies_per_file_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "demo.pdf"
    source.write_bytes(b"%PDF fake")
    recorder = _Recorder()
    _CURRENT["recorder"] = recorder
    _install_httpx(monkeypatch)
    monkeypatch.setenv("PADDLEOCR_VL_USE_CHART_RECOGNITION", "false")

    await PaddleOCRVLRawClient(
        overrides={
            "use_chart_recognition": True,
            "layout_threshold": 0.7,
            "prettify_markdown": True,
        }
    ).download_into(tmp_path / "demo.paddleocr_vl_raw", source)

    assert json.loads(recorder.post_calls[0]["data"]["optionalPayload"]) == {
        **DEFAULT_PAYLOAD,
        # Overridden values
        "useChartRecognition": True,
        "layoutThreshold": 0.7,
        "prettifyMarkdown": True,
        # Env-set value (false) overridden to True — already covered above
    }


async def test_client_sends_async_submit_top_level_parameters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "demo.pdf"
    source.write_bytes(b"%PDF fake")
    recorder = _Recorder()
    _CURRENT["recorder"] = recorder
    _install_httpx(monkeypatch)
    monkeypatch.setenv("PADDLEOCR_VL_PAGE_RANGES", "2,4-6")
    monkeypatch.setenv("PADDLEOCR_VL_BATCH_ID", "batch-20260703")

    await PaddleOCRVLRawClient().download_into(
        tmp_path / "demo.paddleocr_vl_raw", source
    )

    assert recorder.post_calls[0]["data"]["pageRanges"] == "2,4-6"
    assert recorder.post_calls[0]["data"]["batchId"] == "batch-20260703"
    assert "pageRanges" not in json.loads(
        recorder.post_calls[0]["data"]["optionalPayload"]
    )
    assert "batchId" not in json.loads(
        recorder.post_calls[0]["data"]["optionalPayload"]
    )


async def test_official_requests_use_official_endpoint_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "demo.pdf"
    source.write_bytes(b"%PDF fake")
    recorder = _Recorder()
    _CURRENT["recorder"] = recorder
    _install_httpx(monkeypatch)
    monkeypatch.setenv(
        "PADDLEOCR_VL_OFFICIAL_ENDPOINT",
        "http://official-paddle.test/api/v2/ocr/jobs",
    )

    client = PaddleOCRVLRawClient()
    client.endpoint = "http://generic-endpoint.test/should-not-be-used"
    await client.download_into(tmp_path / "demo.paddleocr_vl_raw", source)

    assert recorder.post_calls[0]["url"] == client.official_endpoint
    assert recorder.get_calls[0]["url"] == f"{client.official_endpoint}/job-123"


def test_client_requires_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PADDLEOCR_VL_API_TOKEN", raising=False)

    with pytest.raises(ValueError, match="PADDLEOCR_VL_API_TOKEN is required"):
        PaddleOCRVLRawClient()


def test_client_records_mode_specific_endpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "PADDLEOCR_VL_ENDPOINT", "http://legacy-paddle.test/api/v2/ocr/jobs"
    )
    monkeypatch.setenv(
        "PADDLEOCR_VL_OFFICIAL_ENDPOINT",
        "http://official-paddle.test/api/v2/ocr/jobs/",
    )
    monkeypatch.setenv("PADDLEOCR_VL_LOCAL_ENDPOINT", "http://local-paddle.test/")

    client = PaddleOCRVLRawClient()

    assert client.official_endpoint == "http://official-paddle.test/api/v2/ocr/jobs"
    assert client.local_endpoint == "http://local-paddle.test"
    assert client.endpoint == client.official_endpoint


def test_client_rejects_invalid_api_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PADDLEOCR_VL_API_MODE", "remote")

    with pytest.raises(ValueError, match="PADDLEOCR_VL_API_MODE must be one of"):
        PaddleOCRVLRawClient()


def test_client_local_mode_requires_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PADDLEOCR_VL_API_MODE", "local")
    monkeypatch.delenv("PADDLEOCR_VL_LOCAL_ENDPOINT", raising=False)

    with pytest.raises(ValueError, match="PADDLEOCR_VL_LOCAL_ENDPOINT is required"):
        PaddleOCRVLRawClient()


async def test_local_mode_posts_synchronous_layout_parsing_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "demo.pdf"
    source.write_bytes(b"%PDF local")
    recorder = _Recorder()
    _CURRENT["recorder"] = recorder
    _install_httpx(monkeypatch)
    monkeypatch.setenv("PADDLEOCR_VL_API_MODE", "local")
    monkeypatch.setenv("PADDLEOCR_VL_LOCAL_ENDPOINT", "http://local-paddle")
    monkeypatch.setenv("PADDLEOCR_VL_API_TOKEN", "")

    raw_dir = tmp_path / "demo.paddleocr_vl_raw"
    manifest = await PaddleOCRVLRawClient().download_into(
        raw_dir, source, upload_name="canonical.pdf"
    )

    assert recorder.post_calls[0]["url"] == "http://local-paddle/layout-parsing"
    assert recorder.post_calls[0]["json"]["file"] == base64.b64encode(
        b"%PDF local"
    ).decode("ascii")
    assert recorder.post_calls[0]["json"]["fileType"] == 0
    assert recorder.post_calls[0]["json"]["useLayoutDetection"] is True
    assert "model" not in recorder.post_calls[0]["json"]
    assert "optionalPayload" not in recorder.post_calls[0]["json"]
    assert recorder.get_calls == []
    assert json.loads((raw_dir / "content_list.json").read_text()) == [
        {
            "prunedResult": {"parsing_res_list": []},
            "markdown": {
                "text": "# local",
                "images": {
                    "imgs/fig.jpg": base64.b64encode(b"local-fig").decode("ascii")
                },
            },
            "outputImages": {
                "layout_det_res": base64.b64encode(b"local-layout").decode("ascii")
            },
        }
    ]
    assert (raw_dir / "imgs" / "fig.jpg").read_bytes() == b"local-fig"
    assert (raw_dir / "outputImages" / "layout_det_res_0.jpg").read_bytes() == (
        b"local-layout"
    )
    assert manifest.task_id == "local-log-1"
    assert manifest.api_mode == "local"


@pytest.mark.parametrize("suffix", ["jpeg", "jpg", "png", "tiff", "tif", "bmp", "webp"])
async def test_local_mode_sends_image_file_type(
    suffix: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / f"demo.{suffix}"
    source.write_bytes(b"image local")
    recorder = _Recorder()
    _CURRENT["recorder"] = recorder
    _install_httpx(monkeypatch)
    monkeypatch.setenv("PADDLEOCR_VL_API_MODE", "local")
    monkeypatch.setenv("PADDLEOCR_VL_LOCAL_ENDPOINT", "http://local-paddle")
    monkeypatch.setenv("PADDLEOCR_VL_API_TOKEN", "")

    await PaddleOCRVLRawClient().download_into(
        tmp_path / "demo.paddleocr_vl_raw", source
    )

    assert recorder.post_calls[0]["json"]["fileType"] == 1
