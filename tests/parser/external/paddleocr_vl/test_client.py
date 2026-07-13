"""Tests for PaddleOCR-VL official async API client."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from lightrag.parser.external.paddleocr_vl.client import PaddleOCRVLRawClient
from lightrag.parser.external.paddleocr_vl.client import _indexed_image_path
from lightrag.parser.external.paddleocr_vl.client import _safe_name

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
    "returnMarkdownImages": True,
    "restructurePages": True,
    "mergeTables": True,
    "relevelTitles": True,
    "visualize": False,
}


@pytest.mark.parametrize(
    ("name", "value", "expected"),
    [
        (
            "imgs/figure.jpg",
            "https://example.test/figure.png",
            "imgs/figure_2.png",
        ),
        ("imgs/figure.webp", "base64-payload", "imgs/figure_2.webp"),
        ("layout_det_res", "base64-payload", "layout_det_res_2.jpg"),
    ],
)
def test_indexed_image_path_uses_url_then_name_then_jpg_suffix(
    name: str, value: str, expected: str
) -> None:
    assert _indexed_image_path(name, value, 2).as_posix() == expected


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
_RESULT_PAYLOAD: dict[str, Any] | None = None


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
            result_payload = _RESULT_PAYLOAD or {
                "result": {
                    "layoutParsingResults": [
                        {
                            "prunedResult": {"parsing_res_list": []},
                            "markdown": {
                                "text": "# hello",
                                "images": {
                                    "imgs/fig.jpg": (
                                        "https://pplines-online.bj.bcebos.com/"
                                        "deploy/official/paddleocr/"
                                        "pp-ocr-vl-16-online/default/"
                                        "markdown_0/imgs/fig.jpg"
                                    )
                                },
                            },
                            "outputImages": {
                                "layout_det_res": (
                                    "https://pplines-online.bj.bcebos.com/"
                                    "deploy/official/paddleocr/"
                                    "pp-ocr-vl-16-online/default/"
                                    "layout.jpg"
                                )
                            },
                        }
                    ]
                }
            }
            return _Response(text=json.dumps(result_payload))
        if url in {
            (
                "https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/"
                "pp-ocr-vl-16-online/default/markdown_0/imgs/fig.jpg"
            ),
            (
                "https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/"
                "pp-ocr-vl-16-online/default/layout.jpg"
            ),
        }:
            return _Response(content=b"\xff\xd8fake")
        if url == (
            "https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/"
            "pp-ocr-vl-16-online/f74decfd-07d8-4cf9-aa27-4c4ddeaf309e/"
            "markdown_0/imgs/img_in_image_box_93_222_125_263.jpg?"
            "authorization=bce-auth-v1"
        ):
            return _Response(content=b"bos-fig")
        if url == (
            "https://pplines-online.gz.bcebos.com/deploy/official/paddleocr/"
            "pp-ocr-vl-16-online/f74decfd-07d8-4cf9-aa27-4c4ddeaf309e/"
            "markdown_0/imgs/img_in_image_box_93_222_125_263.jpg?"
            "authorization=bce-auth-v1"
        ):
            return _Response(content=b"gz-bos-fig")
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


_PADDLEOCR_VL_ENV_VARS = (
    "PADDLEOCR_VL_API_MODE",
    "PADDLEOCR_VL_API_TOKEN",
    "PADDLEOCR_VL_ENDPOINT",
    "PADDLEOCR_VL_OFFICIAL_ENDPOINT",
    "PADDLEOCR_VL_LOCAL_ENDPOINT",
    "PADDLEOCR_VL_MODEL",
    "PADDLEOCR_VL_PAGE_RANGES",
    "PADDLEOCR_VL_BATCH_ID",
    "PADDLEOCR_VL_USE_DOC_ORIENTATION_CLASSIFY",
    "PADDLEOCR_VL_USE_DOC_UNWARPING",
    "PADDLEOCR_VL_USE_LAYOUT_DETECTION",
    "PADDLEOCR_VL_USE_CHART_RECOGNITION",
    "PADDLEOCR_VL_USE_SEAL_RECOGNITION",
    "PADDLEOCR_VL_USE_OCR_FOR_IMAGE_BLOCK",
    "PADDLEOCR_VL_LAYOUT_THRESHOLD",
    "PADDLEOCR_VL_LAYOUT_NMS",
    "PADDLEOCR_VL_LAYOUT_UNCLIP_RATIO",
    "PADDLEOCR_VL_LAYOUT_MERGE_BBOXES_MODE",
    "PADDLEOCR_VL_LAYOUT_SHAPE_MODE",
    "PADDLEOCR_VL_PROMPT_LABEL",
    "PADDLEOCR_VL_FORMAT_BLOCK_CONTENT",
    "PADDLEOCR_VL_REPETITION_PENALTY",
    "PADDLEOCR_VL_TEMPERATURE",
    "PADDLEOCR_VL_TOP_P",
    "PADDLEOCR_VL_MIN_PIXELS",
    "PADDLEOCR_VL_MAX_PIXELS",
    "PADDLEOCR_VL_MAX_NEW_TOKENS",
    "PADDLEOCR_VL_MERGE_LAYOUT_BLOCKS",
    "PADDLEOCR_VL_MARKDOWN_IGNORE_LABELS",
    "PADDLEOCR_VL_VLM_EXTRA_ARGS",
    "PADDLEOCR_VL_PRETTIFY_MARKDOWN",
    "PADDLEOCR_VL_SHOW_FORMULA_NUMBER",
    "PADDLEOCR_VL_RETURN_MARKDOWN_IMAGES",
    "PADDLEOCR_VL_RESTRUCTURE_PAGES",
    "PADDLEOCR_VL_MERGE_TABLES",
    "PADDLEOCR_VL_RELEVEL_TITLES",
    "PADDLEOCR_VL_VISUALIZE",
    "PADDLEOCR_VL_ENGINE_VERSION",
    "PADDLEOCR_VL_POLL_INTERVAL_SECONDS",
    "PADDLEOCR_VL_MAX_POLLS",
)


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    global _RESULT_PAYLOAD
    _RESULT_PAYLOAD = None
    for name in _PADDLEOCR_VL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PADDLEOCR_VL_API_MODE", "official")
    monkeypatch.setenv("PADDLEOCR_VL_API_TOKEN", "token-1")
    monkeypatch.setenv("PADDLEOCR_VL_ENDPOINT", "http://paddle.test/api/v2/ocr/jobs")


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
    assert (raw_dir / "imgs" / "fig_0.jpg").read_bytes() == b"\xff\xd8fake"
    assert (
        raw_dir / "outputImages" / "layout_det_res_0.jpg"
    ).read_bytes() == b"\xff\xd8fake"
    assert (raw_dir / "_manifest.json").is_file()


async def test_mandatory_images_with_same_bbox_are_scoped_by_page_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    global _RESULT_PAYLOAD
    source = tmp_path / "demo.pdf"
    source.write_bytes(b"%PDF fake")
    _CURRENT["recorder"] = _Recorder()
    _install_httpx(monkeypatch)
    raw_dir = tmp_path / "demo.paddleocr_vl_raw"
    image_path = "imgs/img_in_image_box_10_20_30_40.jpg"
    _RESULT_PAYLOAD = {
        "result": {
            "layoutParsingResults": [
                {
                    "markdown": {
                        "images": {
                            image_path: base64.b64encode(payload).decode("ascii")
                        }
                    }
                }
                for payload in (b"page-zero", b"page-one")
            ]
        }
    }

    await PaddleOCRVLRawClient().download_into(raw_dir, source)

    pages = json.loads((raw_dir / "content_list.json").read_text(encoding="utf-8"))
    first_ref = "imgs/img_in_image_box_10_20_30_40_0.jpg"
    second_ref = "imgs/img_in_image_box_10_20_30_40_1.jpg"
    assert (raw_dir / first_ref).read_bytes() == b"page-zero"
    assert (raw_dir / second_ref).read_bytes() == b"page-one"
    assert not (raw_dir / image_path).exists()
    assert list(pages[0]["markdown"]["images"]) == [first_ref]
    assert list(pages[1]["markdown"]["images"]) == [second_ref]


async def test_client_downloads_only_bos_remote_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    global _RESULT_PAYLOAD
    source = tmp_path / "demo.pdf"
    source.write_bytes(b"%PDF fake")
    recorder = _Recorder()
    _CURRENT["recorder"] = recorder
    _RESULT_PAYLOAD = {
        "result": {
            "layoutParsingResults": [
                {
                    "prunedResult": {"parsing_res_list": []},
                    "markdown": {
                        "text": "# hello",
                        "images": {
                            "imgs/metadata.jpg": (
                                "https://pplines-online.bj.bcebos.com/deploy/"
                                "official/paddleocr/pp-ocr-vl-16-online/"
                                "f74decfd-07d8-4cf9-aa27-4c4ddeaf309e/"
                                "markdown_0/imgs/"
                                "img_in_image_box_93_222_125_263.jpg?"
                                "authorization=bce-auth-v1"
                            )
                        },
                    },
                    "outputImages": {"layout_det_res": "http://evil.test/layout.jpg"},
                }
            ]
        }
    }
    _install_httpx(monkeypatch)

    raw_dir = tmp_path / "demo.paddleocr_vl_raw"
    await PaddleOCRVLRawClient().download_into(
        raw_dir, source, upload_name="canonical.pdf"
    )

    urls = [call["url"] for call in recorder.get_calls]
    assert (
        "https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/"
        "pp-ocr-vl-16-online/f74decfd-07d8-4cf9-aa27-4c4ddeaf309e/"
        "markdown_0/imgs/img_in_image_box_93_222_125_263.jpg?"
        "authorization=bce-auth-v1"
    ) in urls
    assert "http://evil.test/layout.jpg" not in urls
    assert (raw_dir / "imgs" / "metadata_0.jpg").read_bytes() == b"bos-fig"
    assert not (raw_dir / "outputImages" / "layout_det_res_0.jpg").exists()


async def test_mandatory_markdown_image_with_disallowed_host_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    global _RESULT_PAYLOAD
    source = tmp_path / "demo.pdf"
    source.write_bytes(b"%PDF fake")
    recorder = _Recorder()
    _CURRENT["recorder"] = recorder
    # markdown.images URL on a non-allowed host -> mandatory -> must raise.
    _RESULT_PAYLOAD = {
        "result": {
            "layoutParsingResults": [
                {
                    "prunedResult": {"parsing_res_list": []},
                    "markdown": {
                        "text": "# hello",
                        "images": {"imgs/fig.jpg": "http://evil.test/fig.jpg"},
                    },
                }
            ]
        }
    }
    _install_httpx(monkeypatch)

    with pytest.raises(RuntimeError, match="not an allowed asset host"):
        await PaddleOCRVLRawClient().download_into(
            tmp_path / "demo.paddleocr_vl_raw", source
        )


async def test_mandatory_markdown_image_with_undecodable_payload_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    global _RESULT_PAYLOAD
    source = tmp_path / "demo.pdf"
    source.write_bytes(b"%PDF fake")
    _CURRENT["recorder"] = _Recorder()
    _install_httpx(monkeypatch)
    # markdown.images value that is neither a URL nor valid Base64 -> mandatory
    # -> must raise.
    _RESULT_PAYLOAD = {
        "result": {
            "layoutParsingResults": [
                {
                    "prunedResult": {"parsing_res_list": []},
                    "markdown": {
                        "text": "# hello",
                        "images": {"imgs/fig.jpg": "!!!not-base64!!!"},
                    },
                }
            ]
        }
    }

    with pytest.raises(RuntimeError, match="could not be decoded"):
        await PaddleOCRVLRawClient().download_into(
            tmp_path / "demo.paddleocr_vl_raw", source
        )


async def test_allowed_asset_hosts_env_admits_custom_domain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    global _RESULT_PAYLOAD
    source = tmp_path / "demo.pdf"
    source.write_bytes(b"%PDF fake")
    monkeypatch.setenv("PADDLEOCR_VL_ALLOWED_ASSET_HOSTS", "*.my-cdn.example.com")

    class _CdnFakeAsyncClient(_FakeAsyncClient):
        async def get(self, url: str, **kwargs: Any) -> _Response:
            if url == "https://assets.my-cdn.example.com/fig.jpg":
                return _Response(content=b"cdn-fig")
            return await super().get(url, **kwargs)

    monkeypatch.setattr(
        "lightrag.parser.external.paddleocr_vl.client.httpx.AsyncClient",
        _CdnFakeAsyncClient,
    )
    monkeypatch.setattr(
        "lightrag.parser.external.paddleocr_vl.client.httpx.Timeout",
        lambda *_, **__: None,
    )
    _CURRENT["recorder"] = _Recorder()
    _RESULT_PAYLOAD = {
        "result": {
            "layoutParsingResults": [
                {
                    "prunedResult": {"parsing_res_list": []},
                    "markdown": {
                        "text": "# hello",
                        "images": {
                            "imgs/fig.jpg": "https://assets.my-cdn.example.com/fig.jpg"
                        },
                    },
                }
            ]
        }
    }

    client = PaddleOCRVLRawClient()
    assert "*.my-cdn.example.com" in client.allowed_asset_host_patterns
    raw_dir = tmp_path / "demo.paddleocr_vl_raw"
    await client.download_into(raw_dir, source)

    assert (raw_dir / "imgs" / "fig_0.jpg").read_bytes() == b"cdn-fig"


@pytest.mark.parametrize(
    ("configured_host", "asset_url", "expected"),
    [
        ("example.com", "https://example.com/fig.jpg", True),
        ("example.com", "https://assets.example.com/fig.jpg", False),
        ("*.example.com", "https://example.com/fig.jpg", False),
        ("*.example.com", "https://assets.example.com/fig.jpg", True),
        ("*.example.com", "https://nested.assets.example.com/fig.jpg", True),
        ("*.example.com", "https://notexample.com/fig.jpg", False),
    ],
)
def test_allowed_asset_hosts_distinguish_exact_and_wildcard_patterns(
    configured_host: str,
    asset_url: str,
    expected: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PADDLEOCR_VL_ALLOWED_ASSET_HOSTS", configured_host)
    client = PaddleOCRVLRawClient()

    assert client._is_allowed_asset_url(asset_url) is expected


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
    monkeypatch.setenv("PADDLEOCR_VL_MAX_NEW_TOKENS", "8192")
    monkeypatch.setenv("PADDLEOCR_VL_SHOW_FORMULA_NUMBER", "true")
    monkeypatch.setenv("PADDLEOCR_VL_RETURN_MARKDOWN_IMAGES", "false")
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
        "maxNewTokens": 8192,
        "showFormulaNumber": True,
        "returnMarkdownImages": False,
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
                    "imgs/fig_0.jpg": base64.b64encode(b"local-fig").decode("ascii")
                },
            },
            "outputImages": {
                "layout_det_res": base64.b64encode(b"local-layout").decode("ascii")
            },
        }
    ]
    assert (raw_dir / "imgs" / "fig_0.jpg").read_bytes() == b"local-fig"
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


# ---------------------------------------------------------------------------
# _safe_name (mirrors api/routers/document_routes.py::sanitize_filename)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("normal.jpg", "normal.jpg"),
        ("with/slash.jpg", "withslash.jpg"),
        ("with\\backslash.jpg", "withbackslash.jpg"),
        ("..traversal", "traversal"),
        ("foo..bar", "foobar"),
        ("name\x00null", "namenull"),
        ("name\x7fdel", "namedel"),
        (".leading", "leading"),
        ("trailing.", "trailing"),
        ("  spaced  ", "spaced"),
        ("", "asset"),
        ("...", "asset"),
        ("..", "asset"),
        ("/", "asset"),
        ("\\", "asset"),
    ],
)
def test_safe_name_strips_path_traversal_and_control(raw: str, expected: str) -> None:
    assert _safe_name(raw) == expected


def test_safe_name_preserves_normal_filename_with_dots() -> None:
    # Dots inside a name (not leading/trailing, not "..") are preserved.
    assert _safe_name("report.v2.pdf") == "report.v2.pdf"


# ---------------------------------------------------------------------------
# Error-path coverage (httpx RequestError wrapping, HTTP non-2xx, failed state)
# ---------------------------------------------------------------------------
class _RequestErrorFakeAsyncClient:
    """Fake that raises an httpx.RequestError on POST (upload)."""

    def __init__(self, *_: Any, **__: Any) -> None:
        pass

    async def __aenter__(self) -> "_RequestErrorFakeAsyncClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        pass

    async def post(self, url: str, **kwargs: Any) -> _Response:
        raise httpx.ConnectError("connection refused")

    async def get(self, url: str, **kwargs: Any) -> _Response:
        raise AssertionError(f"unexpected GET {url}")


async def test_client_wraps_httpx_request_error_with_endpoint_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "demo.pdf"
    source.write_bytes(b"%PDF fail")
    monkeypatch.setattr(
        "lightrag.parser.external.paddleocr_vl.client.httpx.AsyncClient",
        _RequestErrorFakeAsyncClient,
    )
    monkeypatch.setattr(
        "lightrag.parser.external.paddleocr_vl.client.httpx.Timeout",
        lambda *_, **__: None,
    )

    with pytest.raises(RuntimeError, match="PaddleOCR-VL backend request failed"):
        await PaddleOCRVLRawClient().download_into(
            tmp_path / "demo.paddleocr_vl_raw", source
        )


class _HttpErrorFakeAsyncClient:
    """Fake whose upload returns a non-2xx status."""

    def __init__(self, *_: Any, **__: Any) -> None:
        pass

    async def __aenter__(self) -> "_HttpErrorFakeAsyncClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        pass

    async def post(self, url: str, **kwargs: Any) -> _Response:
        return _Response(status_code=503, text="service unavailable")

    async def get(self, url: str, **kwargs: Any) -> _Response:
        raise AssertionError(f"unexpected GET {url}")


async def test_upload_http_error_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "demo.pdf"
    source.write_bytes(b"%PDF fail")
    monkeypatch.setattr(
        "lightrag.parser.external.paddleocr_vl.client.httpx.AsyncClient",
        _HttpErrorFakeAsyncClient,
    )
    monkeypatch.setattr(
        "lightrag.parser.external.paddleocr_vl.client.httpx.Timeout",
        lambda *_, **__: None,
    )

    with pytest.raises(RuntimeError, match="HTTP 503"):
        await PaddleOCRVLRawClient().download_into(
            tmp_path / "demo.paddleocr_vl_raw", source
        )


class _FailedStateFakeAsyncClient:
    """Fake that reports a 'failed' job state during polling."""

    def __init__(self, *_: Any, **__: Any) -> None:
        pass

    async def __aenter__(self) -> "_FailedStateFakeAsyncClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        pass

    async def post(self, url: str, **kwargs: Any) -> _Response:
        return _Response(payload={"data": {"jobId": "job-fail"}})

    async def get(self, url: str, **kwargs: Any) -> _Response:
        if url.endswith("/job-fail"):
            return _Response(
                payload={
                    "data": {
                        "state": "failed",
                        "errorMsg": "page count exceeded limit",
                    }
                }
            )
        raise AssertionError(f"unexpected GET {url}")


async def test_failed_poll_state_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "demo.pdf"
    source.write_bytes(b"%PDF fail")
    monkeypatch.setattr(
        "lightrag.parser.external.paddleocr_vl.client.httpx.AsyncClient",
        _FailedStateFakeAsyncClient,
    )
    monkeypatch.setattr(
        "lightrag.parser.external.paddleocr_vl.client.httpx.Timeout",
        lambda *_, **__: None,
    )
    # Avoid sleeping during the poll loop.
    monkeypatch.setenv("PADDLEOCR_VL_POLL_INTERVAL_SECONDS", "0")

    with pytest.raises(RuntimeError, match="job job-fail failed"):
        await PaddleOCRVLRawClient().download_into(
            tmp_path / "demo.paddleocr_vl_raw", source
        )


# ---------------------------------------------------------------------------
# Concurrent image download (mandatory error surfaces from gather)
# ---------------------------------------------------------------------------
async def test_concurrent_mandatory_image_error_surfaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A mandatory markdown image on a disallowed host must raise even though
    # image materialization now runs concurrently via asyncio.gather.
    global _RESULT_PAYLOAD
    source = tmp_path / "demo.pdf"
    source.write_bytes(b"%PDF bad-host")
    recorder = _Recorder()
    _CURRENT["recorder"] = recorder
    _install_httpx(monkeypatch)
    _RESULT_PAYLOAD = {
        "result": {
            "layoutParsingResults": [
                {
                    "prunedResult": {"parsing_res_list": []},
                    "markdown": {
                        "text": "# bad",
                        "images": {
                            "imgs/evil.jpg": "https://evil.test/not-allowed.jpg"
                        },
                    },
                }
            ]
        }
    }

    with pytest.raises(RuntimeError, match="not an allowed asset host"):
        await PaddleOCRVLRawClient().download_into(
            tmp_path / "demo.paddleocr_vl_raw", source
        )


# ---------------------------------------------------------------------------
# HTTPS-only gate for asset URLs
# ---------------------------------------------------------------------------
def test_asset_url_must_be_https(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PADDLEOCR_VL_ALLOWED_ASSET_HOSTS", "evil.test")
    client = PaddleOCRVLRawClient()
    # Same host, but http:// is rejected regardless of host allowlist.
    assert client._is_allowed_asset_url("http://evil.test/fig.jpg") is False
    assert client._is_allowed_asset_url("https://evil.test/fig.jpg") is True
