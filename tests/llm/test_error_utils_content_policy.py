"""Classifier cases for provider content-moderation verdicts.

Wire shapes follow the official Aliyun DashScope / Azure OpenAI error docs and
Alibaba's own SDK tests: the openai SDK sets ``BadRequestError.body`` to the
INNER error dict (``body.get("error", body)``), ``error.type`` is not reliable
(DashScope models it as ``invalid_request_error``), ``param`` may be null or
the string "None", and messages drift (URL suffixes, ``<400>`` prefixes), so
the CODE is what a verdict is matched on.
"""

import httpx
import pytest
from openai import BadRequestError

from lightrag.llm._error_utils import is_content_policy_error

pytestmark = pytest.mark.offline


# DashScope compatible-mode moderation verdict (HTTP 400). The message carries
# no slug, so only the body-code probe can answer for these cases.
DASHSCOPE_INPUT_BODY = {
    "code": "data_inspection_failed",
    "type": "data_inspection_failed",
    "param": None,
    "message": (
        "Input data may contain inappropriate content. For details, see: "
        "https://help.aliyun.com/zh/model-studio/error-code#inappropriate-content"
    ),
}

AZURE_CONTENT_FILTER_BODY = {
    "code": "content_filter",
    "innererror": {"code": "ResponsibleAIPolicyViolation"},
    "message": (
        "The response was filtered due to the prompt triggering Azure OpenAI's "
        "content management policy. Please modify your prompt and retry."
    ),
}


def _make_error(body: object, message: str | None = None) -> BadRequestError:
    request = httpx.Request(
        "POST", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    )
    response = httpx.Response(status_code=400, request=request)
    return BadRequestError(
        message or f"Error code: 400 - {body}", response=response, body=body
    )


@pytest.mark.parametrize(
    "body, message",
    [
        pytest.param(DASHSCOPE_INPUT_BODY, None, id="dashscope-compatible-code"),
        # Alibaba's SDK test models type="invalid_request_error" with the
        # verdict in code, and param as the STRING "None": matching type or
        # requiring param=None would miss this real shape.
        pytest.param(
            {
                "code": "data_inspection_failed",
                "type": "invalid_request_error",
                "param": "None",
                "message": "Output data may contain inappropriate content.",
            },
            None,
            id="dashscope-type-invalid-request-param-none-string",
        ),
        pytest.param(
            {
                "code": "DataInspectionFailed",
                "message": "Input or output data may contain inappropriate content.",
            },
            None,
            id="dashscope-native-camelcase-code",
        ),
        pytest.param(
            {
                "code": "InternalError.Algo.DataInspectionFailed",
                "message": "Input text data may contain inappropriate content.",
            },
            None,
            id="algo-prefixed-code",
        ),
        # Azure: the code is the verdict; the message prose ("content
        # management policy") must not be what matches.
        pytest.param(AZURE_CONTENT_FILTER_BODY, None, id="azure-content-filter-code"),
        pytest.param({"error": DASHSCOPE_INPUT_BODY}, None, id="nested-envelope"),
        # A proxy that flattens the body: no usable code, the verdict slug
        # survives only in the message (observed "<400>" prefix included).
        pytest.param(
            {
                "message": (
                    "<400> InternalError.Algo.DataInspectionFailed: "
                    "Input data may contain inappropriate content."
                )
            },
            None,
            id="body-message-slug-fallback",
        ),
        # No body at all: the rendered message is the last resort.
        pytest.param(
            None,
            "Error code: 400 - {'code': 'data_inspection_failed', "
            "'message': 'Input data may contain inappropriate content.'}",
            id="rendered-message-fallback",
        ),
    ],
)
def test_policy_verdict_is_detected(body, message):
    assert is_content_policy_error(_make_error(body, message)) is True


@pytest.mark.parametrize(
    "body, message",
    [
        # Moderation service UNAVAILABLE (HTTP 500 in the wild): a transient
        # outage that must stay retryable, not a verdict on the content.
        pytest.param(
            {
                "code": "InternalError.DataInspection",
                "message": "Data inspection service unavailable.",
            },
            None,
            id="internal-error-data-inspection-stays-retryable",
        ),
        # Media download/decode failure, not a content verdict.
        pytest.param(
            {
                "code": "InvalidParameter.DataInspection",
                "message": "Failed to download or decode the media file.",
            },
            None,
            id="invalid-parameter-data-inspection",
        ),
        pytest.param(
            {
                "code": None,
                "type": "invalid_request_error",
                "param": "temperature",
                "message": "Invalid value for 'temperature': must be <= 2.",
            },
            None,
            id="generic-parameter-400",
        ),
        pytest.param({}, None, id="empty-body"),
        pytest.param("plain text error page", None, id="non-dict-body"),
        # "content_filter" appears in Azure annotation payloads on choice-less
        # chunks; the rendered-message fallback must match only the DashScope
        # slugs, never that word.
        pytest.param(
            None,
            "Received chunk without choices (likely Azure content filter): "
            "prompt_filter_results={'violence': {'filtered': False}}",
            id="rendered-content-filter-word-is-not-a-verdict",
        ),
    ],
)
def test_non_verdict_is_not_detected(body, message):
    assert is_content_policy_error(_make_error(body, message)) is False
