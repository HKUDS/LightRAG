"""``is_vertex_ai_mode`` must read GOOGLE_GENAI_USE_VERTEXAI like google-genai.

The SDK enables Vertex AI for "true" or "1" in any case. When LightRAG reads
"1" as off, it takes the AI Studio branch and demands GOOGLE_API_KEY, while the
client it would build switches to Vertex AI on its own anyway.
"""

import pytest

from lightrag.parser.docx.utils import is_vertex_ai_mode

pytestmark = pytest.mark.offline


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("true", True),
        ("True", True),
        ("1", True),
        ("false", False),
        ("0", False),
        ("", False),
    ],
)
def test_is_vertex_ai_mode_matches_sdk_parsing(monkeypatch, value, expected):
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", value)

    assert is_vertex_ai_mode() is expected


def test_is_vertex_ai_mode_off_when_unset(monkeypatch):
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)

    assert is_vertex_ai_mode() is False
