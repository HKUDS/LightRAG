"""The docx Gemini client must agree with google-genai about the mode.

The SDK enables Vertex AI for "true" or "1" in any case. When LightRAG reads
"1" as off, it takes the AI Studio branch and demands GOOGLE_API_KEY, while the
client it would build switches to Vertex AI on its own anyway. The AI Studio
client is therefore built with an explicit ``vertexai=False``.
"""

from types import SimpleNamespace

import pytest

from lightrag.parser.docx import utils
from lightrag.parser.docx.utils import create_gemini_client, is_vertex_ai_mode

pytestmark = pytest.mark.offline


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("True", True),
        ("1", True),
        ("0", False),
        (None, False),
    ],
)
def test_is_vertex_ai_mode_matches_sdk_parsing(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    else:
        monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", value)

    assert is_vertex_ai_mode() is expected


def test_ai_studio_client_pins_vertexai_off(monkeypatch):
    """The SDK must not switch an AI Studio client to Vertex AI from its env."""
    monkeypatch.setattr(
        utils, "genai", SimpleNamespace(Client=lambda **kw: SimpleNamespace(kw=kw))
    )
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

    client = create_gemini_client()

    assert client.kw == {"vertexai": False, "api_key": "test-key"}
