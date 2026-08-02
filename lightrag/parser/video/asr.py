"""OpenAI-compatible Qwen ASR client used by the local video parser."""

from __future__ import annotations

import base64
from typing import Any

import httpx


class AsrError(RuntimeError):
    """Raised when the configured ASR service cannot transcribe audio."""


class VideoAsrClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        timeout_seconds: float = 180.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds

    def transcribe(self, wav_bytes: bytes, *, language: str = "") -> str:
        encoded = base64.b64encode(wav_bytes).decode("ascii")
        instruction = "Transcribe the audio accurately. Preserve names, numbers, and technical terms."
        if language:
            instruction += f" The expected spoken language is {language}."
        content = [
            {"type": "text", "text": instruction},
            {
                "type": "input_audio",
                "input_audio": {"data": encoded, "format": "wav"},
            },
        ]
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": 2048,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        url = f"{self.base_url}/chat/completions"
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(url, headers=headers, json=payload)
            # A few Qwen ASR builds expose the equivalent OpenAI audio_url
            # content shape instead of input_audio. Retry only for a client
            # validation error; server errors must remain visible.
            if response.status_code == 400:
                fallback = dict(payload)
                fallback["messages"] = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": instruction},
                            {
                                "type": "audio_url",
                                "audio_url": {"url": f"data:audio/wav;base64,{encoded}"},
                            },
                        ],
                    }
                ]
                response = client.post(url, headers=headers, json=fallback)
            if response.status_code >= 400:
                detail = response.text.strip().replace("\n", " ")
                raise AsrError(f"ASR request failed ({response.status_code}): {detail[:800]}")
            try:
                data = response.json()
            except ValueError as exc:
                raise AsrError("ASR returned invalid JSON") from exc
        text = _response_text(data)
        if not text:
            raise AsrError("ASR returned an empty transcript")
        return text


def _response_text(data: Any) -> str:
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise AsrError("ASR response has no choices[0].message.content") from exc
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
        return "\n".join(parts).strip()
    return str(content).strip()
