from lightrag.parser.video.asr import VideoAsrClient, _response_text


class _Response:
    def __init__(self, status_code, payload, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload


class _Client:
    responses = []
    calls = []

    def __init__(self, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.pop(0)


def test_asr_uses_openai_compatible_chat_endpoint(monkeypatch):
    _Client.responses = [_Response(200, {"choices": [{"message": {"content": "hello"}}]})]
    _Client.calls = []
    monkeypatch.setattr("lightrag.parser.video.asr.httpx.Client", _Client)

    result = VideoAsrClient("http://asr/v1", "secret", "qwen3-asr-1.7b").transcribe(b"wav")

    assert result == "hello"
    url, kwargs = _Client.calls[0]
    assert url == "http://asr/v1/chat/completions"
    assert kwargs["headers"]["Authorization"] == "Bearer secret"
    assert kwargs["json"]["messages"][0]["content"][1]["type"] == "input_audio"


def test_asr_falls_back_to_audio_url_for_validation_error(monkeypatch):
    _Client.responses = [
        _Response(400, {}, "input_audio unsupported"),
        _Response(200, {"choices": [{"message": {"content": [{"text": "ok"}]}}]}),
    ]
    _Client.calls = []
    monkeypatch.setattr("lightrag.parser.video.asr.httpx.Client", _Client)

    assert VideoAsrClient("http://asr/v1", "", "model").transcribe(b"wav") == "ok"
    assert _Client.calls[1][1]["json"]["messages"][0]["content"][1]["type"] == "audio_url"


def test_response_text_rejects_missing_choices():
    try:
        _response_text({})
    except Exception as exc:
        assert "choices" in str(exc)
    else:
        raise AssertionError("missing choices should fail")
