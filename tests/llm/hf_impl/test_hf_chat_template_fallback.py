"""Prompt fallback must preserve all turns without changing caller history."""

from copy import deepcopy

import pytest

from .test_hf_model_if_cache_device_placement import FakeTokenizer, make_hf_module

pytestmark = [pytest.mark.offline, pytest.mark.asyncio]


class PromptTokenizer(FakeTokenizer):
    def __init__(self, mode: str):
        super().__init__([11, 12, 13], host_has_cuda=False)
        self.mode = mode
        self.rendered_messages = []
        self.input_prompt = None

    def apply_chat_template(self, messages, **kwargs):
        self.rendered_messages.append(deepcopy(messages))
        if self.mode == "missing":
            raise ValueError("tokenizer.chat_template is not set")
        if self.mode == "no_system" and messages[0]["role"] == "system":
            raise ValueError("System role not supported")
        return " | ".join(message["content"] for message in messages)

    def __call__(self, text, **kwargs):
        self.input_prompt = text
        return super().__call__(text, **kwargs)


@pytest.mark.parametrize("mode", ["native", "no_system", "missing"])
@pytest.mark.parametrize("system_prompt", [None, "Be concise"])
@pytest.mark.parametrize("with_history", [False, True])
async def test_chat_template_fallback_preserves_prompt_and_history(
    monkeypatch, mode, system_prompt, with_history
):
    hf, model, _ = make_hf_module(monkeypatch, model_device="cpu", host_has_cuda=False)
    tokenizer = PromptTokenizer(mode)
    monkeypatch.setattr(hf, "initialize_hf_model", lambda _: (model, tokenizer))
    history = (
        [
            {"role": "user", "content": "Earlier question"},
            {"role": "assistant", "content": "Earlier answer"},
        ]
        if with_history
        else []
    )
    original_history = deepcopy(history)

    result = await hf.hf_model_if_cache(
        "fake-model", "Current question", system_prompt, history
    )

    assert result == "decoded:[901, 902]"
    assert history == original_history
    if mode == "missing":
        expected = "<system>Be concise</system>\n" if system_prompt else ""
        if with_history:
            expected += (
                "<user>Earlier question</user>\n<assistant>Earlier answer</assistant>\n"
            )
        expected += "<user>Current question</user>\n"
    else:
        contents = [message["content"] for message in original_history]
        contents.append("Current question")
        if system_prompt:
            if mode == "native":
                contents.insert(0, "Be concise")
            else:
                contents[0] = "<system>Be concise</system>\n" + contents[0]
        expected = " | ".join(contents)
    assert tokenizer.input_prompt == expected
    assert len(tokenizer.rendered_messages) == (
        2 if system_prompt and mode != "native" else 1
    )
