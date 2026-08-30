"""OpenAI Responses API (``/v1/responses``) LLM binding.

This is a sibling of the ``openai`` binding, not a replacement for it. The
``openai`` binding doubles as the generic OpenAI-*compatible* binding, and much
of that ecosystem (vLLM, assorted gateways and third-party providers) does not
implement ``/v1/responses``. Changing that binding's transport would regress
those deployments, so the Responses transport lives here instead and is opted
into explicitly with ``LLM_BINDING=openai_responses``.

The public contract matches ``openai_complete_if_cache``: same arguments, a
``str`` return for non-streaming calls and an ``AsyncIterator[str]`` for
streaming ones, ``<think>`` tags around reasoning content, ``TruncatedResponse``
for token-limit truncation, and ``token_tracker`` accounting. The *implementation*
of those four behaviours cannot be shared with the Chat Completions driver
because the wire shapes differ:

===================  ==============================  ================================
Concern              Chat Completions                Responses
===================  ==============================  ================================
System prompt        ``messages[0]`` role=system     ``instructions`` parameter
Output budget        ``max_completion_tokens``       ``max_output_tokens``
Reasoning control    ``reasoning_effort`` (flat)     ``reasoning: {"effort": ...}``
Structured output    ``response_format``             ``text: {"format": ...}``
Body text            ``choices[].message.content``   ``output[].content[].text``
Reasoning text       ``message.reasoning_content``   ``output[]`` items of type
                                                     ``reasoning``
Truncation signal    ``finish_reason == "length"``   ``status == "incomplete"`` with
                                                     ``incomplete_details.reason``
Refusal              ``message.refusal``             ``refusal`` content parts /
                                                     ``response.refusal.*`` events
Usage keys           ``prompt_tokens`` /             ``input_tokens`` /
                     ``completion_tokens``           ``output_tokens``
Streaming unit       ``choices[].delta``             typed events
                                                     (``response.output_text.delta``)
===================  ==============================  ================================

Client construction *is* shared: ``create_openai_async_client`` is imported from
the Chat Completions module so the two bindings cannot drift on base-URL
resolution, default headers or the DashScope workspace header.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any, Union

import pipmaster as pm

if not pm.is_installed("openai"):
    pm.install("openai")

from openai import (
    APIConnectionError,
    APITimeoutError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from lightrag.exceptions import EmptyTruncatedResponseError
from lightrag.utils import (
    VERBOSE_DEBUG,
    TruncatedResponse,
    empty_length_truncated_hint,
    logger,
    safe_unicode_decode,
    verbose_debug,
)

from .openai import (
    InvalidResponseError,
    TransientBadRequestError,
    create_openai_async_client,
)

__all__ = [
    "openai_responses_complete_if_cache",
    "openai_responses_complete",
]

# Chat Completions spellings accepted for convenience, mapped onto their
# Responses equivalents. Callers configured for the `openai` binding therefore
# do not silently lose their output budget when switching binding.
_MAX_TOKEN_ALIASES = ("max_output_tokens", "max_completion_tokens", "max_tokens")

# Chat Completions parameters with no Responses equivalent. Forwarding them
# would make the API reject the whole request, so they are dropped with a
# warning rather than passed through.
_UNSUPPORTED_KWARGS = (
    "frequency_penalty",
    "presence_penalty",
    "logit_bias",
    "n",
    "stop",
)


def _normalize_text_format(response_format: dict[str, Any]) -> dict[str, Any]:
    """Reshape a Chat Completions ``response_format`` into a Responses ``text.format``.

    ``{"type": "json_object"}`` is spelled identically in both APIs. A json
    schema is not: Chat Completions nests ``name`` / ``schema`` / ``strict``
    under a ``json_schema`` key, while Responses expects them directly on the
    format object. Forwarding the nested form unchanged is rejected with a 400,
    so every schema-constrained request would fail on this binding.
    """
    if response_format.get("type") != "json_schema":
        return response_format

    json_schema = response_format.get("json_schema")
    if not isinstance(json_schema, dict):
        # Already flat -- i.e. the caller wrote the Responses spelling -- or a
        # shape this driver does not recognise. Forward it rather than guess.
        return response_format

    flattened = {
        key: value for key, value in response_format.items() if key != "json_schema"
    }
    flattened.update(json_schema)
    return flattened


def _translate_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Translate Chat Completions kwargs into Responses parameters.

    Only the spellings that genuinely differ are rewritten; anything else is
    forwarded untouched so provider-specific extras keep working.
    """
    translated = dict(kwargs)

    # Output budget: first alias present wins, in declared precedence order.
    max_tokens_value = None
    for alias in _MAX_TOKEN_ALIASES:
        if translated.get(alias) is not None and max_tokens_value is None:
            max_tokens_value = translated[alias]
        translated.pop(alias, None)
    if max_tokens_value is not None:
        translated["max_output_tokens"] = max_tokens_value

    # Reasoning: flat `reasoning_effort` becomes nested `reasoning.effort`.
    # An explicit `reasoning` dict always wins; the flat alias only fills a gap.
    reasoning_effort = translated.pop("reasoning_effort", None)
    if reasoning_effort:
        reasoning = translated.get("reasoning")
        if isinstance(reasoning, dict):
            reasoning.setdefault("effort", reasoning_effort)
        elif reasoning is None:
            translated["reasoning"] = {"effort": reasoning_effort}

    # Structured output: `response_format` becomes `text.format`, after the
    # json_schema payload is reshaped for this API.
    response_format = translated.pop("response_format", None)
    if isinstance(response_format, dict):
        text_format = _normalize_text_format(response_format)
        text_param = translated.get("text")
        if isinstance(text_param, dict):
            text_param.setdefault("format", text_format)
        elif text_param is None:
            translated["text"] = {"format": text_format}

    for unsupported in _UNSUPPORTED_KWARGS:
        if unsupported in translated:
            value = translated.pop(unsupported)
            # An unset default is not worth warning about; only a value the
            # operator actually chose is being discarded.
            if value not in (None, "", 0, 0.0, [], {}):
                logger.warning(
                    f"Dropping '{unsupported}' — the OpenAI Responses API has no "
                    f"equivalent parameter (value was {value!r})"
                )

    return translated


def _build_input(
    prompt: str,
    history_messages: list[dict[str, Any]],
    image_inputs: list[Any] | None,
) -> list[dict[str, Any]]:
    """Build the Responses ``input`` list.

    Responses distinguishes ``input_text`` from ``output_text``: history turns
    spoken by the assistant must be tagged as output, everything else as input.
    Tagging an assistant turn as ``input_text`` is rejected by the API.
    """
    api_input: list[dict[str, Any]] = []

    for message in history_messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        part_type = "output_text" if role == "assistant" else "input_text"
        if isinstance(content, str):
            api_input.append(
                {"role": role, "content": [{"type": part_type, "text": content}]}
            )
        else:
            # Already in content-part form; forward it untouched.
            api_input.append({"role": role, "content": content})

    user_content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
    if image_inputs:
        from lightrag.llm._vision_utils import normalize_image_inputs

        for img in normalize_image_inputs(image_inputs):
            user_content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:{img.mime_type};base64,{img.base64_str}",
                }
            )
    api_input.append({"role": "user", "content": user_content})
    return api_input


def _extract_text(response: Any) -> str:
    """Extract assistant body text from a Responses payload.

    Prefers the SDK's ``output_text`` convenience property and falls back to
    walking ``output[]`` so the driver still works against OpenAI-compatible
    servers whose SDK objects lack it.
    """
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text:
        return output_text

    parts: list[str] = []
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) != "message":
            continue
        for part in getattr(item, "content", None) or []:
            if getattr(part, "type", None) == "output_text":
                text = getattr(part, "text", None)
                if text:
                    parts.append(text)
    return "".join(parts)


def _extract_reasoning(response: Any) -> str:
    """Extract reasoning-summary text from a Responses payload.

    Reasoning arrives as ``output[]`` items of type ``reasoning`` carrying a
    ``summary`` list. Raw reasoning tokens are never returned by the API, so
    this is a summary rather than the full trace.
    """
    parts: list[str] = []
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) != "reasoning":
            continue
        for summary in getattr(item, "summary", None) or []:
            text = getattr(summary, "text", None)
            if text:
                parts.append(text)
    return "".join(parts)


def _is_truncated(response: Any) -> bool:
    """Whether the response stopped because it hit the output-token budget."""
    if getattr(response, "status", None) != "incomplete":
        return False
    details = getattr(response, "incomplete_details", None)
    return getattr(details, "reason", None) == "max_output_tokens"


def _incomplete_reason(terminal: Any) -> str:
    """Why a Responses generation reported ``status: incomplete``."""
    details = getattr(terminal, "incomplete_details", None)
    reason = getattr(details, "reason", None)
    return str(reason) if reason is not None else "unknown"


def _extract_refusal(response: Any) -> str:
    """Extract refusal text from a Responses payload.

    A refused turn carries ``refusal`` content parts instead of ``output_text``
    ones, so the body walk in :func:`_extract_text` sees nothing and the
    payload would otherwise be diagnosed as empty output.
    """
    parts: list[str] = []
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) != "message":
            continue
        for part in getattr(item, "content", None) or []:
            if getattr(part, "type", None) == "refusal":
                text = getattr(part, "refusal", None)
                if text:
                    parts.append(text)
    return "".join(parts)


def _usage_counts(usage: Any) -> dict[str, int]:
    """Map Responses usage onto the TokenTracker's Chat Completions key names."""
    return {
        "prompt_tokens": getattr(usage, "input_tokens", 0) or 0,
        "completion_tokens": getattr(usage, "output_tokens", 0) or 0,
        "total_tokens": getattr(usage, "total_tokens", 0) or 0,
    }


def _failure_reason(terminal: Any) -> str:
    """Human-readable reason a Responses generation reported ``status: failed``."""
    error = getattr(terminal, "error", None)
    message = getattr(error, "message", None) or "unknown failure"
    code = getattr(error, "code", None)
    return f"code={code if code is not None else 'n/a'}, {message}"


def _account_stream_usage(token_tracker: Any, usage: Any) -> None:
    """Record what the attempt spent, before any validation raises.

    The request consumed its budget whether or not the output turned out to be
    usable, and a failed generation is billed the same way a truncated one is.
    """
    if token_tracker is None or usage is None:
        return
    counts = _usage_counts(usage)
    token_tracker.add_usage(counts)
    logger.debug(f"Streaming token usage (from API): {counts}")


async def _close_quietly(client: Any) -> None:
    """Close the client, downgrading any close failure to a warning.

    A failure to close must never mask the exception that is already
    propagating, nor turn a successful call into a failed one.
    """
    try:
        await client.close()
    except Exception as close_error:  # noqa: BLE001 - cleanup must not raise
        logger.warning(f"Failed to close OpenAI Responses client: {close_error}")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=(
        retry_if_exception_type(RateLimitError)
        | retry_if_exception_type(APIConnectionError)
        | retry_if_exception_type(APITimeoutError)
        | retry_if_exception_type(InvalidResponseError)
        | retry_if_exception_type(InternalServerError)
        | retry_if_exception_type(TransientBadRequestError)
    ),
)
async def openai_responses_complete_if_cache(
    model: str,
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    enable_cot: bool = False,
    base_url: str | None = None,
    api_key: str | None = None,
    token_tracker: Any | None = None,
    stream: bool | None = None,
    timeout: int | None = None,
    keyword_extraction: bool = False,
    image_inputs: list[Any] | None = None,
    **kwargs: Any,
) -> Union[str, AsyncIterator[str]]:
    """Complete a prompt through the OpenAI Responses API.

    Mirrors the ``openai_complete_if_cache`` contract. See the module docstring
    for the wire-shape differences that force a separate implementation.

    Args:
        model: Model identifier, e.g. ``gpt-5.6``.
        prompt: The user prompt.
        system_prompt: Sent as the Responses ``instructions`` parameter rather
            than as a system message.
        history_messages: Prior conversation turns.
        enable_cot: Wrap reasoning-summary text in ``<think>`` tags.
        base_url: API base URL. Defaults to ``OPENAI_API_BASE`` or the public
            OpenAI endpoint.
        api_key: API key. Defaults to ``OPENAI_API_KEY``.
        token_tracker: Optional usage tracker.
        stream: Whether to stream the response.
        timeout: Request timeout in seconds.
        keyword_extraction: Deprecated shim; maps to a JSON-object output
            format when no explicit ``response_format`` is given.
        image_inputs: Optional images, sent as ``input_image`` content parts.
        **kwargs: Forwarded to the API. Chat Completions spellings
            (``max_tokens``, ``reasoning_effort``, ``response_format``) are
            translated; parameters with no Responses equivalent are dropped
            with a warning.

    Returns:
        The completion text, or an async iterator of text chunks when
        streaming. Token-limit-truncated non-streaming output is wrapped in
        ``TruncatedResponse`` so the cache layer skips persisting it; a
        truncated stream logs a warning instead, since chunks already yielded
        cannot be re-wrapped.

    Raises:
        InvalidResponseError: The response carried no usable text, the
            generation failed after the request was accepted, the model
            refused, or the generation was stopped for a reason other than
            the output budget.
        EmptyTruncatedResponseError: Empty output caused by budget exhaustion.
    """
    if history_messages is None:
        history_messages = []

    # Keep the SDK's own DEBUG chatter out of the log unless explicitly asked
    # for, matching the Chat Completions driver.
    if not VERBOSE_DEBUG and logger.level == logging.DEBUG:
        logging.getLogger("openai").setLevel(logging.INFO)

    client_configs = kwargs.pop("openai_client_configs", {})
    kwargs.pop("hashing_kv", None)
    # Accepted for signature parity with the Chat Completions driver, which
    # uses it only to pick a default response_format.
    entity_extraction = kwargs.pop("entity_extraction", False)

    if (entity_extraction or keyword_extraction) and kwargs.get(
        "response_format"
    ) is None:
        kwargs["response_format"] = {"type": "json_object"}

    response_format = kwargs.get("response_format")
    if response_format is not None and not isinstance(response_format, dict):
        raise ValueError(
            "openai_responses binding accepts only dict response_format payloads; "
            f"got {type(response_format).__name__}"
        )
    if response_format is not None:
        # Structured output and <think> wrapping are mutually exclusive: the
        # tags would corrupt the JSON the caller is about to parse.
        enable_cot = False

    api_kwargs = _translate_kwargs(kwargs)

    client = create_openai_async_client(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        client_configs=client_configs,
    )

    api_input = _build_input(prompt, history_messages, image_inputs)

    if system_prompt:
        api_kwargs["instructions"] = system_prompt
    if stream is not None:
        api_kwargs["stream"] = stream
    if timeout is not None:
        api_kwargs["timeout"] = timeout

    logger.debug("===== Entering func of LLM (openai_responses) =====")
    logger.debug(f"Model: {model}   Base URL: {base_url}")
    logger.debug(f"Additional kwargs: {api_kwargs}")
    logger.debug(f"Num of history messages: {len(history_messages)}")
    verbose_debug(f"System prompt: {system_prompt}")
    verbose_debug(f"Query: {prompt}")
    logger.debug("===== Sending Query to LLM =====")

    try:
        response = await client.responses.create(
            model=model, input=api_input, **api_kwargs
        )
    except (APITimeoutError, APIConnectionError, RateLimitError) as e:
        logger.error(f"OpenAI Responses API error: {e}")
        await _close_quietly(client)
        raise
    except BadRequestError as e:
        await _close_quietly(client)
        # Same transient-400 recovery as the Chat Completions driver: a
        # corrupted request body in transit surfaces as an unparseable-JSON
        # 400 and succeeds on retry. Genuine 400s still fail fast.
        if "could not parse" in str(e).lower():
            logger.warning(
                f"Transient JSON-parse 400 from OpenAI Responses, will retry: {e}"
            )
            raise TransientBadRequestError(str(e)) from e
        raise
    except Exception as e:
        body = getattr(e, "body", None)
        request_id = getattr(e, "request_id", None)
        extra_parts = []
        if body:
            extra_parts.append(f"Response body: {body}")
        if request_id:
            extra_parts.append(f"Request ID: {request_id}")
        extra = ("\n" + "\n".join(extra_parts)) if extra_parts else ""
        logger.error(
            f"OpenAI Responses API Call Failed,\nModel: {model},\n"
            f"Params: {api_kwargs}, Got: {e}{extra}"
        )
        await _close_quietly(client)
        raise

    if hasattr(response, "__aiter__"):
        return _stream_response(response, client, enable_cot, token_tracker)

    return await _handle_non_streaming(response, client, enable_cot, token_tracker)


async def _stream_response(
    response: Any,
    client: Any,
    enable_cot: bool,
    token_tracker: Any | None,
) -> AsyncIterator[str]:
    """Yield text chunks from a Responses event stream.

    Responses emits typed events rather than ``choices[].delta`` chunks, so COT
    state is driven by event *type* instead of by which delta field is
    populated.

    Truncation: a stream that exhausts ``max_output_tokens`` terminates with
    ``response.incomplete`` rather than ``response.completed``. Partial output
    is kept and a warning logged; a truncated stream that yielded nothing
    raises ``EmptyTruncatedResponseError``, matching the non-streaming path's
    rule that budget exhaustion with no usable text is non-retryable.

    Failure: a generation that fails after the request was accepted terminates
    with ``response.failed``, which is neither of those terminals nor the
    separate ``error`` event. It raises, so a provider failure never reaches
    the consumer as a short-but-successful answer.

    Stopped generations: ``response.incomplete`` also carries reasons other
    than the output budget, ``content_filter`` chief among them. Those are not
    truncation, so they raise rather than ending the iterator on whatever text
    had been yielded.

    Refusal: a refused turn streams ``response.refusal.*`` events and can still
    end with ``response.completed``. The refusal text is collected and raised
    after the terminal, rather than being dropped for want of an
    ``output_text`` delta.
    """
    cot_active = False
    body_text_seen = False
    reasoning_text_seen = False
    truncated = False
    incomplete_reason: str | None = None
    refusal_parts: list[str] = []
    refusal_seen = False
    final_terminal = None
    final_usage = None
    closing_via_generator_exit = False

    try:
        async for event in response:
            event_type = getattr(event, "type", "") or ""

            if event_type in ("response.completed", "response.incomplete"):
                # Exhausting max_output_tokens terminates the stream with
                # `response.incomplete` instead of `response.completed`. Both
                # carry the terminal usage; only the latter is a clean finish.
                terminal = getattr(event, "response", None)
                final_terminal = terminal
                usage = getattr(terminal, "usage", None)
                if usage is not None:
                    final_usage = usage
                if event_type == "response.incomplete":
                    truncated = _is_truncated(terminal)
                    if not truncated:
                        # An incomplete terminal for any reason other than the
                        # output budget is not truncation; it is a generation
                        # that was stopped, most often by a content filter. A
                        # terminal this driver cannot read reports "unknown"
                        # and takes the same path: incomplete is not complete,
                        # and guessing which reason it was would be worse.
                        incomplete_reason = _incomplete_reason(terminal)
                continue

            if event_type == "response.reasoning_summary_text.delta":
                if not enable_cot or body_text_seen:
                    continue
                delta = getattr(event, "delta", "") or ""
                if not delta:
                    continue
                if not cot_active:
                    yield "<think>"
                    cot_active = True
                if r"\u" in delta:
                    delta = safe_unicode_decode(delta.encode("utf-8"))
                reasoning_text_seen = True
                yield delta
                continue

            if event_type == "response.output_text.delta":
                delta = getattr(event, "delta", "") or ""
                if not delta:
                    continue
                # First body text closes any open reasoning block, matching the
                # Chat Completions rule that COT stops once content arrives.
                if cot_active:
                    yield "</think>"
                    cot_active = False
                body_text_seen = True
                if r"\u" in delta:
                    delta = safe_unicode_decode(delta.encode("utf-8"))
                yield delta
                continue

            if event_type == "response.refusal.delta":
                # A refusal streams on its own event type and never as body
                # text, so an unhandled one ends the iterator successfully with
                # nothing yielded. It is collected here and raised after the
                # terminal, so the terminal usage is still accounted.
                #
                # The event and its text are tracked apart: an empty delta is
                # still a refusal, and letting it stand in for the text would
                # both suppress the raise and disarm the `.done` fallback.
                refusal_seen = True
                delta = getattr(event, "delta", "") or ""
                if delta:
                    refusal_parts.append(delta)
                continue

            if event_type == "response.refusal.done":
                # Carries the whole refusal. Only used when the deltas carried
                # no text, since otherwise it would duplicate them.
                refusal_seen = True
                if not refusal_parts:
                    done_text = getattr(event, "refusal", "") or ""
                    if done_text:
                        refusal_parts.append(done_text)
                continue

            if event_type == "response.failed":
                # A request the transport accepted can still fail during
                # generation. That terminates the stream with `response.failed`
                # -- neither a completed/incomplete terminal nor the separate
                # `error` event -- so without this branch the iterator would end
                # successfully on whatever partial text had already been
                # yielded, discarding the provider's reason.
                terminal = getattr(event, "response", None)
                _account_stream_usage(token_tracker, getattr(terminal, "usage", None))
                raise InvalidResponseError(
                    f"OpenAI Responses stream failed ({_failure_reason(terminal)})"
                )

            if event_type == "error":
                message = getattr(event, "message", "") or "unknown streaming error"
                raise InvalidResponseError(f"OpenAI Responses stream error: {message}")

        if cot_active:
            yield "</think>"
            cot_active = False

        _account_stream_usage(token_tracker, final_usage)

        refusal = "".join(refusal_parts).strip()
        if not refusal:
            # Same fallback as `_extract_text`: a server that populates the
            # terminal payload but does not emit the refusal events would
            # otherwise end the iterator empty and successful.
            refusal = _extract_refusal(final_terminal).strip()
        if refusal or refusal_seen:
            error_message = (
                "OpenAI Responses stream refused the request: "
                f"{refusal or 'no reason given'}"
            )
            logger.error(error_message)
            # Retryable, like the `error` and `response.failed` terminals: a
            # refusal is a provider verdict this driver cannot distinguish from
            # a transient safety-stack failure, and the retry set is the
            # module's default for everything except budget exhaustion, whose
            # outcome is fixed by the caller's own max_output_tokens.
            raise InvalidResponseError(error_message)

        if incomplete_reason is not None:
            # Text already yielded is the tail of a generation that did not
            # finish, so ending here would present it as a complete answer. The
            # budget case below is the one incomplete reason that keeps its
            # partial output, because the model chose where to stop within a
            # limit the caller set.
            error_message = (
                "OpenAI Responses stream stopped before completion "
                f"(incomplete_reason={incomplete_reason})"
            )
            logger.error(error_message)
            raise InvalidResponseError(error_message)

        if truncated:
            if body_text_seen or reasoning_text_seen:
                # Partial output already reached the consumer, so raising here
                # would discard text it can still salvage. Mirrors the
                # non-streaming path, which returns partial content wrapped in
                # TruncatedResponse - a wrapper a generator cannot apply to
                # chunks it has already yielded.
                logger.warning(
                    "OpenAI Responses stream truncated by token limit, "
                    "returning partial content"
                )
            else:
                output_tokens = getattr(final_usage, "output_tokens", None)
                usage_details = getattr(final_usage, "output_tokens_details", None)
                reasoning_tokens = getattr(usage_details, "reasoning_tokens", None)
                hint = empty_length_truncated_hint(
                    "consider raising max_output_tokens or lowering reasoning effort",
                    reasoning_consumed_budget=bool(reasoning_tokens),
                )
                error_message = (
                    "Received empty content from OpenAI Responses stream "
                    "(status=incomplete, incomplete_reason=max_output_tokens, "
                    f"output_tokens={output_tokens if output_tokens is not None else 'n/a'}, "
                    f"reasoning_tokens={reasoning_tokens if reasoning_tokens is not None else 'n/a'}"
                    f"): {hint}"
                )
                logger.error(error_message)
                # Budget exhaustion is deterministic for a given prompt and
                # budget, so it raises the non-retryable type rather than
                # buying two more full-budget generations.
                raise EmptyTruncatedResponseError(error_message)
    except GeneratorExit:
        # Consumer disconnected: the finally block must not yield, or cleanup
        # aborts with "async generator ignored GeneratorExit".
        closing_via_generator_exit = True
        raise
    except Exception as e:
        if cot_active:
            try:
                yield "</think>"
                cot_active = False
            except Exception as close_error:  # noqa: BLE001 - cleanup only
                logger.warning(f"Failed to close COT tag on error: {close_error}")
        logger.error(f"Error in Responses stream: {e}")
        raise
    finally:
        if cot_active and not closing_via_generator_exit:
            try:
                yield "</think>"
            except Exception as final_error:  # noqa: BLE001 - cleanup only
                logger.warning(f"Failed to close COT tag in finally: {final_error}")
        aclose = getattr(response, "aclose", None)
        if callable(aclose):
            try:
                await response.aclose()
            except Exception as close_error:  # noqa: BLE001 - cleanup only
                logger.debug(f"Stream close not supported by wrapper: {close_error}")
        await _close_quietly(client)


async def _handle_non_streaming(
    response: Any,
    client: Any,
    enable_cot: bool,
    token_tracker: Any | None,
) -> str:
    """Turn a non-streaming Responses payload into the driver's ``str`` result."""
    try:
        # Count usage before any validation raises: the request consumed its
        # budget whether or not the output turned out to be usable, and a
        # reasoning model that spent the whole budget on its trace is exactly
        # the case that raises below.
        usage = getattr(response, "usage", None)
        if token_tracker and usage is not None:
            token_tracker.add_usage(_usage_counts(usage))

        # Same terminal as `response.failed` on the streaming path: a request
        # accepted at the transport layer that failed during generation returns
        # status "failed" with the reason in `error`, rather than raising from
        # responses.create(). Checked before the content walk so the provider's
        # reason is reported instead of a bare empty-output diagnosis, and so a
        # failed generation that carried partial output is not returned as a
        # successful short answer.
        if getattr(response, "status", None) == "failed":
            error_message = (
                f"OpenAI Responses request failed ({_failure_reason(response)})"
            )
            logger.error(error_message)
            raise InvalidResponseError(error_message)

        # Sibling of the `response.refusal.*` events on the streaming path: a
        # refused turn carries `refusal` content parts, which the body walk
        # skips, so without this the refusal would be reported as "model
        # produced no output" and its reason discarded.
        refusal = _extract_refusal(response).strip()
        if refusal:
            error_message = f"OpenAI Responses request was refused: {refusal}"
            logger.error(error_message)
            raise InvalidResponseError(error_message)

        truncated = _is_truncated(response)

        # Sibling of the non-budget `response.incomplete` terminal on the
        # streaming path. Budget truncation keeps its partial output below,
        # because the model chose where to stop within a limit the caller set;
        # any other incomplete reason is a generation that was stopped, and
        # returning its tail would present it as a complete answer.
        if getattr(response, "status", None) == "incomplete" and not truncated:
            error_message = (
                "OpenAI Responses request stopped before completion "
                f"(incomplete_reason={_incomplete_reason(response)})"
            )
            logger.error(error_message)
            raise InvalidResponseError(error_message)

        content = _extract_text(response)
        reasoning = _extract_reasoning(response) if enable_cot else ""

        final_content = content or ""
        if enable_cot and reasoning.strip() and not final_content.strip():
            if r"\u" in reasoning:
                reasoning = safe_unicode_decode(reasoning.encode("utf-8"))
            final_content = f"<think>{reasoning}</think>{final_content}"

        if not final_content or not final_content.strip():
            status = getattr(response, "status", None)
            details = getattr(response, "incomplete_details", None)
            reason = getattr(details, "reason", None)
            output_tokens = getattr(usage, "output_tokens", None)
            usage_details = getattr(usage, "output_tokens_details", None)
            reasoning_tokens = getattr(usage_details, "reasoning_tokens", None)
            reasoning_len = len(reasoning.strip()) if reasoning else 0
            diagnostics = (
                f"status={status if status is not None else 'n/a'}, "
                f"incomplete_reason={reason if reason is not None else 'n/a'}, "
                f"output_tokens={output_tokens if output_tokens is not None else 'n/a'}, "
                f"reasoning_tokens={reasoning_tokens if reasoning_tokens is not None else 'n/a'}, "
                f"reasoning_content_len={reasoning_len}"
            )
            if truncated:
                hint = empty_length_truncated_hint(
                    "consider raising max_output_tokens or lowering reasoning effort",
                    reasoning_consumed_budget=bool(reasoning_tokens),
                )
            elif reasoning_len:
                hint = (
                    "model returned reasoning-only output; "
                    "consider disabling thinking mode for this role"
                )
            else:
                hint = "model produced no output"
            error_message = (
                f"Received empty content from OpenAI Responses API "
                f"({diagnostics}): {hint}"
            )
            logger.error(error_message)
            # Budget exhaustion is deterministic for a given prompt and budget,
            # so it raises the non-retryable type rather than buying two more
            # full-budget generations. Other empty modes stay retryable.
            if truncated:
                raise EmptyTruncatedResponseError(error_message)
            raise InvalidResponseError(error_message)

        if r"\u" in final_content:
            final_content = safe_unicode_decode(final_content.encode("utf-8"))

        if truncated:
            logger.warning(
                "OpenAI Responses output truncated by token limit "
                f"(content_len={len(final_content)}), returning partial content"
            )
            final_content = TruncatedResponse(final_content)

        logger.debug(f"Response content len: {len(final_content)}")
        verbose_debug(f"Response: {response}")
        return final_content
    finally:
        await _close_quietly(client)


async def openai_responses_complete(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    keyword_extraction: bool = False,
    entity_extraction: bool = False,
    **kwargs: Any,
) -> Union[str, AsyncIterator[str]]:
    """LightRAG ``llm_model_func`` entry point for the Responses binding."""
    if history_messages is None:
        history_messages = []
    entity_extraction = kwargs.pop("entity_extraction", entity_extraction)
    model_name = kwargs["hashing_kv"].global_config["llm_model_name"]
    return await openai_responses_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        keyword_extraction=keyword_extraction,
        entity_extraction=entity_extraction,
        **kwargs,
    )
