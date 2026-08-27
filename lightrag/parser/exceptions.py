"""Parser-wide cancellation exception hierarchy.

Native parser ``extract`` hooks and other blocking parse paths raise these to
signal cancellation (or shutdown) while a document is being processed.
``pipeline.py`` catches :class:`ParsePipelineCancelled` to mark a document as
cancelled rather than failed, so any blocking parse path that wants that
treatment must raise from this family.
"""

from __future__ import annotations

import threading


class ParseCancelled(RuntimeError):
    """The parse was cancelled (or the rag shut down) during a blocking wait.

    This is the repo-wide "a blocking parse wait was cancelled" signal.
    Originally introduced for the LLM bridge, the hierarchy now covers any
    blocking parse path — not just LLM calls.
    """


class ParsePipelineCancelled(ParseCancelled):
    """The active document pipeline was cancelled by its caller."""


class ParseShutdown(ParseCancelled):
    """The RAG parser executor is shutting down."""


def normalize_cancel_events(
    cancel_events: tuple[
        threading.Event | tuple[threading.Event, type[ParseCancelled]] | None, ...
    ],
) -> tuple[tuple[threading.Event, type[ParseCancelled]], ...]:
    """Normalize a cancel-event spec into ``(event, exception_type)`` pairs.

    An entry may be a bare Event (mapped to :class:`ParseCancelled`) or an
    ``(Event, exception_type)`` pair when the caller needs the cancellation
    source preserved. ``None`` events are dropped so callers can pass optional
    events positionally without filtering first.
    """
    normalized: list[tuple[threading.Event, type[ParseCancelled]]] = []
    for entry in cancel_events:
        if isinstance(entry, tuple):
            event, exception_type = entry
        else:
            event, exception_type = entry, ParseCancelled
        if event is not None:
            normalized.append((event, exception_type))
    return tuple(normalized)


def first_cancellation(
    cancel_events: tuple[tuple[threading.Event, type[ParseCancelled]], ...],
    message: str,
) -> ParseCancelled | None:
    """Return the exception for the first set event, or ``None`` if none is set."""
    for event, exception_type in cancel_events:
        if event.is_set():
            return exception_type(message)
    return None
