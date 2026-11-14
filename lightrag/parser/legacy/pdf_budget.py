"""Document-wide extraction budget for the legacy PDF extractor.

pypdf limits each stream it decodes, not the document. Pages can share one
content stream, and one form XObject can be invoked many times. A tiny PDF can
therefore demand gigabytes of parsing, and ``MAX_UPLOAD_SIZE`` bounds only the
file.

Rules for callers:

* Use one :class:`PdfExtractionBudget` per document. Call
  :meth:`~PdfExtractionBudget.check_page_contents` with EVERY page before
  extracting any. Then extract each page with
  ``page.extract_text(**budget.visitors(page))`` and pass the returned text to
  :meth:`~PdfExtractionBudget.finish_page`. Skipping ``finish_page`` can lose
  an error that pypdf swallowed inside a form XObject.
* Do not catch :class:`PdfExtractionBudgetError` to keep partial text. It is a
  ``LegacyExtractionError``, and the document must be recorded FAILED.
* The limits are ``PDF_MAX_CONTENT_BYTES``, ``PDF_MAX_CONTENT_RATIO`` and
  ``PDF_MAX_EXTRACTED_CHARS``, read at construction. Defaults are in
  :data:`lightrag.constants.DEFAULT_PDF_MAX_CONTENT_BYTES` and its neighbours.

The mechanism is in docs/design/PdfExtractionBudget.md: what is charged, how
form XObjects are resolved, why a tripped budget re-raises, the accepted
limits and the rejected alternatives.
"""

from __future__ import annotations

from typing import Any

from pypdf.generic import NameObject

from lightrag.constants import (
    DEFAULT_PDF_CONTENT_RATIO_FLOOR_BYTES,
    DEFAULT_PDF_MAX_CONTENT_BYTES,
    DEFAULT_PDF_MAX_CONTENT_RATIO,
    DEFAULT_PDF_MAX_EXTRACTED_CHARS,
)
from lightrag.parser.legacy.extractors import LegacyExtractionError
from lightrag.utils import get_env_value

_TEXT_OPERATORS = frozenset({b"Tj", b"'", b'"', b"TJ"})
_MAX_PARENT_DEPTH = 100


class PdfExtractionBudgetError(LegacyExtractionError):
    """A PDF needs more parsing or text than the extractor will spend on it.

    A ``LegacyExtractionError`` (thus a ``ValueError``), so the pipeline
    records the document FAILED with this message.
    """


def _resolve(obj: Any) -> Any:
    get_object = getattr(obj, "get_object", None)
    return get_object() if get_object is not None else obj


def _decoded_length(stream: Any) -> int:
    """Decoded size of ``stream``, or 0 when it is not a readable stream.

    A stream pypdf cannot decode is left for pypdf to report or skip, exactly
    as before this budget existed; it costs pypdf no parsing either way.
    """
    stream = _resolve(stream)
    get_data = getattr(stream, "get_data", None)
    if get_data is None:
        return 0
    try:
        return len(get_data())
    except MemoryError:
        raise
    except Exception:
        return 0


def _page_resources(page: Any) -> Any:
    """``/Resources`` of ``page``, inherited through ``/Parent`` like pypdf."""
    node = page
    for _ in range(_MAX_PARENT_DEPTH):
        if node is None:
            return None
        if "/Resources" in node:
            return _resolve(node["/Resources"])
        node = _resolve(node.get("/Parent"))
    return None


class PdfExtractionBudget:
    """Counters for one ``_extract_pdf_pypdf`` call.

    Call :meth:`check_page_contents` once with every page before extracting
    any, then extract each page with :meth:`visitors` and pass its text to
    :meth:`finish_page`.
    """

    def __init__(self, file_size: int, file_path: str) -> None:
        self._file_path = file_path
        self._content_cap, self._content_env = self._content_limit(file_size)
        self._max_chars = get_env_value(
            "PDF_MAX_EXTRACTED_CHARS", DEFAULT_PDF_MAX_EXTRACTED_CHARS, int
        )
        self.content_bytes = 0
        self.chars = 0
        self._page_chars = 0
        self._resources: list[Any] = []
        self._tripped: str | None = None

    @staticmethod
    def _content_limit(file_size: int) -> tuple[int, str]:
        """Return ``(cap, env name)``; a cap of 0 means unlimited."""
        max_bytes = get_env_value(
            "PDF_MAX_CONTENT_BYTES", DEFAULT_PDF_MAX_CONTENT_BYTES, int
        )
        max_ratio = get_env_value(
            "PDF_MAX_CONTENT_RATIO", DEFAULT_PDF_MAX_CONTENT_RATIO, int
        )
        gates: list[tuple[int, str]] = []
        if max_bytes > 0:
            gates.append((max_bytes, "PDF_MAX_CONTENT_BYTES"))
        if max_ratio > 0:
            allowance = max(
                DEFAULT_PDF_CONTENT_RATIO_FLOOR_BYTES, max_ratio * max(0, file_size)
            )
            gates.append((allowance, "PDF_MAX_CONTENT_RATIO"))
        return min(gates) if gates else (0, "")

    def _trip(self, message: str) -> None:
        self._tripped = f"Refusing {self._file_path}: {message}"
        raise PdfExtractionBudgetError(self._tripped)

    def _check(self) -> None:
        if self._tripped is not None:
            raise PdfExtractionBudgetError(self._tripped)

    def _add_content(self, length: int, what: str) -> None:
        self.content_bytes += length
        if self._content_cap and self.content_bytes > self._content_cap:
            self._trip(
                f"its content streams decode to more than {self._content_cap} "
                f"bytes (reached at {what}; env {self._content_env})"
            )

    def _check_chars(self) -> None:
        total = self.chars + self._page_chars
        if self._max_chars > 0 and total > self._max_chars:
            self._trip(
                f"it extracts more than {self._max_chars} characters "
                "(env PDF_MAX_EXTRACTED_CHARS)"
            )

    def check_page_contents(self, pages: Any) -> None:
        """Measure every page's ``/Contents`` before anything is parsed."""
        for number, page in enumerate(pages, start=1):
            contents = _resolve(page.get("/Contents"))
            streams = contents if isinstance(contents, list) else [contents]
            for stream in streams:
                self._add_content(_decoded_length(stream), f"page {number}")

    def visitors(self, page: Any) -> dict[str, Any]:
        """Keyword arguments for ``page.extract_text``."""
        self._check()
        self._page_chars = 0
        self._resources = [_page_resources(page)]
        return {
            "visitor_operand_before": self._before,
            "visitor_operand_after": self._after,
        }

    def finish_page(self, text: str) -> None:
        """Count the text a page returned; raise if the budget tripped."""
        self._check()
        self._page_chars = 0
        self.chars += len(text)
        self._check_chars()

    def _before(self, operator: bytes, operands: Any, *_: Any) -> None:
        self._check()
        if operator in _TEXT_OPERATORS:
            self._page_chars += _shown_text_length(operator, operands)
            self._check_chars()
        elif operator == b"Do":
            self._enter_xobject(operands)

    def _after(self, operator: bytes, *_: Any) -> None:
        if operator == b"Do" and len(self._resources) > 1:
            self._resources.pop()

    def _enter_xobject(self, operands: Any) -> None:
        """Charge a form XObject's stream; push the resources it runs with.

        Pushes exactly one entry, popped by :meth:`_after`, unless the charge
        trips the budget, in which case pypdf never calls ``_after`` for it.
        """
        current = self._resources[-1] if self._resources else None
        xform = None
        try:
            xobjects = _resolve(current.get("/XObject")) if current else None
            if xobjects and operands:
                xform = _resolve(xobjects.get(operands[0]))
        except MemoryError:
            raise
        except Exception:
            xform = None
        if xform is None or xform.get("/Subtype") == "/Image":
            self._resources.append(None)
            return
        self._add_content(_decoded_length(xform), f"form XObject {operands[0]}")
        # A form without its own /Resources is charged anyway (pypdf skips
        # it) and runs with the caller's, a superset of what pypdf resolves.
        own = _resolve(xform.get("/Resources"))
        self._resources.append(own if own else current)


def _shown_text_length(operator: bytes, operands: Any) -> int:
    if operator == b"TJ":
        shown = _resolve(operands[0]) if operands else []
        if not isinstance(shown, list):
            return 0
    elif operator == b'"':
        shown = operands[2:3]
    else:
        shown = operands[:1]
    # NameObject subclasses str but is not shown text.
    return sum(
        len(item)
        for item in shown
        if isinstance(item, (str, bytes)) and not isinstance(item, NameObject)
    )


__all__ = ["PdfExtractionBudget", "PdfExtractionBudgetError"]
