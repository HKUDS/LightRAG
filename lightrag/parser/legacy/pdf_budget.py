"""Document-wide extraction budget for the legacy PDF extractor.

pypdf bounds each stream it decodes (75 MB by default) and, in recent
releases, the form XObjects one page may enter. Nothing bounds the DOCUMENT.
Pages can share one content stream, and a page can invoke the same form
XObject thousands of times, which pypdf parses again at every invocation. A
few tens of kilobytes of PDF therefore make pypdf parse and return gigabytes,
and ``MAX_UPLOAD_SIZE`` bounds only the file.

Two document-wide budgets apply, read from the environment for each parse
(see :data:`lightrag.constants.DEFAULT_PDF_MAX_CONTENT_BYTES`):

* **Content bytes** are the decoded bytes of every content stream pypdf
  parses. Every page's ``/Contents`` is measured BEFORE any page is parsed,
  which costs decompression only. A form XObject's stream is added at each
  ``Do`` that invokes it, before pypdf parses it. This is the CPU bound: a
  content stream need not show a single character to be expensive.
* **Extracted characters** bound the text held in memory. While a page is
  interpreted, the string operands of the text-showing operators are counted.
  The text each page returns is then counted exactly.

How the abort works: both counters run in pypdf's ``extract_text`` visitor
callbacks. pypdf catches ``Exception`` around a form XObject, so an error
raised inside one is logged and swallowed. The budget therefore stays
tripped and raises again from every later callback. The page-level loop has
no such catch, so one of those raises propagates. The budget is also checked
again after each page returns.

Accepted limits: work inside ONE stream is bounded by pypdf's own per-stream
caps, not by this budget, so the budget can be overrun by at most one
stream's worth. Counting is conservative: a form XObject pypdf would skip
(a cycle, or pypdf's per-page invocation cap) is still counted, which can
only make the budget stricter.
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
