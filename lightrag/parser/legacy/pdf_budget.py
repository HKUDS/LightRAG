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

from pypdf.generic import DictionaryObject, NameObject, StreamObject

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


def _detached(exc: Exception) -> Exception:
    """``exc`` with no traceback on it or on the exceptions it chains.

    A decoder's frames hold its buffers, and re-raising one instance appends
    frames on every raise; a memoized failure must keep neither.
    """
    seen: set[int] = set()
    node: BaseException | None = exc
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        node.__traceback__ = None
        node = node.__cause__ or node.__context__
    return exc


def _inherited_resources(obj: Any) -> Any:
    """``/Resources`` of a page or form, inherited through ``/Parent`` like pypdf.

    pypdf looks a form's resources up exactly as a page's, ``/Parent``
    included. ``None`` when no node declares it, the one case every supported
    pypdf release treats as "nothing to parse", and also when the lookup
    itself fails: pypdf then fails or skips the object on its own, and either
    way parses nothing.
    """
    try:
        node = obj
        for _ in range(_MAX_PARENT_DEPTH):
            if not isinstance(node, DictionaryObject):
                return None
            if "/Resources" in node:
                return _resolve(node["/Resources"])
            node = _resolve(node.get("/Parent"))
    except MemoryError:
        raise
    except Exception:
        return None
    return None


def _lookup_form(resources: Any, operands: Any) -> tuple[Any, Any] | None:
    """``(form, its /Resources)`` for a ``Do``, or ``None`` if pypdf parses nothing.

    Never raises (except ``MemoryError``): it runs in a visitor callback,
    outside pypdf's own catch, so an error here would fail a page that pypdf
    extracts. Images, non-streams, streams without ``/Subtype``, forms whose
    ``/Resources`` cannot be found even through ``/Parent``, and anything that
    fails to resolve are all "nothing", as they are for pypdf.
    """
    try:
        xobjects = _resolve(resources.get("/XObject")) if resources else None
        if not xobjects or not operands:
            return None
        form = _resolve(xobjects.get(operands[0]))
        # pypdf reads /Subtype as a required key, so a stream without one is
        # skipped inside its catch.
        if not isinstance(form, StreamObject) or form["/Subtype"] == "/Image":
            return None
        resources = _inherited_resources(form)
        if resources is None:
            return None
        return form, resources
    except MemoryError:
        raise
    except Exception:
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
        # Keyed by id(); the object is kept so the id cannot be reused.
        self._decoded: dict[int, tuple[Any, int | Exception]] = {}
        self._failed_forms: set[int] = set()

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

    def _decoded_length(self, stream: Any) -> int:
        """Decoded size of ``stream``; 0 when it is not a stream.

        Raises the decode error. Memoized per object, failures included:
        pypdf caches a successful decode but not a failed one, and a shared
        stream must not be decompressed again for every page that names it.
        """
        stream = _resolve(stream)
        get_data = getattr(stream, "get_data", None)
        if get_data is None:
            return 0
        key = id(stream)
        if key not in self._decoded:
            try:
                result: int | Exception = len(get_data())
            except MemoryError:
                raise
            except Exception as exc:
                result = exc
            self._decoded[key] = (stream, result)
        result = self._decoded[key][1]
        if isinstance(result, Exception):
            raise _detached(result)
        return result

    def check_page_contents(self, pages: Any) -> None:
        """Measure every page's ``/Contents`` before anything is parsed.

        A decode error propagates for a page pypdf would parse, as pypdf's own
        error would have at that page.
        """
        for number, page in enumerate(pages, start=1):
            resources = _inherited_resources(page)
            if resources is None:
                continue
            try:
                contents = _resolve(page.get("/Contents"))
                streams = contents if isinstance(contents, list) else [contents]
                for stream in streams:
                    self._add_content(self._decoded_length(stream), f"page {number}")
            except PdfExtractionBudgetError:
                raise
            except (AttributeError, KeyError):
                # pypdf returns no text for such a page.
                continue
            except MemoryError:
                raise
            except Exception:
                if resources:
                    raise
                # Empty /Resources: recent pypdf skips the page.
                continue

    def visitors(self, page: Any) -> dict[str, Any]:
        """Keyword arguments for ``page.extract_text``."""
        self._check()
        self._page_chars = 0
        self._resources = [_inherited_resources(page)]
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
            try:
                shown = _shown_text_length(operator, operands)
            except MemoryError:
                raise
            except Exception:
                # Never raise from a visitor except to stop the budget.
                shown = 0
            self._page_chars += shown
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
        found = _lookup_form(current, operands)
        if found is None:
            self._resources.append(None)
            return
        xform, own = found
        name = operands[0]
        try:
            length = self._decoded_length(xform)
        except MemoryError:
            raise
        except Exception:
            # pypdf swallows the failure and decodes the form again at the
            # next invocation; one failure per form is the most it may cost.
            # Recent pypdf does not decode a form with empty /Resources at
            # all, so only a form it would decode can trip here.
            if own and id(xform) in self._failed_forms:
                self._trip(
                    f"form XObject {name} cannot be decoded and is invoked again"
                )
            self._failed_forms.add(id(xform))
            length = 0
        self._add_content(length, f"form XObject {name}")
        self._resources.append(own)


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
