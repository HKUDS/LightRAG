"""pypdf text extraction, runnable as a resource-limited child process.

Rules:

* **Import nothing but the standard library at module level, and never
  ``lightrag``.** ``lightrag.parser.legacy.pdf_budget`` runs this file by path
  as ``python -P pdf_worker.py``; importing the package would add seconds and
  hundreds of megabytes to every PDF. pypdf is imported inside functions.
* The parent passes everything through the header (see :func:`main`). Do not
  read LightRAG configuration from the environment here.
* :func:`extract` is also called in-process when isolation is switched off, so
  it must not touch process-wide state. Limits that act on the whole process
  belong in :func:`main`.

Protocol, limits and the reasons for them: docs/design/PdfExtractionBudget.md.
"""

from __future__ import annotations

import io
import json
import os
import sys
import threading
import time
from collections.abc import Callable
from typing import Any

# How a refusal is classified for the parent. "budget" is a limit this module
# enforces; "password" is an encrypted PDF that cannot be opened.
KIND_BUDGET = "budget"
KIND_PASSWORD = "password"
KIND_ERROR = "error"

_WATCHDOG_INTERVAL_SECONDS = 0.2
_MAX_REPORTED_WARNINGS = 5


class PdfRefusal(Exception):
    """Extraction stopped for a reason that has a message for the user."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


def _resolve(obj: Any) -> Any:
    get_object = getattr(obj, "get_object", None)
    return get_object() if get_object is not None else obj


def _check_page_contents(pages: list[Any], cap: int, env: str, name: str) -> None:
    """Refuse when the pages' content streams decode to more than ``cap`` bytes.

    Costs decompression only, and each object is decoded and traversed once
    however many pages share it. A stream that cannot be read counts as 0 and
    is left for pypdf to report or skip. This is a cheap, deterministic first
    gate, not the security boundary: the process limits are.
    """
    sizes: dict[int, tuple[Any, int]] = {}
    total = 0

    def charge(size: int, number: int) -> None:
        nonlocal total
        total += size
        if total > cap:
            raise PdfRefusal(
                KIND_BUDGET,
                f"Refusing {name}: its page content streams decode to more than "
                f"{cap} bytes (reached at page {number}; env {env})",
            )

    def stream_size(obj: Any) -> int:
        try:
            obj = _resolve(obj)
            key = id(obj)
            if key not in sizes:
                get_data = getattr(obj, "get_data", None)
                sizes[key] = (obj, len(get_data()) if get_data else 0)
            return sizes[key][1]
        except MemoryError:
            raise
        except Exception:
            sizes[id(obj)] = (obj, 0)
            return 0

    for number, page in enumerate(pages, start=1):
        try:
            contents = _resolve(page.get("/Contents"))
        except MemoryError:
            raise
        except Exception:
            continue
        key = id(contents)
        if key in sizes:
            charge(sizes[key][1], number)
            continue
        if isinstance(contents, list):
            size = 0
            for entry in contents:
                entry_size = stream_size(entry)
                size += entry_size
                charge(entry_size, number)
            sizes[key] = (contents, size)
        else:
            charge(stream_size(contents), number)


def extract(
    data: bytes,
    password: str | None,
    limits: dict[str, Any],
    emit: Callable[[str], None],
    name: str = "pdf",
) -> None:
    """Extract ``data`` page by page, passing each page's text to ``emit``.

    Raises :class:`PdfRefusal` for an unopenable encrypted PDF and for the two
    deterministic limits in ``limits``: ``content_cap`` (with ``content_env``
    naming the knob) and ``max_chars``. Any pypdf error propagates unchanged.
    """
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        # Try empty password first (covers permission-only encrypted PDFs)
        if reader.decrypt(password or "") == 0:
            raise PdfRefusal(
                KIND_PASSWORD,
                "Incorrect PDF password"
                if password
                else "PDF is encrypted but no password provided",
            )

    pages = list(reader.pages)
    content_cap = int(limits.get("content_cap") or 0)
    if content_cap > 0:
        _check_page_contents(
            pages, content_cap, str(limits.get("content_env") or ""), name
        )

    max_chars = int(limits.get("max_chars") or 0)
    chars = 0
    for page in pages:
        text = page.extract_text()
        chars += len(text)
        if max_chars > 0 and chars > max_chars:
            raise PdfRefusal(
                KIND_BUDGET,
                f"Refusing {name}: it extracts more than {max_chars} characters "
                "(env PDF_MAX_EXTRACTED_CHARS)",
            )
        emit(text + "\n")


# --- Child-process entry point ----------------------------------------------

_status_lock = threading.Lock()


def _finish(code: int, status: dict[str, Any]) -> None:
    """Write the one status line and exit. Whoever takes the lock first wins."""
    _status_lock.acquire()
    try:
        sys.stdout.flush()
    except Exception:
        pass
    os.write(2, b"\n" + json.dumps(status).encode("ascii", "replace") + b"\n")
    os._exit(code)


def _peak_rss_bytes() -> int | None:
    try:
        import resource
    except ImportError:  # Windows
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def _memory_in_use(inherited_peak: int) -> int | None:
    """Resident memory of this process, or ``None`` when it cannot be told.

    Never the raw peak from ``getrusage``: on Linux a child starts with its
    PARENT's peak, so a large server process would make every PDF look over
    the limit. Linux reports the current figure in ``/proc``. Elsewhere only
    the peak exists; it is trusted once it has grown past the value this
    process started with.
    """
    try:
        with open("/proc/self/statm", "rb") as statm:
            return int(statm.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError, AttributeError):
        pass
    peak = _peak_rss_bytes()
    return peak if peak is not None and peak > inherited_peak else None


def _watchdog(
    cpu_seconds: float,
    memory_bytes: int,
    name: str,
    inherited_peak: int,
    parent_pid: int,
) -> None:
    """Stop the process once it has used its CPU time or memory.

    A thread, so it works on every platform and can say why it stopped. It
    cannot run while a C call holds the GIL; the rlimits set in
    :func:`_apply_rlimits` and the parent's wall-clock timeout cover that.
    On POSIX it also ends the process when the parent has died.
    """
    while True:
        if parent_pid and os.name == "posix" and os.getppid() != parent_pid:
            # The server is gone: nobody will read the result. Without this,
            # every restart during a parse would leave children running.
            os._exit(4)
        time.sleep(_WATCHDOG_INTERVAL_SECONDS)
        if cpu_seconds > 0 and time.process_time() > cpu_seconds:
            _finish(
                3,
                {
                    "kind": KIND_BUDGET,
                    "message": f"Refusing {name}: extraction used more than "
                    f"{cpu_seconds:g} CPU seconds (env PDF_MAX_CPU_SECONDS)",
                },
            )
        used = _memory_in_use(inherited_peak) if memory_bytes > 0 else None
        if used is not None and used > memory_bytes:
            _finish(
                3,
                {
                    "kind": KIND_BUDGET,
                    "message": f"Refusing {name}: extraction used more than "
                    f"{memory_bytes // (1024 * 1024)} MiB of memory "
                    "(env PDF_MAX_MEMORY_MB)",
                },
            )


def _apply_rlimits(cpu_seconds: float, memory_bytes: int, output_bytes: int) -> None:
    """Hard operating-system limits behind the watchdog (POSIX only)."""
    try:
        import resource
    except ImportError:  # Windows: the watchdog and the parent's timeout remain
        return

    def limit(which: int, soft: int, hard: int) -> None:
        try:
            _, current_hard = resource.getrlimit(which)
            if current_hard != resource.RLIM_INFINITY:
                soft, hard = min(soft, current_hard), min(hard, current_hard)
            resource.setrlimit(which, (soft, hard))
        except (ValueError, OSError):
            pass  # not supported here (e.g. RLIMIT_AS on macOS)

    limit(resource.RLIMIT_CORE, 0, 0)
    if cpu_seconds > 0:
        # Above the watchdog's limit, so the watchdog normally reports first.
        soft = int(cpu_seconds) + 2
        limit(resource.RLIMIT_CPU, soft, soft + 2)
    if memory_bytes > 0:
        # Address space, not resident memory: twice the watchdog's limit.
        limit(resource.RLIMIT_AS, memory_bytes * 2, memory_bytes * 2)
    if output_bytes > 0:
        limit(resource.RLIMIT_FSIZE, output_bytes, output_bytes)


def main() -> None:
    """Read ``header-line + PDF`` from stdin; write text to stdout, status to fd 2.

    The header is one JSON line: ``path`` (the parent's ``sys.path``),
    ``name``, ``password``, ``limits`` (see :func:`extract`), ``parent_pid``
    and the process limits ``cpu_seconds``, ``memory_bytes`` and
    ``output_bytes``. The status
    is the last line on fd 2: ``{"kind": "ok"}`` or a kind with a message.
    """
    import logging
    import warnings

    stdin = sys.stdin.buffer
    header = json.loads(stdin.readline())
    name = str(header.get("name") or "pdf")
    _apply_rlimits(
        float(header.get("cpu_seconds") or 0),
        int(header.get("memory_bytes") or 0),
        int(header.get("output_bytes") or 0),
    )
    threading.Thread(
        target=_watchdog,
        args=(
            float(header.get("cpu_seconds") or 0),
            int(header.get("memory_bytes") or 0),
            name,
            _peak_rss_bytes() or 0,
            int(header.get("parent_pid") or 0),
        ),
        daemon=True,
    ).start()

    # Import pypdf exactly as the parent would.
    sys.path[:] = [str(entry) for entry in header.get("path") or sys.path]
    warnings.simplefilter("ignore")
    reported: list[str] = []
    warning_count = 0

    class _Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            nonlocal warning_count
            warning_count += 1
            if len(reported) < _MAX_REPORTED_WARNINGS:
                reported.append(record.getMessage()[:300])

    root = logging.getLogger()
    root.handlers[:] = [_Collect(level=logging.WARNING)]

    out = sys.stdout.buffer

    def emit(text: str) -> None:
        out.write(text.encode("utf-8", "surrogatepass"))

    try:
        data = stdin.read()
        extract(data, header.get("password"), header.get("limits") or {}, emit, name)
        out.flush()
    except PdfRefusal as exc:
        _finish(3, {"kind": exc.kind, "message": str(exc)})
    except MemoryError:
        _finish(
            3,
            {
                "kind": KIND_BUDGET,
                "message": f"Refusing {name}: extraction ran out of memory "
                "(env PDF_MAX_MEMORY_MB)",
            },
        )
    except Exception as exc:
        _finish(
            3,
            {
                "kind": KIND_ERROR,
                "message": f"{type(exc).__name__}: {exc}"[:2000],
            },
        )
    _finish(0, {"kind": "ok", "warnings": warning_count, "first_warnings": reported})


if __name__ == "__main__":
    main()
