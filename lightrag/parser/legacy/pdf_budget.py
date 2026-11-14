"""Resource-limited PDF text extraction for the legacy engine.

pypdf limits each object it reads, not the document, and repeats work for
every page and every form XObject invocation. A PDF of a few kilobytes can
therefore cost hours of CPU, and ``MAX_UPLOAD_SIZE`` bounds only the file.
What pypdf will do with a given object cannot be predicted reliably from
outside, so the extraction runs in a child process that is stopped when it
has used its budget.

Rules for callers:

* Extract PDF text only through :func:`extract_pdf_text`. Calling pypdf
  directly in the server process has no CPU or memory bound.
* A refusal is a :class:`PdfExtractionBudgetError`. Do not catch it to keep
  partial text; the document must be recorded FAILED. Every other failure is a
  ``LegacyExtractionError`` carrying pypdf's own message.
* Limits come from the environment at call time: ``PDF_MAX_CPU_SECONDS``,
  ``PDF_MAX_MEMORY_MB``, ``PDF_MAX_CONTENT_BYTES``, ``PDF_MAX_CONTENT_RATIO``
  and ``PDF_MAX_EXTRACTED_CHARS``. A non-positive value disables that limit.
* ``PDF_EXTRACT_IN_SUBPROCESS=false`` runs pypdf in-process. Only the content
  and character limits apply then. It exists for environments that cannot
  start a Python child process, not as a performance option.

Design, protocol and accepted limits: docs/design/PdfExtractionBudget.md.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from lightrag.constants import (
    DEFAULT_PDF_CONTENT_RATIO_FLOOR_BYTES,
    DEFAULT_PDF_MAX_CONTENT_BYTES,
    DEFAULT_PDF_MAX_CONTENT_RATIO,
    DEFAULT_PDF_MAX_CPU_SECONDS,
    DEFAULT_PDF_MAX_EXTRACTED_CHARS,
    DEFAULT_PDF_MAX_MEMORY_MB,
)
from lightrag.parser.legacy import pdf_worker
from lightrag.parser.legacy.extractors import LegacyExtractionError
from lightrag.utils import get_env_value, logger

_WORKER_PATH = str(Path(pdf_worker.__file__).resolve())
# The wall clock is the last resort, for a child that is not using CPU or
# cannot be stopped by the CPU limit (Windows has no RLIMIT_CPU). It allows
# for a host where the child gets only a fraction of a core.
_WALL_CLOCK_FACTOR = 4
_WALL_CLOCK_SLACK_SECONDS = 30
_MAX_STATUS_BYTES = 64 * 1024
_OUTPUT_SLACK_BYTES = 1024 * 1024


class PdfExtractionBudgetError(LegacyExtractionError):
    """A PDF needs more CPU, memory, parsing or text than it is allowed.

    A ``LegacyExtractionError`` (thus a ``ValueError``), so the pipeline
    records the document FAILED with this message.
    """


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


def _raise_refusal(kind: str, message: str) -> None:
    if kind == pdf_worker.KIND_BUDGET:
        raise PdfExtractionBudgetError(message)
    raise LegacyExtractionError(message)


def extract_pdf_text(
    file_bytes: bytes, password: str | None = None, file_path: str = "pdf"
) -> str:
    """Return the text of every page, each followed by a newline."""
    content_cap, content_env = _content_limit(len(file_bytes))
    max_chars = get_env_value(
        "PDF_MAX_EXTRACTED_CHARS", DEFAULT_PDF_MAX_EXTRACTED_CHARS, int
    )
    limits = {
        "content_cap": content_cap,
        "content_env": content_env,
        "max_chars": max_chars,
    }
    if not get_env_value("PDF_EXTRACT_IN_SUBPROCESS", True, bool):
        return _extract_in_process(file_bytes, password, limits, file_path)

    cpu_seconds = get_env_value(
        "PDF_MAX_CPU_SECONDS", DEFAULT_PDF_MAX_CPU_SECONDS, float
    )
    memory_mb = get_env_value("PDF_MAX_MEMORY_MB", DEFAULT_PDF_MAX_MEMORY_MB, int)
    header = {
        "path": [str(entry) for entry in sys.path],
        "name": file_path,
        "password": password,
        "limits": limits,
        "cpu_seconds": max(0.0, cpu_seconds),
        "memory_bytes": max(0, memory_mb) * 1024 * 1024,
        # UTF-8 needs at most 4 bytes per character.
        "output_bytes": max_chars * 4 + _OUTPUT_SLACK_BYTES if max_chars > 0 else 0,
    }
    wall_seconds = (
        cpu_seconds * _WALL_CLOCK_FACTOR + _WALL_CLOCK_SLACK_SECONDS
        if cpu_seconds > 0
        else None
    )
    return _run_worker(file_bytes, header, wall_seconds, file_path)


def _extract_in_process(
    file_bytes: bytes, password: str | None, limits: dict[str, Any], file_path: str
) -> str:
    parts: list[str] = []
    try:
        pdf_worker.extract(file_bytes, password, limits, parts.append, file_path)
    except pdf_worker.PdfRefusal as exc:
        _raise_refusal(exc.kind, str(exc))
    except MemoryError:
        raise
    except Exception as exc:
        raise LegacyExtractionError(f"{type(exc).__name__}: {exc}") from exc
    return "".join(parts)


def _run_worker(
    file_bytes: bytes,
    header: dict[str, Any],
    wall_seconds: float | None,
    file_path: str,
) -> str:
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as status:
        try:
            # -P keeps this directory (which has a parser.py) off sys.path.
            proc = subprocess.Popen(
                [sys.executable, "-P", _WORKER_PATH],
                stdin=subprocess.PIPE,
                stdout=out,
                stderr=status,
            )
        except OSError as exc:
            raise LegacyExtractionError(
                f"Cannot start the PDF extraction process for {file_path} "
                f"({exc}). If this environment cannot run a Python child "
                "process, set PDF_EXTRACT_IN_SUBPROCESS=false; pypdf then runs "
                "in-process without a CPU or memory limit."
            ) from exc
        timed_out = False
        try:
            assert proc.stdin is not None
            try:
                proc.stdin.write(json.dumps(header).encode("ascii") + b"\n")
            except OSError:
                pass  # the child died already; its status says why
            proc.communicate(file_bytes, timeout=wall_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.communicate()

        if timed_out:
            raise PdfExtractionBudgetError(
                f"Refusing {file_path}: extraction did not finish within "
                f"{wall_seconds:g} seconds of wall-clock time "
                "(derived from PDF_MAX_CPU_SECONDS)"
            )
        result = _read_status(status)
        kind = result.get("kind")
        if kind == "ok" and proc.returncode == 0:
            _log_warnings(result, file_path)
            out.seek(0)
            return out.read().decode("utf-8", "surrogatepass")
        if kind and kind != "ok":
            _raise_refusal(kind, str(result.get("message") or "PDF extraction failed"))
        if proc.returncode < 0:
            raise PdfExtractionBudgetError(
                f"Refusing {file_path}: the extraction process was stopped by "
                f"signal {-proc.returncode}, most likely by its CPU or memory "
                "limit (env PDF_MAX_CPU_SECONDS / PDF_MAX_MEMORY_MB)"
            )
        raise LegacyExtractionError(
            f"PDF extraction process failed for {file_path} "
            f"(exit code {proc.returncode}): {result.get('tail', '')}"
        )


def _read_status(status: Any) -> dict[str, Any]:
    """The worker's last status line, or ``{"tail": ...}`` if it wrote none."""
    size = status.seek(0, 2)
    status.seek(max(0, size - _MAX_STATUS_BYTES))
    text = status.read().decode("utf-8", "replace")
    lines = [line for line in text.splitlines() if line.strip()]
    if lines:
        try:
            parsed = json.loads(lines[-1])
            if isinstance(parsed, dict) and "kind" in parsed:
                return parsed
        except ValueError:
            pass
    return {"tail": text[-500:].strip()}


def _log_warnings(result: dict[str, Any], file_path: str) -> None:
    count = result.get("warnings") or 0
    if count:
        logger.warning(
            "pypdf reported %d warning(s) while extracting %s; first: %s",
            count,
            file_path,
            "; ".join(result.get("first_warnings") or []),
        )


__all__ = ["PdfExtractionBudgetError", "extract_pdf_text"]
