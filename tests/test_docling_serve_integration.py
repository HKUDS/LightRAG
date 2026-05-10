"""Real-container integration test for the docling-serve fix in PR #3045.

The offline tests in ``test_protocol_parse_service.py`` mock the docling-serve
contract per the upstream spec (https://docling-project.github.io/docling-serve/usage/).
This file goes one step further: it spins up an actual ``docling-serve`` Docker
container, pushes a freshly-generated PDF through the full
``upload → poll → result`` cycle, and asserts that ``parse_docling`` lands the
extracted Markdown — not the raw JSON wrapper — as the document body.

Skipped by default. Activate with::

    pytest tests/test_docling_serve_integration.py --run-integration

Skips cleanly (rather than failing) when:
  * Docker is unavailable or the daemon is unreachable.
  * The ``quay.io/docling-project/docling-serve-cpu:latest`` image cannot be
    pulled or already-pulled within ``IMAGE_PULL_TIMEOUT``.
  * ``fpdf2`` is missing (used to generate a parseable PDF on the fly).
  * The container fails to become ready within ``CONTAINER_READY_TIMEOUT``.

Tunables (env vars, with sensible defaults):
  * ``LIGHTRAG_DOCLING_IMAGE``    — image ref. Default ``docling-serve-cpu:latest``.
  * ``LIGHTRAG_DOCLING_PULL_S``   — image pull deadline. Default 600s.
  * ``LIGHTRAG_DOCLING_READY_S``  — readiness deadline. Default 300s.
  * ``LIGHTRAG_DOCLING_PARSE_S``  — full parse deadline. Default 600s.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc, Tokenizer


pytestmark = pytest.mark.integration


# --------------------------------------------------------------------------
# Skip gates
# --------------------------------------------------------------------------


DEFAULT_IMAGE = os.getenv(
    "LIGHTRAG_DOCLING_IMAGE", "quay.io/docling-project/docling-serve-cpu:latest"
)
IMAGE_PULL_TIMEOUT = float(os.getenv("LIGHTRAG_DOCLING_PULL_S", "600"))
CONTAINER_READY_TIMEOUT = float(os.getenv("LIGHTRAG_DOCLING_READY_S", "300"))
PARSE_TIMEOUT = float(os.getenv("LIGHTRAG_DOCLING_PARSE_S", "600"))


def _docker_available() -> bool:
    """Return True iff the docker CLI is on PATH and the daemon responds."""
    if shutil.which("docker") is None:
        return False
    try:
        result = subprocess.run(
            ["docker", "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, OSError):
        return False


def _ensure_image(image: str, deadline: float) -> bool:
    """Make ``image`` available locally. Returns False on pull failure / timeout.

    Already-cached images short-circuit the pull. Network failures are
    converted to a graceful skip rather than a test error so CI without
    egress to quay.io still passes the offline suite.
    """
    inspect = subprocess.run(
        ["docker", "image", "inspect", image],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if inspect.returncode == 0:
        return True

    try:
        pull = subprocess.run(
            ["docker", "pull", image],
            timeout=deadline,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except subprocess.TimeoutExpired:
        return False
    return pull.returncode == 0


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# --------------------------------------------------------------------------
# Container lifecycle
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def docling_container() -> Iterator[dict]:
    """Start ``docling-serve-cpu``, yield its base URL, and clean up on exit.

    Skips the test (rather than failing) when docker is unavailable or the
    image cannot be obtained — network outages and missing daemons are
    environmental, not regressions in the code under test.
    """

    if not _docker_available():
        pytest.skip("docker daemon not reachable — skipping real-container test")

    if not _ensure_image(DEFAULT_IMAGE, IMAGE_PULL_TIMEOUT):
        pytest.skip(
            f"unable to pull {DEFAULT_IMAGE} within {IMAGE_PULL_TIMEOUT:.0f}s "
            "(no network or registry unavailable)"
        )

    httpx = pytest.importorskip("httpx")

    name = f"lightrag-docling-it-{int(time.time())}"
    port = _free_port()
    run_args = [
        "docker",
        "run",
        "-d",
        "--rm",
        "--name",
        name,
        "-p",
        f"{port}:5001",
        DEFAULT_IMAGE,
    ]
    started = subprocess.run(
        run_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if started.returncode != 0:
        pytest.skip(f"failed to start docling-serve container: {started.stderr}")

    container_id = started.stdout.strip()
    base = f"http://127.0.0.1:{port}"

    try:
        deadline = time.time() + CONTAINER_READY_TIMEOUT
        while time.time() < deadline:
            try:
                # docling-serve exposes /docs (FastAPI's Swagger UI) once the
                # ASGI app is ready; this is the cheapest readiness probe.
                resp = httpx.get(f"{base}/docs", timeout=2.0)
                if resp.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(2.0)
        else:
            logs = subprocess.run(
                ["docker", "logs", "--tail", "60", container_id],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            pytest.skip(
                "docling-serve container failed readiness within "
                f"{CONTAINER_READY_TIMEOUT:.0f}s. Last logs:\n{logs.stdout}"
            )

        yield {"base": base, "container_id": container_id}

    finally:
        subprocess.run(
            ["docker", "rm", "-f", container_id],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


# --------------------------------------------------------------------------
# PDF generation (skip if fpdf2 missing)
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sample_pdf(tmp_path_factory) -> Path:
    fpdf = pytest.importorskip("fpdf")
    out = tmp_path_factory.mktemp("docling-it") / "sample.pdf"
    pdf = fpdf.FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=14)
    pdf.cell(0, 10, "Hello LightRAG", ln=True)
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(
        0,
        7,
        "This document is generated for the docling-serve integration test "
        "in PR #3045. Its body must round-trip through the upload, poll, and "
        "result endpoints, and the markdown returned by docling-serve must "
        "be persisted by parse_docling instead of the raw JSON wrapper.",
    )
    pdf.output(str(out))
    assert out.stat().st_size > 0
    return out


# --------------------------------------------------------------------------
# LightRAG fixture (mirrors the offline tests)
# --------------------------------------------------------------------------


async def _embed(texts: list[str]) -> np.ndarray:
    return np.zeros((len(texts), 8), dtype=float)


async def _llm(prompt: str, **_) -> str:
    return ""


class _Tok:
    def encode(self, content: str) -> list[int]:
        return [ord(c) for c in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


def _new_rag(tmp_path: Path) -> LightRAG:
    return LightRAG(
        working_dir=str(tmp_path),
        workspace=f"docling-it-{tmp_path.name}",
        llm_model_func=_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8, max_token_size=512, func=_embed
        ),
        tokenizer=Tokenizer("mock", _Tok()),
    )


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


def test_real_docling_serve_returns_extractable_wrapper(
    tmp_path, docling_container, sample_pdf
):
    """Sanity probe: hit the live container directly (no LightRAG in the
    loop) and verify the response shape we coded against still matches the
    runtime contract. If this assertion ever fires, docling-serve has
    diverged from its documented spec and the offline tests are stale.
    """

    httpx = pytest.importorskip("httpx")
    base = docling_container["base"]

    with httpx.Client(timeout=httpx.Timeout(PARSE_TIMEOUT, connect=30.0)) as client:
        with open(sample_pdf, "rb") as f:
            up = client.post(
                f"{base}/v1/convert/file/async",
                files={"files": (sample_pdf.name, f, "application/pdf")},
            )
        assert up.status_code < 400, up.text
        upload_payload = up.json()
        task_id = upload_payload.get("task_id")
        assert task_id, f"no task_id in upload payload: {upload_payload}"

        deadline = time.time() + PARSE_TIMEOUT
        while time.time() < deadline:
            poll = client.get(f"{base}/v1/status/poll/{task_id}")
            poll.raise_for_status()
            status = poll.json().get("task_status")
            if status == "success":
                break
            if status == "failure":
                pytest.fail(f"docling-serve reported failure: {poll.json()}")
            time.sleep(1.0)
        else:
            pytest.fail("docling-serve poll never returned success")

        result = client.get(f"{base}/v1/result/{task_id}")
        result.raise_for_status()
        body = result.json()

    assert isinstance(body, dict), body
    assert "document" in body, body
    document = body["document"]
    assert isinstance(document, dict)
    md = document.get("md_content", "")
    assert isinstance(md, str) and md.strip(), document
    assert "Hello LightRAG" in md or "LightRAG" in md, md


def test_parse_docling_against_real_container_persists_markdown(
    tmp_path, docling_container, sample_pdf, monkeypatch
):
    """Full end-to-end: parse_docling + a real docling-serve container.
    The persisted document content must be the markdown extracted from
    ``document.md_content``, not the raw JSON wrapper. This is the bug
    the codex review on PR #3045 surfaced.
    """

    base = docling_container["base"]
    monkeypatch.setenv("DOCLING_ENDPOINT", f"{base}/v1/convert/file/async")
    monkeypatch.setenv("DOCLING_MAX_POLLS", "600")
    monkeypatch.setenv("DOCLING_POLL_INTERVAL_SECONDS", "1")
    # Default DOCLING_CONTENT_FIELD (document.md_content) — do not override.
    monkeypatch.delenv("DOCLING_CONTENT_FIELD", raising=False)

    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            result = await rag.parse_docling(
                doc_id="docling-it-1",
                file_path=str(sample_pdf),
                content_data={
                    "content": "x",
                    "source_path": str(sample_pdf),
                },
            )
        finally:
            await rag.finalize_storages()

        content = result.get("content", "")
        assert isinstance(content, str)
        # Must be Markdown extracted from document.md_content, not the JSON
        # wrapper. Anchors below are deliberately spec-aware: the wrapper
        # would contain literal ``"document"`` and ``"md_content"`` keys
        # while the extracted markdown will contain neither.
        assert "Hello LightRAG" in content or "LightRAG" in content, content
        assert '"document"' not in content
        assert '"md_content"' not in content
        assert '"task_status"' not in content

    asyncio.run(_run())
