"""A cancelled deletion must not be turned into a DeletionResult.

``adelete_by_doc_id`` flushes storages in its ``finally`` block. When that
flush failed while a ``BaseException`` (e.g. ``asyncio.CancelledError``) was
propagating, the ``finally`` used to ``return`` a failure result, which
silently swallowed the cancellation (and is the ``return``-in-``finally``
pattern that Python 3.14 flags with a SyntaxWarning, PEP 765).
"""

from __future__ import annotations

import asyncio
import warnings
from pathlib import Path
from uuid import uuid4

import pytest

import lightrag.lightrag as lightrag_module

from .test_delete_fail_closed import _build_rag, _ingest

pytestmark = pytest.mark.offline


@pytest.mark.asyncio
async def test_cancel_during_delete_propagates_when_flush_fails(tmp_path):
    rag = await _build_rag(tmp_path, f"dcs-{uuid4().hex[:8]}")
    try:
        doc_id = await _ingest(rag)

        original_insert_done = rag._insert_done
        cancelled = False

        async def cancelled_delete(ids):
            nonlocal cancelled
            cancelled = True
            raise asyncio.CancelledError()

        async def insert_done_failing_after_cancel(*args, **kwargs):
            # Earlier stages flush too; only the finally's flush must fail.
            if cancelled:
                raise RuntimeError("flush boom")
            return await original_insert_done(*args, **kwargs)

        rag.doc_status.delete = cancelled_delete
        rag._insert_done = insert_done_failing_after_cancel

        with pytest.raises(asyncio.CancelledError):
            await rag.adelete_by_doc_id(doc_id)
    finally:
        await rag.finalize_storages()


def test_lightrag_module_has_no_return_in_finally():
    source = Path(lightrag_module.__file__).read_text(encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("error", SyntaxWarning)
        compile(source, lightrag_module.__file__, "exec")
