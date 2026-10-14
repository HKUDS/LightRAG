from uuid import uuid4

import pytest

from lightrag.exceptions import PipelineNotInitializedError
from lightrag.kg import shared_storage
from lightrag.tools.check_initialization import check_lightrag_setup
from tests.pipeline.test_delete_fail_closed import _build_rag


pytestmark = pytest.mark.offline


@pytest.mark.parametrize(
    "failure, expected_message",
    [
        (None, "LightRAG is properly initialized"),
        (
            PipelineNotInitializedError("pipeline_status"),
            "Pipeline status not initialized",
        ),
        (RuntimeError("namespace unavailable"), "namespace unavailable"),
    ],
)
async def test_diagnostic_reports_pipeline_probe_failure(
    tmp_path, monkeypatch, capsys, failure, expected_message
):
    rag = await _build_rag(tmp_path, f"initialization-check-{uuid4().hex}")
    try:
        if failure is not None:

            async def unavailable_namespace(*args, **kwargs):
                raise failure

            monkeypatch.setattr(
                shared_storage, "get_namespace_data", unavailable_namespace
            )

        ready = await check_lightrag_setup(rag)
        assert ready is (failure is None)
        assert expected_message in capsys.readouterr().out
    finally:
        await rag.finalize_storages()
