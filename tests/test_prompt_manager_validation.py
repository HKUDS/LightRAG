import shutil
from pathlib import Path

import pytest

from lightrag.base import PromptType
from lightrag.prompt_template import GitPromptManager
from lightrag.prompt_template.errors import PromptTemplateValidationError


@pytest.mark.offline
async def test_prompt_manager_missing_required_vars_raises(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git not available in PATH")

    pm = GitPromptManager(working_dir=str(tmp_path))
    await pm.initialize()

    # rag_response requires multiple variables; omit one to ensure a clear error.
    with pytest.raises(PromptTemplateValidationError):
        await pm.render(
            PromptType.QUERY,
            "rag_response",
            context_data="CTX",
            response_type="Multiple Paragraphs",
            # user_prompt missing
        )


@pytest.mark.offline
async def test_prompt_manager_illegal_vars_raises(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git not available in PATH")

    pm = GitPromptManager(working_dir=str(tmp_path))
    await pm.initialize()

    with pytest.raises(PromptTemplateValidationError):
        await pm.render(
            PromptType.QUERY,
            "rag_response",
            context_data="CTX",
            response_type="Multiple Paragraphs",
            user_prompt="n/a",
            not_allowed="boom",
        )
