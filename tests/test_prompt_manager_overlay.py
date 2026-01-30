import shutil
from pathlib import Path

import pytest

from lightrag.base import PromptOrigin, PromptType
from lightrag.prompt_template import GitPromptManager


@pytest.mark.offline
async def test_prompt_manager_falls_back_to_system_when_user_template_missing(
    tmp_path: Path,
) -> None:
    if shutil.which("git") is None:
        pytest.skip("git not available in PATH")

    pm = GitPromptManager(working_dir=str(tmp_path))
    await pm.initialize()

    # The repo is seeded with system templates, so the resolved origin is USER.
    tmpl1 = await pm.get_template(PromptType.KG, "default_tuple_delimiter")
    assert tmpl1 is not None
    assert tmpl1.origin == PromptOrigin.USER

    # Remove the user override file and reload. The system template should be used.
    user_file = pm.user_repo_dir / "kg-default_tuple_delimiter.md.j2"
    assert user_file.exists()
    user_file.unlink()

    pm.reload_templates()

    tmpl2 = await pm.get_template(PromptType.KG, "default_tuple_delimiter")
    assert tmpl2 is not None
    assert tmpl2.origin == PromptOrigin.SYSTEM
