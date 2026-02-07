import shutil
import subprocess
from pathlib import Path

import pytest

from lightrag.base import PromptOrigin, PromptType
from lightrag.prompt_template import GitPromptManager


def _git(repo_dir: Path, args: list[str]) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_dir), *args], text=True
    ).strip()


@pytest.mark.offline
async def test_prompt_manager_initialize_seeds_git_repo(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git not available in PATH")

    pm = GitPromptManager(working_dir=str(tmp_path))
    await pm.initialize()

    assert (pm.user_repo_dir / ".git").exists()

    system_templates = pm.list_templates(origin=PromptOrigin.SYSTEM, resolved=False)
    user_templates = pm.list_templates(origin=PromptOrigin.USER, resolved=False)
    assert len(system_templates) >= 1
    assert len(user_templates) >= len(system_templates)

    commit_count = int(_git(pm.user_repo_dir, ["rev-list", "--count", "HEAD"]))
    assert commit_count == 1

    # Spot-check a known default template can be rendered.
    delimiter = await pm.render(PromptType.KG, "default_tuple_delimiter")
    assert delimiter.strip()
