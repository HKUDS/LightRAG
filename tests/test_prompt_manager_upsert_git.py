import shutil
import subprocess
from pathlib import Path

import pytest

from lightrag.base import PromptType
from lightrag.prompt_template import GitPromptManager


def _git(repo_dir: Path, args: list[str]) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_dir), *args], text=True
    ).strip()


@pytest.mark.offline
async def test_prompt_manager_upsert_creates_git_commit(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git not available in PATH")

    pm = GitPromptManager(working_dir=str(tmp_path))
    await pm.initialize()

    before = int(_git(pm.user_repo_dir, ["rev-list", "--count", "HEAD"]))

    await pm.upsert_user_template(
        PromptType.QUERY,
        "fail_response",
        content="CUSTOM_FAIL_RESPONSE",
        author="alice",
    )

    after = int(_git(pm.user_repo_dir, ["rev-list", "--count", "HEAD"]))
    assert after == before + 1

    # Confirm audit info is present (either commit message or author metadata).
    last_author = _git(pm.user_repo_dir, ["log", "-1", "--pretty=%an"])
    assert last_author == "alice"

    last_message = _git(pm.user_repo_dir, ["log", "-1", "--pretty=%B"])
    assert "by alice" in last_message

    rendered = await pm.render(PromptType.QUERY, "fail_response")
    assert rendered.strip() == "CUSTOM_FAIL_RESPONSE"
