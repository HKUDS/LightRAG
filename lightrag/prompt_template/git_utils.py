from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .errors import PromptTemplateGitError


@dataclass(frozen=True)
class GitCommitResult:
    committed: bool
    head: Optional[str]


def _run_git(repo_dir: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", "-C", str(repo_dir), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as e:
        raise PromptTemplateGitError("git executable not found in PATH") from e
    except subprocess.CalledProcessError as e:
        msg = (e.stderr or e.stdout or "").strip()
        raise PromptTemplateGitError(f"git {' '.join(args)} failed: {msg}") from e


def is_git_repo(repo_dir: Path) -> bool:
    if not (repo_dir / ".git").exists():
        return False
    try:
        proc = _run_git(repo_dir, ["rev-parse", "--is-inside-work-tree"])
        return proc.stdout.strip() == "true"
    except PromptTemplateGitError:
        return False


def ensure_git_repo(repo_dir: Path) -> None:
    repo_dir.mkdir(parents=True, exist_ok=True)
    if is_git_repo(repo_dir):
        return
    _run_git(repo_dir, ["init"])


def has_commits(repo_dir: Path) -> bool:
    try:
        _run_git(repo_dir, ["rev-parse", "--verify", "HEAD"])
        return True
    except PromptTemplateGitError:
        return False


def repo_head(repo_dir: Path) -> Optional[str]:
    try:
        proc = _run_git(repo_dir, ["rev-parse", "HEAD"])
        return proc.stdout.strip() or None
    except PromptTemplateGitError:
        return None


def commit_all(
    repo_dir: Path,
    message: str,
    *,
    author: str = "lightrag",
) -> GitCommitResult:
    # Stage everything first.
    _run_git(repo_dir, ["add", "."])

    # No-op if nothing to commit.
    status = _run_git(repo_dir, ["status", "--porcelain"]).stdout.strip()
    if not status:
        return GitCommitResult(committed=False, head=repo_head(repo_dir))

    # Avoid relying on global git config.
    author_email = f"{author}@lightrag.local"
    _run_git(
        repo_dir,
        [
            "-c",
            f"user.name={author}",
            "-c",
            f"user.email={author_email}",
            "commit",
            "-m",
            message,
        ],
    )
    return GitCommitResult(committed=True, head=repo_head(repo_dir))
