"""Shared helpers for interactive setup tests."""

from __future__ import annotations

import re
import os
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PRESERVED_HEADER = (
    "### ----- Preserved custom environment variables from previous .env  -----"
)
PRESERVED_NOTICE = (
    "### ----- Comments in this session will persist across regenerations -----"
)


def _bash_major_version(candidate: str) -> int:
    try:
        result = subprocess.run(
            [candidate, "-c", 'printf "%s" "${BASH_VERSINFO[0]}"'],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return 0
    if result.returncode != 0 or not result.stdout.strip().isdigit():
        return 0
    return int(result.stdout.strip())


def _resolve_bash() -> str:
    candidates = [
        "/opt/homebrew/bin/bash",
        "/usr/local/bin/bash",
        "/opt/local/bin/bash",
    ]
    path_bash = shutil.which("bash")
    if path_bash:
        candidates.append(path_bash)
    candidates.append("bash")

    for candidate in candidates:
        resolved = shutil.which(candidate) or candidate
        if _bash_major_version(resolved) >= 4:
            return resolved
    return shutil.which("bash") or "bash"


BASH_BIN = _resolve_bash()

if Path(BASH_BIN).is_absolute():
    os.environ["PATH"] = f"{Path(BASH_BIN).parent}{os.pathsep}{os.environ.get('PATH', '')}"


def run_bash_process(
    script: str, cwd: Path | None = None, stdin: str | None = ""
) -> subprocess.CompletedProcess[str]:
    """Run a bash snippet and return the completed process."""
    return subprocess.run(
        [BASH_BIN, "--norc", "--noprofile", "-c", script],
        cwd=cwd or REPO_ROOT,
        input=stdin,
        capture_output=True,
        text=True,
        check=False,
    )


def run_bash(script: str, cwd: Path | None = None) -> str:
    """Run a bash snippet and return stdout."""
    result = run_bash_process(script, cwd=cwd)
    if result.returncode != 0:
        raise AssertionError(f"""bash script failed with code {result.returncode}
stdout:
{result.stdout}
stderr:
{result.stderr}""")
    return result.stdout


def parse_lines(output: str) -> dict[str, str]:
    """Parse KEY=value lines into a dictionary."""
    values: dict[str, str] = {}
    for line in output.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def run_bash_lines(script: str, cwd: Path | None = None) -> dict[str, str]:
    """Run a bash snippet and parse KEY=value lines from stdout."""
    return parse_lines(run_bash(script, cwd=cwd))


def write_text_lines(path: Path, lines: list[str]) -> Path:
    """Write lines to a fixture file with a trailing newline."""
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def assert_single_compose_backup(tmp_path: Path, expected_content: str) -> Path:
    """Assert that a single compose backup exists with the expected content."""
    backups = sorted(tmp_path.glob("docker-compose.backup*.yml"))
    assert len(backups) == 1
    assert re.fullmatch("docker-compose\\.backup\\d{8}_\\d{6}\\.yml", backups[0].name)
    assert backups[0].read_text(encoding="utf-8") == expected_content
    return backups[0]


def write_storage_setup_files(
    tmp_path: Path, env_lines: list[str], compose_lines: list[str]
) -> None:
    """Write the minimal env/example/compose fixtures used by storage-flow tests."""
    write_text_lines(tmp_path / ".env", env_lines)
    write_text_lines(
        tmp_path / "env.example",
        (REPO_ROOT / "env.example").read_text(encoding="utf-8").splitlines(),
    )
    write_text_lines(tmp_path / "docker-compose.final.yml", compose_lines)
