"""Shared helpers for interactive setup tests."""

from __future__ import annotations

import functools
import os
import re
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


def _bash_version_major(candidate: str) -> int | None:
    result = subprocess.run(
        [candidate, "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    first_line = result.stdout.splitlines()[0] if result.stdout else ""
    match = re.search(r"version ([0-9]+)", first_line)
    return int(match.group(1)) if match else None


@functools.cache
def bash_bin() -> str:
    """Path to a bash >= 4 interpreter, resolved once per process.

    ``scripts/setup/setup.sh`` uses associative arrays (``declare -A``), which
    need bash 4+. macOS still ships bash 3.2 as ``/bin/bash``, so a plain
    ``"bash"`` picks an interpreter that cannot source the wizard at all.

    Deliberately lazy (and cached): resolution shells out to ``bash --version``
    once per candidate, and a module-level constant made every test session that
    merely imports this helper pay those forks — including sessions that run no
    setup test. Override with ``LIGHTRAG_TEST_BASH``.
    """
    candidates = [
        os.environ.get("LIGHTRAG_TEST_BASH"),
        "/opt/homebrew/bin/bash",
        "/usr/local/bin/bash",
        shutil.which("bash"),
        "/bin/bash",
    ]
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        candidate_path = Path(candidate)
        if candidate_path.is_absolute() and not candidate_path.exists():
            continue
        major = _bash_version_major(candidate)
        if major is not None and major >= 4:
            return candidate
    # No bash 4+ found. Return the best available so the caller fails with the
    # script's own error rather than a confusing FileNotFoundError.
    return shutil.which("bash") or "bash"


def run_bash_process(
    script: str, cwd: Path | None = None, stdin: str | None = ""
) -> subprocess.CompletedProcess[str]:
    """Run a bash snippet and return the completed process."""
    return subprocess.run(
        [bash_bin(), "--norc", "--noprofile", "-c", script],
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
