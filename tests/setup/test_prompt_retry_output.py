"""Validation messages must not become the wizard's captured configuration."""

import shlex

import pytest

from tests.setup._helpers import REPO_ROOT, run_bash_process


@pytest.mark.parametrize(
    ("invocation", "stdin", "expected"),
    [
        ('prompt_until_valid "Port" "9621" validate_port', "invalid\n9622\n", "9622"),
        (
            'prompt_secret_until_valid "Key" validate_non_empty',
            "\nvalid-key\n",
            "valid-key",
        ),
        (
            'prompt_secret_until_valid_with_default "Key" "" validate_non_empty',
            "\nvalid-key\n",
            "valid-key",
        ),
        ('prompt_required_secret "Key"', "\nvalid-key\n", "valid-key"),
    ],
)
def test_retry_diagnostics_stay_out_of_captured_value(invocation, stdin, expected):
    prompts = (REPO_ROOT / "scripts/setup/lib/prompts.sh").as_posix()
    validation = (REPO_ROOT / "scripts/setup/lib/validation.sh").as_posix()
    result = run_bash_process(
        f'''source "{prompts}"
source "{validation}"
value=$(printf '%s' {shlex.quote(stdin)} | {invocation})
printf '%s' "$value"
''',
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == expected
    assert "Please try again." in result.stderr
