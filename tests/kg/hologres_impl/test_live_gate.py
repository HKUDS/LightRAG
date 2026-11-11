import os
from pathlib import Path
import subprocess
import sys
import tempfile


def _run_gate_source(
    source, *extra_args, prefix="test_hologres_gate_sample_"
):
    test_directory = Path(__file__).parent
    project_root = test_directory.parents[2]
    environment = os.environ.copy()
    environment.pop("PYTEST_ADDOPTS", None)
    path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=prefix,
            suffix=".py",
            dir=test_directory,
            delete=False,
        ) as handle:
            handle.write(source)
            path = Path(handle.name)
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                str(path),
                "-p",
                "no:cacheprovider",
                *extra_args,
            ],
            cwd=project_root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        if path is not None:
            path.unlink(missing_ok=True)


def _run_gate_sample(*extra_args):
    return _run_gate_source(
        """
import pytest

@pytest.mark.integration
@pytest.mark.hologres_live
def test_hologres_live_sample():
    pass

@pytest.mark.integration
def test_other_integration_sample():
    pass
""",
        *extra_args,
    )


def test_hologres_live_tests_are_skipped_by_default():
    result = _run_gate_sample()

    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 skipped" in result.stdout


def test_hologres_flag_enables_only_hologres_live_tests():
    result = _run_gate_sample("--run-hologres-live")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed, 1 skipped" in result.stdout


def test_generic_integration_flag_does_not_enable_hologres_live_tests():
    result = _run_gate_sample("--run-integration")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed, 1 skipped" in result.stdout


def test_hologres_live_marker_without_integration_marker_is_collection_error():
    result = _run_gate_source(
        """
import pytest

@pytest.mark.hologres_live
def test_hologres_live_sample():
    pass
""",
        prefix="test_hologres_live_contract_",
    )

    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert "integration and hologres_live markers" in output


def test_hologres_live_test_without_hologres_marker_is_collection_error():
    result = _run_gate_source(
        """
import pytest

@pytest.mark.integration
def test_hologres_live_sample():
    pass
""",
        prefix="test_hologres_live_contract_",
    )

    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert "integration and hologres_live markers" in output
