from pathlib import Path

from lightrag.api.runtime_validation import (
    RuntimeEnvironment,
    validate_runtime_target,
    validate_runtime_target_from_env_file,
)


def test_validate_runtime_target_skips_when_not_declared() -> None:
    is_valid, error_message = validate_runtime_target(None)

    assert is_valid is True
    assert error_message is None


def test_validate_runtime_target_accepts_host_on_host() -> None:
    is_valid, error_message = validate_runtime_target(
        "host",
        RuntimeEnvironment(
            in_container=False,
            in_docker=False,
            in_kubernetes=False,
        ),
    )

    assert is_valid is True
    assert error_message is None


def test_validate_runtime_target_rejects_host_in_container() -> None:
    is_valid, error_message = validate_runtime_target(
        "host",
        RuntimeEnvironment(
            in_container=True,
            in_docker=True,
            in_kubernetes=False,
        ),
    )

    assert is_valid is False
    assert "\n" in error_message
    assert "Configuration error in .env" in error_message
    assert "LIGHTRAG_RUNTIME_TARGET=host" in error_message
    assert "This value from .env" in error_message
    assert "Docker" in error_message


def test_validate_runtime_target_accepts_compose_and_docker_in_container() -> None:
    runtime_environment = RuntimeEnvironment(
        in_container=True,
        in_docker=False,
        in_kubernetes=True,
    )

    for runtime_target in ("compose", "docker"):
        is_valid, error_message = validate_runtime_target(
            runtime_target,
            runtime_environment,
        )

        assert is_valid is True
        assert error_message is None


def test_validate_runtime_target_rejects_container_target_on_host() -> None:
    is_valid, error_message = validate_runtime_target(
        "docker",
        RuntimeEnvironment(
            in_container=False,
            in_docker=False,
            in_kubernetes=False,
        ),
    )

    assert is_valid is False
    assert "\n" in error_message
    assert "Configuration error in .env" in error_message
    assert "LIGHTRAG_RUNTIME_TARGET=docker" in error_message
    assert "This value from .env" in error_message
    assert "Docker or Kubernetes" in error_message


def test_validate_runtime_target_rejects_invalid_value() -> None:
    is_valid, error_message = validate_runtime_target(
        "invalid",
        RuntimeEnvironment(in_container=False, in_docker=False, in_kubernetes=False),
    )

    assert is_valid is False
    assert "\n" in error_message
    assert "Configuration error in .env" in error_message
    assert "must be 'host' or 'compose'" in error_message


def test_validate_runtime_target_from_env_file_uses_raw_env_value(
    tmp_path: Path,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("LIGHTRAG_RUNTIME_TARGET=compose\n", encoding="utf-8")

    is_valid, error_message = validate_runtime_target_from_env_file(
        env_file,
        RuntimeEnvironment(
            in_container=True,
            in_docker=True,
            in_kubernetes=False,
        ),
    )

    assert is_valid is True
    assert error_message is None
