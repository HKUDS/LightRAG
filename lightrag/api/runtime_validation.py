"""Helpers for validating startup runtime expectations from `.env`."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

_CONTAINER_RUNTIME_TARGETS = {"compose", "docker"}


@dataclass(frozen=True)
class RuntimeEnvironment:
    """Describes whether the current process is running in a container runtime."""

    in_container: bool
    in_docker: bool
    in_kubernetes: bool

    @property
    def label(self) -> str:
        if self.in_kubernetes:
            return "Kubernetes"
        if self.in_docker:
            return "Docker"
        return "host"


def _read_cgroup_content() -> str:
    """Best-effort read of cgroup metadata for container detection."""

    for candidate in ("/proc/1/cgroup", "/proc/self/cgroup"):
        try:
            return Path(candidate).read_text(encoding="utf-8")
        except OSError:
            continue
    return ""


def detect_runtime_environment(
    environ: dict[str, str] | None = None,
) -> RuntimeEnvironment:
    """Detect whether the current process is running on host, Docker, or Kubernetes."""

    environ = environ or os.environ
    cgroup_content = _read_cgroup_content().lower()

    in_kubernetes = bool(
        environ.get("KUBERNETES_SERVICE_HOST")
        or Path("/var/run/secrets/kubernetes.io/serviceaccount").exists()
        or "kubepods" in cgroup_content
        or "kubernetes" in cgroup_content
    )
    in_docker = bool(
        Path("/.dockerenv").exists()
        or Path("/run/.containerenv").exists()
        or any(
            marker in cgroup_content
            for marker in ("docker", "containerd", "libpod", "podman")
        )
    )

    return RuntimeEnvironment(
        in_container=in_kubernetes or in_docker,
        in_docker=in_docker,
        in_kubernetes=in_kubernetes,
    )


def load_runtime_target_from_env_file(env_path: str | Path = ".env") -> str | None:
    """Return the raw LIGHTRAG_RUNTIME_TARGET value from the `.env` file, if present."""

    env_values = dotenv_values(str(env_path))
    runtime_target = env_values.get("LIGHTRAG_RUNTIME_TARGET")
    if runtime_target is None:
        return None
    return runtime_target.strip()


def validate_runtime_target(
    runtime_target: str | None,
    runtime_environment: RuntimeEnvironment | None = None,
) -> tuple[bool, str | None]:
    """Validate `.env` runtime target against the current runtime environment."""

    if runtime_target is None:
        return True, None

    normalized_target = runtime_target.strip().lower()
    runtime_environment = runtime_environment or detect_runtime_environment()

    if normalized_target == "host":
        if runtime_environment.in_container:
            return (
                False,
                "Configuration error in .env: LIGHTRAG_RUNTIME_TARGET=host.\n"
                "This .env requires the server process to run on the host, "
                f"but the current process is running inside {runtime_environment.label}.",
            )
        return True, None

    if normalized_target in _CONTAINER_RUNTIME_TARGETS:
        if runtime_environment.in_container:
            return True, None
        return (
            False,
            f"Configuration error in .env: LIGHTRAG_RUNTIME_TARGET={runtime_target}.\n"
            "This .env requires the server process to run inside Docker or "
            "Kubernetes, but the current process is running on the host.",
        )

    return (
        False,
        f"Configuration error in .env: LIGHTRAG_RUNTIME_TARGET={runtime_target!r}.\n"
        "This value from .env must be 'host' or 'compose' (alias: 'docker').",
    )


def validate_runtime_target_from_env_file(
    env_path: str | Path = ".env",
    runtime_environment: RuntimeEnvironment | None = None,
) -> tuple[bool, str | None]:
    """Load LIGHTRAG_RUNTIME_TARGET from `.env` and validate it if declared."""

    runtime_target = load_runtime_target_from_env_file(env_path)
    return validate_runtime_target(runtime_target, runtime_environment)
