"""Empty VLM image budget envs must not crash multimodal analyze setup."""

from __future__ import annotations

from pathlib import Path

import pytest

from lightrag.constants import DEFAULT_MM_IMAGE_MIN_PIXEL
from lightrag.utils import get_env_value

REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE = (REPO_ROOT / "lightrag" / "pipeline.py").read_text(encoding="utf-8")


@pytest.mark.offline
def test_pipeline_routes_vlm_image_envs_through_get_env_value() -> None:
    assert 'get_env_value("VLM_MAX_IMAGE_BYTES"' in PIPELINE
    assert 'get_env_value(\n                    "VLM_MIN_IMAGE_PIXEL"' in PIPELINE or (
        'get_env_value("VLM_MIN_IMAGE_PIXEL"' in PIPELINE
    )
    assert 'int(os.getenv("VLM_MAX_IMAGE_BYTES"' not in PIPELINE
    assert 'int(os.getenv("VLM_MIN_IMAGE_PIXEL"' not in PIPELINE


@pytest.mark.offline
@pytest.mark.parametrize("env_value", ["", "  "])
def test_empty_vlm_image_envs_fall_back(
    env_value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VLM_MAX_IMAGE_BYTES", env_value)
    monkeypatch.setenv("VLM_MIN_IMAGE_PIXEL", env_value)
    max_image_bytes = max(
        256 * 1024,
        get_env_value("VLM_MAX_IMAGE_BYTES", 5 * 1024 * 1024, int),
    )
    min_image_pixel = max(
        1,
        get_env_value("VLM_MIN_IMAGE_PIXEL", DEFAULT_MM_IMAGE_MIN_PIXEL, int),
    )
    assert max_image_bytes == 5 * 1024 * 1024
    assert min_image_pixel == DEFAULT_MM_IMAGE_MIN_PIXEL


@pytest.mark.offline
def test_valid_vlm_image_envs_are_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLM_MAX_IMAGE_BYTES", str(1024 * 1024))
    monkeypatch.setenv("VLM_MIN_IMAGE_PIXEL", "128")
    max_image_bytes = max(
        256 * 1024,
        get_env_value("VLM_MAX_IMAGE_BYTES", 5 * 1024 * 1024, int),
    )
    min_image_pixel = max(
        1,
        get_env_value("VLM_MIN_IMAGE_PIXEL", DEFAULT_MM_IMAGE_MIN_PIXEL, int),
    )
    assert max_image_bytes == 1024 * 1024
    assert min_image_pixel == 128
