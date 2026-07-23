"""Empty VLM image budget envs must not crash multimodal analyze setup."""

from __future__ import annotations

import pytest

from lightrag.constants import DEFAULT_MM_IMAGE_MIN_PIXEL
from lightrag.pipeline import _vlm_image_budget_limits


@pytest.mark.offline
@pytest.mark.parametrize("env_value", ["", "  "])
def test_empty_vlm_image_envs_fall_back(
    env_value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VLM_MAX_IMAGE_BYTES", env_value)
    monkeypatch.setenv("VLM_MIN_IMAGE_PIXEL", env_value)
    max_b, min_p = _vlm_image_budget_limits()
    assert max_b == 5 * 1024 * 1024
    assert min_p == DEFAULT_MM_IMAGE_MIN_PIXEL


@pytest.mark.offline
def test_valid_vlm_image_envs_are_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLM_MAX_IMAGE_BYTES", str(1024 * 1024))
    monkeypatch.setenv("VLM_MIN_IMAGE_PIXEL", "128")
    max_b, min_p = _vlm_image_budget_limits()
    assert max_b == 1024 * 1024
    assert min_p == 128
