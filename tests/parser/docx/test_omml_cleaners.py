"""Tests for ``lightrag/parser/docx/omml/cleaners.py``."""

from __future__ import annotations

import pytest

from lightrag.parser.docx.omml.cleaners import clean_exp

pytestmark = pytest.mark.offline


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            r"\lim\below{x}{y}\lim\below{a}{b}",
            r"\lim_{x}{y}\lim_{a}{b}",
        ),
        (
            r"\min\below{n}{0}\min\below{k}{1}",
            r"\min_{n}{0}\min_{k}{1}",
        ),
        (
            r"\max\below{i}{1} + \max\below{j}{2}",
            r"\max_{i}{1} + \max_{j}{2}",
        ),
        (
            r"E = \lim\below{x \to 0}{f(x)} + \lim\below{y \to 1}{g(y)}",
            r"E = \lim_{x \to 0}{f(x)} + \lim_{y \to 1}{g(y)}",
        ),
        (
            r"\lim\below{\frac{1}{2}}{b}",
            r"\lim_{\frac{1}{2}}{b}",
        ),
        (
            r"\lim\below{a}{b_{1}}",
            r"\lim_{a}{b_{1}}",
        ),
    ],
)
def test_clean_exp_converts_repeated_below_constructs(raw: str, expected: str) -> None:
    assert clean_exp(raw) == expected
