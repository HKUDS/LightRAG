"""Pin the token sample passed to the reproduction query generator."""

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.offline


class WordTokenizer:
    def tokenize(self, text):
        return text.split()

    def convert_tokens_to_string(self, tokens):
        return " ".join(tokens)


@pytest.mark.parametrize(
    "count,budget",
    [(5000, 2000), (10000, 1000), (5000, 1999)],
)
def test_summary_retains_both_existing_offset_windows(count, budget):
    source = Path(__file__).resolve().parents[2] / "reproduce" / "Step_2.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if getattr(n, "name", None) == "get_summary")
    namespace = {"tokenizer": WordTokenizer()}
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    tokens = [str(i) for i in range(count)]
    actual = namespace["get_summary"](" ".join(tokens), budget).split()
    half = budget // 2
    assert actual == tokens[1000 : 1000 + half] + tokens[-(1000 + half) : -1000]
    assert len(actual) <= budget
