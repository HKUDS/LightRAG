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


def load_get_summary():
    # Importing Step_2.py would download GPT-2 and call OpenAI at import time,
    # so compile only the get_summary definition against a stub tokenizer.
    source = Path(__file__).resolve().parents[2] / "reproduce" / "Step_2.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if getattr(n, "name", None) == "get_summary")
    namespace = {"tokenizer": WordTokenizer()}
    exec(  # noqa: S102 - trusted repository source, compiled for the test only
        compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return namespace["get_summary"]


@pytest.mark.parametrize(
    "count,budget,head,tail",
    [
        (5000, 2000, (1000, 2000), (3000, 4000)),
        (10000, 1000, (1000, 1500), (8500, 9000)),
        (5000, 1999, (1000, 1999), (3001, 4000)),
    ],
)
def test_summary_retains_both_existing_offset_windows(count, budget, head, tail):
    tokens = [str(i) for i in range(count)]
    actual = load_get_summary()(" ".join(tokens), budget).split()
    expected = [str(i) for i in range(*head)] + [str(i) for i in range(*tail)]
    assert actual == expected
    assert len(actual) <= budget
