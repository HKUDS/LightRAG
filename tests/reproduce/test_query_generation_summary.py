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


@pytest.mark.parametrize(
    "count,budget,expected",
    [
        (0, 2000, (0, 0)),
        (300, 2000, (0, 300)),
        (1000, 2000, (0, 1000)),
        (1500, 2000, (0, 1500)),
        (3000, 2000, (0, 2000)),
        (3999, 2000, (0, 2000)),
        (2999, 1000, (0, 1000)),
    ],
)
def test_short_context_is_sampled_from_the_start(count, budget, expected):
    # Below 2 * (1000 + budget // 2) tokens the offset windows would be empty
    # or overlap, which dropped or repeated text instead of summarizing it.
    tokens = [str(i) for i in range(count)]
    actual = load_get_summary()(" ".join(tokens), budget).split()
    assert actual == [str(i) for i in range(*expected)]


def test_windows_at_threshold_are_adjacent_and_disjoint():
    tokens = [str(i) for i in range(4000)]
    actual = load_get_summary()(" ".join(tokens), 2000).split()
    assert actual == tokens[1000:3000]
