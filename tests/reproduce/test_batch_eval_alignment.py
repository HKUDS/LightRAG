"""Batch evaluation must preserve the query identity of saved answers."""

import ast
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.offline


@pytest.fixture
def batch_eval():
    source = Path(__file__).resolve().parents[2] / "reproduce" / "batch_eval.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    client = Mock()

    @contextmanager
    def open_jsonlines(path, mode):
        with open(path, mode, encoding="utf-8") as output:
            yield SimpleNamespace(
                write=lambda row: output.write(json.dumps(row) + "\n")
            )

    namespace = {
        "re": __import__("re"),
        "json": json,
        "OpenAI": Mock(return_value=client),
        "jsonlines": SimpleNamespace(open=Mock(side_effect=open_jsonlines)),
    }
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return namespace["batch_eval"], namespace["OpenAI"], namespace["jsonlines"]


def run_batch(tmp_path, batch_eval, first, second, questions=("First?", "Second?")):
    queries = tmp_path / "questions.txt"
    queries.write_text(
        "\n".join(f"- Question {i}: {q}" for i, q in enumerate(questions, 1))
    )
    paths = []
    for name, rows in (("first", first), ("second", second)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(rows))
        paths.append(path)
    batch_eval[0](queries, *paths, tmp_path / "requests.jsonl")


def row(query, answer):
    return {"query": query, "result": answer}


def test_reordered_results_are_matched_by_query(tmp_path, batch_eval):
    run_batch(
        tmp_path,
        batch_eval,
        [row("Second?", "A2"), row("First?", "A1")],
        [row("First?", "B1"), row("Second?", "B2")],
    )
    writes = [
        json.loads(line)
        for line in (tmp_path / "requests.jsonl").read_text().splitlines()
    ]
    assert len(writes) == 2
    prompt = writes[0]["body"]["messages"][1]["content"]
    assert "First?" in prompt and "A1" in prompt and "B1" in prompt
    assert "A2" not in prompt


@pytest.mark.parametrize(
    "rows",
    [
        [row("Second?", "A2")],
        [row("First?", "A1"), row("First?", "duplicate")],
        [row("First?", "A1"), row("Unknown?", "other")],
    ],
)
def test_incomplete_or_ambiguous_results_fail_before_submission(
    tmp_path, batch_eval, rows
):
    with pytest.raises(ValueError):
        run_batch(
            tmp_path, batch_eval, rows, [row("First?", "B1"), row("Second?", "B2")]
        )
    batch_eval[1].assert_not_called()
    batch_eval[2].open.assert_not_called()


def test_complete_results_preserve_compatibility(tmp_path, batch_eval):
    run_batch(
        tmp_path,
        batch_eval,
        [row("First?", "A1"), row("Second?", "A2")],
        [row("First?", "B1"), row("Second?", "B2")],
    )
    batch_eval[1].return_value.batches.create.assert_called_once()


def test_duplicate_questions_fail_before_submission(tmp_path, batch_eval):
    with pytest.raises(ValueError):
        run_batch(
            tmp_path,
            batch_eval,
            [row("First?", "A1")],
            [row("First?", "B1")],
            questions=("First?", "First?"),
        )
    batch_eval[1].assert_not_called()


def test_markdown_questions_match_query_script_normalization(tmp_path, batch_eval):
    run_batch(
        tmp_path,
        batch_eval,
        [row("First?", "A1")],
        [row("First?", "B1")],
        questions=("**First?**",),
    )
    batch_eval[1].return_value.batches.create.assert_called_once()
