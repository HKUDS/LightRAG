"""Batch evaluation must pair each saved answer with the question it answers.

``batch_eval.py`` is imported for real, like the Step_3 scripts in
``test_query_result_output.py``. ``jsonlines`` is not a project dependency, so
a stand-in module is installed before the import; the OpenAI client is
replaced on the loaded module, so nothing reaches the network.
"""

import importlib.util
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.offline

SCRIPT = Path(__file__).resolve().parents[2] / "reproduce" / "batch_eval.py"


@contextmanager
def _open_jsonlines(path, mode):
    with open(path, mode, encoding="utf-8") as output:
        writer = Mock()
        writer.write.side_effect = lambda row: output.write(json.dumps(row) + "\n")
        yield writer


@pytest.fixture
def script(monkeypatch):
    jsonlines = ModuleType("jsonlines")
    jsonlines.open = Mock(side_effect=_open_jsonlines)
    monkeypatch.setitem(sys.modules, "jsonlines", jsonlines)
    spec = importlib.util.spec_from_file_location("reproduce_batch_eval", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "OpenAI", Mock())
    return module


def row(query, answer):
    return {"query": query, "result": answer}


def run_batch(tmp_path, script, first, second, questions=("First?", "Second?")):
    question_file = tmp_path / "questions.txt"
    question_file.write_text(
        "\n".join(f"- Question {i}: {q}" for i, q in enumerate(questions, 1))
    )
    paths = []
    for name, rows in (("first", first), ("second", second)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(rows))
        paths.append(path)
    script.batch_eval(question_file, *paths, tmp_path / "requests.jsonl")


def prompts(tmp_path):
    lines = (tmp_path / "requests.jsonl").read_text().splitlines()
    return [json.loads(line)["body"]["messages"][1]["content"] for line in lines]


def assert_nothing_submitted(script):
    script.OpenAI.assert_not_called()
    script.jsonlines.open.assert_not_called()


def test_reordered_results_are_matched_by_query(tmp_path, script):
    run_batch(
        tmp_path,
        script,
        [row("Second?", "A2"), row("First?", "A1")],
        [row("First?", "B1"), row("Second?", "B2")],
    )
    first, second = prompts(tmp_path)
    assert ("First?" in first, "A1" in first, "B1" in first) == (True, True, True)
    assert ("A2" in first, "B2" in first) == (False, False)
    assert ("Second?" in second, "A2" in second, "B2" in second) == (
        True,
        True,
        True,
    )
    script.OpenAI.return_value.batches.create.assert_called_once()


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        pytest.param(
            [row("Second?", "A2")],
            r"first\.json: 1 of 2 questions have no answer \('First\?'\)",
            id="missing",
        ),
        pytest.param(
            [row("First?", "A1"), row("First?", "again"), row("Second?", "A2")],
            r"first\.json: question answered twice: 'First\?'",
            id="duplicate",
        ),
        pytest.param(
            [row("First?", "A1"), row("Unknown?", "other")],
            r"first\.json: entry 1 answers a question that is not in the question "
            r"file: 'Unknown\?'",
            id="unknown",
        ),
        pytest.param(
            [row("First?", "A1"), {"query": "Second?"}],
            r"first\.json: entry 1 has no 'result': 'Second\?'",
            id="no-result",
        ),
        pytest.param(
            [row("First?", "A1"), "Second?"],
            r"first\.json: entry 1 answers a question that is not in the question "
            r"file: None",
            id="not-an-object",
        ),
    ],
)
def test_bad_results_fail_before_submission(tmp_path, script, rows, message):
    with pytest.raises(ValueError, match=message):
        run_batch(tmp_path, script, rows, [row("First?", "B1"), row("Second?", "B2")])
    assert_nothing_submitted(script)


def test_missing_answers_are_listed_with_a_bound(tmp_path, script):
    questions = [f"Q{i}?" for i in range(1, 6)]
    with pytest.raises(
        ValueError, match=r"5 of 5 .*'Q1\?', 'Q2\?', 'Q3\?' \(\+2 more\)"
    ):
        run_batch(tmp_path, script, [], [], questions=questions)
    assert_nothing_submitted(script)


@pytest.mark.parametrize(
    ("questions", "message"),
    [
        pytest.param(("First?", "First?"), r"repeated: 'First\?'", id="duplicate"),
        pytest.param((), r"no '- Question N: \.\.\.' lines found", id="empty"),
    ],
)
def test_bad_question_file_fails_before_submission(
    tmp_path, script, questions, message
):
    with pytest.raises(ValueError, match=message):
        run_batch(
            tmp_path,
            script,
            [row("First?", "A1")],
            [row("First?", "B1")],
            questions=questions,
        )
    assert_nothing_submitted(script)


def test_markdown_questions_match_query_script_normalization(tmp_path, script):
    run_batch(
        tmp_path,
        script,
        [row("First?", "A1")],
        [row("First?", "B1")],
        questions=("**First?**",),
    )
    (prompt,) = prompts(tmp_path)
    assert ("First?" in prompt, "**First?**" in prompt) == (True, False)
    script.OpenAI.return_value.batches.create.assert_called_once()
