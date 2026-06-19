import json
from pathlib import Path

from lightrag.kb_iteration.zh_artifacts import (
    artifact_zh_relative_path,
    build_zh_json_payload,
    ensure_zh_artifact,
    machine_token_should_be_preserved,
)


class FakeZhClient:
    def __init__(self, response: str = "## 摘要\n\n已生成中文展示稿。"):
        self.response = response
        self.calls: list[tuple[str, str]] = []

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self.response


class FailingZhClient:
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        raise RuntimeError("translation service unavailable")


class KeywordOnlyZhClient:
    def __init__(self, response: str = "## 质量报告\n\n保留 source_id。"):
        self.response = response
        self.calls: list[dict[str, str]] = []

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls.append(
            {"system_prompt": system_prompt, "user_prompt": user_prompt}
        )
        return self.response


def test_artifact_zh_relative_path_inserts_zh_before_final_suffix():
    assert artifact_zh_relative_path(Path("quality_report.md")) == Path(
        "quality_report.zh.md"
    )
    assert artifact_zh_relative_path(Path("snapshots/quality_score.json")) == Path(
        "snapshots/quality_score.zh.json"
    )
    assert artifact_zh_relative_path(Path("proposals.generated.yaml")) == Path(
        "proposals.generated.zh.yaml"
    )


def test_build_zh_json_payload_preserves_keys_values_and_adds_known_labels():
    payload = {
        "overall": 97,
        "metrics": {"hierarchy_missing_branch_count": 0},
        "findings": [{"message": "No blockers", "source_id": "chunk-1"}],
    }

    zh_payload = build_zh_json_payload(payload)

    assert zh_payload["overall"] == 97
    assert zh_payload["metrics"]["hierarchy_missing_branch_count"] == 0
    assert zh_payload["findings"] == [
        {"message": "No blockers", "source_id": "chunk-1"}
    ]
    assert zh_payload["_zh_labels"]["overall"] == "总分"
    assert zh_payload["_zh_labels"]["metrics"] == "指标"
    assert (
        zh_payload["metrics"]["_zh_labels"]["hierarchy_missing_branch_count"]
        == "缺失层级分支数"
    )


def test_build_zh_json_payload_labels_nested_dicts_without_replacing_keys():
    payload = {
        "details": {
            "hierarchy_branches": {
                "required": [{"key": "symptoms", "label": "症状"}],
                "missing": [],
            }
        }
    }

    zh_payload = build_zh_json_payload(payload)

    assert list(zh_payload) == ["details", "_zh_labels"]
    assert list(zh_payload["details"]) == ["hierarchy_branches", "_zh_labels"]
    assert zh_payload["details"]["hierarchy_branches"]["required"] == [
        {"key": "symptoms", "label": "症状"}
    ]
    assert zh_payload["details"]["_zh_labels"]["hierarchy_branches"] == "层级分支"
    assert zh_payload["details"]["hierarchy_branches"]["_zh_labels"]["required"] == (
        "必需项"
    )


def test_build_zh_json_payload_does_not_mutate_original_payload():
    payload = {
        "overall": 97,
        "metrics": {"hierarchy_missing_branch_count": 0},
    }

    build_zh_json_payload(payload)

    assert payload == {
        "overall": 97,
        "metrics": {"hierarchy_missing_branch_count": 0},
    }


def test_build_zh_json_payload_overwrites_reserved_zh_labels():
    payload = {
        "overall": 97,
        "_zh_labels": {"overall": "stale label"},
    }

    zh_payload = build_zh_json_payload(payload)

    assert zh_payload["_zh_labels"] == {"overall": "总分"}


def test_machine_token_should_be_preserved_for_ids_paths_models_and_snake_case():
    preserved = [
        "prop-normalize-relation-keywords",
        "doc-b29c711f27db9ad51c2851d9db562957-chunk-006",
        "snapshots/quality_score.json",
        "deepseek-v4-pro",
        "GPT-4o-mini",
        "API_KEY",
        "hierarchy_missing_branch_count",
    ]

    for token in preserved:
        assert machine_token_should_be_preserved(token), token


def test_machine_token_should_not_be_preserved_for_natural_language_text():
    assert not machine_token_should_be_preserved("Improve hierarchy readability")


def test_ensure_zh_artifact_writes_markdown_translation(tmp_path: Path):
    source_path = tmp_path / "snapshots" / "quality_report.md"
    source_path.parent.mkdir()
    source_path.write_text(
        "# Quality Report\n\nsource_id: chunk-1\n\nImprove hierarchy readability.\n",
        encoding="utf-8",
    )
    client = FakeZhClient("## 质量报告\n\n保留 source_id 和 chunk-1。")

    result = ensure_zh_artifact(
        source_path,
        artifact_key="quality_report.md",
        content_type="text/markdown",
        client=client,
        model="gpt-4o-mini",
    )

    zh_path = tmp_path / "snapshots" / "quality_report.zh.md"
    assert zh_path.read_text(encoding="utf-8") == (
        "## 质量报告\n\n保留 source_id 和 chunk-1。\n"
    )
    assert result.artifact_key == "quality_report.md"
    assert result.source_relative_path == Path("quality_report.md")
    assert result.zh_relative_path == Path("quality_report.zh.md")
    assert result.content_type == "text/markdown"
    assert result.generated is True
    assert result.fallback_to_source is False
    assert result.model == "gpt-4o-mini"
    assert result.error is None
    assert result.content == "## 质量报告\n\n保留 source_id 和 chunk-1。\n"
    assert len(client.calls) == 1
    system_prompt, user_prompt = client.calls[0]
    assert "source_id" in system_prompt
    assert "proposal_id" in system_prompt
    assert "简洁中文" in system_prompt
    assert "Improve hierarchy readability" in user_prompt


def test_ensure_zh_artifact_calls_keyword_only_client_and_preserves_artifact_key(
    tmp_path: Path,
):
    source_path = tmp_path / "snapshots" / "quality_report.md"
    source_path.parent.mkdir()
    source_path.write_text("Improve source_id readability.\n", encoding="utf-8")
    client = KeywordOnlyZhClient("## 质量报告\n\n保留 source_id。")

    result = ensure_zh_artifact(
        source_path,
        artifact_key="snapshots/quality_report.md",
        content_type="text/markdown",
        client=client,
        model="gpt-4o-mini",
    )

    assert result.artifact_key == "snapshots/quality_report.md"
    assert client.calls[0]["system_prompt"]
    assert "Improve source_id readability" in client.calls[0]["user_prompt"]
    assert result.content == "## 质量报告\n\n保留 source_id。\n"


def test_ensure_zh_artifact_writes_json_with_labels_without_llm(tmp_path: Path):
    source_path = tmp_path / "snapshots" / "quality_score.json"
    source_path.parent.mkdir()
    source_path.write_text(
        json.dumps({"overall": 97, "metrics": {"missing": []}}, ensure_ascii=False),
        encoding="utf-8",
    )
    client = FakeZhClient()

    result = ensure_zh_artifact(
        source_path,
        artifact_key="quality_score.json",
        content_type="application/json",
        client=client,
        model="unused-model",
    )

    zh_payload = json.loads(
        (tmp_path / "snapshots" / "quality_score.zh.json").read_text(
            encoding="utf-8"
        )
    )
    assert zh_payload["_zh_labels"]["overall"] == "总分"
    assert zh_payload["metrics"]["_zh_labels"]["missing"] == "缺失项"
    assert client.calls == []
    assert result.artifact_key == "quality_score.json"
    assert result.generated is True
    assert result.fallback_to_source is False
    assert result.payload == zh_payload
    assert result.content is None


def test_ensure_zh_artifact_falls_back_to_source_when_translation_fails(
    tmp_path: Path,
):
    source_path = tmp_path / "quality_report.md"
    source_path.write_text(
        "# Quality Report\n\nKeep source_id unchanged.\n",
        encoding="utf-8",
    )

    result = ensure_zh_artifact(
        source_path,
        artifact_key="quality_report.md",
        content_type="text/markdown",
        client=FailingZhClient(),
        model="gpt-4o-mini",
    )

    assert not (tmp_path / "quality_report.zh.md").exists()
    assert result.artifact_key == "quality_report.md"
    assert result.generated is False
    assert result.fallback_to_source is True
    assert result.error == "translation service unavailable"
    assert result.content == "# Quality Report\n\nKeep source_id unchanged.\n"
    assert result.payload is None


def test_ensure_zh_artifact_falls_back_when_existing_json_artifact_is_corrupt(
    tmp_path: Path,
):
    source_path = tmp_path / "quality_score.json"
    source_payload = {"overall": 91, "metrics": {"missing": []}}
    source_path.write_text(
        json.dumps(source_payload, ensure_ascii=False),
        encoding="utf-8",
    )
    (tmp_path / "quality_score.zh.json").write_text("{not valid json", encoding="utf-8")
    client = FakeZhClient()

    result = ensure_zh_artifact(
        source_path,
        artifact_key="quality_score.json",
        content_type="application/json",
        client=client,
        model="unused-model",
        force=False,
    )

    assert client.calls == []
    assert result.artifact_key == "quality_score.json"
    assert result.generated is False
    assert result.fallback_to_source is True
    assert "Expecting property name enclosed in double quotes" in result.error
    assert result.payload == source_payload
    assert result.content is None


def test_ensure_zh_artifact_reads_existing_zh_without_client_when_force_false(
    tmp_path: Path,
):
    source_path = tmp_path / "quality_report.md"
    source_path.write_text("# Quality Report\n", encoding="utf-8")
    zh_path = tmp_path / "quality_report.zh.md"
    zh_path.write_text("## 已有中文\n", encoding="utf-8")
    client = FakeZhClient("## 新中文")

    result = ensure_zh_artifact(
        source_path,
        artifact_key="quality_report.md",
        content_type="text/markdown",
        client=client,
        model="gpt-4o-mini",
        force=False,
    )

    assert client.calls == []
    assert result.artifact_key == "quality_report.md"
    assert result.generated is False
    assert result.fallback_to_source is False
    assert result.content == "## 已有中文\n"
    assert result.zh_relative_path == Path("quality_report.zh.md")
