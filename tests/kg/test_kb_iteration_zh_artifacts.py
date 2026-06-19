from pathlib import Path

from lightrag.kb_iteration.zh_artifacts import (
    artifact_zh_relative_path,
    build_zh_json_payload,
    machine_token_should_be_preserved,
)


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
