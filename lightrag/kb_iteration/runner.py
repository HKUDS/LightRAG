from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .markdown import write_markdown_memory
from .models import KGSnapshot, QualityScore
from .quality import evaluate_snapshot_quality, write_quality_artifacts
from .snapshot import build_snapshot_from_graphml, write_snapshot_artifacts

GRAPH_FILENAME = "graph_chunk_entity_relation.graphml"
PENDING_REVIEW_PHASE = "pending_user_review"


@dataclass(frozen=True)
class IterationRunResult:
    output_dir: Path
    snapshot: KGSnapshot
    quality_score: QualityScore
    artifact_paths: dict[str, Path]


def run_iteration(
    *,
    workspace: str,
    storage_root: str | Path,
    input_root: str | Path,
    output_root: str | Path,
    profile: str | None = None,
) -> IterationRunResult:
    storage_workspace = Path(storage_root) / workspace
    graph_path = storage_workspace / GRAPH_FILENAME
    if not graph_path.exists():
        raise FileNotFoundError(f"GraphML file not found: {graph_path}")

    input_workspace = Path(input_root) / workspace
    output_dir = Path(output_root) / workspace
    source_files = _discover_source_files(input_workspace)

    snapshot = build_snapshot_from_graphml(
        graph_path,
        workspace=workspace,
        source_files=source_files,
        profile=profile,
    )
    snapshot_artifacts = write_snapshot_artifacts(snapshot, output_dir)
    markdown_artifacts = write_markdown_memory(snapshot, output_dir)

    quality_score = evaluate_snapshot_quality(snapshot)
    quality_artifacts = write_quality_artifacts(quality_score, output_dir)

    artifact_paths = {
        **snapshot_artifacts,
        **markdown_artifacts,
        **quality_artifacts,
    }
    artifact_paths["iteration_log"] = output_dir / "iteration_log.md"
    _append_iteration_log(
        artifact_paths["iteration_log"],
        workspace=workspace,
        artifact_paths=artifact_paths,
        output_dir=output_dir,
    )

    return IterationRunResult(
        output_dir=output_dir,
        snapshot=snapshot,
        quality_score=quality_score,
        artifact_paths=artifact_paths,
    )


def _discover_source_files(input_workspace: Path) -> list[str]:
    if not input_workspace.exists():
        return []

    return sorted(
        path.relative_to(input_workspace).as_posix()
        for path in input_workspace.rglob("*")
        if path.is_file()
    )


def _append_iteration_log(
    path: Path,
    *,
    workspace: str,
    artifact_paths: dict[str, Path],
    output_dir: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write("\n## Run\n\n")
        file.write(f"- workspace: {workspace}\n")
        file.write(f"- phase: {PENDING_REVIEW_PHASE}\n")
        file.write(
            f"- snapshot: {_relative_artifact(artifact_paths['kg_snapshot'], output_dir)}\n"
        )
        file.write(
            "- quality_score: "
            f"{_relative_artifact(artifact_paths['quality_score'], output_dir)}\n"
        )
        file.write(
            "- quality_report: "
            f"{_relative_artifact(artifact_paths['quality_report'], output_dir)}\n"
        )
        file.write("- artifacts:\n")
        for name in sorted(artifact_paths):
            file.write(
                "  - "
                f"{name}: {_relative_artifact(artifact_paths[name], output_dir)}\n"
            )


def _relative_artifact(path: Path, output_dir: Path) -> str:
    return path.relative_to(output_dir).as_posix()
