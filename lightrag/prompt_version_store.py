from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from lightrag.prompt_versions import (
    PROMPT_VERSION_GROUPS,
    build_localized_seed_versions,
    normalize_prompt_group_payload,
    validate_prompt_group_payload,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _seed_record_needs_refresh(existing: dict[str, Any], seed: dict[str, Any]) -> bool:
    fields_to_compare = (
        "group_type",
        "version_name",
        "version_number",
        "comment",
        "source_version_id",
        "payload",
    )
    return any(existing.get(field) != seed.get(field) for field in fields_to_compare)


class PromptVersionStore:
    def __init__(self, working_dir: str | Path, workspace: str = "") -> None:
        self.working_dir = Path(working_dir)
        self.workspace = workspace
        workspace_dir = self.working_dir / workspace if workspace else self.working_dir
        self.registry_dir = workspace_dir / "prompt_versions"
        self.registry_path = self.registry_dir / "registry.json"

    def _build_empty_registry(self) -> dict[str, Any]:
        return {
            "workspace": self.workspace,
            "indexing": {
                "group_type": "indexing",
                "active_version_id": None,
                "versions": [],
            },
            "retrieval": {
                "group_type": "retrieval",
                "active_version_id": None,
                "versions": [],
            },
        }

    def _read_or_default(self) -> dict[str, Any]:
        if not self.registry_path.exists():
            return self._build_empty_registry()

        with self.registry_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _atomic_write(self, payload: dict[str, Any]) -> None:
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self.registry_path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        tmp_path.replace(self.registry_path)

    def initialize(self, locale: str) -> dict[str, Any]:
        registry = self._read_or_default()
        seeds = build_localized_seed_versions(locale)
        changed = False
        for group_type in ("indexing", "retrieval"):
            seed = seeds[group_type]
            existing_versions = registry[group_type]["versions"]
            existing_seed = next(
                (
                    version
                    for version in existing_versions
                    if version["version_id"] == seed["version_id"]
                ),
                None,
            )
            if existing_seed is None:
                existing_versions.append(seed)
                changed = True
                continue

            if _seed_record_needs_refresh(existing_seed, seed):
                existing_seed.update(
                    {
                        "group_type": seed["group_type"],
                        "version_name": seed["version_name"],
                        "version_number": seed["version_number"],
                        "comment": seed["comment"],
                        "source_version_id": seed["source_version_id"],
                        "payload": deepcopy(seed["payload"]),
                    }
                )
                changed = True
        if changed:
            self._atomic_write(registry)
        return registry

    def list_versions(self, group_type: str) -> dict[str, Any]:
        return self._read_or_default()[group_type]

    def get_version(self, group_type: str, version_id: str) -> dict[str, Any]:
        registry = self._read_or_default()
        for version in registry[group_type]["versions"]:
            if version["version_id"] == version_id:
                return deepcopy(version)
        raise ValueError(
            f"Prompt version '{version_id}' not found in group '{group_type}'"
        )

    def _next_version_number(self, group_type: str, registry: dict[str, Any]) -> int:
        versions = registry[group_type]["versions"]
        if not versions:
            return 1
        return max(version["version_number"] for version in versions) + 1

    def create_version(
        self,
        group_type: str,
        payload: dict[str, Any],
        version_name: str,
        comment: str,
        source_version_id: str | None,
    ) -> dict[str, Any]:
        if group_type not in PROMPT_VERSION_GROUPS:
            raise ValueError(f"Unknown prompt version group '{group_type}'")

        normalized_payload = normalize_prompt_group_payload(group_type, payload)
        validate_prompt_group_payload(group_type, normalized_payload)
        registry = self._read_or_default()
        record = {
            "version_id": str(uuid4()),
            "group_type": group_type,
            "version_name": version_name,
            "version_number": self._next_version_number(group_type, registry),
            "comment": comment,
            "source_version_id": source_version_id,
            "created_at": _utc_now(),
            "payload": deepcopy(normalized_payload),
        }
        registry[group_type]["versions"].append(record)
        self._atomic_write(registry)
        return deepcopy(record)

    def copy_version(
        self,
        group_type: str,
        version_id: str,
        version_name: str,
        comment: str,
    ) -> dict[str, Any]:
        source = self.get_version(group_type, version_id)
        return self.create_version(
            group_type,
            source["payload"],
            version_name,
            comment,
            version_id,
        )

    def update_version(
        self,
        group_type: str,
        version_id: str,
        payload: dict[str, Any],
        version_name: str,
        comment: str,
    ) -> dict[str, Any]:
        if group_type not in PROMPT_VERSION_GROUPS:
            raise ValueError(f"Unknown prompt version group '{group_type}'")

        normalized_payload = normalize_prompt_group_payload(group_type, payload)
        validate_prompt_group_payload(group_type, normalized_payload)
        registry = self._read_or_default()
        for version in registry[group_type]["versions"]:
            if version["version_id"] == version_id:
                version["version_name"] = version_name
                version["comment"] = comment
                version["payload"] = deepcopy(normalized_payload)
                self._atomic_write(registry)
                return deepcopy(version)

        raise ValueError(
            f"Prompt version '{version_id}' not found in group '{group_type}'"
        )

    def activate_version(self, group_type: str, version_id: str) -> dict[str, Any]:
        registry = self._read_or_default()
        self.get_version(group_type, version_id)
        registry[group_type]["active_version_id"] = version_id
        self._atomic_write(registry)
        return self.get_version(group_type, version_id)

    def delete_version(self, group_type: str, version_id: str) -> None:
        registry = self._read_or_default()
        if registry[group_type]["active_version_id"] == version_id:
            raise ValueError("Cannot delete the active prompt version")

        registry[group_type]["versions"] = [
            item
            for item in registry[group_type]["versions"]
            if item["version_id"] != version_id
        ]
        self._atomic_write(registry)

    def diff_versions(
        self, group_type: str, version_id: str, base_version_id: str | None
    ) -> dict[str, Any]:
        target = self.get_version(group_type, version_id)["payload"]
        base = (
            self.get_version(group_type, base_version_id)["payload"]
            if base_version_id
            else {}
        )
        changed_keys = sorted(set(base) | set(target))
        return {
            "group_type": group_type,
            "base_version_id": base_version_id,
            "version_id": version_id,
            "changes": {
                key: {"before": base.get(key), "after": target.get(key)}
                for key in changed_keys
                if base.get(key) != target.get(key)
            },
        }
