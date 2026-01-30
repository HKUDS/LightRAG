from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from importlib import resources

from jinja2 import StrictUndefined, meta
from jinja2.sandbox import SandboxedEnvironment

from lightrag.base import PROMPT_TYPE_SPECS, PromptOrigin, PromptTemplate, PromptType

from .errors import (
    PromptTemplateGitError,
    PromptTemplateNotFoundError,
    PromptTemplateValidationError,
)
from .file_lock import FileLock
from .git_utils import commit_all, ensure_git_repo, has_commits, is_git_repo, repo_head


# NOTE: use single escaping in raw strings; `r"\.md\.j2"` matches literal dots.
_FILENAME_RE = re.compile(r"^(kg|query|keyword|ned)-(.+)\.md\.j2$")


@dataclass(frozen=True)
class TemplateKey:
    prompt_type: PromptType
    name: str


def _template_file_name(prompt_type: PromptType, name: str) -> str:
    # Keep names filesystem-friendly; callers can enforce stricter rules if needed.
    if not name or name.strip() != name:
        raise PromptTemplateValidationError(
            "Template name must be a non-empty, trimmed string"
        )
    return f"{prompt_type.value}-{name}.md.j2"


class GitPromptManager:
    """File-based prompt template manager with a framework-managed git repo.

    - System templates: packaged under `lightrag/prompts/system` (read-only).
    - User templates: `working_dir/prompts/user` (read-write, git-managed).
    - Overlay: user overrides system with same (type, name).
    """

    def __init__(self, *, working_dir: str):
        self._working_dir = Path(working_dir)
        self._user_repo_dir = self._working_dir / "prompts" / "user"
        self._lock_path = self._working_dir / "prompts" / ".prompt_repo.lock"

        # Use sandboxed environment: templates are editable by admins, but still should
        # not allow arbitrary attribute traversal / code execution.
        self._jinja_env = SandboxedEnvironment(
            undefined=StrictUndefined,
            autoescape=False,
            keep_trailing_newline=True,
        )

        self._system_index: dict[TemplateKey, PromptTemplate] = {}
        self._user_index: dict[TemplateKey, PromptTemplate] = {}
        self._resolved_index: dict[TemplateKey, PromptTemplate] = {}
        self._template_vars: dict[TemplateKey, set[str]] = {}

    @property
    def user_repo_dir(self) -> Path:
        return self._user_repo_dir

    def get_user_repo_head(self) -> Optional[str]:
        return repo_head(self._user_repo_dir) if self._user_repo_dir.exists() else None

    async def initialize(self) -> None:
        with FileLock(self._lock_path):
            # Ensure directory exists.
            self._user_repo_dir.mkdir(parents=True, exist_ok=True)

            # Ensure git repo.
            ensure_git_repo(self._user_repo_dir)

            # Seed templates (copy missing system templates into user repo).
            self._seed_user_repo_from_system()

            # Ensure repo is clean and audit-friendly.
            # - On first init: create an initial commit.
            # - On later runs: commit any newly seeded templates (no-op if nothing changed).
            if not has_commits(self._user_repo_dir):
                commit_all(
                    self._user_repo_dir,
                    "Initial system prompt generation",
                    author="lightrag",
                )
            else:
                commit_all(
                    self._user_repo_dir,
                    "Sync system prompt templates",
                    author="lightrag",
                )

        # Build indexes after filesystem changes.
        self.reload_templates()

    async def finalize(self) -> None:
        # Nothing to finalize currently; keep API symmetry with storages.
        return

    def _iter_system_templates(self):
        root = resources.files("lightrag.prompts").joinpath("system")

        def _walk(node):
            for child in node.iterdir():
                if child.is_dir():
                    yield from _walk(child)
                else:
                    yield child

        for item in _walk(root):
            if item.name.endswith(".md.j2"):
                yield item

    def _seed_user_repo_from_system(self) -> None:
        # If git is not available, ensure_git_repo would already error, but keep a clear message here.
        if not is_git_repo(self._user_repo_dir):
            raise PromptTemplateGitError(
                f"User prompt directory is not a git repo: {self._user_repo_dir}"
            )

        for sys_file in self._iter_system_templates():
            dest = self._user_repo_dir / sys_file.name
            if dest.exists():
                continue
            dest.write_text(sys_file.read_text(encoding="utf-8"), encoding="utf-8")

    def reload_templates(self) -> None:
        self._system_index = self._scan_system_templates()
        self._user_index = self._scan_user_templates()

        resolved: dict[TemplateKey, PromptTemplate] = {}
        resolved.update(self._system_index)
        resolved.update(self._user_index)  # user overrides
        self._resolved_index = resolved

        # Parse variables for strict validation.
        self._template_vars = {}
        for key, tmpl in self._resolved_index.items():
            self._template_vars[key] = self._extract_and_validate_template_vars(
                key, tmpl
            )

    def _scan_system_templates(self) -> dict[TemplateKey, PromptTemplate]:
        index: dict[TemplateKey, PromptTemplate] = {}
        for item in self._iter_system_templates():
            match = _FILENAME_RE.match(item.name)
            if not match:
                continue
            type_str, name = match.group(1), match.group(2)
            ptype = PromptType(type_str)
            key = TemplateKey(ptype, name)
            index[key] = PromptTemplate(
                type=ptype,
                name=name,
                origin=PromptOrigin.SYSTEM,
                content=item.read_text(encoding="utf-8"),
                file_path=str(item),
            )
        return index

    def _scan_user_templates(self) -> dict[TemplateKey, PromptTemplate]:
        index: dict[TemplateKey, PromptTemplate] = {}
        if not self._user_repo_dir.exists():
            return index

        for path in self._user_repo_dir.glob("*.md.j2"):
            match = _FILENAME_RE.match(path.name)
            if not match:
                continue
            type_str, name = match.group(1), match.group(2)
            ptype = PromptType(type_str)
            key = TemplateKey(ptype, name)
            index[key] = PromptTemplate(
                type=ptype,
                name=name,
                origin=PromptOrigin.USER,
                content=path.read_text(encoding="utf-8"),
                file_path=str(path),
            )
        return index

    def _extract_and_validate_template_vars(
        self, key: TemplateKey, tmpl: PromptTemplate
    ) -> set[str]:
        try:
            ast = self._jinja_env.parse(tmpl.content)
        except Exception as e:
            raise PromptTemplateValidationError(
                f"Failed to parse template {tmpl.origin.value}:{key.prompt_type.value}-{key.name}: {e}"
            ) from e

        vars_used = set(meta.find_undeclared_variables(ast))

        allowed = PROMPT_TYPE_SPECS[key.prompt_type].allowed_variables
        illegal = sorted(vars_used - allowed)
        if illegal:
            raise PromptTemplateValidationError(
                "Template uses variables not allowed for this PromptType. "
                f"template={key.prompt_type.value}-{key.name}, illegal={illegal}, allowed={sorted(allowed)}"
            )

        return vars_used

    async def get_template(
        self, prompt_type: PromptType, template_name: str
    ) -> Optional[PromptTemplate]:
        key = TemplateKey(prompt_type, template_name)
        return self._resolved_index.get(key)

    def list_templates(
        self,
        prompt_type: Optional[PromptType] = None,
        *,
        origin: Optional[PromptOrigin] = None,
        resolved: bool = True,
    ) -> list[PromptTemplate]:
        if resolved:
            src = self._resolved_index
        else:
            if origin == PromptOrigin.SYSTEM:
                src = self._system_index
            elif origin == PromptOrigin.USER:
                src = self._user_index
            else:
                # Combine without overriding so caller can see duplicates if needed.
                src = {**self._system_index, **self._user_index}

        items: list[PromptTemplate] = []
        for key, tmpl in src.items():
            if prompt_type is not None and key.prompt_type != prompt_type:
                continue
            if origin is not None and tmpl.origin != origin:
                continue
            items.append(tmpl)

        # Stable ordering: type, then name.
        items.sort(key=lambda t: (t.type.value, t.name))
        return items

    async def render(
        self,
        prompt_type: PromptType,
        template_name: str,
        **variables: Any,
    ) -> str:
        key = TemplateKey(prompt_type, template_name)
        tmpl = self._resolved_index.get(key)
        if tmpl is None:
            raise PromptTemplateNotFoundError(
                f"Template not found: {prompt_type.value}-{template_name}"
            )

        needed = self._template_vars.get(key)
        if needed is None:
            needed = self._extract_and_validate_template_vars(key, tmpl)
            self._template_vars[key] = needed

        missing = sorted(v for v in needed if v not in variables)
        if missing:
            raise PromptTemplateValidationError(
                "Missing required template variables. "
                f"template={prompt_type.value}-{template_name}, missing={missing}"
            )

        allowed = PROMPT_TYPE_SPECS[prompt_type].allowed_variables
        extra_illegal = sorted(set(variables.keys()) - allowed)
        if extra_illegal:
            raise PromptTemplateValidationError(
                "Render variables contain keys not allowed for this PromptType. "
                f"template={prompt_type.value}-{template_name}, illegal={extra_illegal}, allowed={sorted(allowed)}"
            )

        try:
            compiled = self._jinja_env.from_string(tmpl.content)
            return compiled.render(**variables)
        except Exception as e:
            raise PromptTemplateValidationError(
                f"Failed to render template {prompt_type.value}-{template_name}: {e}"
            ) from e

    async def upsert_user_template(
        self,
        prompt_type: PromptType,
        template_name: str,
        *,
        content: str,
        author: str,
        commit_message: Optional[str] = None,
    ) -> PromptTemplate:
        # Always forbid writes if the user repo isn't a git repo (should not happen if initialize ran).
        with FileLock(self._lock_path):
            ensure_git_repo(self._user_repo_dir)

            filename = _template_file_name(prompt_type, template_name)
            path = self._user_repo_dir / filename
            path.write_text(content, encoding="utf-8")

            msg = commit_message or f"Update {filename} by {author}"
            # Add a timestamp suffix for easier auditing when multiple updates share same message.
            msg = f"{msg} ({datetime.utcnow().isoformat()}Z)"
            commit_all(self._user_repo_dir, msg, author=author)

        self.reload_templates()

        key = TemplateKey(prompt_type, template_name)
        tmpl = self._resolved_index.get(key)
        if tmpl is None:
            raise PromptTemplateNotFoundError(
                f"Template not found after upsert: {prompt_type.value}-{template_name}"
            )
        return tmpl
