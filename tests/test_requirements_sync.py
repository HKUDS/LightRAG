"""The hand-written ``requirements-offline*.txt`` files must track pyproject.toml.

Those files are the pip-only install path for air-gapped deployments
(docs/OfflineDeployment.md). Nothing generates them: they repeat version
constraints that pyproject.toml declares, and the two drift silently. Dependabot
runs on the ``uv`` ecosystem, which rewrites pyproject.toml and uv.lock but is
not relied on to rewrite these files, so a bump that widens a bound in
pyproject.toml lands here as a failure instead of a pip install that resolves a
different version than the one CI tested.

Rules pinned:

1. Every requirement in a requirements file whose package pyproject.toml also
   declares carries exactly pyproject.toml's specifier and marker.
2. pyproject.toml declares each package with one specifier and marker
   everywhere it appears (core dependencies and every extra), so rule 1 has a
   single answer.
3. The layered files list exactly their extra: ``requirements-offline-llm.txt``
   is ``[offline-llm]`` and ``requirements-offline-storage.txt`` is
   ``[offline-storage]``.

Packages pyproject.toml does not declare (the direct-URL spaCy model wheels and
``spacy-pkuseg`` in ``requirements-offline-smart-heading.txt``) are pinned only
in their requirements file and are not checked here.
"""

from __future__ import annotations

import pathlib
import sys
from collections import defaultdict

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - requires-python still allows 3.10
    tomllib = pytest.importorskip("tomli")

pytestmark = pytest.mark.offline

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_PYPROJECT = _REPO_ROOT / "pyproject.toml"
_REQUIREMENTS_FILES = sorted(_REPO_ROOT.glob("requirements-offline*.txt"))
_LAYERED_FILES = {
    "requirements-offline-llm.txt": "offline-llm",
    "requirements-offline-storage.txt": "offline-storage",
}
_SELF = canonicalize_name("lightrag-hku")


def _constraint(req: Requirement) -> tuple[str, str]:
    """The part of a requirement both files must agree on."""
    specifier = ",".join(sorted(str(s) for s in req.specifier))
    marker = str(req.marker) if req.marker else ""
    return specifier, marker


def _load_pyproject() -> dict:
    with _PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)["project"]


def _pyproject_requirements() -> dict[str, list[tuple[str, Requirement]]]:
    """Map canonical name -> [(where, requirement)] across deps and extras."""
    project = _load_pyproject()
    sources = {"dependencies": project.get("dependencies", [])}
    for extra, reqs in project.get("optional-dependencies", {}).items():
        sources[f"[{extra}]"] = reqs
    found: dict[str, list[tuple[str, Requirement]]] = defaultdict(list)
    for where, reqs in sources.items():
        for line in reqs:
            req = Requirement(line)
            name = canonicalize_name(req.name)
            if name != _SELF:
                found[name].append((where, req))
    return found


def _file_requirements(path: pathlib.Path) -> list[Requirement]:
    reqs = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split(" #", 1)[0].strip()
        if not line or line.startswith(("#", "-")):
            continue
        reqs.append(Requirement(line))
    return reqs


def test_requirements_files_exist():
    names = {p.name for p in _REQUIREMENTS_FILES}
    assert set(_LAYERED_FILES) <= names, names


def test_pyproject_declares_each_package_consistently():
    mismatched = {
        name: [(where, str(req)) for where, req in decls]
        for name, decls in _pyproject_requirements().items()
        if len({_constraint(req) for _, req in decls}) > 1
    }
    assert not mismatched, (
        "pyproject.toml declares these packages with different constraints "
        f"in different places: {mismatched}"
    )


@pytest.mark.parametrize("path", _REQUIREMENTS_FILES, ids=lambda p: p.name)
def test_requirements_file_matches_pyproject(path: pathlib.Path):
    declared = _pyproject_requirements()
    drift = []
    for req in _file_requirements(path):
        decls = declared.get(canonicalize_name(req.name))
        if not decls:
            continue
        expected = decls[0][1]
        if _constraint(req) != _constraint(expected):
            drift.append(f"{req}  (pyproject.toml {decls[0][0]}: {expected})")
    assert not drift, (
        f"{path.name} is out of sync with pyproject.toml; copy the "
        "pyproject.toml constraint into the requirements file:\n  " + "\n  ".join(drift)
    )


@pytest.mark.parametrize("filename,extra", sorted(_LAYERED_FILES.items()))
def test_layered_file_lists_exactly_its_extra(filename: str, extra: str):
    in_file = {
        canonicalize_name(r.name) for r in _file_requirements(_REPO_ROOT / filename)
    }
    in_extra = {
        canonicalize_name(Requirement(line).name)
        for line in _load_pyproject()["optional-dependencies"][extra]
    }
    assert in_file == in_extra, (
        f"{filename} must list exactly pyproject.toml's [{extra}] extra; "
        f"missing: {sorted(in_extra - in_file)}, "
        f"extra: {sorted(in_file - in_extra)}"
    )
