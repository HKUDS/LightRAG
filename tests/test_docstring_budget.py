"""Docstrings state rules; contract documents carry the reasoning.

Two failure modes this pins, both of which the repository has already been
through once:

1. **A docstring grows into a design document.** ``NetworkXStorage``'s reached
   592 lines and 41.9% of its module was docstring. Reading the class meant
   scrolling past a specification, and the same facts stated in two places
   drifted apart -- the gate's docstring and the concurrency contract had
   already disagreed about what bounds the admin lock.
2. **A comment cites a GitHub issue as the place something is defined.** The
   referent does not survive a fork, and the sentence keeps claiming something
   is "documented" while nothing in the tree documents it. Provenance-only
   mentions had the same fate and simply carried no information.

Both rules are about the *tree*, not about any one change, which is why they are
tests rather than review habits.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

pytestmark = pytest.mark.offline

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_PACKAGE = _REPO_ROOT / "lightrag"

# A docstring may state rules, obligations and gotchas; past this it is
# explaining a mechanism, which belongs in docs/design/. The limit is generous
# on purpose -- it catches a document, not a thorough docstring.
_MAX_DOCSTRING_LINES = 80

# Raised separately because a class docstring legitimately covers a whole
# surface, while a function's covers one call.
_MAX_CLASS_DOCSTRING_LINES = 100

# Known oversized docstrings OUTSIDE the storage and concurrency code this
# budget was introduced for. Each is a real instance of the same problem and is
# left for a change that can review its subject properly -- splitting them here
# would mean summarising code this one did not read.
#
# This list may SHRINK and must never grow: a new entry means a new design
# document was written into a docstring, which is the thing being prevented.
_KNOWN_OVERSIZED: frozenset[tuple[str, str]] = frozenset(
    {
        ("lightrag/api/routers/graph_routes.py", "update_entity"),
        ("lightrag/api/routers/query_routes.py", "query_text_stream"),
        ("lightrag/api/routers/query_routes.py", "query_data"),
        ("lightrag/base.py", "get_knowledge_graph"),
        ("lightrag/chunker/paragraph_semantic.py", "chunking_by_paragraph_semantic"),
        ("lightrag/lightrag.py", "aquery_data"),
        ("lightrag/llm/openai.py", "openai_complete_if_cache"),
        ("lightrag/tools/source_conflict_repair.py", "<module>"),
    }
)

# CSS colours in the inline WebUI template, not issue references.
_COLOUR_LINE = re.compile(
    r"(background|border-color|color)\s*:\s*#[0-9a-fA-F]{3,8}|greys \(#"
)
_ISSUE_REF = re.compile(r"#\d{3,5}\b")


def _python_files() -> list[pathlib.Path]:
    return sorted(_PACKAGE.rglob("*.py"))


def _oversized(path: pathlib.Path) -> list[tuple[str, int, int]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        doc = ast.get_docstring(node, clean=False)
        if not doc:
            continue
        limit = (
            _MAX_CLASS_DOCSTRING_LINES
            if isinstance(node, ast.ClassDef)
            else _MAX_DOCSTRING_LINES
        )
        lines = doc.count("\n") + 1
        if lines > limit:
            found.append((getattr(node, "name", "<module>"), lines, limit))
    return found


def test_no_docstring_grows_into_a_design_document():
    offenders = []
    for path in _python_files():
        rel = str(path.relative_to(_REPO_ROOT))
        for name, lines, limit in _oversized(path):
            if (rel, name) in _KNOWN_OVERSIZED:
                continue
            offenders.append(f"{rel}:{name} — {lines} lines (limit {limit})")

    assert not offenders, (
        "These docstrings are long enough to be design documents:\n  "
        + "\n  ".join(offenders)
        + "\n\nMove the reasoning to docs/design/ and leave the rules, the "
        "obligations on callers, and a pointer. See "
        "docs/design/NetworkXSingleWriterContract.md for the shape."
    )


def test_the_known_oversized_list_does_not_go_stale():
    """An entry that no longer applies must be removed, not left to rot.

    A stale allowlist quietly re-permits the thing it was recording, so the
    list is only trustworthy while it is exactly the set of remaining cases.
    """
    still_oversized = set()
    for path in _python_files():
        rel = str(path.relative_to(_REPO_ROOT))
        for name, _lines, _limit in _oversized(path):
            still_oversized.add((rel, name))

    stale = sorted(_KNOWN_OVERSIZED - still_oversized)
    assert not stale, (
        "These are no longer oversized; drop them from _KNOWN_OVERSIZED:\n  "
        + "\n  ".join(f"{path}:{name}" for path, name in stale)
    )


def test_no_source_file_cites_a_github_issue():
    offenders = []
    for path in _python_files():
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if _COLOUR_LINE.search(line):
                continue
            if _ISSUE_REF.search(line):
                rel = path.relative_to(_REPO_ROOT)
                offenders.append(f"{rel}:{lineno}: {line.strip()[:90]}")

    assert not offenders, (
        "These lines cite a GitHub issue number:\n  "
        + "\n  ".join(offenders)
        + "\n\nAn issue number does not survive a fork. Name the thing instead "
        "('the two-channel fence'), or move the content into docs/design/ and "
        "cite that."
    )


def test_every_documentation_path_named_in_the_package_exists():
    """A docstring pointer that does not resolve is worse than no pointer.

    The contracts are only useful if the code can send a reader to them, and a
    path typo is silent -- nothing imports these strings.
    """
    pattern = re.compile(r"docs/[A-Za-z0-9_/\-]+\.md")
    missing = []
    for path in _python_files():
        text = path.read_text(encoding="utf-8")
        for match in sorted(set(pattern.findall(text))):
            if not (_REPO_ROOT / match).is_file():
                missing.append(f"{path.relative_to(_REPO_ROOT)} -> {match}")

    assert not missing, "Documentation paths that do not resolve:\n  " + "\n  ".join(
        missing
    )
