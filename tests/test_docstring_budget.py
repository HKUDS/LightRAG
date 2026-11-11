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
# ``_KNOWN_OVERSIZED_SIZE`` below is what makes that mechanical.
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

# The list's own length, asserted below. Without it the allowlist is an
# unguarded opt-out: the limit test SKIPS anything listed, so appending one
# tuple silences a fresh 160-line docstring and every test still passes --
# measured, not assumed. Nothing in a repository can stop a committer from
# editing two lines instead of one; what this buys is that growing the list
# cannot happen as a quiet append, because it also has to move a number whose
# comment says what moving it means.
_KNOWN_OVERSIZED_SIZE = 8


# A BARE ``#NNNN``, which resolves against whatever repository the reader
# happens to be in. Two deliberate narrowings, both about keeping the rule
# simple enough to be reliable rather than maximally broad:
#
# * **Qualified references are exempt** (the ``[\w/]`` lookbehind).
#   ``HKUDS/RAG-Anything#73`` names its repository, so it resolves from a fork
#   exactly as it does here -- it is a real citation, and the thing this rule
#   protects against is a referent that disappears, not a ``#`` character.
# * **Four to five digits.** This repository's issue numbers passed 1000 long
#   ago and every one of the 214 references removed in this series was
#   four-digit, so a shorter number is not a reference here and a rule that
#   chased one would only add false positives. The upper bound is what keeps
#   CSS hex colours out without a content heuristic: every colour in the tree
#   is six hex digits (``#020617``, ``#111827``), a length no issue number
#   will reach for a very long time, and five digits is not a valid colour at
#   all. An earlier revision matched 3-5 digits and needed a fragile
#   "is this line a CSS declaration?" test alongside it; the bound alone is
#   both simpler and stricter.
_ISSUE_REF = re.compile(r"(?<![\w/])#\d{4,5}\b")


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


def test_the_known_oversized_list_is_a_ratchet():
    """Growing the allowlist must be a deliberate edit, not a list append.

    The staleness test above makes entries removable; it does nothing about
    adding them, and the limit test skips whatever is listed. So the
    "may shrink, never grow" rule was a promise in a comment: one appended
    tuple opts a new design-document docstring out of the budget with every
    test still green.
    """
    assert len(_KNOWN_OVERSIZED) == _KNOWN_OVERSIZED_SIZE, (
        f"_KNOWN_OVERSIZED holds {len(_KNOWN_OVERSIZED)} entries but "
        f"_KNOWN_OVERSIZED_SIZE says {_KNOWN_OVERSIZED_SIZE}.\n\n"
        "Shrinking the list: lower the number, and thank you.\n"
        "Growing it: you are opting a docstring out of the budget rather than "
        "moving its reasoning to docs/design/. Do that instead — and if the "
        "entry is genuinely unavoidable, raise the number in the same commit "
        "so the opt-out is visible in the diff."
    )


def test_no_source_file_cites_a_github_issue():
    offenders = []
    for path in _python_files():
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if _ISSUE_REF.search(line):
                rel = path.relative_to(_REPO_ROOT)
                offenders.append(f"{rel}:{lineno}: {line.strip()[:90]}")

    assert not offenders, (
        "These lines cite a GitHub issue number:\n  "
        + "\n  ".join(offenders)
        + "\n\nA bare issue number does not survive a fork. Name the thing "
        "instead ('the two-channel fence'), or move the content into "
        "docs/design/ and cite that. A reference qualified by its repository "
        "('HKUDS/RAG-Anything#73') is fine and is not matched."
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


@pytest.mark.parametrize(
    "line, matched, why",
    [
        ("the gate (issue #3899) raises", True, "bare four-digit reference"),
        ("see #12345 for the detail", True, "bare five-digit reference"),
        ("# NOTE(#3609): truthiness is kept", True, "bare, inside parentheses"),
        (
            "its record is intact (HKUDS/RAG-Anything#73).",
            False,
            "qualified by repository: resolves from a fork",
        ),
        (
            "see HKUDS/RAG-Anything#4567 for the upstream fix",
            False,
            "qualified, and four digits: still not a bare reference",
        ),
        ("            background: #020617;", False, "six-digit CSS colour"),
        ("            border-color: #334155;", False, "CSS colour with letters"),
        ("greys (#505050 / #3b4151) on every title", False, "colours in prose"),
        ("# step #1 of the flow", False, "an ordinal, not a reference"),
    ],
)
def test_what_counts_as_a_bare_issue_reference(line, matched, why):
    """The boundaries of ``_ISSUE_REF``, each with the case it stands for.

    A detector for this has to be simple enough to trust, which means its
    exclusions are decisions rather than accidents: a qualified reference is
    exempt because it survives a fork, and a six-digit number is a colour
    because this repository's issue numbers will not reach that length for a
    long time. Both are load-bearing and neither is visible from the pattern.
    """
    assert bool(_ISSUE_REF.search(line)) is matched, why


# A pointer written as ``see *Section name* in the contract doc``. The phrase is
# a convention this repository introduced along with docs/design/, so matching
# on it is unambiguous -- it is never ordinary prose.
#
# ``(?<!\*)`` / ``(?!\*)`` keep **bold emphasis** out. These docstrings use one
# asterisk for a section reference and two for emphasis, and without the guards
# the inner pair of ``**Not pipeline-gated**`` reads as a section name -- a false
# positive on a line that also carries a perfectly good pointer beside it.
_CONTRACT_POINTER = re.compile(
    r"(?<!\*)\*([A-Z][^*\n]{2,60}?)\*(?!\*)[^.]{0,80}?contract doc|"
    r"contract doc[^.]{0,80}?(?<!\*)\*([A-Z][^*\n]{2,60}?)\*(?!\*)"
)


def _flattened(path: pathlib.Path) -> str:
    """File text with wrapped comment and docstring lines joined.

    Pointers wrap. Five of them rotted through a rename in this series and a
    line-oriented search found none of them, because every one was split as
    ``in the class`` / ``docstring`` across two lines.
    """
    joined = re.sub(r"\n\s*(#\s*)?", " ", path.read_text(encoding="utf-8"))
    return re.sub(r"\s+", " ", joined)


def test_every_contract_section_pointer_resolves():
    """``see *X* in the contract doc`` must name something a reader can find.

    The paths were already checked; the SECTION names were not, and a rename
    inside a contract leaves the pointer looking valid while sending the reader
    nowhere. That is worse than no pointer: it reads as a promise the tree does
    not keep.
    """
    # HEADINGS, not the whole text. A first attempt matched anywhere in the
    # document and was useless: renaming a heading leaves the old wording in the
    # prose that cross-references it, so the pointer still "resolved" while
    # pointing at a section that no longer exists. A pointer names a section, so
    # it is sections it has to be checked against.
    #
    # Scoped to the contract the FILE names, not to every contract at once. A
    # module cites one document -- "the contract doc" is shorthand for the path
    # in its own docstring -- so pooling the headings would let a pointer
    # resolve against a sibling contract that the reader is not being sent to.
    by_document = {
        path.name: {
            re.sub(r"[`*]", "", heading).strip().lower()
            for heading in re.findall(
                r"^#{2,4}\s+(.+)$", path.read_text(encoding="utf-8"), re.M
            )
        }
        for path in sorted((_REPO_ROOT / "docs" / "design").glob("*.md"))
    }
    assert by_document, "no contract documents found — the check would pass vacuously"

    dangling = []
    unnamed = []
    for path in _python_files():
        flat = _flattened(path)
        named = re.findall(r"docs/design/(\w+\.md)", flat)
        # A file that names NO contract path is the defect itself, not a case to
        # wave through. "The contract doc" is shorthand for a path the file is
        # expected to carry somewhere; without it the reader has nowhere to go.
        # An earlier version pooled every contract's headings as a fallback, and
        # that is exactly how ``json_doc_status_impl.py`` passed while saying
        # "see the contract doc" about a document that did not cover it at all.
        headings = set().union(*(by_document.get(doc, set()) for doc in named))

        for match in _CONTRACT_POINTER.finditer(flat):
            name = (match.group(1) or match.group(2)).strip().rstrip(".,;:")
            if not named:
                unnamed.append(f"{path.relative_to(_REPO_ROOT)} -> *{name}*")
            elif name.lower() not in headings:
                dangling.append(
                    f"{path.relative_to(_REPO_ROOT)} -> *{name}* "
                    f"(not in {'/'.join(sorted(set(named)))})"
                )

    assert not unnamed, (
        'These files point at *a section* of "the contract doc" without naming '
        "which document that is:\n  "
        + "\n  ".join(sorted(set(unnamed)))
        + "\n\nName the docs/design/<Name>.md path somewhere in the same file, "
        "so the pointer resolves for a reader and for this check."
    )
    assert not dangling, (
        "These pointers name a section their contract document does not "
        "contain:\n  "
        + "\n  ".join(sorted(set(dangling)))
        + "\n\nA renamed section leaves the pointer looking valid. Retarget it, "
        "or restore the name in docs/design/."
    )


# A parenthetical that opens with whitespace. In prose this is always a typo,
# and the shape it usually takes is a parenthetical whose subject was edited
# away -- "(issue #3400: unsafe" becoming "( unsafe", or worse, "(\n# Phase 3)",
# which keeps its parentheses balanced while naming nothing.
_GUTTED_PARENTHETICAL = re.compile(r"\(\s+[^()]{0,60}?\)")


def _prose(path: pathlib.Path) -> list[str]:
    """Comment blocks and docstrings, each flattened; no code.

    Code is excluded deliberately -- ``foo( x )`` is a formatting question ruff
    already owns, and mentions like ``vars()`` would otherwise dominate.
    """
    source = path.read_text(encoding="utf-8")
    blocks: list[str] = []
    run: list[str] = []
    for line in source.split("\n"):
        stripped = line.strip()
        if stripped.startswith("#"):
            run.append(stripped.lstrip("#").strip())
        elif run:
            blocks.append(" ".join(run))
            run = []
    if run:
        blocks.append(" ".join(run))

    for node in ast.walk(ast.parse(source)):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            doc = ast.get_docstring(node, clean=True)
            if doc:
                blocks.append(re.sub(r"\s+", " ", doc))
    return blocks


def test_no_prose_parenthetical_opens_with_whitespace():
    """Catches a parenthetical whose subject was edited away.

    Removing 214 issue references from comments produced this twice, and
    neither round of checks found it: a balance check passes (the parentheses
    are still matched) and a line-oriented search misses it (the damage
    straddles a line break). Flattening first, and looking at the shape rather
    than the count, is what catches it.
    """
    gutted = []
    for path in _python_files():
        for block in _prose(path):
            for match in _GUTTED_PARENTHETICAL.finditer(block):
                gutted.append(f"{path.relative_to(_REPO_ROOT)}: {match.group(0)!r}")

    assert not gutted, (
        "These parentheticals open with whitespace, which in prose means the "
        "subject was edited away:\n  "
        + "\n  ".join(sorted(set(gutted)))
        + "\n\nRestore what the parenthetical was naming, or drop it."
    )
