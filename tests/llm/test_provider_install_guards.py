"""A pipmaster install guard must precede the import it protects.

Provider modules under ``lightrag/llm/`` install their SDK on demand::

    if not pm.is_installed("ollama"):
        pm.install("ollama")

    import ollama

The order is the whole mechanism. Placed *after* the import, the guard is
dead code: the import raises ``ModuleNotFoundError`` first, so the module is
unimportable on exactly the machines the guard exists for. That is what
``llama_index_impl`` did — its guard sat ten lines below the
``from llama_index.core.llms import ...`` that shadowed it — and the symptom
was a hard collection error for the whole test run rather than an install.

A property of the tree rather than of any one change, so it is pinned here
rather than left to review: a new provider module copied from a sibling
inherits the right order, one written from scratch may not.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

pytestmark = pytest.mark.offline

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_LLM_PACKAGE = _REPO_ROOT / "lightrag" / "llm"
_DISTRIBUTION_IMPORT_NAMES = {
    "google-api-core": "google.api_core",
    "google-genai": "google.genai",
}


def _distribution_import_name(package_spec: str) -> str:
    """Return the import name supplied by a pip distribution specification."""
    distribution = package_spec.partition("[")[0]
    return _DISTRIBUTION_IMPORT_NAMES.get(distribution, distribution.replace("-", "_"))


def _installed_packages(tree: ast.AST) -> list[tuple[str, str, int]]:
    """Return ``(distribution spec, import name, line)`` for install calls.

    The list deliberately retains separate distributions that share a namespace,
    such as ``google-genai`` and ``google-api-core``.
    """
    installs: list[tuple[str, str, int]] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "install"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            package_spec = node.args[0].value
            installs.append(
                (
                    package_spec,
                    _distribution_import_name(package_spec),
                    node.lineno,
                )
            )
    return installs


def _first_import_line(tree: ast.AST, package: str) -> int | None:
    """Line of the first ``import <package>`` / ``from <package> import ...``.

    A package matches itself or a child module. ``from google import genai``
    therefore counts as importing ``google.genai`` as well.
    """
    first: int | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            # ``level`` > 0 is a relative import; it names no third-party package.
            if node.module and not node.level:
                names = [node.module]
                names.extend(
                    f"{node.module}.{alias.name}"
                    for alias in node.names
                    if alias.name != "*"
                )
            else:
                names = []
        else:
            continue
        if any(
            name == package or name.startswith(f"{package}.") for name in names
        ) and (first is None or node.lineno < first):
            first = node.lineno
    return first


def _provider_modules() -> list[pathlib.Path]:
    return sorted(p for p in _LLM_PACKAGE.glob("*.py") if p.name != "__init__.py")


def test_the_llm_package_is_where_it_is_expected():
    """Guards the scan itself: an empty glob would pass every assertion."""
    modules = _provider_modules()
    assert len(modules) >= 5, f"only found {len(modules)} modules in {_LLM_PACKAGE}"


@pytest.mark.parametrize(
    ("package_spec", "import_name"),
    [
        ("llama-index", "llama_index"),
        ("lmdeploy[all]", "lmdeploy"),
        ("google-genai", "google.genai"),
        ("google-api-core", "google.api_core"),
    ],
)
def test_distribution_specs_resolve_to_their_import_names(
    package_spec: str, import_name: str
):
    assert _distribution_import_name(package_spec) == import_name


def test_from_namespace_import_is_matched_to_the_installed_module():
    tree = ast.parse("from google import genai\n")
    assert _first_import_line(tree, "google.genai") == 1


def test_install_guards_sharing_a_namespace_remain_independent():
    tree = ast.parse('pm.install("google-genai")\npm.install("google-api-core")\n')
    assert _installed_packages(tree) == [
        ("google-genai", "google.genai", 1),
        ("google-api-core", "google.api_core", 2),
    ]


def test_every_install_guard_precedes_the_import_it_protects():
    offenders: list[str] = []
    guarded: list[str] = []
    for path in _provider_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for package_spec, package, install_line in _installed_packages(tree):
            import_line = _first_import_line(tree, package)
            if import_line is None:
                continue
            rel = path.relative_to(_REPO_ROOT)
            guarded.append(f"{rel}:{package}")
            if import_line < install_line:
                offenders.append(
                    f"{rel} imports {package} at line {import_line}, but "
                    f"pm.install({package_spec!r}) is at line {install_line} — the "
                    f"guard never runs"
                )

    # The scan must actually be finding guards; a refactor that renames
    # pm.install would otherwise turn this test into a no-op that still passes.
    assert guarded, "no pipmaster install guards found in lightrag/llm/"
    assert not offenders, "install guard placed after its import:\n" + "\n".join(
        offenders
    )
