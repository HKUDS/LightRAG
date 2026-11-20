"""Load single top-level definitions out of the example scripts."""

import ast
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"


def load_demo_definition(demo: str, name: str, namespace: dict):
    """Execute only the top-level definition ``name`` of ``examples/<demo>``.

    Importing a demo would validate API keys, create directories, apply
    ``nest_asyncio`` or configure logging, so the definition is compiled on its
    own. ``namespace`` must stub every global the definition reads.
    """
    path = EXAMPLES_DIR / demo
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )
    exec(  # noqa: S102 - trusted repository source, see the docstring
        compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace[name]
