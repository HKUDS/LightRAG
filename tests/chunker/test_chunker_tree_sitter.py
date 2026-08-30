"""Unit tests for ``chunking_by_tree_sitter`` (process_options=T)."""

import pytest

pytest.importorskip("tree_sitter")
pytest.importorskip("tree_sitter_python")

from lightrag.chunker import chunking_by_tree_sitter  # noqa: E402
from lightrag.utils import Tokenizer, TokenizerInterface  # noqa: E402


class _CharTokenizer(TokenizerInterface):
    """1 char ≈ 1 token; lets assertions reason in terms of input length."""

    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)


def _tok() -> Tokenizer:
    return Tokenizer("char-tokenizer", _CharTokenizer())


def _assert_spans_reconstruct(content: str, chunks: list[dict]) -> None:
    for chunk in chunks:
        span = chunk["_source_span"]
        assert content[span["start"] : span["end"]] == chunk["content"]


@pytest.mark.offline
def test_empty_input_returns_empty_list():
    chunks = chunking_by_tree_sitter(_tok(), "", file_path="a.py")
    assert chunks == []


@pytest.mark.offline
def test_unsupported_language_falls_back_to_fixed_token():
    # No file_path and no language override: nothing to resolve a
    # language from, so this degrades to plain token-window chunking
    # rather than raising.
    body = "just some plain text, not any programming language"
    chunks = chunking_by_tree_sitter(_tok(), body, chunk_token_size=1000)

    assert len(chunks) == 1
    assert chunks[0]["content"] == body
    _assert_spans_reconstruct(body, chunks)


@pytest.mark.offline
def test_unrecognized_language_override_warns_and_falls_back(monkeypatch):
    # An explicit override that doesn't match a known alias is a likely
    # typo, unlike an unrecognized extension -- so it should be visible at
    # warning level, not silently absorbed at debug. ``caplog`` cannot see
    # this: ``lightrag.utils.logger`` sets ``propagate = False`` (see
    # test_recursive_character_bounds.py), so intercept the logger directly.
    import lightrag.chunker.tree_sitter_code as mod

    warnings: list[str] = []
    monkeypatch.setattr(
        mod.logger,
        "warning",
        lambda msg, *a, **k: warnings.append(msg % a if a else msg),
    )

    chunks = chunking_by_tree_sitter(
        _tok(), "def foo():\n    pass\n", 1200, language="phyton"
    )

    assert len(chunks) == 1
    assert any("phyton" in w for w in warnings)


@pytest.mark.offline
def test_unrecognized_extension_falls_back_to_fixed_token():
    body = "fn main() {}"
    chunks = chunking_by_tree_sitter(
        _tok(), body, chunk_token_size=1000, file_path="main.rs"
    )

    assert len(chunks) == 1
    assert chunks[0]["content"] == body


@pytest.mark.offline
def test_python_splits_along_function_and_class_boundaries():
    code = (
        "import os\n"
        "import sys\n"
        "\n"
        "def foo(a, b):\n"
        "    return a + b\n"
        "\n"
        "class Bar:\n"
        "    def method(self):\n"
        "        pass\n"
        "\n"
        "x = 1\n"
    )
    chunks = chunking_by_tree_sitter(_tok(), code, 1200, file_path="module.py")

    contents = [c["content"] for c in chunks]
    assert any(c.startswith("import os") for c in contents)
    assert any(c.startswith("def foo") for c in contents)
    assert any(c.startswith("class Bar") and "def method" in c for c in contents)
    assert any(c == "x = 1" for c in contents)
    # Order is preserved and contiguous.
    assert [c["chunk_order_index"] for c in chunks] == list(range(len(chunks)))
    _assert_spans_reconstruct(code, chunks)


@pytest.mark.offline
def test_python_decorators_stay_with_their_function_and_class():
    # tree-sitter wraps a decorated def/class in a decorated_definition node;
    # without matching that node type too, the decorator would be emitted as
    # a separate filler chunk, detached from the function/class it modifies.
    code = (
        "import functools\n"
        "\n"
        "@functools.wraps\n"
        "def foo():\n"
        "    pass\n"
        "\n"
        "@property\n"
        "class Bar:\n"
        "    pass\n"
    )
    chunks = chunking_by_tree_sitter(_tok(), code, 1200, file_path="m.py")

    contents = [c["content"] for c in chunks]
    assert any(c == "@functools.wraps\ndef foo():\n    pass" for c in contents)
    assert any(c == "@property\nclass Bar:\n    pass" for c in contents)
    _assert_spans_reconstruct(code, chunks)


@pytest.mark.offline
def test_python_explicit_language_override_wins_over_extension():
    chunks = chunking_by_tree_sitter(
        _tok(),
        "def foo():\n    pass\n",
        1200,
        language="python",
        file_path="snippet.txt",
    )

    assert len(chunks) == 1
    assert chunks[0]["content"] == "def foo():\n    pass"


@pytest.mark.offline
def test_python_language_alias_is_accepted():
    chunks = chunking_by_tree_sitter(
        _tok(), "def foo():\n    pass\n", 1200, language="py"
    )

    assert len(chunks) == 1
    assert chunks[0]["content"].startswith("def foo")


@pytest.mark.offline
def test_syntax_error_falls_back_to_fixed_token_for_the_whole_file():
    # Unclosed parenthesis -- tree-sitter's error recovery cannot produce a
    # trustworthy top-level split, so the whole document degrades to
    # token-window chunking rather than mixing structural and fallback
    # chunks from a tree it doesn't trust.
    code = "def foo(:\n    pass\n"
    chunks = chunking_by_tree_sitter(_tok(), code, 1200, file_path="broken.py")

    assert len(chunks) == 1
    assert chunks[0]["content"] == code.strip()


@pytest.mark.offline
def test_oversized_function_is_sub_split_by_token_window():
    body = "\n".join(f"    x{i} = {i}" for i in range(200))
    code = f"def big():\n{body}\n"
    chunks = chunking_by_tree_sitter(
        _tok(),
        code,
        chunk_token_size=50,
        chunk_overlap_token_size=5,
        file_path="big.py",
    )

    assert len(chunks) > 1
    assert all(c["tokens"] <= 50 for c in chunks)
    assert [c["chunk_order_index"] for c in chunks] == list(range(len(chunks)))
    _assert_spans_reconstruct(code, chunks)


@pytest.mark.offline
def test_token_field_matches_tokenizer_encode_length():
    code = "def foo():\n    return 1\n\nclass Bar:\n    pass\n"
    chunks = chunking_by_tree_sitter(_tok(), code, 1200, file_path="m.py")
    tok = _tok()
    for c in chunks:
        assert c["tokens"] == len(tok.encode(c["content"]))


@pytest.mark.offline
def test_javascript_splits_functions_and_classes():
    pytest.importorskip("tree_sitter_javascript")
    code = (
        "const a = 1;\n"
        "function foo(x) {\n"
        "  return x + 1;\n"
        "}\n"
        "class Bar {\n"
        "  method() { return 1; }\n"
        "}\n"
    )
    chunks = chunking_by_tree_sitter(_tok(), code, 1200, file_path="app.js")

    contents = [c["content"] for c in chunks]
    assert any(c.startswith("const a") for c in contents)
    assert any(c.startswith("function foo") for c in contents)
    assert any(c.startswith("class Bar") for c in contents)
    _assert_spans_reconstruct(code, chunks)


@pytest.mark.offline
def test_typescript_splits_interfaces_and_functions():
    pytest.importorskip("tree_sitter_typescript")
    code = "interface Foo {\n  x: number;\n}\nfunction bar(): void {}\n"
    chunks = chunking_by_tree_sitter(_tok(), code, 1200, file_path="app.ts")

    contents = [c["content"] for c in chunks]
    assert any(c.startswith("interface Foo") for c in contents)
    assert any(c.startswith("function bar") for c in contents)


@pytest.mark.offline
def test_falls_back_when_tree_sitter_package_is_unavailable(monkeypatch):
    import lightrag.chunker.tree_sitter_code as mod

    monkeypatch.setattr(mod, "_TREE_SITTER_AVAILABLE", False)
    chunks = mod.chunking_by_tree_sitter(
        _tok(), "def foo():\n    pass\n", 1200, file_path="a.py"
    )

    assert len(chunks) == 1
    assert chunks[0]["content"] == "def foo():\n    pass"


@pytest.mark.offline
def test_falls_back_when_grammar_for_the_language_is_missing(monkeypatch):
    import lightrag.chunker.tree_sitter_code as mod

    monkeypatch.setattr(mod, "_load_grammar", lambda language: None)
    chunks = mod.chunking_by_tree_sitter(
        _tok(), "def foo():\n    pass\n", 1200, file_path="a.py"
    )

    assert len(chunks) == 1
    assert chunks[0]["content"] == "def foo():\n    pass"


@pytest.mark.offline
def test_tsx_extension_is_recognized():
    pytest.importorskip("tree_sitter_typescript")
    chunks = chunking_by_tree_sitter(
        _tok(), "function bar(): void {}\n", 1200, file_path="app.tsx"
    )

    assert len(chunks) == 1
    assert chunks[0]["content"].startswith("function bar")
