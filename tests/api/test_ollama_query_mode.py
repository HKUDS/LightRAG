"""parse_query_mode must honor the mode prefix in the bracket user-prompt form.

The bracket branch rebuilds the query as ``/{mode} {rest}``. The mode table it
is matched against uses space-suffixed keys, so dropping that separator (or
emitting a bare ``/`` when no mode was given) silently downgraded the search
mode to the default and leaked the prefix into the retrieval query text.
"""

import importlib
import sys

import pytest

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_ollama_api = importlib.import_module("lightrag.api.routers.ollama_api")
sys.argv = _original_argv

SearchMode = _ollama_api.SearchMode
parse_query_mode = _ollama_api.parse_query_mode

pytestmark = pytest.mark.offline


@pytest.mark.parametrize(
    "query,expected",
    [
        # Bracket prompt with a query: already worked, must keep working.
        (
            "/local[use mermaid] tell me about X",
            ("tell me about X", SearchMode.local, False, "use mermaid"),
        ),
        # Bracket prompt with no query text: the mode was lost and the literal
        # "/local" became the query.
        ("/local[use mermaid]", ("", SearchMode.local, False, "use mermaid")),
        ("/local[use mermaid]   ", ("", SearchMode.local, False, "use mermaid")),
        # No mode prefix: a stray "/ " was prepended to the user's question.
        (
            "/[use mermaid] tell me about X",
            ("tell me about X", SearchMode.mix, False, "use mermaid"),
        ),
        ("/[use mermaid]", ("", SearchMode.mix, False, "use mermaid")),
        # Context modes use unsuffixed keys, so they never depended on the
        # separator — pin them so the fix does not regress them.
        (
            "/localcontext[use mermaid] tell me about X",
            ("tell me about X", SearchMode.local, True, "use mermaid"),
        ),
        ("/mixcontext[use mermaid]", ("", SearchMode.mix, True, "use mermaid")),
        # An unknown mode keeps its existing pass-through behavior.
        (
            "/nosuchmode[use mermaid] tell me about X",
            ("/nosuchmode tell me about X", SearchMode.mix, False, "use mermaid"),
        ),
        # Non-bracket forms are untouched by the fix.
        ("/local  tell me about X", ("tell me about X", SearchMode.local, False, None)),
        ("/globalcontext X", ("X", SearchMode.global_, True, None)),
        ("tell me about X", ("tell me about X", SearchMode.mix, False, None)),
    ],
)
def test_parse_query_mode(query, expected):
    assert parse_query_mode(query) == expected


@pytest.mark.parametrize(
    "query,expected",
    [
        (
            "/local[be brief] line one\nline two",
            ("line one\nline two", SearchMode.local, False, "be brief"),
        ),
        (
            "/[be brief] line one\nline two",
            ("line one\nline two", SearchMode.mix, False, "be brief"),
        ),
    ],
)
def test_bracket_prompt_keeps_multiline_query(query, expected):
    """A multi-line question must survive the bracket form, as it does without it."""
    assert parse_query_mode(query) == expected
    # The non-bracket path is the reference behavior.
    assert parse_query_mode("/local line one\nline two")[0] == "line one\nline two"


@pytest.mark.parametrize("mode_prefix", ["local", "global", "naive", "hybrid", "mix"])
def test_bracket_prompt_without_query_keeps_mode(mode_prefix):
    """Every space-suffixed mode key survives an empty trailing query."""
    cleaned, mode, only_need_context, user_prompt = parse_query_mode(
        f"/{mode_prefix}[be brief]"
    )
    assert cleaned == ""
    assert mode.value == mode_prefix
    assert only_need_context is False
    assert user_prompt == "be brief"


@pytest.mark.parametrize(
    "query,expected",
    [
        # An English word starting with "context" must pass through as an
        # unknown mode, exactly like "/nosuchmode": the unsuffixed "/context"
        # key swallowed it, truncating the retrieval text and silently
        # flipping the query to context-only retrieval.
        (
            "/contextualize the following passage",
            ("/contextualize the following passage", SearchMode.mix, False, None),
        ),
        (
            "/contexts in which k8s is used",
            ("/contexts in which k8s is used", SearchMode.mix, False, None),
        ),
        # The legitimate unsuffixed forms keep matching: trailing space ...
        (
            "/context how does docker networking work",
            ("how does docker networking work", SearchMode.mix, True, None),
        ),
        # ... the bare key with nothing after it ...
        ("/context", ("", SearchMode.mix, True, None)),
        # ... and sibling whole-word keys.
        ("/localcontext foo", ("foo", SearchMode.local, True, None)),
    ],
)
def test_unsuffixed_context_keys_end_at_a_word_boundary(query, expected):
    """An unsuffixed mode key must not prefix-match a longer word.

    Without a boundary check, "/context" also prefixes "/contextualize" and
    "/contexts": the matched fragment is cut out of the retrieval query text
    and the query is forced into context-only retrieval (no LLM answer) that
    the user never asked for.
    """
    assert parse_query_mode(query) == expected
