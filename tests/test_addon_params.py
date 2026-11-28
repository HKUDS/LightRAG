"""Tests for observable runtime addon parameters."""

from typing import Any

import pytest

from lightrag.addon_params import ObservableAddonParams, normalize_addon_params

pytestmark = pytest.mark.offline


def _ior(params: ObservableAddonParams, other: Any) -> None:
    original = params
    params |= other
    # ``|=`` rebinds the caller's name to whatever ``__ior__`` returns.
    assert params is original


def _delete(params: ObservableAddonParams, key: str) -> None:
    del params[key]


# (operation, mapping afterwards, notifications). Every no-op row starts from
# the same mapping as a writing row of the same method, so a missing short
# circuit and a missing notification are both visible.
_MUTATIONS = [
    pytest.param(
        lambda p: p.update([("language", "French")], entity_types=["person"]),
        {"language": "French", "entity_types": ["person"]},
        1,
        id="update",
    ),
    pytest.param(lambda p: p.update({}), {"language": "English"}, 0, id="update-empty"),
    pytest.param(
        lambda p: _ior(p, {"language": "French"}), {"language": "French"}, 1, id="ior"
    ),
    pytest.param(lambda p: _ior(p, []), {"language": "English"}, 0, id="ior-empty"),
    pytest.param(lambda p: _delete(p, "language"), {}, 1, id="delitem"),
    # The second clear() runs on an empty mapping: it must not notify again.
    pytest.param(lambda p: (p.clear(), p.clear()), {}, 1, id="clear-twice"),
    pytest.param(lambda p: p.pop("language"), {}, 1, id="pop"),
    pytest.param(
        lambda p: p.pop("missing", None), {"language": "English"}, 0, id="pop-missing"
    ),
    pytest.param(lambda p: p.popitem(), {}, 1, id="popitem"),
    pytest.param(
        lambda p: p.setdefault("entity_types", ["person"]),
        {"language": "English", "entity_types": ["person"]},
        1,
        id="setdefault",
    ),
    pytest.param(
        lambda p: p.setdefault("language", "French"),
        {"language": "English"},
        0,
        id="setdefault-existing",
    ),
]


@pytest.mark.parametrize(("mutate", "expected", "notifications"), _MUTATIONS)
def test_every_write_notifies_once_and_a_no_op_never(
    mutate, expected: dict[str, Any], notifications: int
) -> None:
    changes: list[str] = []
    params = ObservableAddonParams(
        {"language": "English"}, on_change=lambda: changes.append("changed")
    )

    mutate(params)

    assert params == expected
    assert len(changes) == notifications


def test_update_is_atomic_when_an_iterable_is_malformed() -> None:
    changes: list[str] = []

    def unexpected_change() -> None:
        changes.append("changed")
        raise AssertionError("failed updates must not trigger callbacks")

    params = ObservableAddonParams({"language": "English"}, on_change=unexpected_change)

    def partly_invalid_items():
        yield ("language", "French")
        yield ("malformed",)

    # Match on the exception type only: the "dictionary update sequence
    # element #1 ..." wording is a CPython implementation detail, and what this
    # regression is about is the mapping's state, not the message.
    with pytest.raises(ValueError):
        params.update(partly_invalid_items())

    # A failed live-config update must not leave the mapping ahead of its
    # derived cache or emit a change notification for an update that failed.
    assert params == {"language": "English"}
    assert changes == []


def test_update_does_not_notify_when_validation_fails_before_a_write() -> None:
    changes: list[str] = []
    params = ObservableAddonParams(
        {"language": "English"}, on_change=lambda: changes.append("changed")
    )

    with pytest.raises(TypeError):
        params.update(42)

    assert params == {"language": "English"}
    assert changes == []


def test_ior_is_atomic_when_an_iterable_is_malformed() -> None:
    """``|=`` takes iterables too (PEP 584), so it needs update()'s guarantee.

    ``dict.__ior__`` is not restricted to the ``Mapping`` its signature
    advertises, so without materializing first it applies leading pairs and then
    raises, skipping the change notification exactly like ``update`` did.
    """
    changes: list[str] = []

    def unexpected_change() -> None:
        changes.append("changed")
        raise AssertionError("failed updates must not trigger callbacks")

    params = ObservableAddonParams({"language": "English"}, on_change=unexpected_change)

    def partly_invalid_items():
        yield ("language", "French")
        yield ("malformed",)

    with pytest.raises(ValueError):
        params |= partly_invalid_items()

    assert params == {"language": "English"}
    assert changes == []


def _addon_params_change_observer() -> Any:
    """A minimal stand-in carrying only LightRAG's addon-params cache hooks.

    Constructing a real ``LightRAG`` needs storages, an LLM func and an
    embedding func; the invalidation contract under test is just these two
    methods plus the two attributes they touch.
    """
    from lightrag.lightrag import LightRAG

    class _CacheOwner:
        _on_addon_params_changed = LightRAG._on_addon_params_changed
        _mark_addon_params_dirty = LightRAG._mark_addon_params_dirty

        def __init__(self) -> None:
            self._addon_params_dirty = False
            self._addon_params = ObservableAddonParams(
                {"chunker": {}}, on_change=self._on_addon_params_changed
            )

    return _CacheOwner()


def test_cache_is_invalidated_even_when_the_change_callback_raises() -> None:
    """A raising callback must not leave the mapping ahead of a clean cache.

    ``separators`` holding non-strings reaches ``len()`` on an ``int`` inside
    ``inspect_r_separators``. The assignment itself has already committed by
    then, so skipping the dirty mark would keep the stale summary language and
    prompt profile alive until some unrelated later mutation happened to
    invalidate them.
    """
    owner = _addon_params_change_observer()
    malformed = {"recursive_character": {"separators": [1, 2]}}

    with pytest.raises(TypeError):
        owner._addon_params["chunker"] = malformed

    assert owner._addon_params["chunker"] == malformed
    assert owner._addon_params_dirty is True


def test_chunk_size_overlay_does_not_mutate_supplied_addon_params() -> None:
    """Resolving runtime defaults must not write into the caller's config.

    A shared mapping reused for a second instance must not carry the first
    instance's resolved sizes, which would otherwise outrank the second
    instance's own ``chunk_token_size`` as an explicit addon_params value.
    """
    from lightrag.lightrag import LightRAG

    supplied = {"chunker": {"fixed_token": {"split_by_character": "|"}}}

    class _OverlayOwner:
        _apply_chunk_size_overlay = LightRAG._apply_chunk_size_overlay

        def __init__(self, chunk_token_size: int) -> None:
            self._addon_params = normalize_addon_params(supplied)
            self.chunk_token_size = chunk_token_size
            self.chunk_overlap_token_size = None

    first = _OverlayOwner(500)
    first._apply_chunk_size_overlay()
    second = _OverlayOwner(2000)
    second._apply_chunk_size_overlay()

    assert supplied == {"chunker": {"fixed_token": {"split_by_character": "|"}}}
    assert (first.chunk_token_size, second.chunk_token_size) == (500, 2000)
    first._addon_params["chunker"]["fixed_token"]["split_by_character"] = "#"
    assert second._addon_params["chunker"]["fixed_token"]["split_by_character"] == "|"


@pytest.mark.parametrize(
    "supplied", [None, {"chunker": {}}], ids=["defaults", "explicit"]
)
def test_env_defaults_backfill_default_and_explicit_addon_params(
    supplied, monkeypatch
) -> None:
    """``LightRAG(addon_params=...)`` skips the defaults factory, so the
    explicit branch must still pick the environment up."""
    monkeypatch.setenv("SUMMARY_LANGUAGE", "Catalan")
    monkeypatch.setenv("ENTITY_TYPE_PROMPT_FILE", "from_env.yml")

    normalized = normalize_addon_params(supplied)

    assert normalized["language"] == "Catalan"
    assert normalized["entity_type_prompt_file"] == "from_env.yml"
