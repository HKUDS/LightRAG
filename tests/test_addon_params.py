"""Tests for observable runtime addon parameters."""

from typing import Any

import pytest

from lightrag.addon_params import ObservableAddonParams

pytestmark = pytest.mark.offline


def test_update_commits_valid_input_and_notifies_once() -> None:
    changes: list[str] = []
    params = ObservableAddonParams(
        {"language": "English"}, on_change=lambda: changes.append("changed")
    )

    params.update([("language", "French")], entity_types=["person"])

    assert params == {"language": "French", "entity_types": ["person"]}
    assert changes == ["changed"]


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


def test_ior_commits_a_mapping_and_notifies_once() -> None:
    changes: list[str] = []
    params = ObservableAddonParams(
        {"language": "English"}, on_change=lambda: changes.append("changed")
    )

    params |= {"language": "French"}

    assert params == {"language": "French"}
    assert changes == ["changed"]


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


def test_cache_is_invalidated_on_a_normal_chunker_replacement() -> None:
    owner = _addon_params_change_observer()

    owner._addon_params["chunker"] = {"recursive_character": {"separators": ["\n"]}}

    assert owner._addon_params_dirty is True
