"""Tests for the `/documents/text(s)` ``chunking`` request object.

Three concerns:

1. **Synchronous validation**: malformed ``chunking`` is rejected at
   request-parse time (HTTP 422 / ``ValidationError``) — never deferred to
   the background indexing task, where the HTTP response is already sent.
   The per-strategy typed params models do full type + value checking, not
   just unknown-key detection.

2. **``_resolve_text_chunking``**: a validated ``chunking`` config is frozen
   into ``(process_options, chunk_options)``; ``chunk_token_size`` and the
   strategy params land in the selected strategy's sub-dict, overriding any
   env-derived value, while the other strategy sub-dicts are dropped (slim).

3. **Route forwarding**: ``/documents/text`` and ``/documents/texts`` forward
   ``request.chunking`` to ``pipeline_index_texts`` and return 422 (without
   scheduling any background work) for a malformed body.
"""

import importlib
import sys
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_dr = importlib.import_module("lightrag.api.routers.document_routes")
sys.argv = _original_argv

TextChunkingConfig = _dr.TextChunkingConfig
InsertTextRequest = _dr.InsertTextRequest
_resolve_text_chunking = _dr._resolve_text_chunking
create_document_routes = _dr.create_document_routes

from lightrag.constants import (  # noqa: E402
    PROCESS_OPTION_CHUNK_FIXED,
    PROCESS_OPTION_CHUNK_PARAGRAH,
    PROCESS_OPTION_CHUNK_RECURSIVE,
    PROCESS_OPTION_CHUNK_VECTOR,
)
from lightrag.parser.routing import default_chunker_config  # noqa: E402

pytestmark = pytest.mark.offline

_ALL_STRATEGY_KEYS = {
    "fixed_token",
    "recursive_character",
    "semantic_vector",
    "paragraph_semantic",
}


# ---------------------------------------------------------------------------
# 1. Synchronous validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        # wrong types (strict rejects lax coercion)
        {"strategy": "fixed_token", "params": {"chunk_token_size": True}},
        {"strategy": "fixed_token", "params": {"chunk_token_size": "5"}},
        {"strategy": "fixed_token", "params": {"chunk_token_size": 1.5}},
        {"strategy": "fixed_token", "params": {"chunk_overlap_token_size": "bad"}},
        {"strategy": "fixed_token", "params": {"split_by_character": 123}},
        {"strategy": "fixed_token", "params": {"split_by_character_only": 1}},
        {"strategy": "recursive_character", "params": {"separators": "abc"}},
        {"strategy": "recursive_character", "params": {"separators": [1, 2]}},
        # value / range
        {"strategy": "fixed_token", "params": {"chunk_token_size": 0}},
        {"strategy": "recursive_character", "params": {"chunk_overlap_token_size": -1}},
        {"strategy": "semantic_vector", "params": {"buffer_size": 0}},
        {"strategy": "semantic_vector", "params": {"buffer_size": True}},
        {
            "strategy": "semantic_vector",
            "params": {"breakpoint_threshold_type": "p99"},
        },
        {
            "strategy": "semantic_vector",
            "params": {"breakpoint_threshold_amount": 0},
        },
        {
            # strict float rejects strings (no lax numeric-string coercion)
            "strategy": "semantic_vector",
            "params": {"breakpoint_threshold_amount": "95"},
        },
        {
            # strict float rejects bool (bool is an int subclass, undesirable here)
            "strategy": "semantic_vector",
            "params": {"breakpoint_threshold_amount": True},
        },
        {
            # > 100 with an explicit percentile/gradient type is rejected at
            # parse time (both fields present, no inheritance ambiguity).
            "strategy": "semantic_vector",
            "params": {
                "breakpoint_threshold_type": "percentile",
                "breakpoint_threshold_amount": 150,
            },
        },
        {
            # malformed regex must be compiled/rejected at parse time
            "strategy": "semantic_vector",
            "params": {"sentence_split_regex": "("},
        },
        # cross-field
        {
            "strategy": "fixed_token",
            "params": {"chunk_token_size": 100, "chunk_overlap_token_size": 200},
        },
        # unknown / wrong-for-strategy keys
        {"strategy": "fixed_token", "params": {"bogus": 1}},
        {"strategy": "fixed_token", "params": {"separators": ["x"]}},
        {"strategy": "recursive_character", "params": {"buffer_size": 1}},
    ],
)
def test_chunking_config_rejects_malformed(body):
    with pytest.raises(ValidationError):
        TextChunkingConfig.model_validate(body)


def test_chunking_config_defaults_to_fixed_token():
    cfg = TextChunkingConfig.model_validate({"params": {"chunk_token_size": 500}})
    assert cfg.strategy == "fixed_token"
    assert cfg.params == {"chunk_token_size": 500}


def test_chunking_config_normalizes_to_supplied_keys_only():
    # int amount is coerced to float; only the supplied key survives.
    cfg = TextChunkingConfig.model_validate(
        {"strategy": "semantic_vector", "params": {"breakpoint_threshold_amount": 95}}
    )
    assert cfg.params == {"breakpoint_threshold_amount": 95.0}


def test_chunking_config_amount_in_range_for_std_deviation():
    # standard_deviation only requires > 0 (no [0, 100] ceiling).
    cfg = TextChunkingConfig.model_validate(
        {
            "strategy": "semantic_vector",
            "params": {
                "breakpoint_threshold_type": "standard_deviation",
                "breakpoint_threshold_amount": 3.5,
            },
        }
    )
    assert cfg.params["breakpoint_threshold_amount"] == 3.5


def test_chunking_config_amount_over_100_without_type_is_deferred():
    # Type omitted -> the (0, 100] ceiling cannot be decided at parse time
    # (the effective type may be inherited), so the model must NOT assume
    # percentile and reject. _resolve_text_chunking applies the ceiling later.
    cfg = TextChunkingConfig.model_validate(
        {"strategy": "semantic_vector", "params": {"breakpoint_threshold_amount": 150}}
    )
    assert cfg.params == {"breakpoint_threshold_amount": 150.0}


def test_chunking_config_accepts_int_amount_widened_to_float():
    # Strict float accepts an int (JSON 95) and widens it to 95.0 — the common
    # documented threshold magnitude. (str/bool are rejected; see the
    # rejection matrix above.) Exercised via both python and JSON validation
    # modes so the FastAPI request path (which parses JSON) stays covered.
    cfg = TextChunkingConfig.model_validate(
        {"strategy": "semantic_vector", "params": {"breakpoint_threshold_amount": 95}}
    )
    assert cfg.params == {"breakpoint_threshold_amount": 95.0}
    assert isinstance(cfg.params["breakpoint_threshold_amount"], float)

    cfg_json = TextChunkingConfig.model_validate_json(
        '{"strategy": "semantic_vector", "params": {"breakpoint_threshold_amount": 95}}'
    )
    assert cfg_json.params == {"breakpoint_threshold_amount": 95.0}


def test_chunking_config_accepts_valid_sentence_split_regex():
    cfg = TextChunkingConfig.model_validate(
        {
            "strategy": "semantic_vector",
            "params": {"sentence_split_regex": r"(?<=[.?!])\s+"},
        }
    )
    assert cfg.params == {"sentence_split_regex": r"(?<=[.?!])\s+"}


def test_chunking_config_drops_explicit_null():
    # Explicit null means "inherit the default" (every param field is
    # Optional/None=inherit), so it must be dropped — not merged over the
    # resolved default, which would later make the chunker do int(None).
    cfg = TextChunkingConfig.model_validate(
        {"strategy": "fixed_token", "params": {"chunk_token_size": None}}
    )
    assert cfg.params == {}


def test_chunking_config_keeps_real_value_drops_sibling_null():
    cfg = TextChunkingConfig.model_validate(
        {
            "strategy": "fixed_token",
            "params": {"chunk_token_size": 500, "split_by_character": None},
        }
    )
    assert cfg.params == {"chunk_token_size": 500}


def test_insert_text_request_rejects_malformed_chunking():
    with pytest.raises(ValidationError):
        InsertTextRequest.model_validate(
            {
                "text": "hi",
                "file_source": "a.md",
                "chunking": {
                    "strategy": "recursive_character",
                    "params": {"separators": "notalist"},
                },
            }
        )


# ---------------------------------------------------------------------------
# 2. _resolve_text_chunking
# ---------------------------------------------------------------------------


def _stub_rag(addon_params=None):
    return SimpleNamespace(
        addon_params=addon_params if addon_params is not None else {}
    )


def test_resolve_none_keeps_default_fixed():
    process_options, chunk_options = _resolve_text_chunking(None, _stub_rag())
    assert process_options == PROCESS_OPTION_CHUNK_FIXED
    assert "fixed_token" in chunk_options


@pytest.mark.parametrize(
    "strategy,expected_po,key",
    [
        ("fixed_token", PROCESS_OPTION_CHUNK_FIXED, "fixed_token"),
        ("recursive_character", PROCESS_OPTION_CHUNK_RECURSIVE, "recursive_character"),
        ("semantic_vector", PROCESS_OPTION_CHUNK_VECTOR, "semantic_vector"),
        ("paragraph_semantic", PROCESS_OPTION_CHUNK_PARAGRAH, "paragraph_semantic"),
    ],
)
def test_resolve_maps_strategy_and_writes_size_into_subdict(strategy, expected_po, key):
    cfg = TextChunkingConfig.model_validate(
        {"strategy": strategy, "params": {"chunk_token_size": 777}}
    )
    process_options, chunk_options = _resolve_text_chunking(cfg, _stub_rag())
    assert process_options == expected_po
    # chunk_token_size lands in the strategy sub-dict for ALL strategies
    # (F included, post-cleanup) — that's where process_single_document reads it.
    assert chunk_options[key]["chunk_token_size"] == 777
    # slim contract: other strategies' sub-dicts are dropped
    for other in _ALL_STRATEGY_KEYS - {key}:
        assert other not in chunk_options


def test_resolve_merges_strategy_params():
    cfg = TextChunkingConfig.model_validate(
        {
            "strategy": "recursive_character",
            "params": {"separators": ["A", "B"], "chunk_overlap_token_size": 0},
        }
    )
    _, chunk_options = _resolve_text_chunking(cfg, _stub_rag())
    assert chunk_options["recursive_character"]["separators"] == ["A", "B"]
    assert chunk_options["recursive_character"]["chunk_overlap_token_size"] == 0


def test_resolve_size_overrides_env_for_recursive(monkeypatch):
    monkeypatch.setenv("CHUNK_R_SIZE", "999")
    addon = {"chunker": default_chunker_config()}
    # sanity: env baked into the R sub-dict
    assert addon["chunker"]["recursive_character"]["chunk_token_size"] == 999
    cfg = TextChunkingConfig.model_validate(
        {"strategy": "recursive_character", "params": {"chunk_token_size": 1234}}
    )
    _, chunk_options = _resolve_text_chunking(cfg, _stub_rag(addon))
    # API value wins over the env-derived sub-dict value.
    assert chunk_options["recursive_character"]["chunk_token_size"] == 1234


def test_resolve_split_by_character_only_false_overrides_env(monkeypatch):
    # The API path can express an explicit False (a plain dict merge), unlike
    # the ainsert positional-arg path. Prove it overrides an env-True default.
    monkeypatch.setenv("CHUNK_F_SPLIT_BY_CHARACTER_ONLY", "true")
    addon = {"chunker": default_chunker_config()}
    assert addon["chunker"]["fixed_token"]["split_by_character_only"] is True
    cfg = TextChunkingConfig.model_validate(
        {"strategy": "fixed_token", "params": {"split_by_character_only": False}}
    )
    _, chunk_options = _resolve_text_chunking(cfg, _stub_rag(addon))
    assert chunk_options["fixed_token"]["split_by_character_only"] is False


def test_resolve_drop_references_request_false_overrides_env_true(monkeypatch):
    # The JSON text API can express an explicit ``drop_references=False`` that
    # overrides ``CHUNK_P_DROP_REFERENCES=true`` (resolved via slim_chunk_options
    # then overlaid by the request params). Mirrors the split_by_character_only
    # override above for the paragraph-semantic switch.
    monkeypatch.setenv("CHUNK_P_DROP_REFERENCES", "true")
    cfg = TextChunkingConfig.model_validate(
        {"strategy": "paragraph_semantic", "params": {"drop_references": False}}
    )
    _, chunk_options = _resolve_text_chunking(cfg, _stub_rag())
    assert chunk_options["paragraph_semantic"]["drop_references"] is False


def test_resolve_drop_references_env_true_without_request_param(monkeypatch):
    # No request param → the env-true default survives into the snapshot.
    monkeypatch.setenv("CHUNK_P_DROP_REFERENCES", "true")
    cfg = TextChunkingConfig.model_validate({"strategy": "paragraph_semantic"})
    _, chunk_options = _resolve_text_chunking(cfg, _stub_rag())
    assert chunk_options["paragraph_semantic"]["drop_references"] is True


def test_resolve_rejects_size_below_inherited_overlap(monkeypatch):
    # Overlap is inherited from addon_params (not in the request), so the
    # request model can't catch it — _resolve_text_chunking must.
    monkeypatch.setenv("CHUNK_F_OVERLAP_SIZE", "100")
    addon = {"chunker": default_chunker_config()}
    assert addon["chunker"]["fixed_token"]["chunk_overlap_token_size"] == 100
    cfg = TextChunkingConfig.model_validate(
        {"strategy": "fixed_token", "params": {"chunk_token_size": 50}}
    )
    with pytest.raises(ValueError, match="chunk_overlap_token_size"):
        _resolve_text_chunking(cfg, _stub_rag(addon))


def test_resolve_allows_size_above_inherited_overlap(monkeypatch):
    monkeypatch.setenv("CHUNK_F_OVERLAP_SIZE", "100")
    addon = {"chunker": default_chunker_config()}
    cfg = TextChunkingConfig.model_validate(
        {"strategy": "fixed_token", "params": {"chunk_token_size": 400}}
    )
    _, chunk_options = _resolve_text_chunking(cfg, _stub_rag(addon))
    assert chunk_options["fixed_token"]["chunk_token_size"] == 400


def test_resolve_skips_overlap_check_for_delimiter_only(monkeypatch):
    # Delimiter-only fixed-token chunking never uses overlap, so a small
    # chunk_token_size below the inherited overlap must NOT be rejected.
    monkeypatch.setenv("CHUNK_F_OVERLAP_SIZE", "100")
    addon = {"chunker": default_chunker_config()}
    cfg = TextChunkingConfig.model_validate(
        {
            "strategy": "fixed_token",
            "params": {
                "split_by_character": "\n\n",
                "split_by_character_only": True,
                "chunk_token_size": 50,
            },
        }
    )
    _, chunk_options = _resolve_text_chunking(cfg, _stub_rag(addon))
    assert chunk_options["fixed_token"]["chunk_token_size"] == 50


def test_resolve_enforces_overlap_when_only_flag_without_delimiter(monkeypatch):
    # split_by_character_only is a no-op without split_by_character: the chunker
    # falls back to normal token windowing, which DOES use overlap — so the
    # overlap < size check must still fire here.
    monkeypatch.setenv("CHUNK_F_OVERLAP_SIZE", "100")
    monkeypatch.delenv("CHUNK_F_SPLIT_BY_CHARACTER", raising=False)
    addon = {"chunker": default_chunker_config()}
    cfg = TextChunkingConfig.model_validate(
        {
            "strategy": "fixed_token",
            "params": {"split_by_character_only": True, "chunk_token_size": 50},
        }
    )
    with pytest.raises(ValueError, match="chunk_overlap_token_size"):
        _resolve_text_chunking(cfg, _stub_rag(addon))


def test_resolve_allows_amount_over_100_with_inherited_std_type():
    # Request overrides only the amount; the standard_deviation type is
    # inherited from addon_params. std/iqr have no (0, 100] ceiling, so this
    # must NOT be rejected (the request model deferred the check here).
    addon = {
        "chunker": {
            "semantic_vector": {"breakpoint_threshold_type": "standard_deviation"}
        }
    }
    cfg = TextChunkingConfig.model_validate(
        {"strategy": "semantic_vector", "params": {"breakpoint_threshold_amount": 150}}
    )
    _, chunk_options = _resolve_text_chunking(cfg, _stub_rag(addon))
    assert chunk_options["semantic_vector"]["breakpoint_threshold_amount"] == 150
    assert (
        chunk_options["semantic_vector"]["breakpoint_threshold_type"]
        == "standard_deviation"
    )


def test_resolve_rejects_amount_over_100_with_inherited_percentile_type():
    # Same partial override, but the effective (inherited) type is percentile,
    # which feeds np.percentile and requires the (0, 100] ceiling.
    addon = {
        "chunker": {"semantic_vector": {"breakpoint_threshold_type": "percentile"}}
    }
    cfg = TextChunkingConfig.model_validate(
        {"strategy": "semantic_vector", "params": {"breakpoint_threshold_amount": 150}}
    )
    with pytest.raises(ValueError, match="breakpoint_threshold_amount"):
        _resolve_text_chunking(cfg, _stub_rag(addon))


def test_resolve_null_size_does_not_erase_inherited_default(monkeypatch):
    # An explicit null in the request must not overwrite the resolved size
    # with None (which would make the chunker do int(None) in the background).
    monkeypatch.setenv("CHUNK_F_SIZE", "640")
    addon = {"chunker": default_chunker_config()}
    cfg = TextChunkingConfig.model_validate(
        {"strategy": "fixed_token", "params": {"chunk_token_size": None}}
    )
    _, chunk_options = _resolve_text_chunking(cfg, _stub_rag(addon))
    # null dropped by the model -> inherited CHUNK_F_SIZE survives, no None.
    assert chunk_options["fixed_token"]["chunk_token_size"] == 640


# ---------------------------------------------------------------------------
# 3. Route forwarding + synchronous 422
# ---------------------------------------------------------------------------


class _FwdDocStatus:
    async def get_doc_by_file_basename(self, basename):
        return None


class _FwdRag:
    workspace = "chunk-fwd-test"
    addon_params: dict = {}

    def __init__(self):
        self.doc_status = _FwdDocStatus()


_HEADERS = {"X-API-Key": "test-key"}


def _make_client(monkeypatch, addon_params=None):
    """Build a TestClient whose enqueue-slot guards are no-ops and whose
    ``pipeline_index_texts`` is a spy recording the forwarded args.

    ``addon_params`` seeds the rag the routes resolve chunking against; the
    handler calls the real ``_resolve_text_chunking`` synchronously, so the
    effective-overlap validation runs against this snapshot.
    """
    captured: dict = {}

    async def _spy(rag, texts, file_sources=None, track_id=None, chunking=None):
        captured["texts"] = texts
        captured["file_sources"] = file_sources
        captured["chunking"] = chunking

    async def _noop_reserve(rag):
        return False

    async def _noop_release(rag):
        return None

    monkeypatch.setattr(_dr, "pipeline_index_texts", _spy)
    monkeypatch.setattr(_dr, "_reserve_enqueue_slot", _noop_reserve)
    monkeypatch.setattr(_dr, "_release_enqueue_slot", _noop_release)

    rag = _FwdRag()
    rag.addon_params = addon_params if addon_params is not None else {}

    app = FastAPI()
    app.include_router(
        create_document_routes(rag, SimpleNamespace(), api_key="test-key")
    )
    return TestClient(app), captured


def test_insert_text_forwards_chunking(monkeypatch):
    client, captured = _make_client(monkeypatch)
    resp = client.post(
        "/documents/text",
        headers=_HEADERS,
        json={
            "text": "hello world",
            "file_source": "a.md",
            "chunking": {
                "strategy": "recursive_character",
                "params": {"chunk_token_size": 1000, "separators": ["X"]},
            },
        },
    )
    assert resp.status_code == 200
    assert captured["chunking"] is not None
    assert captured["chunking"].strategy == "recursive_character"
    assert captured["chunking"].params == {
        "chunk_token_size": 1000,
        "separators": ["X"],
    }


def test_insert_texts_forwards_chunking(monkeypatch):
    client, captured = _make_client(monkeypatch)
    resp = client.post(
        "/documents/texts",
        headers=_HEADERS,
        json={
            "texts": ["one", "two"],
            "file_sources": ["a.md", "b.md"],
            "chunking": {"strategy": "semantic_vector", "params": {"buffer_size": 2}},
        },
    )
    assert resp.status_code == 200
    assert captured["chunking"].strategy == "semantic_vector"
    assert captured["chunking"].params == {"buffer_size": 2}


def test_insert_text_without_chunking_forwards_none(monkeypatch):
    client, captured = _make_client(monkeypatch)
    resp = client.post(
        "/documents/text",
        headers=_HEADERS,
        json={"text": "hello", "file_source": "a.md"},
    )
    assert resp.status_code == 200
    assert captured["chunking"] is None


def test_insert_text_returns_422_on_malformed_chunking_without_scheduling(monkeypatch):
    client, captured = _make_client(monkeypatch)
    resp = client.post(
        "/documents/text",
        headers=_HEADERS,
        json={
            "text": "hello",
            "file_source": "a.md",
            "chunking": {
                "strategy": "recursive_character",
                "params": {"separators": "notalist"},
            },
        },
    )
    assert resp.status_code == 422
    # Body validation fails before the endpoint body runs: no background
    # indexing is scheduled, so the spy never fires.
    assert captured == {}


def test_insert_text_returns_422_when_size_below_inherited_overlap(monkeypatch):
    # chunk_token_size=50 in the request, overlap=100 inherited from the
    # rag's addon_params (not in the request). The model can't catch this;
    # the handler's synchronous _resolve_text_chunking must, BEFORE any
    # background work is scheduled.
    addon = {
        "chunker": {
            "chunk_token_size": 1200,
            "fixed_token": {"chunk_overlap_token_size": 100},
        }
    }
    client, captured = _make_client(monkeypatch, addon_params=addon)
    resp = client.post(
        "/documents/text",
        headers=_HEADERS,
        json={
            "text": "hello",
            "file_source": "a.md",
            "chunking": {"strategy": "fixed_token", "params": {"chunk_token_size": 50}},
        },
    )
    assert resp.status_code == 422
    assert "chunk_overlap_token_size" in resp.json()["detail"]
    # Rejected synchronously: background indexing never scheduled.
    assert captured == {}


def test_insert_text_allows_amount_override_inheriting_std_type(monkeypatch):
    # Reviewer scenario: deployment sets standard_deviation; a request
    # overrides only breakpoint_threshold_amount (> 100). This must be
    # accepted (not 422), since std has no (0, 100] ceiling.
    addon = {
        "chunker": {
            "semantic_vector": {"breakpoint_threshold_type": "standard_deviation"}
        }
    }
    client, captured = _make_client(monkeypatch, addon_params=addon)
    resp = client.post(
        "/documents/text",
        headers=_HEADERS,
        json={
            "text": "hello",
            "file_source": "a.md",
            "chunking": {
                "strategy": "semantic_vector",
                "params": {"breakpoint_threshold_amount": 150},
            },
        },
    )
    assert resp.status_code == 200
    assert captured["chunking"].params == {"breakpoint_threshold_amount": 150.0}


def test_insert_text_rejects_amount_over_100_inheriting_percentile_type(monkeypatch):
    addon = {
        "chunker": {"semantic_vector": {"breakpoint_threshold_type": "percentile"}}
    }
    client, captured = _make_client(monkeypatch, addon_params=addon)
    resp = client.post(
        "/documents/text",
        headers=_HEADERS,
        json={
            "text": "hello",
            "file_source": "a.md",
            "chunking": {
                "strategy": "semantic_vector",
                "params": {"breakpoint_threshold_amount": 150},
            },
        },
    )
    assert resp.status_code == 422
    assert "breakpoint_threshold_amount" in resp.json()["detail"]
    assert captured == {}


def test_insert_text_rejects_malformed_sentence_split_regex(monkeypatch):
    # Malformed regex must 422 at request parse time, before scheduling.
    client, captured = _make_client(monkeypatch)
    resp = client.post(
        "/documents/text",
        headers=_HEADERS,
        json={
            "text": "hello",
            "file_source": "a.md",
            "chunking": {
                "strategy": "semantic_vector",
                "params": {"sentence_split_regex": "("},
            },
        },
    )
    assert resp.status_code == 422
    assert captured == {}


def test_insert_text_drops_explicit_null_param(monkeypatch):
    # "chunk_token_size": null must be treated as "inherit" (dropped), so the
    # request succeeds and the forwarded params carry no None that would later
    # crash the chunker with int(None).
    client, captured = _make_client(monkeypatch)
    resp = client.post(
        "/documents/text",
        headers=_HEADERS,
        json={
            "text": "hello",
            "file_source": "a.md",
            "chunking": {
                "strategy": "fixed_token",
                "params": {"chunk_token_size": None, "chunk_overlap_token_size": 50},
            },
        },
    )
    assert resp.status_code == 200
    assert captured["chunking"].params == {"chunk_overlap_token_size": 50}


def test_insert_text_allows_small_size_for_delimiter_only(monkeypatch):
    # Paragraph splitting with a small chunk_token_size: overlap is inherited
    # (100) but unused in delimiter-only mode, so this must succeed, not 422.
    addon = {"chunker": {"fixed_token": {"chunk_overlap_token_size": 100}}}
    client, captured = _make_client(monkeypatch, addon_params=addon)
    resp = client.post(
        "/documents/text",
        headers=_HEADERS,
        json={
            "text": "hello",
            "file_source": "a.md",
            "chunking": {
                "strategy": "fixed_token",
                "params": {
                    "split_by_character": "\n\n",
                    "split_by_character_only": True,
                    "chunk_token_size": 50,
                },
            },
        },
    )
    assert resp.status_code == 200
    assert captured["chunking"].params["chunk_token_size"] == 50
