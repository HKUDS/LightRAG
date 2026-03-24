import re

import pytest

from lightrag.kg import (
    STORAGE_IMPLEMENTATIONS,
    STORAGE_ENV_REQUIREMENTS,
    STORAGES,
    verify_storage_implementation,
)
from lightrag.kg.nebula_impl import (
    _canonical_edge_pair,
    _normalize_space_name,
    _parse_nebula_hosts,
    _short_hash_suffix,
    NebulaGraphStorage,
)


def test_nebula_graph_storage_is_registered():
    assert (
        "NebulaGraphStorage"
        in STORAGE_IMPLEMENTATIONS["GRAPH_STORAGE"]["implementations"]
    )
    assert STORAGES["NebulaGraphStorage"] == ".kg.nebula_impl"


def test_nebula_graph_storage_env_requirements():
    assert STORAGE_ENV_REQUIREMENTS["NebulaGraphStorage"] == [
        "NEBULA_HOSTS",
        "NEBULA_USER",
        "NEBULA_PASSWORD",
    ]


def test_nebula_graph_storage_verify_compatibility():
    verify_storage_implementation("GRAPH_STORAGE", "NebulaGraphStorage")


def test_normalize_space_name_uses_prefix_and_workspace():
    assert _normalize_space_name("lightrag", "hr-prod") == "lightrag__hr_prod"


def test_normalize_space_name_uses_base_for_empty_workspace():
    assert _normalize_space_name("lightrag", "") == "lightrag__base"


def test_canonical_edge_pair_is_undirected():
    assert _canonical_edge_pair("B", "A") == ("A", "B")


def test_short_hash_suffix_rejects_non_positive_length():
    with pytest.raises(ValueError, match="positive"):
        _short_hash_suffix("abc", length=0)


def test_normalize_space_name_truncates_and_appends_hash_suffix():
    normalized = _normalize_space_name("lightrag", "w" * 180)
    assert len(normalized) == 127
    assert re.search(r"__[0-9a-f]{8}$", normalized)


def test_parse_nebula_hosts_supports_ipv4_and_bracket_ipv6():
    assert _parse_nebula_hosts("127.0.0.1:9669, [::1]:9779") == [
        ("127.0.0.1", 9669),
        ("::1", 9779),
    ]


def test_parse_nebula_hosts_rejects_out_of_range_port():
    with pytest.raises(ValueError, match="must be in 1..65535"):
        _parse_nebula_hosts("127.0.0.1:70000")


def test_parse_nebula_hosts_rejects_unbracketed_ipv6_with_port():
    with pytest.raises(ValueError, match="IPv6 host must use bracket notation"):
        _parse_nebula_hosts("::1:9669")


def test_nebula_graph_storage_sets_initialized_as_instance_attr():
    storage = NebulaGraphStorage(
        namespace="test",
        workspace=None,
        global_config={},
        embedding_func=lambda *args, **kwargs: None,
    )
    assert "_initialized" in storage.__dict__
    assert storage._initialized is False
