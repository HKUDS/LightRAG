"""Extraction-subset protocol and config pins, offline (file backends only)."""

from __future__ import annotations

import pytest

from lightrag.evaluation.hybrid_bench.cli import _base_config, parse_args
from lightrag.evaluation.hybrid_bench.dataset import BenchDataset, DatasetMeta
from lightrag.evaluation.hybrid_bench.rag_setup import (
    BenchConfig,
    build_embedding_func,
    build_llm_func,
    build_rag,
    extraction_subset,
)

pytestmark = pytest.mark.offline


def _dataset() -> BenchDataset:
    return BenchDataset(
        name="tiny",
        meta=DatasetMeta(
            domain="terminology",
            language="zh",
            source="https://example.org",
            license="CC-BY-4.0",
            subset_protocol="test",
            counts={},
        ),
        corpus={f"d{i}": f"text {i}" for i in range(6)},
        queries={"q1": "query one", "q2": "query two"},
        qrels={"q1": {"d2": 1, "d4": 2}, "q2": {"d0": 1}},
    )


def test_subset_always_contains_all_positives_then_sorted_fill():
    # positives {d0, d2, d4}; cap 5 fills with the sorted rest d1, d3.
    assert extraction_subset(_dataset(), 5) == ["d0", "d2", "d4", "d1", "d3"]


def test_subset_cap_equal_to_positives_adds_nothing():
    assert extraction_subset(_dataset(), 3) == ["d0", "d2", "d4"]


def test_subset_cap_zero_means_no_extraction():
    assert extraction_subset(_dataset(), 0) == []


def test_subset_cap_below_positives_fails_loud():
    with pytest.raises(ValueError, match="cannot cover all 3"):
        extraction_subset(_dataset(), 2)


@pytest.mark.parametrize(
    ("embedding_backend", "llm_backend", "synthetic"),
    [
        ("ollama", "ollama", False),
        ("fake-hash", "ollama", True),
        ("ollama", "fake", True),
        ("fake-hash", "fake", True),
    ],
    ids=["real", "fake-embed", "fake-llm", "both-fake"],
)
def test_synthetic_label_follows_the_fakes(embedding_backend, llm_backend, synthetic):
    config = BenchConfig(embedding_backend=embedding_backend, llm_backend=llm_backend)
    assert config.synthetic is synthetic


def test_embedding_model_name_sanitizes_for_table_names():
    config = BenchConfig(embedding_model="bge-m3:latest")
    assert config.embedding_model_name == "bge_m3_latest"
    assert BenchConfig(embedding_backend="fake-hash").embedding_model_name == (
        "fakehash"
    )


async def _refuse_llm(prompt, *args, **kwargs):
    raise AssertionError("gleaning pin must not call the LLM")


def _file_config(tmp_path, **overrides) -> BenchConfig:
    return BenchConfig(
        storage="file",
        working_dir=str(tmp_path / "working"),
        embedding_backend="fake-hash",
        embedding_dim=32,
        **overrides,
    )


async def test_max_gleaning_unset_keeps_product_default(tmp_path):
    from lightrag import LightRAG
    from lightrag.kg.shared_storage import initialize_share_data

    initialize_share_data(workers=1)
    rag = await build_rag(_file_config(tmp_path), _refuse_llm)
    try:
        product_default = LightRAG.__dataclass_fields__[
            "entity_extract_max_gleaning"
        ].default
        assert rag.entity_extract_max_gleaning == product_default
    finally:
        await rag.finalize_storages()


@pytest.mark.parametrize("value", [0, 1, 2])
async def test_max_gleaning_reaches_extraction_global_config(tmp_path, value):
    from lightrag.kg.shared_storage import initialize_share_data

    initialize_share_data(workers=1)
    rag = await build_rag(_file_config(tmp_path, max_gleaning=value), _refuse_llm)
    try:
        assert rag.entity_extract_max_gleaning == value
        # operate.py reads the knob from this exact dict.
        assert rag._build_global_config()["entity_extract_max_gleaning"] == value
    finally:
        await rag.finalize_storages()


def _cli_argv(*extra) -> list[str]:
    return ["--dataset-dir", "d", "--output-dir", "o", *extra]


def test_max_gleaning_cli_default_is_unset():
    args = parse_args(_cli_argv())
    assert args.max_gleaning is None
    assert _base_config(args, "ws").max_gleaning is None


@pytest.mark.parametrize("value", [0, 1])
def test_max_gleaning_cli_flag_reaches_config(value):
    args = parse_args(_cli_argv("--max-gleaning", str(value)))
    assert args.max_gleaning == value
    assert _base_config(args, "ws").max_gleaning == value


def test_max_gleaning_negative_rejected():
    with pytest.raises(SystemExit) as exc:
        parse_args(_cli_argv("--max-gleaning", "-1"))
    assert exc.value.code == 2


def test_llm_num_predict_default_leaves_output_uncapped():
    func = build_llm_func(BenchConfig())
    assert "options" not in func.keywords


@pytest.mark.parametrize("value", [2048, 4096])
def test_llm_num_predict_reaches_ollama_options(value):
    func = build_llm_func(BenchConfig(llm_num_predict=value))
    assert func.keywords["options"] == {"num_predict": value}


def test_llm_num_predict_fake_backend_ignores_the_cap():
    fake = object()
    assert build_llm_func(BenchConfig(llm_backend="fake"), fake) is fake


def test_llm_num_predict_cli_flag_reaches_config():
    args = parse_args(_cli_argv("--llm-num-predict", "4096"))
    assert args.llm_num_predict == 4096
    assert _base_config(args, "ws").llm_num_predict == 4096
    assert parse_args(_cli_argv()).llm_num_predict is None


@pytest.mark.parametrize("value", ["0", "-1"])
def test_llm_num_predict_non_positive_rejected(value):
    with pytest.raises(SystemExit) as exc:
        parse_args(_cli_argv("--llm-num-predict", value))
    assert exc.value.code == 2


def test_ollama_default_host_resolution_unchanged():
    # Empty config host means unset; the ollama path keeps the local URL.
    assert build_llm_func(BenchConfig()).keywords["host"] == ("http://127.0.0.1:11434")
    embedding = build_embedding_func(BenchConfig())
    assert embedding.func.keywords["host"] == "http://127.0.0.1:11434"


def test_ollama_explicit_host_wins_over_the_default():
    config = BenchConfig(llm_host="http://127.0.0.1:99999")
    assert build_llm_func(config).keywords["host"] == "http://127.0.0.1:99999"


@pytest.mark.parametrize(
    ("rag_attr", "config_attr", "default"),
    [
        ("max_parallel_insert", "max_parallel_insert", 8),
        ("llm_model_max_async", "max_async_llm", 16),
        ("embedding_func_max_async", "embedding_func_max_async", 16),
        ("embedding_batch_num", "embedding_batch_num", 32),
    ],
)
async def test_concurrency_defaults_reach_constructor(
    tmp_path, rag_attr, config_attr, default
):
    from lightrag.kg.shared_storage import initialize_share_data

    initialize_share_data(workers=1)
    assert getattr(BenchConfig(), config_attr) == default
    rag = await build_rag(_file_config(tmp_path), _refuse_llm)
    try:
        assert getattr(rag, rag_attr) == default
    finally:
        await rag.finalize_storages()


async def test_concurrency_overrides_reach_constructor(tmp_path):
    from lightrag.kg.shared_storage import initialize_share_data

    initialize_share_data(workers=1)
    config = _file_config(
        tmp_path,
        max_parallel_insert=2,
        max_async_llm=4,
        embedding_func_max_async=3,
        embedding_batch_num=5,
    )
    rag = await build_rag(config, _refuse_llm)
    try:
        assert rag.max_parallel_insert == 2
        assert rag.llm_model_max_async == 4
        assert rag.embedding_func_max_async == 3
        assert rag.embedding_batch_num == 5
    finally:
        await rag.finalize_storages()


@pytest.mark.parametrize(
    ("flag", "attr", "default"),
    [
        ("--max-parallel-insert", "max_parallel_insert", 8),
        ("--max-async-llm", "max_async_llm", 16),
        ("--embedding-func-max-async", "embedding_func_max_async", 16),
        ("--embedding-batch-num", "embedding_batch_num", 32),
    ],
)
def test_concurrency_cli_flag_reaches_config(flag, attr, default):
    assert getattr(parse_args(_cli_argv()), attr) == default
    args = parse_args(_cli_argv(flag, "7"))
    assert getattr(args, attr) == 7
    assert getattr(_base_config(args, "ws"), attr) == 7


@pytest.mark.parametrize(
    "flag",
    [
        "--max-parallel-insert",
        "--max-async-llm",
        "--embedding-func-max-async",
        "--embedding-batch-num",
    ],
)
@pytest.mark.parametrize("value", ["0", "-1"])
def test_concurrency_non_positive_rejected(flag, value):
    with pytest.raises(SystemExit) as exc:
        parse_args(_cli_argv(flag, value))
    assert exc.value.code == 2
