"""A7: default-off retrieval is byte-identical to the pre-hybrid baseline.

Pinned-manifest regression for the hybrid-retrieval zero-change guarantee:
55 documents / 220 CJK+EN chunks / 66 seeded entities / 65 relations,
20 queries x 5 modes, ``only_need_context=True`` with injected hl/ll
keywords so no LLM is ever called, and deterministic bag-of-tokens hash
embeddings. Synthetic embeddings are legitimate here: A7 is an identity
proof, not a quality proof.

Provenance: both pinned manifests were generated on pristine origin/main
35a3a52cf AND on the feature branch with all hybrid wiring plus the
answer-cache key partition, then compared byte-level: empty diff on the
default file backends and on all-PostgreSQL backends. The file and PG
manifests differ from EACH OTHER only in the order of relations tied on
(degree, weight): the stable sort keeps each backend's deterministic edge
enumeration order, NetworkX insertion order versus PGTable neighbor-id
order. Each backend is deterministic across runs, so each has its own
pinned manifest.

Rules: a fixture or query change invalidates the proof -- regenerate BOTH
manifests against a fresh origin/main baseline checkout and re-compare
before updating the JSON. Never normalize a difference away; a hash
mismatch is a retrieval behavior change and must be explained. The shared
harness copy in the PR artifacts omits ``enable_hybrid`` on the baseline,
where the field does not exist; its absence IS default-off.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.base import QueryParam
from lightrag.prompt import PROMPTS
from lightrag.utils import EmbeddingFunc

# ---------------------------------------------------------------------------
# Fixture: shared verbatim with the baseline/branch harness runs.
# ---------------------------------------------------------------------------

EMBEDDING_DIM = 32
MODEL_NAME = "a7proof"
MODES = ("local", "global", "hybrid", "mix", "naive")
COSINE_THRESHOLD = 0.1

FAMILIES = [
    ("传感器", "sensors", [
        ("光耦合器", "optocoupler"),
        ("超声波传感器", "ultrasonic sensor"),
        ("陀螺仪", "gyroscope"),
        ("触碰传感器", "touch sensor"),
        ("颜色传感器", "color sensor"),
    ]),
    ("电机", "motors", [
        ("伺服电机", "servo motor"),
        ("步进电机", "stepper motor"),
        ("直流电机", "dc motor"),
        ("无刷电机", "brushless motor"),
        ("编码电机", "encoded motor"),
    ]),
    ("齿轮", "gears", [
        ("行星齿轮", "planetary gear"),
        ("蜗轮蜗杆", "worm gear"),
        ("锥齿轮", "bevel gear"),
        ("齿条", "gear rack"),
        ("差速器", "differential"),
    ]),
    ("控制器", "controllers", [
        ("主控制器", "main controller"),
        ("扩展板", "expansion board"),
        ("电机驱动板", "motor driver board"),
        ("传感器集线器", "sensor hub"),
        ("遥控器", "remote controller"),
    ]),
    ("编程", "programming", [
        ("图形化编程", "block programming"),
        ("Python编程", "python programming"),
        ("状态机", "state machine"),
        ("事件循环", "event loop"),
        ("固件升级", "firmware update"),
    ]),
    ("结构", "structure", [
        ("底盘", "chassis"),
        ("机械臂", "robotic arm"),
        ("履带", "track"),
        ("万向轮", "omni wheel"),
        ("连接销", "connector pin"),
    ]),
    ("能源", "power", [
        ("锂电池", "lithium battery"),
        ("稳压模块", "voltage regulator"),
        ("太阳能板", "solar panel"),
        ("电源管理", "power management"),
        ("充电底座", "charging dock"),
    ]),
    ("通信", "communication", [
        ("蓝牙模块", "bluetooth module"),
        ("Wi-Fi模块", "wifi module"),
        ("红外通信", "infrared communication"),
        ("串口总线", "serial bus"),
        ("CAN总线", "can bus"),
    ]),
    ("算法", "algorithms", [
        ("PID控制", "pid control"),
        ("路径规划", "path planning"),
        ("卡尔曼滤波", "kalman filter"),
        ("视觉识别", "vision recognition"),
        ("避障算法", "obstacle avoidance"),
    ]),
    ("材料", "materials", [
        ("ABS塑料", "abs plastic"),
        ("碳纤维", "carbon fiber"),
        ("铝合金", "aluminum alloy"),
        ("橡胶轮胎", "rubber tire"),
        ("润滑脂", "lubricant grease"),
    ]),
    ("测试", "testing", [
        ("单元测试", "unit testing"),
        ("标定流程", "calibration procedure"),
        ("老化测试", "burn-in testing"),
        ("故障注入", "fault injection"),
        ("回归测试", "regression testing"),
    ]),
]

ASPECTS = [
    ("工作原理", "working principle"),
    ("技术参数", "technical specifications"),
    ("应用场景", "application scenarios"),
    ("维护保养", "maintenance"),
]

CROSS_LINKS = [
    ("PID控制", "伺服电机"),
    ("卡尔曼滤波", "陀螺仪"),
    ("路径规划", "避障算法"),
    ("视觉识别", "颜色传感器"),
    ("电机驱动板", "直流电机"),
    ("锂电池", "电源管理"),
    ("蓝牙模块", "遥控器"),
    ("状态机", "事件循环"),
    ("机械臂", "伺服电机"),
    ("标定流程", "陀螺仪"),
]

# (query text, hl_keywords, ll_keywords). Keywords are injected through the
# QueryParam escape hatch so no keyword-extraction LLM call ever happens.
QUERIES = [
    ("光耦合器的工作原理是什么？", ["传感器", "信号检测"], ["光耦合器"]),
    ("超声波传感器如何测量距离？", ["传感器", "测距"], ["超声波传感器"]),
    ("陀螺仪在机器人中的作用", ["传感器", "姿态"], ["陀螺仪"]),
    ("伺服电机的控制方式", ["电机", "控制"], ["伺服电机"]),
    ("行星齿轮的传动特点", ["齿轮", "传动"], ["行星齿轮"]),
    ("主控制器如何连接扩展板？", ["控制器", "连接"], ["主控制器", "扩展板"]),
    ("图形化编程适合哪些场景？", ["编程", "教育"], ["图形化编程"]),
    ("锂电池的充电与维护", ["能源", "电池"], ["锂电池", "充电底座"]),
    ("蓝牙模块的通信距离", ["通信", "无线"], ["蓝牙模块"]),
    ("PID控制参数如何整定？", ["算法", "控制"], ["PID控制"]),
    ("How does the optocoupler isolate signals?", ["sensors", "isolation"], ["optocoupler"]),
    ("What is the role of the gyroscope?", ["sensors", "orientation"], ["gyroscope"]),
    ("Explain planetary gear transmission", ["gears", "transmission"], ["planetary gear"]),
    ("How to calibrate the ultrasonic sensor?", ["testing", "calibration"], ["ultrasonic sensor", "calibration procedure"]),
    ("Which motor suits a robotic arm?", ["motors", "structure"], ["servo motor", "robotic arm"]),
    ("机器人底盘用什么材料？", ["材料", "结构"], ["底盘", "ABS塑料", "铝合金"]),
    ("卡尔曼滤波如何融合陀螺仪数据？", ["算法", "传感器"], ["卡尔曼滤波", "陀螺仪"]),
    ("固件升级的流程与风险", ["编程", "维护"], ["固件升级"]),
    ("避障算法与路径规划的关系", ["算法", "导航"], ["避障算法", "路径规划"]),
    ("Difference between stepper motor and dc motor", ["motors", "comparison"], ["stepper motor", "dc motor"]),
]


def build_fixture() -> dict:
    """Build the custom-KG payload: 220 chunks, 66 entities, 65 relations."""
    chunks: list[dict] = []
    entities: list[dict] = []
    relationships: list[dict] = []
    doc_of: dict[str, tuple[int, str]] = {}

    doc_index = 0
    for fam_idx, (fam_zh, fam_en, subs) in enumerate(FAMILIES):
        for zh, en in subs:
            d = doc_index
            doc_index += 1
            file_path = (
                f"corpus/{fam_en.replace(' ', '_')}/{en.replace(' ', '_')}.md"
            )
            aliases = [f"src-{d:03d}-{k}" for k in range(4)]
            doc_of[zh] = (d, aliases[0])
            for k, (asp_zh, asp_en) in enumerate(ASPECTS):
                content = (
                    f"{zh}（{en}）是{fam_zh}家族的关键组件。本节介绍{asp_zh}。"
                    f"{zh}在乐高机器人平台中承担{asp_zh}相关功能，编号 {d:03d}-{k}。"
                    f"The {en} belongs to the {fam_en} family of the LEGO robotics "
                    f"platform. This section covers {asp_en} for document {d:03d}, "
                    f"chunk {k}. 额定值 {100 + d * 7 + k * 13} 单位，"
                    f"公差 ±{(d + k) % 5 + 1}%，采样率 {50 + d * 3} Hz。"
                )
                chunks.append(
                    {
                        "content": content,
                        "source_id": aliases[k],
                        "file_path": file_path,
                        "chunk_order_index": k,
                    }
                )
            entities.append(
                {
                    "entity_name": zh,
                    "entity_type": "组件",
                    "description": (
                        f"{zh}（{en}）：{fam_zh}家族的组件，用于乐高机器人。"
                        f"The {en} is a key {fam_en} component."
                    ),
                    "source_id": aliases[0],
                    "file_path": file_path,
                }
            )
            relationships.append(
                {
                    "src_id": zh,
                    "tgt_id": fam_zh,
                    "description": (
                        f"{zh}属于{fam_zh}类别。"
                        f"The {en} is part of the {fam_en} family."
                    ),
                    "keywords": f"{fam_zh},{fam_en},组件关系",
                    "weight": 1.0,
                    "source_id": aliases[0],
                    "file_path": file_path,
                }
            )
        entities.append(
            {
                "entity_name": fam_zh,
                "entity_type": "类别",
                "description": (
                    f"{fam_zh}（{fam_en}）类别，包含五个关键组件。"
                    f"The {fam_en} family contains five key components."
                ),
                "source_id": f"src-{fam_idx * 5:03d}-0",
                "file_path": f"corpus/{fam_en.replace(' ', '_')}/overview.md",
            }
        )

    for src, tgt in CROSS_LINKS:
        d, alias = doc_of[src]
        relationships.append(
            {
                "src_id": src,
                "tgt_id": tgt,
                "description": f"{src}与{tgt}跨域协作。{src} cooperates with {tgt}.",
                "keywords": "跨域协作,cross-domain",
                "weight": 1.0,
                "source_id": alias,
                "file_path": f"corpus/cross_links/link_{d:03d}.md",
            }
        )

    assert len(chunks) == 220 and len(entities) == 66 and len(relationships) == 65
    return {"chunks": chunks, "entities": entities, "relationships": relationships}


_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")


@lru_cache(maxsize=8192)
def _token_vector(token: str) -> tuple[float, ...]:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return tuple(b / 127.5 - 1.0 for b in digest)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Deterministic bag-of-tokens hash projection, L2-normalized, float32."""
    rows = []
    for text in texts:
        normalized = unicodedata.normalize("NFKC", text).lower()
        vec = np.zeros(EMBEDDING_DIM, dtype=np.float64)
        for token in _TOKEN_RE.findall(normalized):
            vec += np.asarray(_token_vector(token), dtype=np.float64)
        norm = np.linalg.norm(vec)
        rows.append(vec / norm if norm > 0 else vec)
    return np.asarray(rows, dtype=np.float32)


async def a7_embed(texts: list[str]) -> np.ndarray:
    return embed_texts(texts)


LLM_CALLS: list[str] = []


async def refuse_llm(prompt, *args, **kwargs):
    """Record AND raise: a context-only A7 run must never reach the LLM."""
    LLM_CALLS.append(str(prompt)[:80])
    raise AssertionError("A7 context-only run must not call the LLM")


def make_param(mode: str, hl: list[str], ll: list[str]) -> QueryParam:
    """Pin every retrieval knob so no environment default leaks into the proof."""
    return QueryParam(
        mode=mode,
        only_need_context=True,
        enable_rerank=False,
        enable_hybrid=False,
        stream=False,
        top_k=10,
        chunk_top_k=8,
        max_entity_tokens=4000,
        max_relation_tokens=4000,
        max_total_tokens=12000,
        hl_keywords=list(hl),
        ll_keywords=list(ll),
    )


# ---------------------------------------------------------------------------
# Regeneration and manifest comparison.
# ---------------------------------------------------------------------------

_FIXTURE_DIR = Path(__file__).resolve().parent


async def _regenerate_manifest(backend: str, working_dir: str, workspace: str) -> dict:
    """Seed the fixture and hash every retrieved context.

    Fails loud on any LLM call, any non-string context, and any canned
    no-result response: the fixture must give every mode real content.
    """
    kwargs = dict(
        working_dir=working_dir,
        workspace=workspace,
        llm_model_func=refuse_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=8192,
            func=a7_embed,
            model_name=MODEL_NAME,
        ),
        vector_db_storage_cls_kwargs={
            "cosine_better_than_threshold": COSINE_THRESHOLD
        },
    )
    if backend == "pg":
        kwargs.update(
            kv_storage="PGKVStorage",
            vector_storage="PGVectorStorage",
            graph_storage="PGTableGraphStorage",
            doc_status_storage="PGDocStatusStorage",
        )
    rag = LightRAG(**kwargs)
    await rag.initialize_storages()
    try:
        fixture = build_fixture()
        await asyncio.wait_for(rag.ainsert_custom_kg(fixture), timeout=600)
        entries = []
        for qi, (text, hl, ll) in enumerate(QUERIES):
            for mode in MODES:
                context = await asyncio.wait_for(
                    rag.aquery(text, make_param(mode, hl, ll)), timeout=120
                )
                assert isinstance(context, str), f"q{qi:02d}-{mode}: non-str context"
                assert context != PROMPTS["fail_response"], (
                    f"q{qi:02d}-{mode}: fixture must retrieve content"
                )
                raw = context.encode("utf-8")
                entries.append(
                    {
                        "id": f"q{qi:02d}-{mode}",
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "bytes": len(raw),
                    }
                )
        assert not LLM_CALLS, f"LLM was called: {LLM_CALLS}"
        return {
            "version": 1,
            "backend": backend,
            "fixture": {
                "documents": 55,
                "chunks": len(fixture["chunks"]),
                "entities": len(fixture["entities"]),
                "relations": len(fixture["relationships"]),
                "embedding_dim": EMBEDDING_DIM,
                "model_name": MODEL_NAME,
                "queries": len(QUERIES),
                "modes": list(MODES),
            },
            "entries": entries,
        }
    finally:
        await rag.finalize_storages()


def _assert_matches_pinned(manifest: dict, pinned_name: str) -> None:
    pinned = json.loads(
        (_FIXTURE_DIR / pinned_name).read_text(encoding="utf-8")
    )
    assert manifest["version"] == pinned["version"]
    assert manifest["backend"] == pinned["backend"]
    assert manifest["fixture"] == pinned["fixture"]
    assert len(manifest["entries"]) == len(pinned["entries"]) == 100
    for got, want in zip(manifest["entries"], pinned["entries"]):
        assert got == want, f"retrieval context changed for {want['id']}"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_file_backend_matches_pinned_baseline(tmp_path):
    manifest = await _regenerate_manifest(
        "file", str(tmp_path / "wd"), "a7off_file"
    )
    _assert_matches_pinned(manifest, "a7_default_off_manifest_file.json")


@pytest.mark.requires_db
@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skipif(
    not (os.environ.get("POSTGRES_HOST") and os.environ.get("POSTGRES_PASSWORD")),
    reason="POSTGRES_HOST/POSTGRES_PASSWORD not set",
)
async def test_pg_backend_matches_pinned_baseline(tmp_path):
    from lightrag.kg import shared_storage
    from lightrag.kg.postgres_impl import ClientManager
    from lightrag.kg.shared_storage import initialize_share_data

    if ClientManager.get_config()["workspace"]:
        pytest.skip(
            "POSTGRES_WORKSPACE / [postgres] workspace would remap the test workspace"
        )
    initialize_share_data(workers=1)
    # asyncio.Lock binds to the loop of its first CONTENDED acquire; these
    # two primitives are process-wide while pytest loops are per-test (the
    # live PG suites' reset pattern). Nothing holds them at setup.
    shared_storage._data_init_lock = asyncio.Lock()
    ClientManager._lock = asyncio.Lock()

    workspace = f"a7off_{uuid4().hex[:8]}"
    try:
        manifest = await _regenerate_manifest("pg", str(tmp_path / "wd"), workspace)
        _assert_matches_pinned(manifest, "a7_default_off_manifest_pg.json")
    finally:
        await _cleanup_pg(workspace)


async def _cleanup_pg(workspace: str) -> None:
    """Delete every row this workspace wrote, then drop the scratch vector
    tables: the a7proof model name is exclusive to this test."""
    import asyncpg

    from lightrag.kg.postgres_impl import ClientManager

    config = ClientManager.get_config()
    conn = await asyncpg.connect(
        host=config["host"],
        port=int(config["port"]),
        user=config["user"],
        password=config["password"],
        database=config["database"],
    )
    try:
        tables = await conn.fetch(
            "SELECT tablename FROM pg_tables "
            "WHERE schemaname='public' AND tablename LIKE 'lightrag_%'"
        )
        for row in tables:
            table = row["tablename"]
            has_workspace = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM information_schema.columns "
                "WHERE table_schema='public' AND table_name=$1 "
                "AND column_name='workspace')",
                table,
            )
            if has_workspace:
                await conn.execute(
                    f'DELETE FROM "{table}" WHERE workspace=$1', workspace
                )
        for namespace in ("chunks", "entity", "relation"):
            await conn.execute(
                f"DROP TABLE IF EXISTS "
                f"lightrag_vdb_{namespace}_{MODEL_NAME}_{EMBEDDING_DIM}d"
            )
    finally:
        await conn.close()
