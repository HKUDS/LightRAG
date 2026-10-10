"""过滤式向量检索：HNSW 迭代扫描的会话设置必须真的下发到每条连接。

回归点（2026-09-29 实测）：分块检索是
``WHERE workspace = $1 AND content_vector <=> $q < $t ORDER BY … LIMIT $k``，
而 HNSW 索引是**全局**的（一张表装所有 workspace 的向量）。规划器可以先按距离走索引、
再按 workspace 过滤——那样只看到索引吐出的 ~``ef_search`` 个候选，对"少数派 workspace"
结果是 **0 行且不报错**：

* 5009 代码库 70,654 块（占 935,602 行的 7.5%），问一个最近邻距离 0.43（阈值 0.8）
  的问题 → `_dense_query` 返回 0；
* 同一现象在跨库联动侧早已踩过（``application/scoped_retrieval.py`` 用
  ``MATERIALIZED`` 把范围先物化），但代码库自己的问答没人验证过，所以一直没人发现。

修法：连接初始化时 ``LOAD 'vector'`` + ``SET hnsw.iterative_scan = strict_order``，
让索引扫描继续越过被过滤掉的候选；池回收（``RESET ALL``）后必须重放，否则下一位借出者
又回到"静默返回空"。这两个点各一条测试，因为第二种失效只在复用连接时才出现。
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from lightrag.kg.postgres_impl import PostgreSQLDB


def _db() -> PostgreSQLDB:
    db = PostgreSQLDB.__new__(PostgreSQLDB)  # 不走 __init__：只测配置方法本身
    db.workspace = "gw"
    return db


@pytest.mark.asyncio
async def test_iterative_scan_is_set_after_load() -> None:
    connection = AsyncMock()
    db = _db()

    await db.configure_hnsw_iterative_scan(connection)

    statements = [call.args[0] for call in connection.execute.await_args_list]
    # LOAD 在前：GUC 由扩展的共享库注册，本部署 shared_preload_libraries 只有 age，
    # 不先 LOAD 的话 SET 会被当成未知占位符**静默忽略**（看起来像成功）。
    assert statements == ["LOAD 'vector'", "SET hnsw.iterative_scan = strict_order"]


@pytest.mark.asyncio
async def test_missing_pgvector_or_guc_is_not_fatal() -> None:
    """老 pgvector 没有这个 GUC，不该因此连不上库（退回旧行为，但要留痕）。"""
    connection = AsyncMock()
    connection.execute.side_effect = [None, Exception("unrecognized configuration parameter")]
    db = _db()

    # LightRAG 的 logger 自带 handler，caplog 抓不到 —— 直接盯调用
    with patch("lightrag.kg.postgres_impl.logger") as logger:
        await db.configure_hnsw_iterative_scan(connection)  # 不抛

    assert logger.warning.called
    assert "iterative_scan" in str(logger.warning.call_args)


@pytest.mark.asyncio
async def test_load_failure_stops_before_set() -> None:
    """LOAD 都失败（库不在 $libdir）时不必再 SET —— 那只会是一次无声的 no-op。"""
    connection = AsyncMock()
    connection.execute.side_effect = Exception("could not open extension control file")
    db = _db()

    await db.configure_hnsw_iterative_scan(connection)

    assert [call.args[0] for call in connection.execute.await_args_list] == ["LOAD 'vector'"]
