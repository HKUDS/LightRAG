#!/usr/bin/env python
"""
通用图存储测试程序

该程序根据.env中的LIGHTRAG_GRAPH_STORAGE配置选择使用的图存储类型，
并对其进行基本操作和高级操作的测试。

支持的图存储类型包括：
- NetworkXStorage
- Neo4JStorage
- PGGraphStorage
"""

import asyncio
import os
import sys
import importlib
import numpy as np
from dotenv import load_dotenv
from ascii_colors import ASCIIColors

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lightrag.types import KnowledgeGraph
from lightrag.kg import (
    STORAGE_IMPLEMENTATIONS,
    STORAGE_ENV_REQUIREMENTS,
    STORAGES,
    verify_storage_implementation,
)
from lightrag.kg.shared_storage import initialize_share_data


# 模拟的嵌入函数，返回随机向量
async def mock_embedding_func(texts):
    return np.random.rand(len(texts), 10)  # 返回10维随机向量


def check_env_file():
    """
    检查.env文件是否存在，如果不存在则发出警告
    返回True表示应该继续执行，False表示应该退出
    """
    if not os.path.exists(".env"):
        warning_msg = "警告: 当前目录中没有找到.env文件，这可能会影响存储配置的加载。"
        ASCIIColors.yellow(warning_msg)

        # 检查是否在交互式终端中运行
        if sys.stdin.isatty():
            response = input("是否继续执行? (yes/no): ")
            if response.lower() != "yes":
                ASCIIColors.red("测试程序已取消")
                return False
    return True


async def initialize_graph_storage():
    """
    根据环境变量初始化相应的图存储实例
    返回初始化的存储实例
    """
    # 从环境变量中获取图存储类型
    graph_storage_type = os.getenv("LIGHTRAG_GRAPH_STORAGE", "NetworkXStorage")

    # 验证存储类型是否有效
    try:
        verify_storage_implementation("GRAPH_STORAGE", graph_storage_type)
    except ValueError as e:
        ASCIIColors.red(f"错误: {str(e)}")
        ASCIIColors.yellow(
            f"支持的图存储类型: {', '.join(STORAGE_IMPLEMENTATIONS['GRAPH_STORAGE']['implementations'])}"
        )
        return None

    # 检查所需的环境变量
    required_env_vars = STORAGE_ENV_REQUIREMENTS.get(graph_storage_type, [])
    missing_env_vars = [var for var in required_env_vars if not os.getenv(var)]

    if missing_env_vars:
        ASCIIColors.red(
            f"错误: {graph_storage_type} 需要以下环境变量，但未设置: {', '.join(missing_env_vars)}"
        )
        return None

    # 动态导入相应的模块
    module_path = STORAGES.get(graph_storage_type)
    if not module_path:
        ASCIIColors.red(f"错误: 未找到 {graph_storage_type} 的模块路径")
        return None

    try:
        module = importlib.import_module(module_path, package="lightrag")
        storage_class = getattr(module, graph_storage_type)
    except (ImportError, AttributeError) as e:
        ASCIIColors.red(f"错误: 导入 {graph_storage_type} 失败: {str(e)}")
        return None

    # 初始化存储实例
    global_config = {
        "embedding_batch_num": 10,  # 批处理大小
        "vector_db_storage_cls_kwargs": {
            "cosine_better_than_threshold": 0.5  # 余弦相似度阈值
        },
        "working_dir": os.environ.get("WORKING_DIR", "./rag_storage"),  # 工作目录
    }

    # 如果使用 NetworkXStorage，需要先初始化 shared_storage
    if graph_storage_type == "NetworkXStorage":
        initialize_share_data()  # 使用单进程模式

    try:
        storage = storage_class(
            namespace="test_graph",
            global_config=global_config,
            embedding_func=mock_embedding_func,
        )

        # 初始化连接
        await storage.initialize()
        return storage
    except Exception as e:
        ASCIIColors.red(f"错误: 初始化 {graph_storage_type} 失败: {str(e)}")
        return None


async def test_graph_basic(storage):
    """
    测试图数据库的基本操作:
    1. 使用 upsert_node 插入两个节点
    2. 使用 upsert_edge 插入一条连接两个节点的边
    3. 使用 get_node 读取一个节点
    4. 使用 get_edge 读取一条边
    """
    try:
        # 1. 插入第一个节点
        node1_id = "人工智能"
        node1_data = {
            "entity_id": node1_id,
            "description": "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。",
            "keywords": "AI,机器学习,深度学习",
            "entity_type": "技术领域",
        }
        print(f"插入节点1: {node1_id}")
        await storage.upsert_node(node1_id, node1_data)

        # 2. 插入第二个节点
        node2_id = "机器学习"
        node2_data = {
            "entity_id": node2_id,
            "description": "机器学习是人工智能的一个分支，它使用统计学方法让计算机系统在不被明确编程的情况下也能够学习。",
            "keywords": "监督学习,无监督学习,强化学习",
            "entity_type": "技术领域",
        }
        print(f"插入节点2: {node2_id}")
        await storage.upsert_node(node2_id, node2_data)

        # 3. 插入连接边
        edge_data = {
            "relationship": "包含",
            "weight": 1.0,
            "description": "人工智能领域包含机器学习这个子领域",
        }
        print(f"插入边: {node1_id} -> {node2_id}")
        await storage.upsert_edge(node1_id, node2_id, edge_data)

        # 4. 读取节点属性
        print(f"读取节点属性: {node1_id}")
        node1_props = await storage.get_node(node1_id)
        if node1_props:
            print(f"成功读取节点属性: {node1_id}")
            print(f"节点描述: {node1_props.get('description', '无描述')}")
            print(f"节点类型: {node1_props.get('entity_type', '无类型')}")
            print(f"节点关键词: {node1_props.get('keywords', '无关键词')}")
            # 验证返回的属性是否正确
            assert (
                node1_props.get("entity_id") == node1_id
            ), f"节点ID不匹配: 期望 {node1_id}, 实际 {node1_props.get('entity_id')}"
            assert (
                node1_props.get("description") == node1_data["description"]
            ), "节点描述不匹配"
            assert (
                node1_props.get("entity_type") == node1_data["entity_type"]
            ), "节点类型不匹配"
        else:
            print(f"读取节点属性失败: {node1_id}")
            assert False, f"未能读取节点属性: {node1_id}"

        # 5. 读取边属性
        print(f"读取边属性: {node1_id} -> {node2_id}")
        edge_props = await storage.get_edge(node1_id, node2_id)
        if edge_props:
            print(f"成功读取边属性: {node1_id} -> {node2_id}")
            print(f"边关系: {edge_props.get('relationship', '无关系')}")
            print(f"边描述: {edge_props.get('description', '无描述')}")
            print(f"边权重: {edge_props.get('weight', '无权重')}")
            # 验证返回的属性是否正确
            assert (
                edge_props.get("relationship") == edge_data["relationship"]
            ), "边关系不匹配"
            assert (
                edge_props.get("description") == edge_data["description"]
            ), "边描述不匹配"
            assert edge_props.get("weight") == edge_data["weight"], "边权重不匹配"
        else:
            print(f"读取边属性失败: {node1_id} -> {node2_id}")
            assert False, f"未能读取边属性: {node1_id} -> {node2_id}"

        # 5.1 验证无向图特性 - 读取反向边属性
        print(f"读取反向边属性: {node2_id} -> {node1_id}")
        reverse_edge_props = await storage.get_edge(node2_id, node1_id)
        if reverse_edge_props:
            print(f"成功读取反向边属性: {node2_id} -> {node1_id}")
            print(f"反向边关系: {reverse_edge_props.get('relationship', '无关系')}")
            print(f"反向边描述: {reverse_edge_props.get('description', '无描述')}")
            print(f"反向边权重: {reverse_edge_props.get('weight', '无权重')}")
            # 验证正向和反向边属性是否相同
            assert (
                edge_props == reverse_edge_props
            ), "正向和反向边属性不一致，无向图特性验证失败"
            print("无向图特性验证成功：正向和反向边属性一致")
        else:
            print(f"读取反向边属性失败: {node2_id} -> {node1_id}")
            assert (
                False
            ), f"未能读取反向边属性: {node2_id} -> {node1_id}，无向图特性验证失败"

        print("基本测试完成，数据已保留在数据库中")
        return True

    except Exception as e:
        ASCIIColors.red(f"测试过程中发生错误: {str(e)}")
        return False


async def test_graph_advanced(storage):
    """
    测试图数据库的高级操作:
    1. 使用 node_degree 获取节点的度数
    2. 使用 edge_degree 获取边的度数
    3. 使用 get_node_edges 获取节点的所有边
    4. 使用 get_all_labels 获取所有标签
    5. 使用 get_knowledge_graph 获取知识图谱
    6. 使用 delete_node 删除节点
    7. 使用 remove_nodes 批量删除节点
    8. 使用 remove_edges 删除边
    9. 使用 drop 清理数据
    """
    try:
        # 1. 插入测试数据
        # 插入节点1: 人工智能
        node1_id = "人工智能"
        node1_data = {
            "entity_id": node1_id,
            "description": "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。",
            "keywords": "AI,机器学习,深度学习",
            "entity_type": "技术领域",
        }
        print(f"插入节点1: {node1_id}")
        await storage.upsert_node(node1_id, node1_data)

        # 插入节点2: 机器学习
        node2_id = "机器学习"
        node2_data = {
            "entity_id": node2_id,
            "description": "机器学习是人工智能的一个分支，它使用统计学方法让计算机系统在不被明确编程的情况下也能够学习。",
            "keywords": "监督学习,无监督学习,强化学习",
            "entity_type": "技术领域",
        }
        print(f"插入节点2: {node2_id}")
        await storage.upsert_node(node2_id, node2_data)

        # 插入节点3: 深度学习
        node3_id = "深度学习"
        node3_data = {
            "entity_id": node3_id,
            "description": "深度学习是机器学习的一个分支，它使用多层神经网络来模拟人脑的学习过程。",
            "keywords": "神经网络,CNN,RNN",
            "entity_type": "技术领域",
        }
        print(f"插入节点3: {node3_id}")
        await storage.upsert_node(node3_id, node3_data)

        # 插入边1: 人工智能 -> 机器学习
        edge1_data = {
            "relationship": "包含",
            "weight": 1.0,
            "description": "人工智能领域包含机器学习这个子领域",
        }
        print(f"插入边1: {node1_id} -> {node2_id}")
        await storage.upsert_edge(node1_id, node2_id, edge1_data)

        # 插入边2: 机器学习 -> 深度学习
        edge2_data = {
            "relationship": "包含",
            "weight": 1.0,
            "description": "机器学习领域包含深度学习这个子领域",
        }
        print(f"插入边2: {node2_id} -> {node3_id}")
        await storage.upsert_edge(node2_id, node3_id, edge2_data)

        # 2. 测试 node_degree - 获取节点的度数
        print(f"== 测试 node_degree: {node1_id}")
        node1_degree = await storage.node_degree(node1_id)
        print(f"节点 {node1_id} 的度数: {node1_degree}")
        assert node1_degree == 1, f"节点 {node1_id} 的度数应为1，实际为 {node1_degree}"

        # 2.1 测试所有节点的度数
        print("== 测试所有节点的度数")
        node2_degree = await storage.node_degree(node2_id)
        node3_degree = await storage.node_degree(node3_id)
        print(f"节点 {node2_id} 的度数: {node2_degree}")
        print(f"节点 {node3_id} 的度数: {node3_degree}")
        assert node2_degree == 2, f"节点 {node2_id} 的度数应为2，实际为 {node2_degree}"
        assert node3_degree == 1, f"节点 {node3_id} 的度数应为1，实际为 {node3_degree}"

        # 3. 测试 edge_degree - 获取边的度数
        print(f"== 测试 edge_degree: {node1_id} -> {node2_id}")
        edge_degree = await storage.edge_degree(node1_id, node2_id)
        print(f"边 {node1_id} -> {node2_id} 的度数: {edge_degree}")
        assert (
            edge_degree == 3
        ), f"边 {node1_id} -> {node2_id} 的度数应为3，实际为 {edge_degree}"

        # 3.1 测试反向边的度数 - 验证无向图特性
        print(f"== 测试反向边的度数: {node2_id} -> {node1_id}")
        reverse_edge_degree = await storage.edge_degree(node2_id, node1_id)
        print(f"反向边 {node2_id} -> {node1_id} 的度数: {reverse_edge_degree}")
        assert (
            edge_degree == reverse_edge_degree
        ), "正向边和反向边的度数不一致，无向图特性验证失败"
        print("无向图特性验证成功：正向边和反向边的度数一致")

        # 4. 测试 get_node_edges - 获取节点的所有边
        print(f"== 测试 get_node_edges: {node2_id}")
        node2_edges = await storage.get_node_edges(node2_id)
        print(f"节点 {node2_id} 的所有边: {node2_edges}")
        assert (
            len(node2_edges) == 2
        ), f"节点 {node2_id} 应有2条边，实际有 {len(node2_edges)}"

        # 4.1 验证节点边的无向图特性
        print("== 验证节点边的无向图特性")
        # 检查是否包含与node1和node3的连接关系（无论方向）
        has_connection_with_node1 = False
        has_connection_with_node3 = False
        for edge in node2_edges:
            # 检查是否有与node1的连接（无论方向）
            if (edge[0] == node1_id and edge[1] == node2_id) or (
                edge[0] == node2_id and edge[1] == node1_id
            ):
                has_connection_with_node1 = True
            # 检查是否有与node3的连接（无论方向）
            if (edge[0] == node2_id and edge[1] == node3_id) or (
                edge[0] == node3_id and edge[1] == node2_id
            ):
                has_connection_with_node3 = True

        assert (
            has_connection_with_node1
        ), f"节点 {node2_id} 的边列表中应包含与 {node1_id} 的连接"
        assert (
            has_connection_with_node3
        ), f"节点 {node2_id} 的边列表中应包含与 {node3_id} 的连接"
        print(f"无向图特性验证成功：节点 {node2_id} 的边列表包含所有相关的边")

        # 5. 测试 get_all_labels - 获取所有标签
        print("== 测试 get_all_labels")
        all_labels = await storage.get_all_labels()
        print(f"所有标签: {all_labels}")
        assert len(all_labels) == 3, f"应有3个标签，实际有 {len(all_labels)}"
        assert node1_id in all_labels, f"{node1_id} 应在标签列表中"
        assert node2_id in all_labels, f"{node2_id} 应在标签列表中"
        assert node3_id in all_labels, f"{node3_id} 应在标签列表中"

        # 6. 测试 get_knowledge_graph - 获取知识图谱
        print("== 测试 get_knowledge_graph")
        kg = await storage.get_knowledge_graph("*", max_depth=2, max_nodes=10)
        print(f"知识图谱节点数: {len(kg.nodes)}")
        print(f"知识图谱边数: {len(kg.edges)}")
        assert isinstance(kg, KnowledgeGraph), "返回结果应为 KnowledgeGraph 类型"
        assert len(kg.nodes) == 3, f"知识图谱应有3个节点，实际有 {len(kg.nodes)}"
        assert len(kg.edges) == 2, f"知识图谱应有2条边，实际有 {len(kg.edges)}"

        # 7. 测试 delete_node - 删除节点
        print(f"== 测试 delete_node: {node3_id}")
        await storage.delete_node(node3_id)
        node3_props = await storage.get_node(node3_id)
        print(f"删除后查询节点属性 {node3_id}: {node3_props}")
        assert node3_props is None, f"节点 {node3_id} 应已被删除"

        # 重新插入节点3用于后续测试
        await storage.upsert_node(node3_id, node3_data)
        await storage.upsert_edge(node2_id, node3_id, edge2_data)

        # 8. 测试 remove_edges - 删除边
        print(f"== 测试 remove_edges: {node2_id} -> {node3_id}")
        await storage.remove_edges([(node2_id, node3_id)])
        edge_props = await storage.get_edge(node2_id, node3_id)
        print(f"删除后查询边属性 {node2_id} -> {node3_id}: {edge_props}")
        assert edge_props is None, f"边 {node2_id} -> {node3_id} 应已被删除"

        # 8.1 验证删除边的无向图特性
        print(f"== 验证删除边的无向图特性: {node3_id} -> {node2_id}")
        reverse_edge_props = await storage.get_edge(node3_id, node2_id)
        print(f"删除后查询反向边属性 {node3_id} -> {node2_id}: {reverse_edge_props}")
        assert (
            reverse_edge_props is None
        ), f"反向边 {node3_id} -> {node2_id} 也应被删除，无向图特性验证失败"
        print("无向图特性验证成功：删除一个方向的边后，反向边也被删除")

        # 9. 测试 remove_nodes - 批量删除节点
        print(f"== 测试 remove_nodes: [{node2_id}, {node3_id}]")
        await storage.remove_nodes([node2_id, node3_id])
        node2_props = await storage.get_node(node2_id)
        node3_props = await storage.get_node(node3_id)
        print(f"删除后查询节点属性 {node2_id}: {node2_props}")
        print(f"删除后查询节点属性 {node3_id}: {node3_props}")
        assert node2_props is None, f"节点 {node2_id} 应已被删除"
        assert node3_props is None, f"节点 {node3_id} 应已被删除"

        print("\n高级测试完成")
        return True

    except Exception as e:
        ASCIIColors.red(f"测试过程中发生错误: {str(e)}")
        return False


async def test_graph_batch_operations(storage):
    """
    测试图数据库的批量操作:
    1. 使用 get_nodes_batch 批量获取多个节点的属性
    2. 使用 node_degrees_batch 批量获取多个节点的度数
    3. 使用 edge_degrees_batch 批量获取多个边的度数
    4. 使用 get_edges_batch 批量获取多个边的属性
    5. 使用 get_nodes_edges_batch 批量获取多个节点的所有边
    """
    try:
        # 1. 插入测试数据
        # 插入节点1: 人工智能
        node1_id = "人工智能"
        node1_data = {
            "entity_id": node1_id,
            "description": "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。",
            "keywords": "AI,机器学习,深度学习",
            "entity_type": "技术领域",
        }
        print(f"插入节点1: {node1_id}")
        await storage.upsert_node(node1_id, node1_data)

        # 插入节点2: 机器学习
        node2_id = "机器学习"
        node2_data = {
            "entity_id": node2_id,
            "description": "机器学习是人工智能的一个分支，它使用统计学方法让计算机系统在不被明确编程的情况下也能够学习。",
            "keywords": "监督学习,无监督学习,强化学习",
            "entity_type": "技术领域",
        }
        print(f"插入节点2: {node2_id}")
        await storage.upsert_node(node2_id, node2_data)

        # 插入节点3: 深度学习
        node3_id = "深度学习"
        node3_data = {
            "entity_id": node3_id,
            "description": "深度学习是机器学习的一个分支，它使用多层神经网络来模拟人脑的学习过程。",
            "keywords": "神经网络,CNN,RNN",
            "entity_type": "技术领域",
        }
        print(f"插入节点3: {node3_id}")
        await storage.upsert_node(node3_id, node3_data)

        # 插入节点4: 自然语言处理
        node4_id = "自然语言处理"
        node4_data = {
            "entity_id": node4_id,
            "description": "自然语言处理是人工智能的一个分支，专注于使计算机理解和处理人类语言。",
            "keywords": "NLP,文本分析,语言模型",
            "entity_type": "技术领域",
        }
        print(f"插入节点4: {node4_id}")
        await storage.upsert_node(node4_id, node4_data)

        # 插入节点5: 计算机视觉
        node5_id = "计算机视觉"
        node5_data = {
            "entity_id": node5_id,
            "description": "计算机视觉是人工智能的一个分支，专注于使计算机能够从图像或视频中获取信息。",
            "keywords": "CV,图像识别,目标检测",
            "entity_type": "技术领域",
        }
        print(f"插入节点5: {node5_id}")
        await storage.upsert_node(node5_id, node5_data)

        # 插入边1: 人工智能 -> 机器学习
        edge1_data = {
            "relationship": "包含",
            "weight": 1.0,
            "description": "人工智能领域包含机器学习这个子领域",
        }
        print(f"插入边1: {node1_id} -> {node2_id}")
        await storage.upsert_edge(node1_id, node2_id, edge1_data)

        # 插入边2: 机器学习 -> 深度学习
        edge2_data = {
            "relationship": "包含",
            "weight": 1.0,
            "description": "机器学习领域包含深度学习这个子领域",
        }
        print(f"插入边2: {node2_id} -> {node3_id}")
        await storage.upsert_edge(node2_id, node3_id, edge2_data)

        # 插入边3: 人工智能 -> 自然语言处理
        edge3_data = {
            "relationship": "包含",
            "weight": 1.0,
            "description": "人工智能领域包含自然语言处理这个子领域",
        }
        print(f"插入边3: {node1_id} -> {node4_id}")
        await storage.upsert_edge(node1_id, node4_id, edge3_data)

        # 插入边4: 人工智能 -> 计算机视觉
        edge4_data = {
            "relationship": "包含",
            "weight": 1.0,
            "description": "人工智能领域包含计算机视觉这个子领域",
        }
        print(f"插入边4: {node1_id} -> {node5_id}")
        await storage.upsert_edge(node1_id, node5_id, edge4_data)

        # 插入边5: 深度学习 -> 自然语言处理
        edge5_data = {
            "relationship": "应用于",
            "weight": 0.8,
            "description": "深度学习技术应用于自然语言处理领域",
        }
        print(f"插入边5: {node3_id} -> {node4_id}")
        await storage.upsert_edge(node3_id, node4_id, edge5_data)

        # 插入边6: 深度学习 -> 计算机视觉
        edge6_data = {
            "relationship": "应用于",
            "weight": 0.8,
            "description": "深度学习技术应用于计算机视觉领域",
        }
        print(f"插入边6: {node3_id} -> {node5_id}")
        await storage.upsert_edge(node3_id, node5_id, edge6_data)

        # 2. 测试 get_nodes_batch - 批量获取多个节点的属性
        print("== 测试 get_nodes_batch")
        node_ids = [node1_id, node2_id, node3_id]
        nodes_dict = await storage.get_nodes_batch(node_ids)
        print(f"批量获取节点属性结果: {nodes_dict.keys()}")
        assert len(nodes_dict) == 3, f"应返回3个节点，实际返回 {len(nodes_dict)} 个"
        assert node1_id in nodes_dict, f"{node1_id} 应在返回结果中"
        assert node2_id in nodes_dict, f"{node2_id} 应在返回结果中"
        assert node3_id in nodes_dict, f"{node3_id} 应在返回结果中"
        assert (
            nodes_dict[node1_id]["description"] == node1_data["description"]
        ), f"{node1_id} 描述不匹配"
        assert (
            nodes_dict[node2_id]["description"] == node2_data["description"]
        ), f"{node2_id} 描述不匹配"
        assert (
            nodes_dict[node3_id]["description"] == node3_data["description"]
        ), f"{node3_id} 描述不匹配"

        # 3. 测试 node_degrees_batch - 批量获取多个节点的度数
        print("== 测试 node_degrees_batch")
        node_degrees = await storage.node_degrees_batch(node_ids)
        print(f"批量获取节点度数结果: {node_degrees}")
        assert (
            len(node_degrees) == 3
        ), f"应返回3个节点的度数，实际返回 {len(node_degrees)} 个"
        assert node1_id in node_degrees, f"{node1_id} 应在返回结果中"
        assert node2_id in node_degrees, f"{node2_id} 应在返回结果中"
        assert node3_id in node_degrees, f"{node3_id} 应在返回结果中"
        assert (
            node_degrees[node1_id] == 3
        ), f"{node1_id} 度数应为3，实际为 {node_degrees[node1_id]}"
        assert (
            node_degrees[node2_id] == 2
        ), f"{node2_id} 度数应为2，实际为 {node_degrees[node2_id]}"
        assert (
            node_degrees[node3_id] == 3
        ), f"{node3_id} 度数应为3，实际为 {node_degrees[node3_id]}"

        # 4. 测试 edge_degrees_batch - 批量获取多个边的度数
        print("== 测试 edge_degrees_batch")
        edges = [(node1_id, node2_id), (node2_id, node3_id), (node3_id, node4_id)]
        edge_degrees = await storage.edge_degrees_batch(edges)
        print(f"批量获取边度数结果: {edge_degrees}")
        assert (
            len(edge_degrees) == 3
        ), f"应返回3条边的度数，实际返回 {len(edge_degrees)} 条"
        assert (
            node1_id,
            node2_id,
        ) in edge_degrees, f"边 {node1_id} -> {node2_id} 应在返回结果中"
        assert (
            node2_id,
            node3_id,
        ) in edge_degrees, f"边 {node2_id} -> {node3_id} 应在返回结果中"
        assert (
            node3_id,
            node4_id,
        ) in edge_degrees, f"边 {node3_id} -> {node4_id} 应在返回结果中"
        # 验证边的度数是否正确（源节点度数 + 目标节点度数）
        assert (
            edge_degrees[(node1_id, node2_id)] == 5
        ), f"边 {node1_id} -> {node2_id} 度数应为5，实际为 {edge_degrees[(node1_id, node2_id)]}"
        assert (
            edge_degrees[(node2_id, node3_id)] == 5
        ), f"边 {node2_id} -> {node3_id} 度数应为5，实际为 {edge_degrees[(node2_id, node3_id)]}"
        assert (
            edge_degrees[(node3_id, node4_id)] == 5
        ), f"边 {node3_id} -> {node4_id} 度数应为5，实际为 {edge_degrees[(node3_id, node4_id)]}"

        # 5. 测试 get_edges_batch - 批量获取多个边的属性
        print("== 测试 get_edges_batch")
        # 将元组列表转换为Neo4j风格的字典列表
        edge_dicts = [{"src": src, "tgt": tgt} for src, tgt in edges]
        edges_dict = await storage.get_edges_batch(edge_dicts)
        print(f"批量获取边属性结果: {edges_dict.keys()}")
        assert len(edges_dict) == 3, f"应返回3条边的属性，实际返回 {len(edges_dict)} 条"
        assert (
            node1_id,
            node2_id,
        ) in edges_dict, f"边 {node1_id} -> {node2_id} 应在返回结果中"
        assert (
            node2_id,
            node3_id,
        ) in edges_dict, f"边 {node2_id} -> {node3_id} 应在返回结果中"
        assert (
            node3_id,
            node4_id,
        ) in edges_dict, f"边 {node3_id} -> {node4_id} 应在返回结果中"
        assert (
            edges_dict[(node1_id, node2_id)]["relationship"]
            == edge1_data["relationship"]
        ), f"边 {node1_id} -> {node2_id} 关系不匹配"
        assert (
            edges_dict[(node2_id, node3_id)]["relationship"]
            == edge2_data["relationship"]
        ), f"边 {node2_id} -> {node3_id} 关系不匹配"
        assert (
            edges_dict[(node3_id, node4_id)]["relationship"]
            == edge5_data["relationship"]
        ), f"边 {node3_id} -> {node4_id} 关系不匹配"

        # 5.1 测试反向边的批量获取 - 验证无向图特性
        print("== 测试反向边的批量获取")
        # 创建反向边的字典列表
        reverse_edge_dicts = [{"src": tgt, "tgt": src} for src, tgt in edges]
        reverse_edges_dict = await storage.get_edges_batch(reverse_edge_dicts)
        print(f"批量获取反向边属性结果: {reverse_edges_dict.keys()}")
        assert (
            len(reverse_edges_dict) == 3
        ), f"应返回3条反向边的属性，实际返回 {len(reverse_edges_dict)} 条"

        # 验证正向和反向边的属性是否一致
        for (src, tgt), props in edges_dict.items():
            assert (
                tgt,
                src,
            ) in reverse_edges_dict, f"反向边 {tgt} -> {src} 应在返回结果中"
            assert (
                props == reverse_edges_dict[(tgt, src)]
            ), f"边 {src} -> {tgt} 和反向边 {tgt} -> {src} 的属性不一致"

        print("无向图特性验证成功：批量获取的正向和反向边属性一致")

        # 6. 测试 get_nodes_edges_batch - 批量获取多个节点的所有边
        print("== 测试 get_nodes_edges_batch")
        nodes_edges = await storage.get_nodes_edges_batch([node1_id, node3_id])
        print(f"批量获取节点边结果: {nodes_edges.keys()}")
        assert (
            len(nodes_edges) == 2
        ), f"应返回2个节点的边，实际返回 {len(nodes_edges)} 个"
        assert node1_id in nodes_edges, f"{node1_id} 应在返回结果中"
        assert node3_id in nodes_edges, f"{node3_id} 应在返回结果中"
        assert (
            len(nodes_edges[node1_id]) == 3
        ), f"{node1_id} 应有3条边，实际有 {len(nodes_edges[node1_id])} 条"
        assert (
            len(nodes_edges[node3_id]) == 3
        ), f"{node3_id} 应有3条边，实际有 {len(nodes_edges[node3_id])} 条"

        # 6.1 验证批量获取节点边的无向图特性
        print("== 验证批量获取节点边的无向图特性")

        # 检查节点1的边是否包含所有相关的边（无论方向）
        node1_outgoing_edges = [
            (src, tgt) for src, tgt in nodes_edges[node1_id] if src == node1_id
        ]
        node1_incoming_edges = [
            (src, tgt) for src, tgt in nodes_edges[node1_id] if tgt == node1_id
        ]
        print(f"节点 {node1_id} 的出边: {node1_outgoing_edges}")
        print(f"节点 {node1_id} 的入边: {node1_incoming_edges}")

        # 检查是否包含到机器学习、自然语言处理和计算机视觉的边
        has_edge_to_node2 = any(tgt == node2_id for _, tgt in node1_outgoing_edges)
        has_edge_to_node4 = any(tgt == node4_id for _, tgt in node1_outgoing_edges)
        has_edge_to_node5 = any(tgt == node5_id for _, tgt in node1_outgoing_edges)

        assert has_edge_to_node2, f"节点 {node1_id} 的边列表中应包含到 {node2_id} 的边"
        assert has_edge_to_node4, f"节点 {node1_id} 的边列表中应包含到 {node4_id} 的边"
        assert has_edge_to_node5, f"节点 {node1_id} 的边列表中应包含到 {node5_id} 的边"

        # 检查节点3的边是否包含所有相关的边（无论方向）
        node3_outgoing_edges = [
            (src, tgt) for src, tgt in nodes_edges[node3_id] if src == node3_id
        ]
        node3_incoming_edges = [
            (src, tgt) for src, tgt in nodes_edges[node3_id] if tgt == node3_id
        ]
        print(f"节点 {node3_id} 的出边: {node3_outgoing_edges}")
        print(f"节点 {node3_id} 的入边: {node3_incoming_edges}")

        # 检查是否包含与机器学习、自然语言处理和计算机视觉的连接（忽略方向）
        has_connection_with_node2 = any(
            (src == node2_id and tgt == node3_id)
            or (src == node3_id and tgt == node2_id)
            for src, tgt in nodes_edges[node3_id]
        )
        has_connection_with_node4 = any(
            (src == node3_id and tgt == node4_id)
            or (src == node4_id and tgt == node3_id)
            for src, tgt in nodes_edges[node3_id]
        )
        has_connection_with_node5 = any(
            (src == node3_id and tgt == node5_id)
            or (src == node5_id and tgt == node3_id)
            for src, tgt in nodes_edges[node3_id]
        )

        assert (
            has_connection_with_node2
        ), f"节点 {node3_id} 的边列表中应包含与 {node2_id} 的连接"
        assert (
            has_connection_with_node4
        ), f"节点 {node3_id} 的边列表中应包含与 {node4_id} 的连接"
        assert (
            has_connection_with_node5
        ), f"节点 {node3_id} 的边列表中应包含与 {node5_id} 的连接"

        print("无向图特性验证成功：批量获取的节点边包含所有相关的边（无论方向）")

        print("\n批量操作测试完成")
        return True

    except Exception as e:
        ASCIIColors.red(f"测试过程中发生错误: {str(e)}")
        return False


async def test_graph_special_characters(storage):
    """
    测试图数据库对特殊字符的处理:
    1. 测试节点名称和描述中包含单引号、双引号和反斜杠
    2. 测试边的描述中包含单引号、双引号和反斜杠
    3. 验证特殊字符是否被正确保存和检索
    """
    try:
        # 1. 测试节点名称中的特殊字符
        node1_id = "包含'单引号'的节点"
        node1_data = {
            "entity_id": node1_id,
            "description": "这个描述包含'单引号'、\"双引号\"和\\反斜杠",
            "keywords": "特殊字符,引号,转义",
            "entity_type": "测试节点",
        }
        print(f"插入包含特殊字符的节点1: {node1_id}")
        await storage.upsert_node(node1_id, node1_data)

        # 2. 测试节点名称中的双引号
        node2_id = '包含"双引号"的节点'
        node2_data = {
            "entity_id": node2_id,
            "description": "这个描述同时包含'单引号'和\"双引号\"以及\\反斜杠\\路径",
            "keywords": "特殊字符,引号,JSON",
            "entity_type": "测试节点",
        }
        print(f"插入包含特殊字符的节点2: {node2_id}")
        await storage.upsert_node(node2_id, node2_data)

        # 3. 测试节点名称中的反斜杠
        node3_id = "包含\\反斜杠\\的节点"
        node3_data = {
            "entity_id": node3_id,
            "description": "这个描述包含Windows路径C:\\Program Files\\和转义字符\\n\\t",
            "keywords": "反斜杠,路径,转义",
            "entity_type": "测试节点",
        }
        print(f"插入包含特殊字符的节点3: {node3_id}")
        await storage.upsert_node(node3_id, node3_data)

        # 4. 测试边描述中的特殊字符
        edge1_data = {
            "relationship": "特殊'关系'",
            "weight": 1.0,
            "description": "这个边描述包含'单引号'、\"双引号\"和\\反斜杠",
        }
        print(f"插入包含特殊字符的边: {node1_id} -> {node2_id}")
        await storage.upsert_edge(node1_id, node2_id, edge1_data)

        # 5. 测试边描述中的更复杂特殊字符组合
        edge2_data = {
            "relationship": '复杂"关系"\\类型',
            "weight": 0.8,
            "description": "包含SQL注入尝试: SELECT * FROM users WHERE name='admin'--",
        }
        print(f"插入包含复杂特殊字符的边: {node2_id} -> {node3_id}")
        await storage.upsert_edge(node2_id, node3_id, edge2_data)

        # 6. 验证节点特殊字符是否正确保存
        print("\n== 验证节点特殊字符")
        for node_id, original_data in [
            (node1_id, node1_data),
            (node2_id, node2_data),
            (node3_id, node3_data),
        ]:
            node_props = await storage.get_node(node_id)
            if node_props:
                print(f"成功读取节点: {node_id}")
                print(f"节点描述: {node_props.get('description', '无描述')}")

                # 验证节点ID是否正确保存
                assert (
                    node_props.get("entity_id") == node_id
                ), f"节点ID不匹配: 期望 {node_id}, 实际 {node_props.get('entity_id')}"

                # 验证描述是否正确保存
                assert (
                    node_props.get("description") == original_data["description"]
                ), f"节点描述不匹配: 期望 {original_data['description']}, 实际 {node_props.get('description')}"

                print(f"节点 {node_id} 特殊字符验证成功")
            else:
                print(f"读取节点属性失败: {node_id}")
                assert False, f"未能读取节点属性: {node_id}"

        # 7. 验证边特殊字符是否正确保存
        print("\n== 验证边特殊字符")
        edge1_props = await storage.get_edge(node1_id, node2_id)
        if edge1_props:
            print(f"成功读取边: {node1_id} -> {node2_id}")
            print(f"边关系: {edge1_props.get('relationship', '无关系')}")
            print(f"边描述: {edge1_props.get('description', '无描述')}")

            # 验证边关系是否正确保存
            assert (
                edge1_props.get("relationship") == edge1_data["relationship"]
            ), f"边关系不匹配: 期望 {edge1_data['relationship']}, 实际 {edge1_props.get('relationship')}"

            # 验证边描述是否正确保存
            assert (
                edge1_props.get("description") == edge1_data["description"]
            ), f"边描述不匹配: 期望 {edge1_data['description']}, 实际 {edge1_props.get('description')}"

            print(f"边 {node1_id} -> {node2_id} 特殊字符验证成功")
        else:
            print(f"读取边属性失败: {node1_id} -> {node2_id}")
            assert False, f"未能读取边属性: {node1_id} -> {node2_id}"

        edge2_props = await storage.get_edge(node2_id, node3_id)
        if edge2_props:
            print(f"成功读取边: {node2_id} -> {node3_id}")
            print(f"边关系: {edge2_props.get('relationship', '无关系')}")
            print(f"边描述: {edge2_props.get('description', '无描述')}")

            # 验证边关系是否正确保存
            assert (
                edge2_props.get("relationship") == edge2_data["relationship"]
            ), f"边关系不匹配: 期望 {edge2_data['relationship']}, 实际 {edge2_props.get('relationship')}"

            # 验证边描述是否正确保存
            assert (
                edge2_props.get("description") == edge2_data["description"]
            ), f"边描述不匹配: 期望 {edge2_data['description']}, 实际 {edge2_props.get('description')}"

            print(f"边 {node2_id} -> {node3_id} 特殊字符验证成功")
        else:
            print(f"读取边属性失败: {node2_id} -> {node3_id}")
            assert False, f"未能读取边属性: {node2_id} -> {node3_id}"

        print("\n特殊字符测试完成，数据已保留在数据库中")
        return True

    except Exception as e:
        ASCIIColors.red(f"测试过程中发生错误: {str(e)}")
        return False


async def test_graph_undirected_property(storage):
    """
    专门测试图存储的无向图特性:
    1. 验证插入一个方向的边后，反向查询是否能获得相同的结果
    2. 验证边的属性在正向和反向查询中是否一致
    3. 验证删除一个方向的边后，另一个方向的边是否也被删除
    4. 验证批量操作中的无向图特性
    """
    try:
        # 1. 插入测试数据
        # 插入节点1: 计算机科学
        node1_id = "计算机科学"
        node1_data = {
            "entity_id": node1_id,
            "description": "计算机科学是研究计算机及其应用的科学。",
            "keywords": "计算机,科学,技术",
            "entity_type": "学科",
        }
        print(f"插入节点1: {node1_id}")
        await storage.upsert_node(node1_id, node1_data)

        # 插入节点2: 数据结构
        node2_id = "数据结构"
        node2_data = {
            "entity_id": node2_id,
            "description": "数据结构是计算机科学中的一个基础概念，用于组织和存储数据。",
            "keywords": "数据,结构,组织",
            "entity_type": "概念",
        }
        print(f"插入节点2: {node2_id}")
        await storage.upsert_node(node2_id, node2_data)

        # 插入节点3: 算法
        node3_id = "算法"
        node3_data = {
            "entity_id": node3_id,
            "description": "算法是解决问题的步骤和方法。",
            "keywords": "算法,步骤,方法",
            "entity_type": "概念",
        }
        print(f"插入节点3: {node3_id}")
        await storage.upsert_node(node3_id, node3_data)

        # 2. 测试插入边后的无向图特性
        print("\n== 测试插入边后的无向图特性")

        # 插入边1: 计算机科学 -> 数据结构
        edge1_data = {
            "relationship": "包含",
            "weight": 1.0,
            "description": "计算机科学包含数据结构这个概念",
        }
        print(f"插入边1: {node1_id} -> {node2_id}")
        await storage.upsert_edge(node1_id, node2_id, edge1_data)

        # 验证正向查询
        forward_edge = await storage.get_edge(node1_id, node2_id)
        print(f"正向边属性: {forward_edge}")
        assert forward_edge is not None, f"未能读取正向边属性: {node1_id} -> {node2_id}"

        # 验证反向查询
        reverse_edge = await storage.get_edge(node2_id, node1_id)
        print(f"反向边属性: {reverse_edge}")
        assert reverse_edge is not None, f"未能读取反向边属性: {node2_id} -> {node1_id}"

        # 验证正向和反向边属性是否一致
        assert (
            forward_edge == reverse_edge
        ), "正向和反向边属性不一致，无向图特性验证失败"
        print("无向图特性验证成功：正向和反向边属性一致")

        # 3. 测试边的度数的无向图特性
        print("\n== 测试边的度数的无向图特性")

        # 插入边2: 计算机科学 -> 算法
        edge2_data = {
            "relationship": "包含",
            "weight": 1.0,
            "description": "计算机科学包含算法这个概念",
        }
        print(f"插入边2: {node1_id} -> {node3_id}")
        await storage.upsert_edge(node1_id, node3_id, edge2_data)

        # 验证正向和反向边的度数
        forward_degree = await storage.edge_degree(node1_id, node2_id)
        reverse_degree = await storage.edge_degree(node2_id, node1_id)
        print(f"正向边 {node1_id} -> {node2_id} 的度数: {forward_degree}")
        print(f"反向边 {node2_id} -> {node1_id} 的度数: {reverse_degree}")
        assert (
            forward_degree == reverse_degree
        ), "正向和反向边的度数不一致，无向图特性验证失败"
        print("无向图特性验证成功：正向和反向边的度数一致")

        # 4. 测试删除边的无向图特性
        print("\n== 测试删除边的无向图特性")

        # 删除正向边
        print(f"删除边: {node1_id} -> {node2_id}")
        await storage.remove_edges([(node1_id, node2_id)])

        # 验证正向边是否被删除
        forward_edge = await storage.get_edge(node1_id, node2_id)
        print(f"删除后查询正向边属性 {node1_id} -> {node2_id}: {forward_edge}")
        assert forward_edge is None, f"边 {node1_id} -> {node2_id} 应已被删除"

        # 验证反向边是否也被删除
        reverse_edge = await storage.get_edge(node2_id, node1_id)
        print(f"删除后查询反向边属性 {node2_id} -> {node1_id}: {reverse_edge}")
        assert (
            reverse_edge is None
        ), f"反向边 {node2_id} -> {node1_id} 也应被删除，无向图特性验证失败"
        print("无向图特性验证成功：删除一个方向的边后，反向边也被删除")

        # 5. 测试批量操作中的无向图特性
        print("\n== 测试批量操作中的无向图特性")

        # 重新插入边
        await storage.upsert_edge(node1_id, node2_id, edge1_data)

        # 批量获取边属性
        edge_dicts = [
            {"src": node1_id, "tgt": node2_id},
            {"src": node1_id, "tgt": node3_id},
        ]
        reverse_edge_dicts = [
            {"src": node2_id, "tgt": node1_id},
            {"src": node3_id, "tgt": node1_id},
        ]

        edges_dict = await storage.get_edges_batch(edge_dicts)
        reverse_edges_dict = await storage.get_edges_batch(reverse_edge_dicts)

        print(f"批量获取正向边属性结果: {edges_dict.keys()}")
        print(f"批量获取反向边属性结果: {reverse_edges_dict.keys()}")

        # 验证正向和反向边的属性是否一致
        for (src, tgt), props in edges_dict.items():
            assert (
                tgt,
                src,
            ) in reverse_edges_dict, f"反向边 {tgt} -> {src} 应在返回结果中"
            assert (
                props == reverse_edges_dict[(tgt, src)]
            ), f"边 {src} -> {tgt} 和反向边 {tgt} -> {src} 的属性不一致"

        print("无向图特性验证成功：批量获取的正向和反向边属性一致")

        # 6. 测试批量获取节点边的无向图特性
        print("\n== 测试批量获取节点边的无向图特性")

        nodes_edges = await storage.get_nodes_edges_batch([node1_id, node2_id])
        print(f"批量获取节点边结果: {nodes_edges.keys()}")

        # 检查节点1的边是否包含所有相关的边（无论方向）
        node1_edges = nodes_edges[node1_id]
        node2_edges = nodes_edges[node2_id]

        # 检查节点1是否有到节点2和节点3的边
        has_edge_to_node2 = any(
            (src == node1_id and tgt == node2_id) for src, tgt in node1_edges
        )
        has_edge_to_node3 = any(
            (src == node1_id and tgt == node3_id) for src, tgt in node1_edges
        )

        assert has_edge_to_node2, f"节点 {node1_id} 的边列表中应包含到 {node2_id} 的边"
        assert has_edge_to_node3, f"节点 {node1_id} 的边列表中应包含到 {node3_id} 的边"

        # 检查节点2是否有到节点1的边
        has_edge_to_node1 = any(
            (src == node2_id and tgt == node1_id)
            or (src == node1_id and tgt == node2_id)
            for src, tgt in node2_edges
        )
        assert (
            has_edge_to_node1
        ), f"节点 {node2_id} 的边列表中应包含与 {node1_id} 的连接"

        print("无向图特性验证成功：批量获取的节点边包含所有相关的边（无论方向）")

        print("\n无向图特性测试完成")
        return True

    except Exception as e:
        ASCIIColors.red(f"测试过程中发生错误: {str(e)}")
        return False


async def main():
    """主函数"""
    # 显示程序标题
    ASCIIColors.cyan("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                  通用图存储测试程序                          ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    # 检查.env文件
    if not check_env_file():
        return

    # 加载环境变量
    load_dotenv(dotenv_path=".env", override=False)

    # 获取图存储类型
    graph_storage_type = os.getenv("LIGHTRAG_GRAPH_STORAGE", "NetworkXStorage")
    ASCIIColors.magenta(f"\n当前配置的图存储类型: {graph_storage_type}")
    ASCIIColors.white(
        f"支持的图存储类型: {', '.join(STORAGE_IMPLEMENTATIONS['GRAPH_STORAGE']['implementations'])}"
    )

    # 初始化存储实例
    storage = await initialize_graph_storage()
    if not storage:
        ASCIIColors.red("初始化存储实例失败，测试程序退出")
        return

    try:
        # 显示测试选项
        ASCIIColors.yellow("\n请选择测试类型:")
        ASCIIColors.white("1. 基本测试 (节点和边的插入、读取)")
        ASCIIColors.white("2. 高级测试 (度数、标签、知识图谱、删除操作等)")
        ASCIIColors.white("3. 批量操作测试 (批量获取节点、边属性和度数等)")
        ASCIIColors.white("4. 无向图特性测试 (验证存储的无向图特性)")
        ASCIIColors.white("5. 特殊字符测试 (验证单引号、双引号和反斜杠等特殊字符)")
        ASCIIColors.white("6. 全部测试")

        choice = input("\n请输入选项 (1/2/3/4/5/6): ")

        # 在执行测试前清理数据
        if choice in ["1", "2", "3", "4", "5", "6"]:
            ASCIIColors.yellow("\n执行测试前清理数据...")
            await storage.drop()
            ASCIIColors.green("数据清理完成\n")

        if choice == "1":
            await test_graph_basic(storage)
        elif choice == "2":
            await test_graph_advanced(storage)
        elif choice == "3":
            await test_graph_batch_operations(storage)
        elif choice == "4":
            await test_graph_undirected_property(storage)
        elif choice == "5":
            await test_graph_special_characters(storage)
        elif choice == "6":
            ASCIIColors.cyan("\n=== 开始基本测试 ===")
            basic_result = await test_graph_basic(storage)

            if basic_result:
                ASCIIColors.cyan("\n=== 开始高级测试 ===")
                advanced_result = await test_graph_advanced(storage)

                if advanced_result:
                    ASCIIColors.cyan("\n=== 开始批量操作测试 ===")
                    batch_result = await test_graph_batch_operations(storage)

                    if batch_result:
                        ASCIIColors.cyan("\n=== 开始无向图特性测试 ===")
                        undirected_result = await test_graph_undirected_property(
                            storage
                        )

                        if undirected_result:
                            ASCIIColors.cyan("\n=== 开始特殊字符测试 ===")
                            await test_graph_special_characters(storage)
        else:
            ASCIIColors.red("无效的选项")

    finally:
        # 关闭连接
        if storage:
            await storage.finalize()
            ASCIIColors.green("\n存储连接已关闭")


if __name__ == "__main__":
    asyncio.run(main())
