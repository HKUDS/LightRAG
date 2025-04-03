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
        # 清理之前的测试数据
        print("清理之前的测试数据...")
        await storage.drop()

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
        # 清理之前的测试数据
        print("清理之前的测试数据...\n")
        await storage.drop()

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

        # 3. 测试 edge_degree - 获取边的度数
        print(f"== 测试 edge_degree: {node1_id} -> {node2_id}")
        edge_degree = await storage.edge_degree(node1_id, node2_id)
        print(f"边 {node1_id} -> {node2_id} 的度数: {edge_degree}")
        assert (
            edge_degree == 3
        ), f"边 {node1_id} -> {node2_id} 的度数应为2，实际为 {edge_degree}"

        # 4. 测试 get_node_edges - 获取节点的所有边
        print(f"== 测试 get_node_edges: {node2_id}")
        node2_edges = await storage.get_node_edges(node2_id)
        print(f"节点 {node2_id} 的所有边: {node2_edges}")
        assert (
            len(node2_edges) == 2
        ), f"节点 {node2_id} 应有2条边，实际有 {len(node2_edges)}"

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

        # 9. 测试 remove_nodes - 批量删除节点
        print(f"== 测试 remove_nodes: [{node2_id}, {node3_id}]")
        await storage.remove_nodes([node2_id, node3_id])
        node2_props = await storage.get_node(node2_id)
        node3_props = await storage.get_node(node3_id)
        print(f"删除后查询节点属性 {node2_id}: {node2_props}")
        print(f"删除后查询节点属性 {node3_id}: {node3_props}")
        assert node2_props is None, f"节点 {node2_id} 应已被删除"
        assert node3_props is None, f"节点 {node3_id} 应已被删除"

        # 10. 测试 drop - 清理数据
        print("== 测试 drop")
        result = await storage.drop()
        print(f"清理结果: {result}")
        assert (
            result["status"] == "success"
        ), f"清理应成功，实际状态为 {result['status']}"

        # 验证清理结果
        all_labels = await storage.get_all_labels()
        print(f"清理后的所有标签: {all_labels}")
        assert len(all_labels) == 0, f"清理后应没有标签，实际有 {len(all_labels)}"

        print("\n高级测试完成")
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
        ASCIIColors.white("3. 全部测试")

        choice = input("\n请输入选项 (1/2/3): ")

        if choice == "1":
            await test_graph_basic(storage)
        elif choice == "2":
            await test_graph_advanced(storage)
        elif choice == "3":
            ASCIIColors.cyan("\n=== 开始基本测试 ===")
            basic_result = await test_graph_basic(storage)

            if basic_result:
                ASCIIColors.cyan("\n=== 开始高级测试 ===")
                await test_graph_advanced(storage)
        else:
            ASCIIColors.red("无效的选项")

    finally:
        # 关闭连接
        if storage:
            await storage.finalize()
            ASCIIColors.green("\n存储连接已关闭")


if __name__ == "__main__":
    asyncio.run(main())
