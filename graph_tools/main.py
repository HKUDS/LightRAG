"""
主入口模块

作为程序的入口点，负责解析命令行参数和协调调用其他模块
"""

import os
import sys
import logging
import argparse
import asyncio
from typing import Dict, Any, List

from graph_tools.models import Config, Entity
from graph_tools.config_loader import load_app_config, setup_logging, load_schema_config
from graph_tools.utils import load_json_data, save_cypher_to_file, normalize_data_format
from graph_tools.processing import process_document_structure, extract_semantic_info, create_contains_relations
from graph_tools.cypher_generator import generate_cypher_statements
from graph_tools.llm_client import LLMClient
from graph_tools.vector_service import get_embedding, SILICONFLOW_API_KEY

# 用于格式化实体文本以生成嵌入
def format_entity_text_for_embedding(entity: Entity) -> str:
    """根据实体属性格式化文本用于嵌入"""
    text_parts = []
    
    if entity.type:
        text_parts.append(f"类型: {entity.type}")
    if entity.name:
        text_parts.append(f"名称: {entity.name}")
    if entity.description and entity.description != entity.name:  # 避免重复
        text_parts.append(f"描述: {entity.description}")
    if entity.main_category:
        text_parts.append(f"主分类: {entity.main_category}")
    if entity.source:
        text_parts.append(f"来源: {entity.source}")
        
    return ", ".join(text_parts)

# 批量生成实体嵌入向量
async def generate_entity_embeddings(entities: List[Entity], batch_size: int = 8) -> List[Entity]:
    """
    为实体生成嵌入向量
    
    Args:
        entities: 实体列表
        batch_size: 批处理大小
        
    Returns:
        更新了嵌入向量的实体列表
    """
    updated_entities = []
    total = len(entities)
    
    logging.info(f"开始为 {total} 个实体生成嵌入向量...")
    
    for i in range(0, total, batch_size):
        batch = entities[i:i+batch_size]
        batch_texts = [format_entity_text_for_embedding(entity) for entity in batch]
        
        try:
            # 批量获取嵌入向量
            embeddings = await get_embedding(batch_texts, SILICONFLOW_API_KEY)
            
            # 更新实体的嵌入向量
            for j, entity in enumerate(batch):
                entity.vector = embeddings[j]
                updated_entities.append(entity)
                
            logging.info(f"已处理 {min(i+batch_size, total)}/{total} 个实体")
            
            # 添加延迟避免API请求过快
            if i + batch_size < total:
                await asyncio.sleep(0.5)
                
        except Exception as e:
            logging.error(f"处理实体批次时出错: {str(e)}")
            # 仍然添加未处理的实体以保持完整性
            for entity in batch:
                if entity.vector is None:
                    updated_entities.append(entity)
    
    logging.info(f"嵌入向量生成完成，成功处理 {len([e for e in updated_entities if e.vector is not None])}/{total} 个实体")
    return updated_entities

async def main_async(input_json_path: str, config_path: str, output_cypher_path: str, app_config: Dict[str, Any]):
    """
    主处理流程
    
    Args:
        input_json_path: 输入JSON文件路径
        config_path: 配置YAML文件路径
        output_cypher_path: 输出Cypher文件路径
        app_config: 应用程序配置
    """
    try:
        # 1. 加载图谱模式配置
        logging.info(f"从 {config_path} 加载配置")
        schema_config = load_schema_config(config_path)
        
        # 2. 创建LLM客户端
        llm_client = LLMClient(
            api_key=app_config["llm_api_key"],
            api_host=app_config["llm_api_host"],
            model=app_config["llm_model"],
            max_concurrent=app_config["max_concurrent_requests"],
            request_delay=app_config["request_delay"],
            timeout=app_config["request_timeout"],
            max_retries=app_config["max_retries"]
        )

        # 3. 加载输入数据
        logging.info(f"从 {input_json_path} 加载输入数据")
        raw_data = load_json_data(input_json_path)
        if not raw_data:
            logging.error("加载输入数据失败。退出。")
            return
            
        # 标准化数据格式
        data = normalize_data_format(raw_data)
        
        # 4. 处理文档结构
        logging.info("处理文档结构...")
        structure_entities, structure_relations = await process_document_structure(data, schema_config)
        logging.info(f"创建了 {len(structure_entities)} 个结构实体和 {len(structure_relations)} 个结构关系。")
        
        # 过滤出Section实体，用于后续处理
        section_entities = [entity for entity in structure_entities if entity.type == "Section"]
        
        # 5. 使用LLM提取语义信息
        logging.info("使用LLM提取语义信息...")
        semantic_entities, semantic_relations = await extract_semantic_info(data, section_entities, schema_config, llm_client)
        logging.info(f"提取了 {len(semantic_entities)} 个语义实体和 {len(semantic_relations)} 个语义关系。")
        
        # 6. 创建CONTAINS关系
        logging.info("创建章节-实体CONTAINS关系...")
        contains_relations = create_contains_relations(section_entities, semantic_entities)
        logging.info(f"创建了 {len(contains_relations)} 个CONTAINS关系。")
        
        # 7. 合并所有实体和关系
        all_entities = structure_entities + semantic_entities
        all_relations = structure_relations + semantic_relations + contains_relations
        
        logging.info(f"总共有 {len(all_entities)} 个实体和 {len(all_relations)} 个关系。")
        
        # 8. 为实体生成嵌入向量
        logging.info("为实体生成嵌入向量...")
        if app_config.get("generate_embeddings", True):
            # 为所有实体生成嵌入向量，包括结构实体和语义实体
            updated_entities = await generate_entity_embeddings(all_entities)
            all_entities = updated_entities
            logging.info(f"已为 {len([e for e in all_entities if e.vector is not None])} 个实体生成嵌入向量。")
        else:
            logging.info("嵌入向量生成已禁用。")
        
        # 9. 生成Cypher语句并保存
        logging.info("生成Cypher语句...")
        cypher_statements = generate_cypher_statements(all_entities, all_relations, schema_config)
        save_cypher_to_file(cypher_statements, output_cypher_path)
        logging.info(f"Cypher语句已保存到 {output_cypher_path}")
        logging.info("处理完成。")
        
    except Exception as e:
        logging.error(f"处理过程中发生错误: {e}")
        logging.exception("详细错误信息:")

def main():
    """
    主函数，协调整个处理流程
    
    解析命令行参数，设置日志，调用处理函数
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='知识图谱实体关系提取工具')
    parser.add_argument('-i', '--input', required=True, help='输入JSON文件路径 (必须)')
    parser.add_argument('-o', '--output', required=True, help='输出Cypher文件路径 (必须)')
    parser.add_argument('-c', '--config', required=True, help='配置YAML文件路径 (必须)')
    parser.add_argument('-l', '--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default=None, help='日志级别，默认使用环境变量LOG_LEVEL或INFO')
    parser.add_argument('--embeddings', action='store_true', help='为实体生成嵌入向量')
    parser.add_argument('--no-embeddings', action='store_true', help='不为实体生成嵌入向量')
    
    args = parser.parse_args()
    
    # 加载应用程序配置
    app_config = load_app_config()
    
    # 如果命令行指定了日志级别，覆盖配置中的设置
    if args.log_level:
        log_level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        app_config["log_level"] = args.log_level
        app_config["logging_level"] = log_level_map.get(args.log_level, logging.INFO)
    
    # 处理嵌入向量生成设置
    if args.embeddings and args.no_embeddings:
        logging.warning("--embeddings和--no-embeddings选项不能同时使用，将使用默认设置。")
    elif args.embeddings:
        app_config["generate_embeddings"] = True
    elif args.no_embeddings:
        app_config["generate_embeddings"] = False
    
    # 设置日志
    setup_logging(app_config)
    
    # 获取绝对路径
    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)
    config_path = os.path.abspath(args.config)
    
    # 检查API密钥是否设置
    if not app_config["llm_api_key"]:
        logging.error("环境变量 LLM_BINDING_API_KEY 或 SILICONFLOW_API_KEY 未设置，API调用将会失败")
        logging.error("请设置环境变量: export LLM_BINDING_API_KEY='your_api_key_here'")
        return
    
    # 检查API主机和模型是否设置
    if not app_config["llm_api_host"]:
        logging.error("环境变量 LLM_BINDING_HOST 未设置")
        logging.error("请设置环境变量: export LLM_BINDING_HOST='https://api.siliconflow.cn/v1'")
        return
        
    if not app_config["llm_model"]:
        logging.error("环境变量 LLM_MODEL 未设置")
        logging.error("请设置环境变量: export LLM_MODEL='Qwen/Qwen2.5-14B-Instruct'")
        return
    
    # 验证文件是否存在
    if not os.path.exists(config_path):
        logging.error(f"配置文件不存在: {config_path}")
        return
        
    if not os.path.exists(input_path):
        logging.error(f"输入文件不存在: {input_path}")
        return
    
    # 记录生成嵌入向量的设置
    if app_config["generate_embeddings"]:
        logging.info("将为实体生成嵌入向量")
    else:
        logging.info("不生成嵌入向量")
    
    # 启动异步处理流程
    try:
        asyncio.run(main_async(input_path, config_path, output_path, app_config))
    except KeyboardInterrupt:
        logging.info("用户中断程序")
    except Exception as e:
        logging.error(f"程序执行过程中发生错误: {e}")
        logging.exception("详细错误信息:")
        sys.exit(1)

if __name__ == "__main__":
    main() 