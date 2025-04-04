"""
Cypher 生成模块

负责将实体和关系转换为Neo4j的Cypher查询语句
"""

import logging
from typing import List, Dict, Set
from datetime import datetime

from graph_tools.models import Entity, Relation, Config
from graph_tools.normalization import normalize_relation_type, escape_cypher_string

def generate_cypher_statements(entities: List[Entity], relations: List[Relation], config: Config) -> List[str]:
    """
    根据实体和关系生成Cypher语句
    
    Args:
        entities: 实体列表
        relations: 关系列表
        config: 配置对象
        
    Returns:
        List[str]: Cypher语句列表
    """
    cypher_statements = []

    # 添加唯一性约束
    cypher_statements.append("\n// --- Uniqueness Constraints ---")
    cypher_statements.append("// 注意：以下约束语句应在数据库初始化时执行一次")
    cypher_statements.append("// 多次执行会报错，建议在首次运行时使用，后续导入数据时注释掉")
    for entity_type_cn in config.entity_types_llm:
        entity_type_cypher = config.entity_type_map_cypher.get(entity_type_cn, entity_type_cn) # Use mapped or original
        # Neo4j 4.x+
        cypher_statements.append(f"// CREATE CONSTRAINT IF NOT EXISTS ON (n:`{entity_type_cypher}`) ASSERT n.name IS UNIQUE;")
        # 旧版本Neo4j或Memgraph
        cypher_statements.append(f"// CREATE CONSTRAINT ON (n:`{entity_type_cypher}`) ASSERT n.name IS UNIQUE;")

    cypher_statements.append("\n// --- Entity Creation ---")
    
    # 创建实体类型到实体名称的映射，方便后续查找
    entity_type_mapping = {}
    for entity in entities:
        if not entity.name: # Skip empty names
            continue
        if entity.name not in entity_type_mapping:
            entity_type_mapping[entity.name] = set()
        entity_type_mapping[entity.name].add(entity.type)
        
        escaped_name = escape_cypher_string(entity.name)
        escaped_description = escape_cypher_string(entity.description)
        escaped_source = escape_cypher_string(entity.source)
        escaped_main_category = escape_cypher_string(entity.main_category)
        escaped_issuing_authority = escape_cypher_string(entity.issuing_authority)
        
        # 格式化日期时间
        created_at_str = entity.created_at.isoformat() if isinstance(entity.created_at, datetime) else str(entity.created_at)
        updated_at_str = entity.updated_at.isoformat() if isinstance(entity.updated_at, datetime) else str(entity.updated_at)
        
        # 添加唯一ID属性以提高唯一性识别能力，并添加所有Entity属性
        cypher_statements.append(f"MERGE (n:`{entity.type}` {{name: '{escaped_name}'}}) "
                               f"ON CREATE SET n.uuid = '{entity.type}_' + timestamp() + '_' + toString(rand()), "
                               f"n.context_id = '{entity.context_id}', "
                               f"n.entity_type = '{entity.type}', "
                               f"n.main_category = '{escaped_main_category}', "
                               f"n.description = '{escaped_description}', "
                               f"n.source = '{escaped_source}', "
                               f"n.created_at = '{created_at_str}', "
                               f"n.updated_at = '{updated_at_str}', "
                               f"n.issuing_authority = '{escaped_issuing_authority}';")

    cypher_statements.append("\n// --- Relationship Creation ---")
    # 记录无法匹配类型的关系数量
    untyped_relations_count = 0
    ambiguous_relations_count = 0
    
    for relation in relations:
        if not relation.source or not relation.target: # Skip if source or target is missing
            continue
        
        escaped_source = escape_cypher_string(relation.source)
        escaped_target = escape_cypher_string(relation.target)

        # 查找实体的类型
        source_types = entity_type_mapping.get(relation.source, set())
        target_types = entity_type_mapping.get(relation.target, set())
        
        # 针对系统名称的特殊处理
        # 处理源实体，如果是系统名称并且包含多个类型，优先使用Topic
        if len(source_types) > 1 and "Topic" in source_types and "Section" in source_types:
            if "系统" in relation.source or "12306" in relation.source:
                source_type = "Topic"
                logging.info(f"实体 '{relation.source}' 歧义解决为 Topic 类型（系统名称）")
            else:
                source_type = None
        else:
            source_type = next(iter(source_types)) if len(source_types) == 1 else None
        
        # 处理目标实体，如果是系统名称并且包含多个类型，优先使用Topic
        if len(target_types) > 1 and "Topic" in target_types and "Section" in target_types:
            if "系统" in relation.target or "12306" in relation.target:
                target_type = "Topic"
                logging.info(f"实体 '{relation.target}' 歧义解决为 Topic 类型（系统名称）")
            else:
                target_type = None
        else:
            target_type = next(iter(target_types)) if len(target_types) == 1 else None
        
        # 检查歧义
        source_ambiguous = len(source_types) > 1 and source_type is None
        target_ambiguous = len(target_types) > 1 and target_type is None

        if source_type and target_type:
            # 使用精确的类型标签进行匹配，并添加属性
            # 规范化关系类型
            normalized_rel_type = normalize_relation_type(relation.type, config)
            cypher_statements.append(
                f"MATCH (a:`{source_type}` {{name: '{escaped_source}'}}), "
                f"(b:`{target_type}` {{name: '{escaped_target}'}}) "
                f"MERGE (a)-[r:`{normalized_rel_type}`]->(b) "
                f"ON CREATE SET r.created_at = timestamp(), "
                f"r.context_id = '{relation.context_id}', "
                f"r.relation_type = '{normalized_rel_type}';")
        else:
            # 如果存在歧义，添加警告
            if source_ambiguous or target_ambiguous:
                relation_note = f"Ambiguous entities in relationship: ({relation.source})"
                if source_ambiguous:
                    relation_note += f"[多种类型: {', '.join(source_types)}]"
                relation_note += f"-[{relation.type}]->({relation.target})"
                if target_ambiguous:
                    relation_note += f"[多种类型: {', '.join(target_types)}]"
                ambiguous_relations_count += 1
                logging.warning(relation_note)
            else:
                untyped_relations_count += 1
                
            # 规范化关系类型
            normalized_rel_type = normalize_relation_type(relation.type, config)
                
            # 改进的匹配逻辑，添加属性
            if source_type and not target_type:
                cypher_statements.append(
                    f"MATCH (a:`{source_type}` {{name: '{escaped_source}'}}), "
                    f"(b {{name: '{escaped_target}'}}) "
                    f"MERGE (a)-[r:`{normalized_rel_type}`]->(b) "
                    f"ON CREATE SET r.match_type = 'source_typed', r.created_at = timestamp(), "
                    f"r.context_id = '{relation.context_id}', "
                    f"r.relation_type = '{normalized_rel_type}';")
            elif not source_type and target_type:
                cypher_statements.append(
                    f"MATCH (a {{name: '{escaped_source}'}}), "
                    f"(b:`{target_type}` {{name: '{escaped_target}'}}) "
                    f"MERGE (a)-[r:`{normalized_rel_type}`]->(b) "
                    f"ON CREATE SET r.match_type = 'target_typed', r.created_at = timestamp(), "
                    f"r.context_id = '{relation.context_id}', "
                    f"r.relation_type = '{normalized_rel_type}';")
            else:
                # 如果两边都没有明确类型，使用模糊匹配但添加警告标记
                cypher_statements.append(
                    f"MATCH (a {{name: '{escaped_source}'}}), (b {{name: '{escaped_target}'}}) "
                    f"MERGE (a)-[r:`{normalized_rel_type}`]->(b) "
                    f"ON CREATE SET r.match_type = 'untyped', r.reliability = 'low', "
                    f"r.created_at = timestamp(), r.warning = '实体类型未知，可能存在错误匹配', "
                    f"r.context_id = '{relation.context_id}', "
                    f"r.relation_type = '{normalized_rel_type}';")

    if untyped_relations_count > 0:
        logging.warning(f"发现 {untyped_relations_count} 个关系的实体类型无法确定，已使用通用匹配并标记为低可靠性。")
    if ambiguous_relations_count > 0:
        logging.warning(f"发现 {ambiguous_relations_count} 个关系中存在实体类型歧义，请检查日志获取详情。")
        
    return cypher_statements 