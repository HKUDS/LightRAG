"""
数据处理模块

包含核心数据处理逻辑，如文档结构处理、语义信息提取、关系创建等
"""

import json
import logging
import asyncio
from typing import List, Dict, Set, Tuple, Optional, Any
from datetime import datetime
from pathlib import Path

from graph_tools.models import Entity, Relation, Config, LLMTask
from graph_tools.normalization import normalize_entity_name, normalize_entity_type, normalize_relation_type, clean_heading
from graph_tools.prompt_builder import create_entity_prompt, create_relation_prompt
from graph_tools.llm_client import LLMClient

async def process_document_structure(data: Dict[str, Any], config: Config) -> Tuple[
    List[Entity],  # Document和Section实体
    List[Relation]  # 结构关系
]:
    """
    解析文档结构，创建Document和Section实体及其关系
    
    Args:
        data: 数据字典，包含chunks和document_info
        config: 配置对象
        
    Returns:
        Tuple: (实体列表, 关系列表)
    """
    document_entities = []  # Document实体列表
    section_entities = []  # Section实体列表
    structure_relations = []  # 结构关系列表
    
    # 存储ID到实体的映射，方便后续创建关系
    id_to_entity_map = {}
    
    # 获取chunks和document_info
    chunks = data.get("chunks", [])
    document_info = data.get("document_info", {})
    
    # --- 文档标题启发式处理 ---
    # 尝试找到主文档标题
    doc_titles = {}
    
    if chunks:
        first_chunk = chunks[0]
        doc_id = first_chunk.get("full_doc_id", "unknown_doc_0")
        title = "未知文档"
        
        # 尝试从文件路径提取
        if "file_path" in first_chunk:
            # 从文件名中提取，可能需要进一步优化
            title = Path(first_chunk["file_path"]).stem.split('-')[-1].replace('_', ' ')
            # 应用规范化
            title = normalize_entity_name(title, config)
        
        # 回退方案：如果标题看起来太过通用，尝试使用第一个chunk内容的第一行
        if title == "未知文档" and "content" in first_chunk:
            first_line = first_chunk["content"].split('\n')[0].strip()
            if first_line and len(first_line) > 5: # 避免使用过短或通用的行
                title = normalize_entity_name(first_line, config)

        doc_titles[doc_id] = title
    
    # --- 处理Chunks以获取章节和关系 ---
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        doc_id = chunk.get("full_doc_id")
        heading = chunk.get("heading")
        parent_id = chunk.get("parent_id") # 可能是父级的chunk_id

        if not chunk_id or not doc_id or not heading:
            logging.warning(f"由于缺少关键信息，跳过chunk: {chunk.get('chunk_id', 'N/A')}")
            continue

        # 创建/更新文档实体
        doc_title = doc_titles.get(doc_id, f"未知文档 {doc_id}")
        
        # 创建文档实体
        if doc_id not in id_to_entity_map:
            # 从document_info获取对应属性
            main_category = document_info.get("main_category", "文档管理")
            source = document_info.get("document_name", "系统自动生成")
            description = document_info.get("document_summary", f"文档：{doc_title}")
            created_at_str = document_info.get("created_at", "")
            updated_at_str = document_info.get("updated_at", "")
            
              
            # 处理日期字符串转换为datetime对象
            created_at = datetime.now()
            if created_at_str:
                try:
                    created_at = datetime.strptime(created_at_str, "%Y-%m-%d")
                except (ValueError, TypeError):
                    pass
                    
            updated_at = datetime.now()
            if updated_at_str:
                try:
                    updated_at = datetime.strptime(updated_at_str, "%Y-%m-%d")
                except (ValueError, TypeError):
                    pass
            
            document_entity = Entity(
                name=doc_title,
                type="Document",
                context_id=doc_id,
                main_category=main_category,
                description=description,
                source=source,
                created_at=created_at,
                updated_at=updated_at,
                issuing_authority=issuing_authority
            )
            document_entities.append(document_entity)
            id_to_entity_map[doc_id] = document_entity
            logging.debug(f"识别到文档: ID={doc_id}, 标题={doc_title}")

        # 创建章节实体
        section_name = clean_heading(heading, config)
        
        # 从chunk的summary获取描述
        section_description = chunk.get("summary", f"章节：{section_name}")
        if not section_description or section_description == "无子文档内容，无法生成摘要":
            section_description = f"章节：{section_name}"
            
        # 获取document_entity以便复制其元数据
        document_entity = id_to_entity_map.get(doc_id)
        
        section_entity = Entity(
            name=section_name,
            type="Section",
            context_id=chunk_id,
            main_category=document_entity.main_category if document_entity else document_info.get("main_category", ""),
            description=section_description,
            source=document_entity.source if document_entity else document_info.get("document_name", ""),
            created_at=document_entity.created_at if document_entity else (created_at if 'created_at' in locals() else datetime.now()),
            updated_at=document_entity.updated_at if document_entity else (updated_at if 'updated_at' in locals() else datetime.now()),
            issuing_authority=document_entity.issuing_authority if document_entity else issuing_authority
        )
        section_entities.append(section_entity)
        id_to_entity_map[chunk_id] = section_entity
        logging.debug(f"识别到章节: ID={chunk_id}, 名称={section_name}, 文档ID={doc_id}")

        # 创建 HAS_SECTION 关系(Document -> Section)
        if doc_id in id_to_entity_map:
            has_section_relation = Relation(
                source=doc_title,
                target=section_name,
                type="HAS_SECTION",
                context_id=chunk_id
            )
            structure_relations.append(has_section_relation)

        # 创建 HAS_PARENT_SECTION 关系(Section -> Section)
        if parent_id and parent_id != chunk_id and parent_id in id_to_entity_map:
            parent_entity = id_to_entity_map[parent_id]
            if parent_entity.type == "Section":
                has_parent_relation = Relation(
                    source=section_name,
                    target=parent_entity.name,
                    type="HAS_PARENT_SECTION",
                    context_id=chunk_id
                )
                structure_relations.append(has_parent_relation)
                logging.debug(f"识别到父子关系: 子={chunk_id}, 父={parent_id}")
    
    return document_entities + section_entities, structure_relations

async def extract_semantic_info(data: Dict[str, Any], section_entities: List[Entity], config: Config, llm_client: LLMClient) -> Tuple[
    List[Entity],  # 语义实体
    List[Relation]  # 语义关系
]:
    """
    使用LLM提取语义实体和关系
    
    Args:
        data: 数据字典，包含chunks和document_info
        section_entities: 章节实体列表
        config: 配置对象
        llm_client: LLM客户端
        
    Returns:
        Tuple: (实体列表, 关系列表)
    """
    semantic_entities = []  # 语义实体列表
    semantic_relations = []  # 语义关系列表
    processed_chunk_entities = {}  # 存储每个chunk的实体，用于关系提示
    
    # 创建section_id到section实体的映射
    section_map = {entity.context_id: entity for entity in section_entities if entity.type == "Section"}
    
    # 获取document_info和chunks
    document_info = data.get("document_info", {})
    chunks = data.get("chunks", [])
    
    # 获取document_info中的属性，用于填充实体
    main_category = document_info.get("main_category", "") 
    source = document_info.get("document_name", "")
    created_at_str = document_info.get("created_at", "")
    updated_at_str = document_info.get("updated_at", "")
    
    # 处理issuing_authority拼写不一致问题
    issuing_authority = ""
    if "issuing_authority" in document_info:
        issuing_authority = document_info.get("issuing_authority", "")
    elif "issuin_authority" in document_info:
        # 处理拼写错误的情况
        issuing_authority = document_info.get("issuin_authority", "")
        logging.warning("extract_semantic_info: 检测到document_info中的'issuin_authority'拼写错误，已自动处理")
    elif "IssuingAuthority" in document_info:
        # 处理大写开头的情况
        issuing_authority = document_info.get("IssuingAuthority", "")
        logging.warning("extract_semantic_info: 检测到document_info中的'IssuingAuthority'大小写不一致，已自动处理")
    
    # 处理日期字符串转换为datetime对象
    created_at = datetime.now()
    if created_at_str:
        try:
            created_at = datetime.strptime(created_at_str, "%Y-%m-%d")
        except (ValueError, TypeError):
            pass
            
    updated_at = datetime.now()
    if updated_at_str:
        try:
            updated_at = datetime.strptime(updated_at_str, "%Y-%m-%d")
        except (ValueError, TypeError):
            pass
    
    # --- 第1阶段: 实体提取 ---
    entity_tasks = []
    for i, chunk in enumerate(chunks):
        # 确保chunk是字典类型
        if not isinstance(chunk, dict):
            logging.warning(f"数据项 {i} 不是字典类型，跳过处理: {type(chunk)}")
            continue
            
        chunk_id = chunk.get("chunk_id")
        content = chunk.get("content")
        doc_id = chunk.get("full_doc_id")

        if not chunk_id or not content or chunk_id not in section_map:
            logging.warning(f"由于缺少数据或在结构中未找到，跳过chunk {chunk_id or i}的实体提取")
            continue

        # 准备提示的上下文
        section_entity = section_map[chunk_id]
        context = {
            "document_title": doc_id,  # 简化处理，使用doc_id作为文档标题
            "current_heading": section_entity.name,
            "section_path": "N/A",  # 简化，不构建章节路径
            "parent_section_summary": ""
        }

        entity_prompt = create_entity_prompt(content, context, config)
        entity_tasks.append(LLMTask(
            chunk_id=chunk_id,
            prompt_type="entity",
            content=content,
            prompt=entity_prompt
        ))

    logging.info(f"开始为 {len(entity_tasks)} 个chunks进行LLM实体提取...")
    processed_entity_tasks = await llm_client.process_tasks(entity_tasks)
    logging.info("完成LLM实体提取。")

    # 处理实体结果
    for task in processed_entity_tasks:
        if task.result and 'entities' in task.result:
            chunk_entities = []
            for entity_dict in task.result['entities']:
                raw_name = entity_dict.get('name')
                raw_type = entity_dict.get('type')
                
                # 验证类型
                valid_type = False
                for cn_type in config.entity_types_llm:
                    en_type = config.entity_type_map_cypher.get(cn_type, cn_type)
                    if raw_type == en_type or raw_type == cn_type:
                        valid_type = True
                        break
                
                if raw_name and raw_type and valid_type:
                    normalized_name = normalize_entity_name(raw_name, config)
                    normalized_type = normalize_entity_type(raw_type, config)
                    
                    # 获取此chunk对应的section实体，从中继承文档信息
                    section_entity = section_map.get(task.chunk_id)
                    
                    # 如果找到对应的section实体，从中获取文档信息
                    if section_entity:
                        entity = Entity(
                            name=normalized_name,
                            type=normalized_type,
                            context_id=task.chunk_id,
                            main_category=section_entity.main_category,
                            description=f"{normalized_type}：{normalized_name}",
                            source=section_entity.source,
                            created_at=section_entity.created_at,
                            updated_at=section_entity.updated_at,
                            issuing_authority=section_entity.issuing_authority
                        )
                    else:
                        # 如果找不到section实体，使用document_info
                        entity = Entity(
                            name=normalized_name,
                            type=normalized_type,
                            context_id=task.chunk_id,
                            main_category=main_category,
                            description=f"{normalized_type}：{normalized_name}",
                            source=source,
                            created_at=created_at,
                            updated_at=updated_at,
                            issuing_authority=issuing_authority
                        )
                    
                    semantic_entities.append(entity)
                    chunk_entities.append(entity_dict)
                else:
                    logging.warning(f"LLM在chunk {task.chunk_id}中返回了无效的实体: {entity_dict}")
            processed_chunk_entities[task.chunk_id] = chunk_entities
    
    # --- 第2阶段: 关系提取 ---
    relation_tasks = []
    for i, chunk in enumerate(chunks):
        # 确保chunk是字典类型
        if not isinstance(chunk, dict):
            logging.warning(f"关系提取：数据项 {i} 不是字典类型，跳过处理: {type(chunk)}")
            continue
            
        chunk_id = chunk.get("chunk_id")
        content = chunk.get("content")
        doc_id = chunk.get("full_doc_id")

        if not chunk_id or not content or chunk_id not in section_map:
            continue

        # 获取为此chunk提取的语义实体
        chunk_semantic_entities = processed_chunk_entities.get(chunk_id, [])
        if not chunk_semantic_entities:
            continue

        entities_json_str = json.dumps({"entities": chunk_semantic_entities}, ensure_ascii=False, indent=2)

        # 准备上下文
        section_entity = section_map[chunk_id]
        context = {
            "document_title": doc_id,
            "current_heading": section_entity.name,
            "section_path": "N/A"
        }

        relation_prompt = create_relation_prompt(content, entities_json_str, context, config)
        relation_tasks.append(LLMTask(
            chunk_id=chunk_id,
            prompt_type="relation",
            content=content,
            prompt=relation_prompt
        ))

    logging.info(f"开始为 {len(relation_tasks)} 个chunks进行LLM关系提取...")
    processed_relation_tasks = await llm_client.process_tasks(relation_tasks)
    logging.info("完成LLM关系提取。")

    # 处理关系结果
    for task in processed_relation_tasks:
        if task.result and 'relations' in task.result:
            for relation_dict in task.result['relations']:
                raw_source = relation_dict.get('source')
                raw_target = relation_dict.get('target')
                raw_type = relation_dict.get('type')
                
                # 验证关系类型
                valid_type = False
                for cn_type in config.relation_types_llm:
                    en_type = config.relation_type_map_cypher.get(cn_type, cn_type)
                    if raw_type == en_type or raw_type == cn_type:
                        valid_type = True
                        break
                
                if raw_source and raw_target and raw_type and valid_type:
                    normalized_source = normalize_entity_name(raw_source, config)
                    normalized_target = normalize_entity_name(raw_target, config)
                    normalized_type = normalize_relation_type(raw_type, config)
                    
                    relation = Relation(
                        source=normalized_source,
                        target=normalized_target,
                        type=normalized_type,
                        context_id=task.chunk_id
                    )
                    semantic_relations.append(relation)
                    
                    # 检查源和目标是否有对应的实体，如果没有，添加默认实体
                    source_exists = any(e.name == normalized_source for e in semantic_entities)
                    target_exists = any(e.name == normalized_target for e in semantic_entities)
                    
                    # 从关系中获取源和目标类型
                    source_type = relation_dict.get('source_type')
                    target_type = relation_dict.get('target_type')
                    
                    # 获取此chunk对应的section实体
                    section_entity = section_map.get(task.chunk_id)
                    
                    if source_type and not source_exists:
                        normalized_source_type = normalize_entity_type(source_type, config)
                        
                        # 如果找到对应的section实体，从中获取文档信息
                        if section_entity:
                            semantic_entities.append(Entity(
                                name=normalized_source,
                                type=normalized_source_type,
                                context_id=task.chunk_id,
                                main_category=section_entity.main_category,
                                description=f"{normalized_source_type}：{normalized_source}",
                                source=section_entity.source,
                                created_at=section_entity.created_at,
                                updated_at=section_entity.updated_at,
                                issuing_authority=section_entity.issuing_authority
                            ))
                        else:
                            # 如果找不到section实体，使用document_info
                            semantic_entities.append(Entity(
                                name=normalized_source,
                                type=normalized_source_type,
                                context_id=task.chunk_id,
                                main_category=main_category,
                                description=f"{normalized_source_type}：{normalized_source}",
                                source=source,
                                created_at=created_at,
                                updated_at=updated_at,
                                issuing_authority=issuing_authority
                            ))
                    
                    if target_type and not target_exists:
                        normalized_target_type = normalize_entity_type(target_type, config)
                        
                        # 如果找到对应的section实体，从中获取文档信息
                        if section_entity:
                            semantic_entities.append(Entity(
                                name=normalized_target,
                                type=normalized_target_type,
                                context_id=task.chunk_id,
                                main_category=section_entity.main_category,
                                description=f"{normalized_target_type}：{normalized_target}",
                                source=section_entity.source,
                                created_at=section_entity.created_at,
                                updated_at=section_entity.updated_at,
                                issuing_authority=section_entity.issuing_authority
                            ))
                        else:
                            # 如果找不到section实体，使用document_info
                            semantic_entities.append(Entity(
                                name=normalized_target,
                                type=normalized_target_type,
                                context_id=task.chunk_id,
                                main_category=main_category,
                                description=f"{normalized_target_type}：{normalized_target}",
                                source=source,
                                created_at=created_at,
                                updated_at=updated_at,
                                issuing_authority=issuing_authority
                            ))
                else:
                    logging.warning(f"LLM在chunk {task.chunk_id}中返回了无效的关系: {relation_dict}")
    
    return semantic_entities, semantic_relations

def create_contains_relations(section_entities: List[Entity], semantic_entities: List[Entity]) -> List[Relation]:
    """
    创建CONTAINS关系，将语义实体连接到它们所属的章节实体
    
    Args:
        section_entities: 章节实体列表
        semantic_entities: 语义实体列表
        
    Returns:
        List[Relation]: CONTAINS关系列表
    """
    contains_relations = []
    
    # 创建section_id到section实体的映射
    section_map = {entity.context_id: entity for entity in section_entities if entity.type == "Section"}
    
    # 查找文档实体，用作回退选项
    document_entities = [entity for entity in section_entities if entity.type == "Document"]
    default_document = document_entities[0] if document_entities else None
    
    # 为每个语义实体创建CONTAINS关系
    for entity in semantic_entities:
        # 跳过Document和Section类型的实体
        if entity.type in ["Document", "Section"]:
            continue
        
        # 检查实体的context_id是否属于某个section
        if entity.context_id in section_map:
            section_entity = section_map[entity.context_id]
            contains_relation = Relation(
                source=section_entity.name,
                target=entity.name,
                type="CONTAINS",
                context_id=entity.context_id
            )
            contains_relations.append(contains_relation)
        else:
            # 如果找不到对应的section，使用一个默认的文档实体
            # 寻找与实体有相同main_category、source或issuing_authority的section
            matching_section = None
            
            # 按优先级寻找匹配的section
            for section_entity in section_entities:
                if section_entity.type != "Section":
                    continue
                    
                # 优先级1: 完全匹配所有文档信息
                if (section_entity.main_category == entity.main_category and
                    section_entity.source == entity.source and
                    section_entity.issuing_authority == entity.issuing_authority):
                    matching_section = section_entity
                    break
                    
                # 优先级2: 匹配source和issuing_authority
                elif (section_entity.source == entity.source and
                      section_entity.issuing_authority == entity.issuing_authority):
                    matching_section = section_entity
                    continue
                    
                # 优先级3: 匹配main_category
                elif section_entity.main_category == entity.main_category:
                    if not matching_section:
                        matching_section = section_entity
            
            if matching_section:
                logging.debug(f"找到与实体'{entity.name}'匹配的section: {matching_section.name}")
                contains_relation = Relation(
                    source=matching_section.name,
                    target=entity.name,
                    type="CONTAINS",
                    context_id=entity.context_id
                )
                contains_relations.append(contains_relation)
            elif default_document:
                # 如果没有匹配的section，使用文档作为容器
                logging.debug(f"使用文档'{default_document.name}'作为实体'{entity.name}'的容器")
                contains_relation = Relation(
                    source=default_document.name,
                    target=entity.name,
                    type="CONTAINS",
                    context_id=entity.context_id
                )
                contains_relations.append(contains_relation)
            else:
                logging.warning(f"无法为实体'{entity.name}'找到合适的容器，此实体将成为孤立节点")
    
    # 统计生成的关系
    logging.info(f"总共为 {len(semantic_entities)} 个语义实体创建了 {len(contains_relations)} 个CONTAINS关系")
    if len(semantic_entities) > len(contains_relations):
        logging.warning(f"有 {len(semantic_entities) - len(contains_relations)} 个语义实体没有对应的CONTAINS关系")
    
    return contains_relations 