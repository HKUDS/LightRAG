"""
提示词构建模块

负责构建用于实体和关系提取的提示词
"""

from typing import Dict, Any, List, Optional
from graph_tools.models import Config

def create_entity_prompt(chunk_content: str, context: Dict[str, Any], config: Config, category: Optional[str] = None) -> str:
    """
    创建实体提取的提示词
    
    Args:
        chunk_content: 要处理的文本内容
        context: 上下文信息，包含文档标题、当前章节等
        config: 配置对象
        category: 文档分类，用于选择特定的schema和prompt扩展
        
    Returns:
        str: 用于实体提取的完整提示词
    """
    # 从配置中获取基础模板和定义
    base_template = config.prompt_templates.get('entity_extraction', {}).get('template', '')
    base_definitions = config.prompt_templates.get('entity_extraction', {}).get('definitions', '')
    
    # 初始化实体类型列表（基础类型）
    entity_types = list(config.entity_types_llm)
    
    # 如果指定了分类，获取分类特定的模板、定义和schema扩展
    category_template = ''
    category_definitions = ''
    if category and category in config.category_configs:
        # 获取分类特定的schema扩展
        category_config = config.category_configs.get(category, {})
        schema_extension = category_config.get('schema_extension', {})
        
        # 扩展实体类型列表
        category_entity_types = schema_extension.get('entity_types', [])
        entity_types.extend(category_entity_types)
        
        # 获取分类特定的prompt扩展
        category_prompts = category_config.get('prompts', {})
        category_entity_extraction = category_prompts.get('entity_extraction', {})
        
        # 获取分类特定的模板和定义
        category_template = category_entity_extraction.get('template', '')
        category_definitions = category_entity_extraction.get('definitions', '')
    
    # 使用分类特定的模板，如果有的话；否则使用基础模板
    template = category_template if category_template else base_template
    
    # 合并基础定义和分类特定的定义
    definitions = base_definitions
    if category_definitions:
        definitions = f"{base_definitions}\n\n{category_definitions}"
    
    # 获取实体类型的英文名称
    entity_types_english = [config.entity_type_map_cypher.get(t, t) for t in entity_types]
    
    # 获取文档上下文信息
    document_title = context.get('document_title', "未知文档")
    current_heading = context.get('current_heading', "未知章节")
    section_path = context.get('section_path', "")
    
    # 使用配置中的模板
    prompt = template.format(
        definitions=definitions,
        entity_types=', '.join(entity_types),
        document_title=document_title,
        current_heading=current_heading,
        section_path=section_path,
        content=chunk_content
    )
    
    return prompt

def create_relation_prompt(chunk_content: str, entities_json: str, context: Dict[str, Any], config: Config, category: Optional[str] = None) -> str:
    """
    创建关系提取的提示词
    
    Args:
        chunk_content: 要处理的文本内容
        entities_json: 已提取实体的JSON字符串
        context: 上下文信息，包含文档标题、当前章节等
        config: 配置对象
        category: 文档分类，用于选择特定的schema和prompt扩展
        
    Returns:
        str: 用于关系提取的完整提示词
    """
    # 从配置中获取基础模板和定义
    base_template = config.prompt_templates.get('relation_extraction', {}).get('template', '')
    base_definitions = config.prompt_templates.get('relation_extraction', {}).get('definitions', '')
    
    # 初始化关系类型列表（基础类型）
    relation_types = list(config.relation_types_llm)
    
    # 如果指定了分类，获取分类特定的模板、定义和schema扩展
    category_template = ''
    category_definitions = ''
    if category and category in config.category_configs:
        # 获取分类特定的schema扩展
        category_config = config.category_configs.get(category, {})
        schema_extension = category_config.get('schema_extension', {})
        
        # 扩展关系类型列表
        category_relation_types = schema_extension.get('relation_types', [])
        relation_types.extend(category_relation_types)
        
        # 获取分类特定的prompt扩展
        category_prompts = category_config.get('prompts', {})
        category_relation_extraction = category_prompts.get('relation_extraction', {})
        
        # 获取分类特定的模板和定义
        category_template = category_relation_extraction.get('template', '')
        category_definitions = category_relation_extraction.get('definitions', '')
    
    # 使用分类特定的模板，如果有的话；否则使用基础模板
    template = category_template if category_template else base_template
    
    # 合并基础定义和分类特定的定义
    definitions = base_definitions
    if category_definitions:
        definitions = f"{base_definitions}\n\n{category_definitions}"
    
    # 获取关系类型的英文名称
    relation_types_english = [config.relation_type_map_cypher.get(t, t) for t in relation_types]
    
    # 获取所有实体类型（包括Document和Section，以及分类特定的类型）
    all_entity_types = list(config.all_entity_types)
    
    # 如果指定了分类，获取分类特定的实体类型
    if category and category in config.category_configs:
        category_config = config.category_configs.get(category, {})
        schema_extension = category_config.get('schema_extension', {})
        category_entity_types = schema_extension.get('entity_types', [])
        # 添加分类特定的实体类型到所有实体类型列表
        for entity_type in category_entity_types:
            if entity_type not in all_entity_types:
                all_entity_types.append(entity_type)
    
    all_entity_types_english = [config.entity_type_map_cypher.get(t, t) for t in all_entity_types]
    
    # 获取文档上下文信息
    document_title = context.get('document_title', "未知文档")
    current_heading = context.get('current_heading', "未知章节")
    
    # 使用配置中的模板
    prompt = template.format(
        definitions=definitions,
        relation_types=', '.join(relation_types),
        all_entity_types_english=', '.join(all_entity_types_english),
        document_title=document_title,
        current_heading=current_heading,
        content=chunk_content,
        entities_json=entities_json
    )
    
    return prompt 