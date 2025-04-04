"""
数据规范化模块

负责数据清洗、规范化和字符串处理
"""

import re
import logging
from graph_tools.models import Config

def normalize_entity_name(raw_name: str, config: Config) -> str:
    """
    规范化实体名称：
    1. 使用配置的canonical_map规范化实体名称
    2. 去除名称开头的章节序号（如"4.4 "）
    3. 去除名称中的页码标签（如"(p.12)"）
    
    Args:
        raw_name: 原始实体名称
        config: 配置对象
        
    Returns:
        str: 规范化后的实体名称
    """
    if not isinstance(raw_name, str):
        logging.warning(f"Attempted to normalize non-string value: {raw_name}. Returning as is.")
        return str(raw_name)
    
    # 基础清理
    cleaned_name = raw_name.strip().replace('\n', ' ')
    
    # 去除开头的章节序号（如"4.4 "）
    cleaned_name = re.sub(r'^[\d\.]+\s+', '', cleaned_name)
    
    # 去除页码标签（如"(p.12)"）
    cleaned_name = re.sub(r'\s*\(p\.\d+\)\s*', '', cleaned_name)
    
    return config.canonical_map.get(cleaned_name, cleaned_name)

def normalize_entity_type(raw_type: str, config: Config) -> str:
    """
    规范化实体类型，确保使用配置中定义的标准类型。
    
    将尝试匹配中英文类型名称，确保使用entity_type_map_cypher中定义的标准类型。
    
    Args:
        raw_type: 原始实体类型
        config: 配置对象
        
    Returns:
        str: 规范化后的实体类型
    """
    if not isinstance(raw_type, str):
        logging.warning(f"Attempted to normalize non-string entity type: {raw_type}. Returning as is.")
        return str(raw_type)
    
    cleaned_type = raw_type.strip().replace('\n', ' ')
    
    # 1. 先检查是否已是有效的实体类型
    if cleaned_type in config.entity_type_map_cypher.values() or cleaned_type in config.entity_types_llm:
        # 如果是中文类型且存在于entity_type_map_cypher中，返回对应的英文类型
        if cleaned_type in config.entity_types_llm:
            return config.entity_type_map_cypher.get(cleaned_type, cleaned_type)
        return cleaned_type
    
    # 2. 检查是否误将关系类型作为实体类型传入（只在类型不在有效实体类型列表中时检查）
    if cleaned_type in config.relation_type_map_cypher.values() or cleaned_type in config.relation_types_llm:
        # 对于误传入的关系类型，根据关系类型返回可能的实体类型，或者返回一个默认值
        # 但不记录普通日志，因为这种情况可能很常见，只在调试级别记录
        logging.debug(f"识别到关系类型 '{cleaned_type}' 被误用作实体类型，尝试返回适当的实体类型")
        if cleaned_type in ["BELONGS_TO", "RESPONSIBLE_FOR"]:
            return "Organization"  # 这些关系通常与组织实体相关
        elif cleaned_type == "APPLIES_TO":
            return "Statement"  # APPLIES_TO通常源自Statement
        elif cleaned_type == "MENTIONS":
            return "Section"  # MENTIONS通常源自Section
        elif cleaned_type == "REFERENCES":
            return "Statement"  # REFERENCES通常源自Statement
        elif cleaned_type == "HAS_PURPOSE":
            return "Statement"  # HAS_PURPOSE通常与Statement相关
        return "Topic"  # 返回一个默认实体类型
    
    # 3. 尝试通过反向查找，从英文映射回中文再映射回标准英文
    for cn_type, en_type in config.entity_type_map_cypher.items():
        if cleaned_type.lower() == en_type.lower():
            return en_type  # 返回正确大小写的英文类型名
    
    # 4. 如果无法匹配，记录警告并返回原始类型
    logging.warning(f"实体类型 '{cleaned_type}' 未在配置中定义，无法规范化")
    return cleaned_type

def normalize_relation_type(raw_type: str, config: Config) -> str:
    """
    规范化关系类型，确保使用配置中定义的标准类型。
    
    将尝试匹配中英文类型名称，确保使用relation_type_map_cypher中定义的标准类型。
    
    Args:
        raw_type: 原始关系类型
        config: 配置对象
        
    Returns:
        str: 规范化后的关系类型
    """
    if not isinstance(raw_type, str):
        logging.warning(f"Attempted to normalize non-string relation type: {raw_type}. Returning as is.")
        return str(raw_type)
    
    cleaned_type = raw_type.strip().replace('\n', ' ')
    
    # 1. 先检查是否已是有效的关系类型
    if cleaned_type in config.relation_type_map_cypher.values() or cleaned_type in config.relation_types_llm:
        # 如果是中文类型且存在于relation_type_map_cypher中，返回对应的英文类型
        if cleaned_type in config.relation_types_llm:
            return config.relation_type_map_cypher.get(cleaned_type, cleaned_type)
        return cleaned_type
    
    # 2. 检查是否误将实体类型作为关系类型传入（只在类型不在有效关系类型列表中时检查）
    if cleaned_type in config.entity_type_map_cypher.values() or cleaned_type in config.entity_types_llm:
        # 对于误传入的实体类型，根据实体类型返回可能的关系类型，或者返回一个默认值
        # 但不记录普通日志，因为这种情况可能很常见，只在调试级别记录
        logging.debug(f"识别到实体类型 '{cleaned_type}' 被误用作关系类型，尝试返回适当的关系类型")
        if cleaned_type in ["Organization", "Role"]:
            return "BELONGS_TO"  # 这些实体类型通常与隶属关系相关
        elif cleaned_type == "Statement":
            return "HAS_PURPOSE"  # Statement通常与HAS_PURPOSE相关
        elif cleaned_type == "Topic":
            return "MENTIONS"  # Topic通常与MENTIONS相关
        elif cleaned_type == "Section":
            return "CONTAINS"  # Section通常与CONTAINS相关
        elif cleaned_type == "Document":
            return "HAS_SECTION"  # Document通常与HAS_SECTION相关
        return "RELATED_TO"  # 返回一个通用关系类型
    
    # 3. 尝试通过反向查找，从英文映射回中文再映射回标准英文
    for cn_type, en_type in config.relation_type_map_cypher.items():
        if cleaned_type.lower() == en_type.lower():
            return en_type  # 返回正确大小写的英文类型名
    
    # 4. 如果无法匹配，记录警告并返回原始类型
    logging.warning(f"关系类型 '{cleaned_type}' 未在配置中定义，无法规范化")
    return cleaned_type

def clean_heading(heading: str, config: Config) -> str:
    """
    清理章节标题，移除页码和其他噪声。
    
    Args:
        heading: 原始章节标题
        config: 配置对象
        
    Returns:
        str: 清理后的章节标题
    """
    if not heading:
        return "未知章节"
    # 移除 (p.XX)
    cleaned = re.sub(r'\s*\(p\.\d+\)\s*$', '', heading).strip()
    # 应用规范化映射
    return config.canonical_map.get(cleaned, cleaned)

def escape_cypher_string(value: str) -> str:
    """
    转义Cypher查询中的字符串值
    
    Args:
        value: 原始字符串
        
    Returns:
        str: 转义后的字符串
    """
    return value.replace("\\", "\\\\").replace("'", "\\'") 