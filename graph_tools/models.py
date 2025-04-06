"""
定义知识图谱处理的数据模型

包含 Entity, Relation, Config 和 LLMTask 数据类的定义
"""

import logging
import yaml
from typing import List, Dict, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class Entity:
    """表示知识图谱中的一个实体"""
    name: str  # 实体名称
    type: str  # 实体类型
    context_id: str  # 上下文ID（chunk_id 或 doc_id）
    main_category: str = ""  # 实体的主类型，例如：客运管理、货运管理、财务管理等
    description: str = ""  # 实体简述，将会被向量化用于语义检索
    source: str = ""  # 实体的出处/来源
    created_at: datetime = field(default_factory=lambda: datetime.now())  # 首次生成时间
    updated_at: datetime = field(default_factory=lambda: datetime.now())  # 最后修改时间
    issuing_authority: str = ""  # 颁发机构
    vector: Optional[List[float]] = None  # 实体嵌入向量
    
    def __hash__(self):
        return hash((self.name, self.type, self.context_id))
    
    def __eq__(self, other):
        if not isinstance(other, Entity):
            return False
        return (self.name == other.name and 
                self.type == other.type and 
                self.context_id == other.context_id)

@dataclass
class Relation:
    """表示知识图谱中的一个关系"""
    source: str  # 源实体名称
    target: str  # 目标实体名称
    type: str  # 关系类型
    context_id: str  # 上下文ID（通常是chunk_id）
    
    def __hash__(self):
        return hash((self.source, self.target, self.type, self.context_id))
    
    def __eq__(self, other):
        if not isinstance(other, Relation):
            return False
        return (self.source == other.source and 
                self.target == other.target and 
                self.type == other.type and 
                self.context_id == other.context_id)

@dataclass
class Config:
    """配置类，用于存储从YAML文件加载的配置"""
    # 实体类型相关配置
    entity_types_llm: List[str]  # LLM提取的实体类型
    all_entity_types: List[str]  # 所有概念实体类型
    entity_type_map_cypher: Dict[str, str]  # 实体类型映射到Cypher
    
    # 关系类型相关配置
    relation_types_llm: List[str]  # LLM提取的关系类型
    all_relation_types: List[str]  # 所有概念关系类型
    relation_type_map_cypher: Dict[str, str]  # 关系类型映射到Cypher
    
    # 规范化映射
    canonical_map: Dict[str, str]  # 实体名称规范化映射
    
    # 提示词模板
    prompt_templates: Dict[str, Any]
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'Config':
        """从YAML文件加载配置"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            # 基本验证
            if 'schema' not in data:
                raise ValueError("配置文件缺少必要的键: 'schema'")
            if 'normalization' not in data:
                raise ValueError("配置文件缺少必要的键: 'normalization'")
            if 'prompts' not in data:
                raise ValueError("配置文件缺少必要的键: 'prompts'")
            
            schema = data.get('schema', {})
            # 使用配置数据创建Config实例
            config = cls(
                entity_types_llm=schema.get('entity_types_llm', []),
                all_entity_types=schema.get('all_entity_types', []),
                entity_type_map_cypher=schema.get('entity_type_map_cypher', {}),
                relation_types_llm=schema.get('relation_types_llm', []),
                all_relation_types=schema.get('all_relation_types', []),
                relation_type_map_cypher=schema.get('relation_type_map_cypher', {}),
                canonical_map=((data.get('normalization') or {}).get('canonical_map')) or {},
                prompt_templates=data.get('prompts', {})
            )
            
            # 验证必要的配置项
            if not config.entity_types_llm or not config.all_entity_types:
                raise ValueError("配置文件缺少必要的实体类型定义")
            if not config.relation_types_llm or not config.all_relation_types:
                raise ValueError("配置文件缺少必要的关系类型定义")
            if not config.prompt_templates.get('entity_extraction') or not config.prompt_templates.get('relation_extraction'):
                raise ValueError("配置文件缺少必要的提示词模板")
            
            # 输出加载的配置摘要
            logging.info(f"加载了 {len(config.entity_types_llm)} 个LLM抽取的实体类型和 {len(config.relation_types_llm)} 个LLM抽取的关系类型")
            logging.info(f"概念模型中共有 {len(config.all_entity_types)} 个实体类型和 {len(config.all_relation_types)} 个关系类型")
            logging.info(f"加载了 {len(config.canonical_map)} 个规范化映射条目")
            
            return config
            
        except FileNotFoundError:
            logging.error(f"配置文件未找到: {config_path}")
            raise
        except yaml.YAMLError as e:
            logging.error(f"解析配置文件时出错: {e}")
            raise
        except Exception as e:
            logging.error(f"加载配置时发生未预期的错误: {e}")
            logging.exception("详细错误信息:")
            raise

@dataclass
class LLMTask:
    """封装LLM请求的任务类"""
    chunk_id: str        # 当前处理的chunk ID
    prompt_type: str     # 提示类型：'entity' 或 'relation'
    content: str         # 要处理的文本内容
    prompt: str          # 完整的提示信息
    result: Optional[Dict] = None  # 处理结果 