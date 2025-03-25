from typing import Dict, Any, Optional, List, Union
from copy import deepcopy
import json

from lightrag.prompt import PROMPTS


class PromptFactory:
    """
    工厂类，用于根据参数灵活配置prompt，适应不同领域的需求。
    """
    
    def __init__(self, default_prompts: Optional[Dict[str, Any]] = None):
        """
        初始化PromptFactory
        
        Args:
            default_prompts: 可选的默认提示字典，如果不提供则使用内置PROMPTS
        """
        self.prompts = deepcopy(default_prompts or PROMPTS)
        self.domain_configs = {}
    
    def register_domain(self, domain_name: str, config: Dict[str, Any]) -> None:
        """
        注册新的领域配置
        
        Args:
            domain_name: 领域名称
            config: 该领域的配置参数
        """
        self.domain_configs[domain_name] = config
    
    def get_domain_config(self, domain_name: str) -> Dict[str, Any]:
        """
        获取特定领域的配置
        
        Args:
            domain_name: 领域名称
            
        Returns:
            领域配置字典
        """
        if domain_name not in self.domain_configs:
            raise ValueError(f"Domain '{domain_name}' not registered")
        return self.domain_configs[domain_name]
    
    def get_prompt(self, prompt_key: str, domain: Optional[str] = None, **kwargs) -> str:
        """
        获取指定key的提示，可以使用特定领域的配置和其他参数进行格式化
        
        Args:
            prompt_key: 提示模板的键名
            domain: 可选的领域名称，用于应用领域特定配置
            **kwargs: 额外的格式化参数
            
        Returns:
            格式化后的提示字符串
        """
        if prompt_key not in self.prompts:
            raise ValueError(f"Prompt key '{prompt_key}' not found")
        
        prompt_template = self.prompts[prompt_key]
        
        # 合并领域配置和额外参数
        format_args = {}
        if domain and domain in self.domain_configs:
            format_args.update(self.domain_configs[domain])
        format_args.update(kwargs)
        
        # 根据提示类型处理格式化
        if isinstance(prompt_template, str):
            return prompt_template.format(**format_args)
        elif isinstance(prompt_template, list):
            # 如果是列表类型（如examples列表），处理每个元素
            return [item.format(**format_args) if isinstance(item, str) else item for item in prompt_template]
        else:
            return prompt_template
    
    def add_prompt(self, key: str, prompt: Union[str, List, Dict]) -> None:
        """
        添加新的提示模板
        
        Args:
            key: 提示模板的键名
            prompt: 提示模板内容
        """
        self.prompts[key] = prompt
    
    def update_prompt(self, key: str, prompt: Union[str, List, Dict]) -> None:
        """
        更新现有的提示模板
        
        Args:
            key: 要更新的提示模板键名
            prompt: 新的提示模板内容
        """
        if key not in self.prompts:
            raise ValueError(f"Prompt key '{key}' not found")
        self.prompts[key] = prompt
    
    def customize_entity_types(self, entity_types: List[str], domain: Optional[str] = None) -> None:
        """
        为特定领域自定义实体类型
        
        Args:
            entity_types: 实体类型列表
            domain: 可选的领域名称
        """
        if domain:
            if domain not in self.domain_configs:
                self.domain_configs[domain] = {}
            self.domain_configs[domain]["entity_types"] = ", ".join(entity_types)
        else:
            self.prompts["DEFAULT_ENTITY_TYPES"] = entity_types
    
    def set_language(self, language: str, domain: Optional[str] = None) -> None:
        """
        设置提示的语言
        
        Args:
            language: 语言名称
            domain: 可选的领域名称
        """
        if domain:
            if domain not in self.domain_configs:
                self.domain_configs[domain] = {}
            self.domain_configs[domain]["language"] = language
        else:
            self.prompts["DEFAULT_LANGUAGE"] = language
    
    def export_domain_config(self, domain: str) -> Dict[str, Any]:
        """
        导出特定领域的配置为字典
        
        Args:
            domain: 领域名称
            
        Returns:
            领域配置字典
        """
        if domain not in self.domain_configs:
            raise ValueError(f"Domain '{domain}' not found")
        return deepcopy(self.domain_configs[domain])
    
    def export_domain_config_json(self, domain: str) -> str:
        """
        导出特定领域的配置为JSON字符串
        
        Args:
            domain: 领域名称
            
        Returns:
            JSON格式的领域配置
        """
        config = self.export_domain_config(domain)
        return json.dumps(config, ensure_ascii=False, indent=2)
    
    def import_domain_config(self, domain: str, config: Dict[str, Any]) -> None:
        """
        从字典导入领域配置
        
        Args:
            domain: 领域名称
            config: 配置字典
        """
        self.domain_configs[domain] = deepcopy(config)
    
    def import_domain_config_json(self, domain: str, json_str: str) -> None:
        """
        从JSON字符串导入领域配置
        
        Args:
            domain: 领域名称
            json_str: JSON格式的配置字符串
        """
        config = json.loads(json_str)
        self.import_domain_config(domain, config)
    
    def register_relationship_types(self, domain_name: str, relationships: List[Dict[str, str]]) -> None:
        """
        为特定领域注册关系类型
        
        Args:
            domain_name: 领域名称
            relationships: 关系类型列表，每个类型包含name、code和description
        """
        if domain_name not in self.domain_configs:
            self.domain_configs[domain_name] = {}
        
        # 提取关系类型名称并设置到配置中
        relationship_names = [r["name"] for r in relationships]
        self.domain_configs[domain_name]["relationship_types"] = ", ".join(relationship_names) 