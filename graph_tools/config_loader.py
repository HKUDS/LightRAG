"""
配置加载和处理模块

负责加载配置文件并初始化应用程序配置
"""

import os
import logging
from typing import Dict, Any

# 从models模块导入Config类
from graph_tools.models import Config

# 默认配置
DEFAULT_CONFIG = {
    # LLM API配置
    "llm_api_key": os.environ.get("LLM_BINDING_API_KEY") or os.environ.get("SILICONFLOW_API_KEY"),
    "llm_api_host": os.environ.get("LLM_BINDING_HOST"),
    "llm_model": os.environ.get("LLM_MODEL"),
    
    # API并发设置
    "max_concurrent_requests": int(os.environ.get("MAX_CONCURRENT_REQUESTS", "5")),
    "request_delay": float(os.environ.get("REQUEST_DELAY", "0.2")),
    "request_timeout": float(os.environ.get("REQUEST_TIMEOUT", "60.0")),
    "max_retries": int(os.environ.get("MAX_RETRIES", "3")),
    
    # 日志设置
    "log_level": os.environ.get("LOG_LEVEL", "INFO").upper(),
}

def load_app_config() -> Dict[str, Any]:
    """
    加载应用程序配置
    
    从环境变量中读取配置，提供默认值
    
    Returns:
        Dict[str, Any]: 应用程序配置字典
    """
    # 复制默认配置
    config = DEFAULT_CONFIG.copy()
    
    # 验证关键配置
    if not config["llm_api_key"]:
        logging.warning("环境变量 LLM_BINDING_API_KEY 或 SILICONFLOW_API_KEY 未设置，API调用将会失败")
    
    if not config["llm_api_host"]:
        logging.warning("环境变量 LLM_BINDING_HOST 未设置")
    
    if not config["llm_model"]:
        logging.warning("环境变量 LLM_MODEL 未设置")
    
    # 设置日志级别
    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    config["logging_level"] = log_level_map.get(config["log_level"], logging.INFO)
    
    return config

def setup_logging(config: Dict[str, Any]) -> None:
    """
    设置日志配置
    
    Args:
        config: 应用程序配置字典
    """
    logging.basicConfig(
        level=config["logging_level"], 
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info(f"日志级别设置为: {config['log_level']}")

def load_schema_config(config_path: str) -> Config:
    """
    加载图谱模式配置
    
    从YAML文件加载实体和关系类型等配置
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        Config: 配置对象
    """
    # 使用Config类的from_yaml方法加载
    return Config.from_yaml(config_path) 