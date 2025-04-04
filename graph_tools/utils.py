"""
工具函数模块

提供各种辅助功能函数
"""

import json
import logging
from typing import List, Dict, Optional, Any
from pathlib import Path

def load_json_data(file_path: str) -> Optional[List[Dict[str, Any]]]:
    """
    从 JSON 文件加载数据
    
    Args:
        file_path: JSON文件路径
        
    Returns:
        Optional[List[Dict[str, Any]]]: 加载的数据，如果失败则返回None
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()
            
        # 检查文件内容，记录前几行内容
        preview_lines = file_content.splitlines()[:5]
        logging.debug(f"JSON 文件内容预览 ({file_path}):\n" + "\n".join(preview_lines))
        
        try:
            # 尝试解析 JSON
            data = json.loads(file_content)
            logging.info(f"Successfully loaded data from {file_path}")
            
            # 返回整个数据结构
            return data
                
        except json.JSONDecodeError as e:
            logging.error(f"Error: Could not decode JSON from {file_path}: {e}")
            logging.debug(f"JSON 解析失败，文件内容预览:\n{file_content[:1000]}...")
            return None
            
    except FileNotFoundError:
        logging.error(f"Error: Input file not found at {file_path}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during JSON loading: {e}")
        logging.exception("详细错误信息:")
        return None

def save_cypher_to_file(cypher_statements: List[str], output_path: str) -> bool:
    """
    将Cypher语句保存到文件
    
    Args:
        cypher_statements: Cypher语句列表
        output_path: 输出文件路径
        
    Returns:
        bool: 是否保存成功
    """
    try:
        # 确保输出目录存在
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 将Cypher语句写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(";\n".join(cypher_statements) + ";\n")
        
        logging.info(f"Cypher语句已保存到 {output_path}")
        return True
        
    except Exception as e:
        logging.error(f"保存Cypher语句到文件时发生错误: {e}")
        logging.exception("详细错误信息:")
        return False

def extract_document_info(data: Any) -> Dict[str, Any]:
    """
    从数据中提取文档信息
    
    处理不同的数据格式，确保返回有效的文档信息
    
    Args:
        data: 输入数据，可能是不同格式
        
    Returns:
        Dict[str, Any]: 文档信息字典
    """
    # 初始化空的文档信息
    document_info = {}
    
    # 检查数据格式
    if isinstance(data, dict):
        # 如果是字典格式，直接提取document_info
        document_info = data.get("document_info", {})
        
    elif isinstance(data, list) and len(data) > 0:
        # 如果是数组格式，尝试从第一个元素中提取文档信息
        first_item = data[0]
        if isinstance(first_item, dict):
            # 尝试从common_fields或其他字段中提取
            document_info = first_item.get("document_info", {})
            
            # 如果没有document_info字段，尝试构建一个
            if not document_info:
                document_info = {
                    "document_name": first_item.get("file_path", "").split("/")[-1] if "file_path" in first_item else "未知文档",
                    "full_doc_id": first_item.get("full_doc_id", "unknown_doc")
                }
    
    # 确保返回字典
    if not isinstance(document_info, dict):
        logging.warning(f"提取的文档信息不是字典类型: {type(document_info)}")
        document_info = {}
    
    return document_info

def normalize_data_format(data: Any) -> Dict[str, Any]:
    """
    标准化数据格式
    
    确保输入数据格式统一，便于后续处理
    
    Args:
        data: 输入数据，可能是不同格式
        
    Returns:
        Dict[str, Any]: 标准化后的数据
    """
    # 初始化返回结构
    normalized_data = {
        "chunks": [],
        "document_info": {}
    }
    
    # 处理不同的数据格式
    if isinstance(data, dict):
        # 如果是字典格式，提取chunks和document_info
        normalized_data["chunks"] = data.get("chunks", [])
        normalized_data["document_info"] = data.get("document_info", {})
    elif isinstance(data, list):
        # 如果直接是chunks数组
        normalized_data["chunks"] = data
        normalized_data["document_info"] = extract_document_info(data)
    else:
        logging.error(f"不支持的数据类型: {type(data)}")
        
    # 验证数据有效性
    if not normalized_data["chunks"]:
        logging.warning("标准化后的数据不包含任何chunks")
    
    return normalized_data 