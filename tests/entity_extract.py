import json
import os
import logging
import time # For potential rate limiting
import requests  # 用于HTTP请求，调用LLM API
import yaml  # 用于加载配置文件
import asyncio  # 用于异步并发处理
import httpx  # 用于异步HTTP请求
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass  # 用于定义数据类
from pathlib import Path  # 用于处理路径
import re  # 用于正则表达式

# --- 配置 ---

# 1. LLM Configuration
# API密钥可以通过环境变量传入，但不再有默认值
LLM_API_KEY = os.environ.get("LLM_BINDING_API_KEY") or os.environ.get("SILICONFLOW_API_KEY")

# SiliconFlow API配置（使用环境变量，无默认值）
LLM_API_HOST = os.environ.get("LLM_BINDING_HOST")
LLM_MODEL = os.environ.get("LLM_MODEL")

# 仅保留API并发设置的默认值
MAX_CONCURRENT_REQUESTS = int(os.environ.get("MAX_CONCURRENT_REQUESTS", "5"))  # 最大并发请求数
REQUEST_DELAY = float(os.environ.get("REQUEST_DELAY", "0.2"))  # 请求间隔时间（秒）
REQUEST_TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT", "60.0"))  # 请求超时时间（秒）

# 日志级别
LOG_LEVEL = os.environ.get("LOG_LEVEL", "DEBUG").upper()
log_level_map = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}
LOGGING_LEVEL = log_level_map.get(LOG_LEVEL, logging.DEBUG)

# 4. Logging Configuration
logging.basicConfig(
    level=LOGGING_LEVEL, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 尝试加载配置，如果失败则直接报错退出
try:
    # 变量声明，实际值将从配置文件加载
    ENTITY_TYPES = None
    ENTITY_TYPE_MAP_CYPHER = None
    RELATION_TYPES = None
    RELATION_TYPE_MAP_CYPHER = None 
    CANONICAL_MAP = None
    PROMPT_TEMPLATES = None
    
    # 加载配置文件
    def load_config(config_path: str) -> Dict[str, Any]:
        """从YAML配置文件加载配置信息"""
        try:
            # 尝试从指定路径加载配置
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logging.info(f"成功从 {config_path} 加载配置信息")
            
            # 基本验证
            if 'schema' not in config:
                raise ValueError("配置文件缺少必要的键: 'schema'")
            if 'normalization' not in config:
                raise ValueError("配置文件缺少必要的键: 'normalization'")
            if 'prompts' not in config:
                raise ValueError("配置文件缺少必要的键: 'prompts'")
            
            # 检查 schema 部分
            schema = config.get('schema', {})
            if 'entity_types' not in schema:
                raise ValueError("schema 部分缺少必要的键: 'entity_types'")
            if 'relation_types' not in schema:
                raise ValueError("schema 部分缺少必要的键: 'relation_types'")
            
            # 检查实体类型和关系类型的格式
            entity_types = schema.get('entity_types', [])
            relation_types = schema.get('relation_types', [])
            
            if not isinstance(entity_types, list) or not all(isinstance(t, str) for t in entity_types):
                raise ValueError("entity_types 必须是字符串列表")
            if not isinstance(relation_types, list) or not all(isinstance(t, str) for t in relation_types):
                raise ValueError("relation_types 必须是字符串列表")
            
            # 检查 normalization 部分
            normalization = config.get('normalization', {})
            if 'canonical_map' not in normalization:
                raise ValueError("normalization 部分缺少必要的键: 'canonical_map'")
            
            # 检查 prompts 部分
            prompts = config.get('prompts', {})
            if 'entity_extraction' not in prompts:
                raise ValueError("prompts 部分缺少必要的键: 'entity_extraction'")
            if 'relation_extraction' not in prompts:
                raise ValueError("prompts 部分缺少必要的键: 'relation_extraction'")
            
            # 打印加载的关键配置信息
            logging.debug(f"加载的实体类型: {entity_types}")
            logging.debug(f"加载的关系类型: {relation_types}")
            
            entity_map = schema.get('entity_type_map_cypher', {})
            relation_map = schema.get('relation_type_map_cypher', {})
            logging.debug(f"实体类型映射: {entity_map}")
            logging.debug(f"关系类型映射: {relation_map}")
            
            # 打印规范化映射条目数量
            canonical_map = normalization.get('canonical_map', {})
            logging.debug(f"规范化映射条目数: {len(canonical_map)}")
            
            # 检查 prompts 中的必要字段
            entity_extraction = prompts.get('entity_extraction', {})
            relation_extraction = prompts.get('relation_extraction', {})
            
            if 'template' not in entity_extraction or 'definitions' not in entity_extraction:
                raise ValueError("entity_extraction 缺少必要的键: 'template' 或 'definitions'")
            if 'template' not in relation_extraction or 'definitions' not in relation_extraction:
                raise ValueError("relation_extraction 缺少必要的键: 'template' 或 'definitions'")
            
            # 检查模板是否包含必要的占位符
            entity_template = entity_extraction.get('template', '')
            if '{entity_types}' not in entity_template or '{content}' not in entity_template:
                raise ValueError("entity_extraction template 缺少必要的占位符: '{entity_types}' 或 '{content}'")
            
            relation_template = relation_extraction.get('template', '')
            if '{relation_types}' not in relation_template or '{content}' not in relation_template:
                raise ValueError("relation_extraction template 缺少必要的占位符: '{relation_types}' 或 '{content}'")
            
            return config
        except FileNotFoundError:
            logging.error(f"错误: 配置文件未找到: {config_path}")
            raise  # 重新抛出异常以停止执行
        except yaml.YAMLError as e:
            logging.error(f"解析配置文件 {config_path} 时出错: {e}")
            raise
        except Exception as e:
            logging.error(f"加载配置时发生未预期的错误: {e}")
            logging.exception("详细错误信息:")
            raise

    def load_config_and_setup_globals(config_path: str):
        """加载配置并设置全局变量"""
        global ENTITY_TYPES, ENTITY_TYPE_MAP_CYPHER, RELATION_TYPES
        global RELATION_TYPE_MAP_CYPHER, CANONICAL_MAP, PROMPT_TEMPLATES
        
        # 加载配置文件
        CONFIG = load_config(config_path)
        
        # 从配置获取值
        ENTITY_TYPES = CONFIG['schema']['entity_types']
        ENTITY_TYPE_MAP_CYPHER = CONFIG['schema']['entity_type_map_cypher']
        RELATION_TYPES = CONFIG['schema']['relation_types']
        RELATION_TYPE_MAP_CYPHER = CONFIG['schema']['relation_type_map_cypher']
        CANONICAL_MAP = CONFIG['normalization']['canonical_map']
        
        # 加载Prompt模板配置
        PROMPT_TEMPLATES = CONFIG.get('prompts', {})
        
        # 验证必要的配置项存在
        if not ENTITY_TYPES or not RELATION_TYPES:
            raise ValueError("配置文件缺少必要的实体类型或关系类型定义")
        
        if not PROMPT_TEMPLATES.get('entity_extraction') or not PROMPT_TEMPLATES.get('relation_extraction'):
            raise ValueError("配置文件缺少必要的提示词模板")
            
        # 输出加载的配置摘要
        logging.info(f"加载了 {len(ENTITY_TYPES)} 个实体类型和 {len(RELATION_TYPES)} 个关系类型")
        logging.info(f"加载了 {len(CANONICAL_MAP)} 个规范化映射条目")
        
        # 返回加载的配置
        return CONFIG
except Exception as e:
    logging.error(f"初始化失败，程序无法继续: {e}")
    # 这里不抛出异常，因为这是在模块级别的代码，抛出异常会阻止脚本运行
    # 实际错误检查会在main函数中进行

# --- Helper Functions ---

def load_json_data(file_path: str) -> Optional[List[Dict[str, Any]]]:
    """从 JSON 文件加载数据"""
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
            
            # 如果数据是一个包含 chunk 数组的对象，返回 chunk 数组
            if isinstance(data, dict) and 'chunk' in data:
                chunks = data['chunk']
                logging.info(f"找到 {len(chunks)} 个 chunks")
                # 验证 chunks 格式
                for i, chunk in enumerate(chunks[:3]):  # 只打印前3个以避免日志过长
                    if not isinstance(chunk, dict):
                        logging.warning(f"Chunk {i} 不是字典格式: {type(chunk)}")
                    elif 'content' not in chunk:
                        logging.warning(f"Chunk {i} 缺少 'content' 字段")
                    elif 'chunk_id' not in chunk:
                        logging.warning(f"Chunk {i} 缺少 'chunk_id' 字段")
                return chunks
            
            # 如果数据本身就是一个数组，直接返回
            elif isinstance(data, list):
                logging.info(f"数据是列表格式，包含 {len(data)} 个项目")
                # 验证数据格式
                for i, item in enumerate(data[:3]):  # 只打印前3个以避免日志过长
                    if not isinstance(item, dict):
                        logging.warning(f"项目 {i} 不是字典格式: {type(item)}")
                    elif 'content' not in item:
                        logging.warning(f"项目 {i} 缺少 'content' 字段")
                return data
            
            else:
                logging.error(f"Unexpected JSON format in {file_path}. Expected a list or an object with 'chunk' array.")
                logging.debug(f"JSON 结构: {type(data)}")
                return None
                
        except json.JSONDecodeError as e:
            logging.error(f"Error: Could not decode JSON from {file_path}: {e}")
            
            # 如果解析失败，尝试手动修复一些常见问题
            if '"chunk"' in file_content and '"entities"' in file_content:
                logging.warning("尝试手动解析 JSON 数据...")
                
                # 尝试提取 chunk 数组
                chunk_start = file_content.find('"chunk"')
                if chunk_start != -1:
                    square_bracket_start = file_content.find('[', chunk_start)
                    if square_bracket_start != -1:
                        # 找到匹配的右方括号
                        level = 0
                        for i in range(square_bracket_start, len(file_content)):
                            if file_content[i] == '[':
                                level += 1
                            elif file_content[i] == ']':
                                level -= 1
                                if level == 0:
                                    # 找到了匹配的右方括号
                                    chunk_content = file_content[square_bracket_start:i+1]
                                    try:
                                        chunks = json.loads(chunk_content)
                                        logging.info(f"手动解析成功，找到 {len(chunks)} 个 chunks")
                                        return chunks
                                    except json.JSONDecodeError:
                                        logging.error("手动解析 chunk 数组失败")
                                    break
            
            # 记录文件内容以便调试
            logging.debug(f"JSON 解析失败，文件内容预览:\n{file_content[:1000]}...")
            return None
            
    except FileNotFoundError:
        logging.error(f"Error: Input file not found at {file_path}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during JSON loading: {e}")
        logging.exception("详细错误信息:")
        return None


def normalize_entity_name(raw_name: str) -> str:
    """使用配置的CANONICAL_MAP规范化实体名称。"""
    if not isinstance(raw_name, str):
        logging.warning(f"Attempted to normalize non-string value: {raw_name}. Returning as is.")
        return str(raw_name)
    cleaned_name = raw_name.strip().replace('\n', ' ')
    return CANONICAL_MAP.get(cleaned_name, cleaned_name)

def normalize_entity_type(raw_type: str) -> str:
    """规范化实体类型，确保使用配置中定义的标准类型。
    
    将尝试匹配中英文类型名称，确保使用ENTITY_TYPE_MAP_CYPHER中定义的标准类型。
    """
    if not isinstance(raw_type, str):
        logging.warning(f"Attempted to normalize non-string entity type: {raw_type}. Returning as is.")
        return str(raw_type)
    
    cleaned_type = raw_type.strip().replace('\n', ' ')
    
    # 如果是中文类型且存在于ENTITY_TYPE_MAP_CYPHER中，返回对应的英文类型
    if cleaned_type in ENTITY_TYPES:
        return ENTITY_TYPE_MAP_CYPHER.get(cleaned_type, cleaned_type)
    
    # 如果是英文类型且是ENTITY_TYPE_MAP_CYPHER的值的一部分，保持原样返回
    if cleaned_type in ENTITY_TYPE_MAP_CYPHER.values():
        return cleaned_type
    
    # 尝试通过反向查找，从英文映射回中文再映射回标准英文
    for cn_type, en_type in ENTITY_TYPE_MAP_CYPHER.items():
        if cleaned_type.lower() == en_type.lower():
            return en_type  # 返回正确大小写的英文类型名
    
    # 如果无法匹配，记录警告并返回原始类型
    logging.warning(f"实体类型 '{cleaned_type}' 未在配置中定义，无法规范化")
    return cleaned_type

def escape_cypher_string(value: str) -> str:
    """Escapes single quotes and backslashes for Cypher strings."""
    if not isinstance(value, str):
        return str(value) # Return as string if not already
    return value.replace('\\', '\\\\').replace("'", "\\'")

def parse_llm_response(response_text: Optional[str]) -> Optional[Dict[str, List[Dict[str, str]]]]:
    """Safely parses the LLM's JSON response."""
    if not response_text: 
        return None
        
    try:
        # 1. 首先尝试直接解析完整响应，以处理格式良好的 JSON
        try:
            parsed_data = json.loads(response_text.strip())
            if isinstance(parsed_data, dict) and \
               (('entities' in parsed_data and isinstance(parsed_data['entities'], list)) or \
                ('relations' in parsed_data and isinstance(parsed_data['relations'], list))):
                return parsed_data
        except json.JSONDecodeError:
            # 如果直接解析失败，继续使用更细致的方法
            pass
        
        # 2. 处理可能包含在代码块中的 JSON
        cleaned_text = response_text
        if "```" in cleaned_text:
            # 如果有多个代码块，取最大的一个
            blocks = []
            start = 0
            while True:
                start_mark = cleaned_text.find("```", start)
                if start_mark == -1:
                    break
                end_mark = cleaned_text.find("```", start_mark + 3)
                if end_mark == -1:
                    break
                blocks.append((start_mark, end_mark + 3, end_mark - start_mark))
                start = end_mark + 3
            
            if blocks:
                # 取最长的代码块
                largest_block = max(blocks, key=lambda x: x[2])
                start_content = cleaned_text.find("\n", largest_block[0]) + 1
                if start_content > 0 and start_content < largest_block[1]:
                    end_content = largest_block[1]
                    cleaned_text = cleaned_text[start_content:end_content].strip()
        
        # 3. 处理 JSON 可能嵌入在其他文本中的情况
        json_start = cleaned_text.find('{')
        json_end = cleaned_text.rfind('}')
        
        if json_start == -1 or json_end == -1:
            # 尝试一种更宽松的方式查找 "entities" 或 "relations" 关键字
            entities_pos = cleaned_text.find('"entities"')
            relations_pos = cleaned_text.find('"relations"')
            
            if entities_pos != -1 or relations_pos != -1:
                # 向前寻找最近的 {，向后寻找匹配的 }
                pos = min(p for p in [entities_pos, relations_pos] if p != -1)
                bracket_count = 0
                json_start = pos
                while json_start >= 0:
                    if cleaned_text[json_start] == '{':
                        bracket_count += 1
                        if bracket_count > 0:
                            break
                    elif cleaned_text[json_start] == '}':
                        bracket_count -= 1
                    json_start -= 1
                
                if json_start == -1:
                    # 如果找不到起始的 {，尝试使用其他解析策略
                    logging.warning(f"Could not find JSON structure in response: {response_text}")
                    return None
                    
                # 找到匹配的结束 }
                bracket_count = 1  # 已经找到一个 {
                json_end = pos
                while json_end < len(cleaned_text):
                    if cleaned_text[json_end] == '{':
                        bracket_count += 1
                    elif cleaned_text[json_end] == '}':
                        bracket_count -= 1
                        if bracket_count == 0:
                            break
                    json_end += 1
                
                if bracket_count != 0:
                    # 括号不匹配，返回错误
                    logging.warning(f"Unbalanced brackets in JSON: {cleaned_text}")
                    return None
            else:
                logging.warning(f"No JSON structure indicators found in response: {response_text}")
                return None
        
        # 提取并尝试解析 JSON 部分
        json_text = cleaned_text[json_start:json_end + 1]
        
        # 4. 最终清理和解析
        try:
            # 有时候 JSON 文本中可能有额外的空格、注释或格式不正确
            # 尝试更宽松的处理
            cleaned_json = re.sub(r'//.*?$', '', json_text, flags=re.MULTILINE)  # 移除注释
            cleaned_json = re.sub(r'/\*.*?\*/', '', cleaned_json, flags=re.DOTALL)  # 移除块注释
            parsed_data = json.loads(cleaned_json)
        except:
            # 尝试一种更宽松的方式创建 JSON
            try:
                if '"entities"' in json_text:
                    # 手动构建简单的实体 JSON
                    entities_match = re.search(r'"entities"\s*:\s*\[(.*?)\]', json_text, re.DOTALL)
                    if entities_match:
                        entities_content = entities_match.group(1).strip()
                        if entities_content:
                            # 尝试解析实体列表
                            fixed_json = f'{{"entities": [{entities_content}]}}'
                            parsed_data = json.loads(fixed_json)
                        else:
                            parsed_data = {"entities": []}
                    else:
                        parsed_data = {"entities": []}
                elif '"relations"' in json_text:
                    # 手动构建简单的关系 JSON
                    relations_match = re.search(r'"relations"\s*:\s*\[(.*?)\]', json_text, re.DOTALL)
                    if relations_match:
                        relations_content = relations_match.group(1).strip()
                        if relations_content:
                            # 尝试解析关系列表
                            fixed_json = f'{{"relations": [{relations_content}]}}'
                            parsed_data = json.loads(fixed_json)
                        else:
                            parsed_data = {"relations": []}
                    else:
                        parsed_data = {"relations": []}
                else:
                    raise ValueError("Neither entities nor relations found in JSON")
            except Exception as inner_e:
                logging.error(f"Failed to repair JSON: {inner_e}\nOriginal JSON: {json_text}")
                return None
        
        # 5. 验证解析结果是否符合预期格式
        if isinstance(parsed_data, dict) and \
           (('entities' in parsed_data and isinstance(parsed_data['entities'], list)) or \
            ('relations' in parsed_data and isinstance(parsed_data['relations'], list))):
            return parsed_data
        else:
            logging.warning(f"LLM response is valid JSON but not the expected structure: {json_text}")
            return None
    except json.JSONDecodeError as e:
        logging.error(f"Failed to decode LLM JSON response: {e}\nResponse text:\n{response_text}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during LLM response parsing: {e}\nResponse text:\n{response_text}")
        return None

def create_entity_prompt(chunk_content: str) -> str:
    """使用配置的实体类型创建实体提取提示。"""
    # 获取模板配置
    entity_config = PROMPT_TEMPLATES.get('entity_extraction', {})
    definitions = entity_config.get('definitions', '')
    template = entity_config.get('template', '')
    
    # 如果没有找到模板，返回默认消息
    if not template:
        logging.warning("未找到实体提取模板，使用默认空模板")
        return "请提取实体"
    
    # 预处理内容字符串，避免格式占位符冲突
    # 使用使用命名参数的格式化方法，避免内容中的花括号被误解为格式占位符
    try:
        # 格式化模板
        prompt = template.format(
            definitions=definitions,
            entity_types=', '.join(ENTITY_TYPES),
            content=chunk_content
        )
    except KeyError as e:
        # 如果发生KeyError，可能是内容中含有花括号或特殊字符，尝试使用安全的替代方法
        logging.warning(f"格式化模板时出现KeyError: {e}，尝试使用替代方法")
        # 将模板中的{content}替换为实际内容，而不使用format方法
        placeholder = "{content}"
        if placeholder in template:
            prompt = template.replace(placeholder, chunk_content)
            # 然后手动替换其他占位符
            prompt = prompt.replace("{definitions}", definitions)
            prompt = prompt.replace("{entity_types}", ', '.join(ENTITY_TYPES))
        else:
            # 如果没有找到{content}占位符，记录错误并返回简单提示
            logging.error(f"模板中找不到{{content}}占位符，使用简单提示。错误: {e}")
            prompt = f"请从以下内容中提取实体类型: {', '.join(ENTITY_TYPES)}\n\n{chunk_content}"
    except Exception as e:
        # 捕获其他可能的异常
        logging.error(f"创建实体提取提示时发生错误: {e}")
        prompt = f"请从以下内容中提取实体类型: {', '.join(ENTITY_TYPES)}\n\n{chunk_content}"
    
    return prompt

def create_relation_prompt(chunk_content: str) -> str:
    """使用配置的关系类型创建关系提取提示。"""
    # 获取模板配置
    relation_config = PROMPT_TEMPLATES.get('relation_extraction', {})
    definitions = relation_config.get('definitions', '')
    template = relation_config.get('template', '')
    
    # 如果没有找到模板，返回默认消息
    if not template:
        logging.warning("未找到关系提取模板，使用默认空模板")
        return "请提取关系"
    
    # 使用安全的模板格式化方法，避免内容中的花括号被误解为格式占位符
    try:
        # 格式化模板
        prompt = template.format(
            definitions=definitions,
            relation_types=', '.join(RELATION_TYPES),
            entity_types=', '.join(ENTITY_TYPES),
            content=chunk_content
        )
    except KeyError as e:
        # 如果发生KeyError，可能是内容中含有花括号或特殊字符，尝试使用安全的替代方法
        logging.warning(f"格式化关系提取模板时出现KeyError: {e}，尝试使用替代方法")
        # 将模板中的{content}替换为实际内容，而不使用format方法
        placeholder = "{content}"
        if placeholder in template:
            prompt = template.replace(placeholder, chunk_content)
            # 然后手动替换其他占位符
            prompt = prompt.replace("{definitions}", definitions)
            prompt = prompt.replace("{relation_types}", ', '.join(RELATION_TYPES))
            prompt = prompt.replace("{entity_types}", ', '.join(ENTITY_TYPES))
        else:
            # 如果没有找到{content}占位符，记录错误并返回简单提示
            logging.error(f"关系提取模板中找不到{{content}}占位符，使用简单提示。错误: {e}")
            prompt = f"请从以下内容中提取关系类型: {', '.join(RELATION_TYPES)}\n\n实体类型包括: {', '.join(ENTITY_TYPES)}\n\n{chunk_content}"
    except Exception as e:
        # 捕获其他可能的异常
        logging.error(f"创建关系提取提示时发生错误: {e}")
        prompt = f"请从以下内容中提取关系类型: {', '.join(RELATION_TYPES)}\n\n实体类型包括: {', '.join(ENTITY_TYPES)}\n\n{chunk_content}"
    
    return prompt


# --- LLM Interaction ---

# 定义请求任务数据类
@dataclass
class LLMTask:
    """封装LLM请求的任务类"""
    chunk_id: str        # 当前处理的chunk ID
    prompt_type: str     # 提示类型：'entity' 或 'relation'
    content: str         # 要处理的文本内容
    prompt: str          # 完整的提示信息
    result: Optional[Dict] = None  # 处理结果

async def call_llm_async(task: LLMTask, max_retries: int = 3, retry_delay: float = 2.0) -> LLMTask:
    """
    异步调用LLM API进行推理
    
    Args:
        task: LLM任务对象
        max_retries: 最大重试次数
        retry_delay: 重试延迟（秒）
        
    Returns:
        处理后的LLM任务对象
    """
    # 异步调用 LLM API，包含重试逻辑，确保网络请求稳定可靠
    logging.info(f"--- 发送Prompt到LLM ({task.prompt_type} - chunk {task.chunk_id}) ---")

    if not LLM_API_KEY:
        logging.error(f"LLM API密钥未设置，无法调用API (chunk {task.chunk_id})")
        return task
        
    # 重试计数器
    retry_count = 0
    
    while retry_count <= max_retries:
        try:
            # 使用SiliconFlow API (OpenAI兼容接口)调用Qwen模型
            api_url = f"{LLM_API_HOST}/chat/completions"
            
            headers = {
                "Authorization": f"Bearer {LLM_API_KEY}",
                "Content-Type": "application/json"
            }
            
            # OpenAI兼容接口的参数
            payload = {
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": "你是一个实体关系提取助手，善于从文本中提取结构化信息并以JSON格式输出。"},
                    {"role": "user", "content": task.prompt}
                ],
                "max_tokens": 2000,
                "temperature": 0.1,  # 设置较低的温度以获得确定性结果
                "response_format": {"type": "text"}
            }
            
            logging.info(f"--- 等待LLM响应 (chunk {task.chunk_id}) ---")
            
            # 使用httpx进行异步请求
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    api_url, 
                    headers=headers, 
                    json=payload, 
                    timeout=REQUEST_TIMEOUT
                )
                
            if response.status_code == 200:
                response_json = response.json()
                # 解析OpenAI API返回的JSON结果
                llm_response = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
                logging.info(f"--- 成功接收LLM响应 (chunk {task.chunk_id}) ---")
                
                # 记录原始响应，便于调试
                logging.debug(f"LLM原始响应 (chunk {task.chunk_id}):\n{llm_response[:500]}...")
                
                try:
                    # 尝试解析响应
                    task.result = parse_llm_response(llm_response)
                    
                    # 检查是否成功解析结果
                    if task.result is None:
                        logging.warning(f"LLM响应无法解析为有效JSON (chunk {task.chunk_id})。将尝试重试。")
                        logging.debug(f"完整响应内容 (chunk {task.chunk_id}):\n{llm_response}")
                        retry_count += 1
                        if retry_count <= max_retries:
                            logging.info(f"重试 {retry_count}/{max_retries}...")
                            await asyncio.sleep(retry_delay * retry_count)  # 指数退避
                            continue
                        else:
                            logging.error(f"达到最大重试次数，放弃处理 chunk {task.chunk_id}")
                            break
                except Exception as parse_error:
                    logging.error(f"解析LLM响应时发生错误 (chunk {task.chunk_id}): {parse_error}")
                    logging.debug(f"引发解析错误的响应内容 (chunk {task.chunk_id}):\n{llm_response}")
                    retry_count += 1
                    if retry_count <= max_retries:
                        logging.info(f"重试 {retry_count}/{max_retries}...")
                        await asyncio.sleep(retry_delay * retry_count)
                        continue
                    else:
                        logging.error(f"达到最大重试次数，放弃处理 chunk {task.chunk_id}")
                        break
                        
                # 成功解析结果，返回
                return task
                
            elif response.status_code == 401:
                logging.error(f"API认证失败: 无效的API密钥 (chunk {task.chunk_id})。请检查API密钥是否正确设置。")
                break  # 认证失败，不重试
            elif response.status_code == 429:
                logging.warning(f"API请求过于频繁 (chunk {task.chunk_id})。将尝试重试。")
                retry_count += 1
                if retry_count <= max_retries:
                    logging.info(f"重试 {retry_count}/{max_retries}...")
                    await asyncio.sleep(retry_delay * retry_count * 2)  # 对于速率限制错误增加更长的延迟
                    continue
                else:
                    logging.error(f"达到最大重试次数，放弃处理 chunk {task.chunk_id}")
                    break
            else:
                logging.error(f"API调用失败 (chunk {task.chunk_id}): {response.status_code} - {response.text}")
                retry_count += 1
                if retry_count <= max_retries:
                    logging.info(f"重试 {retry_count}/{max_retries}...")
                    await asyncio.sleep(retry_delay * retry_count)  # 指数退避
                    continue
                else:
                    logging.error(f"达到最大重试次数，放弃处理 chunk {task.chunk_id}")
                    break
                    
        except httpx.TimeoutException:
            logging.error(f"API请求超时 (chunk {task.chunk_id})")
            retry_count += 1
            if retry_count <= max_retries:
                logging.info(f"重试 {retry_count}/{max_retries}...")
                await asyncio.sleep(retry_delay * retry_count)
                continue
            else:
                logging.error(f"达到最大重试次数，放弃处理 chunk {task.chunk_id}")
                break
                
        except httpx.RequestError as e:
            logging.error(f"API请求错误 (chunk {task.chunk_id}): {e}")
            retry_count += 1
            if retry_count <= max_retries:
                logging.info(f"重试 {retry_count}/{max_retries}...")
                await asyncio.sleep(retry_delay * retry_count)
                continue
            else:
                logging.error(f"达到最大重试次数，放弃处理 chunk {task.chunk_id}")
                break
                
        except json.JSONDecodeError as e:
            logging.error(f"JSON解析错误 (chunk {task.chunk_id}): {e}")
            retry_count += 1
            if retry_count <= max_retries:
                logging.info(f"重试 {retry_count}/{max_retries}...")
                await asyncio.sleep(retry_delay * retry_count)
                continue
            else:
                logging.error(f"达到最大重试次数，放弃处理 chunk {task.chunk_id}")
                break
                
        except asyncio.CancelledError:
            logging.warning(f"任务被取消 (chunk {task.chunk_id})")
            raise  # 重新抛出取消异常
            
        except Exception as e:
            logging.error(f"调用LLM时发生未知错误 (chunk {task.chunk_id}): {e}")
            logging.exception("详细错误信息:")
            retry_count += 1
            if retry_count <= max_retries:
                logging.info(f"重试 {retry_count}/{max_retries}...")
                await asyncio.sleep(retry_delay * retry_count)
                continue
            else:
                logging.error(f"达到最大重试次数，放弃处理 chunk {task.chunk_id}")
                break
    
    return task

# 为了兼容原有代码，保留同步调用接口
def call_llm(prompt: str) -> Optional[str]:
    """
    同步调用LLM API进行推理，兼容原有代码
    实际调用异步方法并等待其完成
    """
    logging.info(f"--- 使用同步接口发送Prompt到LLM (长度: {len(prompt)}) ---")
    
    # 创建一个任务（chunk_id和content仅用于测试模式）
    task = LLMTask(
        chunk_id="sync_call",
        prompt_type="unknown",
        content=prompt,  # 用于测试模式条件判断
        prompt=prompt
    )
    
    # 运行异步调用并等待结果
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # 如果没有事件循环，创建一个新的
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    task = loop.run_until_complete(call_llm_async(task))
    
    # 如果有结果对象，返回其JSON字符串表示
    if task.result:
        if 'entities' in task.result:
            return json.dumps({"entities": task.result['entities']})
        elif 'relations' in task.result:
            return json.dumps({"relations": task.result['relations']})
    
    return None

# 限流处理器
class RateLimiter:
    """简单的速率限制实现，控制并发请求数和请求间隔"""
    
    def __init__(self, max_concurrent: int, delay_seconds: float):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.delay = delay_seconds
    
    async def acquire(self):
        """获取一个请求槽位"""
        await self.semaphore.acquire()
    
    async def release(self):
        """释放一个槽位并等待指定延迟"""
        await asyncio.sleep(self.delay)  # 等待指定的时间间隔
        self.semaphore.release()

# 异步处理多个任务
async def process_tasks(tasks: List[LLMTask]) -> List[LLMTask]:
    """异步并发处理多个LLM任务，并进行速率限制"""
    # 创建限流器
    limiter = RateLimiter(MAX_CONCURRENT_REQUESTS, REQUEST_DELAY)
    
    async def process_with_rate_limit(task):
        """使用速率限制处理单个任务"""
        try:
            await limiter.acquire()
            logging.debug(f"开始处理任务: {task.chunk_id} ({task.prompt_type})")
            result = await call_llm_async(task)
            logging.debug(f"完成任务: {task.chunk_id} ({task.prompt_type}), 结果状态: {'成功' if result.result else '失败'}")
            return result
        except asyncio.CancelledError:
            logging.info(f"任务被取消: {task.chunk_id}")
            raise
        except Exception as e:
            logging.error(f"处理任务 {task.chunk_id} 时出错: {e}")
            logging.exception("详细错误信息:")
            # 返回原始任务，但不设置结果
            return task
        finally:
            await limiter.release()
    
    # 创建所有任务的协程
    coroutines = [process_with_rate_limit(task) for task in tasks]
    
    try:
        # 一个任务一个任务地处理，而不是使用 gather，以便更好地定位错误
        # (在生产环境可能会使用 gather 以提高效率)
        processed_tasks = []
        for i, task in enumerate(tasks):
            try:
                logging.debug(f"处理任务 {i+1}/{len(tasks)}: {task.chunk_id} ({task.prompt_type})")
                result = await process_with_rate_limit(task)
                processed_tasks.append(result)
                if (i+1) % 10 == 0:
                    logging.info(f"已处理 {i+1}/{len(tasks)} 个任务")
            except Exception as e:
                logging.error(f"处理任务 {i+1}/{len(tasks)} 时发生异常: {e}")
                logging.exception("详细错误信息:")
                # 保留原任务，但不设置结果
                processed_tasks.append(task)
        
        return processed_tasks
    except asyncio.CancelledError:
        logging.info("正在取消所有任务...")
        raise
    except Exception as e:
        logging.error(f"处理任务时发生错误: {e}")
        logging.exception("详细错误信息:")
        raise


# --- Cypher Generation ---

def generate_cypher_statements(entities: Set[Tuple[str, str, str]], relations: Set[Tuple[str, str, str, str]]) -> List[str]:
    """
    使用配置的映射生成Memgraph/Neo4j Cypher MERGE语句。
    
    现在支持实体的chunk_id和entity_type属性，关系的chunk_id和relation_type属性。
    
    Args:
        entities: 包含(name, type, chunk_id)的实体集合
        relations: 包含(source, target, type, chunk_id)的关系集合
        
    Returns:
        Cypher语句列表
    """
    cypher_statements = []

    # 添加唯一性约束（取消注释，建议执行）
    cypher_statements.append("\n// --- Uniqueness Constraints ---")
    cypher_statements.append("// 注意：以下约束语句应在数据库初始化时执行一次")
    cypher_statements.append("// 多次执行会报错，建议在首次运行时使用，后续导入数据时注释掉")
    for entity_type_cn in ENTITY_TYPES:
        entity_type_cypher = ENTITY_TYPE_MAP_CYPHER.get(entity_type_cn, entity_type_cn) # Use mapped or original
        # 根据Neo4j/Memgraph版本选择适当的约束语法
        # Neo4j 4.x+
        cypher_statements.append(f"// CREATE CONSTRAINT IF NOT EXISTS ON (n:`{entity_type_cypher}`) ASSERT n.name IS UNIQUE;")
        # 旧版本Neo4j或Memgraph
        cypher_statements.append(f"// CREATE CONSTRAINT ON (n:`{entity_type_cypher}`) ASSERT n.name IS UNIQUE;")

    cypher_statements.append("\n// --- Entity Creation ---")
    
    # 规范化实体类型，创建新的规范化实体集合
    normalized_entities = set()
    for name, type_cn, chunk_id in entities:
        if not name:  # Skip empty names
            continue
        normalized_type = normalize_entity_type(type_cn)
        normalized_entities.add((name, normalized_type, chunk_id))
    
    sorted_entities = sorted(list(normalized_entities))
    # 创建实体类型到实体名称的映射，方便后续查找
    entity_type_mapping = {}
    for name, type_cypher, chunk_id in sorted_entities:
        if not name: # Skip empty names
            continue
        if name not in entity_type_mapping:
            entity_type_mapping[name] = set()
        entity_type_mapping[name].add(type_cypher)
        
        escaped_name = escape_cypher_string(name)
        # 添加唯一ID属性以提高唯一性识别能力，并添加chunk_id和entity_type属性
        cypher_statements.append(f"MERGE (n:`{type_cypher}` {{name: '{escaped_name}'}}) "
                                 f"ON CREATE SET n.uuid = '{type_cypher}_' + timestamp() + '_' + toString(rand()), "
                                 f"n.chunk_id = '{chunk_id}', "
                                 f"n.entity_type = '{type_cypher}';")

    cypher_statements.append("\n// --- Relationship Creation ---")
    # 记录无法匹配类型的关系数量
    untyped_relations_count = 0
    ambiguous_relations_count = 0
    
    sorted_relations = sorted(list(relations))
    for source, target, type_cn, chunk_id in sorted_relations:
        if not source or not target: # Skip if source or target is missing
            continue
        relation_type_cypher = RELATION_TYPE_MAP_CYPHER.get(type_cn, type_cn.upper().replace(" ", "_")) # Map or sanitize
        escaped_source = escape_cypher_string(source)
        escaped_target = escape_cypher_string(target)

        # 查找实体的类型
        source_types = entity_type_mapping.get(source, set())
        target_types = entity_type_mapping.get(target, set())
        
        # 从规范化实体集合中查找对应的实体类型
        source_type_cypher = next((t for n, t, c in sorted_entities if n == source), None)
        target_type_cypher = next((t for n, t, c in sorted_entities if n == target), None)
        
        # 如果实体有多种可能的类型，记录为歧义关系
        source_ambiguous = len(source_types) > 1
        target_ambiguous = len(target_types) > 1

        if source_type_cypher and target_type_cypher:
            # 使用规范化的类型标签进行精确匹配，并添加属性
            cypher_statements.append(
                f"MATCH (a:`{source_type_cypher}` {{name: '{escaped_source}'}}), "
                f"(b:`{target_type_cypher}` {{name: '{escaped_target}'}}) "
                f"MERGE (a)-[r:`{relation_type_cypher}`]->(b) "
                f"ON CREATE SET r.created_at = timestamp(), "
                f"r.chunk_id = '{chunk_id}', "
                f"r.relation_type = '{type_cn}';")
        else:
            # 如果存在歧义，添加警告
            if source_ambiguous or target_ambiguous:
                relation_note = f"Ambiguous entities in relationship: ({source})"
                if source_ambiguous:
                    relation_note += f"[多种类型: {', '.join(source_types)}]"
                relation_note += f"-[{type_cn}]->({target})"
                if target_ambiguous:
                    relation_note += f"[多种类型: {', '.join(target_types)}]"
                ambiguous_relations_count += 1
                logging.warning(relation_note)
            else:
                untyped_relations_count += 1
                
            # 改进的匹配逻辑，添加属性
            if source_type_cypher and not target_type_cypher:
                # source_type_cypher已经是规范化后的类型，不需要再次转换
                cypher_statements.append(
                    f"MATCH (a:`{source_type_cypher}` {{name: '{escaped_source}'}}), "
                    f"(b {{name: '{escaped_target}'}}) "
                    f"MERGE (a)-[r:`{relation_type_cypher}`]->(b) "
                    f"ON CREATE SET r.match_type = 'source_typed', r.created_at = timestamp(), "
                    f"r.chunk_id = '{chunk_id}', "
                    f"r.relation_type = '{type_cn}';")
            elif not source_type_cypher and target_type_cypher:
                # target_type_cypher已经是规范化后的类型，不需要再次转换
                cypher_statements.append(
                    f"MATCH (a {{name: '{escaped_source}'}}), "
                    f"(b:`{target_type_cypher}` {{name: '{escaped_target}'}}) "
                    f"MERGE (a)-[r:`{relation_type_cypher}`]->(b) "
                    f"ON CREATE SET r.match_type = 'target_typed', r.created_at = timestamp(), "
                    f"r.chunk_id = '{chunk_id}', "
                    f"r.relation_type = '{type_cn}';")
            else:
                # 如果两边都没有明确类型，使用模糊匹配但添加警告标记
                cypher_statements.append(
                    f"MATCH (a {{name: '{escaped_source}'}}), (b {{name: '{escaped_target}'}}) "
                    f"MERGE (a)-[r:`{relation_type_cypher}`]->(b) "
                    f"ON CREATE SET r.match_type = 'untyped', r.reliability = 'low', "
                    f"r.created_at = timestamp(), r.warning = '实体类型未知，可能存在错误匹配', "
                    f"r.chunk_id = '{chunk_id}', "
                    f"r.relation_type = '{type_cn}';")

    if untyped_relations_count > 0:
        logging.warning(f"发现 {untyped_relations_count} 个关系的实体类型无法确定，已使用通用匹配并标记为低可靠性。")
    if ambiguous_relations_count > 0:
        logging.warning(f"发现 {ambiguous_relations_count} 个关系中存在实体类型歧义，请检查日志获取详情。")
        
    return cypher_statements

# --- Main Processing Logic ---

async def extract_entities_and_relations(data: List[Dict[str, Any]]) -> Tuple[Set[Tuple[str, str, str]], Set[Tuple[str, str, str, str]]]:
    """
    并发处理多个文本块，提取实体和关系
    
    Args:
        data: 包含多个文本块的列表
        
    Returns:
        一个元组，包含两个集合：实体集合(name, type, chunk_id)和关系集合(source, target, type, chunk_id)
    """
    # 关键步骤：遍历文本块数据，创建并调度 LLM 提取任务，最后整合和过滤提取结果
    # 首先验证输入数据是否符合预期
    if not data:
        raise ValueError("输入数据为空，无法处理")
    
    if not isinstance(data, list):
        raise TypeError(f"输入数据必须是列表类型，但实际是 {type(data)}")
    
    # 验证数据项格式
    for i, chunk in enumerate(data[:5]):  # 只检查前5个
        if not isinstance(chunk, dict):
            raise TypeError(f"数据项 {i} 必须是字典类型，但实际是 {type(chunk)}")
        
        if 'chunk_id' not in chunk:
            logging.warning(f"数据项 {i} 缺少 'chunk_id' 字段，将使用自动生成ID")
        
        if 'content' not in chunk:
            raise ValueError(f"数据项 {i} 缺少必需的 'content' 字段")
        
        if not isinstance(chunk.get('content', ''), str):
            raise TypeError(f"数据项 {i} 的 'content' 字段必须是字符串类型，但实际是 {type(chunk.get('content'))}")
    
    # 修改返回类型，增加chunk_id属性
    entities_set: Set[Tuple[str, str, str]] = set()  # (name, type, chunk_id)
    relations_set: Set[Tuple[str, str, str, str]] = set()  # (source, target, type, chunk_id)
    # 存储在处理关系时获取的类型信息
    relation_type_info: Dict[str, Set[str]] = {}
    
    # 调试信息：打印前几个数据项的内容
    for i in range(min(2, len(data))):
        chunk = data[i]
        logging.debug(f"数据样本 {i+1}:")
        logging.debug(f"chunk_id: {chunk.get('chunk_id')}")
        if 'content' in chunk:
            content_preview = chunk['content'][:100] + '...' if len(chunk['content']) > 100 else chunk['content']
            logging.debug(f"content: {content_preview}")
    
    # 准备所有任务
    all_tasks = []
    try:
        for i, chunk in enumerate(data):
            chunk_id = chunk.get("chunk_id", f"unknown_{i}")
            content = chunk.get("content")
            if not content:
                logging.warning(f"Chunk {chunk_id}（索引 {i}）没有内容，跳过")
                continue
            
            # 创建实体提取任务
            try:
                entity_prompt = create_entity_prompt(content)
                if i < 1:  # 只打印第一个任务的提示词，避免日志过多
                    logging.debug(f"实体提取提示词示例 (chunk {chunk_id}):\n{entity_prompt[:200]}...")
                
                all_tasks.append(LLMTask(
                    chunk_id=chunk_id,
                    prompt_type="entity",
                    content=content,
                    prompt=entity_prompt
                ))
            except Exception as e:
                logging.error(f"为 chunk {chunk_id} 创建实体提取任务时出错: {e}")
                logging.exception("详细错误信息:")
                continue
            
            # 创建关系提取任务
            try:
                relation_prompt = create_relation_prompt(content)
                if i < 1:  # 只打印第一个任务的提示词，避免日志过多
                    logging.debug(f"关系提取提示词示例 (chunk {chunk_id}):\n{relation_prompt[:200]}...")
                
                all_tasks.append(LLMTask(
                    chunk_id=chunk_id,
                    prompt_type="relation",
                    content=content,
                    prompt=relation_prompt
                ))
            except Exception as e:
                logging.error(f"为 chunk {chunk_id} 创建关系提取任务时出错: {e}")
                logging.exception("详细错误信息:")
                continue
        
        logging.info(f"创建了 {len(all_tasks)} 个任务，准备开始处理...")
    except Exception as e:
        logging.error(f"创建任务时发生错误: {e}")
        logging.exception("详细错误信息:")
        raise
    
    try:
        start_time = time.time()
        logging.info(f"开始并发处理 {len(all_tasks)} 个任务，最大并发数: {MAX_CONCURRENT_REQUESTS}，请求延迟: {REQUEST_DELAY}秒...")
        processed_tasks = await process_tasks(all_tasks)
        elapsed_time = time.time() - start_time
        logging.info(f"完成所有任务，耗时: {elapsed_time:.2f}秒")
    except Exception as e:
        logging.error(f"处理任务时发生错误: {e}")
        logging.exception("详细错误信息:")
        raise
    
    # 处理任务结果
    logging.info("处理任务结果...")
    
    # 先处理实体提取结果
    for task in processed_tasks:
        if not task.result:
            continue
            
        if task.prompt_type == "entity" and 'entities' in task.result:
            for entity in task.result['entities']:
                raw_name = entity.get('name')
                raw_type = entity.get('type')
                if raw_name and raw_type and raw_type in ENTITY_TYPES:
                    normalized_name = normalize_entity_name(raw_name)
                    # 规范化实体类型
                    normalized_type = normalize_entity_type(raw_type)
                    # 将实体添加到集合中，包含chunk_id属性
                    entities_set.add((normalized_name, normalized_type, task.chunk_id))
                else: 
                    logging.warning(f"Invalid entity format or type in chunk {task.chunk_id}: {entity}")
    
    # 然后处理关系提取结果（可能包含额外的实体类型信息）
    relation_count = 0
    enriched_relation_count = 0
    
    for task in processed_tasks:
        if not task.result or task.prompt_type != "relation" or 'relations' not in task.result:
            continue
            
        for relation in task.result['relations']:
            raw_source = relation.get('source')
            raw_target = relation.get('target')
            raw_type = relation.get('type')
            
            # 检查新格式中的类型信息
            source_type = relation.get('source_type')
            target_type = relation.get('target_type')
            
            if raw_source and raw_target and raw_type and raw_type in RELATION_TYPES:
                normalized_source = normalize_entity_name(raw_source)
                normalized_target = normalize_entity_name(raw_target)
                
                # 添加关系，包含chunk_id属性
                relations_set.add((normalized_source, normalized_target, raw_type, task.chunk_id))
                relation_count += 1
                
                # 存储实体类型信息并添加到实体集合，包含chunk_id属性
                if source_type and source_type in ENTITY_TYPES:
                    # 规范化源实体类型
                    normalized_source_type = normalize_entity_type(source_type)
                    # 使用关系中提供的类型丰富实体集合
                    entities_set.add((normalized_source, normalized_source_type, task.chunk_id))
                    # 记录类型信息，用于后续关系处理
                    if normalized_source not in relation_type_info:
                        relation_type_info[normalized_source] = set()
                    relation_type_info[normalized_source].add(normalized_source_type)
                    enriched_relation_count += 1
                
                if target_type and target_type in ENTITY_TYPES:
                    # 规范化目标实体类型
                    normalized_target_type = normalize_entity_type(target_type)
                    entities_set.add((normalized_target, normalized_target_type, task.chunk_id))
                    if normalized_target not in relation_type_info:
                        relation_type_info[normalized_target] = set()
                    relation_type_info[normalized_target].add(normalized_target_type)
                    enriched_relation_count += 1
            else: 
                logging.warning(f"Invalid relation format or type in chunk {task.chunk_id}: {relation}")

    logging.info(f"完成LLM提取。找到 {len(entities_set)} 个唯一实体和 {relation_count} 个唯一关系。")
    if enriched_relation_count > 0:
        logging.info(f"从关系提取中获取了 {enriched_relation_count} 个额外的实体类型信息。")
    
    # 记录具有多种类型的实体
    ambiguous_entities = {name: types for name, types in relation_type_info.items() if len(types) > 1}
    if ambiguous_entities:
        for entity, types in ambiguous_entities.items():
            logging.warning(f"实体 '{entity}' 在不同关系中被赋予了多种类型: {', '.join(types)}")
        logging.warning(f"发现 {len(ambiguous_entities)} 个具有类型歧义的实体。在关系处理中将优先使用更具体的类型。")
    
    return entities_set, relations_set

def add_structural_relations(data: List[Dict[str, Any]], entities_set: Set[Tuple[str, str, str]], 
                            relations_set: Set[Tuple[str, str, str, str]]) -> int:
    """
    添加文档结构关系，确保所有实体都与其源文档或章节相连
    
    1. 添加文档结构关系（章节隶属关系）
    2. 为所有孤立实体添加CONTAINS关系，确保可从文档结构遍历所有实体
    
    Args:
        data: 包含多个文本块的列表
        entities_set: 实体集合，会被此函数修改
        relations_set: 关系集合，会被此函数修改
        
    Returns:
        添加的结构关系数量
    """
    logging.info("添加结构化关系...")
    chunk_map: Dict[str, Dict[str, Any]] = {chunk['chunk_id']: chunk for chunk in data}
    
    structural_relations_added = 0
    
    # 第一步：添加已有的文档结构隶属关系
    for chunk in data:
        chunk_id = chunk.get("chunk_id")
        parent_id = chunk.get("parent_id")
        heading = chunk.get("heading")
        
        if not chunk_id:
            continue
            
        if parent_id and heading:
            parent_chunk = chunk_map.get(parent_id)
            if parent_chunk and parent_chunk.get("heading"):
                child_entity_name = normalize_entity_name(heading)
                parent_entity_name = normalize_entity_name(parent_chunk["heading"])
                
                if child_entity_name and parent_entity_name:
                    child_type = "章节"  # 默认假设非根标题为章节
                    parent_type = "章节" if parent_chunk.get("parent_id") else "文档"
                    
                    # 规范化实体类型
                    normalized_child_type = normalize_entity_type(child_type)
                    normalized_parent_type = normalize_entity_type(parent_type)
                    
                    # 添加实体，包含chunk_id
                    entities_set.add((child_entity_name, normalized_child_type, chunk_id))
                    entities_set.add((parent_entity_name, normalized_parent_type, parent_id))
                    
                    # 添加关系，包含chunk_id
                    relation_tuple = (child_entity_name, parent_entity_name, "隶属关系", chunk_id)
                    if relation_tuple not in relations_set:
                        relations_set.add(relation_tuple)
                        structural_relations_added += 1
    
    # 新增步骤：明确添加Document和Section之间的HAS_SECTION关系
    logging.info("添加文档和章节之间的HAS_SECTION关系...")
    documents = set()
    top_level_sections = set()
    all_sections = set()  # 记录所有章节实体
    document_section_added = 0
    
    # 识别所有Document实体、顶级Section实体和所有Section实体
    for name, entity_type, chunk_id in entities_set:
        normalized_type = normalize_entity_type(entity_type)
        
        if normalized_type.lower() == "document":
            documents.add((name, chunk_id))
        
        # 记录所有章节实体
        if normalized_type.lower() == "section":
            all_sections.add((name, normalized_type, chunk_id))
        
        # 找出顶级章节（通过parent_id指向文档的章节）
        if normalized_type.lower() == "section" and chunk_id in chunk_map:
            parent_id = chunk_map[chunk_id].get("parent_id")
            if parent_id and parent_id in chunk_map:
                parent_entity_type = None
                # 查找父级实体的类型
                for parent_name, parent_type, parent_chunk_id in entities_set:
                    if parent_chunk_id == parent_id:
                        normalized_parent_type = normalize_entity_type(parent_type)
                        parent_entity_type = normalized_parent_type
                        break
                        
                # 如果父级是文档，则此章节是顶级章节
                if parent_entity_type and parent_entity_type.lower() == "document":
                    top_level_sections.add((name, chunk_id))
    
    # 添加Document到顶级Section的HAS_SECTION关系
    relation_type_map = RELATION_TYPE_MAP_CYPHER
    has_section_type = relation_type_map.get("HAS_SECTION", "HAS_SECTION")
    
    # 首先，为每个文档明确查找和关联其顶级章节
    for doc_name, doc_chunk_id in documents:
        for section_name, section_chunk_id in top_level_sections:
            # 检查章节的父级是否为当前文档
            if section_chunk_id in chunk_map and chunk_map[section_chunk_id].get("parent_id") == doc_chunk_id:
                has_section_relation = (doc_name, section_name, has_section_type, doc_chunk_id)
                if has_section_relation not in relations_set:
                    relations_set.add(has_section_relation)
                    document_section_added += 1
    
    # 如果通过父级链接没有找到足够的关系，尝试添加所有文档到所有顶级章节的关系
    if document_section_added == 0 and documents and top_level_sections:
        logging.info("未找到明确的文档-章节关系，尝试添加所有文档到顶级章节的关系...")
        for doc_name, doc_chunk_id in documents:
            for section_name, section_chunk_id in top_level_sections:
                has_section_relation = (doc_name, section_name, has_section_type, doc_chunk_id)
                if has_section_relation not in relations_set:
                    relations_set.add(has_section_relation)
                    document_section_added += 1
    
    # 确保所有Section实体都有关系
    # 如果没有Document实体，则创建一个默认的Document实体
    default_doc_name = "默认文档"
    default_doc_chunk_id = "default_document"
    
    # 检查Section孤立实体，确保每个Section都有与Document的关系
    isolated_sections = []
    for section_name, section_type, section_chunk_id in all_sections:
        # 检查此Section是否有任何关系
        section_has_relation = False
        for src, tgt, rel_type, rel_chunk_id in relations_set:
            if (src == section_name or tgt == section_name) and rel_chunk_id == section_chunk_id:
                section_has_relation = True
                break
        
        if not section_has_relation:
            isolated_sections.append((section_name, section_type, section_chunk_id))
    
    # 如果有孤立的Section，添加它们到现有文档或创建默认文档
    if isolated_sections:
        logging.info(f"发现 {len(isolated_sections)} 个孤立的Section实体，将添加关系...")
        isolated_section_count = 0
        
        # 如果有文档实体，则使用第一个文档
        if documents:
            doc_name, doc_chunk_id = next(iter(documents))
            for section_name, section_type, section_chunk_id in isolated_sections:
                # 添加HAS_SECTION关系
                has_section_relation = (doc_name, section_name, has_section_type, doc_chunk_id)
                if has_section_relation not in relations_set:
                    relations_set.add(has_section_relation)
                    isolated_section_count += 1
        else:
            # 没有文档实体，创建默认文档
            logging.info(f"没有找到文档实体，创建默认文档并关联孤立Section...")
            # 规范化文档类型
            normalized_doc_type = normalize_entity_type("文档")
            entities_set.add((default_doc_name, normalized_doc_type, default_doc_chunk_id))
            
            for section_name, section_type, section_chunk_id in isolated_sections:
                # 添加HAS_SECTION关系
                has_section_relation = (default_doc_name, section_name, has_section_type, default_doc_chunk_id)
                if has_section_relation not in relations_set:
                    relations_set.add(has_section_relation)
                    isolated_section_count += 1
        
        logging.info(f"为 {isolated_section_count} 个孤立Section添加了关系")
        document_section_added += isolated_section_count
    
    logging.info(f"添加了 {structural_relations_added} 个结构化隶属关系")
    logging.info(f"添加了 {document_section_added} 个Document-Section HAS_SECTION关系")
    
    # 第二步：确保所有实体都与其源文档或章节有CONTAINS关系
    # 先根据chunk_id创建实体到章节的映射
    chunk_to_section_map = {}  # 记录chunk_id -> (section_name, section_type)
    section_entities = set()   # 记录所有章节和文档实体
    
    # 找出所有章节和文档实体
    for name, entity_type, chunk_id in entities_set:
        # 考虑所有可能的Section和Document类型变体
        if entity_type.lower() in ["section", "章节", "document", "文档"]:
            section_entities.add((name, entity_type, chunk_id))
            # 每个章节映射到自己
            chunk_to_section_map[chunk_id] = (name, entity_type)
    
    # 获取实体类型映射
    entity_type_map = ENTITY_TYPE_MAP_CYPHER
    relation_type_map = RELATION_TYPE_MAP_CYPHER

    # 为每个非章节非文档实体添加与其源chunk的章节的CONTAINS关系
    contains_added = 0
    
    # 创建关系集合的副本用于遍历，避免在遍历时修改集合
    relations_list = list(relations_set)
    
    # 记录已经添加了CONTAINS关系的实体-章节对
    processed_entity_section_pairs = set()

    for name, entity_type, chunk_id in entities_set:
        # 判断是否为章节或文档，考虑所有可能的变体
        if entity_type.lower() not in ["section", "章节", "document", "文档"]:
            # 每个实体必须与其所在的章节建立CONTAINS关系，即使实体名称相同但在不同chunk中
            entity_section_key = (name, chunk_id)
            
            if entity_section_key in processed_entity_section_pairs:
                continue
                
            # 检查此实体是否已与其章节建立了关系
            entity_has_section_relation = False
            
            # 获取Cypher中使用的实体类型标签
            cypher_entity_type = entity_type_map.get(entity_type, entity_type)
            
            # 检查是否已存在链接到章节的关系
            for src, tgt, rel_type, rel_chunk_id in relations_list:
                # 仅检查当前实体的关系
                if tgt == name and chunk_id == rel_chunk_id:
                    # 检查源是否为章节/文档
                    src_is_section = False
                    for sect_name, sect_type, sect_chunk_id in section_entities:
                        if src == sect_name:
                            src_is_section = True
                            break
                    
                    # 如果源是章节且关系类型已经是CONTAINS，则已经有关系
                    if src_is_section and rel_type.upper() == "CONTAINS":
                        entity_has_section_relation = True
                        processed_entity_section_pairs.add(entity_section_key)
                        break
            
            # 如果没有章节关系，则添加到其源chunk的章节
            if not entity_has_section_relation:
                # 找到实体所在chunk的章节
                section = chunk_to_section_map.get(chunk_id)
                
                if section:
                    section_name, section_type = section
                    # 添加CONTAINS关系
                    cypher_section_type = entity_type_map.get(section_type, section_type)
                    # 使用CONTAINS的Cypher映射
                    contains_rel_type = relation_type_map.get("CONTAINS", "CONTAINS")
                    contains_relation = (section_name, name, contains_rel_type, chunk_id)
                    
                    if contains_relation not in relations_set:
                        relations_set.add(contains_relation)
                        contains_added += 1
                        processed_entity_section_pairs.add(entity_section_key)
                else:
                    # 如果找不到对应章节，则查找此chunk的父chunk
                    if chunk_id in chunk_map:
                        parent_id = chunk_map[chunk_id].get("parent_id")
                        if parent_id and parent_id in chunk_to_section_map:
                            section_name, section_type = chunk_to_section_map[parent_id]
                            # 添加CONTAINS关系
                            cypher_section_type = entity_type_map.get(section_type, section_type)
                            # 使用CONTAINS的Cypher映射
                            contains_rel_type = relation_type_map.get("CONTAINS", "CONTAINS")
                            contains_relation = (section_name, name, contains_rel_type, chunk_id)
                            
                            if contains_relation not in relations_set:
                                relations_set.add(contains_relation)
                                contains_added += 1
                                processed_entity_section_pairs.add(entity_section_key)
                        else:
                            # 如果找不到父chunk，使用任何可用的章节/文档
                            # 这是确保没有孤立实体的最后手段
                            if section_entities:
                                # 使用第一个可用的文档
                                doc_found = False
                                for sect_name, sect_type, sect_chunk_id in section_entities:
                                    if sect_type.lower() in ["document", "文档"]:
                                        cypher_section_type = entity_type_map.get(sect_type, sect_type)
                                        # 使用CONTAINS的Cypher映射
                                        contains_rel_type = relation_type_map.get("CONTAINS", "CONTAINS")
                                        contains_relation = (sect_name, name, contains_rel_type, chunk_id)
                                        
                                        if contains_relation not in relations_set:
                                            relations_set.add(contains_relation)
                                            contains_added += 1
                                            processed_entity_section_pairs.add(entity_section_key)
                                            doc_found = True
                                            break
                                
                                # 如果没有找到文档，使用任何章节
                                if not doc_found:
                                    # 取第一个章节
                                    for sect_name, sect_type, sect_chunk_id in section_entities:
                                        cypher_section_type = entity_type_map.get(sect_type, sect_type)
                                        # 使用CONTAINS的Cypher映射
                                        contains_rel_type = relation_type_map.get("CONTAINS", "CONTAINS")
                                        contains_relation = (sect_name, name, contains_rel_type, chunk_id)
                                        
                                        if contains_relation not in relations_set:
                                            relations_set.add(contains_relation)
                                            contains_added += 1
                                            processed_entity_section_pairs.add(entity_section_key)
                                            break
    
    logging.info(f"添加了 {structural_relations_added} 个结构化隶属关系")
    logging.info(f"添加了 {document_section_added} 个Document-Section HAS_SECTION关系")
    logging.info(f"添加了 {contains_added} 个CONTAINS关系，确保实体可通过文档结构访问")
    
    # 最终检查：确保没有孤立实体
    all_entities_with_relations = set()
    for src, tgt, rel_type, rel_chunk_id in relations_set:
        # 找到源实体和目标实体
        for name, entity_type, chunk_id in entities_set:
            if name == src and chunk_id == rel_chunk_id:
                all_entities_with_relations.add((name, entity_type, chunk_id))
            if name == tgt and chunk_id == rel_chunk_id:
                all_entities_with_relations.add((name, entity_type, chunk_id))
    
    # 找出所有仍然孤立的实体
    isolated_entities = entities_set - all_entities_with_relations
    
    if isolated_entities:
        logging.warning(f"发现 {len(isolated_entities)} 个仍然孤立的实体，将添加默认关系")
        final_fixes = 0
        
        # 获取或创建一个根文档节点
        root_doc_entity = None
        for name, entity_type, chunk_id in entities_set:
            if entity_type.lower() in ["document", "文档"]:
                root_doc_entity = (name, entity_type, chunk_id)
                break
        
        # 如果没有文档实体，创建一个默认文档
        if root_doc_entity is None:
            default_doc_name = "默认文档"
            default_doc_chunk_id = "default_document"
            root_doc_entity = (default_doc_name, "文档", default_doc_chunk_id)
            entities_set.add(root_doc_entity)
            logging.info(f"创建默认文档实体: {default_doc_name}")
        
        # 为每个孤立实体添加到根文档的CONTAINS关系
        contains_rel_type = relation_type_map.get("CONTAINS", "CONTAINS")
        for name, entity_type, chunk_id in isolated_entities:
            contains_relation = (root_doc_entity[0], name, contains_rel_type, chunk_id)
            
            if contains_relation not in relations_set:
                relations_set.add(contains_relation)
                final_fixes += 1
        
        logging.info(f"为 {final_fixes} 个孤立实体添加了与根文档的CONTAINS关系")
        contains_added += final_fixes
    
    return structural_relations_added + document_section_added + contains_added

def print_extraction_results(entities_set: Set[Tuple[str, str, str]], relations_set: Set[Tuple[str, str, str, str]]):
    """
    在控制台打印提取结果
    
    Args:
        entities_set: 实体集合 (name, type, chunk_id)
        relations_set: 关系集合 (source, target, type, chunk_id)
    """
    print("\n--- Final Extracted Entities ---")
    # 按照实体类型和名称排序
    sorted_entities = sorted(list(entities_set), key=lambda x: (x[1], x[0]))  # 按类型和名称排序
    
    for name, entity_type, chunk_id in sorted_entities:
        print(f"- ({entity_type}) {name} [chunk_id: {chunk_id}]")
    
    print(f"\nTotal Unique Entities: {len(sorted_entities)}")

    print("\n--- Final Extracted Relations ---")
    # 按照关系类型和源实体名称排序
    sorted_relations = sorted(list(relations_set), key=lambda x: (x[2], x[0]))  # 按关系类型和源实体排序
    
    for source, target, relation_type, chunk_id in sorted_relations:
        print(f"- {source} --[{relation_type}]--> {target} [chunk_id: {chunk_id}]")
    
    print(f"\nTotal Unique Relations: {len(sorted_relations)}")

def save_cypher_statements(entities_set: Set[Tuple[str, str, str]], relations_set: Set[Tuple[str, str, str, str]], 
                          output_path: str):
    """
    生成并保存Cypher语句
    
    Args:
        entities_set: 实体集合 (name, type, chunk_id)
        relations_set: 关系集合 (source, target, type, chunk_id)
        output_path: 输出文件路径
    """
    print(f"\n--- Generating Cypher Statements (Memgraph/Neo4j) ---")
    cypher_statements = generate_cypher_statements(entities_set, relations_set)

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(";\n".join(cypher_statements) + ";\n")  # 每个语句后添加分号
        print(f"\nCypher statements saved to: {output_path}")
    except IOError as e:
        print(f"\nError writing Cypher statements to file {output_path}: {e}")
        print("\nCypher Statements:\n")
        print(";\n".join(cypher_statements) + ";\n")  # 作为备选方案打印到控制台

def create_example_config_file(config_path: str) -> bool:
    """创建示例配置文件，包含所有图谱模式定义"""
    example_content = """# LightRAG 知识图谱提取配置文件

# 图谱模式定义
schema:
  # 实体类型定义
  entity_types:
    - 文档
    - 章节
    - 主题
    - 关键词
    - 人员
    - 角色
    - 组织
    - 时间
    - 事件
    - 法规

  # 实体类型映射到Cypher中的标签
  entity_type_map_cypher:
    文档: Document
    章节: Section
    主题: Topic
    关键词: Keyword
    人员: Person
    角色: Role
    组织: Organization
    时间: Time
    事件: Event
    法规: Regulation

  # 关系类型定义
  relation_types:
    - 隶属关系
    - 版本关系
    - 引用
    - 依据
    - 责任
    - 审批
    - 时间
    - 生效
    - 关联

  # 关系类型映射到Cypher中的关系类型
  relation_type_map_cypher:
    隶属关系: BELONGS_TO
    版本关系: HAS_VERSION
    引用: REFERENCES
    依据: BASED_ON
    责任: RESPONSIBLE_FOR
    审批: APPROVED_BY
    时间: OCCURS_AT
    生效: EFFECTIVE_FROM
    关联: RELATED_TO

# 实体名称规范化映射
normalization:
  canonical_map:
    客运部: 集团公司客运部
    信息技术所: 集团公司信息技术所
    科信部: 集团公司科信部
    财务部: 集团公司财务部
    计统部: 集团公司计统部
    电务部: 集团公司电务部
    供电部: 集团公司供电部
    宣传部: 集团公司宣传部
    调度所: 集团公司调度所
    集团公司应急领导小组办公室: 集团公司应急领导小组办公室
    集团公司应急领导小组: 集团公司应急领导小组
    国铁集团应急领导小组办公室: 国铁集团应急领导小组办公室
    国铁集团应急领导小组: 国铁集团应急领导小组
    国铁集团客运部: 国铁集团客运部
    12306科创中心: 12306科创中心
    广铁集团: 中国铁路广州局集团有限公司
    集团公司: 中国铁路广州局集团有限公司
    本预案: 《广州局集团公司客票发售和预订系统（含互联网售票部分）应急预案》
    客票系统: 客票发售和预订系统

# 提示词模板
prompts:
  entity_extraction:
    definitions: |
      实体类型定义:
      - 文档：管理规定的文件名称，如《应急预案》。
      - 章节：文档中的具体章节标题，如"1 总则"。
      - 主题：文档或章节的核心议题，如"应急组织机构"。
      - 关键词：文本中重要的名词或术语，如"客票系统"、"应急响应"、"电子客票"。
      - 人员：具体的人名（此文档中可能较少）。
      - 角色：指代具有特定职责的职位或岗位，如"客运部主任"、"售票员"。
      - 组织：涉及的单位、部门或公司，如"中国铁路广州局集团有限公司"、"集团公司客运部"、"信息技术所"、"各车务站段"。
      - 时间：具体的日期、时间点或时间段，如"2021年"、"4小时及以上"、"每年3月"。
      - 事件：文档中描述的具体活动或状况，如"系统突发事件"、"启动应急预案"、"应急演练"、"售票故障"。
      - 法规：引用的其他法规或文件名称及其编号，如"《铁路客票发售和预订系统(含互联网售票部分)应急预案》（铁办客〔2021〕92号）"。

    template: |
      请从以下文本中提取定义的实体类型。尽可能完整提取每个实体，确保不遗漏文本中的关键实体。

      {definitions}

      预定义的实体类型列表: {entity_types}

      文本：
      \"\"\"
{content}
      \"\"\"

      请以严格的 JSON 格式输出，包含一个名为 "entities" 的列表，其中每个对象包含 "name" (实体名称) 和 "type" (实体类型)。确保实体名称是文本中实际出现的词语。

      注意事项：
      1. 确保每个实体完整识别，尤其是组织、文档和法规名称不要截断
      2. 识别实体时考虑可能存在的缩写和全称（如"集团公司"与"中国铁路广州局集团有限公司"）
      3. 对于相同实体的不同表述，保留所有出现形式
      4. 确保每个实体都准确分配了正确的实体类型

      例如:
      {
        "entities": [
          {"name": "集团公司客运部", "type": "组织"},
          {"name": "售票故障", "type": "事件"},
          {"name": "《铁路客票发售和预订系统(含互联网售票部分)应急预案》（铁办客〔2021〕92号）", "type": "法规"}
        ]
      }

  relation_extraction:
    definitions: |
      关系类型定义 (请仅提取文本段落内明确描述的关系):
      - 隶属关系 (BelongsTo): 通常是结构化的，此提示词主要关注文本内描述，如"办公室设在客运部"。(结构化部分将后处理)
      - 版本关系 (HasVersion): 指明文档的版本信息或与其他版本的关系 (如"修订版"、"废止旧版")。
      - 引用 (References): 一个实体提到了另一个实体或文件，如"详见附件5"。
      - 依据 (BasedOn): 指出制定某文件或采取某行动所依据的法规或原则，如"根据...制定本预案"。
      - 责任 (ResponsibleFor): 指明某个角色或组织负责某项任务或职责，如"客运部负责协调"。
      - 审批 (ApprovedBy): 指出某事项需要经过哪个组织或角色批准，如"经...同意后"。
      - 时间 (OccursAt): 事件发生的时间，或规定适用的时间点/段，如"事件影响4小时"、"每年3月开展演练"。
      - 生效 (EffectiveFrom): 规定或文件的生效日期，如"自发布之日起实施"。
      - 关联 (RelatedTo): 实体间的其他关联，如"与...不一致时，以此为准"。

    template: |
      请从以下文本中提取实体之间的关系。请专注于在文本段落中**直接陈述**的关系。

      {definitions}

      预定义的关系类型列表: {relation_types}
      预定义的实体类型列表: {entity_types}

      文本：
      \"\"\"
{content}
      \"\"\"

      请以严格的 JSON 格式输出，包含一个名为 "relations" 的列表，其中每个对象必须包含以下字段:
      - "source" (源实体名称)
      - "source_type" (源实体类型，必须是预定义的实体类型之一)
      - "target" (目标实体名称)
      - "target_type" (目标实体类型，必须是预定义的实体类型之一)
      - "type" (关系类型，必须是预定义的关系类型之一)

      确保实体名称是文本中实际出现的词语，并且为每个实体提供正确的类型。
      注意事项:
      1. 每个关系必须同时包含source_type和target_type
      2. 实体类型必须从预定义列表中选择
      3. 确保实体名称完整，不要截断组织、文档或法规名称

      例如:
      {
        "relations": [
          {
            "source": "集团公司应急领导小组办公室", 
            "source_type": "组织",
            "target": "集团公司客运部", 
            "target_type": "组织",
            "type": "隶属关系"
          },
          {
            "source": "本预案", 
            "source_type": "文档",
            "target": "《铁路客票发售和预订系统(含互联网售票部分)应急预案》", 
            "target_type": "法规",
            "type": "依据"
          },
          {
            "source": "客运部", 
            "source_type": "组织",
            "target": "协调各相关部门", 
            "target_type": "责任",
            "type": "责任"
          }
        ]
      }

# 图数据库配置
database:
  # 是否启用唯一性约束（推荐在数据库初始化时启用）
  enable_uniqueness_constraints: true
  # 是否为关系添加额外的元数据（创建时间、可靠性等）
  enable_relation_metadata: true
"""
    try:
        if not os.path.exists(config_path):
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(example_content)
            logging.info(f"已创建示例配置文件: {config_path}")
            return True
        return False
    except Exception as e:
        logging.warning(f"创建示例配置文件失败: {e}")
        return False

def main(input_file: str, output_file: str, config_file: str):
    """
    Main function to orchestrate the KG extraction process.
    
    Args:
        input_file: Input JSON file path (required)
        output_file: Output Cypher file path (required)
        config_file: Config YAML file path (required)
    """
    # 主函数：协调加载配置、读取数据、抽取实体和关系、构建结构化关系以及生成和保存 Cypher 语句
    # 确保所有参数都已提供
    if not input_file or not output_file or not config_file:
        logging.error("Error: 必须提供输入文件、输出文件和配置文件路径")
        print("用法: python entity_extract.py -i INPUT_FILE -o OUTPUT_FILE -c CONFIG_FILE")
        return
    
    # 获取绝对路径
    input_path = os.path.abspath(input_file)
    output_path = os.path.abspath(output_file)
    config_path = os.path.abspath(config_file)
    
    # 检查API密钥是否设置
    if not LLM_API_KEY:
        logging.error("环境变量 LLM_BINDING_API_KEY 或 SILICONFLOW_API_KEY 未设置，API调用将会失败")
        logging.error("请设置环境变量: export LLM_BINDING_API_KEY='your_api_key_here'")
        return
    
    # 检查API主机和模型是否设置
    if not LLM_API_HOST:
        logging.error("环境变量 LLM_BINDING_HOST 未设置")
        logging.error("请设置环境变量: export LLM_BINDING_HOST='https://api.siliconflow.cn/v1'")
        return
        
    if not LLM_MODEL:
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
    
    # 加载配置
    try:
        load_config_and_setup_globals(config_path)
    except Exception as e:
        logging.error(f"加载配置失败: {e}")
        logging.exception("详细错误信息:")
        return
    
    # 加载数据
    data = load_json_data(input_path)
    if not data:
        logging.error(f"无法加载数据，程序退出")
        return

    # 限制处理的数据为前10个文本分块
    # test_data = data[:10]
    test_data = data
    logging.info(f"为了快速测试，仅处理前 {len(test_data)} 个文本分块（共 {len(data)} 个）")

    # --- Phase 1: 并发LLM抽取实体和关系 ---
    logging.info(f"准备并发处理 {len(test_data)} 个文本块的实体和关系提取...")
    try:
        entities_set, relations_set = asyncio.run(extract_entities_and_relations(test_data))
    except KeyboardInterrupt:
        logging.info("\n任务被用户中断。正在清理...")
        return
    except Exception as e:
        logging.error(f"处理过程中发生错误: {e}")
        logging.exception("详细错误信息，请检查以下堆栈跟踪:")
        # 尝试获取更多信息
        if hasattr(e, '__traceback__'):
            import traceback
            stack_trace = ''.join(traceback.format_tb(e.__traceback__))
            logging.error(f"错误堆栈跟踪:\n{stack_trace}")
        
        # 如果可能，尝试打印原因链
        if hasattr(e, '__cause__') and e.__cause__:
            logging.error(f"原因: {e.__cause__}")
        
        # 如果需要，还可以尝试通过诊断脚本验证各种组件
        logging.info("您可以尝试运行单独的脚本检查以下内容:")
        logging.info("1. 检查 JSON 文件格式: python -m json.tool <your_json_file>")
        logging.info("2. 检查 YAML 配置: python -c 'import yaml; yaml.safe_load(open(\"<your_yaml_file>\"))'")
        logging.info("3. 检查 API 连接: curl -H \"Authorization: Bearer $LLM_BINDING_API_KEY\" $LLM_BINDING_HOST/models")
        
        return
            
    # --- Phase 2: 添加文档结构关系 ---
    logging.info("添加文档结构化关系...")
    
    # 确保每个实体都与其来源文档/章节建立连接
    added_relations_count = add_structural_relations(test_data, entities_set, relations_set)
    logging.info(f"已添加 {added_relations_count} 个结构化关系，确保实体可以从文档结构中遍历访问")
    
    # --- Phase 3: 生成和保存Cypher语句 ---
    logging.info("正在生成Cypher语句...")
    
    # --- 输出结果 ---
    # 1. 控制台输出
    print_extraction_results(entities_set, relations_set)
    
    # 2. 生成并保存Cypher语句
    save_cypher_statements(entities_set, relations_set, output_path)
    
    print("\n--- 注意事项 ---")
    print("1. 所生成的Cypher语句包含唯一性约束创建语句（已被注释）")
    print("   首次导入数据到空数据库时，建议取消注释并执行这些约束语句")
    print("   后续导入时应保持这些语句被注释，否则会因约束已存在而报错")
    print("2. 实体关系类型不确定的情况下，关系已被标记为低可靠性")
    print("   您可以在图数据库中查询并找出这些关系：")
    print("   MATCH ()-[r]->() WHERE r.reliability = 'low' RETURN r")
    print("3. 所有实体和关系都包含以下属性：")
    print("   - chunk_id：用于追溯实体和关系的来源文本块，可在ES服务器中查找原始内容")
    print("   - entity_type：实体的类型属性，方便按类型查询和筛选实体")
    print("   - relation_type：关系的类型属性，方便按类型查询和筛选关系")
    print("   示例查询：")
    print("   MATCH (n) WHERE n.chunk_id = 'chunk_123' RETURN n")
    print("   MATCH (n)-[r]->(m) WHERE r.chunk_id = 'chunk_123' RETURN n, r, m")
    print("\n恭喜！知识图谱数据已成功提取并准备就绪。")
    print(f"Cypher语句已保存到: {output_path}")
    print("您可以将这些语句导入Neo4j或Memgraph等图数据库进行可视化和查询。")


if __name__ == "__main__":
    import argparse
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='知识图谱实体关系提取工具')
    parser.add_argument('-i', '--input', required=True, help='输入JSON文件路径 (必须)')
    parser.add_argument('-o', '--output', required=True, help='输出Cypher文件路径 (必须)')
    parser.add_argument('-c', '--config', required=True, help='配置YAML文件路径 (必须)')
    
    args = parser.parse_args()
    
    # 使用命令行参数调用main函数（所有参数都是必需的）
    main(input_file=args.input, output_file=args.output, config_file=args.config)

# README:
# 
# 本脚本用于从文本中提取实体和关系，构建知识图谱
# 
# 使用说明:
# 1. 设置必要的环境变量:
#    export LLM_BINDING_API_KEY="your_api_key_here"
#    export LLM_BINDING_HOST="https://api.siliconflow.cn/v1"
#    export LLM_MODEL="Qwen/Qwen2.5-14B-Instruct"
#
# 2. 准备配置文件:
#    创建YAML格式的配置文件，包含:
#    - schema: 实体类型和关系类型定义
#    - normalization: 实体名称规范化映射
#    - prompts: 提示词模板
#
# 3. 运行脚本:
#    python entity_extract.py -i INPUT_FILE -o OUTPUT_FILE -c CONFIG_FILE
#    所有参数都是必需的:
#    - INPUT_FILE: 输入JSON文件路径
#    - OUTPUT_FILE: 输出Cypher文件路径
#    - CONFIG_FILE: 配置YAML文件路径
#
# 4. 查看输出结果:
#    - 控制台将显示提取的实体和关系
#    - Cypher语句将被保存到指定的输出文件
#
# 依赖:
# - requests
# - httpx (用于异步HTTP请求)
# - PyYAML (配置文件解析)
