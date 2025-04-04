import json
import os
import logging
import time # For potential rate limiting
import requests  # 用于HTTP请求，调用LLM API
import yaml  # 用于加载配置文件
import asyncio  # 用于异步并发处理
import httpx  # 用于异步HTTP请求
import regex
import re
from typing import List, Dict, Set, Tuple, Optional, Any, NamedTuple, Union
from dataclasses import dataclass, field  # 用于定义数据类
from pathlib import Path  # 用于处理路径
import argparse
from datetime import datetime

# --- 数据类定义 ---

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

# 全局配置对象，在加载配置时设置
config = None

# --- LLM客户端类 ---
class LLMClient:
    """LLM API客户端，封装所有与API交互的细节"""
    
    def __init__(self, api_key: str, api_host: str, model: str, 
                max_concurrent: int = 5, request_delay: float = 0.2,
                timeout: float = 60.0, max_retries: int = 3):
        """
        初始化LLM客户端
        
        Args:
            api_key: API密钥
            api_host: API主机URL
            model: 使用的模型名称
            max_concurrent: 最大并发请求数
            request_delay: 请求之间的延迟（秒）
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
        """
        self.api_key = api_key
        self.api_host = api_host
        self.model = model
        self.max_concurrent = max_concurrent
        self.request_delay = request_delay
        self.timeout = timeout
        self.max_retries = max_retries
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_tasks(self, tasks: List[LLMTask]) -> List[LLMTask]:
        """异步并发处理多个LLM任务，并进行速率限制"""
        async def process_with_rate_limit(task):
            """使用速率限制处理单个任务"""
            try:
                await self.semaphore.acquire()
                logging.debug(f"开始处理任务: {task.chunk_id} ({task.prompt_type})")
                result = await self.call_llm(task)
                logging.debug(f"完成任务: {task.chunk_id} ({task.prompt_type}), 结果状态: {'成功' if result.result else '失败'}")
                return result
            except asyncio.CancelledError:
                logging.info(f"任务被取消: {task.chunk_id}")
                raise
            except Exception as e:
                logging.error(f"处理任务 {task.chunk_id} 时出错: {e}")
                logging.exception("详细错误信息:")
                return task
            finally:
                # 释放一个槽位并等待指定延迟
                await asyncio.sleep(self.request_delay)
                self.semaphore.release()
        
        try:
            # 创建任务列表
            tasks_list = []
            for i, task in enumerate(tasks):
                try:
                    logging.debug(f"处理任务 {i+1}/{len(tasks)}: {task.chunk_id} ({task.prompt_type})")
                    tasks_list.append(process_with_rate_limit(task))
                    if (i+1) % 10 == 0:
                        logging.info(f"已创建 {i+1}/{len(tasks)} 个任务")
                except Exception as e:
                    logging.error(f"创建任务 {i+1}/{len(tasks)} 时发生异常: {e}")
                    logging.exception("详细错误信息:")
            
            # 并发执行所有任务
            processed_tasks = await asyncio.gather(*tasks_list, return_exceptions=True)
            
            # 处理结果，确保任何异常都被适当处理
            results = []
            for i, result in enumerate(processed_tasks):
                if isinstance(result, Exception):
                    logging.error(f"任务 {i} 执行异常: {result}")
                    # 保留原任务，但不设置结果
                    results.append(tasks[i])
                else:
                    results.append(result)
            
            return results
        except asyncio.CancelledError:
            logging.info("正在取消所有任务...")
            raise
        except Exception as e:
            logging.error(f"处理任务时发生错误: {e}")
            logging.exception("详细错误信息:")
            raise
    
    async def call_llm(self, task: LLMTask) -> LLMTask:
        """异步调用LLM API进行推理"""
        logging.info(f"--- 发送Prompt到LLM ({task.prompt_type} - chunk {task.chunk_id}) ---")

        if not self.api_key:
            logging.error(f"LLM API密钥未设置，无法调用API (chunk {task.chunk_id})")
            return task
        
        # 重试计数器
        retry_count = 0
        
        while retry_count <= self.max_retries:
            try:
                # API URL和认证
                api_url = f"{self.api_host}/chat/completions"
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                # API请求参数
                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "你是一个实体关系提取助手，善于从文本中提取结构化信息并以JSON格式输出。"},
                        {"role": "user", "content": task.prompt}
                    ],
                    "max_tokens": 4096,
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
                        timeout=self.timeout
                    )
                
                if response.status_code == 200:
                    response_json = response.json()
                    # 解析OpenAI API返回的JSON结果
                    llm_response = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
                    logging.info(f"--- 成功接收LLM响应 (chunk {task.chunk_id}) ---")
                    
                    # 尝试解析响应
                    task.result = self._parse_llm_response(llm_response)
                    
                    # 检查是否成功解析结果
                    if task.result is None:
                        logging.warning(f"LLM响应无法解析为有效JSON (chunk {task.chunk_id})。将尝试重试。")
                        retry_count += 1
                        if retry_count <= self.max_retries:
                            logging.info(f"重试 {retry_count}/{self.max_retries}...")
                            await asyncio.sleep(self.request_delay * retry_count)  # 指数退避
                            continue
                        else:
                            logging.error(f"达到最大重试次数，放弃处理 chunk {task.chunk_id}")
                            break
                    
                    # 成功解析结果，返回
                    return task
                
                # 处理各种错误状态码
                elif response.status_code == 401:
                    logging.error(f"API认证失败: 无效的API密钥 (chunk {task.chunk_id})。请检查API密钥是否正确设置。")
                    break  # 认证失败，不重试
                elif response.status_code == 429:
                    logging.warning(f"API请求过于频繁 (chunk {task.chunk_id})。将尝试重试。")
                    retry_count += 1
                    if retry_count <= self.max_retries:
                        logging.info(f"重试 {retry_count}/{self.max_retries}...")
                        await asyncio.sleep(self.request_delay * retry_count * 2)  # 对于速率限制错误增加更长的延迟
                        continue
                    else:
                        logging.error(f"达到最大重试次数，放弃处理 chunk {task.chunk_id}")
                        break
                else:
                    logging.error(f"API调用失败 (chunk {task.chunk_id}): {response.status_code} - {response.text}")
                    retry_count += 1
                    if retry_count <= self.max_retries:
                        logging.info(f"重试 {retry_count}/{self.max_retries}...")
                        await asyncio.sleep(self.request_delay * retry_count)  # 指数退避
                        continue
                    else:
                        logging.error(f"达到最大重试次数，放弃处理 chunk {task.chunk_id}")
                        break
                
            except (httpx.TimeoutException, httpx.RequestError, json.JSONDecodeError) as e:
                error_type = type(e).__name__
                logging.error(f"{error_type} (chunk {task.chunk_id}): {e}")
                retry_count += 1
                if retry_count <= self.max_retries:
                    logging.info(f"重试 {retry_count}/{self.max_retries}...")
                    await asyncio.sleep(self.request_delay * retry_count)
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
                if retry_count <= self.max_retries:
                    logging.info(f"重试 {retry_count}/{self.max_retries}...")
                    await asyncio.sleep(self.request_delay * retry_count)
                    continue
                else:
                    logging.error(f"达到最大重试次数，放弃处理 chunk {task.chunk_id}")
                    break
        
        return task
    
    def _parse_llm_response(self, response_text: Optional[str]) -> Optional[Dict[str, List[Dict[str, str]]]]:
        """解析LLM的JSON响应"""
        if not response_text:
            return None
        
        try:
            # 首先尝试直接解析
            try:
                data = json.loads(response_text.strip())
                if isinstance(data, dict) and \
                   (('entities' in data and isinstance(data['entities'], list)) or \
                    ('relations' in data and isinstance(data['relations'], list))):
                    return data
            except json.JSONDecodeError:
                pass  # 继续使用更复杂的解析方法
            
            # 提取代码块中的内容
            if "```" in response_text:
                # 找到最大的代码块
                code_blocks = []
                start_pos = 0
                while True:
                    start_marker = response_text.find("```", start_pos)
                    if start_marker == -1:
                        break
                    end_marker = response_text.find("```", start_marker + 3)
                    if end_marker == -1:
                        break
                    code_blocks.append((start_marker, end_marker + 3, end_marker - start_marker))
                    start_pos = end_marker + 3
                
                if code_blocks:
                    # 获取最长的代码块
                    largest_block = max(code_blocks, key=lambda x: x[2])
                    start_content = response_text.find("\n", largest_block[0]) + 1
                    if start_content > 0 and start_content < largest_block[1]:
                        end_content = largest_block[1]
                        response_text = response_text[start_content:end_content].strip()
            
            # 查找JSON结构
            json_start = response_text.find('{')
            json_end = response_text.rfind('}')
            
            if json_start != -1 and json_end != -1:
                json_text = response_text[json_start:json_end + 1]
                
                # 尝试解析JSON
                try:
                    data = json.loads(json_text)
                    if isinstance(data, dict) and \
                       (('entities' in data and isinstance(data['entities'], list)) or \
                        ('relations' in data and isinstance(data['relations'], list))):
                        return data
                except:
                    pass  # 继续尝试其他方法
            
            # 查找entities或relations关键字
            entities_pos = response_text.find('"entities"')
            relations_pos = response_text.find('"relations"')
            
            if entities_pos != -1 or relations_pos != -1:
                # 使用正则表达式提取实体或关系部分
                entity_pattern = r'"entities"\s*:\s*\[(.*?)\]'
                relation_pattern = r'"relations"\s*:\s*\[(.*?)\]'
                
                if entities_pos != -1:
                    matches = re.search(entity_pattern, response_text, re.DOTALL)
                    if matches:
                        content = matches.group(1).strip()
                        if content:
                            fixed_json = f'{{"entities": [{content}]}}'
                            try:
                                return json.loads(fixed_json)
                            except:
                                pass
                
                if relations_pos != -1:
                    matches = re.search(relation_pattern, response_text, re.DOTALL)
                    if matches:
                        content = matches.group(1).strip()
                        if content:
                            fixed_json = f'{{"relations": [{content}]}}'
                            try:
                                return json.loads(fixed_json)
                            except:
                                pass
            
            logging.warning(f"无法从LLM响应中提取有效的JSON结构: {response_text[:200]}...")
            return None
            
        except Exception as e:
            logging.error(f"解析LLM响应时发生错误: {e}")
            logging.debug(f"问题响应: {response_text[:200]}...")
            return None

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
                return chunks
            
            # 如果数据本身就是一个数组，直接返回
            elif isinstance(data, list):
                logging.info(f"数据是列表格式，包含 {len(data)} 个项目")
                return data
            
            else:
                logging.error(f"Unexpected JSON format in {file_path}. Expected a list or an object with 'chunk' array.")
                logging.debug(f"JSON 结构: {type(data)}")
                return None
                
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


def normalize_entity_name(raw_name: str, config: Config) -> str:
    """
    规范化实体名称：
    1. 使用配置的CANONICAL_MAP规范化实体名称
    2. 去除名称开头的章节序号（如"4.4 "）
    3. 去除名称中的页码标签（如"(p.12)"）
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
    """规范化实体类型，确保使用配置中定义的标准类型。
    
    将尝试匹配中英文类型名称，确保使用ENTITY_TYPE_MAP_CYPHER中定义的标准类型。
    """
    if not isinstance(raw_type, str):
        logging.warning(f"Attempted to normalize non-string entity type: {raw_type}. Returning as is.")
        return str(raw_type)
    
    cleaned_type = raw_type.strip().replace('\n', ' ')
    
    # 1. 先检查是否已是有效的实体类型
    if cleaned_type in config.entity_type_map_cypher.values() or cleaned_type in config.entity_types_llm:
        # 如果是中文类型且存在于ENTITY_TYPE_MAP_CYPHER中，返回对应的英文类型
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
    """规范化关系类型，确保使用配置中定义的标准类型。
    
    将尝试匹配中英文类型名称，确保使用RELATION_TYPE_MAP_CYPHER中定义的标准类型。
    """
    if not isinstance(raw_type, str):
        logging.warning(f"Attempted to normalize non-string relation type: {raw_type}. Returning as is.")
        return str(raw_type)
    
    cleaned_type = raw_type.strip().replace('\n', ' ')
    
    # 1. 先检查是否已是有效的关系类型
    if cleaned_type in config.relation_type_map_cypher.values() or cleaned_type in config.relation_types_llm:
        # 如果是中文类型且存在于RELATION_TYPE_MAP_CYPHER中，返回对应的英文类型
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

def escape_cypher_string(value: str) -> str:
    """Escapes single quotes and backslashes for Cypher strings."""
    if not isinstance(value, str):
        return str(value) # Return as string if not already
    return value.replace('\\', '\\\\').replace("'", "\\'")

def parse_llm_response(response_text: Optional[str]) -> Optional[Dict[str, List[Dict[str, str]]]]:
    """安全地解析LLM的JSON响应，处理各种格式和边缘情况。"""
    if not response_text: 
        return None
    
    # 记录原始响应的前100个字符，用于调试
    logging.debug(f"解析LLM响应: {response_text[:100]}...")
    
    try:
        # 1. 清理Markdown代码块标记
        cleaned_text = response_text.strip()
        
        # 特别处理包含```json的代码块 (常见于大部分LLM响应)
        json_block_pattern = r"```(?:json)?\s*([\s\S]*?)```"
        json_blocks = re.findall(json_block_pattern, cleaned_text, re.DOTALL)
        
        if json_blocks:
            # 如果有多个代码块，选择最长的一个
            longest_block = max(json_blocks, key=len).strip()
            logging.debug(f"从Markdown代码块提取JSON: {longest_block[:50]}...")
            cleaned_text = longest_block
        
        # 2. 尝试直接解析清理后的文本
        try:
            parsed_data = json.loads(cleaned_text)
            if isinstance(parsed_data, dict) and \
               (('entities' in parsed_data and isinstance(parsed_data['entities'], list)) or \
                ('relations' in parsed_data and isinstance(parsed_data['relations'], list))):
                logging.debug("成功直接解析JSON响应")
                return parsed_data
        except json.JSONDecodeError as e:
            # 如果直接解析失败，继续使用更复杂的方法
            logging.debug(f"直接解析JSON失败: {e}，尝试修复...")
        
        # 3. 定位JSON对象的边界
        # 查找第一个左大括号和最后一个右大括号
        json_start = cleaned_text.find('{')
        json_end = cleaned_text.rfind('}')
        
        if json_start != -1 and json_end != -1 and json_end > json_start:
            # 提取潜在的JSON文本
            json_text = cleaned_text[json_start:json_end + 1]
            logging.debug(f"提取潜在JSON文本: {json_text[:50]}...")
            
            # 4. 尝试解析提取的JSON
            try:
                parsed_data = json.loads(json_text)
                if isinstance(parsed_data, dict) and \
                   (('entities' in parsed_data and isinstance(parsed_data['entities'], list)) or \
                    ('relations' in parsed_data and isinstance(parsed_data['relations'], list))):
                    logging.debug("成功解析提取的JSON文本")
                    return parsed_data
            except json.JSONDecodeError:
                # 如果仍然失败，尝试更高级的修复
                pass
            
            # 5. 尝试修复可能不完整的JSON
            # 查找实体或关系数组
            try:
                if '"entities"' in json_text:
                    entities_match = re.search(r'"entities"\s*:\s*\[(.*?)(?:\]|$)', json_text, re.DOTALL)
                    if entities_match:
                        entities_content = entities_match.group(1).strip()
                        # 检查最后一个对象是否完整
                        if entities_content.endswith(','):
                            entities_content = entities_content[:-1]  # 移除尾部逗号
                        
                        # 确保JSON数组内容有效
                        if not entities_content.endswith('}'):
                            # 查找最后一个完整的对象
                            last_complete_obj = entities_content.rfind('}')
                            if last_complete_obj != -1:
                                entities_content = entities_content[:last_complete_obj+1]
                        
                        # 构建完整的JSON字符串
                        fixed_json = f'{{"entities": [{entities_content}]}}'
                        try:
                            parsed_data = json.loads(fixed_json)
                            logging.debug(f"成功修复并解析entities JSON: {fixed_json[:50]}...")
                            return parsed_data
                        except json.JSONDecodeError as e:
                            logging.warning(f"修复entities JSON失败: {e}")
                
                elif '"relations"' in json_text:
                    relations_match = re.search(r'"relations"\s*:\s*\[(.*?)(?:\]|$)', json_text, re.DOTALL)
                    if relations_match:
                        relations_content = relations_match.group(1).strip()
                        # 检查最后一个对象是否完整
                        if relations_content.endswith(','):
                            relations_content = relations_content[:-1]  # 移除尾部逗号
                        
                        # 处理可能存在问题的关系内容
                        # 分析和提取每个完整的对象
                        objects = []
                        brace_count = 0
                        start_pos = 0
                        in_object = False
                        
                        for i, char in enumerate(relations_content):
                            if char == '{':
                                if brace_count == 0:
                                    start_pos = i
                                    in_object = True
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0 and in_object:
                                    # 找到一个完整的对象
                                    obj_str = relations_content[start_pos:i+1]
                                    try:
                                        # 尝试解析单个对象
                                        obj = json.loads(obj_str)
                                        # 检查对象是否包含所有必要的字段
                                        required_fields = ["source", "source_type", "target", "target_type", "type"]
                                        if all(field in obj for field in required_fields):
                                            objects.append(obj)
                                    except json.JSONDecodeError:
                                        pass  # 忽略无法解析的对象
                                    in_object = False
                        
                        if objects:
                            logging.debug(f"从关系内容中提取了 {len(objects)} 个完整的关系对象")
                            return {"relations": objects}
                            
                        # 旧的方法如果上面方法提取不了单个对象
                        # 构建完整的JSON字符串
                        fixed_json = f'{{"relations": [{relations_content}]}}'
                        try:
                            parsed_data = json.loads(fixed_json)
                            logging.debug(f"成功修复并解析relations JSON: {fixed_json[:50]}...")
                            return parsed_data
                        except json.JSONDecodeError as e:
                            logging.warning(f"修复relations JSON失败: {e}")
            except Exception as repair_error:
                logging.error(f"尝试修复JSON时出错: {repair_error}")
        
        # 6. 使用正则表达式直接提取实体或关系对象
        try:
            # 匹配实体对象
            entity_pattern = r'\{\s*"name"\s*:\s*"((?:\\.|[^"\\])*?)"\s*,\s*"type"\s*:\s*"((?:\\.|[^"\\])*?)"\s*\}'
            entities = re.findall(entity_pattern, cleaned_text)
            if entities:
                entity_objects = [{"name": name, "type": type_} for name, type_ in entities]
                logging.debug(f"通过正则表达式提取了 {len(entity_objects)} 个实体")
                return {"entities": entity_objects}
            
            # 匹配关系对象 - 使用非贪婪模式
            relation_pattern = r'\{\s*"source"\s*:\s*"((?:\\.|[^"\\])*?)"\s*,\s*"source_type"\s*:\s*"((?:\\.|[^"\\])*?)"\s*,\s*"target"\s*:\s*"((?:\\.|[^"\\])*?)"\s*,\s*"target_type"\s*:\s*"((?:\\.|[^"\\])*?)"\s*,\s*"type"\s*:\s*"((?:\\.|[^"\\])*?)"\s*\}'
            relations = re.findall(relation_pattern, cleaned_text)
            if relations:
                relation_objects = [
                    {
                        "source": source, 
                        "source_type": source_type, 
                        "target": target, 
                        "target_type": target_type, 
                        "type": rel_type
                    } 
                    for source, source_type, target, target_type, rel_type in relations
                ]
                logging.debug(f"通过正则表达式提取了 {len(relation_objects)} 个关系")
                return {"relations": relation_objects}
                
            # 尝试一种更灵活的关系模式，允许字段顺序不同
            # 找到所有的JSON对象
            obj_pattern = r'\{[^\{\}]+\}'
            objs = re.findall(obj_pattern, cleaned_text)
            relation_fields = ["source", "source_type", "target", "target_type", "type"]
            relations_list = []
            
            for obj in objs:
                # 检查是否包含所有关系字段
                if all(f'"{field}"' in obj for field in relation_fields):
                    # 提取每个字段
                    relation = {}
                    for field in relation_fields:
                        match = re.search(f'"{field}"\\s*:\\s*"((?:\\\\.|[^"\\\\])*?)"', obj)
                        if match:
                            relation[field] = match.group(1)
                    
                    # 只有包含所有字段才添加
                    if len(relation) == len(relation_fields):
                        relations_list.append(relation)
            
            if relations_list:
                logging.debug(f"通过灵活匹配提取了 {len(relations_list)} 个关系")
                return {"relations": relations_list}
                
        except Exception as regex_error:
            logging.error(f"使用正则表达式提取JSON时出错: {regex_error}")
        
        # 所有方法都失败，记录警告并输出完整响应
        logging.warning(f"无法从LLM响应中提取有效的JSON结构:\n{response_text}")
        return None
    except Exception as e:
        logging.error(f"解析LLM响应时发生未预期的错误: {e}")
        logging.error(f"问题响应完整内容:\n{response_text}")
        return None

def create_entity_prompt(chunk_content: str, context: Dict[str, Any], config: Config) -> str:
    """创建实体提取的提示词"""
    # 从配置中获取模板
    template = config.prompt_templates.get('entity_extraction', {}).get('template', '')
    definitions = config.prompt_templates.get('entity_extraction', {}).get('definitions', '')
    
    # 如果模板为空，使用默认模板
    if not template:
        logging.warning("配置中没有找到实体提取模板，使用默认模板")
        template = """请从以下文本中提取定义的实体类型。\n\n{definitions}\n\n要提取的实体类型列表: {entity_types}\n\n文本：\n\"\"\"\n{content}\n\"\"\"\n\n请以JSON格式输出，包含一个名为 "entities" 的列表，其中每个对象包含 "name" 和 "type"。"""
    
    if not definitions:
        logging.warning("配置中没有找到实体类型定义，使用空定义")
        definitions = ""
    
    # 获取实体类型列表
    entity_types = config.entity_types_llm
    entity_types_english = [config.entity_type_map_cypher.get(t, t) for t in entity_types]
    
    # 获取文档上下文信息
    document_title = context.get('document_title', "未知文档")
    current_heading = context.get('current_heading', "未知章节")
    section_path = context.get('section_path', "")
    parent_section_summary = context.get('parent_section_summary', "")
    
    # 强化对实体类型的约束
    strict_entity_type_guidance = """
【特别强调】：
1. 必须严格使用以下英文实体类型，不得使用其他类型：{entity_types_english}
2. 请勿将关系类型（如RESPONSIBLE_FOR, APPLIES_TO, HAS_PURPOSE等）错误地用作实体类型
3. 请确保每个提取的实体都分配了正确的实体类型
"""
    
    # 先使用安全的手工构建的提示词，完全避开模板格式化的问题
    prompt = f"""请从以下文本中提取定义的实体类型。专注于识别组织、角色、具体规定陈述、以及预定义的规定类型(Topic)。

{definitions}

要提取的实体类型列表: {', '.join(entity_types)}

文档信息 (仅供参考，不要提取为实体):
文档标题: {document_title}
当前章节: {current_heading}
章节路径: {section_path}
父章节内容摘要: {parent_section_summary}

文本：
\"\"\"
{chunk_content}
\"\"\"

【重要】：必须使用英文实体类型！返回的JSON中，实体类型必须为英文，且仅包含以下类型: {', '.join(entity_types_english)}

请以严格的JSON格式输出，包含一个名为"entities"的列表，其中每个对象包含"name"(实体名称)和"type"(实体类型)。确保实体名称是文本中实际出现的词语。

注意事项：
1. 不要提取文档标题或章节标题作为实体。
2. 确保每个实体完整识别。
3. 识别实体时考虑缩写和全称。
4. 保留相同实体的不同表述。
5. 确保每个实体都准确分配了正确的英文实体类型。
6. 严格使用以下英文类型名: {', '.join(entity_types_english)}
7. 类型区分指南：
   - Topic与Statement的区别：Topic是规定内容的分类标签（如"服务质量"），Statement是具体的规定本身（如"应使用规范用语"）。
   - Organization与Role的区别：Organization是部门/单位，Role是岗位/职责。
   - 文档名称、预案名称：带有《》括号的文件名不是Organization，如"《广州局集团公司网络安全事件应急预案》"不是Organization。
   - 信息系统：各种信息系统如"客票发售和预订系统"、"12306系统"等不是Organization，应考虑作为Statement或Topic提取（如果符合这些类型的定义）。
8. 利用文档上下文信息辅助判断。

{strict_entity_type_guidance.format(entity_types_english=', '.join(entity_types_english))}

JSON输出示例：
{{
  "entities": [
    {{"name": "集团公司客运部", "type": "Organization"}},
    {{"name": "应使用规范用语，保持微笑服务", "type": "Statement"}},
    {{"name": "服务质量", "type": "Topic"}},
    {{"name": "站务员", "type": "Role"}}
  ]
}}"""
    
    return prompt

def create_relation_prompt(chunk_content: str, entities_json: str, context: Dict[str, Any], config: Config) -> str:
    """创建关系提取的提示词"""
    # 从配置中获取模板
    template = config.prompt_templates.get('relation_extraction', {}).get('template', '')
    definitions = config.prompt_templates.get('relation_extraction', {}).get('definitions', '')
    
    # 如果模板为空，使用默认模板
    if not template:
        logging.warning("配置中没有找到关系提取模板，使用默认模板")
        template = """请从以下文本中提取定义的关系类型。\n\n{definitions}\n\n要提取的关系类型列表: {relation_types}\n\n文本：\n\"\"\"\n{content}\n\"\"\"\n\n已识别的实体列表：\n\"\"\"\n{entities_json}\n\"\"\"\n\n请以JSON格式输出，包含一个名为 "relations" 的列表，其中每个对象包含 "source", "source_type", "target", "target_type", "type"。"""
    
    if not definitions:
        logging.warning("配置中没有找到关系类型定义，使用空定义")
        definitions = ""
    
    # 获取关系类型列表
    relation_types = config.relation_types_llm
    relation_types_english = [config.relation_type_map_cypher.get(t, t) for t in relation_types]
    
    # 获取所有实体类型（包括Document和Section）
    all_entity_types = config.all_entity_types
    all_entity_types_english = [config.entity_type_map_cypher.get(t, t) for t in all_entity_types]
    
    # 获取文档上下文信息
    document_title = context.get('document_title', "未知文档")
    current_heading = context.get('current_heading', "未知章节")
    section_path = context.get('section_path', "")
    
    # 强化对关系类型的约束
    strict_relation_type_guidance = """
【特别强调】：
1. 必须严格使用以下英文关系类型，不得使用其他类型：{relation_types_english}
2. source_type和target_type必须严格使用以下英文实体类型：{all_entity_types_english}
3. 请勿将实体类型（如Organization, Statement, Topic等）错误地用作关系类型
4. 确保每个关系的source和target都是文本中实际提取的实体
"""
    
    # 直接使用安全的手工构建的提示词，完全避开模板格式化的问题
    prompt = f"""请从以下文本中提取定义的关系类型。根据预定义的实体列表提取这些实体之间符合定义的关系类型。请专注于在文本段落中直接陈述的语义关系。

{definitions}

要提取的关系类型列表: {', '.join(relation_types)}
预定义的实体类型列表 (用于关系端点): {', '.join(all_entity_types_english)}

文档信息 (仅供参考):
文档标题: {document_title}
当前章节: {current_heading}
章节路径: {section_path}

文本：
\"\"\"
{chunk_content}
\"\"\"

文本中已识别的语义实体列表 (用于构建关系):
\"\"\"
{entities_json}
\"\"\"

【重要】：必须使用英文实体类型和关系类型！

请以严格的JSON格式输出，包含一个名为"relations"的列表，其中每个对象必须包含"source", "source_type", "target", "target_type", "type"字段。

注意事项:
1. 关系必须连接上面语义实体列表中的实体，或者连接语义实体到已知的章节(Section)或文档(Document)（如果文本明确引用）。
2. source_type和target_type必须是以下英文类型之一: {', '.join(all_entity_types_english)}
3. 关系类型必须是预定义的英文类型之一: {', '.join(relation_types_english)}
4. 不要提取Document -> Section或Section -> Section的结构关系。
5. CONTAINS关系将由脚本自动处理（连接章节和其包含的语义实体），不要让LLM提取CONTAINS。
6. 关系类型使用指南：
   - RESPONSIBLE_FOR：描述对具体规则(Statement)或规则类型(Topic)的责任。
   - MENTIONS：用于连接章节(Section)和其明确提及的规定类型(Topic)。
   - REFERENCES: 用于文本中明确提到的对其他章节或文档的引用 (e.g., "详见章节 3.1")。不要用于连接非文档/章节的实体。例如，系统名称、业务流程等非章节/文档的引用应使用RELATED_TO关系。
   - RELATED_TO: 用于表达其他关系类型无法清晰表达的联系，包括提及系统、业务流程等非文档/章节的情况。

{strict_relation_type_guidance.format(
    relation_types_english=', '.join(relation_types_english),
    all_entity_types_english=', '.join(all_entity_types_english)
)}

JSON输出示例：
{{
  "relations": [
    {{
      "source": "集团公司客运部",
      "source_type": "Organization",
      "target": "服务质量",
      "target_type": "Topic",
      "type": "RESPONSIBLE_FOR"
    }},
    {{
      "source": "应使用规范用语，保持微笑服务",
      "source_type": "Statement",
      "target": "站务员",
      "target_type": "Role",
      "type": "APPLIES_TO"
    }}
  ]
}}"""
    
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

def generate_cypher_statements(entities: List[Entity], relations: List[Relation], config: Config) -> List[str]:
    """
    使用配置的映射生成Memgraph/Neo4j Cypher MERGE语句。
    
    现在支持实体的context_id和entity_type属性，关系的context_id和relation_type属性。
    
    Args:
        entities: Entity数据类对象列表
        relations: Relation数据类对象列表
        config: 配置对象
        
    Returns:
        Cypher语句列表
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
        # 添加唯一ID属性以提高唯一性识别能力，并添加chunk_id和entity_type属性
        cypher_statements.append(f"MERGE (n:`{entity.type}` {{name: '{escaped_name}'}}) "
                                 f"ON CREATE SET n.uuid = '{entity.type}_' + timestamp() + '_' + toString(rand()), "
                                 f"n.context_id = '{entity.context_id}', "
                                 f"n.entity_type = '{entity.type}';")

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
            doc_id = chunk.get("full_doc_id", "unknown_doc")
            heading = chunk.get("heading", "Unknown Section")
            
            if not content:
                logging.warning(f"Chunk {chunk_id}（索引 {i}）没有内容，跳过")
                continue
            
            # 创建上下文信息
            context = {
                "document_title": f"Document {doc_id}",
                "current_heading": heading,
                "section_path": "N/A",
                "parent_section_summary": ""
            }
            
            # 创建实体提取任务
            try:
                entity_prompt = create_entity_prompt(content, context, config)
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
            
            # 创建关系提取任务 - 空实体列表，稍后在处理时会填充
            try:
                empty_entities_json = json.dumps({"entities": []}, ensure_ascii=False)
                relation_prompt = create_relation_prompt(content, empty_entities_json, context, config)
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
                if raw_name and raw_type and raw_type in [config.entity_type_map_cypher.get(t, t) for t in config.entity_types_llm]:
                    normalized_name = normalize_entity_name(raw_name, config)
                    # 规范化实体类型
                    normalized_type = normalize_entity_type(raw_type, config)
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
            
            if raw_source and raw_target and raw_type and raw_type in [config.relation_type_map_cypher.get(t, t) for t in config.relation_types_llm]:
                normalized_source = normalize_entity_name(raw_source, config)
                normalized_target = normalize_entity_name(raw_target, config)
                
                # 规范化关系类型
                normalized_type = normalize_relation_type(raw_type, config)
                
                # 添加关系，包含chunk_id属性
                relations_set.add((normalized_source, normalized_target, normalized_type, task.chunk_id))
                relation_count += 1
                
                # 对各种关系类型进行特殊处理，确保关系两端的实体类型正确
                # REFERENCES关系的目标必须是Document或Section
                if normalized_type == 'REFERENCES' and target_type and target_type not in ['Document', 'Section']:
                    logging.warning(f"REFERENCES关系的目标实体类型应为Document或Section，但实际为 '{target_type}'，将其更正为Section类型")
                    target_type = "Section"
                
                # APPLIES_TO关系的目标必须是Organization或Role
                elif normalized_type == 'APPLIES_TO' and target_type and target_type not in ['Organization', 'Role']:
                    logging.warning(f"APPLIES_TO关系的目标实体类型应为Organization或Role，但实际为 '{target_type}'，将其更正为Organization类型")
                    target_type = "Organization"
                
                # RESPONSIBLE_FOR关系的源必须是Organization或Role
                elif normalized_type == 'RESPONSIBLE_FOR' and source_type and source_type not in ['Organization', 'Role']:
                    logging.warning(f"RESPONSIBLE_FOR关系的源实体类型应为Organization或Role，但实际为 '{source_type}'，将其更正为Organization类型")
                    source_type = "Organization"
                
                # BELONGS_TO关系的源和目标必须是Organization或Role
                elif normalized_type == 'BELONGS_TO':
                    if source_type and source_type not in ['Organization', 'Role']:
                        logging.warning(f"BELONGS_TO关系的源实体类型应为Organization或Role，但实际为 '{source_type}'，将其更正为Organization类型")
                        source_type = "Organization"
                    if target_type and target_type not in ['Organization', 'Role']:
                        logging.warning(f"BELONGS_TO关系的目标实体类型应为Organization或Role，但实际为 '{target_type}'，将其更正为Organization类型")
                        target_type = "Organization"
                
                # MENTIONS关系的源必须是Section，目标必须是Topic
                elif normalized_type == 'MENTIONS':
                    if source_type and source_type != 'Section':
                        logging.warning(f"MENTIONS关系的源实体类型应为Section，但实际为 '{source_type}'，将其更正为Section类型")
                        source_type = "Section"
                    if target_type and target_type != 'Topic':
                        logging.warning(f"MENTIONS关系的目标实体类型应为Topic，但实际为 '{target_type}'，将其更正为Topic类型")
                        target_type = "Topic"
                
                # HAS_PURPOSE关系的源和目标必须是Statement
                elif normalized_type == 'HAS_PURPOSE':
                    if source_type and source_type != 'Statement':
                        logging.warning(f"HAS_PURPOSE关系的源实体类型应为Statement，但实际为 '{source_type}'，将其更正为Statement类型")
                        source_type = "Statement"
                    if target_type and target_type != 'Statement':
                        logging.warning(f"HAS_PURPOSE关系的目标实体类型应为Statement，但实际为 '{target_type}'，将其更正为Statement类型")
                        target_type = "Statement"
                
                # 存储实体类型信息并添加到实体集合，包含chunk_id属性
                if source_type and source_type in [config.entity_type_map_cypher.get(t, t) for t in config.entity_types_llm]:
                    # 规范化源实体类型
                    normalized_source_type = normalize_entity_type(source_type, config)
                    # 使用关系中提供的类型丰富实体集合
                    entities_set.add((normalized_source, normalized_source_type, task.chunk_id))
                    # 记录类型信息，用于后续关系处理
                    if normalized_source not in relation_type_info:
                        relation_type_info[normalized_source] = set()
                    relation_type_info[normalized_source].add(normalized_source_type)
                    enriched_relation_count += 1
                
                if target_type and target_type in [config.entity_type_map_cypher.get(t, t) for t in config.entity_types_llm]:
                    # 规范化目标实体类型
                    normalized_target_type = normalize_entity_type(target_type, config)
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
    添加文档结构关系，优先使用结构化数据而非依赖LLM来创建文档结构
    
    1. 从结构化数据创建Document和Section节点
    2. 从结构化数据建立HAS_SECTION和HAS_PARENT_SECTION关系 
    3. 将LLM提取的实体与其源Section通过CONTAINS关系连接
    
    Args:
        data: 包含多个文本块的列表
        entities_set: 实体集合，会被此函数修改
        relations_set: 关系集合，会被此函数修改
        
    Returns:
        添加的结构关系数量
    """
    logging.info("添加结构化关系，优先使用结构化数据...")
    chunk_map: Dict[str, Dict[str, Any]] = {chunk['chunk_id']: chunk for chunk in data}
    
    # 获取实体类型和关系类型映射
    entity_type_map = config.entity_type_map_cypher
    relation_type_map = config.relation_type_map_cypher
    has_section_type = relation_type_map.get("HAS_SECTION", "HAS_SECTION")
    has_parent_section_type = relation_type_map.get("HAS_PARENT_SECTION", "HAS_PARENT_SECTION")
    contains_rel_type = relation_type_map.get("CONTAINS", "CONTAINS")
    
    # 计数器
    document_nodes_added = 0
    section_nodes_added = 0
    has_section_relations_added = 0
    has_parent_section_relations_added = 0
    contains_relations_added = 0
    
    # --- 步骤1: 直接从结构化数据创建Document和Section节点 ---
    document_entities = set()  # 存储文档实体 (name, chunk_id)
    section_entities = {}  # chunk_id -> (name, type, chunk_id)
    
    # 查找文档级节点和信息
    document_title = None
    document_chunk_id = None
    
    # 1.1 首先查找文档信息 (通常位于根节点或有full_doc_id的节点)
    for chunk in data:
        # 查找文档标题
        if 'full_doc_id' in chunk and chunk.get('full_doc_id') and 'heading' in chunk:
            document_title = chunk.get('heading')
            document_chunk_id = chunk.get('chunk_id')
            break  # 找到文档信息就跳出
    
    # 如果没有找到明确的文档信息，尝试使用没有parent_id的根节点
    if not document_title:
        for chunk in data:
            if not chunk.get('parent_id') and chunk.get('heading'):
                document_title = chunk.get('heading')
                document_chunk_id = chunk.get('chunk_id')
                break
    
    # 如果仍然没有找到文档信息，使用文件名或默认名称
    if not document_title:
        if 'file_path' in data[0]:
            # 从文件路径提取文件名
            file_path = data[0].get('file_path')
            document_title = os.path.basename(file_path) if file_path else "未知文档"
        else:
            document_title = "未知文档"
        
        document_chunk_id = "document_root"
    
    # 1.2 创建Document节点
    normalized_document_type = normalize_entity_type("Document", config)
    entities_set.add((document_title, normalized_document_type, document_chunk_id))
    document_entities.add((document_title, document_chunk_id))
    document_nodes_added += 1
    logging.info(f"从结构化数据创建文档实体: '{document_title}' [chunk_id: {document_chunk_id}]")
    
    # 1.3 从结构化数据创建Section节点
    for chunk in data:
        chunk_id = chunk.get('chunk_id')
        heading = chunk.get('heading')
        
        # 跳过非有效节点
        if not chunk_id or not heading:
            continue
        
        # 创建Section节点 (所有非文档根节点都视为章节)
        if chunk_id != document_chunk_id:
            normalized_section_type = normalize_entity_type("Section", config)
            normalized_heading = normalize_entity_name(heading, config)
            # 添加到实体集
            entities_set.add((normalized_heading, normalized_section_type, chunk_id))
            # 记录所有Section实体，用于后续创建关系
            section_entities[chunk_id] = (normalized_heading, normalized_section_type, chunk_id)
            section_nodes_added += 1
    
    logging.info(f"从结构化数据创建了 {section_nodes_added} 个章节实体")
    
    # --- 步骤2: 建立文档结构关系 (HAS_SECTION & HAS_PARENT_SECTION) ---
    
    # 2.1 建立Document -> Section (HAS_SECTION)关系
    for chunk in data:
        chunk_id = chunk.get('chunk_id')
        parent_id = chunk.get('parent_id')
        
        # 跳过非有效节点
        if chunk_id not in section_entities:
            continue
        
        section_name = section_entities[chunk_id][0]
        
        # 如果父级是文档或没有父级(表示顶级章节)
        if parent_id == document_chunk_id or not parent_id:
            # 为所有顶级章节添加HAS_SECTION关系
            has_section_relation = (document_title, section_name, has_section_type, document_chunk_id)
            if has_section_relation not in relations_set:
                relations_set.add(has_section_relation)
                has_section_relations_added += 1
    
    # 2.2 建立Section -> Section (HAS_PARENT_SECTION)关系
    for chunk in data:
        chunk_id = chunk.get('chunk_id')
        parent_id = chunk.get('parent_id')
        
        # 跳过非有效节点或没有父级的节点
        if chunk_id not in section_entities or not parent_id or parent_id not in section_entities:
            continue
        
        child_section_name = section_entities[chunk_id][0]
        parent_section_name = section_entities[parent_id][0]
        
        # 添加HAS_PARENT_SECTION关系
        has_parent_relation = (child_section_name, parent_section_name, has_parent_section_type, chunk_id)
        if has_parent_relation not in relations_set:
            relations_set.add(has_parent_relation)
            has_parent_section_relations_added += 1
    
    logging.info(f"从结构化数据创建了 {has_section_relations_added} 个Document-Section HAS_SECTION关系")
    logging.info(f"从结构化数据创建了 {has_parent_section_relations_added} 个Section-Section HAS_PARENT_SECTION关系")
    
    # --- 步骤3: 为LLM提取的实体添加CONTAINS关系 ---
    
    # 3.1 为每个非结构性实体(非Document/Section)添加与其所在Section的CONTAINS关系
    processed_entity_section_pairs = set()  # 用于跟踪已处理的实体-章节对
    
    for name, entity_type, chunk_id in entities_set:
        # 跳过Document和Section类型的实体
        if entity_type.lower() in ["document", "section"]:
            continue
        
        # 每个实体都应该与其所在的章节建立CONTAINS关系
        entity_section_key = (name, chunk_id)
        
        if entity_section_key in processed_entity_section_pairs:
            continue
        
        # 检查此实体是否已有CONTAINS关系
        entity_has_contains_relation = False
        for src, tgt, rel_type, rel_chunk_id in relations_set:
            if tgt == name and rel_chunk_id == chunk_id and rel_type == contains_rel_type:
                entity_has_contains_relation = True
                processed_entity_section_pairs.add(entity_section_key)
                break
        
        # 如果没有CONTAINS关系，则添加
        if not entity_has_contains_relation:
            # 找到实体所在chunk对应的Section
            if chunk_id in section_entities:
                # 当前chunk就是一个Section，直接关联
                section_name = section_entities[chunk_id][0]
                contains_relation = (section_name, name, contains_rel_type, chunk_id)
                if contains_relation not in relations_set:
                    relations_set.add(contains_relation)
                    contains_relations_added += 1
                    processed_entity_section_pairs.add(entity_section_key)
            elif chunk_id in chunk_map:
                # 查找当前chunk的父级，应该是一个Section
                parent_id = chunk_map[chunk_id].get('parent_id')
                if parent_id and parent_id in section_entities:
                    section_name = section_entities[parent_id][0]
                    contains_relation = (section_name, name, contains_rel_type, chunk_id)
                    if contains_relation not in relations_set:
                        relations_set.add(contains_relation)
                        contains_relations_added += 1
                        processed_entity_section_pairs.add(entity_section_key)
                else:
                    # 如果找不到父section，使用文档作为容器
                    contains_relation = (document_title, name, contains_rel_type, chunk_id)
                    if contains_relation not in relations_set:
                        relations_set.add(contains_relation)
                        contains_relations_added += 1
                        processed_entity_section_pairs.add(entity_section_key)
            else:
                # 找不到chunk，使用文档作为容器
                contains_relation = (document_title, name, contains_rel_type, chunk_id)
                if contains_relation not in relations_set:
                    relations_set.add(contains_relation)
                    contains_relations_added += 1
                    processed_entity_section_pairs.add(entity_section_key)
    
    logging.info(f"为LLM提取的实体添加了 {contains_relations_added} 个CONTAINS关系")
    
    # --- 步骤4: 最终检查，确保没有孤立实体 ---
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
        logging.warning(f"发现 {len(isolated_entities)} 个仍然孤立的实体，将添加到文档的CONTAINS关系")
        isolated_fixes = 0
        
        # 为每个孤立实体添加到文档的CONTAINS关系
        for name, entity_type, chunk_id in isolated_entities:
            contains_relation = (document_title, name, contains_rel_type, chunk_id)
            
            if contains_relation not in relations_set:
                relations_set.add(contains_relation)
                isolated_fixes += 1
                contains_relations_added += isolated_fixes
        
        logging.info(f"为 {isolated_fixes} 个孤立实体添加了与文档的CONTAINS关系")
    
    # 返回所有添加的关系总数
    total_relations_added = has_section_relations_added + has_parent_section_relations_added + contains_relations_added
    
    logging.info(f"总共添加了 {total_relations_added} 个结构化关系")
    return total_relations_added

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
    cypher_statements = generate_cypher_statements(entities_set, relations_set, config)

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(";\n".join(cypher_statements) + ";\n")  # 每个语句后添加分号
        print(f"\nCypher statements saved to: {output_path}")
    except IOError as e:
        print(f"\nError writing Cypher statements to file {output_path}: {e}")
        print("\nCypher Statements:\n")
        print(";\n".join(cypher_statements) + ";\n")  # 作为备选方案打印到控制台


def create_contains_relations(section_entities: List[Entity], semantic_entities: List[Entity]) -> List[Relation]:
    """为语义实体创建CONTAINS关系（Section -> 语义实体）"""
    contains_relations = []
    
    # 创建section_id到section名称的映射
    section_map = {entity.context_id: entity.name for entity in section_entities if entity.type == "Section"}
    
    # 为每个语义实体创建CONTAINS关系
    for entity in semantic_entities:
        # 跳过Document和Section类型的实体
        if entity.type in ["Document", "Section"]:
            continue
        
        # 检查实体的context_id是否属于某个section
        if entity.context_id in section_map:
            section_name = section_map[entity.context_id]
            contains_relation = Relation(
                source=section_name,
                target=entity.name,
                type="CONTAINS",
                context_id=entity.context_id
            )
            contains_relations.append(contains_relation)
    
    return contains_relations

async def main_async(input_json_path: str, config_path: str, output_cypher_path: str):
    """主处理流程"""
    try:
        # 1. 加载配置
        global config
        logging.info(f"从 {config_path} 加载配置")
        config = Config.from_yaml(config_path)
        
        # 2. 创建LLM客户端
        llm_client = LLMClient(
            api_key=LLM_API_KEY,
            api_host=LLM_API_HOST,
            model=LLM_MODEL,
            max_concurrent=MAX_CONCURRENT_REQUESTS,
            request_delay=REQUEST_DELAY,
            timeout=REQUEST_TIMEOUT
        )

        # 3. 加载输入数据
        logging.info(f"从 {input_json_path} 加载输入数据")
        data = load_json_data(input_json_path)
        if not data:
            logging.error("加载输入数据失败。退出。")
            return
        
        # 4. 处理文档结构
        logging.info("处理文档结构...")
        start_time = time.time()
        structure_entities, structure_relations = await process_document_structure(data, config)
        logging.info(f"结构处理在 {time.time() - start_time:.2f} 秒内完成。")
        logging.info(f"创建了 {len(structure_entities)} 个结构实体和 {len(structure_relations)} 个结构关系。")
        
        # 过滤出Section实体，用于后续处理
        section_entities = [entity for entity in structure_entities if entity.type == "Section"]
        
        # 5. 使用LLM提取语义信息
        logging.info("使用LLM提取语义信息...")
        start_time = time.time()
        semantic_entities, semantic_relations = await extract_semantic_info(data, section_entities, config, llm_client)
        logging.info(f"语义提取在 {time.time() - start_time:.2f} 秒内完成。")
        logging.info(f"提取了 {len(semantic_entities)} 个语义实体和 {len(semantic_relations)} 个语义关系。")
        
        # 6. 创建CONTAINS关系
        logging.info("创建章节-实体CONTAINS关系...")
        contains_relations = create_contains_relations(section_entities, semantic_entities)
        logging.info(f"创建了 {len(contains_relations)} 个CONTAINS关系。")
        
        # 7. 合并所有实体和关系
        all_entities = structure_entities + semantic_entities
        all_relations = structure_relations + semantic_relations + contains_relations
        
        logging.info(f"总共有 {len(all_entities)} 个实体和 {len(all_relations)} 个关系。")
        
        # 8. 生成Cypher语句
        logging.info("生成Cypher语句...")
        cypher_statements = generate_cypher_statements(all_entities, all_relations, config)
        
        # 9. 保存Cypher语句
        output_dir = Path(output_cypher_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_cypher_path, 'w', encoding='utf-8') as f:
            f.write(";\n".join(cypher_statements) + ";\n")
        
        logging.info(f"Cypher语句已保存到 {output_cypher_path}")
        logging.info("处理完成。")
        
    except Exception as e:
        logging.error(f"处理过程中发生错误: {e}")
        logging.exception("详细错误信息:")

def main(input_file: str, output_file: str, config_file: str):
    """主函数，协调整个处理流程"""
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
    
    # 运行异步主函数
    try:
        asyncio.run(main_async(input_path, config_path, output_path))
    except KeyboardInterrupt:
        logging.info("\n任务被用户中断。")
    except Exception as e:
        logging.error(f"执行过程中发生错误: {e}")
        logging.exception("详细错误信息:")


def clean_heading(heading: str, config: Config) -> str:
    """移除标题中的页码和可能的其他噪声。"""
    if not heading:
        return "未知章节"
    # 移除 (p.XX)
    cleaned = re.sub(r'\s*\(p\.\d+\)\s*$', '', heading).strip()
    # 可选：如果为标题定义了规范化映射，应用它
    cleaned = config.canonical_map.get(cleaned, cleaned)
    return cleaned


async def process_document_structure(data: List[Dict[str, Any]], config: Config) -> Tuple[
    List[Entity],  # Document和Section实体
    List[Relation]  # 结构关系
]:
    """解析文档结构，创建Document和Section实体及其关系"""
    document_entities = []  # Document实体列表
    section_entities = []  # Section实体列表
    structure_relations = []  # 结构关系列表
    
    # 存储ID到实体的映射，方便后续创建关系
    id_to_entity_map = {}
    
    # --- 文档标题启发式处理 ---
    # 尝试找到主文档标题
    doc_titles = {}
    if data:
        first_chunk = data[0]
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
    for chunk in data:
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
            document_entity = Entity(
                name=doc_title,
                type="Document",
                context_id=doc_id,
                main_category="文档管理",
                description=f"文档：{doc_title}",
                source="系统自动生成"
            )
            document_entities.append(document_entity)
            id_to_entity_map[doc_id] = document_entity
            logging.debug(f"识别到文档: ID={doc_id}, 标题={doc_title}")

        # 创建章节实体
        section_name = clean_heading(heading, config)
        section_entity = Entity(
            name=section_name,
            type="Section",
            context_id=chunk_id,
            main_category="内容结构",
            description=f"章节：{section_name}",
            source="系统自动生成"
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

async def extract_semantic_info(data: List[Dict[str, Any]], section_entities: List[Entity], config: Config, llm_client: LLMClient) -> Tuple[
    List[Entity],  # 语义实体
    List[Relation]  # 语义关系
]:
    """使用LLM提取语义实体和关系"""
    semantic_entities = []  # 语义实体列表
    semantic_relations = []  # 语义关系列表
    processed_chunk_entities = {}  # 存储每个chunk的实体，用于关系提示
    
    # 创建section_id到section实体的映射
    section_map = {entity.context_id: entity for entity in section_entities if entity.type == "Section"}
    
    # --- 第1阶段: 实体提取 ---
    entity_tasks = []
    for i, chunk in enumerate(data):
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
                    
                    entity = Entity(
                        name=normalized_name,
                        type=normalized_type,
                        context_id=task.chunk_id,
                        main_category="实体概念",
                        description=f"{normalized_type}：{normalized_name}",
                        source="LLM提取"
                    )
                    semantic_entities.append(entity)
                    chunk_entities.append({"name": normalized_name, "type": raw_type})
                else:
                    logging.warning(f"LLM在chunk {task.chunk_id}中返回了无效的实体: {entity_dict}")
            processed_chunk_entities[task.chunk_id] = chunk_entities
    
    # --- 第2阶段: 关系提取 ---
    relation_tasks = []
    for i, chunk in enumerate(data):
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
                    normalized_type = normalize_entity_type(raw_type, config)
                    
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
                    
                    if source_type and not source_exists:
                        normalized_source_type = normalize_entity_type(source_type, config)
                        semantic_entities.append(Entity(
                            name=normalized_source,
                            type=normalized_source_type,
                            context_id=task.chunk_id,
                            main_category="关系实体",
                            description=f"{normalized_source_type}：{normalized_source}",
                            source="关系提取"
                        ))
                    
                    if target_type and not target_exists:
                        normalized_target_type = normalize_entity_type(target_type, config)
                        semantic_entities.append(Entity(
                            name=normalized_target,
                            type=normalized_target_type,
                            context_id=task.chunk_id,
                            main_category="关系实体",
                            description=f"{normalized_target_type}：{normalized_target}",
                            source="关系提取"
                        ))
                else:
                    logging.warning(f"LLM在chunk {task.chunk_id}中返回了无效的关系: {relation_dict}")
    
    return semantic_entities, semantic_relations

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
