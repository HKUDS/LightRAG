"""
LLM API 客户端模块

提供与大型语言模型 API 交互的功能
"""

import json
import logging
import asyncio
import time
import httpx
import re
from typing import List, Dict, Optional, Any

from graph_tools.models import LLMTask

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

# 为了兼容原有代码，提供同步调用接口
def call_llm_sync(api_key: str, api_host: str, model: str, prompt: str) -> Optional[str]:
    """
    同步调用LLM API进行推理，兼容原有代码
    实际使用异步方法并等待其完成
    
    Args:
        api_key: API密钥
        api_host: API主机URL
        model: 使用的模型名称
        prompt: 提示文本
        
    Returns:
        Optional[str]: JSON响应字符串，如果失败则为None
    """
    logging.info(f"--- 使用同步接口发送Prompt到LLM (长度: {len(prompt)}) ---")
    
    # 创建LLM客户端和任务
    client = LLMClient(api_key, api_host, model)
    task = LLMTask(
        chunk_id="sync_call",
        prompt_type="unknown",
        content=prompt,
        prompt=prompt
    )
    
    # 运行异步调用并等待结果
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # 如果没有事件循环，创建一个新的
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    task = loop.run_until_complete(client.call_llm(task))
    
    # 如果有结果对象，返回其JSON字符串表示
    if task.result:
        if 'entities' in task.result:
            return json.dumps({"entities": task.result['entities']})
        elif 'relations' in task.result:
            return json.dumps({"relations": task.result['relations']})
    
    return None 