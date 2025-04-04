import json
import os
from openai import OpenAI
import sys
from typing import Dict, List, Any
import time
import re

# 初始化API参数
LLM_API_KEY = os.environ.get("LLM_BINDING_API_KEY") or os.environ.get("SILICONFLOW_API_KEY") or os.environ.get("OPENAI_API_KEY")
LLM_API_HOST = os.environ.get("LLM_BINDING_HOST")
LLM_MODEL = os.environ.get("LLM_MODEL") or "gpt-3.5-turbo-16k"

# 初始化OpenAI客户端
if LLM_API_HOST:
    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_API_HOST)
else:
    client = OpenAI(api_key=LLM_API_KEY)

def call_llm_for_summary(content: str, max_tokens: int = 200) -> str:
    """
    调用LLM生成摘要
    
    Args:
        content: 需要摘要的内容
        max_tokens: 摘要最大token数
        
    Returns:
        生成的摘要文本
    """
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,  # 使用环境变量中的模型
            messages=[
                {"role": "system", "content": "请对以下文本生成摘要，要求内容精准和把握实质，摘要字数不超过200字。"},
                {"role": "user", "content": content}
            ],
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"调用LLM出错: {e}")
        return f"摘要生成失败: {str(e)}" 