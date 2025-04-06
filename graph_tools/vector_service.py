import os
import asyncio
import aiohttp
import logging
import sys
from typing import List, Dict, Any
from dotenv import load_dotenv

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("vector_embedding.log")
    ]
)
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()

# 获取API密钥与验证
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
if not SILICONFLOW_API_KEY:
    logger.error("未设置SILICONFLOW_API_KEY环境变量")
    sys.exit(1)

# API配置
MAX_RETRIES = 3
RETRY_DELAY = 2
BATCH_API_SIZE = 8  # API批量调用大小

# 模型配置
DEFAULT_MODEL = "netease-youdao/bce-embedding-base_v1"
MAX_TOKEN_SIZE = 512

async def get_embedding(texts: List[str], api_key: str, retry_count: int = 0) -> List[List[float]]:
    """调用API获取多个文本的嵌入向量，支持重试机制"""
    api_url = "https://api.siliconflow.cn/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": DEFAULT_MODEL,
        "input": texts,
        "max_token_size": MAX_TOKEN_SIZE
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ValueError(f"API错误: {response.status}, {error_text}")
                
                result = await response.json()
                return [item["embedding"] for item in result["data"]]
    except Exception as e:
        if retry_count < MAX_RETRIES:
            logger.warning(f"获取嵌入向量失败，尝试重试 ({retry_count+1}/{MAX_RETRIES}): {str(e)}")
            await asyncio.sleep(RETRY_DELAY * (2 ** retry_count))  # 指数退避
            return await get_embedding(texts, api_key, retry_count + 1)
        else:
            logger.error(f"获取嵌入向量失败，超过最大重试次数: {str(e)}")
            raise

def format_entity_text(entity: Dict[str, Any]) -> str:
    """根据实体属性格式化文本用于嵌入"""
    props = entity["properties"]
    label = entity["labels"][0] if entity["labels"] else "Unknown"
    name = props.get("name", "")
    description = props.get("description", "")
    
    # 组合文本
    text_parts = []
    if label:
        text_parts.append(f"类型: {label}")
    if name:
        text_parts.append(f"名称: {name}")
    if description and description != name:
        text_parts.append(f"描述: {description}")
    
    return ", ".join(text_parts) 

async def test_embedding_service():
    """测试嵌入服务并输出关键信息"""
    try:
        # 测试文本
        test_texts = ["这是一个测试文本，用于验证嵌入服务是否正常工作。", 
                      "这是第二个测试文本，用于检查嵌入向量的维度。"]
        
        logger.info(f"正在使用模型 {DEFAULT_MODEL} 测试嵌入服务...")
        logger.info(f"发送 {len(test_texts)} 条测试文本...")
        
        # 获取嵌入向量
        embeddings = await get_embedding(test_texts, SILICONFLOW_API_KEY)
        
        # 输出关键信息
        dimensions = len(embeddings[0])
        logger.info(f"嵌入服务测试成功!")
        logger.info(f"模型: {DEFAULT_MODEL}")
        logger.info(f"嵌入向量维度: {dimensions}")
        logger.info(f"最大标记大小: {MAX_TOKEN_SIZE}")
        
        # 显示部分向量值作为示例
        sample_values = embeddings[0][:5]
        logger.info(f"向量样例(前5个值): {sample_values}")
        
        return True
    except Exception as e:
        logger.error(f"测试嵌入服务失败: {str(e)}")
        return False

def main():
    """主函数，用于测试服务"""
    logger.info("开始测试向量嵌入服务...")
    
    # 运行测试
    result = asyncio.run(test_embedding_service())
    
    if result:
        logger.info("服务测试完成，嵌入功能正常。")
    else:
        logger.error("服务测试失败，请检查配置和连接。")

if __name__ == "__main__":
    main() 