import asyncio
import os
import numpy as np
import logging
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Any, Optional
import aiohttp
import json
from gqlalchemy import Memgraph

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("entity_indexer")

# 加载环境变量
load_dotenv()

# 获取环境变量
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "5"))
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", "0.2"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "60.0"))
VECTOR_STORAGE_PATH = os.getenv("WORKING_DIR", "/app/data/rag_passenger") + "/vector_storage"

async def siliconcloud_embedding(
    texts: List[str],
    model: str = "netease-youdao/bce-embedding-base_v1",
    api_key: Optional[str] = None,
    max_token_size: int = 512,
) -> np.ndarray:
    """
    调用SiliconFlow API获取文本的嵌入向量
    
    Args:
        texts: 文本列表
        model: 模型名称
        api_key: API密钥
        max_token_size: 最大token大小
        
    Returns:
        numpy数组形式的嵌入向量
    """
    if not api_key:
        api_key = os.getenv("SILICONFLOW_API_KEY")
        if not api_key:
            raise ValueError("必须提供SILICONFLOW_API_KEY")
    
    api_url = "https://api.siliconflow.cn/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": model,
        "input": texts,
        "max_token_size": max_token_size
    }
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                api_url, 
                headers=headers, 
                json=payload,
                timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"API错误: 状态码 {response.status}, 错误: {error_text}")
                    raise ValueError(f"API错误: {error_text}")
                
                result = await response.json()
                embeddings = [item["embedding"] for item in result["data"]]
                return np.array(embeddings)
        except Exception as e:
            logger.error(f"嵌入请求失败: {str(e)}")
            raise

async def embedding_func(texts: list[str]) -> np.ndarray:
    """
    获取文本的嵌入向量，支持批处理
    
    Args:
        texts: 文本列表
    
    Returns:
        嵌入向量的numpy数组
    """
    # 添加批处理大小限制，最大批大小为32
    max_batch_size = 32
    
    # 如果输入小于或等于最大批大小，直接处理
    if len(texts) <= max_batch_size:
        return await siliconcloud_embedding(
            texts,
            model="netease-youdao/bce-embedding-base_v1",
            api_key=SILICONFLOW_API_KEY,
            max_token_size=512,
        )
    
    # 如果输入超过最大批大小，分批处理
    all_embeddings = []
    for i in range(0, len(texts), max_batch_size):
        batch_texts = texts[i:i+max_batch_size]
        batch_embeddings = await siliconcloud_embedding(
            batch_texts,
            model="netease-youdao/bce-embedding-base_v1",
            api_key=SILICONFLOW_API_KEY,
            max_token_size=512,
        )
        all_embeddings.append(batch_embeddings)
        # 添加延迟避免API速率限制
        await asyncio.sleep(REQUEST_DELAY)
    
    # 合并所有批次的结果
    return np.vstack(all_embeddings)

def format_entity_text(entity: dict) -> str:
    """
    根据实体属性格式化文本用于嵌入
    
    Args:
        entity: 实体数据
        
    Returns:
        格式化后的文本
    """
    props = entity.get("properties", {})
    # 获取第一个标签作为实体类型
    label = entity.get("labels", ["Unknown"])[0]
    name = props.get("name", "")
    description = props.get("description", "")
    main_category = props.get("main_category", "")
    source = props.get("source", "")
    
    # 组合文本，包括更多属性以提高嵌入质量
    text_parts = []
    if label:
        text_parts.append(f"类型: {label}")
    if name:
        text_parts.append(f"名称: {name}")
    if description and description != name:  # 避免重复
        text_parts.append(f"描述: {description}")
    if main_category:
        text_parts.append(f"主分类: {main_category}")
    if source:
        text_parts.append(f"来源: {source}")
        
    return ", ".join(text_parts)

class NanoVectorDBStorage:
    """简化的向量数据库存储实现"""
    
    def __init__(self, storage_path: str):
        """
        初始化向量存储
        
        Args:
            storage_path: 存储路径
        """
        self.storage_path = storage_path
        self.vectors = []
        self.metadata = []
        os.makedirs(os.path.dirname(storage_path), exist_ok=True)
        self.load()
        
    def load(self):
        """从磁盘加载向量存储"""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.vectors = [np.array(v) for v in data.get('vectors', [])]
                    self.metadata = data.get('metadata', [])
                    logger.info(f"从 {self.storage_path} 加载了 {len(self.vectors)} 个向量")
            else:
                logger.info(f"未找到向量存储文件 {self.storage_path}，将创建新存储")
        except Exception as e:
            logger.error(f"加载向量存储时出错: {str(e)}")
            self.vectors = []
            self.metadata = []
    
    def save(self):
        """保存向量存储到磁盘"""
        try:
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                data = {
                    'vectors': [v.tolist() for v in self.vectors],
                    'metadata': self.metadata
                }
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"向量存储保存到 {self.storage_path}")
        except Exception as e:
            logger.error(f"保存向量存储时出错: {str(e)}")
    
    async def add_items(self, items: List[Tuple[list, dict]]):
        """
        添加向量项到存储
        
        Args:
            items: (向量, 元数据)元组的列表
        """
        for vector, metadata in items:
            # 检查是否已存在相同UUID的实体
            entity_uuid = metadata.get('entity_uuid')
            existing_index = None
            
            if entity_uuid:
                for i, meta in enumerate(self.metadata):
                    if meta.get('entity_uuid') == entity_uuid:
                        existing_index = i
                        break
            
            if existing_index is not None:
                # 更新现有实体
                self.vectors[existing_index] = np.array(vector)
                self.metadata[existing_index] = metadata
            else:
                # 添加新实体
                self.vectors.append(np.array(vector))
                self.metadata.append(metadata)

class MemgraphEntityFetcher:
    """从Memgraph获取实体的工具类"""
    
    def __init__(self, host: str = "localhost", port: int = 7687):
        """
        初始化Memgraph连接
        
        Args:
            host: Memgraph服务器主机
            port: Memgraph服务器端口
        """
        # 创建Memgraph连接
        connection_params = {"host": host, "port": port}
        self.client = Memgraph(**connection_params)
        
    async def get_nodes_batch(self, skip: int = 0, limit: int = 100) -> List[Dict]:
        """
        批量获取节点
        
        Args:
            skip: 跳过的节点数
            limit: 返回的节点数限制
            
        Returns:
            节点列表
        """
        # Cypher查询，获取所有节点，包括标签和属性
        query = f"""
        MATCH (n)
        RETURN 
            id(n) AS id,
            labels(n) AS labels,
            properties(n) AS properties
        ORDER BY id(n)
        SKIP {skip}
        LIMIT {limit}
        """
        
        try:
            result = self.client.execute_and_fetch(query)
            nodes = []
            
            for record in result:
                node_id = record["id"]
                node_labels = record["labels"]
                node_properties = record["properties"]
                
                node = {
                    "id": node_id,
                    "labels": node_labels,
                    "properties": node_properties,
                    "type": "node"
                }
                nodes.append(node)
                
            return nodes
        except Exception as e:
            logger.error(f"从Memgraph获取节点时出错: {str(e)}")
            return []

async def build_entity_index(graph_fetcher: MemgraphEntityFetcher, vector_storage: NanoVectorDBStorage, batch_size: int = 100):
    """
    构建实体索引
    
    Args:
        graph_fetcher: 图数据库实体获取器
        vector_storage: 向量存储
        batch_size: 每批处理的实体数量
    """
    has_more_entities = True
    skip = 0
    total_indexed = 0
    
    logger.info("开始构建实体索引...")
    
    while has_more_entities:
        logger.info(f"获取实体批次: skip={skip}, limit={batch_size}")
        
        # 从图数据库批量获取实体
        entity_batch = await graph_fetcher.get_nodes_batch(skip=skip, limit=batch_size)
        
        if not entity_batch:
            logger.info("没有更多实体，索引构建完成")
            has_more_entities = False
            break
        
        texts_to_embed = []
        entity_ids = []
        valid_entities_in_batch = []
        
        # 准备文本和ID
        for entity in entity_batch:
            entity_uuid = entity.get("properties", {}).get("uuid")
            if entity_uuid:  # 确保实体有UUID
                text = format_entity_text(entity)
                texts_to_embed.append(text)
                entity_ids.append(entity_uuid)
                valid_entities_in_batch.append(entity)  # 保留有效实体以供元数据使用
        
        if not texts_to_embed:
            logger.info(f"批次中没有有效的带UUID的实体 (skip={skip})，跳到下一批")
            skip += len(entity_batch)  # 即使无效也要增加skip
            continue
        
        logger.info(f"计算 {len(texts_to_embed)} 个实体的嵌入向量...")
        
        try:
            # 计算向量嵌入
            embeddings = await embedding_func(texts_to_embed)
            
            logger.info(f"存储 {len(embeddings)} 个嵌入向量到向量数据库...")
            
            # 准备存入向量数据库的数据
            items_to_store = []
            for i in range(len(entity_ids)):
                entity = valid_entities_in_batch[i]
                props = entity.get("properties", {})
                
                metadata = {
                    "entity_uuid": entity_ids[i],
                    "text": texts_to_embed[i],  # 存储原始文本供参考
                    "label": entity.get("labels", ["Unknown"])[0],
                    "name": props.get("name", ""),
                    "entity_id": entity.get("id"),  # 图数据库中的ID
                    "description": props.get("description", ""),
                    "main_category": props.get("main_category", ""),
                    "source": props.get("source", "")
                }
                
                items_to_store.append((embeddings[i].tolist(), metadata))
            
            # 存储到向量数据库
            await vector_storage.add_items(items_to_store)
            
            # 每处理N批次保存一次向量存储
            total_indexed += len(items_to_store)
            if total_indexed % 500 == 0:
                vector_storage.save()
                
            logger.info(f"已存储批次。总计已索引: {total_indexed}")
            
        except Exception as e:
            logger.error(f"处理批次时出错: {str(e)}")
            # 可以在这里添加重试逻辑，或者继续处理下一批
        
        skip += len(entity_batch)
    
    logger.info(f"实体索引构建完成。总计索引了 {total_indexed} 个实体。")
    # 持久化向量索引
    vector_storage.save()
    logger.info("向量索引已保存。")

async def main():
    """主函数"""
    # 从环境变量获取配置
    memgraph_host = os.getenv("MEMGRAPH_HOST", "localhost")
    memgraph_port = int(os.getenv("MEMGRAPH_PORT", "7687"))
    vector_db_path = os.getenv("VECTOR_STORAGE_PATH", VECTOR_STORAGE_PATH + "/entity_vectors.json")
    
    # 初始化图数据库和向量数据库客户端
    graph_fetcher = MemgraphEntityFetcher(host=memgraph_host, port=memgraph_port)
    vector_storage = NanoVectorDBStorage(storage_path=vector_db_path)
    
    # 构建索引
    batch_size = int(os.getenv("ENTITY_BATCH_SIZE", "100"))
    await build_entity_index(graph_fetcher, vector_storage, batch_size=batch_size)

if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main()) 