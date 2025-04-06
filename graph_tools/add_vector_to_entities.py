import asyncio
import os
import numpy as np
from dotenv import load_dotenv
from gqlalchemy import Memgraph
import logging
import time
import sys
from typing import List, Dict, Any, Optional
from vector_service import get_embedding, format_entity_text, SILICONFLOW_API_KEY, BATCH_API_SIZE, logger

# 加载环境变量
load_dotenv()

# 连接Memgraph
memgraph_host = os.getenv("MEMGRAPH_HOST", "localhost")
memgraph_port = int(os.getenv("MEMGRAPH_PORT", "7687"))

# API配置
MAX_RETRIES = 3
RETRY_DELAY = 2

async def connect_to_memgraph(max_retries: int = 3) -> Memgraph:
    """连接到Memgraph数据库，带重试机制"""
    retry_count = 0
    while retry_count < max_retries:
        try:
            memgraph = Memgraph(host=memgraph_host, port=memgraph_port)
            # 测试连接
            list(memgraph.execute_and_fetch("RETURN 1"))
            logger.info(f"成功连接到Memgraph: {memgraph_host}:{memgraph_port}")
            return memgraph
        except Exception as e:
            retry_count += 1
            logger.warning(f"连接Memgraph失败 (尝试 {retry_count}/{max_retries}): {str(e)}")
            if retry_count < max_retries:
                await asyncio.sleep(RETRY_DELAY)
            else:
                logger.error(f"无法连接到Memgraph，已达最大重试次数: {str(e)}")
                raise

async def update_entities_with_vectors(memgraph: Memgraph, entity_vectors: List[Dict[str, Any]]) -> None:
    """批量更新实体向量属性"""
    try:
        # 准备参数列表
        params_list = []
        for entity in entity_vectors:
            params_list.append({"id": entity["id"], "vector": entity["vector"]})
        
        # 批量执行更新
        batch_query = """
        UNWIND $params AS param
        MATCH (n)
        WHERE id(n) = param.id
        SET n.vector = param.vector
        """
        memgraph.execute(batch_query, {"params": params_list})
        
        entity_ids = [entity["id"] for entity in entity_vectors]
        logger.info(f"已批量更新 {len(entity_ids)} 个实体向量: {entity_ids}")
    except Exception as e:
        logger.error(f"批量更新实体向量属性时出错: {str(e)}")
        raise

async def process_entity_batch(memgraph: Memgraph, entities: List[Dict[str, Any]], batch_size: int = 32) -> int:
    """处理一批实体，生成嵌入并更新到数据库，返回成功处理的实体数量"""
    total_processed = 0
    
    for i in range(0, len(entities), batch_size):
        batch = entities[i:i+batch_size]
        logger.info(f"处理批次 {i//batch_size + 1}/{(len(entities)-1)//batch_size + 1}，共 {len(batch)} 个实体")
        
        # 对批次内的实体进行多次API调用（每次最多BATCH_API_SIZE个）
        for j in range(0, len(batch), BATCH_API_SIZE):
            sub_batch = batch[j:j+BATCH_API_SIZE]
            formatted_texts = [format_entity_text(entity) for entity in sub_batch]
            
            try:
                # 批量获取嵌入向量
                vectors = await get_embedding(formatted_texts, SILICONFLOW_API_KEY)
                
                # 准备批量更新数据
                entities_to_update = []
                for k, entity in enumerate(sub_batch):
                    entities_to_update.append({
                        "id": entity["id"],
                        "vector": vectors[k]
                    })
                
                # 批量更新实体向量
                await update_entities_with_vectors(memgraph, entities_to_update)
                total_processed += len(sub_batch)
                
            except Exception as e:
                entity_ids = [entity["id"] for entity in sub_batch]
                logger.error(f"处理实体批次时出错: {str(e)}，实体IDs: {entity_ids}")
            
            # 添加延迟避免API请求过快
            if j + BATCH_API_SIZE < len(batch):
                await asyncio.sleep(0.5)
    
    return total_processed

def create_vector_index(memgraph: Memgraph) -> bool:
    """创建向量索引，返回是否成功创建"""
    try:
        # 检查索引是否已存在
        index_info = list(memgraph.execute_and_fetch("SHOW INDEX INFO;"))
        if not any(record.get("name") == "entity_vector_idx" for record in index_info):
            memgraph.execute("CREATE VECTOR INDEX entity_vector_idx ON (n)(n.vector);")
            logger.info("已创建向量索引entity_vector_idx")
            return True
        else:
            logger.info("向量索引entity_vector_idx已存在")
            return False
    except Exception as e:
        logger.error(f"创建索引时出错: {str(e)}")
        return False

async def process_document_entities(memgraph: Memgraph, batch_size: int = 32) -> int:
    """优先处理Document类型的实体，为其生成嵌入并更新，返回成功处理的数量"""
    try:
        # 查询所有没有向量属性的Document类型实体
        query = """
        MATCH (n:Document)
        WHERE n.vector IS NULL
        RETURN id(n) AS id, labels(n) AS labels, properties(n) AS properties
        """
        
        entities = list(memgraph.execute_and_fetch(query))
        if not entities:
            logger.info("没有找到需要添加向量属性的Document实体")
            return 0
        
        logger.info(f"找到 {len(entities)} 个需要添加向量属性的Document实体")
        
        # 处理实体
        processed_count = await process_entity_batch(memgraph, entities, batch_size)
        
        logger.info(f"处理完成, 成功为 {processed_count}/{len(entities)} 个Document实体添加了嵌入向量")
        return processed_count
        
    except Exception as e:
        logger.error(f"处理Document实体时出错: {str(e)}")
        return 0

async def main():
    """主函数"""
    try:
        # 连接到数据库
        memgraph = await connect_to_memgraph()
        
        # 记录总开始时间
        total_start_time = time.time()
        
        # 先处理Document实体
        logger.info("优先处理Document实体...")
        doc_processed = await process_document_entities(memgraph)
        
        # 查询其他类型没有向量属性的实体
        query = """
        MATCH (n)
        WHERE n.vector IS NULL AND NOT 'Document' IN labels(n)
        RETURN id(n) AS id, labels(n) AS labels, properties(n) AS properties
        LIMIT 1000
        """
        
        entities = list(memgraph.execute_and_fetch(query))
        other_processed = 0
        
        if not entities:
            logger.info("没有找到其他需要添加向量属性的实体")
        else:
            logger.info(f"找到 {len(entities)} 个需要添加向量属性的其他实体")
            
            # 处理其他实体
            other_processed = await process_entity_batch(memgraph, entities)
        
        # 创建向量索引
        create_vector_index(memgraph)
        
        # 计算总处理时间
        total_elapsed_time = time.time() - total_start_time
        
        logger.info(f"全部处理完成, 总计处理了 {doc_processed + other_processed} 个实体 "
                   f"(Document: {doc_processed}, 其他: {other_processed}), "
                   f"总耗时: {total_elapsed_time:.2f}秒")
        
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())