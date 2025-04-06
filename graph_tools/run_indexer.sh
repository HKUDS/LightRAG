#!/bin/bash

# 设置环境变量（如果不存在于.env文件中）
export MEMGRAPH_HOST=${MEMGRAPH_HOST:-"localhost"}
export MEMGRAPH_PORT=${MEMGRAPH_PORT:-"7687"}
export ENTITY_BATCH_SIZE=${ENTITY_BATCH_SIZE:-"100"}
export VECTOR_STORAGE_PATH=${VECTOR_STORAGE_PATH:-"../data/vector_storage/entity_vectors.json"}

# 确保有日志目录
mkdir -p logs

# 运行索引构建脚本
echo "开始构建实体向量索引..."
python entity_embedding_indexer.py > logs/indexing_$(date +%Y%m%d_%H%M%S).log 2>&1

# 检查是否成功完成
if [ $? -eq 0 ]; then
    echo "实体向量索引构建完成！输出文件: $VECTOR_STORAGE_PATH"
else
    echo "实体向量索引构建失败，请查看日志文件以获取详细信息。"
fi 