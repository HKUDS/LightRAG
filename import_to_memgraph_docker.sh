#!/bin/bash

# 导入Cypher文件到Docker中运行的Memgraph实例
# 用法: ./import_to_memgraph_docker.sh [cypher文件] [容器名称/ID]

# 默认值
CYPHER_FILE="output_graph.cypher"
CONTAINER_NAME_OR_ID=""

# 解析命令行参数
if [ $# -ge 1 ]; then
  CYPHER_FILE=$1
fi

if [ $# -ge 2 ]; then
  CONTAINER_NAME_OR_ID=$2
else
  # 如果没有提供容器ID/名称，尝试自动获取
  CONTAINER_ID=$(docker ps | grep memgraph | awk '{print $1}' | head -n 1)
  if [ -z "$CONTAINER_ID" ]; then
    echo "错误：未找到运行中的Memgraph容器。请提供容器ID或名称作为第二个参数。"
    echo "用法: $0 [cypher文件] [容器名称/ID]"
    exit 1
  fi
  CONTAINER_NAME_OR_ID=$CONTAINER_ID
  echo "已自动识别Memgraph容器: $CONTAINER_NAME_OR_ID"
fi

# 检查文件是否存在
if [ ! -f "$CYPHER_FILE" ]; then
  echo "错误：找不到文件 $CYPHER_FILE"
  exit 1
fi

echo "开始导入 $CYPHER_FILE 到容器 $CONTAINER_NAME_OR_ID..."

# 执行导入命令
docker exec -i $CONTAINER_NAME_OR_ID mgconsole < $CYPHER_FILE

if [ $? -eq 0 ]; then
  echo "导入完成！请检查Memgraph中的数据。"
else
  echo "导入过程中出错。请检查错误信息。"
  exit 1
fi 