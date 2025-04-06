#!/bin/bash

# 设置默认输入文件
DEFAULT_INPUT_FILE="tests/demo3_with_summary.json"

# 检查是否提供了输入文件参数
if [ $# -lt 1 ]; then
  INPUT_FILE="$DEFAULT_INPUT_FILE"
  echo "使用默认输入文件: $INPUT_FILE"
else
  INPUT_FILE="$1"
fi

# 检查输入文件是否存在
if [ ! -f "$INPUT_FILE" ]; then
  echo "错误: 输入文件 '$INPUT_FILE' 不存在"
  exit 1
fi

# 设置默认配置文件，如果提供了第二个参数则使用该参数
CONFIG_FILE="${2:-tests/config_simple.yaml}"

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
  echo "错误: 配置文件 '$CONFIG_FILE' 不存在"
  exit 1
fi

# 从输入文件中提取文件名（不含路径和扩展名）
FILENAME=$(basename "$INPUT_FILE" .json)
# 提取输入文件的目录路径
DIR_PATH=$(dirname "$INPUT_FILE")
# 创建输出目录（如果不存在）
OUTPUT_DIR="$DIR_PATH/output_cypher"
mkdir -p "$OUTPUT_DIR"
# 构建输出文件路径
OUTPUT_FILE="$OUTPUT_DIR/$FILENAME.cypher"


# 运行实体提取工具
if ! ./graph_tools/entity_extract -i "$INPUT_FILE" -o "$OUTPUT_FILE" -c "$CONFIG_FILE"; then
  echo "错误: 实体提取失败"
  exit 1
fi

# 在导入数据到Memgraph前添加人工确认步骤
echo "导入数据到Memgraph: $OUTPUT_FILE"
echo "是否继续执行? (y/n)"
read -r confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
  echo "操作已取消"
  exit 0
fi

echo "导入数据到Memgraph..."
# 检查Python脚本是否存在
if [ ! -f "tests/import_to_memgraph.py" ]; then
  echo "错误: 导入脚本 'tests/import_to_memgraph.py' 不存在"
  exit 1
fi

# 运行导入脚本
if ! python tests/import_to_memgraph.py "$OUTPUT_FILE"; then
  echo "错误: 数据导入失败"
  exit 1
fi

echo "处理完成!"

