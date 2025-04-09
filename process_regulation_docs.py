#!/usr/bin/env python

"""
自动处理data/RegulationDocuments目录下的所有JSON文件
生成知识图谱的Cypher文件
"""

import os
import subprocess
import glob
import sys
from pathlib import Path

# 配置参数
CONFIG_FILE = "tests/config.yaml"
INPUT_DIR = "data/RegulationDocuments"
OUTPUT_DIR = "data/RegulationDocuments/output_cypher"

def process_json_files():
    """处理所有JSON文件并生成Cypher文件"""
    # 检查配置文件是否存在
    if not os.path.isfile(CONFIG_FILE):
        print(f"错误: 配置文件 '{CONFIG_FILE}' 不存在")
        return False
    
    # 检查输入目录是否存在
    if not os.path.isdir(INPUT_DIR):
        print(f"错误: 输入目录 '{INPUT_DIR}' 不存在")
        return False
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 获取所有JSON文件
    json_files = glob.glob(os.path.join(INPUT_DIR, "*.json"))
    
    if not json_files:
        print(f"警告: 在 '{INPUT_DIR}' 中没有找到JSON文件")
        return False
    
    print(f"找到 {len(json_files)} 个JSON文件进行处理")
    
    # 处理每个JSON文件
    for json_file in json_files:
        filename = os.path.basename(json_file)
        output_filename = os.path.splitext(filename)[0] + ".cypher"
        output_file = os.path.join(OUTPUT_DIR, output_filename)
        
        print(f"处理文件: {filename}")
        
        # 运行实体提取工具
        cmd = ["./graph_tools/entity_extract", "-i", json_file, "-o", output_file, "-c", CONFIG_FILE]
        try:
            subprocess.run(cmd, check=True)
            print(f"成功生成Cypher文件: {output_file}")
        except subprocess.CalledProcessError:
            print(f"错误: 处理文件 '{json_file}' 时失败")
            continue
    
    return True

def import_to_memgraph():
    """导入所有Cypher文件到Memgraph"""
    # 检查是否有Cypher文件
    cypher_files = glob.glob(os.path.join(OUTPUT_DIR, "*.cypher"))
    
    if not cypher_files:
        print(f"警告: 在 '{OUTPUT_DIR}' 中没有找到Cypher文件")
        return False
    
    print(f"找到 {len(cypher_files)} 个Cypher文件可以导入到Memgraph")
    
    # 询问是否导入到Memgraph
    while True:
        answer = input("是否导入数据到Memgraph? (y/n): ").lower()
        if answer in ['y', 'n']:
            break
    
    if answer == 'n':
        print("跳过导入到Memgraph")
        return True
    
    # 导入每个Cypher文件
    for cypher_file in cypher_files:
        print(f"导入文件: {os.path.basename(cypher_file)}")
        
        cmd = ["python", "graph_tools/import_to_memgraph.py", cypher_file]
        try:
            subprocess.run(cmd, check=True)
            print(f"成功导入文件到Memgraph: {cypher_file}")
        except subprocess.CalledProcessError:
            print(f"错误: 导入文件 '{cypher_file}' 时失败")
            continue
    
    return True

if __name__ == "__main__":
    print("开始处理文档...")
    
    if process_json_files():
        print("所有JSON文件处理完成!")
        import_to_memgraph()
    else:
        print("处理失败!")
        sys.exit(1)
    
    print("处理完成!") 