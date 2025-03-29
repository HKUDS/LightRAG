#!/usr/bin/env python3
# coding: utf-8

import os
import sys
import time
from typing import List, Optional



from gqlalchemy import Memgraph

def read_cypher_file(file_path: str) -> List[str]:
    """
    读取Cypher文件并返回所有语句列表
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 根据分号和新行分割语句
    statements = []
    for statement in content.split(';;'):
        statement = statement.strip()
        if statement and not statement.startswith('//'):
            # 删除注释并清理空白
            clean_statement = statement.replace('\n', ' ').strip()
            if clean_statement:
                statements.append(clean_statement + ';')
    
    return statements

def import_to_memgraph(cypher_file: str, host: str = '127.0.0.1', port: int = 7687, 
                      username: Optional[str] = None, password: Optional[str] = None):
    """
    将Cypher文件导入到Memgraph数据库
    """
    print(f"正在从文件 {cypher_file} 导入数据到 Memgraph ({host}:{port})...")
    start_time = time.time()
    
    try:
        memgraph = Memgraph(host=host, port=port)
        
        # 读取文件内容
        statements = read_cypher_file(cypher_file)
        print(f"已读取 {len(statements)} 条Cypher语句")
        
        # 执行语句
        success_count = 0
        for i, statement in enumerate(statements):
            try:
                memgraph.execute(statement)
                success_count += 1
                
                if (i + 1) % 100 == 0:
                    print(f"进度: {i + 1}/{len(statements)} 条语句已执行")
                
            except Exception as e:
                print(f"执行语句时出错 (第 {i + 1} 条): {statement}")
                print(f"错误信息: {str(e)}")
        
        end_time = time.time()
        print(f"导入完成! 总计 {success_count}/{len(statements)} 条语句成功执行")
        print(f"耗时: {end_time - start_time:.2f} 秒")
        
    except Exception as e:
        print(f"导入过程中出错: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='将Cypher文件导入到Memgraph数据库')
    parser.add_argument('cypher_file', help='Cypher文件路径')
    parser.add_argument('--host', default='127.0.0.1', help='Memgraph主机地址 (默认: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=7687, help='Memgraph端口 (默认: 7687)')
    parser.add_argument('--username', help='用户名 (如果需要)')
    parser.add_argument('--password', help='密码 (如果需要)')
    
    args = parser.parse_args()
    
    import_to_memgraph(
        args.cypher_file, 
        host=args.host, 
        port=args.port,
        username=args.username, 
        password=args.password
    ) 