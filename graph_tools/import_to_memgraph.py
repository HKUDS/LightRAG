#!/usr/bin/env python3
# coding: utf-8

import os
import sys
import time
import json
import logging
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum



from gqlalchemy import Memgraph

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class ConflictStrategy(Enum):
    """处理实体或关系冲突的策略"""
    SKIP = "skip"       # 跳过已存在的实体/关系
    UPDATE = "update"   # 更新已存在的实体/关系
    ERROR = "error"     # 遇到冲突时报错

def read_cypher_file(file_path: str) -> List[str]:
    """
    读取Cypher文件并返回所有语句列表
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 根据分号分割语句
    statements = []
    current_statement = []
    
    for line in content.split('\n'):
        # 跳过空行和注释行
        if not line.strip() or line.strip().startswith('//'):
            continue
            
        # 添加当前行到当前语句
        current_statement.append(line.strip())
        
        # 如果行以分号结尾，说明是一个完整的语句
        if line.strip().endswith(';'):
            # 合并当前语句的所有行
            full_statement = ' '.join(current_statement)
            # 清理语句
            full_statement = full_statement.replace('\n', ' ').strip()
            # 移除多余的分号
            while full_statement.endswith(';'):
                full_statement = full_statement[:-1].strip()
            if full_statement:
                statements.append(full_statement)
            current_statement = []
    
    # 处理最后一个可能没有分号的语句
    if current_statement:
        full_statement = ' '.join(current_statement).strip()
        if full_statement:
            statements.append(full_statement)
    
    return statements

def convert_create_to_merge(statement: str) -> str:
    """
    将CREATE语句转换为MERGE语句以处理已存在的实体和关系
    简单替换，复杂情况可能需要更精细的解析
    """
    if statement.strip().upper().startswith('CREATE '):
        return statement.replace('CREATE ', 'MERGE ', 1).replace('create ', 'MERGE ', 1)
    return statement

def analyze_cypher_error(error_msg: str) -> Dict[str, Any]:
    """
    分析Cypher错误信息，返回错误类型和详情
    """
    error_info = {
        "type": "unknown",
        "detail": error_msg,
        "is_conflict": False
    }
    
    error_msg = error_msg.lower()
    
    if "already exists" in error_msg or "constraint" in error_msg:
        error_info["type"] = "conflict"
        error_info["is_conflict"] = True
    elif "syntax error" in error_msg:
        error_info["type"] = "syntax"
    elif "property" in error_msg and "not found" in error_msg:
        error_info["type"] = "property_not_found"
    elif "node" in error_msg and "not found" in error_msg:
        error_info["type"] = "node_not_found"
    elif "relationship" in error_msg and "not found" in error_msg:
        error_info["type"] = "relationship_not_found"
    elif "timeout" in error_msg:
        error_info["type"] = "timeout"
    elif "out of memory" in error_msg:
        error_info["type"] = "out_of_memory"
    
    return error_info

def execute_statement_with_retry(memgraph, statement: str, max_retries: int = 3) -> Tuple[bool, str]:
    """
    执行单个Cypher语句，支持重试机制
    返回: (是否成功, 错误信息)
    """
    retries = 0
    last_error = ""
    
    while retries <= max_retries:
        try:
            memgraph.execute(statement)
            return True, ""
        except Exception as e:
            last_error = str(e)
            error_info = analyze_cypher_error(last_error)
            
            # 如果是超时错误，重试
            if error_info["type"] == "timeout":
                retries += 1
                if retries <= max_retries:
                    time.sleep(1)  # 重试前等待一秒
                    continue
            else:
                # 其他错误不重试
                break
    
    return False, last_error

def execute_batch(memgraph, statements: List[str], use_transactions: bool, 
                 conflict_strategy: ConflictStrategy) -> Dict[str, Any]:
    """
    执行一批Cypher语句，返回执行结果统计
    """
    results = {
        "success": 0,
        "skipped": 0,
        "error": 0,
        "errors": []  # 详细错误信息
    }
    
    if use_transactions:
        # 使用事务执行批处理
        try:
            # 开始事务
            memgraph.execute("BEGIN")
            transaction_success = True
            
            for i, statement in enumerate(statements):
                success, error_msg = execute_statement_with_retry(memgraph, statement)
                
                if success:
                    results["success"] += 1
                else:
                    error_info = analyze_cypher_error(error_msg)
                    
                    if conflict_strategy == ConflictStrategy.SKIP and error_info["is_conflict"]:
                        results["skipped"] += 1
                        logger.debug(f"跳过已存在的实体/关系 (第 {i + 1} 条)")
                    elif conflict_strategy == ConflictStrategy.ERROR and error_info["is_conflict"]:
                        # 回滚事务并记录错误
                        memgraph.execute("ROLLBACK")
                        error_detail = {
                            "index": i,
                            "statement": statement,
                            "error": error_msg,
                            "type": error_info["type"]
                        }
                        results["errors"].append(error_detail)
                        results["error"] += 1
                        transaction_success = False
                        break
                    else:
                        results["error"] += 1
                        error_detail = {
                            "index": i,
                            "statement": statement,
                            "error": error_msg,
                            "type": error_info["type"]
                        }
                        results["errors"].append(error_detail)
            
            # 如果事务成功，提交事务
            if transaction_success:
                memgraph.execute("COMMIT")
            else:
                # 确保事务被回滚
                try:
                    memgraph.execute("ROLLBACK")
                except Exception:
                    pass
                
        except Exception as e:
            # 确保事务被回滚
            try:
                memgraph.execute("ROLLBACK")
            except Exception:
                pass
            
            logger.error(f"批处理执行出错: {str(e)}")
            # 记录整个批处理错误
            results["error"] += 1
            results["errors"].append({
                "index": -1,  # -1表示整个批处理错误
                "statement": "TRANSACTION",
                "error": str(e),
                "type": "transaction"
            })
    else:
        # 不使用事务，逐条执行
        for i, statement in enumerate(statements):
            success, error_msg = execute_statement_with_retry(memgraph, statement)
            
            if success:
                results["success"] += 1
            else:
                error_info = analyze_cypher_error(error_msg)
                
                if conflict_strategy == ConflictStrategy.SKIP and error_info["is_conflict"]:
                    results["skipped"] += 1
                    logger.debug(f"跳过已存在的实体/关系 (第 {i + 1} 条)")
                elif conflict_strategy == ConflictStrategy.ERROR and error_info["is_conflict"]:
                    results["error"] += 1
                    error_detail = {
                        "index": i,
                        "statement": statement,
                        "error": error_msg,
                        "type": error_info["type"]
                    }
                    results["errors"].append(error_detail)
                    break
                else:
                    results["error"] += 1
                    error_detail = {
                        "index": i,
                        "statement": statement,
                        "error": error_msg,
                        "type": error_info["type"]
                    }
                    results["errors"].append(error_detail)
    
    return results

def create_memgraph_connection(host: str, port: int, username: Optional[str], password: Optional[str]) -> Memgraph:
    """
    创建到Memgraph的连接，正确处理None值的用户名和密码
    """
    # 准备连接参数，过滤掉None值
    connection_params = {"host": host, "port": port}
    
    # 只有在用户名和密码都不为None时才添加到连接参数
    if username is not None and password is not None:
        connection_params["username"] = username
        connection_params["password"] = password
        
    return Memgraph(**connection_params)

def import_to_memgraph(cypher_file: str, host: str = '127.0.0.1', port: int = 7687, 
                      username: Optional[str] = None, password: Optional[str] = None,
                      conflict_strategy: ConflictStrategy = ConflictStrategy.SKIP,
                      batch_size: int = 100, use_transactions: bool = True,
                      error_log_file: Optional[str] = None,
                      debug_mode: bool = False):
    """
    将Cypher文件导入到Memgraph数据库
    
    参数:
        cypher_file: Cypher文件路径
        host: Memgraph主机地址
        port: Memgraph端口
        username: 用户名（如需）
        password: 密码（如需）
        conflict_strategy: 处理冲突的策略
        batch_size: 批处理大小
        use_transactions: 是否使用事务
        error_log_file: 错误日志文件路径
        debug_mode: 是否启用调试模式
    """
    # 设置日志级别
    if debug_mode:
        logger.setLevel(logging.DEBUG)
    
    print(f"正在从文件 {cypher_file} 导入数据到 Memgraph ({host}:{port})...")
    logger.info(f"使用配置: 冲突策略={conflict_strategy.value}, 批处理大小={batch_size}, 事务={use_transactions}")
    start_time = time.time()
    
    # 统计结果
    total_results = {
        "success": 0,
        "skipped": 0,
        "error": 0,
        "total": 0,
        "error_types": {},
        "errors": []
    }
    
    try:
        # 创建Memgraph连接，特别处理None值的用户名和密码
        try:
            memgraph = create_memgraph_connection(host, port, username, password)
            logger.info(f"成功创建到Memgraph的连接")
        except Exception as e:
            raise Exception(f"创建Memgraph连接时出错: {str(e)}")
        
        # 测试连接
        try:
            memgraph.execute("RETURN 1")
            logger.info("Memgraph连接成功")
        except Exception as e:
            raise Exception(f"无法连接到Memgraph: {str(e)}")
        
        # 读取文件内容
        statements = read_cypher_file(cypher_file)
        logger.info(f"已读取 {len(statements)} 条Cypher语句")
        total_results["total"] = len(statements)
        
        # 根据冲突策略处理语句
        if conflict_strategy == ConflictStrategy.UPDATE:
            # 将CREATE语句转换为MERGE语句
            statements = [convert_create_to_merge(stmt) for stmt in statements]
            logger.info("已将CREATE语句转换为MERGE语句以处理实体和关系冲突")
        
        # 使用批处理提高性能
        batch_count = 0
        
        for i in range(0, len(statements), batch_size):
            batch_count += 1
            batch = statements[i:i+batch_size]
            
            # 执行批处理
            batch_results = execute_batch(
                memgraph=memgraph,
                statements=batch,
                use_transactions=use_transactions,
                conflict_strategy=conflict_strategy
            )
            
            # 更新总体统计
            total_results["success"] += batch_results["success"]
            total_results["skipped"] += batch_results["skipped"]
            total_results["error"] += batch_results["error"]
            
            # 记录错误明细
            for error in batch_results["errors"]:
                if error["type"] not in total_results["error_types"]:
                    total_results["error_types"][error["type"]] = 0
                total_results["error_types"][error["type"]] += 1
                
                # 添加批次信息
                error["batch"] = batch_count
                error["global_index"] = i + error.get("index", 0) if error.get("index", 0) >= 0 else -1
                total_results["errors"].append(error)
            
            # 显示进度
            current_progress = min(i + batch_size, len(statements))
            print(f"进度: {current_progress}/{len(statements)} 条语句已处理 "
                  f"(成功: {batch_results['success']}, 跳过: {batch_results['skipped']}, "
                  f"错误: {batch_results['error']})")
            
            # 如果有严重错误且策略是ERROR，则中断
            if batch_results["error"] > 0 and conflict_strategy == ConflictStrategy.ERROR:
                critical_errors = [e for e in batch_results["errors"] 
                                  if e["type"] not in ["conflict", "duplicate"]]
                if critical_errors:
                    logger.error(f"发现严重错误，中断导入过程")
                    break
        
        end_time = time.time()
        
        # 将错误详情保存到文件
        if error_log_file and total_results["errors"]:
            try:
                with open(error_log_file, 'w', encoding='utf-8') as f:
                    json.dump(total_results["errors"], f, ensure_ascii=False, indent=2)
                logger.info(f"错误详情已保存到 {error_log_file}")
            except Exception as e:
                logger.error(f"保存错误日志时出错: {str(e)}")
        
        # 输出错误类型统计
        if total_results["error_types"]:
            print("\n错误类型统计:")
            for error_type, count in total_results["error_types"].items():
                print(f"  - {error_type}: {count}个")
        
        print(f"\n导入完成!")
        print(f"成功: {total_results['success']}, 跳过: {total_results['skipped']}, "
              f"错误: {total_results['error']}, 总计: {total_results['total']}")
        print(f"耗时: {end_time - start_time:.2f} 秒")
        
    except Exception as e:
        logger.error(f"导入过程中出错: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='将Cypher文件导入到Memgraph数据库')
    parser.add_argument('cypher_file', help='Cypher文件路径')
    parser.add_argument('--host', default='127.0.0.1', help='Memgraph主机地址 (默认: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=7687, help='Memgraph端口 (默认: 7687)')
    parser.add_argument('--username', help='用户名 (如果需要)')
    parser.add_argument('--password', help='密码 (如果需要)')
    parser.add_argument('--conflict-strategy', choices=['skip', 'update', 'error'], default='skip',
                      help='冲突处理策略: skip=跳过已存在的实体/关系, update=使用MERGE更新, error=遇到冲突时报错 (默认: skip)')
    parser.add_argument('--batch-size', type=int, default=100, help='批处理大小 (默认: 100)')
    parser.add_argument('--no-transactions', action='store_true', help='不使用事务')
    parser.add_argument('--error-log', help='错误日志文件路径')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    parser.add_argument('--validate-queries', action='store_true', help='验证Cypher语句而不执行')
    
    args = parser.parse_args()
    
    conflict_strategy = ConflictStrategy(args.conflict_strategy)
    
    if args.validate_queries:
        # 只验证语句，不执行
        statements = read_cypher_file(args.cypher_file)
        print(f"已验证 {len(statements)} 条Cypher语句")
        sys.exit(0)
    
    import_to_memgraph(
        args.cypher_file, 
        host=args.host, 
        port=args.port,
        username=args.username, 
        password=args.password,
        conflict_strategy=conflict_strategy,
        batch_size=args.batch_size,
        use_transactions=not args.no_transactions,
        error_log_file=args.error_log,
        debug_mode=args.debug
    ) 