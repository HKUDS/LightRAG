#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
客票发售和预订系统应急预案知识图谱查询工具 (向量检索增强版)
基于实际查询结果，对查询逻辑进行了调整和扩展，并加入了向量检索能力
"""

import sys
from typing import List, Dict, Any, Optional, Tuple
from py2neo import Graph  # 也支持 Memgraph
import pandas as pd
from tabulate import tabulate
from colorama import init, Fore, Style
import argparse
import os
import datetime
import asyncio
import aiohttp
from dotenv import load_dotenv
import logging

# 初始化colorama
init()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("kg_query.log")
    ]
)
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()

# 向量API配置
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
MAX_RETRIES = 3
RETRY_DELAY = 2

class EmergencyResponseKG:
    """客票发售和预订系统应急预案知识图谱查询工具"""
    
    def __init__(self, uri: str = "bolt://localhost:7687", 
                 user: str = "neo4j", 
                 password: str = "password",
                 export_dir: str = "./exports"):
        """初始化图数据库连接
        
        Args:
            uri: 图数据库连接URI
            user: 数据库用户名
            password: 数据库密码
            export_dir: 导出目录
        """
        self.export_dir = export_dir
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
        
        # 验证API密钥    
        if not SILICONFLOW_API_KEY:
            print(f"{Fore.YELLOW}警告: 未设置SILICONFLOW_API_KEY环境变量，向量检索功能将不可用{Style.RESET_ALL}")
            self.vector_search_enabled = False
        else:
            self.vector_search_enabled = True
            
        try:
            self.graph = Graph(uri, auth=(user, password))
            print(f"{Fore.GREEN}已成功连接到图数据库{Style.RESET_ALL}")
            
            # 检查数据库内容概况
            self._check_database_stats()
            
            # 检查向量功能支持情况
            self._check_vector_support()
        except Exception as e:
            print(f"{Fore.RED}连接图数据库失败: {e}{Style.RESET_ALL}")
            sys.exit(1)
    
    def _check_database_stats(self):
        """检查数据库基本统计信息，帮助诊断问题"""
        try:
            # 节点类型统计
            node_counts = self.execute_query("""
            MATCH (n)
            RETURN labels(n) AS 节点类型, count(n) AS 数量
            ORDER BY 数量 DESC
            """)
            
            # 关系类型统计
            rel_counts = self.execute_query("""
            MATCH ()-[r]->()
            RETURN type(r) AS 关系类型, count(r) AS 数量
            ORDER BY 数量 DESC
            """)
            
            print(f"{Fore.CYAN}数据库节点统计:{Style.RESET_ALL}")
            print(tabulate(node_counts, headers="keys", tablefmt="pretty", showindex=False))
            
            print(f"{Fore.CYAN}数据库关系统计:{Style.RESET_ALL}")
            print(tabulate(rel_counts, headers="keys", tablefmt="pretty", showindex=False))
            
        except Exception as e:
            print(f"{Fore.YELLOW}无法获取数据库统计信息: {e}{Style.RESET_ALL}")
    
    def _check_vector_support(self):
        """检查数据库的向量支持情况"""
        try:
            # 检查是否有节点带有向量属性和VectorEntity标签
            vector_nodes = self.execute_query("""
            MATCH (n:VectorEntity)
            WHERE n.vector IS NOT NULL
            RETURN COUNT(n) AS count
            """)
            
            if vector_nodes.empty or vector_nodes.iloc[0]['count'] == 0:
                print(f"{Fore.YELLOW}警告: 数据库中没有找到带VectorEntity标签的节点{Style.RESET_ALL}")
                self.has_vector_nodes = False
            else:
                self.has_vector_nodes = True
                print(f"{Fore.GREEN}找到 {vector_nodes.iloc[0]['count']} 个带VectorEntity标签的节点{Style.RESET_ALL}")
            
            # 检查vector_index_all索引是否存在
            try:
                test_query = "CALL vector_search.show_index_info() YIELD * RETURN *;"
                result = self.execute_query(test_query)
                
                # 检查是否存在我们创建的索引
                if not result.empty and any('vector_index_all' in str(row) for _, row in result.iterrows()):
                    self.vector_function_available = True
                    self.vector_search_method = "vector_search.search"
                    self.vector_index_name = "vector_index_all"
                    print(f"{Fore.GREEN}向量索引可用: vector_index_all{Style.RESET_ALL}")
                else:
                    print(f"{Fore.YELLOW}警告: 未找到向量索引vector_index_all{Style.RESET_ALL}")
                    self.vector_function_available = False
            except Exception as e:
                self.vector_function_available = False
                print(f"{Fore.YELLOW}警告: 向量搜索过程不可用: {str(e)}{Style.RESET_ALL}")
                
        except Exception as e:
            print(f"{Fore.YELLOW}无法检查向量支持情况: {str(e)}{Style.RESET_ALL}")
            self.has_vector_nodes = False
            self.vector_function_available = False
    
    async def get_embedding(self, text: str) -> List[float]:
        """获取文本的嵌入向量
        
        Args:
            text: 需要嵌入的文本
            
        Returns:
            嵌入向量
        """
        if not self.vector_search_enabled:
            raise ValueError("未配置向量检索API密钥")
            
        api_url = "https://api.siliconflow.cn/v1/embeddings"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {SILICONFLOW_API_KEY}"
        }
        
        payload = {
            "model": "netease-youdao/bce-embedding-base_v1",
            "input": [text],
            "max_token_size": 512
        }
        
        retry_count = 0
        while retry_count < MAX_RETRIES:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(api_url, headers=headers, json=payload) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            raise ValueError(f"API错误: {response.status}, {error_text}")
                        
                        result = await response.json()
                        embedding = result["data"][0]["embedding"]
                        
                        # 只输出向量长度和部分示例，不输出完整向量
                        vector_length = len(embedding)
                        print(f"{Fore.CYAN}获取到向量, 维度: {vector_length}, 前三个元素: {embedding[:3]}...{Style.RESET_ALL}")
                        
                        return embedding
            except Exception as e:
                retry_count += 1
                if retry_count < MAX_RETRIES:
                    print(f"{Fore.YELLOW}获取嵌入向量失败，尝试重试 ({retry_count}/{MAX_RETRIES}): {str(e)}{Style.RESET_ALL}")
                    await asyncio.sleep(RETRY_DELAY * (2 ** retry_count))  # 指数退避
                else:
                    print(f"{Fore.RED}获取嵌入向量失败，超过最大重试次数: {str(e)}{Style.RESET_ALL}")
                    raise
    
    async def vector_similarity_search(self, query_text: str, top_k: int = 10, node_labels: Optional[List[str]] = None) -> pd.DataFrame:
        """使用向量相似度检索相关内容
        
        Args:
            query_text: 查询文本
            top_k: 返回结果数量
            node_labels: 限制搜索的节点类型
            
        Returns:
            相似节点DataFrame
        """
        # 检查向量搜索条件
        if not self.vector_search_enabled:
            print(f"{Fore.YELLOW}向量检索功能未启用，请设置API密钥{Style.RESET_ALL}")
            return pd.DataFrame()
            
        if not hasattr(self, 'has_vector_nodes') or not self.has_vector_nodes:
            print(f"{Fore.YELLOW}数据库中没有向量属性，将使用备选搜索方法{Style.RESET_ALL}")
            return await self._fallback_keyword_search(query_text, top_k, node_labels)
            
        if not hasattr(self, 'vector_function_available') or not self.vector_function_available:
            print(f"{Fore.YELLOW}向量搜索过程不可用，将使用备选搜索方法{Style.RESET_ALL}")
            return await self._fallback_keyword_search(query_text, top_k, node_labels)
        
        try:
            # 获取查询文本的向量表示
            query_vector = await self.get_embedding(query_text)
            
            # 检查向量索引是否存在
            index_query = "CALL vector_search.show_index_info() YIELD * RETURN *"
            index_results = self.execute_query(index_query)
            
            if index_results.empty:
                print(f"{Fore.YELLOW}未找到向量索引，将使用备选搜索方法{Style.RESET_ALL}")
                return await self._fallback_keyword_search(query_text, top_k, node_labels)
            
            # 使用第一个找到的索引名称
            index_name = index_results.iloc[0]['index_name']
            print(f"{Fore.CYAN}使用向量索引: {index_name}{Style.RESET_ALL}")
            
            # 直接使用向量搜索过程，然后与节点进行匹配
            # 先获取搜索结果
            search_query = f"""
            CALL vector_search.search("{index_name}", $top_k, $query_vector) 
            YIELD * RETURN *
            """
            search_params = {"query_vector": query_vector, "top_k": top_k}
            search_results = self.execute_query(search_query, search_params)
            # search_results = self.execute_query(search_query)
            
            if search_results.empty:
                print(f"{Fore.YELLOW}向量搜索没有返回结果，使用备选搜索方法{Style.RESET_ALL}")
                return await self._fallback_keyword_search(query_text, top_k, node_labels)
            
            # 获取节点ID列表
            node_ids = []
            for node in search_results['node'].tolist():
                # 获取节点ID - 使用py2neo的方式获取节点ID
                node_id = node.identity
                node_ids.append(node_id)
            
            # 打印节点ID列表，但限制长度以避免日志过长
            if len(node_ids) > 5:
                print(f"{Fore.CYAN}调试 - 找到 {len(node_ids)} 个节点, ID示例: {node_ids[:5]}...{Style.RESET_ALL}")
            else:
                print(f"{Fore.CYAN}调试 - 找到 {len(node_ids)} 个节点, ID列表: {node_ids}{Style.RESET_ALL}")
            
            # 向search_results添加node_id列
            search_results['node_id'] = [node.identity for node in search_results['node'].tolist()]
            
            # 移除vector列(如果存在)
            vector_columns = [col for col in search_results.columns if 'vector' in col.lower()]
            if vector_columns:
                search_results = search_results.drop(columns=vector_columns)
                print(f"{Fore.CYAN}从搜索结果中移除向量列: {', '.join(vector_columns)}{Style.RESET_ALL}")
            
            # 处理搜索结果中的节点对象，确保不含向量属性
            if 'node' in search_results.columns:
                # 创建不包含node列的新DataFrame
                columns_to_keep = [col for col in search_results.columns if col != 'node']
                search_results_clean = search_results[columns_to_keep].copy()
                search_results = search_results_clean
            
            node_ids_str = ', '.join([str(nid) for nid in node_ids])
            
            # 构建标签过滤条件
            label_filter = ""
            if node_labels and len(node_labels) > 0:
                label_filter = "AND (" + " OR ".join([f"n:{label}" for label in node_labels]) + ")"
            
            # 获取节点详细信息
            node_query = f"""
            MATCH (n)
            WHERE id(n) IN [{node_ids_str}] {label_filter}
            RETURN id(n) AS node_id,
                CASE WHEN n:Statement THEN 'Statement' 
                     WHEN n:Section THEN 'Section' 
                     WHEN n:Organization THEN 'Organization'
                     WHEN n:Role THEN 'Role'
                     ELSE labels(n)[0] 
                END AS 类型,
                CASE WHEN n:Statement THEN n.name
                     WHEN n:Section THEN n.title
                     WHEN n:Organization THEN n.name
                     WHEN n:Role THEN n.name
                     ELSE COALESCE(n.name, n.title, toString(id(n)))
                END AS 内容
            """
            print(node_query)  
            node_details = self.execute_query(node_query)
            
            if node_details.empty:
                print(f"{Fore.YELLOW}找不到匹配的节点详细信息，使用备选搜索方法{Style.RESET_ALL}")
                return await self._fallback_keyword_search(query_text, top_k, node_labels)
            
            # 合并搜索结果和节点详细信息
            merged_results = pd.merge(node_details, search_results, on='node_id')
            
            # # 添加调试输出
            # print(f"{Fore.CYAN}合并后的结果列: {', '.join(merged_results.columns)}{Style.RESET_ALL}")
            
            # 删除node_id列和vector列
            columns_to_drop = ['node_id']
            if 'vector' in merged_results.columns:
                columns_to_drop.append('vector')
            merged_results = merged_results.drop(columns=columns_to_drop)
            
            # 检查相似度列名
            score_column = None
            for col in merged_results.columns:
                if 'score' in col.lower() or 'similarity' in col.lower() or 'distance' in col.lower():
                    score_column = col
                    break
            
            # 如果没有找到相似度列，使用一个默认值
            if score_column is None:
                print(f"{Fore.YELLOW}没有找到相似度列，添加默认值{Style.RESET_ALL}")
                merged_results['相似度'] = 1.0
            else:
                # 按相似度排序
                print(f"{Fore.CYAN}使用列 '{score_column}' 作为相似度{Style.RESET_ALL}")
                merged_results = merged_results.sort_values(score_column, ascending=False)
                merged_results = merged_results.rename(columns={score_column: '相似度'})
            
            return merged_results
            
        except Exception as e:
            print(f"{Fore.RED}向量相似度查询出错: {str(e)}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}使用备选搜索方法{Style.RESET_ALL}")
            return await self._fallback_keyword_search(query_text, top_k, node_labels)
    
    async def _fallback_keyword_search(self, query_text: str, top_k: int = 10, node_labels: Optional[List[str]] = None) -> pd.DataFrame:
        """当向量搜索不可用时的备选关键词搜索方法
        
        Args:
            query_text: 查询文本
            top_k: 返回结果数量
            node_labels: 限制搜索的节点类型
            
        Returns:
            相似节点DataFrame
        """
        print(f"{Fore.CYAN}使用基于关键词的备选搜索方法{Style.RESET_ALL}")
        
        # 更智能的关键词提取 - 针对中文优化
        # 1. 首先尝试基于常见分隔符分词
        raw_keywords = []
        for sep in ['，', ',', '。', '.', '；', ';', '：', ':', '、', ' ']:
            if sep in query_text:
                raw_keywords.extend([k.strip() for k in query_text.split(sep) if k.strip()])
        
        # 如果没有找到分隔符，把整个查询当作一个关键词
        if not raw_keywords:
            raw_keywords = [query_text]
            
        # 2. 再提取2-4个字的关键短语
        keywords = []
        for raw_kw in raw_keywords:
            if 2 <= len(raw_kw) <= 4:
                keywords.append(raw_kw)
            elif len(raw_kw) > 4:
                # 对较长的词进行滑动窗口提取
                for i in range(len(raw_kw) - 1):
                    if i + 3 <= len(raw_kw):
                        keywords.append(raw_kw[i:i+3])
        
        # 3. 针对特定领域术语进行扩展
        domain_terms = {
            "崩溃": ["故障", "失败", "宕机", "中断", "异常", "错误", "不可用"],
            "系统": ["应用", "程序", "软件", "平台", "客票系统", "预订系统"],
            "恢复": ["修复", "还原", "重启", "启动", "处理", "解决", "恢复运行"],
            "服务": ["功能", "运行", "操作", "业务"],
            "应急": ["预案", "处置", "措施", "方案", "应对", "响应"]
        }
        
        expanded_keywords = keywords.copy()
        for kw in keywords:
            # 对每个关键词，检查是否有同义词扩展
            for term, synonyms in domain_terms.items():
                if term in kw:
                    # 将同义词添加到扩展关键词列表
                    expanded_keywords.extend(synonyms)
        
        # 去重
        expanded_keywords = list(set(expanded_keywords))
        
        print(f"{Fore.CYAN}提取的关键词: {', '.join(expanded_keywords)}{Style.RESET_ALL}")
        
        if not expanded_keywords:
            return pd.DataFrame()
            
        # 构建Cypher查询 - 使用OR连接所有关键词条件
        label_filter = ""
        if node_labels and len(node_labels) > 0:
            label_filter = "WHERE " + " OR ".join([f"n:{label}" for label in node_labels])
            
        keyword_conditions = []
        for keyword in expanded_keywords:
            if len(keyword) >= 2:
                # 针对Statement节点的查询
                keyword_conditions.append(f"(n:Statement AND n.name CONTAINS '{keyword}')")
                # 针对Section节点的查询
                keyword_conditions.append(f"(n:Section AND n.title CONTAINS '{keyword}')")
                # 针对其他节点的查询
                keyword_conditions.append(f"(NOT n:Statement AND NOT n:Section AND n.name CONTAINS '{keyword}')")
        
        if not keyword_conditions:
            return pd.DataFrame()
            
        keyword_filter = " OR ".join(keyword_conditions)
        
        # 如果有标签过滤，将其与关键词条件结合
        filter_clause = f"WHERE {keyword_filter}"
        if label_filter:
            filter_clause = f"{label_filter} AND ({keyword_filter})"
        
        cypher_query = f"""
        MATCH (n)
        {filter_clause}
        RETURN 
            CASE WHEN n:Statement THEN 'Statement' 
                 WHEN n:Section THEN 'Section' 
                 WHEN n:Organization THEN 'Organization'
                 WHEN n:Role THEN 'Role'
                 ELSE labels(n)[0] 
            END AS 类型,
            CASE WHEN n:Statement THEN n.name
                 WHEN n:Section THEN n.title
                 WHEN n:Organization THEN n.name
                 WHEN n:Role THEN n.name
                 ELSE COALESCE(n.name, n.title, toString(id(n)))
            END AS 内容,
            1.0 AS 相似度
        LIMIT $top_k
        """
        
        return self.execute_query(cypher_query, {"top_k": top_k})

    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """执行Cypher查询并返回结果
        
        Args:
            query: Cypher查询语句
            params: 查询参数字典
            
        Returns:
            结果DataFrame
        """
        try:
            result = self.graph.run(query, parameters=params or {})
            df = pd.DataFrame([dict(record) for record in result])
            
            # 自动移除vector列
            vector_columns = [col for col in df.columns if 'vector' in col.lower()]
            if vector_columns:
                df = df.drop(columns=vector_columns)
                print(f"{Fore.CYAN}已移除向量列: {', '.join(vector_columns)}{Style.RESET_ALL}")
            
            # # 添加调试输出
            # print(f"{Fore.CYAN}查询执行成功，返回 {len(df)} 条结果{Style.RESET_ALL}")
            # if not df.empty:
            #     print(f"{Fore.CYAN}结果数据示例 (前5行):{Style.RESET_ALL}")
            #     print(tabulate(df.head(5), headers="keys", tablefmt="pretty", showindex=False))
            #     print(f"{Fore.CYAN}数据列: {', '.join(df.columns)}{Style.RESET_ALL}")
                
            return df
        except Exception as e:
            print(f"{Fore.RED}查询执行失败: {e}{Style.RESET_ALL}")
            print(f"查询语句: {query}")
            return pd.DataFrame()

    def display_results(self, df: pd.DataFrame, title: str, export: bool = False) -> None:
        """美观地显示查询结果
        
        Args:
            df: 结果DataFrame
            title: 显示标题
            export: 是否导出结果
        """
        print(f"\n{Fore.CYAN}============ {title} ============{Style.RESET_ALL}")
        
        if df.empty:
            print(f"{Fore.YELLOW}未找到结果{Style.RESET_ALL}")
            return
        
        # 处理过长的文本
        def truncate_text(text, max_len=80):
            if isinstance(text, str) and len(text) > max_len:
                return text[:max_len-3] + "..."
            return text
        
        # 对长文本进行截断处理，避免表格过宽
        display_df = df.copy()
        for col in display_df.columns:
            if display_df[col].dtype == 'object':
                display_df[col] = display_df[col].apply(truncate_text)
        
        # 使用更美观的表格格式
        headers = [f"{Fore.GREEN}{h}{Style.RESET_ALL}" for h in display_df.columns]
        
        # 使用fancy_grid格式，增加表格边框清晰度
        table = tabulate(
            display_df, 
            headers=headers, 
            tablefmt="fancy_grid", 
            showindex=False,
            numalign="left",
            stralign="left"
        )
        
        # 给表格的行添加交替颜色，增强可读性
        lines = table.split('\n')
        formatted_lines = []
        data_line_idx = 0
        
        for i, line in enumerate(lines):
            # 表头和分隔符保持原样
            if '═' in line or '╒' in line or '╕' in line or '╘' in line or '╛' in line or '╞' in line or '╡' in line:
                formatted_lines.append(line)
            else:
                # 数据行使用交替色
                color = Fore.CYAN if data_line_idx % 2 == 0 else Fore.WHITE
                formatted_lines.append(f"{color}{line}{Style.RESET_ALL}")
                data_line_idx += 1
        
        print('\n'.join(formatted_lines))
        print(f"\n{Fore.YELLOW}共 {len(df)} 条结果{Style.RESET_ALL}\n")
        
        if export and not df.empty:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.export_dir}/{title.replace(' ', '_')}_{timestamp}.csv"
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f"{Fore.GREEN}结果已导出至: {filename}{Style.RESET_ALL}")

    async def semantic_keyword_search(self, keyword: str, export: bool = False) -> None:
        """使用语义向量搜索，根据用户提供的关键词检索知识图谱
        
        应用场景: 更智能地查找与用户意图相关的内容，即使没有包含完全相同的关键词
        
        Args:
            keyword: 搜索关键词
        """
        print(f"{Fore.CYAN}正在进行语义向量搜索: '{keyword}'...{Style.RESET_ALL}")
        results = await self.vector_similarity_search(keyword, top_k=20)
        self.display_results(results, f"'{keyword}'的语义检索结果", export)
        
        # 如果找到结果中包含Statement，尝试查找相关章节和部门
        if not results.empty:
            statements = results[results['类型'] == 'Statement']
            if not statements.empty:
                # 仅使用前3个最相关的Statement查找详细信息
                stmt_contents = statements['内容'].head(3).tolist()
                stmt_params = ', '.join([f"'{s}'" for s in stmt_contents])
                
                detail_query = f"""
                MATCH (st:Statement)
                WHERE st.name IN [{stmt_params}]
                OPTIONAL MATCH (s:Section)-[:CONTAINS]->(st)
                OPTIONAL MATCH (org)-[:RESPONSIBLE_FOR]->(st)
                WHERE org:Organization
                RETURN st.name AS 内容, s.title AS 所在章节, collect(DISTINCT org.name) AS 负责部门
                """
                detail_results = self.execute_query(detail_query)
                if not detail_results.empty:
                    self.display_results(detail_results, f"'{keyword}'的相关章节和责任部门", export)

    # ========== 场景1: 应急响应人员查询（增加向量检索功能）==========
    async def find_system_maintainers(self, export: bool = False) -> None:
        """查询负责客票发售和预订系统的部门和角色
        
        应用场景: 系统发生故障时，急需确定负责处理的部门和角色
        """
        # 原有的关键词查询
        query = """
        MATCH (actor)-[:RESPONSIBLE_FOR]->(st:Statement)
        WHERE st.name CONTAINS "客票发售和预订系统" OR st.name CONTAINS "客票系统"
        AND (actor:Organization OR actor:Role)
        RETURN labels(actor)[0] AS 责任主体类型, actor.name AS 责任主体名称
        """
        results = self.execute_query(query)
        self.display_results(results, "系统故障时的责任主体", export)
        
        # 补充向量相似度查询
        if self.vector_search_enabled:
            semantic_query = "客票发售和预订系统的负责部门和人员"
            semantic_results = await self.vector_similarity_search(
                semantic_query, 
                top_k=10,
                node_labels=["Organization", "Role"]
            )
            
            if not semantic_results.empty:
                self.display_results(
                    semantic_results, 
                    "系统故障时的责任主体（语义检索补充）", 
                    export
                )
    
    # ========== 场景2: 旅客投诉处理流程查询（未找到结果，增加向量检索）==========
    async def find_complaint_procedures(self, export: bool = False) -> None:
        """查询与旅客投诉处理相关的所有规定和流程
        
        应用场景: 大量旅客投诉涌入时，查询应急预案中的处理程序
        """
        # 原有的关键词查询
        query = """
        MATCH (st:Statement)
        WHERE st.name CONTAINS "旅客" OR st.name CONTAINS "投诉" 
              OR st.name CONTAINS "乘客" OR st.name CONTAINS "服务质量"
              OR st.name CONTAINS "客服" OR st.name CONTAINS "应对旅客"
              OR st.name CONTAINS "退票" OR st.name CONTAINS "改签"
        RETURN st.name AS 投诉处理相关规定
        """
        results = self.execute_query(query)
        self.display_results(results, "旅客投诉处理相关规定", export)
        
        # 补充向量相似度查询
        if self.vector_search_enabled:
            semantic_query = "如何处理旅客投诉和退改签申请的流程和规定"
            semantic_results = await self.vector_similarity_search(
                semantic_query, 
                top_k=10,
                node_labels=["Statement"]
            )
            
            if not semantic_results.empty:
                filtered_results = semantic_results[semantic_results['类型'] == 'Statement']
                if not filtered_results.empty:
                    self.display_results(
                        filtered_results, 
                        "旅客投诉处理相关规定（语义检索补充）", 
                        export
                    )
    
    # ========== 场景3: 值班负责人职责查询（未找到结果，改为查询所有角色）==========
    def find_roles_and_responsibilities(self, export: bool = False) -> None:
        """查询预案中定义的所有角色及其职责
        
        应用场景: 了解预案中涉及的各类人员角色及其职责
        """
        query = """
        MATCH (r:Role)
        OPTIONAL MATCH (r)-[:RESPONSIBLE_FOR]->(st:Statement)
        RETURN r.name AS 角色名称, collect(st.name) AS 负责事项
        UNION
        MATCH (r:Role)
        OPTIONAL MATCH (st:Statement)-[:APPLIES_TO]->(r)
        RETURN r.name AS 角色名称, collect(st.name) AS 相关规定
        """
        results = self.execute_query(query)
        self.display_results(results, "角色及职责", export)
        
        # 如果没有找到角色，尝试查询与"值班"相关的Statement
        if results.empty:
            query = """
            MATCH (st:Statement)
            WHERE st.name CONTAINS "值班" OR st.name CONTAINS "当班" OR st.name CONTAINS "负责人"
            RETURN st.name AS 值班相关规定
            """
            results = self.execute_query(query)
            self.display_results(results, "值班相关规定", export)
    
    # ========== 场景4: 网络故障相关章节查询（返回空值，增强查询）==========
    def find_failure_related_content(self, export: bool = False) -> None:
        """查询与各类故障相关的内容
        
        应用场景: 系统或网络出现问题时，查找相关处理指导
        """
        # 先查找相关Statement
        query = """
        MATCH (st:Statement)
        WHERE st.name CONTAINS "网络" OR st.name CONTAINS "故障" 
              OR st.name CONTAINS "中断" OR st.name CONTAINS "系统异常"
              OR st.name CONTAINS "应急处置" OR st.name CONTAINS "恢复"
        RETURN st.name AS 故障相关规定
        """
        print(query)
        results = self.execute_query(query)
        self.display_results(results, "故障相关规定", export)
        
        # 再查找这些Statement所在的Section
        if not results.empty:
            statement_list = results['故障相关规定'].tolist()
            statement_params = ', '.join([f'"{s}"' for s in statement_list[:5]])  # 限制数量避免查询过大
            
            query = f"""
            MATCH (s:Section)-[:CONTAINS]->(st:Statement)
            WHERE st.name IN [{statement_params}]
            RETURN DISTINCT s.id AS 章节ID, s.title AS 章节标题, collect(st.name) AS 包含的相关规定
            """
            section_results = self.execute_query(query)
            self.display_results(section_results, "故障相关章节", export)
    
    # ========== 场景5: 跨部门协作程序（优化查询策略）==========
    def find_cooperation_procedures(self, export: bool = False) -> None:
        """查询涉及多个部门的应急措施
        
        应用场景: 识别需要多部门配合的复杂程序，提前做好协调准备
        """
        # 查询被多个部门引用的Statement
        query = """
        MATCH (st:Statement)
        WITH st, [(st)<-[:RESPONSIBLE_FOR]-(o:Organization) | o.name] AS responsible_orgs
        WHERE size(responsible_orgs) > 1
        RETURN st.name AS 协作程序, responsible_orgs AS 相关部门
        UNION
        MATCH (st:Statement)
        WITH st, [(st)-[:APPLIES_TO]->(o:Organization) | o.name] AS applicable_orgs
        WHERE size(applicable_orgs) > 1
        RETURN st.name AS 协作程序, applicable_orgs AS 相关部门
        """
        print(query)
        results = self.execute_query(query)
        self.display_results(results, "跨部门协作程序", export)
        
        # 如果上面的查询没有结果，查找提及多个部门的Statement
        if results.empty:
            # 先获取所有部门名称
            orgs_query = """
            MATCH (o:Organization)
            RETURN o.name AS dept_name
            """
            orgs_df = self.execute_query(orgs_query)
            
            if not orgs_df.empty:
                # 构建一个查询，查找Statement中包含多个部门名称的情况
                # 为简化，这里只选取最常见的几个部门
                top_depts = orgs_df['dept_name'].tolist()[:10]  # 取前10个部门
                
                mentions_conditions = []
                for dept in top_depts:
                    if len(dept) > 1:  # 避免单字部门名引起的误匹配
                        mentions_conditions.append(f'st.name CONTAINS "{dept}"')
                
                if mentions_conditions:
                    combined_query = """
                    MATCH (st:Statement)
                    WHERE """ + " OR ".join(mentions_conditions) + """
                    RETURN st.name AS 可能的协作规定
                    """
                    mention_results = self.execute_query(combined_query)
                    self.display_results(mention_results, "可能提及多部门的规定", export)
    
    # ========== 场景6: 部门职责统计（工作良好，增加可视化）==========
    def count_department_responsibilities(self, export: bool = False) -> None:
        """统计各部门在应急预案中负责的规定数量
        
        应用场景: 评估各部门在应急响应中的职责权重和工作量
        """
        query = """
        MATCH (o:Organization)-[:RESPONSIBLE_FOR]->(st:Statement)
        RETURN o.name AS 部门名称, count(st) AS 负责规定数量
        ORDER BY 负责规定数量 DESC
        """
        print(query)
        results = self.execute_query(query)
        self.display_results(results, "各部门职责统计", export)
        
        # 为前10个部门生成负责的具体规定
        if not results.empty:
            top_depts = results['部门名称'].head(10).tolist()
            
            for dept in top_depts:
                detail_query = f"""
                MATCH (o:Organization {{name: "{dept}"}})-[:RESPONSIBLE_FOR]->(st:Statement)
                RETURN st.name AS 具体规定
                LIMIT 20
                """
                dept_details = self.execute_query(detail_query)
                if not dept_details.empty:
                    self.display_results(dept_details, f"{dept}的具体职责", export)
    
    # ========== 场景7: 数据与系统恢复相关流程（扩大关键词范围）==========
    def find_system_recovery_processes(self, export: bool = False) -> None:
        """查询与系统恢复相关的流程及责任部门
        
        应用场景: 系统故障后，查找恢复流程和方法
        """
        query = """
        MATCH (st:Statement)
        WHERE st.name CONTAINS "恢复" OR st.name CONTAINS "备份" 
              OR st.name CONTAINS "容灾" OR st.name CONTAINS "系统还原"
              OR st.name CONTAINS "数据保护" OR st.name CONTAINS "故障处理"
              OR st.name CONTAINS "应急响应" OR st.name CONTAINS "恢复运行"
        OPTIONAL MATCH (actor)-[:RESPONSIBLE_FOR]->(st)
        WHERE actor:Organization
        RETURN st.name AS 系统恢复规定, collect(DISTINCT actor.name) AS 责任部门
        """
        print(query)
        results = self.execute_query(query)
        self.display_results(results, "系统恢复相关流程", export)
    
    # ========== 场景8: 预案结构查询（改为查找所有文档和章节）==========
    def browse_document_structure(self, export: bool = False) -> None:
        """查询知识图谱中的所有文档及其章节结构
        
        应用场景: 了解预案整体结构和组织
        """
        # 找出所有Document节点
        query = """
        MATCH (d:Document)
        RETURN d.title AS 文档标题
        """
        docs = self.execute_query(query)
        self.display_results(docs, "知识图谱中的文档", export)
        
        # 查询章节结构
        query = """
        MATCH (d:Document)-[:HAS_SECTION]->(s:Section)
        RETURN d.title AS 文档标题, s.id AS 章节ID, s.title AS 章节标题
        ORDER BY d.title, s.id
        """
        print(query)
        sections = self.execute_query(query)
        self.display_results(sections, "文档章节结构", export)
        
        # 如果没有获取到足够的章节信息，尝试直接查找所有Section
        if sections.empty or len(sections) < 5:
            query = """
            MATCH (s:Section)
            OPTIONAL MATCH (s)-[:HAS_PARENT_SECTION]->(parent:Section)
            RETURN s.id AS 章节ID, s.title AS 章节标题, 
                   parent.id AS 父章节ID, parent.title AS 父章节标题
            ORDER BY s.id
            """
            print(query)
            all_sections = self.execute_query(query)
            self.display_results(all_sections, "所有章节信息", export)
    
    # ========== 新增场景9: 抽取预案中的应急级别与响应步骤 ==========
    def extract_emergency_levels_and_steps(self, export: bool = False) -> None:
        """抽取预案中定义的应急级别和关键响应步骤
        
        应用场景: 了解不同等级故障的处理流程和时间要求
        """
        # 查询与应急级别相关的Statement
        query = """
        MATCH (st:Statement)
        WHERE st.name CONTAINS "级别" OR st.name CONTAINS "等级" 
              OR st.name CONTAINS "一级" OR st.name CONTAINS "二级" 
              OR st.name CONTAINS "三级" OR st.name CONTAINS "四级"
              OR st.name CONTAINS "重大" OR st.name CONTAINS "严重"
              OR st.name CONTAINS "较大" OR st.name CONTAINS "一般"
        RETURN st.name AS 应急级别描述
        """
        print(query)
        level_results = self.execute_query(query)
        self.display_results(level_results, "应急级别定义", export)
        
        # 查询与响应步骤相关的Statement
        query = """
        MATCH (st:Statement)
        WHERE st.name CONTAINS "步骤" OR st.name CONTAINS "流程" 
              OR st.name CONTAINS "程序" OR st.name CONTAINS "措施"
              OR st.name CONTAINS "处置" OR st.name CONTAINS "响应"
              OR st.name CONTAINS "先" OR st.name CONTAINS "后"
        RETURN st.name AS 响应步骤描述
        """
        print(query)
        step_results = self.execute_query(query)
        self.display_results(step_results, "应急响应步骤", export)
    
    # ========== 新增场景10: 关键词全文搜索 ==========
    def keyword_search(self, keyword: str, export: bool = False) -> None:
        """根据用户提供的关键词，全文搜索知识图谱
        
        应用场景: 灵活查询特定关键词相关的所有内容
        
        Args:
            keyword: 搜索关键词
        """
        # 在Statement中搜索
        query = f"""
        MATCH (st:Statement)
        WHERE st.name CONTAINS "{keyword}"
        RETURN "Statement" AS 实体类型, st.name AS 内容
        
        UNION
        
        MATCH (o:Organization)
        WHERE o.name CONTAINS "{keyword}"
        RETURN "Organization" AS 实体类型, o.name AS 内容
        
        UNION
        
        MATCH (r:Role)
        WHERE r.name CONTAINS "{keyword}"
        RETURN "Role" AS 实体类型, r.name AS 内容
        
        UNION
        
        MATCH (s:Section)
        WHERE s.title CONTAINS "{keyword}"
        RETURN "Section" AS 实体类型, s.title AS 内容
        
        UNION
        
        MATCH (d:Document)
        WHERE d.title CONTAINS "{keyword}"
        RETURN "Document" AS 实体类型, d.title AS 内容
        """

        print(query)
        results = self.execute_query(query)
        self.display_results(results, f"关键词'{keyword}'搜索结果", export)
        
        # 如果找到的Statement不为空，查找其所在章节和负责部门
        if not results.empty:
            st_results = results[results['实体类型'] == 'Statement']
            if not st_results.empty:
                # 取前5个Statement，避免查询过大
                statements = st_results['内容'].head(5).tolist()
                statement_params = ', '.join([f'"{s}"' for s in statements])
                
                detail_query = f"""
                MATCH (st:Statement)
                WHERE st.name IN [{statement_params}]
                OPTIONAL MATCH (s:Section)-[:CONTAINS]->(st)
                OPTIONAL MATCH (org)-[:RESPONSIBLE_FOR]->(st)
                WHERE org:Organization
                RETURN st.name AS 内容, s.title AS 所在章节, collect(DISTINCT org.name) AS 负责部门
                """
                print(query)
                detail_results = self.execute_query(detail_query)
                self.display_results(detail_results, f"关键词'{keyword}'详细信息", export)

    # ========== 新增场景12: 查询特定故障的处理链条 ==========
    def find_fault_handling_chain(self, fault_keyword: str, export: bool = False) -> None:
        """查询特定故障相关的章节、恢复步骤及责任人
        
        应用场景: 快速定位某个故障的处理上下文、恢复方法及负责人
        
        Args:
            fault_keyword: 描述故障的关键词 (例如 "数据库连接", "网络中断")
            export: 是否导出结果
        """
        print(f"{Fore.CYAN}正在查询与故障 '{fault_keyword}' 相关的处理链条...{Style.RESET_ALL}")
        
        # 查找包含关键词的故障Statement，找到其所在Section，再找Section下的恢复Statement及其责任人
        query = """
        MATCH (fault_st:Statement)
        WHERE fault_st.name CONTAINS $fault_keyword
        WITH fault_st LIMIT 5 // 限制初始故障Statement数量，防止结果过多
        
        MATCH (s:Section)-[:CONTAINS]->(fault_st) // 找到故障所在章节
        MATCH (s)-[:CONTAINS]->(recovery_st:Statement) // 找到同章节下其他Statement
        WHERE (recovery_st.name CONTAINS "恢复" OR recovery_st.name CONTAINS "修复" 
               OR recovery_st.name CONTAINS "启动" OR recovery_st.name CONTAINS "处理") // 筛选恢复相关的步骤
          AND recovery_st <> fault_st // 排除故障本身
        
        OPTIONAL MATCH (actor)-[:RESPONSIBLE_FOR]->(recovery_st) // 找到恢复步骤的责任人
        WHERE actor:Organization OR actor:Role
        
        RETURN fault_st.name AS 相关故障描述, 
               s.title AS 所在章节, 
               recovery_st.name AS 相关恢复步骤, 
               labels(actor)[0] AS 责任主体类型, 
               actor.name AS 责任主体名称
        ORDER BY s.title, recovery_st.name
        LIMIT 30 // 限制最终结果数量
        """
        print(query)
        results = self.execute_query(query, params={"fault_keyword": fault_keyword})
        self.display_results(results, f"故障 '{fault_keyword}' 的处理链条", export)

    # ========== 新增场景13: 查询共同负责领域事务的部门与章节 ==========
    def find_shared_responsibility_areas(self, domain_keyword: str, export: bool = False) -> None:
        """查询共同负责特定领域事务的部门与章节
        
        应用场景: 识别跨部门协作的关键领域和相关文档章节
        
        Args:
            domain_keyword: 描述事务领域的关键词 (例如 "恢复", "备份", "投诉")
            export: 是否导出结果
        """
        print(f"{Fore.CYAN}正在查询负责 '{domain_keyword}' 事务的部门与章节...{Style.RESET_ALL}")
        
        # 查找包含领域关键词的Statement，通过CONTAINS找到Section，通过RESPONSIBLE_FOR找到Organization
        # 然后按Section和Organization分组，聚合共同负责的Statement
        query = """
        MATCH (s:Section)-[:CONTAINS]->(st:Statement)<-[:RESPONSIBLE_FOR]-(o:Organization)
        WHERE st.name CONTAINS $domain_keyword // 筛选特定领域的Statement
        
        WITH s, o, collect(st.name) AS procedures // 按章节和部门聚合规定
        WHERE size(procedures) >= 1 // 确保至少有一个相关规定
        
        RETURN s.title AS 章节标题, 
               o.name AS 负责部门, 
               size(procedures) AS 相关规定数量,
               procedures AS 相关规定列表 // 显示具体规定列表
        ORDER BY size(procedures) DESC, s.title, o.name
        LIMIT 30 // 限制结果数量
        """
        print(query)
        results = self.execute_query(query, params={"domain_keyword": domain_keyword})
        self.display_results(results, f"共同负责 '{domain_keyword}' 事务的部门与章节", export)

async def main():
    """主函数: 解析命令行参数并执行相应的查询"""
    parser = argparse.ArgumentParser(description="客票发售和预订系统应急预案知识图谱查询工具")
    
    # 添加数据库连接参数
    parser.add_argument("--uri", default="bolt://localhost:7687", help="图数据库URI")
    parser.add_argument("--user", default="neo4j", help="数据库用户名")
    parser.add_argument("--password", default="password", help="数据库密码")
    parser.add_argument("--export", action="store_true", help="是否导出查询结果为CSV文件")
    parser.add_argument("--export-dir", default="./kg_exports", help="导出目录")
    
    # 添加查询场景选择 (范围更新到13)
    parser.add_argument("--scenario", type=int, choices=range(1, 14), 
                        help="要执行的场景编号(1-13)，不提供则执行所有场景")
    
    # 添加关键词搜索参数 (需要为新场景指定关键词)
    parser.add_argument("--keyword", type=str, help="关键词搜索(用于场景10, 11)")
    parser.add_argument("--fault-keyword", type=str, help="故障关键词(用于场景12)")
    parser.add_argument("--domain-keyword", type=str, help="领域关键词(用于场景13)")
    parser.add_argument("--semantic", action="store_true", help="使用语义向量搜索(用于场景11)")
    
    args = parser.parse_args()
    
    # 初始化知识图谱查询工具
    kg = EmergencyResponseKG(uri=args.uri, user=args.user, password=args.password, 
                            export_dir=args.export_dir)
    
    # 定义所有场景 (增加12, 13)
    scenarios = {
        1: (kg.find_system_maintainers, "查询系统故障责任主体"),
        2: (kg.find_complaint_procedures, "查询旅客投诉处理相关规定"),
        3: (kg.find_roles_and_responsibilities, "查询角色及职责"),
        4: (kg.find_failure_related_content, "查询故障相关内容"),
        5: (kg.find_cooperation_procedures, "查询跨部门协作程序"),
        6: (kg.count_department_responsibilities, "统计各部门职责"),
        7: (kg.find_system_recovery_processes, "查询系统恢复相关流程"),
        8: (kg.browse_document_structure, "浏览预案结构"),
        9: (kg.extract_emergency_levels_and_steps, "抽取应急级别与响应步骤"),
        10: (lambda: kg.keyword_search(args.keyword or "应急", args.export), "关键词搜索"),
        11: (kg.semantic_keyword_search, "语义向量搜索"),
        12: (lambda: kg.find_fault_handling_chain(args.fault_keyword or "数据库", args.export), "查询特定故障处理链条"),
        13: (lambda: kg.find_shared_responsibility_areas(args.domain_keyword or "恢复售票", args.export), "查询共同负责领域事务的部门与章节")
    }
    
    # 执行选定的场景或所有场景
    if args.scenario:
        # 为需要关键词的场景获取输入
        if args.scenario == 10 and not args.keyword:
            print(f"{Fore.YELLOW}场景10需要提供 --keyword 参数{Style.RESET_ALL}")
            args.keyword = input("请输入搜索关键词: ")
        elif args.scenario == 11 and not args.keyword:
            print(f"{Fore.YELLOW}场景11需要提供 --keyword 参数{Style.RESET_ALL}")
            args.keyword = input("请输入语义搜索关键词: ")
        elif args.scenario == 12 and not args.fault_keyword:
            print(f"{Fore.YELLOW}场景12需要提供 --fault-keyword 参数{Style.RESET_ALL}")
            args.fault_keyword = input("请输入故障关键词 (例如 '数据库', '网络'): ")
        elif args.scenario == 13 and not args.domain_keyword:
            print(f"{Fore.YELLOW}场景13需要提供 --domain-keyword 参数{Style.RESET_ALL}")
            args.domain_keyword = input("请输入领域关键词 (例如 '恢复', '备份'): ")
            
        func, desc = scenarios[args.scenario]
        print(f"{Fore.YELLOW}执行场景 {args.scenario}: {desc}{Style.RESET_ALL}")
        
        # 处理同步/异步函数调用
        # 注意: 需要将新场景的 lambda 包装改为非异步，因为新添加的方法是同步的
        # 或者将新方法改为 async (如果它们内部有 await 调用，但目前没有)
        # 这里我们假设新方法是同步的
        if asyncio.iscoroutinefunction(func):
             # 异步场景调用 (目前只有1, 2, 11是异步的)
            if args.scenario == 11:
                 await func(args.keyword, args.export) # 场景11需要keyword
            else:
                 await func(args.export) # 场景1和2需要export参数
        else:
             # 同步场景调用
             func() # 其他场景包括新的12, 13
    else:
        print(f"{Fore.YELLOW}执行所有应用场景 (1-13){Style.RESET_ALL}")
        # 简化为仅执行几个核心场景 + 新场景，避免输出过多
        # core_scenarios = [1, 5, 11, 12, 13] # 选择部分场景执行
        core_scenarios = list(scenarios.keys()) # 或者执行全部
        
        default_keyword = "应急"
        default_fault_keyword = "数据库"
        default_domain_keyword = "恢复"
        
        for i in core_scenarios:
            func, desc = scenarios[i]
            print(f"\n{Fore.MAGENTA}--- 执行场景 {i}: {desc} ---{Style.RESET_ALL}")
            
            try:
                 if asyncio.iscoroutinefunction(func):
                     # 异步场景调用
                     if i == 11:
                         await func(args.keyword or default_keyword, args.export)
                     else:
                         await func(args.export)
                 else:
                     # 同步场景调用 (包括 10, 12, 13 等)
                     if i == 10:
                         func() # 使用默认关键词或命令行参数
                     elif i == 12:
                          # 需要正确调用lambda
                         scenarios[i][0]() # 调用lambda (使用默认或命令行参数)
                     elif i == 13:
                          # 需要正确调用lambda
                         scenarios[i][0]() # 调用lambda (使用默认或命令行参数)
                     else:
                         func() # 其他同步场景
            except Exception as e:
                print(f"{Fore.RED}执行场景 {i} 时出错: {e}{Style.RESET_ALL}")

    print(f"\n{Fore.GREEN}所有查询已完成{Style.RESET_ALL}")

if __name__ == "__main__":
    # 注意：如果KG类中有async方法，并且main中有同步方法调用async方法，
    # 或者反之，可能需要调整事件循环的处理。
    # 目前的设计是main是async，它调用同步或异步的KG方法。
    # 调用同步方法没问题，调用异步方法用 await。
    # 在同步方法中如果需要调用异步方法，则需要特殊处理（如 asyncio.run_coroutine_threadsafe）。
    # 目前添加的方法是同步的，应该没问题。
    asyncio.run(main())