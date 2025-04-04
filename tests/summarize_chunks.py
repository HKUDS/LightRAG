import json
import os
from openai import OpenAI
import sys
from typing import Dict, List, Any
import time
import re
import logging
from dotenv import load_dotenv
from tqdm import tqdm

# 设置日志格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 加载.env文件中的环境变量
load_dotenv()

# 常量定义
MAX_CONTENT_LENGTH_FOR_DIRECT_SUMMARY = 15000
API_CALL_DELAY_SECONDS = 1
HEADING_SUMMARY_MAX_TOKENS = 30
DEFAULT_SUMMARY_MAX_TOKENS = 200
CHUNK_SIZE = 12000

# 提示词
HEADING_GENERATION_PROMPT = "你是一个标题生成器。请为以下法规内容生成一个简明扼要的标题（不超过15字），标题应当概括内容的主旨。"
LEVEL_2_SUMMARY_PROMPT = """请对以下内容生成一个全面但简洁的摘要。
这些内容是一个文档的多个部分的摘要集合，请综合这些摘要生成一个更高层次的摘要。
摘要应当涵盖所有重要信息，但避免不必要的细节。

内容如下：
"""
LEVEL_1_SUMMARY_PROMPT = """请基于以下二级文档摘要，生成一个全面但简洁的高级摘要。
这些摘要代表了整个文档的不同部分，你需要提取并综合其中的核心信息。
摘要应当抓住文档的主要内容和关键点，展现其整体框架和主旨。

内容如下：
"""

# 预编译正则表达式
LEVEL_1_PATTERN = re.compile(r'[a-f0-9]+_chunk_1_\d+(?:_\d+)*')
LEVEL_2_PATTERN = re.compile(r'[a-f0-9]+_chunk_2_\d+(?:_\d+)*')
LEVEL_3_PATTERN = re.compile(r'[a-f0-9]+_chunk_3_\d+(?:_\d+)*')
PAGE_NUMBER_PATTERN = re.sub(r'\s*\(p\.\d+\)\s*$', '', '')
REGULATION_PATTERN = re.compile(r'^(第[一二三四五六七八九十百千万零〇]+条)\s')

# 初始化OpenAI客户端
LLM_API_KEY = os.environ.get("LLM_BINDING_API_KEY") or os.environ.get("SILICONFLOW_API_KEY")
LLM_API_HOST = os.environ.get("LLM_BINDING_HOST")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-3.5-turbo-16k")  # 默认使用gpt-3.5-turbo-16k

# 初始化客户端，优先使用LLM_API_KEY，其次使用OPENAI_API_KEY
client_params = {
    "api_key": LLM_API_KEY or os.environ.get("OPENAI_API_KEY")
}

# 如果设置了API主机地址，添加到客户端参数中
if LLM_API_HOST:
    client_params["base_url"] = LLM_API_HOST

client = OpenAI(**client_params)

def call_llm_for_summary(content: str, max_tokens: int = DEFAULT_SUMMARY_MAX_TOKENS, system_prompt: str = "请对以下文本生成摘要，要求内容精准和把握实质，摘要字数不超过200字。") -> str:
    """
    调用LLM生成摘要
    
    Args:
        content: 需要摘要的内容
        max_tokens: 摘要最大token数
        system_prompt: 系统提示词
        
    Returns:
        生成的摘要文本
    """
    if not content:
        return "无内容，无法生成摘要"
    
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,  # 使用环境变量中设置的模型
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content}
            ],
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"调用LLM出错: {e}")
        return f"摘要生成失败: {str(e)}"

def chunk_content(content: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """
    当内容太长时，将内容分块处理
    
    Args:
        content: 需要分块的内容
        chunk_size: 每块的大小(字符数)
        
    Returns:
        分块后的内容列表
    """
    if len(content) <= chunk_size:
        return [content]
    
    # 按段落分割
    paragraphs = content.split("\n\n")
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        # 如果当前段落加上已有内容超过限制，先保存当前块
        if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
            chunks.append(current_chunk)
            current_chunk = paragraph
        else:
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
    
    # 添加最后一块
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def summarize_long_content(content: str) -> str:
    """
    对较长内容进行分块摘要后再综合摘要
    
    Args:
        content: 原始内容
        
    Returns:
        最终摘要
    """
    # 分块处理
    chunks = chunk_content(content)
    
    # 如果只有一块，直接摘要
    if len(chunks) == 1:
        return call_llm_for_summary(content)
    
    # 对每块生成摘要
    chunk_summaries = []
    for i, chunk in enumerate(tqdm(chunks, desc="处理长内容分块")):
        logging.info(f"正在处理第{i+1}/{len(chunks)}块长内容...")
        summary = call_llm_for_summary(chunk)
        chunk_summaries.append(summary)
        # 避免API调用过快
        time.sleep(API_CALL_DELAY_SECONDS)
    
    # 合并摘要后再次摘要
    combined_summaries = "\n\n".join(chunk_summaries)
    final_summary = call_llm_for_summary(combined_summaries)
    
    return final_summary

def clean_heading(heading: str, content: str = "") -> str:
    """
    清洗heading文本，包括去除页码标签和为法规类内容生成简明标题
    
    Args:
        heading: 原始heading文本
        content: 相关联的内容文本，用于生成摘要标题
        
    Returns:
        清洗后的heading文本
    """
    if not heading:
        return ""
        
    # 去除页码标签 (p.xx)
    cleaned_heading = re.sub(r'\s*\(p\.\d+\)\s*$', '', heading)
    
    # 检测是否是"第**条"开头的法规内容
    if re.match(r'^第[一二三四五六七八九十百千万零〇]+条\s', cleaned_heading):
        # 如果内容较短，直接使用
        if len(cleaned_heading) < 30:
            return cleaned_heading
        
        try:
            # 调用LLM生成简明标题
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": HEADING_GENERATION_PROMPT},
                    {"role": "user", "content": cleaned_heading + "\n" + content}
                ],
                max_tokens=HEADING_SUMMARY_MAX_TOKENS
            )
            
            # 获取生成的标题
            generated_title = response.choices[0].message.content.strip()
            
            # 保留原始条款编号，并添加生成的标题
            regulation_number = re.match(r'^(第[一二三四五六七八九十百千万零〇]+条)\s', cleaned_heading).group(1)
            final_heading = f"{regulation_number} {generated_title}"
            
            return final_heading
        except Exception as e:
            logging.error(f"生成标题时出错: {e}")
            return cleaned_heading
    
    return cleaned_heading

def clean_all_headings(chunks: List[Dict], chunk_map: Dict) -> None:
    """
    清洗所有块的heading
    
    Args:
        chunks: 块列表
        chunk_map: chunk_id到chunk的映射
    """
    logging.info("开始清洗所有块的heading...")
    for chunk in tqdm(chunks, desc="清洗heading"):
        if "heading" in chunk:
            content = chunk.get("content", "")
            chunk["heading"] = clean_heading(chunk["heading"], content)
            # 更新chunk_map
            chunk_map[chunk['chunk_id']] = chunk

def summarize_level_3_chunks(chunks: List[Dict], chunk_map: Dict) -> None:
    """
    处理三级文本块，生成摘要
    
    Args:
        chunks: 块列表
        chunk_map: chunk_id到chunk的映射
    """
    level_3_chunks = [chunk for chunk in chunks if LEVEL_3_PATTERN.match(chunk['chunk_id'])]
    logging.info(f"开始处理{len(level_3_chunks)}个三级文本块...")
    
    for i, chunk in enumerate(tqdm(level_3_chunks, desc="处理三级块")):
        # 如果已经有摘要，则跳过
        if 'summary' in chunk and chunk['summary']:
            continue
            
        content = chunk.get('content', '')
        if not content:
            chunk['summary'] = "无内容，无法生成摘要"
            chunk_map[chunk['chunk_id']] = chunk
            continue
        
        # 生成摘要
        try:
            summary = call_llm_for_summary(content)
            chunk['summary'] = summary
            chunk_map[chunk['chunk_id']] = chunk
            logging.info(f"三级块摘要生成完成: {chunk['chunk_id']}")
            
            # 避免API调用过快
            time.sleep(API_CALL_DELAY_SECONDS)
        except Exception as e:
            logging.error(f"处理三级块{chunk['chunk_id']}时出错: {e}")
            chunk['summary'] = f"摘要生成失败: {str(e)}"
            chunk_map[chunk['chunk_id']] = chunk

def summarize_level_2_chunks(chunks: List[Dict], chunk_map: Dict) -> None:
    """
    处理二级文本块，生成摘要
    
    Args:
        chunks: 块列表
        chunk_map: chunk_id到chunk的映射
    """
    level_2_chunks = [chunk for chunk in chunks if LEVEL_2_PATTERN.match(chunk['chunk_id'])]
    logging.info(f"开始处理{len(level_2_chunks)}个二级文本块...")
    
    for i, chunk in enumerate(tqdm(level_2_chunks, desc="处理二级块")):
        logging.info(f"处理二级块 {i+1}/{len(level_2_chunks)}: {chunk['chunk_id']}")
        
        # 获取该二级块的所有子块（三级块）
        child_ids = chunk.get('child_ids', [])
        # 过滤出三级块的ID
        level_3_child_ids = [child_id for child_id in child_ids if child_id in chunk_map and LEVEL_3_PATTERN.match(child_id)]
        child_chunks = [chunk_map.get(child_id) for child_id in level_3_child_ids if chunk_map.get(child_id)]
        
        # 如果没有子块，跳过
        if not child_chunks:
            chunk['summary'] = "无子文档内容，无法生成摘要"
            chunk_map[chunk['chunk_id']] = chunk
            continue
        
        # 使用三级块的摘要和heading来生成二级块的摘要
        try:
            # 收集所有有效的三级块摘要和标题
            summaries_with_headings = []
            for child in child_chunks:
                heading = child.get('heading', '')
                summary = child.get('summary', '')
                
                # 只使用有效的摘要
                if summary and summary != "无内容，无法生成摘要" and not summary.startswith("摘要生成失败"):
                    if heading:
                        summaries_with_headings.append(f"【{heading}】\n{summary}")
                    else:
                        summaries_with_headings.append(summary)
            
            # 如果没有有效的摘要，则使用原来的方法（使用内容）
            if not summaries_with_headings:
                combined_content = "\n\n".join([child['content'] for child in child_chunks])
            else:
                combined_content = "\n\n".join(summaries_with_headings)
            
            # 生成摘要
            if len(combined_content) > MAX_CONTENT_LENGTH_FOR_DIRECT_SUMMARY:  # 如果内容过长
                summary = summarize_long_content(combined_content)
            else:
                summary = call_llm_for_summary(LEVEL_2_SUMMARY_PROMPT + combined_content)
            
            # 将摘要添加到二级块
            chunk['summary'] = summary
            chunk_map[chunk['chunk_id']] = chunk
            logging.info(f"二级块摘要生成完成: {chunk['chunk_id']}")
            
            # 避免API调用过快
            time.sleep(API_CALL_DELAY_SECONDS)
        except Exception as e:
            logging.error(f"处理二级块{chunk['chunk_id']}时出错: {e}")
            chunk['summary'] = f"摘要生成失败: {str(e)}"
            chunk_map[chunk['chunk_id']] = chunk

def summarize_level_1_chunks(chunks: List[Dict], chunk_map: Dict) -> None:
    """
    处理一级文本块，生成摘要
    
    Args:
        chunks: 块列表
        chunk_map: chunk_id到chunk的映射
    """
    level_1_chunks = [chunk for chunk in chunks if LEVEL_1_PATTERN.match(chunk['chunk_id'])]
    logging.info(f"开始处理{len(level_1_chunks)}个一级文本块...")
    
    for i, chunk in enumerate(tqdm(level_1_chunks, desc="处理一级块")):
        logging.info(f"处理一级块 {i+1}/{len(level_1_chunks)}: {chunk['chunk_id']}")
        
        # 获取该一级块的所有子块（二级块）
        child_ids = chunk.get('child_ids', [])
        # 过滤出二级块的ID
        level_2_child_ids = [child_id for child_id in child_ids if child_id in chunk_map and LEVEL_2_PATTERN.match(child_id)]
        child_chunks = [chunk_map.get(child_id) for child_id in level_2_child_ids if chunk_map.get(child_id)]
        
        # 如果没有子块，跳过
        if not child_chunks:
            chunk['summary'] = "无子文档内容，无法生成摘要"
            chunk_map[chunk['chunk_id']] = chunk
            continue
        
        # 合并所有二级块的摘要（改进后只使用摘要，不再使用内容）
        try:
            # 收集所有有效的二级块标题和摘要
            summaries_with_headings = []
            for child in child_chunks:
                heading = child.get('heading', '')
                summary = child.get('summary', '')
                
                # 只使用有效的摘要
                if summary and summary != "无子文档内容，无法生成摘要" and not summary.startswith("摘要生成失败"):
                    if heading:
                        summaries_with_headings.append(f"【{heading}】\n{summary}")
                    else:
                        summaries_with_headings.append(summary)
            
            # 如果没有有效的摘要，则尝试使用内容
            if not summaries_with_headings:
                logging.warning(f"一级块 {chunk['chunk_id']} 的所有子块都没有有效摘要，尝试使用内容")
                summaries_with_headings = [f"【{child.get('heading', '')}】\n{child.get('content', '')}" 
                                           for child in child_chunks if child.get('content')]
            
            # 如果仍然没有内容，则跳过
            if not summaries_with_headings:
                chunk['summary'] = "无有效子文档内容或摘要，无法生成摘要"
                chunk_map[chunk['chunk_id']] = chunk
                continue
                
            combined_text = "\n\n".join(summaries_with_headings)
            
            # 生成摘要
            if len(combined_text) > MAX_CONTENT_LENGTH_FOR_DIRECT_SUMMARY:  # 如果内容过长
                summary = summarize_long_content(combined_text)
            else:
                summary = call_llm_for_summary(LEVEL_1_SUMMARY_PROMPT + combined_text)
            
            # 将摘要添加到一级块
            chunk['summary'] = summary
            chunk_map[chunk['chunk_id']] = chunk
            logging.info(f"一级块摘要生成完成: {chunk['chunk_id']}")
            
            # 避免API调用过快
            time.sleep(API_CALL_DELAY_SECONDS)
        except Exception as e:
            logging.error(f"处理一级块{chunk['chunk_id']}时出错: {e}")
            chunk['summary'] = f"摘要生成失败: {str(e)}"
            chunk_map[chunk['chunk_id']] = chunk

def process_chunks(file_path: str) -> Dict:
    """
    处理chunks文件，生成两级摘要
    
    Args:
        file_path: JSON文件路径
        
    Returns:
        处理后的chunks数据
    """
    try:
        # 读取JSON文件
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logging.error(f"读取JSON文件出错: {e}")
        raise
    
    chunks = data.get('chunks', [])
    if not chunks:
        logging.warning("JSON文件中没有找到chunks数据")
        return data
    
    # 创建chunk_id到chunk的映射，方便查找
    chunk_map = {chunk['chunk_id']: chunk for chunk in chunks if 'chunk_id' in chunk}
    
    # 清洗所有heading
    clean_all_headings(chunks, chunk_map)
    
    # 处理三级文本块，生成摘要
    summarize_level_3_chunks(chunks, chunk_map)
    
    # 处理二级文本块，生成摘要
    summarize_level_2_chunks(chunks, chunk_map)
    
    # 处理一级文本块，生成摘要
    summarize_level_1_chunks(chunks, chunk_map)
    
    # 更新原始数据中的chunks
    updated_chunks = []
    for chunk in chunks:
        if 'chunk_id' in chunk and chunk['chunk_id'] in chunk_map:
            updated_chunks.append(chunk_map[chunk['chunk_id']])
        else:
            updated_chunks.append(chunk)
    
    # 获取所有一级块的摘要，写入文档摘要
    create_document_summary(updated_chunks, data)
    
    # 返回更新后的数据
    data['chunks'] = updated_chunks
    return data

def create_document_summary(chunks: List[Dict], data: Dict) -> None:
    """
    整合所有一级文本块的摘要，写入document_info.DocumentSummary属性
    
    Args:
        chunks: 所有文本块
        data: 文档数据
    """
    # 获取所有一级文本块
    level_1_chunks = [chunk for chunk in chunks if LEVEL_1_PATTERN.match(chunk.get('chunk_id', ''))]
    
    if not level_1_chunks:
        logging.warning("未找到一级文本块，无法生成文档摘要")
        return
    
    # 收集所有有效的一级块摘要
    summaries = []
    for chunk in level_1_chunks:
        heading = chunk.get('heading', '')
        summary = chunk.get('summary', '')
        
        # 只使用有效的摘要
        if summary and summary != "无子文档内容，无法生成摘要" and not summary.startswith("摘要生成失败"):
            if heading:
                summaries.append(f"【{heading}】\n{summary}")
            else:
                summaries.append(summary)
    
    if not summaries:
        logging.warning("没有找到有效的一级块摘要，无法生成文档摘要")
        return
    
    # 合并所有一级块摘要
    combined_summary = "\n\n".join(summaries)
    
    # 如果摘要太长，再生成一个综合摘要
    if len(combined_summary) > MAX_CONTENT_LENGTH_FOR_DIRECT_SUMMARY:
        logging.info("一级块摘要合并后超长，进行再次摘要...")
        document_summary = summarize_long_content(f"以下是文档的主要部分摘要，请生成一个整体摘要：\n\n{combined_summary}")
    else:
        prompt = """请基于以下文档各部分的摘要，生成一个全面而简洁的文档整体摘要。
这个摘要应当概括文档的主要内容和核心要点，帮助读者快速了解文档的主旨和框架。

摘要内容如下：
"""
        document_summary = call_llm_for_summary(prompt + combined_summary)
    
    # 确保document_info字典存在
    if 'document_info' not in data:
        data['document_info'] = {}
    
    # 将文档摘要写入DocumentSummary属性
    data['document_info']['document_summary'] = document_summary
    logging.info("已生成文档整体摘要并写入document_info.document_summary")

def save_result(data: Dict, output_path: str):
    """保存处理结果到文件"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logging.info(f"结果已保存到: {output_path}")
    except Exception as e:
        logging.error(f"保存结果出错: {e}")
        raise

def main():
    if len(sys.argv) < 2:
        logging.error("请提供JSON文件路径作为参数")
        print("用法: python summarize_chunks.py <input_json_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # 生成输出文件路径
    base_name = os.path.basename(input_file)
    file_name, ext = os.path.splitext(base_name)
    output_file = f"{file_name}_with_summary{ext}"
    
    try:
        # 处理chunks
        logging.info(f"开始处理文件: {input_file}")
        processed_data = process_chunks(input_file)
        
        # 保存结果
        save_result(processed_data, output_file)
        logging.info("处理完成!")
    except Exception as e:
        logging.error(f"处理过程中发生错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 