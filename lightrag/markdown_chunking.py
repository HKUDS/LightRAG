import re
import logging
from typing import List, Dict, Any, Optional
import hashlib
import argparse
import os
import json
from lightrag.utils import encode_string_by_tiktoken, decode_tokens_by_tiktoken
import uuid

logger = logging.getLogger(__name__)

def remove_markdown(text: str) -> str:
    """移除文本中的 Markdown 标签"""
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'^\s*#+\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'(\*\*|\*|__|_)(.*?)\1', r'\2', text)
    text = re.sub(r'`(.+?)`', r'\1', text)
    text = re.sub(r'[>\*_~`]', '', text)
    return text

def remove_page_numbers(text: str) -> str:
    """移除文本中的页码标签
    
    Args:
        text: 需要处理的文本
        
    Returns:
        移除页码标签后的文本
    """
    # 移除形如 (p.1) 的页码标签
    return re.sub(r'\s*\(p\.\d+\)\s*', ' ', text).strip()

def extract_page_numbers(text: str) -> List[int]:
    """提取文本中的页码标签
    
    Args:
        text: 需要提取页码的文本
        
    Returns:
        提取到的不重复页码列表（整型）
    """
    # 匹配形如 (p.1) 的页码标签
    pattern = r'\(p\.(\d+)\)'
    matches = re.findall(pattern, text)
    
    # 如果没有找到页码，返回空列表
    if not matches:
        return []
    
    # 转换为整型列表并去重
    page_numbers = list(set(int(page) for page in matches))
    return page_numbers

def preprocess_markdown_headings(content: str) -> str:
    """预处理Markdown标题，确保文档只有一个一级标题"""
    lines = content.split('\n')
    h1_indices = [i for i, line in enumerate(lines) if re.match(r'^\s*#\s+', line)]
    
    if len(h1_indices) <= 1:
        return content
    
    first_h1_index = h1_indices[0]
    new_lines = lines.copy()
    
    for i in range(len(lines)):
        if i <= first_h1_index:
            continue
        
        heading_match = re.match(r'^(\s*)([#]+)(\s+.*)', lines[i])
        if heading_match:
            indent, hashes, rest = heading_match.groups()
            level = len(hashes)
            
            if level == 1:
                new_lines[i] = f"{indent}##" + rest
            elif level > 1:
                new_lines[i] = f"{indent}{'#' * (level + 1)}" + rest
    
    return '\n'.join(new_lines)

def generate_chunk_id(level: int, index: int) -> str:
    """生成块的唯一ID"""
    # 生成唯一标识前缀
    unique_prefix = hashlib.md5(f"{level}_{index}_{uuid.uuid4()}".encode()).hexdigest()[:8]
    return f"{unique_prefix}_chunk_{level}_{index}"

def generate_full_doc_id(content: str) -> str:
    """生成文档的唯一ID"""
    return f"doc-{hashlib.md5(content.encode()).hexdigest()}"

def process_level(
    content: str,
    level: int,
    parent_id: Optional[str] = None,
    chunk_index: int = 0,
    file_path: str = "unknown_source",
    full_doc_id: str = None,
    overlap_token_size: int = 128,
    max_token_size: int = 1024,
    tiktoken_model: str = "gpt-4o",
    doc_title: str = None,  # 添加文档标题参数
    parent_title: str = None  # 添加父级标题参数
) -> tuple[List[Dict[str, Any]], int]:
    """
    递归处理每个标题级别的内容
    
    Args:
        content: 当前级别的内容
        level: 当前处理的标题级别
        parent_id: 父文档的ID
        chunk_index: 当前块的索引
        file_path: 文件路径
        full_doc_id: 完整文档ID
        overlap_token_size: 重叠的token大小
        max_token_size: 最大token大小
        tiktoken_model: token化模型名称
        doc_title: 文档标题（第一个一级标题）
        parent_title: 上一级标题（用于三级标题）
    
    Returns:
        (chunks, new_chunk_index) 元组，包含当前级别处理的所有块和新的块索引
    """
    chunks = []
    current_chunk_index = chunk_index
    
    # 构建当前级别的标题模式
    heading_marker = "#" * level
    heading_pattern = re.compile(f"^{heading_marker}\\s+(.+)", re.MULTILINE)
    
    # 分割内容
    heading_splits = heading_pattern.split(content)
    
    # 如果没有找到当前级别的标题，则将整个内容作为一个块处理
    if len(heading_splits) <= 1:
        if level == 1:
            # 顶级没有标题，整个文档作为一个块
            heading_chunks = [("Document Top Level", content)]
        else:
            # 当前级别没有标题，内容作为上一级的一部分
            return chunks, current_chunk_index
    else:
        # 提取标题和内容
        heading_titles = heading_splits[1::2]  # 标题
        heading_contents = heading_splits[2::2]  # 内容
        
        # 确保标题和内容数量匹配
        if len(heading_titles) > len(heading_contents):
            heading_contents.extend([""] * (len(heading_titles) - len(heading_contents)))
            
        heading_chunks = list(zip(heading_titles, heading_contents))
    
    # 处理每个标题块
    for heading_title_text, heading_content_text in heading_chunks:
        heading_title_text = heading_title_text.strip()
        current_heading = f"{heading_marker} {heading_title_text}" if heading_title_text else "Document Top Level"
        
        # 如果是一级标题的第一个，保存为文档标题
        if level == 1 and doc_title is None and heading_title_text:
            doc_title = heading_title_text
        
        # 生成当前块的ID
        current_chunk_id = generate_chunk_id(level, current_chunk_index)
        current_chunk_index += 1
        
        # 创建当前块
        current_chunk = {
            "tokens": 0,  # 稍后更新
            "content": "",  # 稍后更新
            "chunk_order_index": current_chunk_index - 1,
            "chunk_id": current_chunk_id,
            "parent_id": parent_id,
            "child_ids": [],  # 稍后更新
            "file_path": file_path,
            "full_doc_id": full_doc_id,
            "heading": remove_markdown(current_heading),
            "page_numbers": []  # 添加页码属性
        }
        
        # 先添加当前块到chunks列表中
        chunks.append(current_chunk)
        
        # 查找下一级标题
        next_level = level + 1
        next_heading_marker = "#" * next_level
        next_heading_pattern = re.compile(f"^{next_heading_marker}\\s+(.+)", re.MULTILINE)
        next_heading_match = next_heading_pattern.search(heading_content_text)
        
        # 提取当前级别的内容（直到下一级标题之前）
        if next_heading_match:
            current_content = heading_content_text[:next_heading_match.start()].strip()
        else:
            current_content = heading_content_text.strip()
        
        # 收集下一级标题，这将用于当前级别的内容丰富
        next_titles = []
        next_splits = next_heading_pattern.split(heading_content_text)
        if len(next_splits) > 1:
            next_titles = next_splits[1::2]
        
        # 拼接下一级标题文本
        next_titles_text = ", ".join(next_titles) if next_titles else ""
        
        # 为二级标题添加标题内容和文档标题
        if level == 2:
            # 如果有文档标题，先去除页码后再在二级标题前添加
            if doc_title:
                clean_doc_title = remove_page_numbers(doc_title)
                heading_content = f"{clean_doc_title}\n{heading_title_text}"
            else:
                heading_content = heading_title_text
                
            if not current_content:
                current_content = heading_content
            else:
                current_content = f"{heading_content}\n\n{current_content}"
                
            # 调整content_with_context的格式以匹配样板
            if next_titles:
                # 使用去除页码后的文档标题
                clean_doc_title = remove_page_numbers(doc_title) if doc_title else ""
                content_with_context = f"{clean_doc_title}\n{heading_title_text}\n\n下一级标题是：{next_titles_text}\n"
            else:
                content_with_context = f"{heading_content}\n\n{current_content}"
        # 为三级标题添加文档标题和父级标题
        elif level == 3:
            # 如果有文档标题和父级标题，先去除文档标题和父级标题中的页码后再在三级标题前添加
            if doc_title and parent_title:
                clean_doc_title = remove_page_numbers(doc_title)
                clean_parent_title = remove_page_numbers(parent_title)
                heading_content = f"{clean_doc_title}\n{clean_parent_title}\n{heading_title_text}"
            elif doc_title:
                clean_doc_title = remove_page_numbers(doc_title)
                heading_content = f"{clean_doc_title}\n{heading_title_text}"
            else:
                heading_content = heading_title_text
                
            if not current_content:
                current_content = heading_content
            else:
                current_content = f"{heading_content}\n\n{current_content}"
                
            content_with_context = current_content
        # 处理当前级别的内容 - 将标题内容也加入到内容中
        # 如果是最底层级别(level > 3)且没有内容，则将标题作为内容
        elif level > 3 and not current_content and heading_title_text:
            current_content = heading_title_text
            content_with_context = current_content
        # 如果有下一级标题，添加到内容中
        elif next_titles and level <= 2:  # 只为一级和二级标题添加下一级标题列表
            content_with_context = f"{current_heading}\n\n下一级标题是：{next_titles_text}\n"
        else:
            content_with_context = current_content
        
        # 提取页码
        heading_page_numbers = extract_page_numbers(heading_title_text)
        content_page_numbers = extract_page_numbers(content_with_context)
        all_page_numbers = list(set(heading_page_numbers + content_page_numbers))
        
        # 将页码添加到当前块
        current_chunk["page_numbers"] = all_page_numbers
        
        # 先去除内容中的页码标签，再移除Markdown标签
        content_without_page_numbers = remove_page_numbers(content_with_context)
        cleaned_content = remove_markdown(content_without_page_numbers)
        
        # 更新当前块的内容
        current_chunk["content"] = cleaned_content
        
        # 使用token_size进行分块，如果内容非空
        if current_content:
            token_size_chunks = chunking_by_token_size_v2(
                current_content,
                overlap_token_size=overlap_token_size,
                max_token_size=max_token_size,
                tiktoken_model=tiktoken_model,
                doc_title=doc_title,
                parent_title=parent_title
            )
            
            if token_size_chunks:
                # 更新当前块的tokens
                current_chunk["tokens"] = token_size_chunks[0]["tokens"]
                
                # 如果分块结果超过1个，为额外的块创建新条目
                if len(token_size_chunks) > 1:
                    for i, chunk_data in enumerate(token_size_chunks[1:], 1):
                        # 为子块生成唯一标识前缀
                        unique_prefix = hashlib.md5(f"{level}_{current_chunk_index}_{i}_{uuid.uuid4()}".encode()).hexdigest()[:8]
                        sub_chunk_id = f"{unique_prefix}_chunk_{level}_{current_chunk_index}_{i}"
                        sub_chunk = {
                            **chunk_data,
                            "chunk_order_index": current_chunk_index - 1,
                            "content": chunk_data["content"],  # 使用实际分块内容而不是cleaned_content
                            "chunk_id": sub_chunk_id,
                            "parent_id": parent_id,
                            "child_ids": [],
                            "file_path": file_path,
                            "full_doc_id": full_doc_id,
                            "heading": remove_markdown(current_heading),
                            "page_numbers": all_page_numbers  # 添加页码属性
                        }
                        chunks.append(sub_chunk)
        
        # 处理下一级内容，如果有
        sub_chunks = []
        if next_heading_match:
            # 如果当前是二级标题，将当前标题作为父级标题传递给下一级
            current_parent_title = heading_title_text if level == 2 else parent_title
            
            sub_chunks, new_chunk_index = process_level(
                heading_content_text,
                next_level,
                current_chunk_id,  # 使用当前块ID作为父ID
                current_chunk_index,
                file_path,
                full_doc_id,
                overlap_token_size,
                max_token_size,
                tiktoken_model,
                doc_title,  # 传递文档标题
                current_parent_title  # 传递父级标题
            )
            
            # 更新当前块的child_ids，且只包含直接子级（即下一级标题）
            direct_child_ids = []
            for sub_chunk in sub_chunks:
                # 只添加直接子级（level等于当前level+1）
                sub_chunk_id = sub_chunk["chunk_id"]
                # 使用正则匹配新的ID格式中的level部分
                match = re.search(r'_chunk_(\d+)_', sub_chunk_id)
                if match and int(match.group(1)) == level+1:
                    direct_child_ids.append(sub_chunk_id)
            
            # 更新当前块的child_ids
            chunks[-1]["child_ids"] = direct_child_ids
            
            # 添加所有子块
            chunks.extend(sub_chunks)
            current_chunk_index = new_chunk_index
    
    return chunks, current_chunk_index

def chunking_by_markdown_hierarchical(
    content: str,
    overlap_token_size: int = 128,
    max_token_size: int = 1024,
    tiktoken_model: str = "gpt-4o",
    file_path: str = "unknown_source",
) -> List[Dict[str, Any]]:
    """
    按照Markdown标题层级进行分块
    
    Args:
        content: Markdown文档内容
        overlap_token_size: 重叠的token大小
        max_token_size: 最大token大小
        tiktoken_model: token化模型名称
        file_path: 文件路径
    
    Returns:
        分块结果列表
    """
    # 预处理标题
    content = preprocess_markdown_headings(content)
    
    # 生成文档ID
    full_doc_id = generate_full_doc_id(content)
    
    # 从第一级开始处理
    chunks, _ = process_level(
        content,
        level=1,
        parent_id=None,
        chunk_index=0,
        file_path=file_path,
        full_doc_id=full_doc_id,
        overlap_token_size=overlap_token_size,
        max_token_size=max_token_size,
        tiktoken_model=tiktoken_model,
        doc_title=None,  # 初始文档标题为空，在处理过程中会自动获取
        parent_title=None  # 初始父级标题为空
    )
    
    return chunks

def chunking_by_token_size_v2(
    content: str,
    overlap_token_size: int = 128,
    max_token_size: int = 1024,
    tiktoken_model: str = "gpt-4o",
    doc_title: Optional[str] = None,  # 添加文档标题参数
    parent_title: Optional[str] = None,  # 添加父级标题参数
) -> list[dict[str, Any]]:
    """
    Split text by token size using recursive/hierarchical character splitting.
    
    核心思想：优先使用最能代表语义完整的大分隔符（如段落符），如果拆分后的块仍然超过
    最大长度限制，则在该块内部使用次级分隔符（如句号、问号、感叹号），再不行则用更小
    的分隔符（如逗号、空格），最后实在没办法才按字符数硬切。
    
    同时在分割后尽量合并相邻段落，保证在不超过最大token数的情况下最大限度保持文本连贯性。
    
    Args:
        content: 要分割的文本内容
        overlap_token_size: 重叠的token大小
        max_token_size: 最大token大小
        tiktoken_model: token化模型名称
        doc_title: 文档标题
        parent_title: 父级标题
    
    Returns:
        分块结果列表
    """
    # 先移除页码标签
    content = remove_page_numbers(content)
    
    # 创建标题前缀
    title_prefix = ""
    if doc_title:
        title_prefix += f"{remove_page_numbers(doc_title)}\n"
    if parent_title:
        title_prefix += f"{remove_page_numbers(parent_title)}\n"
    
    # 计算标题前缀的token长度
    title_prefix_tokens = 0
    if title_prefix:
        title_prefix_tokens = len(encode_string_by_tiktoken(title_prefix, model_name=tiktoken_model))
        # 调整最大token大小，为标题预留空间
        adjusted_max_token_size = max_token_size - title_prefix_tokens
    else:
        adjusted_max_token_size = max_token_size
    
    # 按优先级排序的分隔符列表（从高到低）
    delimiters = ["\n\n", "\n", "。", "！", "？", ".", "!", "?", "，", ",", " ", ""]
    
    # 递归分割函数
    def split_text_recursive(text: str, delimiter_index: int = 0) -> list[str]:
        """递归地使用不同级别的分隔符拆分文本"""
        # 检查当前文本的token长度
        tokens = encode_string_by_tiktoken(text, model_name=tiktoken_model)
        
        # 如果文本已经小于调整后的最大长度，直接返回
        if len(tokens) <= adjusted_max_token_size:
            return [text]
            
        # 如果已经用完了所有分隔符，使用token硬切
        if delimiter_index >= len(delimiters) - 1:
            chunks = []
            for start in range(0, len(tokens), adjusted_max_token_size - overlap_token_size):
                end = min(start + adjusted_max_token_size, len(tokens))
                chunk_tokens = tokens[start:end]
                chunk_text = decode_tokens_by_tiktoken(chunk_tokens, model_name=tiktoken_model)
                chunks.append(chunk_text)
            return chunks
        
        # 获取当前级别的分隔符
        delimiter = delimiters[delimiter_index]
        
        # 使用当前分隔符拆分文本
        segments = text.split(delimiter)
        
        # 如果分隔符不存在于文本中（拆分后仍然只有一段），尝试下一级分隔符
        if len(segments) == 1:
            return split_text_recursive(text, delimiter_index + 1)
        
        # 处理所有分段
        result_chunks = []
        for segment in segments:
            if not segment.strip():  # 跳过空段
                continue
                
            # 检查分段大小
            segment_tokens = encode_string_by_tiktoken(segment, model_name=tiktoken_model)
            
            if len(segment_tokens) <= adjusted_max_token_size:
                # 分段已经足够小，直接添加
                result_chunks.append(segment)
            else:
                # 分段仍然太大，使用下一级分隔符递归分割
                sub_chunks = split_text_recursive(segment, delimiter_index + 1)
                result_chunks.extend(sub_chunks)
        
        return result_chunks
    
    # 执行递归分割，获取初步的分块
    initial_chunks = split_text_recursive(content)
    
    # 贪心合并函数：在不超过adjusted_max_token_size的前提下尽可能合并相邻段落
    def merge_chunks(chunks: list[str]) -> list[str]:
        if not chunks or len(chunks) == 1:
            return chunks
            
        merged_chunks = []
        current_chunk = chunks[0]
        current_tokens = encode_string_by_tiktoken(current_chunk, model_name=tiktoken_model)
        
        for i in range(1, len(chunks)):
            next_chunk = chunks[i]
            next_tokens = encode_string_by_tiktoken(next_chunk, model_name=tiktoken_model)
            
            # 计算合并后的token长度（加上分隔符的长度）
            # 这里假设我们使用原始分隔符合并相邻分块
            delimiter_to_use = delimiters[0] if delimiters[0] in ["\n\n", "\n"] else "\n"
            combined_chunk = current_chunk + delimiter_to_use + next_chunk
            combined_tokens_len = len(encode_string_by_tiktoken(combined_chunk, model_name=tiktoken_model))
            
            # 如果合并后不超过调整后的最大长度，则合并
            if combined_tokens_len <= adjusted_max_token_size:
                current_chunk = combined_chunk
                current_tokens = encode_string_by_tiktoken(current_chunk, model_name=tiktoken_model)
            else:
                # 如果合并会超过最大长度，先保存当前块，然后从下一个块开始
                merged_chunks.append(current_chunk)
                current_chunk = next_chunk
                current_tokens = next_tokens
        
        # 添加最后一个块
        if current_chunk:
            merged_chunks.append(current_chunk)
            
        return merged_chunks
    
    # 应用合并策略，可以多次合并直到不能再合并
    merged_chunks = initial_chunks
    prev_len = 0
    
    # 连续合并直到没有更多可以合并的相邻块
    while len(merged_chunks) != prev_len:
        prev_len = len(merged_chunks)
        merged_chunks = merge_chunks(merged_chunks)
    
    # 处理重叠（如果需要）
    if overlap_token_size > 0 and len(merged_chunks) > 1:
        overlapped_chunks = []
        for i in range(len(merged_chunks)):
            current_chunk = merged_chunks[i]
            
            # 如果不是第一个块，添加前一个块的末尾部分
            if i > 0:
                prev_chunk = merged_chunks[i-1]
                prev_tokens = encode_string_by_tiktoken(prev_chunk, model_name=tiktoken_model)
                
                # 如果前一个块足够长，提取末尾部分作为重叠
                if len(prev_tokens) >= overlap_token_size:
                    overlap_tokens = prev_tokens[-overlap_token_size:]
                    overlap_text = decode_tokens_by_tiktoken(overlap_tokens, model_name=tiktoken_model)
                    
                    # 将重叠部分添加到当前块的开头
                    current_chunk = overlap_text + current_chunk
            
            overlapped_chunks.append(current_chunk)
        
        merged_chunks = overlapped_chunks
    
    # 构建最终结果
    results = []
    for i, chunk in enumerate(merged_chunks):
        # 确保没有超出最大token数
        # 添加标题前缀到内容
        chunk_with_title = title_prefix + chunk if title_prefix else chunk
        chunk_tokens = encode_string_by_tiktoken(chunk_with_title, model_name=tiktoken_model)
        
        if len(chunk_tokens) > max_token_size:
            # 如果添加标题后超出最大token数，需要重新调整内容
            # 计算可用于内容的token数
            available_content_tokens = max_token_size - title_prefix_tokens
            
            # 对原始内容进行切分
            original_tokens = encode_string_by_tiktoken(chunk, model_name=tiktoken_model)
            for start in range(0, len(original_tokens), available_content_tokens - overlap_token_size):
                end = min(start + available_content_tokens, len(original_tokens))
                sub_chunk_tokens = original_tokens[start:end]
                sub_chunk = decode_tokens_by_tiktoken(sub_chunk_tokens, model_name=tiktoken_model)
                
                # 构建带标题的最终分块
                final_sub_chunk = title_prefix + sub_chunk
                final_sub_chunk_tokens = encode_string_by_tiktoken(final_sub_chunk, model_name=tiktoken_model)
                
                results.append({
                    "tokens": len(final_sub_chunk_tokens),
                    "content": final_sub_chunk.strip(),
                    "chunk_order_index": len(results),
                })
        else:
            # 如果不超出，直接添加
            results.append({
                "tokens": len(chunk_tokens),
                "content": chunk_with_title.strip(),
                "chunk_order_index": len(results),
            })
    
    return results

def main():
    """主函数：读取文件，执行分块，打印结果"""
    parser = argparse.ArgumentParser(description='Markdown文档分块工具')
    parser.add_argument('file_path', type=str, help='Markdown文件路径')
    parser.add_argument('--overlap', type=int, default=128, help='块重叠的token大小')
    parser.add_argument('--max_size', type=int, default=1024, help='块的最大token大小')
    parser.add_argument('--model', type=str, default="gpt-4o", help='tiktoken模型名称')
    parser.add_argument('--output', type=str, help='输出JSON文件路径(可选)')
    args = parser.parse_args()

    # 检查文件是否存在
    if not os.path.exists(args.file_path):
        print(f"错误: 文件 '{args.file_path}' 不存在")
        return

    # 读取文件内容
    try:
        with open(args.file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return

    # 执行分块
    print(f"正在处理文件: {args.file_path}")
    print(f"参数: 重叠大小={args.overlap}, 最大大小={args.max_size}, 模型={args.model}")
    chunks = chunking_by_markdown_hierarchical(
        content, 
        overlap_token_size=args.overlap,
        max_token_size=args.max_size,
        tiktoken_model=args.model,
        file_path=args.file_path
    )

    # 打印分块结果
    print(f"\n共生成了 {len(chunks)} 个分块:")
    for i, chunk in enumerate(chunks):
        print(f"\n{'='*50}")
        print(f"分块 {i+1}/{len(chunks)}")
        print(f"ID: {chunk['chunk_id']}")
        print(f"父ID: {chunk.get('parent_id', 'None')}")
        print(f"子ID列表: {chunk.get('child_ids', [])}")
        print(f"Tokens: {chunk.get('tokens', 'N/A')}")
        print(f"序号: {chunk.get('chunk_order_index', 'N/A')}")
        print(f"页码：{chunk.get('page_numbers',[])}")
        print(f"{'='*50}")
        print(f"内容:\n{chunk['content']}")
        print(f"{'='*50}")

    # 如果指定了输出文件，则将结果保存为JSON
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
            print(f"\n分块结果已保存到: {args.output}")
        except Exception as e:
            print(f"保存结果时出错: {e}")

if __name__ == "__main__":
    main() 