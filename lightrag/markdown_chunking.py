import re
import logging
from typing import List, Dict, Any, Optional
import hashlib
from lightrag.utils import encode_string_by_tiktoken, decode_tokens_by_tiktoken

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
    return f"chunk_{level}_{index}"

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
    tiktoken_model: str = "gpt-4o"
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
        heading_chunks = list(zip(heading_titles, heading_contents))
    
    # 处理每个标题块
    for heading_title_text, heading_content_text in heading_chunks:
        heading_title_text = heading_title_text.strip()
        current_heading = f"{heading_marker} {heading_title_text}" if heading_title_text else "Document Top Level"
        
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
        
        # 收集下一级标题
        next_titles = []
        next_splits = next_heading_pattern.split(heading_content_text)
        if len(next_splits) > 1:
            next_titles = next_splits[1::2]
        
        # 去除当前标题中的 Markdown 标签
        cleaned_heading = remove_markdown(current_heading)
        
        # 生成当前块的ID
        current_chunk_id = generate_chunk_id(level, current_chunk_index)
        
        # 处理当前级别的内容
        if current_content:
            # 如果有下一级标题，添加到内容中
            next_titles_text = ", ".join(next_titles)
            content_with_context = current_content
            if next_titles:
                content_with_context = f"下一级标题是：{next_titles_text}\n内容是：{current_content}"
            
            cleaned_content = remove_markdown(content_with_context)
            
            # 使用token_size进行分块
            token_size_chunks = chunking_by_token_size_v2(
                current_content,
                overlap_token_size=overlap_token_size,
                max_token_size=max_token_size,
                tiktoken_model=tiktoken_model
            )
            
            for i, chunk_data in enumerate(token_size_chunks):
                # 为分割后的每个块生成唯一ID
                sub_chunk_id = f"{current_chunk_id}_{i}" if len(token_size_chunks) > 1 else current_chunk_id
                
                chunk = {
                    **chunk_data,
                    "chunk_order_index": current_chunk_index,
                    "content": cleaned_content,
                    "chunk_id": sub_chunk_id,
                    "parent_id": parent_id,
                    "child_ids": [],
                    "file_path": file_path,
                    "full_doc_id": full_doc_id
                }
                
                chunks.append(chunk)
                current_chunk_index += 1
        
        # 处理下一级内容
        if next_heading_match:
            sub_chunks, current_chunk_index = process_level(
                heading_content_text,
                next_level,
                current_chunk_id,
                current_chunk_index,
                file_path,
                full_doc_id,
                overlap_token_size,
                max_token_size,
                tiktoken_model
            )
            
            # 更新当前块的child_ids
            if chunks and chunks[-1]["chunk_id"] == current_chunk_id:
                chunks[-1]["child_ids"] = [chunk["chunk_id"] for chunk in sub_chunks]
            
            chunks.extend(sub_chunks)
    
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
        tiktoken_model=tiktoken_model
    )
    
    return chunks 

def chunking_by_token_size_v2(
    content: str,
    overlap_token_size: int = 128,
    max_token_size: int = 1024,
    tiktoken_model: str = "gpt-4o",
) -> list[dict[str, Any]]:
    """Split text by token size."""
    tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)
    results: list[dict[str, Any]] = []
    
    for index, start in enumerate(range(0, len(tokens), max_token_size - overlap_token_size)):
        chunk_content = decode_tokens_by_tiktoken(
            tokens[start : start + max_token_size],
            model_name=tiktoken_model
        )
        results.append({
            "tokens": min(max_token_size, len(tokens) - start),
            "content": chunk_content.strip(),
            "chunk_order_index": index,
        })
    
    return results 