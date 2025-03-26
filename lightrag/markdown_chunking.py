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
                tiktoken_model=tiktoken_model
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
                            "content": cleaned_content,
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
) -> list[dict[str, Any]]:
    """Split text by token size."""
    # 先移除页码标签
    content = remove_page_numbers(content)
    
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