# markdown_chunking.py
import re
import logging
from typing import List, Dict, Any, Optional, Tuple

from lightrag.operate import chunking_by_token_size


def remove_markdown(text: str) -> str:
    """辅助函数：移除文本中的 Markdown 标签"""
    # 将 Markdown 链接 [text](url) 替换为 text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # 将图片 ![alt](url) 替换为 alt
    text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'\1', text)
    # 移除标题前缀的 #
    text = re.sub(r'^\s*#+\s*', '', text, flags=re.MULTILINE)
    # 移除强调符号 **、*、__、_
    text = re.sub(r'(\*\*|\*|__|_)(.*?)\1', r'\2', text)
    # 移除行内代码标记
    text = re.sub(r'`(.+?)`', r'\1', text)
    # 移除其他 Markdown 符号
    text = re.sub(r'[>\*_~`]', '', text)
    return text


def preprocess_markdown_headings(content: str) -> str:
    """
    预处理Markdown标题，确保文档只有一个一级标题。
    
    如果存在多个一级标题，则保留第一个作为真正的文章标题，
    将其余的一级标题降级为二级标题，并相应地调整所有子标题的级别。
    
    Args:
        content: Markdown 文档内容
        
    Returns:
        处理后的 Markdown 文档内容，保证标题层级正确
    """
    # 分割文档为行
    lines = content.split('\n')
    
    # 查找所有一级标题的行号
    h1_indices = [i for i, line in enumerate(lines) if re.match(r'^\s*#\s+', line)]
    
    # 如果一级标题少于2个，不需要处理
    if len(h1_indices) <= 1:
        return content
    
    # 保留第一个一级标题，其余的需要降级
    first_h1_index = h1_indices[0]
    
    # 创建一个新的行列表，用于存储修改后的内容
    new_lines = lines.copy()
    
    # 处理第一个一级标题之后的部分
    for i in range(len(lines)):
        if i <= first_h1_index:
            # 保持第一个一级标题及之前的内容不变
            continue
        
        # 查找行开头的标题标记
        heading_match = re.match(r'^(\s*)([#]+)(\s+.*)', lines[i])
        if heading_match:
            indent, hashes, rest = heading_match.groups()
            level = len(hashes)
            
            # 如果是一级标题，降级为二级
            if level == 1:
                new_lines[i] = f"{indent}##" + rest
            # 其他标题也相应增加一级
            elif level > 1:
                new_lines[i] = f"{indent}{'#' * (level + 1)}" + rest
    
    # 合并处理后的行
    return '\n'.join(new_lines)


def preprocess_attachment_headings(content: str) -> str:
    """
    预处理Markdown中的附件标题，确保附件标题按照二级标题处理。
    
    扫描文档中的所有标题，如果标题中包含"附件"字样，将其调整为二级标题，
    其下级标题也相应调整层级。
    
    Args:
        content: Markdown 文档内容
        
    Returns:
        处理后的 Markdown 文档内容，保证附件标题为二级标题
    """
    # 分割文档为行，以便逐行处理
    lines = content.split('\n')
    processed_lines = []
    
    # 一次迭代标识所有附件标题的位置、层级和类型
    attachment_headers = []
    for i, line in enumerate(lines):
        heading_match = re.match(r'^(\s*)([#]+)(\s+.*)$', line)
        if heading_match:
            indent, hashes, rest = heading_match.groups()
            level = len(hashes)
            rest_content = rest.strip()
            
            # 判断是否为主附件标题
            if "附件" in rest_content:
                is_sub_attachment = False
                for attach in attachment_headers:
                    # 检查是否为某个已处理的附件的子标题
                    if rest_content.startswith(attach['title'].split('：')[0]) and rest_content != attach['title']:
                        is_sub_attachment = True
                        break
                
                if not is_sub_attachment:
                    attachment_headers.append({
                        'index': i,
                        'level': level,
                        'title': rest_content,
                        'is_main': True
                    })
    
    # 现在处理文档，应用附件标题级别调整
    i = 0
    current_attachment = None
    level_adjustment = 0
    
    while i < len(lines):
        line = lines[i]
        heading_match = re.match(r'^(\s*)([#]+)(\s+.*)$', line)
        
        if heading_match:
            indent, hashes, rest = heading_match.groups()
            level = len(hashes)
            rest_content = rest.strip()
            
            # 检查是否为主附件标题
            is_main_attachment = False
            for attach in attachment_headers:
                if i == attach['index']:
                    is_main_attachment = True
                    current_attachment = attach
                    # 计算需要调整的级别值
                    level_adjustment = 2 - level
                    # 将附件标题调整为二级标题
                    if level != 2:
                        line = f"{indent}##" + rest
                    break
            
            # 如果不是主附件标题，且当前在附件范围内
            if not is_main_attachment and current_attachment:
                # 判断是否退出当前附件范围
                if level <= current_attachment['level']:
                    # 新的相同或更高级别标题，退出附件处理
                    if "附件" not in rest_content or any(rest_content == attach['title'] for attach in attachment_headers):
                        current_attachment = None
                        level_adjustment = 0
                else:
                    # 在附件范围内的子标题，需要调整级别
                    new_level = level + level_adjustment
                    # 确保级别在合法范围内
                    new_level = max(1, min(6, new_level))
                    line = f"{indent}{'#' * new_level}" + rest
        
        processed_lines.append(line)
        i += 1
    
    return '\n'.join(processed_lines)


def chunking_by_markdown_hierarchical(
    content: str,
    split_by_character: str | None = None,  # 忽略此参数，Markdown 结构化分块不依赖字符分割
    split_by_character_only: bool = False, # 忽略此参数
    overlap_token_size: int = 128,
    max_token_size: int = 1024,
    tiktoken_model: str = "gpt-4o",
    heading_levels: int = 2,  # 新增参数：指定要处理的标题级别数量，默认为2（章节和节）
    parent_level: int = 1,  # 新增参数：指定父文档的级别，默认为1级标题
    preprocess_headings: bool = True,  # 新增参数：是否预处理标题层级
    preprocess_attachments: bool = True,  # 新增参数：是否预处理附件标题
) -> List[Dict[str, Any]]:
    """
    Markdown 结构化分块函数，按照标题层次进行分块，并处理超长内容。

    Args:
        content: Markdown 文档内容
        split_by_character: 忽略此参数，Markdown 结构化分块不依赖字符分割
        split_by_character_only: 忽略此参数
        overlap_token_size: 块之间的 token 重叠大小
        max_token_size: 每个块的最大 token 数量
        tiktoken_model: 用于 token 化的模型名称
        heading_levels: 要处理的标题级别数量，例如：2表示处理到##级别，3表示处理到###级别
        parent_level: 指定父文档的级别，例如：1表示#级别为父文档，2表示##级别为父文档
        preprocess_headings: 是否预处理标题层级，确保文档只有一个一级标题
        preprocess_attachments: 是否预处理附件标题，确保附件标题为二级标题

    Returns:
        分块结果列表，每个元素是一个字典，包含 "content", "tokens", "chunk_order_index", "structure_level", "heading_text", 
        "chunk_id", "parent_id", "child_ids"
    """
    # 设置日志记录器，替代print语句
    logger = logging.getLogger(__name__)
    
    # 验证heading_levels参数
    if heading_levels < 1 or heading_levels > 6:
        raise ValueError("heading_levels参数必须在1到6之间")
    
    # 验证parent_level参数
    if parent_level < 1 or parent_level > heading_levels:
        raise ValueError("parent_level参数必须在1到heading_levels之间")
    
    # 预处理附件标题，确保附件标题为二级标题
    if preprocess_attachments:
        content = preprocess_attachment_headings(content)
    
    # 预处理标题层级，确保只有一个一级标题
    if preprocess_headings:
        content = preprocess_markdown_headings(content)
    
    chunks = []
    chunk_index = 0
    
    # 定义标题级别名称映射
    level_names = ["chapter", "section", "subsection", "subsubsection", "paragraph", "subparagraph"]
    
    # 存储父子文档关系
    parent_child_map = {}
    
    # 定义处理标题层级的辅助函数
    def process_level(content_text: str, current_level: int, parent_heading: str = "", parent_id: Optional[str] = None) -> List[str]:
        nonlocal chunk_index
        
        # 返回当前级别处理的所有chunk_ids
        current_level_chunk_ids = []
        
        if current_level > heading_levels:
            # 已达到最大处理级别，将内容作为最后一级的块
            if content_text.strip():
                cleaned_parent = remove_markdown(parent_heading)
                base_chunk_meta = {
                    "structure_level": level_names[current_level - 2] if current_level - 2 < len(level_names) else f"level_{current_level}",
                    "heading_text": cleaned_parent
                }
                
                token_size_chunks = chunking_by_token_size(
                    content_text,
                    overlap_token_size=overlap_token_size,
                    max_token_size=max_token_size,
                    tiktoken_model=tiktoken_model
                )
                
                for chunk_data in token_size_chunks:
                    cleaned_content = remove_markdown(f"本内容属于{parent_heading}")
                    chunk_id = f"chunk_{current_level}_{chunk_index}"
                    
                    chunk_meta = {
                        **chunk_data,
                        "chunk_order_index": chunk_index,
                        "content": cleaned_content,
                        **base_chunk_meta,
                        "chunk_id": chunk_id,
                        "parent_id": parent_id,
                        "child_ids": []
                    }
                    
                    chunks.append(chunk_meta)
                    current_level_chunk_ids.append(chunk_id)
                    
                    # 更新父文档的子文档列表
                    if parent_id and parent_id in parent_child_map:
                        parent_child_map[parent_id].append(chunk_id)
                    
                    chunk_index += 1
            return current_level_chunk_ids
        
        # 构建当前级别的标题模式
        heading_marker = "#" * current_level
        heading_pattern = re.compile(f"^{heading_marker}\\s+(.+)", re.MULTILINE)
        
        # 分割内容
        heading_splits = heading_pattern.split(content_text)
        
        # 如果没有找到当前级别的标题，则将整个内容作为一个块处理
        if len(heading_splits) <= 1:
            if current_level == 1:
                # 顶级没有标题，整个文档作为一个块
                heading_chunks = [("Document Top Level", content_text)]
            else:
                # 当前级别没有标题，内容作为上一级的一部分
                child_ids = process_level(content_text, current_level + 1, parent_heading, parent_id)
                current_level_chunk_ids.extend(child_ids)
                return current_level_chunk_ids
        else:
            # 提取标题和内容
            heading_titles = heading_splits[1::2]  # 标题
            heading_contents = heading_splits[2::2]  # 内容
            heading_chunks = list(zip(heading_titles, heading_contents))
        
        # 处理每个标题块
        for heading_title_text, heading_content_text in heading_chunks:
            heading_title_text = heading_title_text.strip()
            current_heading = f"{heading_marker} {heading_title_text}" if heading_title_text else parent_heading
            
            logger.info(f"Processing Level {current_level}: {current_heading}")
            
            # 查找下一级标题
            next_level = current_level + 1
            if next_level <= heading_levels:
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
            else:
                # 已经是最后一级，所有内容都属于当前级别
                current_content = heading_content_text.strip()
                next_titles = []
            
            # 去除当前标题中的 Markdown 标签
            cleaned_heading = remove_markdown(current_heading)
            
            # 创建当前级别的块
            level_name = level_names[current_level - 1] if current_level - 1 < len(level_names) else f"level_{current_level}"
            
            # 判断当前级别是否为父文档级别
            is_parent_level = (current_level == parent_level)
            
            # 生成当前块的ID
            current_chunk_id = f"chunk_{current_level}_{chunk_index}"
            
            # 初始化父子文档映射
            parent_child_map[current_chunk_id] = []
            
            # 添加当前级别的内容块
            if current_content:
                # 如果有下一级标题，添加到内容中
                next_titles_text = ", ".join(next_titles)
                content_with_context = current_content
                if next_titles and next_level <= heading_levels:
                    content_with_context = f"下一级标题是：{next_titles_text}\n内容是：{current_content}"
                if parent_heading:
                    content_with_context = f"{content_with_context}\n本内容属于{parent_heading}"
                
                cleaned_content = remove_markdown(content_with_context)
                
                token_size_chunks = chunking_by_token_size(
                    current_content,
                    overlap_token_size=overlap_token_size,
                    max_token_size=max_token_size,
                    tiktoken_model=tiktoken_model
                )
                
                # 根据是否为父级别决定新的parent_id
                new_parent_id = current_chunk_id if is_parent_level else parent_id
                
                for i, chunk_data in enumerate(token_size_chunks):
                    # 为分割后的每个块生成唯一ID
                    sub_chunk_id = f"{current_chunk_id}_{i}" if len(token_size_chunks) > 1 else current_chunk_id
                    
                    base_chunk_meta = {
                        "structure_level": level_name,
                        "heading_text": cleaned_heading,
                        "chunk_id": sub_chunk_id,
                        "parent_id": parent_id,
                        "child_ids": []
                    }
                    
                    # 如果是父级别，parent_id为None
                    if is_parent_level:
                        base_chunk_meta["parent_id"] = None
                    
                    chunks.append({
                        **chunk_data,
                        "chunk_order_index": chunk_index,
                        "content": cleaned_content,
                        **base_chunk_meta
                    })
                    
                    current_level_chunk_ids.append(sub_chunk_id)
                    
                    # 更新父文档的子文档列表
                    if parent_id and parent_id in parent_child_map:
                        parent_child_map[parent_id].append(sub_chunk_id)
                    
                    chunk_index += 1
            
            # 处理下一级内容，并获取子文档ID
            if next_level <= heading_levels:
                # 确定下一级处理使用的parent_id
                next_parent_id = current_chunk_id if is_parent_level else parent_id
                child_ids = process_level(heading_content_text, next_level, current_heading, next_parent_id)
                
                # 如果是父级别，记录子文档
                if is_parent_level:
                    parent_child_map[current_chunk_id].extend(child_ids)
                    
                # 将子级的chunk_ids添加到当前级别的结果中
                current_level_chunk_ids.extend(child_ids)
        
        return current_level_chunk_ids
    
    # 从第一级开始处理
    process_level(content, 1)
    
    # 更新每个块的child_ids
    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        if chunk_id in parent_child_map:
            chunk["child_ids"] = parent_child_map[chunk_id]
    
    return chunks

