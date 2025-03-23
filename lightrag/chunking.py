from typing import Any, Callable, List, Dict
import re
from .base import ChunkingMode
from .utils import encode_string_by_tiktoken, decode_tokens_by_tiktoken
from .markdown_chunking import chunking_by_markdown_hierarchical

# 添加一个配置字典，用于存储层级分块的参数
_hierarchical_chunk_config = {
    "heading_levels": 2,  # 默认处理到二级标题
    "parent_level": 1,    # 默认一级标题为父文档
    "preprocess_headings": True,  # 默认预处理标题层级
    "preprocess_attachments": True,  # 默认预处理附件标题
}

# 添加一个函数来设置层级分块的参数
def set_hierarchical_chunking_config(
    heading_levels: int = None,
    parent_level: int = None,
    preprocess_headings: bool = None,
    preprocess_attachments: bool = None
) -> dict:
    """
    设置层级分块的参数配置。
    
    Args:
        heading_levels: 要处理的标题级别数量
        parent_level: 指定父文档的级别
        preprocess_headings: 是否预处理标题层级
        preprocess_attachments: 是否预处理附件标题
    
    Returns:
        当前的配置字典
    """
    global _hierarchical_chunk_config
    
    if heading_levels is not None:
        _hierarchical_chunk_config["heading_levels"] = heading_levels
    if parent_level is not None:
        _hierarchical_chunk_config["parent_level"] = parent_level
    if preprocess_headings is not None:
        _hierarchical_chunk_config["preprocess_headings"] = preprocess_headings
    if preprocess_attachments is not None:
        _hierarchical_chunk_config["preprocess_attachments"] = preprocess_attachments
    
    return _hierarchical_chunk_config.copy()

# 添加一个函数来获取层级分块的参数
def get_hierarchical_chunking_config() -> dict:
    """
    获取当前层级分块的参数配置。
    
    Returns:
        当前的配置字典
    """
    return _hierarchical_chunk_config.copy()

def chunking_by_token_size(
    content: str,
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    overlap_token_size: int = 128,
    max_token_size: int = 1024,
    tiktoken_model: str = "gpt-4o",
) -> list[dict[str, Any]]:
    """Split text by token size."""
    tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)
    results: list[dict[str, Any]] = []
    
    if split_by_character:
        raw_chunks = content.split(split_by_character)
        new_chunks = []
        if split_by_character_only:
            for chunk in raw_chunks:
                _tokens = encode_string_by_tiktoken(chunk, model_name=tiktoken_model)
                new_chunks.append((len(_tokens), chunk))
        else:
            for chunk in raw_chunks:
                _tokens = encode_string_by_tiktoken(chunk, model_name=tiktoken_model)
                if len(_tokens) > max_token_size:
                    for start in range(0, len(_tokens), max_token_size - overlap_token_size):
                        chunk_content = decode_tokens_by_tiktoken(
                            _tokens[start : start + max_token_size],
                            model_name=tiktoken_model,
                        )
                        new_chunks.append((min(max_token_size, len(_tokens) - start), chunk_content))
                else:
                    new_chunks.append((len(_tokens), chunk))
        
        for index, (_len, chunk) in enumerate(new_chunks):
            results.append({
                "tokens": _len,
                "content": chunk.strip(),
                "chunk_order_index": index,
            })
    else:
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

def chunking_by_hierarchical(
    content: str,
    split_by_character: str | None = None,  # 忽略此参数，Markdown 结构化分块不依赖字符分割
    split_by_character_only: bool = False, # 忽略此参数
    overlap_token_size: int = 128,
    max_token_size: int = 1024,
    tiktoken_model: str = "gpt-4o",
    heading_levels: int = None,  # 可以覆盖全局配置
    parent_level: int = None,  # 可以覆盖全局配置
) -> List[Dict[str, Any]]:
    """
    层级分块函数的包装器，使用全局配置或调用时指定的参数。
    
    Args:
        content: Markdown 文档内容
        split_by_character: 忽略此参数
        split_by_character_only: 忽略此参数
        overlap_token_size: 块之间的 token 重叠大小
        max_token_size: 每个块的最大 token 数量
        tiktoken_model: 用于 token 化的模型名称
        heading_levels: 要处理的标题级别数量，覆盖全局配置
        parent_level: 指定父文档的级别，覆盖全局配置
    
    Returns:
        分块结果列表
    """
    # 获取当前配置
    config = get_hierarchical_chunking_config()
    
    # 如果传入了参数，覆盖当前配置
    if heading_levels is not None:
        config["heading_levels"] = heading_levels
    if parent_level is not None:
        config["parent_level"] = parent_level
    
    # 调用实际的分块函数
    return chunking_by_markdown_hierarchical(
        content=content,
        split_by_character=split_by_character,
        split_by_character_only=split_by_character_only,
        overlap_token_size=overlap_token_size,
        max_token_size=max_token_size,
        tiktoken_model=tiktoken_model,
        heading_levels=config["heading_levels"],
        parent_level=config["parent_level"],
        preprocess_headings=config["preprocess_headings"],
        preprocess_attachments=config["preprocess_attachments"]
    )

def chunking_by_markdown(
    content: str,
    overlap_token_size: int = 128,
    max_token_size: int = 1024,
    tiktoken_model: str = "gpt-4o",
    min_chunk_size: int = 100,
) -> list[dict[str, Any]]:
    """Split text by markdown headers while respecting token size limits."""
    # Markdown header pattern
    header_pattern = r'^#{1,6}\s+.+$'
    
    # Split content into lines
    lines = content.split('\n')
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for line in lines:
        line_tokens = len(encode_string_by_tiktoken(line + '\n', model_name=tiktoken_model))
        
        # If line is a header and we have a non-empty chunk, save it
        if re.match(header_pattern, line) and current_chunk and current_tokens >= min_chunk_size:
            chunks.append('\n'.join(current_chunk))
            current_chunk = []
            current_tokens = 0
        
        # Add line to current chunk
        current_chunk.append(line)
        current_tokens += line_tokens
        
        # If chunk exceeds max size, save it
        if current_tokens >= max_token_size:
            chunks.append('\n'.join(current_chunk))
            current_chunk = []
            current_tokens = 0
    
    # Add remaining chunk if any
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    
    # Process chunks to ensure token size limits and add overlap
    results = []
    for index, chunk in enumerate(chunks):
        chunk_tokens = encode_string_by_tiktoken(chunk, model_name=tiktoken_model)
        
        if len(chunk_tokens) > max_token_size:
            # Split large chunks further
            for start in range(0, len(chunk_tokens), max_token_size - overlap_token_size):
                sub_chunk = decode_tokens_by_tiktoken(
                    chunk_tokens[start : start + max_token_size],
                    model_name=tiktoken_model
                )
                results.append({
                    "tokens": min(max_token_size, len(chunk_tokens) - start),
                    "content": sub_chunk.strip(),
                    "chunk_order_index": len(results),
                })
        else:
            results.append({
                "tokens": len(chunk_tokens),
                "content": chunk.strip(),
                "chunk_order_index": len(results),
            })
    
    return results

def get_chunking_function(mode: ChunkingMode) -> Callable:
    """Get the appropriate chunking function based on the mode."""
    if mode == ChunkingMode.TOKEN:
        return chunking_by_token_size
    elif mode == ChunkingMode.MARKDOWN:
        return chunking_by_markdown
    elif mode == ChunkingMode.CHARACTER:
        return lambda content, **kwargs: chunking_by_token_size(content, split_by_character=kwargs.get('split_by_character', '\n'), **kwargs)
    elif mode == ChunkingMode.HYBRID:
        # Hybrid mode first splits by markdown headers, then ensures token size limits
        return lambda content, **kwargs: chunking_by_markdown(content, **kwargs)
    elif mode == ChunkingMode.HIREARCHIACL:
        return lambda content, **kwargs: chunking_by_hierarchical(content, **kwargs)
    else:
        raise ValueError(f"Unsupported chunking mode: {mode}") 