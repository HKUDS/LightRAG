# Markdown 层级分块参数配置指南

LightRAG 支持 Markdown 文档的层级分块，通过识别标题层级结构来创建更加语义化的文档块。最新版本中，我们添加了更多可配置参数，使用户可以根据自己的需求自定义分块行为。

## 新增参数说明

`chunking_by_markdown_hierarchical` 函数新增了以下参数：

- `heading_levels`：要处理的标题级别数量，默认为 2（处理到 ## 二级标题）
- `parent_level`：指定父文档的级别，默认为 1（# 一级标题作为父文档）
- `preprocess_headings`：是否预处理标题层级，确保文档只有一个一级标题，默认为 `True`
- `preprocess_attachments`：是否预处理附件标题，确保附件标题为二级标题，默认为 `True`

## 不修改 lightrag.py 的情况下设置参数

我们提供了一种方式，让用户可以在不修改 `lightrag.py` 的情况下设置这些参数。具体方法是通过 `set_hierarchical_chunking_config` 函数设置全局配置参数。

### 基本用法

```python
from lightrag import LightRAG
from lightrag.chunking import set_hierarchical_chunking_config, get_hierarchical_chunking_config
from lightrag.base import ChunkingMode

# 初始化 LightRAG 实例，指定使用层级分块策略
rag = LightRAG(
    chunking_mode=ChunkingMode.HIREARCHIACL,
    # 其他参数...
)

# 设置层级分块参数
set_hierarchical_chunking_config(
    heading_levels=4,         # 处理到 #### 四级标题
    parent_level=2,           # ## 二级标题作为父文档
    preprocess_headings=True, # 预处理标题层级
    preprocess_attachments=False # 不预处理附件标题
)

# 然后正常使用 LightRAG 进行文档处理，将应用上述配置
rag.insert_document("您的文档内容", file_path="example.md")
```

### 查看当前配置

您可以随时查看当前的层级分块参数配置：

```python
from lightrag.chunking import get_hierarchical_chunking_config

# 查看当前配置
current_config = get_hierarchical_chunking_config()
print(current_config)
```

### 参数说明和使用建议

#### heading_levels

该参数决定分块时要处理到哪一级标题。值越大，分块颗粒度越细。

- `heading_levels=1`：仅处理一级标题（#），适合大纲类文档
- `heading_levels=2`：处理到二级标题（##），适合一般技术文档（默认）
- `heading_levels=3`：处理到三级标题（###），适合内容细节较多的文档
- `heading_levels=4` 及以上：处理更深层级标题，适合结构复杂的大型文档

#### parent_level

该参数指定哪一级标题被视为父文档，用于构建文档的层次结构。必须小于等于 `heading_levels`。

- `parent_level=1`：一级标题（#）作为父文档（默认）
- `parent_level=2`：二级标题（##）作为父文档，适合多章节复杂文档
- `parent_level=3` 及以上：更深层级标题作为父文档，适合特定场景的复杂文档

#### preprocess_headings

该参数控制是否对文档标题进行预处理，确保文档只有一个一级标题。

- `preprocess_headings=True`：自动处理多个一级标题，保留第一个作为真正的一级标题，其余降级（默认）
- `preprocess_headings=False`：保持原始标题结构不变，适合已经规范化的文档

#### preprocess_attachments

该参数控制是否预处理附件标题，确保附件标题为二级标题。

- `preprocess_attachments=True`：自动将"附件"相关标题调整为二级标题（默认）
- `preprocess_attachments=False`：保持原始附件标题不变，适合已经规范化的文档

## 完整示例

```python
from lightrag import LightRAG
from lightrag.chunking import set_hierarchical_chunking_config, get_hierarchical_chunking_config
from lightrag.base import ChunkingMode

# 初始化 LightRAG
rag = LightRAG(
    chunking_mode=ChunkingMode.HIREARCHIACL,
    # 其他参数...
)

# 查看默认配置
print("默认配置：")
print(get_hierarchical_chunking_config())

# 设置新配置
set_hierarchical_chunking_config(
    heading_levels=3,
    parent_level=2,
    preprocess_headings=False,
    preprocess_attachments=True
)

# 查看更新后的配置
print("更新后配置：")
print(get_hierarchical_chunking_config())

# 处理文档
rag.insert_document("# 文档标题\n\n## 第一章\n\n内容...\n\n## 第二章\n\n内容...", file_path="example.md")
```

## 注意事项

1. 参数设置是全局的，会影响所有使用层级分块的文档处理。
2. 如果需要为不同文档设置不同参数，请在处理文档前调用 `set_hierarchical_chunking_config` 进行配置更新。
3. 确保 `parent_level` 不大于 `heading_levels`，否则会抛出异常。
4. 参数修改不会影响已经处理过的文档，只对新处理的文档生效。 