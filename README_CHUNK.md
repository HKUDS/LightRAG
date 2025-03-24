# Markdown结构化分块工具

这个工具集用于将Markdown文档按照标题结构进行智能分块，并输出为JSON格式。

## 功能特点

- 自动处理Markdown文档的层次结构
- 预处理多个一级标题，确保层级合理
- 识别并调整附件标题为适当的层级
- 按标题层次进行分块，保留结构关系
- 支持块内容的token计数
- 维护块之间的父子关系

## 使用方法

### 1. 处理Markdown文件并输出为JSON

```bash
python test_output.py <markdown文件路径> [输出json文件路径]
```

示例：
```bash
python test_output.py test_markdown.md test.json
```

### 2. 查看生成的JSON文件

```bash
python view_chunks.py <json文件路径> [选项]
```

可用选项：
- `--search`, `-s`: 按标题关键词搜索块
- `--id`, `-i`: 按ID查找块
- `--tree`, `-t`: 显示层级结构
- `--no-content`, `-n`: 不显示内容
- `--detail`, `-d`: 显示详细级别(1或2)

示例：
```bash
# 查看文档层级结构
python view_chunks.py test.json --tree

# 搜索包含特定关键词的块
python view_chunks.py test.json --search "附件"

# 查看特定ID的块详情
python view_chunks.py test.json --id chunk_1_0

# 只显示基本信息，不显示内容
python view_chunks.py test.json --no-content
```

## 输出JSON格式

每个块包含以下字段：

- `content`: 块的文本内容
- `tokens`: 块的token数量
- `chunk_order_index`: 块在文档中的顺序索引
- `structure_level`: 块的结构级别(chapter, section, subsection等)
- `heading_text`: 块的标题文本
- `chunk_id`: 块的唯一ID
- `parent_id`: 父块的ID
- `child_ids`: 子块ID列表

## 注意事项

- 处理大型文档时，建议将heading_levels参数设置为较小的值(2或3)以避免生成过多的小块
- 默认的最大块大小为1024个token，可以根据需要调整
- 块之间的重叠大小默认为128个token，可以根据需要调整 