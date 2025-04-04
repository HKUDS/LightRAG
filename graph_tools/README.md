# 知识图谱工具 (graph_tools)

这个模块提供了从文本中提取实体和关系，并构建知识图谱的功能。重构自原始的 `tests/entity_extract.py` 脚本，使用了更加模块化、易于维护的结构。

## 功能

- 从文本数据中提取实体和关系
- 规范化实体和关系类型
- 处理文档结构，创建层次化知识图谱
- 生成用于Neo4j图数据库的Cypher查询语句

## 模块结构

- `models.py` - 数据模型定义 (Entity, Relation, Config, LLMTask)
- `config_loader.py` - 配置加载和处理
- `llm_client.py` - LLM API调用客户端
- `prompt_builder.py` - 提示词构建
- `normalization.py` - 数据规范化和清洗
- `cypher_generator.py` - Cypher查询语句生成
- `processing.py` - 核心数据处理逻辑
- `utils.py` - 通用辅助函数
- `main.py` - 程序入口点

## 安装和依赖

要运行此工具，需要先安装必要的依赖：

```bash
pip install pyyaml httpx asyncio
```

## 使用方法

### 环境变量设置

首先设置必要的环境变量：

```bash
export LLM_BINDING_API_KEY="your_api_key_here"
export LLM_BINDING_HOST="https://api.siliconflow.cn/v1"
export LLM_MODEL="Qwen/Qwen2.5-14B-Instruct"
```

可选的环境变量：

```bash
export MAX_CONCURRENT_REQUESTS=5
export REQUEST_DELAY=0.2
export REQUEST_TIMEOUT=60.0
export MAX_RETRIES=3
export LOG_LEVEL=INFO
```

### 命令行使用

使用命令行工具提取实体和关系：

```bash
python -m graph_tools.entity_extract -i INPUT_FILE -o OUTPUT_FILE -c CONFIG_FILE
```

或者使用可执行脚本：

```bash
./graph_tools/entity_extract -i INPUT_FILE -o OUTPUT_FILE -c CONFIG_FILE
```

参数说明：
- `-i, --input`: 输入JSON文件路径 (必须)
- `-o, --output`: 输出Cypher文件路径 (必须)
- `-c, --config`: 配置YAML文件路径 (必须)
- `-l, --log-level`: 日志级别 (可选, 默认使用环境变量LOG_LEVEL或INFO)

### 作为库使用

```python
import asyncio
from graph_tools.config_loader import load_app_config, setup_logging, load_schema_config
from graph_tools.utils import load_json_data, save_cypher_to_file, normalize_data_format
from graph_tools.llm_client import LLMClient
from graph_tools.main import main_async

# 加载配置
app_config = load_app_config()
setup_logging(app_config)

# 异步调用
asyncio.run(main_async(
    input_json_path="path/to/input.json",
    config_path="path/to/config.yaml",
    output_cypher_path="path/to/output.cypher",
    app_config=app_config
))
```

## 配置文件格式

配置文件使用YAML格式，需要包含以下主要部分：

```yaml
schema:
  entity_types_llm: ["组织", "角色", "规定", "主题"]
  all_entity_types: ["Document", "Section", "Organization", "Role", "Statement", "Topic"]
  entity_type_map_cypher: 
    组织: "Organization"
    角色: "Role"
    规定: "Statement"
    主题: "Topic"
  
  relation_types_llm: ["隶属于", "负责", "适用于", "有目的", "提及", "引用", "相关"]
  all_relation_types: ["HAS_SECTION", "HAS_PARENT_SECTION", "CONTAINS", "BELONGS_TO", "RESPONSIBLE_FOR", "APPLIES_TO", "HAS_PURPOSE", "MENTIONS", "REFERENCES", "RELATED_TO"]
  relation_type_map_cypher:
    隶属于: "BELONGS_TO"
    负责: "RESPONSIBLE_FOR"
    适用于: "APPLIES_TO"
    有目的: "HAS_PURPOSE"
    提及: "MENTIONS"
    引用: "REFERENCES"
    相关: "RELATED_TO"

normalization:
  canonical_map:
    "客票发售与预订": "客票发售和预订"
    "客运服务": "旅客服务"

prompts:
  entity_extraction:
    template: "请从以下文本中提取定义的实体类型..."
    definitions: "实体类型定义：组织-指部门或单位..."
  
  relation_extraction:
    template: "请从以下文本中提取定义的关系类型..."
    definitions: "关系类型定义：隶属于-表示组织或角色之间的从属关系..."
```

## 输入数据格式

输入JSON文件应包含chunks数组和document_info：

```json
{
  "document_info": {
    "document_name": "文档标题",
    "main_category": "文档类别",
    "issuing_authority": "发布机构"
  },
  "chunks": [
    {
      "chunk_id": "chunk1",
      "full_doc_id": "doc1",
      "heading": "第一章 总则",
      "content": "本文档内容...",
      "parent_id": null
    },
    {
      "chunk_id": "chunk2",
      "full_doc_id": "doc1",
      "heading": "1.1 目的",
      "content": "本节内容...",
      "parent_id": "chunk1"
    }
  ]
}
```

## 输出格式

输出文件是一系列Cypher语句，可以直接在Neo4j中执行，用于创建知识图谱。 