# 图数据库实体向量索引构建工具

这个工具用于从Memgraph图数据库中获取实体数据，计算它们的向量嵌入，并将结果存储在向量数据库中。

## 功能特性

- 从Memgraph图数据库批量获取实体
- 使用SiliconFlow API计算实体文本的嵌入向量
- 自动处理大批量数据，包括批处理和重试机制
- 将向量索引保存到磁盘，支持增量更新

## 安装

1. 确保您已安装Python 3.8或更高版本

2. 安装所需依赖：

```bash
pip install -r requirements.txt
```

## 配置

在使用前，请确保正确配置以下环境变量（可以在.env文件中设置）：

```
# 必需的配置
SILICONFLOW_API_KEY=your_api_key_here     # SiliconFlow API密钥
MEMGRAPH_HOST=localhost                   # Memgraph服务器主机
MEMGRAPH_PORT=7687                        # Memgraph服务器端口

# 可选配置
WORKING_DIR=/app/data/rag_passenger       # 工作目录
VECTOR_STORAGE_PATH=/path/to/storage      # 向量存储路径
ENTITY_BATCH_SIZE=100                     # 每批处理的实体数量
MAX_CONCURRENT_REQUESTS=5                 # 最大并发请求数
REQUEST_DELAY=0.2                         # 请求间隔时间(秒)
REQUEST_TIMEOUT=60.0                      # 请求超时时间(秒)
```

## 使用方法

直接运行主脚本：

```bash
python entity_embedding_indexer.py
```

程序将自动：
1. 连接到Memgraph数据库
2. 分批获取实体节点
3. 为每个实体生成文本表示
4. 调用SiliconFlow API计算嵌入向量
5. 将向量和元数据存储到磁盘

## 向量表示

对于每个实体，将提取以下属性组合成文本进行嵌入：
- 实体类型（标签）
- 名称
- 描述（如果与名称不同）
- 主分类
- 来源

例如：
```
类型: Statement, 名称: 在窗口售票故障，互联网正常的情况下, 主分类: 客运管理, 来源: 中国铁路广州局集团有限公司关于发布《广州局集团公司客票发售和预订系统（含互联网售票部分）应急预案》的通知
```

## 向量存储

向量存储使用JSON格式保存，包含两个主要部分：
- vectors: 所有实体的向量数组
- metadata: 与向量对应的元数据信息（包含entity_uuid、实体标签、名称等）

对于已存在的实体（基于uuid匹配），系统会更新其向量和元数据，而不是创建重复记录。
