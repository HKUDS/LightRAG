
## API 服务器实现

LightRAG also provides a FastAPI-based server implementation for RESTful API access to RAG operations. This allows you to run LightRAG as a service and interact with it through HTTP requests.
LightRAG 还提供基于 FastAPI 的服务器实现，用于对 RAG 操作进行 RESTful API 访问。这允许您将 LightRAG 作为服务运行并通过 HTTP 请求与其交互。

### 设置 API 服务器
<details>
<summary>单击展开设置说明</summary>

1. 首先，确保您具有所需的依赖项:
```bash
pip install fastapi uvicorn pydantic
```

2. 设置您的环境变量:
```bash
export RAG_DIR="your_index_directory"  # Optional: Defaults to "index_default"
export OPENAI_BASE_URL="Your OpenAI API base URL"  # Optional: Defaults to "https://api.openai.com/v1"
export OPENAI_API_KEY="Your OpenAI API key"  # Required
export LLM_MODEL="Your LLM model" # Optional: Defaults to "gpt-4o-mini"
export EMBEDDING_MODEL="Your embedding model" # Optional: Defaults to "text-embedding-3-large"
```

3. 运行API服务器:
```bash
python examples/lightrag_api_openai_compatible_demo.py
```

服务器将启动于 `http://0.0.0.0:8020`.
</details>

### API端点

API服务器提供以下端点:

#### 1. 查询端点
<details>
<summary>点击查看查询端点详情</summary>

- **URL:** `/query`
- **Method:** POST
- **Body:**
```json
{
    "query": "Your question here",
    "mode": "hybrid",  // Can be "naive", "local", "global", or "hybrid"
    "only_need_context": true // Optional: Defaults to false, if true, only the referenced context will be returned, otherwise the llm answer will be returned
}
```
- **Example:**
```bash
curl -X POST "http://127.0.0.1:8020/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "What are the main themes?", "mode": "hybrid"}'
```
</details>

#### 2. 插入文本端点
<details>
<summary>单击可查看插入文本端点详细信息</summary>

- **URL:** `/insert`
- **Method:** POST
- **Body:**
```json
{
    "text": "Your text content here"
}
```
- **Example:**
```bash
curl -X POST "http://127.0.0.1:8020/insert" \
     -H "Content-Type: application/json" \
     -d '{"text": "Content to be inserted into RAG"}'
```
</details>

#### 3. 插入文件端点
<details>
<summary>单击查看插入文件端点详细信息</summary>

- **URL:** `/insert_file`
- **Method:** POST
- **Body:**
```json
{
    "file_path": "path/to/your/file.txt"
}
```
- **Example:**
```bash
curl -X POST "http://127.0.0.1:8020/insert_file" \
     -H "Content-Type: application/json" \
     -d '{"file_path": "./book.txt"}'
```
</details>

#### 4. 健康检查端点
<details>
<summary>点击查看健康检查端点详细信息</summary>

- **URL:** `/health`
- **Method:** GET
- **Example:**
```bash
curl -X GET "http://127.0.0.1:8020/health"
```
</details>

### 配置

可以使用环境变量配置API服务器:
- `RAG_DIR`: 存放RAG索引的目录 (default: "index_default")
- 应在代码中为您的特定 LLM 和嵌入模型提供商配置 API 密钥和基本 URL
