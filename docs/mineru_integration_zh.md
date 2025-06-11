# MinerU 集成指南

### 关于 MinerU

MinerU 是一个强大的开源工具，用于从 PDF、图像和 Office 文档中提取高质量的结构化数据。它提供以下功能：

- 保留文档结构（标题、段落、列表等）的文本提取
- 处理包括多列格式在内的复杂布局
- 自动识别并将公式转换为 LaTeX 格式
- 提取图像、表格和脚注
- 自动检测扫描文档并应用 OCR
- 支持多种输出格式（Markdown、JSON）

### 安装

#### 安装 MinerU 依赖

如果您已经安装了 LightRAG，但没有 MinerU 支持，您可以通过安装 magic-pdf 包来直接添加 MinerU 支持：

```bash
pip install "magic-pdf[full]>=1.2.2" huggingface_hub
```

这些是 LightRAG 所需的 MinerU 相关依赖项。

#### MinerU 模型权重

MinerU 需要模型权重文件才能正常运行。安装后，您需要下载所需的模型权重。您可以使用 Hugging Face 或 ModelScope 下载模型。

##### 选项 1：从 Hugging Face 下载

```bash
pip install huggingface_hub
wget https://github.com/opendatalab/MinerU/raw/master/scripts/download_models_hf.py -O download_models_hf.py
python download_models_hf.py
```

##### 选项 2：从 ModelScope 下载（推荐中国用户使用）

```bash
pip install modelscope
wget https://github.com/opendatalab/MinerU/raw/master/scripts/download_models.py -O download_models.py
python download_models.py
```

两种方法都会自动下载模型文件并在配置文件中配置模型目录。配置文件位于用户目录中，名为 `magic-pdf.json`。

> **Windows 用户注意**：用户目录位于 `C:\Users\用户名`
> **Linux 用户注意**：用户目录位于 `/home/用户名`
> **macOS 用户注意**：用户目录位于 `/Users/用户名`

#### 可选：安装 LibreOffice

要处理 Office 文档（DOC、DOCX、PPT、PPTX），您需要安装 LibreOffice：

**Linux/macOS：**
```bash
apt-get/yum/brew install libreoffice
```

**Windows：**
1. 安装 LibreOffice
2. 将安装目录添加到 PATH 环境变量：`安装目录\LibreOffice\program`

### 使用 MinerU 解析器

#### 基本用法

```python
from lightrag.mineru_parser import MineruParser

# 解析 PDF 文档
content_list, md_content = MineruParser.parse_pdf('path/to/document.pdf', 'output_dir')

# 解析图像
content_list, md_content = MineruParser.parse_image('path/to/image.jpg', 'output_dir')

# 解析 Office 文档
content_list, md_content = MineruParser.parse_office_doc('path/to/document.docx', 'output_dir')

# 自动检测并解析任何支持的文档类型
content_list, md_content = MineruParser.parse_document('path/to/file', 'auto', 'output_dir')
```

#### RAGAnything 集成

在 RAGAnything 中，您可以直接使用文件路径作为 `process_document_complete` 方法的输入来处理文档。以下是一个完整的配置示例：

```python
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.raganything import RAGAnything


# 初始化 RAGAnything
rag = RAGAnything(
    working_dir="./rag_storage",  # 工作目录
    llm_model_func=lambda prompt, system_prompt=None, history_messages=[], **kwargs: openai_complete_if_cache(
        "gpt-4o-mini",  # 使用的模型
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key="your-api-key",  # 替换为您的 API 密钥
        base_url="your-base-url",  # 替换为您的 API 基础 URL
        **kwargs,
    ),
    vision_model_func=lambda prompt, system_prompt=None, history_messages=[], image_data=None, **kwargs: openai_complete_if_cache(
        "gpt-4o",  # 视觉模型
        "",
        system_prompt=None,
        history_messages=[],
        messages=[
            {"role": "system", "content": system_prompt} if system_prompt else None,
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}"
                    }
                }
            ]} if image_data else {"role": "user", "content": prompt}
        ],
        api_key="your-api-key",  # 替换为您的 API 密钥
        base_url="your-base-url",  # 替换为您的 API 基础 URL
        **kwargs,
    ) if image_data else openai_complete_if_cache(
        "gpt-4o-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key="your-api-key",  # 替换为您的 API 密钥
        base_url="your-base-url",  # 替换为您的 API 基础 URL
        **kwargs,
    ),
    embedding_func=lambda texts: openai_embed(
        texts,
        model="text-embedding-3-large",
        api_key="your-api-key",  # 替换为您的 API 密钥
        base_url="your-base-url",  # 替换为您的 API 基础 URL
    ),
    embedding_dim=3072,
    max_token_size=8192
)

# 处理单个文件
await rag.process_document_complete(
    file_path="path/to/document.pdf",
    output_dir="./output",
    parse_method="auto"
)

# 查询处理后的文档
result = await rag.query_with_multimodal(
    "What is the main content of the document?",
    mode="hybrid"
)
```

MinerU 会将文档内容分类为文本、公式、图像和表格，分别使用相应的摄入类型进行处理：
- 文本内容：`ingestion_type='text'`
- 图像内容：`ingestion_type='image'`
- 表格内容：`ingestion_type='table'`
- 公式内容：`ingestion_type='equation'`

#### 查询示例

以下是一些常见的查询示例：

```python
# 查询文本内容
result = await rag.query_with_multimodal(
    "What is the main topic of the document?",
    mode="hybrid"
)

# 查询图片相关内容
result = await rag.query_with_multimodal(
    "Describe the images and figures in the document",
    mode="hybrid"
)

# 查询表格相关内容
result = await rag.query_with_multimodal(
    "Tell me about the experimental results and data tables",
    mode="hybrid"
)
```

#### 命令行工具

我们还提供了一个用于文档解析的命令行工具：

```bash
python examples/mineru_example.py path/to/document.pdf
```

可选参数：
- `--output` 或 `-o`：指定输出目录
- `--method` 或 `-m`：选择解析方法（auto、ocr、txt）
- `--stats`：显示内容统计信息

### 输出格式

MinerU 为每个解析的文档生成三个文件：

1. `{文件名}.md` - 文档的 Markdown 表示
2. `{文件名}_content_list.json` - 结构化 JSON 内容
3. `{文件名}_model.json` - 详细的模型解析结果

`content_list.json` 文件包含从文档中提取的所有结构化内容，包括：
- 文本块（正文、标题等）
- 图像（路径和可选的标题）
- 表格（表格内容和可选的标题）
- 列表
- 公式

### 疑难解答

如果您在使用 MinerU 时遇到问题：

1. 检查模型权重是否正确下载
2. 确保有足够的内存（建议 16GB+）
3. 对于 CUDA 加速问题，请参阅 [MinerU 文档](https://mineru.readthedocs.io/en/latest/additional_notes/faq.html)
4. 如果解析 Office 文档失败，请验证 LibreOffice 是否正确安装
5. 如果遇到 `pickle.UnpicklingError: invalid load key, 'v'.`，可能是因为模型下载不完整。尝试重新下载模型。
6. 对于使用较新显卡（H100 等）并出现 OCR 文本乱码的用户，请尝试升级 Paddle 使用的 CUDA 版本：
   ```bash
   pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu123/
   ```
7. 如果遇到 "文件名太长" 错误，最新版本的 MineruParser 已经包含了自动处理此问题的逻辑。

#### 更新现有模型

如果您之前已经下载了模型并需要更新它们，只需再次运行下载脚本即可。脚本将更新模型目录到最新版本。

### 高级配置

MinerU 配置文件 `magic-pdf.json` 支持多种自定义选项，包括：

- 模型目录路径
- OCR 引擎选择
- GPU 加速设置
- 缓存设置

有关完整的配置选项，请参阅 [MinerU 官方文档](https://mineru.readthedocs.io/)。

### 直接使用模态处理器

您也可以直接使用 LightRAG 的模态处理器，而不需要通过 MinerU。这在您想要处理特定类型的内容或对处理流程有更多控制时特别有用。

每个模态处理器都会返回一个包含以下内容的元组：
1. 处理后内容的描述
2. 可用于进一步处理或存储的实体信息

处理器支持不同类型的内容：
- `ImageModalProcessor`：处理带有标题和脚注的图像
- `TableModalProcessor`：处理带有标题和脚注的表格
- `EquationModalProcessor`：处理 LaTeX 格式的数学公式
- `GenericModalProcessor`：可用于扩展自定义内容类型的基础处理器

> **注意**：完整的可运行示例可以在 `examples/modalprocessors_example.py` 中找到。您可以使用以下命令运行它：
> ```bash
> python examples/modalprocessors_example.py --api-key YOUR_API_KEY
> ```

<details>
<summary> 使用不同模态处理器的示例 </summary>

```python
from lightrag.modalprocessors import (
    ImageModalProcessor,
    TableModalProcessor,
    EquationModalProcessor,
    GenericModalProcessor
)

# 初始化 LightRAG
lightrag = LightRAG(
    working_dir="./rag_storage",
    embedding_func=lambda texts: openai_embed(
        texts,
        model="text-embedding-3-large",
        api_key="your-api-key",
        base_url="your-base-url",
    ),
    llm_model_func=lambda prompt, system_prompt=None, history_messages=[], **kwargs: openai_complete_if_cache(
        "gpt-4o-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key="your-api-key",
        base_url="your-base-url",
        **kwargs,
    ),
)

# 处理图像
image_processor = ImageModalProcessor(
    lightrag=lightrag,
    modal_caption_func=vision_model_func
)

image_content = {
    "img_path": "image.jpg",
    "img_caption": ["示例图像标题"],
    "img_footnote": ["示例图像脚注"]
}

description, entity_info = await image_processor.process_multimodal_content(
    modal_content=image_content,
    content_type="image",
    file_path="image_example.jpg",
    entity_name="示例图像"
)

# 处理表格
table_processor = TableModalProcessor(
    lightrag=lightrag,
    modal_caption_func=llm_model_func
)

table_content = {
    "table_body": """
    | 姓名 | 年龄 | 职业 |
    |------|-----|------|
    | 张三 | 25  | 工程师 |
    | 李四 | 30  | 设计师 |
    """,
    "table_caption": ["员工信息表"],
    "table_footnote": ["数据更新至2024年"]
}

description, entity_info = await table_processor.process_multimodal_content(
    modal_content=table_content,
    content_type="table",
    file_path="table_example.md",
    entity_name="员工表格"
)

# 处理公式
equation_processor = EquationModalProcessor(
    lightrag=lightrag,
    modal_caption_func=llm_model_func
)

equation_content = {
    "text": "E = mc^2",
    "text_format": "LaTeX"
}

description, entity_info = await equation_processor.process_multimodal_content(
    modal_content=equation_content,
    content_type="equation",
    file_path="equation_example.txt",
    entity_name="质能方程"
)
```
</details>
