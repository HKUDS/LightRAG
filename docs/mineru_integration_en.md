# MinerU Integration Guide

### About MinerU

MinerU is a powerful open-source tool for extracting high-quality structured data from PDF, image, and office documents. It provides the following features:

- Text extraction while preserving document structure (headings, paragraphs, lists, etc.)
- Handling complex layouts including multi-column formats
- Automatic formula recognition and conversion to LaTeX format
- Image, table, and footnote extraction
- Automatic scanned document detection and OCR application
- Support for multiple output formats (Markdown, JSON)

### Installation

#### Installing MinerU Dependencies

If you have already installed LightRAG but don't have MinerU support, you can add MinerU support by installing the magic-pdf package directly:

```bash
pip install "magic-pdf[full]>=1.2.2" huggingface_hub
```

These are the MinerU-related dependencies required by LightRAG.

#### MinerU Model Weights

MinerU requires model weight files to function properly. After installation, you need to download the required model weights. You can use either Hugging Face or ModelScope to download the models.

##### Option 1: Download from Hugging Face

```bash
pip install huggingface_hub
wget https://github.com/opendatalab/MinerU/raw/master/scripts/download_models_hf.py -O download_models_hf.py
python download_models_hf.py
```

##### Option 2: Download from ModelScope (Recommended for users in China)

```bash
pip install modelscope
wget https://github.com/opendatalab/MinerU/raw/master/scripts/download_models.py -O download_models.py
python download_models.py
```

Both methods will automatically download the model files and configure the model directory in the configuration file. The configuration file is located in your user directory and named `magic-pdf.json`.

> **Note for Windows users**: User directory is at `C:\Users\username`
> **Note for Linux users**: User directory is at `/home/username`
> **Note for macOS users**: User directory is at `/Users/username`

#### Optional: LibreOffice Installation

To process Office documents (DOC, DOCX, PPT, PPTX), you need to install LibreOffice:

**Linux/macOS:**
```bash
apt-get/yum/brew install libreoffice
```

**Windows:**
1. Install LibreOffice
2. Add the installation directory to your PATH: `install_dir\LibreOffice\program`

### Using MinerU Parser

#### Basic Usage

```python
from lightrag.mineru_parser import MineruParser

# Parse a PDF document
content_list, md_content = MineruParser.parse_pdf('path/to/document.pdf', 'output_dir')

# Parse an image
content_list, md_content = MineruParser.parse_image('path/to/image.jpg', 'output_dir')

# Parse an Office document
content_list, md_content = MineruParser.parse_office_doc('path/to/document.docx', 'output_dir')

# Auto-detect and parse any supported document type
content_list, md_content = MineruParser.parse_document('path/to/file', 'auto', 'output_dir')
```

#### RAGAnything Integration

In RAGAnything, you can directly use file paths as input to the `process_document_complete` method to process documents. Here's a complete configuration example:

```python
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.raganything import RAGAnything


# Initialize RAGAnything
rag = RAGAnything(
    working_dir="./rag_storage",  # Working directory
    llm_model_func=lambda prompt, system_prompt=None, history_messages=[], **kwargs: openai_complete_if_cache(
        "gpt-4o-mini",  # Model to use
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key="your-api-key",  # Replace with your API key
        base_url="your-base-url",  # Replace with your API base URL
        **kwargs,
    ),
    vision_model_func=lambda prompt, system_prompt=None, history_messages=[], image_data=None, **kwargs: openai_complete_if_cache(
        "gpt-4o",  # Vision model
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
        api_key="your-api-key",  # Replace with your API key
        base_url="your-base-url",  # Replace with your API base URL
        **kwargs,
    ) if image_data else openai_complete_if_cache(
        "gpt-4o-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key="your-api-key",  # Replace with your API key
        base_url="your-base-url",  # Replace with your API base URL
        **kwargs,
    ),
    embedding_func=lambda texts: openai_embed(
        texts,
        model="text-embedding-3-large",
        api_key="your-api-key",  # Replace with your API key
        base_url="your-base-url",  # Replace with your API base URL
    ),
    embedding_dim=3072,
    max_token_size=8192
)

# Process a single file
await rag.process_document_complete(
    file_path="path/to/document.pdf",
    output_dir="./output",
    parse_method="auto"
)

# Query the processed document
result = await rag.query_with_multimodal(
    "What is the main content of the document?",
    mode="hybrid"
)

```

MinerU categorizes document content into text, formulas, images, and tables, processing each with its corresponding ingestion type:
- Text content: `ingestion_type='text'`
- Image content: `ingestion_type='image'`
- Table content: `ingestion_type='table'`
- Formula content: `ingestion_type='equation'`

#### Query Examples

Here are some common query examples:

```python
# Query text content
result = await rag.query_with_multimodal(
    "What is the main topic of the document?",
    mode="hybrid"
)

# Query image-related content
result = await rag.query_with_multimodal(
    "Describe the images and figures in the document",
    mode="hybrid"
)

# Query table-related content
result = await rag.query_with_multimodal(
    "Tell me about the experimental results and data tables",
    mode="hybrid"
)
```

#### Command Line Tool

We also provide a command-line tool for document parsing:

```bash
python examples/mineru_example.py path/to/document.pdf
```

Optional parameters:
- `--output` or `-o`: Specify output directory
- `--method` or `-m`: Choose parsing method (auto, ocr, txt)
- `--stats`: Display content statistics

### Output Format

MinerU generates three files for each parsed document:

1. `{filename}.md` - Markdown representation of the document
2. `{filename}_content_list.json` - Structured JSON content
3. `{filename}_model.json` - Detailed model parsing results

The `content_list.json` file contains all structured content extracted from the document, including:
- Text blocks (body text, headings, etc.)
- Images (paths and optional captions)
- Tables (table content and optional captions)
- Lists
- Formulas

### Troubleshooting

If you encounter issues with MinerU:

1. Check that model weights are correctly downloaded
2. Ensure you have sufficient RAM (16GB+ recommended)
3. For CUDA acceleration issues, see [MinerU documentation](https://mineru.readthedocs.io/en/latest/additional_notes/faq.html)
4. If parsing Office documents fails, verify LibreOffice is properly installed
5. If you encounter `pickle.UnpicklingError: invalid load key, 'v'.`, it might be due to an incomplete model download. Try re-downloading the models.
6. For users with newer graphics cards (H100, etc.) and garbled OCR text, try upgrading the CUDA version used by Paddle:
   ```bash
   pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu123/
   ```
7. If you encounter a "filename too long" error, the latest version of MineruParser includes logic to automatically handle this issue.

#### Updating Existing Models

If you have previously downloaded models and need to update them, you can simply run the download script again. The script will update the model directory to the latest version.

### Advanced Configuration

The MinerU configuration file `magic-pdf.json` supports various customization options, including:

- Model directory path
- OCR engine selection
- GPU acceleration settings
- Cache settings

For complete configuration options, refer to the [MinerU official documentation](https://mineru.readthedocs.io/).

### Using Modal Processors Directly

You can also use LightRAG's modal processors directly without going through MinerU. This is useful when you want to process specific types of content or have more control over the processing pipeline.

Each modal processor returns a tuple containing:
1. A description of the processed content
2. Entity information that can be used for further processing or storage

The processors support different types of content:
- `ImageModalProcessor`: Processes images with captions and footnotes
- `TableModalProcessor`: Processes tables with captions and footnotes
- `EquationModalProcessor`: Processes mathematical equations in LaTeX format
- `GenericModalProcessor`: A base processor that can be extended for custom content types

> **Note**: A complete working example can be found in `examples/modalprocessors_example.py`. You can run it using:
> ```bash
> python examples/modalprocessors_example.py --api-key YOUR_API_KEY
> ```

<details>
<summary> Here's an example of how to use different modal processors: </summary>

```python
from lightrag.modalprocessors import (
    ImageModalProcessor,
    TableModalProcessor,
    EquationModalProcessor,
    GenericModalProcessor
)

# Initialize LightRAG
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

# Process an image
image_processor = ImageModalProcessor(
    lightrag=lightrag,
    modal_caption_func=vision_model_func
)

image_content = {
    "img_path": "image.jpg",
    "img_caption": ["Example image caption"],
    "img_footnote": ["Example image footnote"]
}

description, entity_info = await image_processor.process_multimodal_content(
    modal_content=image_content,
    content_type="image",
    file_path="image_example.jpg",
    entity_name="Example Image"
)

# Process a table
table_processor = TableModalProcessor(
    lightrag=lightrag,
    modal_caption_func=llm_model_func
)

table_content = {
    "table_body": """
    | Name | Age | Occupation |
    |------|-----|------------|
    | John | 25  | Engineer   |
    | Mary | 30  | Designer   |
    """,
    "table_caption": ["Employee Information Table"],
    "table_footnote": ["Data updated as of 2024"]
}

description, entity_info = await table_processor.process_multimodal_content(
    modal_content=table_content,
    content_type="table",
    file_path="table_example.md",
    entity_name="Employee Table"
)

# Process an equation
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
    entity_name="Mass-Energy Equivalence"
)
```

</details>
