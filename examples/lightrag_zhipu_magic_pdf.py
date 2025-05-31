import os

from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod
from lightrag import LightRAG
from lightrag.llm import zhipu_complete, zhipu_embedding
from lightrag.utils import EmbeddingFunc

# args
pdf_file_name = "abc.pdf"  # replace with the real pdf path
name_without_suff = pdf_file_name.split(".")[0]

# prepare env
local_image_dir, local_md_dir = "output/images", "output"
image_dir = str(os.path.basename(local_image_dir))

os.makedirs(local_image_dir, exist_ok=True)

image_writer, md_writer = (
    FileBasedDataWriter(local_image_dir),
    FileBasedDataWriter(local_md_dir),
)

# read bytes
reader1 = FileBasedDataReader("")
pdf_bytes = reader1.read(pdf_file_name)  # read the pdf content

# proc
## Create Dataset Instance
ds = PymuDocDataset(pdf_bytes)

## inference
if ds.classify() == SupportedPdfParseMethod.OCR:
    infer_result = ds.apply(doc_analyze, ocr=True)

    ## pipeline
    pipe_result = infer_result.pipe_ocr_mode(image_writer)

else:
    infer_result = ds.apply(doc_analyze, ocr=False)

    ## pipeline
    pipe_result = infer_result.pipe_txt_mode(image_writer)

### draw model result on each page
infer_result.draw_model(os.path.join(local_md_dir, f"{name_without_suff}_model.pdf"))

### get model inference result
model_inference_result = infer_result.get_infer_res()

### draw layout result on each page
pipe_result.draw_layout(os.path.join(local_md_dir, f"{name_without_suff}_layout.pdf"))

### draw spans result on each page
pipe_result.draw_span(os.path.join(local_md_dir, f"{name_without_suff}_spans.pdf"))

### get markdown content
md_content = pipe_result.get_markdown(image_dir)

### dump markdown
pipe_result.dump_md(md_writer, f"{name_without_suff}.md", image_dir)

### get content list content
content_list_content = pipe_result.get_content_list(image_dir)

### dump content list
pipe_result.dump_content_list(
    md_writer, f"{name_without_suff}_content_list.json", image_dir
)

### get middle json
middle_json_content = pipe_result.get_middle_json()

### dump middle json
pipe_result.dump_middle_json(md_writer, f"{name_without_suff}_middle.json")

# 添加 LightRAG 相关配置
WORKING_DIR = "./dickens"
os.environ["ZHIPUAI_API_KEY"] = ""

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

api_key = os.environ.get("ZHIPUAI_API_KEY")
if api_key is None:
    raise Exception("Please set ZHIPU_API_KEY in your environment")

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=zhipu_complete,
    llm_model_name="glm-4-flashx",  # Using the most cost/performance balance model, but you can change it here.
    llm_model_max_async=4,
    llm_model_max_token_size=32768,
    embedding_func=EmbeddingFunc(
        embedding_dim=768,  # Zhipu embedding-3 dimension
        max_token_size=8192,
        func=lambda texts: zhipu_embedding(texts),
    ),
)

# 将转换后的 Markdown 内容写入 book3.txt
book_path = "./book.txt"
with open(book_path, "w", encoding="utf-8") as f:
    f.write(md_content)

# 把 book.txt 文件内容插入到 LightRAG 实例中
with open(book_path, "r", encoding="utf-8") as f:
    rag.insert(f.read())
