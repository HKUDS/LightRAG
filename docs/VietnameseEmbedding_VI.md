# Tích hợp Vietnamese Embedding cho LightRAG

Tài liệu này hướng dẫn sử dụng mô hình **AITeamVN/Vietnamese_Embedding** với LightRAG để tăng cường khả năng truy xuất thông tin tiếng Việt.

## Thông tin Mô hình

- **Mô hình**: [AITeamVN/Vietnamese_Embedding](https://huggingface.co/AITeamVN/Vietnamese_Embedding)
- **Mô hình gốc**: [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)
- **Loại**: Sentence Transformer
- **Độ dài tối đa**: 2048 tokens
- **Số chiều embedding**: 1024
- **Hàm tương đồng**: Dot product similarity
- **Ngôn ngữ**: Tiếng Việt (cũng hỗ trợ các ngôn ngữ khác từ BGE-M3)
- **Dữ liệu huấn luyện**: ~300,000 bộ ba (query, văn bản dương, văn bản âm) tiếng Việt

## Tính năng

✅ **Embedding tiếng Việt chất lượng cao** - Được fine-tune đặc biệt cho văn bản tiếng Việt  
✅ **Hỗ trợ đa ngôn ngữ** - Kế thừa khả năng đa ngôn ngữ từ BGE-M3  
✅ **Xử lý văn bản dài** - Hỗ trợ tới 2048 tokens mỗi đầu vào  
✅ **Xử lý hiệu quả** - Tự động phát hiện thiết bị (CUDA/MPS/CPU)  
✅ **Embedding chuẩn hóa** - Sẵn sàng cho dot product similarity  
✅ **Tích hợp dễ dàng** - Thay thế trực tiếp các hàm embedding khác  

## Cài đặt

### 1. Cài đặt LightRAG

```bash
cd LightRAG
pip install -e .
```

### 2. Cài đặt các thư viện cần thiết

Các thư viện sau sẽ được tự động cài đặt:
- `transformers`
- `torch`
- `numpy`

### 3. Thiết lập HuggingFace Token

Bạn cần token HuggingFace để truy cập mô hình:

```bash
export HUGGINGFACE_API_KEY=
# hoặc
export HF_TOKEN=
```

Lấy token tại: https://huggingface.co/settings/tokens

## Bắt đầu nhanh

### Ví dụ đơn giản

```python
import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete
from lightrag.llm.vietnamese_embed import vietnamese_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import EmbeddingFunc

WORKING_DIR = "./vietnamese_rag_storage"

async def main():
    # Lấy HuggingFace token
    hf_token = os.environ.get("HUGGINGFACE_API_KEY")
    
    # Khởi tạo LightRAG với Vietnamese embedding
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=2048,
            func=lambda texts: vietnamese_embed(
                texts,
                model_name="AITeamVN/Vietnamese_Embedding",
                token=hf_token
            )
        ),
    )
    
    # Khởi tạo storage và pipeline
    await rag.initialize_storages()
    await initialize_pipeline_status()
    
    # Chèn văn bản tiếng Việt
    await rag.ainsert("Việt Nam là một quốc gia nằm ở Đông Nam Á.")
    
    # Truy vấn
    result = await rag.aquery(
        "Việt Nam ở đâu?",
        param=QueryParam(mode="hybrid")
    )
    print(result)
    
    await rag.finalize_storages()

if __name__ == "__main__":
    asyncio.run(main())
```

### Sử dụng với file `.env`

Tạo file `.env` trong thư mục dự án:

```env
# HuggingFace Token cho Vietnamese Embedding
HUGGINGFACE_API_KEY=

# Cấu hình LLM
OPENAI_API_KEY=your_openai_key_here
LLM_BINDING=openai
LLM_MODEL=gpt-4o-mini

# Cấu hình Embedding
EMBEDDING_MODEL=AITeamVN/Vietnamese_Embedding
EMBEDDING_DIM=1024
```

## Các script ví dụ

### 1. Ví dụ đơn giản
```bash
python examples/lightrag_vietnamese_embedding_simple.py
```

Ví dụ tối thiểu về xử lý văn bản tiếng Việt.

### 2. Demo đầy đủ
```bash
python examples/vietnamese_embedding_demo.py
```

Demo toàn diện bao gồm:
- Xử lý văn bản tiếng Việt
- Xử lý văn bản tiếng Anh (hỗ trợ đa ngôn ngữ)
- Xử lý văn bản hỗn hợp
- Nhiều ví dụ truy vấn

## Tài liệu API

### `vietnamese_embed()`

Tạo embeddings cho văn bản sử dụng mô hình Vietnamese Embedding.

```python
async def vietnamese_embed(
    texts: list[str],
    model_name: str = "AITeamVN/Vietnamese_Embedding",
    token: str | None = None,
) -> np.ndarray
```

**Tham số:**
- `texts` (list[str]): Danh sách các văn bản cần embedding
- `model_name` (str): Tên mô hình trên HuggingFace
- `token` (str, optional): HuggingFace API token (đọc từ env nếu không cung cấp)

**Trả về:**
- `np.ndarray`: Mảng embeddings với shape (len(texts), 1024)

**Ví dụ:**
```python
from lightrag.llm.vietnamese_embed import vietnamese_embed

texts = ["Xin chào", "Tạm biệt", "Cảm ơn"]
embeddings = await vietnamese_embed(texts)
print(embeddings.shape)  # (3, 1024)
```

## Sử dụng nâng cao

### Lựa chọn thiết bị

Mô hình tự động phát hiện và sử dụng thiết bị tốt nhất:
1. CUDA (nếu có)
2. MPS (cho Apple Silicon)
3. CPU (dự phòng)

Bật debug logging để xem thiết bị đang sử dụng:

```python
from lightrag.utils import setup_logger

setup_logger("lightrag", level="DEBUG")
```

### Xử lý batch

Hàm embedding hỗ trợ xử lý batch hiệu quả:

```python
# Xử lý nhiều văn bản hiệu quả
large_batch = ["Văn bản 1", "Văn bản 2", ..., "Văn bản 1000"]
embeddings = await vietnamese_embed(large_batch)
```

## Khắc phục sự cố

### Vấn đề: "No HuggingFace token found"

**Giải pháp:** Thiết lập biến môi trường:
```bash
export HUGGINGFACE_API_KEY="your_token"
# hoặc
export HF_TOKEN="your_token"
```

### Vấn đề: "Model download fails"

**Giải pháp:** 
1. Kiểm tra kết nối internet
2. Xác thực token HuggingFace hợp lệ
3. Đảm bảo đủ dung lượng ổ đĩa (~2 GB)

### Vấn đề: "Out of memory error"

**Giải pháp:**
1. Giảm kích thước batch
2. Sử dụng CPU thay vì GPU (chậm hơn nhưng dùng ít bộ nhớ hơn)
3. Đóng các ứng dụng khác đang dùng GPU/RAM

### Vấn đề: "Embedding generation chậm"

**Giải pháp:**
1. Đảm bảo đang sử dụng GPU (kiểm tra logs)
2. Cài đặt PyTorch với CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu118`
3. Giảm max_token_size nếu văn bản ngắn hơn

## So sánh với các mô hình embedding khác

| Mô hình | Số chiều | Max Tokens | Ngôn ngữ | Fine-tuned cho tiếng Việt |
|---------|----------|------------|----------|---------------------------|
| Vietnamese_Embedding | 1024 | 2048 | Đa ngôn ngữ | ✅ Có |
| BGE-M3 | 1024 | 8192 | Đa ngôn ngữ | ❌ Không |
| text-embedding-3-large | 3072 | 8191 | Đa ngôn ngữ | ❌ Không |
| text-embedding-3-small | 1536 | 8191 | Đa ngôn ngữ | ❌ Không |

## Hỗ trợ

Để báo cáo vấn đề về tích hợp Vietnamese embedding:
- Mở issue trên [LightRAG GitHub](https://github.com/HKUDS/LightRAG/issues)
- Gắn tag `vietnamese-embedding`

Để báo cáo vấn đề về mô hình:
- Truy cập [AITeamVN/Vietnamese_Embedding](https://huggingface.co/AITeamVN/Vietnamese_Embedding)

## Giấy phép

Tích hợp này tuân theo giấy phép của LightRAG. Mô hình Vietnamese_Embedding có thể có điều khoản giấy phép riêng - vui lòng kiểm tra [trang mô hình](https://huggingface.co/AITeamVN/Vietnamese_Embedding) để biết chi tiết.

## Lời cảm ơn

- **AITeamVN** đã huấn luyện và phát hành mô hình Vietnamese_Embedding
- **BAAI** cho mô hình gốc BGE-M3
- **Nhóm LightRAG** cho framework RAG xuất sắc
