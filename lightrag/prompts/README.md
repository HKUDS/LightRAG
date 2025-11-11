# LightRAG Prompts

Thư mục này chứa tất cả các prompt templates được sử dụng trong LightRAG.

## Cấu trúc

Các prompts được tổ chức thành các file text riêng biệt thay vì hardcode trong Python code:

### Main Prompts

- `entity_extraction_system_prompt.md` - System prompt cho entity extraction
- `entity_extraction_user_prompt.md` - User prompt cho entity extraction
- `entity_continue_extraction_user_prompt.md` - Prompt để tiếp tục extraction
- `summarize_entity_descriptions.md` - Prompt để tổng hợp entity descriptions
- `fail_response.md` - Response khi không tìm thấy context
- `rag_response.md` - Prompt cho RAG response với knowledge graph
- `naive_rag_response.md` - Prompt cho naive RAG response
- `kg_query_context.md` - Template cho knowledge graph query context
- `naive_query_context.md` - Template cho naive query context
- `keywords_extraction.md` - Prompt cho keyword extraction

### Examples

- `entity_extraction_example_1.md` - Ví dụ 1: Narrative text
- `entity_extraction_example_2.md` - Ví dụ 2: Financial/market data
- `entity_extraction_example_3.md` - Ví dụ 3: Sports event
- `keywords_extraction_example_1.md` - Ví dụ về international trade
- `keywords_extraction_example_2.md` - Ví dụ về deforestation
- `keywords_extraction_example_3.md` - Ví dụ về education

## Cách sử dụng

Các prompts được load tự động khi import module `lightrag.prompt`:

```python
from lightrag.prompt import PROMPTS

# Truy cập prompt
system_prompt = PROMPTS["entity_extraction_system_prompt"]

# Format với parameters
formatted = system_prompt.format(
    entity_types="person, organization, location",
    tuple_delimiter="<|#|>",
    language="English",
    # ...
)
```

## Chỉnh sửa Prompts

Để chỉnh sửa prompts:

1. Mở file `.md` tương ứng trong thư mục này
2. Chỉnh sửa nội dung (giữ nguyên các placeholder như `{entity_types}`, `{language}`, etc.)
3. Lưu file
4. Khởi động lại application để load prompt mới

**Lưu ý:** 
- Các placeholder trong dấu ngoặc nhọn `{}` sẽ được thay thế bằng giá trị thực tế khi runtime. Không xóa hoặc đổi tên các placeholder này trừ khi bạn cũng cập nhật code tương ứng.
- File format là Markdown (`.md`) để hỗ trợ syntax highlighting và format đẹp hơn trong editors.

## Lợi ích của việc tách prompts ra file riêng

1. **Dễ chỉnh sửa:** Không cần chạm vào Python code để thay đổi prompts
2. **Version control:** Dễ dàng track changes trong prompts
3. **Collaboration:** Người không biết code có thể chỉnh sửa prompts
4. **Testing:** Có thể test nhiều phiên bản prompts khác nhau
5. **Reusability:** Prompts có thể được sử dụng bởi các công cụ khác
6. **Maintainability:** Code Python gọn gàng hơn, dễ maintain hơn

## Backward Compatibility

Cấu trúc PROMPTS dictionary được giữ nguyên 100% như cũ, nên tất cả code hiện tại vẫn hoạt động bình thường mà không cần thay đổi gì.

