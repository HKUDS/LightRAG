# Prompt Customization Guide

LightRAG cho phép bạn tùy chỉnh tất cả các prompts được sử dụng trong hệ thống thông qua các file Markdown.

## 📁 Vị trí Prompts

Tất cả prompts được lưu trong thư mục:

```
lightrag/prompts/
├── entity_extraction_system_prompt.md
├── entity_extraction_user_prompt.md
├── entity_continue_extraction_user_prompt.md
├── entity_extraction_example_1.md
├── entity_extraction_example_2.md
├── entity_extraction_example_3.md
├── summarize_entity_descriptions.md
├── rag_response.md
├── naive_rag_response.md
├── keywords_extraction.md
├── keywords_extraction_example_1.md
├── keywords_extraction_example_2.md
├── keywords_extraction_example_3.md
├── kg_query_context.md
├── naive_query_context.md
├── fail_response.md
└── README.md
```

## 🔧 Cách Tùy Chỉnh

### Local Development

1. **Mở file prompt cần chỉnh sửa:**
   ```bash
   code lightrag/prompts/entity_extraction_system_prompt.md
   ```

2. **Chỉnh sửa nội dung** (giữ nguyên placeholders)

3. **Restart application:**
   ```bash
   # Nếu chạy trực tiếp
   # Ctrl+C và chạy lại
   
   # Nếu dùng lightrag-server
   pkill lightrag-server
   lightrag-server
   ```

### Docker Deployment

Với Docker, prompts được mount từ host vào container:

1. **Chỉnh sửa file trên host:**
   ```bash
   nano lightrag/prompts/entity_extraction_system_prompt.md
   ```

2. **Restart container:**
   ```bash
   docker-compose restart lightrag
   ```

**Lợi ích:** Không cần rebuild Docker image!

Chi tiết xem: [lightrag/prompts/DOCKER_USAGE.md](../lightrag/prompts/DOCKER_USAGE.md)

## 📝 Prompt Variables

Các prompts sử dụng placeholders được thay thế runtime:

### Entity Extraction Prompts

- `{entity_types}` - Danh sách các entity types
- `{tuple_delimiter}` - Delimiter giữa các fields (mặc định: `<|#|>`)
- `{completion_delimiter}` - Signal kết thúc (mặc định: `<|COMPLETE|>`)
- `{language}` - Ngôn ngữ output (English, Vietnamese, etc.)
- `{input_text}` - Text cần extract entities
- `{examples}` - Examples được insert từ example files

### RAG Response Prompts

- `{response_type}` - Kiểu response (paragraphs, bullet points, etc.)
- `{user_prompt}` - Additional instructions từ user
- `{context_data}` - Knowledge graph + document chunks
- `{entities_str}` - JSON entities
- `{relations_str}` - JSON relationships
- `{text_chunks_str}` - Document chunks
- `{reference_list_str}` - Reference documents

### Summary Prompts

- `{description_type}` - Entity hoặc Relation
- `{description_name}` - Tên của entity/relation
- `{description_list}` - JSON list các descriptions
- `{summary_length}` - Max tokens cho summary
- `{language}` - Output language

### Keyword Extraction Prompts

- `{query}` - User query
- `{examples}` - Examples từ example files

## 💡 Best Practices

### 1. Backup trước khi thay đổi

```bash
git checkout -b custom-prompts
# ... make changes ...
git commit -am "Customize entity extraction for medical domain"
```

### 2. Giữ nguyên placeholders

❌ **SAI:**
```
Entity types: organization, person, location
```

✅ **ĐÚNG:**
```
Entity types: {entity_types}
```

### 3. Test incremental changes

- Thay đổi một prompt tại một thời điểm
- Test thoroughly trước khi deploy production
- Monitor quality metrics

### 4. Document your changes

Thêm comment hoặc note trong prompt:

```markdown
---Role---
You are a Knowledge Graph Specialist...

<!-- Custom modification for medical domain - 2024-11-11 -->
<!-- Added specific instructions for medical entity types -->
```

### 5. Version control

```bash
# Tag phiên bản prompts
git tag -a prompts-v1.0 -m "Production prompts version 1.0"

# Rollback nếu cần
git checkout prompts-v1.0 -- lightrag/prompts/
```

## 🎯 Common Customization Scenarios

### Scenario 1: Thêm Entity Type mới

**File:** `entity_extraction_system_prompt.md`

```markdown
entity_type: Categorize the entity using one of the following types: 
{entity_types}. If none of the provided entity types apply, 
do not add new entity type and classify it as `Other`.
```

Thay đổi:
```markdown
entity_type: Categorize the entity using one of the following types: 
{entity_types}, MEDICAL_TERM, DRUG_NAME, DISEASE. 
If none apply, classify as `Other`.
```

### Scenario 2: Thay đổi Response Format

**File:** `rag_response.md`

Tìm section về References và customize format:

```markdown
4. References Section Format:
  - The References section should be under heading: `### References`
  - Reference list entries should adhere to the format: `* [n] Document Title`
```

### Scenario 3: Multi-language Support

**File:** `entity_extraction_system_prompt.md`

Thêm instructions cho ngôn ngữ cụ thể:

```markdown
7. Language & Proper Nouns:
  - The entire output must be written in `{language}`.
  - For Vietnamese: Use diacritics correctly and proper Vietnamese grammar.
  - For Chinese: Use simplified or traditional based on context.
```

### Scenario 4: Domain-specific Instructions

Thêm domain knowledge vào prompts:

```markdown
---Domain Context---

For financial documents:
- Identify financial metrics (revenue, profit, loss, etc.)
- Extract temporal information (quarters, fiscal years)
- Recognize financial entities (stocks, bonds, derivatives)
```

## 🔍 Testing Customized Prompts

### Unit Test

```python
from lightrag.prompt import PROMPTS

# Verify prompt loaded correctly
assert "{entity_types}" in PROMPTS["entity_extraction_system_prompt"]
assert len(PROMPTS["entity_extraction_examples"]) == 3

# Test formatting
formatted = PROMPTS["entity_extraction_system_prompt"].format(
    entity_types="person, organization",
    tuple_delimiter="<|#|>",
    language="English",
    examples="...",
    completion_delimiter="<|COMPLETE|>",
    input_text="Test text"
)
print(formatted)
```

### Integration Test

```python
from lightrag import LightRAG

# Initialize with custom prompts
rag = LightRAG(working_dir="./test_dir")

# Insert test data
rag.insert("Your test document here")

# Query and validate
result = rag.query("Test query", mode="hybrid")
print(result)
```

## 🚨 Troubleshooting

### Prompt không load

```python
# Debug prompt loading
from lightrag.prompt import _PROMPT_DIR
print(f"Prompt directory: {_PROMPT_DIR}")
print(f"Directory exists: {_PROMPT_DIR.exists()}")
print(f"Files: {list(_PROMPT_DIR.glob('*.md'))}")
```

### Syntax error trong prompt

- Check placeholders: `{variable_name}`
- Không dùng `{{` hoặc `}}`
- Đảm bảo file UTF-8 encoding

### Performance degradation

- So sánh với baseline metrics
- A/B test với prompts cũ
- Review prompt complexity

## 📚 Related Documentation

- [lightrag/prompts/README.md](../lightrag/prompts/README.md) - Prompt structure overview
- [lightrag/prompts/DOCKER_USAGE.md](../lightrag/prompts/DOCKER_USAGE.md) - Docker-specific usage
- [Algorithm.md](Algorithm.md) - Understanding how prompts are used in the pipeline

## 🤝 Contributing Custom Prompts

Nếu prompts của bạn improve quality đáng kể:

1. Fork repository
2. Create feature branch
3. Test thoroughly
4. Document changes
5. Submit PR with:
   - Before/after metrics
   - Use cases
   - Example outputs

## 📞 Support

Có câu hỏi về prompt customization? 

- Check [lightrag/prompts/README.md](../lightrag/prompts/README.md)
- Open issue on GitHub
- Discuss in community channels

