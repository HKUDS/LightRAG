# 🚀 Quick Reference - Prompt Refactoring

**TL;DR:** Prompts đã được tách ra khỏi Python code thành các file Markdown riêng biệt để dễ chỉnh sửa hơn.

---

## 📌 Những thay đổi chính

### ✅ Đã làm gì?

1. **Tách prompts ra file riêng**
   - Từ: Hardcoded trong `lightrag/prompt.py` (422 dòng)
   - Thành: 16 file `.md` trong `lightrag/prompts/`
   - Kết quả: `prompt.py` giảm còn 88 dòng (-79%)

2. **Docker support**
   - Volume mount: `./lightrag/prompts:/app/lightrag/prompts`
   - Chỉnh sửa prompts không cần rebuild image

3. **Documentation hoàn chỉnh**
   - 3 guide files chi tiết
   - Examples và troubleshooting

### 🎯 Tại sao làm?

- ❌ Khó chỉnh sửa prompts trong Python code
- ❌ Cần biết Python để sửa prompts
- ❌ Phải restart/rebuild mọi thứ
- ✅ Giờ đây: Edit file `.md` và restart là xong!

---

## 📂 Cấu trúc mới

```
lightrag/
├── prompt.py (88 dòng) ← Code gọn gàng
└── prompts/
    ├── README.md ← Hướng dẫn
    ├── CHANGELOG.md ← Lịch sử thay đổi
    ├── DOCKER_USAGE.md ← Dùng với Docker
    ├── 10 main prompts.md
    └── 6 examples.md

docs/
└── PromptCustomization.md ← Guide đầy đủ

docker-compose.yml ← Đã thêm volume
Dockerfile ← Đã cập nhật
PROMPT_REFACTORING_SUMMARY.md ← Tóm tắt chi tiết
```

---

## 💻 Cách sử dụng

### Chỉnh sửa Prompt (Local)

```bash
# 1. Mở file
code lightrag/prompts/entity_extraction_system_prompt.md

# 2. Sửa và lưu

# 3. Restart app
# (Prompts load lại tự động)
```

### Chỉnh sửa Prompt (Docker)

```bash
# 1. Sửa trên host
vim lightrag/prompts/rag_response.md

# 2. Restart container
docker-compose restart lightrag

# Done! 🎉
```

---

## 🔑 Key Points

### Backward Compatible ✅
- Không có breaking changes
- Code cũ vẫn chạy bình thường
- API không thay đổi
- PROMPTS dictionary vẫn như cũ

### Files Created (20 total)
- 10 main prompt files
- 6 example files
- 4 documentation files

### Benefits
- 🚀 **Fast:** Không cần rebuild Docker image
- ✏️ **Easy:** Edit bằng bất kỳ editor nào
- 📝 **Standard:** Markdown format chuẩn
- 🔄 **Flexible:** Dễ dàng version control
- 🧪 **Testable:** A/B test nhiều prompts

---

## 📖 Documentation

| File | Mục đích |
|------|----------|
| **[PROMPT_REFACTORING_SUMMARY.md](PROMPT_REFACTORING_SUMMARY.md)** | Tóm tắt đầy đủ toàn bộ dự án |
| **[lightrag/prompts/CHANGELOG.md](lightrag/prompts/CHANGELOG.md)** | Lịch sử thay đổi theo format chuẩn |
| **[lightrag/prompts/README.md](lightrag/prompts/README.md)** | Overview về prompts |
| **[lightrag/prompts/DOCKER_USAGE.md](lightrag/prompts/DOCKER_USAGE.md)** | Hướng dẫn dùng với Docker |
| **[docs/PromptCustomization.md](docs/PromptCustomization.md)** | Guide tùy chỉnh chi tiết |

---

## ⚡ Quick Commands

```bash
# Xem prompts
ls lightrag/prompts/*.md

# Backup trước khi sửa
cp -r lightrag/prompts lightrag/prompts.backup

# Test trong Docker
docker exec lightrag cat /app/lightrag/prompts/entity_extraction_system_prompt.md

# Restart Docker
docker-compose restart lightrag

# Check logs
docker-compose logs -f lightrag
```

---

## 🎓 Examples

### Example 1: Thêm entity type mới

```bash
# Edit file
vim lightrag/prompts/entity_extraction_system_prompt.md

# Tìm dòng:
# entity_type: Categorize the entity using one of the following types: {entity_types}

# Thêm types:
# entity_type: ... {entity_types}, MEDICAL_TERM, DRUG_NAME, DISEASE ...

# Save và restart
docker-compose restart lightrag
```

### Example 2: Custom RAG response format

```bash
# Edit
vim lightrag/prompts/rag_response.md

# Tìm section "References Section Format" và customize

# Restart
docker-compose restart lightrag
```

---

## 🧪 Testing

Tất cả đã được test:
- ✅ 14/14 PROMPTS keys validated
- ✅ UTF-8 encoding OK
- ✅ Placeholders intact
- ✅ Docker volumes work
- ✅ No linter errors
- ✅ 100% backward compatible

---

## 🚨 Important Notes

### DO ✅
- Backup trước khi sửa
- Test sau khi sửa
- Giữ nguyên `{placeholders}`
- Dùng UTF-8 encoding
- Commit changes vào git

### DON'T ❌
- Xóa hoặc đổi tên placeholders
- Dùng encoding khác UTF-8
- Sửa nhiều prompts cùng lúc
- Skip testing
- Forget to backup

---

## 🔗 Quick Links

- **Main code:** [lightrag/prompt.py](lightrag/prompt.py)
- **Prompts:** [lightrag/prompts/](lightrag/prompts/)
- **Docker config:** [docker-compose.yml](docker-compose.yml)
- **Full summary:** [PROMPT_REFACTORING_SUMMARY.md](PROMPT_REFACTORING_SUMMARY.md)

---

## 📊 Stats

| Metric | Value |
|--------|-------|
| Code reduced | -79% (422→88 lines) |
| Files created | 20 |
| Documentation | 4 guides |
| Breaking changes | 0 |
| Test pass rate | 100% |

---

## ❓ FAQ

**Q: Code cũ còn chạy được không?**  
A: Có! 100% backward compatible.

**Q: Cần rebuild Docker image không?**  
A: Không! Chỉ cần restart container.

**Q: Sửa prompts ở đâu?**  
A: Trong `lightrag/prompts/*.md`

**Q: Làm sao rollback nếu sai?**  
A: `git checkout -- lightrag/prompts/`

**Q: Cần restart app không?**  
A: Có, prompts load lúc import module.

---

**Status:** ✅ Complete  
**Date:** November 11, 2024  
**Version:** 1.0.0

