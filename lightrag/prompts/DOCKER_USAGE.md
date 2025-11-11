# Using Prompts with Docker

## Overview

Thư mục prompts được mount từ host vào Docker container, cho phép bạn chỉnh sửa prompts mà **không cần rebuild Docker image**.

## Volume Mapping

Trong `docker-compose.yml`:

```yaml
volumes:
  - ./lightrag/prompts:/app/lightrag/prompts
```

**Host Path:** `./lightrag/prompts` (trên máy của bạn)  
**Container Path:** `/app/lightrag/prompts` (trong Docker container)

## Cách sử dụng

### 1. Chỉnh sửa Prompts

Chỉ cần edit file `.md` trong thư mục `lightrag/prompts/` trên host:

```bash
# Ví dụ: Chỉnh sửa entity extraction prompt
notepad lightrag/prompts/entity_extraction_system_prompt.md
# hoặc
code lightrag/prompts/entity_extraction_system_prompt.md
```

### 2. Áp dụng thay đổi

Sau khi chỉnh sửa, restart container để load prompts mới:

```bash
docker-compose restart lightrag
```

**Lưu ý:** Container sẽ tự động đọc file prompts mới khi khởi động lại.

### 3. Kiểm tra thay đổi

Prompts được load lại mỗi khi:
- Container restart
- Application restart
- Module `lightrag.prompt` được import lại

## Lợi ích

✅ **Không cần rebuild image:** Tiết kiệm thời gian và bandwidth  
✅ **Edit nhanh:** Thay đổi prompts ngay trên host  
✅ **Version control:** Track changes với git  
✅ **Rollback dễ dàng:** Git revert nếu cần  
✅ **Test A/B:** Dễ dàng test nhiều phiên bản prompts  

## Ví dụ Workflow

### Tùy chỉnh Entity Extraction Prompt

```bash
# 1. Mở file prompt
code lightrag/prompts/entity_extraction_system_prompt.md

# 2. Chỉnh sửa (ví dụ: thêm entity type mới)
# Thay đổi các instructions, format, examples...

# 3. Lưu file

# 4. Restart container
docker-compose restart lightrag

# 5. Test với API
curl -X POST http://localhost:9621/insert \
  -H "Content-Type: application/json" \
  -d '{"text": "Your test text here"}'
```

### Backup Prompts trước khi thay đổi

```bash
# Tạo backup
cp -r lightrag/prompts lightrag/prompts.backup

# Hoặc sử dụng git
git checkout -b custom-prompts
# ... make changes ...
git commit -am "Custom entity extraction prompts"
```

### Restore về default prompts

```bash
# Nếu đã backup
rm -rf lightrag/prompts
mv lightrag/prompts.backup lightrag/prompts

# Hoặc dùng git
git checkout main -- lightrag/prompts/
```

## Troubleshooting

### Prompt không được update sau khi chỉnh sửa

**Giải pháp:**
```bash
# Restart container
docker-compose restart lightrag

# Hoặc recreate container
docker-compose down
docker-compose up -d
```

### Permission issues (Linux/Mac)

Nếu gặp lỗi permission:

```bash
# Đảm bảo quyền đọc
chmod -R 644 lightrag/prompts/*.md
chmod 755 lightrag/prompts

# Hoặc chown nếu cần
sudo chown -R $USER:$USER lightrag/prompts
```

### File không tồn tại trong container

Check volume mapping:

```bash
# Xem volumes
docker-compose config

# Inspect container
docker exec lightrag ls -la /app/lightrag/prompts/

# Check file content
docker exec lightrag cat /app/lightrag/prompts/entity_extraction_system_prompt.md
```

## Best Practices

1. **Backup trước khi thay đổi:** Luôn backup hoặc commit vào git
2. **Test từng prompt:** Không thay đổi nhiều prompts cùng lúc
3. **Document changes:** Ghi chú lý do thay đổi trong commit message
4. **Monitor performance:** Theo dõi quality và performance sau khi thay đổi
5. **Keep placeholders intact:** Không xóa hoặc đổi tên `{variable_name}`

## Environment-specific Prompts

Nếu cần prompts khác nhau cho các môi trường:

```yaml
# docker-compose.dev.yml
volumes:
  - ./lightrag/prompts.dev:/app/lightrag/prompts

# docker-compose.prod.yml  
volumes:
  - ./lightrag/prompts.prod:/app/lightrag/prompts
```

Sau đó:
```bash
# Dev
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# Prod
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up
```

## Hot Reload (Future Enhancement)

Hiện tại cần restart container. Trong tương lai có thể implement:
- File watcher để auto-reload prompts
- API endpoint để reload prompts without restart
- Cache invalidation mechanism

## Support

Nếu gặp vấn đề với prompt customization, check:
1. File có tồn tại trên host không
2. Volume mount đúng trong docker-compose.yml
3. Container có quyền đọc file không
4. Syntax trong prompt file có đúng không (placeholders intact)

