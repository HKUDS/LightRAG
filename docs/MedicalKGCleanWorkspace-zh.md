# 医学 KG 干净 Workspace 重建指南

本项目当前建议优先使用新 workspace 重建医学 KG；如果确认旧 workspace 不再作为验收依据，旧 workspace 可以删除，或先备份后删除。重建后的验收应以新 workspace 的抽取、归一化、层级补全和浏览视图为准。

## 推荐配置

```env
WORKSPACE=influenza_medical_v1
MEDICAL_KG_PROFILE=clinical_guideline_zh
ENTITY_EXTRACTION_USE_JSON=true
```

如果还切换了 embedding 模型，也应使用新 workspace，避免旧向量空间和新向量空间混用。

## 删除原则

删除或备份前，先停止正在运行的文档处理、扫描、上传和重建任务，确认没有后台 pipeline 正在写入。

只处理 LightRAG 配置的 `working_dir` 和 `input_dir` 下，对应目标 workspace 的数据。不要删除配置目录之外的路径，也不要清理这些目录的上级目录、共享父目录或无法确认归属的路径。

建议先记录当前 `.env` 或启动参数中的 `WORKSPACE`、`WORKING_DIR`、`INPUT_DIR`，再核对实际路径。若路径不确定，先备份或暂停操作，不要为了快速重建而扩大删除范围。

## 重建步骤

1. 停止正在运行的文档处理，等待已有任务退出。
2. 备份旧 workspace 数据，或在确认无保留价值后删除旧 workspace 数据。
3. 设置新 workspace 名称，并启用 `MEDICAL_KG_PROFILE=clinical_guideline_zh` 与 `ENTITY_EXTRACTION_USE_JSON=true`。
4. 启动服务后，重新导入原始医学文档，不要依赖旧 workspace 中已经抽取过的图谱结果。
5. 打开 WebUI 图谱，搜索 `流行性感冒`，并使用 `medical_view=true` 与 `medical_browse=true` 验证医学分组和层级浏览视图。

## 验收重点

- 病原体、症状和并发症能进入对应医学分组。
- 剂量、页码、表格碎片、阈值等值型内容不应成为主要实体节点。
- `流行性感冒` 的邻接关系应保留医学概念和证据描述，而不是旧 workspace 残留的孤立值节点。
- 重建完成后如仍看到旧抽取结果，优先检查是否复用了旧 workspace、旧缓存或旧 embedding 数据。
