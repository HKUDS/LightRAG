# Docling 内容提取引擎 Sidecar 改造方案

## 一、目标

当前 `parse_docling` 只从 Docling Serve 的 `document.md_content` 提取 Markdown 文本，并以 `raw` 格式写入 `full_docs`。改造后，Docling 应与 MinerU 一样：

- 调用 Docling Serve 获取完整结构化 JSON、Markdown 和引用图片资源；
- 将 Docling 原始产物落盘到可缓存的 raw bundle；
- 通过 `DoclingAdapter` 将 DoclingDocument JSON **忠实**归一化为 LightRAG IR：只转换 Docling 已明确表达的结构，不做版面识别纠错、相邻文本推断、图表编号补全或续表 caption 继承；
- 复用 `lightrag.sidecar.write_sidecar` 写出 `*.blocks.jsonl`、`*.tables.json`、`*.drawings.json`、`*.equations.json` 和 `*.blocks.assets/`；
- `full_docs` 最终保存 `parse_format=lightrag`、`lightrag_document_path` 和纯文本 `content`，供后续多模态分析、P 分块、实体抽取和删除清理使用。

## 二、已确认事实

### 2.1 Docling Serve API

本地 OpenAPI：`http://l4ai:5001/openapi.json`，服务版本为 `Docling Serve 1.18.0`。

异步转换入口：

```http
POST /v1/convert/file/async
Content-Type: multipart/form-data
```

后续流程：

```http
GET /v1/status/poll/{task_id}?wait=<seconds>
GET /v1/result/{task_id}
```

关键 multipart 字段分为两组：客户端固定常量（不暴露 env，写死在 `DoclingRawClient`）和用户可调 env。

**固定常量**（修改需走代码变更）：

| 字段 | 固定值 | 理由 |
| --- | --- | --- |
| `files` | 原始文件 | 表单文件字段，固定为 `files` |
| `pipeline` | `standard` | 与本项目其它解析路径假设一致；vlm pipeline 走另一条改造线 |
| `target_type` | `zip` | sidecar 流水线必须拿到完整 bundle；让 `/v1/result/{task_id}` 返回 `application/zip` |
| `to_formats` | `["json", "md"]` | JSON 用于机器归一化，MD 用于人工检查；两者缺一不可 |
| `image_export_mode` | `referenced` | 图片以引用方式写入 zip 的 `artifacts/`；base64 内嵌无法落盘到 sidecar assets |

**可调 env**（运维通过 `.env` 控制）：

| 字段 | env / 默认 | 说明 |
| --- | --- | --- |
| `do_ocr` | `DOCLING_DO_OCR=true` | 是否启用 OCR |
| `force_ocr` | `DOCLING_FORCE_OCR=true` | 是否强制全文 OCR；false 时仅图片走 OCR，其余按常规提取；true 时所有内容都走 OCR，有助于提高内容识别质量 |
| `ocr_engine` | `DOCLING_OCR_ENGINE=auto` | OpenAPI 已标记 DEPRECATED，但与 `ocr_preset` 同时发送 |
| `ocr_preset` | `DOCLING_OCR_PRESET=auto` | OCR 引擎 preset（推荐取代 `ocr_engine`） |
| `ocr_lang` | `DOCLING_OCR_LANG=`（空 / null） | 空时不发送字段；非空按 JSON 数组解析，按服务端实测兼容形式提交 |
| `do_formula_enrichment` | `DOCLING_DO_FORMULA_ENRICHMENT=false` | 公式 OCR/LaTeX 增强；默认 false 是保守值，部署侧确认 code-formula 模型就绪后可开启 |

实际 zip 产物布局示例：

```text
<task-result>.zip
├── artifacts/
│   ├── image_000000_<hash>.png
│   └── ...
├── <上传文件名>.json
└── <上传文件名>.md
```

对于PDF和Word文档中嵌入的图片，json文件的 `pictures[*].image.uri` 会指向 `artifacts/image_....png`。

### 2.2 Docling JSON 格式

Docling JSON 是 `DoclingDocument`，OpenAPI 对顶层只声明为 object，但实测结构如下：

```json
{
  "schema_name": "DoclingDocument",
  "version": "...",
  "name": "README",
  "origin": {
    "mimetype": "text/html",
    "binary_hash": 2214225749219141834,
    "filename": "README.md"
  },
  "furniture": {...},
  "body": {
    "self_ref": "#/body",
    "children": [{"$ref": "#/texts/0"}],
    "content_layer": "body",
    "name": "_root_",
    "label": "unspecified"
  },
  "groups": [...],
  "texts": [...],
  "pictures": [...],
  "tables": [...],
  "key_value_items": [],
  "form_items": [],
  "pages": {
    "1": {
      "size": {"width": 2828.0, "height": 1212.0},
      "image": {"uri": "data:image/png;base64,..."},
      "page_no": 1
    }
  }
}
```

内容对象通过 `self_ref` 和 `{"$ref": "#/texts/5"}` 互相引用。常见对象字段：

- `texts[*]`：`label` 可能为 `title`、`section_header`、`text`、`list_item`、`caption`、`footnote`、`formula`、`code`、`page_header`、`page_footer` 等；正文在 `text` / `orig`；标题层级只使用 Docling 给出的 `level`。若 PDF / OCR 路径下层级较粗或误标，adapter 不纠正，保持 Docling 输出语义。
- `tables[*]`：`label="table"`，结构在 `data.grid`、`data.table_cells`、`data.num_rows`、`data.num_cols`；表题/脚注只通过 `captions` / `footnotes` ref 或对象自身 `children` 中直接标为 `caption` / `footnote` 的 ref 获取。实测 Force OCR/PDF 会产生 `caption` label 和 table caption refs，但 `footnotes` refs 仍可能为空。
- `pictures[*]`：`label="picture"`，图片通常在 `image.uri`，也可能 `image=null` 或缺失 `uri`；标题/脚注只通过 `captions` / `footnotes` ref 或对象自身 `children` 中直接标为 `caption` / `footnote` 的 ref 获取；`children` 数组可能包含图内 OCR 文本（非 caption/footnote 的 children 首版不进入正文，仅保留审计信息，见 §5.4）。
- `prov[*]`：位置数组，包含 `page_no`、`bbox.{l,t,r,b,coord_origin}`、`charspan`。`page_no` 是 1-based；`coord_origin` 可能是 `TOPLEFT` 或 `BOTTOMLEFT`。实测：PDF 路径下顶级 prov（text / table / picture）几乎全为 `BOTTOMLEFT`；表 cell 内部 bbox 全为 `TOPLEFT`（混合坐标系，per-position override 主要为 markdown / docx / html / 混合输入服务）。
- `groups[*]`：用于 list、inline、section、key_value_area、form_area、table cell 内部结构；需要递归展开其 `children`。**判定结构语义统一使用 `label` 字段；`name` 仅作调试日志使用。**
- `content_layer`：所有顶级对象（texts / tables / pictures / groups）均含此 enum 字段：`{body, furniture, background}`。**`content_layer == "furniture" / "background"` 一律跳过，不进入 block，不写 position，不参与去重；即使在 `body.children` 中显式列出也跳过。**

> * 以下是一个真实的Docling JSON文件，如需了解格式细节可以直接查看: converted_docs-ForceOCR/m012-manual.json （由于文件体积较大，建议通过JSON工具查看）
> * 如果需要测试 Docling API的实际响应，可以通过以下URL:  http://l4ai:5001/v1/convert/file/async

### 2.3 MinerU 可复用逻辑

`MinerUAdapter` 已经实现了需要复用的核心行为：

- 标题检测：`title` / `section_header` 或 `text_level > 0` 作为 heading。
- 标题栈：`heading_stack[level - 1]` 保存当前层级，`parent_headings` 是当前 heading 之前的祖先链。
- 按标题合并 block：每个 heading 打开一个 block；其下的普通文本、列表、代码、表格、图片、公式都追加到当前 block。
- 相邻标题处理：如果当前 block 只有 heading、还没有正文，且下一个 heading 更深，则把更深 heading 作为正文行合并到当前 block，而不是立即切出空 block。
- heading 行保留在 `content_template` 中，使用 Markdown 前缀，例如 `# Heading`、`## Subheading`。
- 首个 heading 前的内容进入合成块 `Preface/Uncategorized`，`level=0`。
- 位置处理：有 bbox 的 item 生成细粒度 `IRPosition(type="bbox", anchor=<page>, range=[...])`；无 bbox 但有页码的 item 按页聚合为 anchor-only position；同一 block 的 positions 汇总到 block 级别。

Docling Adapter 应直接按这个行为实现，避免 Docling 和 MinerU 在 P 分块、溯源和多模态上下文中表现不一致。

## 三、目标目录布局

以 `demo.[docling-R].pdf` 为例，解析后保留两类目录：

```text
inputs/<workspace>/__parsed__/
├── demo.[docling-R].pdf
├── demo.pdf.docling_raw/
│   ├── _manifest.json
│   ├── demo.json
│   ├── demo.md
│   └── artifacts/
│       └── image_000000_<hash>.png
└── demo.pdf.parsed/
    ├── demo.blocks.jsonl
    ├── demo.drawings.json
    ├── demo.tables.json
    ├── demo.equations.json
    └── demo.blocks.assets/
        └── image_000000_<hash>.png
```

`*.docling_raw/` 是 Docling 原始 bundle 缓存，便于复查 JSON/MD 和避免重复上传；`*.parsed/` 是 LightRAG Sidecar，供后续流水线使用。

## 四、模块改造

### 4.1 新增 `lightrag/docling_raw/`

建议按 MinerU 拆同构模块：

```text
lightrag/docling_raw/
├── __init__.py
├── cache.py
├── client.py
└── manifest.py
```

职责：

- `client.py::DoclingRawClient`
  - 读取 §六 列出的 6 个 `DOCLING_*` env；其余字段（`pipeline / target_type / to_formats / image_export_mode`）作为模块级常量硬编码；
  - 上传文件到 `/v1/convert/file/async`，multipart 字段同时包含固定常量与可调 env；`ocr_engine` 与 `ocr_preset` 都按各自 env 值发送（默认 `auto`）；`ocr_lang` 为空时跳过该字段；
  - 长轮询 `/v1/status/poll/{task_id}?wait=5`；按 §七 的状态枚举映射判定终态，命中 `failure / partial_success / skipped` 时**不再下载 result**，直接抛 `RuntimeError` 并附 `task_id` / `error_message` / 截断 payload；
  - 仅 `success` 时从 `/v1/result/{task_id}` 下载 zip；
  - safe extract 到 `raw_dir`，拒绝绝对路径和 `..` 路径；
  - 校验有且仅有一个 `.json` 和一个 `.md`；
  - 写 `_manifest.json`。
- `cache.py`
  - `raw_dir_for_parsed_dir(parsed_dir) -> <base>.docling_raw/`；
  - `is_bundle_valid(raw_dir, source_file)`；
  - `clear_dir_contents(raw_dir)`；
  - 复用 MinerU 的 size/hash 策略；
  - `options_signature` 必须覆盖：可调 env（`DOCLING_DO_OCR / DOCLING_FORCE_OCR / DOCLING_OCR_ENGINE / DOCLING_OCR_PRESET / DOCLING_OCR_LANG / DOCLING_DO_FORMULA_ENRICHMENT`）+ 固定常量（`pipeline / target_type / to_formats / image_export_mode`）。任一字段值变化即 cache miss；常量也写入是为防止未来值变更后老 bundle 被复用。
- `manifest.py`
  - engine 固定为 `docling`；
  - critical file 为主 JSON；
  - files 记录 MD、artifacts 和其它 zip 文件；
  - 记录 `endpoint_signature`、`engine_version`、`options_signature`、`task_id`。

缓存失效条件：

- 原始文件大小或 sha256 改变；
- `DOCLING_ENDPOINT` 改变；
- `DOCLING_ENGINE_VERSION` 设置且与 manifest 不一致；
- OCR / image / formula / output 参数签名改变；
- 主 JSON 缺失、大小/hash 不一致；
- artifacts 文件缺失或大小不一致；
- `LIGHTRAG_FORCE_REPARSE_DOCLING=true`。

### 4.2 新增 `lightrag/parser_adapters/docling.py`

入口：

```python
class DoclingAdapter:
    def normalize_from_workdir(self, raw_dir: Path, *, document_name: str) -> IRDoc:
        ...
```

核心处理步骤：

1. 选择 raw bundle 中的主 JSON：
   - 优先 `<stem>.json`；
   - 否则选择 raw dir 根目录下唯一 `.json`；
   - 如果没有或多个候选无法判定，抛出明确错误。
2. 建立 ref 索引：
   - `#/body` → `body`；
   - `#/texts/<i>`、`#/tables/<i>`、`#/pictures/<i>`、`#/groups/<i>`；
   - `key_value_items`、`form_items`：**首版完全跳过** — 不进入 body block、不写 sidecar、不写入 `consumed_refs`；只在 `IRDoc.split_option.docling_extras` 里记录 `{"key_value_items": <count>, "form_items": <count>}` 以备审计；将来如需支持再单开扩展提案。
3. **预计算 `consumed_refs`**（去重集合，遍历前一次性准备）：
   - 收集所有 `tables[*].captions / footnotes` 中的 ref；
   - 收集所有 `pictures[*].captions / footnotes` 中的 ref；
   - 对 `tables[*].children` / `pictures[*].children`，只收集直接指向 `label="caption"` 或 `label="footnote"` 的 text ref；
   - 对 `pictures[*].children` 中其它 OCR / 图内文本 ref，加入 `consumed_refs` 并在 drawing extras 中记录审计信息，不进入正文；
   - 不收集相邻 sibling、`key_value_area` 内注释、编号文本或续表候选；这些不是 Docling 对象显式关联，必须按普通阅读流处理；
   - 这些 ref 仅通过 `IRTable.caption / IRDrawing.caption / footnotes` 字段或 drawing extras 渲染，不再作为独立 IRBlock。
4. **按阅读顺序遍历**：
   - 从 `body.children` 开始按顺序处理；
   - 对每个 ref，若 `content_layer == "furniture"` 或 `"background"` 直接跳过（防御性二次检查，因为 §4.2 step 4 已建立 ref 索引但未过滤）；
   - 若 ref ∈ `consumed_refs` 直接跳过；
   - 对 `groups` 递归展开其 `children`（按 `label` 分派，见 §5 后文）；
   - 对 `texts / tables / pictures` 生成 block 内容；
   - 维护 `visited_refs` 防止环路（理论上 Docling 树是 DAG，但防御性使用）。
5. 输出 `IRDoc`，交给 `write_sidecar`。

### 4.3 修改 `parse_docling`

`parse_docling` 不再调用 `_call_protocol_parse_service` 抽 `document.md_content`，而是改为：

```python
parsed_dir = parsed_artifact_dir_for_source(...)
raw_dir = docling_raw.raw_dir_for_parsed_dir(parsed_dir)

if not force_reparse and is_bundle_valid(raw_dir, source_file_path):
    use cache
else:
    clear raw_dir
    await DoclingRawClient().download_into(raw_dir, source_file_path)

ir = DoclingAdapter().normalize_from_workdir(raw_dir, document_name=document_name)
parsed_data = write_sidecar(ir, parsed_dir=parsed_dir, doc_id=doc_id, engine="docling")
persist full_docs as FULL_DOCS_FORMAT_LIGHTRAG
archive source
return parsed_data with blocks_path
```

**`_call_protocol_parse_service` 处理策略**：**彻底删除**。已用 `rg "_call_protocol_parse_service"` 全量核对：

- 生产代码唯一调用点是 `pipeline.py:2605`（`parse_docling`），本次重构即移除；
- `lightrag/mineru_raw/client.py:117-122` 的 docstring 亲口承认"故意复制 ~70 行 upload/poll choreography 而不复用该 helper，因为它不暴露 `result_url` + `Content-Type`"——证明该 helper 已不够通用；
- `lightrag/mineru_raw/client.py:225` 一行旁注 `(mirrors _call_protocol_parse_service)`；
- 测试侧：2 个 Docling parse 测试（`tests/test_pipeline_release_closure.py:2897, 2928`）使用 monkeypatch，在 §8.4 中本就要重写；3 个直测（`3014 / 3101 / 3196`）是"为这个 helper 本身存在的通用协议工具测试"，目标删除后即为死代码。

**统一清理清单**（与本节 patch 一并提交）：

1. 删除 `lightrag/pipeline.py:2644-2761` 的 `_call_protocol_parse_service` 方法及其辅助；
2. 删除 `tests/test_pipeline_release_closure.py` 的 3 个直测（lines ~3014 / 3101 / 3196 含周边 test 方法）；
3. 删除 2 个旧 Docling monkeypatch 测试（lines ~2897 / 2928）— 由 §8.3 的新 `test_parse_docling_sidecar.py` 取代；
4. 改写 `lightrag/mineru_raw/client.py:117-122` 的 docstring，去掉对 `:meth:_PipelineMixin._call_protocol_parse_service` 的悬挂引用，仅说明"为暴露 `result_url` + `Content-Type` 而独立实现 upload/poll"；
5. 删除 `lightrag/mineru_raw/client.py:225` 的 `(mirrors _call_protocol_parse_service)` 旁注；
6. 删除 `tests/test_pipeline_release_closure.py:2488` 测试 docstring 中提及 "the legacy `_call_protocol_parse_service` helper" 的一句；
7. `env.example` 中 §六"遗留兼容"列举的 10 个废弃 env（`DOCLING_CONTENT_FIELD / DOCLING_POLL_* / DOCLING_ID_FIELD / DOCLING_STATUS_FIELD / DOCLING_RESULT_URL_FIELD / DOCLING_RESULT_ENDPOINT / DOCLING_FILE_FIELD / DOCLING_SUCCESS_VALUES / DOCLING_FAILED_VALUES / DOCLING_POLL_INTERVAL_SECONDS / DOCLING_MAX_POLLS`）一并删除 — 它们的唯一消费者就是被删的 helper。

理由（YAGNI）：项目 `AGENTS.md` 明确反对为假设的未来保留抽象；MinerU 与 Docling 现均有专用 raw client，没有第三个 protocol-based parser 在 roadmap 上；保留 helper 只会持续引发"哪个是现代路径"的认知负担。

## 五、Docling → IR 映射规则

### 5.1 标题与 block 合并

#### 5.1.1 heading 检测（label 维度）

Docling 通常能够识别 `title` / `section_header` 并给出 `level`。Adapter 忠实使用 Docling 输出的 `label` 与 `level`：如果 PDF / OCR 路径下所有 `section_header.level` 都是 1，sidecar 也保留该结果，不根据文本编号做层级纠正。

| Docling item | IR heading 行为 |
| --- | --- |
| `label="title"` | 首个标记为 `doc_title` 候选，sidecar heading level=1 |
| `label="section_header"` | 标记为 heading 候选；sidecar heading level 为 Docling item level + 1；缺失 level 时回退到 2 |
| `label="caption"` | 不打开 heading；若该 ref 已被 table/picture 的 `captions` 或直系 caption child 消费则跳过，否则作为普通文本追加 |
| `label="footnote"` | 不打开 heading；若该 ref 已被 table/picture 的 `footnotes` 或直系 footnote child 消费则跳过，否则作为普通文本追加（保留 `注:` / `注1:` 等原前缀） |
| 其它 `text / list_item / formula / code` | 不打开 heading，追加到当前 block |

#### 5.1.2 heading 栈与 parent_headings

完全复用 MinerU：

- 维护 `heading_stack: list[str]`，索引 `i` 对应 level `i+1`；
- 遇到新 heading（按 §5.1.2 推导出的 level）：
  - 截断 `heading_stack` 到 `level - 1`；
  - 把当前 heading 文本 append 到 `heading_stack[level-1]`；
- **当前 block 的 `heading`** = 该 block 打开时的 heading 文本；
- **当前 block 的 `parent_headings`** = `heading_stack[:level-1]` 的拷贝（祖先链，不含自己）；
- 首个 heading 之前的内容进入合成块：`heading=""`、`level=0`、`parent_headings=[]`、`session_type="body"`。

#### 5.1.3 content_template 的 Markdown 前缀

被识别为 heading 的段落写入 `IRBlock.content_template` 时，**必须**带 Markdown `#` 前缀：

- 前缀 = `"#" * level + " "`；
- level=1 → 首行 `# 1 产品用途和功能`
- level=2 → 首行 `## 2.1 电气性能指标`
- level=3 → 首行 `### 2.4.5 温度冲击（随系统进行）`
- heading 行**永远**作为该 block content 的**第一行**，保证按顺序拼接所有 content 行可复原全文；
- writer 不会再次添加前缀，前缀由 adapter 唯一负责。

#### 5.1.4 block 合并规则（与 MinerU 同构）

- 普通文本、列表、代码、表格、图片、公式追加到当前 block；
- **相邻 heading 合并**：若当前 block 仅含 heading 行、尚无正文，且下一个 heading 的 level 更深，则把下一个 heading 作为正文行（带其推导出的 `#` 前缀）追加到当前 block，不立即切出空 block；
- block flush 时丢弃完全空（无文本、无模态）的块；
- 首个 heading 前内容进入合成块（`heading=""`），与 MinerU 行为一致。

#### 5.1.5 session_type

- 首版统一写 `"body"`；
- 不识别 `preface / TOC / references / appendix`，与 MinerU 当前行为对齐，留作后续扩展。

### 5.2 文本与列表

| Docling label | IR 内容 |
| --- | --- |
| `text` | `item.text`，为空时递归渲染 `children` |
| `list_item` | 优先 `item.text`；为空时渲染 children；有 `marker` 时保留 marker（拼成 `<marker> <text>`，如 `a) 外廓尺寸...`） |
| `code` | 作为普通代码文本追加，暂不新增代码 sidecar |
| `caption` | 如果该 ref ∈ `consumed_refs`（即被某 table/picture 的 `captions` 或直系 caption child 消费）则跳过；否则作为普通文本追加 |
| `footnote` | 如果该 ref ∈ `consumed_refs`（即被某 table/picture 的 `footnotes` 或直系 footnote child 消费）则跳过；否则作为普通文本追加（保留原前缀 `注:` / `注1:` 等） |

**忠实映射边界**：不根据文本内容（例如 `^图\d+`、`^表\d+`、`^注\d*[:：]`）重新归类，不根据相邻 sibling 把普通文本绑定到前一个/后一个表格或图片，不补全编号，不继承续表 caption。Docling 未显式关联的 `caption` / `footnote` / `text` 都按阅读流保留。

**furniture / background 过滤**：`page_header / page_footer` 等版面装饰类内容的判定**不依赖 label**，统一以 `item.content_layer == "furniture"`（或 `"background"`）作为唯一判据 — 任何 `content_layer != "body"` 的 ref 一律跳过，不进入 block、不写 position、不写 consumed_refs。这与 §2.2 的整体过滤策略对齐。

**group 展开（按 `group.label` 分派）**：

- `label == "list"`：children 顺序追加，每条 `list_item` 占一行；
- `label == "inline"`：children 文本以空格连接为同一行；轻量清理标点前后多余空格；`formatting` / `hyperlink` 暂不改变正文（可放入 adapter 内部调试 extras）；
- `label == "section"`：children 自然展开，正常触发 heading 栈；
- `label == "form_area"`：children 顺序追加（实测为首页登记栏多段文本，无结构化语义）；
- `label == "key_value_area"`：children 顺序追加；KV 结构化抽取留待后续；
- 其它未知 label：默认按"顺序展开 children"，并 `logger.warning` 记录原 label，避免静默吞掉。

### 5.3 表格

Docling `tables[*]` → `IRTable`：

**`rows: list[list[str]]` 构造**（首选路径）：

- 优先以 `data.grid` 构造：`rows[i][j] = grid[i][j].text`；
- 当 cell 跨行/跨列时，Docling 在每个被占据格子放**同一 cell 对象**，因此 `rows` 中相同文本会自然冗余 — 这是预期行为，与 LightRAG `tables.json` 的 `rows: list[list[str]]` 纯字符串 2D 数组形态兼容；
- 若 `data.grid` 为空数组而 `data.table_cells` 非空：按 `[num_rows][num_cols]` 初始化空字符串矩阵，遍历 `table_cells`，按 `start_row_offset_idx / start_col_offset_idx` 写入；跨行/跨列 cell 在被占据范围内重复填同一文本。

**其它字段映射**：

- `data.num_rows` / `data.num_cols` 写入 `IRTable.num_rows` / `num_cols`（writer 据此生成 `dimension`）；
- `table_header`：抽取 `column_header == True` 且 `start_row_offset_idx == 0` 的连续顶部行（双条件避免拼错位）；
- `captions` refs 解析为 `IRTable.caption`（多条拼接为单字符串，分隔符 `" / "`）；
- 若 `captions` 为空，仅检查该 table 对象自身 `children` 中直接引用的 `label="caption"` text，并按同样规则写入 `IRTable.caption`；不扫描相邻 body/group sibling；
- `footnotes` refs 解析为 `IRTable.footnotes: list[str]`；
- 若 `footnotes` 为空，仅检查该 table 对象自身 `children` 中直接引用的 `label="footnote"` text，并按同样规则写入 `IRTable.footnotes`；不把 body 或 `key_value_area` 中的 `注:` 文本归入表格；
- Docling item 的 `self_ref`（形如 `#/tables/2`）透传到 `IRTable.self_ref`，writer 写入 `tables.json` item 顶层 `self_ref` 字段（spec §五），便于溯源回查 `.docling_raw/<doc>.json`；
- **`IRTable.extras`**：
  - `extras.parent = tables[k].parent`；
  - `extras.children_refs = tables[k].children`；
  - `extras.references = tables[k].references`；
  - `extras.annotations = tables[k].annotations`；
  - `extras.cells`：按 `[i, j]` 顺序保存每个 cell 的 `row_span / col_span / row_header / row_section / fillable / start_*_offset_idx / end_*_offset_idx / bbox`，便于将来需要 HTML 渲染时回查；不污染 `tables.json` 顶层 schema；
- block 内容追加 `{{TBL:k}}`，writer 渲染为 `<table id="tb-..." format="json">...</table>`。

**不做续表/误标纠正**：如果 Docling 没有把后续分页 table 关联到 caption，`IRTable.caption` 保持空；如果 Docling 把 `表16 与调节器内部接口` 标为 `section_header` 而不是 table caption，adapter 保持 heading 语义，不降级为 caption。

### 5.4 图片

Docling `pictures[*]` → `IRDrawing`：

**`image.uri` 处理**（四种形态）：

- `artifacts/...`（`image_export_mode=referenced` 的常规形态）：
  - 作为 `AssetSpec.ref`；
  - `AssetSpec.source = raw_dir / image.uri`；
  - `suggested_name = Path(uri).name`；
  - writer 复制到 `<base>.blocks.assets/`。
- `data:image/...;base64,...`（罕见，base64 内嵌）：
  - 解码 bytes 作为 `AssetSpec.source`；
  - 根据 `image.mimetype` 推断扩展名。
- 外部 URL：
  - 不下载；
  - 使用 `IRDrawing.path_override` 保留 URL；
  - `drawings.json.path` 写 URL，后续 VLM 会因本地文件不可用而跳过或按现有逻辑处理。
- `image` / `image.uri` 缺失：
  - 跳过该 Docling `pictures[*]` 对象，不生成 `IRDrawing`；
  - 不创建 `AssetSpec`，不在 block 内容中追加 `{{IMG:k}}`；
  - 这样 `drawings.json.path` 对已输出 drawing 始终指向可用本地 asset 或外部 URL。

**字段映射**：

- `fmt` 推断：**优先用 `image.mimetype`**（如 `image/png` → `png`，`image/jpeg` → `jpg`），仅当 mimetype 缺失时回退到 `Path(uri).suffix.lstrip('.')`；
- `captions` refs 解析为 `IRDrawing.caption`（多条拼接为单字符串，分隔符 `" / "`）；
- 若 `captions` 为空，仅检查该 picture 对象自身 `children` 中直接引用的 `label="caption"` text，并按同样规则写入 `IRDrawing.caption`；不扫描相邻 body/group sibling；
- `footnotes` refs 解析为 `IRDrawing.footnotes: list[str]`；
- 若 `footnotes` 为空，仅检查该 picture 对象自身 `children` 中直接引用的 `label="footnote"` text，并按同样规则写入 `IRDrawing.footnotes`；
- `src` 字段（spec §四）：保留给 Docling item 自身的 `src` 字段（多数情况下 Docling 不提供，写空字符串）；
- `self_ref`（形如 `#/pictures/3`）透传到 `IRDrawing.self_ref`，writer 写入 `drawings.json` item 顶层 `self_ref` 字段（spec §四）。

**`IRDrawing.extras` 字段保留**（全部进 extras 不污染顶层 schema）：

- `extras.intrinsic_size = [image.size.width, image.size.height]`（VLM 下游可据此快速决定是否跳过过小图）；
- `extras.dpi = image.dpi`；
- `extras.mimetype = image.mimetype`；
- `extras.parent = pictures[k].parent`；
- `extras.children_refs = pictures[k].children`；
- `extras.ocr_child_count = 非 caption/footnote child ref 数量`；
- `extras.annotations = pictures[k].annotations`（Docling 1.10+ VLM annotation 在此处；fixture 中为空，未来非空时直接透传）；
- `extras.references = pictures[k].references`（与 captions/footnotes 并列的引用回指字段；fixture 中为空但保留）；
- block 内容追加 `{{IMG:k}}`，writer 渲染为 `<drawing id="im-..." format="..." path="..." caption="..." src="..." />`。

**`pictures[*].children` 处理**（首版决策）：

- `pictures[k].children` 中 `label="caption"` / `label="footnote"` 的 ref 可以作为对象自身显式结构写入 `IRDrawing.caption` / `footnotes`；
- 其它 child ref 视为图内 OCR / 内部文本，不进入正文、不生成独立 block；只通过 `extras.children_refs` 和 `extras.ocr_child_count` 保留审计信息；
- 不根据图片前后的相邻文本推断 caption；Docling 未显式关联时 `IRDrawing.caption` 保持空。

**注意**：`pages[*].image` 是页面渲染图（PDF 路径下为 base64，单页约 100KB），**不作为 drawing 输出**；它只在 raw bundle 中保留，供未来可视化复查。

### 5.5 公式

Docling 公式表现为 `label="formula"` 的 text item（启用 `DOCLING_DO_FORMULA_ENRICHMENT=true` 后 `text` 字段为 LaTeX，`orig` 字段为乱码原文）。首版规则：

**LaTeX 字段优先级**：

- **优先使用 `item.text`**（启用 enrichment 后这里是 LaTeX）；
- `orig` 仅作 fallback：若 `text` 缺失，或 `text` 与 `orig` 完全一致（说明 enrichment 未运行），则视为 LaTeX 不可用，**退化为普通文本块**，按 `label="text"` 处理（不产生 IREquation，不写 equations.json）；
- 这意味着 `DOCLING_DO_FORMULA_ENRICHMENT=false`（首版默认）时，公式自动以普通文本进入正文，不报错。

**`$$` 包裹**（adapter 与 spec 的桥接）：

- Docling 的 `text` 字段是裸 LaTeX（如 `C = 2 * \frac { P * T } { ... }`），**不含** `$` / `$$` 包裹；
- adapter 在传给 `IREquation.latex` 前**显式补上 `$$ ... $$`**，与 MinerU adapter 行为对齐；
- writer 写 `equations.json.content` 时按现有逻辑剥离外层 `$$`（与 spec §六"无外层 $"约定一致）；
- writer 写 blocks.jsonl 的 `<equation>` 标签时，标签 body 保持 `$$ ... $$`（与 spec §三.3 一致）。

**block 与 inline 区分**：

- Docling 顶级 `texts[k] label="formula"` 通常视为 `is_block=True`（占独立 prov，独立 page 位置）；
- 若公式 item 出现在 inline group（`groups[k].label="inline"`）中，视为 `is_block=False`，仅进入 block 文本（`{{EQI:k}}`），不写 `equations.json`，`self_ref` 在该路径上无意义；
- block 公式：Docling text item 的 `self_ref`（形如 `#/texts/15`）透传到 `IREquation.self_ref`，writer 写入 `equations.json` item 顶层 `self_ref` 字段（spec §六）。

### 5.6 位置与页码

策略要点：**文档级 origin 固定为 `LEFTBOTTOM`**（Docling 默认坐标系），凡是 prov 自身 `coord_origin=TOPLEFT` 的，在 position 对象上写入 `origin="LEFTTOP"` 作为 per-position override。坐标值**原样落盘，绝不翻转**。该方案借助 spec §八 既有的两级 origin 模型（meta 全局默认 + position 级覆盖）解决混合坐标系问题，需要 IR 扩展 `IRPosition.origin` 可选字段才能落盘。

Docling `prov[]` → `IRPosition`：

```json
{
  "type": "bbox",
  "anchor": "1",
  "range": [l, t, r, b],
  "charspan": [start, end],
  "origin": "LEFTTOP"
}
```

字段规则：

- `anchor`：`str(prov.page_no)`，`page_no` 已是 1-based；
- `range`：`bbox` 坐标按 Docling 原值原样写入 `[l, t, r, b]`，**不做坐标翻转**；
- `charspan`：存在时原样写入；
- `origin`：
  - `prov.bbox.coord_origin == "BOTTOMLEFT"` → 不写入（继承 meta 行 `bbox_attributes.origin`）；
  - `prov.bbox.coord_origin == "TOPLEFT"` → 写入 `"LEFTTOP"`（per-position override）；
  - 其它未知值 → 写入对应映射并 `logger.warning` 记录原始值；
- bbox 缺失但 page_no 存在时，按 MinerU 逻辑聚合为 anchor-only position；
- 一个 block 合并多个 item 时，positions 按 item 顺序累积；anchor-only 页码去重并排序后放在 bbox 位置前。

`IRDoc.bbox_attributes`：

```json
{
  "origin": "LEFTBOTTOM"
}
```

- **不写 `max` 字段**。MinerU 的 `max=1000` 表示坐标归一化到 0–1000；Docling 是页面像素坐标，没有归一化。
- **不写 `page_sizes` 字段**。页面尺寸消费者直接读 `.docling_raw/<doc>.json` 的 `pages[N].size` 即可，避免 sidecar 与 raw bundle 之间冗余。
- 支持 `DOCLING_BBOX_ATTRIBUTES`（JSON 字符串）env 覆盖默认 origin，与 [`parser_adapters/mineru.py`](../lightrag/parser_adapters/mineru.py) 的 `MINERU_BBOX_ATTRIBUTES` 对齐。env 不影响 per-position override 逻辑：若设置 `origin=LEFTTOP`，所有 BOTTOMLEFT 的 prov 在 position level 携带 `origin=LEFTBOTTOM` 进行 override。

**实测验证**：PDF 路径下 354 个 texts + 24 个 tables + 4 个 pictures 的顶级 prov 全部为 `BOTTOMLEFT`，因此 per-position `origin` override 在 PDF 路径下几乎不触发；它主要为 markdown / docx / html 与混合输入服务。表 cell 内部 bbox 全部为 `TOPLEFT`，但**不会**以 `IRPosition` 形式写入 sidecar（只在 `IRTable.extras.cells` 里），所以不会因为坐标系不同产生混淆。

**优势对比旧方案**：

- 精度无损：坐标值不翻转，避免依赖浮点页面高度的 round-trip 误差；
- 混合 origin 不再产生 warning：spec 既有的两级模型本身就是为这种场景设计的；
- 实现简单：单遍扫描即可完成，无需统计多数派或翻转坐标。


## 六、环境变量

新增/调整 `.env` 和 `env.example`，**仅暴露 6 个可调字段**（其它管线相关字段已在 `DoclingRawClient` 中固化为常量，见 §2.1 / §4.1）：

```bash
# Docling raw bundle handling
# DOCLING_ENGINE_VERSION=
# LIGHTRAG_FORCE_REPARSE_DOCLING=false

# Docling OCR / formula enrichment
DOCLING_DO_OCR=true
DOCLING_FORCE_OCR=true
DOCLING_OCR_ENGINE=auto
DOCLING_OCR_PRESET=auto
DOCLING_OCR_LANG=
DOCLING_DO_FORMULA_ENRICHMENT=false

# Optional: override bbox_attributes (JSON) — default {"origin":"LEFTBOTTOM"}
# DOCLING_BBOX_ATTRIBUTES={"origin":"LEFTBOTTOM"}
```

实现注意：

- `DOCLING_DO_OCR` / `DOCLING_FORCE_OCR` / `DOCLING_DO_FORMULA_ENRICHMENT` 按布尔解析（与 MinerU 的 `_env_bool` 一致）。
- `DOCLING_OCR_ENGINE` 与 `DOCLING_OCR_PRESET` 都默认 `auto`，client 同时发送两个 multipart 字段；OpenAPI 已将 `ocr_engine` 标记 DEPRECATED，但按反馈保留以兼容现有部署。
- `DOCLING_OCR_LANG` 为空（默认）时**不发送 `ocr_lang` 字段**，由 OCR 引擎走默认行为；非空时按 JSON 数组解析（例如 `["en","zh"]`），multipart 发送形式以服务端实测兼容方式为准（JSON 字符串或重复 form 字段，见 client 实现注释）。
- **已删除字段**：`DOCLING_TARGET_TYPE` / `DOCLING_TO_FORMATS` / `DOCLING_IMAGE_EXPORT_MODE`（固化为常量）；`DOCLING_OCR_CUSTOM_CONFIG` / `DOCLING_CODE_FORMULA_PRESET` / `DOCLING_CODE_FORMULA_CUSTOM_CONFIG`（首版不开放透传，后续按需扩展）。
- **遗留兼容**：`DOCLING_CONTENT_FIELD` / `DOCLING_POLL_*` / `DOCLING_ID_FIELD` / `DOCLING_STATUS_FIELD` / `DOCLING_RESULT_URL_FIELD` / `DOCLING_RESULT_ENDPOINT` / `DOCLING_FILE_FIELD` / `DOCLING_SUCCESS_VALUES` / `DOCLING_FAILED_VALUES` / `DOCLING_POLL_INTERVAL_SECONDS` / `DOCLING_MAX_POLLS` 在新主路径上不再被读取，需要把他们从 env.example 文件中删除。由于当前版本尚未公开发布，不需要考虑向后兼容的问题。

## 七、错误处理与日志

- 上传失败：报 `Docling upload failed: <status> <body prefix>`。
- **轮询状态判定**（基于 OpenAPI `ConversionStatus` 枚举）：

  | `task_status` | 客户端行为 |
  | --- | --- |
  | `success` | 唯一成功状态，继续下载 `/v1/result/{task_id}` |
  | `failure` | 立即抛 `RuntimeError`；**不下载 result** |
  | `partial_success` | 立即抛 `RuntimeError`；按失败处理（避免下游拿到不完整 sidecar） |
  | `skipped` | 立即抛 `RuntimeError`（文件未被实际处理，等同失败） |
  | `pending` / `started` | 继续轮询 |
  | 其它未识别值 | 继续轮询并 `logger.warning` 记录原始值 |

  错误日志格式：

  ```
  Docling task {task_id} ended in {status}: {error_message or '<no error_message>'}; payload={truncated_payload}
  ```

  其中 `truncated_payload` 为 `json.dumps(poll_payload)[:400]`。
- 优先使用长轮询 `/v1/status/poll/{task_id}?wait=5` 降低请求数；`wait` 超时仍按非终态对待，下一轮继续。
- result 下载非 zip：如果 content-type 是 JSON，解析并报出 payload；否则提示 content-type 和 body 前 400 字符。
- zip 缺少 JSON/MD：报清晰错误，不写 manifest。
- JSON malformed：报主 JSON 路径。
- 图片资源缺失：不中断解析，写 warning；跳过对应 picture，避免生成空 `drawing path`。
- 解析结果没有任何 block：报错，避免把空 sidecar 写入成功状态。

## 八、测试计划

### 8.1 Adapter 单测

新增 `tests/parser_adapters/test_docling_adapter.py`：

- `test_docling_adapter_simple_heading_hierarchy`
  - title + section_header + text；
  - 校验 level、parent_headings、heading 行 Markdown 前缀。
- `test_docling_adapter_merges_payloads_under_heading`
  - 同一 heading 下 text + table + picture + formula；
  - 校验一个 block 内按顺序出现 `{{TBL}}` / `{{IMG}}` / `{{EQ}}`。
- `test_docling_adapter_inline_groups_and_empty_list_items`
  - list_item text 为空但 children 指向 inline group；
  - 校验渲染后列表文本不丢失。
- `test_docling_adapter_table_grid_and_header`
  - `data.grid`、`column_header`、caption/footnotes refs。
- `test_docling_adapter_picture_referenced_asset`
  - `image.uri=artifacts/a.png`；
  - 校验 `AssetSpec.source` 指向 raw dir 内文件。
- `test_docling_adapter_positions_and_bbox_attributes`
  - `prov.page_no`、`bbox.coord_origin=BOTTOMLEFT`；
  - 校验 `IRPosition(anchor="1", range=[...])`；
  - 校验 `IRDoc.bbox_attributes == {"origin": "LEFTBOTTOM"}`（**无 `max` 字段、无 `page_sizes` 字段**）；
  - 同时构造一条 `bbox.coord_origin=TOPLEFT` 的 prov（mock 模拟非 PDF 输入混合坐标系），校验该 position 的 `origin=="LEFTTOP"` per-position override 字段；其余 BOTTOMLEFT prov 不写 `origin` 字段。
- `test_docling_adapter_bbox_attributes_env_override`
  - 设置 `DOCLING_BBOX_ATTRIBUTES='{"origin":"LEFTTOP"}'`；
  - 校验 env 值覆盖默认；同时校验所有 BOTTOMLEFT prov 的 position 都携带 `origin="LEFTBOTTOM"` 作为 override。
- `test_docling_adapter_preserves_docling_heading_level`
  - 输入 3 个 `section_header level=1`：`1 产品用途和功能` / `2.1 电气性能指标` / `2.4.5 温度冲击`；
  - 校验 adapter 不根据文本编号重算层级，三者均按 Docling level 输出同级 heading；
  - 校验 `表16 与调节器内部接口` 若被 Docling 标为 `section_header`，保持 heading，不降级为 table caption。
- `test_docling_adapter_caption_refs_only`
  - 输入 `tables[0].captions=[#/texts/cap]`，`texts/cap.label="caption"`；
  - 校验 `IRTable.caption` 来自该 ref，且 caption ref 不进入正文；
  - 再输入相邻但未被 ref 关联的 `label="caption"` / `label="text"`，校验其按普通阅读流保留，不被归入 table。
- `test_docling_adapter_footnotes_refs_only`
  - 输入 `tables[0].footnotes=[#/texts/fn]`，校验 `IRTable.footnotes` 来自该 ref；
  - 再输入 body 或 `key_value_area` 中的 `注：...` 文本但不放入 table `footnotes`，校验不进入 `IRTable.footnotes`。
- `test_docling_adapter_furniture_skipped_by_content_layer`
  - 输入若干 `texts[]` 含 `content_layer="furniture"` 的 page_footer；
  - 校验 0 个 IRBlock 的 content_template 含这些 page_footer 文本，0 条相关 position。
- `test_docling_adapter_picture_children_dropped`
  - 输入 `pictures[0].children = [#/texts/A, #/texts/B]`，A/B 为图内 OCR 文本（不在 captions / footnotes 里）；
  - 校验 A/B 不在任何 IRBlock.content 里；
  - 校验 `IRDrawing.extras.children_refs` / `ocr_child_count` 记录这些 child refs；
  - 校验直系 caption child（若存在且 `label="caption"`）正常进入 `IRDrawing.caption`。
- `test_docling_adapter_picture_missing_image_skipped`
  - 输入 `pictures[0].image = null` 或无 `image.uri`；
  - 校验不生成 `IRDrawing`，也不创建空路径 drawing。

### 8.2 DoclingRawClient 单测

新增 `tests/mineru_raw/test_docling_raw_client.py`（与 mineru_raw 测试目录同构）：

- `test_docling_client_sends_fixed_constants`
  - 拦截 `httpx.AsyncClient.post`；
  - 校验 multipart 中包含 `pipeline=standard` / `target_type=zip` / `to_formats=[json,md]` / `image_export_mode=referenced`，且这些值不受 env 影响。
- `test_docling_client_partial_success_aborts`
  - mock 状态轮询返回 `partial_success`；
  - 校验抛 `RuntimeError`，错误信息含 `task_id` 与 `error_message`；
  - 校验**不**触发 `/v1/result/{task_id}` 请求。
- `test_docling_client_failure_aborts`
  - mock 状态轮询返回 `failure`；
  - 同上：抛错且不下载 result。
- `test_docling_client_ocr_lang_omitted_when_empty`
  - env `DOCLING_OCR_LANG` 为空；
  - 校验 multipart 中不含 `ocr_lang` 字段。

### 8.3 parse_docling 集成单测

新增 `tests/test_parse_docling_sidecar.py`，仿照 `tests/test_parse_mineru_sidecar.py`：

- stub `DoclingRawClient.download_into` 写入 fake zip 解压后的 raw bundle；
- 校验 `parse_docling` 写出 `.parsed/` sidecar、`.docling_raw/_manifest.json`、`full_docs.parse_format=lightrag`；
- 校验 cache hit 不重复下载；
- 校验 `LIGHTRAG_FORCE_REPARSE_DOCLING=true` 会重新下载；
- 校验 source hash 改变会 cache miss；
- 校验 `options_signature` 字段变化触发 cache miss（例如改 `DOCLING_OCR_LANG`）；
- 校验缺少 JSON/MD 会失败；
- Force OCR fixture：校验 `tables[3].caption == "表1 供氧抗荷调节器温度-湿度-高度要求"`、`pictures[1].caption == "图2 振动图谱"`；同时校验未被 `footnotes` refs 关联的 `注：...` 不进入 table footnotes；
- 非 Force OCR/docx fixture：若 `captions` / `footnotes` 为空，则 table/drawing caption/footnotes 为空，相邻图题/表题文本保留在正文阅读流中。

### 8.4 现有测试调整

- `test_parse_docling_uses_docling_serve_async_defaults` 改为覆盖 `DoclingRawClient` 默认 endpoint 推导和 multipart 字段，而不是期望 `document.md_content`。
- `test_parse_docling_empty_service_result_raises_without_fallback` 改为 raw bundle 缺关键文件失败。
- **删除** `_call_protocol_parse_service` 的 3 个直测（`tests/test_pipeline_release_closure.py:3014 / 3101 / 3196`）— 被测函数本身已删除，单测失去意义。
- **删除** 2 个用 monkeypatch 替换 `_call_protocol_parse_service` 的旧 Docling 测试（`tests/test_pipeline_release_closure.py:2897 / 2928`）— 由 §8.3 新建的 `test_parse_docling_sidecar.py` 取代。
- 更新 `tests/test_pipeline_release_closure.py:2488` 处的 MinerU 测试 docstring，去掉对该 helper 的历史性提及。

## 九、实施顺序

1. 新增 `docling_raw` client/cache/manifest，并完成离线单测。
2. 新增 `DoclingAdapter`，用小型 DoclingDocument fixture 做 golden 单测。
3. 修改 `parse_docling` 接入 raw client + adapter + `write_sidecar`。
4. 更新 `parser_adapters/__init__.py` 导出 `DoclingAdapter`。
5. 更新 `env.example`、`.env` 注释段和 `docs/FileProcessingConfiguration-zh.md` 的 Docling 配置说明。
6. 调整旧 Docling 测试并新增 parse_docling sidecar 集成测试。
7. 跑：

```bash
./scripts/test.sh tests/parser_adapters/test_docling_adapter.py
./scripts/test.sh tests/test_parse_docling_sidecar.py
./scripts/test.sh tests/test_pipeline_release_closure.py
ruff check lightrag tests
```

## 十、开放风险

- **Docling PDF 输入下 `level` 可能很粗** — 实测 fixture（DoclingDocument 1.10.0）50/50 `section_header level=1`、0 个 `title`。adapter 忠实保留 Docling level，不根据文本编号修正；若需要更准层级，应在 Docling 版面识别/转发侧解决。
- **PDF 顶层 prov 几乎全为 BOTTOMLEFT，per-position origin override 主要为非 PDF / 混合输入服务** — fixture 中 354 texts + 24 tables + 4 pictures 顶层 prov 全部 BOTTOMLEFT；TOPLEFT 仅出现在 table_cells 内部 bbox（不进入 IRPosition）。但 markdown 等输入下 TOPLEFT 可能出现在顶层，per-position override 机制不能省。
- Docling JSON schema 对 `TextItem / PictureItem / TableItem` 在 OpenAPI 中是 `additionalProperties`，字段稳定性应以真实 fixture + 适配器容错为准。
- `pages[*].image.uri` 是巨大 base64（PDF 路径下单页约 100KB，整篇文档可能 10MB+），不应复制到 sidecar assets；只处理 `pictures[*].image`。首版保留完整 raw JSON 以维持 hash 与审计语义；如未来需要减小 raw bundle 体积，可单开提案引入 `DOCLING_STRIP_PAGE_IMAGES` 开关，首版不实现。
- **Docling 可能把图表标题误标为 `section_header` 或普通 `text`** — adapter 不纠正；未通过 `captions` / `footnotes` refs 或对象直系 caption/footnote child 显式关联的文本会保留在正文阅读流中。原始 Docling JSON 保存在 `.docling_raw/`，供审计与上游识别规则迭代。
- **公式 enrichment 双轨**：`DOCLING_DO_FORMULA_ENRICHMENT=true` 时 `text` 为 LaTeX，false 时与 `orig` 一致；adapter 必须双路兼容，否则未启用 enrichment 部署直接 crash。
