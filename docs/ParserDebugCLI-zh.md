# Parser CLI Debuger使用指南

本工具用于本地调试 LightRAG 注册表中的任意内容解析引擎（内置 `native` / `legacy` / `mineru` / `docling`，以及通过 `lightrag.parsers` entry point 注册的第三方引擎，见 `docs/ThirdPartyParser-zh.md`），针对**单个文件**触发与 pipeline worker 相同的注册表派发路径（`get_parser(engine).parse(...)`），并把解析产物（sidecar 与 raw 缓存）输出到一个**扁平目录布局**——与生产入库目录相比，区别仅在于：

- **无 `__parsed__/` 中间层**：产物直接落在指定父目录下，便于查看；
- **源文件不会被归档**：源文件保留在原位置（生产路径会把源文件移到 `<INPUT_DIR>/__parsed__/`）；
- **raw 缓存只看目录是否存在**：`mineru` / `docling` 的 raw 目录非空即视为有效，跳过 `_manifest.json` 校验。

其余流程（IR 构建、sidecar 写入、对 `full_docs` 的同步逻辑）与生产入库完全一致，便于排查解析阶段问题。

## 命令格式

```bash
python -m lightrag.parser.cli <input_file> \
    --engine <engine> \
    [-o <sidecar_parent_dir>] \
    [--doc-id <doc-id>] \
    [--force-reparse] \
    [--preview N]
```

| 参数 | 说明 |
|---|---|
| `input_file` | 待解析的源文件路径（位置参数，必填）。文件必须实际存在。 |
| `--engine` | 必填，可选值来自注册表：内置 `native`（仅 `.docx`，本地解析）/ `legacy`（纯文本抽取，无 sidecar）/ `mineru`（PDF/办公文档，调 MinerU 服务）/ `docling`（PDF/办公文档，调 docling-serve），以及任何已注册的第三方引擎。 |
| `-o / --sidecar-parent-dir` | sidecar 与 raw 目录的父目录，默认 = 源文件所在目录。 |
| `--doc-id` | 自定义文档 ID，默认 `doc-<md5(源文件绝对路径)>`（同一文件多次跑结果稳定）。 |
| `--force-reparse` | 仅对外部服务引擎（`mineru` / `docling` 及继承 `ExternalParserBase` 的第三方引擎）生效：清空 raw 目录、强制重新下载与解析。默认行为是 raw 目录非空即复用。 |
| `--preview N` | 解析完成后打印前 N 个 block 的预览（headings + 内容片段），默认 5；`0` 关闭。对无 sidecar 的引擎（如 `legacy`），改为打印解析文本的前 400 字符。 |

## 输出目录布局

以输入 `./inputs/workspace/sample.pdf` + 默认 sidecar 父目录（即 `./inputs/workspace/`）为例：

```
./inputs/workspace/
├── sample.pdf                       # 原文件，不动
├── sample.pdf.parsed/               # ← sidecar 输出
│   ├── sample.blocks.jsonl          # JSONL：首行 meta，后续每行一个 block
│   ├── sample.blocks.assets/        # native 抽取的图片/媒体资产（若有）
│   ├── sample.tables.json           # 表格 sidecar（若 IR 含 tables）
│   ├── sample.drawings.json         # 图纸/图片 sidecar（若 IR 含 drawings）
│   └── sample.equations.json        # 公式 sidecar（若 IR 含 equations）
└── sample.pdf.<engine>_raw/         # ← mineru / docling 的 raw 缓存（native 无此目录）
    ├── _manifest.json               # 由引擎下载流程写入；CLI 缓存校验不读
    └── <bundle files>               # 引擎特定 raw 产物（content_list.json / *.json / 资产等）
```

`native` 引擎不产生 raw 目录（解析是本地的，无外部服务参与）。

## 典型用例

### A. 本地解析 `.docx`（零网络依赖）

```bash
python -m lightrag.parser.cli ./inputs/workspace/sample.docx --engine native
# 产出：./inputs/workspace/sample.docx.parsed/  （含 blocks.jsonl + assets）
```

### B. 用 MinerU 解析 PDF（首次会下载 raw）

```bash
# 第一次：下载 raw bundle + 生成 sidecar
python -m lightrag.parser.cli ./inputs/workspace/sample.pdf --engine mineru
# 第二次（无任何修改）：raw 目录非空 → 直接复用 → 仅重建 sidecar，速度快
python -m lightrag.parser.cli ./inputs/workspace/sample.pdf --engine mineru
# 日志会显示： [parse_mineru] raw cache hit doc_id=... raw_dir=.../sample.pdf.mineru_raw
```

### C. 用 Docling 解析 PDF + 复用已有 raw 目录

```bash
# 已有 ./inputs/workspace/sample.pdf.docling_raw/ （含 docling 产物的 JSON 等文件）
python -m lightrag.parser.cli ./inputs/workspace/sample.pdf --engine docling
# CLI 不查 manifest，只要 raw 目录非空就跳过 docling-serve 调用
```

> 注：这是旧 `python -m lightrag.parser.external.docling` 调试入口「从已有 raw 重建 sidecar」场景的等价替代——只需把 raw 目录放到约定位置（`<sidecar_parent>/<source>.docling_raw/`）即可触发缓存命中分支。

### D. 输出到自定义目录

```bash
python -m lightrag.parser.cli ./inputs/workspace/sample.docx \
    --engine native -o /tmp/debug_sidecar
# 产出：/tmp/debug_sidecar/sample.docx.parsed/
# 原文件 ./inputs/workspace/sample.docx 不会被移动
```

### E. 强制重新解析（清空 raw 后重新下载）

```bash
python -m lightrag.parser.cli ./inputs/workspace/sample.pdf \
    --engine docling --force-reparse
# raw 目录被清空 → 重新调 docling-serve 下载 → 重新生成 sidecar
```

## 环境变量

`mineru` / `docling` 引擎在 **缓存未命中**（首次解析或 `--force-reparse`）时会调用外部服务，所需环境变量与生产入库一致：

- **MinerU**：`MINERU_API_MODE`（`local` / `official`）、`MINERU_API_TOKEN`、`MINERU_LOCAL_ENDPOINT` 或 `MINERU_OFFICIAL_ENDPOINT`，可选 `MINERU_ENGINE_VERSION` / `MINERU_MODEL_VERSION` / `MINERU_POLL_INTERVAL_SECONDS` / `MINERU_MAX_POLLS`。
- **Docling**：`DOCLING_ENDPOINT`，可选 `DOCLING_ENGINE_VERSION` / `DOCLING_DO_OCR` / `DOCLING_FORCE_OCR` / `DOCLING_OCR_ENGINE` / `DOCLING_OCR_PRESET` / `DOCLING_OCR_LANG` / `DOCLING_DO_FORMULA_ENRICHMENT` / `DOCLING_POLL_INTERVAL_SECONDS` / `DOCLING_MAX_POLLS`。

详见 [FileProcessingConfiguration-zh.md](./FileProcessingConfiguration-zh.md)。

**缓存命中**时（raw 目录已存在且非空，且未传 `--force-reparse`）无需任何外部服务环境变量——可用于离线复现解析输出。

## 常见排障

| 现象 | 处理 |
|---|---|
| `error: input file does not exist: ...` | 检查 `input_file` 路径，必须是已存在的文件（不是 raw 目录）。 |
| raw 目录存在但 sidecar 内容仍是旧的 | 默认会**复用** raw 重建 sidecar。如果 raw 本身就过期或被替换，加 `--force-reparse` 清空重下。 |
| MinerU 报 `MINERU_API_TOKEN` 缺失 / Docling 连接 `DOCLING_ENDPOINT` 失败 | 缓存未命中触发了外部服务调用——核对对应环境变量；或确认 raw 目录是否非空（命中缓存时无需服务）。 |
| 源文件被意外移动 | 不应发生：CLI 已 mock 归档函数。若复现请提 issue（可能是 pipeline 内增加了新的归档调用点）。 |
| `parse_docling` 报 `produced zero blocks` | docling raw 中的主 JSON 内容不可解析或为空。检查 raw 目录的 `*.json` 是否合法。 |

## 与 `LightRAG.parse_*` 生产路径的等价性

本 CLI 直接调用生产代码路径 `LightRAG.parse_native` / `parse_mineru` / `parse_docling`（通过 `lightrag/parser/debug.py` 的轻量 RAG 替身），因此：

- sidecar 字段、命名、内容格式与生产入库完全一致；
- IR 构建器、`write_sidecar` 调用、`_persist_parsed_full_docs` 行为完全一致；
- 三处差异均由 CLI 内的 `monkey-patch` 实现，**不修改任何生产代码**：
  1. `parsed_artifact_dir_for_source` → 返回扁平路径（无 `__parsed__/`）；
  2. `is_bundle_valid` → 「raw 非空即有效」；
  3. `archive_docx_source_after_full_docs_sync` → no-op，保留源文件。

可与 `tests/parser/docx/golden/native_docx/` 下的 golden fixture 对比验证（CLI 不冻结时间戳，比对时排除 `created_at` 等时间字段即可）。
