# 文件处理方式配置指南

从 LightRAG Server v1.5.0 开始，文件索引流水线支持使用外部的 MinerU 和 Docling 引擎进行文件内容的分析抽取，抽取内容将以 `LightRAG Document` 格式保存到`INPUT`输入目录下的 `__parsed__`子目录。`LightRAG Document`文件格式支持表格和图片等多模态数据，同时包含文章的章节段落元数据，方便日后进行内容溯源。LightRAG还提供了一个内置的 native 文件抽取引擎，可以高效地实现 docx 的智能内容抽取，支持章节标题、自动编号、表格和图片等内容的精确提取。

## 支持的抽取引擎类型

| 引擎 | 说明 | 支持的文件格式（后缀） |
| --- | --- | --- |
| `legacy` | 旧版提取方式，在加入管线前集中提取内容 | `txt` `md` `mdx` `pdf` `docx` `pptx` `xlsx` `rtf` `odt` `tex` `epub` `html` `htm` `csv` `json` `xml` `yaml` `yml` `log` `conf` `ini` `properties` `sql` `bat` `sh` `c` `h` `cpp` `hpp` `py` `java` `js` `ts` `swift` `go` `rb` `php` `css` `scss` `less` |
| `native` | 内置智能结构化内容抽取器 | `docx` |
| `mineru` | 外部 MinerU 内容提取引擎 | `pdf`  `docx`  `pptx`  `xlsx` |
| `docling` | 外部 Docling 内容提取引擎 | `pdf` `docx` `pptx` `xlsx` `md` `html` `xhtml` `png` `jpg` `jpeg` `tiff` `webp` `bmp` |

> 为了向后兼容，在未修改配置的情况下升级系统文件内容提取方式方式会维持原来的legacy不变。如需启用新的内容处理引擎，需要按照以下方式来配置

## 修改默认内容抽取方式

使用环境变量 `LIGHTRAG_PARSER`可以给不同的文件后最配资默认的文件内容提取方式：

```bash
LIGHTRAG_PARSER=pdf:mineru,docx:docling,pptx:docling,xlsx:docling,*:legacy
MINERU_ENDPOINT=http://localhost:8000/api/v1/task
DOCLING_ENDPOINT=http://localhost:8081/v1/convert/file/async
```

规则格式：

```text
后缀:引擎,后缀:引擎,*:legacy
后缀:引擎;后缀:引擎;*:legacy
```

注意事项：

- 左侧匹配的是文件后缀，不是完整文件名；应写 `pdf:mineru`，不要写 `*.pdf:mineru`。
- 规则可以使用英文逗号 `,` 或分号 `;` 分隔。
- 规则按从左到右的顺序检查；优先规则放在前面；通配符规则应放在最后。
- 启动时会严格校验规则：未知内容提取引擎、错误后缀写法、显式使用不支持的后缀、外部引擎缺少 endpoint 都会导致启动失败。
- 通配符规则只会让引擎处理其能力表支持的后缀。例如 `*:mineru;html:docling` 中，MinerU 只接管 MinerU 支持的后缀，`html` 会继续匹配到后续 `docling` 规则。
- 如果所有规则都不可用，文件内容提取方式会回退到 `legacy`，如果legacy也不支持对应的文件后缀，会向系统加一个错误条目，上传文件保留在`INPUT`目录。

## 对单文件指定内容抽取方式

可以在文件名中使用中括号临时指定单个文件的处理方式：

```text
paper.[mineru].pdf
slides.[docling].pptx
memo.[native].docx
report.[legacy].pdf
```

文件名 hint 的优先级高于 `LIGHTRAG_PARSER`。如果指定的引擎不支持该后缀，系统会回退到默认规则继续选择可用引擎。如果所有规则都不可用，文件内容提取方式会回退到 `legacy`，如果legacy也不支持对应的文件后缀，会向系统加一个错误条目，上传文件保留在`INPUT`目录。

## 文档重复判定规则

文件上传、目录扫描、文件解析入队和文本接口在入队前会按照「文件名 + 内容 hash」两道关卡来判断是否重复，命中任一即视为重复并写入一条 `FAILED` 记录，不会覆盖已有的 `full_docs`。

### 1) 文件名（basename）查重

- 判断粒度为 basename，不包含目录路径和 workspace 路径。例如 `/data/a.pdf`、`inputs/a.pdf` 和 `a.pdf` 都视为同一个文件名 `a.pdf`。
- 只要 `doc_status` 中已经存在同名文件记录，无论该记录当前处于 `PENDING`、`PARSING`、`ANALYZING`、`PROCESSING`、`FAILED` 还是 `PROCESSED`，同名文件都会被视为重复。
- 同名文件即使内容已经变化，也需要先删除旧文档记录后再重新上传或入队。
- 文本接口必须提供有效的 `file_source`，并按 `file_source` 的 basename 判断重复；缺少有效 `file_source` 时直接返回 400。

存储后端通过 `get_doc_by_file_basename` 提供 basename 直查能力。`JsonDocStatusStorage` 已经实现了内存级遍历；其它后端目前回落到默认实现（扫描全部状态后比对 basename），将在后续 PR 中补齐原生索引。

### 2) 内容 hash 查重

- 文件名不同但内容完全相同的文档同样视为重复，不会再次入队。
- `full_docs` 与 `doc_status` 在入队时都会写入 `content_hash` 字段：
  - `format=raw`：取经过 `sanitize_text_for_encoding` 之后的文本 MD5。
  - `format=lightrag`：取 `lightrag_document_path` 解析出的 `*.blocks.jsonl` 文件 MD5（与 `_load_lightrag_document_content` 的解析规则一致）。
  - `format=pending_parse`：暂不写入 hash，等到真正完成解析后由后续步骤补上（避免按空内容误判）。
- 重复记录会在 `metadata.duplicate_kind` 中标记为 `filename` 或 `content_hash`，便于排查。
- 存储后端通过 `get_doc_by_content_hash` 进行 hash 直查；命名约定与 `get_doc_by_file_basename` 一致。

> 入队批次内（同一次 `apipeline_enqueue_documents` 调用）也会做 basename 与 content_hash 去重，命中时把后续条目直接写为 `FAILED` 并标记 `existing_status=batch_duplicate`。

## 推荐配置

### 保持旧版行为

不配置 `LIGHTRAG_PARSER`：

```bash
# LIGHTRAG_PARSER=
```

所有文件按旧版 `legacy` 本地抽取方式处理。

使用Native引擎处理docx文件，其余保持旧版行为：

```bash
LIGHTRAG_PARSER=docx:native
```

### 梦幻组合

* 使用Legacy处理md（写在最前面是为了避免用Docling处理md文件）
* 使用Native处理docx文件（写在最前面是为了避免用MinerU处理docx文件）
* 用MinerU 处理其自此的（自此的 PDF和其余Office)
* 让Docling 处理其余它自此的文件格式（HTML和图片等）
* 企业文件格式用Legacy

```bash
LIGHTRAG_PARSER=md:legacy,docx:native,*:mineru,*:docling,*:legacy
MINERU_ENDPOINT=http://localhost:8000/api/v1/task
DOCLING_ENDPOINT=http://localhost:8081/v1/convert/file/async
```

## `full_docs`存储存储说明

文件入队和抽取结果会写入 `full_docs`：

| 字段 | 说明 |
| --- | --- |
| `parsed_engine` | 实际完成抽取的引擎：`legacy`, `native`, `mineru`, `docling`。对于待抽取文件，也可暂存目标引擎。 |
| `format` | 内容格式：`pending_parse`, `raw`, `lightrag`。 |
| `content` | `raw` 时保存抽取文本；`pending_parse` 时为空字符串；`lightrag` 时固定为 `{{stored-in-lightrag-doucment}}`。 |
| `lightrag_document_path` | `format=lightrag` 时保存结构化 LightRAG Document 的相对路径。 |

其中 `file_path` 是文件名重复判定和内容溯源的关键字段，建议保持稳定、可读的文件名来源。

`pending_parse` 表示文件已经入队，但还没有完成抽取。抽取成功后会改写为 `raw` 或 `lightrag`。抽取失败时保留 `pending_parse` 和空 `content`，便于后续排查和重试。

## 原始文件归档

- `legacy`：本地抽取成功并入队后，原文件会移动到同级 `__parsed__` 目录。
- `native` / `mineru` / `docling`：文件先保留在原位置供 pipeline 解析；解析成功并写入 `full_docs` 后，原文件再移动到 `__parsed__`。
- 解析失败时，原文件不会移动，便于修复配置后重新处理。
- 同名重复文件不会作为新文档入队，也不应为了重复判定而覆盖或移动既有文档源文件。
