# 文件处理方式配置说明

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

## `full_docs` 存储说明

文件入队和抽取结果会写入 `full_docs`：

| 字段 | 说明 |
| --- | --- |
| `file_path` | 文件名 basename（不含目录）。未提供有效来源时保存为 `unknown_source`；有效文件名的重复判定与内容溯源都基于该字段。 |
| `source_path` | 入队时提供的原始路径（仅当与 `file_path` 不同才会写入），供 `native` / `mineru` / `docling` 解析器定位真实文件位置。 |
| `format` | 内容格式：`pending_parse`, `raw`, `lightrag`。 |
| `content` | `raw` 时保存抽取文本；`pending_parse` 时为空字符串；`lightrag` 时固定为以 `{{LRdoc}}`开头的一段内容摘要。 |
| `content_hash` | 内容 MD5，用于跨文件名查重。`format=raw` 取 `sanitize_text_for_encoding` 后文本的 hash；`format=lightrag` 取 `*.blocks.jsonl` 文件 hash；`format=pending_parse` 不写入，待抽取完成后补上。 |
| `lightrag_document_path` | `format=lightrag` 时保存结构化 LightRAG Document 的路径；新记录优先保存为相对 `INPUT_DIR` 的路径，例如 `__parsed__/report.docx.parsed/report.blocks.jsonl`。 |
| `parsed_engine` | 实际完成抽取的引擎：`legacy`, `native`, `mineru`, `docling`。对于待抽取文件，也可暂存目标引擎。 |

`pending_parse` 表示文件已经入队，但还没有完成抽取。抽取成功后会改写为 `raw` 或 `lightrag`，并补齐 `content_hash`。抽取失败时保留 `pending_parse` 和空 `content`，便于后续排查和重试。

> `doc_status` 中也会同步保存 `file_path`（basename）与 `content_hash`，作为 `get_doc_by_file_basename` / `get_doc_by_content_hash` 的查重索引来源。

## 分析结果目录结构

`__parsed__` 是输入目录旁的归档与分析结果目录。它同时保存已经处理过的原始文档，以及结构化解析产生的 LightRAG Document 文件和图片等资源。

- 原始文件归档：`legacy` 本地抽取成功并入队后，原文件会移动到同级 `__parsed__` 目录；`native` / `mineru` / `docling` 会先保留原文件供 pipeline 解析，解析成功并写入 `full_docs` 后再移动到 `__parsed__`。
- 分析结果目录：结构化解析结果会写入以原始文件名加 `.parsed` 后缀命名的子目录，避免与归档原文件同名冲突。例如 `report.docx` 的分析结果目录为 `__parsed__/report.docx.parsed/`。
- 分析结果文件：LightRAG Document blocks 文件使用原文件主干命名，例如 `__parsed__/report.docx.parsed/report.blocks.jsonl`；同一目录下还可能包含 `report.tables.json`、`report.drawings.json`、`report.equations.json` 和 `report.blocks.assets/` 图片资源目录。
- 解析失败时，原文件不会移动，便于修复配置后重新处理。
- `/documents/scan` 扫描到同名且已 `PROCESSED` 的文件时，该输入文件会被视为已处理并移动到 `__parsed__`，不会作为新文档入队。
- 扫描或解析过程中发现内容 hash 重复时，该输入文件同样会移动到 `__parsed__`；本次 `doc_status` 保留为 `FAILED duplicate` 以便追踪。
- 移动文件只作用于当前输入文件，不会覆盖或移动既有文档源文件。若目标目录已存在同名文件，系统会自动追加 `_001`、`_002` 等编号，例如 `report.pdf` 会依次归档为 `report_001.pdf`、`report_002.pdf`。若分析结果目录名已被普通文件占用，也会追加编号，例如 `report.docx.parsed_001/`。

## 文档重复判定规则

文件上传、文件解析入队和文本接口会按照「文件名 + 内容 hash」两道关卡判断是否重复，命中任一即视为重复并写入一条 `FAILED` 记录，不会覆盖已有的 `full_docs`。`/documents/scan` 目录扫描也使用同一套索引，但为了便于自动重试未完成文件，对文件名重复有单独的归档与重处理规则。

### 1) 文件名（basename）查重

- 判断粒度为 basename，不包含目录路径和 workspace 路径。例如 `/data/a.pdf`、`inputs/a.pdf` 和 `a.pdf` 都视为同一个文件名 `a.pdf`。
- 文件名查重需要把文件名中的处理指引部分去掉，即认为 `abc.docx` 与 `abc.[native].docx` 是文件名重复。
- 对普通上传、文本接口和核心入队 API，只要 `doc_status` 中已经存在同名文件记录，无论该记录当前处于 `PENDING`、`PARSING`、`ANALYZING`、`PROCESSING`、`FAILED` 还是 `PROCESSED`，同名文件都会被视为重复。
- 对 `/documents/scan` 目录扫描：
  - 如果同名记录已经是 `PROCESSED`，当前扫描到的文件视为已处理文件，系统会输出 warning，将该输入文件移动到同级 `__parsed__` 目录，并跳过入队。
  - 如果同名记录不是 `PROCESSED`，当前扫描文件不会仅因文件名相同而跳过；系统会按新的扫描文件从头提取、入队并覆盖/重置未完成的同名状态。
- 普通上传和核心入队 API 中，同名文件即使内容已经变化，也需要先删除旧文档记录后再重新上传或入队；扫描路径的非 `PROCESSED` 同名重处理只用于目录扫描自动恢复。
- 文本接口必须提供有效的 `file_source`，并按 `file_source` 的 basename 判断重复；缺少有效 `file_source` 时直接返回 400。
- 核心 API `insert` / `ainsert` / `apipeline_enqueue_documents` 仍兼容未传 `file_paths` 的调用；这类文档的 `file_path` 会保存为 `unknown_source`，不会参与文件名查重，文档 ID 继续按文本内容生成。
- 为兼容遗留数据，空字符串、`no-file-path` 和 `unknown_source` 都会被视为未知来源；它们不会阻止新的无来源文本入队，也不会作为同名文件互相去重。
- 遗留数据中如果存在多个有效 basename 相同的 `file_path`，系统不会自动合并或清理；后续再次入队同名文件时会命中第一条匹配记录并按重复处理，建议先删除或修正旧记录。

存储后端通过 `get_doc_by_file_basename` 提供 basename 直查能力。`JsonDocStatusStorage` 已经实现了内存级遍历；其它后端目前回落到默认实现（扫描全部状态后比对 basename），将在后续 PR 中补齐原生索引。

### 2) 内容 hash 查重

- 文件名不同但抽取后的内容完全相同的文档同样视为重复。这里的 hash 是按配置的抽取引擎得到最终文本或 LightRAG Document 后计算的内容 hash，不是原始文件字节 hash。
- `full_docs` 与 `doc_status` 会按内容格式写入或补齐 `content_hash` 字段：
  - `format=raw`：取经过 `sanitize_text_for_encoding` 之后的文本 MD5。
  - `format=lightrag`：取 `lightrag_document_path` 解析出的 `*.blocks.jsonl` 文件 MD5。相对路径按 `INPUT_DIR` 解析。
  - `format=pending_parse`：暂不写入 hash，等到真正完成解析后由后续步骤补上（避免按空内容误判）。
- `legacy` 路径会在本地提取文本后、入队时进行内容 hash 查重；命中重复时，本次记录写为 `FAILED duplicate`，不会生成新的 `full_docs`、chunks 或图数据。
- `native` / `mineru` / `docling` 路径会先以 `pending_parse` 入队；真正完成解析并补齐 `content_hash` 后，如果发现其它文档已有相同 hash，本次记录会在进入分析、切块、实体抽取和图写入前停止。
- 重复记录会在 `metadata.duplicate_kind` 中标记为 `filename` 或 `content_hash`，便于排查。内容 hash 重复还会记录 `metadata.is_duplicate=true`、`metadata.original_doc_id` 和 `metadata.original_track_id`；解析后才发现的重复会删除本次临时写入的 `full_docs`。
- 相关 warning 会尽量减少重复噪音：扫描发现已 `PROCESSED` 的同名文件时会写入日志和 pipeline status；入队阶段重复使用 LightRAG 层的 `Duplicate document detected (...)` 日志；解析完成后才发现的内容重复使用 `Duplicate content skipped after parsing`，并写入 pipeline status。扫描归档不会额外输出 `[File Extraction]Duplicate skipped`。
- 存储后端通过 `get_doc_by_content_hash` 进行 hash 直查；命名约定与 `get_doc_by_file_basename` 一致。

> 入队批次内（同一次 `apipeline_enqueue_documents` 调用）也会做 basename 与 content_hash 去重，命中时把后续条目直接写为 `FAILED` 并标记 `existing_status=batch_duplicate`。其中 basename 去重只对有效文件名生效；`unknown_source`、`no-file-path` 和空来源只参与内容 hash 去重。

