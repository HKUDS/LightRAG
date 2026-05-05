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

使用环境变量 `LIGHTRAG_PARSER`可以给不同的文件后缀配置默认的文件内容提取方式以及默认的处理选项：

```bash
LIGHTRAG_PARSER=pdf:mineru,docx:docling,pptx:docling,xlsx:docling,*:legacy
MINERU_ENDPOINT=http://localhost:8000/api/v1/task
DOCLING_ENDPOINT=http://localhost:8081/v1/convert/file/async
```

规则格式：

```text
后缀:引擎,后缀:引擎,*:legacy
后缀:引擎;后缀:引擎;*:legacy
后缀:引擎-选项                # 在引擎后追加默认处理选项（见下文“处理选项”一节）
```

注意事项：

- 左侧匹配的是文件后缀，不是完整文件名；应写 `pdf:mineru`，不要写 `*.pdf:mineru`。
- 规则可以使用英文逗号 `,` 或分号 `;` 分隔。
- 规则按从左到右的顺序检查；优先规则放在前面；通配符规则应放在最后。
- 启动时会严格校验规则：未知内容提取引擎、错误后缀写法、显式使用不支持的后缀、外部引擎缺少 endpoint、处理选项中的非法字符都会导致启动失败。
- 通配符规则只会让引擎处理其能力表支持的后缀。例如 `*:mineru;html:docling` 中，MinerU 只接管 MinerU 支持的后缀，`html` 会继续匹配到后续 `docling` 规则。
- 如果所有规则都不可用，文件内容提取方式会回退到 `legacy`，如果legacy也不支持对应的文件后缀，会向系统加一个错误条目，上传文件保留在`INPUT`目录。
- 引擎后缀 `-选项` 部分作为该规则匹配文件的默认 `process_options`，会被文件名 hint 中的 `[...]` 覆盖。例如 `LIGHTRAG_PARSER=docx:native-iet` 表示所有 `.docx` 默认采用 `native` 引擎并开启图、表、公式分析。

## 对单文件指定内容抽取方式

可以在文件名中使用中括号临时指定单个文件的处理方式：

```text
paper.[mineru].pdf
slides.[docling].pptx
memo.[native].docx
report.[legacy].pdf
```

中括号内的内容支持三种形式：

```text
[ENGINE]              # 仅指定引擎，处理选项使用默认或 LIGHTRAG_PARSER 提供的默认
[ENGINE-OPTIONS]      # 引擎 + 处理选项
[OPTIONS]             # 仅指定处理选项，引擎仍按 LIGHTRAG_PARSER / 默认规则解析
```

仅当首段以 `-` 分隔出第二段时第一段才被作为引擎候选；否则若整段能整体匹配引擎名（`mineru` / `native` / `docling` / `legacy`），视为只指定引擎；否则整段视为选项串。文件名 hint 的优先级高于 `LIGHTRAG_PARSER`。如果指定的引擎不支持该后缀，系统会回退到默认规则继续选择可用引擎。如果所有规则都不可用，文件内容提取方式会回退到 `legacy`，如果legacy也不支持对应的文件后缀，会向系统加一个错误条目，上传文件保留在`INPUT`目录。

## 处理选项

处理选项控制单个文件在多模态分析、知识图谱构建和文本分块上的行为。所有选项都是可选的；缺省值见下表。同一文件最多指定一种分块方式（`F` / `R` / `S`），其它选项可任意组合。

| 选项 | 类型 | 默认 | 含义 |
| --- | --- | --- | --- |
| `i` | 多模态 | 关闭 | 启用图像分析（VLM） |
| `t` | 多模态 | 关闭 | 启用表格分析（VLM） |
| `e` | 多模态 | 关闭 | 启用公式分析（VLM） |
| `!` | 流水线 | 关闭 | 禁止实体/关系抽取，不构建知识图谱（仅保留 chunks 向量索引，naive / mix 检索仍可用） |
| `F` | 分块 | 默认 | 固定长度或按分隔符机械分割（按分隔符分割时块不重叠） |
| `R` | 分块 | — | 递归语义分块（优先按段落、句子分割）；当前版本回退至 `F`，行为等同于固定分块 |
| `S` | 分块 | — | 标题语义分块（优先按标题分割，标题块不重叠）；要求 `native` 抽取出的结构化输出，否则降级到 `F` |

举例：

```text
my-proposal.[native-iet].docx   # 使用 native 引擎，开启图、表、公式分析
my-memo.[native-R!].md          # 使用 native 引擎，递归语义分块，禁止知识图谱构建，多模态默认关
my-proposal.[!].docx            # 使用默认引擎（按 LIGHTRAG_PARSER 解析），仅禁止知识图谱构建
my-proposal.[mineru].docx       # 使用 MinerU 引擎，多模态、分块、KG 全部默认（即多模态关、F 分块、构建 KG）
```

校验与解析规则：

- `F`/`R`/`S` 至多出现一个；同一选项重复时只生效一次但不报错。
- 大小写敏感：分块选项 `F`/`R`/`S` 必须大写；其它选项 `i`/`t`/`e`/`!` 小写。
- 中括号内出现非法字符时，整个 hint 失效，引擎按默认规则解析、选项按 `LIGHTRAG_PARSER` 默认或全部默认；同时落日志 warning。
- 如果文件名 hint 提供了非空选项串，则以 hint 为准；否则使用 `LIGHTRAG_PARSER` 规则中匹配项的默认选项；都没有则使用全部默认。
- `S` 仅对 `native` 抽取出的结构化结果（interchange JSONL）有效；对 `legacy` 路径或非结构化输出会自动降级到 `F` 并记录 warning。

> 多模态全局开关 `addon_params["enable_multimodal_pipeline"]` 已废弃，相关行为统一由文件级 `i` / `t` / `e` 选项控制。如启动配置仍包含该字段，会在日志输出 deprecation warning 并被忽略。

### 选项作用阶段

处理选项的不同字符在流水线的不同阶段生效，具体如下：

| 选项 | 作用阶段 | 说明 |
| --- | --- | --- |
| `i` / `t` / `e` | ANALYZING（VLM 分析） | 决定是否对 sidecar 中的图像 / 表格 / 公式调用 VLM 做摘要分析。**抽取阶段不受影响**：内容提取引擎按文档实际内容输出 `drawings.json` / `tables.json` / `equations.json` sidecar 文件。这样后续仅修改 `i`/`t`/`e` 选项触发"再分析"即可补做 VLM，无须重新解析原始文件。 |
| `!` | EXTRACTION（实体关系抽取） | 跳过实体/关系抽取与图谱写入；chunks 仍写入向量库以保留 naive / mix 检索能力。 |
| `F` / `R` / `S` | CHUNKING（文本分块） | 决定使用哪种分块策略；对解析阶段输出无影响。 |

> 模态可用性以"sidecar 文件是否存在"为唯一信号，内容提取引擎不需要在 meta 中声明能力。某文档若没有任何图像/表格/公式，对应 sidecar 不会写入；用户即使开启了 `i`/`t`/`e`，对应模态也只会被静默跳过，但 `analyze_multimodal` 会在该篇文档落一行 INFO 级日志（`[analyze_multimodal] process_options opted into i:drawings ... but the parser produced no such sidecar`），便于排查"VLM 为何没跑"。这种情况不会报错。

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

为 docx 默认开启图、表、公式分析（处理选项默认）：

```bash
LIGHTRAG_PARSER=docx:native-iet
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
| `file_path` | 文件名 basename（不含目录），**保留用户提供的原始名（含中括号 hint）**，例如 `abc.[native-iet].docx` 原样写入。未提供有效来源时保存为 `unknown_source`。文件名 hint 不会被剥离，方便管理 UI 直接展示用户原本的命名意图。 |
| `canonical_basename` | 去掉处理提示 hint 后的规范化 basename（例如 `abc.docx`）。文件名查重以此字段为索引 key，保证 `abc.docx` 与 `abc.[native-iet].docx` 视为同一逻辑文档。 |
| `source_path` | 入队时提供的原始路径（仅当含目录分隔符或绝对路径时才写入），供 `native` / `mineru` / `docling` 解析器定位真实文件位置。 |
| `format` | 内容格式：`pending_parse`, `raw`, `lightrag`。 |
| `content` | `raw` 时保存抽取文本；`pending_parse` 时为空字符串；`lightrag` 时固定为以 `{{LRdoc}}`开头的一段内容摘要。 |
| `content_hash` | 内容 MD5，用于跨文件名查重。`format=raw` 取 `sanitize_text_for_encoding` 后文本的 hash；`format=lightrag` 取 `*.blocks.jsonl` 文件 hash；`format=pending_parse` 不写入，待抽取完成后补上。 |
| `lightrag_document_path` | `format=lightrag` 时保存结构化 LightRAG Document 的路径；新记录优先保存为相对 `INPUT_DIR` 的路径，例如 `__parsed__/report.docx.parsed/report.blocks.jsonl`。注意路径中的子目录与 blocks 文件名都使用规范化 basename（不含 hint）。 |
| `parsed_engine` | 实际完成抽取的引擎：`legacy`, `native`, `mineru`, `docling`。对于待抽取文件，也可暂存目标引擎。 |
| `process_options` | 入队时记录的原始处理选项串（不含引擎名和分隔 `-`），例如 `"iet"`、`"R!"`、`""`。下游各阶段以此字段为权威源，决定是否启用图像/表格/公式分析（`i`/`t`/`e`）、是否禁止知识图谱构建（`!`）以及分块方式（`F`/`R`/`S`）。空字符串等价于全部默认值。 |

`pending_parse` 表示文件已经入队，但还没有完成抽取。抽取成功后会改写为 `raw` 或 `lightrag`，并补齐 `content_hash`。抽取失败时保留 `pending_parse` 和空 `content`，便于后续排查和重试。

> `doc_status` 中也同步保存原始 `file_path`（含 hint）、`canonical_basename` 与 `content_hash`，作为 `get_doc_by_file_basename` / `get_doc_by_content_hash` 的查重索引来源。`get_doc_by_file_basename` 内部把传入参数先经 `canonicalize_parser_hinted_basename` 规范化后再与 `canonical_basename` 比对，因此 `abc.docx` 与 `abc.[native-iet].docx` 总是命中同一文档。
> `process_options` 同时镜像写入 `doc_status.metadata["process_options"]`，便于管理 UI 直接展示当前文件的处理策略。

## 内容提取结果目录结构

`__parsed__` 是输入目录旁的归档与分析结果目录。它同时保存已经处理过的原始文档，以及结构化解析产生的 LightRAG Document 文件和图片等资源。

- 原始文件归档：`legacy` 本地抽取成功并入队后，原文件会移动到同级 `__parsed__` 目录；`native` / `mineru` / `docling` 会先保留原文件供 pipeline 解析，解析成功并写入 `full_docs` 后再移动到 `__parsed__`。**归档时保留原始文件名（含 `[hint]`）**，例如 `report.[native-iet].docx` 归档为 `__parsed__/report.[native-iet].docx`，便于追溯用户最初的命名与处理选项。
- 分析结果目录：结构化解析结果会写入以**规范化文件名**（去掉 `[hint]`）加 `.parsed` 后缀命名的子目录，避免与归档原文件同名冲突，并保证当文件名 hint 或处理选项变化时同一逻辑文档继续指向同一目录。例如 `report.docx`、`report.[native].docx`、`report.[native-iet].docx` 的分析结果都写入 `__parsed__/report.docx.parsed/`。
- 分析结果文件：LightRAG Document blocks 文件以及 sidecar 都使用规范化文件名的主干命名，例如 `__parsed__/report.docx.parsed/report.blocks.jsonl`；同一目录下还可能包含 `report.tables.json`、`report.drawings.json`、`report.equations.json` 和 `report.blocks.assets/` 图片资源目录。**sidecar 是否生成由文档内容决定**：解析器只在文档实际包含表格/图片/公式时写出对应文件。这是模态可用性的唯一信号 —— 引擎不需要在 meta 中声明能力。`i`/`t`/`e` 选项只决定下一阶段是否对已存在的 sidecar 调用 VLM 做摘要分析。
- 解析失败时，原文件不会移动，便于修复配置后重新处理。
- `/documents/scan` 扫描到同名且已 `PROCESSED` 的文件时，该输入文件会被视为已处理并移动到 `__parsed__`，不会作为新文档入队。
- `/documents/scan` 同一次扫描中发现多个规范化后同名的文件时，会优先保留带支持引擎 hint 的文件以尊重用户的引擎选择；如果没有任何变体带 hint，则按排序处理第一个文件。其余变体会输出 warning 并移动到 `__parsed__`，避免同批文件互相覆盖。例如 `abc.docx` 和 `abc.[native].docx` 同时存在时只会处理 `abc.[native].docx`。
- 扫描或解析过程中发现内容 hash 重复时，该输入文件同样会移动到 `__parsed__`；本次 `doc_status` 保留为 `FAILED duplicate` 以便追踪。
- 移动文件只作用于当前输入文件，不会覆盖或移动既有文档源文件。若目标目录已存在同名文件，系统会自动追加 `_001`、`_002` 等编号，例如 `report.pdf` 会依次归档为 `report_001.pdf`、`report_002.pdf`。若分析结果目录名已被普通文件占用，也会追加编号，例如 `report.docx.parsed_001/`。

## 文档重复判定规则

文件上传、文件解析入队和文本接口会按照「文件名 + 内容 hash」两道关卡判断是否重复，命中任一即视为重复并写入一条 `FAILED` 记录，不会覆盖已有的 `full_docs`。`/documents/scan` 目录扫描也使用同一套索引，但为了便于自动重试未完成文件，对文件名重复有单独的归档与重处理规则。

### 1) 文件名（basename）查重

- 判断粒度为 basename，不包含目录路径和 workspace 路径。例如 `/data/a.pdf`、`inputs/a.pdf` 和 `a.pdf` 都视为同一个文件名 `a.pdf`。
- 文件名查重以 `canonical_basename` 为索引：将文件名末尾的支持引擎处理提示 hint 剥离后再比对，因此 `abc.docx`、`abc.[native].docx`、`abc.[native-iet].docx` 之间互相视为同名；不支持的 hint 不会被剥离，例如 `abc.[draft].docx` 仍按原文件名处理。
- 对普通上传、文本接口和核心入队 API，只要 `doc_status` 中已经存在同名文件记录，无论该记录当前处于 `PENDING`、`PARSING`、`ANALYZING`、`PROCESSING`、`FAILED` 还是 `PROCESSED`，同名文件都会被视为重复。
- 对 `/documents/scan` 目录扫描：
  - 同一次扫描中如果有多个文件规范化后同名，优先处理带支持引擎 hint 的文件；若无任何 hint 变体，则处理排序后的第一个文件，其余文件会归档到 `__parsed__` 并跳过。
  - 如果同名记录已经是 `PROCESSED`，当前扫描到的文件视为已处理文件，系统会输出 warning，将该输入文件移动到同级 `__parsed__` 目录，并跳过入队。
  - 如果同名记录不是 `PROCESSED`，扫描文件**不**仅因文件名相同而跳过，但**也不**会重新提取/覆盖既有记录。具体路径取决于既有记录的形态（与下文"为什么 scan 仍是独占写者"一节列举的分类规则一致）：
    - 同名非 PROCESSED 且 `full_docs` 存在 → **resume 路径**：doc_status 现状保留，源文件留在 `INPUT/`，由处理循环按状态查询接走（不重新提取、不覆盖既有状态）。
    - 同名 `FAILED` 且 `full_docs` 缺失 → 视为 `apipeline_enqueue_error_documents` 写下的提取错误 stub：scan 删掉这条 stub 后**把当前文件按新文件重新入队**。这是唯一会重新提取的子分支，目的是让"修好源文件再 scan 一次"自动生效。
- 普通上传和核心入队 API 中，同名文件即使内容已经变化，也需要先删除旧文档记录后再重新上传或入队；扫描路径上述两种自动恢复仅用于目录扫描场景。
- 文本接口必须提供有效的 `file_source`，并按 `file_source` 的 basename 判断重复；缺少有效 `file_source` 时直接返回 400。
- 核心 API `insert` / `ainsert` / `apipeline_enqueue_documents` 仍兼容未传 `file_paths` 的调用；这类文档的 `file_path` 会保存为 `unknown_source`，不会参与文件名查重，文档 ID 继续按文本内容生成。
- 空字符串、`no-file-path` 和 `unknown_source` 都会被视为未知来源；它们不会阻止新的无来源文本入队，也不会作为同名文件互相去重。

存储后端通过 `get_doc_by_file_basename` 提供 basename 直查能力，内部按 `canonical_basename` 字段比对（传入参数会先经 `canonicalize_parser_hinted_basename` 规范化）。`JsonDocStatusStorage` 已经实现了内存级遍历；其它后端目前回落到默认实现（扫描全部状态后比对 `canonical_basename`），将在后续 PR 中补齐原生索引。

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
>
> **跨调用并发去重**也由 workspace 级串行锁保证（详见 [enqueue 串行锁（防并发去重穿透）](#enqueue-串行锁防并发去重穿透)）：两次相同内容、不同文件名的并发入队不会双双穿透 `content_hash` 检查。

## 并发与重入约束

为防止 `scan` / `upload` / `insert` 与运行中的流水线相互覆盖 `doc_status` / `full_docs` 记录，所有写入入口在 `pipeline_status` 共享字典上协调。同一 workspace 下的 `pipeline_status_lock` 保证下表所有 transition 都在锁内原子完成：

| 字段 | 语义 |
| --- | --- |
| `busy` | 流水线繁忙的笼统标志。处理循环和破坏性作业（clear/delete）都会设它。**仅有 `busy=True`（处理循环）不阻塞 enqueue**——循环按 batch 拉取 `doc_status` 快照处理，每批结束后通过 `request_pending` 检查是否还有新工作。 |
| `destructive_busy` | `busy` 的破坏性子集：`/documents/clear` 或 `/documents/{doc_id}`（删除）正在 drop 存储 / 删源文件。reservation 和 enqueue last-line guard 都会拒绝——并发 enqueue 会写入正被 drop 的存储，已接受的文档会静默丢失。处理循环不会设此字段。 |
| `scanning` | `/documents/scan` 后台任务运行中（整个生命周期：分类阶段 + 处理阶段）。仅 `/scan` 端点用它拒绝重叠 scan，本身**不**阻塞 upload/insert。 |
| `scanning_exclusive` | `scanning` 的独占子集：只在 scan 的**分类阶段**为 True——run_scanning_process 在读 doc_status 分类（已处理 / 续跑 / 删 stub / 归档），不能与并发写者交错。reservation 和 enqueue last-line guard 都会拒绝。分类完成后会立即清旗，scan 进入处理阶段后允许并发 upload。 |
| `pending_enqueues` | 已通过 `_reserve_enqueue_slot` 但 bg task 未完成的 upload/insert 数。仅给 scan 端点参考——决定是否能拿独占。bg task 在 `finally` 里释放 slot。 |
| `request_pending` | 让运行中的处理循环再扫一轮的信号。enqueue 在 `busy=True` 时写完 `doc_status` 后置位；处理循环每个 batch 结束后检查并重新拉快照。 |

### 入口行为

| 入口 | 条件 | 行为 |
| --- | --- | --- |
| `/documents/upload` / `/documents/text` / `/documents/texts` | `scanning_exclusive=True` 或 `destructive_busy=True` | 抛 HTTP 409，不写文件、不调入队 |
| 同上 | 否则（含纯 `busy=True`、scan 处理阶段 `scanning=True` 但 `scanning_exclusive=False`） | 锁内 `pending_enqueues++` 预留 slot → 严格名字预检 → 保存文件 → schedule bg task；bg task 在 `finally` 释放 slot |
| `/documents/scan` | `busy=True` 或 `scanning=True` 或 `pending_enqueues>0` | 落 warning 后立即返回 `scanning_skipped_pipeline_busy`，不 schedule 后台任务 |
| 同上 | 全部 idle | 锁内设 `scanning=True` 后 schedule，task 结束在 `finally` 清旗 |
| `/documents/clear` / `/documents/delete_document` | `busy=True` 或 `scanning=True` 或 `pending_enqueues>0` | 端点同步返回 `status="busy"`，不 schedule 后台任务 |
| 同上 | 全部 idle | 端点**同步**在锁内设 `busy=True` + `destructive_busy=True`（`delete_document` 在返回 `deletion_started` 之前），bg task 的 finally 一并清旗 |
| `apipeline_enqueue_documents` 内部 (last-line guard) | `scanning_exclusive=True` 且 `from_scan=False`，或 `destructive_busy=True` | 抛 `RuntimeError("Cannot enqueue while scan is classifying / clearing or deleting")` |
| 同上 | 任何其它情况（含纯 `busy=True`、scan 处理阶段） | 正常入队；写完 `doc_status` 后若 `busy=True` 自动 nudge `request_pending=True` |

`from_scan=True` 是 scan 后台任务自身入队时的旁路：scan 已持有 `scanning` 旗标，必须允许它把扫到的文件入队。

### 为什么 `busy` 不再阻塞 enqueue

旧版本里 `busy=True` 一律拒绝任何新入队，理由是"修改 `doc_status` 会与流水线工作线程交错"。但实际上：

1. **写入顺序保证一致性**：`apipeline_enqueue_documents` 总是先 upsert `full_docs`、再 upsert `doc_status`。处理循环开头的 consistency check 仅删除"`doc_status` 行没有对应 `full_docs`"的孤儿——这种状态在并发 enqueue 中不可能出现。
2. **批次级快照**：处理循环每个 batch 拉一次 `get_docs_by_statuses` 快照，新写入的 `PENDING` 行不会破坏当前 batch；下一轮通过 `request_pending` 重拉快照即可看到新工作。
3. **`request_pending` 设计本就为此**：旧版同时存在 `request_pending` 字段——它就是为"运行中又有新工作"设计的，但被 busy 守护堵死了。

新契约把这个机制启用起来后，**用户在长批次处理过程中仍可继续上传新文档**，bg task 写完 `doc_status` 后由运行中的循环自动接管。

### 为什么 scan 仍是独占写者

scan 不仅 enqueue 自己扫到的新文件，还会读 `doc_status` 决定每个文件去向：

- 同名 `PROCESSED` 行 → 归档源文件、跳过入队。
- 同名非 PROCESSED 且 `full_docs` 存在 → resume 路径，源文件**保留在 `INPUT/`**，不归档（pending-parse 解析器仍可能需要它），由处理循环按状态查询接走。
- 同名 `FAILED` 且 `full_docs` 缺失 → 识别为之前 `apipeline_enqueue_error_documents` 写下的提取错误 stub（一致性检查会保留这种行供人工 review），scan 自动删除该 stub 并把当前文件按新文件重新入队，让用户"修好源文件再 scan 一次"能直接生效。

这些"读—决策—写"组合不能与其它写者交错，否则分类决策会基于过期视图。所以 scan 必须独占，且 scan 端点会在 `busy` / `scanning` / `pending_enqueues>0` 任一存在时拒绝。

### 严格名字预检（upload 路径）

upload 通过 reservation 后、保存文件前必须双道检查：

1. **INPUT 目录扫描**：把要保存的 basename 经 `canonicalize_parser_hinted_basename` 规范化，遍历 INPUT 目录里现有任何同 canonical 变体（含 hint / 不含 hint），命中即 409。
2. **doc_status 查重**：用规范化 basename 调 `get_existing_doc_by_file_basename`，命中即 409。

两道都过 → 保存文件 → schedule bg task → bg task 调 `apipeline_enqueue_documents` 写库 + 调 `apipeline_process_enqueue_documents` 触发处理。

> 旧版本曾允许 upload 在已有同名记录时悄悄写入 FAILED 重复条目；新规则改为 fail-fast，不在 doc_status 留下任何重复痕迹。如需替换同名文档，请先调用 `/documents/{doc_id}` 的删除接口。

### 多 reservation 并发的协调

两个 upload 同时进来时（scan 此时拿不到独占）：

1. A `_reserve_enqueue_slot` → `pending_enqueues=1`，写文件，schedule bg task A，返回 success。
2. B `_reserve_enqueue_slot` → `pending_enqueues=2`，写文件，schedule bg task B，返回 success。
3. bg task A `apipeline_enqueue_documents` → 写 `doc_status` → 调 `apipeline_process_enqueue_documents` → 设 `busy=True` 处理 A 的文档。
4. bg task B `apipeline_enqueue_documents` → 看到 `scanning=False`，正常写入；写完后看到 `busy=True`，自动设 `request_pending=True`。
5. bg task B 调 `apipeline_process_enqueue_documents` → 看到 `busy=True`，设 `request_pending=True` 立即返回。
6. A 的处理循环跑完当前 batch，看到 `request_pending=True`，重拉快照，把 B 的 `PENDING` 行接上处理。
7. 全部完成后 `busy=False`、`pending_enqueues=0`。

任何一个 bg task 都不会因为 busy 被误拒——因为 enqueue 不再检查 busy；处理循环也不会重复处理同一份 batch——`request_pending` 只在 batch 间生效，且每次重拉前清零。

### enqueue 串行锁（防并发去重穿透）

`apipeline_enqueue_documents` 内部"读 doc_status 做去重 → 写 `full_docs` / `doc_status`"这一段在 workspace 级 `enqueue_serialize` 锁内串行执行。原因：放开 busy/scan-processing 阶段允许并发 enqueue 之后，两次相同内容、不同文件名的入队（典型场景：scan 处理阶段的 enqueue 与 upload 同时进来）若在没有锁的情况下并发执行——

1. A 读 `doc_status` 查 `content_hash`：未命中。
2. B 读 `doc_status` 查 `content_hash`：仍未命中（A 还没 upsert）。
3. A upsert `full_docs` + `doc_status`。
4. B upsert `full_docs` + `doc_status`。

结果：同 `content_hash` 的两条 `PENDING` 都进入流水线后续处理，原本应当被识别为 `duplicate_kind=content_hash` 的那条**没**被识别。

加上串行锁后第二次 enqueue 一定能在去重读时看到第一次已 upsert 的行，正常走"无新唯一文档"的早返回路径并把本次记为 `duplicate_kind=content_hash` 的 FAILED 行。锁的作用范围**只覆盖**：

- `filter_keys`（按 doc_id 排除已存在）
- 文件名 / 内容 hash 去重读
- 重复 FAILED 行的 upsert
- `full_docs.upsert` + `doc_status.upsert`

锁**不**覆盖 `request_pending` nudge（在锁外，只取一下 `pipeline_status_lock`），也**不**阻塞处理循环的 `get_docs_by_statuses` 读（处理循环走的是 `doc_status` 自身的并发读，与 enqueue 写是 KV 级原子，不抢同一把锁）。锁顺序：`enqueue_serialize → pipeline_status_lock`，无死锁路径。

### 移除 `reprocess_existing_non_processed`

旧 `apipeline_enqueue_documents` 的 `reprocess_existing_non_processed=True` 行为会在 scan 时直接删除非 PROCESSED 的旧记录并重建，与本规则相冲突，已整段移除。scan 改为按"为什么 scan 仍是独占写者"一节的分类规则处理同名文件：归档 / 续跑 / 删 stub 后重入队，由"流水线启动时的续跑规则"在处理循环里统一接管。

## 流水线启动时的续跑规则

每次 `apipeline_process_enqueue_documents` 起步时，会拉取所有处于 `PARSING` / `ANALYZING` / `PROCESSING` / `PENDING` / `FAILED` 状态的文档继续处理。续跑路径**根据"内容是否已抽取"分流**，保证同一个文档无论之前进度如何，按当前 `process_options` 续跑都有幂等结果。

### 判断"内容已抽取"

读 `full_docs[doc_id]`：

| `format` | 判定 |
| --- | --- |
| `lightrag` 且 `lightrag_document_path` 文件存在 | ✅ 已抽取 |
| `raw` 且 `content` 非空 | ✅ 已抽取 |
| 其它（含 `pending_parse`、记录缺失） | ❌ 未抽取 |

### 分支 A：未抽取

走完整流水线（`parse_native` / `parse_mineru` / `parse_docling` → `analyze_multimodal` → 分块 → 实体抽取），按 `full_docs.process_options` 决定每一阶段的行为。这是"首次入队"的常规流。

### 分支 B：已抽取

**一律跳过解析**（不重新调 `parse_*`），从 ANALYZING 阶段重启，并清光旧 chunks / entities 后按当前 `process_options` 重做：

| 子步骤 | 行为 |
| --- | --- |
| 引擎对比 | 若 `process_options` 隐含的引擎 ≠ `full_docs.parsed_engine`，**仅 warn**，不重新解析。已抽取的内容是不可变事实，重新跑不同引擎会产生不一致。要切换引擎请先 delete 整个文档再重传。 |
| 旧 chunks 清理 | 读 `doc_status.chunks_list`，从 `chunks_vdb` 与 `text_chunks` 全部 delete。理由：流水线产物中无法可靠区分"普通文本块 vs 多模态附加块"，按 chunk id 一律重新生成最简单也最可靠 |
| 旧实体 / 关系清理 | 复用 `adelete_by_doc_id` 内部清理逻辑（抽出为 `_purge_doc_chunks_and_kg(doc_id)` helper），删除 `entity_chunks` / `relation_chunks` 中以这些 chunk id 为 source 的条目，并把图谱里因之失去全部源的孤立节点一并删除 |
| `analyze_multimodal` | **不再看 `meta.analyze_time`**：按新 `process_options.{i,t,e}` 与 sidecar 中各 item 的 `llm_analyze_result` 取交集做增量分析（已分析的 item 跳过，新启用的模态从空状态开始分析）。`analyze_time` 改为"最近一次成功分析时间"语义，仅供观测 |
| 重新分块 | 按新 `process_options.chunking` 重跑（interchange path 用 native heading-driven，legacy path 用 fixed） |
| 实体抽取 / KG-skip | 按新 `process_options.skip_kg` 决定 |

> 这条规则保证：用户改 `i/t/e` 重传同名文档（先删旧 doc 再上传带新 hint 的文件）时，多模态分析能增量补齐；改 `F`/`R`/`S` 时 chunks 与图谱重建；改 `!` 时停掉或恢复 KG 构建。引擎变更被视为"重大变更"，统一由 delete + 重传完成，不在续跑路径里隐式发生。

### 与"文档重复判定规则"的关系

续跑规则只对 `doc_id` 已经存在于 `doc_status` 的文档生效。新文件入队仍然走"并发与重入约束"中的严格名字预检 + "文档重复判定规则"中的 `canonical_basename` / `content_hash` 查重；续跑分支不会被用来"新文件挤掉旧 PROCESSED 记录"。
