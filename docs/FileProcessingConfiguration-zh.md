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
  - 如果同名记录不是 `PROCESSED`，当前扫描文件不会仅因文件名相同而跳过；系统会按新的扫描文件从头提取、入队并覆盖/重置未完成的同名状态。
- 普通上传和核心入队 API 中，同名文件即使内容已经变化，也需要先删除旧文档记录后再重新上传或入队；扫描路径的非 `PROCESSED` 同名重处理只用于目录扫描自动恢复。
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
