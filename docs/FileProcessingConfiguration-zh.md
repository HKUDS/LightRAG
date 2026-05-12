# 文件处理方式配置说明

从版本 v1.5.0 （目前在dev分支）开始，LightRAG的文件处理管线进行了重大的升级：

* 支持多种文件内容抽引擎：legacy、native、mineru、docling
* 支持多种文本块分块方法：Fix、Recursive、Vector、Paragraph
* 支持对个别文件关闭实体关系抽取

LightRAG Server引入了一个文件处理的中间格式： `LightRAG Document` 。该格式支持表格和图片等多模态数据，同时包含文章的章节段落元数据，方便日后进行内容溯源。

本文以 **LightRAG Server** 的部署与使用视角组织：先给出快速开始可直接套用的配置，再展开内容抽取与分块的配置语法、存储 / 目录布局、去重、并发以及续跑规则。直接通过 Python 代码调用 `LightRAG` 类的开发者请翻到[第八章 Python SDK 调用](#八、Python SDK 调用)。

## 一、快速开始

### 保持旧版行为

不配置 `LIGHTRAG_PARSER`：

```bash
# LIGHTRAG_PARSER=
```

所有文件按旧版 `legacy` 本地抽取方式处理。

### 仅启用 Native 处理 docx

```bash
# 使用默认的F分块策略
LIGHTRAG_PARSER=docx:native

# 使用R分块策略
LIGHTRAG_PARSER=docx:native-R
```

为 docx 默认开启图、表、公式分析（`-iet` 后缀给该规则的所有匹配文件加上默认处理选项）：

```bash
# 使用默认的F分块策略
LIGHTRAG_PARSER=docx:native-iet

# 使用V分块策略
LIGHTRAG_PARSER=docx:native-ietV
```

### 梦幻组合

* 用 Legacy 处理 md（写在最前面是为了避免用 Docling 处理 md 文件）
* 用 Native 处理 docx 文件（写在最前面是为了避免用 MinerU 处理 docx 文件）
* 用 MinerU 处理它擅长的（PDF 和其余 Office）
* 让 Docling 处理它擅长的其他文件格式（HTML 和图片等）
* 其余文件回退 Legacy

```bash
LIGHTRAG_PARSER=md:legacy-R,docx:native-R,*:mineru-R,*:docling-R,*:legacy-R
MINERU_ENDPOINT=http://localhost:8000/api/v1/task
DOCLING_ENDPOINT=http://localhost:8081/v1/convert/file/async
```

## 二、内容抽取与处理选项配置

LightRAG 的文件处理配置由两部分合成：内容抽取引擎决定原始文件如何被解析，处理选项决定解析后是否执行多模态分析、使用哪种分块方式，以及是否构建知识图谱。通常先用环境变量 `LIGHTRAG_PARSER` 按文件后缀设置默认规则，再用文件名中的 `[hint]` 覆盖单个文件。引擎和选项可以写在同一个配置片段里，例如 `docx:native-iet` 或 `report.[native-R!].docx`。

为了向后兼容，在未修改配置的情况下，升级后的文件内容提取方式会维持原来的 `legacy` 行为。如需启用新的内容处理引擎，请按本节说明配置。

### 2.1 配置语法总览

完整配置模型如下：

```text
LIGHTRAG_PARSER=后缀:引擎-选项,后缀:引擎,*:legacy-R
filename.[ENGINE].ext
filename.[ENGINE-OPTIONS].ext
filename.[OPTIONS].ext
```

- `LIGHTRAG_PARSER` 是默认规则表，按文件后缀匹配，例如 `pdf:mineru`、`docx:native-iet`。
- 文件名 `[hint]` 是单文件覆盖规则，例如 `paper.[mineru].pdf`、`memo.[native-R!].docx`。
- `ENGINE` 是内容抽取引擎：`legacy`、`native`、`mineru` 或 `docling`。
- `OPTIONS` 是处理选项字符组合，例如 `iet`、`R!`、`P`。选项最终写入 `process_options`，由后续流水线阶段读取。
- `ENGINE-OPTIONS` 中的连字符只用于分隔引擎和选项，不属于选项本身。

常见组合示例：

```bash
LIGHTRAG_PARSER=pdf:mineru-R,docx:native-ietP,*:legacy-R
MINERU_ENDPOINT=http://localhost:8000/api/v1/task
DOCLING_ENDPOINT=http://localhost:8081/v1/convert/file/async
```

```text
my-proposal.[native-iet].docx   # 使用 native 引擎，开启图、表、公式分析
my-memo.[native-R!].docx        # 使用 native 引擎，递归语义分块，禁止知识图谱构建
my-proposal.[!].docx            # 使用默认引擎，仅禁止知识图谱构建
my-proposal.[mineru].docx       # 使用 MinerU 引擎，处理选项全部默认
```

### 2.2 默认规则：`LIGHTRAG_PARSER`

`LIGHTRAG_PARSER` 用来为不同文件后缀配置默认内容抽取引擎，也可以在引擎后追加该规则的默认处理选项：

```text
后缀:引擎,后缀:引擎,*:legacy
后缀:引擎;后缀:引擎;*:legacy
后缀:引擎-选项
```

- 左侧匹配的是文件后缀，不是完整文件名；应写 `pdf:mineru`，不要写 `*.pdf:mineru`。
- 规则可以使用英文逗号 `,` 或分号 `;` 分隔。
- 规则按从左到右的顺序检查；优先规则放在前面，通配符规则通常放在最后。
- 引擎后缀 `-选项` 部分作为该规则匹配文件的默认 `process_options`。例如 `LIGHTRAG_PARSER=docx:native-iet` 表示所有 `.docx` 默认采用 `native` 引擎，并开启图像、表格、公式分析。

### 2.3 单文件覆盖：文件名 hint

文件名中可以使用中括号临时指定单个文件的处理方式：

```text
paper.[mineru-R].pdf
slides.[docling].pptx
memo.[native-P].docx
notes.[R].md
```

中括号内的内容支持三种形式：

```text
[ENGINE]              # 仅指定引擎，处理选项使用默认或 LIGHTRAG_PARSER 提供的默认
[ENGINE-OPTIONS]      # 同时指定引擎和处理选项
[OPTIONS]             # 仅指定处理选项，引擎仍按 LIGHTRAG_PARSER / 默认规则解析
```

解析 hint 时，仅当首段以 `-` 分隔出第二段时，第一段才会被作为引擎候选；否则若整段能整体匹配引擎名（`mineru` / `native` / `docling` / `legacy`），视为只指定引擎；否则整段视为选项串。

### 2.4 内容抽取引擎

| 引擎 | 说明 | 支持的文件格式（后缀） |
| --- | --- | --- |
| `legacy` | 旧版提取方式，在加入管线前集中提取内容 | `txt` `md` `mdx` `pdf` `docx` `pptx` `xlsx` `rtf` `odt` `tex` `epub` `html` `htm` `csv` `json` `xml` `yaml` `yml` `log` `conf` `ini` `properties` `sql` `bat` `sh` `c` `h` `cpp` `hpp` `py` `java` `js` `ts` `swift` `go` `rb` `php` `css` `scss` `less` |
| `native` | 内置智能结构化内容抽取器 | `docx` |
| `mineru` | 外部 MinerU 内容提取引擎 | `pdf` `docx` `pptx` `xlsx` |
| `docling` | 外部 Docling 内容提取引擎 | `pdf` `docx` `pptx` `xlsx` `md` `html` `xhtml` `png` `jpg` `jpeg` `tiff` `webp` `bmp` |

`mineru` 和 `docling` 是外部内容提取引擎。启用相关规则时，必须在服务启动前配置对应 endpoint，例如：

```bash
MINERU_ENDPOINT=http://localhost:8000/api/v1/task
DOCLING_ENDPOINT=http://localhost:8081/v1/convert/file/async
```

`legacy` 内容提取引擎抽取的内容为 `raw` 格式，即仅保存在 `full_docs` 存储的 `content` 字段。`native` / `mineru` / `docling` 内容提取引擎抽取的内容格式为 `raw` + `lightrag` 双格式，完整的内容以文件形式保存在 sidecar 目录，纯文本内容同时保存在 `content` 字段。

### 2.5 文件处理选项

处理选项控制单个文件在多模态分析、知识图谱构建和文本分块上的行为。所有选项都是可选的；缺省值见下表。同一文件最多指定一种分块方式（F/R/V/P），其它选项可任意组合。

| 选项 | 类型 | 默认 | 含义 |
| --- | --- | --- | --- |
| `i` | 多模态 | 关闭 | 启用图像分析（VLM） |
| `t` | 多模态 | 关闭 | 启用表格分析（VLM） |
| `e` | 多模态 | 关闭 | 启用公式分析（VLM） |
| `!` | 流水线 | 关闭 | 禁止实体/关系抽取，不构建知识图谱（仅保留 chunks 向量索引，naive / mix 检索仍可用） |
| `F` | 分块 | 默认 | Fix/固定长度分块：遗留方法, 按固定Token长度或按分隔符机械分割（按分隔符分割时文本块不会出现重叠） |
| `R` | 分块 | - | Recursive/递归字符分块(RecursiveCharacterTextSplitter@LangChain)：接收一个分隔符列表（默认是 `["\n\n","\n","。","！","？","；","，"," ",""]`，按从语义最强到最弱排列）。优先按段落（双换行符）切分；如果切出的块依然超过 Token 限制，逐级降级使用单换行符 → 中文句末标点（`。！？`）→ 中文句中标点（`；，`）→ 空格 → 逐字符切分。**默认 cascade 包含中文标点**，使中文 / 中英混合文档能在语义边界切分。英文 `.?!` 故意排除（字面量匹配会误切 `0.95` / `e.g.`）。 |
| `V` | 分块 | - | Vector/向量语义分块(SemanticChunker@LangChain)：首先按句子拆分文本（默认句子切分正则同时识别英文 `.?!` 与中文 `。？！`，使中文 / 中英混合文档能正确切句），计算相邻句子的 Embedding，然后根据指定的阈值策略（如百分位 percentile、标准差 standard_deviation 或四分位距 interquartile）寻找语义断层进行切分。`SemanticChunker` 本身没有 chunk size 上限——任何超过 `chunk_token_size` 的语义块在落库前会自动通过 R 二次切分（保留 V 的非重叠语义）。此分块策略不会出现文本块重叠的情况。 |
| `P` | 分块 | - | Paragraph/段落语义分块（native）；优先按标题分割，严格避免上一标题底部内容与下一个标题内容混合破坏语义。适合对能够准确识别标题且标题结构清晰的文档进行分块。同一标题下的超长正文 fallback 到 R 时允许按 `CHUNK_P_OVERLAP_SIZE` 保留重叠；相邻大表格之间的桥接文字也可按该预算重复进入前后表格块。此分块方法只能运用在保存在 sidecar 目录的 `lightrag` 内容。如果 `lightrag` 内容不存在，将退化为使用 `R` 方法进行文本分块。此分块方法出现文本块重叠的情况远少于 `R策略` 和 `F策略`。 |

> 多模态全局开关 `addon_params["enable_multimodal_pipeline"]` 已废弃，相关行为统一由文件级 `i/t/e` 选项控制。详见[附录 A](#附录-a从旧版升级的注意事项)。

#### 选项生效阶段

处理选项的不同字符在流水线的不同阶段生效：

| 选项 | 作用阶段 | 说明 |
| :-: | --- | --- |
| i/t/e | Analyzing多模态分析 | 决定是否对 sidecar 中的图像 / 表格 / 公式调用 VLM 做摘要分析。**抽取阶段不受影响**：内容提取引擎按文档实际内容输出 `drawings.json` / `tables.json` / `equations.json` sidecar 文件。这样后续仅修改 `i`/`t`/`e` 选项触发"再分析"即可补做 VLM，无须重新解析原始文件。 |
| ! | Extraction实体关系抽取 | 跳过实体/关系抽取与图谱写入；chunks 仍写入向量库以保留 naive / mix 检索能力。 |
| F/R/V/P | Chunking文本分块 | 决定使用哪种分块策略；对解析阶段输出无影响。 |

> 模态可用性以"sidecar 文件是否存在"为唯一信号，内容提取引擎不需要在 meta 中声明能力。某文档若没有任何图像/表格/公式，对应 sidecar 不会写入；用户即使开启了 `i/t/e`，对应模态也只会被静默跳过，但 `analyze_multimodal` 会在该篇文档落一行 INFO 级日志（`[analyze_multimodal] process_options opted into i:drawings ... but the parser produced no such sidecar`），便于排查"VLM 为何没跑"。这种情况不会报错。

### 2.6 校验、优先级与回退

- 启动时会严格校验 `LIGHTRAG_PARSER`：未知内容提取引擎、错误后缀写法、显式使用不支持的后缀、外部引擎缺少 endpoint、处理选项中的非法字符都会导致启动失败。
- **通配符规则匹配某后缀时**，引擎需通过两道可用性检查（见 `parser_routing._engine_is_usable`）：(a) 该引擎能力表支持此后缀；(b) 若是外部引擎（`mineru` / `docling`），对应 endpoint 环境变量已配置。任一检查不过，本规则跳过，继续匹配下一条规则。例如 `*:mineru;html:docling` 中：MinerU 不支持 `html` 后缀（条件 a 不过），`html` 继续命中 `docling`；如果未设置 `MINERU_ENDPOINT`，所有 PDF 也会跳过 `*:mineru` 落到下一条规则（条件 b 不过）。这一行为对 `LIGHTRAG_PARSER` 规则匹配和文件名 hint 引擎选择都生效。
- 文件名 hint 的优先级高于 `LIGHTRAG_PARSER`。如果 hint 指定的引擎不支持该后缀，系统会回退到默认规则继续选择可用引擎。
- 如果文件名 hint 提供了非空选项串，则以 hint 为准；否则使用 `LIGHTRAG_PARSER` 规则中匹配项的默认选项；都没有则使用全部默认。
- 如果所有规则都不可用，文件内容提取方式会回退到 `legacy`；如果 `legacy` 也不支持对应的文件后缀，会向系统添加一个错误条目，上传文件保留在 `INPUT` 目录。
- F/R/V/P至多出现一个；同一选项重复时只生效一次但不报错。
- 大小写敏感：分块选项 F/R/V/P必须大写；其它选项 i/t/e小写。
- 中括号内出现非法字符时，整个 hint 失效，引擎按默认规则解析，选项按 `LIGHTRAG_PARSER` 默认或全部默认；同时落日志 warning。
- `P` 仅对 `native` 抽取出的 LightRAG Document 结构化结果有效；对 `legacy` 路径或非结构化输出会自动降级到 `R` 并记录 warning。

## 三、分块器参数配置（chunk_options）

### 3.1 process_options vs chunk_options 的职责

`process_options` 选**用哪种**分块策略（F/R/V/P），`chunk_options` 决定那一路分块器**用哪些参数**。两者职责正交：前者是单字符 selector，后者是结构化字典。

```
env vars                                                  (启动期一次性读取)
   │
   ▼
addon_params["chunker"]                                   (LightRAG 实例字段，由 env 与 legacy 兜底填入)
   │
   ▼  resolve_chunk_options(addon_params, split_by_character=…, split_by_character_only=…)
   │
full_docs[doc_id]["chunk_options"]                       (入队时冻结，每文件独立快照)
   │
   ▼
chunker(tokenizer, content, chunk_token_size, **strategy_kwargs)   (分块时按 selector 派发)
```

- **env vars** 在 `LightRAG.__init__` 阶段（由 `default_chunker_config()` 读取 strategy 特定 env，再由 `_apply_chunk_size_overlay` 兜底 legacy env）灌进 `addon_params["chunker"]`。
- **`addon_params["chunker"]`** 是 `ObservableAddonParams` 字段；Server 部署只需通过 env / 重启即可让新值生效。若需要在 Python 进程内运行时改它（不重启）以及 per-file 覆盖，请见[第八章 Python SDK 调用](#八python-sdk-调用)。
- **`full_docs.chunk_options`** 在 `apipeline_enqueue_documents` 入队时冻结：默认由 `resolve_chunk_options(self.addon_params, ...)` 现场拼装；若调用方传入 `chunk_options` 参数则原样持久化（SDK 用法，见 §8.4）。
- **分块器调用**从 `full_docs.chunk_options` 取对应子字典，按 `process_options.chunking` selector 派发到 F/R/V/P。

### 3.2 环境变量

下表所有变量在 `LightRAG` 实例化时一次性读入 `addon_params["chunker"]`：strategy 特定 env 由 `default_chunker_config()` 读取，legacy env (`CHUNK_SIZE` / `CHUNK_OVERLAP_SIZE`) 由 `_apply_chunk_size_overlay` 在 strategy env 与 legacy 构造字段都没填的槽位上兜底。修改 env 后需要重启服务（或新建 `LightRAG` 实例）才生效；已入队的文档持有冻结快照不受影响。

| 变量 | 默认 | 类型 | 作用域 |
|---|---|---|---|
| `CHUNK_SIZE` | `1200` | int | legacy 顶层 `chunk_token_size` 兜底；优先级低于 strategy 特定 env 与 SDK 路径设置的 `addon_params["chunker"]["chunk_token_size"]` |
| `CHUNK_OVERLAP_SIZE` | `100` | int | legacy overlap 兜底；当某 strategy 既无特定 env (`CHUNK_F_OVERLAP_SIZE` / `CHUNK_R_OVERLAP_SIZE` / `CHUNK_P_OVERLAP_SIZE`) 又无 SDK 路径的 `LightRAG(chunk_overlap_token_size=…)` 时填入 |
| `CHUNK_F_OVERLAP_SIZE` | 未设 | int | F strategy 特定 overlap；高于 legacy 构造字段与 `CHUNK_OVERLAP_SIZE` |
| `CHUNK_F_SPLIT_BY_CHARACTER` | （未设 = `null`） | str? | F 预切分隔符；`null` / 空串 = 仅按 token 窗 |
| `CHUNK_F_SPLIT_BY_CHARACTER_ONLY` | `false` | bool | F 严格模式：不二次按 token 切，超长抛错 |
| `CHUNK_R_SIZE` | 未设 | int | R strategy 特定 `chunk_token_size`；高于顶层 legacy 兜底（`CHUNK_SIZE` 与 SDK 路径的 `LightRAG(chunk_token_size=…)`）。未设时 R 沿用顶层解析结果 |
| `CHUNK_R_OVERLAP_SIZE` | 未设 | int | R strategy 特定 overlap；高于 legacy 构造字段与 `CHUNK_OVERLAP_SIZE` |
| `CHUNK_R_SEPARATORS` | `["\n\n","\n","。","！","？","；","，"," ",""]` | JSON 数组字符串 | R 分隔符级联，按从语义最强到最弱排列。默认包含中文句末（`。！？`）和句中（`；，`）标点，使中文 / 中英混合文档能在语义边界切分。英文 `.?!` 故意排除（字面量匹配会误切数字与缩写） |
| `CHUNK_V_SIZE` | 未设 | int | V strategy 特定 `chunk_token_size`（hard cap，超过时自动通过 R 二次切分）；高于顶层 legacy 兜底。未设时 V 沿用顶层解析结果 |
| `CHUNK_V_BREAKPOINT_THRESHOLD_TYPE` | `percentile` | str | V 阈值类型；可选 `percentile` / `standard_deviation` / `interquartile` / `gradient` |
| `CHUNK_V_BREAKPOINT_THRESHOLD_AMOUNT` | （未设 = `null`） | float? | V 阈值大小；`null` 让 LangChain 按类型自选默认（如 percentile=95） |
| `CHUNK_V_BUFFER_SIZE` | `1` | int | V 句子缓冲窗，距离计算时合并的相邻句数 |
| `CHUNK_V_SENTENCE_SPLIT_REGEX` | `(?<=[.?!])\s+\|(?<=[。？！])` | str | V 的句子切分正则，喂给 LangChain `SemanticChunker`。默认同时识别英文 `.?!`（要求后接空白，避免误切 `0.95`）和中文 `。？！`（不要求空白，适应中文连写）。env 值为原始正则字符串，无需 JSON 引号 |
| `CHUNK_P_SIZE` | 未设 | int | P strategy 特定 `chunk_token_size`；高于顶层 legacy 兜底（`CHUNK_SIZE` 与 SDK 路径的 `LightRAG(chunk_token_size=…)`）。未设时 P 沿用顶层解析结果 |
| `CHUNK_P_OVERLAP_SIZE` | 未设 | int | P strategy 特定 overlap；高于 legacy 构造字段与 `CHUNK_OVERLAP_SIZE`。用于同一 JSONL content 行内长正文 fallback 到 R 时的文本重叠，以及相邻大表格之间桥接文字复制到前后表格块的单侧预算 |

P 的内部比例常量是算法刻度，会随 `chunk_token_size` 自动按比例推导；通过设置 `CHUNK_P_SIZE` 可让 P 用独立的 `chunk_token_size`，避免共享 `CHUNK_SIZE` 时被其他策略的偏好（如 F 偏小）拖累。`CHUNK_P_OVERLAP_SIZE` 只影响 P 内部普通文本 fallback 与表格桥接上下文，不会让表格行级切片互相重叠。`CHUNK_R_SIZE` / `CHUNK_V_SIZE` 同理：R 偏向较小目标利于句段切分，V 作为 advisory ceiling 通常希望放大以减少过度拆分。

### 3.3 优先级链

每个分块槽位的最终值按 specificity-ordered 链解析（高 → 低）：

1. **`addon_params["chunker"]` 显式值** —— 通过 SDK 路径运行时设置或在构造时显式写入的字段值（见 §8.3）。Server-only 部署通常不会出现这一档。最直接，赢一切。
2. **strategy 特定 env** —— 如 `CHUNK_F_OVERLAP_SIZE` / `CHUNK_R_OVERLAP_SIZE` / `CHUNK_P_OVERLAP_SIZE` / `CHUNK_R_SIZE` / `CHUNK_V_SIZE` / `CHUNK_P_SIZE`（尚无 strategy 特定的 `CHUNK_F_SIZE`，F 复用顶层 `chunk_token_size`）。仅当槽位未被 ① 显式占用时填入。
3. **legacy 构造字段** —— `LightRAG(chunk_token_size=…, chunk_overlap_token_size=…)`，仅 SDK 路径生效，详见 §8.2。strategy 无关，"粗粒度缺省"，只填仍空的槽位。
4. **legacy env** —— `CHUNK_SIZE` / `CHUNK_OVERLAP_SIZE`。最终回退。

举例：`CHUNK_R_OVERLAP_SIZE=42` + `LightRAG(chunk_overlap_token_size=2)` → R 子字典 `chunk_overlap_token_size=42`（strategy env 胜出），F / P 子字典 `chunk_overlap_token_size=2`（无 F / P 特定 env，legacy 构造字段填入）。

三层语义保证：

1. **复现性**：env 改了，重启后老文档仍按入队那一刻的快照分块，结果不变。
2. **续跑一致性**：续跑分支 B（内容已抽取，按当前 `process_options` 重做分块）读的也是 `full_docs.chunk_options`，避免 env 漂移破坏一致性。
3. **per-file 个性化**：调用方可以为每个文件传不同的 `chunk_options`（典型用法：管理 UI 单独配置某个文件的 separators 或 V 阈值）。这是 SDK 路径的入参语义，详见 §8.4。

### 3.4 字段结构

```jsonc
{
  "chunk_token_size": 1200,                                   // 通用 token 上限
  "fixed_token": {                                            // F 专属
    "chunk_overlap_token_size": 100,
    "split_by_character": null,
    "split_by_character_only": false
  },
  "recursive_character": {                                    // R 专属
    "chunk_token_size": 1200,                                 // 可选；不写沿用顶层 chunk_token_size
    "chunk_overlap_token_size": 100,
    "separators": ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]   // 默认 cascade 含中文标点
  },
  "semantic_vector": {                                        // V 专属
    "chunk_token_size": 1200,                                 // 可选 hard cap；超过时通过 R 二次切分
    "breakpoint_threshold_type": "percentile",                // percentile | standard_deviation | interquartile | gradient
    "breakpoint_threshold_amount": null,                      // null = LangChain 默认
    "buffer_size": 1,
    "sentence_split_regex": "(?<=[.?!])\\s+|(?<=[。？！])"      // 默认正则兼容中英文句末标点
  },
  "paragraph_semantic": {                                     // P 专属
    "chunk_token_size": 3000,                                 // 不写沿用通用设置，建议设置大一些的分块大小
    "chunk_overlap_token_size": 100                           // 不写沿用 legacy overlap 解析链
  }
}
```

各子字典与对应分块器函数的 keyword-only 参数一一对应；新增参数时无需改 dispatcher，只在 chunker 函数添加 kwarg 即可。

### 3.5 缺失兼容

老文档入队时还没有 `chunk_options` 字段；分块时 dispatcher 会调用 `resolve_chunk_options(self.addon_params, …)` 兜底现场拼装。建议在升级后通过 reprocess 一次让老文档拿到 `chunk_options` 快照。

## 四、存储与目录布局

### 4.1 `full_docs` 字段

文件入队和抽取结果会写入 `full_docs`：

| 字段 | 说明 |
| --- | --- |
| `file_path` | 文件名 basename（不含目录），**保留用户提供的原始名（含中括号 hint）**，例如 `abc.[native-iet].docx` 原样写入。未提供有效来源时保存为 `unknown_source`。文件名 hint 不会被剥离，方便管理 UI 直接展示用户原本的命名意图。 |
| `canonical_basename` | 去掉处理提示 hint 后的规范化 basename（例如 `abc.docx`）。文件名查重以此字段为索引 key，保证 `abc.docx` 与 `abc.[native-iet].docx` 视为同一逻辑文档。 |
| `source_path` | 入队时提供的原始路径（仅当含目录分隔符或绝对路径时才写入），供 `native` / `mineru` / `docling` 解析器定位真实文件位置。 |
| `parse_format` | 内容格式：`pending_parse`, `raw`, `lightrag`。 |
| `content` | `raw` 时保存抽取文本；`pending_parse` 时为空字符串；`lightrag` 时存储以 `{{LRdoc}}` 开头的**完整合并文本**（拼接 `.blocks.jsonl` 中所有 `type=="content"` 行的 body 段），分块阶段 `parse_native` 会剥离前缀后再交给 chunking_func，与 `raw` 走完全相同的代码路径。 |
| `content_hash` | 内容 MD5，用于跨文件名查重。`parse_format=raw` 取 `sanitize_text_for_encoding` 后文本的 hash；`parse_format=lightrag` 取 `*.blocks.jsonl` 文件 hash；`parse_format=pending_parse` 不写入，待抽取完成后补上。 |
| `lightrag_document_path` | `parse_format=lightrag` 时保存结构化 LightRAG Document 的路径；新记录优先保存为相对 `INPUT_DIR` 的路径，例如 `__parsed__/report.docx.parsed/report.blocks.jsonl`。注意路径中的子目录与 blocks 文件名都使用规范化 basename（不含 hint）。 |
| `parse_engine` | 实际完成抽取的引擎：`legacy`, `native`, `mineru`, `docling`。对于待抽取文件，也可暂存目标引擎。 |
| `process_options` | 入队时记录的原始处理选项串（不含引擎名和分隔 `-`），例如 `"iet"`、`"R!"`、`""`。下游各阶段以此字段为权威源，决定是否启用图像/表格/公式分析（`i/t/e`）、是否禁止知识图谱构建（`!`）以及分块方式（`F/R/V/P`）。空字符串等价于全部默认值。 |
| `chunk_options` | 入队时**冻结**的分块器参数快照（嵌套字典）。由 SDK 路径调用方传入或由 `resolve_chunk_options(self.addon_params, ...)` 从实例字段（含 env 默认）兜底（见 §3.1）。`process_options` 选哪种分块策略（F/R/V/P），`chunk_options` 决定那一路分块器使用哪些参数。下游 `process_single_document` 在分块前从此字段读取专属 kwargs；持久化保证 env 变化、续跑、重启后老文档行为可复现。 |

`pending_parse` 表示文件已经入队，但还没有完成抽取。抽取成功后会改写为 `raw` 或 `lightrag`，并补齐 `content_hash`。抽取失败时保留 `pending_parse` 和空 `content`，便于后续排查和重试。

> `doc_status` 中也同步保存原始 `file_path`（含 hint）、`canonical_basename` 与 `content_hash`，作为 `get_doc_by_file_basename` / `get_doc_by_content_hash` 的查重索引来源。`get_doc_by_file_basename` 内部把传入参数先经 `canonicalize_parser_hinted_basename` 规范化后再与 `canonical_basename` 比对，因此 `abc.docx` 与 `abc.[native-iet].docx` 总是命中同一文档。
> `process_options` 同时镜像写入 `doc_status.metadata["process_options"]`，便于管理 UI 直接展示当前文件的处理策略。

### 4.2 `__parsed__` 目录结构

`__parsed__` 是输入目录旁的归档与分析结果目录。它同时保存已经处理过的原始文档，以及结构化解析产生的 LightRAG Document （lightrag格式）的文件和图片等资源。

- 原始文件归档：`legacy` 本地抽取成功并入队后，原文件会移动到同级 `__parsed__` 目录；`native` / `mineru` / `docling` 会先保留原文件供 pipeline 解析，解析成功并写入 `full_docs` 后再移动到 `__parsed__`。**归档时保留原始文件名（含 `[hint]`）**，例如 `report.[native-iet].docx` 归档为 `__parsed__/report.[native-iet].docx`，便于追溯用户最初的命名与处理选项。
- 分析结果目录：结构化解析结果会写入以**规范化文件名**（去掉 `[hint]`）加 `.parsed` 后缀命名的子目录，避免与归档原文件同名冲突，并保证当文件名 hint 或处理选项变化时同一逻辑文档继续指向同一目录。例如 `report.docx`、`report.[native].docx`、`report.[native-iet].docx` 的分析结果都写入 `__parsed__/report.docx.parsed/`。
- 分析结果文件：LightRAG Document blocks 文件以及 sidecar 都使用规范化文件名的主干命名，例如 `__parsed__/report.docx.parsed/report.blocks.jsonl`；同一目录下还可能包含 `report.tables.json`、`report.drawings.json`、`report.equations.json` 和 `report.blocks.assets/` 图片资源目录。**sidecar 是否生成由文档内容决定**：解析器只在文档实际包含表格/图片/公式时写出对应文件。这是模态可用性的唯一信号 —— 引擎不需要在 meta 中声明能力。`i`/`t`/`e` 选项只决定下一阶段是否对已存在的 sidecar 调用 VLM 做摘要分析。
- 解析失败时，原文件不会移动，便于修复配置后重新处理。
- `/documents/scan` 扫描到同名且已 `PROCESSED` 的文件时，该输入文件会被视为已处理并移动到 `__parsed__`，不会作为新文档入队。
- `/documents/scan` 同一次扫描中发现多个规范化后同名的文件时，会优先保留带支持引擎 hint 的文件以尊重用户的引擎选择；如果没有任何变体带 hint，则按排序处理第一个文件。其余变体会输出 warning 并移动到 `__parsed__`，避免同批文件互相覆盖。例如 `abc.docx` 和 `abc.[native].docx` 同时存在时只会处理 `abc.[native].docx`。
- 扫描或解析过程中发现内容 hash 重复时，该输入文件同样会移动到 `__parsed__`；本次 `doc_status` 保留为 `FAILED duplicate` 以便追踪。
- 移动文件只作用于当前输入文件，不会覆盖或移动既有文档源文件。若目标目录已存在同名文件，系统会自动追加 `_001`、`_002` 等编号，例如 `report.pdf` 会依次归档为 `report_001.pdf`、`report_002.pdf`。若分析结果目录名已被普通文件占用，也会追加编号，例如 `report.docx.parsed_001/`。

## 五、文档重复判定规则

文件上传、文件解析入队和文本接口会按照「文件名 + 内容 hash」两道关卡判断是否重复，命中任一即视为重复并写入一条 `FAILED` 记录，不会覆盖已有的 `full_docs`。`/documents/scan` 目录扫描也使用同一套索引，但为了便于自动重试未完成文件，对文件名重复有单独的归档与重处理规则。

### 5.1 文件名（basename）查重

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
- SDK 路径调用 `insert` / `ainsert` / `apipeline_enqueue_documents` 时不传 `file_paths` 是被允许的，相关行为详见 §8.4。这类无来源文档的 `file_path` 保存为 `unknown_source`。
- 空字符串、`no-file-path` 和 `unknown_source` 都会被视为未知来源；它们不会阻止新的无来源文本入队，也不会作为同名文件互相去重。

存储后端通过 `get_doc_by_file_basename` 提供 basename 直查能力，内部按 `canonical_basename` 字段比对（传入参数会先经 `canonicalize_parser_hinted_basename` 规范化）。`JsonDocStatusStorage` 已经实现了内存级遍历；其它后端目前回落到默认实现（扫描全部状态后比对 `canonical_basename`），将在后续 PR 中补齐原生索引。

### 5.2 内容 hash 查重

- 文件名不同但抽取后的内容完全相同的文档同样视为重复。这里的 hash 是按配置的抽取引擎得到最终文本或 LightRAG Document 后计算的内容 hash，不是原始文件字节 hash。
- `full_docs` 与 `doc_status` 会按内容格式写入或补齐 `content_hash` 字段：
  - `parse_format=raw`：取经过 `sanitize_text_for_encoding` 之后的文本 MD5。
  - `parse_format=lightrag`：取 `lightrag_document_path` 解析出的 `*.blocks.jsonl` 文件 MD5。相对路径按 `INPUT_DIR` 解析。
  - `parse_format=pending_parse`：暂不写入 hash，等到真正完成解析后由后续步骤补上（避免按空内容误判）。
- `legacy` 路径会在本地提取文本后、入队时进行内容 hash 查重；命中重复时，本次记录写为 `FAILED duplicate`，不会生成新的 `full_docs`、chunks 或图数据。
- `native` / `mineru` / `docling` 路径会先以 `pending_parse` 入队；真正完成解析并补齐 `content_hash` 后，如果发现其它文档已有相同 hash，本次记录会在进入分析、切块、实体抽取和图写入前停止。
- 重复记录会在 `metadata.duplicate_kind` 中标记为 `filename` 或 `content_hash`，便于排查。内容 hash 重复还会记录 `metadata.is_duplicate=true`、`metadata.original_doc_id` 和 `metadata.original_track_id`；解析后才发现的重复会删除本次临时写入的 `full_docs`。
- 相关 warning 会尽量减少重复噪音：扫描发现已 `PROCESSED` 的同名文件时会写入日志和 pipeline status；入队阶段重复使用 LightRAG 层的 `Duplicate document detected (...)` 日志；解析完成后才发现的内容重复使用 `Duplicate content skipped after parsing`，并写入 pipeline status。扫描归档不会额外输出 `[File Extraction]Duplicate skipped`。
- 存储后端通过 `get_doc_by_content_hash` 进行 hash 直查；命名约定与 `get_doc_by_file_basename` 一致。

> 入队批次内（同一次 `apipeline_enqueue_documents` 调用）也会做 basename 与 content_hash 去重，命中时把后续条目直接写为 `FAILED` 并标记 `existing_status=batch_duplicate`。其中 basename 去重只对有效文件名生效；`unknown_source`、`no-file-path` 和空来源只参与内容 hash 去重。
>
> **跨调用并发去重**也由 workspace 级串行锁保证（详见 [§6.7 enqueue 串行锁（防并发去重穿透）](#67-enqueue-串行锁防并发去重穿透)）：两次相同内容、不同文件名的并发入队不会双双穿透 `content_hash` 检查。

## 六、并发与重入约束

为防止 `scan` / `upload` / `insert` 与运行中的流水线相互覆盖 `doc_status` / `full_docs` 记录，所有写入入口在 `pipeline_status` 共享字典上协调。同一 workspace 下的 `pipeline_status_lock` 保证下表所有 transition 都在锁内原子完成。

### 6.1 `pipeline_status` 字段

| 字段 | 语义 |
| --- | --- |
| `busy` | 流水线繁忙的笼统标志。处理循环和破坏性作业（clear/delete）都会设它。**仅有 `busy=True`（处理循环）不阻塞 enqueue**——循环按 batch 拉取 `doc_status` 快照处理，每批结束后通过 `request_pending` 检查是否还有新工作。 |
| `destructive_busy` | `busy` 的破坏性子集：`/documents/clear` 或 `/documents/{doc_id}`（删除）正在 drop 存储 / 删源文件。reservation 和 enqueue last-line guard 都会拒绝——并发 enqueue 会写入正被 drop 的存储，已接受的文档会静默丢失。处理循环不会设此字段。 |
| `scanning` | `/documents/scan` 后台任务运行中（整个生命周期：分类阶段 + 处理阶段）。仅 `/scan` 端点用它拒绝重叠 scan，本身**不**阻塞 upload/insert。 |
| `scanning_exclusive` | `scanning` 的独占子集：只在 scan 的**分类阶段**为 True——run_scanning_process 在读 doc_status 分类（已处理 / 续跑 / 删 stub / 归档），不能与并发写者交错。reservation 和 enqueue last-line guard 都会拒绝。分类完成后会立即清旗，scan 进入处理阶段后允许并发 upload。 |
| `pending_enqueues` | 已通过 `_reserve_enqueue_slot` 但 bg task 未完成的 upload/insert 数。仅给 scan 端点参考——决定是否能拿独占。bg task 在 `finally` 里释放 slot。 |
| `request_pending` | 让运行中的处理循环再扫一轮的信号。enqueue 在 `busy=True` 时写完 `doc_status` 后置位；处理循环每个 batch 结束后检查并重新拉快照。 |

### 6.2 入口行为

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

### 6.3 为什么 `busy` 不再阻塞 enqueue

旧版本里 `busy=True` 一律拒绝任何新入队，理由是"修改 `doc_status` 会与流水线工作线程交错"。但实际上：

1. **写入顺序保证一致性**：`apipeline_enqueue_documents` 总是先 upsert `full_docs`、再 upsert `doc_status`。处理循环开头的 consistency check 仅删除"`doc_status` 行没有对应 `full_docs`"的孤儿——这种状态在并发 enqueue 中不可能出现。
2. **批次级快照**：处理循环每个 batch 拉一次 `get_docs_by_statuses` 快照，新写入的 `PENDING` 行不会破坏当前 batch；下一轮通过 `request_pending` 重拉快照即可看到新工作。
3. **`request_pending` 设计本就为此**：旧版同时存在 `request_pending` 字段——它就是为"运行中又有新工作"设计的，但被 busy 守护堵死了。

新契约把这个机制启用起来后，**用户在长批次处理过程中仍可继续上传新文档**，bg task 写完 `doc_status` 后由运行中的循环自动接管。

### 6.4 为什么 scan 仍是独占写者

scan 不仅 enqueue 自己扫到的新文件，还会读 `doc_status` 决定每个文件去向：

- 同名 `PROCESSED` 行 → 归档源文件、跳过入队。
- 同名非 PROCESSED 且 `full_docs` 存在 → resume 路径，源文件**保留在 `INPUT/`**，不归档（pending-parse 解析器仍可能需要它），由处理循环按状态查询接走。
- 同名 `FAILED` 且 `full_docs` 缺失 → 识别为之前 `apipeline_enqueue_error_documents` 写下的提取错误 stub（一致性检查会保留这种行供人工 review），scan 自动删除该 stub 并把当前文件按新文件重新入队，让用户"修好源文件再 scan 一次"能直接生效。

这些"读—决策—写"组合不能与其它写者交错，否则分类决策会基于过期视图。所以 scan 必须独占，且 scan 端点会在 `busy` / `scanning` / `pending_enqueues>0` 任一存在时拒绝。

### 6.5 严格名字预检（upload 路径）

upload 通过 reservation 后、保存文件前必须双道检查：

1. **INPUT 目录扫描**：把要保存的 basename 经 `canonicalize_parser_hinted_basename` 规范化，遍历 INPUT 目录里现有任何同 canonical 变体（含 hint / 不含 hint），命中即 409。
2. **doc_status 查重**：用规范化 basename 调 `get_existing_doc_by_file_basename`，命中即 409。

两道都过 → 保存文件 → schedule bg task → bg task 调 `apipeline_enqueue_documents` 写库 + 调 `apipeline_process_enqueue_documents` 触发处理。

> 旧版本曾允许 upload 在已有同名记录时悄悄写入 FAILED 重复条目；新规则改为 fail-fast，不在 doc_status 留下任何重复痕迹。如需替换同名文档，请先调用 `/documents/{doc_id}` 的删除接口。

### 6.6 多 reservation 并发的协调

两个 upload 同时进来时（scan 此时拿不到独占）：

1. A `_reserve_enqueue_slot` → `pending_enqueues=1`，写文件，schedule bg task A，返回 success。
2. B `_reserve_enqueue_slot` → `pending_enqueues=2`，写文件，schedule bg task B，返回 success。
3. bg task A `apipeline_enqueue_documents` → 写 `doc_status` → 调 `apipeline_process_enqueue_documents` → 设 `busy=True` 处理 A 的文档。
4. bg task B `apipeline_enqueue_documents` → 看到 `scanning=False`，正常写入；写完后看到 `busy=True`，自动设 `request_pending=True`。
5. bg task B 调 `apipeline_process_enqueue_documents` → 看到 `busy=True`，设 `request_pending=True` 立即返回。
6. A 的处理循环跑完当前 batch，看到 `request_pending=True`，重拉快照，把 B 的 `PENDING` 行接上处理。
7. 全部完成后 `busy=False`、`pending_enqueues=0`。

任何一个 bg task 都不会因为 busy 被误拒——因为 enqueue 不再检查 busy；处理循环也不会重复处理同一份 batch——`request_pending` 只在 batch 间生效，且每次重拉前清零。

### 6.7 enqueue 串行锁（防并发去重穿透）

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

## 七、流水线启动时的续跑规则

每次 `apipeline_process_enqueue_documents` 起步时，会拉取所有处于 `PARSING` / `ANALYZING` / `PROCESSING` / `PENDING` / `FAILED` 状态的文档继续处理。续跑路径**根据"内容是否已抽取"分流**，保证同一个文档无论之前进度如何，按当前 `process_options` 续跑都有幂等结果。

续跑规则只对 `doc_id` 已经存在于 `doc_status` 的文档生效。新文件入队需要"并发与重入约束"中的文件查重逻辑，避免新文件挤掉旧的已经成功提取内容的文件记录。

### 7.1 判断"内容已抽取"

读 `full_docs[doc_id]`：

| `parse_format` | 判定 |
| --- | --- |
| `lightrag` 且 `lightrag_document_path` 文件存在 | ✅ 已抽取 |
| `raw` 且 `content` 非空 | ✅ 已抽取 |
| 其它（含 `pending_parse`、记录缺失） | ❌ 未抽取 |

### 7.2 分支 A：未抽取

走完整流水线（`parse_native` / `parse_mineru` / `parse_docling` → `analyze_multimodal` → 分块 → 实体抽取），按 `full_docs.process_options` 决定每一阶段的行为。这是"首次入队"的常规流。

### 7.3 分支 B：已抽取

**一律跳过解析**（不重新调 `parse_*`），从 ANALYZING 阶段重启，并清光旧 chunks / entities 后按当前 `process_options` 重做：

| 子步骤 | 行为 |
| --- | --- |
| 引擎对比 | 若 `process_options` 隐含的引擎 ≠ `full_docs.parse_engine`，**仅 warn**，不重新解析。已抽取的内容是不可变事实，重新跑不同引擎会产生不一致。要切换引擎请先 delete 整个文档再重传。 |
| 旧 chunks / 实体 / 关系清理 | 读 `status_doc.chunks_list` 收集旧 chunk id 集，调 `_purge_doc_chunks_and_kg(doc_id, chunk_ids)`：从 `chunks_vdb` / `text_chunks` 删除 chunk 行；按 `entity_chunks` / `relation_chunks` 反查受影响的实体 / 关系，对失去全部源的条目直接从图谱与向量库删除，对仍有其它文档贡献的条目调 `rebuild_knowledge_from_chunks` 用剩余 chunks 重建；最后删除 `full_entities` / `full_relations` 中本 doc 的索引行。purge 完成后 `status_doc.chunks_list = []` / `chunks_count = 0` 重置，避免后续 state-machine upsert 写回旧 ID。 |
| `analyze_multimodal` | "是否已分析"的判定**仅由 sidecar item 的 `llm_analyze_result` 子字段决定**：已写有 `llm_analyze_result` 的 item 跳过；新启用模态对应 item 缺失该字段，从空状态分析。代码中无 `meta.analyze_time` 字段读写。 |
| 重新分块 | 按新 `process_options.chunking` 选策略，参数从 `full_docs.chunk_options` 读取（入队快照，不会因续跑被覆盖；env 改动后老文档仍按入队那一刻的参数分块）。LightRAG Document path 在 `process_options=P` 时走 paragraph_semantic，否则按 selector 分发到 F/R/V。 |
| 实体抽取 / KG-skip | 按新 `process_options.skip_kg` 决定 |

> 这条规则保证：用户改 `i/t/e` 重传同名文档（先删旧 doc 再上传带新 hint 的文件）时，多模态分析能增量补齐；改 `F/R/V/P` 时 chunks 与图谱重建；改 `!` 时停掉或恢复 KG 构建。引擎变更被视为"重大变更"，统一由 delete + 重传完成，不在续跑路径里隐式发生。

## 八、Python SDK 调用

本章针对**直接 import `LightRAG` 类**进行集成的开发者，覆盖 Server 部署不会用到的运行时 API、构造期参数和已移除的旧接口。Server 用户通常无须阅读本章。

### 8.1 适用对象

```python
from lightrag import LightRAG
rag = LightRAG(working_dir="./rag_storage", ...)
await rag.initialize_storages()
await rag.ainsert("text", file_paths="doc.pdf")
```

这种调用方式以下行为与 Server 路径不同：可在不重启进程的情况下改 `addon_params["chunker"]`，可向 `apipeline_enqueue_documents` 传入 per-file `chunk_options`，可在 `ainsert` 调用时动态覆盖 F 策略的预切分参数。

### 8.2 LightRAG 构造期参数

`LightRAG(chunk_token_size=…, chunk_overlap_token_size=…)` 是 §3.3 优先级链中的**第 3 档**："legacy 构造字段"。strategy 无关、粗粒度缺省，只填仍空的槽位：

- 优先级低于 `addon_params["chunker"]` 显式值（§8.3）和 strategy 特定 env（§3.2）。
- 优先级高于 legacy env `CHUNK_SIZE` / `CHUNK_OVERLAP_SIZE`。
- 实例字段 `self.chunk_token_size` / `self.chunk_overlap_token_size` 在 `__post_init__` 之后总会被回填为 `int`，方便仍读这两个字段的旧路径（如 `pipeline.py` 中 `chunk_opts.get("chunk_token_size") or self.chunk_token_size` 兜底）继续工作。

### 8.3 运行时改 `addon_params["chunker"]`

`addon_params["chunker"]` 是 `ObservableAddonParams` 字段，可以**运行时改**：

```python
rag.addon_params["chunker"]["recursive_character"]["separators"] = ["##", "\n", " "]
```

改完后，**后续入队**的文档拿到新默认；已入队文档保留入队时的快照不变（参见 §3.3 三层语义保证）。这是 §3.3 优先级链的第 1 档："`addon_params["chunker"]` 显式值"，赢一切。

Server 部署没有这个能力 —— 改 env 后必须重启服务才生效。

### 8.4 `apipeline_enqueue_documents(chunk_options=…)`

`apipeline_enqueue_documents` 接受可选的 `chunk_options` 参数，调用方传入 `dict` / `list[dict]` 即原样持久化到 `full_docs[doc_id]["chunk_options"]`；不传则由 `resolve_chunk_options(self.addon_params, ...)` 现场拼装一份。

典型用法：

```python
await rag.apipeline_enqueue_documents(
    input=["text A", "text B"],
    file_paths=["a.txt", "b.txt"],
    chunk_options=[
        {"chunk_token_size": 800, "recursive_character": {"separators": ["\n\n", "\n"]}},
        {"chunk_token_size": 1500},
    ],
)
```

per-file 个性化的典型场景：管理 UI 单独配置某个文件的 separators 或 V 阈值；将来上传 API 也可在 form / hint 中接收覆盖。

**不传 `file_paths` 的兼容**：核心 API `insert` / `ainsert` / `apipeline_enqueue_documents` 仍兼容未传 `file_paths` 的调用；这类文档的 `file_path` 会保存为 `unknown_source`，不会参与文件名查重，文档 ID 继续按文本内容生成。

`apipeline_enqueue_documents` 自身的并发约束（last-line guard、`from_scan=True` 旁路）见 §6.2 入口行为表。

### 8.5 `ainsert(split_by_character=…, split_by_character_only=…)`

`LightRAG.ainsert(split_by_character=…, split_by_character_only=…)` 的运行时参数在入队时由 `resolve_chunk_options` 覆写到 `chunk_options.fixed_token`：

- `split_by_character` 非 `None` 即覆盖 env 默认；
- `split_by_character_only=True` 即覆盖（`False` 是签名默认值，与"未指定"无法区分，所以 env 默认胜出）。

仅对 F 策略生效；其它策略的子字典不受影响。

### 8.6 已移除的 SDK 入参：`reprocess_existing_non_processed`

旧 `apipeline_enqueue_documents` 的 `reprocess_existing_non_processed=True` 行为会在 scan 时直接删除非 PROCESSED 的旧记录并重建，与 §五 / §六 的规则相冲突，已整段移除。替代路径：

- 自动续跑：scan 按 §6.4 的分类规则处理同名文件（归档 / 续跑 / 删 stub 后重入队），由 §七 续跑规则在处理循环里统一接管。
- 强制刷新：先调 `/documents/{doc_id}` 删旧文档，再上传同名新文件。

## 附录 A：从旧版升级的注意事项

### 多模态全局开关已废弃

`addon_params["enable_multimodal_pipeline"]` 已废弃，相关行为统一由文件级 `i/t/e` 选项控制（§2.5）。如启动配置仍包含该字段，会在日志输出 deprecation warning 并被忽略。
