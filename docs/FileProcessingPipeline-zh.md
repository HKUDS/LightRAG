# 文件处理流水线工作方式说明

从版本 v1.5.0 （目前在dev分支）开始，LightRAG的文件处理流水线进行了重大的升级：

* 支持多种文件内容抽引擎：legacy、native、mineru、docling
* 支持多种文本块分块方法：Fix、Recursive、Vector、Paragraph
* 支持对个别文件关闭实体关系抽取

LightRAG Server引入了一个文件处理的中间格式： `LightRAG Document` 。该格式支持表格和图片等多模态数据，同时包含文章的章节段落元数据，方便日后进行内容溯源。

本文以 **LightRAG Server** 的部署与使用视角组织：先给出快速开始可直接套用的配置，再展开内容抽取与分块的配置语法、存储 / 目录布局、去重、并发以及续跑规则。直接通过 Python 代码调用 `LightRAG` 类的开发者请翻到[第十一章 Python SDK 调用](#11-python-sdk-调用)。

## 目录

- [1. 快速开始](#1-快速开始)
- [2. 处理选项与配置语法](#2-处理选项与配置语法)
- [3. 文件解析引擎](#3-文件解析引擎)
- [4. 多模态分析（VLM）](#4-多模态分析vlm)
- [5. 分块器参数配置（chunk_options）](#5-分块器参数配置chunk_options)
- [6. 存储与目录布局](#6-存储与目录布局)
- [7. 同名与重复文档](#7-同名与重复文档)
- [8. 运行控制：边跑边传、停止与重试](#8-运行控制边跑边传停止与重试)
- [9. 流水线启动时的续跑规则](#9-流水线启动时的续跑规则)
- [10. 常见问题排查](#10-常见问题排查)
- [11. Python SDK 调用](#11-python-sdk-调用)
- [附录 A：从旧版升级的注意事项](#附录-a从旧版升级的注意事项)
- [附录 B：环境变量速查](#附录-b环境变量速查)
## 1. 快速开始

### 1.1 保持旧版文件处理行为

所有文件按旧版的文档解析和分块策略处理所有文档。不配置 `LIGHTRAG_PARSER` 或把它配置为如下值：

```bash
LIGHTRAG_PARSER=*:legacy-F
```

### 1.2 推荐起步文件处理行为

不依赖外部文档解析服务，不依赖`VLM`视觉模型。使用新版原生的 `Native` 解析 `docx` 文档，开启表格(t)和公式(e)的模态分析，搭配`P`分块策略；其余文档使用老版本的内容解析器，搭配效果更好的`R`分块策略。

```bash
LIGHTRAG_PARSER=*:native-teP,*:legacy-R
```

### 1.3 开启多模态处理能力

开启多模态处理能力需要依赖 `MinerU` 文件解析服务和 `VLM` 视觉识别模型。使用 `Native` 解释 `docx` 文件，使用 `MinerU` 解析 `pdf`、`office` 和各种图片文件。以上文件都开启图片(i)、表格(t)和公式(e)的模态分析，并并搭配`P`分块策略。其余文档回退到老版本的内容解析器并搭配`R`分块策略。

```bash
LIGHTRAG_PARSER=*:native-iteP,*:mineru-iteP,*:legacy-R
VLM_PROCESS_ENABLE=true
VLM_LLM_MODEL=kimi-k2.6
MINERU_API_MODE=local
MINERU_LOCAL_ENDPOINT=http://localhost:8000
```

> `P`分块策略是LightRAG原生的分块策略，详情请参阅[Paragraph Semantic 分块策略](ParagraphSemanticChunking-zh.md)。VLM的配资请参阅[基于角色的 LLM/VLM 配置指南](RoleSpecificLLMConfiguration-zh.md)


### 1.4 逐字段读懂一条预设

`LIGHTRAG_PARSER` 用逗号分隔的每一项都是一条路由规则，按文件后缀从左到右匹配，第一条可用的规则胜出。把 `*:native-iteP,*:legacy-R` 拆开看：

```
*  :  native  -  iteP
│     │          │
│     │          └─ 处理选项：i/t/e 分别开启图片、表格、公式分析；
│     │             P 选择段落语义分块器
│     └─ 内容抽取引擎
└─ 规则适用的后缀；`*` 匹配任意后缀
```

所以第一条规则的意思是"任何文件都交给 `native` 解析，三种模态全部分析，用 `P` 分块"，第二条是"第一条处理不了的，退回 `legacy` 并用 `R` 分块"。当引擎不支持该后缀、或外部引擎没有配置服务端点时，该规则会被跳过——这正是 `*:native-iteP` 实际只接管 `docx` / `md` / `textpack`、其余文件全部落到 `legacy` 的原因。

完整语法见 §2.3，选项字母见 §2.1，各引擎见 §3。

### 1.5 第一次运行：跑通一个文档并确认成功

一条不依赖任何外部服务的冒烟路径——一个 `.docx`，不需要 MinerU、docling、VLM：

```bash
LIGHTRAG_PARSER=*:native-teP,*:legacy-R
```

**1. 先验证路由，别浪费一次解析。** 写错的 `LIGHTRAG_PARSER` 在启动时就会失败（§2.7）。服务起来后，`GET /documents/supported_file_types` 返回**服务端实际解析出的**后缀白名单与引擎-后缀映射，不入库一个文件就能确认规则是否生效。

**2. 把文件送进来。** 要么 `POST /documents/upload`（文件存入 `INPUT_DIR`，响应带 `track_id`），要么把文件放进 `INPUT_DIR` 后调 `POST /documents/scan`。想对单个文件试不同配置又不改 `.env`，就用文件名 hint：`report.[native-teP].docx`（§2.5）。

**3. 看进度。** `GET /documents/track_status/{track_id}` 看这次上传，`GET /documents/pipeline_status` 看实时日志，`GET /documents/status_counts` 看整批。状态阶梯是 `PENDING → PARSING → ANALYZING → PROCESSING → PROCESSED`。

**4. 产物落在哪。** 全部在 `INPUT_DIR/__parsed__/` 下（§6.2）。以 `report.[native-teP].docx` 为例：

```
__parsed__/report.[native-teP].docx   # 归档的原件，保留 hint
__parsed__/report.docx.parsed/        # 规范名，已剥掉 hint
    report.blocks.jsonl               # 首行是 meta，其余是内容块
    report.tables.json                # 仅当文档确实有表格
    report.equations.json             # 仅当文档确实有公式
    report.blocks.assets/             # 导出的图片
```

**5. 验证解析**，按这个顺序：

1. `*.blocks.jsonl` 首行是 `meta` 记录。
2. 内容行的标题结构合理——这正是 `P` 的切分依据。
3. 你预期的 sidecar 存在。缺失意味着文档没有该类内容，或该引擎不产出它（`legacy` 一个都不产出，见 §3.2）。
4. 你启用了分析的那些 sidecar item 带有 `"llm_analyze_result": {"status": "success"}`。这里是 `skipped` 或 `failure`，就是"为什么我的图片/表格没进知识图谱"的答案（§4.3）。

**6. 验证分块与入库。** `GET /documents/paginated` 的 `chunks_count` 是分块数；doc_status 记录里的 `metadata.parse_engine` 与 `metadata.process_options` 表明**实际生效的**引擎和选项，这是发现"hint 被静默判为无效"最快的办法（§2.7）。

**失败了怎么办：** 源文件**留在 `INPUT_DIR`**，改完配置可以重新扫描。doc_status 记录上的 `error_msg` 是诊断入口，§10 列出了常见的几种。注意引擎与处理选项都在入队时就已冻结——改 `LIGHTRAG_PARSER` 或改 hint 都不影响已有记录，要用新配置重跑就得删除文档后重新上传（§9.3）。想脱离服务器离线复现一次解析，用 `python -m lightrag.parser.cli`（[ParserDebugCLI-zh.md](./ParserDebugCLI-zh.md)）。

## 2. 处理选项与配置语法

LightRAG 的文件处理配置由两部分合成：内容抽取引擎决定原始文件如何被解析，处理选项决定解析后是否执行多模态分析、使用哪种分块方式，以及是否构建知识图谱。通常先用环境变量 `LIGHTRAG_PARSER` 按文件后缀设置默认规则，再用文件名中的 `[hint]` 覆盖单个文件。引擎和选项可以写在同一个配置片段里，例如 `docx:native-iet` 或 `report.[native-R!].docx`。

为了向后兼容，在未修改配置的情况下，升级后的文件内容提取方式会维持原来的 `legacy` 行为。如需启用新的内容处理引擎，请按本章说明配置；各引擎自身的能力与设置见 §3。

### 2.1 文件处理选项

处理选项以文件为粒度控制多模态分析、知识图谱构建和文本分块的行为；既可在 `LIGHTRAG_PARSER` 中作为规则默认值批量设置（见 [§2.4](#24-默认规则lightrag_parser)），也可通过文件名 hint 对单个文件覆盖（见 [§2.5](#25-单文件覆盖文件名-hint)）。所有选项都是可选的；缺省值见下表。同一文件最多指定一种分块方式（F/R/V/P），其它选项可任意组合。

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

### 2.2 选项生效阶段

处理选项的不同字符在流水线的不同阶段生效：

| 选项 | 作用阶段 | 说明 |
| :-: | --- | --- |
| i/t/e | Analyzing多模态分析 | 决定是否对 sidecar 中的图像 / 表格 / 公式调用 VLM 做摘要分析。**抽取阶段不受影响**：内容提取引擎按文档实际内容输出 `drawings.json` / `tables.json` / `equations.json` sidecar 文件。这样后续仅修改 `i`/`t`/`e` 选项触发"再分析"即可补做 VLM，无须重新解析原始文件。 |
| ! | Extraction实体关系抽取 | 跳过实体/关系抽取与图谱写入；chunks 仍写入向量库以保留 naive / mix 检索能力。 |
| F/R/V/P | Chunking文本分块 | 决定使用哪种分块策略；对解析阶段输出无影响。 |

> 模态可用性以"sidecar 文件是否存在"为唯一信号，内容提取引擎不需要在 meta 中声明能力。某文档若没有任何图像/表格/公式，对应 sidecar 不会写入；用户即使开启了 `i/t/e`，对应模态也只会被静默跳过，但 `analyze_multimodal` 会在该篇文档落一行 INFO 级日志（`[analyze_multimodal] sidecar e:equations empty: doc—id ...`），便于排查"VLM 为何没跑"。这种情况不会报错。

### 2.3 配置语法总览

完整配置模型如下：

```text
LIGHTRAG_PARSER=后缀:引擎-选项,后缀:引擎,*:legacy-R
filename.[ENGINE].ext
filename.[ENGINE-OPTIONS].ext
filename.[-OPTIONS].ext
```

- `LIGHTRAG_PARSER` 是默认规则表，按文件后缀匹配，例如 `pdf:mineru`、`docx:native-iet`。
- 文件名 `[hint]` 是单文件覆盖规则，例如 `paper.[mineru].pdf`、`memo.[native-R!].docx`。
- `ENGINE` 是内容抽取引擎。选哪个决定了解析阶段**能**产出什么：

  | 引擎 | 是什么 | 什么时候用它 |
  | --- | --- | --- |
  | `legacy` | 纯文本抽取，不产出 sidecar | 想保持升级前的行为，或文档本来就是纯文本 |
  | `native` | 内置结构化抽取器，完全本地，不依赖外部服务 | `docx` / `md` / `textpack`，且希望不部署任何东西就用上 `P` 分块或模态分析 |
  | `mineru` | 外部 MinerU 服务 | PDF、扫描件，以及需要版面识别与 OCR 的 office / 图片格式 |
  | `docling` | 外部 docling-serve | MinerU 的替代方案；也是内置路径里唯一能出 LaTeX 公式的 |

  只有 `native`、`mineru`、`docling` 会产出 sidecar，因此也只有它们能支撑 `i` / `t` / `e` 选项与 `P` 分块器。各引擎的能力、配置与部署见 [§3](#3-文件解析引擎)。
- `OPTIONS` 是处理选项字符组合，例如 `iet`、`R!`、`P`。选项最终写入 `process_options`，由后续流水线阶段读取。
- `ENGINE-OPTIONS` 中的连字符只用于分隔引擎和选项，不属于选项本身。
- 仅指定处理选项时必须写成 `[-OPTIONS]`，例如 `[-!]`。无横线的 `[abc]` 会被严格解释为引擎名并报错，不会回退为选项串。

常见组合示例：

```bash
LIGHTRAG_PARSER=pdf:mineru-R,docx:native-ietP,*:legacy-R
MINERU_API_MODE=local
MINERU_LOCAL_ENDPOINT=http://localhost:8000
DOCLING_ENDPOINT=http://localhost:5001
```

```text
my-proposal.[native-iet].docx   # 使用 native 引擎，开启图、表、公式分析
my-memo.[native-R!].docx        # 使用 native 引擎，递归语义分块，禁止知识图谱构建
my-proposal.[-!].docx           # 使用默认引擎，仅禁止知识图谱构建
my-proposal.[mineru].docx       # 使用 MinerU 引擎，处理选项全部默认
```

### 2.4 默认规则：`LIGHTRAG_PARSER`

`LIGHTRAG_PARSER` 用来为不同文件后缀配置默认内容抽取引擎，也可以在引擎后追加该规则的默认处理选项：

```text
后缀:引擎,后缀:引擎,*:legacy
后缀:引擎;后缀:引擎;*:legacy
后缀:引擎-选项
```

- 左侧匹配的是文件后缀，不是完整文件名；应写 `pdf:mineru`，不要写 `*.pdf:mineru`。
- 规则使用分号 `;`（推荐）或英文逗号 `,` 分隔。
- 规则按从左到右的顺序检查；优先规则放在前面，通配符规则通常放在最后。
- 引擎后缀 `-选项` 部分作为该规则匹配文件的默认 `process_options`。例如 `LIGHTRAG_PARSER=docx:native-iet` 表示所有 `.docx` 默认采用 `native` 引擎，并开启图像、表格、公式分析。

### 2.5 单文件覆盖：文件名 hint

文件名中可以使用中括号临时指定单个文件的处理方式：

```text
paper.[mineru-R].pdf
slides.[docling].pptx
memo.[native-P].docx
notes.[-R].md
```

中括号内的内容支持三种形式：

```text
[ENGINE]              # 仅指定引擎，处理选项使用默认或 LIGHTRAG_PARSER 提供的默认
[ENGINE-OPTIONS]      # 同时指定引擎和处理选项
[-OPTIONS]            # 仅指定处理选项，引擎仍按 LIGHTRAG_PARSER / 默认规则解析
```

解析 hint 时，无横线内容必须整体匹配引擎名（`mineru` / `native` / `docling` / `legacy`）；带横线且横线前有内容时，横线前是引擎、横线后是选项；以横线开头时表示仅指定选项。旧式 `[OPTIONS]` 写法不再合法，例如 `[iet]` 应改为 `[-iet]`。

### 2.6 为分块策略附加参数

分块策略选择符（`F` / `R` / `V` / `P`）——无论在 `LIGHTRAG_PARSER` 规则还是文件名 hint 中——都可以用圆括号附加该策略的分块参数。括号内逗号**只**用于分隔参数；规则切分是括号感知的，因此该逗号绝不会被误判为规则分隔符（`;` 与 `,` 都是合法的规则分隔符，但推荐 `;`）。

```text
notes.[-R(chunk_ts=800,chunk_ol=80)].md                            # 文件名 hint
LIGHTRAG_PARSER=pdf:legacy-R(chunk_ts=800,chunk_ol=80);*:legacy-R  # 规则
```

当前支持的参数（全称 / 短别名）：

| 参数 | 别名 | 适用策略 | 类型 | 含义 |
| --- | --- | --- | --- | --- |
| `chunk_token_size` | `chunk_ts` | F / R / V / P | int（≥ 1） | 各策略的块大小 |
| `chunk_overlap_token_size` | `chunk_ol` | F / R / P | int（≥ 0） | 块间重叠（V 无重叠） |
| `drop_references` | `drop_rf` | P | bool | 分块前丢弃匹配的参考文献块，如 `paper.[-P(drop_rf=true)].pdf`；布尔参数可省略取值，`paper.[-P(drop_rf)].pdf` 等价于 `drop_rf=true` |

- `process_options` 仍是纯选择符字符串；每个参数会写入该策略的 `chunk_options`（见 §5），策略其它来自环境变量的参数保持不变。别名在内部统一归一化为全称。
- 合并优先级：选择符仍遵循“文件名 hint 的非空选项整体覆盖规则选项”；参数按**同一策略**叠加——先规则参数，再文件名 hint 参数（同一键以文件名为准）。
- 启动期（`LIGHTRAG_PARSER`）与上传期（文件名 hint）均严格校验：未知参数、类型错误、取值越界、把参数加到不支持的策略（如 `V` 上的 `chunk_ol`）都会给出友好报错。

> `drop_references` 检测调参 `CHUNK_P_REFERENCES_TAIL_N`（默认 `0`：扫描全部内容块；正数表示只扫描文末最后 N 块）/ `CHUNK_P_REFERENCES_HEADINGS`（竖线分隔，默认 `References\|Bibliography\|参考文献`）仅经环境变量、运行时实时读取。drop_references可以通过环境变量 `CHUNK_P_DROP_REFERENCES` 设置为全局默认值.

### 2.7 校验、优先级与回退

- 启动时会严格校验 `LIGHTRAG_PARSER`：未知内容提取引擎、错误后缀写法、显式使用不支持的后缀、外部引擎缺少 endpoint、处理选项中的非法字符都会导致启动失败。
- **通配符规则匹配某后缀时**，引擎需通过两道可用性检查（见 `parser_routing._engine_is_usable`）：(a) 该引擎能力表支持此后缀；(b) 若是外部引擎（`mineru` / `docling`），对应 endpoint/token 环境变量已配置。任一检查不过，本规则跳过，继续匹配下一条规则。例如 `*:mineru;html:docling` 中：MinerU 不支持 `html` 后缀（条件 a 不过），`html` 继续命中 `docling`；如果 `MINERU_API_MODE=local` 但未设置 `MINERU_LOCAL_ENDPOINT`，所有 PDF 也会跳过 `*:mineru` 落到下一条规则（条件 b 不过）。这一行为对 `LIGHTRAG_PARSER` 规则匹配和文件名 hint 引擎选择都生效。
- 文件名 hint 的优先级高于 `LIGHTRAG_PARSER`。如果 hint 指定的引擎不支持该后缀，系统会回退到默认规则继续选择可用引擎。
- 如果文件名 hint 提供了非空选项串，则以 hint 为准；否则使用 `LIGHTRAG_PARSER` 规则中匹配项的默认选项；都没有则使用全部默认。
- 如果所有规则都不可用，文件内容提取方式会回退到 `legacy`；如果 `legacy` 也不支持对应的文件后缀，会向系统添加一个错误条目，上传文件保留在 `INPUT` 目录。
- F/R/V/P至多出现一个；同一选项重复时只生效一次但不报错。
- 大小写敏感：分块选项 F/R/V/P必须大写；其它选项 i/t/e小写。
- 中括号内出现非法字符时，整个 hint 失效，引擎按默认规则解析，选项按 `LIGHTRAG_PARSER` 默认或全部默认；同时落日志 warning。
- `P` 对任何能产出 `.blocks.jsonl` sidecar 的引擎（`native` / `mineru` / `docling`）抽取出的结构化结果有效；对 `legacy` 路径或无 sidecar 的输出会自动降级到 `R` 并记录 warning。

## 3. 文件解析引擎

### 3.1 引擎能力矩阵

| 引擎 | 说明 | 支持的文件格式（后缀） |
| --- | --- | --- |
| `legacy` | 旧版提取方式，在加入流水线前集中提取内容 | `txt` `md` `mdx` `pdf` `docx` `pptx` `xlsx` `rtf` `odt` `tex` `epub` `html` `htm` `csv` `json` `xml` `yaml` `yml` `log` `conf` `ini` `properties` `sql` `bat` `sh` `c` `h` `cpp` `hpp` `py` `java` `js` `ts` `swift` `go` `rb` `php` `css` `scss` `less` |
| `native` | 内置智能结构化内容抽取器 | `docx` `md` `textpack` |
| `mineru` | 外部 MinerU 内容提取引擎 | `pdf` `docx` `pptx` `xlsx` `png` `jpg` `jpeg` `jp2` `webp` `gif` `bmp`（可扩展，见 `MINERU_ADDITIONAL_SUFFIXES`） |
| `docling` | 外部 Docling 内容提取引擎 | `pdf` `docx` `pptx` `xlsx` `md` `html` `xhtml` `png` `jpg` `jpeg` `tiff` `webp` `bmp`（可扩展，见 `DOCLING_ADDITIONAL_SUFFIXES`） |

`mineru` 和 `docling` 是外部内容提取引擎，启用相关规则前必须先把服务跑起来，再在 LightRAG 配置对应 endpoint/token。

两个外部引擎上表列的都是**基线**格式集 —— 即引擎开箱即可处理的格式。它们其余的输入格式（旧版 Office `doc` / `xls` / `ppt`，docling 还有 ODF、EPUB、AsciiDoc、LaTeX、CSV 等）取决于**服务侧而非 LightRAG 侧**安装的组件 —— 旧版 Office 转换需要服务侧装有 LibreOffice —— MinerU 还额外取决于当前的 `MINERU_API_MODE`。因此这些格式不全局对外声明：请用 `MINERU_ADDITIONAL_SUFFIXES` / `DOCLING_ADDITIONAL_SUFFIXES` 声明你自己的部署实际能处理哪些（见下文各引擎小节，以及 [env.example](https://github.com/HKUDS/LightRAG/blob/main/env.example) 里的 MinerU / Docling 配置块），再用 `LIGHTRAG_PARSER` 规则或文件名 hint 把这些后缀路由过去 —— 只声明后缀并不会让裸 `x.doc` 变成可上传。

LightRAG 在本地会缓存 `mineru` 和 `docling` 引擎的解析结果。重复上传相同的文件通常不会重新调用引擎解析文档。如果需要删除解析缓存，必须在文档管理界面删除文件弹窗中点击“同时删除文件”选项。修改 `mineru` 和 `docling` 引擎的端点地址和有效提取参数也会导致缓存失效，下次上传相同文件的时候会重新调用引擎解析文件内容。

### 3.2 使用 legacy 内容抽取器

`legacy` 是除 `.textpack`（路由层强制交给 `native`）之外所有后缀的兜底引擎，也是不配置 `LIGHTRAG_PARSER` 时的默认行为。它只抽取纯文本，由此带来四个在把文件路由给它之前值得先知道的后果：

- **它永不产出 sidecar。** 输出是 `parse_format=raw`，不存在 `drawings.json` / `tables.json` / `equations.json` 供分析阶段读取。因此 `i` / `t` / `e` 选项对 legacy 解析的文档完全无效，`P` 也会因为没有标题结构可切而退化为 `R`（§2.7）。
- **它没有原始缓存目录**，所以 `LIGHTRAG_FORCE_REPARSE_*` 对它不适用（§3.7）。
- **它唯一的配置项是 `PDF_DECRYPT_PASSWORD`**，用于打开受密码保护的 PDF。
- **无文本层的扫描版 PDF 会硬失败**，而不是产出一个空文档。这类文件应改路由到能做 OCR 的 `mineru` 或 `docling`。

### 3.3 使用 Native 文件解析引擎

`native` 是 LightRAG 内置的结构化内容抽取引擎，**纯本地运行**：不依赖 MinerU / Docling 等外部服务，抽取阶段也不调用 VLM，开箱即用无需任何部署。运行依赖仅 `python-docx` + `defusedxml`（必备）；其中 markdown 路径的 SVG 栅格化额外依赖**可选**的 `cairosvg`（缺失时跳过该 SVG 并记 warning，不影响其余内容）。为 docx 启用可选的 `smart_heading` 引擎参数时，额外需要钉定版本的 `zh_core_web_sm` / `en_core_web_sm` spaCy 模型（`spacy` 运行时已随 `api` extra 一并安装，模型用 `lightrag-download-cache --spacy-install` 安装——Docker 主镜像已内置）；从不启用该参数的部署无需模型，另外 smart_heading 路径会在解析阶段调用 EXTRACT 角色 LLM。设置环境变量 `DOCX_SMART_HEADING=true` 后，路由到 native 引擎的 `.docx` 文件默认启用 smart_heading——单个文件/规则可用显式 `native(smart_heading=false)` 关闭——同时服务器会在启动时校验 spaCy 模型并 fail-fast（而不是等到首次解析才报错）；`LIGHTRAG_PARSER` 规则中携带 `native(smart_heading=true)`（或其省值写法 `native(smart_heading)`）时同样触发启动校验。该默认值仅作用于新上传：已入库文档重解析时沿用其持久化的引擎参数。

支持后缀：`docx` / `md` / `textpack`。启用方式：

- `docx`、`md` 默认仍走 `legacy`，需显式选择 native，例如默认规则 `LIGHTRAG_PARSER=docx:native`、`LIGHTRAG_PARSER=md:native`，或文件名 hint `report.[native-iet].docx`、`notes.[native].md`（语法见 [§2.4](#24-默认规则lightrag_parser) / [§2.5](#25-单文件覆盖文件名-hint)）。
- `textpack` 为 native 独占后缀，无需 hint/规则即自动路由到 native。

#### docx 抽取能力

native 直接解析 OOXML，能识别以下结构并写入对应 sidecar（sidecar 是否生成由文档实际内容决定，见 [§6.2](#62-__parsed__-目录结构)）：

| 元素 | 抽取行为 | 落盘 |
| --- | --- | --- |
| 标题层级 | Heading 1–9（`pPr/outlineLvl` 或样式继承链推断），供 `P` 分块策略按标题切分 | `blocks.jsonl` |
| 段落 | 含超链接文本、列表自动编号；修订追踪只保留最终文本（去掉删除部分） | `blocks.jsonl` |
| 表格 | 2D 结构，自动展开合并单元格（colspan/rowspan）、提取跨页重复表头 | `tables.json` |
| 图片 / drawing | 嵌入图片导出到资源目录，正文留占位符 | `drawings.json` + `<base>.blocks.assets/` |
| 公式 | OMML → LaTeX，区分块级与行内 | `equations.json` |

图片落盘细节：

- 嵌入图片导出到 `blocks.jsonl` 同级的 `<base>.blocks.assets/` 目录，支持 `png` `jpeg` `gif` `bmp` `tiff` `webp` `emf` `wmf`。
- **SVG 图片**：Word 在保存 SVG 时会同时存矢量 `.svg` 与一张 PNG 位图回退，native docx 落盘的是这张 **PNG 回退**（读取 `<a:blip>` 的 `r:embed`，指向 PNG），不导出 SVG 矢量原图。对下游 VLM 消费而言 PNG 通常已足够，无需再做栅格化。（注意这与下文 md 路径「SVG 经 cairosvg 栅格化」是不同实现：docx 直接取 Word 已生成的 PNG。）
- **VML / OLE 对象**（旧版 Word 图片、Visio 图、公式编辑器预览等）：通过 `v:imagedata` 导出其渲染预览，常见为 EMF/WMF，落入同一 assets 目录；若关系标记为外部链接（`TargetMode="External"`），只记录 URL 不导出字节。**注意：EMF/WMF（及 Visio 等 OLE 对象的预览）目前只能"提取落盘"，无法进入多模态分析**——下游 VLM 图像分析只接受栅格格式 `png` / `jpg` / `jpeg` / `gif` / `webp`，其余格式（EMF/WMF/SVG 等）会被静默跳过（标记 `skipped`，不报错、不影响整篇文档）。例外是**公式**：它以 LaTeX 文本而非图片存储，走文本（EXTRACT）角色分析而非 VLM，因此能被正常处理。

#### docx 段落溯源（paraId）提示

native docx 会采集 Word 2013+ 写入的 `w14:paraId` 作为段落级溯源锚点。若文档由 LibreOffice / WPS / 旧版 Word 生成，或被手工改过 docx 内部 XML，部分段落会缺少 paraId，此时会在日志输出一次提示：

```text
[parse_native] <文件名>: N paragraphs lack paraId; Re-saving file in Word 2013+ to regenerate ids.
```

受影响块的 `positions` 退化为 `[{"type": "paraid", "range": null}]`。这只是提示，**不影响解析成功**；如需精确段落溯源，按提示在 Word 2013+ 中「另存为 .docx」即可重建 id。

#### md / textpack 抽取能力

`native` 引擎除 `docx` 外还支持 Markdown：

- `md`：按标题（ATX `#`）分块，识别 md 原生竖线表格（含表头）、HTML `<table>`（含 `<thead>`，保留 colspan/rowspan）、段落级公式（以 `$$` 开头并以 `$$` 结束的段落；行内 `$...$` 不识别）、内嵌图片（base64 data URL）。代码围栏（```` ``` ````）内的内容原样保留，不参与识别。与 `docx` 一样，`md` 默认仍走 `legacy`，需用 `LIGHTRAG_PARSER=md:native` 或文件名 `[native]` hint 选择 native。
- `textpack`：TextBundle 规范的 zip 包（md 正文 + 资源目录，约定为 `assets/`，Bear / Ulysses 等导出格式）。只有 `native` 支持该后缀，因此无需 hint/规则即自动路由到 native。
  - **包内结构要求**（正文按扩展名定位，不要求固定叫 `text.markdown`，方便用任意 zip 工具自行打包）：
    - 正文文件名任意，扩展名为 `.md` 或 `.markdown` 即可。
    - 若包内含 `*.textbundle` 后缀的子目录，则**最多只能有 1 个**（多于 1 个报错），且正文**只从该 `.textbundle` 子目录查找**（忽略根目录的 md）。
    - 若包内**不含** `*.textbundle` 子目录，则正文**只从压缩包根目录查找**。
    - 查找目录内 `.md` / `.markdown` 文件**必须恰好 1 个**：0 个或多于 1 个均报错。
    - 正文所在目录即资源解析的"包根"（`bundle_root`）。
  - 包内以相对路径（文件引用）内嵌的图片按相对包根目录解析，**允许放在包内任意子目录**（不限于 `assets/`），但禁止目录穿越（`..`、绝对路径、越出包根的引用会被记 warning 跳过）；解析出的字节须通过图片 magic bytes 校验，否则跳过。独立 `.md`（非 textpack）中的相对路径图片不解析（记 warning 跳过）。
- SVG 图片（base64 / textpack 包内文件 / 在线下载）会先经 cairosvg 栅格化为 PNG 再写入 sidecar；cairosvg 不可用或渲染失败时跳过该图（记 warning）。**系统依赖**：`cairosvg` 是对 cairo 的 cffi 绑定——`pip install cairosvg`（随 `api` extra 安装）总能成功，但只有宿主机同时装了原生的 `libcairo` 共享库，栅格化才真正能跑（Debian/Ubuntu 用 `sudo apt-get install libcairo2`，RHEL/Fedora 用 `sudo dnf install cairo`，macOS 用 `brew install cairo`，Windows 需安装内含 `libcairo-2.dll` 的 GTK3 运行时）——`pip`/`uv` 装不了系统库，所以除官方 Docker 镜像外这一步永远不会自动完成。服务器会在启动时探测栅格化能力，缺失时记一条 warning，避免这个缺口一直藏到某篇文档处理时才被发现。
- 外部 URL 图片（`![](http://...)`）**默认下载并内嵌**（`NATIVE_MD_IMAGE_DOWNLOAD_ENABLED` 默认 `true`）；无论下载成功与否都会生成 drawing（成功内嵌资源，失败回退为外链）。下载默认仅允许可全球路由的公网 IP（DNS 解析结果与每一跳重定向目标都校验，且 socket 直连已校验 IP 以防 DNS rebinding，忽略环境 `HTTP(S)_PROXY`），私网 / 环回 / 链路本地 / 保留 / CGNAT（`100.64.0.0/10`）等一律拒绝；如需放行特定内网段，用 `NATIVE_MD_IMAGE_ALLOWED_NON_PUBLIC_CIDRS` 配置 CIDR 白名单。若设为 `false`，外链图片整个丢弃（不生成对应 drawing，故仅含外链图片的文档不会生成 `drawings.json`）。
  - 下载还受**每文档**（而不只是每图片）预算约束：单文档保留的图片总字节数（`NATIVE_MD_IMAGE_MAX_TOTAL_BYTES`）、含重定向跳转在内的远程获取尝试数（`NATIVE_MD_IMAGE_MAX_REQUESTS`）、以及覆盖全部下载的墙钟总预算（`NATIVE_MD_IMAGE_DOWNLOAD_TOTAL_TIMEOUT`）。超预算的远程图片降级为外链并记录解析告警，不会让文档失败；每个请求都有真实的墙钟截止时间（`NATIVE_MD_IMAGE_DOWNLOAD_TIMEOUT`），慢速滴流的对端无法将其无限重置。进行中的下载可被 `POST /documents/cancel_pipeline` 中断。

#### 环境变量

native 的所有 `NATIVE_*` 环境变量与 `.native_raw/` 缓存目录**仅作用于 markdown / textpack 引擎的外链图片下载**；**docx 路径不读取任何 `NATIVE_*` 变量**。最常用的两个：

- `LIGHTRAG_FORCE_REPARSE_NATIVE`（默认 `false`）：强制丢弃 `.native_raw/` 缓存、重新联网下载外链图片。
- `NATIVE_MD_IMAGE_DOWNLOAD_ENABLED`（默认 `true`）：外链图片下载总开关，设为 `false` 时丢弃所有外链图片。

其余下载/大小/预算/SSRF 相关变量（`NATIVE_MD_IMAGE_DOWNLOAD_TIMEOUT` / `NATIVE_MD_IMAGE_DOWNLOAD_REQUIRED` / `NATIVE_MD_IMAGE_MAX_BYTES` / `NATIVE_MD_IMAGE_MAX_SVG_PIXELS` / `NATIVE_MD_IMAGE_MAX_TOTAL_BYTES` / `NATIVE_MD_IMAGE_MAX_REQUESTS` / `NATIVE_MD_IMAGE_DOWNLOAD_TOTAL_TIMEOUT` / `NATIVE_MD_IMAGE_ALLOWED_NON_PUBLIC_CIDRS`）含义与默认值见仓库根目录 [env.example](https://github.com/HKUDS/LightRAG/blob/main/env.example)。

下载的外链图片缓存到 `<文件>.native_raw/`（与 `.parsed/` 同级，类比 `.mineru_raw`/`.docling_raw`），重新解析同一未改动文件时直接复用、不再联网；源文件内容或上述大小 / SVG 像素 / CIDR 配置变化时缓存自动失效。删除文档（删除弹窗勾选「同时删除文件」）时该缓存目录会与 `.parsed/` 一并清理。

### 3.4 使用 MinerU 文件解析引擎

LightRAG文档处理管线支持使用MinerU作为文件解析器。支持使用两种MinerU访问模式：

- `official`模式：使用MinerU云端的 API v4 服务。需要先到 [MinerU官网](https://mineru.net/) 注册账号并创建API-KEY。然后在LightRAG的 `.env` 文件中添加以下配置：

```bash
MINERU_API_MODE=official
MINERU_API_TOKEN=<your_token>
# MINERU_OFFICIAL_ENDPOINT=https://mineru.net   # 默认值，通常无需修改
```

* `local`模式：使用本地部署的 MinerU 服务，部署方式见 [ParserServiceDeployment-zh.md §1](./ParserServiceDeployment-zh.md#1-本地部署-mineru-服务)。本地MinerU服务启动后在LightRAG的 `.env` 文件中添加以下配置：

```bash
MINERU_API_MODE=local
MINERU_LOCAL_ENDPOINT=http://<your_mineru_local_server_ip>:8000
```

两种模式都生效的共享参数：

| 环境变量 | 缺省 | 含义 |
| --- | --- | --- |
| `MINERU_LANGUAGE` | `ch` | OCR / 解析语言 |
| `MINERU_ENABLE_TABLE` | `true` | 表格识别。**它决定 `tables.json` 是否会被写出**——比 §2.1 的 `t` 选项早一个阶段，后者只决定已存在的 sidecar 是否被分析 |
| `MINERU_ENABLE_FORMULA` | `true` | 公式识别；与 `e` 选项是同样的关系 |
| `MINERU_PAGE_RANGES` | （空） | 页码范围。`official` 原样透传，支持 `1-3,5,7-9`；`local` 只接受单页或一个简单区间，逗号列表在启动期就会被拒绝 |
| `MINERU_ADDITIONAL_SUFFIXES` | （空） | 本部署在 §3.1 基线之外还能处理的后缀——为什么只声明它还不够，见该节说明 |

仅 `local` 模式：

| 环境变量 | 缺省 | 含义 |
| --- | --- | --- |
| `MINERU_LOCAL_ENDPOINT` | `http://127.0.0.1:8000` | mineru-api / mineru-router 基础 URL |
| `MINERU_LOCAL_BACKEND` | `hybrid-auto-engine` | 由哪个后端执行解析：`hybrid-auto-engine`（pipeline + VLM，需要 GPU 及配套推理引擎）、`pipeline`（对 CPU 友好，无 VLM 环节）、`vlm-auto-engine` |
| `MINERU_LOCAL_PARSE_METHOD` | `auto` | pipeline 部分的解析策略：`auto` / `txt` / `ocr`。**纯 VLM 后端会忽略它**——版面与 OCR 由模型自身处理 |
| `MINERU_LOCAL_IMAGE_ANALYSIS` | `false` | MinerU **自己的** VLM 轮次，用于 caption 与脚注——见 §4.7，它与 `i` 选项无关。`pipeline` 后端会静默丢弃该标志 |
| `MINERU_LOCAL_START_PAGE_ID` | `0` | 起始页 |
| `MINERU_LOCAL_END_PAGE_ID` | `99999` | 结束页 |

仅 `official` 模式：

| 环境变量 | 缺省 | 含义 |
| --- | --- | --- |
| `MINERU_API_TOKEN` | — | MinerU 官网申请的 API key，必填 |
| `MINERU_OFFICIAL_ENDPOINT` | `https://mineru.net` | 服务端点 |
| `MINERU_MODEL_VERSION` | `vlm` | 向云服务请求的模型版本 |
| `MINERU_IS_OCR` | `false` | 云端 OCR 开关 |

轮询预算与缓存：

| 环境变量 | 缺省 | 含义 |
| --- | --- | --- |
| `MINERU_POLL_INTERVAL_SECONDS` | `2` | 任务状态轮询间隔 |
| `MINERU_MAX_POLLS` | `600` | 放弃前的最大轮询次数；缺省预算约 20 分钟 |
| `MINERU_ENGINE_VERSION` | （空） | 记入原始产物包 manifest，不一致即缓存失效；留空则跳过该检查（§6.3） |
| `LIGHTRAG_FORCE_REPARSE_MINERU` | `false` | 绕过原始缓存，每次解析都重新上传（§3.7） |
| `MINERU_BBOX_ATTRIBUTES` | `{"origin":"LEFTTOP","max":1000}` | 记入 sidecar meta 的坐标系。注意其缺省值与 `DOCLING_BBOX_ATTRIBUTES` 不同 |

> **本地部署 MinerU 服务**（Docker 镜像构建、vLLM 预加载、标题层级修正）见 [ParserServiceDeployment-zh.md §1](./ParserServiceDeployment-zh.md#1-本地部署-mineru-服务)。

### 3.5 使用 Docling 文件解析引擎

`docling` 内容提取引擎需要外部的 [docling-serve](https://github.com/DS4SD/docling-serve) 服务（v1 异步 API）。最少配置：

```bash
DOCLING_ENDPOINT=http://localhost:5001
```

`DOCLING_ENDPOINT` 只填 base URL（**不**带 `/v1/convert/file/async`）。目前LightRAG固定使用 Docling 的 standard 流水线处理文件。用户可以通过以下环境环境变量来控制 Docling 流水线的行为：

| Env | 默认 | 含义 |
| --- | --- | --- |
| `DOCLING_DO_OCR` | `true` | OCR 总开关 |
| `DOCLING_FORCE_OCR` | `true` | 强制对每页 OCR（扫描件必须开，非扫描件开启通常也有助于提高版面识别质量） |
| `DOCLING_OCR_ENGINE` | `auto` | OCR 引擎选择（不建议修改） |
| `DOCLING_OCR_PRESET` | `auto` | OCR 引擎 preset（不建议修改） |
| `DOCLING_OCR_LANG` | （空） | 按照OCR引擎要求设置（不建议修改） |
| `DOCLING_DO_FORMULA_ENRICHMENT` | `false` | 是识别文档中的公式并按LaTex格式输出；启用前需要确保Docling后台下载了公式识别模型（见 [ParserServiceDeployment-zh.md §2](./ParserServiceDeployment-zh.md#2-本地部署-docling-serve启用-latex-公式识别)） |

未配置 `DOCLING_OCR_ENGINE` / `DOCLING_OCR_PRESET` 时等同于 `auto`；未配置 `DOCLING_OCR_LANG` 时不向 docling-serve 传递语言列表，由 OCR 引擎使用自身默认值。解析缓存按这些有效参数计算签名，因此“未配置”和“显式填写默认值”不会导致缓存失效。

可选输入格式 1 个 env：

| Env | 默认 | 含义 |
| --- | --- | --- |
| `DOCLING_ADDITIONAL_SUFFIXES` | （空） | 本部署的 docling-serve 在 §3.1 基线格式集之外还能处理的后缀，逗号分隔，例如 `doc,ppt,xls`。Docling 的旧版 Office 支持需要 docling-serve 侧装有 LibreOffice，因此这些格式按部署逐个声明，不全局对外宣告 |

`DOCLING_ADDITIONAL_SUFFIXES` 使用要点：

- 用 `,` 分隔的裸小写后缀；允许带前导点和两侧空格（` .DOC ` 等同 `doc`）。其它写法（按 glob 习惯写成 `*.doc`、或用 `;` 分隔）会在启动时报错，不会被静默忽略。
- 它只让该后缀**可以路由到 docling**，本身并不会让裸 `x.doc` 变成可上传。必须配合路由规则（`LIGHTRAG_PARSER=doc:docling`）或文件名 hint（`x.[docling].doc`）使用 —— 否则这类文件仍会落到默认的 `legacy` 引擎并被判为不支持的后缀。反过来，只写 `doc:docling` 规则而不配这个 env，会因 `doc` 不在 docling 能力表内而启动校验失败。
- 该变量在使用时实时读取，因此写在父进程环境或 `.env` 里都同样生效。

轮询预算 2 个 env（docling-serve 是 server-side long-poll，客户端不再额外 sleep）：

| Env | 默认 | 含义 |
| --- | --- | --- |
| `DOCLING_POLL_INTERVAL_SECONDS` | `5` | 等待解析结果的轮询间隔时间 |
| `DOCLING_MAX_POLLS` | `240` | 最大轮询轮次，超过抛 `TimeoutError`；<br />默认等待时间 ≈ 5 x 240（约20 分钟） |

Bundle 缓存 3 个 env：

| Env | 默认 | 含义 |
| --- | --- | --- |
| `DOCLING_ENGINE_VERSION` | （空） | Docling引擎版本；版本变化会导致解析缓存失效 |
| `LIGHTRAG_FORCE_REPARSE_DOCLING` | `false` | 设为 `true`/`1` 时不启用解析缓存 |
| `DOCLING_BBOX_ATTRIBUTES` | `{"origin":"LEFTBOTTOM"}` | Docling 版面默认坐标系 |

**`DOCLING_DO_FORMULA_ENRICHMENT` 启用前提**：docling-serve 侧需就绪 code-formula 模型权重。adapter 双轨兼容 —— 启用时 `text` 字段为 LaTeX，关闭或权重缺失导致 `text == orig` 时自动按普通文本处理，不写 `equations.json`。因此默认 `false` 是保守值，部署侧确认模型就绪后再开启。

> **本地部署 docling-serve**（含 `DOCLING_DO_FORMULA_ENRICHMENT` 所需的公式识别模型下载）见 [ParserServiceDeployment-zh.md §2](./ParserServiceDeployment-zh.md#2-本地部署-docling-serve启用-latex-公式识别)。

### 3.6 为解析引擎附加参数

参数也可以附加到**引擎 token** 上，按文件覆盖外部引擎的行为。它们被编码进持久化的 `parse_engine` 字段，同时作用于引擎请求与其原始包缓存签名（因此改动参数会触发重解析，而非复用旧缓存包）。

```text
paper.[mineru(page_range=1-3,language=en,local_parse_method=ocr)].pdf   # 文件名 hint
scan.[docling(force_ocr=true)].pdf
report.[native(smart_heading)].docx                                      # 布尔参数的省值写法
LIGHTRAG_PARSER=pdf:mineru(language=en);*:legacy-R                       # 规则
```

当前支持的引擎参数（全称 / 别名）：

| 引擎 | 参数 | 别名 | 类型 | 说明 |
| --- | --- | --- | --- | --- |
| `mineru` | `page_range` | `pr` | 列表 | 一个或多个页码范围；**见下方列表说明** |
| `mineru` | `language` | — | str | OCR / 模型语言（如 `en`、`ch`） |
| `mineru` | `local_parse_method` | `local_pm` | 枚举 | `auto` / `txt` / `ocr`（local 模式） |
| `docling` | `force_ocr` | `ocr` | bool | `true` / `false`；可省值裸写，`docling(ocr)` 等价于 `docling(force_ocr=true)` |
| `native` | `smart_heading` | — | bool | 可选开启的 docx 智能标题识别（见 [§3.3](#33-使用-native-文件解析引擎)）；可省值裸写，`native(smart_heading)` 等价于 `=true`；markdown 路径会告警并忽略 |

- **`page_range` 可写多个页码段——每段都单独写一个 `page_range=...`。** 括号 `(...)` 内逗号只分隔参数，因此多段页码要写成 `page_range=1-3,page_range=5,page_range=7-9`，不要写成环境变量里的单串形式 `MINERU_PAGE_RANGES="1-3,5,7-9"`。**多段** `page_range` 需要 `MINERU_API_MODE=official`；`local` 模式只接受单页/单段（如 `page_range=1-3`）。
- **`local_parse_method` 仅限 local 模式。** 它只影响本地 MinerU 请求，因此在 `MINERU_API_MODE=official` 下会被**拒绝**（official API 既不发送它、也不计入缓存键——接受它将静默无效）。
- **布尔型引擎参数可以省略取值、写成裸开关**，用于缩短规则与文件名：`native(smart_heading)` 等价于 `native(smart_heading=true)`，`docling(ocr)` 等价于 `docling(force_ocr=true)`。只有布尔参数可以这样写（`mineru(language)` 会友好报错）；持久化的 `parse_engine` 始终按全称 `key=value` 重新编码（`native(smart_heading=true)`），因此省值写法不会改变缓存签名。关闭布尔参数仍需显式写出（`native(smart_heading=false)`）。
- 引擎参数只被声明了参数的引擎接受（`mineru` / `docling` / `native`）；给 `legacy` 附加参数、或给任一引擎附加未知参数，都会友好报错。校验在启动期（`LIGHTRAG_PARSER`）与上传期均执行。
- 合并优先级：引擎参数按**最终引擎**解析——当文件名 hint 选中了另一个可用引擎时，规则的引擎参数会被丢弃。
- `parse_engine` 以 hint 语法存储（如 `mineru(page_range=1-3)`），并展示在 `doc_status` metadata 中，便于查看文档当时使用的解析参数。

### 3.7 解析缓存与强制重解析

`native`、`mineru`、`docling` 三个引擎各自在解析产物旁保留一份原始产物缓存，这样重新解析未变更的文件时不必重复昂贵的工作——一次外部服务往返，或一轮 markdown 外链图片下载。目录布局与失效规则见 §6.3；配置时需要知道的是：

| 引擎 | 缓存目录 | 内容 | 强制重解析 |
| --- | --- | --- | --- |
| `native` | `<base>.native_raw/` | markdown / textpack 路径下载的外部图片 | `LIGHTRAG_FORCE_REPARSE_NATIVE=true` |
| `mineru` | `<base>.mineru_raw/` | MinerU 服务返回的产物包 | `LIGHTRAG_FORCE_REPARSE_MINERU=true` |
| `docling` | `<base>.docling_raw/` | docling-serve 返回的产物包 | `LIGHTRAG_FORCE_REPARSE_DOCLING=true` |
| `legacy` | — | 无缓存 | 不适用 |

每个产物包都在 manifest 里记录引擎版本与生效参数签名，所以改服务端点、改抽取参数、或改 `MINERU_ENGINE_VERSION` / `DOCLING_ENGINE_VERSION` 本身就会让缓存失效。强制标志是为 manifest 察觉不到的那种情况准备的：**服务**变了而它的版本串没变。删除文档时勾选"同时删除文件"会把缓存目录连同 `.parsed/` 一并移除。

## 4. 多模态分析（VLM）

### 4.1 这一阶段做什么

解析阶段写 sidecar；ANALYZING 阶段读取它、把每个 item 送给模型、再把结果写回该 item 的 `llm_analyze_result`；随后 PROCESS 阶段据此构建多模态分块。这个顺序带来两个结论：

- **分析可以不重新解析就重跑。** 抽取阶段不受 `i` / `t` / `e` 影响——引擎按文档实际内容产出 sidecar——所以事后再启用某个模态，只会补做 VLM 工作，不会重新碰原始文件（§9.3）。
- **没有对应 sidecar 的模态是静默空操作**，仅记一条 INFO 日志。这不是错误，它意味着文档没有该类内容，或该引擎不产出它。

### 4.2 每个选项实际需要什么

`VLM_PROCESS_ENABLE` 常被读成三种模态的总开关。它不是——它只闸控图片：

| 选项 | 分析角色 | 需要 `VLM_PROCESS_ENABLE` | 读取的 sidecar |
| --- | --- | --- | --- |
| `i` 图片 | VLM 角色 | **是** | `*.drawings.json` |
| `t` 表格 | EXTRACT 角色 | 否 | `*.tables.json` |
| `e` 公式 | EXTRACT 角色 | 否 | `*.equations.json` |

这也是推荐预设 `*:native-teP` 在完全不配置 VLM 的情况下也能工作的原因。而 sidecar 是否存在还取决于引擎：MinerU 侧是 `MINERU_ENABLE_TABLE` / `MINERU_ENABLE_FORMULA`，docling 侧是 `DOCLING_DO_FORMULA_ENRICHMENT`，`legacy` 则一个都不产出。

### 4.3 失败与跳过

本节内容**仅适用于 `process_options` 含 `i` 的文档**。不含 `i` 的文档根本不会进入图片分析，因此无论它带多少张图片、无论 `VLM_PROCESS_ENABLE` 如何设置，都会正常处理完。

对于确实含 `i` 的文档，每张图片要走一条过滤链，而 **VLM 闸门位于这条链的中间**。闸门之前的每一步都是跳过、文档安然无恙；闸门本身则让整个文档失败：

| 顺序 | 条件 | 结局 |
| :-: | --- | --- |
| 1 | 图片文件不存在 | item `skipped`，文档继续 |
| 2 | 不是受支持的栅格格式（emf / wmf / svg 等） | item `skipped`，文档继续 |
| 3 | 任一边小于 `VLM_MIN_IMAGE_PIXEL`（64） | item `skipped`，文档继续 |
| 4 | **VLM 闸门**：`VLM_PROCESS_ENABLE=false` 或没有 VLM 角色 | **文档 FAILED** —— `error_msg` 为 "VLM analysis required but VLM role is not available" |
| 5 | 图片文件为空 | 文档 FAILED |
| 6 | 超过 `VLM_MAX_IMAGE_BYTES`（5 MB） | item `skipped` —— 注意此检查**在闸门之后**，超大图并不能绕过第 4 步 |

文本模态一侧：内容为空的**表格** item 会先被记为 `skipped` 并打 WARNING，**不会走到 format 校验**；只有**有内容**的表格 item，其 `format` 缺失或非法时才让文档失败，那意味着 sidecar 已损坏。内容为空的**公式** item 则直接让文档失败。

### 4.4 token 预算

| 变量 | 缺省 | 含义 |
| --- | --- | --- |
| `MAX_EXTRACT_INPUT_TOKENS` | `20480` | 单次抽取 / 分析提示词的输入总预算 |
| `SURROUNDING_LEADING_MAX_TOKENS` | `2000` | 注入到 item **之前**的上下文的单侧上限 |
| `SURROUNDING_TRAILING_MAX_TOKENS` | `2000` | 注入到 item **之后**的上下文的单侧上限 |
| `MM_EXTRACT_CONTENT_MIN_TOKENS` | `100` | 为 item 自身内容保留的下限 |

上下文预算从总预算里扣，设得过高会挤占 item 本身。发生时服务会在启动阶段告警并点名该调哪个变量：要么调高 `MAX_EXTRACT_INPUT_TOKENS`，要么调低这一对上下文上限。

### 4.5 图片限制与吞吐

`VLM_MAX_IMAGE_BYTES`（缺省 5 MB）与 `VLM_MIN_IMAGE_PIXEL`（缺省 64）界定什么值得送出去：下限的存在是为了跳过图标和分隔线，而不是为它们付一次 VLM 调用。分析阶段的并发是 `MAX_PARALLEL_ANALYZE`（§8.6），与解析、入库阶段互不影响。

### 4.6 模型配置不在本章

`VLM_PROCESS_ENABLE` 是流水线开关，归本章。**具体哪个模型**承担 VLM 角色——`VLM_LLM_MODEL`、`VLM_LLM_BINDING`、`VLM_LLM_BINDING_HOST`、`VLM_LLM_BINDING_API_KEY`、`VLM_MAX_ASYNC_LLM`、`VLM_LLM_TIMEOUT`——以及支持视觉输入的 provider 列表，归 [基于角色的 LLM/VLM 配置指南](RoleSpecificLLMConfiguration-zh.md)，本文刻意不重复，以免两处漂移。

### 4.7 容易混淆的另一件事：`MINERU_LOCAL_IMAGE_ANALYSIS`

MinerU 自己也有一轮 VLM，由 `MINERU_LOCAL_IMAGE_ANALYSIS` 开启。它在**解析期跑在 MinerU 主机上**，改善的是 MinerU 自身输出里的 caption 与脚注，消耗那台机器的 GPU，且被 `pipeline` backend 忽略。它与 `i` 选项、与 `VLM_PROCESS_ENABLE` 都无关。

### 4.8 一个安全的启用顺序

1. 先只开 `te`。它只需要 EXTRACT 角色——不引入新基础设施，也不引入新的失败模式。
2. 确认 sidecar item 的 `llm_analyze_result.status == "success"`。
3. 再把 `i` 与 `VLM_PROCESS_ENABLE=true` 以及一个支持视觉的 binding **一起**加上。

把第 3 步拆成两半，就会撞上 §4.3 第 4 行那种失败。而且因为处理选项在入队时已冻结，已经这样失败的文档无法靠改配置重试救回来：`/documents/reprocess_failed` 会按原来的 `i` 重跑。要么把 VLM 配好再重试，要么删除该文档、以不带 `i` 的配置重新上传（§8.3）。

## 5. 分块器参数配置（chunk_options）

### 5.1 process_options vs chunk_options 的职责

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
- **`addon_params["chunker"]`** 是 `ObservableAddonParams` 字段；Server 部署只需通过 env / 重启即可让新值生效。若需要在 Python 进程内运行时改它（不重启）以及 per-file 覆盖，请见[第十一章 Python SDK 调用](#11-python-sdk-调用)。
- **`full_docs.chunk_options`** 在 `apipeline_enqueue_documents` 入队时冻结：默认由 `resolve_chunk_options(self.addon_params, ...)` 现场拼装；若调用方传入 `chunk_options` 参数则原样持久化（SDK 用法，见 §11.4）。
- **分块器调用**从 `full_docs.chunk_options` 取对应子字典，按 `process_options.chunking` selector 派发到 F/R/V/P。

### 5.2 环境变量

以下变量都在 `LightRAG` 实例化时一次性读入 `addon_params["chunker"]`：strategy 特定 env 由 `default_chunker_config()` 读取，legacy env（`CHUNK_SIZE` / `CHUNK_OVERLAP_SIZE`）由 `_apply_chunk_size_overlay` 在 strategy env 与 legacy 构造字段都没填的槽位上兜底。修改 env 后需要重启服务（或新建 `LightRAG` 实例）才生效；已入队的文档持有冻结快照不受影响。

下面按各变量配置的策略分组。同一个槽位上，strategy 特定变量始终高于全局兜底。

#### 全局兜底

- **`CHUNK_SIZE`** —— `1200`，int。
  顶层 `chunk_token_size` 兜底。低于 strategy 特定 env，也低于 SDK 路径设置的 `addon_params["chunker"]["chunk_token_size"]`。
- **`CHUNK_OVERLAP_SIZE`** —— `100`，int。
  overlap 兜底。只有当某策略既没有自己的 env（`CHUNK_F_OVERLAP_SIZE` / `CHUNK_R_OVERLAP_SIZE` / `CHUNK_P_OVERLAP_SIZE`）、也没有 SDK 路径的 `LightRAG(chunk_overlap_token_size=…)` 时才填入该槽位。

#### F —— 定长

- **`CHUNK_F_SIZE`** —— 未设，int。
  F 自己的 `chunk_token_size`，高于顶层兜底（`CHUNK_SIZE` 与 SDK 路径的 `LightRAG(chunk_token_size=…)`）。未设时 F 沿用顶层解析结果。
- **`CHUNK_F_OVERLAP_SIZE`** —— 未设，int。
  F 自己的 overlap，高于 legacy 构造字段与 `CHUNK_OVERLAP_SIZE`。
- **`CHUNK_F_SPLIT_BY_CHARACTER`** —— 未设（`null`），str。
  预切分隔符。`null` 或空串表示仅按 token 窗切分。
- **`CHUNK_F_SPLIT_BY_CHARACTER_ONLY`** —— `false`，bool。
  严格模式：不再按 token 二次切分，遇到超长片段直接抛错。

#### R —— 递归字符

- **`CHUNK_R_SIZE`** —— 未设，int。
  R 自己的 `chunk_token_size`，高于顶层兜底。未设时 R 沿用顶层解析结果。
- **`CHUNK_R_OVERLAP_SIZE`** —— 未设，int。
  R 自己的 overlap，高于 legacy 构造字段与 `CHUNK_OVERLAP_SIZE`。
- **`CHUNK_R_SEPARATORS`** —— JSON 数组字符串。默认值：

  ```json
  ["\n\n","\n","。","！","？","；","，"," ",""]
  ```

  分隔符级联，按语义边界从最强到最弱排列。默认包含中文句末（`。！？`）与句中（`；，`）标点，使中文 / 中英混合文档能在语义边界切分。英文 `.?!` 被刻意排除——按字面匹配会切断数字与缩写。

  **上限为 64 条、单条最长 256 字符。** 切分器每保留一个分隔符就下降一层，且每层都会重扫全文，因此过长的级联付出 `O(len(separators) × len(text))` 的代价却切不出更多内容。两条上限的行为并不相同，且差异会体现在输出上：单条超过 256 字符的分隔符会被**整条丢弃**而非截短——一条 300 字符的分隔符是消失，而不是退化成匹配它的前 256 字符；列表超过 64 条则**截断**到 64 条，若原列表末尾有字符级 `""` 哨兵则予以保留。若无一条存活，回退目标由消费方决定，且**不是**本变量的默认值：`chunking_by_recursive_character` 走它文档化的 `separators=None` 路径，即切分器自带的四条纯英文级联 `["\n\n", "\n", " ", ""]`，而非九条的 `DEFAULT_R_SEPARATORS`；`load_chunk_separators` 则回退到去掉哨兵的 `DEFAULT_R_SEPARATORS`。

  对非 HTTP 配置，收敛只会在值被缓存时告警一次：`CHUNK_R_SEPARATORS` 在配置装载时，显式提供或整体替换的 `addon_params['chunker']` 会立即处理（为兼容而保留的嵌套原地修改，则在第一次入队时处理）。规范化后的值会**原地**写回并供后续文档使用——因此按文档化的运行时改法持有的那个嵌套 `recursive_character` 字典引用，在收敛之后仍然生效。直接 SDK 调用与旧版本已持久化的快照会保留原值，并在执行时静默收敛——包括全部丢弃后回退到 `separators=None` 这一分支（它只可能由直接 SDK 调用触发）——避免一个历史错误值按文档重复告警。

  有一种修正并不是收敛：`addon_params['chunker']` 的 `separators` 若既不是 list/tuple 也不是 `None`，该键会被**整个移除**并单独告警。原因是裸 `str` 满足 `Sequence[str]`——对它做边界收敛会按字符迭代，把一处笔误改写成 64 个单字符的级联，此后看上去完全像是有意为之。移除该键则让切分器走它文档化的 `separators=None` 路径。**HTTP 请求体则是直接拒绝**：请求模型对两条上限都会抛出校验错误，因此 `/documents/text` 等接口返回 **422**，不会走丢弃、截断或回退中的任何一条。数值上限各处相同，但对越界的响应刻意不同——发起 HTTP 请求的调用方就在现场能读到错误，写环境变量的人不在。

#### V —— 语义向量

- **`CHUNK_V_SIZE`** —— 未设，int。
  V 自己的 `chunk_token_size`。它是硬上限：超过的部分会通过 R 二次切分。高于顶层兜底；未设时 V 沿用顶层解析结果。
- **`CHUNK_V_BREAKPOINT_THRESHOLD_TYPE`** —— `percentile`，str。
  取值为 `percentile` / `standard_deviation` / `interquartile` / `gradient` 之一。
- **`CHUNK_V_BREAKPOINT_THRESHOLD_AMOUNT`** —— 未设（`null`），float。
  阈值大小。`null` 表示让 LangChain 按类型自选默认值（percentile 用 95）。
- **`CHUNK_V_BUFFER_SIZE`** —— `1`，int。
  计算距离时合并的相邻句数。
- **`CHUNK_V_SENTENCE_SPLIT_REGEX`** —— str。默认值：

  ```text
  (?<=[.?!])\s+|(?<=[。？！])
  ```

  喂给 LangChain `SemanticChunker` 的句子切分正则。默认同时识别英文 `.?!`（要求后接空白，因此 `0.95` 不会被切开）和中文 `。？！`（不要求空白，适应中文连写）。env 值是原始正则，无需 JSON 引号。

#### P —— 段落语义

- **`CHUNK_P_SIZE`** —— `2000`（`DEFAULT_CHUNK_P_SIZE`），int。
  P 自己的 `chunk_token_size`，也是唯一**不**继承的槽位。未设时 P 不会回退到顶层 `CHUNK_SIZE` / `LightRAG(chunk_token_size=…)`，而是始终携带 `DEFAULT_CHUNK_P_SIZE`——段落语义合并需要比全局默认更大的上限，才能把相关段落保留在一起。部署需要别的上限时在这里覆盖。P 的内部比例常量是算法刻度，会按这里解析出的值等比推导。
- **`CHUNK_P_OVERLAP_SIZE`** —— 未设，int。
  P 自己的 overlap，高于 legacy 构造字段与 `CHUNK_OVERLAP_SIZE`。它管两件事：同一 JSONL content 行内长正文 fallback 到 R 时的文本重叠，以及大表格前后桥接文字复制进相邻块的单侧预算。它**不会**让表格行级切片互相重叠。

> `CHUNK_F_SIZE` / `CHUNK_R_SIZE` / `CHUNK_V_SIZE` 的行为与 `CHUNK_P_SIZE` 相反：不设时它们**会**沿用顶层 `chunk_token_size`。这通常正是你想要的——F 就是默认全局窗口，R 偏向较小目标以便按句段切分，而 V 作为 advisory ceiling 通常是被调大而非调小，以减少过度拆分。

### 5.3 优先级链

每个分块槽位的最终值按 specificity-ordered 链解析（高 → 低）：

1. **`addon_params["chunker"]` 显式值** —— 通过 SDK 路径运行时设置或在构造时显式写入的字段值（见 §11.3）。Server-only 部署通常不会出现这一档。最直接，赢一切。
2. **strategy 特定 env** —— 如 `CHUNK_F_SIZE` / `CHUNK_R_SIZE` / `CHUNK_V_SIZE`（各策略 `chunk_token_size`）、`CHUNK_F_OVERLAP_SIZE` / `CHUNK_R_OVERLAP_SIZE` / `CHUNK_P_OVERLAP_SIZE`（overlap）、`CHUNK_P_SIZE`（P 专属）。未设对应 size env 时，F/R/V 沿用顶层 `chunk_token_size`。仅当槽位未被 ① 显式占用时填入。
3. **legacy 构造字段** —— `LightRAG(chunk_token_size=…, chunk_overlap_token_size=…)`，仅 SDK 路径生效，详见 §11.2。strategy 无关，"粗粒度缺省"，只填仍空的槽位。
4. **legacy env** —— `CHUNK_SIZE` / `CHUNK_OVERLAP_SIZE`。最终回退。

举例：`CHUNK_R_OVERLAP_SIZE=42` + `LightRAG(chunk_overlap_token_size=2)` → R 子字典 `chunk_overlap_token_size=42`（strategy env 胜出），F / P 子字典 `chunk_overlap_token_size=2`（无 F / P 特定 env，legacy 构造字段填入）。

**P 的 `chunk_token_size` 特例**：P 的 `chunk_token_size` 槽位**不**走完整的四档链。当 ① 未显式提供时，直接按 `CHUNK_P_SIZE` env > `DEFAULT_CHUNK_P_SIZE`（2000）解析，**跳过** ③ legacy 构造字段 `LightRAG(chunk_token_size=…)` 与 ④ legacy env `CHUNK_SIZE`。理由参见 §5.2 `CHUNK_P_SIZE` 行。

三层语义保证：

1. **复现性**：env 改了，重启后老文档仍按入队那一刻的快照分块，结果不变。
2. **续跑一致性**：续跑分支 B（内容已抽取，按当前 `process_options` 重做分块）读的也是 `full_docs.chunk_options`，避免 env 漂移破坏一致性。
3. **per-file 个性化**：调用方可以为每个文件传不同的 `chunk_options`（典型用法：管理 UI 单独配置某个文件的 separators 或 V 阈值）。这是 SDK 路径的入参语义，详见 §11.4。

### 5.4 字段结构

`addon_params["chunker"]`（实例字段）保留全部四种策略的子字典作为运行时基线；`full_docs[doc_id]["chunk_options"]` 是**精简快照**——入队时只保留 `process_options` 选中的那一路策略子字典（缺省 F），其它策略的参数会被丢弃，因为处理阶段不会读它们。重新解析时 `process_options` 与 `chunk_options` 一同改写，避免旧策略的参数残留。

**`addon_params["chunker"]` 全量基线**（运行时可由 SDK 修改，影响后续入队）：

```jsonc
{
  "chunk_token_size": 1200,                                   // 通用 token 上限
  "fixed_token": {                                            // F 专属
    "chunk_token_size": 1200,                                 // 可选;不写沿用顶层 chunk_token_size(可由 CHUNK_F_SIZE 种子化)
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
                                                              // 仅可通过 env/SDK 配置（CHUNK_V_SENTENCE_SPLIT_REGEX）；REST 的
                                                              // `chunking.params` 传入该键会返回 422 —— 见 GHSA-32jh-39m7-8x84（ReDoS）
  },
  "paragraph_semantic": {                                     // P 专属
    "chunk_token_size": 2000,                                 // 不写则按 CHUNK_P_SIZE 或 DEFAULT_CHUNK_P_SIZE（2000）解析；
                                                              // **不**继承通用 chunk_token_size
    "chunk_overlap_token_size": 100                           // 不写沿用 legacy overlap 解析链
  }
}
```

**`full_docs[doc_id]["chunk_options"]` 精简快照**（按 selector 投影；下例为 `process_options="R"`）：

```jsonc
{
  "chunk_token_size": 1200,                                   // 通用 token 上限（保留为顶层 fallback）
  "recursive_character": {                                    // 唯一保留的策略子字典
    "chunk_overlap_token_size": 100,
    "separators": ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
  }
}
```

selector → 子字典映射：F → `fixed_token`，R → `recursive_character`，V → `semantic_vector`，P → `paragraph_semantic`；无 selector 默认 F。各子字典与对应分块器函数的 keyword-only 参数一一对应；新增参数时无需改 dispatcher，只在 chunker 函数添加 kwarg 即可。

### 5.5 缺失兼容

老文档入队时还没有 `chunk_options` 字段；分块时 dispatcher 会按当前 `process_options` 调用 `resolve_chunk_options(self.addon_params, process_options=…)` 兜底拼装一份精简快照。建议在升级后通过 reprocess 一次让老文档拿到精简的 `chunk_options` 快照（且与当前 `process_options` 对齐）。

## 6. 存储与目录布局

### 6.1 `full_docs` 字段

文件入队和抽取结果会写入 `full_docs`：

| 字段 | 说明 |
| --- | --- |
| `file_path` | 文件名 basename（不含目录），**保留用户提供的原始名（含中括号 hint）**，例如 `abc.[native-iet].docx` 原样写入。未提供有效来源时保存为 `unknown_source`。文件名 hint 不会被剥离，方便管理 UI 直接展示用户原本的命名意图。 |
| `canonical_basename` | 去掉处理提示 hint 后的规范化 basename（例如 `abc.docx`）。文件名查重以此字段为索引 key，保证 `abc.docx` 与 `abc.[native-iet].docx` 视为同一逻辑文档。 |
| `source_path` | 入队时提供的原始路径（仅当含目录分隔符或绝对路径时才写入），供 `native` / `mineru` / `docling` 解析器定位真实文件位置。 |
| `parse_format` | 内容格式：`pending_parse`, `raw`, `lightrag`。 |
| `content` | `raw` 时保存抽取文本；`pending_parse` 时为空字符串；`lightrag` 时存储以 `{{LRdoc}}` 开头的**完整合并文本**（拼接 `.blocks.jsonl` 中所有 `type=="content"` 行的 body 段），解析阶段的 reuse handler（`ReuseParser`）会剥离前缀后再交给 chunking_func，与 `raw` 走完全相同的代码路径。 |
| `content_hash` | 内容 MD5，用于跨文件名查重。`parse_format=raw` 取 `sanitize_text_for_encoding` 后文本的 hash；`parse_format=lightrag` 取 `*.blocks.jsonl` 文件 hash；`parse_format=pending_parse` 不写入，待抽取完成后补上。 |
| `lightrag_document_path` | `parse_format=lightrag` 时保存结构化 LightRAG Document 的路径；新记录优先保存为相对 `INPUT_DIR` 的路径，例如 `__parsed__/report.docx.parsed/report.blocks.jsonl`。注意路径中的子目录与 blocks 文件名都使用规范化 basename（不含 hint）。 |
| `parse_engine` | 实际完成抽取的引擎：`legacy`, `native`, `mineru`, `docling`。对于待抽取文件，也可暂存目标引擎。 |
| `process_options` | 入队时记录的原始处理选项串（不含引擎名和分隔 `-`），例如 `"iet"`、`"R!"`、`""`。下游各阶段以此字段为权威源，决定是否启用图像/表格/公式分析（`i/t/e`）、是否禁止知识图谱构建（`!`）以及分块方式（`F/R/V/P`）。空字符串等价于全部默认值。 |
| `chunk_options` | 入队时**冻结**的分块器参数快照（精简字典：只保留 `process_options` 选中的那一路策略子字典，其它策略丢弃）。由 SDK 路径调用方传入或由 `resolve_chunk_options(self.addon_params, process_options=…)` 从实例字段（含 env 默认）兜底（见 §5.1）。`process_options` 选哪种分块策略（F/R/V/P），`chunk_options` 决定那一路分块器使用哪些参数。下游 `process_single_document` 在分块前从此字段读取专属 kwargs；持久化保证 env 变化、续跑、重启后老文档行为可复现。重新解析时与 `process_options` 一同改写。 |

`pending_parse` 表示文件已经入队，但还没有完成抽取。抽取成功后会改写为 `raw` 或 `lightrag`，并补齐 `content_hash`。抽取失败时保留 `pending_parse` 和空 `content`，便于后续排查和重试。

> `doc_status` 中也同步保存原始 `file_path`（含 hint）、`canonical_basename` 与 `content_hash`，作为 `get_doc_by_file_basename` / `get_doc_by_content_hash` 的查重索引来源。`get_doc_by_file_basename` 内部把传入参数先经 `canonicalize_parser_hinted_basename` 规范化后再与 `canonical_basename` 比对，因此 `abc.docx` 与 `abc.[native-iet].docx` 总是命中同一文档。
> `process_options` 同时镜像写入 `doc_status.metadata["process_options"]`，便于管理 UI 直接展示当前文件的处理策略。

### 6.2 `__parsed__` 目录结构

`__parsed__` 是输入目录旁的归档与分析结果目录。它同时保存已经处理过的原始文档，以及结构化解析产生的 LightRAG Document （lightrag格式）的文件和图片等资源。

- 原始文件归档：`legacy` 本地抽取成功并入队后，原文件会移动到同级 `__parsed__` 目录；`native` / `mineru` / `docling` 会先保留原文件供 pipeline 解析，解析成功并写入 `full_docs` 后再移动到 `__parsed__`。**归档时保留原始文件名（含 `[hint]`）**，例如 `report.[native-iet].docx` 归档为 `__parsed__/report.[native-iet].docx`，便于追溯用户最初的命名与处理选项。
- 分析结果目录：结构化解析结果会写入以**规范化文件名**（去掉 `[hint]`）加 `.parsed` 后缀命名的子目录，避免与归档原文件同名冲突，并保证当文件名 hint 或处理选项变化时同一逻辑文档继续指向同一目录。例如 `report.docx`、`report.[native].docx`、`report.[native-iet].docx` 的分析结果都写入 `__parsed__/report.docx.parsed/`。
- 分析结果文件：LightRAG Document blocks 文件以及 sidecar 都使用规范化文件名的主干命名，例如 `__parsed__/report.docx.parsed/report.blocks.jsonl`；同一目录下还可能包含 `report.tables.json`、`report.drawings.json`、`report.equations.json` 和 `report.blocks.assets/` 图片资源目录。**sidecar 是否生成由文档内容决定**：解析器只在文档实际包含表格/图片/公式时写出对应文件。这是模态可用性的唯一信号 —— 引擎不需要在 meta 中声明能力。`i`/`t`/`e` 选项只决定下一阶段是否对已存在的 sidecar 调用 VLM 做摘要分析。
- 解析失败时，原文件不会移动，便于修复配置后重新处理。
- `/documents/scan` 扫描到同名且已 `PROCESSED` 的文件时，该输入文件会被视为已处理并移动到 `__parsed__`，不会作为新文档入队。
- `/documents/scan` 同一次扫描中发现多个规范化后同名的文件时，**先取得 canonical claim 的文件获胜**（即目录流式遍历时先到达的那个）。可丢弃磁盘 spool 中的 scan-wide UNIQUE claim 会在任何候选落库前把后续变体归档并输出 warning。例如 `abc.docx` 和 `abc.[native].docx` 同时存在时只处理先被遍历到的那一个。hint 只决定引擎，不再提供调度优先级。
- 扫描或解析过程中发现内容 hash 重复时，该输入文件同样会移动到 `__parsed__`；本次 `doc_status` 保留为 `FAILED duplicate` 以便追踪。
- 移动文件只作用于当前输入文件，不会覆盖或移动既有文档源文件。若目标目录已存在同名文件，系统会自动追加 `_001`、`_002` 等编号，例如 `report.pdf` 会依次归档为 `report_001.pdf`、`report_002.pdf`。若分析结果目录名已被普通文件占用，也会追加编号，例如 `report.docx.parsed_001/`。

### 6.3 原始产物包（`<base>.*_raw/`）

有三个引擎会在 `<规范文件名>.parsed/` 旁保留一份原始产物包，使得重新解析未变更的文件时不必重复该引擎那一步昂贵的工作。三者遵循同一套设计：产物目录里存放原始产物，外加一个 `_manifest.json`，它同时充当原子成功标记与缓存键。

| 产物包 | 由谁写出 | 避免的昂贵步骤 | 内容 |
| --- | --- | --- | --- |
| `<base>.native_raw/` | `native`（markdown / textpack 路径） | 下载并栅格化外部 `http(s)` 图片 | 每张缓存图片一个文件，名为 `sha256(url)[:16]`，存放栅格化之后的字节 |
| `<base>.mineru_raw/` | `mineru` | 与 MinerU 服务的上传 / 轮询 / 下载往返 | `content_list.json`，以及可选的 `full.md` / `middle.json` / `layout.pdf` / `images/` |
| `<base>.docling_raw/` | `docling` | 与 docling-serve 的上传 / 轮询 / 下载往返 | `<base>.json`（DoclingDocument，含 `pages[].image` base64）、供人工查看的 `<base>.md`、以及被 `pictures[*].image.uri` 引用的 `artifacts/image_*.png` |

产物包是 `.parsed/` 的**兄弟目录**而非子目录，因此能在每次重新抽取前的目录清空中存活下来。

设计目标：

- **避免重复那一步昂贵的工作。** 重新解析时先用源文件的 hash 与大小校验 `_manifest.json`；命中则完全跳过网络工作，直接把已存产物送进 adapter 与 sidecar writer。
- **保留诊断材料。** 当引擎解析有误、或下游 sidecar 字段异常时，产物包就是与"引擎实际返回了什么"对照的地方。
- **支持对象溯源**（外部引擎）。由 MinerU 产出的 `drawings.json` / `tables.json` / `equations.json` 会在 `self_ref` 里记录 `content_list.json#/N`，可据此回查对应的 MinerU 原始对象及其 `page_idx` / `bbox`。
- **上传文件名去 hint**（MinerU）。当源文件名带有 `[mineru-...]` / `[-iet]` 之类处理 hint 时，调用 MinerU API 使用的是规范化后的文件名，避免带 hint 的文件名出现在返回的产物包里。

Docling 产物包目录结构：

```text
__parsed__/<base>.docling_raw/
├── _manifest.json
├── <base>.json        # DoclingDocument JSON（含 pages[].image base64）
├── <base>.md          # Markdown 形式，供人工查看
└── artifacts/
    └── image_*.png    # 被 pictures[*].image.uri 引用的图片资源
```

生命周期，三个产物包完全一致：

| 操作 | 行为 |
|---|---|
| 首次解析 | 取回产物，然后原子写入 `_manifest.json`。docling 的取回过程是 `POST /v1/convert/file/async` → 长轮询 `/v1/status/poll/{task_id}?wait=N` → `GET /v1/result/{task_id}` → 安全解压 zip（拒绝绝对路径与 `..`）。 |
| 重新解析（缓存命中） | 不调用外部服务，不重写产物；仅重跑 adapter + writer 重新生成 sidecar（这正是 adapter 升级代价很低的原因）。 |
| 重新解析（缓存未命中） | 清空目录，重新取回并写 manifest。 |
| `DELETE /documents` 且 `delete_file=True` | `*.parsed/`、原始产物包、源文件一并删除。 |
| `DELETE /documents` 且 `delete_file=False` | 保留全部产物，仅删除 doc_status 与 KG 数据。 |
| `clear_documents` / 整体清空 `__parsed__` | 随之一并清除。 |
| scan 周期 | **不会**回收孤立的产物包——只有用户显式删除时才移除，避免误扫掉调试现场。 |

强制重解析（完全绕过缓存）：`LIGHTRAG_FORCE_REPARSE_NATIVE` / `LIGHTRAG_FORCE_REPARSE_MINERU` / `LIGHTRAG_FORCE_REPARSE_DOCLING`（§3.7）。

并发安全：LightRAG 强制同一 workspace 内 `canonical_basename` 唯一（upload / enqueue 时返回 HTTP 409），加上流水线对每个文档的串行处理，任何产物包都不会被并发写入，无需额外加锁。

#### manifest 失效条件

以下任一项不匹配即缓存未命中。三个产物包共有的：

- 源文件大小或 sha256 与 manifest 不符；
- 记录在案的产物缺失，或其大小 / sha256 与 manifest 不符。

在此之上各引擎特有的条件：

| 引擎 | 额外的失效条件 |
| --- | --- |
| `native` | 下载参数签名变化——即 §3.3 的大小 / SVG 像素 / CIDR 相关选项 |
| `mineru` | `MINERU_ENGINE_VERSION` 与记录的 `engine_version` 不一致；`MINERU_API_MODE` 与记录的 `api_mode` 不一致；当前模式对应的端点（`MINERU_OFFICIAL_ENDPOINT` / `MINERU_LOCAL_ENDPOINT`）与记录的 `endpoint_signature` 不一致；`content_list.json` 大小或 sha256 不符；任一记录在案的非关键文件（图片、`middle.json` 等）大小不符 |
| `docling` | `DOCLING_ENDPOINT` 与记录的 `endpoint_signature` 不一致；已设置 `DOCLING_ENGINE_VERSION` 且与记录的 `engine_version` 不一致；`options_signature` 不一致——它既覆盖可调环境变量（`DOCLING_DO_OCR` / `DOCLING_FORCE_OCR` / `DOCLING_OCR_ENGINE` / `DOCLING_OCR_PRESET` / `DOCLING_OCR_LANG` / `DOCLING_DO_FORMULA_ENRICHMENT`），**也**覆盖硬编码常量 `pipeline` / `target_type` / `to_formats` / `image_export_mode`，把它们写进签名是为了将来改动这些值时不会悄悄复用旧产物包；主 JSON 缺失或大小 / sha256 不符；`artifacts/` 中任一图片缺失或大小不符 |

> **"任一侧为空即跳过"** 适用于两个外部引擎的 `engine_version` 与 `endpoint_signature`。若写 manifest 时该字段为空（例如首次解析时未配置 `MINERU_ENGINE_VERSION`），或当前环境变量未设置，则跳过该项检查。因此，在产物包已经存在**之后**才设置版本号变量，并不会追溯性地让它失效——这种情况需要用对应的 `LIGHTRAG_FORCE_REPARSE_*` 标志。

## 7. 同名与重复文档

文件上传、目录扫描和文本接口都会查重。同一份内容重复入库会浪费 LLM 配额，还会在知识图谱里造出重复实体，所以 LightRAG 用文件名和内容两条规则查重：命中任一即判为重复，本次记录写为 `FAILED`，**已有文档永远不会被覆盖**。所有查重都在分块、实体抽取和图谱写入**之前**完成——但这不等于"落库之前"：`native` / `mineru` / `docling` 必须先持久化待解析记录、再持久化解析结果，才拿得到内容 hash，所以这条路径上会短暂存在一条随后被撤销的记录（见 §7.1、§7.2）。

### 7.1 两条查重规则

**规则一：文件名重复。** 只比对文件名本身（不含目录，也不含 workspace 路径），因此 `/data/a.pdf`、`inputs/a.pdf` 和 `a.pdf` 是同一个名字。比对前会先剥掉能识别的 `[引擎-选项]` hint，所以 `abc.docx`、`abc.[native].docx`、`abc.[native-iet].docx` 互相视为同名；无法识别的 hint 不剥离，`abc.[draft].docx` 仍是另一个名字。只要 `doc_status` 里已有同名记录，**无论它处于 `PENDING`、`PARSING`、`ANALYZING`、`PROCESSING`、`FAILED` 还是 `PROCESSED`**，本次都算重复。

**规则二：内容重复。** 比对的是**抽取之后的正文**，不是原始文件的字节。所以换个文件名、甚至换成另一种格式，只要抽出来的内容一致就算重复。取值口径按内容格式区分：

| `parse_format` | 内容 hash 取自 |
| --- | --- |
| `raw` | 编码规范化之后的正文文本 MD5 |
| `lightrag` | sidecar 中 `*.blocks.jsonl` 文件的 MD5（相对路径按 `INPUT_DIR` 解析） |
| `pending_parse` | 暂不计算，等真正解析完成后补上（避免按空内容误判） |

这也决定了内容查重发生在什么时候：`legacy` 在本地抽完文本、入队时就能判；`native` / `mineru` / `docling` 要等解析真正完成才判，此时本次记录会**停在建库之前**——不分块、不抽实体、不写图谱，并删掉本次临时写入的 `full_docs`。

两条补充规则：

- **文本接口**（`/documents/text`、`/documents/texts`）必须提供有效的 `file_source`，并按它的 basename 判定同名；缺失时直接返回 400。
- **无来源文档**（SDK 调用 `insert` / `ainsert` 时不传 `file_paths`，`file_path` 记为 `unknown_source`）不参与同名查重——`unknown_source` 只是占位名，两份无来源文档不会因为它互相冲突——但**仍然按正文内容查重**：重复插入同一段正文依旧会被判为重复。空字符串与 `no-file-path` 同理。

同一批入队内部、以及并发发起的两次入队，都不会双双穿透查重：后到的那条一定会被识别为重复并写为 `FAILED`。

### 7.2 你会看到什么

| 症状 | 含义 |
| --- | --- |
| 上传返回 409 `Document storage already contains '<name>' (Status: …)` | `doc_status` 里已有同名记录（任何状态，包括 `FAILED`） |
| 上传返回 409 `Input directory already contains a file with the same canonical basename …` | 库里没有记录，但 `INPUT_DIR` 里还留着同名文件 |
| 文档列表多出一条 `FAILED`，`error_msg` 为 `File name already exists. Original doc_id: …, Status: …` | 批量入队或扫描时命中了文件名重复 |
| `FAILED`，`error_msg` 为 `Identical content already exists under another filename. Original doc_id: …` | 命中了内容重复 |
| `FAILED`，`error_msg` 为 `N existing documents already share this file name …` | 历史遗留的"一个文件名对应多条主记录"冲突，需要先按 doc id 修复 |

上传路径是 **fail-fast** 的：前两种情况直接 409，不写文件、也不在 `doc_status` 里留下任何痕迹。后三种会在文档列表里留下记录，但**记录形态不同**——运维时不能只按 `dup-` 前缀查找：

- **入队时发现的重复**（同名，或 `legacy` 路径上正文 hash 已存在）：新建一条 doc id 以 `dup-` 开头的 `FAILED` 记录，原文档不受影响。`metadata` 带 `duplicate_kind`（`filename` / `content_hash`）、`original_doc_id` 和 `original_track_id`，指向保留下来的那份正本。
- **解析完成后才发现的内容重复**（`native` / `mineru` / `docling` 入队时还没有正文 hash）：**不会**新建 `dup-*` 记录，而是把本文档原有的 `doc-*` 记录就地改成 `FAILED`，写入 `metadata.is_duplicate=true` 与 `duplicate_kind=content_hash`，删掉本次临时写入的 `full_docs`，并把源文件归档到 `__parsed__`。
- **`filename_conflict`**：库里已有多条主记录占用同一个文件名，系统**故意不替你选正本**，因此这条记录上的 `original_doc_id` 并不指向某个"保留下来的正本"——必须先按 [§7.3](#73-怎么处理) 修复冲突。

### 7.3 怎么处理

- **想用新内容替换同名文档**：先删除（`POST /documents/delete_document`，或 WebUI 文档列表里的删除对话框），再上传。上传永远不会覆盖已有记录。
- **想清掉重复记录**：`dup-*` 记录、以及解析后被就地改成 `FAILED` 的那条原记录，都是**惰性记录**——没有正文内容，`/documents/scan` 与 `/documents/reprocess_failed` 都会跳过它们，既不会重跑，也不会再失败一次。要清掉只能显式删除；留着不管也不影响检索。
- **只是想换引擎或换处理选项重跑同一份文件**：查重不是障碍，被冻结的配置才是——见 [§8.3](#83-出错后如何重试) 与 [§9.3](#93-分支-b已抽取)，这种情况必须删除后重新上传。
- **`filename_conflict`（一名多记录）**：用 `GET /documents/source_conflicts` 列出冲突，`POST /documents/source_conflicts/repair` 指定要保留的 `primary_doc_id`（默认 dry-run，拿到 `candidate_count` / `fingerprint` 后回填提交；候选集有变动则返回 409），或离线执行 `python -m lightrag.tools.source_conflict_repair`。修好之后再扫描。

### 7.4 目录扫描的特殊处理

`/documents/scan` 用的是同一套查重口径，但它同时面对"磁盘上的文件"和"库里的记录"两侧，所以对同名文件有额外的自动处理——目的就是让"改好配置、修好源文件，再扫一次"能直接生效，而不必逐个手工删除：

| 库里的记录 | 扫描时的动作 |
| --- | --- |
| 没有同名记录 | 作为新文档入队；整批按文件修改时间从旧到新排序 |
| 已 `PROCESSED` | 不重复处理；源文件归档到 `__parsed__`，并落一条 warning |
| `FAILED`，且确认从未成功抽取出内容 | 删掉这条失败记录，把文件当作新文档重新走一遍——"修好源文件再扫一次"靠的就是它 |
| 处理到一半被中断（`PENDING` / `PARSING` / `ANALYZING` / `PROCESSING`） | 记录和源文件都原样保留，由流水线接着跑，不重新抽取 |
| 同名的另一个物理文件 / 记录来源不明 / 一名多记录 | 既不入队也不删记录，只归档或告警——这几类需要人工判断，见 [§7.3](#73-怎么处理) |

两点运维须知：

- **扫描被中断（取消、崩溃、重启）时一条都不会入队。** 发现与入队分两步走，源文件始终留在 `INPUT_DIR`，下次扫描重新发现即可。唯一回不来的是上表第三行已经删掉的失败记录——文件会作为新文档重跑，但那条留作人工复核的记录没了。
- **`SCAN_SPOOL_DIR` 必须指向真实可写的本地磁盘。** 扫描期间的候选清单落在 `SCAN_SPOOL_DIR`，未设置时取 `WORKING_DIR/scan_spool`，两者都再按 workspace 分子目录。不要指向 tmpfs（很多 Linux 主机上 `/tmp` 是内存盘）；`WORKING_DIR` 本身是网络卷时应显式设置这个变量。目录不可用时扫描**直接失败且一条都不入队**，错误信息会点名该变量，输入文件原封不动，改完配置重扫不丢任何东西。

## 8. 运行控制：边跑边传、停止与重试

本章回答三个运行期问题：流水线跑起来之后还能不能继续上传、怎么把它停下来、以及文档失败之后该怎么重试。并发调优参数与请求准入限制放在本章末尾两节。

### 8.1 处理进行中还能做什么

| 你想做的事 | 正在处理文档 | 扫描的分类阶段 | 正在清空 / 删除 | 手动重试正在排空 |
| --- | :-: | :-: | :-: | :-: |
| 上传文件 / 插入文本 | ✅ 允许 | ❌ 409 | ❌ 409 | ❌ 409 |
| `/documents/scan` | ⛔ 拒绝 | ⛔ 拒绝 | ⛔ 拒绝 | ⛔ 拒绝 |
| 删除文档 / 清空 | ⛔ 拒绝 | ⛔ 拒绝 | ⛔ 拒绝 | ⛔ 拒绝 |
| 检索 / 查询 | ✅ 不被拒绝 | ✅ 不被拒绝 | ⚠️ 不被拒绝，但结果不保证 | ✅ 不被拒绝 |

要点：

- **长批次处理期间可以继续上传，不用等。** 这是最常被误解的一点：文档处理本身**不**阻塞上传。新文档写完记录后由运行中的流水线自动接手——可能被并进当前批次，也可能在批次边界被接上，两种都不需要你做任何额外操作。
- **扫描、删除、清空需要流水线空闲。** 扫描要一边读库一边决定每个磁盘文件的去向（[§7.4](#74-目录扫描的特殊处理)），删除和清空会直接拆掉存储，这两类都不能和并发写入交错。被拒时 `/documents/scan` 返回 HTTP 200 但 `status="scanning_skipped_pipeline_busy"`，删除 / 清空返回 `status="busy"`——都不是报错，等空闲后重试即可。
- **查询不受流水线准入约束，但清空 / 删除期间不保证结果。** 查询接口不检查流水线状态，所以不会像上传那样被 409 拒绝；但 `/documents/clear` 会用 `asyncio.gather` 并发 drop 文本分块、实体 / 关系 / chunk 向量、知识图谱、`full_docs` 与 `doc_status` 等十余个存储，而查询既不持有一致性快照，也不会等它结束——中途发起的查询可能拿到空结果、部分结果或存储报错。"请求会被受理"和"能返回一致结果"是两回事：破坏性操作期间应当等它跑完再查。
- **上传被拒时按报错文案定位原因**，三种 409 各自对应一种状态：
  - `Document scan is classifying files. …` —— 扫描正在分类阶段。
  - `Pipeline is clearing or deleting documents. …` —— 正在清空或删除文档。
  - `A retry of failed documents is draining the pipeline. …` —— 手动重试正在排空流水线（[§8.3](#83-出错后如何重试)）。
- 上传返回 **413 / 429** 与流水线是否繁忙无关，那是请求准入限制，见 [§8.7](#87-准入与请求限制)；返回 **503** 见 [§8.5](#85-所有接口都返回-503recovery_required-栅栏)。

### 8.2 如何停止正在运行的流水线

```text
POST /documents/cancel_pipeline      # 无请求体
```

WebUI 入口是文档管理页「流水线状态」对话框里的**「中断」**按钮，仅在流水线运行中且尚未请求中断时可点。接口返回 `{"status": "cancellation_requested" | "not_busy", "message": …}`：`not_busy` 表示当前没有在跑，无需中断。

**返回 200 不代表已经停了。** 它只是置了一个中断请求标志。中断是**协作式**的：只在阶段之间、批次边界、以及多模态分析每 0.5 秒的轮询点生效，**不会打断已经发出的 LLM 调用**。所以点完之后还要等当前这一步跑完，`GET /documents/pipeline_status` 上的 `cancellation_requested` 可以确认请求已经收到。

停下来之后，文档会落在这几种状态：

| 文档当时的位置 | 停止后的状态 |
| --- | --- |
| 已经处理完 | `PROCESSED`，正常保留，不受影响 |
| 已被本批取走（正在解析 / 分析 / 抽取，或在队列里排队） | `FAILED`，`error_msg` 形如 `User cancelled during parse: <文件名>` |
| 还没被本批取走 | 保持 `PENDING`，下一次触发时继续 |

几点补充：

- **已完成的部分工作不会浪费。** LLM 缓存和已经分析成功的多模态条目都会落盘，之后重试会命中缓存，不会重复花钱。
- ⚠️ **被标成 `FAILED` 的文档不会自动重试**，必须用 [§8.3](#83-出错后如何重试) 的显式重试手段捞回来。
- **中断期间仍可继续上传。** 新上传的文档会在这一轮退出之后，由下一轮自动接上。
- **什么时候它不管用**：扫描的分类阶段不算"流水线繁忙"，此时调用会返回 `not_busy`，而且目前没有中断扫描任务的接口——等它跑完即可。删除 / 清空作业**可以**被中断，粒度是逐个文档：已删除的保持删除，剩余的会在响应里报告为未删除。

**直接重启或杀掉服务**是另一种停法，后果不同：

- 进行中的文档会停在 `PARSING` / `ANALYZING` / `PROCESSING`，**不会**被写成 `FAILED`。
- 这些中断态在下一次流水线启动时会**自动恢复为 `PENDING` 重跑**，不需要手工干预。
- 但**服务启动本身不会自动开跑**：需要一次上传，或调一次 `POST /documents/scan` 来触发。所以重启后看到文档卡在 `PROCESSING` 不必惊慌，触发一次即可。

### 8.3 出错后如何重试

失败文档有三种捞回方式，按覆盖面从大到小：

| 手段 | 做什么 | 什么时候用 |
| --- | --- | --- |
| `POST /documents/scan`（WebUI「扫描/重试」按钮） | 先把所有可恢复的 `FAILED` 重置为 `PENDING`，再扫描 `INPUT_DIR` 里的新文件；还能处理"从未成功抽取出内容"的失败记录（删记录 + 把文件当新文档重跑） | **首选**，覆盖面最广 |
| `POST /documents/reprocess_failed` | 只按存储里的记录重试 `FAILED`，不做目录发现；但解析阶段就失败的文档会**重新解析**，那时仍要读取记录指向的源文件 | 不想顺带触发一次全目录扫描时 |
| 删除 + 重新上传 | 彻底重来一遍 | 需要换引擎 / 换处理选项，或上面两种都救不回来 |

四条必须知道的口径：

1. **每次重试请求对每个文档只给一次机会。** 再失败就停在 `FAILED`，等下一次显式重试，不会自旋。
2. **自动续跑不碰 `FAILED`。** 上传新文件触发的那一轮只会恢复中断态（`PENDING` / `PARSING` / `ANALYZING` / `PROCESSING`）；`FAILED` 只能靠上表前两个显式入口。
3. **是否重新解析，取决于正文有没有抽取成功。** 已经抽出正文的文档不会重解析：重试从多模态分析阶段起跑，先清掉上一轮写进去的 chunks 与图谱贡献再重做（细节见 [§9.3](#93-分支-b已抽取)），所以"修好 VLM 配置 / 等限流恢复之后重试"是有效的。而**在解析阶段就失败的文档**（`full_docs` 里只留着 `pending_parse` 占位记录）会被重置为 `PENDING` 后重新进入解析阶段，再次读取 `INPUT_DIR` 里的源文件、再次调用对应引擎——修好 MinerU / Docling 后重试因此有效，但源文件已被删除时这类重试会再次失败。
4. **重试不会改引擎与处理选项。** `parse_engine`、`process_options`、`chunk_options` 在入队那一刻就冻结进记录了；改 `.env` 或改文件名 hint 只对新上传生效。

哪些情况重试有效、哪些必须删除重传：

| 失败原因 | 就地重试？ |
| --- | --- |
| LLM / VLM / 存储 / 网络临时故障，或撞上限流 | ✅ 直接重试 |
| 被用户中断（`User cancelled during …`） | ✅ 直接重试 |
| VLM 没配好（`VLM analysis required but VLM role is not available`） | ⚠️ 把 VLM 配好后重试有效；若想改成不做图片分析（去掉 `i`），必须删除重传 |
| 外部解析服务没起来 / 端点写错 | ⚠️ 修好配置后重试；命中原始产物包缓存时不会重复调用外部服务（§6.3） |
| 想换解析引擎、换分块策略、改 `i/t/e/!` | ❌ 删除后重新上传（[§9.3](#93-分支-b已抽取)） |
| 文件名 hint 写错、分块参数非法（`[File Extraction]` 开头、从未抽出正文的记录） | ❌ `reprocess_failed` 会跳过它；改好文件名后用 `/documents/scan`，或删掉记录后重传 |
| 扫描版 PDF 走了 `legacy`，抽不出文字 | ❌ 先把该后缀路由到 `mineru` / `docling`，再删除重传（[§3.2](#32-使用-legacy-内容抽取器)） |
| 重复文档 `dup-*` | ❌ 惰性记录，重试不会动它，直接删除（[§7.3](#73-怎么处理)） |
| 删除文档时报 409（缺少 recovery anchor） | ❌ 原样重试仍会被拒；先跑 `audit_kg_integrity(..., apply=True)` 修复 |

`/documents/reprocess_failed` 还有两点值得知道。它会先冻结入口、把流水线排空到空闲——这段时间上传返回 409、`/documents/scan` 被拒——再在没有任何 worker 运行的前提下把 `FAILED` 改回 `PENDING`，然后恢复正常处理；文档数量多时这个过程需要一点时间。它可能返回 429（未确认的重试请求过多，上限由 `MAX_UNACKED_MANUAL_RETRIES` 控制）或 503（栅栏已升起，或正在清空 / 删除）；排空迟迟到不了空闲时会升起栅栏，见 [§8.5](#85-所有接口都返回-503recovery_required-栅栏)。

### 8.4 删除文档：两个勾选项的区别

删除对话框里的两个复选框默认都不勾，它们决定磁盘上的东西是否一起清掉：

- **都不勾**：只删存储里的状态——chunks、向量、图谱贡献、`doc_status`、`full_docs`。磁盘上的源文件、`__parsed__` 里的归档件与 `.parsed/` sidecar、外部引擎的原始产物缓存**全部保留**。
- **勾「同时删除上传文件」**（API 参数 `delete_file=true`）：连同 `INPUT_DIR` 里的源文件、`__parsed__` 下的归档件与 `<base>.parsed/` sidecar，以及 `<base>.mineru_raw/` / `<base>.docling_raw/` / `<base>.native_raw/` 原始产物缓存一起删除。**想换引擎重跑、或想让外部引擎真的重新解析一次，必须勾这个**，否则重新上传会命中缓存拿回旧结果（[§3.7](#37-解析缓存与强制重解析)、§6.3）。
- **勾「同时删除实体关系抽取 LLM 缓存」**：额外清掉该文档抽取阶段的 LLM 缓存，重传时会真实重跑 LLM 而不是命中缓存。想验证换了模型或提示词之后的效果，需要勾它。

### 8.5 所有接口都返回 503：`recovery_required` 栅栏

有些故障会让 workspace 处于"继续下去只能靠猜"的状态。管线不做这种猜测，而是升起 `recovery_required` 栅栏：此后**所有**写操作（upload / text / scan / 手动重试 / delete / clear）一律返回 **HTTP 503**，直到运维显式解除。三种情况会升起它——其中 1 和 3 依赖跨进程的死进程检测，只在 **Linux + Gunicorn 多 worker** 部署下才会发生（单进程 uvicorn 的协调状态随进程一起消失）：

1. **worker 在 `custom_chunks` / `delete` / `clear` 执行途中死亡。** 这些操作可能已经半提交，不能简单重跑。（`processing` / `scan` 的执行者死亡是可重跑的，会被静默回收，不设栅栏。）
2. **手动重试的排空无法到达空闲。** 有两种卡法：同一批文档反复回来且状态毫无变化（再重扫只会自旋），以及排空根本无法推进的文档——持有**未完成 custom-chunk 操作**的行，只有 `/documents/scan` 的回滚能处理它们。两种情况下重置都不执行，重试请求保持未确认（那一次机会仍然欠着），栅栏消息里带阻塞文档 ID 的有界样本。`recovery_kind` 区分二者：`manual_drain_stalled` 和 `manual_drain_blocked`。
3. **无法判定某个执行者是否已经死亡。** 回收一个执行权必须先证明其所属进程确已死亡；不带进程身份的记录永远无法证明，管线不靠猜回收，改由栅栏提供出口。

`GET /documents/pipeline_status` 会返回 `recovery_required`（布尔）、`recovery_kind`（粗粒度原因）和 `recovery_message`（与 503 相同的文案，某些原因还带阻塞文档的有界样本）。

解除栅栏：

```text
POST /documents/recovery/force_reset
```

这是**不安全的人工覆盖**——它不修复任何东西。除栅栏之外，它还会**取消该 workspace 排队中的手动重试请求**，这一点是必需的而非附带：只要还有排队请求，`/documents/scan` 就会拒绝执行（scan 自己也要跑独占的 `FAILED` 重置，不能插队），所以只清栅栏会让恢复路径照样被堵。响应里返回 `cancelled_manual_retries`。不会丢文档——失败文档仍是 `FAILED`，由下一次重试请求或 scan 自己的重置处理。由于两半都是必需的，这个调用是**全有或全无**的：如果排队请求无法取消，接口返回 **503** 且栅栏保持不变，这样可以重试来完成恢复，而不是让 API 报告一次并未发生的恢复。

恢复顺序：

| 原因 | 操作 |
|---|---|
| `manual_drain_blocked` | `POST /documents/recovery/force_reset`，然后 `POST /documents/scan` —— scan 会回滚未完成的操作**并且**自己执行 `FAILED` 重置，无需再单独调重试。 |
| `manual_drain_stalled` | `POST /documents/recovery/force_reset`，然后排查 `recovery_message` 中列出的文档——它们卡住的原因这个栅栏无法给出。处理完后重新调 `POST /documents/reprocess_failed`。 |
| worker 死于 `custom_chunks` / `delete` / `clear` 途中 | 不要直接 `force_reset`——按下面「半提交存储怎么修」处理。 |

**能不能直接重启服务了事？** 判断标准只有一条——**堵塞源在内存里还是在存储里**。重启清掉的只有运行时协调状态（`pipeline_status` 里的栅栏与 owner 记录、ingress 里排队的手动重试请求，都不持久化），写进 `doc_status` / `full_docs` / 存储的东西一样都清不掉。

| 成因 | 重启能否解决 | 说明 |
| --- | :-: | --- |
| owner 生死无法判定 | ✅ 能 | 卡住的 reservation 记录本身就活在跨进程共享状态里，进程组一起重启就没了，且没有任何存储被动过——这是最干净的办法 |
| `manual_drain_stalled` | ⚠️ 多半能 | 栅栏与排队请求随重启消失；那些"反复回来又不变状态"的活跃行若是死进程遗留的 `PROCESSING` / `PARSING` / `ANALYZING` 孤儿，重启后会被自动重置为 `PENDING` 并在下次触发时重跑，堵塞源自然消失。但重启不诊断根因：若它们是因别的原因每轮都停在同一状态，下一次 `/documents/reprocess_failed` 会再次 stall |
| `manual_drain_blocked` | ❌ 不能 | 阻塞源是 `doc_status.metadata` 里未完成的 custom-chunk 操作日志，它**是持久化的**，重启原样还在。重启只清掉栅栏和排队请求（这一步仍是必要的，否则排队请求会让 `/scan` 拒绝自己的 reservation），真正的回滚必须由 `POST /documents/scan` 执行 |
| worker 死于 `custom_chunks` / `delete` / `clear` | ❌ 不能 | 半提交的是存储本身，见下 |

所以重启相当于一次"温和版 `force_reset`"：两者都清栅栏与排队请求、都不修复任何东西，区别是重启还会顺带把中断态文档重置为 `PENDING`（这正是 stalled 常能自愈的原因），代价是一次服务中断。能停服务就优先重启；不能停、且已确认存储未被动过（成因 2 / 3）才用 `force_reset`。

#### 半提交存储怎么修：两个离线工具与 `force_reset` 的分工

栅栏本身**不持久化**（它活在 `pipeline_status` 里），**整体重启服务就会清掉它和排队请求**。这一点决定了 `force_reset` 该用在哪：

- **成因 2 / 3 用它。** 这两种情况下重置根本没执行、存储一字未动，卡住的只是调度——为此停服务不值得。
- **成因 1 尽量别用它。** 它只清标志、不修任何东西；清完之后所有写入（包括并发上传）立刻在一个可能半提交的存储上重新开工。而下面两个离线工具本来就要求先停服务——停下来栅栏自然消失，`force_reset` 根本不必出场。

两个离线工具能修什么、修不了什么：

| 工具 | 真相源与修复对象 | 修不了 | 成本 |
| --- | --- | --- | --- |
| `python -m lightrag.tools.kg_integrity_repair` | 沿 graph → chunk `source_id` → `text_chunks` → `full_doc_id` 反查，重建缺失或不可用的 `full_entities` / `full_relations` 锚点行；也能给确实什么都不拥有的文档写空锚点行 | 图↔向量漂移、`doc_status` / `full_docs` 自身、custom-chunk 日志、栅栏本身 | 不调 LLM / embedder，近乎免费 |
| `lightrag-rebuild-vdb` | drop 后按图重建 `entities_vdb` / `relationships_vdb`、按 `text_chunks` 重建 `chunks_vdb`；能清掉反向孤儿（向量里有、图里没有），这是增量修复做不到的 | 锚点行、`doc_status`、图本身的正确性、栅栏本身 | 全量重新嵌入，真金白银 |

两个顺序陷阱，弄反了会把可修复的数据变成不可修复的：

- **`kg_integrity_repair` 必须在任何进一步删除之前跑。** 它只能从**幸存的** chunk provenance 重建锚点；半截的 delete 若已经删掉 `text_chunks`，那部分贡献就成了不可恢复孤儿，工具只能报告、绝不自动改。先跑一次报告模式（不加 `--apply`）没有任何理由不跑。
- **`lightrag-rebuild-vdb` 必须最后跑。** 它把图与 `text_chunks` 当作真相：半截 delete 留下的、本该删掉的图对象，会被它忠实地重新嵌入回向量库。它保证的是向量与图**一致**，而不是图本身**正确**。所以先把图这一侧的真相定下来，再考虑重建向量。

因此成因 1 的完整顺序是：

1. **停服务**（栅栏随之消失，别急着 `force_reset`）。
2. `python -m lightrag.tools.kg_integrity_repair --verbose` 先看报告：锚点缺口、以及已经救不回来的不可恢复孤儿。有缺口再 `--apply`。
3. 起服务，把没做完的那次操作重跑一遍——whole-document purge 带 journal，会从已完成的阶段之后接着做；`clear` 直接重跑；`custom_chunks` 的回滚由 `POST /documents/scan` 触发。
4. 稳定之后若怀疑图↔向量漂移，再停服务跑 `lightrag-rebuild-vdb`，先用菜单里的只读一致性检查决定值不值得付嵌入成本。

两个工具都必须在服务已停止、workspace 空闲时运行（并发写入会让它们读到移动的目标）；`lightrag-rebuild-vdb` 还必须使用与服务相同的 `.env`，否则重建出的向量落在另一个嵌入空间里。详见 [README_KG_INTEGRITY_REPAIR.md](../lightrag/tools/README_KG_INTEGRITY_REPAIR.md) 与 [README_REBUILD_VDB.md](../lightrag/tools/README_REBUILD_VDB.md)。

### 8.6 流水线并发参数

上面几节讲的是"谁能写"的正确性问题，本节这一组参数解决的是"同时跑几个 worker"的吞吐量问题。流水线分为 3 个阶段，每个阶段的 worker 池数量独立可调：

```
          ┌─ parse_queues["native"]  ─► [native 池  × N1] ─┐   ← legacy 共享此池
PENDING ─►├─ parse_queues["mineru"]  ─► [mineru 池  × N2] ─┼─► q_analyze ─►[analyzer × N4] ─► q_process ─►[processor × N5]
          ├─ parse_queues["docling"] ─► [docling 池 × N3] ─┤
          └─ parse_queues[<第三方组>] ─► [自定义并发池]  ──┘   ← 按 ParserSpec.queue_group 动态创建
```

解析队列**按注册表的 `ParserSpec.queue_group` 动态创建**（每批取一次注册表快照）：内置 native/mineru/docling 各占一组，legacy 共享 native 池（本地、无网络），第三方引擎可声明独立组与自定义并发数（见 [ThirdPartyParser-zh.md](./ThirdPartyParser-zh.md)）。入队时根据每个文档的解析引擎（来自 `LIGHTRAG_PARSER` 默认值或文件 hint）把它放入对应解析队列；各解析队列**完全互不阻塞**——mineru 占满不会拖慢 docling 或 native。解析完成后统一进入 `q_analyze`（多模态分析），再进入 `q_process`（实体/关系抽取 + 入库）。

| 环境变量 | 默认值 | 作用 | 调优建议 |
| --- | --- | --- | --- |
| `MAX_PARALLEL_PARSE_NATIVE` | `5` | N1: native 解析（docx / pdf / txt 等纯本地处理）并发 worker 数 | 纯 CPU、内存占用低，可按 CPU 核数提高 |
| `MAX_PARALLEL_PARSE_MINERU` | `2` | N2: MinerU 解析并发 worker 数 | MinerU 占用 GPU/CPU 显著，**默认 2 为适度并发**。资源紧张时可降到 1；本地部署且显存充足时可设 2-3；走 MinerU 官方云端服务时可适当提高（受云端配额限制） |
| `MAX_PARALLEL_PARSE_DOCLING` | `2` | N3: Docling 解析并发 worker 数 | Docling 同样资源敏感，**默认 2 为适度并发**。资源紧张时可降到 1；本地部署且 CPU/GPU 充足时可设 2-3 |
| `MAX_PARALLEL_ANALYZE` | `5` | N4: 多模态分析（VLM 图片 / 表格描述）并发 worker 数 | 直接消耗 VLM 配额。建议 ≤ VLM 服务并发上限 |
| `MAX_PARALLEL_INSERT` | `3` | N5: 实体 / 关系抽取 + 入库阶段并发文档数 | 推荐 `MAX_ASYNC_LLM / 3`，区间 2~10。该阶段每个文档会触发多次 LLM 调用，过高会撞 LLM 限流。同时该值还作为 `asyncio.Semaphore` 用于二次约束（worker 数和信号量值一致） |
| `QUEUE_SIZE_PARSE` | `20` | parse（native/MinerU/Docling）输入队列长度 | 一般无需调整。队列内仅为轻量 doc_id（大文档体在进入 analyze 前已剥离），仅限制 pipeline 一次预派发给 parse worker 的待处理文档数，调整影响很小 |
| `QUEUE_SIZE_ANALYZE` | `100` | analyze 队列（parse → analyze 阶段）的有界容量 | 一般无需调整。极少量大批量任务（成千上万）可适当提高，避免 enqueue 端反压；内存紧张时可调低 |
| `QUEUE_SIZE_INSERT` | `4` | analyze → process 阶段间的队列容量 | process 是流水线中最慢、最耗内存的阶段，队列特意做小，给上游提供反压防止内存堆积 |

**几个要点：**

1. **解析阶段按引擎隔离**，所以混用 native/mineru/docling 时不必担心一种引擎慢拖累另一种。
2. **mineru / docling 默认 2**：两者资源占用高，默认保持适度并发。资源紧张时可降到 1（避免 OOM / 显存竞争 / 失败重试）；如果你部署了多 GPU 或专门的解析服务器，可手动调高。
3. **`MAX_PARALLEL_INSERT` 兼任 worker 池大小和信号量上限**：流水线创建 `Semaphore(max_parallel_insert)`，每个 process worker 在抽取入库前还要拿一次信号量。所以哪怕你把 worker 数手动改大，实际并发上限仍由这个值决定——直接调它就够了。
4. **queue size 与背压**：`QUEUE_SIZE_INSERT=4` 这个偏小的默认值是有意为之——process 阶段慢且占内存，让 analyze 阶段在队列写满时阻塞、再反压到 parse 阶段，避免一次性把成千上万份解析结果堆在内存里。
5. **改后生效方式**：所有参数通过 `.env`（或环境变量）传入，仅在 `LightRAG` 实例构造时读取一次；改完需要重启服务。
6. **分块不随并发增长**：分块在一个专用的单 worker 线程池里执行，目的是不阻塞事件循环，并发度不随 `MAX_PARALLEL_INSERT` 提高，调大并发不会让分块更快。自定义 `chunking_func` 仍在事件循环上执行（它的契约允许触碰运行中的事件循环），CPU 密集的实现应自行 `asyncio.to_thread`。

**典型调优场景：**

- 大量 PDF + 本地 MinerU 单 GPU：`MAX_PARALLEL_PARSE_MINERU=2`、`MAX_PARALLEL_ANALYZE=5`、`MAX_PARALLEL_INSERT=3`（默认即可；显存紧张时把 MINERU 降到 1）。
- 大量 PDF + MinerU 云端服务：`MAX_PARALLEL_PARSE_MINERU=3~5`（视云端配额），其它保持默认。
- 纯 docx / txt（仅走 native）：`MAX_PARALLEL_PARSE_NATIVE=10`、`MAX_PARALLEL_INSERT` 按 `MAX_ASYNC_LLM/3` 推算。
- LLM 限流明显：先降 `MAX_PARALLEL_INSERT`（process 阶段每文档多次 LLM 调用），再降 `MAX_PARALLEL_ANALYZE`（VLM 是独立配额）。

### 8.7 准入与请求限制

在文档触及上述任何机制之前，服务端就可能直接拒收。下面这些变量决定一次上传是被拒绝还是被排队：

| 变量 | 缺省 | 拒绝方式 |
| --- | --- | --- |
| `MAX_UPLOAD_SIZE` | `104857600`（100 MB） | `413` —— 单个上传文件过大 |
| `MAX_REQUEST_BODY_BYTES` | `1048576`（1 MiB） | `413` —— 原始请求体过大。作用于**所有**路由且分档：普通路由取该值，`/documents/text` 与 `/documents/texts` 在**未设置该变量时**取内置 50 MiB，`/documents/upload` 由 `MAX_UPLOAD_SIZE` + 1 MiB 派生。显式配置任意正值（含 1 MiB 默认值）会让它统一作用于所有非上传路由。设为 `0` 则全部关闭 |
| `MAX_TEXTS_PER_REQUEST` | `0`（关闭） | `413` —— 单次 `/documents/texts` 携带的文本条数过多 |
| `MAX_PENDING_DOCUMENTS` | `0`（关闭） | `429` —— 处于 PENDING / PARSING / ANALYZING / PROCESSING 的文档已过多 |

它们的完整语义（含 `MAX_UPLOAD_SIZE` 与反向代理自身请求体上限的关系）见 [LightRAG Server](./LightRAG-API-Server-zh.md)。

另有三个属于流水线自身、而非请求准入的旋钮：

| 变量 | 缺省 | 含义 |
| --- | --- | --- |
| `PIPELINE_SCHEDULING_PAGE_SIZE` | `500` | doc_status 积压扫描的 keyset 分页大小；`0` 关闭分页 |
| `PIPELINE_REQUIRE_STRICT_STORAGE_READS` | `false` | doc_status 后端无法提供严格读时拒绝启动。它直接决定 [§7.4](#74-目录扫描的特殊处理) 里"删掉失败记录、把文件当新文档重跑"那一行能否生效——没有可靠的点读，就什么都不会删 |
| `MAX_UNACKED_MANUAL_RETRIES` | `64` | 每个 workspace 已发布但未确认的手动重试请求上限（[§8.3](#83-出错后如何重试)） |

## 9. 流水线启动时的续跑规则

每次 `apipeline_process_enqueue_documents` 起步时，会拉取所有处于 `PARSING` / `ANALYZING` / `PROCESSING` / `PENDING` / `FAILED` 状态的文档继续处理。续跑路径**根据"内容是否已抽取"分流**，保证同一个文档无论之前进度如何，按当前 `process_options` 续跑都有幂等结果。

续跑规则只对 `doc_id` 已经存在于 `doc_status` 的文档生效。新文件入队需要 §7 的文件查重逻辑，避免新文件挤掉旧的已经成功提取内容的文件记录。

### 9.1 判断"内容已抽取"

读 `full_docs[doc_id]`：

| `parse_format` | 判定 |
| --- | --- |
| `lightrag` 且 `lightrag_document_path` 文件存在 | ✅ 已抽取 |
| `raw` 且 `content` 非空 | ✅ 已抽取 |
| 其它（含 `pending_parse`、记录缺失） | ❌ 未抽取 |

### 9.2 分支 A：未抽取

走完整流水线（注册表派发解析 `get_parser(engine).parse(...)` → `analyze_multimodal` → 分块 → 实体抽取），按 `full_docs.process_options` 决定每一阶段的行为。这是"首次入队"的常规流。

### 9.3 分支 B：已抽取

**一律跳过解析**（不重新调 `parse_*`），从 ANALYZING 阶段重启，并清光旧 chunks / entities 后按当前 `process_options` 重做：

| 子步骤 | 行为 |
| --- | --- |
| 引擎对比 | 若 `process_options` 隐含的引擎 ≠ `full_docs.parse_engine`，**仅 warn**，不重新解析。已抽取的内容是不可变事实，重新跑不同引擎会产生不一致。要切换引擎请先 delete 整个文档再重传。 |
| 旧 chunks / 实体 / 关系清理 | 读 `status_doc.chunks_list` 收集旧 chunk id 集，调 `_purge_doc_chunks_and_kg(doc_id, chunk_ids)`：从 `chunks_vdb` / `text_chunks` 删除 chunk 行；按 `entity_chunks` / `relation_chunks` 反查受影响的实体 / 关系，对失去全部源的条目直接从图谱与向量库删除，对仍有其它文档贡献的条目调 `rebuild_knowledge_from_chunks` 用剩余 chunks 重建；最后删除 `full_entities` / `full_relations` 中本 doc 的索引行。purge 完成后 `status_doc.chunks_list = []` / `chunks_count = 0` 重置，避免后续 state-machine upsert 写回旧 ID。 |
| `analyze_multimodal` | 对已启用模态，每次运行都会重新计算 sidecar item 分析并覆盖已有的 `llm_analyze_result`。由于 LLM cache 的存在重复计算通常会保持语义字段不变，只会重写 `analyze_time` 等运行时字段；cache miss，例如更换模型和提示词等，保存内容才可能与上次不同。 |
| 重新分块 | 按新 `process_options.chunking` 选策略，参数从 `full_docs.chunk_options` 读取（入队快照，不会因续跑被覆盖；env 改动后老文档仍按入队那一刻的参数分块）。LightRAG Document path 在 `process_options=P` 时走 paragraph_semantic，否则按 selector 分发到 F/R/V。 |
| 实体抽取 / KG-skip | 按新 `process_options.skip_kg` 决定 |

> 这条规则保证：用户改 `i/t/e` 重传同名文档（先删旧 doc 再上传带新 hint 的文件）时，多模态分析能增量补齐；改 `F/R/V/P` 时 chunks 与图谱重建；改 `!` 时停掉或恢复 KG 构建。引擎变更被视为"重大变更"，统一由 delete + 重传完成，不在续跑路径里隐式发生。

## 10. 常见问题排查

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| 上传被拒，提示不支持该类型 | 该后缀不在生效的白名单里。只声明 `MINERU_ADDITIONAL_SUFFIXES` / `DOCLING_ADDITIONAL_SUFFIXES` 并不能让裸 `x.doc` 变成可上传（§3.1） | 补一条路由规则或文件名 hint；用 `GET /documents/supported_file_types` 确认 |
| 改完 `LIGHTRAG_PARSER` 后服务起不来 | 启动校验是严格的：未知引擎、语法错误、外部引擎缺服务端点、引擎/分块参数非法（§2.6、§2.7、§3.6） | 读启动报错，它会指出是哪条规则 |
| 规则像是被忽略，所有文件都走了 `legacy` | 引擎未通过可用性检查，该规则被跳过：要么不支持该后缀，要么服务端点/凭据未配置（§2.7） | 配置服务端点，或把该后缀路由给支持它的引擎 |
| 文件名 hint 被忽略并打了告警 | 方括号内出现非法字符，或只写选项时漏了前导连字符（§2.5） | 使用 `[-OPTIONS]` 形式，例如 `report.[-teP].docx` |
| 开了 `i` / `t` / `e` 却什么都没分析 | sidecar 不存在——文档没有该类内容，或引擎不产出它（`MINERU_ENABLE_TABLE` / `MINERU_ENABLE_FORMULA`、`DOCLING_DO_FORMULA_ENRICHMENT`；`legacy` 一个都不产出） | 查找那条指出 sidecar 为空的 INFO 日志，并检查 `*.parsed/`（§4.2） |
| 文档 FAILED，报 "VLM analysis required but VLM role is not available" | 启用了 `i`，且有图片通过了前置过滤，而 VLM 未配置（§4.3） | 设 `VLM_PROCESS_ENABLE=true` 并配好支持视觉的 binding，**或者**删除该文档、改用 `te` 重新上传——直接重试不会去掉 `i` |
| 部分图片被分析，另一些被静默跳过 | 非栅格格式、小于 `VLM_MIN_IMAGE_PIXEL`、或大于 `VLM_MAX_IMAGE_BYTES`（§4.3） | 看该 item 的 `llm_analyze_result` 消息；图片确实需要就调高字节上限 |
| 多模态 item 因 token 预算失败 | 上下文预算挤占了 item 自身的空间（§4.4） | 调高 `MAX_EXTRACT_INPUT_TOKENS`，或调低 `SURROUNDING_LEADING_MAX_TOKENS` / `SURROUNDING_TRAILING_MAX_TOKENS` |
| 指定了 `P`，分块结果却像 `R` | 没有产出结构化的 `LightRAG Document`，`P` 已退化（§2.7、§3.2） | 把文件路由给 `native` / `mineru` / `docling`，而不是 `legacy` |
| `.docx` 标题识别不准，导致 `P` 切分很差 | 该文档的 Word 大纲不可靠 | 试试 `native(smart_heading=true)`（§3.3），配合 parser CLI 迭代 |
| 日志提示部分段落缺少 `paraId` | 由 LibreOffice / WPS / 旧版 Word 产生（§3.3） | 仅提示性。只有确实需要段落级溯源时，才用 Word 2013+ 另存一次 |
| 文档 FAILED 且标记为重复 | 文件名查重（规范化并剥掉 hint 后）或内容 hash 查重（§7.1、§7.2） | 先删除已有文档，或给新文件改名；重复记录（`dup-*`，或解析后就地改为 FAILED 的原记录）只能显式删除（§7.3） |
| 上传返回 `409` | 输入目录或 doc_status 里已存在同规范名的文档（§7.2） | 用 `POST /documents/delete_document` 删除后再上传 |
| 上传返回 `413` 或 `429` | 触发了准入限制（§8.7） | 对照该节的限制表判断是哪一条 |
| 所有接口都返回 `503` | `recovery_required` 栅栏已升起（§8.5） | 按该节给出的恢复顺序处理 |
| `/documents/scan` 返回 `scanning_skipped_pipeline_busy` | 流水线正忙或正在扫描、有上传在途、或有手动重试排队中（§8.1） | 等待空闲；`POST /documents/reprocess_failed` 是卡住的重试请求的一键恢复 |
| 点了「中断」，一批文档变成 `FAILED` | 已进入本批的文档在中断时会被标记为 FAILED（§8.2） | 用 `POST /documents/scan` 或 `POST /documents/reprocess_failed` 重试；已完成的文档不受影响 |
| 点了「中断」但流水线还在跑 | 中断是协作式的，不会打断已经发出的 LLM 调用（§8.2） | 等当前阶段结束；`GET /documents/pipeline_status` 的 `cancellation_requested` 可确认请求已收到 |
| 重启服务后文档卡在 `PROCESSING` / `PARSING` | 中断态会自动恢复，但服务启动本身不会触发流水线（§8.2） | 调一次 `POST /documents/scan`，或上传任意文件触发 |
| 改了引擎或选项，输出却没变 | 引擎与 `process_options` 在入队时就冻结进 doc_status 记录。自动续跑与 `/documents/reprocess_failed` 都沿用存量值；改 `LIGHTRAG_PARSER` 或 hint 只对新上传生效（§8.3、§9.3） | 删除该文档（勾选"同时删除文件"）后重新上传 |
| 修好了外部引擎的服务配置，它却一直返回旧结果 | 命中了原始产物包缓存（§6.3） | 打开对应的 `LIGHTRAG_FORCE_REPARSE_*`（§3.7），或删除文档时勾选"同时删除文件" |
| 扫描版 PDF 报 "extracted no usable text" | `legacy` 读不了没有文本层的 PDF（§3.2） | 改路由到开启 OCR 的 `mineru` 或 `docling` |
| MinerU 拒绝多段页码范围 | 多段范围只有 `official` 模式支持，`local` 只接受单页或一个简单区间（§3.6） | `local` 下改用单个区间，或切换模式 |
| 启动时报缺少 spaCy 模型 | `DOCX_SMART_HEADING=true` 或规则里带 `native(smart_heading=true)` 触发了启动期快速失败检查（§3.3） | 执行 `lightrag-download-cache --spacy-install`，或改用已内置模型的主 Docker 镜像 |

## 11. Python SDK 调用

本章针对**直接 import `LightRAG` 类**进行集成的开发者，覆盖 Server 部署不会用到的运行时 API、构造期参数和已移除的旧接口。Server 用户通常无须阅读本章。

### 11.1 适用对象

```python
from lightrag import LightRAG
rag = LightRAG(working_dir="./rag_storage", ...)
await rag.initialize_storages()
await rag.ainsert("text", file_paths="doc.pdf")
```

这种调用方式以下行为与 Server 路径不同：可在不重启进程的情况下改 `addon_params["chunker"]`，可向 `apipeline_enqueue_documents` 传入 per-file `chunk_options`，可在 `ainsert` 调用时动态覆盖 F 策略的预切分参数。

### 11.2 LightRAG 构造期参数

`LightRAG(chunk_token_size=…, chunk_overlap_token_size=…)` 是 §5.3 优先级链中的**第 3 档**："legacy 构造字段"。strategy 无关、粗粒度缺省，只填仍空的槽位：

- 优先级低于 `addon_params["chunker"]` 显式值（§11.3）和 strategy 特定 env（§5.2）。
- 优先级高于 legacy env `CHUNK_SIZE` / `CHUNK_OVERLAP_SIZE`。
- 实例字段 `self.chunk_token_size` / `self.chunk_overlap_token_size` 在 `__post_init__` 之后总会被回填为 `int`，方便仍读这两个字段的旧路径（如 `pipeline.py` 中 `chunk_opts.get("chunk_token_size") or self.chunk_token_size` 兜底）继续工作。

### 11.3 运行时改 `addon_params["chunker"]`

`addon_params["chunker"]` 是 `ObservableAddonParams` 字段，可以**运行时改**：

```python
rag.addon_params["chunker"]["recursive_character"]["separators"] = ["##", "\n", " "]
```

改完后，**后续入队**的文档拿到新默认；已入队文档保留入队时的快照不变（参见 §5.3 三层语义保证）。这是 §5.3 优先级链的第 1 档："`addon_params["chunker"]` 显式值"，赢一切。

Server 部署没有这个能力 —— 改 env 后必须重启服务才生效。

### 11.4 `apipeline_enqueue_documents(chunk_options=…)`

`apipeline_enqueue_documents` 接受可选的 `chunk_options` 参数，调用方传入 `dict` / `list[dict]` 会按当前文档的 `process_options` 投影为精简快照（只保留对应策略子字典 + 顶层 `chunk_token_size`）后持久化到 `full_docs[doc_id]["chunk_options"]`；不传则由 `resolve_chunk_options(self.addon_params, process_options=…)` 现场拼装一份。调用方可以放心传入全量字典——其它策略子字典会被 dispatcher 丢弃，不会污染存储。

典型用法：

```python
await rag.apipeline_enqueue_documents(
    input=["text A", "text B"],
    file_paths=["a.[native-R].txt", "b.txt"],
    process_options=["R", ""],
    chunk_options=[
        {"chunk_token_size": 800, "recursive_character": {"separators": ["\n\n", "\n"]}},
        {"chunk_token_size": 1500},
    ],
)
```

per-file 个性化的典型场景：管理 UI 单独配置某个文件的 separators 或 V 阈值；将来上传 API 也可在 form / hint 中接收覆盖。

**不传 `file_paths` 的兼容**：核心 API `insert` / `ainsert` / `apipeline_enqueue_documents` 仍兼容未传 `file_paths` 的调用；这类文档的 `file_path` 会保存为 `unknown_source`，不会参与文件名查重，文档 ID 继续按文本内容生成。

`apipeline_enqueue_documents` 自身的并发约束见 §8.1 的状态-操作表。

### 11.5 `ainsert(split_by_character=…, split_by_character_only=…)`

`LightRAG.ainsert(split_by_character=…, split_by_character_only=…)` 的运行时参数在入队时由 `resolve_chunk_options` 覆写到 `chunk_options.fixed_token`：

- `split_by_character` 非 `None` 即覆盖 env 默认；
- `split_by_character_only=True` 即覆盖（`False` 是签名默认值，与"未指定"无法区分，所以 env 默认胜出）。

仅对 F 策略生效；其它策略的子字典不受影响。

### 11.6 已移除的 SDK 入参：`reprocess_existing_non_processed`

旧 `apipeline_enqueue_documents` 的 `reprocess_existing_non_processed=True` 行为会在 scan 时直接删除非 PROCESSED 的旧记录并重建，与 §7 / §8 的规则相冲突，已整段移除。替代路径：

- 自动续跑：scan 按 §7.4 的规则处理同名文件（归档 / 续跑 / 删记录后重入队），由 §9 续跑规则在处理循环里统一接管。
- 强制刷新：先调 `/documents/delete_document` 删旧文档，再上传同名新文件。

## 附录 A：从旧版升级的注意事项

### A.1 多模态全局开关已被文件级选项取代

`addon_params["enable_multimodal_pipeline"]` 已废弃。多模态分析现在按文档、通过 `i` / `t` / `e` 处理选项选择（§2.1），既可在 `LIGHTRAG_PARSER` 中作为规则默认值设置（§2.4），也可通过文件名 hint 指定（§2.5）。不再存在"全部分析"式的全局开关，因为一篇文档实际含有哪些模态由解析引擎决定、以 sidecar 是否存在来表达，而不是由配置决定。

迁移方式：把原来的全局开关换成路由规则上对应的字母，例如 `LIGHTRAG_PARSER=*:native-iteP,*:legacy-R`。注意 `VLM_PROCESS_ENABLE` 是一个正交的独立开关：它只闸控**图片**分析（表格与公式由 `EXTRACT` 角色分析）；若启用了 `i` 而 VLM 不可用，通过前置过滤的图片会让该文档失败，而不是被跳过。

升级前已入库的文档，其 `process_options` 在入队时就已冻结在 `doc_status` 记录里；改规则或改 hint 都不会追溯修改它们。要让已有文档按新选项重跑，只能删除后重新上传（§9.3）。

### A.2 不主动开启则行为不变

不配置 `LIGHTRAG_PARSER` 时，所有扩展名仍然走 `legacy` 内容抽取器与旧的分块行为，与升级前完全一致。新引擎与新分块策略都是 opt-in 的；§1 给出了三套可直接套用的预设。

### A.3 已移除的 SDK 入参

`apipeline_enqueue_documents(reprocess_existing_non_processed=...)` 已移除，替代路径见 §11.6。

## 附录 B：环境变量速查

本表说明各类文件处理相关环境变量该去哪里查，是索引而非复述：逐个变量的权威说明在 [env.example](https://github.com/HKUDS/LightRAG/blob/main/env.example) 的注释里。

| 类别 | 变量 | 说明所在 |
| --- | --- | --- |
| 路由 | `LIGHTRAG_PARSER` | §2.3、§2.4、§2.5 |
| 分块 | `CHUNK_SIZE`、`CHUNK_OVERLAP_SIZE`、`CHUNK_{F,R,V,P}_*` | §5.2 |
| 多模态 | `VLM_PROCESS_ENABLE`、`VLM_MAX_IMAGE_BYTES`、`VLM_MIN_IMAGE_PIXEL`、`MAX_EXTRACT_INPUT_TOKENS`、`SURROUNDING_*_MAX_TOKENS`、`MM_EXTRACT_CONTENT_MIN_TOKENS` | §4 |
| VLM / 角色模型 | `VLM_LLM_*`、`VLM_MAX_ASYNC_LLM` | [RoleSpecificLLMConfiguration-zh.md](RoleSpecificLLMConfiguration-zh.md) |
| legacy 引擎 | `PDF_DECRYPT_PASSWORD` | §3.2 |
| native 引擎 | `NATIVE_MD_IMAGE_*` | §3.3 |
| native docx smart_heading | `DOCX_SMART_HEADING`、`DOCX_SMART_*` 调优项 | §3.3 与 `env.example` 的 smart_heading 注释块 |
| MinerU | `MINERU_*` | §3.4 |
| Docling | `DOCLING_*` | §3.5 |
| 解析缓存 | `LIGHTRAG_FORCE_REPARSE_{NATIVE,MINERU,DOCLING}`、`{MINERU,DOCLING}_ENGINE_VERSION` | §3.7、§6.3 |
| 目录 | `INPUT_DIR`、`WORKING_DIR`、`SCAN_SPOOL_DIR` | §6、§7.4 |
| 并发 | `MAX_PARALLEL_*`、`QUEUE_SIZE_*` | §8.6 |
| 准入与限制 | `MAX_UPLOAD_SIZE`、`MAX_REQUEST_BODY_BYTES`、`MAX_TEXTS_PER_REQUEST`、`MAX_PENDING_DOCUMENTS`、`PIPELINE_*`、`MAX_UNACKED_MANUAL_RETRIES`、`SCAN_ENQUEUE_BATCH_SIZE` | §8.7 |
| 查询期（不影响分块） | `ENABLE_CONTENT_HEADINGS` —— 组装回答上下文时为每个分块追加其标题路径；它不改变分块边界，也不改变已存储的分块文本 | [LightRAG Server](./LightRAG-API-Server-zh.md) |
| 离线 / 分词器 | `TIKTOKEN_CACHE_DIR` | [OfflineDeployment.md](./OfflineDeployment.md) |
