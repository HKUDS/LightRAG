# Paragraph Semantic 分块策略

## 1. 适用场景与策略选择

### 1.1 P 策略要解决什么问题

Paragraph Semantic Chunking（下文简称 **P 策略**）面向 DOCX、PDF 等具有清晰章节结构的文档。其核心目标是：**让分块边界尽可能对齐文档原生的语义边界**（标题、段落、表格行），而不是仅由 token 长度计数决定切点。

P 策略主要解决以下四类问题：

1. **表格语境断裂**：大表被拆分后，首尾切片容易脱离前置说明、后置解释或中间桥接文字，召回时无法独立理解。
2. **层级信息利用不足**：仅看相邻段落的方法无法利用父标题路径、同级条款之间的关系。
3. **细碎章节尺寸失衡**：规章、标准、合同等文档常包含大量 100~300 token 的细碎条款，若不合并则块过短、语义稀薄；若仅按相邻长度合并又会跨主题污染。
4. **长块二次拆分破坏结构**：章节过长时，常规字符切分会忽略表格行边界和标题层级。

P 策略对**任何能生成 `.blocks.jsonl` sidecar 的 parser**（`native` / `mineru` / `docling`）产出的结构化产物有效——三者经统一的 `write_sidecar()` 落盘相同结构的 `.blocks.jsonl`（含 `heading` / `level` / `parent_headings`）。仅 `legacy` 引擎不产出 sidecar；无 sidecar 的输入（legacy 路径或解析失败）会自动降级为 R 策略（见 §6）。

### 1.2 P / R / V 三种策略对比

| 维度 | R 策略（Recursive） | V 策略（SemanticVector） | P 策略（ParagraphSemantic） |
|---|---|---|---|
| 切分依据 | 字符分隔符级联（段落 → 换行 → 中文标点 → 空格 → 字符）+ token 预算 | 句子级 embedding 距离阈值（百分位 / 标准差 / 四分位距 / 梯度）寻找语义断层 | 标题 outline level 与 `parent_headings` + 表格行边界 + 锚点 + 层级感知合并 |
| 块大小控制 | `chunk_token_size` 硬上限 | `chunk_token_size` 仅为 advisory ceiling，超限时通过 R 二次切分 | `target_max` 硬上限 + `target_ideal` 软目标 + 表格阈值 + 尾部吸收阈值多重协同 |
| 表格处理 | 不感知表格，可能在表格中间切断 | 不感知表格 | 表格小于 `table_max` 保持完整；大表按 JSON 行数组 / HTML `<tr>` 行边界切片，并重新包裹为合法 `<table>` |
| 表格上下文 | 依赖窗口偶然覆盖 | 依赖 embedding 距离 | 首切片粘连前置说明、末切片粘连后置解释、连续大表桥接文字双向重叠 |
| 块间重叠 | 全局 `chunk_overlap_token_size` | 不会出现重叠 | 章节边界不会重叠；同章节长正文 fallback 到 R 时按 `CHUNK_P_OVERLAP_SIZE` 重叠；连续大表桥接文字可同时进入前后两个表格块 |
| heading 元数据 | 通常无 | 通常无 | 继承或提升 heading；拆分后追加 `[part n]` 后缀；保留 `parent_headings` 和 `level` |
| 嵌入计算开销 | 无 | 高（需对每个句子计算 embedding） | 无 |
| 依赖输入 | 任意文本 | 任意文本 + Embedding 模型 | 必须有 `.blocks.jsonl` sidecar（`native` / `mineru` / `docling` 任一引擎产出），否则降级为 R |

### 1.3 怎么选

| 场景 | 推荐 | 理由 |
|---|---|---|
| 章节层级清晰（内容解析引擎需要能够生成Sidecar文件） | **P** | 充分利用标题层级与表格行边界，块边界最贴合语义；避免跨主题污染 |
| 文档以散文 / 评论 / 长篇正文为主，没有明确章节结构 | **V** | 按语义相似度切分能在话题切换点形成自然边界，比字符切分更稳定 |
| 输入是纯文本、Markdown、代码、日志，或追求最低算力开销 | **R** | 无嵌入开销，分隔符级联对中英文混合文本足够稳定 |
| 通用配置（不确定文件类型） | **R** | P 在无 sidecar 时自动降级到 R；V 在无 Embedding 模型时也降级到 R |
| 标题样式混乱、正文中大量伪标题的文档 | **R** 或 **V** | P 依赖 parser 正确识别标题，标题错乱会导致基础块边界偏移 |
| 单行超大表格或不可解析表格 | 任意 | 三种策略最终都会走字符级 fallback；P 仍保留表格上下文粘连优势 |

## 2. 设计目标与核心不变量

P 策略的全部规则都服务于一个目标：**让块边界对齐文档原生语义边界，并让每个块在召回时能被独立理解**。它把这个目标拆成针对三类场景的具体规则（表格、长块、细碎章节），逐条在 §3 展开。无论规则如何组合，以下四条**重叠不变量**始终成立——它们界定了「哪里允许文字复制、哪里绝不允许」：

1. **章节边界不会重叠**：不同 `.blocks.jsonl` 内容行之间的文本绝不会被复制到对方块里，避免“张冠李戴”。
2. **章节内长正文可重叠**：同一个内容行内拆分的多个片段允许按 `chunk_overlap_token_size` 保留 R 风格 overlap，减少长正文中途切断。
3. **表格之间桥接文字可双向重叠**：唯一的跨段落复制场景，专门服务连续大表的上下文保留。
4. **表格行不互相重叠**：行级切片本身是非重叠的，与 R 的 overlap 概念不同。

### 2.1 规则与效果速览

下表把 §3 的每条规则映射到它达到的效果，以及实现它的内部阶段（阶段名同时是代码注释、日志关键字与排查的交叉引用标识，见 §7.6）：

| 分块规则 | 达到的效果 | 实现阶段 | 详见 |
|---|---|---|---|
| 标题级基础块 | 块边界对齐文档原生结构，而非 token 计数 | HeadingBlocks | §3.1 |
| 表格完整性 + 行边界切片 | 表格不被从中间截断，切片仍是合法 `<table>` | TableRowSplit | §3.2 |
| 表格上下文粘连（角色 + 桥接双向重叠） | 表格前置说明、后置解释、桥接文字不脱离表格 | TableRowSplit / TableBridge | §3.3 |
| 表头恢复（切分时补回中段/末段表头） | 被拆表的 `middle`/`last` 切片单独召回时不丢失列名，且不触发长度超限 | HeaderRecovery | §3.3.3 |
| 锚点驱动长块再切分 | 超长章节按语义点切分，保留标题层级 | AnchorSplit | §3.4 |
| 无正文标题粘连 | 父标题不与其子内容失散 | HeadingGlue | §3.5 |
| 层级感知合并 | 细碎条款聚到理想大小，又不跨主题污染 | LevelMerge | §3.6 |
| 重叠规则 | 召回上下文充分，但章节/表格边界不“张冠李戴” | 贯穿全程 | §3.7 |
| 尺寸阈值协同 | 大多数块落在 `[target_ideal, target_max]` | 贯穿全程 | §3.8 |

### 2.2 处理流水线总览

上述规则串成一条以 `.blocks.jsonl` 为输入的流水线（`fixlevel=0` 模式，**每个 `type == "content"` 行被视为一个标题级基础块**）：

```text
DOCX / PDF / PPTX / …
  ↓  native(docx, fixlevel=0) / mineru / docling parser —— 按标题输出基础块，不做 token 拆分
.blocks.jsonl + sidecar (.tables.json / .equations.json / .drawings.json / .blocks.assets/)
  ↓  TableRowSplit：超大表格按行边界切片并赋予 first/middle/last 角色      → §3.2
  ↓  HeaderRecovery：切分时把重复表头预扣预算后注入中段/末段切片      → §3.3.3
  ↓  TableBridge：连续大表之间桥接文字双向重叠                        → §3.3
  ↓  AnchorSplit：锚点驱动的长文本块再切分                              → §3.4
  ↓  PartLabeling：[part n] 行级来源追溯编号（按原始内容行独立编号，故在跨行合并之前）
  ↓  HeadingGlue：无正文标题块向前并入严格更深的子块               → §3.5
  ↓  LevelMerge：层级感知的双相位合并                                  → §3.6
最终 chunk 列表
```

## 3. 分块规则与效果

### 3.1 标题级基础块——边界对齐文档原生语义 〔HeadingBlocks〕

**规则**：每个 `.blocks.jsonl` 的 `type == "content"` 行就是一个基础块，即「一条标题下的正文作为一个块」。标题识别完全由 **parser** 完成，**P chunker 自身不扫描文档 body、也不判断标题样式**，更不在解析阶段做 token 阈值拆分。

**达到的效果**：块的初始边界天然落在文档大纲结构上（标题切换处），而不是任意 token 位置；后续所有阶段都在这个语义对齐的基础上做加工。

三个能产出 sidecar 的引擎殊途同归地按标题切出基础块，各自得到 `heading` / `level` / `parent_headings`：

- **native（docx，`fixlevel=0`）**：读取 `styles.xml`，按 `<w:basedOn>` 建立样式继承链回溯有效 `<w:outlineLvl>`；遍历 `document.xml` 段落沿继承链解析大纲级别，原始 outline level 0~8 映射为内部 `level` 1~9；维护 `current_heading_stack`，遇新标题清理不浅于当前 level 的旧标题并计算 `parent_headings`。
- **mineru**：按条目的 `text_level > 0` 或 `label` 为 `title` / `section_header` 检测标题，用 heading_stack 维护父链。
- **docling**：`label="title"` → level 1，`label="section_header"` → `item.level + 1`（默认 level 2），同样维护父链。

三者最终都产出统一的 `IRBlock`（携带 `heading` / `level` / `parent_headings`），并由 `write_sidecar()` 落为相同结构的 `.blocks.jsonl`；表格、公式、图形被提取为单行标签（`<table id="..." format="json">...</table>` 等）写入对应 sidecar。所有可识别标题均触发基础块边界，**不**执行 token 阈值拆分。

P chunker 直接读取 `.blocks.jsonl`，每个 content 行作为后续 TableRowSplit/AnchorSplit 的独立处理单元——这也意味着 `[part n]` 编号按每个原始 content 行**独立重置**（见 §3.4 与 §4.4）。

### 3.2 表格完整性与行边界切片——不从中间截断表格 〔TableRowSplit〕

**规则**：token 数不超过 `table_max` 的表格**保持完整**；只有超过 `table_max` 的表格才切片，且**优先按行边界切**，只有收敛到单行仍无法在上限内表达时，整张表才退化为字符级切分。

**达到的效果**：表格永远不会在「单元格中间」被截断；每个切片都重新包裹为合法的 `<table>` 标签，下游解析与 LLM 阅读都能把它当作表格理解，而非破碎的标记片段。

#### 3.2.1 行边界优先切片

- `format="json"`：按 JSON 顶层行数组切片。
- `format="html"`：按 `<tr>...</tr>` 行切片。
- 未显式标注但内容可嗅探为 JSON / HTML 的表格同样按上述规则处理。

切片前预扣 `<table {attrs}></table>` 外壳 token 开销，使重新包裹后的切片尽量不超过 `table_max`。每个切片重新包裹为合法的 `<table>` 标签，便于下游解析。

#### 3.2.2 行级递归二次切片

若某个行子集重新包裹后仍超过 `table_max`，则在该行子集内继续细分。**当切片收敛到单行、且该单行无法在 `target_max` 内与表头并存（单行内容本身超上限，或没超但加上表头后超上限）时，整张表退化为对原始 `<table>` 文本（其 body 天然含表头）的 R 递归字符切分，并打一条 `logger.warning` 警告**——表头内容随原表文本以纯文本形式保留，绝不被静默丢弃，也不产生「部分 `<table>` 切片 + 部分孤立字符片段」的混合输出。不需要注入表头、且单行本身装得进 `target_max` 的切片仍原样保留为合法 `<table>` 标记。该机制使可被行边界表达的表格内容尽量保留合法表格结构。

#### 3.2.3 末片回吞

若表格末片 token 数低于 `table_min_last`，且与前一切片合并后不超过 `table_max`，则将末片回吞至前一切片，减少无效短表格块。

### 3.3 表格上下文粘连——前后说明、桥接与表头不失散 〔TableRowSplit / TableBridge / HeaderRecovery〕

**规则**：被切片的表格按「首/中/末」角色与周围段落做差异化粘连；连续两张大表之间的短桥接文字按预算**双向**分配到两侧表格块；丢失表头的中段/末段切片在**切分时**就把该表的重复表头拼回自身的 `<table>`（表头 token 已在切分前预扣进每片上限）。

**达到的效果**：表格的**前置说明**进入首切片块、**后置解释**进入末切片块、**桥接文字**同时作为左表后文与右表前文——任一表格切片被召回时都带着足以独立理解的上下文，不会出现「表格在这、解释在另一块」的断裂。被拆表的中段/末段切片即使脱离了承载表头的首切片，也会把表头行重新拼回自身的 `<table>` 开头，使其单独召回时仍能理解每列含义。

#### 3.3.1 表格切片角色与物理粘连

每个表格切片被赋予内部字段 `table_chunk_role`，并按角色决定与周围段落的粘连方式：

| 角色 | 含义 | 粘连策略 |
|---|---|---|
| `first` | 原始表格的首切片 | 追加到当前累积块尾部，使表格**前置说明**与首切片进入同一块 |
| `middle` | 原始表格的中间切片 | 独立输出，避免与无关正文合并 |
| `last` | 原始表格的末切片 | 作为新累积块起点，使**后置解释**自动追加到末切片之后 |
| `none` | 非表格切片或未拆分的完整表格 | 按普通文本块处理 |

`table_chunk_role` 是内部字段，最终输出不会保留，**但在 LevelMerge 中继续作为合并约束使用**（见 §3.6.1）。

#### 3.3.2 连续大表桥接文字双向重叠 〔TableBridge〕

当同一原始内容行中出现「大表 A、短桥接文字、大表 B」的模式，且两张表均被拆分时，桥接文字按上下文预算进行双向分配：

1. 将桥接文字按 token 编码。
2. 计算左侧预算 `prev_budget = min(chunk_overlap_token_size, target_max - 左侧末切片当前 token 数)`。
3. 计算右侧预算 `next_budget = min(chunk_overlap_token_size, target_max - 右侧首切片当前 token 数)`。
4. **若桥接文字长度同时不超过两侧预算**：左右两个表格边界块都包含**完整桥接文字**。
5. **若桥接文字较长**：前缀进入左侧末切片块，后缀进入右侧首切片块；超出两侧预算的中间段独立成为普通文本块。该中间段块**与左右两侧各保留 `chunk_overlap_token_size` 的 R 风格 overlap**：向左回吞已进入左表块的前缀尾部、向右多含已进入右表块的后缀头部。由于每侧前缀/后缀长度本身就 ≤ overlap 预算，overlap 区间会覆盖整段前缀与后缀，**结果中间段块实际承载完整桥接文字**（桥接文字因此从不被切散，只是其首尾**额外**复制进相邻表格块）。overlap 索引始终夹在桥接文字 token 内，**绝不会把 `<table>` 内容拷进中间段块**。

单侧预算还会被限制到不超过 `chunk_token_size / 2`，避免桥接文字主导整个块。

这与普通相邻 chunk overlap 的差异：

- 普通 overlap 按前后顺序复制字符或 token，与边界类型无关。
- TableBridge 机制以表格切片角色为触发条件，把桥接文字同时作为左表后文上下文和右表前文上下文，避免桥接说明只归属一侧表格或被单独切散后难以召回。

#### 3.3.3 中段/末段切片表头恢复 〔HeaderRecovery〕

大表按行边界切片后，表头行只保留在**首切片**内；`middle` / `last` 切片因此丢失列名，单独召回时无法判断每列含义。为此在 **TableRowSplit 切分的同时**，把表头行直接拼回非首切片自身的 `<table>`，使每个切片重新成为带表头的完整表格。

1. **表头来源**：解析期已把每张表的「跨页重复表头」写入同目录的 `.tables.json`（条目字段 `table_header`；**只有真正带重复表头的表才有该字段**）。**该字段按表格自身格式原生存储，使合并单元格语义全程存活**：`format="json"` 表存为 JSON 二维数组字符串（如 `[["H1","H2"]]`），`format="html"` 表存为原始 `<thead>…</thead>` 片段（保留 `rowspan` / `colspan`）。P 按待拆 `<table>` 标签保留的 `id` 关联回对应表条目，取其 `table_header`。
2. **预扣预算、切分时注入**：表头的 token 数在切分**之前**就从每片 body 上限中预扣（与 `<table {attrs}></table>` 包裹开销一并扣除）。`_split_table_text` 据此切分，再把表头拼回每个非首切片——`format="json"` 切片把表头行 prepend 到行数组，`format="html"` 切片把存储的原始 `<thead>` 片段 **verbatim 拼回 body 开头**（保留 `rowspan` / `colspan` 合并单元格语义，不再展开为无 span 的网格）；若 HTML 切片已自带 `<thead>`（切点落在多行表头内部）则跳过，避免重复。切片原有 `attrs`（含首位的 `id`）保持不变。由于已预扣，**切片含表头后仍 ≤ `target_max`**，硬上限由所有下游阶段自然保证，不存在事后回填导致的超限。首切片自带真实表头行，不重复注入。若某切片收敛到单行后已无法在 `target_max` 内同时容纳行内容与表头（见 §3.2.2），则**整张表退化为 R 递归字符切分（含表头）并打 warning**，绝不保留无表头的孤立切片。
3. **绝不臆造表头**——以下情形均不注入：源表在 `.tables.json` 中没有 `table_header` 字段（无重复表头）、`.tables.json` 缺失/不可读、切片已退化为字符级非 `<table>` 片段（无 `id` 可关联），或表格未发生真正的多片切分。
4. **格式一致性硬校验（损坏即报错）**：注入前先判定 `table_header` 的格式（JSON 二维数组 vs `<thead>` 片段）并与待拆表格自身的 `format` 比对。两者明确冲突（如 HTML 表却拿到 JSON 数组表头，反之亦然）意味着 sidecar 已损坏或张冠李戴，此时 **`_split_table_text` 直接 `raise ValueError` 中断该文档分块**，而非用错位表头产出畸形切片。这是刻意的「损坏即硬报错」语义，与第 3 点「表头缺失则静默跳过」相区分——缺失是可容忍的常态，格式冲突是数据损坏信号。

> 因为表头在切分时即进入切片，被拆表的各切片在 LevelMerge 中被**完全冻结、互不重新合并**（见 §3.6.1）——否则把同一张表的两个切片重新合并会在表中重复一次表头。表头**进入 `content`**，计入该 chunk 的 token 数（表头通常很小）；不写入 `heading`。

### 3.4 锚点驱动的长块再切分——按语义点切、保留标题 〔AnchorSplit〕

**规则**：对 TableRowSplit 后仍超过 `target_max` 的内容块，优先在「短段落锚点」处均衡切分，被选中的锚点晋升为子块新标题；无合格锚点时按「表格优先 → 贪心打包 → 字符切分」三级降级。

**达到的效果**：超长章节不是被硬切在任意 token 位置，而是切在短小的小标题/过渡句这类**自然语义点**上，子块继承可读的标题与父标题路径；同时保证算法**永不丢内容**，且尽量遵守用户配置的块大小上限。

#### 3.4.1 短段落锚点

把内容按段落恢复，选择满足以下条件的段落作为候选锚点：

- 段落不是表格（不以 `<table` 开头）。
- 段落文本长度不超过 `max_anchor_candidate_length`（100 字符）。
- 段落不是该块的第一个段落（避免递归无法收敛）。

#### 3.4.2 均衡选锚

根据目标子块数量计算理想切分位置，从候选锚点中选择距离理想位置最近的锚点。被选中的锚点**晋升为后续子块的新 `heading`**，原 heading 写入该子块的 `parent_headings`。

#### 3.4.3 无锚点降级

若不存在合格锚点：

1. **表格优先**：若块内仍存在超限表格，优先调用 TableRowSplit 的行边界切片。
2. **贪心打包**：其余文本按段落贪心打包到接近 `target_max`。
3. **递归字符切分**：单一过长普通文本段落降级到 R 策略（`chunking_by_recursive_character`），使用 `chunk_overlap_token_size` 保持相邻文本片段的连续性。

无锚点 fallback 路径保证算法**不会丢弃内容**，并尽量遵守用户配置的块大小上限。

### 3.5 无正文标题粘连——父标题不与子内容失散 〔HeadingGlue〕

**规则**：当一个块是 heading-only（只有标题、没有自己的正文）且紧邻的下一块层级**严格更深**时，把它**向前并入**那个更深的子块，并保留较浅的**父标题**身份；其余情况原样留给 LevelMerge。

**达到的效果**：像 `## 2.4`（无正文）这样的父标题，绝不会被单独切成孤块、再被 LevelMerge 向后吞进上一个同级块 `## 2.3` 而与它真正的子内容 `### 2.4.1` 失散——标题始终随其子内容走，标题路径层级无损。

某些章节只有标题、没有自己的正文（heading-only），例如：

```
## 2.3   结构尺寸及重量 .....   （level 2，有正文）
## 2.4   环境适应性指标          （level 2，heading-only，无正文）
### 2.4.1   概述                 （level 3，有正文）
```

若直接进入 LevelMerge，`## 2.4` 会作为一个独立的同级小块，被 Phase A 同级合并或尾部整批吸收**向后吞进上一个同级块 `## 2.3` 的末端**，使这个父标题与它真正的子内容 `### 2.4.1` 失散。

因此在 LevelMerge 之前增加一个前置步骤（`_glue_heading_only_blocks`）。当前块为 heading-only（`content` 仅由标题行构成，由 `^#{1,6} +` 判定）时，**仅向前粘连**：

- **触发条件**：紧邻的下一块层级**严格更深**（`level` 更大），且其 `table_chunk_role` 为 `none` 或 `first`。`first` 即「被切大表的首切片」——子节正文若是超大表格，TableRowSplit 切片后其首个产出块的角色为 `first`；紧跟 heading-only 行的只可能是下一行的首个产出块，故角色必为 `none` 或 `first`（`middle`/`last` 只在同一行表格内部出现）。
- **并入 `first` 切片时保留其角色**：把 `## 2.4` 并入 `first` 切片后，合并块**仍标记 `first`**（`## 2.4` 标题正是表格的前置上下文，本就该由 `first` 切片承载）。这样 LevelMerge 不会把它向后吸回 `## 2.3`（`first` 不可被向后吸收），表格边界保护得以维持；`none` 子块的行为与现状完全一致。
- **动作**：向前并入该子块，保留**父标题**身份（`heading` / `level` / `parent_headings` 取自较浅父块）。即 `## 2.4` 与 `### 2.4.1` 绑成一块，标题路径仍以 `2.4` 为主——子块 2.4.1 的 `parent_headings` 本就含 2.4，层级信息无损。链式标题（`# 2` → `## 2.4` → `### 2.4.1`）沿链折叠、保留**最浅**身份，直到遇到首个含正文的子块。
- **不做向后粘连**：当下一块**不更深**（更浅/同级标题，或已到末尾）时，该 heading-only 块原样留给 LevelMerge。**不会**把它向后并入更深的前块（如 `### 2.3.9`）——把更浅的 `## 2.4` 标题吞进更深的 L3 块会倒置层级（深吞浅）、压低标题层级。这类孤立标题直接交给 LevelMerge 正常处理。
- **保住硬上限**：子块来自 AnchorSplit、本在 `target_max` 内，但前缀拼入父标题行后可能超限。由于下游无人会重切超限块（LevelMerge 只阻止其继续变大），绑定后超限的块在此重切：**先剥离开头的标题行**，正文按**完整 `target_max`** 切分（使后续不含前缀的正文片保持完整预算），再把标题前缀拼回**第一个正文片**。仅当第一个正文片大到放不下前缀时，才单独对它用缩减后的上限再切——因此大前缀不会把整个子节切得过碎。这样标题始终随真实正文，绝不会被单独切成一个 heading-only 孤块（否则 LevelMerge 又会把它向后吸走），且每个产出片段仍 ≤ `target_max`。（退化情形：当前缀本身已吃满上限——极长标题或极小 `chunk_token_size`——无法保持完整，则整块直接切分、对超长标题行做字符级切分;此时 cap 优先于保持标题完整。）
- **不额外回填前块**：由于 `keep="left"` 保留父块的 `level`，绑定后的整体只是个普通小块（并非锁定独立）。是否并回前块 `2.3` 完全沿用 LevelMerge 既有规则——前块仍 < `target_ideal` 时走同级合并，或整体小于 `small_tail_threshold` 时走尾部吸收（即便前块已饱和也能被吸入），二者都以重测后的真实 token ≤ `target_max` 为界。本前置步骤只保证标题**不脱离其子内容**，并不把整体锁为独立块——因此在尺寸允许时让 `2.3 + 2.4 + 2.4.1` 同块正是期望的防过碎行为。

> 边界歧义：正文行若真以 `#␠` 开头会被误判为标题行——这是 `lightrag/parser/_markdown.py` 已记录并接受的同一启发式歧义，实际语料中概率极低。

### 3.6 层级感知合并——细碎条款聚到理想大小、不跨主题污染 〔LevelMerge〕

**规则**：**自深层级向浅层级处理**，先合并同级小块（Phase A），再尾部整批吸收，最后允许浅层块吸收深层块（Phase B）；每次合并都要同时满足尺寸、表格角色、层级、父标题路径四类约束。

**达到的效果**：大量 100~300 token 的细碎条款被合并到接近 `target_ideal` 的尺寸（块不再过短、语义稀薄），同时**绝不把分属不同主题/不同父章节的相邻小块揉在一起**——既治「块过碎」，又防「跨主题污染」。

#### 3.6.1 合并约束（每次合并都要满足）

1. **尺寸约束**：合并后的真实文本 token 数不超过 `target_max`；已达到 `target_ideal` 的块原则上不继续参与普通同级合并。
2. **角色约束（切片冻结）**：被拆表的所有切片 `first` / `middle` / `last` 一律**锁定独立、不参与任何合并**（既不向后吸收、也不被前块吸收、也不进入尾部整批吸收）。原因：表头已在 TableRowSplit 切分时注入各切片，若把同一张表的两个切片重新合并会在表中重复一次表头（§3.3.3）。表格边界的前后说明已在切分阶段粘进首/末切片，冻结不影响上下文粘连，只放弃小的首/末切片块与无关邻块的事后整合。仅 `none`（普通块/未拆分的完整表）可参与合并。
3. **层级约束**：同级合并在相同 `level` 之间发生；跨级吸收只允许浅层吸收深层，**禁止深层反向吸收浅层**。
4. **父标题路径一致性约束**：避免跨主题污染的关键，按合并方向取严格语义——
   - **同级合并（Phase A / 尾部吸收）**：两块 `parent_headings` 必须**完全相等**（真·兄弟）。仅 `level` 相同但父链不同（如 `2.4.1` 与 `2.5.1`）不允许合并。
   - **跨级吸收（Phase B，浅吸深）**：深块必须是浅块的**后代**——浅块的完整标题路径（`parent_headings` + 自身 `heading`，已剥离 `[part n]`）是深块 `parent_headings` 的前缀。浅块吞并不同分支的深块被禁止。
   - `parent_headings` 为空（preamble / 无层级输入）的块视为路径相容，放行（无层级可污染）。

#### 3.6.2 Phase A：同级合并

针对当前 level 的相邻块，当**当前块**低于 `target_ideal`、合并后真实 token ≤ `target_max`，且满足上述约束时，合并为一个块（被吸收的邻块不要求低于 `target_ideal`；反向合并时另要求前块也 < `target_ideal`）。

表格切片角色的方向规则（被拆表切片全部冻结，仅 `none` 可合并）：

| 块角色 | 可向后吸收下一块 | 可被前一块吸收 |
|---|:-:|:-:|
| `none` | 是 | 是 |
| `first` | 否 | 否 |
| `middle` | 否 | 否 |
| `last` | 否 | 否 |

#### 3.6.3 尾部整批吸收

若一个**普通块（`none`）**且已达到 `target_ideal` 的块后面紧跟一串同级小块，且该串小块总 token 数低于 `small_tail_threshold`、合并后真实 token 数不超过 `target_max`，则**一次性吸收**该串小块。遇到**任何被拆表切片**（`first` / `middle` / `last`），或父标题路径发生分叉时停止；被拆表切片自身也不会发起尾部吸收。

#### 3.6.4 Phase B：跨级吸收

对于 Phase A 后仍未饱和的小块，尝试跨级合并，但仅允许浅层吸收深层：

- 当前块比后一块更浅时，当前块可向后吸收后一块。
- 当前块比前一块更深时，前一浅层块可吸收当前块。
- 反方向合并被禁止。
- 被拆表切片（`first` / `middle` / `last`）在跨级阶段同样冻结，不参与合并；仅 `none` 块参与跨级吸收。

#### 3.6.5 合并后真实 token 复测

由于合并时会插入换行连接符，逐块 token 数相加可能低估合并结果。**每次提交合并前，都要对拼接后的真实文本重新计算 token 数**，确认不超过 `target_max` 后再提交。

合并后保留主块的 `heading`。如果多个 part 片段被合并，最终 heading 保留主块的 part 后缀，**不会**额外拼接多个 part 标签。

### 3.7 重叠规则汇总——哪里重叠、哪里绝不

**规则 + 效果**：P 策略对「文字复制（overlap）」有精确的边界划分，既保证召回上下文充分，又杜绝跨章节/跨表格的“张冠李戴”。把散落在各阶段的重叠行为集中如下：

| 场景 | 是否重叠 | 预算 / 机制 | 服务的效果 |
|---|---|---|---|
| 不同 `.blocks.jsonl` 内容行（章节边界） | **绝不重叠** | —— | 章节边界清晰，不张冠李戴 |
| 同一内容行内长正文 fallback 到 R | 可重叠 | `chunk_overlap_token_size` | 长正文中途切断处保持语义连续 |
| 连续大表之间的桥接文字 | 双向重叠 | 两侧各 `min(overlap, …, target_max/2)` | 桥接说明同时作为左右两表的上下文 |
| 桥接长文本的独立中间段块 | 与左右各重叠 | `chunk_overlap_token_size`（夹在桥接 token 内，绝不含 `<table>`） | 中间段与相邻表格块阅读连续 |
| 表格行级切片之间 | **绝不重叠** | —— | 行切片非重叠，避免重复行 |

### 3.8 尺寸阈值协同——大多数块落在 [ideal, max]

**规则**：P 策略的阈值不是固定常量，而是按 `chunk_token_size`（记为 N）动态推导，多个阈值协同控制文本块与表格切片的大小。

**达到的效果**：理想分布下，大多数 chunk 落在 `[target_ideal, target_max]` 区间（N=2000 时约 1500~2000 token）；明显偏小的块通常只是锁定独立的 `middle` 表格切片或章节边界尾块。

| 名称 | 计算式 | N = 2000 时取值 | 技术含义 |
|---|---|---:|---|
| `target_max` | N | 2000 | 文本块硬上限 |
| `target_ideal` | 0.75 × N | 1500 | 文本块理想目标，达到此值后停止参与普通同级合并 |
| `table_max` | 0.625 × N | 1250 | 表格触发切片阈值 |
| `table_ideal` | 0.375 × N | 750 | 表格切片理想大小 |
| `table_min_last` | 0.32 × `table_max` | 400 | 表格末片回吞阈值（小于此值且能合并则回吞至前一切片） |
| `small_tail_threshold` | 0.125 × N | 250 | 尾部碎块吸收阈值 |
| `max_anchor_candidate_length` | 固定 | 100 字符 | 长块拆分锚点候选段落长度上限 |

比例约束关系：`table_max < target_ideal < target_max`、`table_ideal < table_max`。这些比例源自审计模式经验值（`大块 8000、小表 5000、理想表 3000、表格尾块 1600`），现按 `chunk_token_size` 等比缩放。

## 4. 输入与输出

### 4.1 输入

`chunking_by_paragraph_semantic()` 接收以下输入：

| 参数 | 来源 | 说明 |
|---|---|---|
| `content` | `full_docs[doc_id].content` | 拼接后的合并文本，用于 sidecar 缺失时降级 |
| `blocks_path` | `full_docs[doc_id].lightrag_document_path` | `.blocks.jsonl` 路径，是 P 策略的主输入 |
| `.tables.json`（隐式） | 由 `blocks_path` 推导（`<base>.blocks.jsonl` → `<base>.tables.json`） | HeaderRecovery（§3.3.3）的表头数据源；缺失时静默跳过表头注入 |
| `chunk_token_size` | `chunk_options.chunk_token_size` / `CHUNK_P_SIZE` | 目标硬上限 N，默认 `2000` |
| `chunk_overlap_token_size` | `CHUNK_P_OVERLAP_SIZE` / `chunk_overlap_token_size` | 同一内容行内长正文 fallback 与表格桥接预算的上限，默认 `100` |
| `tokenizer` | LightRAG 已解析好的 tokenizer | 所有 token 计数与文本 overlap 截取的基准 |

P 策略**不接收** `split_by_character` / `split_by_character_only`，因为正常路径由标题和段落结构驱动。

### 4.2 `.blocks.jsonl` 约定

P 策略只处理 `type == "content"` 行。每个内容行通常包含：

- `content`：该标题下的正文文本，可能包含普通段落、`<table ... />` 标签、`<equation ... />` 公式、`<drawing ... />` 图形。
- `heading`：当前标题。
- `parent_headings`：父级标题链。
- `level`：标题级别（1~9，对应原始 outline level 0~8）。
- `positions`：原始段落定位（用于追溯）。
- `blockid`：该内容行的稳定标识（可选）。存在时会被带入最终 chunk 的 `sidecar` 字段，供多模态管线与文档删除按源 block 回溯；缺失时（raw / legacy 输入）输出不含 `sidecar`。

parser 保证「一条标题下的正文作为一个基础块」（native 经 `fixlevel=0` 模式，mineru / docling 经各自 IR builder），不在解析阶段做 token 阈值拆分。表格保持完整插入到 `content` 中。

### 4.3 输出

最终输出为有序 chunk 列表，每个元素：

```python
{
    "tokens": int,                    # 真实 token 数（合并后会复测）
    "content": str,                   # 块文本（可能包含 <table> 标签）
    "chunk_order_index": int,         # 块顺序索引
    "heading": {                      # 标题元数据（嵌套 dict，非扁平字段）
        "level": int,                 # 标题层级
        "heading": str,               # 拆分后追加 [part n] 后缀
        "parent_headings": list[str], # 父级标题链，不追加后缀
    },
    # 可选：仅当输入 .blocks.jsonl 行带 blockid 时出现，
    # 供多模态管线与文档删除按源 block 回溯。
    "sidecar": {
        "type": "block",
        "id": str,                    # 主块 blockid（refs[0]）
        "refs": [{"type": "block", "id": str}, ...],  # 去重后的全部源 blockid
    },
}
```

注意：`level` 与 `parent_headings` 现已收进 `heading` 嵌套 dict，顶层不再单独提供；`[part n]` 后缀落在 `heading["heading"]` 上。

实现内部还会临时使用 `paragraphs`、`content`、`table_chunk_role`、`blockids` 等字段辅助拆分和合并，但**不会**以这些名字进入最终输出（`blockids` 经转换后体现为 `sidecar`）。

### 4.4 `[part n]` 后缀规则

- 同一个原始 `.blocks.jsonl` 内容行被拆成多个片段时，所有片段的 `heading` 字段追加 `[part 1]`、`[part 2]` …
- 未发生拆分的内容行保持原 heading 不变。
- `parent_headings` 不追加后缀。
- 编号在每个原始内容行内**独立重置**（因 PartLabeling 在跨行合并之前编号，见 §2.2）。
- 旧的 `[表格片段N]` 后缀已统一由 `[part n]` 替代。

## 5. 配置项

| 配置 | 默认 | 说明 |
|---|---|---|
| `CHUNK_P_SIZE` | `2000`（未设时使用 `DEFAULT_CHUNK_P_SIZE`，**不**沿用 `CHUNK_SIZE`） | P 专用 `chunk_token_size`；段落语义合并需要比全局默认更大的上限，因此独立默认而非回退到 `CHUNK_SIZE` |
| `CHUNK_P_OVERLAP_SIZE` | 未设（沿用 `CHUNK_OVERLAP_SIZE`） | P 专用 overlap；只影响同一内容行内长正文 fallback 和表格桥接预算，**不**让表格行级切片互相重叠 |
| `CHUNK_OVERLAP_SIZE` / `LightRAG(chunk_overlap_token_size=…)` | `100` | 未设 P 专用 overlap 时的全局兜底 |

配置语法、优先级链、`addon_params["chunker"]` 运行时改值等详见 [FileProcessingConfiguration-zh.md](FileProcessingConfiguration-zh.md) §3。

`P` 是与引擎正交的 chunking 选项（`后缀:引擎-选项`），可与任何产出 sidecar 的引擎组合。启用 P 的典型 `LIGHTRAG_PARSER` 写法：

```bash
# docx 用 native，pdf 用 mineru，其余支持格式用 docling，都启用 P；不支持的格式回退 legacy-R
LIGHTRAG_PARSER=docx:native-teP,pdf:mineru-iteP,*:docling-iteP,*:legacy-R
CHUNK_P_SIZE=2000
CHUNK_P_OVERLAP_SIZE=100
```

（选项位 `i`/`t`/`e` 分别为图/表/公式分析，`P` 为 chunking 策略，可按需组合。）或在单文件覆盖：

```text
my-proposal.[native-P].docx
paper.[mineru-P].pdf
```

## 6. 降级保护——永不丢内容

**规则 + 效果**：P 策略有多层降级保护，任何结构化能力失效时都退到字符级切分，**保证文档仍产生检索块，不因结构化 sidecar 缺失而被静默丢弃**。

| 触发条件 | 降级行为 |
|---|---|
| `blocks_path` 缺失、不可读、无有效 content 行 | 整体降级到 `chunking_by_recursive_character()`，传入解析出的 `chunk_overlap_token_size` |
| TableRowSplit 中表格无法识别 JSON / HTML 结构 | 该表格调用 R 策略字符切分 |
| TableRowSplit 中单行无法在 `target_max` 内与表头并存（单行内容超上限，或加表头后超上限） | **整张表（含表头）退化为 R 策略字符切分，并打 `logger.warning`**；表头内容随原表文本以纯文本保留 |
| AnchorSplit 中长块没有合格短段落锚点 | 表格优先 → 贪心打包 → 单段落超长再降级 R 字符切分 |
| HeaderRecovery 时 `.tables.json` 缺失/不可读、源表无 `table_header` | 跳过表头注入（该表本就无重复表头，不影响其余分块） |

**重要**：整体 fallback 后不再具备标题层级、表格角色和桥接文字双向重叠能力；但能保证文档仍产生检索块。

## 7. 效果检验与调试

### 7.1 检查 sidecar 是否生成

确认 parser 是否成功产生 `.blocks.jsonl`：

```bash
ls -l INPUT/__parsed__/<doc>.<ext>.parsed/<doc>.blocks.jsonl
```

若文件不存在或为空，P 策略会整体降级为 R，不会获得 P 的任何收益。常见原因：

- 未给该格式配置能产出 sidecar 的引擎（如 `LIGHTRAG_PARSER=docx:native-...` / `pdf:mineru-...` / `*:docling-...`），实际走了 `legacy` 路径。
- 解析失败（看 `pipeline_status` 错误条目）。
- 该格式不被所选引擎支持（如 native 仅支持 docx；换用 mineru / docling 覆盖更多格式）。

### 7.2 检查 blocks.jsonl 内容

每行一个 JSON，过滤 `type == "content"` 后查看 heading / level / parent_headings 是否符合预期：

```bash
jq -c 'select(.type=="content") | {level, heading, parent_headings}' \
   INPUT/__parsed__/<doc>.<ext>.parsed/<doc>.blocks.jsonl | head
```

若 heading 大量为空或 level 异常，说明 parser 没正确识别标题 —— 此时 P 策略的层级合并和锚点提升都会失效。

### 7.3 检查最终 chunks 是否达到预期效果

查看 `text_chunks` 存储中的 chunk 元数据：

```bash
jq '.[] | {heading, level, tokens, parent_headings}' \
   rag_storage/kv_store_text_chunks.json | head -30
```

应观察到以下「规则生效」的迹象：

- 大表前后块的 heading 通常对应 `[part 1]` / `[part n]`（§3.2 表格切片发生）。
- 细碎条款被合并到接近 `target_ideal` 的块（§3.6 层级合并生效）。
- `parent_headings` 在不同章节切换处发生跳变，同章节内保持稳定（§3.1 / §3.6 父路径约束）。
- 大多数 chunk 落在 `[target_ideal, target_max]` 区间（§3.8）；明显偏小的块通常是 `middle` 表格切片（锁定独立）或紧靠章节边界的尾块。

若出现大量低于 `small_tail_threshold` 的尾块，可能是：

- 父标题路径一致性约束过严（不同 `parent_headings` 的相邻小块无法合并，§3.6.1）。
- 大量 `middle` 表格切片堆积（表格本身就很大）。

### 7.4 常见问题排查

#### 7.4.1 P 没生效，输出与 R 一致

按以下顺序排查：

1. `full_docs[doc_id].process_options` 是否包含 `P`？
2. `full_docs[doc_id].parse_format` 是否为 `lightrag`？若为 `raw`，说明走的是 legacy 路径，P 会自动降级到 R。
3. `lightrag_document_path` 指向的 `.blocks.jsonl` 是否存在、是否非空？
4. 日志中是否有 `paragraph_semantic ... fallback to recursive_character` 字样？

#### 7.4.2 表格被切散、前后说明分离（§3.2 / §3.3 未生效）

- 检查表格是否真的被识别为 `<table format="json">` 或 `<table format="html">`（看 `.blocks.jsonl`）。未识别格式的表格只能走字符切分，无法启动 TableRowSplit 的角色机制。
- 检查表格 token 数是否真的超过 `table_max`。低于阈值的表格保持完整，不会触发首/中/末切片。
- 若是连续大表，确认两张表之间的桥接文字是否在**同一 content 行**内 —— 跨 content 行的桥接不参与 B.1 双向重叠。

#### 7.4.3 细碎条款没有被合并（§3.6 未生效）

- 检查相邻条款的 `parent_headings` 是否一致：父标题路径一致性约束会阻止跨主题合并。
- 检查 `level` 是否一致：同级合并要求相同 `level`，跨级吸收只允许浅吸深。
- 检查中间是否插入了 `middle` 表格切片：会阻断尾部整批吸收。

#### 7.4.4 出现单个超过 `target_max` 的块

正常情况下 LevelMerge 的真实 token 复测会拒绝超限合并，但以下场景仍可能出现超限块：

- 单行表格自身超过 `target_max`，无锚点可拆，最终走 R 字符切分但单 chunk 仍超限。
- `enforce_chunk_token_limit_before_embedding` 在 embedding 前会做最后的硬切分，下游不会真把超限 chunk 嵌入向量库。

#### 7.4.5 `[part n]` 后缀异常（§3.4 / §4.4）

- 同一原始 content 行拆出多片但只看到一个 `[part 1]`：检查是否在 LevelMerge 中被合并 —— 合并后保留主块的 part 后缀，不拼接多个。
- 出现旧式 `[表格片段N]` 后缀：说明使用了旧版 chunker 输出的数据，新版统一为 `[part n]`，需要重新分块。

### 7.5 日志关键字

P 策略相关日志关键字（用于 `grep` 排查）：

- `paragraph_semantic` — 模块入口
- `fallback to recursive_character` — 整体或单段落降级
- `table_chunk_role` — 表格角色相关（§3.3）
- `bridge` — TableBridge 桥接文字处理（§3.3.2）
- `table_header` / `tables.json` — HeaderRecovery 表头恢复（§3.3.3）
- `anchor` — AnchorSplit 锚点选择（§3.4）

### 7.6 阶段名 ↔ 规则对照

代码注释、docstring、日志与测试中使用下列**阶段名**作为交叉引用标识。「曾用名」列给出旧版字母编号（仍可能出现在历史 commit / issue / PR 讨论中）：

| 阶段名 | 曾用名 | 对应规则 | 章节 |
|---|---|---|---|
| `HeadingBlocks` | Stage A | 标题级基础块 | §3.1 |
| `TableRowSplit` | Stage B | 表格完整性与行边界切片 | §3.2 |
| `HeaderRecovery` | Stage B.2 | 切分时为中段/末段切片补回表头 | §3.3.3 |
| `TableBridge` | Stage B.1 | 连续大表桥接文字双向重叠 | §3.3.2 |
| `AnchorSplit` | Stage C | 锚点驱动的长块再切分 | §3.4 |
| `PartLabeling` | Stage C.1 | `[part n]` 行级来源追溯编号 | §4.4 |
| `HeadingGlue` | Stage D 前置 | 无正文标题粘连 | §3.5 |
| `LevelMerge` | Stage D | 层级感知的双相位合并 | §3.6 |
