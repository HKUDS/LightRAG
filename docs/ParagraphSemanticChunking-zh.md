# Paragraph Semantic 分块策略

## 1. 适用场景与策略选择

### 1.1 P 策略要解决什么问题

Paragraph Semantic Chunking（下文简称 **P 策略**）面向 DOCX 等具有清晰章节结构的文档。其核心目标是：**让分块边界尽可能对齐文档原生的语义边界**（标题、段落、表格行），而不是仅由 token 长度计数决定切点。

P 策略主要解决以下四类问题：

1. **表格语境断裂**：大表被拆分后，首尾切片容易脱离前置说明、后置解释或中间桥接文字，召回时无法独立理解。
2. **层级信息利用不足**：仅看相邻段落的方法无法利用父标题路径、同级条款之间的关系。
3. **细碎章节尺寸失衡**：规章、标准、合同等文档常包含大量 100~300 token 的细碎条款，若不合并则块过短、语义稀薄；若仅按相邻长度合并又会跨主题污染。
4. **长块二次拆分破坏结构**：章节过长时，常规字符切分会忽略表格行边界和标题层级。

P 策略仅对 `native` 抽取引擎生成的 `.blocks.jsonl` 结构化产物有效；对非结构化输入会自动降级为 R 策略（见 §8）。

### 1.2 P / R / V 三种策略对比

| 维度 | R 策略（Recursive） | V 策略（SemanticVector） | P 策略（ParagraphSemantic） |
|---|---|---|---|
| 切分依据 | 字符分隔符级联（段落 → 换行 → 中文标点 → 空格 → 字符）+ token 预算 | 句子级 embedding 距离阈值（百分位 / 标准差 / 四分位距 / 梯度）寻找语义断层 | DOCX outline level 与 `parent_headings` + 表格行边界 + 锚点 + 层级感知合并 |
| 块大小控制 | `chunk_token_size` 硬上限 | `chunk_token_size` 仅为 advisory ceiling，超限时通过 R 二次切分 | `target_max` 硬上限 + `target_ideal` 软目标 + 表格阈值 + 尾部吸收阈值多重协同 |
| 表格处理 | 不感知表格，可能在表格中间切断 | 不感知表格 | 表格小于 `table_max` 保持完整；大表按 JSON 行数组 / HTML `<tr>` 行边界切片，并重新包裹为合法 `<table>` |
| 表格上下文 | 依赖窗口偶然覆盖 | 依赖 embedding 距离 | 首切片粘连前置说明、末切片粘连后置解释、连续大表桥接文字双向重叠 |
| 块间重叠 | 全局 `chunk_overlap_token_size` | 不会出现重叠 | 章节边界不会重叠；同章节长正文 fallback 到 R 时按 `CHUNK_P_OVERLAP_SIZE` 重叠；连续大表桥接文字可同时进入前后两个表格块 |
| heading 元数据 | 通常无 | 通常无 | 继承或提升 heading；拆分后追加 `[part n]` 后缀；保留 `parent_headings` 和 `level` |
| 嵌入计算开销 | 无 | 高（需对每个句子计算 embedding） | 无 |
| 依赖输入 | 任意文本 | 任意文本 + Embedding 模型 | 必须有 `.blocks.jsonl` sidecar（即 `native` 引擎抽取结果），否则降级为 R |

### 1.3 怎么选

| 场景 | 推荐 | 理由 |
|---|---|---|
| DOCX 且章节层级清晰、含大表格、含细碎条款 | **P** | 充分利用标题层级与表格行边界，块边界最贴合语义；避免跨主题污染 |
| 文档以散文 / 评论 / 长篇正文为主，没有明确章节结构 | **V** | 按语义相似度切分能在话题切换点形成自然边界，比字符切分更稳定 |
| 输入是纯文本、Markdown、代码、日志，或追求最低算力开销 | **R** | 无嵌入开销，分隔符级联对中英文混合文本足够稳定 |
| 通用配置（不确定文件类型） | **R** | P 在无 sidecar 时自动降级到 R；V 在无 Embedding 模型时也降级到 R |
| 标题样式混乱、正文中大量伪标题的文档 | **R** 或 **V** | P 依赖 native parser 正确识别标题，标题错乱会导致基础块边界偏移 |
| 单行超大表格或不可解析表格 | 任意 | 三种策略最终都会走字符级 fallback；P 仍保留表格上下文粘连优势 |

### 1.4 P 策略的代价

- 必须搭配 `native` 引擎：在 `LIGHTRAG_PARSER` 中显式声明，例如 `docx:native-P`；否则即使写了 `P`，也会因为缺少 `.blocks.jsonl` 退化到 R。
- 仅支持 DOCX：其他格式没有 `.blocks.jsonl` 产物。
- 算法路径多、阈值多：调试时需要先确认输入 sidecar 是否正确，再看各阶段输出。

## 2. 工作原理总览

P 策略以 native parser 在 `fixlevel=0` 模式下产生的 `.blocks.jsonl` 为输入，**每个 `type == "content"` 行被视为一个标题级基础块**，然后在该基础上执行表格切片、长块拆分和层级合并：

```text
DOCX
  ↓  native parser (fixlevel=0)
.blocks.jsonl + sidecar (.tables.json / .equations.json / .drawings.json / .blocks.assets/)
  ↓  Stage B：超大表格按行边界切片并赋予 first/middle/last 角色
  ↓  Stage B.1：连续大表之间桥接文字双向重叠
  ↓  Stage C：锚点驱动的长文本块再切分
  ↓  Stage C.1：[part n] 行级来源追溯编号（按原始内容行独立编号，故在跨行合并之前）
  ↓  Stage D 前置：无正文标题块向前并入严格更深的子块
  ↓  Stage D：层级感知的双相位合并
最终 chunk 列表
```

**P 策略的关键不变量**：

1. **章节边界不会重叠**：不同 `.blocks.jsonl` 内容行之间的文本绝不会被复制到对方块里，避免“张冠李戴”。
2. **章节内长正文可重叠**：同一个内容行内拆分的多个片段允许按 `chunk_overlap_token_size` 保留 R 风格 overlap，减少长正文中途切断。
3. **表格之间桥接文字可双向重叠**：唯一的跨段落复制场景，专门服务连续大表的上下文保留。
4. **表格行不互相重叠**：行级切片本身是非重叠的，与 R 的 overlap 概念不同。

## 3. 输入与输出

### 3.1 输入

`chunking_by_paragraph_semantic()` 接收以下输入：

| 参数 | 来源 | 说明 |
|---|---|---|
| `content` | `full_docs[doc_id].content` | 拼接后的合并文本，用于 sidecar 缺失时降级 |
| `blocks_path` | `full_docs[doc_id].lightrag_document_path` | `.blocks.jsonl` 路径，是 P 策略的主输入 |
| `chunk_token_size` | `chunk_options.chunk_token_size` / `CHUNK_P_SIZE` | 目标硬上限 N，默认 `2000` |
| `chunk_overlap_token_size` | `CHUNK_P_OVERLAP_SIZE` / `chunk_overlap_token_size` | 同一内容行内长正文 fallback 与表格桥接预算的上限，默认 `100` |
| `tokenizer` | LightRAG 已解析好的 tokenizer | 所有 token 计数与文本 overlap 截取的基准 |

P 策略**不接收** `split_by_character` / `split_by_character_only`，因为正常路径由标题和段落结构驱动。

### 3.2 `.blocks.jsonl` 约定

P 策略只处理 `type == "content"` 行。每个内容行通常包含：

- `content`：该标题下的正文文本，可能包含普通段落、`<table ... />` 标签、`<equation ... />` 公式、`<drawing ... />` 图形。
- `heading`：当前标题。
- `parent_headings`：父级标题链。
- `level`：标题级别（1~9，对应原始 outline level 0~8）。
- `positions`：原始段落定位（用于追溯）。
- `blockid`：该内容行的稳定标识（可选）。存在时会被带入最终 chunk 的 `sidecar` 字段，供多模态管线与文档删除按源 block 回溯；缺失时（raw / legacy 输入）输出不含 `sidecar`。

native parser 的 `fixlevel=0` 模式保证「一条标题下的正文作为一个基础块」，不在解析阶段做 token 阈值拆分。表格保持完整插入到 `content` 中。

### 3.3 输出

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

### 3.4 `[part n]` 后缀规则

- 同一个原始 `.blocks.jsonl` 内容行被拆成多个片段时，所有片段的 `heading` 字段追加 `[part 1]`、`[part 2]` …
- 未发生拆分的内容行保持原 heading 不变。
- `parent_headings` 不追加后缀。
- 编号在每个原始内容行内**独立重置**。
- 旧的 `[表格片段N]` 后缀已统一由 `[part n]` 替代。

## 4. 关键阈值

P 策略的阈值不是固定常量，而是按 `chunk_token_size`（记为 N）动态推导：

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

## 5. Stage A：标题级基础块

标题识别由 native parser 完成，**P chunker 自身不扫描 docx body、也不判断标题样式**。

native parser 在 `fixlevel=0` 模式下：

1. 读取 `styles.xml`，按 `<w:basedOn>` 建立样式继承链，回溯有效 `<w:outlineLvl>`。
2. 遍历 `document.xml` 段落，沿继承链解析大纲级别；原始 outline level 0~8 映射为内部 `level` 1~9。
3. 维护 `current_heading_stack`，遇新标题时清理不浅于当前 level 的旧标题，计算 `parent_headings`。
4. 将表格、公式、图形分别提取为单行标签（`<table id="..." format="json">...</table>` 等），写入对应 sidecar。
5. 所有可识别标题均触发基础块边界，**不**执行 token 阈值拆分。

P chunker 直接读取 `.blocks.jsonl`，每个 content 行作为后续 Stage B/C 的独立处理单元。这意味着 `[part n]` 编号按每个原始 content 行独立重置。

## 6. Stage B：超大表格行边界切片

Stage B 只处理 token 数超过 `table_max` 的表格。其目标**不是单纯拆表**，而是在行边界优先拆分的基础上保留表格边界上下文。

### 6.1 行边界优先切片

- `format="json"`：按 JSON 顶层行数组切片。
- `format="html"`：按 `<tr>...</tr>` 行切片。
- 未显式标注但内容可嗅探为 JSON / HTML 的表格同样按上述规则处理。

切片前预扣 `<table {attrs}></table>` 外壳 token 开销，使重新包裹后的切片尽量不超过 `table_max`。每个切片重新包裹为合法的 `<table>` 标签，便于下游解析。

### 6.2 行级递归二次切片

若某个行子集重新包裹后仍超过 `table_max`，则在该行子集内继续细分。**只有切片已经收敛到单行、且该单行自身超过限制时，才退化为字符级切分**。该机制使可被行边界表达的表格内容尽量保留合法表格结构。

### 6.3 末片回吞

若表格末片 token 数低于 `table_min_last`，且与前一切片合并后不超过 `table_max`，则将末片回吞至前一切片，减少无效短表格块。

### 6.4 表格切片角色与物理粘连

每个表格切片被赋予内部字段 `table_chunk_role`，并按角色决定与周围段落的粘连方式：

| 角色 | 含义 | 粘连策略 |
|---|---|---|
| `first` | 原始表格的首切片 | 追加到当前累积块尾部，使表格**前置说明**与首切片进入同一块 |
| `middle` | 原始表格的中间切片 | 独立输出，避免与无关正文合并 |
| `last` | 原始表格的末切片 | 作为新累积块起点，使**后置解释**自动追加到末切片之后 |
| `none` | 非表格切片或未拆分的完整表格 | 按普通文本块处理 |

`table_chunk_role` 是内部字段，最终输出不会保留，**但在 Stage D 中继续作为合并约束使用**（见 §9.1）。

## 7. Stage B.1：连续大表桥接文字双向重叠

当同一原始内容行中出现「大表 A、短桥接文字、大表 B」的模式，且两张表均被拆分时，桥接文字按上下文预算进行双向分配：

1. 将桥接文字按 token 编码。
2. 计算左侧预算 `prev_budget = min(chunk_overlap_token_size, target_max - 左侧末切片当前 token 数)`。
3. 计算右侧预算 `next_budget = min(chunk_overlap_token_size, target_max - 右侧首切片当前 token 数)`。
4. **若桥接文字长度同时不超过两侧预算**：左右两个表格边界块都包含**完整桥接文字**。
5. **若桥接文字较长**：前缀进入左侧末切片块，后缀进入右侧首切片块；超出两侧预算的中间段独立成为普通文本块。该中间段块**与左右两侧各保留 `chunk_overlap_token_size` 的 R 风格 overlap**：向左回吞已进入左表块的前缀尾部、向右多含已进入右表块的后缀头部。由于每侧前缀/后缀长度本身就 ≤ overlap 预算，overlap 区间会覆盖整段前缀与后缀，**结果中间段块实际承载完整桥接文字**（桥接文字因此从不被切散，只是其首尾**额外**复制进相邻表格块）。overlap 索引始终夹在桥接文字 token 内，**绝不会把 `<table>` 内容拷进中间段块**。

单侧预算还会被限制到不超过 `chunk_token_size / 2`，避免桥接文字主导整个块。

这与普通相邻 chunk overlap 的差异：

- 普通 overlap 按前后顺序复制字符或 token，与边界类型无关。
- B.1 机制以表格切片角色为触发条件，把桥接文字同时作为左表后文上下文和右表前文上下文，避免桥接说明只归属一侧表格或被单独切散后难以召回。

## 8. Stage C：锚点驱动的长文本块再切分

Stage C 处理 Stage B 后仍超过 `target_max` 的内容块。

### 8.1 短段落锚点

把内容按段落恢复，选择满足以下条件的段落作为候选锚点：

- 段落不是表格（不以 `<table` 开头）。
- 段落文本长度不超过 `max_anchor_candidate_length`（100 字符）。
- 段落不是该块的第一个段落（避免递归无法收敛）。

### 8.2 均衡选锚

根据目标子块数量计算理想切分位置，从候选锚点中选择距离理想位置最近的锚点。被选中的锚点**晋升为后续子块的新 `heading`**，原 heading 写入该子块的 `parent_headings`。

### 8.3 无锚点降级

若不存在合格锚点：

1. **表格优先**：若块内仍存在超限表格，优先调用 Stage B 的行边界切片。
2. **贪心打包**：其余文本按段落贪心打包到接近 `target_max`。
3. **递归字符切分**：单一过长普通文本段落降级到 R 策略（`chunking_by_recursive_character`），使用 `chunk_overlap_token_size` 保持相邻文本片段的连续性。

无锚点 fallback 路径保证算法**不会丢弃内容**，并尽量遵守用户配置的块大小上限。

## 8.5 Stage D 前置：无正文标题块的粘连

某些章节只有标题、没有自己的正文（heading-only），例如：

```
## 2.3   结构尺寸及重量 .....   （level 2，有正文）
## 2.4   环境适应性指标          （level 2，heading-only，无正文）
### 2.4.1   概述                 （level 3，有正文）
```

若直接进入 Stage D，`## 2.4` 会作为一个独立的同级小块，被 Phase A 同级合并或尾部整批吸收**向后吞进上一个同级块 `## 2.3` 的末端**，使这个父标题与它真正的子内容 `### 2.4.1` 失散。

因此在 Stage D 之前增加一个前置步骤（`_glue_heading_only_blocks`）。当前块为 heading-only（`content` 仅由标题行构成，由 `^#{1,6} +` 判定）时，**仅向前粘连**：

- **触发条件**：紧邻的下一块层级**严格更深**（`level` 更大），且其 `table_chunk_role` 为 `none` 或 `first`。`first` 即「被切大表的首切片」——子节正文若是超大表格，Stage B 切片后其首个产出块的角色为 `first`；紧跟 heading-only 行的只可能是下一行的首个产出块，故角色必为 `none` 或 `first`（`middle`/`last` 只在同一行表格内部出现）。
- **并入 `first` 切片时保留其角色**：把 `## 2.4` 并入 `first` 切片后，合并块**仍标记 `first`**（`## 2.4` 标题正是表格的前置上下文，本就该由 `first` 切片承载）。这样 Stage D 不会把它向后吸回 `## 2.3`（`first` 不可被向后吸收），表格边界保护得以维持；`none` 子块的行为与现状完全一致。
- **动作**：向前并入该子块，保留**父标题**身份（`heading` / `level` / `parent_headings` 取自较浅父块）。即 `## 2.4` 与 `### 2.4.1` 绑成一块，标题路径仍以 `2.4` 为主——子块 2.4.1 的 `parent_headings` 本就含 2.4，层级信息无损。链式标题（`# 2` → `## 2.4` → `### 2.4.1`）沿链折叠、保留**最浅**身份，直到遇到首个含正文的子块。
- **不做向后粘连**：当下一块**不更深**（更浅/同级标题，或已到末尾）时，该 heading-only 块原样留给 Stage D。**不会**把它向后并入更深的前块（如 `### 2.3.9`）——把更浅的 `## 2.4` 标题吞进更深的 L3 块会倒置层级（深吞浅）、压低标题层级。这类孤立标题直接交给 Stage D 正常处理。
- **保住硬上限**：子块来自 Stage C、本在 `target_max` 内，但前缀拼入父标题行后可能超限。由于下游无人会重切超限块（Stage D 只阻止其继续变大），绑定后超限的块在此重切：**先剥离开头的标题行**，正文按**完整 `target_max`** 切分（使后续不含前缀的正文片保持完整预算），再把标题前缀拼回**第一个正文片**。仅当第一个正文片大到放不下前缀时，才单独对它用缩减后的上限再切——因此大前缀不会把整个子节切得过碎。这样标题始终随真实正文，绝不会被单独切成一个 heading-only 孤块（否则 Stage D 又会把它向后吸走），且每个产出片段仍 ≤ `target_max`。（退化情形：当前缀本身已吃满上限——极长标题或极小 `chunk_token_size`——无法保持完整，则整块直接切分、对超长标题行做字符级切分;此时 cap 优先于保持标题完整。）
- **不额外回填前块**：由于 `keep="left"` 保留父块的 `level`，绑定后的整体只是个普通小块（并非锁定独立）。是否并回前块 `2.3` 完全沿用 Stage D 既有规则——前块仍 < `target_ideal` 时走同级合并，或整体小于 `small_tail_threshold` 时走尾部吸收（即便前块已饱和也能被吸入），二者都以重测后的真实 token ≤ `target_max` 为界。本前置步骤只保证标题**不脱离其子内容**，并不把整体锁为独立块——因此在尺寸允许时让 `2.3 + 2.4 + 2.4.1` 同块正是期望的防过碎行为。

> 边界歧义：正文行若真以 `#␠` 开头会被误判为标题行——这是 `lightrag/parser/_markdown.py` 已记录并接受的同一启发式歧义，实际语料中概率极低。

## 9. Stage D：层级感知的双相位合并

Stage D 解决细碎章节场景下「块过碎」和「跨主题污染」的矛盾。核心思想是**自深层级向浅层级处理**，先合并同级小块，再允许浅层块吸收深层块，同时引入尺寸约束、表格切片角色约束和标题路径约束。

### 9.1 D.0 合并约束（每次合并都要满足）

1. **尺寸约束**：合并后的真实文本 token 数不超过 `target_max`；已达到 `target_ideal` 的块原则上不继续参与普通同级合并。
2. **角色约束**：`middle` 表格切片锁定独立；`first`、`last` 按方向参与合并，防止表格边界上下文被错误吞并。
3. **层级约束**：同级合并在相同 `level` 之间发生；跨级吸收只允许浅层吸收深层，**禁止深层反向吸收浅层**。
4. **父标题路径一致性约束**：避免跨主题污染的关键，按合并方向取严格语义——
   - **同级合并（Phase A / 尾部吸收）**：两块 `parent_headings` 必须**完全相等**（真·兄弟）。仅 `level` 相同但父链不同（如 `2.4.1` 与 `2.5.1`）不允许合并。
   - **跨级吸收（Phase B，浅吸深）**：深块必须是浅块的**后代**——浅块的完整标题路径（`parent_headings` + 自身 `heading`，已剥离 `[part n]`）是深块 `parent_headings` 的前缀。浅块吞并不同分支的深块被禁止。
   - `parent_headings` 为空（preamble / 无层级输入）的块视为路径相容，放行（无层级可污染）。

### 9.2 D.1 Phase A：同级合并

针对当前 level 的相邻块，当**当前块**低于 `target_ideal`、合并后真实 token ≤ `target_max`，且满足上述约束时，合并为一个块（被吸收的邻块不要求低于 `target_ideal`；反向合并时另要求前块也 < `target_ideal`）。

表格切片角色的方向规则：

| 块角色 | 可向后吸收下一块 | 可被前一块吸收 |
|---|:-:|:-:|
| `none` | 是 | 是 |
| `first` | 是 | 否 |
| `middle` | 否 | 否 |
| `last` | 否 | 是 |

### 9.3 D.2 尾部整批吸收

若一个已达到 `target_ideal` 的块后面紧跟一串同级小块，且该串小块总 token 数低于 `small_tail_threshold`、合并后真实 token 数不超过 `target_max`，则**一次性吸收**该串小块。遇到 `middle` 表格切片时停止。

### 9.4 D.3 Phase B：跨级吸收

对于 Phase A 后仍未饱和的小块，尝试跨级合并，但仅允许浅层吸收深层：

- 当前块比后一块更浅时，当前块可向后吸收后一块。
- 当前块比前一块更深时，前一浅层块可吸收当前块。
- 反方向合并被禁止。
- 跨级阶段允许 `last` 角色向后吸收；`middle` 仍不参与合并。

### 9.5 D.4 合并后真实 token 复测

由于合并时会插入换行连接符，逐块 token 数相加可能低估合并结果。**每次提交合并前，都要对拼接后的真实文本重新计算 token 数**，确认不超过 `target_max` 后再提交。

合并后保留主块的 `heading`。如果多个 part 片段被合并，最终 heading 保留主块的 part 后缀，**不会**额外拼接多个 part 标签。

## 10. Fallback 与降级路径

P 策略有多层降级保护：

| 触发条件 | 降级行为 |
|---|---|
| `blocks_path` 缺失、不可读、无有效 content 行 | 整体降级到 `chunking_by_recursive_character()`，传入解析出的 `chunk_overlap_token_size` |
| Stage B 中表格无法识别 JSON / HTML 结构 | 该表格调用 R 策略字符切分 |
| Stage B 中单行表格自身超过 `table_max` | 该单行调用 R 策略字符切分 |
| Stage C 中长块没有合格短段落锚点 | 表格优先 → 贪心打包 → 单段落超长再降级 R 字符切分 |

**重要**：整体 fallback 后不再具备标题层级、表格角色和桥接文字双向重叠能力；但能保证文档仍产生检索块，不因结构化 sidecar 缺失而被静默丢弃。

## 11. 配置项

| 配置 | 默认 | 说明 |
|---|---|---|
| `CHUNK_P_SIZE` | `2000`（未设时使用 `DEFAULT_CHUNK_P_SIZE`，**不**沿用 `CHUNK_SIZE`） | P 专用 `chunk_token_size`；段落语义合并需要比全局默认更大的上限，因此独立默认而非回退到 `CHUNK_SIZE` |
| `CHUNK_P_OVERLAP_SIZE` | 未设（沿用 `CHUNK_OVERLAP_SIZE`） | P 专用 overlap；只影响同一内容行内长正文 fallback 和表格桥接预算，**不**让表格行级切片互相重叠 |
| `CHUNK_OVERLAP_SIZE` / `LightRAG(chunk_overlap_token_size=…)` | `100` | 未设 P 专用 overlap 时的全局兜底 |

配置语法、优先级链、`addon_params["chunker"]` 运行时改值等详见 [FileProcessingConfiguration-zh.md](FileProcessingConfiguration-zh.md) §3。

启用 P 的典型 `LIGHTRAG_PARSER` 写法：

```bash
LIGHTRAG_PARSER=docx:native-P,*:legacy-R
CHUNK_P_SIZE=2000
CHUNK_P_OVERLAP_SIZE=100
```

或在单文件覆盖：

```text
my-proposal.[native-P].docx
```

## 12. 分块效果检验

### 12.1 检查 sidecar 是否生成

确认 native parser 是否成功产生 `.blocks.jsonl`：

```bash
ls -l INPUT/__parsed__/<doc>.docx.parsed/<doc>.blocks.jsonl
```

若文件不存在或为空，P 策略会整体降级为 R，不会获得 P 的任何收益。常见原因：

- 未配置 `LIGHTRAG_PARSER=docx:native-...`。
- 解析失败（看 `pipeline_status` 错误条目）。
- 文档不是 DOCX（其他格式不支持 P）。

### 12.2 检查 blocks.jsonl 内容

每行一个 JSON，过滤 `type == "content"` 后查看 heading / level / parent_headings 是否符合预期：

```bash
jq -c 'select(.type=="content") | {level, heading, parent_headings}' \
   INPUT/__parsed__/<doc>.docx.parsed/<doc>.blocks.jsonl | head
```

若 heading 大量为空或 level 异常，说明 native parser 没正确识别标题样式 —— 此时 P 策略的层级合并和锚点提升都会失效。

### 12.3 检查最终 chunks

查看 `text_chunks` 存储中的 chunk 元数据：

```bash
jq '.[] | {heading, level, tokens, parent_headings}' \
   rag_storage/kv_store_text_chunks.json | head -30
```

应观察到：

- 大表前后块的 heading 通常对应 `[part 1]` / `[part n]`（说明 Stage B 拆分发生）。
- 细碎条款被合并到接近 `target_ideal` 的块（说明 Stage D 生效）。
- `parent_headings` 在不同章节切换处发生跳变，同章节内保持稳定。

### 12.4 块尺寸分布检验

理想分布：大多数 chunk 落在 `[target_ideal, target_max]` 区间（即 N=2000 时约 1500~2000 token）；明显偏小的块通常是 `middle` 表格切片（锁定独立）或紧靠章节边界的尾块。

若出现大量低于 `small_tail_threshold` 的尾块，可能是：

- 父标题路径一致性约束过严（不同 `parent_headings` 的相邻小块无法合并）。
- 大量 `middle` 表格切片堆积（表格本身就很大）。

## 13. 错误调试

### 13.1 P 没生效，输出与 R 一致

按以下顺序排查：

1. `full_docs[doc_id].process_options` 是否包含 `P`？
2. `full_docs[doc_id].parse_format` 是否为 `lightrag`？若为 `raw`，说明走的是 legacy 路径，P 会自动降级到 R。
3. `lightrag_document_path` 指向的 `.blocks.jsonl` 是否存在、是否非空？
4. 日志中是否有 `paragraph_semantic ... fallback to recursive_character` 字样？

### 13.2 表格被切散、前后说明分离

- 检查表格是否真的被识别为 `<table format="json">` 或 `<table format="html">`（看 `.blocks.jsonl`）。未识别格式的表格只能走字符切分，无法启动 Stage B 的角色机制。
- 检查表格 token 数是否真的超过 `table_max`。低于阈值的表格保持完整，不会触发首/中/末切片。
- 若是连续大表，确认两张表之间的桥接文字是否在**同一 content 行**内 —— 跨 content 行的桥接不参与 B.1 双向重叠。

### 13.3 细碎条款没有被合并

- 检查相邻条款的 `parent_headings` 是否一致：父标题路径一致性约束会阻止跨主题合并。
- 检查 `level` 是否一致：同级合并要求相同 `level`，跨级吸收只允许浅吸深。
- 检查中间是否插入了 `middle` 表格切片：会阻断尾部整批吸收。

### 13.4 出现单个超过 `target_max` 的块

正常情况下 Stage D 的真实 token 复测会拒绝超限合并，但以下场景仍可能出现超限块：

- 单行表格自身超过 `target_max`，无锚点可拆，最终走 R 字符切分但单 chunk 仍超限。
- `enforce_chunk_token_limit_before_embedding` 在 embedding 前会做最后的硬切分，下游不会真把超限 chunk 嵌入向量库。

### 13.5 `[part n]` 后缀异常

- 同一原始 content 行拆出多片但只看到一个 `[part 1]`：检查是否在 Stage D 中被合并 —— 合并后保留主块的 part 后缀，不拼接多个。
- 出现旧式 `[表格片段N]` 后缀：说明使用了旧版 chunker 输出的数据，新版统一为 `[part n]`，需要重新分块。

### 13.6 日志关键字

P 策略相关日志关键字（用于 `grep` 排查）：

- `paragraph_semantic` — 模块入口
- `fallback to recursive_character` — 整体或单段落降级
- `table_chunk_role` — 表格角色相关
- `bridge` — Stage B.1 桥接文字处理
- `anchor` — Stage C 锚点选择
