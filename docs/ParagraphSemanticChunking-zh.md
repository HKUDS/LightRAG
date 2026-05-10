# Paragraph Semantic Chunking 设计说明

## 1. 设计目标

Paragraph Semantic Chunking（下文简称 **P 策略**）面向 Word / DOCX 等包含标题层级、正文段落和大表格的文档，目标是在尽量保留文档结构语义的前提下生成适合 RAG 检索的文本块。**P 策略**仅针对内容提取引生成的生成 结构化`.blocks.jsonl` 段落块的进行重新分块。

核心目标：

1. **标题驱动**：优先尊重 `blocks.jsonl` 中由标题层级形成的内容块，而不是只按固定 token 数硬切。
2. **表格优先完整**：表格低于 `table_max` 时保持完整；超出 `table_max` 的大表格按行边界拆分，并为表格片段保留上下文。
3. **允许必要重叠**：同一个 `blocks.jsonl` 内容行过长需要拆分时，允许文本重叠；表格之间的桥接文字也允许同时出现在前后两个表格块中。
4. **块大小受控**：尽量接近目标大小，并避免超过 `chunk_token_size`；不可再结构化拆分的单元会降级为递归字符切分。
5. **层级感知合并**：先同级合并小块，再尝试跨级吸收小块，减少碎片。

**P 策略**是 Recursive Character 分块方法的一个改进版。可以保证章节边界的文本块不会出现内容重叠，从而避免分块内容的张冠李戴。结构化表格行拆分本身仍按行边界非重叠。章节内的长文本与 Recursive Character 分块方法相同。本策略最适合用于章节层次结果的文档分块。

> 代码实现位置：`lightrag/chunker/paragraph_semantic.py`。

## 2. 输入与输出

### 2.1 输入

`chunking_by_paragraph_semantic()` 的主要输入包括：

- `content`：原文内容或 fallback 文本。
- `blocks_path`：native parser 生成的 `.blocks.jsonl` 路径。
- `chunk_token_size`：目标最大 token 数，默认 `2000`。
- `chunk_overlap_token_size`：同一原始块内长文本 fallback 与表格桥接上下文使用的最大 overlap，默认 `100`，可由 P 专用配置覆盖。
- `tokenizer`：LightRAG 已解析好的 tokenizer 对象，用于所有 token 计数和文本 overlap 截取。

P chunker 不接收 `split_by_character` / `split_by_character_only` 这类分隔符参数，因为正常路径是标题和段落结构驱动。若 `blocks_path` 缺失、不可读或没有可用内容行，P 策略会降级调用 `chunking_by_recursive_character()`，并传入 P 策略解析出的 `chunk_overlap_token_size`。

### 2.2 `.blocks.jsonl` 约定

P 策略只处理 `type == "content"` 的行。每个内容行通常包含：

- `content`：该标题下的正文文本，可能包含普通段落、表格标签、公式或绘图标签。
- `heading`：当前标题。
- `parent_headings`：父级标题链。
- `level`：标题级别。

native parser 的 `fixlevel=0` 保持一条标题下的正文作为一个基础内容块。P 策略会在此基础上继续拆分过大的内容块。

### 2.3 输出

最终输出为 chunk 列表，每个元素包含：

```python
{
    "tokens": int,
    "content": str,
    "chunk_order_index": int,
    "heading": str,
    "parent_headings": list[str],
    "level": int,
}
```

实现内部还会临时使用 `paragraphs`、`table_chunk_role` 等字段辅助拆分和合并，但这些字段不会进入最终公开输出。

当同一个 `blocks.jsonl` 内容行最终拆成多个片段时，所有片段的 `heading` 会追加行内编号后缀：

- 第一片：`原标题 [part 1]`
- 第二片：`原标题 [part 2]`
- 依此类推

未发生拆分的内容行不会追加 `[part 1]`。`parent_headings` 不追加 part 后缀。旧的 `[表格片段N]` 后缀已由统一的 `[part n]` 后缀替代。

## 3. 关键阈值

P 策略的阈值不是固定 token 常量，而是根据 `chunk_token_size` 动态计算：

| 名称 | 计算方式 | `chunk_token_size=2000` 时 |
|---|---:|---:|
| `target_max` | `chunk_token_size` | 2000 |
| `target_ideal` | `0.75 * chunk_token_size` | 1500 |
| `table_max` | `0.625 * chunk_token_size` | 1250 |
| `table_ideal` | `0.375 * chunk_token_size` | 750 |
| `table_min_last` | `0.32 * table_max` | 400 |
| `small_tail_threshold` | `0.125 * chunk_token_size` | 250 |
| `max_anchor_candidate_length` | 固定字符数 | 100 |

这些比例来源于早期审计模式中“大块 8000、小表 5000、理想表 3000、表格尾块 1600”等经验值，但当前实现会随 `chunk_token_size` 缩放。

## 4. 总体流程

```text
native parser fixlevel=0
  ↓
.blocks.jsonl 标题级基础块
  ↓
Stage B：超大表格按结构化行边界拆分，并处理表格间桥接上下文
  ↓
Stage C：仍超大的普通内容块按短段落锚点拆分；无锚点时降级为递归字符切分
  ↓
按原始 content 行追加 [part n] 后缀
  ↓
Stage D：同级合并小块，必要时跨级吸收
  ↓
最终 chunks
```

## 5. Stage A：标题级基础块

当前 P chunker 不直接扫描 docx body，也不在 chunker 内判断标题样式。标题识别和基础内容块生成由 native parser 完成。

对于 docx 输入：

1. native parser 识别标题层级。
2. `fixlevel=0` 按标题边界生成 `.blocks.jsonl`。
3. P chunker 读取其中的 `content` 行。
4. 每个 content 行作为后续 Stage B / C 的独立处理单元。

这意味着 `[part n]` 编号会按每个原始 content 行独立重置。

## 6. Stage B：超大表格拆分

Stage B 只处理超过 `table_max` 的表格。表格在文本中以 `<table ...>...</table>` 包裹，当前实现支持：

- `format="json"`：按 JSON 顶层行数组拆分。
- `format="html"`：按 HTML `<tr>` 行拆分。
- 未显式标注但内容可嗅探为 JSON / HTML 的表格。

如果表格可结构化拆分，算法会尽量按行组合出接近 `table_ideal` 的表格片段，并保留必要的 table wrapper 开销。若某个行组合仍然过大，会继续按行递归拆分；如果最终只剩不可再拆的一行或无法识别结构，则降级为递归字符切分。

表格片段内部使用 `table_chunk_role` 辅助后续合并：

| 角色 | 含义 |
|---|---|
| `first` | 原始表格的第一段 |
| `middle` | 原始表格的中间段 |
| `last` | 原始表格的最后一段 |
| `none` | 非表格片段或完整表格 |

`table_chunk_role` 是内部字段，最终输出不会保留。

### 6.1 表格之间的桥接文字

当一个原始内容行中存在：

```text
大表格 A
短文字或中等长度文字
大表格 B
```

并且 A、B 都被拆成表格片段时，中间文字会按上下文预算分配：

- 前表格最后一片获得桥接文字前缀作为后文上下文。
- 后表格第一片获得桥接文字后缀作为前文上下文。
- 如果桥接文字足够短，前后两侧可以同时包含完整桥接文字。
- 如果桥接文字超过两侧上下文预算，剩余中间部分会作为独立文本块保留。

单侧上下文预算按 token 计算，最大为 `chunk_overlap_token_size`，并会被限制到不超过 `chunk_token_size / 2`。实际可用预算还受当前表格片段距离 `chunk_token_size` 的剩余容量限制，避免表格上下文块超过最大 chunk 大小。

## 7. Stage C：长普通内容块拆分

Stage C 处理 Stage B 后仍超过 `chunk_token_size` 的块。

### 7.1 短段落锚点

实现会先把内容按段落拆开，并寻找可作为锚点的短段落：

- 段落不是表格。
- 段落文本长度不超过 `100` 个字符。
- 锚点不能是第一个段落，避免递归无法收敛。

若存在锚点，算法优先在靠近目标大小的位置切分，并把锚点提升为后半段的 `heading`；原 heading 进入后半段的 `parent_headings`。

### 7.2 无锚点 fallback

如果没有合适锚点，当前实现不会报错退出，而是按以下方式降级：

1. 表格段落优先走 Stage B 的行边界拆分能力。
2. 可组合的小段落会贪心打包。
3. 单个过长的普通文本段落会调用 `chunking_by_recursive_character()`。

这里的递归字符切分会传入 `chunk_overlap_token_size`，因此相邻文本块之间会保留配置化 overlap。

## 8. Stage D：小块合并

Stage D 用于减少过碎的小块，并尽量不破坏表格片段顺序。

### 8.1 同级合并

优先在相同 `level` 的相邻块之间合并。常规同级合并受 `table_chunk_role` 约束：

| 当前块角色 | 可向后吸收下一块 | 可被前一块吸收 |
|---|---|---|
| `none` | 是 | 是 |
| `first` | 是 | 否 |
| `middle` | 否 | 否 |
| `last` | 否 | 是 |

另外，若一个已达到 `target_ideal` 的同级块后面紧跟一小段同级尾块，且这段尾块总量低于 `small_tail_threshold`、合并后不超过 `chunk_token_size`，实现会一次性吸收这段尾块；遇到 `middle` 表格片段会停止吸收。

### 8.2 跨级吸收

如果同级合并后仍有小块，会尝试和相邻块跨级合并：

1. 当当前块比后一个块更浅时，优先让当前块向后吸收后续深层块。
2. 若不能向后吸收，且前一个块更浅，则让前一个浅层块吸收当前块。
3. 合并仍必须满足 token 上限与表格角色约束；跨级阶段允许 `last` 角色向后吸收，但 `middle` 仍不参与合并。

合并后保留主块的 `heading`。如果多个 part 片段被合并，最终 heading 保留被保留下来的主块 part 后缀，不额外拼接多个 part 标签。

## 9. Fallback 与配置

当无法使用 `.blocks.jsonl` 时，P 策略会直接降级到 `chunking_by_recursive_character()`。这使得 P 策略可作为通用 chunker 配置使用，但只有 native parser 提供结构化 sidecar 时才具备标题和表格感知能力。

相关配置：

- `CHUNK_P_SIZE`：P 策略专用 chunk 大小。
- `CHUNK_P_OVERLAP_SIZE`：P 策略专用 overlap 大小。
- `CHUNK_OVERLAP_SIZE` / `LightRAG(chunk_overlap_token_size=...)`：未设置 P 专用 overlap 时的全局 fallback。

## 10. 与 R 策略对比

| 维度 | R：Recursive Character | P：Paragraph Semantic |
|---|---|---|
| 切分依据 | 字符分隔符 + token 预算 | 标题层级、段落、表格行边界、短段落锚点 |
| 表格处理 | 可能在任意位置切断 | 低于 `table_max` 保持完整；超大表格按 JSON / HTML 行拆分 |
| 上下文重叠 | 由 recursive overlap 控制 | 章节边界不会出现重叠；段落内普通长文本 fallback 使用 overlap；表格按行切分无重叠；表格之前的桥接文字可复制到两侧 |
| heading | 通常无结构化 heading | 继承或提升 heading，拆分后追加 `[part n]` |

> 无结构 sidecar时 P 策略自动 fallback 回 R 策略。

## 11. 当前适用场景

适合：

- DOCX 中有清晰标题层级。
- 文档包含大表格，需要尽量按行保留结构。
- 检索时需要利用 `heading` / `parent_headings` 提供语义上下文。
- 同一标题下长段落拆分后允许少量重叠，以减少语义断裂。

不适合或收益有限：

- 没有 `.blocks.jsonl` sidecar 的输入，此时会退化为 R 策略。
- 标题样式混乱、正文中大量伪标题的文档。
- 单行超大表格或不可解析表格，这类内容最终仍可能走字符级 fallback。
