# Native 多模态 surrounding context 修改计划

## 目标

在 native / LightRAG Document 内容提取链路生成 sidecar 文件时，为 `drawings.json`、`tables.json`、`equations.json` 中对应条目补充同一 `blocks.jsonl` content 行内的前后文：

```json
"surrounding": {
  "leading": "前导文本",
  "trailing": "尾随文本"
}
```

`leading` 和 `trailing` 分别不超过 2000 tokens，截断优先按段落 / 句子边界，无法满足时才按字符兜底截断。

## 当前代码入口

- DOCX native 解析和 sidecar 生成集中在 `lightrag/native_parser/docx/lightrag_adapter.py`。
- 通用 parser content-list 到 LightRAG Document 的 sidecar 生成集中在 `lightrag/pipeline.py::_write_lightrag_document_from_content_list`。
- `blocks.jsonl` 的 content 行保存同一章节段落内容，sidecar 条目通过 `blockid` 指回该行。
- `i` / `t` / `e` 开关在 `lightrag/pipeline.py::analyze_multimodal` 中生效，只控制是否写入 `llm_analyze_result`；`surrounding` 不再受这些开关控制。
- recursive character 的分隔符配置来自 `CHUNK_R_SEPARATORS`，默认值在 `lightrag/constants.py::DEFAULT_R_SEPARATORS`。
- 表格 JSON / HTML 按行切分已有参考实现位于 `lightrag/chunker/paragraph_semantic.py`。

## 总体方案

把 surrounding context 作为 parse-stage sidecar enrichment 实现，而不是作为 multimodal analysis 前置步骤：

1. DOCX native parser 或通用 content-list writer 写出 `.blocks.jsonl` 和 sidecar JSON 后，立即调用 sidecar enrichment helper。
2. helper 读取 `blocks.jsonl` content 行和所有已存在的 `drawings` / `tables` / `equations` sidecar 文件。
3. 通过 sidecar 条目的 `blockid` 定位同一 content 行，再通过条目 `id` 定位该行中的目标 `<drawing ... />`、`<table ...>...</table>` 或 `<equation ...>...</equation>` 标签。
4. 仅在目标标签前后的同一 content 字符串内抽取 `leading` / `trailing`，不跨 content 行。
5. 将结果写回 sidecar，随后再持久化 `full_docs` 并进入后续 VLM / chunking / KG 流程。

选择该入口的原因：

- sidecar 文件在内容抽取后即保持稳定，不会因为是否启用 VLM 分析而变化。
- 可以使用 `self.tokenizer` 精确计算 2000 token 上限，而不是 DOCX parser 里的估算 token。
- sidecar 已经完成 id 重写，目标对象定位稳定。
- `i` / `t` / `e` 只决定后续是否生成 `llm_analyze_result`，不影响 parse artifacts 的完整性。

## 新增模块建议

新增 `lightrag/multimodal_context.py`，放置纯 helper，避免继续膨胀 `pipeline.py`。

建议公开函数：

```python
def enrich_sidecars_with_surrounding(
    *,
    blocks_path: str,
    enabled_modalities: set[str],  # parse-stage 调用传入 {"drawings", "tables", "equations"}
    tokenizer: Tokenizer,
    max_tokens: int = 2000,
    separators: list[str] | None = None,
) -> dict[str, int]:
    """补写 sidecar surrounding，返回每类 modality 更新条目数。"""
```

内部主要 helper：

- `load_content_rows_by_blockid(blocks_path) -> dict[str, str]`
- `load_chunk_separators() -> list[str]`：读取 `CHUNK_R_SEPARATORS`，失败时回退 `DEFAULT_R_SEPARATORS`。
- `find_target_span(kind, item_id, block_content) -> tuple[int, int] | None`
- `build_surrounding(kind, block_content, span, tokenizer, max_tokens, separators)`
- `trim_leading(...)` / `trim_trailing(...)`
- `remove_table_tags(text)`：只用于表格对象的 surrounding 候选文本预处理。
- `trim_table_atom_by_rows(...)`：只用于图片 / 公式 surrounding 中的表格片段按行裁剪。

## 对象定位规则

按 sidecar root key 映射目标标签：

- `drawings`：匹配完整自闭合标签 `<drawing ... id="dr-..." ... />`
- `tables`：匹配完整标签 `<table ... id="tb-..." ...>...</table>`；通用 parser writer 产生的 `<cite type="table" refid="tb-...">...</cite>` 也应作为 table marker 处理。
- `equations`：匹配完整标签 `<equation ... id="eq-..." ...>...</equation>`

实现细节：

- `id` 用 `re.escape` 转义。
- 属性顺序不能假设固定，只要求标签内存在对应 `id` 属性。
- 匹配结果必须返回完整标签 span，后续切分不得把 `<drawing />` 或 `<equation />` 切开。

## surrounding 抽取规则

对目标 span：

- `leading_source = block_content[:span_start]`
- `trailing_source = block_content[span_end:]`

然后按 modality 处理：

- 图片 / 公式：保留候选文本中的 `<drawing />`、`<equation>...</equation>` 和 `<table>...</table>`，但切分时保护完整标签。
- 表格：在 token 计算前，从候选文本中删除所有其它 `<table ...>...</table>` 和 table `<cite ...>` marker，删除后的文本再参与分段和 token 预算。

所有抽取都限制在当前 `block_content` 内，不读取前一个或后一个 blocks 行。

## 分段与截断算法

### 文本分段

使用与 recursive character 类似的 separator cascade：

1. 从 `CHUNK_R_SEPARATORS` 读取 JSON 数组。
2. 未配置或解析失败时使用 `DEFAULT_R_SEPARATORS`。
3. 分段顺序保持 strongest boundary first，例如 `\n\n`、`\n`、`。`、`！`、`？`、`；`、`，`、` `、`""`。

建议实现为“保护标签的 recursive splitter”：

- 先扫描文本，把 `<drawing ... />`、`<equation ...>...</equation>`、`<table ...>...</table>` 变成 atom。
- 普通文本 atom 可继续按 separators 递归切分。
- multimodal tag atom 默认不可被切开。
- 表格 atom 在图片 / 公式 surrounding 中允许用 row-aware table trim 生成较小的完整 `<table ...>...</table>` 片段。

### leading

从目标前方最近的 segment 开始，反向累积：

1. 如果加入下一个完整 segment 后 token 数仍不超过 2000，则保留。
2. 如果会超限，停止。
3. 如果最近的 1 个 segment 自身已超过 2000 tokens，则允许按字符截断该 segment 的尾部，但必须修复/避开半截 `<drawing />`、`<equation>` 标签。

### trailing

从目标后方最近的 segment 开始，正向累积：

1. 如果加入下一个完整 segment 后 token 数仍不超过 2000，则保留。
2. 如果会超限，停止。
3. 如果最近的 1 个 segment 自身已超过 2000 tokens，则允许按字符截断该 segment 的头部，但必须修复/避开半截 `<drawing />`、`<equation>` 标签。

### 字符兜底截断

字符截断用二分查找找到最长字符前缀/后缀，使 `len(tokenizer.encode(candidate)) <= max_tokens`。

截断后执行标签完整性修复：

- 如果结果落在 `<drawing ... />` 内部，删除整个不完整标签。
- 如果结果落在 `<equation ...>...</equation>` 内部，删除整个不完整标签。
- 表格对象的 surrounding 已预先删除所有表格标签，不需要修复表格。
- 图片 / 公式 surrounding 中的表格 atom 不直接字符切原始表格，优先使用行结构裁剪。

## 表格处理细节

### 表格对象 surrounding

对 `tables.json` 条目：

- `leading_source` / `trailing_source` 在任何 token 计算前删除 `<table ...>...</table>`。
- 删除包含目标外的所有其它表格，也包含候选文本里任何残留表格。
- 删除后再分段、截断、计 token，保证 token 数和最终保存内容一致。

### 图片 / 公式 surrounding 中的表格

对 `drawings.json` 和 `equations.json` 条目：

- `<table ...>...</table>` 可以保留在 surrounding 中。
- 如果完整表格加入后仍不超过 token 预算，直接保留完整表格。
- 如果表格超预算，优先按行结构裁剪：
  - JSON 表格：解析 body 为 list，leading 取靠近目标的尾部 rows，trailing 取靠近目标的头部 rows，再用原 attrs 重包成 `<table ...>{rows_json}</table>`。
  - HTML 表格：参考 `paragraph_semantic.py` 的 `<tr>` 扫描逻辑，leading 取尾部 rows，trailing 取头部 rows，保留 `<thead>` / `<tbody>` / `<tfoot>` 分组结构后重包。
  - 单行本身超限时，保留外层 `<table ...></table>` 完整性，内部按字符兜底生成不超过预算的片段；优先保持 JSON 可序列化，HTML 则尽量不切开 `<tr>` / `<td>` 标签。

为减少重复代码，可以把 `paragraph_semantic.py` 中的 HTML row scanner 和 JSON row token split 逻辑抽到共享模块，例如 `lightrag/table_markup.py`，再由 paragraph semantic chunker 和 surrounding extractor 共同使用。

## pipeline 集成点

在 parse-stage writer 完成 `.blocks.jsonl` 和 sidecar JSON 写出后调用：

1. `parse_native` 的 DOCX pending-parse 分支：`parse_docx_to_lightrag_document(...)` 返回后、`_persist_parsed_full_docs(...)` 前调用。
2. `_write_lightrag_document_from_content_list(...)`：写出 `.blocks.jsonl`、`.drawings.json`、`.tables.json`、`.equations.json` 后、`_persist_parsed_full_docs(...)` 前调用。
3. 调用时传入全部 modality；helper 会自动跳过不存在的 sidecar：

```python
enrich_sidecars_with_surrounding(
    blocks_path=str(blocks_path),
    enabled_modalities={"drawings", "tables", "equations"},
    tokenizer=self.tokenizer,
)
```

4. 记录 debug/info 日志，例如每类 modality 更新数量。
5. `analyze_multimodal` 不再负责补写 `surrounding`，只按 `i` / `t` / `e` 决定是否生成或跳过 `llm_analyze_result`。

可选增强：VLM prompt 中加入 `surrounding.leading` / `surrounding.trailing`，让模型分析图片、表格、公式时能利用同一章节段落上下文。该增强不影响本次 sidecar schema 目标，但建议同一 PR 内完成。

## sidecar schema

每个条目新增可选字段：

```json
{
  "id": "dr-doc-0001",
  "blockid": "...",
  "heading": "...",
  "surrounding": {
    "leading": "...",
    "trailing": "..."
  }
}
```

兼容性：

- 旧 sidecar 没有 `surrounding` 时仍可被读取。
- 下游现有字段不变。
- `version` 可保持 `1.0`，因为这是向后兼容的可选字段；如果项目希望显式标记 schema 变更，可在同一改动中升到 `1.1`。

## 测试计划

新增单元测试优先覆盖 helper，不依赖真实 DOCX：

1. `drawings.json`：同一 block 内图片前后文本写入 `surrounding`，不跨 blocks 行。
2. `equations.json`：公式 surrounding 保留完整 `<drawing />` / `<equation>` 标签，不出现半截标签。
3. `tables.json`：表格 surrounding 在 token 计算前删除候选文本中的其它 `<table ...>...</table>`。
4. `CHUNK_R_SEPARATORS`：覆盖自定义 separator，确认按配置边界截断。
5. 超长最近句子：最近 segment 超 2000 tokens 时按字符截断，最终 token 数不超过 2000。
6. 图片 / 公式 surrounding 含 JSON 表格：超预算时按 rows 从靠近目标的一侧裁剪并重包 `<table>`。
7. 图片 / 公式 surrounding 含 HTML 表格：超预算时优先按 `<tr>` 裁剪，保留 row wrapper。
8. 幂等性：parse-stage 重跑时已有 valid `surrounding` 可被重算或保留，行为固定；`llm_analyze_result` 不应影响 `surrounding` 补写。
9. 通用 parser writer 的 table `<cite type="table" refid="...">` marker 可定位并生成 `tables.json` surrounding。

集成测试建议放在 `tests/test_parse_native_lightrag_e2e.py` 或新增 `tests/test_multimodal_surrounding_context.py`：

- stub DOCX native parser 输出包含 drawing / table / equation 的 block，调用 `parse_native(...)`，断言三类 sidecar 在 parse 完成后已经补写 `surrounding`，无需启用 `i` / `t` / `e`。
- 构造 `_write_lightrag_document_from_content_list(...)` content-list，断言通用 writer 的 sidecar 在写出后已经补写 `surrounding`。

## 实施顺序

1. 新增 helper 模块和基础 tag 扫描、blockid 映射、separator 配置读取。
2. 实现表格对象的预删除逻辑，并先覆盖 `tables.json` surrounding。
3. 实现图片 / 公式的 tag atom 保护和基本 leading/trailing 截断。
4. 增加 JSON / HTML 表格 row-aware 裁剪，必要时抽共享表格 helper。
5. 集成到 parse-stage sidecar writer，确保 `full_docs` 持久化前完成 sidecar 补写。
6. 从 `analyze_multimodal` 移除 `surrounding` 补写职责，保持 VLM 阶段只处理 `llm_analyze_result`。
7. 增加单元测试和最小集成测试。
8. 如选择让 VLM 使用 surrounding，再更新 prompt 并补一条 prompt 输入断言测试。

## 风险与注意点

- 不要在 DOCX parser 内部用估算 token 做最终限制；parse-stage enrichment 必须使用 `self.tokenizer` 保证 2000 tokens 上限。
- 不要用简单字符串切片直接截断包含标签的文本，必须先保护 tag atom。
- 表格对象 surrounding 必须先删除表格再计 token，不能先算 token 后删除。
- HTML 表格用 regex 行扫描只能覆盖常规 `<tr>` 结构；异常 HTML 需要降级但不能破坏外层 `<table>` 标签完整性。
- sidecar 写回应保持 UTF-8、`ensure_ascii=False`、`indent=2`，与现有文件风格一致。
- parse artifacts 应在内容抽取后保持稳定，不应依赖后续 `i` / `t` / `e` VLM 开关才补写 `surrounding`。
