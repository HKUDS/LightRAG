# LightRAG Sidecar 文件格式说明

本文介绍内解析引擎输出的**LightRAG Sidecar**文件格式。LightRAG 在使用native/mineru/docling这些支持多模态内容解析引擎提取文件内容的时候，会把"正文 + 多模态对象 + 解析元数据"拆开写到一个 `*.parsed/` 目录中，目录内的每个 JSON / JSONL 文件统称为 **sidecar** 文件。Sidecar 是后续流水线（多模态分析 → 多模态 chunk 构造 → 实体抽取 → 文档删除时的缓存清理）唯一可靠的依据。Sidecar的文件格式是LightRAG内置的通用文件交换格式，新的多模态内容提取引擎都需要遵循这个格式。公开**LightRAG Sidecar**文件格式的目的是给社区开发者编写字节的内容解析引擎提供方便。

## 一、概述

| 关注点 | 文件 | 存放内容 | 说明 |
|---|---|---|---|
| 主文件 | `<doc>.blocks.jsonl` | 存放 Block 正文 | 所有 Block 的 content字段内容拼接后形成完整的原文 |
| 图形对象 | `<doc>.drawings.json` | 文件中抽取出来的图形对象 | 送VLM进行分析后回填分析结果 |
| 表格对象 | `<doc>.tables.json` | 文件中抽取出来的表格对象 | 送LLM进行分析后回填分析结果 |
| 公式对象 | `<doc>.equations.json` | 文件中抽取出来的公司对象 | 送LLM进行分析后回填分析结果 |
| 原始图像资源 | `<doc>.blocks.assets/` | 文件中抽取出来的图片原始文件 | 送VLM进行图片分析 |

Sidecar 的设计意图：

- 解析阶段 内容提取引擎(native/mineru/docling) **只**负责生成 `blockid / heading / content / surrounding` 等"客观"字段；
- 多模态分析阶段 (`analyze_multimodal`) **只**追加分析结果 `llm_analyze_result` 字典

## 二、目录布局

```
inputs/space1/__parsed__/<规范文件名>.parsed/
├── <规范文件名>.blocks.jsonl        正文块序列 + 文档级 meta（首行）
├── <规范文件名>.drawings.json       图形 sidecar（dict 容器，键 = 图形 id）
├── <规范文件名>.tables.json         表格 sidecar
├── <规范文件名>.equations.json      公式 sidecar
└── <规范文件名>.blocks.assets/      原始资源目录（存放drawings.json中的图片文件放这里）
    ├── image1.wmf
    ├── image2.wmf
    ├── image3.wmf
    ├── image4.png
    ├── image5.png
    ├── image6.png
    └── image7.emf
```

## 三、blocks.jsonl

`blocks.jsonl` 是按行序列化的 JSON，**第一行 `type="meta"`**，其余每行是一个内容块 `type="content"`。

### 3.1 meta 行实例

```json
{
  "type": "meta",
  "format": "lightrag",
  "version": "1.0",
  "document_name": "m012-manual.docx",
  "document_format": "docx",
  "document_hash": "sha256:4840...3f9543d9db0822d2d59",
  "table_file": true,
  "equation_file": true,
  "drawing_file": true,
  "asset_dir": true,
  "split_option": { "fixlevel": 0 },
  "blocks": 39,
  "doc_id": "doc-f1bee60173d067d88595c00e7d9b0ce5",
  "parse_engine": "native",
  "parse_time": "2026-05-13T18:42:25.943490+00:00",
  "doc_title": "m012-manual"
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `type` | `"meta"` | 行类型，固定值，校验位 |
| `format` | `"lightrag"` | sidecar 大版本族标识 |
| `version` | `str` | sidecar schema 版本 |
| `document_name` | `str` | 规范文件名（含后缀，不含处理指示） |
| `document_format` | `str` | 文件格式（目前以文件后缀表示） |
| `document_hash` | `"sha256:<hex>"` | 原始文件内容哈希，用于跨次解析去重 |
| `table_file` / `equation_file` / `drawing_file` | `bool` | 是否存在对应 sidecar 文件（为真时对应文件必然存在） |
| `asset_dir` | `bool` | 是否存在`blocks.assets`资源目录 |
| `split_option` | `object` | 文件提取时的分块参数。此字段留给文件提取引擎自己记录和使用 |
| `blocks` | `int` | content 行数（不含 meta） |
| `doc_id` | `"doc-<md5>"` | 文档全局 id。sidecar item id（`dr-/tb-/eq-`）使用 `doc_id` 去掉 `doc-` 前缀后的哈希部分，以缩短嵌入正文中的占位标签 |
| `parse_engine` | `str` | 解析引擎`native/mineru/docling/legacy` |
| `parse_time` | `str` | 解析完成时间; 格式：ISO-8601 UTC |
| `doc_title` | `str` | 文档标题（通常为首个 H1）；可选 |
| `doc_summary` | `str` | 文档摘要；可选 |
| `doc_attributes` | `object` | 文章扩展属性对象；可选 |
| `bbox_attributes` | `object` | 版面bbox框属性；可选<br />origin: `LEFTTOP` `LEFTBOTTOM`<br />max: `1000` # 宽或高坐标最大值 |

> LightRAG要求同一个workspace（知识库）内的文件名（document_name）必须唯一。

### 3.2 content 行

每个 content 行是一个原始文档"块"的最小可寻址单位，至少包含：

```json
{
  "type": "content",
  "blockid": "462c6364584a7ba4bdae6853f85ac429",
  "format": "plain_text",
  "content": "1 产品用途和功能\nMI012模块用于支撑供氧抗荷调节器的供氧抗荷控制功能...",
  "heading": "1 产品用途和功能",
  "parent_headings": [],
  "level": 1,
  "session_type": "body",
  "table_slice": "none",
  "positions": [
    {
      "type": "paraid",
      "range": ["5EA4577A", "6555DDCB"]
    }
  ]
}
```

| 字段 | 含义 |
|---|---|
| `type` | `"content"` |
| `blockid` | 全局唯一的Block ID |
| `format` | 内容形态，目前固定为 `"plain_text"` |
| `content` | 文本内容；**公式和图片此以占位标签出现，表格以带table标签的JSON或HTLM格式出现**（见 3.3） |
| `heading` | content所在章节的最高层级标题；heading真实存在时，应该同时出现在content的开头；如果heading之后紧接着下一个层级的heading，则把下一个层级的heading正文看待。这样做的目的是需要保证所有 Block 的 content字段内容拼接后形成完整的原文。 |
| `parent_headings` | 字符串数组: 自顶向下的祖先标题列表，不含当前 `heading` |
| `level` | 整数: `heading` 在文档大纲中的层级（`1` = H1 / 一级标题，0表示无标题） |
| `session_type` | Block所处区域：`body` `preface` `TOC` `references` `appendix` |
| `table_slice` | 可选保留字段；表示Block是否仅包括表格片段。目前分析引擎不会拆分长表格。因此本字段固定为 `"none"`（表示表格不会被分片） |
| `table_header` | 可选保留字段；在当前块位表格片段的时候，保存识别出来的表格头。目前不存在 |
| `positions` | `position` 对象数组：标识文本块的版面位置；文本块来与版面的多个位置的时候，则会出现多个`position` 对象。参见[§八](#八、position) |

> - blockid计算方式：`md5(doc_id + ":" + block_index + ":" + heading + ":" + content)`。文档经过分块策略处理得到的 chunk 将保存 blockid 用于溯源 chunk 在s idecar 中的位置。
> - 不关系文档章节结构的分块策略 `F` `R` `V` 使用的就是 content 字段拼接后的内容进行分块。因此需要保证所有 Block 的 content字段合并在一起能够构成完整的文档内容，不会缺少内容，不会出现重叠的内容。

### 3.3 content 内嵌占位标签

为了让 P 分块策略在不破坏多模态对象的前提下对正文做切分，`content` 文本里使用如下三种 XML 风格的占位标签：

| 标签 | 含义 | 标签属性 |
|---|---|---|
| `<table id="tb-…" format="json">…</table>` | 表格占位，包体是表格原始 JSON / HTML | `id` 指向 `tables.json` 里对应 item；`format` ∈ `json` / `html` |
| `<drawing id="dr-…" format="png" path="..." src="..." caption="..." />` | 自闭合图形占位 | `id` 指向 `drawings.json`；`path` 相对 `*.parsed/` 目录；`src` 是原文档里的引用名 |
| `<equation id="eq-…" format="latex">…</equation>` | 公式占位 | 行内公式同样用 `<equation format="latex">` 但**不**带 `id`，不会进 sidecar； 仅块公式（独占一行或多行）时携带 `id` |

在实体关系抽取的时候喂给大模型的文本会把 `id / path / src` 等内部属性剥掉，但为保留键属性（`format / caption`）。目的是避免抽取出文章不可见的实体，给抽取结果注入过多的噪声。

### 3.4 blockid 与 chunk sidecar.refs 的对应

葛总分块策略在sidecar文件存在时，会在其输出的每个 chunk 都会带上 `sidecar = {"type": "block", "id": <主来源 blockid>, "refs": [{"type": "block", "id": <blockid>}, ...]}`，其中：

- 未合并的 chunk → `sidecar.refs` 只有一个元素，等于该 chunk 来自的 blocks.jsonl 行的 `blockid`；
- Stage D 合并后的 chunk → `refs` 顺序保留所有来源 `blockid`（去重）；
- hard fallback split 后的子 chunk → 共享父 chunk 的 `sidecar`。

这条链路是文档级追溯（chunk ↔ block ↔ 原段落 paraId）的基础。

## 四、drawings.json

顶层是 `{"version": "1.0", "drawings": { <id>: <item>, ... }}` 形态的 dict 容器，**键 = `id` 字段**，便于按 id 查找。每个 item 形如：

```json
{
  "id": "dr-f1bee60173d067d88595c00e7d9b0ce5-0004",
  "blockid": "2f52b70839d13a936d97955916820147",
  "heading": "2.3 结构尺寸及重量",
  "format": "png",
  "path": "m012-manual.blocks.assets/image4.png",
  "src": "",
  "caption": "",
  "footnotes": [],
  "surrounding": {
    "leading": "2.3 结构尺寸及重量\n尺寸及重量要求如下：\na) 外廓尺寸长度为：<drawing …",
    "trailing": "\n图1　外廓尺寸示意\nb) 重量不大于0.85kg。\nc) 测试结果：实测电路噪声Vpp=1.526mV…"
  },
  "llm_analyze_result": {
    "name": "产品外廓尺寸工程图纸",
    "type": "Illustration",
    "description": "该图纸为产品的外廓尺寸示意图，展示了一个电子设备或电源模块的三视图设计…",
    "analyze_time": 1778697752,
    "status": "success",
    "message": ""
  },
}
```

| 字段 | 说明 |
|---|---|
| `id` | `dr-<doc_hash>-<NNNN>` 形式（`doc_hash` 为 `doc_id` 去掉 `doc-` 前缀后的 32 位 md5） |
| `blockid` | 指向产生该图形的 content 行 |
| `heading` | 所在章节标题 |
| `format` | 原始扩展名（去点）：`png` / `jpeg` / `gif` / `webp` / `wmf` / `emf` / … |
| `path` | 相对 `*.parsed/` 目录的资源路径，**永远**指向 `*.blocks.assets/` 内文件 |
| `src` | 原文档里图形的引用别名（多数情况下为空） |
| `caption` | 可见标题（解析器可能留空） |
| `footnotes` | 脚注字符串列表 |
| `surrounding` | 上下文对象：参见[§七](#七、surrounding) |
| `llm_analyze_result` | 模态分析结果对象：详见 [§八](#七、surrounding) |

**只有图形支持的 raster 格式（png / jpeg / gif / webp）才会进入 VLM 分析**；其他格式（wmf / emf / svg 等）写 `llm_analyze_result.status="skipped"`，下游不生成多模态 chunk，文档继续处理。图片大小超过环境变量`VLM_MAX_IMAGE_BYTES`规定的大小后，图片同样不会进入VLM分析。

## 五、tables.json

顶层是 `{"version": "1.0", "tables": { <id>: <item>, ... }}` 形态的 dict 容器，**键 = `id` 字段**，便于按 id 查找。每个 item 形如：

```json
{
  "id": "tb-f1bee60173d067d88595c00e7d9b0ce5-0007",
  "blockid": "3f33897b5e105d254addc655f1efbf8c",
  "heading": "2.4.4 温度-湿度-高度（随系统进行）",
  "dimension": [16, 8],
  "format": "json",
  "content": "[[\"试验步骤\", \"温度(℃)\", \"高度(m)\", \"相对湿度\", \"时间(min)\", \"辅助冷却\", \"系统电源\", \"功能、性能检查\"],…",
  "caption": "",
  "footnotes": [],
  "table_header": "[[\"试验步骤\", \"温度(℃)\", \"高度(m)\", \"相对湿度\", \"时间(min)\", \"辅助冷却\", \"系统电源\", \"功能、性能检查\"]]"
  "surrounding": {
    "leading": "2.4.4 温度-湿度-高度（随系统进行）\n产品应能承受执行任务期间的温度、湿度、高度环境综合作用…",
    "trailing": "\n注：以上步骤重复10个循环。a成品及附件达到温度稳定或240min，以长者为准；b成品及附件达到温度稳定或120min，以长者为准。…"
  },
  "llm_analyze_result": {
    "name": "文档管理元数据表",
    "description": "这是一份文档管理信息表，用于记录技术文档的基本元数据和版本控制信息 …",
    "analyze_time": 1778697759,
    "status": "success",
    "message": ""
  },
}
```

tables.json 文件的 `blockid` `heading` `surrounding` `llm_analyze_result` 字段与drawings.json相同。不同或新添加的字段说明如下：

| 字段 | 说明 |
|---|---|
| `id` | `tb-<doc_hash>-<NNNN>` 形式（`doc_hash` 为 `doc_id` 去掉 `doc-` 前缀后的 32 位 md5） |
| `dimension` | 整数数组：`[num_rows, num_cols]`，包含表头行 |
| `format` | `"json"` (二维数组) 或 `"html"` (负载 `<table>…</table>` 片段，含起止标签) |
| `content` | 字符串：表格正文，按 `format` 决定结构；这是后续多模态 chunk 真正使用的字符串。 |
| `table_header` | 字符串：可选；识别出来的作为表格头的行内容 |

在模态分析阶段，如果`content`字段长度超过大模型的上下文长度时，表格内容会被机械地截断后在喂给模型。

## 六、equations.json

顶层是 `{"version": "1.0", "equations": { <id>: <item>, ... }}` 形态的 dict 容器，**键 = `id` 字段**，便于按 id 查找。每个 item 形如：

```json
{
  "id": "eq-f1bee60173d067d88595c00e7d9b0ce5-0001",
  "blockid": "2f52b70839d13a936d97955916820147",
  "heading": "2.3 结构尺寸及重量",
  "format": "latex",
  "content": "C=2∗\\frac{P∗T}{\\left( {V}_{H}^{2}−{V}_{L}^{2} \\right)∗η}",
  "caption": "",
  "footnotes": [],
  "surrounding": {
    "leading": "2.3 结构尺寸及重量\n尺寸及重量要求如下：\n …",
    "trailing": "\n其中P为供电异常时维持的功率28W,T为期望储能时间，V<sub>H</sub>为电容放电前…"
  },
  "llm_analyze_result": {
    "name": "电容储能时间计算公式",
    "description": "该公式用于计算在电源异常情况下维持系统正常工作所需的电容储能值 …",
    "analyze_time": 1778697783,
    "status": "success",
    "message": "",
    "equation": "C=2\\cdot\\frac{P\\cdot T}{(V_{H}^{2}-V_{L}^{2})\\cdot\\eta}"
  }
}
```

equations.json 文件的 `blockid` `heading` `surrounding` `llm_analyze_result` 字段与drawings.json相同。不同或新添加的字段说明如下：

| 字段 | 说明 |
|---|---|
| `id` | `eq-<doc_hash>-<NNNN>` 形式（`doc_hash` 为 `doc_id` 去掉 `doc-` 前缀后的 32 位 md5） |
| `format` | 固定为 `"latex"` |
| `content` | 字符串：是**原始** LaTeX（可能包含 Unicode 运算符、外层 `\[ \]`），不包含两头的`$`分割符；模态分析阶段直接读这里 |
| `llm_analyze_result.equation` | 字符串：是大模型输出的**规范化**后的 LaTeX公式（外层 `$ / \[ \] / equation` 环境，Unicode 转 LaTeX，不包含联投的`$`分割符），这是后续多模态 chunk 真正使用的字符串； |

在模态分析阶段，如果`content`字段长度超过大模型的上下文长度时，表格内容会被机械地截断后在喂给模型。行内公式（与正文连续的 `<equation format="latex">…</equation>`）**不会**保存到 equations.json 文件，它仅会在 blocks 文本里以无 `id` 形式留存。这样做的目的是避免给抽取结果注入过多的噪音。

## 七、surrounding

`surrounding.leading` 和 `surrounding.trailing` 是 sidecar item 的可分析上下文窗口，目的是提供图片、表格和公式所在段落的上下文信息，提高多模态分析的质量。

- 取自同一 `blockid` 对应的 content 行文本，以多模态占位标签的位置为切分点；
- 每一侧不超过 `2000` tokens（按 tokenizer 截断，倾向保留靠近目标的句子）；
- 文本中保留**同行其他**多模态对象的占位标签，这让模型能感知"图 1 之后还有公式 1"这种上下文；
- 当目标对象本身位于 block 起点 / 终点时，对应一侧为 `""` 而不是 `"n/a"`（提示词组装时再把空字符串显示为 `n/a`）。

## 八、positions

`positions`是一个对象数组，用于标识`blockid`的内容来之文件中的哪一个文字，用于内容溯源的时候能够在原始文件中找到和显示对应的内容。当`blockid`的内容是由版面的多个栏目合并而成时，会出现多个`position` 对象，每个`position` 对象对应1个版面方框或栏目。为了适应不同的文档格式的内容定位方式，系统提供了以下几种`position` 对象对象类型。

`position` 对象的`type`字段决定了其类型：

* paraid

用于docx格式文件；按`段落id`（paraid）定位内容。`rang`字段指定起止`段落id`；`charspan`为可选字段，指定内容从段落的m个字符开始到底n个字符结束。不提供`charspan`表示`blockid`为起止段落的全部内容。示例：

```
"positions": [
{
    "type": "paraid",
    "range": ["5EA4577A", "6555DDCB"]
    "charspan": [10,999]
}]
```

* bbox

用于PDF格式文件，按矩形框定位内容。`anchor`字段指定页码（可以是数字也可以字符串）；`range`是矩形坐标数组`[h1,w1,h2,w2]`，如如`[174, 155, 818, 333]`。

`position` 对象有多种类型，docx文档的`position` 对象的`type`为 `"paraid",`使用`rang`数组表示文本所属的起止段`paraid`位置。PDF文档的`position` 对象类型为`"bbox"`，标识的文本所在的版面矩形位置。`charspan`为可选字段，指定内容从段落的m个字符开始到底n个字符结束。不提供`charspan`表示`blockid`为起止段落的全部内容。示例：

```
"positions": [
{
    "type": "bbox",
    "anchor": "ii"
    "range": [174, 155, 818, 333]
    "charspan": [10, 999]
}]
```

* heading

用于Markdown格式文件，按标题定位内容。`anchor`是起始标题（标题重复是的处理方式查到markdown anchor规范）；`charspan`为可选字段，指定内容从段落的m个字符开始到底n个字符结束。不提供`charspan`表示`blockid`为起止段落的全部内容。

```
"positions": [
{
    "type": "heading",
    "anchor": "ii"
    "range": [174, 155, 818, 333]
    "charspan": [10, 999]
}]
```

* absolute

用于text格式文件，按字符绝对位置定位。`charspan`指定内容从段落的m个字符开始到底n个字符结束。

```
"positions": [
{
    "charspan": [10, 999]
}]
```

## 八、`llm_analyze_result` 状态机

| `status` | 触发场景 | 字段说明 |
|---|---|---|
| `success` | 模型成功返回合法 JSON 且必需字段齐全 | 图形：`name / type / description`；表格：`name / description`；公式：`name / description / equation` |
| `skipped` | 期跳过多模态分析：图片格式不支持、像素 < `VLM_MIN_IMAGE_PIXEL`（默认 32px）、大于 `VLM_MAX_IMAGE_BYTES`（默认 5 MB）、未启用VLM | `message` 写跳过原因 |
| `failure` | 必需字段缺失、JSON 修复后仍不合法、VLM/EXTRACT role 未配置而对应模态被启用、模型调用异常 | `message` 写诊断 |

补充：

- `analyze_time` 是 epoch 秒，每个 status 都有；
- `message` 在 `status="success"` 时**恒为空串**，便于过滤；
- 已经 `status="success"` 或 `status="skipped"` 的 item，下一次 `analyze_multimodal` 默认**跳过**重新分析（幂等）。`status="failure"` 在下一次仍会被视为失败抛错，避免静默吞掉错误。

图形 `type` 受 12 项枚举约束（见 [`IMAGE_TYPE_ENUM`](../lightrag/prompt_multimodal.py)：`Photo / Illustration / Screenshot / Icon / Chart / Table / Infographic / Flowchart / Chat Log / Wireframe / Texture / Other`）；模型若返回枚举外的值，会被规整成 `Other` 而不是失败。
