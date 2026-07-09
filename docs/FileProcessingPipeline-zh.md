# 文件处理流水线工作方式说明

从版本 v1.5.0 （目前在dev分支）开始，LightRAG的文件处理流水线进行了重大的升级：

* 支持多种文件内容抽引擎：legacy、native、mineru、docling
* 支持多种文本块分块方法：Fix、Recursive、Vector、Paragraph
* 支持对个别文件关闭实体关系抽取

LightRAG Server引入了一个文件处理的中间格式： `LightRAG Document` 。该格式支持表格和图片等多模态数据，同时包含文章的章节段落元数据，方便日后进行内容溯源。

本文以 **LightRAG Server** 的部署与使用视角组织：先给出快速开始可直接套用的配置，再展开内容抽取与分块的配置语法、存储 / 目录布局、去重、并发以及续跑规则。直接通过 Python 代码调用 `LightRAG` 类的开发者请翻到[第八章 Python SDK 调用](#八、Python SDK 调用)。

## 一、快速开始

### 保持旧版文件处理行为

所有文件按旧版的文档解析和分块策略处理所有文档。不配置 `LIGHTRAG_PARSER` 或把它配置为如下值：

```bash
LIGHTRAG_PARSER=*:legacy-F
```

### 推荐起步文件处理行为

不依赖外部文档解析服务，不依赖`VLM`视觉模型。使用新版原生的 `Native` 解析 `docx` 文档，开启表格(t)和公式(e)的模态分析，搭配`P`分块策略；其余文档使用老版本的内容解析器，搭配效果更好的`R`分块策略。

```bash
LIGHTRAG_PARSER=*:native-teP,*:legacy-R
```

### 开启多模态处理能力

开启多模态处理能力需要依赖 `MinerU` 文件解析服务和 `VLM` 视觉识别模型。使用 `Native` 解释 `docx` 文件，使用 `MinerU` 解析 `pdf`、`office` 和各种图片文件。以上文件都开启图片(i)、表格(t)和公式(e)的模态分析，并并搭配`P`分块策略。其余文档回退到老版本的内容解析器并搭配`R`分块策略。

```bash
LIGHTRAG_PARSER=*:native-iteP,*:mineru-iteP,*:legacy-R
VLM_PROCESS_ENABLE=true
VLM_LLM_MODEL=kimi-k2.6
MINERU_API_MODE=local
MINERU_LOCAL_ENDPOINT=http://localhost:8000
```

> `P`分块策略是LightRAG原生的分块策略，详情请参阅[Paragraph Semantic 分块策略](ParagraphSemanticChunking-zh.md)。VLM的配资请参阅[基于角色的 LLM/VLM 配置指南](RoleSpecificLLMConfiguration-zh.md)

## 二、文件处理方式配置

LightRAG 的文件处理配置由两部分合成：内容抽取引擎决定原始文件如何被解析，处理选项决定解析后是否执行多模态分析、使用哪种分块方式，以及是否构建知识图谱。通常先用环境变量 `LIGHTRAG_PARSER` 按文件后缀设置默认规则，再用文件名中的 `[hint]` 覆盖单个文件。引擎和选项可以写在同一个配置片段里，例如 `docx:native-iet` 或 `report.[native-R!].docx`。

为了向后兼容，在未修改配置的情况下，升级后的文件内容提取方式会维持原来的 `legacy` 行为。如需启用新的内容处理引擎，请按本节说明配置。

### 2.1 配置语法总览

完整配置模型如下：

```text
LIGHTRAG_PARSER=后缀:引擎-选项,后缀:引擎,*:legacy-R
filename.[ENGINE].ext
filename.[ENGINE-OPTIONS].ext
filename.[-OPTIONS].ext
```

- `LIGHTRAG_PARSER` 是默认规则表，按文件后缀匹配，例如 `pdf:mineru`、`docx:native-iet`。
- 文件名 `[hint]` 是单文件覆盖规则，例如 `paper.[mineru].pdf`、`memo.[native-R!].docx`。
- `ENGINE` 是内容抽取引擎：`legacy`、`native`、`mineru` 或 `docling`。
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

### 2.2 默认规则：`LIGHTRAG_PARSER`

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

### 2.3 单文件覆盖：文件名 hint

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

#### 为分块策略附加参数

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
| `drop_references` | `drop_rf` | P | bool | 分块前丢弃文末参考文献节，如 `paper.[-P(drop_rf=true)].pdf`；布尔参数可省略取值，`paper.[-P(drop_rf)].pdf` 等价于 `drop_rf=true` |

- `process_options` 仍是纯选择符字符串；每个参数会写入该策略的 `chunk_options`（见 §3），策略其它来自环境变量的参数保持不变。别名在内部统一归一化为全称。
- 合并优先级：选择符仍遵循“文件名 hint 的非空选项整体覆盖规则选项”；参数按**同一策略**叠加——先规则参数，再文件名 hint 参数（同一键以文件名为准）。
- 启动期（`LIGHTRAG_PARSER`）与上传期（文件名 hint）均严格校验：未知参数、类型错误、取值越界、把参数加到不支持的策略（如 `V` 上的 `chunk_ol`）都会给出友好报错。

> `drop_references` 检测调参 `CHUNK_P_REFERENCES_TAIL_N`（默认 2）/ `CHUNK_P_REFERENCES_HEADINGS`（竖线分隔，默认 `References\|Bibliography\|参考文献`）仅经环境变量、运行时实时读取。drop_references可以通过环境变量 `CHUNK_P_DROP_REFERENCES` 设置为全局默认值.

#### 为解析引擎附加参数

参数也可以附加到**引擎 token** 上，按文件覆盖外部引擎的行为。它们被编码进持久化的 `parse_engine` 字段，同时作用于引擎请求与其原始包缓存签名（因此改动参数会触发重解析，而非复用旧缓存包）。

```text
paper.[mineru(page_range=1-3,language=en,local_parse_method=ocr)].pdf   # 文件名 hint
paddle.[paddleocr_vl(page_range=1-3,page_range=5,useOcrForImageBlock=true)].pdf
scan.[docling(force_ocr=true)].pdf
LIGHTRAG_PARSER=pdf:mineru(language=en);*:legacy-R                       # 规则
```

当前支持的引擎参数（全称 / 别名）：

| 引擎 | 参数 | 别名 | 类型 | 说明 |
| --- | --- | --- | --- | --- |
| `mineru` | `page_range` | `pr` | 列表 | 一个或多个页码范围；**见下方列表说明** |
| `mineru` | `language` | — | str | OCR / 模型语言（如 `en`、`ch`） |
| `mineru` | `local_parse_method` | `local_pm` | 枚举 | `auto` / `txt` / `ocr`（local 模式） |
| `docling` | `force_ocr` | `ocr` | bool | `true` / `false` |
| `paddleocr_vl` | `page_range` | `pr` | 列表 | 一个或多个页码范围；发送为 PaddleOCR-VL 请求字段 `pageRanges` |
| `paddleocr_vl` | `use_ocr_for_image_block` | `useOcrForImageBlock` | bool | 按文件覆盖 `PADDLEOCR_VL_USE_OCR_FOR_IMAGE_BLOCK` |
| `paddleocr_vl` | `use_seal_recognition` | `useSealRecognition` | bool | 按文件覆盖 `PADDLEOCR_VL_USE_SEAL_RECOGNITION` |
| `paddleocr_vl` | `use_doc_unwarping` | `useDocUnwarping` | bool | 按文件覆盖 `PADDLEOCR_VL_USE_DOC_UNWARPING` |

- **`page_range` 可写多个页码段——每段都单独写一个 `page_range=...`。** 括号 `(...)` 内逗号只分隔参数，因此多段页码要写成 `page_range=1-3,page_range=5,page_range=7-9`，不要写成环境变量里的单串形式 `MINERU_PAGE_RANGES="1-3,5,7-9"`。**多段** `page_range` 需要 `MINERU_API_MODE=official`；`local` 模式只接受单页/单段（如 `page_range=1-3`）。
- PaddleOCR-VL 有自己的 `pageRanges` 字段；它的 `page_range` hint 使用同样的重复键语法，但不继承 MinerU local 模式的单段限制。
- **`local_parse_method` 仅限 local 模式。** 它只影响本地 MinerU 请求，因此在 `MINERU_API_MODE=official` 下会被**拒绝**（official API 既不发送它、也不计入缓存键——接受它将静默无效）。
- 只有 `mineru`、`docling` 与 `paddleocr_vl` 接受引擎参数；把参数加到 `legacy`/`native` 会友好报错。校验在启动期（`LIGHTRAG_PARSER`）与上传期均执行。
- 合并优先级：引擎参数按**最终引擎**解析——当文件名 hint 选中了另一个可用引擎时，规则的引擎参数会被丢弃。
- `parse_engine` 以 hint 语法存储（如 `mineru(page_range=1-3)`），并展示在 `doc_status` metadata 中，便于查看文档当时使用的解析参数。

### 2.4 文件解析引擎

| 引擎 | 说明 | 支持的文件格式（后缀） |
| --- | --- | --- |
| `legacy` | 旧版提取方式，在加入流水线前集中提取内容 | `txt` `md` `mdx` `pdf` `docx` `pptx` `xlsx` `rtf` `odt` `tex` `epub` `html` `htm` `csv` `json` `xml` `yaml` `yml` `log` `conf` `ini` `properties` `sql` `bat` `sh` `c` `h` `cpp` `hpp` `py` `java` `js` `ts` `swift` `go` `rb` `php` `css` `scss` `less` |
| `native` | 内置智能结构化内容抽取器 | `docx` `md` `textpack` |
| `mineru` | 外部 MinerU 内容提取引擎 | `pdf` `doc` `docx` `ppt` `pptx` `xls` `xlsx` `png` `jpg` `jpeg` `jp2` `webp` `gif` `bmp` |
| `docling` | 外部 Docling 内容提取引擎 | `pdf` `docx` `pptx` `xlsx` `md` `html` `xhtml` `png` `jpg` `jpeg` `tiff` `webp` `bmp` |
| `paddleocr_vl` | 外部 PaddleOCR-VL 内容提取引擎 | `pdf` `jpeg` `jpg` `png` `tiff` `tif` `bmp` `webp` |

`mineru`、`docling` 和 `paddleocr_vl` 是外部内容提取引擎，启用相关规则前必须先把服务跑起来，再在 LightRAG 配置对应 endpoint/token。

LightRAG 在本地会缓存 `mineru`、`docling` 和 `paddleocr_vl` 引擎的解析结果。重复上传相同的文件通常不会重新调用引擎解析文档。如果需要删除解析缓存，必须在文档管理界面删除文件弹窗中点击“同时删除文件”选项。修改这些引擎的端点地址和有效提取参数也会导致缓存失效，下次上传相同文件的时候会重新调用引擎解析文件内容。

#### 使用 Native 文件解析引擎

`native` 是 LightRAG 内置的结构化内容抽取引擎，**纯本地运行**：不依赖 MinerU / Docling 等外部服务，抽取阶段也不调用 VLM，开箱即用无需任何部署。运行依赖仅 `python-docx` + `defusedxml`（必备）；其中 markdown 路径的 SVG 栅格化额外依赖**可选**的 `cairosvg`（缺失时跳过该 SVG 并记 warning，不影响其余内容）。

支持后缀：`docx` / `md` / `textpack`。启用方式：

- `docx`、`md` 默认仍走 `legacy`，需显式选择 native，例如默认规则 `LIGHTRAG_PARSER=docx:native`、`LIGHTRAG_PARSER=md:native`，或文件名 hint `report.[native-iet].docx`、`notes.[native].md`（语法见 [§2.2](#22-默认规则lightrag_parser) / [§2.3](#23-单文件覆盖文件名-hint)）。
- `textpack` 为 native 独占后缀，无需 hint/规则即自动路由到 native。

##### docx 抽取能力

native 直接解析 OOXML，能识别以下结构并写入对应 sidecar（sidecar 是否生成由文档实际内容决定，见 [§4.2](#42-__parsed__-目录结构)）：

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

##### docx 段落溯源（paraId）提示

native docx 会采集 Word 2013+ 写入的 `w14:paraId` 作为段落级溯源锚点。若文档由 LibreOffice / WPS / 旧版 Word 生成，或被手工改过 docx 内部 XML，部分段落会缺少 paraId，此时会在日志输出一次提示：

```text
[parse_native] <文件名>: N paragraphs lack paraId; Re-saving file in Word 2013+ to regenerate ids.
```

受影响块的 `positions` 退化为 `[{"type": "paraid", "range": null}]`。这只是提示，**不影响解析成功**；如需精确段落溯源，按提示在 Word 2013+ 中「另存为 .docx」即可重建 id。

##### md / textpack 抽取能力

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
- SVG 图片（base64 / textpack 包内文件 / 在线下载）会先经 cairosvg 栅格化为 PNG 再写入 sidecar；cairosvg 不可用或渲染失败时跳过该图（记 warning）。
- 外部 URL 图片（`![](http://...)`）**默认下载并内嵌**（`NATIVE_MD_IMAGE_DOWNLOAD_ENABLED` 默认 `true`）；无论下载成功与否都会生成 drawing（成功内嵌资源，失败回退为外链）。下载默认仅允许可全球路由的公网 IP（DNS 解析结果与每一跳重定向目标都校验，且 socket 直连已校验 IP 以防 DNS rebinding，忽略环境 `HTTP(S)_PROXY`），私网 / 环回 / 链路本地 / 保留 / CGNAT（`100.64.0.0/10`）等一律拒绝；如需放行特定内网段，用 `NATIVE_MD_IMAGE_ALLOWED_NON_PUBLIC_CIDRS` 配置 CIDR 白名单。若设为 `false`，外链图片整个丢弃（不生成对应 drawing，故仅含外链图片的文档不会生成 `drawings.json`）。

##### 环境变量

native 的所有 `NATIVE_*` 环境变量与 `.native_raw/` 缓存目录**仅作用于 markdown / textpack 引擎的外链图片下载**；**docx 路径不读取任何 `NATIVE_*` 变量**。最常用的两个：

- `LIGHTRAG_FORCE_REPARSE_NATIVE`（默认 `false`）：强制丢弃 `.native_raw/` 缓存、重新联网下载外链图片。
- `NATIVE_MD_IMAGE_DOWNLOAD_ENABLED`（默认 `true`）：外链图片下载总开关，设为 `false` 时丢弃所有外链图片。

其余下载/大小/SSRF 相关变量（`NATIVE_MD_IMAGE_DOWNLOAD_TIMEOUT` / `NATIVE_MD_IMAGE_DOWNLOAD_REQUIRED` / `NATIVE_MD_IMAGE_MAX_BYTES` / `NATIVE_MD_IMAGE_MAX_SVG_PIXELS` / `NATIVE_MD_IMAGE_ALLOWED_NON_PUBLIC_CIDRS`）含义与默认值见仓库根目录 [env.example](https://github.com/HKUDS/LightRAG/blob/main/env.example)。

下载的外链图片缓存到 `<文件>.native_raw/`（与 `.parsed/` 同级，类比 `.mineru_raw`/`.docling_raw`），重新解析同一未改动文件时直接复用、不再联网；源文件内容或上述大小 / SVG 像素 / CIDR 配置变化时缓存自动失效。删除文档（删除弹窗勾选「同时删除文件」）时该缓存目录会与 `.parsed/` 一并清理。

#### 使用 MinerU 文件解析引擎

LightRAG文档处理管线支持使用MinerU作为文件解析器。支持使用两种MinerU访问模式：

- `official`模式：使用MinerU云端的 API v4 服务。需要先到 [MinerU官网](https://mineru.net/) 注册账号并创建API-KEY。然后在LightRAG的 `.env` 文件中添加以下配置：

```bash
MINERU_API_MODE=official
MINERU_API_TOKEN=<your_token>
# MINERU_OFFICIAL_ENDPOINT=https://mineru.net   # 默认值，通常无需修改
```

* `local`模式：使用本地部署的MInerU服务。部署方式见后面的说明。本地MinerU服务启动后在LightRAG的 `.env` 文件中添加以下配置：

```bash
MINERU_API_MODE=local
MINERU_LOCAL_ENDPOINT=http://<your_mineru_local_server_ip>:8000
```

其余MinerU的详细配置请参考仓库根目录环境变量示例文件 [env.example](https://github.com/HKUDS/LightRAG/blob/main/env.example) 中的 MinerU 小节。针对 `official` 和 `local` 两种模式，分别有不同的环境变量配置。需要仔细阅读示例文件中的说明。

#### **本地部署 MinerU 服务**

从 Github官方仓库   [opendatalab/MinerU](https://github.com/opendatalab/MinerU) 把 Dockerfile 和 compose.yaml 拷贝到本地。这两个文件应该在仓库的 docker 目录可以找到。针对中国供应商的特殊显卡需要选择相应的 Dockerfile 。

准备好上诉两个文件后通过以下命令构建 docker 镜像:

```bash
docker build --tag mineru:latest .
```

镜像构建好之后通过以下命令启动 API 服务（参数 `--profile api` 标识仅启动MinerU的 API 服务，服务默认监听 8000 端口）：

```bash
docker compose -f compose.yaml --profile api up -d
```

镜像构建细节、GPU 驱动准备、模型权重位置等请参考官方 README：<https://github.com/opendatalab/MinerU>。

**进阶配置：开启 vLLM 预加载与标题层级修正（可选）**

在基础部署之上，建议为本地 MinerU 额外开启两项 MinerU **服务端**功能。这两项都改的是 MinerU 容器侧配置（容器内 `mineru.json` 与官方 `compose.yaml`），不涉及 LightRAG 的 env 变量；其中标题层级修正还需要一个可用的 LLM API。

- **vLLM 启动预加载**：让容器启动时就把 VLM 模型加载进显存，避免首个解析请求承担模型加载延迟。
- **标题层级修正（`title_aided`）**：MinerU 借助一个外部 LLM 修正解析输出的标题层级，提升结构化产物质量。这对依赖标题结构的 [P（段落语义）分块策略](#25-文件处理选项)尤其有帮助；`P分块策略` 优先按标题分割，标题层级越准确，分块语义越好。

**步骤1：导出并修改 `mineru-lightrag.json`**

从官方镜像中把 `/root/mineru.json` 拷到宿主机当前目录的 `mineru-lightrag.json`（用固定容器名 `temp_mineru`，无需运行容器）：

```bash
docker create --name temp_mineru mineru:latest
docker cp temp_mineru:/root/mineru.json ./mineru-lightrag.json
docker rm temp_mineru
```

然后修改 `mineru-lightrag.json` 中的 `llm-aided-config.title_aided`：填入 `api_key`，并把 `enable` 改为 `true`：

```json
"llm-aided-config": {
    "title_aided": {
        "api_key": "your_api_key",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen3.5-plus",
        "enable_thinking": false,
        "enable": true
    }
}
```

> `api_key` / `base_url` / `model` 需替换为用户自己可用的 LLM 服务（示例使用阿里云 DashScope 的 OpenAI 兼容接口）。

**步骤2：修改官方 `compose.yaml` 的 `api` profile 服务（`mineru-api`）**

在 `mineru-api` 服务上做三处改动：`environment` 增加 `MINERU_TOOLS_CONFIG_JSON`（让 MinerU 读改过的配置而非镜像内置 `mineru.json`），`volumes` 把宿主机 `mineru-lightrag.json` 挂进容器，`command` 追加 `--enable-vlm-preload true` 开启 vLLM 预加载。改好后的完整 `mineru-api` profile 如下（以 `# <-- 新增` 标注三处增量）：

```yaml
  mineru-api:
    image: mineru:latest
    container_name: mineru-api
    restart: always
    profiles: ["api"]
    ports:
      - 8000:8000
    environment:
      MINERU_MODEL_SOURCE: local
      MINERU_TOOLS_CONFIG_JSON: /root/mineru-lightrag.json   # <-- Added
    volumes:
      - ./mineru-lightrag.json:/root/mineru-lightrag.json    # <-- Added
    entrypoint: mineru-api
    command:
      --host 0.0.0.0
      --port 8000
      --allow-public-http-client
      --gpu-memory-utilization 0.45                          # Reserved 10GB is fine, preventing OOM errors
      --enable-vlm-preload true                              # <-- Added
    ulimits:
      memlock: -1
      stack: 67108864
    ipc: host
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["0"]
              capabilities: [gpu]
```

> 示范中请按实际显卡情况调整 `gpu-memory-utilization` ；`environment` / `volumes` / `command` 三处为本次新增项，其余保持官方原样。

**步骤3：重启生效**

改完后重新启动 API 服务让改动生效：

```bash
docker compose -f compose.yaml --profile api up -d
```

#### 使用 PaddleOCR-VL 文件解析引擎

LightRAG 也可以把 PDF 和常见图片（`jpeg`、`jpg`、`png`、`tiff`、`tif`、`bmp`、`webp`）交给 `paddleocr_vl` 引擎解析。PaddleOCR-VL 与 MinerU / Docling 一样属于外部解析服务：启用路由规则前，需要先配置云端 token 或本地服务 endpoint。

- `official` 模式：使用 PaddleOCR AIStudio 云服务异步 API。先获取 access token，再在 LightRAG 的 `.env` 中配置：

```bash
LIGHTRAG_PARSER=pdf:paddleocr_vl-iteP;*:legacy-R
PADDLEOCR_VL_API_MODE=official
PADDLEOCR_VL_API_TOKEN=<your_access_token>
# PADDLEOCR_VL_OFFICIAL_ENDPOINT=https://paddleocr.aistudio-app.com/api/v2/ocr/jobs
```

- `local` 模式：使用本地部署的 PaddleOCR-VL 服务。服务启动后，在 LightRAG 的 `.env` 中配置：

```bash
LIGHTRAG_PARSER=pdf:paddleocr_vl-iteP;*:legacy-R
PADDLEOCR_VL_API_MODE=local
PADDLEOCR_VL_LOCAL_ENDPOINT=http://<your_paddleocr_vl_server_ip>:8080
```

如果 LightRAG API Server 跑在 Docker 容器中，而 PaddleOCR-VL 服务跑在宿主机上，`PADDLEOCR_VL_LOCAL_ENDPOINT` 不要写 `localhost`，应写容器可访问的地址，例如 Linux 下的宿主机网关 IP，或 Docker Desktop 环境中的 `http://host.docker.internal:8080`。

`paddleocr_vl` 支持的请求参数、缓存目录和 cache 失效规则见后文 [4.5 PaddleOCR-VL 原始产物目录](#45-paddleocr-vl-原始产物目录-basepaddleocr_vl_raw)。

#### **本地部署 PaddleOCR-VL 服务**

PaddleOCR-VL 本地部署有两种常见形态。两者使用同一套核心 Pipeline：文档解码、可选方向/去扭曲预处理、PP-DocLayoutV3 版面分析、区域裁剪合并、PaddleOCR-VL-1.6-0.9B VLM 识别、Markdown/JSON 后处理、可选跨页表格合并和标题层级重排。区别主要在服务化层和并发调度。本节只说明 LightRAG 侧如何选择 endpoint、如何验证服务、以及如何接入 `paddleocr_vl` 引擎；具体镜像构建、`.env` 参数含义、模型路径、批处理大小、设备参数、各类 accelerator 的部署差异，请以 PaddleOCR 官方教程和对应目录 README 为准。官方教程见 [PaddleOCR-VL 使用教程](https://www.paddleocr.ai/latest/version3.x/pipeline_usage/PaddleOCR-VL.html)。

| 部署方式 | 官方目录 | 容器结构 | 说明 |
| --- | --- | --- | --- |
| 简化版（加速算子部署） | `deploy/paddleocr_vl_docker/accelerators/<accelerator>` | `paddleocr-vl-api` + `paddleocr-vlm-server` | 按硬件类型分别提供部署目录，PaddleX 内置 HTTP 服务直接调用 Pipeline |
| HPS 高性能服务化部署 | `deploy/paddleocr_vl_docker/hps` | `paddleocr-vl-api` + `paddleocr-vl-pipeline` + `paddleocr-vlm-server` | FastAPI 网关 + Triton 动态批处理 + vLLM 连续批处理；根据当前官方文档，该方案目前仅支持 NVIDIA GPU |

**前置条件**

- 已安装 Docker / Docker Compose；
- 推理设备、驱动、运行时和容器工具链与所选官方部署目录匹配；
- 使用 HPS 方案时，根据当前官方文档，需要 x64 CPU、NVIDIA GPU（Compute Capability >= 8.0 且 < 10.0）、支持 CUDA 12.6 的 NVIDIA 驱动、Docker >= 19.03 和 Docker Compose >= 2.0；
- 显存足以加载 `PaddleOCR-VL-1.6-0.9B` 和版面分析模型；
- 如果服务器在内网或离线环境，需要提前准备模型权重和镜像源。

**方式一：简化版部署**

从 PaddleOCR 官方仓库复制 `deploy/paddleocr_vl_docker/accelerators` 下与你的加速卡匹配的目录。官方目前在该目录下按硬件划分了多个子目录，例如 `amd-gpu`、`huawei-npu`、`hygon-dcu`、`iluvatar-gpu`、`intel-gpu`、`kunlunxin-xpu`、`metax-gpu`、`nvidia-gpu`、`nvidia-gpu-sm120` 等。每个子目录的镜像、环境变量、启动参数和硬件要求可能不同，应以对应子目录中的官方说明为准。

```bash
git clone https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR/deploy/paddleocr_vl_docker/accelerators/<your_accelerator>

# 按实际环境修改 .env，例如模型路径、GPU、镜像源等
docker compose -f compose.yaml up -d --build
```

启动后应有两个服务：

- `paddleocr-vl-api`：对外暴露文档解析 API；
- `paddleocr-vlm-server`：提供 VLM 推理服务，通常以 OpenAI 兼容的 `/v1` 接口被 Pipeline 调用。

**方式二：HPS 高性能部署**

HPS 目录通常位于 PaddleOCR 官方仓库的 `deploy/paddleocr_vl_docker/hps`。根据当前官方 README，该方案目前暂时只支持 NVIDIA GPU，对其他推理设备的支持仍在完善中。该方式会启动三层服务：FastAPI 网关、Triton Pipeline 和 vLLM Server。

```bash
git clone https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR/deploy/paddleocr_vl_docker/hps

# 按官方 README 修改 .env，例如 HPS_PIPELINE_NAME、PaddleX 版本、SDK 目录、并发、超时和 VLM 服务地址
cp .env.example .env
bash prepare.sh
docker compose -f compose.yaml up -d --build
```

HPS 的默认对外入口一般是网关端口 `8080`。网关会把 `/layout-parsing` 请求转发给 Triton Pipeline，Pipeline 再调用 vLLM Server 完成区域识别。`HPS_PIPELINE_NAME`、`HPS_PADDLEX_VERSION`、`HPS_SDK_DIR`、`HPS_MAX_CONCURRENT_INFERENCE_REQUESTS`、`HPS_MAX_CONCURRENT_NON_INFERENCE_REQUESTS`、`HPS_INFERENCE_TIMEOUT`、`HPS_UVICORN_WORKERS`、`HPS_DEVICE_ID` 等参数的含义和取值建议，以官方 HPS README 为准。吞吐较高时优先调 HPS 网关和 Triton/vLLM 侧参数，而不是盲目提高 LightRAG 的 `MAX_PARALLEL_PARSE_PADDLEOCR_VL`。

**检查服务状态**

```bash
curl http://localhost:8080/health
curl http://localhost:8080/health/ready
```

就绪接口返回 `errorCode=0` 后再开始解析。首次启动时 VLM 模型需要加载到显存，可能需要等待数分钟。

**直接调用 PaddleOCR-VL 本地 API 验证**

官方 HPS 网关常见接口是 `multipart/form-data`：

```bash
curl -X POST http://localhost:8080/layout-parsing \
  -F "file=@/path/to/document.pdf" \
  -F "useLayoutDetection=true" \
  -F "useChartRecognition=true" \
  -F "formatBlockContent=true" \
  -F "prettifyMarkdown=true" \
  -F "restructurePages=true" \
  -F "mergeTables=true" \
  -F "temperature=0.0" \
  -F "maxNewTokens=4096"
```

响应成功时应包含 `layoutParsingResults`，其中每页结果里有 Markdown 文本、版面 JSON 和可选图片资源。

**接入 LightRAG**

确认本地服务可用后，把 LightRAG 配成 `local` 模式：

```bash
LIGHTRAG_PARSER=pdf:paddleocr_vl-iteP;*:legacy-R
PADDLEOCR_VL_API_MODE=local
PADDLEOCR_VL_LOCAL_ENDPOINT=http://localhost:8080
MAX_PARALLEL_PARSE_PADDLEOCR_VL=1
```

当前 LightRAG 的 `paddleocr_vl` local client 会向 `POST {PADDLEOCR_VL_LOCAL_ENDPOINT}/layout-parsing` 发送同步 JSON 请求，并把文件内容放在 base64 编码的 `file` 字段中；其它 PaddleOCR-VL 选项作为顶层 JSON 字段发送。如果你使用的官方网关只接受 `multipart/form-data`，需要在 PaddleOCR-VL 网关前加一个轻量兼容适配层，或把本地网关调整为同时接受 JSON/base64 请求。适配层的职责只有两点：把 JSON 中的 base64 `file` 转成上传文件，并把服务返回结果规范化为 `{"errorCode": 0, "result": {"layoutParsingResults": [...]}}`。

用 parser CLI 做端到端验证：

```bash
PADDLEOCR_VL_API_MODE=local \
PADDLEOCR_VL_LOCAL_ENDPOINT=http://localhost:8080 \
python -m lightrag.parser.cli ./inputs/sample.pdf \
  --engine paddleocr_vl \
  --force-reparse
```

成功后会生成 `sample.pdf.paddleocr_vl_raw/` 和 `sample.pdf.parsed/`。其中 `content_list.json` 是 PaddleOCR-VL 原始结果，`*.blocks.jsonl` / `tables.json` / `drawings.json` / `equations.json` 是 LightRAG 后续流水线使用的 sidecar。

**常见问题**

| 现象 | 排查方向 |
| --- | --- |
| `/health/ready` 长时间不 ready | 等待 VLM 模型加载；检查 GPU 显存、模型路径、容器日志 |
| LightRAG 容器连不上 `localhost:8080` | Docker 内的 `localhost` 指向 LightRAG 容器自身，改用宿主机网关 IP 或 `host.docker.internal` |
| 直接 `curl -F` 成功，但 LightRAG local 失败 | 当前 LightRAG local client 使用 JSON/base64 请求；给本地网关加兼容适配层 |
| 首次解析很慢 | VLM 冷启动、PDF 页数多或 `restructurePages=true`；先用小 PDF 验证 |
| 解析成功但没有重新调用服务 | 命中了 `*.paddleocr_vl_raw/` cache；设置 `LIGHTRAG_FORCE_REPARSE_PADDLEOCR_VL=true` 或 CLI 使用 `--force-reparse` |

#### 使用 Docling 文件解析引擎

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
| `DOCLING_DO_FORMULA_ENRICHMENT` | `false` | 是识别文档中的公式并按LaTex格式输出；启用前需要确保Docling后台下载了公式识别模型（见后面说明） |

未配置 `DOCLING_OCR_ENGINE` / `DOCLING_OCR_PRESET` 时等同于 `auto`；未配置 `DOCLING_OCR_LANG` 时不向 docling-serve 传递语言列表，由 OCR 引擎使用自身默认值。解析缓存按这些有效参数计算签名，因此“未配置”和“显式填写默认值”不会导致缓存失效。

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

#### Docling本地部署(启用 LaTeX 公式识别)

下面以 Docker 部署 docling-serve 为例，给出从镜像下载到模型挂载的完整步骤，部署完成后将 `DOCLING_DO_FORMULA_ENRICHMENT=true` 写入 LightRAG 的 `.env` 即可启用 LaTeX 公式识别。

> **重要提示**：以下步骤基于显卡支持 CUDA 13 的环境。如果显卡较老旧、不支持 CUDA 13，需要把命令与 compose 文件中的镜像名 `docling-serve-cu130:main` 替换为对应 CUDA 版本的标签。可选镜像列表参见 [docling-serve Packages](https://github.com/orgs/docling-project/packages?repo_name=docling-serve)。

**1. 下载镜像**

```bash
docker pull ghcr.io/docling-project/docling-serve-cu130:main
```

**2. 下载模型**

```bash
# 创建 docling 工作目录
mkdir docling
cd docling

# 创建模型挂载目录
mkdir models

# 把容器内的原有模型拷贝到 models 目录
docker run --rm -it \
  -v "$(pwd)/models:/opt/app-root/src/models" \
  ghcr.io/docling-project/docling-serve-cu130:main \
  cp -r /opt/app-root/src/.cache/docling/models /opt/app-root/src/

# 下载公式识别模型
docker run --rm \
  -v "$(pwd)/models:/opt/app-root/src/models" \
  -e DOCLING_SERVE_ARTIFACTS_PATH="/opt/app-root/src/models" \
  ghcr.io/docling-project/docling-serve-cu130:main \
  docling-tools models download-hf-repo docling-project/CodeFormulaV2 -o models
```

**3. 创建 `docker-compose.yaml` 文件**

在上一步的 `docling` 目录下创建 `docker-compose.yaml`，内容如下：

```yaml
services:
  docling-serve:
    image: ghcr.io/docling-project/docling-serve-cu130:main
    container_name: docling-serve
    ports:
      - "5001:5001"
    environment:
      DOCLING_SERVE_ENABLE_UI: "true"
      NVIDIA_VISIBLE_DEVICES: "all"
      DOCLING_SERVE_ARTIFACTS_PATH: "/opt/app-root/src/models"
    # deploy:  # This section is for compatibility with Swarm
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: all
    #           capabilities: [gpu]
    runtime: nvidia
    restart: always
    volumes:
      - ./models:/opt/app-root/src/models
```

随后在该目录执行 `docker compose up -d` 启动服务。容器就绪后，在 LightRAG 的 `.env` 中设置：

```bash
DOCLING_ENDPOINT=http://localhost:5001
DOCLING_DO_FORMULA_ENRICHMENT=true
```

即可让 LightRAG 通过本地 docling-serve 识别文档中的公式并以 LaTeX 形式输出。

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

> 模态可用性以"sidecar 文件是否存在"为唯一信号，内容提取引擎不需要在 meta 中声明能力。某文档若没有任何图像/表格/公式，对应 sidecar 不会写入；用户即使开启了 `i/t/e`，对应模态也只会被静默跳过，但 `analyze_multimodal` 会在该篇文档落一行 INFO 级日志（`[analyze_multimodal] sidecar e:equations empty: doc—id ...`），便于排查"VLM 为何没跑"。这种情况不会报错。

### 2.6 校验、优先级与回退

- 启动时会严格校验 `LIGHTRAG_PARSER`：未知内容提取引擎、错误后缀写法、显式使用不支持的后缀、外部引擎缺少 endpoint、处理选项中的非法字符都会导致启动失败。
- **通配符规则匹配某后缀时**，引擎需通过两道可用性检查（见 `parser_routing._engine_is_usable`）：(a) 该引擎能力表支持此后缀；(b) 若是外部引擎（`mineru` / `docling`），对应 endpoint/token 环境变量已配置。任一检查不过，本规则跳过，继续匹配下一条规则。例如 `*:mineru;html:docling` 中：MinerU 不支持 `html` 后缀（条件 a 不过），`html` 继续命中 `docling`；如果 `MINERU_API_MODE=local` 但未设置 `MINERU_LOCAL_ENDPOINT`，所有 PDF 也会跳过 `*:mineru` 落到下一条规则（条件 b 不过）。这一行为对 `LIGHTRAG_PARSER` 规则匹配和文件名 hint 引擎选择都生效。
- 文件名 hint 的优先级高于 `LIGHTRAG_PARSER`。如果 hint 指定的引擎不支持该后缀，系统会回退到默认规则继续选择可用引擎。
- 如果文件名 hint 提供了非空选项串，则以 hint 为准；否则使用 `LIGHTRAG_PARSER` 规则中匹配项的默认选项；都没有则使用全部默认。
- 如果所有规则都不可用，文件内容提取方式会回退到 `legacy`；如果 `legacy` 也不支持对应的文件后缀，会向系统添加一个错误条目，上传文件保留在 `INPUT` 目录。
- F/R/V/P至多出现一个；同一选项重复时只生效一次但不报错。
- 大小写敏感：分块选项 F/R/V/P必须大写；其它选项 i/t/e小写。
- 中括号内出现非法字符时，整个 hint 失效，引擎按默认规则解析，选项按 `LIGHTRAG_PARSER` 默认或全部默认；同时落日志 warning。
- `P` 对任何能产出 `.blocks.jsonl` sidecar 的引擎（`native` / `mineru` / `docling`）抽取出的结构化结果有效；对 `legacy` 路径或无 sidecar 的输出会自动降级到 `R` 并记录 warning。

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
| `CHUNK_F_SIZE` | 未设 | int | F strategy 特定 `chunk_token_size`；高于顶层 legacy 兜底（`CHUNK_SIZE` 与 SDK 路径的 `LightRAG(chunk_token_size=…)`）。未设时 F 沿用顶层解析结果 |
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
| `CHUNK_P_SIZE` | `2000`（`DEFAULT_CHUNK_P_SIZE`） | int | P strategy 特定 `chunk_token_size`。与 R/V 不同，未设时 P **不**沿用顶层 `CHUNK_SIZE` / `LightRAG(chunk_token_size=…)`——段落语义合并需要比全局默认更大的上限才能将相关段落保留在一起，因此槽位始终携带 `DEFAULT_CHUNK_P_SIZE`（2000） |
| `CHUNK_P_OVERLAP_SIZE` | 未设 | int | P strategy 特定 overlap；高于 legacy 构造字段与 `CHUNK_OVERLAP_SIZE`。用于同一 JSONL content 行内长正文 fallback 到 R 时的文本重叠，以及相邻大表格之间桥接文字复制到前后表格块的单侧预算 |

P 的内部比例常量是算法刻度，会随 `chunk_token_size` 自动按比例推导。P 始终使用独立于全局链的 `chunk_token_size`——即使 `CHUNK_P_SIZE` 未设，P 也会回退到 `DEFAULT_CHUNK_P_SIZE`（2000）而**不**沿用全局 `CHUNK_SIZE`，因为段落语义合并需要比全局默认更大的上限才能将相关段落保留在一起。需要按部署调整时通过 `CHUNK_P_SIZE` 覆盖该默认。`CHUNK_P_OVERLAP_SIZE` 只影响 P 内部普通文本 fallback 与表格桥接上下文，不会让表格行级切片互相重叠。`CHUNK_F_SIZE` / `CHUNK_R_SIZE` / `CHUNK_V_SIZE` 行为不同——未设时**仍会**沿用顶层 `chunk_token_size`（F 即默认全局窗口，R 偏向较小目标利于句段切分，V 作为 advisory ceiling 通常希望放大以减少过度拆分）。

### 3.3 优先级链

每个分块槽位的最终值按 specificity-ordered 链解析（高 → 低）：

1. **`addon_params["chunker"]` 显式值** —— 通过 SDK 路径运行时设置或在构造时显式写入的字段值（见 §8.3）。Server-only 部署通常不会出现这一档。最直接，赢一切。
2. **strategy 特定 env** —— 如 `CHUNK_F_SIZE` / `CHUNK_R_SIZE` / `CHUNK_V_SIZE`（各策略 `chunk_token_size`）、`CHUNK_F_OVERLAP_SIZE` / `CHUNK_R_OVERLAP_SIZE` / `CHUNK_P_OVERLAP_SIZE`（overlap）、`CHUNK_P_SIZE`（P 专属）。未设对应 size env 时，F/R/V 沿用顶层 `chunk_token_size`。仅当槽位未被 ① 显式占用时填入。
3. **legacy 构造字段** —— `LightRAG(chunk_token_size=…, chunk_overlap_token_size=…)`，仅 SDK 路径生效，详见 §8.2。strategy 无关，"粗粒度缺省"，只填仍空的槽位。
4. **legacy env** —— `CHUNK_SIZE` / `CHUNK_OVERLAP_SIZE`。最终回退。

举例：`CHUNK_R_OVERLAP_SIZE=42` + `LightRAG(chunk_overlap_token_size=2)` → R 子字典 `chunk_overlap_token_size=42`（strategy env 胜出），F / P 子字典 `chunk_overlap_token_size=2`（无 F / P 特定 env，legacy 构造字段填入）。

**P 的 `chunk_token_size` 特例**：P 的 `chunk_token_size` 槽位**不**走完整的四档链。当 ① 未显式提供时，直接按 `CHUNK_P_SIZE` env > `DEFAULT_CHUNK_P_SIZE`（2000）解析，**跳过** ③ legacy 构造字段 `LightRAG(chunk_token_size=…)` 与 ④ legacy env `CHUNK_SIZE`。理由参见 §3.2 `CHUNK_P_SIZE` 行。

三层语义保证：

1. **复现性**：env 改了，重启后老文档仍按入队那一刻的快照分块，结果不变。
2. **续跑一致性**：续跑分支 B（内容已抽取，按当前 `process_options` 重做分块）读的也是 `full_docs.chunk_options`，避免 env 漂移破坏一致性。
3. **per-file 个性化**：调用方可以为每个文件传不同的 `chunk_options`（典型用法：管理 UI 单独配置某个文件的 separators 或 V 阈值）。这是 SDK 路径的入参语义，详见 §8.4。

### 3.4 字段结构

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

### 3.5 缺失兼容

老文档入队时还没有 `chunk_options` 字段；分块时 dispatcher 会按当前 `process_options` 调用 `resolve_chunk_options(self.addon_params, process_options=…)` 兜底拼装一份精简快照。建议在升级后通过 reprocess 一次让老文档拿到精简的 `chunk_options` 快照（且与当前 `process_options` 对齐）。

## 四、存储与目录布局

### 4.1 `full_docs` 字段

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
| `chunk_options` | 入队时**冻结**的分块器参数快照（精简字典：只保留 `process_options` 选中的那一路策略子字典，其它策略丢弃）。由 SDK 路径调用方传入或由 `resolve_chunk_options(self.addon_params, process_options=…)` 从实例字段（含 env 默认）兜底（见 §3.1）。`process_options` 选哪种分块策略（F/R/V/P），`chunk_options` 决定那一路分块器使用哪些参数。下游 `process_single_document` 在分块前从此字段读取专属 kwargs；持久化保证 env 变化、续跑、重启后老文档行为可复现。重新解析时与 `process_options` 一同改写。 |

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

### 4.3 MinerU 原始产物目录 `<base>.mineru_raw/`

`mineru` 引擎在解析过程中会把 MinerU 服务返回的完整产物（`content_list.json` + 可选的 `full.md` / `middle.json` / `layout.pdf` / `images/` 等）落到 `__parsed__/<规范文件名>.mineru_raw/` 目录下，并写入 `_manifest.json` 作为完整性校验文件。

设计目的：

- **避免重复上传**。再次解析同一文件时，先用源文件的内容 hash + 文件大小校验 `_manifest.json`，命中即跳过 MinerU 服务调用，直接从本地 `content_list.json` 走 adapter → SidecarWriter 流程。
- **保留诊断信息**。MinerU 解析出错或者下游 sidecar 字段异常时，可以直接到 `*.mineru_raw/` 比对原始 content_list 与图片资源。
- **支持对象溯源**。MinerU 生成的 `drawings.json` / `tables.json` / `equations.json` 会在 `self_ref` 中保存 `content_list.json#/N`，用于回查对应的 MinerU 原始对象及其 `page_idx` / `bbox` 等定位信息。
- **上传文件名去 hint**。源文件名包含 `[mineru-...]` / `[-iet]` 等处理 hint 时，调用 MinerU API 使用去 hint 后的规范文件名，避免 MinerU 返回的 raw bundle 内部文件名携带 hint。

生命周期：

| 操作 | 行为 |
|---|---|
| 首次解析 | 下载所有产物 → 原子写入 `_manifest.json`。 |
| 重复解析（cache 命中） | 不调用 MinerU 服务；不重写产物；走 adapter+Writer 重生成 sidecar（适用于 adapter 升级场景）。 |
| 重复解析（cache miss） | 清空目录内所有文件后重新下载并写入 manifest。 |
| `DELETE /documents` 且 `delete_file=True` | `*.parsed/` 与 `*.mineru_raw/` 与原始文件一并删除。 |
| `DELETE /documents` 且 `delete_file=False` | 保留所有产物，仅删 doc_status 与 KG 数据。 |
| `clear_documents` / `__parsed__` 整体清理 | 自然一并清除。 |
| scan 周期 | 不主动 GC 孤儿 `*.mineru_raw/`（用户显式删除时才清，避免误删调试现场）。 |

强制重新解析（绕过 cache）：设置 `LIGHTRAG_FORCE_REPARSE_MINERU=true`。

并发安全：LightRAG 强制要求同一 workspace 下 `canonical_basename` 唯一（上传/入队时返回 HTTP 409），加上流水线对单个文档的串行化处理，因此 `*.mineru_raw/` 不会出现并发写入冲突，无需额外锁。

`_manifest.json` 失效条件（任一触发即 cache miss）：

- 源文件大小或 sha256 与 manifest 记录不符；
- `MINERU_ENGINE_VERSION` 环境变量与 manifest 记录的 `engine_version` 都非空且不一致；
- 当前 `MINERU_API_MODE` 与 manifest 记录的 `api_mode` 都非空且不一致；
- 当前 mode 对应 endpoint（`MINERU_OFFICIAL_ENDPOINT` / `MINERU_LOCAL_ENDPOINT`）与 manifest 记录的 `endpoint_signature` 都非空且不一致；
- `content_list.json` 大小或 sha256 与 manifest 不符；
- 任一记录的非关键文件（图片、`middle.json` 等）大小与 manifest 不符。

> 关于 `engine_version` / `endpoint_signature` 的"任一侧为空即跳过"语义：当 manifest 写入时该字段为空（例如首次解析时未配置 `MINERU_ENGINE_VERSION`），或当前环境变量未设置时，该项不参与失效判断。如果首次解析时未设置版本环境变量，事后再补上并不会自动让历史缓存失效——这类场景需要手动设置 `LIGHTRAG_FORCE_REPARSE_MINERU=true` 触发重新解析。

### 4.4 Docling 原始产物目录 `<base>.docling_raw/`

`docling` 引擎在解析过程中会把 docling-serve 返回的 zip 产物（DoclingDocument JSON、Markdown 和引用图片）解压到 `__parsed__/<规范文件名>.docling_raw/` 目录下，并写入 `_manifest.json` 作为完整性校验文件。IR builder 在二次解析时会读取该目录的 `.json` 文件喂给 `DoclingIRBuilder`，不再走 docling-serve 服务。

目录布局：

```text
__parsed__/<base>.docling_raw/
├── _manifest.json
├── <base>.json        # DoclingDocument JSON（含 pages[].image base64）
├── <base>.md          # Markdown 形态，供人工检查
└── artifacts/
    └── image_*.png    # pictures[*].image.uri 指向的图片资源
```

设计目的：

- **避免重复上传/转换**。再次解析同一文件时，先用源文件 hash + 文件大小校验 `_manifest.json`，命中即跳过对 docling-serve 的上传 / 轮询 / 下载，直接从本地 `.json` 走 DoclingIRBuilder → SidecarWriter 流程。
- **保留诊断信息**。docling-serve 解析出错或下游 sidecar 字段异常时，可以直接到 `*.docling_raw/` 比对原始 DoclingDocument JSON、Markdown 与 `artifacts/` 图片。

生命周期：

| 操作 | 行为 |
|---|---|
| 首次解析 | `POST /v1/convert/file/async` 上传 → 长轮询 `/v1/status/poll/{task_id}?wait=N` → `GET /v1/result/{task_id}` 下载 zip → 安全解压（拒绝绝对路径与 `..`）→ 原子写入 `_manifest.json`。 |
| 重复解析（cache 命中） | 不调用 docling-serve；不重写产物；走 adapter+Writer 重生成 sidecar（适用于 adapter 升级场景）。 |
| 重复解析（cache miss） | 清空目录内所有文件后重新上传 / 下载 / 写入 manifest。 |
| `DELETE /documents` 且 `delete_file=True` | `*.parsed/` 与 `*.docling_raw/` 与原始文件一并删除。 |
| `DELETE /documents` 且 `delete_file=False` | 保留所有产物，仅删 doc_status 与 KG 数据。 |
| `clear_documents` / `__parsed__` 整体清理 | 自然一并清除。 |
| scan 周期 | 不主动 GC 孤儿 `*.docling_raw/`（用户显式删除时才清，避免误删调试现场）。 |

强制重新解析（绕过 cache）：设置 `LIGHTRAG_FORCE_REPARSE_DOCLING=true`。

并发安全：与 MinerU 路径一致 —— LightRAG 强制要求同一 workspace 下 `canonical_basename` 唯一（上传 / 入队时返回 HTTP 409），加上流水线对单个文档的串行化处理，因此 `*.docling_raw/` 不会出现并发写入冲突，无需额外锁。

`_manifest.json` 失效条件（任一触发即 cache miss）：

- 源文件大小或 sha256 与 manifest 记录不符；
- `DOCLING_ENDPOINT` 与 manifest 记录的 `endpoint_signature` 不一致；
- `DOCLING_ENGINE_VERSION` 设置且与 manifest 记录的 `engine_version` 不一致；
- `options_signature` 不一致 —— 任一 OCR / 公式 / pipeline 字段变化都会触发，覆盖范围包括：
  - 可调 env：`DOCLING_DO_OCR` / `DOCLING_FORCE_OCR` / `DOCLING_OCR_ENGINE` / `DOCLING_OCR_PRESET` / `DOCLING_OCR_LANG` / `DOCLING_DO_FORMULA_ENRICHMENT`；
  - 固化常量：`pipeline` / `target_type` / `to_formats` / `image_export_mode`（写入 signature 是为了防止未来值变更后老 bundle 被误复用）；
- 主 JSON 缺失、大小或 sha256 不一致；
- `artifacts/` 内任一图片缺失或大小不一致；
- `LIGHTRAG_FORCE_REPARSE_DOCLING=true`。

> `engine_version` / `endpoint_signature` 的"任一侧为空即跳过"语义与 MinerU §4.3 一致：manifest 写入时该字段为空（首次未配置 `DOCLING_ENGINE_VERSION`）或当前环境变量未设置时，该项不参与失效判断；事后补上版本号不会自动让历史缓存失效，需要 `LIGHTRAG_FORCE_REPARSE_DOCLING=true` 触发。

### 4.5 PaddleOCR-VL 原始产物目录 `<base>.paddleocr_vl_raw/`

`paddleocr_vl` 引擎调用 PaddleOCR-VL，把返回的版面解析结果保存为 `content_list.json`，并下载或解码 Markdown / outputImages 中引用的图片资源，统一写入 `__parsed__/<规范文件名>.paddleocr_vl_raw/`，同时用 `_manifest.json` 作为完整性校验文件。

最小配置：

```bash
LIGHTRAG_PARSER=pdf:paddleocr_vl-iteP;*:legacy-R
PADDLEOCR_VL_API_MODE=official
PADDLEOCR_VL_API_TOKEN=<your_access_token>
# PADDLEOCR_VL_OFFICIAL_ENDPOINT=https://paddleocr.aistudio-app.com/api/v2/ocr/jobs
```

`PADDLEOCR_VL_API_MODE` 支持 `official` 和 `local`。`official` 对接 PaddleOCR 云端异步 API：提交任务到 `PADDLEOCR_VL_OFFICIAL_ENDPOINT`，轮询完成后再下载结果 JSON/JSONL。`local` 对接自部署、且兼容 LightRAG 请求约定的 PaddleOCR-VL 服务，向 `POST {PADDLEOCR_VL_LOCAL_ENDPOINT}/layout-parsing` 发送同步 JSON 请求，服务会在文档解析完成后直接返回结果。`PADDLEOCR_VL_ENDPOINT` 仍作为 `PADDLEOCR_VL_OFFICIAL_ENDPOINT` 的兼容别名保留。

PaddleOCR-VL 默认把 `outputImages`、`inputImage`、`markdown.images`、
`exports` 等二进制字段以 Base64 内联返回。当服务端启用
`Serving.return_urls=true` 时，这些字段的结构不变，但值会变成预签名对象存储
URL。PaddleOCR 当前的 URL 返回仅支持 BOS（百度智能云对象存储），因此
LightRAG 只下载 host 为 `bj.bcebos.com` 或以 `.bj.bcebos.com` 结尾的 HTTPS
图片 URL，例如：

```bash
https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online/.../markdown_0/imgs/example.jpg?authorization=...
```

这些字段里的其他远程图片 URL 会被忽略；Base64 内联图片仍会正常解码。

local 模式最小配置：

```bash
LIGHTRAG_PARSER=pdf:paddleocr_vl-iteP;*:legacy-R
PADDLEOCR_VL_API_MODE=local
PADDLEOCR_VL_LOCAL_ENDPOINT=http://localhost:8080
```

official 异步提交任务还支持顶层 `pageRanges` 和 `batchId` 字段。`pageRanges` 可通过引擎 hint 按文件覆盖，`batchId` 和默认模型则保留为全局 client 配置：

```bash
PADDLEOCR_VL_MODEL=PaddleOCR-VL-1.6
PADDLEOCR_VL_PAGE_RANGES=
PADDLEOCR_VL_BATCH_ID=
```

PaddleOCR-VL 的可选请求参数从环境变量读取，并写入 official API 的
`optionalPayload`。其中 `useOcrForImageBlock`、`useSealRecognition` 和
`useDocUnwarping` 也可通过文件 hint 按文件覆盖，例如
`paddleocr_vl(page_range=1-3,useOcrForImageBlock=true)`。local 模式下，同一组有效参数会按照 PaddleOCR-VL
`POST /layout-parsing` 兼容接口的要求，作为 base64 `file` 旁边的顶层 JSON 字段发送。
client 会对 PDF 输入发送 `fileType=0`，对图片输入发送 `fileType=1`。下面这些参数未设置时会随当前 client 默认值发送：

```bash
PADDLEOCR_VL_USE_DOC_ORIENTATION_CLASSIFY=false
PADDLEOCR_VL_USE_DOC_UNWARPING=false
PADDLEOCR_VL_USE_LAYOUT_DETECTION=true
PADDLEOCR_VL_USE_CHART_RECOGNITION=true
PADDLEOCR_VL_USE_SEAL_RECOGNITION=true
PADDLEOCR_VL_USE_OCR_FOR_IMAGE_BLOCK=false
PADDLEOCR_VL_LAYOUT_NMS=true
PADDLEOCR_VL_LAYOUT_SHAPE_MODE=auto
PADDLEOCR_VL_PROMPT_LABEL=ocr
PADDLEOCR_VL_FORMAT_BLOCK_CONTENT=false
PADDLEOCR_VL_REPETITION_PENALTY=1
PADDLEOCR_VL_TEMPERATURE=0
PADDLEOCR_VL_TOP_P=1
PADDLEOCR_VL_MIN_PIXELS=147384
PADDLEOCR_VL_MAX_PIXELS=2822400
PADDLEOCR_VL_MERGE_LAYOUT_BLOCKS=true
PADDLEOCR_VL_MARKDOWN_IGNORE_LABELS=["header","header_image","footer","footer_image","number","footnote","aside_text"]
PADDLEOCR_VL_SHOW_FORMULA_NUMBER=false
PADDLEOCR_VL_RESTRUCTURE_PAGES=true
PADDLEOCR_VL_MERGE_TABLES=true
PADDLEOCR_VL_RELEVEL_TITLES=true
PADDLEOCR_VL_PRETTIFY_MARKDOWN=true
PADDLEOCR_VL_VISUALIZE=false
```

下面这些官方 API 参数未设置时不会传给服务，由服务端使用部署默认值：

```bash
PADDLEOCR_VL_LAYOUT_THRESHOLD=
PADDLEOCR_VL_LAYOUT_UNCLIP_RATIO=
PADDLEOCR_VL_LAYOUT_MERGE_BBOXES_MODE=
PADDLEOCR_VL_VLM_EXTRA_ARGS=
```

支持 number/object/array 形式的参数（例如
`PADDLEOCR_VL_LAYOUT_UNCLIP_RATIO`、`PADDLEOCR_VL_MARKDOWN_IGNORE_LABELS` 和
`PADDLEOCR_VL_VLM_EXTRA_ARGS`）可以用 JSON 写法配置。

目录布局：

```text
__parsed__/<base>.paddleocr_vl_raw/
├── _manifest.json
├── content_list.json
├── imgs/
│   └── *.jpg
└── outputImages/
    └── *.jpg
```

强制重新解析（绕过 cache）：设置 `LIGHTRAG_FORCE_REPARSE_PADDLEOCR_VL=true`。

缓存失效条件与其它外部引擎一致：源文件大小/hash、API mode、endpoint 签名、参数签名（全局模型、`pageRanges`、`batchId` 以及上面列出的所有 PaddleOCR-VL 请求参数）、可选 `PADDLEOCR_VL_ENGINE_VERSION`、`content_list.json` 大小/sha256，以及记录的图片资源大小。cache 命中时，LightRAG 不再调用 PaddleOCR-VL API，而是直接从本地 `content_list.json` 重建 sidecar。

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

## 六、流水线并发与重入约束

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

### 6.8 流水线并发参数

`pipeline_status` 相关的锁解决的是"谁能写"的正确性问题，本节这一组参数解决的是"同时跑几个 worker"的吞吐量问题。流水线分为 3 个阶段，每个阶段的 worker 池数量独立可调：

```
          ┌─ parse_queues["native"]  ─► [native 池  × N1] ─┐   ← legacy 共享此池
PENDING ─►├─ parse_queues["mineru"]  ─► [mineru 池  × N2] ─┼─► q_analyze ─►[analyzer × N5] ─► q_process ─►[processor × N6]
          ├─ parse_queues["docling"] ─► [docling 池 × N3] ─┤
          ├─ parse_queues["paddleocr_vl"] ─► [PaddleOCR-VL 池 × N4] ─┤
          └─ parse_queues[<第三方组>] ─► [自定义并发池]  ──┘   ← 按 ParserSpec.queue_group 动态创建
```

解析队列**按注册表的 `ParserSpec.queue_group` 动态创建**（每批取一次注册表快照）：内置 native/mineru/docling/paddleocr_vl 各占一组，legacy 共享 native 池（本地、无网络），第三方引擎可声明独立组与自定义并发数（见 `docs/ThirdPartyParser-zh.md`）。入队时 `resolve_stored_document_parser_engine` 根据每个文档的 `parser_engine`（来自 `LIGHTRAG_PARSER` 默认值或文件 hint）把它放入对应解析队列；各解析队列**完全互不阻塞**——mineru 占满不会拖慢 docling 或 native。解析完成后统一进入 `q_analyze`（多模态分析），再进入 `q_process`（实体/关系抽取 + 入库）。

| 环境变量 | 默认值 | 作用 | 调优建议 |
| --- | --- | --- | --- |
| `MAX_PARALLEL_PARSE_NATIVE` | `5` | N1: native 解析（docx / pdf / txt 等纯本地处理）并发 worker 数 | 纯 CPU、内存占用低，可按 CPU 核数提高 |
| `MAX_PARALLEL_PARSE_MINERU` | `2` | N2: MinerU 解析并发 worker 数 | MinerU 占用 GPU/CPU 显著，**默认 2 为适度并发**。资源紧张时可降到 1；本地部署且显存充足时可设 2-3；走 MinerU 官方云端服务时可适当提高（受云端配额限制） |
| `MAX_PARALLEL_PARSE_DOCLING` | `2` | N3: Docling 解析并发 worker 数 | Docling 同样资源敏感，**默认 2 为适度并发**。资源紧张时可降到 1；本地部署且 CPU/GPU 充足时可设 2-3 |
| `MAX_PARALLEL_PARSE_PADDLEOCR_VL` | `2` | N4: PaddleOCR-VL 解析并发 worker 数 | 外部云/API 配额约束，除非账号并发额度足够，否则保持默认即可 |
| `MAX_PARALLEL_ANALYZE` | `5` | N5: 多模态分析（VLM 图片 / 表格描述）并发 worker 数 | 直接消耗 VLM 配额。建议 ≤ VLM 服务并发上限 |
| `MAX_PARALLEL_INSERT` | `3` | N6: 实体 / 关系抽取 + 入库阶段并发文档数 | 推荐 `MAX_ASYNC_LLM / 3`，区间 2~10。该阶段每个文档会触发多次 LLM 调用，过高会撞 LLM 限流。同时该值还作为 `asyncio.Semaphore` 用于二次约束（worker 数和信号量值一致） |
| `QUEUE_SIZE_PARSE` | `20` | parse（native/MinerU/Docling）输入队列长度 | 一般无需调整。队列内仅为轻量 doc_id（大文档体在进入 analyze 前已剥离），仅限制 pipeline 一次预派发给 parse worker 的待处理文档数，调整影响很小 |
| `QUEUE_SIZE_ANALYZE` | `100` | analyze 队列（parse → analyze 阶段）的有界容量 | 一般无需调整。极少量大批量任务（成千上万）可适当提高，避免 enqueue 端反压；内存紧张时可调低 |
| `QUEUE_SIZE_INSERT` | `4` | analyze → process 阶段间的队列容量 | process 是流水线中最慢、最耗内存的阶段，队列特意做小，给上游提供反压防止内存堆积 |

**几个要点：**

1. **解析阶段按引擎隔离**，所以混用 native/mineru/docling/paddleocr_vl 时不必担心一种引擎慢拖累另一种。
2. **mineru / docling 默认 2**：两者资源占用高，默认保持适度并发。资源紧张时可降到 1（避免 OOM / 显存竞争 / 失败重试）；如果你部署了多 GPU 或专门的解析服务器，可手动调高。
3. **`MAX_PARALLEL_INSERT` 兼任 worker 池大小和信号量上限**：流水线创建 `Semaphore(max_parallel_insert)`，每个 process worker 在抽取入库前还要拿一次信号量。所以哪怕你把 worker 数手动改大，实际并发上限仍由这个值决定——直接调它就够了。
4. **queue size 与背压**：`QUEUE_SIZE_INSERT=4` 这个偏小的默认值是有意为之——process 阶段慢且占内存，让 analyze 阶段在队列写满时阻塞、再反压到 parse 阶段，避免一次性把成千上万份解析结果堆在内存里。
5. **改后生效方式**：所有参数通过 `.env`（或环境变量）传入，仅在 `LightRAG` 实例构造时读取一次；改完需要重启服务。

**典型调优场景：**

- 大量 PDF + 本地 MinerU 单 GPU：`MAX_PARALLEL_PARSE_MINERU=2`、`MAX_PARALLEL_ANALYZE=5`、`MAX_PARALLEL_INSERT=3`（默认即可；显存紧张时把 MINERU 降到 1）。
- 大量 PDF + MinerU 云端服务：`MAX_PARALLEL_PARSE_MINERU=3~5`（视云端配额），其它保持默认。
- 纯 docx / txt（仅走 native）：`MAX_PARALLEL_PARSE_NATIVE=10`、`MAX_PARALLEL_INSERT` 按 `MAX_ASYNC_LLM/3` 推算。
- LLM 限流明显：先降 `MAX_PARALLEL_INSERT`（process 阶段每文档多次 LLM 调用），再降 `MAX_PARALLEL_ANALYZE`（VLM 是独立配额）。

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

走完整流水线（注册表派发解析 `get_parser(engine).parse(...)` → `analyze_multimodal` → 分块 → 实体抽取），按 `full_docs.process_options` 决定每一阶段的行为。这是"首次入队"的常规流。

### 7.3 分支 B：已抽取

**一律跳过解析**（不重新调 `parse_*`），从 ANALYZING 阶段重启，并清光旧 chunks / entities 后按当前 `process_options` 重做：

| 子步骤 | 行为 |
| --- | --- |
| 引擎对比 | 若 `process_options` 隐含的引擎 ≠ `full_docs.parse_engine`，**仅 warn**，不重新解析。已抽取的内容是不可变事实，重新跑不同引擎会产生不一致。要切换引擎请先 delete 整个文档再重传。 |
| 旧 chunks / 实体 / 关系清理 | 读 `status_doc.chunks_list` 收集旧 chunk id 集，调 `_purge_doc_chunks_and_kg(doc_id, chunk_ids)`：从 `chunks_vdb` / `text_chunks` 删除 chunk 行；按 `entity_chunks` / `relation_chunks` 反查受影响的实体 / 关系，对失去全部源的条目直接从图谱与向量库删除，对仍有其它文档贡献的条目调 `rebuild_knowledge_from_chunks` 用剩余 chunks 重建；最后删除 `full_entities` / `full_relations` 中本 doc 的索引行。purge 完成后 `status_doc.chunks_list = []` / `chunks_count = 0` 重置，避免后续 state-machine upsert 写回旧 ID。 |
| `analyze_multimodal` | 对已启用模态，每次运行都会重新计算 sidecar item 分析并覆盖已有的 `llm_analyze_result`。由于 LLM cache 的存在重复计算通常会保持语义字段不变，只会重写 `analyze_time` 等运行时字段；cache miss，例如更换模型和提示词等，保存内容才可能与上次不同。 |
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
