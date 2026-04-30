# 文件处理方式配置指南

LightRAG Server 支持按文件后缀或文件名为不同文件选择不同的内容抽取引擎。默认行为保持向后兼容：如果不配置 `LIGHTRAG_PARSER`，文件会走 `legacy` 本地抽取方式。

## 抽取引擎

| 引擎 | 说明 | 默认支持后缀 |
| --- | --- | --- |
| `legacy` | 旧版兼容方式，在文件入队前由 API 进程本地抽取文本。 | Server 上传支持的全部后缀 |
| `native` | LightRAG 内置结构化抽取器。当前仅支持 DOCX。 | `docx` |
| `mineru` | 通过外部 MinerU 服务抽取文件内容。 | `pdf`, `docx`, `pptx`, `xlsx` |
| `docling` | 通过外部 Docling 服务抽取文件内容。 | `pdf`, `docx`, `pptx`, `xlsx`, `md`, `html`, `xhtml`, `png`, `jpg`, `jpeg`, `tiff`, `webp`, `bmp` |

外部引擎需要配置对应 endpoint。只要 `LIGHTRAG_PARSER` 中启用了 `mineru` 或 `docling`，服务启动时就会检查 `MINERU_ENDPOINT` 或 `DOCLING_ENDPOINT` 是否已配置；缺失时启动失败，避免文件静默回退到其他引擎。

## 默认路由规则

使用 `.env` 中的 `LIGHTRAG_PARSER` 配置默认文件处理方式：

```bash
LIGHTRAG_PARSER=pdf:mineru,docx:docling,pptx:docling,xlsx:docling,*:legacy
MINERU_ENDPOINT=http://localhost:8000/api/v1/task
DOCLING_ENDPOINT=http://localhost:8081/v1/convert/file/async
```

规则格式：

```text
后缀:引擎,后缀:引擎,*:legacy
后缀:引擎;后缀:引擎;*:legacy
```

注意事项：

- 左侧匹配的是文件后缀，不是完整文件名；应写 `pdf:mineru`，不要写 `*.pdf:mineru`。
- 规则可以使用英文逗号 `,` 或分号 `;` 分隔。
- 规则按从左到右的顺序检查；通配符规则应放在最后。
- 启动时会严格校验规则：未知引擎、错误后缀写法、显式使用不支持的后缀、外部引擎缺少 endpoint 都会导致启动失败。
- 通配符规则只会让引擎处理其能力表支持的后缀。例如 `*:mineru;html:docling` 中，MinerU 只接管 MinerU 支持的后缀，`html` 会继续匹配到后续 `docling` 规则。
- 如果所有规则都不可用，最终回退到 `legacy`。

## 单文件覆盖

可以在文件名中使用中括号临时指定单个文件的处理方式：

```text
paper.[mineru].pdf
slides.[docling].pptx
memo.[native].docx
report.[legacy].pdf
```

文件名 hint 的优先级高于 `LIGHTRAG_PARSER`。如果指定的引擎不支持该后缀，系统会回退到默认规则继续选择可用引擎。外部引擎仍需要配置对应 endpoint。

## 推荐配置

### 保持旧版行为

不配置 `LIGHTRAG_PARSER`：

```bash
# LIGHTRAG_PARSER=
```

所有文件按旧版 `legacy` 本地抽取方式处理。

### 使用 MinerU 处理 PDF，Docling 处理 Office

```bash
LIGHTRAG_PARSER=pdf:mineru,docx:docling,pptx:docling,xlsx:docling,*:legacy
MINERU_ENDPOINT=http://localhost:8000/api/v1/task
DOCLING_ENDPOINT=http://localhost:8081/v1/convert/file/async
```

适合包含复杂 PDF、PPTX、XLSX、DOCX 的场景。

### 使用 LightRAG native DOCX 解析

```bash
LIGHTRAG_PARSER=docx:native
```

DOCX 会进入 LightRAG 内置 DOCX 解析流程，其余文件继续按旧版方式处理。

### 使用通配符启用 MinerU，再指定 HTML 走 Docling

```bash
LIGHTRAG_PARSER=*:mineru;html:docling
MINERU_ENDPOINT=http://localhost:8000/api/v1/task
DOCLING_ENDPOINT=http://localhost:8081/v1/convert/file/async
```

该配置是合法的。MinerU 只会处理其能力表支持的后缀；`html` 不在 MinerU 能力表内，因此会继续匹配到 `html:docling`。其他不被 MinerU 或 Docling 接管的后缀会回退到 `legacy`。

## 处理状态和存储字段

文件入队和抽取结果会写入 `full_docs`：

| 字段 | 说明 |
| --- | --- |
| `parsed_engine` | 实际完成抽取的引擎：`legacy`, `native`, `mineru`, `docling`。对于待抽取文件，也可暂存目标引擎。 |
| `format` | 内容格式：`pending_parse`, `raw`, `lightrag`。 |
| `content` | `raw` 时保存抽取文本；`pending_parse` 时为空字符串；`lightrag` 时固定为 `{{stored-in-lightrag-doucment}}`。 |
| `lightrag_document_path` | `format=lightrag` 时保存结构化 LightRAG Document 的相对路径。 |

`pending_parse` 表示文件已经入队，但还没有完成抽取。抽取成功后会改写为 `raw` 或 `lightrag`。抽取失败时保留 `pending_parse` 和空 `content`，便于后续排查和重试。

## 原始文件归档

- `legacy`：本地抽取成功并入队后，原文件会移动到同级 `__parsed__` 目录。
- `native` / `mineru` / `docling`：文件先保留在原位置供 pipeline 解析；解析成功并写入 `full_docs` 后，原文件再移动到 `__parsed__`。
- 解析失败时，原文件不会移动，便于修复配置后重新处理。

## DOCX 配置

DOCX 处理方式统一使用 `LIGHTRAG_PARSER` 控制：

```bash
# 使用旧版本地文本抽取
LIGHTRAG_PARSER=docx:legacy

# 使用 LightRAG native DOCX 解析
LIGHTRAG_PARSER=docx:native
```
