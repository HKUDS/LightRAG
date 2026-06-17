# RAG-Anything 解析插件

本文说明 LightRAG 内置的 `lightrag_raganything_parser` 插件。它通过
`lightrag.parsers` entry point 注册一个独立解析引擎：`raganything`。

这个插件只把 RAG-Anything 当作“文档解析器”使用：调用
`raganything.parser.get_parser(...).parse_document(...)` 得到
MinerU 兼容的 `content_list`，然后复用 LightRAG 现有 sidecar、缓存和
`full_docs` 持久化流程。它不会调用 RAG-Anything 的入库流程，也不会替换
LightRAG 内置的 `mineru` 引擎。

## 适用场景

- 你已经在本机准备好了 `HKUDS/RAG-Anything`，例如 `D:\RAG-Anything`。
- 你希望解析阶段走 RAG-Anything 的 parser，但后续 chunk、实体关系抽取、
  graph 写入和查询仍然走 LightRAG。
- 你不想依赖 LightRAG 内置 `mineru` 引擎的 `MINERU_LOCAL_ENDPOINT` HTTP 服务。

## .env 配置

最小配置：

```env
RAGANYTHING_PATH=D:\RAG-Anything
RAGANYTHING_PARSER=mineru
RAGANYTHING_PARSE_METHOD=auto
LIGHTRAG_PARSER=*:raganything,*:legacy
```

可选配置：

```env
# OCR/解析语言，例如 ch、en、ja；留空则交给 RAG-Anything 默认逻辑
RAGANYTHING_LANG=ch

# 透传给 RAG-Anything parser.parse_document 的额外参数，必须是 JSON 对象
# 示例：MinerU backend/device/source/formula/table/vlm_url 等
RAGANYTHING_PARSE_KWARGS={"backend":"pipeline","device":"cuda","source":"local","formula":true,"table":true}

# 强制忽略已有 raw 缓存并重新解析
LIGHTRAG_FORCE_REPARSE_RAGANYTHING=false
```

`RAGANYTHING_PATH` 指向 RAG-Anything 仓库根目录即可。也可以不设置该变量，
但需要保证当前 Python 环境已经能直接 `import raganything`。

## 路由选择

把所有文件优先交给 RAG-Anything，失败时由规则尾部继续回退到 legacy：

```env
LIGHTRAG_PARSER=*:raganything,*:legacy
```

只让 PDF 和图片走 RAG-Anything：

```env
LIGHTRAG_PARSER=pdf:raganything,png:raganything,jpg:raganything,jpeg:raganything,*:legacy
```

也可以通过文件名 hint 指定：

```text
report.[raganything].pdf
report.[raganything-iet].pdf
```

## 输出与缓存

解析成功后会生成：

```text
<input-dir>/__parsed__/<file>.parsed/
<input-dir>/__parsed__/<file>.raganything_raw/
```

`*.raganything_raw/content_list.json` 是从 RAG-Anything 返回值归一化后的
`content_list`。图片、表格截图、公式图等本地资产会被复制到
`*.raganything_raw/images/`，再由 LightRAG 的 sidecar builder 统一写入
`*.parsed/` 目录。

raw 缓存会校验源文件大小、源文件哈希、关键文件哈希和 RAG-Anything 解析选项。
源文件或解析参数变化时会自动重新解析。

## 本地调试

```powershell
$env:RAGANYTHING_PATH="D:\RAG-Anything"
$env:RAGANYTHING_PARSER="mineru"
$env:RAGANYTHING_PARSE_METHOD="auto"
python -m lightrag.parser.cli D:\docs\demo.pdf --engine raganything --preview 5
```

如果看到 `raganything source file not found`，优先检查传入文件路径是否真实存在。
如果看到 `No module named raganything`，检查 `RAGANYTHING_PATH` 或当前 Python
环境是否能导入 RAG-Anything。
