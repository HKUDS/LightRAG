# 第三方 Parser 引擎开发与注册指南

LightRAG 的解析层通过统一的 `BaseParser` 契约 + 中央引擎注册表(`lightrag/parser/registry.py`)派发所有解析引擎。内置引擎(`native` / `legacy` / `mineru` / `docling`)与第三方引擎走完全相同的派发路径:pipeline worker 与调试 CLI 都通过 `get_parser(engine).parse(ParseContext(...))` 驱动,**没有任何针对内置引擎的特判**。因此第三方包只需做两件事:

1. **实现**一个 `BaseParser` 子类;
2. **注册**一个 `ParserSpec`(推荐通过 `lightrag.parsers` entry point 自动发现)。

完成后,该引擎自动获得:独立(或共享)的解析并发池、文件名 hint / `LIGHTRAG_PARSER` 路由规则 / API `parse_engine` 参数三种选择方式、后缀能力校验、以及 `python -m lightrag.parser.cli --engine <name>` 单文件调试支持。

> 架构背景见 RFC #3197;sidecar 文件格式见 `docs/LightRAGSidecarFormat-zh.md`;CLI 用法见 `docs/ParserDebugCLI-zh.md`。

---

## 1. 派发流程总览

```
上传/扫描 → enqueue(PENDING_PARSE, parse_engine=<engine>)
    → pipeline 按 ParserSpec.queue_group 选并发池
    → parse worker: get_parser(engine).parse(ParseContext(rag, doc_id, file_path, content_data))
        ├─ 成功 → ParseResult → 进入 analyze / chunk / KG 流水线
        └─ 抛异常 → 该文档 doc_status=FAILED(只影响这一篇)
```

引擎对单篇文档的全部职责都收敛在一次 `parse(ctx)` 调用里:解析、持久化 `full_docs`、归档源文件、返回结构化结果。

## 2. 实现 Parser

### 2.1 契约(`lightrag/parser/base.py`)

```python
class MyParser(BaseParser):
    engine_name = "myengine"          # 必须与 ParserSpec.engine_name 一致

    async def parse(self, ctx: ParseContext) -> ParseResult: ...
```

`ParseContext` 提供:

| 成员 | 说明 |
|---|---|
| `ctx.rag` | LightRAG 实例(用于 `_persist_parsed_full_docs` 等) |
| `ctx.doc_id` / `ctx.file_path` / `ctx.content_data` | 文档标识、规范化文件路径、`full_docs` 行 |
| `ctx.resolve(engine_name)` | 返回 `ResolvedSource(source_path, document_name, parsed_dir)` —— 解析磁盘源文件路径、规范化文档名、推导 `__parsed__/<base>.parsed/` 产物目录 |
| `ctx.archive_source(path)` | 解析成功并完成 `full_docs` 同步后,把源文件归档进 `__parsed__/` |

`ParseResult` 字段:`doc_id` / `file_path` / `parse_format`(`"raw"` 或 `"lightrag"`)/ `content` / `blocks_path`(无 sidecar 则 `""`)/ `parse_engine` / `parse_stage_skipped`(缓存命中等跳过场景)/ `parse_warnings`(非致命警告,会落到 `doc_status.metadata`)。

### 2.2 三条实现路径(按引擎形态选基类)

**A. 纯文本引擎(无 sidecar)— 直接继承 `BaseParser`**

参考 `lightrag/parser/legacy/parser.py`(`LegacyParser`),核心骨架:

```python
class MyTextParser(BaseParser):
    engine_name = "myengine"

    async def parse(self, ctx: ParseContext) -> ParseResult:
        rs = ctx.resolve(self.engine_name)
        source = rs.source_path
        if not source.is_file():
            raise FileNotFoundError(f"myengine source not found: {source}")

        text = await asyncio.to_thread(my_extract, source)  # CPU 活进线程
        if not text.strip():
            raise ValueError(f"extracted no usable text from {ctx.file_path}")

        await ctx.rag._persist_parsed_full_docs(ctx.doc_id, {
            "content": text,
            "file_path": ctx.file_path,
            "parse_format": FULL_DOCS_FORMAT_RAW,
            "parse_engine": self.engine_name,
            "update_time": int(time.time()),
        })
        await ctx.archive_source(str(source))
        return ParseResult(
            doc_id=ctx.doc_id, file_path=ctx.file_path,
            parse_format=FULL_DOCS_FORMAT_RAW, content=text,
            blocks_path="", parse_engine=self.engine_name,
        )
```

**B. 本地解析、产 sidecar — 继承 `NativeParserBase`**(`lightrag/parser/native_base.py`)

模板固定了「预清理产物目录(带回滚)→ 线程内抽取 → 构建 IR → 写 sidecar → 持久化 → 归档」的完整流程,只需实现两个钩子:

```python
class MyNativeParser(NativeParserBase):
    engine_name = "myengine"

    def extract(self, source, *, parsed_dir, asset_dir, base_name):
        """同步,在线程中运行;返回 (blocks, warnings, metadata)。
        可在 write_sidecar 之前把图片等资产写入 asset_dir。"""

    def build_ir(self, blocks, *, document_name, asset_dir_name, metadata) -> IRDoc:
        """blocks → IRDoc(交给统一的 sidecar writer)。"""
```

可选覆写:`validate_source`(默认仅要求文件存在)、`surface_warnings`(把抽取警告映射为 `parse_warnings`)。参考实现:`lightrag/parser/docx/parser.py`。

**C. 外部解析服务(下载 raw bundle + 缓存)— 继承 `ExternalParserBase`**(`lightrag/parser/external/_base.py`)

模板固定了「raw 缓存命中检查 → 未命中则清目录重新下载 → 构建 IR → 写 sidecar → 持久化 → 归档」,实现三个钩子 + 两个类属性:

```python
class MyExternalParser(ExternalParserBase):
    engine_name = "myengine"
    raw_dir_suffix = ".myengine_raw"          # raw bundle 目录后缀(以 . 开头)
    force_reparse_env = "LIGHTRAG_FORCE_REPARSE_MYENGINE"

    def is_bundle_valid(self, raw_dir, source_path) -> bool: ...   # 缓存命中检查
    async def download_into(self, raw_dir, source_path, *, upload_name): ...
    def build_ir(self, raw_dir, document_name) -> IRDoc: ...
```

可选覆写 `validate_ir`(构建后校验,如零 block 报错)。参考实现:`lightrag/parser/external/mineru/parser.py`、`.../docling/parser.py`。

### 2.3 失败语义(重要)

- `parse(ctx)` **抛出任何异常 ⇒ 仅该文档标记 FAILED**,错误信息写入 `doc_status.error_msg`,不影响批内其他文档。
- 解析出**空内容时应抛异常**而不是返回空串——否则会产出一篇零知识文档静默进入 chunking(内置引擎均遵循此约定)。
- worker 在调用引擎前会做后缀守门:`PENDING_PARSE` 文档的后缀不在该引擎 `ParserSpec.suffixes` 内 ⇒ 直接 FAILED,引擎代码不会被调用。

## 3. 声明 `ParserSpec`(能力元数据)

```python
from lightrag.parser.registry import ParserSpec, register_parser

register_parser(ParserSpec(
    engine_name="myengine",
    impl="my_pkg.parser:MyParser",       # "module:Class",get_parser 时才懒加载
    suffixes=frozenset({"pdf", "foo"}),  # 小写、不带点
    queue_group="myengine",              # 见下文并发模型
    concurrency=int(os.getenv("MAX_PARALLEL_PARSE_MYENGINE", "2")),
    # 仅外部服务引擎需要(routing 在 endpoint 未配置时跳过该引擎):
    endpoint_configured=lambda: bool(os.getenv("MYENGINE_ENDPOINT", "").strip()),
    endpoint_requirement=lambda: "MYENGINE_ENDPOINT",
))
```

| 字段 | 必填 | 说明 |
|---|---|---|
| `engine_name` | ✓ | 注册表键,也是 `--engine` / 文件名 hint / `LIGHTRAG_PARSER` 里的引擎名。**与已有名字相同会覆盖原注册**(包括内置引擎)——除非有意替换实现,请勿与 `native/legacy/mineru/docling` 撞名。 |
| `impl` | ✓ | `"module:Class"` 字符串。注册表只在文档实际解析时才 import 它,**注册阶段绝不能提前 import 实现**(保持能力查询 import-cheap,这是注册表的设计不变量)。 |
| `suffixes` | ✓ | 该引擎能处理的扩展名(小写无点)。用于路由校验与 worker 端后缀守门。 |
| `queue_group` | | 并发池分组,默认 `"native"`(共享 native 池)。独立池填唯一组名。 |
| `concurrency` | | 该组 worker 数(组的唯一 owner 才需要填)。环境变量覆盖由**注册方在注册时自行烘焙**(如上例 `int(os.getenv(...))`),注册值即权威值。 |
| `endpoint_configured` / `endpoint_requirement` | | 零参闭包(只读 env、不发网络)。前者返回该引擎依赖的外部服务是否已配置;后者返回缺失时提示用户的配置项名。本地引擎不用填(默认恒可用)。 |
| `user_selectable` | | 默认 `True`。`False` 表示内部格式 handler(如 `reuse`/`passthrough`),不会出现在引擎选择面。 |

### 并发模型

- pipeline 每批为**每个 `queue_group` 建一条队列 + 一组 worker**;
- 组的 worker 数:内置组(`native`/`mineru`/`docling`)由 LightRAG 实例字段 `max_parallel_parse_*` 决定(支持构造参数覆盖);第三方独立组取该组唯一 owner spec 的 `concurrency`;一个组**只能有一个**声明了 `concurrency` 的 spec,多个会在批启动时报错;
- `queue_group="native"` 蹭内置池时,`concurrency` 不生效(池大小由 `max_parallel_parse_native` 决定)——本地轻量引擎(如 legacy)适合这种方式,外部服务引擎建议独立组以免慢请求拖住本地解析。

## 4. 注册:entry point 自动发现(推荐)

LightRAG 通过 `lightrag.parsers` entry-point group 自动发现第三方引擎(`lightrag/parser/plugins.py`)。第三方包只需:

**① 在自己的 `pyproject.toml` 声明 entry point:**

```toml
[project.entry-points."lightrag.parsers"]
myengine = "my_pkg.lightrag_plugin:register"
```

**② 提供一个零参注册函数:**

```python
# my_pkg/lightrag_plugin.py —— 保持 import-cheap:不要在这里 import 解析实现
import os
from lightrag.parser.registry import ParserSpec, register_parser

def register() -> None:
    register_parser(ParserSpec(
        engine_name="myengine",
        impl="my_pkg.parser:MyParser",   # 实现走懒加载
        suffixes=frozenset({"foo"}),
        queue_group="myengine",
        concurrency=int(os.getenv("MAX_PARALLEL_PARSE_MYENGINE", "2")),
    ))
```

`pip install my-pkg` 之后即装即用,无需改动 LightRAG 代码:

- **API Server**:`create_app()` 在校验 `LIGHTRAG_PARSER` 路由规则**之前**调用 `load_third_party_parsers()`,因此路由规则可以直接引用第三方引擎名(如 `LIGHTRAG_PARSER="foo:myengine"`);
- **调试 CLI**:`python -m lightrag.parser.cli sample.foo --engine myengine` 直接可用(`main()` 在构建 `--engine` choices 前加载插件)。无 sidecar 的引擎(`blocks_path=""`)CLI 会打印纯文本摘要而非 blocks 摘要;继承 `ExternalParserBase` 的引擎自动获得 raw 缓存展示与 `--force-reparse` 支持;
- **库内嵌用法**(不经 server/CLI 直接用 LightRAG 类):在构建 pipeline 前自行调用一次:

```python
from lightrag.parser.plugins import load_third_party_parsers
load_third_party_parsers()   # 进程内幂等
```

加载语义:每进程幂等(重复调用为 no-op);**单个插件抛异常只会被记录并跳过**,不会影响其他插件或内置引擎,更不会阻断 server 启动——但该引擎将不可用,请关注启动日志中的 `[parser-plugins]` 行。

> 不想发包时也可以跳过 entry point,在自己的启动脚本里直接 `register_parser(...)` 后再启动/调用 LightRAG——注册表是进程内模块单例,效果相同,只是没有"装上即生效"。

## 5. 路由:让文档用上你的引擎

引擎选择优先级(`lightrag/parser/routing.py`):

1. **文件名 hint**:`report.[myengine].foo`(可带处理选项 `report.[myengine-iet].foo`);
2. **`LIGHTRAG_PARSER` 规则**:如 `LIGHTRAG_PARSER="foo:myengine,pdf:mineru"`(按后缀 glob 匹配,首条命中生效);
3. **默认**:`legacy`。

API 上传时显式传 `parse_engine="myengine"` 则直接固定该引擎(存入 PENDING_PARSE 行,worker 原样履约;后缀不支持会 FAILED 而非静默回退)。注册了 `endpoint_configured` 的引擎在 endpoint 未配置时会被路由跳过(hint/规则校验也会给出 `endpoint_requirement` 提示)。

## 6. 测试建议

- **单测引擎本体**:绕开 CLI,直接 `get_parser("myengine").parse(ParseContext(fake_rag, doc_id, file_path, content_data))`。`fake_rag` 只需提供 `_persist_parsed_full_docs` / `_resolve_source_file_for_parser` / `full_docs` / `doc_status`(参考 `lightrag/parser/debug.py` 的 `build_debug_rag()`,或 `tests/parser/test_legacy_parser.py` 的极简 `_FakeRag`);
- **注册表是模块级单例**:测试中 `register_parser` 后,用 `finally: registry._REGISTRY.pop("myengine", None)` 清理(参考 `tests/parser/test_registry.py`);
- **entry-point 加载逻辑**:参考 `tests/parser/test_plugins.py`(monkeypatch `lightrag.parser.plugins.entry_points` 注入 fake entry point;`plugins._loaded` 标志需复位)。
