# 第三方分块器

已安装的 Python 包可以注册六参数分块回调，无需修改 LightRAG。Server 启动时发现注册信息并注入选中的一个回调。本功能对应 [#3868](https://github.com/HKUDS/LightRAG/issues/3868)，不包含 `C(name=...)`、上下文契约、身份强制校验、多活动分块器或 F/R/V/P 的 transform hook。

## 发布轻量注册入口

在插件包的 `pyproject.toml` 中声明：

```toml
[project.entry-points."lightrag.chunkers"]
acme = "acme_chunker.plugin:register"
```

包的 `__init__.py` 和注册模块均不要导入分块实现：

```python
# acme_chunker/plugin.py
from lightrag.chunker.registry import ChunkerSpec, register_chunker

def register():
    register_chunker(ChunkerSpec(
        name="acme",
        impl="acme_chunker.implementation:chunk",
        version="1",
        description="Organization-specific document splitting",
        executor_safe=False,
    ))
```

一个零参数入口函数可以注册多个 spec。注册仅保存元数据，只有选中的 `impl` 才被加载。插件是受信任的已安装 Python 代码，不是沙箱；作者不得在注册入口提前导入实现、访问网络或初始化进程/线程资源。

名称必须满足 `[a-z0-9][a-z0-9_-]{0,63}`，保留名为 `fixed_token`、`recursive_character`、`semantic_vector`、`paragraph_semantic` 以及 `f/r/v/p/c`。`version` 是仅用于诊断的不透明字符串，不能作为兼容性保证；`description` 必须为非空单行说明。以后可增加可选 spec 字段而不改变 entry-point 契约。

发现过程每进程一次；Gunicorn preload 在 fork 前完成解析，worker 继承同一注册/选择快照。实现模块导入也应兼容 preload：与事件循环、网络绑定的资源在回调中创建，不在导入时创建。安装/升级插件或更改选择后需重启部署。入口按发行包及 entry-point 来源确定性排序。重名记录包含双方来源的 ERROR；选中重名则启动失败，未选中重名保留最后一项。失败的注册入口会连同其部分注册结果一起丢弃并记录来源；若它未能提供所选名称，后续校验以未知名称拒绝启动。

## 实现原有六参数契约

```python
def chunk(tokenizer, content, split_by_character, split_by_character_only,
          chunk_overlap_token_size, chunk_token_size):
    # Return ordered dictionaries with tokens, content and chunk_order_index.
    ...
```

支持同步和异步函数，始终传入这六个位置参数，不传 `_emit_source_span` 等内置私有参数。默认在事件循环上调用，包含返回 awaitable 的同步工厂函数。异常导致文档 FAILED，不会静默改用其它算法。现有下游块校验与 embedding 大小处理仍适用；自定义输出不进行 source-span sidecar 回填。

CPU 密集、同步且线程安全的实现可以声明 `executor_safe=True`，复用 LightRAG 有界分块线程池。该实现不得依赖当前事件循环或返回 awaitable。异步函数/异步可调用对象声明该选项会在启动时被拒绝。依赖事件循环的实现保持默认，或在异步回调中自行 offload；这个声明是作者承诺，不是线程安全证明或隔离机制。

## Server 选择

将插件安装进运行 LightRAG 的同一环境，配置：

```dotenv
CUSTOM_CHUNKER=acme
```

也可使用 `lightrag-server --custom-chunker acme` 或 `lightrag-gunicorn --custom-chunker acme`，CLI 优先于 `.env`。值只能是已注册的裸名称，不能是 Python 导入路径、文件路径或代码，HTTP 请求也不能指定实现。

未知/有歧义的名称、导入/属性错误、不可调用对象、可检查但不能接受六个位置参数的签名均导致启动失败。没有可检查签名的原生可调用对象仍允许加载，但运行与输出契约不变。

注入等同于 `LightRAG(chunking_func=...)`，作用于**显式 C 和不带分块 selector 的插入**，不是 C 专用。F/R/V/P 仍选内置策略并保留现有 bypass 告警。文本 API 用 `chunking.strategy="custom"`，文件/路由 hint 用 C。未设置时不覆盖默认回调：新 C 请求仍返回 422，已持久化/后台 C 仍逐次告警并回退精确定长分块。仅安装插件并不会选中它。

## SDK 嵌入

```python
from lightrag import LightRAG
from lightrag.chunker.plugins import load_and_resolve_chunker

callback = load_and_resolve_chunker("acme")
rag = LightRAG(chunking_func=callback)  # Add storage/LLM configuration.
# await rag.initialize_storages() before inserts; finalize on completion.
```

SDK 也可直接 `register_chunker(spec, origin="my-application")`。未配置的解析结果为 `None`：此时应省略构造参数，而非传入 `chunking_func=None`。

若仅需发现插件，可调用 `load_third_party_chunkers()`，再使用 registry 中的 `registered_chunker_names()` 或 `resolve_chunker(name)`。仅发现的接口会记录失败，但不会声称选择已通过校验。

## 诊断与重处理

一条启动 INFO 列出名称、版本、说明、来源和选择结果，并说明不带 selector 的插入也受影响。注册失败是 ERROR，不影响无关的有效选择。Server 的每条失败日志会注明选择校验后的结果：指定的分块器仍可用于 C/无 selector 插入、选择失败导致启动中止，或未配置时维持现有 C 接纳/回退行为。启动校验成功不保证插件之后的运行或输出一定正确。

`doc_status.metadata.custom_chunker` 独立于 `chunk_opts` 保存最近一次尝试的观察值，例如 `{"name": "acme", "version": "1", "authoritative": false}`；`chunk_method` 原有字符串不变。旧观察值被没有注册回调的一次尝试替换时，名称/版本记为 null。

观察值跨重置保留仅供比较：持久化 C 文档的旧名称/版本与当前配置不同时，每次处理尝试记录一条 drift WARNING，随后按**当前**回调执行，或在移除配置后按原有规则告警回退。fallback 告警独立保留原有逐文档/逐次频率。记录中的身份不参与选择或阻止执行；同名同版本无法证明实现未变，包版本及部署可重现性需另外管理。改变配置不会自动重处理已完成文档。
