# 设计提案：将整个 INPUT_DIR 后端化到 S3（面向分布式部署）

> 状态：草案 / 待评审
> 关联：PR #3331（Sidecar 图片资源上传 S3）、Issue #3315
> 适用版本：基于 `main` @ `40a62c9`

## 1. 背景与动机

当前 LightRAG 的 `INPUT_DIR`（默认 `./inputs`）是一个**服务器本地目录**，承担了三类完全不同的职责：

1. **用户上传的原始文件**：`/documents/upload` 把字节流写到 `input_dir/<filename>`。
2. **解析引擎的中间产物 / 缓存**：external parser 的原始包 `*.mineru_raw/`、`*.docling_raw/`、native 的 `*.native_raw/`。
3. **解析后的 Sidecar 产物与归档源文件**：`__parsed__/<name>.parsed/`（`*.blocks.jsonl`、`*.drawings.json`、`*.tables.json`、`*.equations.json`、`*.blocks.assets/`）以及处理完成后被移动到 `__parsed__/` 下的源文件。

这种"全部落本地盘"的设计有三个根本性问题：

- **阻碍分布式 / 多副本部署**。多个 API / worker 副本各自有独立的本地盘。副本 A 接收的上传、A 写出的 Sidecar，副本 B 看不到。`/scan`、去重（按文件名 / content hash）、删除、查询溯源都会在副本间不一致。
- **查询结果无法向客户端溯源原文 / 图片**。Sidecar 里图片资源记录的是 `file://` 本地路径（见 `utils_pipeline.py` 的 `sidecar_uri_for`），浏览器无法访问服务器文件系统。这正是 Issue #3315 的痛点。
- **数据生命周期与容器生命周期耦合**。容器/Pod 重建即丢数据，必须靠 PVC（见 `k8s-deploy/`），运维成本高且难以水平扩展。

### 1.1 PR #3331 为什么不够

PR #3331 的方案是：在 Sidecar 写盘后（Stage 1b），把 `*.blocks.assets/` 里**已经物化的图片**额外上传一份到 S3，并把远程 URL 写进 `blocks.jsonl` 占位符和 `drawings.json` 的 `remote_url` 字段。

它解决了"图片给前端展示"的窄问题，但**没有改变 INPUT_DIR 的本质**：

- 上传的源文件仍然只在本地盘；
- 解析中间产物（mineru_raw / docling_raw）仍然只在本地盘；
- Sidecar 的 `blocks.jsonl` / 各模态 JSON 仍然只在本地盘，VLM 仍然强依赖"从本地磁盘读图"（PR 注释明确写了 `local path is always preserved so VLM analysis reads from disk`）；
- `/scan`、去重、删除、归档仍然全部基于本地 `iterdir/glob/unlink/rmtree`。

因此它无法支撑分布式，也无法做到"任意副本都能溯源任意文档的原文与资源"。本提案的目标是**把整个 INPUT_DIR 抽象为一个可插拔的对象存储后端**，本地盘只是其中一种实现，S3 是面向分布式的实现。

## 2. 目标与非目标

**目标**

- G1：`INPUT_DIR` 承载的全部数据（原始文件、中间产物、Sidecar 产物、归档源文件）都通过统一的**存储抽象层**读写，后端可在"本地文件系统"与"S3 兼容对象存储"间切换。
- G2：默认行为零变化、零配置——不开启 S3 时，行为与今天逐字节一致（向后兼容）。
- G3：多副本下，任一副本都能读到其它副本写入的上传 / Sidecar / 归档，去重、扫描、删除、查询溯源全部一致。
- G4：查询结果可向客户端安全地溯源原文与图片资源（presigned URL 或服务端代理），无需把服务器本地文件暴露给客户端。
- G5：提供从既有本地部署到 S3 的迁移路径，且历史已写入 `full_docs.sidecar_location` 的 `file://` URI 仍可解析。

**非目标**

- 不改动 KV / Vector / Graph / DocStatus 四类**结构化存储**后端（它们已有自己的可插拔体系，见 `kg/`）。本提案只针对**文件 / blob 存储**。
- 不强制要求 S3；本地后端继续是默认与单机推荐。
- 不在本提案内实现对象存储侧的生命周期 / 版本 / 加密策略（交给 S3 自身配置）。

## 3. 现状盘点：本地文件系统触点

下表是需要被抽象层接管的全部触点（已逐一核对源码），按职责分组。这是改造的"工作清单"。

### 3.1 配置与路径解析
- `lightrag/api/config.py:274-277,487-488` — `--input-dir` 参数，`abspath` 解析。
- `lightrag/utils_pipeline.py:540-542` — `configured_input_dir()`，读 `INPUT_DIR` 环境变量。
- `lightrag/utils_pipeline.py:634-676` — `input_dir_path()` / `parsed_dir()` / `parsed_artifact_dir_for()`，路径拼装。
- `lightrag/api/routers/document_routes.py:977-998` — `DocumentManager.__init__` 拼 `base_input_dir / workspace` 并 `mkdir`。

### 3.2 上传写盘
- `document_routes.py:118-161` — `sanitize_filename`（路径穿越校验）。
- `document_routes.py:2692,2715-2729` — 目标路径计算 + 重名预检（`exists` / `find_existing_file_by_file_path` 的 `iterdir`）。
- `document_routes.py:2736-2769` — `aiofiles` 流式写盘 + 超限 `unlink`。
- `document_routes.py:94-95,1746-1750` — `__tmp__` 临时文件前缀与 enqueue 后清理。

### 3.3 扫描发现
- `document_routes.py:1028-1055` — `scan_directory_for_new_files`，按各引擎后缀 `glob("*{ext}")`。
- `document_routes.py:1421-1431` — `find_existing_file_by_file_path`，`iterdir` 找同名。

### 3.4 解析管线（读源 + 写产物 + 缓存）
- `parser/external/_base.py:116-163` — `source.is_file()` 检查；`raw_dir_for_parsed_dir` 解析原始包目录；`mkdir` + `clear_dir_contents`；`download_into` 把字节交给外部 HTTP 服务；`build_ir`；`write_sidecar`。
- `sidecar/writer.py:60-112` — `write_sidecar` 清理/创建 `parsed_dir`（`exists`/`rmtree`/`mkdir`）。
- `sidecar/writer.py:330-367` — `write_text` 写 `blocks.jsonl` 及各模态 JSON。
- `sidecar/writer.py:395-460` — `_materialize_assets`：`mkdir` / `copyfile` / `write_bytes` 物化资源。
- `s3_uploader.py`（PR #3331 新增）— Stage 1b 资源上传。

### 3.5 Sidecar URI / 读产物
- `utils_pipeline.py:690-751` — `sidecar_uri_for` / `resolve_sidecar_uri` / `sidecar_blocks_path` / `sidecar_modality_path` / `sidecar_assets_dir_for_uri`。**这是关键扩展点**——URI scheme 已预留 `s3://`（注释见 `utils_pipeline.py:683-687`，`resolve_sidecar_uri` 对非 `file://` 返回 `None`）。
- `utils_pipeline.py:786-816` — `load_lightrag_document_content`，`open` 读 `blocks.jsonl` 合并正文。

### 3.6 资源 / 图片向客户端服务
- 当前**没有**真正的"文件下发"端点（`document_routes.py` 无 `FileResponse`）。Sidecar 里是 `file://` URI，客户端无法消费 → Issue #3315。PR #3331 用"上传 S3 + 远程 URL"绕过。

### 3.7 删除 / 清空
- `document_routes.py:1479-1558` — `delete_file_variants_by_file_path`：`iterdir` + `unlink`（源文件）+ `rmtree`（`.parsed` / `*_raw` 目录）。
- `document_routes.py:3207-3210` — `/documents/clear` 对 `input_dir.glob("*")` 逐个 `unlink`。

### 3.8 归档与存储中的路径引用
- `utils.py:741-787` — `get_unique_filename_in_parsed` / `move_file_to_parsed_dir`（`rename`）。
- `utils_pipeline.py:759-778` — `archive_source_after_full_docs_sync`。
- `pipeline.py:594,806-837` — 把 `file_path` 写入 doc_status / full_docs。
- `parser/external/_base.py:165-179` — 把 `sidecar_location`（`file://` URI）写入 full_docs。

## 4. 核心设计：文件存储抽象层（FileStore）

引入一个与 `kg/` 的 `BaseKVStorage` 同构的抽象——`BaseFileStorage`（暂名 `FileStore`），把"INPUT_DIR"从一个 `Path` 升级为一个**逻辑命名空间 + 后端实现**。

### 4.1 抽象接口

```python
# lightrag/filestore/base.py
class BaseFileStore(ABC):
    """以 'key'（相对逻辑路径，使用 '/' 分隔）寻址的对象存储抽象。

    key 例： 'report.pdf'
            '__parsed__/report.pdf.parsed/report.blocks.jsonl'
            '__parsed__/report.pdf.parsed/report.blocks.assets/img-001.png'
    workspace 由实现内部前缀化（与 kg 一致），调用方只用相对 key。
    """

    # --- 字节级 ---
    async def put_bytes(self, key: str, data: bytes, *, content_type: str | None = None) -> None: ...
    async def put_stream(self, key: str, fileobj) -> None: ...        # 流式/分片上传
    async def get_bytes(self, key: str) -> bytes: ...
    async def open_stream(self, key: str): ...                         # 返回可异步读的流
    async def exists(self, key: str) -> bool: ...
    async def size(self, key: str) -> int | None: ...

    # --- 列举 / 删除 ---
    async def list(self, prefix: str) -> list[str]: ...                # 替代 glob/iterdir
    async def delete(self, key: str) -> None: ...
    async def delete_prefix(self, prefix: str) -> None: ...            # 替代 rmtree

    # --- 移动 / 复制（S3 无原子 rename → copy+delete）---
    async def copy(self, src_key: str, dst_key: str) -> None: ...
    async def move(self, src_key: str, dst_key: str) -> None: ...      # 归档用

    # --- 与"必须有真实文件路径"的工具交互（关键）---
    @asynccontextmanager
    async def materialize(self, key: str) -> AsyncIterator[Path]:
        """把对象下载到本地临时文件，产出本地 Path，退出时清理。
        local 后端直接产出真实路径（零拷贝）。"""

    @asynccontextmanager
    async def staging_dir(self, prefix: str) -> AsyncIterator[Path]:
        """提供一个本地临时目录用于写一批产物，退出时把目录内文件批量
        put 回 store（local 后端直接就地写）。供 write_sidecar / 外部解析包使用。"""

    # --- 向客户端溯源 ---
    async def public_url(self, key: str, *, expires: int = 3600) -> str:
        """S3：返回 presigned GET URL；local：返回服务端代理端点 URL。"""

    # --- URI 互转（衔接已有 sidecar_location）---
    def uri_for(self, key: str) -> str: ...        # local→file://  s3→s3://bucket/...
    def key_for_uri(self, uri: str) -> str | None: ...
```

### 4.2 两个实现

- **`LocalFileStore`**（默认）：`materialize` 零拷贝（直接返回真实路径），`staging_dir` 就地写，`list` = `glob`，`move` = `rename`，`public_url` 返回服务端代理端点。即把今天的全部本地逻辑收拢进来，行为不变。
- **`S3FileStore`**：基于 `boto3`（复用 PR #3331 已加入的 `s3-upload` 可选依赖）。`put_stream` 用 `upload_fileobj`（自动分片），`materialize`/`staging_dir` 用本地临时目录搭桥，`list` 用 `list_objects_v2`（分页），`move`=copy+delete，`public_url` 用 `generate_presigned_url`。workspace 与 `__parsed__` 都体现为 key 前缀。

### 4.3 工厂与配置

仿照 `kg/factory.py::get_storage_class()`，加 `filestore/factory.py`：

```
INPUT_STORAGE=local            # 默认；或 s3
```

`S3FileStore` 复用一组统一前缀的环境变量（与 PR #3331 命名收敛）：
```
INPUT_STORAGE=s3
S3_ENDPOINT=...           S3_BUCKET=...
S3_ACCESS_KEY=...         S3_SECRET_KEY=...        # 回退 AWS_*
S3_REGION=us-east-1       S3_PREFIX=lightrag/
S3_PUBLIC_URL_PREFIX=...  # 可选 CDN
S3_PRESIGN_EXPIRE=3600    # presigned URL 有效期
S3_VERIFY_TLS=true        # 注意：PR #3331 写死 verify=False，本设计默认 true 并可配
```

### 4.4 单一注入点

`DocumentManager` 不再持有 `input_dir: Path`，改为持有 `store: BaseFileStore` 与逻辑前缀。`utils_pipeline` 中所有 `Path` 拼装函数（`parsed_artifact_dir_for` 等）改为返回 **key** 而非 `Path`。`LightRAG` 实例在 `initialize_storages()` 时从工厂构建 store（与现有 storage 初始化同处）。

### 4.5 VLM 读图：staging 本地副本 + materialize 缓存语义

这是本设计里需要单独讲清的一环，因为多模态分析（VLM）是 INPUT_DIR 里唯一**强依赖"本地真实文件路径"**的消费者，也是 PR #3331 不敢动它、只能"另传一份"的根因。

**现状（严格读本地盘）。** Layer 2 的 `analyze_multimodal` 对每个 drawing 这样取图（`pipeline.py:3591-3654`）：

```python
def _resolve_image_path(path_str, sidecar_dir) -> Path | None:
    candidate = Path(path_str)
    if not candidate.is_absolute():
        candidate = sidecar_dir / path_str       # drawings.json 的 path 相对 sidecar 目录
    if candidate.exists() and candidate.is_file():
        return candidate
    return None
...
raw = candidate.read_bytes()                      # 直接读本地磁盘字节
img_payload = {"base64": base64.b64encode(raw).decode("ascii"), ...}
```

即 `drawings.json.path` → 拼本地路径 → `exists()/is_file()` → `read_bytes()` → base64 → 喂 VLM。全程要求那个图片文件在**本地文件系统**真实存在。这就是 PR #3331 注释 `local path is always preserved so VLM analysis reads from disk` 的来由——它把图额外传 S3 仅供前端，VLM 仍读本地。

**问题。** INPUT_DIR 后端化到 S3 后，`*.blocks.assets/img-001.png` 的权威副本在 S3，本地盘不一定有。若运行 VLM 的副本不是当初解析该文档的副本（重试、事后重分析、解析与分析被调度到不同 worker），`candidate.exists()` 直接 `False` → 跳过或失败。

**设计：一个 `materialize` 调用 + 一层本地缓存,把两种情况收敛。** 把上面"拼本地路径 + read_bytes"改为:

```python
async with store.materialize(asset_key) as local_path:
    raw = local_path.read_bytes()
    ...  # 尺寸校验、_VLM_RASTER_EXTS、max_image_bytes、base64、缓存 key 全部不变
```

- **`LocalFileStore.materialize`** → 零拷贝，直接返回真实路径（行为等于今天）。
- **`S3FileStore.materialize`** → 先查**本地缓存目录**:
  - **热路径(同一次解析批内)**:解析 Layer 1 时,`write_sidecar` 在 `staging_dir()` 的本地临时目录里产出 `blocks.jsonl` 与 `*.blocks.assets/` 的图,再 flush 回 S3;VLM Layer 2 紧接着在**同一 worker、同一文档**上跑,那批本地副本(=缓存)仍在 → 缓存命中,近乎零成本,性能与今天一致。这是绝大多数情况。
  - **冷路径(跨副本 / 事后重分析)**:本地缓存未命中 → 从 S3 下载到缓存目录再返回本地路径,用完按下方清理规则处理。只有这种情况付一次下载。

如此,"staging 本地副本"与"跨副本 materialize"不再是两段不同代码,而是同一个 `materialize` 的**缓存命中 / 未命中**两种结果。

**对 `drawings.json` 的影响。** `path` 不再存"永久本地绝对路径",改存**相对 store 的 key**(相对 sidecar 的文件名,与今天的相对 path 几乎一致)。VLM 路径解析为:`full_docs.sidecar_location` URI → store + 基准 key → 拼出 asset key → `materialize`。由此 **PR #3331"必须永久保留本地 path 供 VLM 读盘"的约束被解除**;给前端的 URL 也由 `store.public_url(key)`(S3=presigned URL)统一产出,不再单独维护 `remote_url` 字段。

**两个工程注意点。**

1. **清理(借用 vs 拥有)**：`materialize` 用上下文管理器。退出时,只删除"本次冷路径为你下载的临时文件";指向 staging 缓存/本地真实路径的"借用"结果**不能删**。实现上需区分 owned/borrowed 两类返回,避免误删尚被同批次其它步骤使用的缓存。
2. **并发**：同一图片可能被并发 `materialize`。缓存写入用 `<key>.tmp` + 原子 rename(或按 key 加锁),避免读到写一半的文件。这与 external parser `*_raw/` 缓存命中判定(`is_bundle_valid`)是同类问题,可复用同一套缓存目录与原子写约定。

## 5. 各环节改造映射

| 环节 | 今天 | 改造后 |
|---|---|---|
| 上传写盘 | `aiofiles.open(path,'wb')` 流式 | `store.put_stream(key, upload.file)`；超限改为上传前/中止并 `store.delete(key)` |
| 临时上传 | `__tmp__` 前缀文件 | `__tmp__` 前缀 key，enqueue 后 `store.delete` |
| 重名预检 | `path.exists()` / `iterdir` | `store.exists(key)` / `store.list(prefix)`（仍优先 content-hash 去重，少依赖列举） |
| 扫描 | `glob("*{ext}")` | `store.list("")` 按后缀过滤；大规模下建议以 doc_status 为权威、扫描为补偿 |
| 读源给解析器 | 传本地 `Path` | `async with store.materialize(key) as p:` 再交给 `download_into(p)` |
| 解析中间包 `*_raw/` | 本地 `mkdir`+写 | `store.staging_dir()` 内构建后回传；缓存命中用 `store.exists`/`list` 判定 |
| 写 Sidecar | `write_text`/`copyfile` 到 `parsed_dir` | `write_sidecar` 在 `staging_dir` 内产出，退出时批量回传；或直接 `put_bytes` |
| 资源物化 | `_materialize_assets` 写盘 | 同上；PR #3331 的"另传一份"被自然取代——资源本就在 store |
| VLM 读图 | 读本地 disk（`read_bytes`） | `async with store.materialize(asset_key)`：热路径命中 staging 本地缓存、冷路径下载（详见 §4.5） |
| Sidecar URI | `file://abs/...` | `store.uri_for(key)`：local→`file://`，s3→`s3://bucket/prefix/...` |
| 读 blocks.jsonl | `open(path)` | `store.get_bytes(key)` / `open_stream`；`load_lightrag_document_content` 按 URI→key→store 读 |
| 向客户端溯源 | 无 / file:// | 新增 `GET /documents/file?uri=...`：s3 后端 302 到 presigned URL，local 后端校验后 `FileResponse` |
| 删除 | `unlink` / `rmtree` | `store.delete` / `store.delete_prefix(parsed_dir_key)` |
| 清空 | `glob("*")`+`unlink` | `store.delete_prefix(workspace_prefix)` |
| 归档 | `rename` 到 `__parsed__/` | `store.move(src_key, parsed_key)` |

关键点：**`sidecar_location` URI 仍是文档与其产物之间的唯一句柄**。我们只是把 URI 的 scheme 从单一 `file://` 扩展为 `file://` + `s3://`，解析时按 scheme 路由到对应 store。`resolve_sidecar_uri` 今天对 `s3://` 返回 `None`，正好是要补齐的 resolver。

## 6. 关键技术问题与决策

**Q1 外部解析器（mineru/docling）如何拿到源文件？**
它们是 HTTP 适配器，`download_into` 需要本地字节。决策：用 `store.materialize(key)` 下载到本地临时文件再交给适配器。不引入"presigned URL 直传给第三方"的复杂性（第三方未必能访问我们的 S3）。

**Q2 VLM 必须读本地磁盘？** 详见 §4.5。
要点:VLM 改为 `async with store.materialize(asset_key) as p:` 取本地路径,其余逻辑不变。解析同批次内命中 `staging_dir` 写下的本地缓存(零额外下载),仅跨副本/事后重分析时才真正从 S3 下载。由此**移除** PR #3331 中"必须永久保留本地 path 供 VLM"的约束。

**Q3 向客户端溯源用 presigned 还是代理？**
两者都支持，由后端决定：S3 用 presigned URL（302 重定向，卸载带宽、支持有效期），local 用服务端 `FileResponse` 代理。统一经 `store.public_url(key)`，对调用方透明。**推荐**：S3 后端默认 presigned，可配 `S3_PUBLIC_URL_PREFIX` 走 CDN。

**Q4 分布式正确性 / 原子性。**
- S3 单对象 PUT 原子，天然适合"上传完才可见"。`__tmp__` 前缀 + enqueue 后 rename(copy+delete) 的两阶段仍成立。
- 无目录概念：所有"目录"操作改为前缀语义；`delete_prefix` 需分页删除。
- 跨副本互斥仍由现有 `shared_storage`（Redis 命名空间锁）负责（见 AGENTS.md 的 pipeline 并发契约），**不**依赖文件系统做锁。
- 去重以 doc_status（content_hash / canonical basename）为权威，`store.list` 仅作补偿，避免 S3 大桶 list 成为热点。

**Q5 中间产物缓存（`*_raw/`）放哪？**
放 store（S3）下的缓存前缀，使任意副本可复用 mineru/docling 的昂贵解析结果。命中判定用 `store.exists`/manifest 比对（沿用现有 `is_bundle_valid`）。可加 `S3_CACHE_RAW=true|false` 让用户在"省解析"与"省存储"间权衡。

**Q6 一致性 / 列举延迟。**
S3 现已读写强一致（含 list-after-write），无需额外处理；对非 AWS 兼容实现（MinIO 等）同样满足。仍坚持"doc_status 权威、list 补偿"以降低对 list 的依赖。

## 7. 兼容性与迁移

- **默认零变化**：`INPUT_STORAGE` 未设或 `local` → `LocalFileStore`，逐字节等价于现状。
- **历史 `file://` URI**：`resolve_sidecar_uri` 继续解析 `file://`；新写入按当前后端生成 `s3://` 或 `file://`。混合存在期间按 scheme 路由，互不影响。
- **迁移工具**：新增 `python -m lightrag.tools.migrate_input_to_s3`，把现有 `INPUT_DIR` 全量上传到 S3，并把 doc_status/full_docs 中的 `file://` `sidecar_location` 改写为 `s3://`（dry-run + 校验）。
- **K8s**：S3 后端下可去掉 inputs 的 PVC（见 `k8s-deploy/`），副本数可 >1。

## 8. 分阶段实施计划

1. **阶段 0 — 抽象层落地（无行为变化）**：新增 `lightrag/filestore/{base,local,factory}.py`；把 `utils_pipeline` 路径函数改为 key 语义；`DocumentManager` 注入 `store`。全程用 `LocalFileStore`，跑通现有测试，**默认行为不变**。
2. **阶段 1 — 改造写/读触点**：上传、扫描、解析读源、`write_sidecar`、`load_lightrag_document_content`、删除、归档全部走 store。仍 local 后端，回归测试。
3. **阶段 2 — S3 后端**：实现 `S3FileStore`（复用 `boto3`），`materialize`/`staging_dir`/`list`/`move`/presigned。补 MinIO 集成测试（mock + `requires_db` 标记）。
4. **阶段 3 — 客户端溯源端点**：`GET /documents/file`（presigned 302 / 代理 FileResponse）；WebUI 用它替换 `file://`，自然兼容并取代 PR #3331 的 `remote_url`。
5. **阶段 4 — 迁移工具 + 文档 + K8s 去 PVC**：迁移脚本、`env.example`、`docs/`（含本文件英文版）、`k8s-deploy` values。

每阶段独立可合并、可回退，默认配置始终是 local。

## 9. 与 PR #3331 的关系

PR #3331 是"图片资源上传 S3"的局部方案；本提案是其**超集**：当 `INPUT_STORAGE=s3` 时，资源本就写在 S3，无需"额外上传一份"，`remote_url` 字段由 `store.public_url` 统一产出。建议：

- 若 #3331 已有用户依赖，可作为 `INPUT_STORAGE=local` 下的"仅资源外发"过渡选项保留一版，并在文档标注被本抽象取代；
- 收敛环境变量命名（`LIGHTRAG_ASSET_UPLOAD_*` → `S3_*`）；
- 修正 `verify=False` 的 TLS 隐患（默认校验，按需 `S3_VERIFY_TLS=false`）。

## 10. 测试策略

- 抽象层契约测试：对 `LocalFileStore` 与 `S3FileStore`（MinIO/mock）跑同一组契约用例（put/get/list/move/delete_prefix/materialize/staging/public_url）。
- 触点回归：上传去重、scan 分类、删除 variants、clear、归档、Sidecar 读写、跨副本溯源。
- 向后兼容：历史 `file://` URI 解析、默认 local 路径逐字节等价。
- 遵循 AGENTS.md：external service 用 mock，集成测试打 `integration`/`requires_db` 标记，并为每个改造点加回归测试。
