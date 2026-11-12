# 探索 LightRAG：从默认配置到生产可用

本指南从 README 安装部分结束的地方接着讲：服务已经用默认 `.env` 跑起来，导入了几篇文档，查询也正常。接下来逐项打开默认配置没有启用的能力：

| 步骤 | 能力 | 带来什么 | 需要什么 |
|---|---|---|---|
| 1 | [PostgreSQL 存储](#第-1-步--postgresql-存储) | 可用于生产环境的存储 | 带 pgvector + Apache AGE 的 PostgreSQL（容器即可） |
| 2 | [账号认证](#第-2-步--账号认证) | WebUI 登录、给客户端用的 API key | 无 |
| 3 | [角色 LLM](#第-3-步--角色-llm) | 回答用更强的模型，抽取用更便宜的模型 | 只需你现有的 LLM 服务 |
| 4 | [Rerank 重排序](#第-4-步--rerank-重排序) | 提高检索精度 | 一个 rerank 接口（兼容 Cohere、Jina 或阿里云格式） |
| 5 | [图片分析（VLM）](#第-5-步--用-vlm-分析图片) | Markdown / Word 文档里的图片也能被检索 | 支持图片输入的模型 |
| 6 | [Word 智能标题](#第-6-步--word-文档智能标题识别) | 没用标题样式的 `.docx` 也能得到正确的章节结构 | 钉定版本的 spaCy 模型 |
| 7 | [MinerU](#第-7-步--用-mineru-解析-pdf-和-office-文件) | 高质量解析 PDF、PPTX、XLSX 和扫描件 | MinerU API token，或带 GPU 的自部署 MinerU |
| 8 | [并发调优](#第-8-步--并发调优) | 导入速度提高数倍 | 无 |

下面每一步都按这个顺序在一台真实部署上做过：2 vCPU / 4 GB、没有 GPU 的云主机，OpenAI 模型（抽取用 `gpt-6-luna`，回答用 `gpt-6-sol`，嵌入用 `text-embedding-3-large`），测试语料是 7 篇文档（Markdown、一个 `.textpack`、一篇 PDF 和一份中文 `.docx`）。文中的耗时数据都来自这台机器。

## 开始之前

**为什么是这个顺序。** 存储放在最前面。图片分析、智能标题和 MinerU 都在上传时起作用，而且只对打开之后上传的文档生效，所以用来测试它们的文档最好直接落在最终要用的存储里。认证排第二，因为配置成本很低，而且在别人能访问服务之前就应该开好。并发调优放在最后，等你能测出一次真实的导入耗时再调。

**每一步的固定流程：**

1. 修改 `.env`。
2. 重启 `lightrag-server`。`.env` 只在启动时读取。
3. 看启动日志。日志会打印存储后端、每个 LLM 角色的模型和参数，以及被拒绝的配置。
4. 按各步给出的方法，用一篇文档或一次查询验证效果。

**上传时就固定下来的设置。** 文档的解析引擎、处理选项（`i`、`t`、`e`、分块方式）和分块参数，都在入队时记录下来。之后再改 `.env`，不会影响已经入库的文档。要让新设置作用到某篇文档，需要删除后重新上传。重新处理一篇内容没变的文档代价很低：抽取缓存（`ENABLE_LLM_CACHE_FOR_EXTRACT`，默认开启）会直接返回之前做过的 LLM 调用结果。

**准备一份小的测试语料**，覆盖你关心的格式：一篇带图片的 Markdown、一份 Word、一篇 PDF。每做完一步重新上传一次，效果一目了然。

## 第 1 步 — PostgreSQL 存储

默认的四种存储（`JsonKVStorage`、`NanoVectorDBStorage`、`NetworkXStorage`、`JsonDocStatusStorage`）把所有数据放在内存里，靠整文件重写来落盘，只适合小规模测试。PostgreSQL 可以在一个数据库里承担全部四种存储：键值和文档状态用普通表，向量用 pgvector，知识图谱用 Apache AGE。

### 1.1 用设置向导生成 PostgreSQL 服务

```bash
make env-storage
```

四种存储分别选择 `PGKVStorage`、`PGVectorStorage`、`PGGraphStorage` 和 `PGDocStatusStorage`，选择由向导通过 Docker 在本地运行 PostgreSQL，再填写用户名、密码和数据库名。向导会：

- 先备份原来的 `.env`，再把存储类型和 `POSTGRES_*` 连接参数写进 `.env`；
- 在 `docker-compose.final.yml` 里加入一个 `postgres` 服务。它使用的镜像 `gzdaniel/postgres-for-rag:pg18-age-pgvector` 内置了 PostgreSQL 18、pgvector 和 Apache AGE。

`.env` 里保持主机上可用的地址（`POSTGRES_HOST=localhost`）；生成的 compose 文件则让其中容器化的 `lightrag` 服务改为连接 `postgres`。向导会问哪些问题，参见 [InteractiveSetup.md](./InteractiveSetup.md)。

启动数据库：

```bash
# LightRAG Server 直接在主机上运行（源码或 PyPI 安装）：只启动数据库
docker compose -f docker-compose.final.yml up -d postgres

# LightRAG Server 也在 Docker 中运行：启动整个服务栈
docker compose -f docker-compose.final.yml up -d
```

确认两个扩展都可用：

```bash
docker compose -f docker-compose.final.yml exec postgres psql -U <用户名> -d <数据库名> -tAc \
  "select name, default_version from pg_available_extensions where name in ('vector','age')"
```

服务模板会在所有网卡上开放 5432 端口。如果主机能从公网访问，请使用强密码，并用防火墙限制这个端口。

### 1.2 检查向量索引类型

向导不会询问 `POSTGRES_VECTOR_INDEX_TYPE`，请在 `.env` 里确认：

```bash
POSTGRES_VECTOR_INDEX_TYPE=HNSW_HALFVEC
```

**根据嵌入维度选择向量索引类型。** pgvector 无法为超过 2000 维的向量建立 `HNSW` 或 `IVFFlat` 索引；`HNSW_HALFVEC` 最多支持 4000 维，需要 pgvector 0.7 或更高版本，向导使用的镜像满足这个要求。默认的 `text-embedding-3-large` 是 3072 维，所以必须用 `HNSW_HALFVEC`。如果用 `HNSW`，索引会建不出来：没有维度检查的版本只打一条 `ERROR` 就继续运行，此时没有向量索引，每次查询都要全表扫描；带维度检查的版本会拒绝启动，并提示修改方法。请在首次上传前定好：已有向量的表切换到 `HNSW_HALFVEC` 时，存储的向量会被原地转成 16 位浮点数，这个精度损失无法恢复。

重启服务后，启动日志里会列出四个 `PG*` 存储。数据表在第一次启动时创建，向量表的表名里带着嵌入模型和维度，例如 `lightrag_vdb_chunks_text_embedding_3_large_3072d`。

### 1.3 迁移已有文档

切换存储不会搬运数据。把已经导入的内容带过来，最省钱的办法是先迁移抽取缓存，再把文档重新上传一遍：

1. PostgreSQL 启动后（1.1），在重启服务之前运行 `python -m lightrag.tools.migrate_llm_cache`。源选 `JsonKVStorage`，目标选 `PGKVStorage`，然后确认。它会迁移抽取、摘要和多模态分析的缓存（参见 [README_MIGRATE_LLM_CACHE.md](../lightrag/tools/README_MIGRATE_LLM_CACHE.md)）。
2. 重启服务，切换到 PostgreSQL。
3. 重新上传原始文件。实体抽取会命中迁移过来的缓存。

在测试部署上，迁移了 139 条缓存记录。重新上传 6 篇文档用了 60 秒，只新增了 17 次 LLM 调用，其余抽取都来自缓存。新存储里的文本块和原来完全一致，都是 56 个。

## 第 2 步 — 账号认证

`AUTH_ACCOUNTS` 和 `LIGHTRAG_API_KEY` 都没设置时，所有接口都是公开的。建议两个都配：账号给使用 WebUI 的人，API key 给脚本和 API 客户端。

```bash
AUTH_ACCOUNTS='admin:{bcrypt}$2b$12$...'   # 用单引号：哈希里有 $
TOKEN_SECRET=<openssl rand -hex 32>         # 配了 AUTH_ACCOUNTS 就必须设置
LIGHTRAG_API_KEY=<一串足够长的随机字符>
WHITELIST_PATHS=/health                     # Ollama 兼容的 /api/* 也要求认证
```

密码建议存 bcrypt 哈希，不要存明文。用环境里已经装好的 `bcrypt` 包生成：

```bash
python -c "import bcrypt, getpass; print(bcrypt.hashpw(getpass.getpass().encode(), bcrypt.gensalt(12)).decode())"
```

配置了 `AUTH_ACCOUNTS` 却没有设置非默认的 `TOKEN_SECRET`，服务会拒绝启动。`WHITELIST_PATHS` 的默认值让 `/api/*` 保持开放，这样 Ollama 客户端不用改就能用；服务一旦能从网络访问，就应该收窄成 `/health`。

**验证：**

```bash
curl -s -o /dev/null -w '%{http_code}\n' http://HOST:9621/documents/status_counts               # 401
curl -s -o /dev/null -w '%{http_code}\n' http://HOST:9621/api/tags                              # 401
curl -s -o /dev/null -w '%{http_code}\n' http://HOST:9621/health                                # 200
curl -s -o /dev/null -w '%{http_code}\n' -H "X-API-Key: $KEY" http://HOST:9621/documents/status_counts  # 200
curl -s -X POST http://HOST:9621/login -d 'username=admin&password=...'                         # 返回 access_token
```

## 第 3 步 — 角色 LLM

LightRAG 会以四种角色调用 LLM：`EXTRACT`（从每个文本块抽取实体和关系）、`KEYWORD`（提取查询关键词）、`QUERY`（生成最终回答）和 `VLM`（分析图片）。它们默认都用 `LLM_MODEL`。抽取对每个文本块都要调用一次，应该快而便宜；回答要从又长又杂的上下文里组织答案，用更强的模型效果更好。

```bash
# 基础模型：EXTRACT、KEYWORD、VLM 使用
LLM_MODEL=gpt-6-luna
OPENAI_LLM_REASONING_EFFORT=low

# 回答：高一档的模型，更多推理
QUERY_LLM_MODEL=gpt-6-sol
QUERY_OPENAI_LLM_REASONING_EFFORT=medium

# 关键词：对延迟敏感，不推理
KEYWORD_OPENAI_LLM_REASONING_EFFORT=none
```

每个角色都继承基础配置的绑定类型、服务地址、API key 和参数，除非用角色前缀覆盖（`QUERY_LLM_BINDING_HOST`、`QUERY_LLM_BINDING_API_KEY` 等），所以某个角色也可以指向另一家服务商。服务商参数同样可以加角色前缀：`<ROLE>_OPENAI_LLM_REASONING_EFFORT`。详见 [RoleSpecificLLMConfiguration-zh.md](./RoleSpecificLLMConfiguration-zh.md)。

**验证**：启动日志会逐个打印每个角色：

```
Role LLM Configuration (initialized):
 - extract: openai/gpt-6-luna, host=https://api.openai.com/v1, max_async=4, timeout=240
 - keyword: openai/gpt-6-luna, ...
 - query: openai/gpt-6-sol, ...
 - vlm: openai/gpt-6-luna, ...
Role LLM Option:
 - extract: OpenAI {'reasoning_effort': 'low'}
 - keyword: OpenAI {'reasoning_effort': 'none'}
 - query: OpenAI {'reasoning_effort': 'medium'}
```

**顺便考虑 `SUMMARY_LANGUAGE`。** 它决定抽取出的实体和关系描述用什么语言，默认 `English`。请在导入之前把它设成文档的主要语言，例如 `Chinese`。之后再改会改变抽取提示词，改动之后重新处理的文档都无法命中缓存。

## 第 4 步 — Rerank 重排序

向量检索返回的是和问题*相似*的文本块，但相似不等于能*回答*问题。Rerank 模型会把每个候选块和问题逐一打分，只保留最相关的几个。它在 `mix` 和 `naive` 模式下作用最大，因为这两种模式的答案主要来自文本块。

```bash
RERANK_BINDING=cohere                          # cohere、jina 或 aliyun
RERANK_MODEL=BAAI/bge-reranker-v2-m3
RERANK_BINDING_HOST=https://your-host/rerank   # 完整的接口地址
RERANK_BINDING_API_KEY=...
# MIN_RERANK_SCORE=0.0       # 低于这个分数的文本块会被丢弃
# RERANK_BY_DEFAULT=True     # 除非查询设置 enable_rerank=false，否则默认重排序
```

绑定类型按接口格式选，不按厂商选：用 vLLM 部署的模型，或任何提供 Cohere 风格 `/rerank` 接口的服务，都用 `cohere`。OpenAI 没有 rerank 接口。配置前先测一下接口：

```bash
curl -s https://your-host/rerank -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model":"BAAI/bge-reranker-v2-m3","query":"什么是知识图谱","documents":["知识图谱是用图结构表示实体及其关系的知识库。","今天天气很好。"]}'
```

启动日志里出现 `Reranking is enabled: BAAI/bge-reranker-v2-m3 using cohere provider` 即为生效。`MAX_ASYNC_RERANK` 未设置时沿用 `MAX_ASYNC_LLM`，`RERANK_TIMEOUT` 默认 30 秒。

**验证**：通过 `POST /query/data` 把同一个问题分别在 `enable_rerank` 关闭和开启时各查一次。这个接口只返回检索到的上下文，不生成回答。在测试语料上，问题"LazyGraphRAG 的索引成本和 GraphRAG 相比如何？"（`mix` 模式，`chunk_top_k=5`）的结果：

| | 前 5 个文本块中来自 LazyGraphRAG 文档的数量 | 耗时 |
|---|---|---|
| 不开 rerank | 0 个（全是泛泛讲 GraphRAG 和 LightRAG 的段落） | 4.3 秒 |
| 开启 rerank | 4 个 | 4.7 秒 |

日志 `Successfully reranked: 5 chunks from 34 original chunks` 说明 rerank 从候选块中做了挑选。

## 第 5 步 — 用 VLM 分析图片

内置的 `native` 解析器能从 Markdown、`.textpack` 和 Word 文件中提取表格、公式和图片。表格（`t`）和公式（`e`）由抽取用的 LLM 分析，默认规则里已经开启；图片（`i`）需要视觉模型，默认关闭。

```bash
LIGHTRAG_PARSER=*:native-iteP,*:legacy-R   # 在 native 规则里加上 i
VLM_PROCESS_ENABLE=true
# VLM_LLM_MODEL=...                        # 可选：默认沿用 LLM_MODEL
```

`VLM` 角色默认使用 `LLM_MODEL`。`gpt-6-luna` 支持图片输入，所以这里不用再配别的。如果基础模型只支持文本，就把 `VLM_LLM_MODEL`（必要时还有 `VLM_LLM_BINDING_HOST` / `VLM_LLM_BINDING_API_KEY`）设成一个视觉模型。

> [!IMPORTANT]
> `VLM_PROCESS_ENABLE` 必须和 `i` 选项一起打开。如果文档带着 `i` 选项处理，而这个开关是关的，文档会在遇到第一张有效图片时**直接失败**（"VLM analysis required but VLM role is not available"），而不是跳过这张图。

**图片从哪里来：**

| 来源 | 能否处理 |
|---|---|
| Word `.docx` 内嵌图片 | 能 |
| Markdown 里的 base64 `data:` 图片 | 能 |
| Markdown 里的远程 `http(s)` 图片 | 会下载（禁止访问内网地址）；下载失败时保留链接并记一条警告 |
| Markdown 里的**本地**图片，如 `./doc.assets/a.png` | **只有打包成 `.textpack` 才行**，即把 `.md` 和图片一起打成 zip。单独上传 `.md` 时图片文件不会一起传上去，解析器会记录 `skipping image (file reference outside .textpack)` |

本地图片路径里有空格时，Typora 等编辑器会直接写成 `![alt](./my doc.assets/a.png)`，不加尖括号。这不是标准 Markdown，但只要路径以图片扩展名（`.png`、`.jpg`、`.jpeg`、`.gif`、`.webp`、`.svg`）结尾，LightRAG 仍会把它识别为图片。其他情况请用标准写法 `![alt](<./my doc.assets/a.png>)`，或把空格写成 `%20`；否则它不算图片，会被当作普通文本保留，也不会有任何警告。

宽或高小于 `VLM_MIN_IMAGE_PIXEL`（64 像素）的图片会被当作装饰图标跳过。

**验证**：重新上传一篇带图片的文档。解析结果的日志会统计识别出的内容，随后是分析阶段的日志：

```
[sidecar] wrote 29 blocks for doc_id=... (6 tables, 6 drawings, 2 equations, assets=True, engine=native)
[analyze_multimodal] drawings=6, tables=6, equations=2 surrounding backfilled: doc-...
```

图片描述会和 `<drawing .../>` 标签一起写进文本块，参与实体抽取和检索。

## 第 6 步 — Word 文档智能标题识别

很多 Word 文档，尤其是公文，标题是靠加粗、加大字号和"一、""（一）"这样的编号排出来的，没有用 Word 的标题样式。内置解析器从 Word 大纲读取标题，所以这类文档解析出来是一整块，段落分块（`P`）也就没有结构可依。`smart_heading` 会根据版式、编号和语言特征识别标题，并请抽取用的 LLM 判断标题块。

```bash
DOCX_SMART_HEADING=true
```

它需要钉定版本的 spaCy 模型：`lightrag-download-cache --spacy-install`（参见 README）。打开开关后，服务会在启动时检查这些模型，缺少时拒绝启动。只想对单个文件启用，可以用文件名提示上传：`report.[native(smart_heading)].docx`。

> [!NOTE]
> **短文档会被跳过。** 正文短于 `DOCX_SMART_MIN_TOKENS`（默认 1800）的文档仍然只用 Word 大纲，因为大约能装进一个文本块的文档本来就不需要按标题切分。所以用一份很短的测试文档，看起来就像开关没起作用一样。请用篇幅接近真实的文档测试，或者对短小的通知类文档调低 `DOCX_SMART_MIN_TOKENS`。

**验证**：识别过程记录在 `INPUT_DIR/__parsed__/` 下该文档 `.parsed/` 目录里的 `<文件名>.smart_audit.json`；被跳过时，这里会记为 `smart_skipped_short_document`。文档状态里的 `parse_engine` 会显示为 `native(smart_heading=true)`。

测试文档有 53 个段落、约 2400 个 token，标题都是加粗的正文段落。普通解析只得到一个名为 `Preface/Uncategorized` 的块；打开 `smart_heading` 后，识别出了文档标题（第 0 级）、4 个"一、"标题（第 1 级）和 15 个"（一）"标题（第 2 级），只调用了 1 次 LLM，文档按章节切成了 4 个文本块。

## 第 7 步 — 用 MinerU 解析 PDF 和 Office 文件

`native` 解析器支持 Markdown、`.textpack` 和 Word。其他格式（PDF、PPTX、XLSX、图片）都会落到 `legacy`，它只提取纯文本，没有表格、图片和版面信息。[MinerU](https://github.com/opendatalab/MinerU) 能做版面分析、OCR、表格和公式识别，输出和 native 解析器相同的结构化结果，所以 `i` / `t` / `e` 分析和 `P` 分块对它的输出同样有效。

MinerU 有两种用法：

- **`official`**：mineru.net 提供的精准解析云 API。在网站的 API 管理页面创建 API token 即可。文件会上传到 MinerU 的云端，服务对单个文件有页数和大小限制（撰写本文时为 200 MB、200 页），并有每日额度。
- **`local`**：自己部署 `mineru-api`，需要 GPU，参见 [ParserServiceDeployment-zh.md](./ParserServiceDeployment-zh.md)。测试机没有 GPU，所以本指南用 `official`。

```bash
MINERU_API_MODE=official
MINERU_API_TOKEN=sk-...
LIGHTRAG_PARSER=*:native-iteP,*:mineru-iteP,*:legacy-R
# MAX_PARALLEL_PARSE_MINERU=1     # 同时交给 MinerU 解析的文件数
```

规则对每个文件从左到右依次匹配：`native` 支持的格式交给 `native`，MinerU 支持的格式交给 `mineru`，剩下的交给 `legacy`。`LIGHTRAG_PARSER` 里写了 `mineru` 时，请在启动服务之前配好它的模式和 token（或本地服务地址）。

**先确认 token，再怀疑 LightRAG。** 如果解析失败，报错 `MinerU official upload URL request failed: HTTP 401 ... "msgCode": "A0202"`，说明 MinerU 拒绝了这个 token（MinerU 错误码表中 A0202 为"Token 错误"，A0211 为"Token 过期"）。用 `.env` 里的值直接调用一次接口：

```bash
curl -s -X POST https://mineru.net/api/v4/file-urls/batch \
  -H "Authorization: Bearer $(grep ^MINERU_API_TOKEN= .env | cut -d= -f2-)" \
  -H 'Content-Type: application/json' -d '{"files":[{"name":"probe.pdf"}],"model_version":"vlm"}'
```

返回 `"code":0` 说明 token 可用。在测试部署上，第一次就是因为 A0202 失败的：`.env` 里填的是另一个更短的字符串，而不是网站上显示的 `sk-...` key。

**重试失败的文档。** 解析失败的文档会停在 `FAILED` 状态，不会自动重试。排除原因后，调用 `POST /documents/reprocess_failed`，每篇失败的文档会重试一次。重试沿用文档原来的解析引擎和处理选项。

**验证**：日志会显示使用的引擎和 MinerU 识别出的内容：

```
[File Extraction]Deferred HybridRAG ... .pdf to mineru parser
[sidecar] wrote 19 blocks for doc_id=... (5 tables, 2 drawings, 0 equations, assets=True, engine=mineru)
```

测试用的 PDF 是一篇研究论文。MinerU 解析出 19 个结构块，其中有 5 个表格、2 张图，都经过了抽取 LLM 和 VLM 的分析，整篇文档约 100 秒处理完。同一个文件之前走 `legacy` 时只有纯文本。

## 第 8 步 — 并发调优

默认的并发参数比较保守。导入的大部分时间都在等待 LLM 和嵌入接口返回，所以提高并发对本机资源的消耗很小，上限通常取决于服务商的限流。

| 参数 | 控制什么 | 默认值 |
|---|---|---|
| `MAX_ASYNC_LLM` | 每个角色的 LLM 并发请求数，以及单篇文档内并行抽取的文本块数 | 4 |
| `MAX_PARALLEL_INSERT` | 同时处理的文档数，建议约为 `MAX_ASYNC_LLM / 3` | 3 |
| `EMBEDDING_FUNC_MAX_ASYNC` | 嵌入接口的并发请求数 | 8 |
| `VLM_MAX_ASYNC_LLM` | 图片分析的并发请求数 | 4 |
| `POSTGRES_MAX_CONNECTIONS` | PostgreSQL 连接池大小，要跟着其他参数一起调大 | 25 |

在测试部署上的实测方法：删除所有文档后重新上传 7 篇测试语料，测试期间设置 `ENABLE_LLM_CACHE_FOR_EXTRACT=false`，保证每次 LLM 调用都是真实请求：

| 方案 | LLM / 文档并行 / 嵌入 / VLM / PG 连接池 | 总耗时 | 服务进程内存峰值 |
|---|---|---|---|
| 默认 | 4 / 3 / 8 / 4 / 25 | 449 秒 | 839 MB |
| 调优 | 12 / 4 / 16 / 8 / 40 | 170 秒 | 866 MB |
| 最终 | 24 / 7 / 24 / 12 / 40 | 120 秒 | 876 MB |

全程没有出现限流错误，内存几乎没变。最后一轮里 7 篇文档已经全部并行，剩下的时间取决于最大的那一篇；语料更多时，调大 `MAX_PARALLEL_INSERT` 还会继续有收益。最终保留的配置：

```bash
MAX_ASYNC_LLM=24
MAX_PARALLEL_INSERT=8
EMBEDDING_FUNC_MAX_ASYNC=24
VLM_MAX_ASYNC_LLM=12
POSTGRES_MAX_CONNECTIONS=40
```

请逐步调大，同时留意日志里有没有 `429` / `RateLimitError`。测试完记得把抽取缓存重新打开。

## 最终效果

8 步全部完成后，把 7 篇测试语料完整重新导入一遍，用时 72 秒，没有错误（此时抽取缓存和 MinerU 的解析结果都已命中缓存）。每篇文档都走了预期的解析引擎：

| 文档 | 引擎 | 处理选项 | 文本块数 |
|---|---|---|---|
| HybridRAG 论文（`.pdf`） | `mineru` | `iteP` | 16 |
| LightRAG 论文（`.textpack`） | `native` | `iteP` | 30 |
| 4 篇 Markdown 文章 | `native` | `iteP` | 1–11 |
| 中文实施方案（`.docx`） | `native(smart_heading=true)` | `iteP` | 4 |

问一个要靠 Word 文档回答的问题："企业知识图谱建设方案分哪几个实施阶段？每个阶段做什么？"，9.5 秒返回。答案把进度表格（每个阶段的月份）和章节正文（每个阶段做什么）结合在一起，并引用了这份 `.docx`。

相对 `env.example` 修改过的配置汇总：

```bash
# 第 1 步 — PostgreSQL
LIGHTRAG_KV_STORAGE=PGKVStorage
LIGHTRAG_DOC_STATUS_STORAGE=PGDocStatusStorage
LIGHTRAG_GRAPH_STORAGE=PGGraphStorage
LIGHTRAG_VECTOR_STORAGE=PGVectorStorage
POSTGRES_HOST=localhost
POSTGRES_USER=lightrag
POSTGRES_PASSWORD='...'
POSTGRES_VECTOR_INDEX_TYPE=HNSW_HALFVEC
POSTGRES_MAX_CONNECTIONS=40

# 第 2 步 — 账号认证
AUTH_ACCOUNTS='admin:{bcrypt}$2b$12$...'
TOKEN_SECRET=...
LIGHTRAG_API_KEY=...
WHITELIST_PATHS=/health

# 第 3 步 — 角色 LLM
LLM_MODEL=gpt-6-luna
OPENAI_LLM_REASONING_EFFORT=low
QUERY_LLM_MODEL=gpt-6-sol
QUERY_OPENAI_LLM_REASONING_EFFORT=medium
KEYWORD_OPENAI_LLM_REASONING_EFFORT=none

# 第 4 步 — Rerank
RERANK_BINDING=cohere
RERANK_MODEL=BAAI/bge-reranker-v2-m3
RERANK_BINDING_HOST=https://your-host/rerank
RERANK_BINDING_API_KEY=...

# 第 5、7 步 — 解析路由、VLM、MinerU
LIGHTRAG_PARSER=*:native-iteP,*:mineru-iteP,*:legacy-R
VLM_PROCESS_ENABLE=true
MINERU_API_MODE=official
MINERU_API_TOKEN=sk-...

# 第 6 步 — Word 智能标题
DOCX_SMART_HEADING=true

# 第 8 步 — 并发
MAX_ASYNC_LLM=24
MAX_PARALLEL_INSERT=8
EMBEDDING_FUNC_MAX_ASYNC=24
VLM_MAX_ASYNC_LLM=12
```

## 过程中遇到的坑

| 现象 | 原因 | 解决办法 |
|---|---|---|
| 切换到 PostgreSQL 后向量索引建不出来，或服务拒绝启动 | 嵌入超过 2000 维却用了 `HNSW` / `IVFFlat` | `POSTGRES_VECTOR_INDEX_TYPE=HNSW_HALFVEC` |
| Markdown 里的图片没有被分析 | 单独上传 `.md` 时本地图片不会一起上传；或路径含空格、又不以图片扩展名结尾，却没加 `<...>` | 改传 `.textpack`；写成 `![](<含空格的路径>)` |
| 带图片的文档处理失败（`FAILED`） | 处理选项带 `i`，但 `VLM_PROCESS_ENABLE=false` | 两者一起打开 |
| `DOCX_SMART_HEADING=true` 没有效果 | 文档正文短于 `DOCX_SMART_MIN_TOKENS` | 查看 `smart_audit.json`；用更长的文档测试 |
| MinerU 解析失败，HTTP 401 `A0202` | MinerU 拒绝了这个 token | 用第 7 步的 `curl` 测试 token，再调用 `POST /documents/reprocess_failed` |
| 失败的文档一直不会自己恢复 | `FAILED` 状态的文档只有在请求时才会重试 | `POST /documents/reprocess_failed` |
| 新配置对旧文档不生效 | 解析引擎和处理选项在上传时就已固定 | 删除后重新上传；有抽取缓存，代价很低 |
