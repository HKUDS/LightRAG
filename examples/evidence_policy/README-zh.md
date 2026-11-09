# 按生效日期选择条款版本：功能演示

[English](README.md) | 简体中文

本分支演示：同一条款同时存在原文与修订时，先确定查询日期适用的版本，再补取
解释该版本所需的原文，交给模型回答。核心入口是
`QueryParam(evidence_policy="authority-v1", as_of="YYYY-MM-DD")`。

2026-10-29 已完成真实问答 **5/5**，相关回归 **45 passed**。
见[验证记录与边界](VALIDATION.md)和[已记录的完整上下文及答案](recorded/2026-10-29.json)。
这是限定范围的实验演示；代码通过 Python SDK 在本地运行。

## 安装

推荐使用本次运行所用的 Python 3.11（记录环境为 3.11.9）。在包含本项改动的
源码仓库根目录执行：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e . -r examples/evidence_policy/requirements.txt
```

项目自身的依赖由根目录 `pyproject.toml` 安装；此目录的 `requirements.txt` 补充
OpenAI 兼容客户端和本地嵌入依赖，并固定本次演示使用的版本。
Windows 激活命令为 `.venv\Scripts\activate`。

## 运行五个场景

只需一个 DeepSeek API key，在当前终端设置：

```bash
export LIGHTRAG_LLM_API_KEY="<your-api-key>"
python examples/evidence_policy/run_live.py \
  --workdir temp/evidence-policy-demo \
  --out temp/evidence-policy-demo/record.json
```

默认生成模型为 `deepseek-flash`，端点为 `https://api.deepseek.com`；本地嵌入采用
`BAAI/bge-small-zh-v1.5`，通过 fastembed / ONNX 输出 512 维向量。
首次运行会下载嵌入模型。预检、文档抽取和答案生成会调用模型 API。

完整运行请指定一个新的 `--workdir`：脚本会重建其中的 `main` 子目录，并创建
`unrecalled` 副本。运行存储与新生成的 `record.json` 保留在已忽略的 `temp/` 下。
密钥仅通过环境变量提供，不要将实际值写进源码或提交文件。

脚本依次入库两篇虚构短文，从真实 `text_chunks` 行生成包含 chunk ID、文档 ID、
字符 span 和 SHA-256 的版本目录，再注入 `LightRAG.evidence_catalog`。
五次问答都走 `aquery_llm(mode="naive")`，答案缓存和 thinking 关闭。
控制台输出每个场景的结果；完整上下文、答案、初始召回和补取 ID 写入 `--out`。

| 场景 | 查询日期 | 预期 |
| --- | --- | --- |
| F01：原文与修订均被召回 | 2026-08-01 | 270 天，旧 180 天作为历史 |
| F02：初始只召回旧文 | 2026-08-01 | 从目录补取修订，回答 270 天 |
| F03：历史日期 | 2026-03-01 | 180 天 |
| F04-pre：修订已公布但未生效 | 2026-06-15 | 180 天 |
| F04-post：修订已生效 | 2026-08-01 | 270 天 |

修订于 2026-05-20 公布、2026-07-01 生效。F02 人为删除副本中修订的向量条目，
正文和目录仍保留，用于复现漏召回后的补取；它没有测量自然检索的漏召回率。

## 复用入库数据和配置

已有完整入库数据时，可以只重复五次问答，使用新的输出文件名：

```bash
python examples/evidence_policy/run_live.py \
  --reuse-ingestion \
  --workdir temp/evidence-policy-demo \
  --out temp/evidence-policy-demo/record-next.json
```

`--preflight-only` 检查模型端点与嵌入维度，不入库；默认仍会发起一次 chat 请求。

| 环境变量 | 默认值或用途 |
| --- | --- |
| `LIGHTRAG_LLM_API_KEY` | 必填，生成模型密钥 |
| `LIGHTRAG_LLM_BASE_URL` | `https://api.deepseek.com` |
| `LIGHTRAG_LLM_MODEL` | `deepseek-flash` |
| `LIGHTRAG_EMBED_BACKEND` | `fastembed`；也支持 `openai` |
| `LIGHTRAG_EMBED_MODEL` | 本地默认 `BAAI/bge-small-zh-v1.5`；托管嵌入时必填 |
| `LIGHTRAG_EMBED_BASE_URL` | 托管嵌入端点，未设置时沿用生成端点 |
| `LIGHTRAG_EMBED_API_KEY` | 托管嵌入密钥，未设置时沿用生成密钥 |
| `LIGHTRAG_EMBED_DIM` | 可选；设置后与预检返回的维度核对 |

脚本的 chat 请求带有 DeepSeek 的 `thinking: {"type": "disabled"}` 参数。
更换兼容端点时需确认其支持该参数；本次真实运行仅覆盖上述 DeepSeek 配置。

## 文件与适用边界

| 文件 | 用途 |
| --- | --- |
| `run_live.py` | 入库、生成目录、执行五次真实问答、保存记录 |
| `corpus_3037.py` | 两篇虚构文档及显式版本关系 |
| `smoke_plumbing.py` | 使用本地模型与嵌入桩的接线辅助脚本 |
| `requirements.txt` | 演示附加依赖 |
| `VALIDATION.md` | 现有验证结果与限定范围 |
| `recorded/2026-10-29.json` | 已完成运行的发布副本 |

实现包含 `naive` 与 `mix` 分支，本次真实演示覆盖 `naive`。
版本关系由运营者显式给定；自动发现修订、REST 参数暴露、生产目录管理和全模式接入
均超出本次范围。`resolved` 表示证据处理结果，不能当作模型答案已通过自动语义或引用校验。
其他已知限制见[核心契约](../../docs/design/EvidencePolicyContract.md)。
