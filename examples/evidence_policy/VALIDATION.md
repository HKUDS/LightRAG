# 已完成的验证

记录日期：2026-10-29（Asia/Shanghai）。本次整理上传复用这些结果，没有重新运行模型或测试。

## 真实问答：5/5

环境：Python 3.11.9、OpenAI 客户端 2.54.0、fastembed 0.9.1、onnxruntime 1.31.1。
生成模型为 DeepSeek `deepseek-flash`，本地嵌入为 `BAAI/bge-small-zh-v1.5`（512 维）。
文档经 `ainsert` 入库，目录从真实 chunk 行生成，问答经 `aquery_llm(mode="naive")` 执行。

| 场景 | 日期 | 初始版本 | 选中版本 | 已记录答案 |
| --- | --- | --- | --- | --- |
| F01 | 2026-08-01 | V-old、V-new | V-new | 270 天；180 天标为历史 |
| F02 | 2026-08-01 | V-old | 补取 V-new，选 V-new | 270 天 |
| F03 | 2026-03-01 | V-old、V-new | V-old | 180 天 |
| F04-pre | 2026-06-15 | V-old、V-new | V-old | 180 天 |
| F04-post | 2026-08-01 | V-old、V-new | V-new | 270 天 |

[完整记录](recorded/2026-10-29.json)包含初始 chunk、补取 chunk、模型实际上下文、答案、
检查项和诊断。发布副本仅将工作机绝对目录替换为 `<local-workdir>`，并更新演示文档指针；
五个场景内容及汇总保持原样，源记录 SHA-256 保存在 `publication_note`。

最终轮复用已有入库，成功完成五次 chat，进程退出码 0。自动检查包含版本选择、
预期数值、模型调用和 F02 补取关系；完整答案另经人工阅读核对。
这些检查不构成通用答案语义校验或独立法律审核。

首次运行发现目录注入导致 `asdict` 将 dataclass scope 键展开为 dict，以及模型上下文
缺少查询日期。现有代码已修复这两项：在浅副本上生成配置并保留同一目录对象，
向上下文加入 `requested_as_of`。表中结果来自修复后的最终轮。

## 相关回归：45 passed

既有命令针对 `tests/evidence_policy/`，包含构造与目录注入、查询接入、上下文日期、
缓存区分和策略运行时检查：

```bash
PYTHONDONTWRITEBYTECODE=1 ./scripts/test.sh tests/evidence_policy \
  -q -p no:cacheprovider --capture=sys \
  --basetemp=/private/tmp/3037-live-date-regression-20261029
```

结果为 **45 passed**。需要在新环境运行这些测试时，先安装项目 `pytest` extra：
`python -m pip install -e ".[pytest]"`。

## 结论的适用范围

这轮结果支持：两篇虚构文档、一个明确版本链、`naive` 检索路径中，能够按生效日期
选择原文或修订，并在初始只召回旧文时补取修订。

F02 的漏召回通过删除向量条目构造；真实正文仍在 KV 中。没有据此声称自然漏召回率、
大规模法规质量、其他模型或存储后端的效果、`mix` 真实运行结果，或上游合并状态。
现有 `authority-v1` 的其他已知缺口保留在[核心契约](../../docs/design/EvidencePolicyContract.md)中。
