# RAGAS 评估框架详解

## 什么是 RAGAS？

**RAGAS**（Retrieval-Augmented Generation Assessment）是一个专门用于评估 **RAG 系统质量**的开源框架。

### 核心概念

```
传统软件测试：
  输入 → 程序 → 输出
  ✅ 确定性：相同输入 → 相同输出
  ✅ 容易测试：断言输出 == 期望值

RAG 系统测试：
  问题 → [检索 + LLM生成] → 答案
  ❌ 非确定性：相同问题 → 不同答案
  ❌ 难以测试：答案正确性难以量化

RAGAS 的作用：
  将"答案质量"量化为 0-1 分数
  可以自动化、可比较、可追踪
```

### 为什么需要 RAGAS？

**问题场景**：你改进了 RAG 系统（如：换了 embedding 模型、调整了 chunk 大小、优化了提示词），如何知道是变好了还是变差了？

**传统方法**：
- 👎 人工阅读：耗时、主观、不可扩展
- 👎 用户反馈：滞后、不全面

**RAGAS 方法**：
- ✅ 自动评估：几分钟评估 100+ 问题
- ✅ 客观量化：0.85 vs 0.78（明确好坏）
- ✅ 可追踪：每次改动都可对比

---

## RAGAS 的四大核心指标

RAGAS 从四个维度评估 RAG 系统：

```
                    RAG 系统
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
    【检索阶段】    【生成阶段】   【整体质量】
        │              │              │
    ┌───┴───┐      ┌───┴───┐      ┌───┴───┐
    │Context│      │Context│      │Answer │
    │Recall │      │Precis.│      │Relev. │
    └───────┘      └───────┘      └───────┘
                                       │
                               ┌───────┴────────┐
                               │  Faithfulness  │
                               └────────────────┘
```

### 1. Context Precision（上下文准确性）

**测量**：检索的上下文中，有多少是**真正相关**的？

**问题**：检索了 10 个文档，但只有 3 个真正相关，其他 7 个是噪音

**计算逻辑**：
```python
# 伪代码
相关文档排名靠前 → 高分
无关文档混在其中 → 低分

# 例子
检索结果：[相关, 相关, 无关, 相关, 无关, ...]
Context Precision ≈ 0.75  # 前面的相关文档多
```

**实际评估方式**：
RAGAS 使用 LLM 判断每个检索片段是否与问题相关：

```
LLM Prompt:
---
Question: "What is the capital of France?"
Retrieved Context 1: "Paris is the capital and largest city of France..."
Retrieved Context 2: "The Eiffel Tower was built in 1889..."

Is Context 1 relevant to answering the question? → Yes
Is Context 2 relevant to answering the question? → No
```

**分数解读**：
- **0.9-1.0**：检索非常精准，几乎没有噪音
- **0.7-0.9**：大部分相关，少量噪音
- **0.5-0.7**：相关和无关混杂
- **< 0.5**：检索质量差，需要优化

**优化方向**：
- 提升 embedding 模型质量
- 调整 top_k 参数（减少检索数量）
- 改进 reranking 策略

---

### 2. Context Recall（上下文召回）

**测量**：黄金标准答案中的信息，有多少出现在检索的上下文中？

**问题**：检索遗漏了关键信息，导致答案不完整

**计算逻辑**：
```python
# 黄金标准答案包含 5 个事实
ground_truth = [
    "巴黎是法国首都",
    "位于塞纳河畔",
    "人口约 220 万",
    "是欧洲最大城市之一",
    "有埃菲尔铁塔等著名地标"
]

# 检索的上下文包含其中 4 个
retrieved_contexts_contain = [
    "巴黎是法国首都",  # ✓
    "位于塞纳河畔",    # ✓
    "人口约 220 万",   # ✓
    # 缺少"欧洲最大城市"
    "有埃菲尔铁塔"     # ✓
]

Context Recall = 4 / 5 = 0.80
```

**实际评估方式**：
RAGAS 将黄金答案拆分为多个"句子/陈述"，然后用 LLM 判断每个陈述是否能从检索的上下文中推导出来：

```
LLM Prompt:
---
Statement from ground truth: "Paris is the capital of France"
Retrieved Contexts: [context1, context2, ...]

Can this statement be inferred from the contexts? → Yes/No
```

**分数解读**：
- **0.9-1.0**：检索到所有关键信息
- **0.7-0.9**：漏掉少量次要信息
- **0.5-0.7**：漏掉一些关键信息
- **< 0.5**：检索不足，严重遗漏

**优化方向**：
- 增加 top_k（检索更多文档）
- 改进 chunking 策略（避免关键信息被切分）
- 优化查询改写（query rewriting）

---

### 3. Faithfulness（忠实度）

**测量**：生成的答案是否**基于检索到的上下文**？还是 LLM 在"幻觉"？

**问题**：LLM 生成了听起来合理但实际错误的信息

**计算逻辑**：
```python
# 将答案拆分为多个陈述
answer = "巴黎是法国首都，位于塞纳河畔，人口约 500 万。"

statements = [
    "巴黎是法国首都",      # ✓ 上下文支持
    "位于塞纳河畔",        # ✓ 上下文支持
    "人口约 500 万"        # ✗ 上下文说是 220 万（幻觉！）
]

Faithfulness = 2 / 3 = 0.67  # 有幻觉，分数低
```

**实际评估方式**：
```
LLM Prompt:
---
Context: "Paris is the capital of France, located on the Seine River,
          with a population of about 2.2 million."

Answer Statement: "Paris has a population of about 5 million."

Can this statement be verified from the context? → No (Hallucination)
```

**分数解读**：
- **0.9-1.0**：答案完全基于事实，无幻觉
- **0.7-0.9**：少量不可验证的陈述
- **0.5-0.7**：有明显幻觉
- **< 0.5**：严重幻觉，不可信

**优化方向**：
- 调整 LLM prompt（强调"仅基于上下文回答"）
- 使用更可靠的 LLM 模型
- 减少生成的 temperature（降低随机性）

---

### 4. Answer Relevancy（答案相关性）

**测量**：答案是否真正回答了用户的问题？

**问题**：答案虽然基于事实，但答非所问

**计算逻辑**：
```python
Question: "What is the capital of France?"

Answer A: "Paris is the capital of France."
→ Answer Relevancy = 1.0  # 完美回答

Answer B: "France is a country in Europe. It has many beautiful cities
           including Paris, Lyon, Marseille..."
→ Answer Relevancy = 0.65  # 包含答案但啰嗦

Answer C: "France has a population of 67 million people."
→ Answer Relevancy = 0.2   # 完全跑题
```

**实际评估方式**：
RAGAS 使用"反向问题生成"：

```
Step 1: 根据答案生成问题
Answer: "Paris is the capital of France"
Generated Question: "What is the capital of France?"

Step 2: 计算原问题和生成问题的相似度
Original Question: "What is the capital of France?"
Generated Question: "What is the capital of France?"
Similarity: 1.0 → High Relevancy

如果答案答非所问：
Answer: "France has 67 million people"
Generated Question: "What is the population of France?"
Similarity to Original: 0.3 → Low Relevancy
```

**分数解读**：
- **0.9-1.0**：直接回答问题，无冗余
- **0.7-0.9**：回答了问题但有冗余信息
- **0.5-0.7**：部分回答或答非所问
- **< 0.5**：完全不相关

**优化方向**：
- 改进 prompt（要求简洁直接的答案）
- 调整生成长度限制
- 使用更好的指令跟随模型

---

## RAGAS 如何工作？

### 工作流程

```
┌─────────────────────────────────────────────────────────────┐
│ 1. 准备测试数据集                                            │
│    - 问题列表                                                │
│    - 黄金标准答案（ground truth）                            │
└────────────────┬────────────────────────────────────────────┘
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. 运行 RAG 系统                                             │
│    Question → RAG System → {answer, contexts}               │
└────────────────┬────────────────────────────────────────────┘
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. RAGAS 自动评估                                            │
│    使用评估 LLM（如 GPT-4o-mini）分析：                      │
│    - 答案是否忠实于上下文？                                  │
│    - 答案是否相关？                                          │
│    - 检索是否完整？                                          │
│    - 检索是否精准？                                          │
└────────────────┬────────────────────────────────────────────┘
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. 输出分数报告                                              │
│    Faithfulness:      0.87                                  │
│    Answer Relevancy:  0.91                                  │
│    Context Recall:    0.82                                  │
│    Context Precision: 0.75                                  │
│    Overall RAGAS:     0.84                                  │
└─────────────────────────────────────────────────────────────┘
```

### 数据格式

RAGAS 需要的输入数据：

```python
from datasets import Dataset

# 测试数据集格式
test_dataset = Dataset.from_dict({
    "question": [
        "What is the capital of France?",
        "Who founded Apple Inc.?",
    ],
    "answer": [
        "Paris is the capital of France.",  # RAG 系统生成的答案
        "Steve Jobs and Steve Wozniak founded Apple Inc.",
    ],
    "contexts": [
        [  # 检索到的上下文（列表）
            "Paris is the capital and largest city of France...",
            "The city is located on the Seine River...",
        ],
        [
            "Apple Inc. was founded by Steve Jobs, Steve Wozniak...",
        ],
    ],
    "ground_truth": [
        "Paris",  # 黄金标准答案
        "Steve Jobs and Steve Wozniak",
    ],
})
```

### 评估 LLM 的作用

**关键点**：RAGAS 使用**另一个 LLM**（评估 LLM）来评估你的 RAG 系统。

```
你的 RAG 系统:
  使用 LLM A（如 Qwen-7B）生成答案
        ↓
RAGAS 评估:
  使用 LLM B（如 GPT-4o-mini）评估答案质量
```

**为什么这样设计？**
- LLM 擅长理解语义和推理
- 可以判断"答案是否基于上下文"、"是否相关"等复杂问题
- 比简单的字符串匹配更智能

**成本**：
- 评估一个问题通常需要 4-8 次 LLM 调用（每个指标 1-2 次）
- 推荐使用便宜的模型如 GPT-4o-mini（$0.15/1M tokens）
- 评估 100 个问题的成本约 $0.50-1.00

---

## LightRAG 中的 RAGAS 使用

### 内置评估工具

LightRAG 已经集成了 RAGAS：`lightrag/evaluation/eval_rag_quality.py`

### 快速开始

#### 步骤 1: 安装依赖

```bash
pip install ragas datasets langchain-openai
```

#### 步骤 2: 准备测试数据集

```bash
# 创建测试数据集
cat > lightrag/evaluation/my_test.json << 'EOF'
{
  "test_cases": [
    {
      "question": "LightRAG 支持哪些向量数据库？",
      "ground_truth": "LightRAG 支持多种向量数据库，包括 Milvus、Qdrant、ChromaDB、Neo4j 等。",
      "project": "lightrag_docs"
    },
    {
      "question": "如何提高 LightRAG 的索引速度？",
      "ground_truth": "可以通过增加 max_async 参数、使用更快的 LLM、禁用 gleaning 等方式提高索引速度。",
      "project": "lightrag_docs"
    }
  ]
}
EOF
```

#### 步骤 3: 配置评估 LLM

```bash
# 创建 .env 文件
cat > .env << 'EOF'
# 评估使用的 LLM（推荐使用便宜的模型）
EVAL_LLM_MODEL=gpt-4o-mini
EVAL_LLM_BINDING_API_KEY=sk-your-api-key-here

# 评估使用的 Embedding 模型
EVAL_EMBEDDING_MODEL=text-embedding-3-small

# 如果使用自定义端点（如 vLLM、Ollama）
# EVAL_LLM_BINDING_HOST=http://localhost:8000/v1

# 并发数（同时评估多少个问题）
EVAL_MAX_CONCURRENT=2
EOF
```

#### 步骤 4: 运行 LightRAG 服务器

```bash
# 启动 LightRAG（确保已经索引了相关文档）
python -m lightrag.api.lightrag_server
```

#### 步骤 5: 运行评估

```bash
# 评估默认数据集
python lightrag/evaluation/eval_rag_quality.py

# 评估自定义数据集
python lightrag/evaluation/eval_rag_quality.py \
    --dataset lightrag/evaluation/my_test.json \
    --ragendpoint http://localhost:9621
```

### 输出示例

```
======================================================================
🔍 RAGAS Evaluation - Using Real LightRAG API
======================================================================
Evaluation Models:
  • LLM Model:            gpt-4o-mini
  • Embedding Model:      text-embedding-3-small
  • LLM Endpoint:         OpenAI Official API

Test Configuration:
  • Total Test Cases:     2
  • Test Dataset:         my_test.json
  • LightRAG API:         http://localhost:9621
  • Results Directory:    results

======================================================================
🚀 Starting RAGAS Evaluation of LightRAG System
🔧 RAGAS Evaluation (Stage 2): 2 concurrent
======================================================================

Eval-01: 100%|████████████████████████| 4/4 [00:12<00:00]
Eval-02: 100%|████████████████████████| 4/4 [00:11<00:00]

======================================================================
📊 EVALUATION RESULTS SUMMARY
======================================================================
#    | Question                                           | Faith  | AnswRel | CtxRec | CtxPrec | RAGAS  | Status
--------------------------------------------------------------------------------------------------------------------------
1    | LightRAG 支持哪些向量数据库？                       | 0.9200 | 0.9500  | 0.8800 | 0.8200  | 0.8925 | ✓
2    | 如何提高 LightRAG 的索引速度？                      | 0.8800 | 0.9100  | 0.8500 | 0.7800  | 0.8550 | ✓
======================================================================

======================================================================
📊 EVALUATION COMPLETE
======================================================================
Total Tests:    2
Successful:     2
Failed:         0
Success Rate:   100.00%
Elapsed Time:   25.43 seconds
Avg Time/Test:  12.72 seconds

======================================================================
📈 BENCHMARK RESULTS (Average)
======================================================================
Average Faithfulness:      0.9000
Average Answer Relevance:  0.9300
Average Context Recall:    0.8650
Average Context Precision: 0.8000
Average RAGAS Score:       0.8738
----------------------------------------------------------------------
Min RAGAS Score:           0.8550
Max RAGAS Score:           0.8925

======================================================================
📁 GENERATED FILES
======================================================================
Results Dir:    /path/to/lightrag/evaluation/results
   • CSV:  results_20250119_143022.csv
   • JSON: results_20250119_143022.json
======================================================================
```

### 结果解读

#### 问题 1 分析

```
Question: "LightRAG 支持哪些向量数据库？"

Faithfulness:      0.92  ✅ 答案基本基于事实
Answer Relevancy:  0.95  ✅ 直接回答了问题
Context Recall:    0.88  👍 检索到大部分相关信息
Context Precision: 0.82  👍 检索较精准但有少量噪音

RAGAS Score: 0.8925  ✅ 整体质量优秀
```

**这意味着什么？**
- ✅ RAG 系统正确回答了问题
- ✅ 答案基于检索到的文档，没有幻觉
- ⚠️ Context Precision 0.82 表示检索了一些不太相关的内容
- 💡 优化建议：可以调整 top_k 或使用 reranking

#### 问题 2 分析

```
Question: "如何提高 LightRAG 的索引速度？"

Faithfulness:      0.88  👍 答案较为忠实
Answer Relevancy:  0.91  ✅ 回答了问题
Context Recall:    0.85  👍 检索到大部分信息
Context Precision: 0.78  ⚠️ 检索有较多噪音

RAGAS Score: 0.8550  👍 质量良好
```

**这意味着什么？**
- ✅ 基本回答正确
- ⚠️ Faithfulness 0.88 表示可能有轻微幻觉或不可验证的陈述
- ⚠️ Context Precision 0.78 表示检索了不少无关内容
- 💡 优化建议：
  - 改进 chunking（确保相关信息在同一 chunk）
  - 调整 embedding 模型
  - 添加元数据过滤

---

## RAGAS 的限制和注意事项

### 1. 依赖评估 LLM 的质量

**问题**：如果评估 LLM 本身有偏见或错误怎么办？

```python
# 示例：评估 LLM 可能犯错
Question: "某专业领域的深度问题"
Answer: "（正确但技术性强的答案）"

评估 LLM（GPT-4o-mini）可能：
- 误判为不相关（因为不理解专业术语）
- 误判为幻觉（因为超出其知识范围）
```

**解决方案**：
- 使用更强的评估模型（如 GPT-4o 而非 GPT-4o-mini）
- 在特定领域使用领域专家人工抽查
- 对关键指标进行人工校准

### 2. Ground Truth 的质量

**RAGAS 严重依赖黄金标准答案的质量**：

```python
# 糟糕的 Ground Truth
{
    "question": "What is LightRAG?",
    "ground_truth": "It's good"  # 太简单，无法评估 Recall
}

# 好的 Ground Truth
{
    "question": "What is LightRAG?",
    "ground_truth": "LightRAG is a retrieval-augmented generation framework
                     that combines knowledge graphs with vector search to
                     improve answer quality and relevance."
}
```

**建议**：
- Ground Truth 应该详细、准确
- 包含所有关键信息点
- 由领域专家撰写或审核

### 3. 成本考虑

**每个问题的评估成本**：

```
单个问题评估:
- Faithfulness:      2-3 次 LLM 调用
- Answer Relevancy:  1-2 次 LLM 调用
- Context Recall:    2-4 次 LLM 调用
- Context Precision: 2-4 次 LLM 调用

总计: 7-13 次 LLM 调用

使用 GPT-4o-mini ($0.15/1M input, $0.60/1M output):
- 每个问题约 $0.01-0.02
- 100 个问题约 $1-2
- 1000 个问题约 $10-20
```

**省钱建议**：
- 使用便宜的评估模型（GPT-4o-mini）
- 使用自托管模型（如 Qwen2.5-7B）
- 减少测试集大小（50-100 个代表性问题足够）

### 4. 非确定性

**同一个问题多次评估可能得到不同分数**：

```python
# 第一次运行
RAGAS Score: 0.8523

# 第二次运行（完全相同的输入）
RAGAS Score: 0.8619

# 差异原因：评估 LLM 的非确定性
```

**解决方案**：
- 关注趋势而非绝对值（0.85 vs 0.75 是有意义的差异）
- 使用足够大的测试集（减少随机波动）
- 降低评估 LLM 的 temperature

---

## 实战建议

### 1. 建立基线

```bash
# 在做任何优化之前，先建立基线
python lightrag/evaluation/eval_rag_quality.py > baseline_results.txt

# 记录基线分数
Baseline RAGAS Score: 0.8234
- Faithfulness:      0.8523
- Answer Relevancy:  0.8756
- Context Recall:    0.7945
- Context Precision: 0.7712
```

### 2. 每次改动后重新评估

```bash
# 改动 1: 换了 embedding 模型
# 重新评估
python lightrag/evaluation/eval_rag_quality.py > after_embedding_change.txt

# 对比
RAGAS Score: 0.8234 → 0.8456 (+2.7%)  ✅ 改进
- Context Recall: 0.7945 → 0.8234 (+3.6%)  # 主要改进点

# 改动 2: 调整了 chunk size
RAGAS Score: 0.8456 → 0.8123 (-3.9%)  ❌ 变差
# 回滚改动
```

### 3. 分析单个失败案例

```python
# 查看详细的 JSON 结果
import json
with open('lightrag/evaluation/results/results_20250119.json') as f:
    results = json.load(f)

# 找出分数最低的问题
lowest_score = min(results['results'], key=lambda x: x['ragas_score'])

print(f"问题: {lowest_score['question']}")
print(f"RAGAS: {lowest_score['ragas_score']}")
print(f"答案: {lowest_score['answer']}")
print(f"指标: {lowest_score['metrics']}")

# 输出示例:
# 问题: "如何配置自定义 LLM？"
# RAGAS: 0.6234
# 答案: "LightRAG supports custom LLMs..."
# 指标: {
#   'faithfulness': 0.5234,       # ← 低！可能有幻觉
#   'answer_relevancy': 0.8234,
#   'context_recall': 0.5123,     # ← 低！检索不足
#   'context_precision': 0.6456
# }
#
# 分析: 检索没找到关键信息，LLM 开始"猜测"答案
# 解决: 确保相关文档被索引，改进检索策略
```

### 4. 针对性优化

根据指标找出优化方向：

| 指标低 | 可能原因 | 优化方向 |
|--------|---------|---------|
| **Context Precision 低** | 检索了太多无关内容 | • 减少 top_k<br>• 添加 reranking<br>• 改进 embedding 模型 |
| **Context Recall 低** | 检索遗漏关键信息 | • 增加 top_k<br>• 优化 chunking 策略<br>• 改进查询改写 |
| **Faithfulness 低** | LLM 产生幻觉 | • 改进 prompt（强调基于事实）<br>• 降低 temperature<br>• 使用更可靠的 LLM |
| **Answer Relevancy 低** | 答案答非所问或冗余 | • 改进 prompt（要求简洁）<br>• 限制生成长度<br>• 使用指令跟随能力更强的 LLM |

---

## 与传统评估方法对比

### 传统方法：人工评估

```
优点:
✅ 最准确（人类判断）
✅ 可以发现意外问题

缺点:
❌ 耗时（每个问题 2-5 分钟）
❌ 成本高（需要专家时间）
❌ 不可扩展（难以评估 100+ 问题）
❌ 主观（不同评估者结果不一致）
```

### RAGAS 方法

```
优点:
✅ 快速（几分钟评估 100 个问题）
✅ 成本低（$1-2 per 100 questions）
✅ 可扩展（轻松评估 1000+ 问题）
✅ 客观（分数可重复、可比较）
✅ 自动化（CI/CD 集成）

缺点:
❌ 依赖评估 LLM（可能有误判）
❌ 需要高质量 Ground Truth
❌ 无法发现所有类型的问题
```

### 推荐做法：混合使用

```
1. 使用 RAGAS 进行大规模自动评估（100+ 问题）
   → 快速发现整体趋势和明显问题

2. 对低分案例进行人工审查（10-20 个）
   → 深入理解问题根源

3. 定期进行人工抽查（每周/每月）
   → 校准 RAGAS 评估，发现边缘案例

4. 收集真实用户反馈
   → 发现 RAGAS 无法覆盖的问题
```

---

## 总结

### RAGAS 核心价值

1. **量化 RAG 质量**：将"答案好不好"从主观判断变为客观分数
2. **快速迭代**：每次改动都能立即看到影响
3. **多维度评估**：从检索、生成、整体三个角度全面评估
4. **自动化**：可集成到 CI/CD 流程

### 使用场景

✅ **适合使用 RAGAS 的场景**：
- 评估 RAG 系统的整体质量
- 对比不同配置/模型的效果
- 持续监控系统性能
- A/B 测试新特性

❌ **不适合使用 RAGAS 的场景**：
- 评估单个组件（如只评估实体提取）→ 用专门的 Precision/Recall
- 极高准确性要求的领域（如医疗）→ 必须人工审核
- 没有明确答案的开放性问题 → 难以创建 Ground Truth

### 下一步

现在你了解了 RAGAS，可以：

1. **运行第一次评估**：
   ```bash
   python lightrag/evaluation/eval_rag_quality.py
   ```

2. **创建自己的测试集**：
   - 收集 20-50 个真实用户问题
   - 为每个问题编写高质量的 Ground Truth

3. **建立评估流程**：
   - 每次重大改动前后都运行评估
   - 追踪 RAGAS 分数变化
   - 结合人工审查

4. **持续优化**：
   - 针对低分指标优化
   - 定期更新测试集
   - 收集新的边缘案例

---

## 参考资源

- **RAGAS 官方文档**: https://docs.ragas.io/
- **RAGAS GitHub**: https://github.com/explodinggradients/ragas
- **论文**: "RAGAS: Automated Evaluation of Retrieval Augmented Generation"
- **LightRAG 评估代码**: `lightrag/evaluation/eval_rag_quality.py`

---

需要我帮你：
- 创建第一个测试数据集？
- 运行实际评估？
- 分析评估结果？
