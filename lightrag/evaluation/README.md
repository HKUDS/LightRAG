# 📊 LightRAG Evaluation Framework

RAGAS-based offline evaluation of your LightRAG system.

## What is RAGAS?

**RAGAS** (Retrieval Augmented Generation Assessment) is a framework for reference-free evaluation of RAG systems using LLMs.

Instead of requiring human-annotated ground truth, RAGAS uses state-of-the-art evaluation metrics:

### Core Metrics

| Metric | What It Measures | Good Score |
|--------|-----------------|-----------|
| **Faithfulness** | Is the answer factually accurate based on retrieved context? | > 0.80 |
| **Answer Relevance** | Is the answer relevant to the user's question? | > 0.80 |
| **Context Recall** | Was all relevant information retrieved from documents? | > 0.80 |
| **Context Precision** | Is retrieved context clean without irrelevant noise? | > 0.80 |
| **RAGAS Score** | Overall quality metric (average of above) | > 0.80 |

---

## 📁 Structure

```
lightrag/evaluation/
├── eval_rag_quality.py      # Main evaluation script
├── sample_dataset.json        # 3 test questions about LightRAG
├── sample_documents/          # Matching markdown files for testing
│   ├── 01_lightrag_overview.md
│   ├── 02_rag_architecture.md
│   ├── 03_lightrag_improvements.md
│   ├── 04_supported_databases.md
│   ├── 05_evaluation_and_deployment.md
│   └── README.md
├── __init__.py              # Package init
├── results/                 # Output directory
│   ├── results_YYYYMMDD_HHMMSS.json    # Raw metrics in JSON
│   └── results_YYYYMMDD_HHMMSS.csv     # Metrics in CSV format
└── README.md                # This file
```

**Quick Test:** Index files from `sample_documents/` into LightRAG, then run the evaluator to reproduce results (~89-100% RAGAS score per question).

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install ragas datasets langfuse
```

Or use your project dependencies (already included in pyproject.toml):

```bash
pip install -e ".[offline-llm]"
```

### 2. Run Evaluation

```bash
cd /path/to/LightRAG
python -m lightrag.evaluation.eval_rag_quality
```

Or directly:

```bash
python lightrag/evaluation/eval_rag_quality.py
```

### 3. View Results

Results are saved automatically in `lightrag/evaluation/results/`:

```
results/
├── results_20241023_143022.json     ← Raw metrics in JSON format
└── results_20241023_143022.csv      ← Metrics in CSV format (for spreadsheets)
```

**Results include:**
- ✅ Overall RAGAS score
- 📊 Per-metric averages (Faithfulness, Answer Relevance, Context Recall, Context Precision)
- 📋 Individual test case results
- 📈 Performance breakdown by question

---

## ⚙️ Configuration

### Environment Variables

The evaluation framework supports customization through environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `EVAL_LLM_MODEL` | `gpt-4o-mini` | LLM model used for RAGAS evaluation |
| `EVAL_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model for evaluation |
| `EVAL_LLM_BINDING_API_KEY` | (falls back to `OPENAI_API_KEY`) | API key for evaluation models |
| `EVAL_LLM_BINDING_HOST` | (optional) | Custom endpoint URL for OpenAI-compatible services |
| `EVAL_MAX_CONCURRENT` | `1` | Number of concurrent test case evaluations (1=serial) |
| `EVAL_QUERY_TOP_K` | `10` | Number of documents to retrieve per query |
| `EVAL_LLM_MAX_RETRIES` | `5` | Maximum LLM request retries |
| `EVAL_LLM_TIMEOUT` | `120` | LLM request timeout in seconds |

### Usage Examples

**Default Configuration (OpenAI):**
```bash
export OPENAI_API_KEY=sk-xxx
python lightrag/evaluation/eval_rag_quality.py
```

**Custom Model:**
```bash
export OPENAI_API_KEY=sk-xxx
export EVAL_LLM_MODEL=gpt-4.1
export EVAL_EMBEDDING_MODEL=text-embedding-3-large
python lightrag/evaluation/eval_rag_quality.py
```

**OpenAI-Compatible Endpoint:**
```bash
export EVAL_LLM_BINDING_API_KEY=your-custom-key
export EVAL_LLM_BINDING_HOST=https://api.openai.com/v1
export EVAL_LLM_MODEL=qwen-plus
python lightrag/evaluation/eval_rag_quality.py
```

### Concurrency Control & Rate Limiting

The evaluation framework includes built-in concurrency control to prevent API rate limiting issues:

**Why Concurrency Control Matters:**
- RAGAS internally makes many concurrent LLM calls for each test case
- Context Precision metric calls LLM once per retrieved document
- Without control, this can easily exceed API rate limits

**Default Configuration (Conservative):**
```bash
EVAL_MAX_CONCURRENT=1    # Serial evaluation (one test at a time)
EVAL_QUERY_TOP_K=10      # OP_K query parameter of LightRAG
EVAL_LLM_MAX_RETRIES=5   # Retry failed requests 5 times
EVAL_LLM_TIMEOUT=180     # 2-minute timeout per request
```

**If You Have Higher API Quotas:**
```bash
EVAL_MAX_CONCURRENT=2    # Evaluate 2 tests in parallel
EVAL_QUERY_TOP_K=20      # OP_K query parameter of LightRAG
```

**Common Issues and Solutions:**

| Issue | Solution |
|-------|----------|
| **Warning: "LM returned 1 generations instead of 3"** | Reduce `EVAL_MAX_CONCURRENT` to 1 or decrease `EVAL_QUERY_TOP_K` |
| **Context Precision returns NaN** | Lower `EVAL_QUERY_TOP_K` to reduce LLM calls per test case |
| **Rate limit errors (429)** | Increase `EVAL_LLM_MAX_RETRIES` and decrease `EVAL_MAX_CONCURRENT` |
| **Request timeouts** | Increase `EVAL_LLM_TIMEOUT` to 180 or higher |

---

## 📝 Test Dataset

`sample_dataset.json` contains 3 generic questions about LightRAG. Replace with questions matching YOUR indexed documents.

**Custom Test Cases:**

```json
{
  "test_cases": [
    {
      "question": "Your question here",
      "ground_truth": "Expected answer from your data",
      "context": "topic"
    }
  ]
}
```

---

## 📊 Interpreting Results

### Score Ranges

- **0.80-1.00**: ✅ Excellent (Production-ready)
- **0.60-0.80**: ⚠️ Good (Room for improvement)
- **0.40-0.60**: ❌ Poor (Needs optimization)
- **0.00-0.40**: 🔴 Critical (Major issues)

### What Low Scores Mean

| Metric | Low Score Indicates |
|--------|-------------------|
| **Faithfulness** | Responses contain hallucinations or incorrect information |
| **Answer Relevance** | Answers don't match what users asked |
| **Context Recall** | Missing important information in retrieval |
| **Context Precision** | Retrieved documents contain irrelevant noise |

### Optimization Tips

1. **Low Faithfulness**:
   - Improve entity extraction quality
   - Better document chunking
   - Tune retrieval temperature

2. **Low Answer Relevance**:
   - Improve prompt engineering
   - Better query understanding
   - Check semantic similarity threshold

3. **Low Context Recall**:
   - Increase retrieval `top_k` results
   - Improve embedding model
   - Better document preprocessing

4. **Low Context Precision**:
   - Smaller, focused chunks
   - Better filtering
   - Improve chunking strategy

---

## 📚 Resources

- [RAGAS Documentation](https://docs.ragas.io/)
- [RAGAS GitHub](https://github.com/explodinggradients/ragas)

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'ragas'"

```bash
pip install ragas datasets
```

### "Warning: LM returned 1 generations instead of requested 3" or Context Precision NaN

**Cause**: This warning indicates API rate limiting or concurrent request overload:
- RAGAS makes multiple LLM calls per test case (faithfulness, relevancy, recall, precision)
- Context Precision calls LLM once per retrieved document (with `EVAL_QUERY_TOP_K=10`, that's 10 calls)
- Concurrent evaluation multiplies these calls: `EVAL_MAX_CONCURRENT × LLM calls per test`

**Solutions** (in order of effectiveness):

1. **Serial Evaluation** (Default):
   ```bash
   export EVAL_MAX_CONCURRENT=1
   python lightrag/evaluation/eval_rag_quality.py
   ```

2. **Reduce Retrieved Documents**:
   ```bash
   export EVAL_QUERY_TOP_K=5  # Halves Context Precision LLM calls
   python lightrag/evaluation/eval_rag_quality.py
   ```

3. **Increase Retry & Timeout**:
   ```bash
   export EVAL_LLM_MAX_RETRIES=10
   export EVAL_LLM_TIMEOUT=180
   python lightrag/evaluation/eval_rag_quality.py
   ```

4. **Use Higher Quota API** (if available):
   - Upgrade to OpenAI Tier 2+ for higher RPM limits
   - Use self-hosted OpenAI-compatible service with no rate limits

### "AttributeError: 'InstructorLLM' object has no attribute 'agenerate_prompt'" or NaN results

This error occurs with RAGAS 0.3.x when LLM and Embeddings are not explicitly configured. The evaluation framework now handles this automatically by:
- Using environment variables to configure evaluation models
- Creating proper LLM and Embeddings instances for RAGAS

**Solution**: Ensure you have set one of the following:
- `OPENAI_API_KEY` environment variable (default)
- `EVAL_LLM_BINDING_API_KEY` for custom API key

The framework will automatically configure the evaluation models.

### "No sample_dataset.json found"

Make sure you're running from the project root:

```bash
cd /path/to/LightRAG
python lightrag/evaluation/eval_rag_quality.py
```

### "LLM API errors during evaluation"

The evaluation uses your configured LLM (OpenAI by default). Ensure:
- API keys are set in `.env`
- Have sufficient API quota
- Network connection is stable

### Evaluation requires running LightRAG API

The evaluator queries a running LightRAG API server at `http://localhost:9621`. Make sure:
1. LightRAG API server is running (`python lightrag/api/lightrag_server.py`)
2. Documents are indexed in your LightRAG instance
3. API is accessible at the configured URL

---

## 📝 Next Steps

1. Index documents into LightRAG (WebUI or API)
2. Start LightRAG API server
3. Run `python lightrag/evaluation/eval_rag_quality.py`
4. Review results (JSON/CSV) in `results/` folder
5. Adjust entity extraction prompts or retrieval settings based on scores

---

**Happy Evaluating! 🚀**
