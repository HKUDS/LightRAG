# 📊 RAGAS-based Evaluation Framework

## What is RAGAS?

**RAGAS** (Retrieval Augmented Generation Assessment) is a framework for reference-free evaluation of RAG systems using LLMs. RAGAS uses state-of-the-art evaluation metrics:

### Core Metrics

| Metric | What It Measures | Good Score |
| ------ | ---------------- | ---------- |
| **Faithfulness** | Is the answer factually accurate based on retrieved context? | > 0.80 |
| **Answer Relevance** | Is the answer relevant to the user's question? | > 0.80 |
| **Context Recall** | Was all relevant information retrieved from documents? | > 0.80 |
| **Context Precision** | Is retrieved context clean without irrelevant noise? | > 0.80 |
| **RAGAS Score** | Overall quality metric (average of above) | > 0.80 |

### 📁 LightRAG Evalua'tion Framework Directory Structure

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



## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install ragas datasets langfuse
```

Or use your project dependencies (already included in pyproject.toml):

```bash
pip install -e ".[evaluation]"
```

### 2. Run Evaluation

**Optional offline sample retrieval check (no API/model calls):**
```bash
python lightrag/evaluation/offline_retrieval_check.py --strict
```

This checks whether the bundled sample questions can lexically retrieve their
expected sample documents before running LightRAG, embeddings, LLM calls, or
RAGAS.

**Basic usage (uses defaults):**
```bash
cd /path/to/LightRAG
python lightrag/evaluation/eval_rag_quality.py
```

**Specify custom dataset:**
```bash
python lightrag/evaluation/eval_rag_quality.py --dataset my_test.json
```

**Specify custom RAG endpoint:**
```bash
python lightrag/evaluation/eval_rag_quality.py --ragendpoint http://my-server.com:9621
```

**Specify both (short form):**
```bash
python lightrag/evaluation/eval_rag_quality.py -d my_test.json -r http://localhost:9621
```

**Get help:**
```bash
python lightrag/evaluation/eval_rag_quality.py --help
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



## 📋 Command-Line Arguments

The evaluation script supports command-line arguments for easy configuration:

| Argument | Short | Default | Description |
| -------- | ----- | ------- | ----------- |
| `--dataset` | `-d` | `sample_dataset.json` | Path to test dataset JSON file |
| `--ragendpoint` | `-r` | `http://localhost:9621` or `$LIGHTRAG_API_URL` | LightRAG API endpoint URL |

### Usage Examples

**Use default dataset and endpoint:**
```bash
python lightrag/evaluation/eval_rag_quality.py
```

**Custom dataset with default endpoint:**
```bash
python lightrag/evaluation/eval_rag_quality.py --dataset path/to/my_dataset.json
```

**Default dataset with custom endpoint:**
```bash
python lightrag/evaluation/eval_rag_quality.py --ragendpoint http://my-server.com:9621
```

**Custom dataset and endpoint:**
```bash
python lightrag/evaluation/eval_rag_quality.py -d my_dataset.json -r http://localhost:9621
```

**Absolute path to dataset:**
```bash
python lightrag/evaluation/eval_rag_quality.py -d /path/to/custom_dataset.json
```

**Show help message:**
```bash
python lightrag/evaluation/eval_rag_quality.py --help
```



## ⚙️ Configuration

### Environment Variables

The evaluation framework supports customization through environment variables:

**⚠️ IMPORTANT: Both LLM and Embedding endpoints MUST be OpenAI-compatible**
- The RAGAS framework requires OpenAI-compatible API interfaces
- Custom endpoints must implement the OpenAI API format (e.g., vLLM, SGLang, LocalAI)
- Non-compatible endpoints will cause evaluation failures

| Variable | Default | Description |
| -------- | ------- | ----------- |
| **LLM Configuration** | | |
| `EVAL_LLM_MODEL` | `gpt-4o-mini` | LLM model used for RAGAS evaluation |
| `EVAL_LLM_BINDING_API_KEY` | falls back to `OPENAI_API_KEY` | API key for LLM evaluation |
| `EVAL_LLM_BINDING_HOST` | (optional) | Custom OpenAI-compatible endpoint URL for LLM |
| **Embedding Configuration** | | |
| `EVAL_EMBEDDING_MODEL` | `text-embedding-3-large` | Embedding model for evaluation |
| `EVAL_EMBEDDING_BINDING_API_KEY` | falls back to `EVAL_LLM_BINDING_API_KEY` → `OPENAI_API_KEY` | API key for embeddings |
| `EVAL_EMBEDDING_BINDING_HOST` | falls back to `EVAL_LLM_BINDING_HOST` | Custom OpenAI-compatible endpoint URL for embeddings |
| **Performance Tuning** | | |
| `EVAL_MAX_CONCURRENT` | 2 | Number of concurrent test case evaluations (1=serial) |
| `EVAL_QUERY_TOP_K` | 10 | Number of documents to retrieve per query |
| `EVAL_LLM_MAX_RETRIES` | 5 | Maximum LLM request retries |
| `EVAL_LLM_TIMEOUT` | 180 | LLM request timeout in seconds |

### Usage Examples

**Example 1: Default Configuration (OpenAI Official API)**
```bash
export OPENAI_API_KEY=sk-xxx
python lightrag/evaluation/eval_rag_quality.py
```
Both LLM and embeddings use OpenAI's official API with default models.

**Example 2: Custom Models on OpenAI**
```bash
export OPENAI_API_KEY=sk-xxx
export EVAL_LLM_MODEL=gpt-4o-mini
export EVAL_EMBEDDING_MODEL=text-embedding-3-large
python lightrag/evaluation/eval_rag_quality.py
```

**Example 3: Same Custom OpenAI-Compatible Endpoint for Both**
```bash
# Both LLM and embeddings use the same custom endpoint
export EVAL_LLM_BINDING_API_KEY=your-custom-key
export EVAL_LLM_BINDING_HOST=http://localhost:8000/v1
export EVAL_LLM_MODEL=qwen-plus
export EVAL_EMBEDDING_MODEL=BAAI/bge-m3
python lightrag/evaluation/eval_rag_quality.py
```
Embeddings automatically inherit LLM endpoint configuration.

**Example 4: Separate Endpoints (Cost Optimization)**
```bash
# Use OpenAI for LLM (high quality)
export EVAL_LLM_BINDING_API_KEY=sk-openai-key
export EVAL_LLM_MODEL=gpt-4o-mini
# No EVAL_LLM_BINDING_HOST means use OpenAI official API

# Use local vLLM for embeddings (cost-effective)
export EVAL_EMBEDDING_BINDING_API_KEY=local-key
export EVAL_EMBEDDING_BINDING_HOST=http://localhost:8001/v1
export EVAL_EMBEDDING_MODEL=BAAI/bge-m3

python lightrag/evaluation/eval_rag_quality.py
```
LLM uses OpenAI official API, embeddings use local custom endpoint.

**Example 5: Different Custom Endpoints for LLM and Embeddings**
```bash
# LLM on one OpenAI-compatible server
export EVAL_LLM_BINDING_API_KEY=key1
export EVAL_LLM_BINDING_HOST=http://llm-server:8000/v1
export EVAL_LLM_MODEL=custom-llm

# Embeddings on another OpenAI-compatible server
export EVAL_EMBEDDING_BINDING_API_KEY=key2
export EVAL_EMBEDDING_BINDING_HOST=http://embedding-server:8001/v1
export EVAL_EMBEDDING_MODEL=custom-embedding

python lightrag/evaluation/eval_rag_quality.py
```
Both use different custom OpenAI-compatible endpoints.

**Example 6: Using Environment Variables from .env File**
```bash
# Create .env file in project root
cat > .env << EOF
EVAL_LLM_BINDING_API_KEY=your-key
EVAL_LLM_BINDING_HOST=http://localhost:8000/v1
EVAL_LLM_MODEL=qwen-plus
EVAL_EMBEDDING_MODEL=BAAI/bge-m3
EOF

# Run evaluation (automatically loads .env)
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
EVAL_MAX_CONCURRENT=2    # Serial evaluation (one test at a time)
EVAL_QUERY_TOP_K=10      # OP_K query parameter of LightRAG
EVAL_LLM_MAX_RETRIES=5   # Retry failed requests 5 times
EVAL_LLM_TIMEOUT=180     # 3-minute timeout per request
```

**Common Issues and Solutions:**

| Issue | Solution |
| ----- | -------- |
| **Warning: "LM returned 1 generations instead of 3"** | Reduce `EVAL_MAX_CONCURRENT` to 1 or decrease `EVAL_QUERY_TOP_K` |
| **Context Precision returns NaN** | Lower `EVAL_QUERY_TOP_K` to reduce LLM calls per test case |
| **Rate limit errors (429)** | Increase `EVAL_LLM_MAX_RETRIES` and decrease `EVAL_MAX_CONCURRENT` |
| **Request timeouts** | Increase `EVAL_LLM_TIMEOUT` to 180 or higher |



## 📝 Test Dataset

`sample_dataset.json` contains 3 generic questions about LightRAG. Replace with questions matching YOUR indexed documents.

**Custom Test Cases:**

```json
{
  "test_cases": [
    {
      "question": "Your question here",
      "ground_truth": "Expected answer from your data",
      "project": "evaluation_project_name"
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
| ------ | ------------------- |
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

### "LightRAG query API errors during evaluation"

The evaluation uses your configured LLM (OpenAI by default). Ensure:
- API keys are set in `.env`
- Network connection is stable

### Evaluation requires running LightRAG API

The evaluator queries a running LightRAG API server at `http://localhost:9621`. Make sure:
1. LightRAG API server is running (`python lightrag/api/lightrag_server.py`)
2. Documents are indexed in your LightRAG instance
3. API is accessible at the configured URL



## 📝 Next Steps

1. Start LightRAG API server
2. Upload sample documents into LightRAG  throught  WebUI
3. Run `python lightrag/evaluation/eval_rag_quality.py`
4. Review results (JSON/CSV) in `results/` folder

Evaluation Result Sample:

```
INFO: ======================================================================
INFO: 🔍 RAGAS Evaluation - Using Real LightRAG API
INFO: ======================================================================
INFO: Evaluation Models:
INFO:   • LLM Model:            gpt-4.1
INFO:   • Embedding Model:      text-embedding-3-large
INFO:   • Endpoint:             OpenAI Official API
INFO: Concurrency & Rate Limiting:
INFO:   • Query Top-K:          10 Entities/Relations
INFO:   • LLM Max Retries:      5
INFO:   • LLM Timeout:          180 seconds
INFO: Test Configuration:
INFO:   • Total Test Cases:     6
INFO:   • Test Dataset:         sample_dataset.json
INFO:   • LightRAG API:         http://localhost:9621
INFO:   • Results Directory:    results
INFO: ======================================================================
INFO: 🚀 Starting RAGAS Evaluation of LightRAG System
INFO: 🔧 RAGAS Evaluation (Stage 2): 2 concurrent
INFO: ======================================================================
INFO:
INFO: ===================================================================================================================
INFO: 📊 EVALUATION RESULTS SUMMARY
INFO: ===================================================================================================================
INFO: #    | Question                                           |  Faith | AnswRel | CtxRec | CtxPrec |  RAGAS | Status
INFO: -------------------------------------------------------------------------------------------------------------------
INFO: 1    | How does LightRAG solve the hallucination probl... | 1.0000 |  1.0000 | 1.0000 |  1.0000 | 1.0000 |      ✓
INFO: 2    | What are the three main components required in ... | 0.8500 |  0.5790 | 1.0000 |  1.0000 | 0.8573 |      ✓
INFO: 3    | How does LightRAG's retrieval performance compa... | 0.8056 |  1.0000 | 1.0000 |  1.0000 | 0.9514 |      ✓
INFO: 4    | What vector databases does LightRAG support and... | 0.8182 |  0.9807 | 1.0000 |  1.0000 | 0.9497 |      ✓
INFO: 5    | What are the four key metrics for evaluating RA... | 1.0000 |  0.7452 | 1.0000 |  1.0000 | 0.9363 |      ✓
INFO: 6    | What are the core benefits of LightRAG and how ... | 0.9583 |  0.8829 | 1.0000 |  1.0000 | 0.9603 |      ✓
INFO: ===================================================================================================================
INFO:
INFO: ======================================================================
INFO: 📊 EVALUATION COMPLETE
INFO: ======================================================================
INFO: Total Tests:    6
INFO: Successful:     6
INFO: Failed:         0
INFO: Success Rate:   100.00%
INFO: Elapsed Time:   161.10 seconds
INFO: Avg Time/Test:  26.85 seconds
INFO:
INFO: ======================================================================
INFO: 📈 BENCHMARK RESULTS (Average)
INFO: ======================================================================
INFO: Average Faithfulness:      0.9053
INFO: Average Answer Relevance:  0.8646
INFO: Average Context Recall:    1.0000
INFO: Average Context Precision: 1.0000
INFO: Average RAGAS Score:       0.9425
INFO: ----------------------------------------------------------------------
INFO: Min RAGAS Score:           0.8573
INFO: Max RAGAS Score:           1.0000
```

---

**Happy Evaluating! 🚀**
