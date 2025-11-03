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
