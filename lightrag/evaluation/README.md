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
├── sample_dataset.json        # Generic LightRAG test cases (not personal data)
├── __init__.py              # Package init
├── results/                 # Output directory
│   ├── results_YYYYMMDD_HHMMSS.json    # Raw metrics in JSON
│   └── results_YYYYMMDD_HHMMSS.csv     # Metrics in CSV format
└── README.md                # This file
```

**Note:** `sample_dataset.json` contains **generic test questions** about LightRAG features (RAG systems, vector databases, deployment, etc.). This is **not personal portfolio data** - you can use these questions directly to test your own LightRAG installation.

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

The included `sample_dataset.json` contains **generic example questions** about LightRAG (RAG systems, vector databases, deployment, etc.). **This is NOT personal data** - it's meant as a template.

**Important:** You should **replace these with test questions based on YOUR data** that you've injected into your RAG system.

### Creating Your Own Test Cases

Edit `sample_dataset.json` with questions relevant to your indexed documents:

```json
{
  "test_cases": [
    {
      "question": "Question based on your documents",
      "ground_truth": "Expected answer from your data",
      "context": "topic_category"
    }
  ]
}
```

**Example (for a technical portfolio):**

```json
{
  "question": "Which projects use PyTorch?",
  "ground_truth": "The Neural ODE Project uses PyTorch with TorchODE library for continuous-time neural networks.",
  "context": "ml_projects"
}
```

---

## 🔧 Integration with Your RAG System

Currently, the evaluation script uses **ground truth as mock responses**. To evaluate your actual LightRAG:

### Step 1: Update `generate_rag_response()`

In `eval_rag_quality.py`, replace the mock implementation:

```python
async def generate_rag_response(self, question: str, context: str = None) -> Dict[str, str]:
    """Generate RAG response using your LightRAG system"""
    from lightrag import LightRAG

    rag = LightRAG(
        working_dir="./rag_storage",
        llm_model_func=your_llm_function
    )

    response = await rag.aquery(question)

    return {
        "answer": response,
        "context": "context_from_kg"  # If available
    }
```

### Step 2: Run Evaluation

```bash
python lightrag/evaluation/eval_rag_quality.py
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

## 📈 Usage Examples

### Python API

```python
import asyncio
from lightrag.evaluation import RAGEvaluator

async def main():
    evaluator = RAGEvaluator()
    results = await evaluator.run()

    # Access results
    for result in results:
        print(f"Question: {result['question']}")
        print(f"RAGAS Score: {result['ragas_score']:.2%}")
        print(f"Metrics: {result['metrics']}")

asyncio.run(main())
```

### Custom Dataset

```python
evaluator = RAGEvaluator(test_dataset_path="custom_tests.json")
results = await evaluator.run()
```

### Batch Evaluation

```python
from pathlib import Path
import json

results_dir = Path("lightrag/evaluation/results")
results_dir.mkdir(exist_ok=True)

# Run multiple evaluations
for i in range(3):
    evaluator = RAGEvaluator()
    results = await evaluator.run()
```

---

## 🎯 Using Evaluation Results

**What the Metrics Tell You:**

1. ✅ **Quality Metrics**: Overall RAGAS score indicates system health
2. ✅ **Evaluation Framework**: Automated quality assessment with RAGAS
3. ✅ **Best Practices**: Offline evaluation pipeline for continuous improvement
4. ✅ **Production-Ready**: Metrics-driven system optimization

**Example Use Cases:**

- Track RAG quality over time as you update your documents
- Compare different retrieval modes (local, global, hybrid, mix)
- Measure impact of chunking strategy changes
- Validate system performance before deployment

---

## 🔗 Related Features

- **LangFuse Integration**: Real-time observability of production RAG calls
- **LightRAG**: Core RAG system with entity extraction and knowledge graphs
- **Metrics**: See `results/` for detailed evaluation metrics

---

## 📚 Resources

- [RAGAS Documentation](https://docs.ragas.io/)
- [RAGAS GitHub](https://github.com/explodinggradients/ragas)
- [LangFuse + RAGAS Guide](https://langfuse.com/guides/cookbook/evaluation_of_rag_with_ragas)

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

### Results showing 0 scores

Current implementation uses ground truth as mock responses. Results will show perfect scores because the "generated answer" equals the ground truth.

**To use actual RAG results:**
1. Implement the `generate_rag_response()` method
2. Connect to your LightRAG instance
3. Run evaluation again

---

## 📝 Next Steps

1. ✅ Review test dataset in `sample_dataset.json`
2. ✅ Run `python lightrag/evaluation/eval_rag_quality.py`
3. ✅ Open the HTML report in browser
4. 🔄 Integrate with actual LightRAG system
5. 📊 Monitor metrics over time
6. 🎯 Use insights for optimization

---

**Happy Evaluating! 🚀**
