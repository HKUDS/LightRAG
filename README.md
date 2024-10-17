<center><h2>🚀 PalmierRAG: State-of-the-Art RAG for Any Codebase</h2></center>

## Install

* Install from source

```bash
cd palmier-lightrag
pip install -e .
```

## Quick Start

* Set OpenAI API key in environment if using OpenAI models: `export OPENAI_API_KEY="sk-...".` OR add .env to `lightrag/`
* Create `ragtest/input` directory containing `.txt` files to be indexed.
* Run `python index.py` to index the documents.
* Modify `query.py` with desired query and run `python query.py` to query the documents.
### Using Hugging Face Models
If you want to use Hugging Face models, you only need to set LightRAG as follows:
```python
from lightrag.llm import hf_model_complete, hf_embedding
from transformers import AutoModel, AutoTokenizer

# Initialize LightRAG with Hugging Face model
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=hf_model_complete,  # Use Hugging Face complete model for text generation
    llm_model_name='meta-llama/Llama-3.1-8B-Instruct',  # Model name from Hugging Face
    # Use Hugging Face embedding function
    embedding_func=EmbeddingFunc(
        embedding_dim=384,
        max_token_size=5000,
        func=lambda texts: hf_embedding(
            texts, 
            tokenizer=AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2"),
            embed_model=AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        )
    ),
)
```
### Batch Insert
```python
# Batch Insert: Insert multiple texts at once
rag.insert(["TEXT1", "TEXT2",...])
```
### Incremental Insert

```python
# Incremental Insert: Insert new documents into an existing LightRAG instance
rag = LightRAG(working_dir="./dickens")

with open("./newText.txt") as f:
    rag.insert(f.read())
```
## Evaluation
### Dataset
The dataset used in LightRAG can be download from [TommyChien/UltraDomain](https://huggingface.co/datasets/TommyChien/UltraDomain).

### Generate Query
LightRAG uses the following prompt to generate high-level queries, with the corresponding code located in `example/generate_query.py`.
```python
Given the following description of a dataset:

{description}

Please identify 5 potential users who would engage with this dataset. For each user, list 5 tasks they would perform with this dataset. Then, for each (user, task) combination, generate 5 questions that require a high-level understanding of the entire dataset.

Output the results in the following structure:
- User 1: [user description]
    - Task 1: [task description]
        - Question 1:
        - Question 2:
        - Question 3:
        - Question 4:
        - Question 5:
    - Task 2: [task description]
        ...
    - Task 5: [task description]
- User 2: [user description]
    ...
- User 5: [user description]
    ...
```
 
 ### Batch Eval
To evaluate the performance of two RAG systems on high-level queries, LightRAG uses the following prompt, with the specific code available in `example/batch_eval.py`.
```python
---Role---
You are an expert tasked with evaluating two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.
---Goal---
You will evaluate two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**. 

- **Comprehensiveness**: How much detail does the answer provide to cover all aspects and details of the question?
- **Diversity**: How varied and rich is the answer in providing different perspectives and insights on the question?
- **Empowerment**: How well does the answer help the reader understand and make informed judgments about the topic?

For each criterion, choose the better answer (either Answer 1 or Answer 2) and explain why. Then, select an overall winner based on these three categories.

Here is the question:
{query}

Here are the two answers:

**Answer 1:**
{answer1}

**Answer 2:**
{answer2}

Evaluate both answers using the three criteria listed above and provide detailed explanations for each criterion.

Output your evaluation in the following JSON format:

{{
    "Comprehensiveness": {{
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Provide explanation here]"
    }},
    "Empowerment": {{
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Provide explanation here]"
    }},
    "Overall Winner": {{
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Summarize why this answer is the overall winner based on the three criteria]"
    }}
}}
```

## Code Structure

```python
.
├── examples
│   ├── batch_eval.py
│   ├── generate_query.py
│   ├── lightrag_openai_demo.py
│   └── lightrag_hf_demo.py
├── lightrag
│   ├── __init__.py
│   ├── base.py
│   ├── lightrag.py
│   ├── llm.py
│   ├── operate.py
│   ├── prompt.py
│   ├── storage.py
│   └── utils.py
├── reproduce
│   ├── Step_0.py
│   ├── Step_1.py
│   ├── Step_2.py
│   └── Step_3.py
├── LICENSE
├── README.md
├── requirements.txt
└── setup.py
```

## Citation

```python
@article{guo2024lightrag,
title={LightRAG: Simple and Fast Retrieval-Augmented Generation}, 
author={Zirui Guo and Lianghao Xia and Yanhua Yu and Tu Ao and Chao Huang},
year={2024},
eprint={2410.05779},
archivePrefix={arXiv},
primaryClass={cs.IR}
}
```