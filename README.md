# LightRAG: Simple and Fast Retrieval-Augmented Generation
![请添加图片描述](https://i-blog.csdnimg.cn/direct/567139f1a36e4564abc63ce5c12b6271.jpeg)



<a href='https://github.com/HKUDS/LightRAG'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a href='https://arxiv.org/abs/2410.05779'><img src='https://img.shields.io/badge/arXiv-2410.05779-b31b1b'></a>

This repository hosts the code of LightRAG. The structure of this code is based on [nano-graphrag](https://github.com/gusye1234/nano-graphrag).
![请添加图片描述](https://i-blog.csdnimg.cn/direct/b2aaf634151b4706892693ffb43d9093.png)
## Install

* Install from source

```bash
cd LightRAG
pip install -e .
```
* Install from PyPI
```bash
pip install lightrag-hku
```

## Quick Start

* Set OpenAI API key in environment: `export OPENAI_API_KEY="sk-...".`
* Download the demo text "A Christmas Carol by Charles Dickens" 
```bash
curl https://raw.githubusercontent.com/gusye1234/nano-graphrag/main/tests/mock_data.txt > ./book.txt
```
Use the below python snippet:

```python
from lightrag import LightRAG, QueryParam

rag = LightRAG(working_dir="./dickens")

with open("./book.txt") as f:
    rag.insert(f.read())

# Perform naive search
print(rag.query("What are the top themes in this story?", param=QueryParam(mode="naive")))

# Perform local search
print(rag.query("What are the top themes in this story?", param=QueryParam(mode="local")))

# Perform global search
print(rag.query("What are the top themes in this story?", param=QueryParam(mode="global")))

# Perform hybird search
print(rag.query("What are the top themes in this story?", param=QueryParam(mode="hybird")))
```
Batch Insert
```python
rag.insert(["TEXT1", "TEXT2",...])
```
Incremental Insert

```python
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
### Overall Performance Table
### Overall Performance Table
|                      | **Agriculture**             |                       | **CS**                    |                       | **Legal**                 |                       | **Mix**                   |                       |
|----------------------|-------------------------|-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|
|                      | NaiveRAG                | **LightRAG**           | NaiveRAG              | **LightRAG**           | NaiveRAG              | **LightRAG**           | NaiveRAG              | **LightRAG**           |
| **Comprehensiveness** | 32.69%                  | **67.31%**             | 35.44%                | **64.56%**             | 19.05%                | **80.95%**             | 36.36%                | **63.64%**             |
| **Diversity**         | 24.09%                  | **75.91%**             | 35.24%                | **64.76%**             | 10.98%                | **89.02%**             | 30.76%                | **69.24%**             |
| **Empowerment**       | 31.35%                  | **68.65%**             | 35.48%                | **64.52%**             | 17.59%                | **82.41%**             | 40.95%                | **59.05%**             |
| **Overall**           | 33.30%                  | **66.70%**             | 34.76%                | **65.24%**             | 17.46%                | **82.54%**             | 37.59%                | **62.40%**             |
|                      | RQ-RAG                  | **LightRAG**           | RQ-RAG                | **LightRAG**           | RQ-RAG                | **LightRAG**           | RQ-RAG                | **LightRAG**           |
| **Comprehensiveness** | 32.05%                  | **67.95%**             | 39.30%                | **60.70%**             | 18.57%                | **81.43%**             | 38.89%                | **61.11%**             |
| **Diversity**         | 29.44%                  | **70.56%**             | 38.71%                | **61.29%**             | 15.14%                | **84.86%**             | 28.50%                | **71.50%**             |
| **Empowerment**       | 32.51%                  | **67.49%**             | 37.52%                | **62.48%**             | 17.80%                | **82.20%**             | 43.96%                | **56.04%**             |
| **Overall**           | 33.29%                  | **66.71%**             | 39.03%                | **60.97%**             | 17.80%                | **82.20%**             | 39.61%                | **60.39%**             |
|                      | HyDE                    | **LightRAG**           | HyDE                  | **LightRAG**           | HyDE                  | **LightRAG**           | HyDE                  | **LightRAG**           |
| **Comprehensiveness** | 24.39%                  | **75.61%**             | 36.49%                | **63.51%**             | 27.68%                | **72.32%**             | 42.17%                | **57.83%**             |
| **Diversity**         | 24.96%                  | **75.34%**             | 37.41%                | **62.59%**             | 18.79%                | **81.21%**             | 30.88%                | **69.12%**             |
| **Empowerment**       | 24.89%                  | **75.11%**             | 34.99%                | **65.01%**             | 26.99%                | **73.01%**             | **45.61%**            | **54.39%**             |
| **Overall**           | 23.17%                  | **76.83%**             | 35.67%                | **64.33%**             | 27.68%                | **72.32%**             | 42.72%                | **57.28%**             |
|                      | GraphRAG                | **LightRAG**           | GraphRAG              | **LightRAG**           | GraphRAG              | **LightRAG**           | GraphRAG              | **LightRAG**           |
| **Comprehensiveness** | 45.56%                  | **54.44%**             | 45.98%                | **54.02%**             | 47.13%                | **52.87%**             | **51.86%**            | 48.14%                |
| **Diversity**         | 19.65%                  | **80.35%**             | 39.64%                | **60.36%**             | 25.55%                | **74.45%**             | 35.87%                | **64.13%**             |
| **Empowerment**       | 36.69%                  | **63.31%**             | 45.09%                | **54.91%**             | 42.81%                | **57.19%**             | **52.94%**            | 47.06%                |
| **Overall**           | 43.62%                  | **56.38%**             | 45.98%                | **54.02%**             | 45.70%                | **54.30%**             | **51.86%**            | 48.14%                |

## Code Structure

```python
.
├── examples
│   ├── batch_eval.py
│   ├── generate_query.py
│   ├── insert.py
│   └── query.py
├── lightrag
│   ├── __init__.py
│   ├── base.py
│   ├── lightrag.py
│   ├── llm.py
│   ├── operate.py
│   ├── prompt.py
│   ├── storage.py
│   └── utils.jpeg
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
