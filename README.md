# LightRAG: Simple and Fast Retrieval-Augmented Generation
<img src='' />

<a href='https://github.com/HKUDS/GraphEdit'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a href='https://arxiv.org/abs/'><img src='https://img.shields.io/badge/arXiv--b31b1b'></a>

This repository hosts the code of LightRAG. The structure of this code is based on [nano-graphrag](https://github.com/gusye1234/nano-graphrag).
## Install

* Install from source

```
cd LightRAG
pip install -e .
```
* Install from PyPI
```
pip install lightrag-hku
```

## Quick Start

* Set OpenAI API key in environment: `export OPENAI_API_KEY="sk-...".`
* Download the demo text "A Christmas Carol by Charles Dickens" 
```
curl https://raw.githubusercontent.com/gusye1234/nano-graphrag/main/tests/mock_data.txt > ./book.txt
```
Use the below python snippet:

```
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
```
rag.insert(["TEXT1", "TEXT2",...])
```
Incremental Insert

```
rag = LightRAG(working_dir="./dickens")

with open("./newText.txt") as f:
    rag.insert(f.read())
```
## Evaluation
### Dataset
The dataset used in LightRAG can be download from [TommyChien/UltraDomain](https://huggingface.co/datasets/TommyChien/UltraDomain).

### Generate Query
LightRAG uses the following prompt to generate high-level queries, with the corresponding code located in `example/generate_query.py`.
```
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
```
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
|                      | **Agriculture**             |                       | **CS**                    |                       | **Legal**                 |                       | **Mix**                   |                       |
|----------------------|-------------------------|-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|
|                      | NaiveRAG                | **LightRAG**           | NaiveRAG              | **LightRAG**           | NaiveRAG              | **LightRAG**           | NaiveRAG              | **LightRAG**           |
| **Comprehensiveness** | 32.69%                  | <u>67.31%</u>         | 35.44%                | <u>64.56%</u>         | 19.05%                | <u>80.95%</u>         | 36.36%                | <u>63.64%</u>         |
| **Diversity**         | 24.09%                  | <u>75.91%</u>         | 35.24%                | <u>64.76%</u>         | 10.98%                | <u>89.02%</u>         | 30.76%                | <u>69.24%</u>         |
| **Empowerment**       | 31.35%                  | <u>68.65%</u>         | 35.48%                | <u>64.52%</u>         | 17.59%                | <u>82.41%</u>         | 40.95%                | <u>59.05%</u>         |
| **Overall**           | 33.30%                  | <u>66.70%</u>         | 34.76%                | <u>65.24%</u>         | 17.46%                | <u>82.54%</u>         | 37.59%                | <u>62.40%</u>         |
|                      | RQ-RAG                  | **LightRAG**           | RQ-RAG                | **LightRAG**           | RQ-RAG                | **LightRAG**           | RQ-RAG                | **LightRAG**           |
| **Comprehensiveness** | 32.05%                  | <u>67.95%</u>         | 39.30%                | <u>60.70%</u>         | 18.57%                | <u>81.43%</u>         | 38.89%                | <u>61.11%</u>         |
| **Diversity**         | 29.44%                  | <u>70.56%</u>         | 38.71%                | <u>61.29%</u>         | 15.14%                | <u>84.86%</u>         | 28.50%                | <u>71.50%</u>         |
| **Empowerment**       | 32.51%                  | <u>67.49%</u>         | 37.52%                | <u>62.48%</u>         | 17.80%                | <u>82.20%</u>         | 43.96%                | <u>56.04%</u>         |
| **Overall**           | 33.29%                  | <u>66.71%</u>         | 39.03%                | <u>60.97%</u>         | 17.80%                | <u>82.20%</u>         | 39.61%                | <u>60.39%</u>         |
|                      | HyDE                    | **LightRAG**           | HyDE                  | **LightRAG**           | HyDE                  | **LightRAG**           | HyDE                  | **LightRAG**           |
| **Comprehensiveness** | 24.39%                  | <u>75.61%</u>         | 36.49%                | <u>63.51%</u>         | 27.68%                | <u>72.32%</u>         | 42.17%                | <u>57.83%</u>         |
| **Diversity**         | 24.96%                  | <u>75.34%</u>         | 37.41%                | <u>62.59%</u>         | 18.79%                | <u>81.21%</u>         | 30.88%                | <u>69.12%</u>         |
| **Empowerment**       | 24.89%                  | <u>75.11%</u>         | 34.99%                | <u>65.01%</u>         | 26.99%                | <u>73.01%</u>         | 45.61%           |<u>54.39%</u>              |
| **Overall**           | 23.17%                  | <u>76.83%</u>         | 35.67%                | <u>64.33%</u>         | 27.68%                | <u>72.32%</u>         | 42.72%                | <u>57.28%</u>         |
|                      | GraphRAG                | **LightRAG**           | GraphRAG              | **LightRAG**           | GraphRAG              | **LightRAG**           | GraphRAG              | **LightRAG**           |
| **Comprehensiveness** | 45.56%                  | <u>54.44%</u>         | 45.98%                | <u>54.02%</u>         | 47.13%                | <u>52.87%</u>         | <u>51.86%</u>            | 48.14%                |
| **Diversity**         | 19.65%                  | <u>80.35%</u>         | 39.64%                | <u>60.36%</u>         | 25.55%                | <u>74.45%</u>         | 35.87%                | <u>64.13%</u>         |
| **Empowerment**       | 36.69%                  | <u>63.31%</u>         | 45.09%                | <u>54.91%</u>         | 42.81%                | <u>57.19%</u>         | <u>52.94%</u>          | 47.06%                |
| **Overall**           | 43.62%                  | <u>56.38%</u>         | 45.98%                | <u>54.02%</u>         | 45.70%                | <u>54.30%</u>         | <u>51.86%</u>          | 48.14%                |

## Code Structure

```
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

```
@article{guo2024lightrag,
title={LightRAG: Simple and Fast Retrieval-Augmented Generation}, 
author={Zirui Guo and Lianghao Xia and Yanhua Yu and Tu Ao and Chao Huang},
year={2024},
eprint={},
archivePrefix={arXiv},
primaryClass={cs.IR}
}
```

