<center><h2>🚀 LightRAG: Simple and Fast Retrieval-Augmented Generation</h2></center>


![请添加图片描述](https://i-blog.csdnimg.cn/direct/567139f1a36e4564abc63ce5c12b6271.jpeg)

<div align='center'>
     <p>
        <a href='https://lightrag.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
        <a href='https://youtu.be/oageL-1I0GE'><img src='https://badges.aleen42.com/src/youtube.svg'></a>
        <a href='https://arxiv.org/abs/2410.05779'><img src='https://img.shields.io/badge/arXiv-2410.05779-b31b1b'></a>
        <a href='https://discord.gg/mvsfu2Tg'><img src='https://discordapp.com/api/guilds/1296348098003734629/widget.png?style=shield'></a>
    </p>
     <p>
          <img src='https://img.shields.io/github/stars/hkuds/lightrag?color=green&style=social' />
        <img src="https://img.shields.io/badge/python->=3.9.11-blue">
        <a href="https://pypi.org/project/lightrag-hku/"><img src="https://img.shields.io/pypi/v/lightrag-hku.svg"></a>
        <a href="https://pepy.tech/project/lightrag-hku"><img src="https://static.pepy.tech/badge/lightrag-hku/month"></a>
    </p>

This repository hosts the code of LightRAG. The structure of this code is based on [nano-graphrag](https://github.com/gusye1234/nano-graphrag).
![请添加图片描述](https://i-blog.csdnimg.cn/direct/b2aaf634151b4706892693ffb43d9093.png)
</div>

## 🎉 News
- [x] [2024.10.20]🎯🎯📢📢We’ve added a new feature to LightRAG: Graph Visualization.
- [x] [2024.10.18]🎯🎯📢📢We’ve added a link to a [LightRAG Introduction Video](https://youtu.be/oageL-1I0GE). Thanks to the author!
- [x] [2024.10.17]🎯🎯📢📢We have created a [Discord channel](https://discord.gg/mvsfu2Tg)! Welcome to join for sharing and discussions! 🎉🎉
- [x] [2024.10.16]🎯🎯📢📢LightRAG now supports [Ollama models](https://github.com/HKUDS/LightRAG?tab=readme-ov-file#quick-start)!
- [x] [2024.10.15]🎯🎯📢📢LightRAG now supports [Hugging Face models](https://github.com/HKUDS/LightRAG?tab=readme-ov-file#quick-start)!

## Install

* Install from source (Recommend)

```bash
cd LightRAG
pip install -e .
```
* Install from PyPI
```bash
pip install lightrag-hku
```

## Quick Start
* All the code can be found in the `examples`.
* Set OpenAI API key in environment if using OpenAI models: `export OPENAI_API_KEY="sk-...".`
* Download the demo text "A Christmas Carol by Charles Dickens":
```bash
curl https://raw.githubusercontent.com/gusye1234/nano-graphrag/main/tests/mock_data.txt > ./book.txt
```
Use the below Python snippet (in a script) to initialize LightRAG and perform queries:

```python
from lightrag import LightRAG, QueryParam
from lightrag.llm import gpt_4o_mini_complete, gpt_4o_complete

#########
# Uncomment the below two lines if running in a jupyter notebook to handle the async nature of rag.insert()
# import nest_asyncio 
# nest_asyncio.apply() 
#########

WORKING_DIR = "./dickens"


WORKING_DIR = "./dickens"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=gpt_4o_mini_complete  # Use gpt_4o_mini_complete LLM model
    # llm_model_func=gpt_4o_complete  # Optionally, use a stronger model
)

with open("./book.txt") as f:
    rag.insert(f.read())

# Perform naive search
print(rag.query("What are the top themes in this story?", param=QueryParam(mode="naive")))

# Perform local search
print(rag.query("What are the top themes in this story?", param=QueryParam(mode="local")))

# Perform global search
print(rag.query("What are the top themes in this story?", param=QueryParam(mode="global")))

# Perform hybrid search
print(rag.query("What are the top themes in this story?", param=QueryParam(mode="hybrid")))
```

<details>
<summary> Using Open AI-like APIs </summary>

* LightRAG also supports Open AI-like chat/embeddings APIs:
```python
async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        "solar-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("UPSTAGE_API_KEY"),
        base_url="https://api.upstage.ai/v1/solar",
        **kwargs
    )

async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embedding(
        texts,
        model="solar-embedding-1-large-query",
        api_key=os.getenv("UPSTAGE_API_KEY"),
        base_url="https://api.upstage.ai/v1/solar"
    )

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=llm_model_func,
    embedding_func=EmbeddingFunc(
        embedding_dim=4096,
        max_token_size=8192,
        func=embedding_func
    )
)
```
</details>

<details>
<summary> Using Hugging Face Models </summary>

* If you want to use Hugging Face models, you only need to set LightRAG as follows:
```python
from lightrag.llm import hf_model_complete, hf_embedding
from transformers import AutoModel, AutoTokenizer

# Initialize LightRAG with Hugging Face model
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=hf_model_complete,  # Use Hugging Face model for text generation
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
</details>

<details>
<summary> Using Ollama Models </summary>
     
* If you want to use Ollama models, you only need to set LightRAG as follows:

```python
from lightrag.llm import ollama_model_complete, ollama_embedding

# Initialize LightRAG with Ollama model
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=ollama_model_complete,  # Use Ollama model for text generation
    llm_model_name='your_model_name', # Your model name
    # Use Ollama embedding function
    embedding_func=EmbeddingFunc(
        embedding_dim=768,
        max_token_size=8192,
        func=lambda texts: ollama_embedding(
            texts,
            embed_model="nomic-embed-text"
        )
    ),
)
```

* Increasing the `num_ctx` parameter:

1. Pull the model:
```python
ollama pull qwen2
```

2. Display the model file:
```python
ollama show --modelfile qwen2 > Modelfile
```

3. Edit the Modelfile by adding the following line:
```python
PARAMETER num_ctx 32768
```

4. Create the modified model:
```python
ollama create -f Modelfile qwen2m
```

</details>

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

### Graph Visualization

<details>
<summary> Graph visualization with html </summary>

* The following code can be found in `examples/graph_visual_with_html.py`

```python
import networkx as nx
from pyvis.network import Network

# Load the GraphML file
G = nx.read_graphml('./dickens/graph_chunk_entity_relation.graphml')

# Create a Pyvis network
net = Network(notebook=True)

# Convert NetworkX graph to Pyvis network
net.from_nx(G)

# Save and display the network
net.show('knowledge_graph.html')
```

</details>

<details>
<summary> Graph visualization with Neo4j </summary>

* The following code can be found in `examples/graph_visual_with_neo4j.py`

```python
import os
import json
from lightrag.utils import xml_to_json
from neo4j import GraphDatabase

# Constants
WORKING_DIR = "./dickens"
BATCH_SIZE_NODES = 500
BATCH_SIZE_EDGES = 100

# Neo4j connection credentials
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "your_password"

def convert_xml_to_json(xml_path, output_path):
    """Converts XML file to JSON and saves the output."""
    if not os.path.exists(xml_path):
        print(f"Error: File not found - {xml_path}")
        return None

    json_data = xml_to_json(xml_path)
    if json_data:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        print(f"JSON file created: {output_path}")
        return json_data
    else:
        print("Failed to create JSON data")
        return None

def process_in_batches(tx, query, data, batch_size):
    """Process data in batches and execute the given query."""
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        tx.run(query, {"nodes": batch} if "nodes" in query else {"edges": batch})

def main():
    # Paths
    xml_file = os.path.join(WORKING_DIR, 'graph_chunk_entity_relation.graphml')
    json_file = os.path.join(WORKING_DIR, 'graph_data.json')

    # Convert XML to JSON
    json_data = convert_xml_to_json(xml_file, json_file)
    if json_data is None:
        return

    # Load nodes and edges
    nodes = json_data.get('nodes', [])
    edges = json_data.get('edges', [])

    # Neo4j queries
    create_nodes_query = """
    UNWIND $nodes AS node
    MERGE (e:Entity {id: node.id})
    SET e.entity_type = node.entity_type,
        e.description = node.description,
        e.source_id = node.source_id,
        e.displayName = node.id  
    REMOVE e:Entity  
    WITH e, node
    CALL apoc.create.addLabels(e, [node.entity_type]) YIELD node AS labeledNode
    RETURN count(*)
    """

    create_edges_query = """
    UNWIND $edges AS edge
    MATCH (source {id: edge.source})
    MATCH (target {id: edge.target})
    WITH source, target, edge,
         CASE
            WHEN edge.keywords CONTAINS 'lead' THEN 'lead'
            WHEN edge.keywords CONTAINS 'participate' THEN 'participate'
            WHEN edge.keywords CONTAINS 'uses' THEN 'uses'
            WHEN edge.keywords CONTAINS 'located' THEN 'located'
            WHEN edge.keywords CONTAINS 'occurs' THEN 'occurs'
           ELSE REPLACE(SPLIT(edge.keywords, ',')[0], '\"', '')
         END AS relType
    CALL apoc.create.relationship(source, relType, {
      weight: edge.weight,
      description: edge.description,
      keywords: edge.keywords,
      source_id: edge.source_id
    }, target) YIELD rel
    RETURN count(*)
    """

    set_displayname_and_labels_query = """
    MATCH (n)
    SET n.displayName = n.id
    WITH n
    CALL apoc.create.setLabels(n, [n.entity_type]) YIELD node
    RETURN count(*)
    """

    # Create a Neo4j driver
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    try:
        # Execute queries in batches
        with driver.session() as session:
            # Insert nodes in batches
            session.execute_write(process_in_batches, create_nodes_query, nodes, BATCH_SIZE_NODES)

            # Insert edges in batches
            session.execute_write(process_in_batches, create_edges_query, edges, BATCH_SIZE_EDGES)

            # Set displayName and labels
            session.run(set_displayname_and_labels_query)

    except Exception as e:
        print(f"Error occurred: {e}")
    
    finally:
        driver.close()

if __name__ == "__main__":
    main()
```

</details>

## Evaluation
### Dataset
The dataset used in LightRAG can be downloaded from [TommyChien/UltraDomain](https://huggingface.co/datasets/TommyChien/UltraDomain).

### Generate Query
LightRAG uses the following prompt to generate high-level queries, with the corresponding code in `example/generate_query.py`.

<details>
<summary> Prompt </summary>

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
</details>

 ### Batch Eval
To evaluate the performance of two RAG systems on high-level queries, LightRAG uses the following prompt, with the specific code available in `example/batch_eval.py`.

<details>
<summary> Prompt </summary>

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
</details>

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

## Reproduce
All the code can be found in the `./reproduce` directory.

### Step-0 Extract Unique Contexts
First, we need to extract unique contexts in the datasets.

<details>
<summary> Code </summary>

```python
def extract_unique_contexts(input_directory, output_directory):

    os.makedirs(output_directory, exist_ok=True)

    jsonl_files = glob.glob(os.path.join(input_directory, '*.jsonl'))
    print(f"Found {len(jsonl_files)} JSONL files.")

    for file_path in jsonl_files:
        filename = os.path.basename(file_path)
        name, ext = os.path.splitext(filename)
        output_filename = f"{name}_unique_contexts.json"
        output_path = os.path.join(output_directory, output_filename)

        unique_contexts_dict = {}

        print(f"Processing file: {filename}")

        try:
            with open(file_path, 'r', encoding='utf-8') as infile:
                for line_number, line in enumerate(infile, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        json_obj = json.loads(line)
                        context = json_obj.get('context')
                        if context and context not in unique_contexts_dict:
                            unique_contexts_dict[context] = None
                    except json.JSONDecodeError as e:
                        print(f"JSON decoding error in file {filename} at line {line_number}: {e}")
        except FileNotFoundError:
            print(f"File not found: {filename}")
            continue
        except Exception as e:
            print(f"An error occurred while processing file {filename}: {e}")
            continue

        unique_contexts_list = list(unique_contexts_dict.keys())
        print(f"There are {len(unique_contexts_list)} unique `context` entries in the file {filename}.")

        try:
            with open(output_path, 'w', encoding='utf-8') as outfile:
                json.dump(unique_contexts_list, outfile, ensure_ascii=False, indent=4)
            print(f"Unique `context` entries have been saved to: {output_filename}")
        except Exception as e:
            print(f"An error occurred while saving to the file {output_filename}: {e}")

    print("All files have been processed.")

```
</details>

### Step-1 Insert Contexts
For the extracted contexts, we insert them into the LightRAG system.

<details>
<summary> Code </summary>

```python
def insert_text(rag, file_path):
    with open(file_path, mode='r') as f:
        unique_contexts = json.load(f)

    retries = 0
    max_retries = 3
    while retries < max_retries:
        try:
            rag.insert(unique_contexts)
            break
        except Exception as e:
            retries += 1
            print(f"Insertion failed, retrying ({retries}/{max_retries}), error: {e}")
            time.sleep(10)
    if retries == max_retries:
        print("Insertion failed after exceeding the maximum number of retries")
```
</details>

### Step-2 Generate Queries

We extract tokens from the first and the second half of each context in the dataset, then combine them as dataset descriptions to generate queries.

<details>
<summary> Code </summary>

```python
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def get_summary(context, tot_tokens=2000):
    tokens = tokenizer.tokenize(context)
    half_tokens = tot_tokens // 2

    start_tokens = tokens[1000:1000 + half_tokens]
    end_tokens = tokens[-(1000 + half_tokens):1000]

    summary_tokens = start_tokens + end_tokens
    summary = tokenizer.convert_tokens_to_string(summary_tokens)

    return summary
```
</details>

### Step-3 Query
For the queries generated in Step-2, we will extract them and query LightRAG.

<details>
<summary> Code </summary>

```python
def extract_queries(file_path):
    with open(file_path, 'r') as f:
        data = f.read()

    data = data.replace('**', '')

    queries = re.findall(r'- Question \d+: (.+)', data)

    return queries
```
</details>

## Code Structure

```python
.
├── examples
│   ├── batch_eval.py
│   ├── graph_visual_with_html.py
│   ├── graph_visual_with_neo4j.py
│   ├── generate_query.py
│   ├── lightrag_azure_openai_demo.py
│   ├── lightrag_bedrock_demo.py
│   ├── lightrag_hf_demo.py
│   ├── lightrag_ollama_demo.py
│   ├── lightrag_openai_compatible_demo.py
│   ├── lightrag_openai_demo.py
│   └── vram_management_demo.py
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
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── README.md
├── requirements.txt
└── setup.py
```

## Star History

<a href="https://star-history.com/#HKUDS/LightRAG&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=HKUDS/LightRAG&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=HKUDS/LightRAG&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=HKUDS/LightRAG&type=Date" />
 </picture>
</a>

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
