<center><h2>🚀 REASONRAG: Simple and Fast Retrieval-Augmented Generation</h2></center>



## Install

* Install from source (Recommend)

```bash
cd REASONRAG
pip install -e .
```
* Install from PyPI
```bash
pip install REASONrag-hku
```

## Quick Start
* [Video demo](https://www.youtube.com/watch?v=g21royNJ4fw) of running REASONRAG locally.
* All the code can be found in the `examples`.
* Set OpenAI API key in environment if using OpenAI models: `export OPENAI_API_KEY="sk-...".`
## Code Structure

```python
.
├── examples
│   ├── batch_eval.py
│   ├── generate_query.py
│   ├── graph_visual_with_html.py
│   ├── graph_visual_with_neo4j.py
│   ├── REASONrag_api_openai_compatible_demo.py
│   ├── REASONrag_azure_openai_demo.py
│   ├── REASONrag_bedrock_demo.py
│   ├── REASONrag_hf_demo.py
│   ├── REASONrag_lmdeploy_demo.py
│   ├── REASONrag_ollama_demo.py
│   ├── REASONrag_openai_compatible_demo.py
│   ├── REASONrag_openai_demo.py
│   ├── REASONrag_siliconcloud_demo.py
│   └── vram_management_demo.py
├── REASONrag
│   ├── kg
│   │   ├── __init__.py
│   │   └── neo4j_impl.py
│   ├── __init__.py
│   ├── base.py
│   ├── REASONrag.py
│   ├── llm.py
│   ├── operate.py
│   ├── prompt.py
│   ├── storage.py
│   └── utils.py
├── reproduce
│   ├── Step_0.py
│   ├── Step_1_openai_compatible.py
│   ├── Step_1.py
│   ├── Step_2.py
│   ├── Step_3_openai_compatible.py
│   └── Step_3.py
├── .gitignore
├── .pre-commit-config.yaml
├── Dockerfile
├── get_all_edges_nx.py
├── LICENSE
├── README.md
├── requirements.txt
├── setup.py
├── test_neo4j.py
└── test.py
```

## Star History

<a href="https://star-history.com/#HKUDS/REASONRAG&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=HKUDS/REASONRAG&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: REASON)" srcset="https://api.star-history.com/svg?repos=HKUDS/REASONRAG&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=HKUDS/REASONRAG&type=Date" />
 </picture>
</a>

## Contribution

Thank you to all our contributors!

<a href="https://github.com/HKUDS/REASONRAG/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=HKUDS/REASONRAG" />
</a>

## 🌟Citation

```python
@article{guo2024REASONrag,
title={REASONRAG: Simple and Fast Retrieval-Augmented Generation},
author={Zirui Guo and Lianghao Xia and Yanhua Yu and Tu Ao and Chao Huang},
year={2024},
eprint={2410.05779},
archivePrefix={arXiv},
primaryClass={cs.IR}
}
```
**Thank you for your interest in our work!**
