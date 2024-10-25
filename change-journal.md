## 🎉 News
- [!] [2024.10.25]🎯🎯📢📢We’ve added a new streamlit template for LightRAG.
- [x] [2024.10.20]🎯🎯📢📢We’ve added a new feature to LightRAG: Graph Visualization.
- [x] [2024.10.18]🎯🎯📢📢We’ve added a link to a [LightRAG Introduction Video](https://youtu.be/oageL-1I0GE). Thanks to the author!
- [x] [2024.10.17]🎯🎯📢📢We have created a [Discord channel](https://discord.gg/mvsfu2Tg)! Welcome to join for sharing and discussions! 🎉🎉
- [x] [2024.10.16]🎯🎯📢📢LightRAG now supports [Ollama models](https://github.com/HKUDS/LightRAG?tab=readme-ov-file#quick-start)!
- [x] [2024.10.15]🎯🎯📢📢LightRAG now supports [Hugging Face models](https://github.com/HKUDS/LightRAG?tab=readme-ov-file#quick-start)!

## Streamlit Template
- [ ] [[2024-10-25]] Make a streamlit template for LightRAG
    - [x] Add a sidebar for document upload, model selection, and API key input
    - [x] Add a main page for query input and result display
    - [x] Add a graph visualization page for the knowledge graph

We have added a new Streamlit template for LightRAG. You can run it directly using (activate .env):

```bash
python3 -m streamlit run example/lightrag_streamlit_demo.py
```

### Streamlit Design
- Sidebar
    - ### Configuration
        - OpenAI API Key
        - Model Name
    - ### Knowledge Graph
        - (Button) Add Document (Upload or Paste)
        - Table of Contents

- Main container
    - Graph Visualization
    - GPTChat Interface


### Knowledge Graph Options
1. Insert text, markdown or PDF file into KGraph, refresh sidebar with new document
2. Add a option to select the AI Model and API Key if not ollama
