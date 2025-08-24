<div align="center">

<div style="margin: 20px 0;">
  <img src="./assets/logo.png" width="120" height="120" alt="LightRAG Logo" style="border-radius: 20px; box-shadow: 0 8px 32px rgba(0, 217, 255, 0.3);">
</div>

# 🚀 LightRAG: Simple and Fast Retrieval-Augmented Generation

<div align="center">
    <a href="https://trendshift.io/repositories/13043" target="_blank"><img src="https://trendshift.io/api/badge/repositories/13043" alt="HKUDS%2FLightRAG | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</div>

<div align="center">
  <div style="width: 100%; height: 2px; margin: 20px 0; background: linear-gradient(90deg, transparent, #00d9ff, transparent);"></div>
</div>

<div align="center">
  <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; padding: 25px; text-align: center;">
    <p>
      <a href='https://github.com/HKUDS/LightRAG'><img src='https://img.shields.io/badge/🔥Project-Page-00d9ff?style=for-the-badge&logo=github&logoColor=white&labelColor=1a1a2e'></a>
      <a href='https://arxiv.org/abs/2410.05779'><img src='https://img.shields.io/badge/📄arXiv-2410.05779-ff6b6b?style=for-the-badge&logo=arxiv&logoColor=white&labelColor=1a1a2e'></a>
      <a href="https://github.com/HKUDS/LightRAG/stargazers"><img src='https://img.shields.io/github/stars/HKUDS/LightRAG?color=00d9ff&style=for-the-badge&logo=star&logoColor=white&labelColor=1a1a2e' /></a>
    </p>
    <p>
      <img src="https://img.shields.io/badge/🐍Python-3.10-4ecdc4?style=for-the-badge&logo=python&logoColor=white&labelColor=1a1a2e">
      <a href="https://pypi.org/project/lightrag-hku/"><img src="https://img.shields.io/pypi/v/lightrag-hku.svg?style=for-the-badge&logo=pypi&logoColor=white&labelColor=1a1a2e&color=ff6b6b"></a>
    </p>
    <p>
      <a href="https://discord.gg/yF2MmDJyGJ"><img src="https://img.shields.io/badge/💬Discord-Community-7289da?style=for-the-badge&logo=discord&logoColor=white&labelColor=1a1a2e"></a>
      <a href="https://github.com/HKUDS/LightRAG/issues/285"><img src="https://img.shields.io/badge/💬WeChat-Group-07c160?style=for-the-badge&logo=wechat&logoColor=white&labelColor=1a1a2e"></a>
    </p>
    <p>
      <a href="README-zh.md"><img src="https://img.shields.io/badge/🇨🇳中文版-1a1a2e?style=for-the-badge"></a>
      <a href="README.md"><img src="https://img.shields.io/badge/🇺🇸English-1a1a2e?style=for-the-badge"></a>
    </p>
  </div>
</div>

</div>

<div align="center" style="margin: 30px 0;">
  <img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="800">
</div>

<div align="center" style="margin: 30px 0;">
    <img src="./README.assets/b2aaf634151b4706892693ffb43d9093.png" width="800" alt="LightRAG Diagram">
</div>

---

## 🌟 WorldClass RAG Module Extension

**NEW ADDITION**: Un módulo RAG (Retrieval-Augmented Generation) de clase mundial diseñado para integrarse fácilmente en cualquier desarrollo, siguiendo las mejores prácticas del estado del arte y extendiendo las capacidades de LightRAG.

### ✨ Características Avanzadas del WorldClass RAG Module
- **🧩 Chunking Inteligente**: Múltiples estrategias (semántico, recursivo, basado en oraciones) con superposición optimizada
- **🔍 Búsqueda Híbrida**: Combina búsqueda semántica y por palabras clave con re-ranking automático
- **📄 Procesamiento Multimodal**: Soporte para texto, imágenes, tablas y documentos complejos
- **🕸️ Graph RAG**: Preservación de relaciones entre entidades usando grafos de conocimiento
- **💾 Gestión de Memoria**: Sistema avanzado para conversaciones largas y contexto persistente
- **📊 Evaluación Continua**: Métricas automáticas de relevancia, fidelidad, calidad y latencia

### 🚀 Facilidad de Integración
- **🔌 API REST**: Interfaz HTTP estándar para cualquier lenguaje
- **🐍 SDK Python**: Cliente nativo con todas las funcionalidades
- **⚙️ Configuración Declarativa**: Setup mediante archivos YAML/JSON
- **🐳 Docker Ready**: Contenedores optimizados para producción
- **📈 Escalabilidad**: Diseño preparado para millones de consultas

### 🔒 Seguridad Empresarial
- **🔐 Control de Acceso**: Autenticación y autorización granular
- **🛡️ Filtrado PII**: Detección y limpieza automática de información sensible
- **📋 Auditoría**: Logs completos para cumplimiento normativo
- **🔒 Encriptación**: Datos en tránsito y en reposo protegidos

### 🏗️ Arquitectura WorldClass RAG

```
worldclass-rag/
├── core/                   # Núcleo del sistema RAG
│   ├── chunking/          # Estrategias de segmentación
│   ├── embeddings/        # Modelos de embeddings
│   ├── retrieval/         # Sistema de recuperación
│   ├── generation/        # Generación de respuestas
│   └── evaluation/        # Métricas y evaluación
├── processors/            # Procesadores de documentos
│   ├── text/             # Procesamiento de texto
│   ├── pdf/              # Manejo de PDFs
│   ├── images/           # Procesamiento de imágenes
│   └── tables/           # Extracción de tablas
├── storage/              # Sistemas de almacenamiento
│   ├── vector/           # Bases de datos vectoriales
│   ├── graph/            # Almacén de grafos
│   └── cache/            # Sistema de caché
├── api/                  # API REST y WebSocket
├── sdk/                  # SDK Python
├── config/               # Configuraciones
├── security/             # Módulos de seguridad
└── monitoring/           # Observabilidad y métricas
```

### 🔧 Quick Start - WorldClass RAG

```python
from worldclass_rag import RAGEngine, SemanticChunker, HybridRetriever

# Inicializar motor RAG
rag = RAGEngine(
    use_local_embeddings=True,  # Sin API keys necesarias
    chunking_strategy="semantic", # Chunking inteligente  
    retrieval_method="hybrid"     # Búsqueda híbrida
)

# Procesar documentos
documents = ["Tu contenido aquí..."]
rag.add_documents(documents)

# Realizar consultas
results = rag.query("¿Qué es RAG?")
print(results.response)
```

---

## 🎉 News

- [X] **[2025.01.15] 🌟 NEW**: WorldClass RAG Module added with advanced chunking strategies, hybrid search, and enterprise features
- [X] [2025.06.16]🎯📢Our team has released [RAG-Anything](https://github.com/HKUDS/RAG-Anything) an All-in-One Multimodal RAG System for seamless text, image, table, and equation processing.
- [X] [2025.06.05]🎯📢LightRAG now supports comprehensive multimodal data handling through [RAG-Anything](https://github.com/HKUDS/RAG-Anything) integration, enabling seamless document parsing and RAG capabilities across diverse formats including PDFs, images, Office documents, tables, and formulas. Please refer to the new [multimodal section](https://github.com/HKUDS/LightRAG/?tab=readme-ov-file#multimodal-document-processing-rag-anything-integration) for details.
- [X] [2025.03.18]🎯📢LightRAG now supports citation functionality, enabling proper source attribution.
- [X] [2025.02.05]🎯📢Our team has released [VideoRAG](https://github.com/HKUDS/VideoRAG) understanding extremely long-context videos.
- [X] [2025.01.13]🎯📢Our team has released [MiniRAG](https://github.com/HKUDS/MiniRAG) making RAG simpler with small models.
- [X] [2025.01.06]🎯📢You can now [use PostgreSQL for Storage](#using-postgresql-for-storage).
- [X] [2024.12.31]🎯📢LightRAG now supports [deletion by document ID](https://github.com/HKUDS/LightRAG?tab=readme-ov-file#delete).
- [X] [2024.11.25]🎯📢LightRAG now supports seamless integration of [custom knowledge graphs](https://github.com/HKUDS/LightRAG?tab=readme-ov-file#insert-custom-kg), empowering users to enhance the system with their own domain expertise.
- [X] [2024.11.19]🎯📢A comprehensive guide to LightRAG is now available on [LearnOpenCV](https://learnopencv.com/lightrag). Many thanks to the blog author.
- [X] [2024.11.11]🎯📢LightRAG now supports [deleting entities by their names](https://github.com/HKUDS/LightRAG?tab=readme-ov-file#delete).
- [X] [2024.11.09]🎯📢Introducing the [LightRAG Gui](https://lightrag-gui.streamlit.app), which allows you to insert, query, visualize, and download LightRAG knowledge.
- [X] [2024.11.04]🎯📢You can now [use Neo4J for Storage](https://github.com/HKUDS/LightRAG?tab=readme-ov-file#using-neo4j-for-storage).
- [X] [2024.10.29]🎯📢LightRAG now supports multiple file types, including PDF, DOC, PPT, and CSV via `textract`.
- [X] [2024.10.20]🎯📢We've added a new feature to LightRAG: Graph Visualization.
- [X] [2024.10.18]🎯📢We've added a link to a [LightRAG Introduction Video](https://youtu.be/oageL-1I0GE). Thanks to the author!
- [X] [2024.10.17]🎯📢We have created a [Discord channel](https://discord.gg/yF2MmDJyGJ)! Welcome to join for sharing and discussions! 🎉🎉
- [X] [2024.10.16]🎯📢LightRAG now supports [Ollama models](https://github.com/HKUDS/LightRAG?tab=readme-ov-file#quick-start)!
- [X] [2024.10.15]🎯📢LightRAG now supports [Hugging Face models](https://github.com/HKUDS/LightRAG?tab=readme-ov-file#quick-start)!

<details>
  <summary style="font-size: 1.4em; font-weight: bold; cursor: pointer; display: list-item;">
    Algorithm Flowchart
  </summary>

![LightRAG Indexing Flowchart](https://learnopencv.com/wp-content/uploads/2024/11/LightRAG-VectorDB-Json-KV-Store-Indexing-Flowchart-scaled.jpg)
*Figure 1: LightRAG Indexing Flowchart - Img Caption : [Source](https://learnopencv.com/lightrag/)*
![LightRAG Retrieval and Querying Flowchart](https://learnopencv.com/wp-content/uploads/2024/11/LightRAG-Querying-Flowchart-Dual-Level-Retrieval-Generation-Knowledge-Graphs-scaled.jpg)
*Figure 2: LightRAG Retrieval and Querying Flowchart - Img Caption : [Source](https://learnopencv.com/lightrag/)*

</details>

## Installation

### Install LightRAG Server

The LightRAG Server is designed to provide Web UI and API support. The Web UI facilitates document indexing, knowledge graph exploration, and a simple RAG query interface. LightRAG Server also provide an Ollama compatible interfaces, aiming to emulate LightRAG as an Ollama chat model. This allows AI chat bot, such as Open WebUI, to access LightRAG easily.

* Install from PyPI

```bash
pip install "lightrag-hku[api]"
cp env.example .env
lightrag-server
```

* Installation from Source

```bash
git clone https://github.com/HKUDS/LightRAG.git
cd LightRAG
# create a Python virtual enviroment if neccesary
# Install in editable mode with API support
pip install -e ".[api]"
cp env.example .env
lightrag-server
```

* Launching the LightRAG Server with Docker Compose

```
git clone https://github.com/HKUDS/LightRAG.git
cd LightRAG
cp env.example .env
# modify LLM and Embedding settings in .env
docker compose up
```

> Historical versions of LightRAG docker images can be found here: [LightRAG Docker Images]( https://github.com/HKUDS/LightRAG/pkgs/container/lightrag)

### Install WorldClass RAG Extension

```bash
# Checkout to genspark_ai_developer branch for WorldClass RAG
git checkout genspark_ai_developer

# Install WorldClass RAG dependencies
pip install -r requirements.txt

# Install in editable mode
pip install -e .
```

### Install LightRAG Core

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

### Using LightRAG

```python
import os
from lightrag import LightRAG, QueryParam
from lightrag.llm import gpt_4o_mini_complete, gpt_4o_complete

WORKING_DIR = "./dickens"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=gpt_4o_mini_complete,  # Use gpt_4o_mini_complete LLM model
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

### Using WorldClass RAG

```python
from worldclass_rag import RAGEngine, SemanticChunker, HybridRetriever

# Initialize WorldClass RAG Engine
rag = RAGEngine(
    use_local_embeddings=True,
    chunking_strategy="semantic",
    retrieval_method="hybrid"
)

# Add documents
documents = [
    "Your document content here...",
    "Another document...",
]

rag.add_documents(documents)

# Query with advanced features
result = rag.query("What is RAG?")
print(f"Answer: {result.response}")
print(f"Sources: {result.sources}")
print(f"Confidence: {result.confidence}")

# Evaluate performance
evaluation = rag.evaluate_query("What is RAG?")
print(f"Relevance: {evaluation.relevance}")
print(f"Quality: {evaluation.quality}")
```

---

> **Note**: This repository now includes both the original LightRAG functionality and the new WorldClass RAG module extension. Switch to the `genspark_ai_developer` branch to access the WorldClass RAG features.