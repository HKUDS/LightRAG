# LightRAG Core Algorithm Overview

**Related Documents**: [System Architecture](SYSTEM_ARCHITECTURE_AND_DATA_FLOW.md) | [Concurrent Processing](LightRAG_concurrent_explain.md) | [Production Guide](../production/PRODUCTION_DEPLOYMENT_COMPLETE.md) | [Documentation Index](../DOCUMENTATION_INDEX.md)

This document provides a visual overview of LightRAG's core algorithms for document indexing and retrieval.

> **💡 For Implementation Details**: See the [System Architecture and Data Flow](SYSTEM_ARCHITECTURE_AND_DATA_FLOW.md) document for comprehensive technical details.
> **💡 For Production Setup**: See the [Complete Production Deployment Guide](../production/PRODUCTION_DEPLOYMENT_COMPLETE.md) for deployment instructions.

## Document Indexing Process

![LightRAG Indexing Flowchart](https://learnopencv.com/wp-content/uploads/2024/11/LightRAG-VectorDB-Json-KV-Store-Indexing-Flowchart-scaled.jpg)

*Figure 1: LightRAG Indexing Flowchart - Shows the complete document processing pipeline from input to storage distribution. [Source](https://learnopencv.com/lightrag/)*

### Key Indexing Stages
1. **Document Input** - Multiple input methods (API, Web UI, MCP tools)
2. **Text Processing** - Chunking, cleaning, preprocessing
3. **Knowledge Extraction** - Entity and relationship identification using LLMs
4. **Storage Distribution** - Multi-backend storage (KV, Vector, Graph, Status)

**Related**: [Enhanced Document Processing](../production/PRODUCTION_DEPLOYMENT_COMPLETE.md#enhanced-document-processing) with Docling service

## Query and Retrieval Process

![LightRAG Retrieval and Querying Flowchart](https://learnopencv.com/wp-content/uploads/2024/11/LightRAG-Querying-Flowchart-Dual-Level-Retrieval-Generation-Knowledge-Graphs-scaled.jpg)

*Figure 2: LightRAG Retrieval and Querying Flowchart - Demonstrates dual-level retrieval combining knowledge graphs and vector search. [Source](https://learnopencv.com/lightrag/)*

### Query Modes Available
- **Naive**: Basic vector search
- **Local**: Context-dependent retrieval
- **Global**: Knowledge graph-based retrieval
- **Hybrid**: Combines local and global methods *(Recommended)*
- **Mix**: Integrates knowledge graph and vector retrieval
- **Bypass**: Direct LLM query without RAG

**Related**: [MCP Query Tools](../integration_guides/MCP_IMPLEMENTATION_SUMMARY.md) for Claude CLI integration

## Performance Considerations

The algorithms support:
- **Concurrent Processing**: See [Concurrent Processing Details](LightRAG_concurrent_explain.md)
- **Multiple Storage Backends**: See [Storage Architecture](SYSTEM_ARCHITECTURE_AND_DATA_FLOW.md#storage-backend-coordination)
- **Horizontal Scaling**: See [Production Scaling](../production/PRODUCTION_DEPLOYMENT_COMPLETE.md#maintenance--scaling)

## Algorithm Optimization

For production environments, consider:
- **Chunking Strategy**: Optimize chunk size for your use case
- **Entity Extraction**: Balance quality vs. performance with gleaning iterations
- **Query Mode Selection**: Hybrid mode provides best results for most use cases
- **Caching**: Multi-level caching significantly improves performance

**Implementation Guide**: [Performance Tuning](../production/PRODUCTION_DEPLOYMENT_COMPLETE.md#performance-tuning)
