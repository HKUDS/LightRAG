# LightRAG LLM Integrations (`lightrag/llm`)

This document provides an overview of the Large Language Model (LLM) integrations available within the `lightrag/llm` subdirectory. Each module facilitates interaction with a specific LLM provider or framework, typically offering standardized interfaces for text completion and embedding generation.

A common pattern across these modules is the use of retry logic, often implemented via a `@retry` decorator, to enhance the robustness of API calls against transient network issues or provider-side errors. Many modules also implement caching mechanisms (e.g., functions like `_call_if_cache`, `_embed_if_cache`) to optimize performance and reduce costs by reusing previous results.

## Modules Overview

### `anthropic.py`
*   **Provider/Framework:** Anthropic (Claude models)
*   **Purpose:** Integrates with Anthropic's API.
*   **Main Functions:** Provides functions for text completion (`_call`) and potentially embedding generation (`_embed`), along with their caching variants if implemented.

### `azure_openai.py`
*   **Provider/Framework:** Microsoft Azure OpenAI Service
*   **Purpose:** Integrates with Azure's hosted OpenAI models.
*   **Main Functions:** Offers functions for text completion (`_call`) and embedding generation (`_embed`), often mirroring the OpenAI API structure, including caching variants.

### `bedrock.py`
*   **Provider/Framework:** Amazon Web Services (AWS) Bedrock
*   **Purpose:** Integrates with AWS Bedrock, which provides access to various foundation models.
*   **Main Functions:** Includes functions for invoking different models available through Bedrock for tasks like completion (`_call`) and embeddings (`_embed`), plus caching.

### `hf.py`
*   **Provider/Framework:** Hugging Face
*   **Purpose:** Integrates with Hugging Face models, potentially using libraries like `transformers` or the Inference API.
*   **Main Functions:** Provides methods for text generation/completion (`_call`) and embedding (`_embed`) using Hugging Face models, with caching.

### `jina.py`
*   **Provider/Framework:** Jina AI
*   **Purpose:** Integrates with Jina AI's embedding models.
*   **Main Functions:** Primarily focuses on embedding generation (`_embed`) using Jina AI's specialized models, likely including caching.

### `llama_index_impl.py`
*   **Provider/Framework:** LlamaIndex
*   **Purpose:** Acts as a wrapper or interface to utilize LLMs configured within the LlamaIndex framework.
*   **Main Functions:** Provides standardized `_call` and `_embed` functions that delegate the actual LLM interaction to the underlying LlamaIndex setup, potentially leveraging LlamaIndex's own caching and retry mechanisms.

### `lmdeploy.py`
*   **Provider/Framework:** LMDeploy (by OpenMMLab)
*   **Purpose:** Integrates with models served via the LMDeploy inference framework.
*   **Main Functions:** Offers `_call` for text completion and potentially `_embed` for embeddings, interacting with an LMDeploy service endpoint, including caching.

### `lollms.py`
*   **Provider/Framework:** LoLLMs (Lord of Large Language Models)
*   **Purpose:** Integrates with the LoLLMs framework, which supports various local and remote models.
*   **Main Functions:** Provides `_call` and `_embed` interfaces to interact with models managed by a LoLLMs instance, with caching.

### `nvidia_openai.py`
*   **Provider/Framework:** NVIDIA NIM (via OpenAI-compatible endpoint)
*   **Purpose:** Integrates with NVIDIA NIM inference microservices using their OpenAI-compatible API endpoint.
*   **Main Functions:** Offers standard OpenAI-like functions for completion (`_call`) and embeddings (`_embed`), tailored for NVIDIA's service, including caching.

### `ollama.py`
*   **Provider/Framework:** Ollama
*   **Purpose:** Integrates with locally running LLMs served via the Ollama tool.
*   **Main Functions:** Provides `_call` for completion and `_embed` for embeddings by interacting with the local Ollama API endpoint, with caching.

### `openai.py`
*   **Provider/Framework:** OpenAI
*   **Purpose:** Integrates directly with the OpenAI API (e.g., GPT models).
*   **Main Functions:** Offers the core functions for text completion (`_call`) and embedding generation (`_embed`) using OpenAI's models, along with caching variants.

### `siliconcloud.py`
*   **Provider/Framework:** SiliconCloud
*   **Purpose:** Integrates with LLM services provided by SiliconCloud.
*   **Main Functions:** Provides `_call` and `_embed` functions to interact with SiliconCloud's API endpoints, including caching.

### `zhipu.py`
*   **Provider/Framework:** Zhipu AI (ChatGLM models)
*   **Purpose:** Integrates with Zhipu AI's API.
*   **Main Functions:** Offers functions for text completion (`_call`) and potentially embeddings (`_embed`) using Zhipu's models, with caching.