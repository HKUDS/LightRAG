import tempfile
from pathlib import Path

# Use a temporary directory for the logs
LOG_DIR = Path(tempfile.gettempdir()) / "lightrag_tests"
LOG_DIR.mkdir(exist_ok=True)

# Use a temporary directory for the working directory
WORKING_DIR = Path(tempfile.gettempdir()) / "lightrag_tests_working_dir"
WORKING_DIR.mkdir(exist_ok=True)

# Use a temporary directory for the input directory
INPUT_DIR = Path(tempfile.gettempdir()) / "lightrag_tests_input_dir"
INPUT_DIR.mkdir(exist_ok=True)

# Define the global arguments for the tests
global_args = {
    "log_level": "INFO",
    "verbose": False,
    "host": "127.0.0.1",
    "port": 8000,
    "workers": 1,
    "cors_origins": "*",
    "ssl": False,
    "ssl_certfile": None,
    "ssl_keyfile": None,
    "simulated_model_name": "test_model",
    "simulated_model_tag": "latest",
    "history_turns": 5,
    "key": None,
    "auth_accounts": None,
    "working_dir": str(WORKING_DIR),
    "input_dir": str(INPUT_DIR),
    "llm_binding": "openai",
    "llm_binding_host": "http://localhost:8080",
    "llm_model": "test_model",
    "temperature": 0.7,
    "max_async": 4,
    "max_tokens": 1024,
    "timeout": 300,
    "enable_llm_cache": True,
    "enable_llm_cache_for_extract": True,
    "embedding_binding": "openai",
    "embedding_binding_host": "http://localhost:8080",
    "embedding_model": "test_embedding_model",
    "embedding_dim": 768,
    "summary_language": "English",
    "max_parallel_insert": 2,
    "chunk_size": 1200,
    "chunk_overlap_size": 100,
    "cosine_threshold": 0.2,
    "top_k": 5,
    "force_llm_summary_on_merge": 0,
    "kv_storage": "JsonKVStorage",
    "vector_storage": "NanoVectorDBStorage",
    "graph_storage": "NetworkXStorage",
    "doc_status_storage": "JsonDocStatusStorage",
    "workspace": "test_workspace",
    "rerank_binding_api_key": None,
    "rerank_binding_host": None,
    "rerank_model": None,
    "min_rerank_score": 0.5,
    "related_chunk_number": 3,
    "embedding_func_max_async": 8,
    "embedding_batch_num": 10,
    "max_graph_nodes": 1000,
    "auto_scan_at_startup": False,
    "whitelist_paths": "/health,/docs,/openapi.json,/redoc",
    "testing": True,
}
