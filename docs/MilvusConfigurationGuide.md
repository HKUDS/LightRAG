# Milvus Configuration via vector_db_storage_cls_kwargs

## Overview

Milvus index parameters can be configured through `vector_db_storage_cls_kwargs`, which is the **recommended approach** for framework integration scenarios (e.g., when using RAGAnything or other frameworks built on top of LightRAG).

## Why Use vector_db_storage_cls_kwargs?

✅ **Framework Integration**: Allows configuration to be passed through framework layers without environment variable changes
✅ **Programmatic Configuration**: Set parameters in code rather than relying on environment variables
✅ **Dynamic Configuration**: Different configurations for different RAG instances
✅ **Clean API**: All parameters passed in one place during initialization

## Supported Parameters

All 11 MilvusIndexConfig parameters can be configured via `vector_db_storage_cls_kwargs`:

### Base Configuration
- `index_type`: Index type (AUTOINDEX, HNSW, HNSW_SQ, IVF_FLAT, etc.)
- `metric_type`: Distance metric (COSINE, L2, IP)

### HNSW Parameters
- `hnsw_m`: Number of connections per layer (2-2048, default: 30)
- `hnsw_ef_construction`: Size of dynamic candidate list during construction (default: 200)
- `hnsw_ef`: Size of dynamic candidate list during search (default: 100)

### HNSW_SQ Parameters (requires Milvus 2.6.8+)
- `sq_type`: Quantization type (SQ4U, SQ6, SQ8, BF16, FP16, default: SQ8)
- `sq_refine`: Enable refinement (default: False)
- `sq_refine_type`: Refinement type (SQ6, SQ8, BF16, FP16, FP32, default: FP32)
- `sq_refine_k`: Number of candidates to refine (default: 10)

### IVF Parameters
- `ivf_nlist`: Number of cluster units (1-65536, default: 1024)
- `ivf_nprobe`: Number of units to query (default: 16)

## Configuration Priority

Configuration is resolved in the following order:
1. **Parameters passed via vector_db_storage_cls_kwargs** (highest priority)
2. Environment variables (MILVUS_INDEX_TYPE, etc.)
3. Default values

## Usage Examples

### Basic Configuration

```python
from lightrag import LightRAG

rag = LightRAG(
    working_dir="./demo",
    vector_storage="MilvusVectorDBStorage",
    vector_db_storage_cls_kwargs={
        "cosine_better_than_threshold": 0.2,
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "hnsw_m": 32,
        "hnsw_ef_construction": 256,
        "hnsw_ef": 150,
    }
)
```

### RAGAnything Framework Integration

```python
# In RAGAnything framework code:
def create_lightrag_instance(user_config):
    """Create LightRAG instance with user-provided Milvus configuration"""

    # User configuration from RAGAnything
    milvus_config = {
        "cosine_better_than_threshold": user_config.get("threshold", 0.2),
        "index_type": user_config.get("index_type", "HNSW"),
        "hnsw_m": user_config.get("hnsw_m", 32),
        # ... other parameters
    }

    # Pass configuration to LightRAG
    rag = LightRAG(
        working_dir=user_config["working_dir"],
        vector_storage="MilvusVectorDBStorage",
        vector_db_storage_cls_kwargs=milvus_config,
    )

    return rag
```

### Advanced Configuration with HNSW_SQ

```python
rag = LightRAG(
    working_dir="./demo",
    vector_storage="MilvusVectorDBStorage",
    vector_db_storage_cls_kwargs={
        "cosine_better_than_threshold": 0.2,
        "index_type": "HNSW_SQ",  # Requires Milvus 2.6.8+
        "metric_type": "COSINE",
        "hnsw_m": 48,
        "hnsw_ef_construction": 400,
        "hnsw_ef": 200,
        "sq_type": "SQ8",
        "sq_refine": True,
        "sq_refine_type": "FP32",
        "sq_refine_k": 20,
    }
)
```

### IVF Configuration

```python
rag = LightRAG(
    working_dir="./demo",
    vector_storage="MilvusVectorDBStorage",
    vector_db_storage_cls_kwargs={
        "cosine_better_than_threshold": 0.2,
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "ivf_nlist": 2048,
        "ivf_nprobe": 32,
    }
)
```

## Implementation Details

### How It Works

1. When `MilvusVectorDBStorage.__post_init__()` is called:
   ```python
   kwargs = self.global_config.get("vector_db_storage_cls_kwargs", {})
   index_config_keys = MilvusIndexConfig.get_config_field_names()
   index_config_params = {
       k: v for k, v in kwargs.items() if k in index_config_keys
   }
   self.index_config = MilvusIndexConfig(**index_config_params)
   ```

2. `MilvusIndexConfig.get_config_field_names()` dynamically extracts all valid parameter names from the dataclass
3. Only valid Milvus index parameters are extracted from kwargs
4. Parameters are passed to `MilvusIndexConfig` which applies defaults and validates them
5. Environment variables are used as fallback for any parameters not provided in kwargs

### Automatic Synchronization

The implementation uses `MilvusIndexConfig.get_config_field_names()` to dynamically extract valid parameters. This means:
- ✅ New parameters added to `MilvusIndexConfig` are **automatically recognized**
- ✅ No need to maintain duplicate parameter lists
- ✅ Single source of truth for configuration parameters

## Testing

The configuration via `vector_db_storage_cls_kwargs` is thoroughly tested:

```bash
# Run all kwargs bridge tests
python -m pytest tests/test_milvus_kwargs_bridge.py -v

# Test RAGAnything integration scenario specifically
python -m pytest tests/test_milvus_kwargs_bridge.py::TestMilvusKwargsParameterBridge::test_raganything_framework_integration_scenario -v

# Test all parameters support
python -m pytest tests/test_milvus_kwargs_bridge.py::TestMilvusKwargsParameterBridge::test_all_milvus_parameters_supported_via_kwargs -v
```

## Examples

See `examples/milvus_kwargs_configuration_demo.py` for a complete working example.

## Backward Compatibility

✅ **100% backward compatible** with existing code
✅ Environment variable configuration still works
✅ All existing tests pass

## FAQ

### Q: Can I mix kwargs and environment variables?
**A:** Yes! Parameters in `vector_db_storage_cls_kwargs` take priority over environment variables.

### Q: What happens to non-Milvus parameters in kwargs?
**A:** They are ignored. Only valid MilvusIndexConfig parameters are extracted. This allows frameworks to pass their own parameters alongside Milvus configuration.

### Q: Do I need to set environment variables?
**A:** No! When using `vector_db_storage_cls_kwargs`, environment variables are optional. They serve as fallback values.

### Q: Is this approach recommended for RAGAnything?
**A:** Yes! This is the **recommended approach** for any framework that builds on top of LightRAG, as it allows clean configuration passing through framework layers.

## References

- Test Suite: `tests/test_milvus_kwargs_bridge.py`
- Implementation: `lightrag/kg/milvus_impl.py` (lines 1237-1272)
- Example: `examples/milvus_kwargs_configuration_demo.py`
- MilvusIndexConfig: `lightrag/kg/milvus_impl.py` (lines 75-303)
