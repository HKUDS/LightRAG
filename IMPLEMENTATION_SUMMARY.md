# MilvusIndexConfig Parameter Bridging - Implementation Summary

## Overview
This PR enables configuration of Milvus index parameters through `vector_db_storage_cls_kwargs`, allowing programmatic configuration instead of relying solely on environment variables.

## Problem Addressed
Previously, `MilvusIndexConfig` could only be configured via environment variables because `MilvusVectorDBStorage.__post_init__()` always created `MilvusIndexConfig()` without parameters. This made it difficult for applications like RAGAnything to configure Milvus index settings programmatically.

## Solution
Modified `lightrag/kg/milvus_impl.py` to:
1. Extract index configuration parameters from `vector_db_storage_cls_kwargs`
2. Pass them to `MilvusIndexConfig` constructor
3. Update fallback index creation to use configured values

## Changes Made

### 1. MilvusVectorDBStorage.__post_init__() (lines 1220-1236)
```python
# Extract MilvusIndexConfig parameters from vector_db_storage_cls_kwargs
kwargs = self.global_config.get("vector_db_storage_cls_kwargs", {})
index_config_keys = {
    "index_type", "metric_type",
    "hnsw_m", "hnsw_ef_construction", "hnsw_ef",
    "sq_type", "sq_refine", "sq_refine_type", "sq_refine_k",
    "ivf_nlist", "ivf_nprobe",
}
index_config_params = {k: v for k, v in kwargs.items() if k in index_config_keys}

# Initialize index configuration (if not already set)
# Priority: init params from kwargs > environment variables > defaults
if not hasattr(self, "index_config") or self.index_config is None:
    self.index_config = MilvusIndexConfig(**index_config_params)
```

### 2. _create_vector_index_fallback() (lines 431-450)
```python
index_params={
    "index_type": self.index_config.index_type if self.index_config.index_type != "AUTOINDEX" else "HNSW",
    "metric_type": self.index_config.metric_type,
    "params": {"M": self.index_config.hnsw_m, "efConstruction": self.index_config.hnsw_ef_construction},
}
```

### 3. New Test Suite
- Created `tests/test_milvus_kwargs_bridge.py` with 8 comprehensive tests
- Tests cover HNSW, IVF, HNSW_SQ parameter passing
- Validates backward compatibility
- Confirms parameter priority (code > env > defaults)

## Usage Examples

### Basic HNSW Configuration
```python
from lightrag import LightRAG

rag = LightRAG(
    working_dir="./rag_storage",
    vector_storage="MilvusVectorDBStorage",
    vector_db_storage_cls_kwargs={
        "cosine_better_than_threshold": 0.3,
        "hnsw_m": 32,
        "hnsw_ef": 256,
        "hnsw_ef_construction": 300,
    }
)
```

### IVF Configuration
```python
rag = LightRAG(
    vector_storage="MilvusVectorDBStorage",
    vector_db_storage_cls_kwargs={
        "cosine_better_than_threshold": 0.3,
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "ivf_nlist": 2048,
    }
)
```

### HNSW_SQ with Quantization
```python
rag = LightRAG(
    vector_storage="MilvusVectorDBStorage",
    vector_db_storage_cls_kwargs={
        "cosine_better_than_threshold": 0.3,
        "index_type": "HNSW_SQ",
        "sq_type": "SQ8",
        "sq_refine": True,
        "sq_refine_type": "FP16",
    }
)
```

## Supported Parameters
All MilvusIndexConfig parameters can be passed through `vector_db_storage_cls_kwargs`:
- `index_type`: Index type (HNSW, IVF_FLAT, HNSW_SQ, etc.)
- `metric_type`: Metric type (COSINE, L2, IP)
- `hnsw_m`: HNSW M parameter
- `hnsw_ef_construction`: HNSW efConstruction parameter
- `hnsw_ef`: HNSW ef search parameter
- `sq_type`: Scalar quantization type (SQ4U, SQ6, SQ8, BF16, FP16)
- `sq_refine`: Enable refinement
- `sq_refine_type`: Refinement type
- `sq_refine_k`: Refinement k value
- `ivf_nlist`: IVF nlist parameter
- `ivf_nprobe`: IVF nprobe parameter

## Backward Compatibility
✅ **100% backward compatible**
- When no index parameters provided: Uses defaults (environment variables → built-in defaults)
- Existing code continues to work without changes
- Priority order: code parameters > environment variables > defaults

## Test Results
- ✅ 8/8 new tests passing
- ✅ 39/39 existing Milvus tests passing
- ✅ Total: 47/47 Milvus tests passing

## Benefits
1. **Programmatic Configuration**: Configure index parameters directly in code
2. **Flexibility**: No need to set environment variables for every deployment
3. **Type Safety**: Parameters validated by MilvusIndexConfig
4. **Backward Compatible**: Existing deployments unaffected
5. **Clear Priority**: Code > Environment > Defaults

## Files Modified
- `lightrag/kg/milvus_impl.py`: 21 lines changed (15 additions, 6 deletions)
- `tests/test_milvus_kwargs_bridge.py`: 258 lines added (new file)

## Validation
All changes have been thoroughly tested:
1. ✅ Unit tests for parameter extraction and passing
2. ✅ Backward compatibility tests
3. ✅ Parameter priority tests
4. ✅ Fallback method integration tests
5. ✅ All existing Milvus tests pass
