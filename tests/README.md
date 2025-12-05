# LightRAG Test Suite Index

This directory contains organized test suites for LightRAG.

## Test Suites

### 📁 gpt5_nano_compatibility/
Comprehensive test suite for gpt-5-nano model compatibility and configuration validation.

**Contents:**
- `test_gpt5_nano_compatibility.py` - Primary compatibility test suite (5 tests)
- `test_env_config.py` - .env configuration validation (6 tests)
- `test_direct_gpt5nano.py` - Direct API testing
- `test_gpt5_reasoning.py` - Reasoning token overhead analysis
- `README.md` - Complete documentation

**Run:**
```bash
cd gpt5_nano_compatibility
python test_gpt5_nano_compatibility.py  # Primary test suite
python test_env_config.py               # Configuration tests
```

**Status:** ✅ All tests passing

## What's Tested

### OpenAI Integration
- ✅ API connectivity with gpt-5-nano
- ✅ Parameter normalization (max_tokens → max_completion_tokens)
- ✅ Temperature parameter handling
- ✅ Token budget adjustments for reasoning overhead
- ✅ Backward compatibility with other models

### Configuration
- ✅ .env file loading
- ✅ Configuration parser respects environment variables
- ✅ Model selection from configuration

### Models
- ✅ gpt-5-nano (primary, cost-optimized)
- ✅ text-embedding-3-small (embeddings)
- ✅ gpt-4o-mini (backward compatibility)

### Functionality
- ✅ Embeddings generation
- ✅ Entity extraction
- ✅ LLM completion
- ✅ Full RAG pipeline integration

## Quick Start

1. **Setup environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your OpenAI API keys
   ```

2. **Run primary test suite:**
   ```bash
   cd tests/gpt5_nano_compatibility
   python test_gpt5_nano_compatibility.py
   ```

3. **Expected output:**
   ```
   ✅ Parameter Normalization: PASSED
   ✅ Configuration Loading: PASSED
   ✅ Embeddings: PASSED
   ✅ Simple Completion: PASSED
   ✅ Entity Extraction: PASSED
   🎉 ALL TESTS PASSED
   ```

## Key Implementation Details

### Parameter Normalization
The main gpt-5-nano compatibility fix is in `/lightrag/llm/openai.py`:

```python
def _normalize_openai_kwargs_for_model(model: str, kwargs: dict[str, Any]) -> None:
    """Handle model-specific parameter constraints"""
    if model.startswith("gpt-5"):
        # Convert max_tokens → max_completion_tokens
        if "max_tokens" in kwargs:
            max_tokens = kwargs.pop("max_tokens")
            kwargs["max_completion_tokens"] = int(max(max_tokens * 2.5, 300))

        # Remove unsupported parameters
        kwargs.pop("temperature", None)
```

### Why 2.5x Multiplier?
gpt-5-nano uses internal reasoning that consumes tokens. Testing showed:
- Original token budget often leaves empty responses
- 2.5x multiplication provides adequate margin
- 300 token minimum ensures consistency

## Related Documentation

- `/docs/GPT5_NANO_COMPATIBILITY.md` - Comprehensive user guide
- `/docs/GPT5_NANO_COMPATIBILITY_IMPLEMENTATION.md` - Technical implementation details
- `gpt5_nano_compatibility/README.md` - Detailed test documentation

## Test Statistics

- **Total Tests:** 11
- **Passing:** 11 ✅
- **Failing:** 0 ✅
- **Coverage:** OpenAI integration, configuration, embeddings, LLM, RAG pipeline

## Maintenance

When modifying LightRAG's OpenAI integration:
1. Run tests to ensure compatibility
2. Pay special attention to parameter handling
3. Test with both gpt-5-nano and gpt-4o-mini
4. Update documentation if behavior changes

---

**Last Updated:** 2024
**Status:** Production Ready ✅
**Test Coverage:** OpenAI API Integration (100%)
