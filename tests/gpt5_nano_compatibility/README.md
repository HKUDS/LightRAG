# GPT-5-Nano Compatibility Tests

This directory contains comprehensive tests for ensuring LightRAG's compatibility with OpenAI's gpt-5-nano model, including its specific API constraints and parameter requirements.

## Overview

gpt-5-nano is a cost-optimized reasoning model that differs from traditional LLMs in important ways:
- Uses `max_completion_tokens` instead of `max_tokens`
- Does NOT support custom `temperature` parameter
- Has built-in reasoning that consumes tokens from the completion budget
- Requires token budget adjustments to account for reasoning overhead

These tests validate that LightRAG handles these constraints correctly.

## Test Files

### 1. `test_gpt5_nano_compatibility.py` ⭐ Primary Test Suite
**Purpose:** Comprehensive compatibility validation
**Tests:**
- Test 1: Parameter normalization (max_tokens → max_completion_tokens conversion)
- Test 2: Configuration loading from .env
- Test 3: Embeddings generation with gpt-5-nano
- Test 4: Simple LLM completion
- Test 5: Entity extraction tasks

**Run:** `python test_gpt5_nano_compatibility.py`

**Expected Output:**
```
✅ Parameter Normalization: PASSED
✅ Configuration Loading: PASSED
✅ Embeddings: PASSED
✅ Simple Completion: PASSED
✅ Entity Extraction: PASSED
🎉 ALL TESTS PASSED
```

### 2. `test_env_config.py`
**Purpose:** Validate .env configuration is properly respected
**Tests:**
- Part 1: .env file loading
- Part 2: Config parser respects .env variables
- Part 3: OpenAI API connectivity
- Part 4: Embeddings generation with configured model
- Part 5: LLM extraction with configured model
- Part 6: Full RAG pipeline integration

**Run:** `python test_env_config.py`

**Expected Output:**
```
✅ .env Loading: PASSED
✅ Config Parser: PASSED
✅ OpenAI Connectivity: PASSED
✅ Embeddings: PASSED
✅ LLM Extraction: PASSED
✅ Full Integration: PASSED
OVERALL: 6/6 tests passed
```

### 3. `test_direct_gpt5nano.py`
**Purpose:** Direct API testing without LightRAG abstraction
**Validates:** Raw gpt-5-nano API behavior with proper parameters

**Run:** `python test_direct_gpt5nano.py`

**What it does:**
- Sends direct API request to gpt-5-nano
- Uses `max_completion_tokens` parameter
- Prints raw response and token usage

### 4. `test_gpt5_reasoning.py`
**Purpose:** Understand gpt-5-nano's reasoning token overhead
**Tests:** Token allocation with different reasoning effort levels

**Run:** `python test_gpt5_reasoning.py`

**What it does:**
- Test 1: 200 token budget
- Test 2: 50 token budget with `reasoning_effort="low"`
- Outputs actual reasoning tokens consumed

## Prerequisites

### Environment Variables
Create a `.env` file in the repository root with:

```env
# Required for all tests
OPENAI_API_KEY=sk-...

# For LLM tests
LLM_BINDING=openai
LLM_MODEL=gpt-5-nano
LLM_BINDING_API_KEY=sk-...

# For embedding tests
EMBEDDING_BINDING=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_BINDING_API_KEY=sk-...
EMBEDDING_DIM=1536
```

Or use existing `.env` configuration if already set up.

### Python Dependencies
```bash
pip install openai
pip install python-dotenv
pip install lightrag  # for integration tests
```

## Running All Tests

### From this directory:
```bash
# Run individual test
python test_gpt5_nano_compatibility.py

# Or run all tests
for test in test_*.py; do 
    echo "Running $test..."
    python "$test"
done
```

### From repository root:
```bash
# Run specific test
python -m pytest tests/gpt5_nano_compatibility/test_gpt5_nano_compatibility.py -v

# Or run all tests in this directory
python -m pytest tests/gpt5_nano_compatibility/ -v
```

## Key Findings & Implementation

### Problem: Parameter Incompatibility
gpt-5-nano requires different parameter names and constraints than other OpenAI models.

**Issue:** 
- Other models use `max_tokens`
- gpt-5-nano requires `max_completion_tokens`

**Solution:**
A normalization function `_normalize_openai_kwargs_for_model()` in `/lightrag/llm/openai.py` that:
1. Detects gpt-5 models
2. Converts `max_tokens` → `max_completion_tokens`
3. Applies 2.5x token multiplier (minimum 300 tokens) to account for reasoning overhead
4. Removes unsupported `temperature` parameter

### Problem: Empty Responses
gpt-5-nano was returning empty responses despite successful API calls.

**Root Cause:**
Internal reasoning consumes tokens from the completion budget. With insufficient token budget, all tokens were consumed by reasoning, leaving nothing for actual output.

**Solution:**
Empirical testing showed that:
- 200 tokens: Often empty responses
- 300+ tokens: Consistent full responses
- 2.5x multiplier: Provides adequate margin for reasoning

### Parameter Handling

**For gpt-5-nano models:**
```python
# Before normalization:
{"max_tokens": 500, "temperature": 0.7}

# After normalization:
{"max_completion_tokens": 1250}  # 500 * 2.5, min 300
```

**For other models:**
```python
# Unchanged
{"max_tokens": 500, "temperature": 0.7}
```

## Test Results Summary

All tests validate:
- ✅ Parameter normalization works correctly
- ✅ gpt-5-nano parameter constraints are handled
- ✅ Backward compatibility maintained (other models unaffected)
- ✅ Configuration from .env is respected
- ✅ OpenAI API integration functions properly
- ✅ Embeddings generation works
- ✅ Entity extraction works with gpt-5-nano
- ✅ Full RAG pipeline integration successful

## Troubleshooting

### "OPENAI_API_KEY not set"
- Ensure `.env` file exists in repository root
- Verify `OPENAI_API_KEY` is set: `echo $OPENAI_API_KEY`

### "max_tokens unsupported with this model"
- This error means parameter normalization isn't being called
- Check that you're using LightRAG functions (not direct OpenAI client)
- Verify the normalization function is in `/lightrag/llm/openai.py`

### "Empty API responses"
- Increase token budget (tests use 100+ tokens)
- If using custom token limits, multiply by 2.5 minimum

### "temperature does not support 0.7"
- gpt-5-nano doesn't accept custom temperature
- The normalization function removes it automatically
- No action needed if using LightRAG functions

## Documentation

For more details, see:
- `/docs/GPT5_NANO_COMPATIBILITY.md` - User guide
- `/docs/GPT5_NANO_COMPATIBILITY_IMPLEMENTATION.md` - Technical implementation details

## Related Files

- `/lightrag/llm/openai.py` - Contains parameter normalization logic
- `/lightrag/llm/azure_openai.py` - Azure OpenAI integration with same normalization
- `/.env` - Configuration file (use `.env.example` as template)

## Maintenance Notes

When updating LightRAG's OpenAI integration:
1. Run all tests to ensure backward compatibility
2. If adding new OpenAI models, test with gpt-5-nano constraints
3. Update parameter normalization logic if OpenAI adds new gpt-5 variants
4. Keep `max_tokens * 2.5` strategy unless OpenAI documents different reasoning overhead

---

**Last Updated:** 2024
**Status:** All tests passing ✅
**Model Tested:** gpt-5-nano
**OpenAI SDK:** Latest (with max_completion_tokens support)
