# LightRAG Offline Deployment Guide

This guide provides comprehensive instructions for deploying LightRAG in offline environments where internet access is limited or unavailable.
> Software packages requiring `transformers`, `torch`, or `cuda` will not be included in the offline dependency group. Consequently, document extraction tools such as Docling, as well as local LLM models like Hugging Face and LMDeploy, are outside the scope of offline installation support. These high-compute-resource-demanding services should not be integrated into LightRAG. Docling will be decoupled and deployed as a standalone service.

If you deploy LightRAG using Docker, there is no need to refer to this document, as the LightRAG Docker image is pre-configured for offline operation.

> Software packages requiring `transformers`, `torch`, or `cuda` will not be included in the offline dependency group. Consequently, document extraction tools such as Docling, as well as local LLM models like Hugging Face and LMDeploy, are outside the scope of offline installation support. These high-compute-resource-demanding services should not be integrated into LightRAG. Docling will be decoupled and deployed as a standalone service.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Layered Dependencies](#layered-dependencies)
- [Tiktoken Cache Management](#tiktoken-cache-management)
- [Complete Offline Deployment Workflow](#complete-offline-deployment-workflow)
- [Troubleshooting](#troubleshooting)

## Overview

LightRAG uses dynamic package installation (`pipmaster`) for optional features based on file types and configurations. In offline environments, these dynamic installations will fail. This guide shows you how to pre-install all necessary dependencies and cache files.

### What Gets Dynamically Installed?

LightRAG dynamically installs packages for:

- **Storage Backends**: `redis`, `neo4j`, `pymilvus`, `pymongo`, `asyncpg`, `qdrant-client`
- **LLM Providers**: `openai`, `anthropic`, `ollama`, `zhipuai`, `aioboto3`, `voyageai`, `llama-index`, `lmdeploy`, `transformers`, `torch`
- **Tiktoken Models**: BPE encoding models downloaded from OpenAI CDN

**Note**: Document processing dependencies (`pypdf`, `python-docx`, `python-pptx`, `openpyxl`) are now pre-installed with the `api` extras group and no longer require dynamic installation.

## Quick Start

### Option 1: Using pip with Offline Extras

```bash
# Online environment: Install all offline dependencies
pip install lightrag-hku[offline]

# Download tiktoken cache
lightrag-download-cache

# Create offline package
pip download lightrag-hku[offline] -d ./offline-packages
tar -czf lightrag-offline.tar.gz ./offline-packages ~/.tiktoken_cache

# Transfer to offline server
scp lightrag-offline.tar.gz user@offline-server:/path/to/

# Offline environment: Install
tar -xzf lightrag-offline.tar.gz
pip install --no-index --find-links=./offline-packages lightrag-hku[offline]
export TIKTOKEN_CACHE_DIR=~/.tiktoken_cache
```

### Option 2: Using Requirements Files

```bash
# Online environment: Download packages
pip download -r requirements-offline.txt -d ./packages

# Transfer to offline server
tar -czf packages.tar.gz ./packages
scp packages.tar.gz user@offline-server:/path/to/

# Offline environment: Install
tar -xzf packages.tar.gz
pip install --no-index --find-links=./packages -r requirements-offline.txt
```

## Layered Dependencies

LightRAG provides flexible dependency groups for different use cases:

### Available Dependency Groups

| Group | Description | Use Case |
|-------|-------------|----------|
| `api` | API server + document processing | FastAPI server with PDF, DOCX, PPTX, XLSX support |
| `offline-storage` | Storage backends | Redis, Neo4j, MongoDB, PostgreSQL, etc. |
| `offline-llm` | LLM providers | OpenAI, Anthropic, Ollama, etc. |
| `offline` | Complete offline package | API + Storage + LLM (all features) |

**Note**: Document processing (PDF, DOCX, PPTX, XLSX) is included in the `api` extras group. The previous `offline-docs` group has been merged into `api` for better integration.

> Software packages requiring `transformers`, `torch`, or `cuda` will not be included in the offline dependency group.

### Installation Examples

```bash
# Install API with document processing
pip install lightrag-hku[api]

# Install API and storage backends
pip install lightrag-hku[api,offline-storage]

# Install all offline dependencies (recommended for offline deployment)
pip install lightrag-hku[offline]
```

### Using Individual Requirements Files

```bash
# Storage backends only
pip install -r requirements-offline-storage.txt

# LLM providers only
pip install -r requirements-offline-llm.txt

# All offline dependencies
pip install -r requirements-offline.txt
```

## Tiktoken Cache Management

Tiktoken downloads BPE encoding models on first use. In offline environments, you must pre-download these models.

### Using the CLI Command

After installing LightRAG, use the built-in command:

```bash
# Download to default location (~/.tiktoken_cache)
lightrag-download-cache

# Download to specific directory
lightrag-download-cache --cache-dir ./tiktoken_cache

# Download specific models only
lightrag-download-cache --models gpt-4o-mini gpt-4
```

### Default Models Downloaded

- `gpt-4o-mini` (LightRAG default)
- `gpt-4o`
- `gpt-4`
- `gpt-3.5-turbo`
- `text-embedding-ada-002`
- `text-embedding-3-small`
- `text-embedding-3-large`

### Setting Cache Location in Offline Environment

```bash
# Option 1: Environment variable (temporary)
export TIKTOKEN_CACHE_DIR=/path/to/tiktoken_cache

# Option 2: Add to ~/.bashrc or ~/.zshrc (persistent)
echo 'export TIKTOKEN_CACHE_DIR=~/.tiktoken_cache' >> ~/.bashrc
source ~/.bashrc

# Option 3: Copy to default location
cp -r /path/to/tiktoken_cache ~/.tiktoken_cache/
```

## Complete Offline Deployment Workflow

### Step 1: Prepare in Online Environment

```bash
# 1. Install LightRAG with offline dependencies
pip install lightrag-hku[offline]

# 2. Download tiktoken cache
lightrag-download-cache --cache-dir ./offline_cache/tiktoken

# 3. Download all Python packages
pip download lightrag-hku[offline] -d ./offline_cache/packages

# 4. Create archive for transfer
tar -czf lightrag-offline-complete.tar.gz ./offline_cache

# 5. Verify contents
tar -tzf lightrag-offline-complete.tar.gz | head -20
```

### Step 2: Transfer to Offline Environment

```bash
# Using scp
scp lightrag-offline-complete.tar.gz user@offline-server:/tmp/

# Or using USB/physical media
# Copy lightrag-offline-complete.tar.gz to USB drive
```

### Step 3: Install in Offline Environment

```bash
# 1. Extract archive
cd /tmp
tar -xzf lightrag-offline-complete.tar.gz

# 2. Install Python packages
pip install --no-index \
    --find-links=/tmp/offline_cache/packages \
    lightrag-hku[offline]

# 3. Set up tiktoken cache
mkdir -p ~/.tiktoken_cache
cp -r /tmp/offline_cache/tiktoken/* ~/.tiktoken_cache/
export TIKTOKEN_CACHE_DIR=~/.tiktoken_cache

# 4. Add to shell profile for persistence
echo 'export TIKTOKEN_CACHE_DIR=~/.tiktoken_cache' >> ~/.bashrc
```

### Step 4: Verify Installation

```bash
# Test Python import
python -c "from lightrag import LightRAG; print('✓ LightRAG imported')"

# Test tiktoken
python -c "from lightrag.utils import TiktokenTokenizer; t = TiktokenTokenizer(); print('✓ Tiktoken working')"

# Test optional dependencies (if installed)
python -c "import docling; print('✓ Docling available')"
python -c "import redis; print('✓ Redis available')"
```

## Troubleshooting

### Issue: Tiktoken fails with network error

**Problem**: `Unable to load tokenizer for model gpt-4o-mini`

**Solution**:
```bash
# Ensure TIKTOKEN_CACHE_DIR is set
echo $TIKTOKEN_CACHE_DIR

# Verify cache files exist
ls -la ~/.tiktoken_cache/

# If empty, you need to download cache in online environment first
```

### Issue: Dynamic package installation fails

**Problem**: `Error installing package xxx`

**Solution**:
```bash
# Pre-install the specific package you need
# For API with document processing:
pip install lightrag-hku[api]

# For storage backends:
pip install lightrag-hku[offline-storage]

# For LLM providers:
pip install lightrag-hku[offline-llm]
```

### Issue: Missing dependencies at runtime

**Problem**: `ModuleNotFoundError: No module named 'xxx'`

**Solution**:
```bash
# Check what you have installed
pip list | grep -i xxx

# Install missing component
pip install lightrag-hku[offline]  # Install all offline deps
```

### Issue: Permission denied on tiktoken cache

**Problem**: `PermissionError: [Errno 13] Permission denied`

**Solution**:
```bash
# Ensure cache directory has correct permissions
chmod 755 ~/.tiktoken_cache
chmod 644 ~/.tiktoken_cache/*

# Or use a user-writable directory
export TIKTOKEN_CACHE_DIR=~/my_tiktoken_cache
mkdir -p ~/my_tiktoken_cache
```

## Best Practices

1. **Test in Online Environment First**: Always test your complete setup in an online environment before going offline.

2. **Keep Cache Updated**: Periodically update your offline cache when new models are released.

3. **Document Your Setup**: Keep notes on which optional dependencies you actually need.

4. **Version Pinning**: Consider pinning specific versions in production:
   ```bash
   pip freeze > requirements-production.txt
   ```

5. **Minimal Installation**: Only install what you need:
   ```bash
   # If you only need API with document processing
   pip install lightrag-hku[api]
   # Then manually add specific LLM: pip install openai
   ```

## Additional Resources

- [LightRAG GitHub Repository](https://github.com/HKUDS/LightRAG)
- [Docker Deployment Guide](./DockerDeployment.md)
- [API Documentation](../lightrag/api/README.md)

## Support

If you encounter issues not covered in this guide:

1. Check the [GitHub Issues](https://github.com/HKUDS/LightRAG/issues)
2. Review the [project documentation](../README.md)
3. Create a new issue with your offline deployment details
