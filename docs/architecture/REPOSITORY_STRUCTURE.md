# LightRAG Repository Structure

**Last Updated**: 2025-01-29
**Status**: Organized and Clean ✅
**Related Documents**: [System Architecture](SYSTEM_ARCHITECTURE_AND_DATA_FLOW.md) | [Algorithm Overview](Algorithm.md) | [Production Guide](../production/PRODUCTION_DEPLOYMENT_COMPLETE.md) | [Documentation Index](../DOCUMENTATION_INDEX.md)

## 📁 Directory Overview

```
LightRAG/
├── 📚 docs/                          # Documentation
│   ├── integration_guides/            # External service integrations
│   └── test_outputs/                  # Generated test results
├── 🧪 testing/                       # Test files and utilities
│   └── docling_tests/                 # Enhanced Docling tests
├── 📦 examples/                       # Usage examples and demos
├── 🐳 k8s-deploy/                     # Kubernetes deployment
├── 🎯 lightrag/                       # Core library code
├── 🌐 lightrag_webui/                 # React/TypeScript frontend
├── 📊 logs/                           # Application logs
├── 🔄 reproduce/                      # Reproducibility scripts
├── 🧩 tests/                          # Unit tests
└── 💾 rag_storage/                    # Runtime data storage
```

## 📚 Documentation Structure

### Core Documentation (`docs/`)
- `README.md` - Documentation index and navigation
- `Algorithm.md` - LightRAG algorithms and flowcharts
- `DockerDeployment.md` - Container deployment guide
- `LightRAG_concurrent_explain.md` - Concurrency details
- `rerank_integration.md` - Reranking model setup

### Integration Guides (`docs/integration_guides/`)
- `MCP_INTEGRATION_PLAN.md` - Model Context Protocol integration
- `MCP_IMPLEMENTATION_GUIDE.md` - MCP step-by-step guide
- `MCP_TOOLS_SPECIFICATION.md` - MCP technical specifications
- `XAI_INTEGRATION_SUMMARY.md` - xAI Grok model integration
- `TROUBLESHOOTING_XAI.md` - xAI troubleshooting guide
- `ENHANCED_DOCLING_TEST_SUMMARY.md` - Docling configuration tests

### Test Outputs (`docs/test_outputs/`)
Automatically generated test results and validation outputs

## 🧪 Testing Structure

### Main Testing Directory (`testing/`)
- `README.md` - Testing overview and guidelines

### Enhanced Docling Tests (`testing/docling_tests/`)
- `test_enhanced_docling.py` - Comprehensive test suite
- `test_api_integration.py` - API integration tests
- `check_docling_api.py` - API compatibility checker
- `create_test_pdf.py` - Test document generator
- `test_document_enhanced_docling.pdf` - Test PDF file

## 🎯 Core Library (`lightrag/`)

### Main Components
- `lightrag.py` - Core RAG implementation
- `base.py` - Base classes and interfaces
- `constants.py` - Configuration constants
- `exceptions.py` - Custom exceptions

### API Server (`lightrag/api/`)
- `lightrag_server.py` - FastAPI application
- `config.py` - Enhanced configuration system
- `routers/` - API route implementations
- `webui/` - Built web UI assets

### Storage Backends (`lightrag/kg/`)
- Multiple storage implementations (Neo4j, PostgreSQL, Redis, etc.)
- KV, Vector, Graph, and Document Status storage

### LLM Integrations (`lightrag/llm/`)
- Support for OpenAI, Ollama, xAI, Azure, and more
- Enhanced configuration and binding options

## 📦 Examples (`examples/`)
- Production-ready usage examples
- Integration demonstrations
- Performance testing scripts

## 🐳 Deployment (`k8s-deploy/`)
- Kubernetes deployment configurations
- Database setup scripts
- Helm charts and configurations

## 📊 Runtime Data

### Logs (`logs/`)
- Application logs and debugging information
- Automatically rotated and managed

### Storage (`rag_storage/`)
- Runtime document storage
- Vector embeddings and knowledge graphs
- Enhanced Docling cache (`docling_cache/`)

## 🔧 Configuration Files

### Environment Configuration
- `.env` - Local environment variables (not in git)
- `env.example` - Template with all options
- `env.ollama-binding-options.example` - Ollama-specific options

### Project Configuration
- `pyproject.toml` - Python project metadata
- `CLAUDE.md` - Claude Code assistant instructions
- `REPOSITORY_STRUCTURE.md` - This file

## 🚫 Ignored Files (.gitignore)

The repository ignores:
- Development artifacts (`__pycache__/`, `*.pyc`)
- Virtual environments (`.venv/`, `env/`)
- Logs and temporary files (`*.log`, `*.tmp`)
- Test outputs (organized in `docs/test_outputs/`)
- Runtime storage (`rag_storage/`)
- IDE files (`.vscode/`, `.idea/`)

## 🗂️ File Organization Principles

### ✅ Clean Structure
- **No temporary files in root** - All test files in `testing/`
- **Organized documentation** - Grouped by type in `docs/`
- **Logical grouping** - Related files in appropriate directories
- **Clear naming** - Descriptive file and directory names

### ✅ Development-Friendly
- **Easy navigation** - README files in each major directory
- **Test isolation** - Tests separated from core code
- **Output management** - Generated files in designated areas
- **Documentation accessibility** - Guides easy to find

### ✅ Production-Ready
- **Clean deployments** - No development artifacts
- **Proper gitignore** - Only essential files tracked
- **Organized configs** - Templates and examples available
- **Scalable structure** - Easy to add new components

## 🔄 Maintenance Guidelines

### Adding New Features
1. Code goes in appropriate `lightrag/` subdirectory
2. Tests go in `testing/{feature_name}/`
3. Documentation in `docs/` or `docs/integration_guides/`
4. Examples in `examples/`

### Managing Test Files
1. Keep organized in `testing/` subdirectories
2. Generate outputs to `docs/test_outputs/`
3. Update README files when adding test suites
4. Clean up temporary files regularly

### Documentation Updates
1. Update relevant README files
2. Keep integration guides current
3. Document new configuration options
4. Maintain this structure document

## 📈 Benefits of Current Structure

### For Developers
- **Easy navigation** - Clear directory structure
- **Isolated testing** - Tests don't clutter main code
- **Comprehensive docs** - Everything documented
- **Clean git history** - No temporary file commits

### For Users
- **Clear examples** - Easy to find usage patterns
- **Complete guides** - Step-by-step integration help
- **Production ready** - Clean deployment structure
- **Troubleshooting** - Organized help documentation

### For Maintainers
- **Organized issues** - Easy to categorize problems
- **Clean releases** - No development artifacts
- **Scalable structure** - Easy to add new features
- **Quality assurance** - Systematic testing approach

---

This structure supports **professional development**, **easy maintenance**, and **production deployment** while keeping the repository clean and well-organized.
