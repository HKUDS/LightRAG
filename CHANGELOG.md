# Changelog

All notable changes to the LightRAG project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - Pre-Release Preparation

### 🚀 Major Features Added
- **Production-Grade Infrastructure**: Complete Docker Compose production deployment with security hardening
- **Enterprise Authentication**: JWT-based authentication with rate limiting and audit logging
- **Multi-Storage Support**: PostgreSQL, Redis, MongoDB, Neo4j, Qdrant, Milvus backends
- **MCP Server Integration**: Model Context Protocol server with 11 tools and 3 resources for Claude CLI
- **React Web UI**: TypeScript/React frontend with Vite build system and Bun support
- **xAI Integration**: Full support for xAI Grok models with timeout handling
- **Comprehensive Documentation**: 40+ documentation files with cross-referencing

### ✨ Enhanced Features
- **Storage Architecture**: 4 storage types (KV, Vector, Graph, Document Status) with multiple implementations
- **API Server**: FastAPI-based with Ollama-compatible interface and health checks
- **Async Processing**: Full async/await architecture with connection pooling
- **Security Hardening**: Non-root containers, read-only filesystems, capability dropping
- **Monitoring & Observability**: Comprehensive logging, audit trails, and health monitoring
- **Kubernetes Support**: Production-grade K8s deployments with Helm charts

### 🔧 Configuration Improvements
- **Environment Management**: Comprehensive `.env` templates and examples
- **Provider Configurations**: Support for OpenAI, Ollama, Azure OpenAI, xAI, and more
- **Production Templates**: Ready-to-use production configuration templates
- **Auto-scaling Options**: Gunicorn with configurable worker processes

### 📚 Documentation Enhancements
- **DOCUMENTATION_INDEX.md**: Complete navigation with role-based paths
- **PRODUCTION_DEPLOYMENT_COMPLETE.md**: Single authoritative production guide
- **Integration Guides**: Detailed guides for MCP, xAI, Docling, and more
- **Security Documentation**: Comprehensive hardening and authentication guides
- **Architecture Documentation**: System flow diagrams and component details

### 🧪 Testing & Quality Assurance
- **Test Framework**: Comprehensive pytest-based testing with async support
- **Health Checking**: Multi-layer health checks for all system components
- **Validation Scripts**: Automated documentation link validation
- **Pre-commit Hooks**: Code formatting and quality enforcement

### 🐛 Bug Fixes
- **xAI Timeout Issues**: Fixed with MAX_ASYNC=2 configuration and retry logic
- **Unicode Handling**: Resolved decode errors in xAI integration
- **Redis Deprecation**: Fixed close() vs aclose() compatibility
- **DateTime Warnings**: Updated to timezone-aware datetime methods
- **Pydantic V2**: Migrated from dict() to model_dump()

### 🔒 Security Enhancements
- **Authentication System**: Complete JWT implementation with configurable expiration
- **Rate Limiting**: Per-endpoint configurable rate limiting
- **Audit Logging**: Complete request/response audit trail
- **Container Security**: Security-hardened Docker containers
- **Network Security**: Internal container networks with controlled access

### ⚠️ Known Issues
- **Test Coverage**: Currently at 9.85% - improvement needed before stable release
- **Deprecated Functions**: Multiple functions marked for removal in v0.2.0
- **Storage Lock-in**: Cannot change storage backends after document addition
- **MCP Direct Client**: Several features marked as NOT_IMPLEMENTED

### 🚨 Breaking Changes
- **Storage Interface**: May change in v0.2.0
- **Deprecated Functions**: Will be removed in v0.2.0
- **Configuration Format**: Minor changes possible in v0.2.0

### 📋 Migration Notes
- **From v1.4.x**: Update configuration files using new templates
- **Storage Backends**: Plan backend selection carefully - cannot be changed later
- **xAI Users**: Add `MAX_ASYNC=2` to environment configuration
- **Authentication**: Review security settings before production deployment

### 🔗 Dependencies
- **Python**: Requires Python 3.10+ (3.11+ recommended)
- **Core Dependencies**: Updated pandas to 2.0.0+, added async support
- **Optional Dependencies**: Extensive API dependencies for production features
- **Test Dependencies**: pytest, pytest-asyncio, pytest-cov for testing

---

## [1.4.5] - 2024-01-XX (Current Version)

### Base Features
- Core LightRAG retrieval-augmented generation functionality
- Knowledge graph processing and vector retrieval
- Basic storage backends and LLM integrations
- Initial API server implementation

---

## Release Planning

### Next Release (v1.5.0 - Stable)
**Target**: Q1 2025
**Requirements**:
- [ ] Test coverage improvement to 60%+
- [ ] Deprecated function cleanup
- [ ] Circuit breaker implementation
- [ ] Complete MCP direct client features

### Future Releases (v2.0.0)
**Planned Features**:
- Storage backend migration support
- Advanced security features
- Performance optimization
- Extended LLM provider support

---

## Development Status

**Current Branch**: `cleanup-and-docs`
**Release Status**: 🚨 Pre-Release - Critical issues must be addressed
**Test Coverage**: 9.85% (TARGET: 60%+)
**Documentation**: ✅ Complete
**Production Ready**: ⚠️ Conditional - see KNOWN_ISSUES.md

---

For detailed information about known issues and limitations, see [KNOWN_ISSUES.md](KNOWN_ISSUES.md).
