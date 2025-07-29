# LightRAG Documentation

This directory contains comprehensive documentation for LightRAG, including core algorithms, deployment guides, and integration documentation.

## Directory Structure

### Core Documentation
- `Algorithm.md` - LightRAG core algorithms and flowcharts
- `DockerDeployment.md` - Docker deployment guide
- `LightRAG_concurrent_explain.md` - Concurrency and parallel processing
- `rerank_integration.md` - Reranking model integration

### Integration Guides (`integration_guides/`)
Complete guides for integrating LightRAG with external services:
- **MCP (Model Context Protocol)** - Complete implementation guide
- **xAI Grok Models** - Integration and troubleshooting 
- **Enhanced Docling** - Advanced document processing configuration

### Test Outputs (`test_outputs/`)
Generated outputs from testing and validation:
- Test results and logs
- Performance benchmarks
- Content quality analysis

## Quick Navigation

### For Developers
- [Algorithm Overview](Algorithm.md) - Understand LightRAG's core approach
- [Concurrent Processing](LightRAG_concurrent_explain.md) - Parallel processing details
- [Integration Guides](integration_guides/) - External service integrations

### For DevOps
- [Docker Deployment](DockerDeployment.md) - Container deployment
- [Rerank Integration](rerank_integration.md) - Model optimization

### For Integration
- [MCP Implementation](integration_guides/MCP_INTEGRATION_PLAN.md) - Model Context Protocol
- [xAI Integration](integration_guides/XAI_INTEGRATION_SUMMARY.md) - Grok model setup
- [Enhanced Docling](integration_guides/ENHANCED_DOCLING_TEST_SUMMARY.md) - Document processing

## Contributing

When adding documentation:
1. Place core LightRAG docs in the root `docs/` directory
2. Place integration guides in `integration_guides/`
3. Test outputs automatically go to `test_outputs/`
4. Update this README when adding new sections
5. Follow existing markdown formatting conventions

## Documentation Standards

- Use clear, descriptive titles
- Include code examples where applicable
- Provide troubleshooting sections
- Link related documentation
- Keep examples up to date