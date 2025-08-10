# LightRAG Integration Guides

**Related Documents**: [Production Deployment](../production/PRODUCTION_DEPLOYMENT_COMPLETE.md) | [System Architecture](../architecture/SYSTEM_ARCHITECTURE_AND_DATA_FLOW.md) | [Documentation Index](../DOCUMENTATION_INDEX.md)

This directory contains comprehensive integration guides for connecting LightRAG with external services, protocols, and platforms.

> **💡 For Production Integration**: All these integrations are covered in the [Complete Production Deployment Guide](../production/PRODUCTION_DEPLOYMENT_COMPLETE.md) with configuration examples.

## Featured Integration: Model Context Protocol (MCP) 🚀

**NEW in January 2025**: Complete MCP integration enables direct use of LightRAG through Claude CLI and other MCP-compatible clients.

### Quick Links
- **[🎯 MCP Implementation Summary](MCP_IMPLEMENTATION_SUMMARY.md)** - **START HERE** - Complete overview and usage guide
- **[📋 MCP Integration Plan](MCP_INTEGRATION_PLAN.md)** - Strategic implementation roadmap
- **[🔧 MCP Implementation Guide](MCP_IMPLEMENTATION_GUIDE.md)** - Step-by-step development guide
- **[⚙️ MCP Tools Specification](MCP_TOOLS_SPECIFICATION.md)** - Technical specifications for all tools

### What is MCP Integration?

The Model Context Protocol integration provides:
- **11 MCP Tools** for complete RAG operations
- **3 MCP Resources** for system monitoring
- **Streaming Support** for real-time responses
- **Production-Ready** error handling and caching
- **Natural Language Interface** through Claude CLI

### Quick Start with MCP

```bash
# 1. Install dependencies
pip install mcp httpx pydantic aiofiles typing-extensions

# 2. Start MCP server
python -m lightrag_mcp

# 3. Configure Claude CLI
claude config mcp add lightrag-mcp python -m lightrag_mcp

# 4. Start querying!
claude mcp lightrag_query "What are the main themes in my documents?"
```

### Available MCP Tools

#### Query Tools (2)
- `lightrag_query` - Execute RAG queries with 6 modes
- `lightrag_stream_query` - Real-time streaming responses

#### Document Management Tools (5)
- `lightrag_insert_text` - Insert text documents
- `lightrag_insert_file` - Process files (PDF, DOCX, etc.)
- `lightrag_list_documents` - List documents with filtering
- `lightrag_delete_documents` - Remove documents
- `lightrag_batch_process` - Batch document processing

#### Knowledge Graph Tools (4)
- `lightrag_get_graph` - Extract graph data
- `lightrag_search_entities` - Search entities
- `lightrag_update_entity` - Modify entities
- `lightrag_get_entity_relationships` - Explore relationships

#### System Tools (3)
- `lightrag_health_check` - System monitoring
- `lightrag_clear_cache` - Cache management
- `lightrag_get_system_stats` - Usage analytics

### MCP Resources

- `lightrag://system/config` - System configuration
- `lightrag://system/health` - Health status
- `lightrag://documents/status` - Document pipeline status

## Other Integrations

### xAI Grok Models
- **Enhanced Error Handling**: Robust timeout and connection management
- **Model Support**: grok-3-mini, grok-2-1212, grok-2-vision-1212
- **Production Ready**: Comprehensive configuration options

### Enhanced Docling
- **19 Configuration Options**: Advanced document processing
- **Multiple Export Formats**: Markdown, JSON, HTML, DocTags, text
- **Intelligent Caching**: 87% performance improvement
- **Comprehensive Testing**: 100% validation success rate

## Integration Architecture

```
LightRAG Core
     ↓
LightRAG API Server (FastAPI)
     ↓
┌─────────────────┬─────────────────┬─────────────────┐
│   MCP Server    │   Web UI        │   Ollama API    │
│   (Claude CLI)  │   (React/TS)    │   (Chat Bots)   │
└─────────────────┴─────────────────┴─────────────────┘
```

## File Organization

### MCP Documentation
- `MCP_IMPLEMENTATION_SUMMARY.md` - Complete overview (recommended starting point)
- `MCP_INTEGRATION_PLAN.md` - Strategic planning document
- `MCP_IMPLEMENTATION_GUIDE.md` - Developer implementation guide
- `MCP_TOOLS_SPECIFICATION.md` - Technical tool specifications

### Legacy Integration Docs
- `XAI_INTEGRATION_SUMMARY.md` - xAI Grok model integration
- `TROUBLESHOOTING_XAI.md` - xAI troubleshooting guide
- `ENHANCED_DOCLING_TEST_SUMMARY.md` - Docling configuration tests

## Getting Started

### For End Users
1. **Start with MCP**: [MCP Implementation Summary](MCP_IMPLEMENTATION_SUMMARY.md)
2. **Install and Configure**: Follow the MCP quick start guide
3. **Begin Querying**: Use Claude CLI to interact with your knowledge base

### for Developers
1. **Understand the Architecture**: Review the integration plan
2. **Follow Implementation Guide**: Step-by-step development instructions
3. **Reference Tool Specs**: Technical specifications for all tools
4. **Extend as Needed**: Framework supports custom tool development

### For DevOps
1. **Review Deployment Options**: Multiple deployment strategies available
2. **Configure Environment**: 25+ environment variables for customization
3. **Monitor Performance**: Built-in health checks and statistics
4. **Scale as Needed**: Supports high-concurrency production deployments

## Integration Benefits

### For Users
- **Natural Language Interface**: Query your knowledge base conversationally
- **Comprehensive Operations**: Full document and graph management
- **Real-time Responses**: Streaming support for long-running queries
- **Rich Visualizations**: Export graph data in multiple formats

### For Developers
- **Production Ready**: Comprehensive error handling and validation
- **Well Documented**: Extensive examples and troubleshooting guides
- **Extensible**: Clean architecture for custom integrations
- **Performance Optimized**: Caching, connection pooling, async operations

### For Organizations
- **Easy Adoption**: Minimal configuration required
- **Scalable**: Supports multiple concurrent users
- **Secure**: Input validation, authentication, rate limiting
- **Maintainable**: Clean code structure and comprehensive logging

## Performance Characteristics

### Response Times
- **Health Checks**: <500ms
- **Simple Queries**: <2s
- **Complex Queries**: <10s
- **Document Operations**: <30s

### Scalability
- **Concurrent Users**: 50+ supported
- **Document Processing**: 100+ docs/hour
- **Memory Usage**: <1GB base footprint
- **Cache Performance**: 87% improvement for repeated queries

## Troubleshooting

### Common Issues
1. **Connection Problems**: Verify LightRAG server is running
2. **Authentication Errors**: Check API key configuration
3. **Tool Not Found**: Refresh MCP configuration in Claude CLI
4. **Performance Issues**: Enable caching and adjust concurrency limits

### Debug Resources
- **Health Checks**: Use `lightrag_health_check` tool
- **System Stats**: Monitor with `lightrag_get_system_stats`
- **Debug Logging**: Enable with `MCP_LOG_LEVEL=DEBUG`
- **Correlation IDs**: Track requests for debugging

## Future Integrations

### Planned
- **Authentication Protocols**: OAuth, SAML integration
- **Monitoring Systems**: Prometheus, OpenTelemetry support
- **Additional LLM Providers**: Expanded model support
- **Custom Storage**: Additional database backends

### Community Contributions
- **Plugin System**: Framework for community integrations
- **Template Library**: Reusable integration patterns
- **Best Practices**: Community-driven guidelines
- **Performance Optimizations**: Shared optimization techniques

## Contributing

### Adding New Integrations
1. **Follow Naming Convention**: `{SERVICE}_INTEGRATION_{TYPE}.md`
2. **Include Complete Guide**: Installation, configuration, usage, troubleshooting
3. **Provide Examples**: Working code samples and usage patterns
4. **Update This README**: Add links and descriptions
5. **Test Thoroughly**: Ensure all examples work correctly

### Documentation Standards
- **Clear Structure**: Use consistent headings and organization
- **Complete Coverage**: Include all configuration options
- **Working Examples**: Test all code samples
- **Troubleshooting**: Address common issues
- **Performance Notes**: Include performance characteristics

---

**Last Updated**: 2025-01-29
**Featured Integration**: Model Context Protocol (MCP) v1.0.0
**Total Integrations**: 3 (MCP, xAI, Enhanced Docling)
**Documentation Status**: Complete and Current

For questions or issues with specific integrations, refer to the individual integration documentation or enable debug logging for detailed error information.
=======
# Integration Guides

This directory contains comprehensive guides for integrating LightRAG with various services and protocols.

## Available Guides

### Model Context Protocol (MCP)
- `MCP_INTEGRATION_PLAN.md` - Complete integration plan for MCP support
- `MCP_IMPLEMENTATION_GUIDE.md` - Step-by-step implementation guide
- `MCP_TOOLS_SPECIFICATION.md` - Technical specification for MCP tools

### xAI Grok Integration
- `XAI_INTEGRATION_SUMMARY.md` - Summary of xAI Grok model integration
- `TROUBLESHOOTING_XAI.md` - Troubleshooting guide for xAI issues

### Enhanced Docling Configuration
- `ENHANCED_DOCLING_TEST_SUMMARY.md` - Complete test results and implementation details

## Usage

Each guide is self-contained and includes:
- Technical specifications
- Configuration examples
- Implementation steps
- Testing procedures
- Troubleshooting information

## Contributing

When adding new integration guides:
1. Follow the existing naming convention
2. Include comprehensive documentation
3. Provide working examples
4. Add troubleshooting sections
5. Update this README
