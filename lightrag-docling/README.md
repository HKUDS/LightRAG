# LightRAG Docling Service

A standalone microservice for document processing using [Docling](https://github.com/DS4SD/docling).

## Overview

This service provides a FastAPI-based REST API for processing documents with advanced features:

- **Multiple Format Support**: PDF, DOCX, PPTX, XLSX, TXT, MD
- **Advanced Processing**: OCR, table extraction, figure detection
- **Flexible Export**: Markdown, JSON, Text, HTML
- **Intelligent Caching**: Disk-based caching with TTL
- **Batch Processing**: Process multiple documents in parallel
- **Health Monitoring**: Comprehensive health checks and metrics
- **Production Ready**: Security hardening, monitoring, logging

## Quick Start

### Using Docker Compose (Recommended)

1. **Clone and setup**:
   ```bash
   cd lightrag-docling
   cp .env.example .env
   # Edit .env as needed
   ```

2. **Start service**:
   ```bash
   docker-compose up -d
   ```

3. **Test service**:
   ```bash
   curl http://localhost:8080/health
   ```

### Using Docker

```bash
# Build image
docker build -t lightrag-docling .

# Run container
docker run -d \
  --name docling-service \
  -p 8080:8080 \
  -e DOCLING_CACHE_ENABLED=true \
  -v docling_cache:/app/cache \
  lightrag-docling
```

### Development Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run service**:
   ```bash
   python -m uvicorn src.docling_service:app --reload --port 8080
   ```

## API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8080/docs
- **ReDoc**: http://localhost:8080/redoc

### Key Endpoints

- `POST /process` - Process single document
- `POST /process/batch` - Process multiple documents
- `GET /health` - Service health check
- `GET /config` - Service configuration
- `GET /formats` - Supported formats
- `GET /metrics` - Service metrics
- `GET /cache/stats` - Cache statistics
- `DELETE /cache` - Clear cache

### Example Usage

```python
import base64
import requests

# Read document
with open("document.pdf", "rb") as f:
    file_content = base64.b64encode(f.read()).decode()

# Process document
response = requests.post("http://localhost:8080/process", json={
    "file_content": file_content,
    "filename": "document.pdf",
    "config": {
        "export_format": "markdown",
        "enable_ocr": True,
        "enable_table_structure": True,
        "enable_figures": True
    }
})

result = response.json()
print(result["content"])
```

## Configuration

Configure via environment variables (see `.env.example`):

### Core Settings
- `DOCLING_HOST` - Service bind host (default: 0.0.0.0)
- `DOCLING_PORT` - Service port (default: 8080)
- `DOCLING_LOG_LEVEL` - Logging level (default: INFO)

### Processing Settings
- `DOCLING_DEFAULT_EXPORT_FORMAT` - Default export format (default: markdown)
- `DOCLING_DEFAULT_ENABLE_OCR` - Enable OCR by default (default: true)
- `DOCLING_DEFAULT_ENABLE_TABLE_STRUCTURE` - Enable table extraction (default: true)
- `DOCLING_DEFAULT_ENABLE_FIGURES` - Enable figure extraction (default: true)

### Cache Settings
- `DOCLING_CACHE_ENABLED` - Enable caching (default: true)
- `DOCLING_CACHE_DIR` - Cache directory (default: ./cache)
- `DOCLING_CACHE_MAX_SIZE_GB` - Maximum cache size (default: 5)
- `DOCLING_CACHE_TTL_HOURS` - Cache TTL in hours (default: 168)

### Limits
- `DOCLING_MAX_FILE_SIZE_MB` - Maximum file size (default: 100)
- `DOCLING_MAX_BATCH_SIZE` - Maximum batch size (default: 10)
- `DOCLING_REQUEST_TIMEOUT_SECONDS` - Request timeout (default: 300)

## Testing

Run tests:
```bash
# Install test dependencies
pip install -r tests/requirements.txt

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Health Monitoring

### Health Check
```bash
curl http://localhost:8080/health
```

Returns service status, uptime, resource usage, and dependency status.

### Metrics
```bash
curl http://localhost:8080/metrics
```

Returns processing statistics, performance metrics, and cache statistics.

### Cache Management
```bash
# Get cache statistics
curl http://localhost:8080/cache/stats

# Clear cache
curl -X DELETE http://localhost:8080/cache
```

## Performance Tuning

### Resource Allocation
- **Memory**: 2-4GB recommended (ML models are memory intensive)
- **CPU**: 2+ cores recommended for parallel processing
- **Storage**: SSD recommended for cache performance

### Configuration Optimization
- Set `DOCLING_DEFAULT_MAX_WORKERS=2` for balanced performance
- Enable caching with appropriate TTL for your use case
- Adjust `DOCLING_MAX_BATCH_SIZE` based on available resources
- Use `parallel_processing=true` for batch operations

### Docker Resource Limits
```yaml
deploy:
  resources:
    limits:
      memory: 4G
      cpus: '2.0'
    reservations:
      memory: 1G
      cpus: '0.5'
```

## Security

### Authentication
Set `DOCLING_API_KEY` to enable API key authentication:
```bash
curl -H "Authorization: Bearer YOUR_API_KEY" http://localhost:8080/process
```

### Container Security
- Runs as non-root user (UID 10001)
- Minimal runtime dependencies
- Read-only filesystem where possible
- Dropped unnecessary capabilities

### Network Security
- Configure `DOCLING_CORS_ORIGINS` appropriately
- Use HTTPS in production
- Restrict network access to required ports only

## Integration with LightRAG

This service is designed to integrate with LightRAG as a separate microservice:

1. **Service Discovery**: LightRAG detects the service via `DOCLING_SERVICE_URL`
2. **Fallback**: If service unavailable, LightRAG uses basic parsers
3. **Compatible API**: Maintains compatibility with existing docling configuration

See the main LightRAG documentation for integration details.

## Troubleshooting

### Common Issues

1. **Out of Memory**:
   - Increase container memory limit
   - Reduce `DOCLING_DEFAULT_MAX_WORKERS`
   - Process smaller batches

2. **Slow Processing**:
   - Disable unnecessary features (OCR, tables, figures)
   - Increase `DOCLING_DEFAULT_MAX_WORKERS`
   - Use SSD storage for cache

3. **Cache Issues**:
   - Check disk space in cache directory
   - Verify cache directory permissions
   - Clear cache and restart service

### Logs
```bash
# Docker logs
docker logs lightrag-docling-service

# Container logs location
/app/logs/
```

### Debug Mode
Set `DOCLING_DEBUG=true` for detailed logging and error information.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

Same as LightRAG main project.
