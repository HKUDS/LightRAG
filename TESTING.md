# LightRAG Testing Guide

This guide provides information on testing your LightRAG production deployment.

## 📋 Quick Start

### Daily Health Check
For quick verification that all services are running:
```bash
./scripts/quick-health-check.sh
```

### Full System Test
For comprehensive end-to-end testing:
```bash
./scripts/test-production-deployment.sh
```

## 🚀 Test Scripts Overview

| Script | Purpose | Duration | When to Use |
|--------|---------|----------|-------------|
| `quick-health-check.sh` | Basic service verification | ~10 seconds | Daily monitoring, quick troubleshooting |
| `test-production-deployment.sh` | Complete system validation | ~2-3 minutes | After deployments, before releases |

## ✅ What Gets Tested

### Infrastructure Tests
- [x] Docker container status
- [x] LightRAG API health
- [x] PostgreSQL connectivity
- [x] Redis connectivity
- [x] Required database extensions
- [x] Monitoring services

### Document Processing Tests
- [x] File upload functionality
- [x] Document parsing and chunking
- [x] Entity and relation extraction (xAI)
- [x] Embedding generation (Ollama)
- [x] Knowledge graph creation
- [x] PostgreSQL data storage

### Query Tests
- [x] Hybrid query mode (KG + Vector)
- [x] Local query mode (Context-dependent)
- [x] Global query mode (Knowledge graph)
- [x] Response quality validation
- [x] Citation tracking

## 🔧 Configuration

Set environment variables for custom configuration:
```bash
# PostgreSQL Settings
export POSTGRES_USER="your_db_user"
export POSTGRES_PASSWORD="your_secure_password"
export POSTGRES_DATABASE="your_db_name"

# API Settings
export LIGHTRAG_URL="http://localhost:9621"

# Run tests
./scripts/test-production-deployment.sh
```

## 📊 Expected Results

### Successful Health Check
```bash
🔍 LightRAG Quick Health Check

✅ lightrag is running
✅ postgres is running
✅ redis is running
✅ LightRAG API is healthy
✅ PostgreSQL database is accessible
✅ Found 3 processed document(s)
✅ Query functionality is working

✅ All critical health checks passed! 🎉
```

### Successful Full Test
```bash
🎉 ALL TESTS PASSED! (11/11)
✅ Production deployment is fully operational and ready for use!

Available Services:
• LightRAG API: http://localhost:9621
• Grafana Dashboard: http://localhost:3000
• Prometheus Metrics: http://localhost:9091
• Jaeger Tracing: http://localhost:16686
```

## 🔍 Troubleshooting

### Common Issues

#### "Service not running" errors
```bash
# Check actual container status
docker compose -f docker-compose.production.yml ps

# Restart if needed
docker compose -f docker-compose.production.yml restart [service_name]
```

#### "Database not accessible" errors
```bash
# Check PostgreSQL logs
docker compose -f docker-compose.production.yml logs postgres

# Test direct connection
docker compose -f docker-compose.production.yml exec postgres psql -U [username] -d [database]
```

#### "API not responding" errors
```bash
# Check LightRAG logs
docker compose -f docker-compose.production.yml logs lightrag

# Verify all dependencies are healthy
docker compose -f docker-compose.production.yml ps
```

### Test Failures

#### Document Processing Fails
1. Check xAI API key validity
2. Verify Ollama is running on host
3. Ensure PostgreSQL extensions are installed
4. Check disk space for document storage

#### Query Tests Fail
1. Verify documents were processed successfully
2. Check if knowledge graph data exists
3. Ensure embeddings were generated
4. Test with simpler queries first

## 🚀 Integration

### CI/CD Pipeline
Add to your GitHub Actions or similar:
```yaml
- name: Test Production Deployment
  run: |
    ./scripts/quick-health-check.sh
    ./scripts/test-production-deployment.sh
```

### Monitoring Integration
Set up automated testing:
```bash
# Add to crontab for regular monitoring
0 */6 * * * /path/to/lightrag/scripts/quick-health-check.sh

# Email alerts on failure
0 */6 * * * /path/to/lightrag/scripts/quick-health-check.sh || mail -s "LightRAG Health Check Failed" admin@company.com
```

## 📚 Related Documentation

- [Production Deployment Guide](PRODUCTION_DEPLOYMENT_GUIDE.md)
- [API Documentation](docs/api.md)
- [Monitoring Setup](monitoring/README.md)
- [Script Documentation](scripts/README.md)
