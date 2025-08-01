# LightRAG Testing Scripts

This directory contains automated testing scripts for validating LightRAG production deployments.

## 📋 Available Scripts

### 1. `test-production-deployment.sh`
**Comprehensive end-to-end testing suite**

This script performs complete validation of the LightRAG production deployment including:

#### Infrastructure Tests
- ✅ Container status verification
- ✅ LightRAG API health check
- ✅ PostgreSQL connection & extensions
- ✅ Monitoring endpoints accessibility

#### Document Processing Tests
- ✅ Document upload functionality
- ✅ Document processing pipeline
- ✅ Processing status tracking
- ✅ PostgreSQL data storage verification

#### RAG Query Tests
- ✅ Hybrid query mode testing
- ✅ Local query mode testing
- ✅ Global query mode testing
- ✅ Response quality validation

#### Usage
```bash
# Run full test suite
./scripts/test-production-deployment.sh

# The script will:
# 1. Create a test document
# 2. Upload and process it
# 3. Verify all components
# 4. Test query functionality
# 5. Clean up test files
# 6. Provide detailed results
```

#### Requirements
- `curl` - For API testing
- `jq` - For JSON parsing
- `docker` and `docker-compose` - For container management
- Running LightRAG production deployment

#### Expected Output
```
======================================================================
              LightRAG Production Deployment Test Suite
======================================================================

=== INFRASTRUCTURE TESTS ===
✅ Container Status Check
✅ LightRAG Health Check
✅ PostgreSQL Connection & Extensions
✅ Monitoring Endpoints

=== DOCUMENT PROCESSING TESTS ===
✅ Document Upload
✅ Document Processing
✅ Document Status Verification
✅ PostgreSQL Data Verification

=== RAG QUERY TESTS ===
✅ Hybrid Query Test
✅ Local Query Test
✅ Global Query Test

======================================================================
                            TEST RESULTS
======================================================================
🎉 ALL TESTS PASSED! (11/11)
✅ Production deployment is fully operational and ready for use!

Available Services:
• LightRAG API: http://localhost:9621
• Grafana Dashboard: http://localhost:3000 (admin/admin)
• Prometheus Metrics: http://localhost:9091
• Jaeger Tracing: http://localhost:16686
```

---

### 2. `quick-health-check.sh`
**Fast health check for daily monitoring**

This script performs essential health checks to verify the system is operational:

#### Quick Checks
- ✅ Container status (LightRAG, PostgreSQL, Redis)
- ✅ API health endpoint
- ✅ Database connectivity
- ✅ Document count summary
- ✅ Basic query functionality

#### Usage
```bash
# Run quick health check
./scripts/quick-health-check.sh
```

#### Expected Output
```
🔍 LightRAG Quick Health Check

✅ lightrag_app is running
✅ lightrag_postgres is running
✅ lightrag_redis is running
✅ LightRAG API is healthy
LLM Model: grok-3-mini
Embedding Model: bge-m3:latest
✅ PostgreSQL database is accessible
✅ Found 3 processed document(s)
✅ Query functionality is working

✅ All critical health checks passed! 🎉

📋 Service URLs:
• LightRAG API: http://localhost:9621
• Health Check: http://localhost:9621/health
• Grafana: http://localhost:3000
• Prometheus: http://localhost:9091
• Jaeger: http://localhost:16686
```

---

## 🔧 Configuration

Both scripts read configuration from environment variables:

```bash
# PostgreSQL Configuration
export POSTGRES_USER="lightrag_user"
export POSTGRES_PASSWORD="your_secure_password"
export POSTGRES_DATABASE="lightrag_db"

# LightRAG Configuration
export LIGHTRAG_URL="http://localhost:9621"
export COMPOSE_FILE="docker-compose.production.yml"
```

## 🚀 Integration with CI/CD

### GitHub Actions Example
```yaml
name: LightRAG Production Test
on: [push, pull_request]

jobs:
  test-deployment:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Start LightRAG Production
        run: |
          cp production.env .env
          docker-compose -f docker-compose.production.yml up -d

      - name: Run Health Check
        run: ./scripts/quick-health-check.sh

      - name: Run Full Test Suite
        run: ./scripts/test-production-deployment.sh
```

### Docker Compose Test Service
```yaml
# Add to docker-compose.production.yml
  test-runner:
    build: .
    volumes:
      - ./scripts:/scripts
      - /var/run/docker.sock:/var/run/docker.sock
    command: /scripts/test-production-deployment.sh
    depends_on:
      - lightrag
```

## 📊 Test Results

### Success Criteria
- All containers running and healthy
- API responding to health checks
- PostgreSQL extensions installed and accessible
- Document processing pipeline functional
- RAG queries returning comprehensive answers
- Monitoring services accessible

### Common Issues and Solutions

#### Container Not Starting
```bash
# Check logs
docker-compose -f docker-compose.production.yml logs lightrag

# Common fixes
docker-compose -f docker-compose.production.yml down --volumes
docker-compose -f docker-compose.production.yml up -d
```

#### Database Connection Issues
```bash
# Check PostgreSQL status
docker-compose -f docker-compose.production.yml exec postgres pg_isready

# Reset database
docker-compose -f docker-compose.production.yml restart postgres
```

#### API Not Responding
```bash
# Check if all dependencies are healthy
docker-compose -f docker-compose.production.yml ps

# Restart LightRAG service
docker-compose -f docker-compose.production.yml restart lightrag
```

## 📝 Customization

### Adding Custom Tests
Edit `test-production-deployment.sh` and add your test function:

```bash
test_custom_feature() {
    log "Testing custom feature..."

    # Your test logic here
    if your_test_command; then
        return 0
    else
        return 1
    fi
}

# Add to main() function
run_test "Custom Feature Test" "test_custom_feature"
```

### Environment-Specific Configuration
Create environment-specific configuration files:

```bash
# scripts/config/staging.env
LIGHTRAG_URL="https://staging.example.com"
POSTGRES_USER="staging_user"

# scripts/config/production.env
LIGHTRAG_URL="https://api.example.com"
POSTGRES_USER="prod_user"

# Usage
source scripts/config/staging.env
./scripts/test-production-deployment.sh
```

## 🔍 Troubleshooting

### Verbose Output
Set environment variable for detailed logging:
```bash
export VERBOSE=1
./scripts/test-production-deployment.sh
```

### Test Individual Components
```bash
# Test only containers
docker-compose -f docker-compose.production.yml ps

# Test only API
curl -sf http://localhost:9621/health | jq .

# Test only database
docker-compose -f docker-compose.production.yml exec postgres pg_isready
```

### Performance Testing
For load testing, consider using these additional tools:
- `ab` (Apache Bench) for API load testing
- `pgbench` for PostgreSQL performance testing
- Custom scripts for document processing throughput

---

## 📚 Related Documentation

- [Production Deployment Guide](../PRODUCTION_DEPLOYMENT_GUIDE.md)
- [API Documentation](../docs/api.md)
- [Monitoring Setup](../monitoring/README.md)
- [Troubleshooting Guide](../docs/troubleshooting.md)
