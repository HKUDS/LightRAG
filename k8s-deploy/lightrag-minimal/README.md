# LightRAG Minimal Helm Chart

This Helm chart deploys a production-ready LightRAG setup with PostgreSQL and pgvector support for Kubernetes environments. It has been tested and validated with complete teardown/rebuild cycles.

## Configuration

This chart provides a comprehensive LightRAG deployment with:
- **PostgreSQL with pgvector**: For vector storage, KV storage, and document status using `pgvector/pgvector:pg16` image
- **NetworkX**: For graph storage (local, no external database required)
- **Persistent Storage**: For data persistence across pod restarts
- **Health Checks**: Automated health monitoring
- **API Endpoints**: Document upload, query, and management
- **Conservative Concurrency**: Optimized OpenAI API usage to prevent rate limiting

## Prerequisites

- Kubernetes 1.19+ (tested with Minikube)
- Helm 3.0+ with Bitnami repository
- OpenAI API key
- Storage class that supports ReadWriteOnce (standard storage class works)
- Minimum resources: 2 CPU cores, 4Gi memory available

## Validated Installation Steps

### Development/Local Setup (Minikube)

1. **Prepare Helm repositories**:
```bash
cd lightrag-minimal
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update
helm dependency update
```

2. **Set your OpenAI API key**:
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

3. **Deploy for development**:
```bash
# Substitute environment variables and deploy
envsubst < values-dev.yaml > values-dev-final.yaml
helm install lightrag-minimal . \
  -f values-dev-final.yaml \
  --namespace lightrag \
  --create-namespace

# Wait for deployment
kubectl wait --namespace lightrag \
  --for=condition=ready pod \
  -l app.kubernetes.io/name=postgresql \
  --timeout=120s

kubectl wait --namespace lightrag \
  --for=condition=ready pod \
  -l app.kubernetes.io/name=lightrag-minimal \
  --timeout=120s

# Clean up temporary file
rm values-dev-final.yaml

# Start port forwarding
kubectl port-forward --namespace lightrag svc/lightrag-minimal 9621:9621 &
```

### Production Setup

```bash
# Customize values-prod.yaml first (domain, storage classes, etc.)
envsubst < values-prod.yaml > values-prod-final.yaml
helm install lightrag-minimal . \
  -f values-prod-final.yaml \
  --namespace lightrag \
  --create-namespace
rm values-prod-final.yaml
```

## Configuration Options

### Validated Environment Configuration

Both `values-dev.yaml` and `values-prod.yaml` include these critical settings:

```yaml
env:
  # OpenAI API Configuration (REQUIRED)
  LLM_BINDING: "openai"
  LLM_BINDING_HOST: "https://api.openai.com/v1"
  EMBEDDING_BINDING: "openai"
  EMBEDDING_BINDING_HOST: "https://api.openai.com/v1"
  EMBEDDING_MODEL: "text-embedding-ada-002"
  EMBEDDING_DIM: "1536"
  
  # Conservative concurrency (prevents API errors)
  MAX_ASYNC: "4"
  MAX_PARALLEL_INSERT: "2"
  
  # LLM Configuration
  ENABLE_LLM_CACHE: "true"
  ENABLE_LLM_CACHE_FOR_EXTRACT: "true"
  TIMEOUT: "240"
  TEMPERATURE: "0"
  MAX_TOKENS: "32768"
```

### PostgreSQL Configuration

```yaml
postgresql:
  # CRITICAL: Use pgvector image for vector support
  image:
    registry: docker.io
    repository: pgvector/pgvector
    tag: pg16
  auth:
    password: "your-secure-password"
```

### Development vs Production

| Setting | Development | Production |
|---------|-------------|------------|
| Resources | 1 CPU, 2Gi RAM | 4 CPU, 8Gi RAM |
| Storage | 5Gi | 100Gi |
| Replicas | 1 | 2-10 (autoscaling) |
| Ingress | Disabled | Enabled with TLS |
| Storage Class | Default | `fast-ssd` |

## Accessing LightRAG

### Development Access
```bash
# Port forward (included in installation steps above)
kubectl port-forward --namespace lightrag svc/lightrag-minimal 9621:9621 &

# Access URLs
echo "Web UI: http://localhost:9621/webui"
echo "API Docs: http://localhost:9621/docs"
echo "Health Check: http://localhost:9621/health"
```

### Verify Deployment
```bash
# Check health
curl http://localhost:9621/health

# Expected response:
{
  "status": "healthy",
  "configuration": {
    "llm_model": "gpt-4o",
    "kv_storage": "PGKVStorage",
    "vector_storage": "PGVectorStorage",
    "graph_storage": "NetworkXStorage"
  }
}
```

### Production (Ingress)
Production uses ingress with TLS (see `values-prod.yaml`):
```yaml
ingress:
  enabled: true
  className: "nginx"
  hosts:
    - host: lightrag.yourdomain.com
```

## Monitoring

### Check Deployment Status
```bash
kubectl get pods -l app.kubernetes.io/name=lightrag-minimal
kubectl get services -l app.kubernetes.io/name=lightrag-minimal
```

### View Logs
```bash
kubectl logs -l app.kubernetes.io/name=lightrag-minimal -f
```

### Health Checks
The deployment includes health checks on `/health` endpoint.

## Scaling

For production workloads, consider enabling autoscaling:
```yaml
autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
```

## Upgrading

```bash
helm upgrade lightrag-minimal ./lightrag-minimal -f values-prod.yaml
```

## Uninstalling

```bash
helm uninstall lightrag-minimal
```

**Note**: This will delete all data unless you have persistent volumes with a retain policy.

## Document Loading

After successful deployment, load your documentation using the included loader. The loader supports two reference modes:

### Reference Modes

**Files Mode (Default)**: Uses file paths in citations
```bash
# Install dependencies (if needed)
pip install httpx

# Load documents with file path references
python ../../../load_docs.py /path/to/your/docs --endpoint http://localhost:9621

# Example with relative path
python ../../../load_docs.py ../docs --endpoint http://localhost:9621
```

**URLs Mode**: Uses website URLs in citations (recommended for public documentation)
```bash
# Load Apolo documentation with URL references
python ../../../load_docs.py ../apolo-copilot/docs/official-apolo-documentation/docs \
  --mode urls --base-url https://docs.apolo.us/index/ --endpoint http://localhost:9621

# Load custom documentation with URL references  
python ../../../load_docs.py /path/to/docs \
  --mode urls --base-url https://your-docs.example.com/docs/ --endpoint http://localhost:9621
```

### Benefits of URL Mode
- **Clickable References**: Query responses include direct links to source documentation
- **Better User Experience**: Users can easily navigate to original content
- **Professional Citations**: References point to live documentation sites

### ⚠️ Important: File Structure Requirements for URL Mode

**Your local file structure must match your documentation site's URL structure:**

```
# Example: GitBook documentation site
docs/
├── getting-started/
│   ├── installation.md          → https://docs.example.com/getting-started/installation
│   └── first-steps.md           → https://docs.example.com/getting-started/first-steps
├── administration/
│   ├── README.md                → https://docs.example.com/administration
│   └── setup.md                 → https://docs.example.com/administration/setup
└── README.md                    → https://docs.example.com/
```

**Quick Setup Guide:**
1. **Analyze your docs site**: Visit URLs and note the path structure
2. **Create matching directories**: `mkdir -p docs/{section1,section2,section3}`
3. **Organize markdown files**: Place files to match URL paths (remove `.md` from URLs)
4. **Verify mapping**: Test a few URLs manually before loading

**URL Mapping Rules:**
- `.md` extension is removed from URLs
- `README.md` files map to their directory URL  
- Subdirectories become URL path segments
- File and folder names should match URL slugs exactly

### Expected Output

Both modes produce similar output with different reference formats:

```bash
🚀 Loading Documentation into LightRAG
============================================================
📁 Documentation path: /path/to/docs
🔧 Reference mode: urls
🌐 Base URL: https://docs.apolo.us/index/
🌐 LightRAG endpoint: http://localhost:9621

✅ LightRAG is healthy: healthy
📚 Found 58 markdown files
🔧 Mode: urls
🌐 Base URL: https://docs.apolo.us/index/
📊 Total content: 244,400 characters
📊 Average length: 4,287 characters

🔄 Starting to load documents...
✅ Loaded: Document Title
📈 Progress: 10/58 (10 success, 0 failed)
...
✅ Loading complete!
📊 Successful: 58
📊 Failed: 0
✅ Query successful!
```

### Query Response Examples

**Files Mode References:**
```
### References
- [DC] getting-started/installation.md
- [KG] administration/cluster-setup.md
```

**URLs Mode References:**
```  
### References
- [DC] https://docs.apolo.us/index/getting-started/installation
- [KG] https://docs.apolo.us/index/administration/cluster-setup
```

## Troubleshooting

### Common Issues

**Issue: `UnsupportedProtocol: Request URL is missing protocol`**
- **Solution**: Ensure `LLM_BINDING_HOST` and `EMBEDDING_BINDING_HOST` are set to `https://api.openai.com/v1`

**Issue: Document processing failures with API connection errors**
- **Solution**: Reduce concurrency with `MAX_ASYNC: "4"` and `MAX_PARALLEL_INSERT: "2"`

**Issue: pgvector extension missing**
- **Solution**: Ensure using `pgvector/pgvector:pg16` image, not standard PostgreSQL

### Validation Commands
```bash
# Check all pods are running
kubectl get pods --namespace lightrag

# Verify API connectivity
kubectl exec --namespace lightrag \
  $(kubectl get pod -l app.kubernetes.io/name=lightrag-minimal --namespace lightrag -o jsonpath='{.items[0].metadata.name}') \
  -- python -c "import requests; print(requests.get('https://api.openai.com/v1/models', headers={'Authorization': 'Bearer ' + open('/dev/null').read()}, timeout=5).status_code)"

# Check document processing status
curl http://localhost:9621/documents | jq '.statuses | to_entries | map({status: .key, count: (.value | length)})'
```

## Clean Teardown and Rebuild

For testing or redeployment:

```bash
# Complete teardown
helm uninstall lightrag-minimal --namespace lightrag
kubectl delete namespace lightrag

# Rebuild (repeat installation steps above)
# This process has been validated multiple times
```

## Validated Features

✅ **Pure Helm Deployment** - No manual kubectl apply commands needed  
✅ **PostgreSQL with pgvector** - Automatic extension creation via proper image  
✅ **Environment Flexibility** - Separate dev/prod configurations  
✅ **Document Loading** - Working API with `file_source` parameter  
✅ **Conservative Concurrency** - Prevents OpenAI API rate limiting  
✅ **Health Monitoring** - Comprehensive health checks and status endpoints  
✅ **Persistent Storage** - Data survives pod restarts and cluster updates  

## Comparison with Docker Compose

| Feature | Docker Compose | Helm Chart |
|---------|----------------|------------|
| PostgreSQL | pgvector/pgvector:pg16 | Same image via subchart |
| Concurrency | MAX_ASYNC=4 | Same settings |
| API Configuration | .env file | Environment variables |
| Scaling | Single container | Kubernetes autoscaling |
| Persistence | Local volumes | PersistentVolumeClaims |
| Monitoring | Manual | Kubernetes native |

This chart maintains the same conservative, working configuration as the Docker Compose setup while adding Kubernetes-native features for production deployment.