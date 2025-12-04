# LightRAG Deployment Guide

## Deployment Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DEPLOYMENT OPTIONS                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │   LOCAL      │    │   DOCKER     │    │   K8S/HELM   │                  │
│  │   (Dev)      │    │   (Staging)  │    │   (Prod)     │                  │
│  ├──────────────┤    ├──────────────┤    ├──────────────┤                  │
│  │ pip install  │    │ docker-      │    │ helm install │                  │
│  │ python -m    │    │ compose up   │    │ kubectl      │                  │
│  │ lightrag.api │    │              │    │ apply        │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
│        │                   │                   │                            │
│        └───────────────────┴───────────────────┘                            │
│                            │                                                 │
│        ┌───────────────────┴───────────────────┐                            │
│        │         STORAGE TOPOLOGY              │                            │
│        ├───────────────────────────────────────┤                            │
│        │  Dev: JSON + NetworkX + NanoVectorDB  │                            │
│        │  Prod: PostgreSQL + Neo4j + pgvector  │                            │
│        └───────────────────────────────────────┘                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Local Development Setup

### Quick Install

```bash
# Clone repository
git clone https://github.com/HKUDS/LightRAG.git
cd LightRAG

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install with API dependencies
pip install -e ".[api]"
```

### Storage Dependencies

```bash
# Lightweight (development)
pip install nano-vectordb networkx

# PostgreSQL (production)
pip install asyncpg psycopg2-binary pgvector

# Neo4j (production graph)
pip install neo4j

# Additional vector stores
pip install pymilvus qdrant-client redis faiss-cpu
```

### Running the Server

```bash
# Set environment variables
export OPENAI_API_KEY="sk-xxx"
export LLM_MODEL="gpt-4o-mini"

# Start server
python -m lightrag.api.lightrag_server
# Server starts at http://localhost:9621
```

---

## 2. Docker Deployment

### Docker Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         DOCKER COMPOSE STACK                              │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                       lightrag:9621                                  │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐│ │
│  │  │  python:3.12-slim + LightRAG API                                ││ │
│  │  │                                                                  ││ │
│  │  │  Volumes:                                                        ││ │
│  │  │    /app/data/rag_storage  → ./data/rag_storage                  ││ │
│  │  │    /app/data/inputs       → ./data/inputs                       ││ │
│  │  │    /app/config.ini        → ./config.ini                        ││ │
│  │  │    /app/.env              → ./.env                              ││ │
│  │  └─────────────────────────────────────────────────────────────────┘│ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                  │                                        │
│                                  ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │              External Services (optional)                           │ │
│  │                                                                      │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │ │
│  │  │  PostgreSQL  │  │    Neo4j     │  │    Redis     │              │ │
│  │  │    :5432     │  │    :7687     │  │    :6379     │              │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

### docker-compose.yml

```yaml
services:
  lightrag:
    container_name: lightrag
    image: ghcr.io/hkuds/lightrag:latest
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "${PORT:-9621}:9621"
    volumes:
      - ./lightrag:/app/lightrag
      - ./data/rag_storage:/app/data/rag_storage
      - ./data/inputs:/app/data/inputs
      - ./data/tiktoken:/app/data/tiktoken
      - ./config.ini:/app/config.ini
      - ./.env:/app/.env
    env_file:
      - .env
    environment:
      - TIKTOKEN_CACHE_DIR=/app/data/tiktoken
      - INIT_DEMO_TENANTS=true
      - AUTH_USER=admin
      - AUTH_PASS=admin123
    restart: unless-stopped
    extra_hosts:
      - "host.docker.internal:host-gateway"
```

### Environment File (.env)

```bash
# LLM Configuration
LLM_BINDING=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-xxx

# Embedding Configuration
EMBEDDING_BINDING=openai
EMBEDDING_MODEL=text-embedding-ada-002
EMBEDDING_DIM=1536

# Storage (lightweight)
LIGHTRAG_KV_STORAGE=JsonKVStorage
LIGHTRAG_VECTOR_STORAGE=NanoVectorDBStorage
LIGHTRAG_GRAPH_STORAGE=NetworkXStorage
LIGHTRAG_DOC_STATUS_STORAGE=JsonDocStatusStorage

# Server
HOST=0.0.0.0
PORT=9621

# Multi-tenancy
ENABLE_MULTI_TENANTS=false
```

### Docker Commands

```bash
# Start with pre-built image
docker-compose up -d

# Build and start locally
docker-compose up -d --build

# View logs
docker-compose logs -f lightrag

# Stop
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

### Development with External Databases

```bash
# Start only PostgreSQL and Redis for local development
docker-compose -f docker-compose.dev-db.yml up -d

# Services started:
# - PostgreSQL: localhost:15432
# - Redis: localhost:16379
```

---

## 3. Kubernetes Deployment (Helm)

### Prerequisites

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Kubernetes | 1.20+ | 1.26+ |
| Helm | 3.x | 3.12+ |
| Memory | 4 GB | 8 GB |
| CPUs | 2 | 4 |

### Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         KUBERNETES CLUSTER                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  namespace: rag                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                                                                          ││
│  │  ┌──────────────────────────────────────────────────────────────────┐   ││
│  │  │  Deployment: lightrag                                             │   ││
│  │  │  ┌──────────────────────────────────────────────────────────────┐│   ││
│  │  │  │  Pod: lightrag-xxxx-yyyy                                     ││   ││
│  │  │  │  ┌────────────────────────────────────────────────────────┐  ││   ││
│  │  │  │  │  Container: lightrag                                   │  ││   ││
│  │  │  │  │  Image: ghcr.io/hkuds/lightrag:latest                 │  ││   ││
│  │  │  │  │  Port: 9621                                            │  ││   ││
│  │  │  │  │                                                         │  ││   ││
│  │  │  │  │  Probes:                                               │  ││   ││
│  │  │  │  │    readiness: GET /health                              │  ││   ││
│  │  │  │  └────────────────────────────────────────────────────────┘  ││   ││
│  │  │  │  Volumes:                                                    ││   ││
│  │  │  │    - rag-storage (PVC)                                      ││   ││
│  │  │  │    - inputs (PVC)                                           ││   ││
│  │  │  │    - env-file (Secret)                                      ││   ││
│  │  │  └──────────────────────────────────────────────────────────────┘│   ││
│  │  └──────────────────────────────────────────────────────────────────┘   ││
│  │                                                                          ││
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐       ││
│  │  │ Service: lightrag│  │ PVC: rag-storage │  │ Secret: env      │       ││
│  │  │ Type: ClusterIP  │  │ Size: 10Gi       │  │ (API keys)       │       ││
│  │  │ Port: 9621       │  │                  │  │                  │       ││
│  │  └──────────────────┘  └──────────────────┘  └──────────────────┘       ││
│  │                                                                          ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  StatefulSet: pg-cluster         │  StatefulSet: neo4j-cluster         ││
│  │  (via KubeBlocks)                │  (via KubeBlocks)                   ││
│  │  - postgresql-0                  │  - neo4j-0                          ││
│  │  - postgresql-1                  │                                      ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Lightweight Deployment (Development)

```bash
# Set API credentials
export OPENAI_API_BASE=https://api.openai.com/v1
export OPENAI_API_KEY=sk-xxx

# Install using convenience script
bash ./k8s-deploy/install_lightrag_dev.sh

# Or using Helm directly
helm upgrade --install lightrag ./k8s-deploy/lightrag \
  --namespace rag --create-namespace \
  --set-string env.LIGHTRAG_KV_STORAGE=JsonKVStorage \
  --set-string env.LIGHTRAG_VECTOR_STORAGE=NanoVectorDBStorage \
  --set-string env.LIGHTRAG_GRAPH_STORAGE=NetworkXStorage \
  --set-string env.LIGHTRAG_DOC_STATUS_STORAGE=JsonDocStatusStorage \
  --set-string env.LLM_BINDING=openai \
  --set-string env.LLM_MODEL=gpt-4o-mini \
  --set-string env.LLM_BINDING_HOST=$OPENAI_API_BASE \
  --set-string env.LLM_BINDING_API_KEY=$OPENAI_API_KEY \
  --set-string env.EMBEDDING_BINDING=openai \
  --set-string env.EMBEDDING_MODEL=text-embedding-ada-002 \
  --set-string env.EMBEDDING_DIM=1536 \
  --set-string env.EMBEDDING_BINDING_API_KEY=$OPENAI_API_KEY
```

### Production Deployment (with Databases)

```bash
# 1. Install KubeBlocks for database management
bash ./k8s-deploy/databases/01-prepare.sh

# 2. Install PostgreSQL and Neo4j
bash ./k8s-deploy/databases/02-install-database.sh

# 3. Verify databases are running
kubectl get clusters -n rag
# NAME            STATUS     AGE
# neo4j-cluster   Running    39s
# pg-cluster      Running    42s

# 4. Install LightRAG (auto-configures database connections)
export OPENAI_API_BASE=https://api.openai.com/v1
export OPENAI_API_KEY=sk-xxx
bash ./k8s-deploy/install_lightrag.sh
```

### Helm Values Configuration

```yaml
# k8s-deploy/lightrag/values.yaml
replicaCount: 1

image:
  repository: ghcr.io/hkuds/lightrag
  tag: latest

resources:
  limits:
    cpu: 1000m
    memory: 2Gi
  requests:
    cpu: 500m
    memory: 1Gi

persistence:
  enabled: true
  ragStorage:
    size: 10Gi
  inputs:
    size: 5Gi

env:
  HOST: 0.0.0.0
  PORT: 9621
  
  # LLM Configuration
  LLM_BINDING: openai
  LLM_MODEL: gpt-4o-mini
  LLM_BINDING_HOST: ""
  LLM_BINDING_API_KEY: ""
  
  # Embedding Configuration
  EMBEDDING_BINDING: openai
  EMBEDDING_MODEL: text-embedding-ada-002
  EMBEDDING_DIM: 1536
  
  # Storage Configuration (Production)
  LIGHTRAG_KV_STORAGE: PGKVStorage
  LIGHTRAG_VECTOR_STORAGE: PGVectorStorage
  LIGHTRAG_GRAPH_STORAGE: Neo4JStorage
  LIGHTRAG_DOC_STATUS_STORAGE: PGDocStatusStorage
```

### Accessing the Application

```bash
# Port-forward to local machine
kubectl --namespace rag port-forward svc/lightrag 9621:9621

# Access at http://localhost:9621
```

---

## 4. Storage Configuration by Environment

### Storage Topology Reference

| Environment | KV Storage | Vector Storage | Graph Storage | Use Case |
|-------------|------------|----------------|---------------|----------|
| Development | JsonKVStorage | NanoVectorDBStorage | NetworkXStorage | Local testing |
| Staging | RedisKVStorage | MilvusVectorDBStorage | Neo4JStorage | Integration testing |
| Production | PGKVStorage | PGVectorStorage | Neo4JStorage | Production workloads |
| High-Scale | RedisKVStorage | MilvusVectorDBStorage | Neo4JStorage | Large datasets |

### PostgreSQL Setup (Production)

```bash
# Required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS age;

# Environment variables
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=lightrag
POSTGRES_PASSWORD=secret
POSTGRES_DATABASE=lightrag

# Storage configuration
LIGHTRAG_KV_STORAGE=PGKVStorage
LIGHTRAG_VECTOR_STORAGE=PGVectorStorage
LIGHTRAG_GRAPH_STORAGE=AGEStorage  # or Neo4JStorage
LIGHTRAG_DOC_STATUS_STORAGE=PGDocStatusStorage
```

### Neo4j Setup (Production Graph)

```bash
# Environment variables
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=secret

# Storage configuration
LIGHTRAG_GRAPH_STORAGE=Neo4JStorage
```

---

## 5. Health & Monitoring

### Health Endpoint

```bash
# Check server health
curl http://localhost:9621/health

# Response
{
  "status": "healthy",
  "version": "1.4.9.1"
}
```

### Docker Health Check

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:9621/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s
```

### Kubernetes Probes

```yaml
readinessProbe:
  httpGet:
    path: /health
    port: 9621
  initialDelaySeconds: 10
  periodSeconds: 5
  timeoutSeconds: 2
  failureThreshold: 3

livenessProbe:
  httpGet:
    path: /health
    port: 9621
  initialDelaySeconds: 30
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3
```

---

## 6. Security Best Practices

### Authentication

```bash
# Enable API authentication
AUTH_ENABLED=true
AUTH_USER=admin
AUTH_PASS=<secure-password>

# JWT Token authentication
AUTH_SECRET_KEY=<32-byte-secret>
AUTH_ALGORITHM=HS256
AUTH_ACCESS_TOKEN_EXPIRE_MINUTES=30
```

### Secrets Management

```bash
# Never commit secrets to version control
# Use environment variables or secret managers

# Kubernetes Secrets
kubectl create secret generic lightrag-secrets \
  --from-literal=OPENAI_API_KEY=sk-xxx \
  --from-literal=POSTGRES_PASSWORD=xxx \
  --namespace rag
```

### Network Security

```yaml
# Kubernetes NetworkPolicy
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: lightrag-network-policy
  namespace: rag
spec:
  podSelector:
    matchLabels:
      app: lightrag
  policyTypes:
    - Ingress
    - Egress
  ingress:
    - from:
        - podSelector: {}
      ports:
        - port: 9621
```

---

## 7. Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Port 9621 in use | Another service running | Change PORT env var or stop conflicting service |
| LLM API errors | Invalid API key | Verify OPENAI_API_KEY or LLM_BINDING_API_KEY |
| Database connection failed | Wrong credentials | Check POSTGRES_* or NEO4J_* env vars |
| Out of memory | Large document processing | Increase container memory limits |
| Tiktoken cache errors | Missing cache directory | Set TIKTOKEN_CACHE_DIR |

### Debug Commands

```bash
# Docker logs
docker logs -f lightrag

# Kubernetes pod logs
kubectl logs -f deployment/lightrag -n rag

# Check pod status
kubectl describe pod -l app=lightrag -n rag

# Database connectivity
kubectl exec -it deployment/lightrag -n rag -- \
  python -c "import asyncpg; print('PostgreSQL OK')"
```

### Log Levels

```bash
# Set log level
LOG_LEVEL=DEBUG  # DEBUG, INFO, WARNING, ERROR

# Enable verbose logging for specific modules
LIGHTRAG_LOG_LEVEL=DEBUG
```

---

## Quick Reference

### Docker Commands

```bash
docker-compose up -d                    # Start
docker-compose down                     # Stop
docker-compose logs -f lightrag         # View logs
docker-compose restart lightrag         # Restart
docker-compose exec lightrag bash       # Shell access
```

### Helm Commands

```bash
helm install lightrag ./k8s-deploy/lightrag -n rag    # Install
helm upgrade lightrag ./k8s-deploy/lightrag -n rag    # Upgrade
helm uninstall lightrag -n rag                        # Uninstall
helm status lightrag -n rag                           # Status
helm get values lightrag -n rag                       # Get values
```

### kubectl Commands

```bash
kubectl get pods -n rag                               # List pods
kubectl logs -f deployment/lightrag -n rag           # View logs
kubectl port-forward svc/lightrag 9621:9621 -n rag   # Port forward
kubectl exec -it deployment/lightrag -n rag -- bash  # Shell access
```

---

**Related Documentation:**
- [Architecture Overview](0002-architecture-overview.md)
- [Configuration Reference](0007-configuration-reference.md)
- [Storage Backends](0004-storage-backends.md)
- [API Reference](0003-api-reference.md)
