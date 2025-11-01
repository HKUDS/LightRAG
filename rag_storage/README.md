# Pre-built Knowledge Graph for Docker Deployments

This directory can contain a pre-built knowledge graph that will be included in Docker images, enabling instant query capability without re-indexing.

## Benefits

- **💰 Cost Savings**: No embedding API costs in production
- **⚡ Fast Startup**: Instant query capability (no indexing delay)
- **🔄 Consistency**: Same embeddings across all deployments
- **📦 Portable**: Ship ready-to-query Docker images

## Usage

### 1. Build Your Knowledge Graph Locally

```bash
# Index your documents locally
python -m lightrag.examples.lightrag_api_openai_compatible_demo

# Or use the API
curl -X POST http://localhost:9621/insert \
  -H "Content-Type: application/json" \
  -d '{"text": "Your document content here"}'
```

This will create `graph_chunk_entity_relation.graphml` in your local `rag_storage/` directory.

### 2. Build Docker Image with Pre-built Graph

```bash
# Ensure graph file exists
ls rag_storage/graph_chunk_entity_relation.graphml

# Build Docker image (graph will be included automatically)
docker build -t lightrag:prebuilt .
```

### 3. Deploy Without Re-indexing

```bash
# Run container - queries work immediately
docker run -p 9621:9621 \
  -e OPENAI_API_KEY=your_key \
  lightrag:prebuilt

# Test query (no indexing needed!)
curl -X POST http://localhost:9621/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is LightRAG?", "mode": "hybrid"}'
```

## How It Works

### Dockerfile Integration

The Dockerfile includes this optional step:

```dockerfile
# Copy pre-built knowledge graph if available (optional)
COPY --chown=root:root rag_storage/graph_chunk_entity_relation.graphml /app/data/rag_storage/
```

### .dockerignore Configuration

```
# Exclude rag_storage but allow pre-built knowledge graph (optional)
/rag_storage/*
!/rag_storage/graph_chunk_entity_relation.graphml
```

### Build Behavior

- **With graph file**: File is copied into image → instant queries
- **Without graph file**: Build continues normally → index at runtime

## File Format

The `graph_chunk_entity_relation.graphml` file contains:
- **Entities**: Extracted from documents
- **Relationships**: Connections between entities
- **Chunks**: Document segments with embeddings
- **Metadata**: Source information and timestamps

## Use Cases

### ✅ Good Use Cases

- **Production deployments** with stable document corpus
- **Demo/POC environments** with sample data
- **Multi-region deployments** with consistent data
- **Offline deployments** without embedding API access
- **Cost optimization** for large document sets

### ⚠️ Consider Alternatives

- **Frequently updated content**: Use volume mounts instead
- **User-specific data**: Mount per-user graph files
- **Dynamic indexing**: Let containers index at runtime

## Advanced Usage

### Multiple Graph Files

To include multiple pre-built graphs:

```dockerfile
# Custom Dockerfile
COPY rag_storage/*.graphml /app/data/rag_storage/
```

Update `.dockerignore`:
```
/rag_storage/*
!/rag_storage/*.graphml
```

### Volume Override

Even with pre-built graph, you can override at runtime:

```bash
# Use custom graph file
docker run -p 9621:9621 \
  -v /path/to/custom/graph:/app/data/rag_storage \
  lightrag:prebuilt
```

### Multi-stage Builds

For CI/CD pipelines:

```dockerfile
# Stage 1: Build graph
FROM lightrag:base AS indexer
COPY documents/ /documents/
RUN python index_documents.py /documents

# Stage 2: Production image with graph
FROM lightrag:base AS production
COPY --from=indexer /app/data/rag_storage/*.graphml /app/data/rag_storage/
```

## Troubleshooting

### Graph Not Loaded

**Symptoms**: Container queries return empty results

**Check**:
```bash
# Verify graph file in image
docker run lightrag:prebuilt ls -lh /app/data/rag_storage/

# Check logs
docker logs <container_id>
```

### Build Fails

**Error**: `COPY failed: file not found`

**Solution**: This means the Dockerfile expects the graph file but it doesn't exist. Either:
1. Create the graph file before building
2. Remove the COPY instruction for optional builds

### Wrong Graph Loaded

**Issue**: Old data in queries

**Solution**:
```bash
# Rebuild image with new graph
rm rag_storage/graph_chunk_entity_relation.graphml
python rebuild_index.py
docker build --no-cache -t lightrag:prebuilt .
```

## Best Practices

1. **Version your graph files**: Tag Docker images with graph versions
   ```bash
   docker build -t lightrag:v1.0-graph-20250101 .
   ```

2. **Document graph contents**: Add metadata file
   ```bash
   echo "Built: 2025-01-01, Documents: 1000, Entities: 5000" > rag_storage/graph_metadata.txt
   ```

3. **Test before deploying**:
   ```bash
   # Validate graph locally
   python -m lightrag.tools.validate_graph rag_storage/graph_chunk_entity_relation.graphml
   ```

4. **Monitor graph size**:
   ```bash
   # Check file size
   du -h rag_storage/graph_chunk_entity_relation.graphml
   ```

## Security Considerations

- **Sensitive Data**: Don't include confidential information in public images
- **Access Control**: Use private registries for graphs with proprietary data
- **Compliance**: Ensure graph data complies with data residency requirements

## Performance Tips

- **Graph Size**: Optimize for < 100MB for faster image pulls
- **Compression**: GraphML compresses well with gzip
- **Caching**: Use Docker layer caching for unchanged graphs

---

**Note**: This feature is optional. LightRAG works without pre-built graphs by indexing at runtime.
