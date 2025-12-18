# FR01: Memory API Ingestion - Configuration & Deployment

## Configuration File Reference

### Complete Configuration Example

```yaml
# config.yaml - Memory Connector Configuration

# ============================================================================
# LightRAG Connection Settings
# ============================================================================
lightrag:
  # Integration mode: "api" (HTTP) or "direct" (library)
  mode: "api"

  # API mode settings (used when mode: "api")
  api:
    url: "http://localhost:9621"
    api_key: "${LIGHTRAG_API_KEY}"  # Environment variable
    workspace: "memories"
    timeout: 300  # Request timeout in seconds

  # Direct mode settings (used when mode: "direct")
  direct:
    working_dir: "./lightrag_storage"
    # LLM settings inherited from environment variables:
    # - LLM_BINDING=openai
    # - LLM_MODEL=gpt-4o-mini
    # - OPENAI_API_KEY=...
    # - EMBEDDING_BINDING=openai
    # - EMBEDDING_MODEL=text-embedding-3-small

# ============================================================================
# Memory API Settings
# ============================================================================
memory_api:
  url: "http://127.0.0.1:8080"
  api_key: "${MEMORY_API_KEY}"  # Environment variable
  timeout: 30  # Request timeout in seconds
  max_retries: 3  # Retry attempts on failure
  retry_backoff: 2.0  # Exponential backoff multiplier

# ============================================================================
# Connector Definitions
# ============================================================================
connectors:
  # Personal memories connector
  - id: "personal-memories"
    enabled: true
    description: "Personal memory sync"
    context_id: "CTX123"

    # Schedule configuration
    schedule:
      type: "interval"  # "interval" or "cron"
      interval_hours: 1  # Run every 1 hour (for interval type)
      # For cron type, use:
      # type: "cron"
      # cron: "0 */1 * * *"  # Hourly at minute 0

    # Ingestion settings
    ingestion:
      query_range: "week"  # API range parameter (day, week, month)
      query_limit: 100     # Max items per query
      batch_size: 10       # Memories to process in parallel
      skip_empty_transcripts: true  # Skip memories with no transcript

    # Transformation settings
    transformation:
      strategy: "standard"  # "standard" or "rich"
      include_audio: false  # Download and process audio files
      include_image: false  # Download and process image files
      geocoding: false      # Reverse geocode locations
      extract_tags: true    # Extract hashtags from transcripts

    # Retry settings for failed items
    retry:
      max_attempts: 3
      backoff_multiplier: 2.0
      max_backoff_seconds: 60

  # Work memories connector (disabled by default)
  - id: "work-memories"
    enabled: false
    description: "Work memory sync (weekdays only)"
    context_id: "CTX456"

    schedule:
      type: "cron"
      cron: "0 9,17 * * 1-5"  # 9am and 5pm on weekdays

    ingestion:
      query_range: "day"
      query_limit: 50
      batch_size: 5
      skip_empty_transcripts: true

    transformation:
      strategy: "standard"
      include_audio: false
      include_image: false

    retry:
      max_attempts: 3

# ============================================================================
# State Management
# ============================================================================
state:
  backend: "json"  # "json" or "sqlite"
  path: "./memory_sync_state.json"

  # For SQLite backend:
  # backend: "sqlite"
  # path: "./memory_sync_state.db"

  # State cleanup settings
  cleanup:
    enabled: true
    retention_days: 90  # Keep history for 90 days
    cleanup_cron: "0 2 * * *"  # Run cleanup at 2am daily

# ============================================================================
# Management API Server
# ============================================================================
api:
  enabled: true
  host: "0.0.0.0"
  port: 9622

  # Authentication
  enable_auth: true
  api_key: "${CONNECTOR_API_KEY}"  # Environment variable

  # CORS settings
  cors:
    enabled: true
    origins:
      - "http://localhost:3000"
      - "http://localhost:9621"

  # Rate limiting (optional)
  rate_limit:
    enabled: false
    requests_per_minute: 60

# ============================================================================
# Logging Configuration
# ============================================================================
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
  format: "json"  # "json" or "text"

  # File logging
  file:
    enabled: true
    path: "./logs/memory_connector.log"
    max_bytes: 10485760  # 10MB
    backup_count: 5  # Keep 5 backup files
    rotation: "size"  # "size" or "time"

  # Console logging
  console:
    enabled: true
    colorize: true

  # Log specific components
  components:
    scheduler: "INFO"
    api_client: "INFO"
    orchestrator: "INFO"
    state_manager: "DEBUG"

# ============================================================================
# Monitoring & Alerting (Optional)
# ============================================================================
monitoring:
  # Prometheus metrics
  prometheus:
    enabled: false
    port: 9623
    path: "/metrics"

  # Health check
  health_check:
    enabled: true
    endpoint: "/health"

  # Alerts (webhook notifications)
  alerts:
    enabled: false
    webhook_url: "${ALERT_WEBHOOK_URL}"
    alert_on:
      - consecutive_failures: 3  # Alert after 3 consecutive failures
      - no_sync_for_hours: 24    # Alert if no sync for 24 hours

# ============================================================================
# Advanced Settings
# ============================================================================
advanced:
  # Graceful shutdown timeout
  shutdown_timeout: 30

  # Pipeline settings
  pipeline:
    max_concurrent_connectors: 2  # Max connectors running simultaneously
    job_timeout_minutes: 60       # Max time for single ingestion job

  # Performance tuning
  performance:
    enable_caching: true
    cache_ttl_seconds: 3600

  # Security
  security:
    encrypt_secrets: false  # Encrypt API keys in state file
    encryption_key: "${ENCRYPTION_KEY}"
```

## Environment Variables

### Required Variables

```bash
# Memory API credentials
export MEMORY_API_KEY="your-memory-api-key"

# LightRAG API credentials (if using API mode)
export LIGHTRAG_API_KEY="your-lightrag-api-key"

# Connector API credentials (if authentication enabled)
export CONNECTOR_API_KEY="your-connector-api-key"
```

### Optional Variables

```bash
# LightRAG LLM settings (for direct mode)
export LLM_BINDING="openai"
export LLM_MODEL="gpt-4o-mini"
export OPENAI_API_KEY="your-openai-api-key"
export EMBEDDING_BINDING="openai"
export EMBEDDING_MODEL="text-embedding-3-small"

# Logging
export LOG_LEVEL="INFO"
export LOG_FORMAT="json"

# Monitoring
export ALERT_WEBHOOK_URL="https://hooks.slack.com/..."

# Security
export ENCRYPTION_KEY="your-32-character-encryption-key"
```

### Using .env File

Create `.env` file:

```bash
# .env
MEMORY_API_KEY=your-memory-api-key
LIGHTRAG_API_KEY=your-lightrag-api-key
CONNECTOR_API_KEY=your-connector-api-key
OPENAI_API_KEY=your-openai-api-key
```

The connector will automatically load from `.env` file.

## Deployment Scenarios

### Scenario 1: Development (Local)

**Setup**:
```bash
# 1. Clone repository
git clone https://github.com/your/lightrag.git
cd lightrag

# 2. Install dependencies
poetry install

# 3. Create config
cp config.example.yaml config.yaml
# Edit config.yaml with your settings

# 4. Create .env file
cat > .env <<EOF
MEMORY_API_KEY=your-key
LIGHTRAG_API_KEY=your-key
CONNECTOR_API_KEY=dev-key
EOF

# 5. Test one-time sync
poetry run python -m memory_connector sync \
  --config config.yaml \
  --connector-id personal-memories

# 6. Start service
poetry run python -m memory_connector serve \
  --config config.yaml
```

**Recommended Settings**:
```yaml
lightrag:
  mode: "api"
  api:
    url: "http://localhost:9621"

state:
  backend: "json"

logging:
  level: "DEBUG"
  format: "text"
  console:
    enabled: true
    colorize: true
```

### Scenario 2: Production (Server)

**Setup**:
```bash
# 1. Install as package
pip install lightrag[connectors]

# 2. Create config directory
sudo mkdir -p /etc/memory-connector
sudo cp config.yaml /etc/memory-connector/

# 3. Create data directory
sudo mkdir -p /var/lib/memory-connector
sudo chown -R connector:connector /var/lib/memory-connector

# 4. Set environment variables
sudo vi /etc/environment
# Add API keys

# 5. Create systemd service
sudo cp memory-connector.service /etc/systemd/system/
sudo systemctl daemon-reload

# 6. Start service
sudo systemctl start memory-connector
sudo systemctl enable memory-connector

# 7. Check status
sudo systemctl status memory-connector
sudo journalctl -u memory-connector -f
```

**Systemd Service File** (`memory-connector.service`):
```ini
[Unit]
Description=Memory Connector Service
After=network.target

[Service]
Type=simple
User=connector
Group=connector
WorkingDirectory=/var/lib/memory-connector
Environment="CONFIG_PATH=/etc/memory-connector/config.yaml"
EnvironmentFile=/etc/memory-connector/.env
ExecStart=/usr/local/bin/memory-connector serve --config ${CONFIG_PATH}
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

**Recommended Settings**:
```yaml
lightrag:
  mode: "api"
  api:
    url: "http://lightrag-service:9621"

state:
  backend: "sqlite"
  path: "/var/lib/memory-connector/state.db"

logging:
  level: "INFO"
  format: "json"
  file:
    enabled: true
    path: "/var/log/memory-connector/app.log"
```

### Scenario 3: Docker Container

**Dockerfile**:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml poetry.lock ./
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev --extras connectors

# Copy application
COPY lightrag/ ./lightrag/
COPY memory_connector/ ./memory_connector/

# Create data directory
RUN mkdir -p /data

# Expose API port
EXPOSE 9622

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:9622/health || exit 1

# Run service
CMD ["python", "-m", "memory_connector", "serve", "--config", "/config/config.yaml"]
```

**docker-compose.yaml**:
```yaml
version: '3.8'

services:
  lightrag:
    image: lightrag:latest
    ports:
      - "9621:9621"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - lightrag-data:/app/storage

  memory-connector:
    build: .
    ports:
      - "9622:9622"
    environment:
      - MEMORY_API_KEY=${MEMORY_API_KEY}
      - LIGHTRAG_API_KEY=${LIGHTRAG_API_KEY}
      - CONNECTOR_API_KEY=${CONNECTOR_API_KEY}
    volumes:
      - ./config.yaml:/config/config.yaml:ro
      - connector-data:/data
    depends_on:
      - lightrag
    restart: unless-stopped

volumes:
  lightrag-data:
  connector-data:
```

**Usage**:
```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f memory-connector

# Check status
curl http://localhost:9622/health

# Stop
docker-compose down
```

### Scenario 4: Kubernetes Deployment

**ConfigMap** (`configmap.yaml`):
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: memory-connector-config
data:
  config.yaml: |
    lightrag:
      mode: "api"
      api:
        url: "http://lightrag-service:9621"
        api_key: "${LIGHTRAG_API_KEY}"
        workspace: "memories"

    memory_api:
      url: "http://memory-api-service:8080"
      api_key: "${MEMORY_API_KEY}"

    connectors:
      - id: "personal-memories"
        enabled: true
        context_id: "CTX123"
        schedule:
          type: "interval"
          interval_hours: 1
        ingestion:
          query_range: "week"
          query_limit: 100

    state:
      backend: "sqlite"
      path: "/data/state.db"

    logging:
      level: "INFO"
      format: "json"
```

**Secret** (`secret.yaml`):
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: memory-connector-secrets
type: Opaque
stringData:
  MEMORY_API_KEY: "your-memory-api-key"
  LIGHTRAG_API_KEY: "your-lightrag-api-key"
  CONNECTOR_API_KEY: "your-connector-api-key"
```

**Deployment** (`deployment.yaml`):
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: memory-connector
spec:
  replicas: 1
  selector:
    matchLabels:
      app: memory-connector
  template:
    metadata:
      labels:
        app: memory-connector
    spec:
      containers:
      - name: memory-connector
        image: memory-connector:latest
        ports:
        - containerPort: 9622
        env:
        - name: MEMORY_API_KEY
          valueFrom:
            secretKeyRef:
              name: memory-connector-secrets
              key: MEMORY_API_KEY
        - name: LIGHTRAG_API_KEY
          valueFrom:
            secretKeyRef:
              name: memory-connector-secrets
              key: LIGHTRAG_API_KEY
        - name: CONNECTOR_API_KEY
          valueFrom:
            secretKeyRef:
              name: memory-connector-secrets
              key: CONNECTOR_API_KEY
        volumeMounts:
        - name: config
          mountPath: /config
          readOnly: true
        - name: data
          mountPath: /data
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 9622
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 9622
          initialDelaySeconds: 5
          periodSeconds: 10
      volumes:
      - name: config
        configMap:
          name: memory-connector-config
      - name: data
        persistentVolumeClaim:
          claimName: memory-connector-data
---
apiVersion: v1
kind: Service
metadata:
  name: memory-connector-service
spec:
  selector:
    app: memory-connector
  ports:
  - protocol: TCP
    port: 9622
    targetPort: 9622
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: memory-connector-data
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
```

**Deploy**:
```bash
kubectl apply -f secret.yaml
kubectl apply -f configmap.yaml
kubectl apply -f deployment.yaml

# Check status
kubectl get pods -l app=memory-connector
kubectl logs -f deployment/memory-connector

# Port forward for testing
kubectl port-forward deployment/memory-connector 9622:9622
```

## CLI Usage

### Commands

#### `serve` - Start service

Start the connector service with scheduler and API:

```bash
memory-connector serve --config config.yaml

# Options:
  --config PATH        Configuration file path
  --api-only          Run only API server (no scheduler)
  --scheduler-only    Run only scheduler (no API)
  --host HOST         API host (default: from config)
  --port PORT         API port (default: from config)
```

#### `sync` - One-time sync

Run a one-time sync for a connector:

```bash
memory-connector sync \
  --config config.yaml \
  --connector-id personal-memories

# Options:
  --config PATH           Configuration file path
  --connector-id ID       Connector ID to sync
  --force                Force full resync (ignore state)
```

#### `status` - Check status

Check sync status for a connector:

```bash
memory-connector status \
  --config config.yaml \
  --connector-id personal-memories

# Output:
# Connector: personal-memories
# Status: completed
# Last Sync: 2025-12-18 15:00:00 UTC
# Processed: 145 memories
# Failed: 2 memories
# Next Run: 2025-12-18 16:00:00 UTC
```

#### `list` - List connectors

List all configured connectors:

```bash
memory-connector list --config config.yaml

# Output:
# ID                   Status    Enabled  Last Sync
# personal-memories    idle      yes      2025-12-18 15:00:00
# work-memories        disabled  no       -
```

#### `trigger` - Manual trigger

Manually trigger a connector:

```bash
memory-connector trigger \
  --config config.yaml \
  --connector-id personal-memories

# This will start the sync immediately (async)
```

#### `reset` - Reset state

Reset sync state for a connector (force full resync):

```bash
memory-connector reset \
  --config config.yaml \
  --connector-id personal-memories

# WARNING: This will reprocess all memories!
```

## Monitoring & Operations

### Health Check

```bash
# Check service health
curl http://localhost:9622/health

# Response:
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 3600,
  "active_jobs": 1,
  "last_error": null
}
```

### View Logs

```bash
# Tail logs (systemd)
sudo journalctl -u memory-connector -f

# Tail logs (docker)
docker-compose logs -f memory-connector

# Tail logs (file)
tail -f /var/log/memory-connector/app.log
```

### Metrics (Prometheus)

Enable Prometheus metrics in `config.yaml`:

```yaml
monitoring:
  prometheus:
    enabled: true
    port: 9623
```

**Metrics available**:
- `memory_connector_syncs_total{connector_id, status}` - Total syncs
- `memory_connector_memories_processed{connector_id}` - Memories processed
- `memory_connector_sync_duration_seconds{connector_id}` - Sync duration
- `memory_connector_errors_total{connector_id, error_type}` - Errors

**Query examples**:
```promql
# Sync success rate
rate(memory_connector_syncs_total{status="success"}[5m])

# Average sync duration
avg(memory_connector_sync_duration_seconds) by (connector_id)

# Error rate
rate(memory_connector_errors_total[5m])
```

### Troubleshooting

#### Issue: Connector not running

```bash
# Check scheduler status
memory-connector status --config config.yaml

# Check logs for errors
sudo journalctl -u memory-connector -n 100

# Check if connector is enabled
cat config.yaml | grep -A 20 "id: personal-memories"
```

#### Issue: Duplicate processing

```bash
# Check state file
cat memory_sync_state.json | jq '.connectors."personal-memories"'

# If corrupted, reset state
memory-connector reset --config config.yaml --connector-id personal-memories
```

#### Issue: Memory API connection failures

```bash
# Test API connectivity
curl -H "X-API-KEY: your-key" http://127.0.0.1:8080/memory/CTX123

# Check DNS resolution
nslookup memory-api-host

# Check network connectivity
ping memory-api-host
```

## Backup & Recovery

### Backup State

```bash
# Backup JSON state
cp memory_sync_state.json memory_sync_state.json.backup

# Backup SQLite state
sqlite3 memory_sync_state.db ".backup memory_sync_state.db.backup"

# Automated backup (cron)
0 2 * * * /usr/local/bin/backup-memory-connector.sh
```

### Restore State

```bash
# Restore JSON state
cp memory_sync_state.json.backup memory_sync_state.json

# Restore SQLite state
cp memory_sync_state.db.backup memory_sync_state.db

# Restart service
sudo systemctl restart memory-connector
```

## Security Best Practices

1. **API Keys**:
   - Store in environment variables or secrets manager
   - Never commit to version control
   - Rotate regularly

2. **File Permissions**:
   ```bash
   chmod 600 config.yaml
   chmod 600 .env
   chmod 700 /var/lib/memory-connector
   ```

3. **Network Security**:
   - Use HTTPS for all API connections
   - Restrict API access with firewall rules
   - Use VPN for remote connections

4. **Encryption**:
   - Enable secrets encryption in config
   - Use TLS for LightRAG API connection
   - Encrypt state files at rest

## Next Steps

See `05-FUTURE-ENHANCEMENTS.md` for roadmap and future features.
