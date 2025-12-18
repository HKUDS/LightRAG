# Memory Connector for LightRAG

A standalone Go service that ingests memory items from the Memory API into LightRAG's knowledge graph.

## Features

- **Automated Ingestion**: Pull memory items on a configurable schedule (hourly, cron, or manual)
- **Incremental Sync**: Only process new memories, avoiding duplicates
- **Transformation Strategies**: Standard and rich transformation options
- **Dead Letter Queue**: Track and retry failed items
- **State Management**: JSON or SQLite storage backends
- **Concurrent Processing**: Configurable concurrency for optimal performance
- **Management API**: HTTP API for status monitoring and manual triggers
- **Multiple Deployment Modes**: Binary, Docker, or systemd service

## Quick Start

### Prerequisites

- Go 1.21+
- Access to Memory API with API key
- LightRAG instance running (API mode)

### Installation

#### Option 1: Install Script

```bash
curl -fsSL https://raw.githubusercontent.com/kamir/memory-connector/main/scripts/install.sh | bash
```

#### Option 2: Build from Source

```bash
git clone https://github.com/kamir/memory-connector.git
cd memory-connector
make build
sudo make install
```

#### Option 3: Docker

```bash
docker pull memory-connector:latest
```

### Configuration

1. Copy the sample configuration:

```bash
cp configs/config.yaml configs/my-config.yaml
```

2. Edit `configs/my-config.yaml`:

```yaml
memory_api:
  url: "https://your-memory-api.com"
  api_key: "your-api-key"  # Or set via MEMCON_MEMORY_API_API_KEY env var

lightrag:
  url: "http://localhost:9621"

connectors:
  - id: "my-connector"
    enabled: true
    context_id: "your-context-id"
    schedule:
      type: "interval"
      interval_hours: 1
    ingestion:
      query_range: "day"
      query_limit: 100
      max_concurrency: 5
    transform:
      strategy: "standard"
      include_metadata: true
```

3. Set API key via environment variable (recommended):

```bash
export MEMCON_MEMORY_API_API_KEY="your-api-key"
```

### Usage

#### Manual Sync

Trigger a one-time sync for a connector:

```bash
memory-connector sync --connector my-connector
```

#### Service Mode

Run as a daemon with automatic scheduling:

```bash
memory-connector serve --config configs/my-config.yaml
```

#### List Connectors

View all configured connectors:

```bash
memory-connector list
```

#### Check Status

View sync status and history:

```bash
memory-connector status --connector my-connector
```

#### JSON Output

All commands support JSON output with the `--json` flag:

```bash
memory-connector status --connector my-connector --json
```

### Docker Usage

```bash
# Run manual sync
docker run --rm \
  -v $(pwd)/configs:/app/configs \
  -v $(pwd)/data:/app/data \
  -e MEMCON_MEMORY_API_API_KEY="your-api-key" \
  memory-connector:latest sync --connector my-connector

# Run as service
docker run -d \
  --name memory-connector \
  -v $(pwd)/configs:/app/configs \
  -v $(pwd)/data:/app/data \
  -e MEMCON_MEMORY_API_API_KEY="your-api-key" \
  -p 8080:8080 \
  memory-connector:latest serve
```

## Development

### Building

```bash
# Build for current platform
make build

# Build for all platforms
make build-all

# Build Docker image
make docker-build
```

### Testing

```bash
# Run tests
make test

# Run tests with coverage (target: 90%+)
make test-coverage
```

### Code Quality

```bash
# Format code
make fmt

# Run linter
make lint
```

### Development Mode

Auto-reload on code changes:

```bash
make dev
```

## Architecture

### Components

- **Memory API Client**: Fetches memory items with retry logic
- **LightRAG Client**: Submits transformed documents
- **Transformer**: Converts memories to LightRAG format (standard or rich)
- **State Manager**: Tracks processed items (JSON or SQLite)
- **Scheduler**: Manages cron-based and interval-based jobs
- **Orchestrator**: Coordinates the entire sync process

### Data Flow

1. Scheduler triggers sync job
2. Fetch memories from Memory API
3. Filter out already-processed items
4. Transform memories to LightRAG document format
5. Submit to LightRAG API with concurrent processing
6. Update state store and generate sync report
7. Failed items go to Dead Letter Queue for retry

## Configuration Reference

### Logging

Supports both JSON and console formats:

```yaml
logging:
  level: "info"  # debug, info, warn, error
  format: "console"  # json or console
  output_path: "stdout"  # stdout or file path
```

### Storage

JSON or SQLite backends:

```yaml
storage:
  type: "json"  # json or sqlite
  path: "./data"  # directory or database path
```

### Schedule Types

- **interval**: Run every N hours
- **cron**: Use cron expression
- **manual**: Trigger via API or CLI only

### Transformation Strategies

- **standard**: Simple transcript extraction
- **rich**: Enhanced with temporal, location, and media context

## Deployment

### Systemd Service

See `deployments/systemd/memory-connector.service`

```bash
sudo cp deployments/systemd/memory-connector.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable memory-connector
sudo systemctl start memory-connector
```

### Kubernetes

See `deployments/k8s/` for deployment manifests.

```bash
kubectl apply -f deployments/k8s/
```

## Troubleshooting

### View Logs

```bash
# If running as service
journalctl -u memory-connector -f

# If running in Docker
docker logs -f memory-connector
```

### Check State

```bash
# JSON backend
cat data/my-connector.json | jq .

# SQLite backend
sqlite3 data/state.db "SELECT * FROM sync_states;"
```

### Failed Items

Check the Dead Letter Queue:

```bash
memory-connector status --connector my-connector --json | jq '.failed_items'
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

MIT License

## Support

- Issues: https://github.com/kamir/memory-connector/issues
- Documentation: See `FEATURES/FR01-memory-ingestion/` in parent repository
