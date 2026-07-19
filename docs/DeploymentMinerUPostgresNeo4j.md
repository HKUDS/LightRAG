# Quality-first MinerU, PostgreSQL, and Neo4j deployment

This deployment keeps the MinerU and LightRAG Compose projects independent. A
small, shared external Docker network gives LightRAG private access to MinerU
without publishing the MinerU API to the LAN or Internet.

## 1. Create the shared network once

```bash
docker network create mineru-lightrag-internal
```

## 2. Attach the existing MinerU Compose service

In `/home/azureuser/minhnion/MinerU/docker/compose.cu128.yaml`, add the
following network to the `mineru-api` service. Keep the existing
`127.0.0.1:8001:8001` port mapping unchanged; it remains useful for host-local
health checks and other host applications.

```yaml
services:
  mineru-api:
    # Keep the existing service settings.
    networks:
      default:
      mineru_lightrag:
        aliases:
          - mineru-api

networks:
  mineru_lightrag:
    name: mineru-lightrag-internal
    external: true
```

Restart only the MinerU project after that edit:

```bash
cd /home/azureuser/minhnion/MinerU
docker compose -f docker/compose.cu128.yaml up -d
curl --fail http://127.0.0.1:8001/health
```

`mineru-api` is then resolvable only by containers joined to
`mineru-lightrag-internal`. This does not merge the two Compose projects and
does not stop other host-local clients from using `127.0.0.1:8001`.

## 3. LightRAG configuration

The repository `.env` is configured as a quality-first profile:

- MinerU: `hybrid-engine`, `effort=high`, automatic text/OCR mode, formula,
  table, and image analysis enabled.
- PDF, supported images, and legacy Office binary formats are parsed by MinerU.
- DOCX, Markdown, and textpack use LightRAG native parsing. TXT, HTML, PPTX,
  and XLSX use LightRAG legacy parsing because the current native parser does
  not support PPTX/XLSX/TXT/HTML.
- `PGKVStorage`, `PGDocStatusStorage`, and `PGVectorStorage` use PostgreSQL;
  `Neo4JStorage` stores the knowledge graph. The configured OpenAI embedding is
  3,072-dimensional, so pgvector uses `HNSW_HALFVEC` (required above 2,000
  dimensions) rather than standard HNSW.
- VLM analysis is enabled and inherits the existing OpenAI credentials/model;
  the configured base model must support image input.

Before starting, set secure, non-placeholder values in `.env` for
`POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DATABASE`, `NEO4J_USERNAME`,
and `NEO4J_PASSWORD`. The same Postgres credentials initialize the database and
are used by LightRAG. Do not publish database ports unless a separate,
authenticated admin workflow requires it.

## 4. Start LightRAG, PostgreSQL, Neo4j, API, and dashboard

```bash
cd /home/azureuser/minhnion/LightRAG
docker compose build lightrag
docker compose up -d
docker compose ps
curl --fail http://127.0.0.1:${PORT:-9621}/health
```

The LightRAG API and WebUI dashboard are served on the same port:

- Dashboard: `http://127.0.0.1:9621/`
- API documentation: `http://127.0.0.1:9621/docs`

Use the `HOST` and `PORT` values from `.env` instead of the defaults when they
are customized. Protect a non-loopback deployment with `LIGHTRAG_API_KEY` or
`AUTH_ACCOUNTS` before exposing it beyond a trusted network.

## Operations

```bash
# Logs and status
docker compose logs -f lightrag
docker compose ps

# Stop the LightRAG stack without deleting persistent data
docker compose down

# Explicitly inspect the private MinerU attachment
docker network inspect mineru-lightrag-internal

# Backup PostgreSQL (choose a host path outside the repository)
docker compose exec -T postgres pg_dump -U "$POSTGRES_USER" "$POSTGRES_DATABASE" > lightrag-postgres.sql
```

Named Docker volumes retain PostgreSQL and Neo4j data across `docker compose
down`. Deleting those volumes destroys the database; re-ingest source documents
rather than manually copying file-based LightRAG storage into the new database
backend.
