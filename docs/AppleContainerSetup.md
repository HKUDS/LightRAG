# Running the LightRAG storage stack on Apple `container`

[Apple `container`](https://github.com/apple/container) is Apple's native,
open-source container runtime for macOS 26 (Tahoe) on Apple Silicon. It runs each
container in its own lightweight Linux VM, with no background daemon.

`scripts/setup/apple-container.sh` brings up the full LightRAG storage stack —
**PostgreSQL**, **Neo4j**, and **Milvus** (standalone, with its `etcd` and
`minio` sidecars) — plus the **LightRAG API server**, on Apple `container`
instead of Docker Compose. LLM and embeddings are reached over normal outbound
HTTPS (e.g. the OpenAI API); there is **no GPU and no vLLM**, so the stack runs
on a CPU-only Apple Silicon Mac.

> This is a development convenience for Apple Silicon users who want the
> production-like Postgres/Neo4j/Milvus backends without Docker Desktop. For
> Docker/Podman deployments, see [DockerDeployment.md](DockerDeployment.md).

## Why a script instead of a Compose file

Apple `container` (1.0.0) has **no Docker Compose support**, and several Compose
features it lacks have to be reimplemented:

| Compose feature | Apple `container` 1.0.0 | How the script handles it |
| --- | --- | --- |
| `depends_on` / `condition: service_healthy` | not supported | explicit start ordering + `wait_for` health loops |
| `healthcheck` | not supported | the repo's `PORT_HEX` `/proc/net/tcp` probe, run via `container exec` |
| Service-name DNS (`postgres`, `neo4j`, …) | **does not resolve** between containers in 1.0.0 (see [apple/container#856](https://github.com/apple/container/issues/856)) | services are wired by the **IP** `container` assigns on the shared network (discovered with `container inspect`) — no DNS, no `sudo` |
| host bind mounts for DB data dirs | **broken** (`chown`/`chmod: Operation not permitted`, [apple/container#333](https://github.com/apple/container/issues/333), wontfix) | **named volumes** (`container volume create`) |
| `restart: unless-stopped` | not supported | `up` restarts stopped containers; a crashed container stays down until the next `up` |

## Prerequisites

- **A clone of the repository.** Every command below is run from the repo root:

  ```bash
  git clone https://github.com/HKUDS/LightRAG.git
  cd LightRAG
  ```

- **macOS 26 (Tahoe) or newer** on **Apple Silicon**. Container-to-container
  networking and the `container network` command do not exist before macOS 26,
  so the stack cannot work on macOS 15 — the script refuses to run there.
- The **`container` CLI**, installed from the
  [signed release](https://github.com/apple/container/releases) and started once:

  ```bash
  container system start          # accept the default kernel install when prompted
  ```

- **Bash 4+** (macOS ships Bash 3.2). Install a modern bash and run the script
  with it:

  ```bash
  brew install bash
  bash scripts/setup/apple-container.sh up
  ```

- **A `.env` file** with your LLM/embedding provider and API key. If you do not
  have one yet, copy the template and edit it:

  ```bash
  cp env.example .env
  # set LLM_BINDING / LLM_MODEL / LLM_BINDING_API_KEY and the EMBEDDING_* keys
  ```

  Only the LLM and embedding settings matter here (e.g. an OpenAI key on both
  `LLM_BINDING_API_KEY` and `EMBEDDING_BINDING_API_KEY`). **Leave the storage
  backend variables as they are** — the script overrides them to point at the
  Postgres / Neo4j / Milvus containers. Do **not** start from
  `env.docker-compose-full`, which is pre-wired for the GPU Docker stack.
  (`make env-base` can also generate a `.env` interactively.)

No `sudo` is required.

## Quick start

```bash
# Start the whole stack (databases + LightRAG server)
bash scripts/setup/apple-container.sh up

# Databases only (run the LightRAG server on the host yourself)
bash scripts/setup/apple-container.sh up --no-lightrag

# See what is running
bash scripts/setup/apple-container.sh status

# Tail a service's logs
bash scripts/setup/apple-container.sh logs lightrag --follow

# Stop and remove containers (keeps data)
bash scripts/setup/apple-container.sh down

# Stop, remove containers AND delete all stored data
bash scripts/setup/apple-container.sh down --purge
```

Equivalent `make` targets are provided (they resolve a bash 4+ interpreter for
you): `make apple-up`, `make apple-down`, `make apple-status`,
`make apple-logs SVC=lightrag`, `make apple-restart SVC=<service>`,
`make apple-pull`. Pass script flags via `SETUP_OPTS`, e.g.
`make apple-up SETUP_OPTS=--no-lightrag` or `make apple-down SETUP_OPTS=--purge`.

When the stack is up:

- LightRAG WebUI: <http://127.0.0.1:9621/webui>
- LightRAG health: <http://127.0.0.1:9621/health>
- Neo4j Browser / MinIO console: on the container's IP, printed by `up`
  (the host can reach the container subnet directly). To recover an IP later,
  re-run `up` (it is idempotent) or `container inspect <service>`.

## What the stack looks like

```
host (macOS 26, Apple Silicon)
  └─ 127.0.0.1:9621 ──▶ [lightrag] ──┐   (--network lightrag)
                                     ├─▶ postgres   :5432   (volume rag_pg)
                                     ├─▶ neo4j       :7687   (volume rag_neo4j)
                                     └─▶ milvus      :19530  (volume rag_milvus)
                                            ├─▶ milvus-etcd  :2379  (volume rag_etcd)
                                            └─▶ milvus-minio :9000  (volume rag_minio)
  [lightrag] ──── outbound HTTPS ────▶ api.openai.com
```

Only the LightRAG server publishes a host port (`127.0.0.1:9621`). The databases
are intentionally **not** published, so the stack never clashes with a Postgres
already listening on the host's `5432`. Each service is reached by its container
IP on the `lightrag` network.

> **Trust boundary.** This is a local, single-user development stack. The
> database containers are not published to a host port, but they are reachable on
> their vmnet IP with the default dev credentials listed under
> [Configuration](#configuration). Do not run it on a shared or multi-user
> machine, and override `POSTGRES_PASSWORD` / `NEO4J_PASSWORD` /
> `MINIO_SECRET_ACCESS_KEY` if others can route to the container subnet.

### Images (all verified to publish a `linux/arm64` manifest)

| Service | Image |
| --- | --- |
| postgres | `pgvector/pgvector:pg18` |
| neo4j | `neo4j:5-community` |
| milvus | `milvusdb/milvus:v2.6.11` (standalone, CPU) |
| milvus-etcd | `quay.io/coreos/etcd:v3.5.25` |
| milvus-minio | `minio/minio:RELEASE.2025-09-07T16-13-09Z` |
| lightrag | `ghcr.io/hkuds/lightrag:latest` |

Two deviations from `docker-compose-full.yml` / `scripts/setup/templates/`, both
forced by Apple Silicon:

- **Postgres** uses `pgvector/pgvector:pg18` (multi-arch) instead of the setup
  template's `gzdaniel/postgres-for-rag:pg18-age-pgvector`, which is **amd64-only**
  (no arm64 manifest). The Apache AGE graph extension it adds is not needed here:
  graph storage is Neo4j and vector storage is Milvus, so Postgres only serves
  `PGKVStorage` + `PGDocStatusStorage`.
- **Milvus** uses the CPU tag `milvusdb/milvus:v2.6.11`, **not** the
  `…-gpu` tag from `docker-compose-full.yml` (which is amd64 + CUDA and cannot run
  on Apple Silicon).

## Configuration

The script reads your existing host `.env` and **never modifies it**. For the
containerized LightRAG server it writes a generated `.apple-container.env`
(git-ignored) that copies your `.env` and overrides only the storage selection
and connection endpoints:

```
LIGHTRAG_KV_STORAGE=PGKVStorage
LIGHTRAG_DOC_STATUS_STORAGE=PGDocStatusStorage
LIGHTRAG_GRAPH_STORAGE=Neo4JStorage
LIGHTRAG_VECTOR_STORAGE=MilvusVectorDBStorage
POSTGRES_HOST=<postgres container IP>
NEO4J_URI=neo4j://<neo4j IP>:7687
MILVUS_URI=http://<milvus IP>:19530
```

Your LLM/embedding settings (`LLM_BINDING`, `EMBEDDING_BINDING`, API keys, model
names, `EMBEDDING_DIM`, …) are inherited unchanged. Set a real LLM API key in
`.env` before ingesting documents or querying.

Because `.apple-container.env` copies your `.env`, it contains your real API
keys. It is git-ignored, created with mode `600`, left in place by `down`, and
removed by `down --purge`.

Common overrides (environment variables, all optional):

| Variable | Default | Purpose |
| --- | --- | --- |
| `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` | `rag` / `rag` / `rag` | Postgres credentials |
| `NEO4J_USERNAME` / `NEO4J_PASSWORD` | `neo4j` / `lightragdev` | Neo4j auth (password ≥ 8 chars) |
| `MINIO_ACCESS_KEY_ID` / `MINIO_SECRET_ACCESS_KEY` | `minioadmin` / `minioadmin` | MinIO / Milvus object store |
| `LIGHTRAG_AC_MEM_HEAVY` | `6G` | memory for Milvus and Neo4j VMs |
| `LIGHTRAG_AC_MEM_LIGHT` | `2G` | memory for Postgres and LightRAG VMs |

## Data persistence

All data lives in named volumes (`rag_pg`, `rag_neo4j`, `rag_milvus`, `rag_etcd`,
`rag_minio`, `rag_lightrag`). `down` removes the containers but keeps the volumes,
so a later `up` restores your data. Only `down --purge` deletes the volumes.

## Troubleshooting

- **`bind(...): Address already in use`** — something on the host already owns
  `9621`. Stop it (e.g. a host `lightrag-server`) and retry. Database ports are
  not published, so a host Postgres on `5432` is fine.
- **A service IP changed after an individual `restart`** — the stack is wired by
  IP at `up` time. If you `restart` a database on its own and its IP changes, run
  `down` then `up` to re-wire dependents.
- **Milvus or Neo4j is killed / slow** — they are memory-hungry; raise
  `LIGHTRAG_AC_MEM_HEAVY` (e.g. `8G`).
- **Ingestion or a query fails while `/health` is green** — the stack is fine;
  the LLM/embedding call failed. Check that a real key is set on **both**
  `LLM_BINDING_API_KEY` and `EMBEDDING_BINDING_API_KEY`, and that the provider has
  quota/billing (OpenAI returns `429 insufficient_quota` when out of credit).
  `logs lightrag` shows the exact HTTP error.
- **`Rerank is enabled but no rerank model is configured`** — this CPU stack
  ships no local reranker. Retrieval still works; to silence it, either set a
  hosted reranker (`RERANK_BINDING` + `RERANK_MODEL` + key) or pass
  `enable_rerank=false` in the query parameters.
- **Inspecting a service** — `bash scripts/setup/apple-container.sh logs <service>`
  or `container exec <service> sh`.

## Cleanup

```bash
bash scripts/setup/apple-container.sh down --purge   # remove containers + data
container network delete lightrag                    # remove the network (optional)
```
