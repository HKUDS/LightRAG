# LightRAG Production Deployment Guide

> **📘 This document has been consolidated into a comprehensive production guide.**
> **Please use: [Complete Production Deployment Guide](PRODUCTION_DEPLOYMENT_COMPLETE.md)**

**Status**: 🚨 **DEPRECATED** - Redirects to consolidated guide
**Replacement**: [PRODUCTION_DEPLOYMENT_COMPLETE.md](PRODUCTION_DEPLOYMENT_COMPLETE.md)
**Action Required**: Update bookmarks and references to use the new consolidated guide.

This document previously provided simple xAI-focused production deployment instructions but has been merged into the consolidated guide under "Quick Start Deployments - xAI + Ollama Stack".

~~This guide provides step-by-step instructions for deploying a production-ready LightRAG environment using Docker Compose. This setup includes:~~

- The **LightRAG application**, built from the current repository code.
- A **PostgreSQL** database for robust data storage (including vector and graph data).
- A **Redis** instance for caching and performance enhancement.
- Configuration to use **xAI's Grok 3 Mini** as the primary Large Language Model.

---

## Prerequisites

Before you begin, ensure you have the following installed and configured:

- **Docker and Docker Compose**: [Install Docker Desktop](https://www.docker.com/products/docker-desktop/)
- **Git**: To clone the repository.
- **xAI API Key**: You must have a valid API key from xAI.
- **Ollama (Optional but Recommended)**: For running the embedding model locally. Ensure Ollama is running and you have pulled the recommended embedding model:
  ```bash
  ollama pull bge-m3:latest
  ```

---

## Step 1: Clone the Repository

First, clone the LightRAG repository to your local machine and navigate into the directory.

```bash
git clone https://github.com/Ajith-82/LightRAG.git
cd LightRAG
```

## Step 2: Prepare the Dockerfile

This repository includes a revised, more secure `Dockerfile.revised` that uses a safer base image and runs as a non-root user. We will replace the original `Dockerfile` with this improved version.

```bash
mv Dockerfile.revised Dockerfile
```

## Step 3: Configure the Environment (`.env` file)

This is the most critical step. You need to create a `.env` file to configure all the services. You can copy the `advanced.env.example` as a starting point.

```bash
cp advanced.env.example .env
```

Now, open the `.env` file and set the following variables. **Remove or comment out any other conflicting settings for `LLM_BINDING`, `EMBEDDING_BINDING`, and `LIGHTRAG_*_STORAGE`**.

```env
# -------------------------------------------------------------------
# Production Deployment Configuration
# -------------------------------------------------------------------

# --- Server Port ---
PORT=9621

# --- LLM Configuration: xAI Grok 3 Mini ---
# Note: MAX_ASYNC=2 and TIMEOUT=240 are recommended for stability with xAI
LLM_BINDING=xai
LLM_MODEL=grok-3-mini
LLM_BINDING_HOST=https://api.x.ai/v1
LLM_BINDING_API_KEY=your_xai_api_key_here  # <-- PASTE YOUR XAI KEY HERE
MAX_ASYNC=2
TIMEOUT=240

# --- Embedding Configuration: Ollama (Recommended) ---
# Ensure Ollama is running and you have pulled this model.
EMBEDDING_BINDING=ollama
EMBEDDING_MODEL=bge-m3:latest
EMBEDDING_DIM=1024
# Use host.docker.internal to allow the container to access the Ollama service running on your host machine.
EMBEDDING_BINDING_HOST=http://host.docker.internal:11434

# --- Storage Configuration: Use PostgreSQL and Redis ---
# Point LightRAG to use the PostgreSQL and Redis containers.
LIGHTRAG_KV_STORAGE=RedisKVStorage
LIGHTRAG_DOC_STATUS_STORAGE=PGDocStatusStorage
LIGHTRAG_GRAPH_STORAGE=PGGraphStorage
LIGHTRAG_VECTOR_STORAGE=PGVectorStorage

# --- PostgreSQL Connection Details ---
# These credentials MUST MATCH the values in docker-compose.prod.yml
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_USER=your_prod_user
POSTGRES_PASSWORD='your_strong_password' # <-- USE A STRONG, UNIQUE PASSWORD
POSTGRES_DATABASE=lightrag_prod_db

# --- Redis Connection Details ---
# This points to the Redis service defined in docker-compose.prod.yml
REDIS_URI=redis://redis:6379

# --- Optional: Secure Your API ---
# Uncomment and set a strong key to protect your API endpoints.
# LIGHTRAG_API_KEY=your-very-secure-and-random-api-key
```

## Step 4: Build and Launch the Services

With the configuration in place, you can now build the custom LightRAG image and launch the entire stack (LightRAG, PostgreSQL, Redis) using the production-ready compose file.

```bash
docker-compose -f docker-compose.prod.yml up --build -d
```

- `-f docker-compose.prod.yml`: Specifies that we are using the production configuration.
- `--build`: Forces Docker to build the `lightrag` image from your local `Dockerfile`.
- `-d`: Runs the containers in detached mode (in the background).

This command might take a few minutes the first time as it downloads the necessary base images and builds the LightRAG application image.

## Step 5: Verify the Deployment

First, check that all containers are running correctly.

```bash
docker-compose -f docker-compose.prod.yml ps
```

You should see three services running: `lightrag_prod`, `postgres_db_prod`, and `redis_cache_prod`. Their status should be `running` or `healthy`.

Next, check the logs of the LightRAG application to ensure it started without errors.

```bash
docker-compose -f docker-compose.prod.yml logs -f lightrag
```

Finally, test the `/health` endpoint to confirm the API is responsive.

```bash
curl http://localhost:9621/health
```

You should receive a JSON response with a `"status": "healthy"` message.

## Step 6: Using Your Production Instance

Your LightRAG instance is now ready. You can start inserting documents and making queries.

Here is an example of how to send a query using `curl`. If you set a `LIGHTRAG_API_KEY` in Step 3, you must include it in the header.

```bash
# Replace YOUR_API_KEY with the one you configured.
# This header is only needed if you set LIGHTRAG_API_KEY.
export LIGHTRAG_API_KEY="your-very-secure-and-random-api-key"

curl -X POST http://localhost:9621/query \
-H "Content-Type: application/json" \
-H "Authorization: Bearer $LIGHTRAG_API_KEY" \
-d '{
  "query": "What is Retrieval-Augmented Generation?",
  "mode": "bypass"
}'
```

## Managing the Deployment

- **To view logs**:
  ```bash
  docker-compose -f docker-compose.prod.yml logs -f <service_name>
  # e.g., docker-compose -f docker-compose.prod.yml logs -f lightrag
  ```

- **To stop the services**:
  ```bash
  docker-compose -f docker-compose.prod.yml down
  ```

- **To restart the services**:
  ```bash
  docker-compose -f docker-compose.prod.yml restart
  ```

## Troubleshooting

- **Connection Errors to xAI**: Double-check that your `LLM_BINDING_API_KEY` is correct and does not have extra characters or spaces.
- **Connection Errors to Ollama**: Ensure the Ollama service is running on your host machine. The `host.docker.internal` setting in the `.env` file is crucial for allowing the Docker container to reach it.
- **xAI Timeouts**: The recommended settings (`MAX_ASYNC=2`, `TIMEOUT=240`) should prevent most timeout issues. For more details, see the [xAI Troubleshooting Guide](./integration_guides/TROUBLESHOOTING_XAI.md).
- **Port Conflicts**: If you get an error that port `9621` or `5432` is already in use, you can change it in your `.env` file (for LightRAG) or the `docker-compose.prod.yml` file (for Postgres).
