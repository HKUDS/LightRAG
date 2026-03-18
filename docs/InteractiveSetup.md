# Interactive Setup Guide

Use the interactive setup wizard when you want LightRAG to guide you through the configuration instead of editing `.env` by hand.

The wizard is exposed through `make` targets:

- `make env-base`
- `make env-storage`
- `make env-server`
- `make env-validate`
- `make env-security-check`
- `make env-backup`
- `make env-base-rewrite`
- `make env-storage-rewrite`

You do not need to call the underlying shell script directly.

## What This Wizard Is For

The setup wizard helps you configure LightRAG in three parts:

- `env-base` sets up the LLM, embedding model, and optional reranker.
- `env-storage` adds or changes storage backends such as PostgreSQL, Neo4j, Redis, Milvus, Qdrant, MongoDB, or Memgraph.
- `env-server` sets server host and port, WebUI labels, authentication, API keys, and SSL.

You can rerun each step later. The wizard loads your existing `.env` and shows current values as defaults, so you only need to change what is different.

## Before You Start

- Run commands from the repository root.
- The `make env-*` targets automatically choose a compatible Bash 4+ interpreter.
- Use the documented `make env-*` targets rather than invoking the setup script yourself.
- `make env-base` is the normal starting point because it creates the initial `.env`.
- `make env-storage` and `make env-server` require an existing `.env`.
- If you choose any wizard-managed Docker service, the wizard also prepares LightRAG for the Docker startup path.

## Choose Your Setup Path

Use this quick guide to decide what to run:

- I want the fastest first run with remote model providers: `make env-base`
- I want embedding or reranking to run locally in Docker: `make env-base`
- I already configured models and now want databases: `make env-storage`
- I already configured models and now want auth, API keys, or SSL: `make env-server`
- I want to check whether my current setup is valid: `make env-validate`
- I want to audit my current setup before exposing it: `make env-security-check`
- I want a standalone backup without changing configuration: `make env-backup`
- I need to repair the generated compose services from the bundled templates: `make env-base-rewrite` or `make env-storage-rewrite`

## Scenario 1: First-Time Local Setup

Use this when you want LightRAG running with the least amount of setup and you already have remote model endpoints or API keys.

**Command**

```bash
make env-base
```

**What the wizard asks**

- LLM provider, model, endpoint, and API key
- Whether the embedding model should run locally via Docker
- If embedding stays remote: embedding provider, model, dimension, endpoint, and API key
- Whether reranking should be enabled
- If reranking is enabled: whether the rerank service should run locally via Docker
- If reranking stays remote: rerank provider, model, endpoint, and API key

**What gets written**

- `.env`
- `docker-compose.final.yml` only if you enabled wizard-managed Docker services

**What to do next**

- If you did not enable wizard-managed Docker services:

```bash
lightrag-server
```

- If you enabled wizard-managed Docker services:

```bash
docker compose -f docker-compose.final.yml up -d
```

## Scenario 2: Local Setup With Docker-Hosted Embedding or Rerank

Use this when you want LightRAG to run local inference services for embedding and/or reranking through Docker.

**Command**

```bash
make env-base
```

**Recommended answers**

- Answer `yes` to `Run embedding model locally via Docker (vLLM)?` if you want local embeddings
- Answer `yes` to `Enable reranking?` and then `yes` to `Run rerank service locally via Docker?` if you want local reranking

**What the wizard asks after you enable local services**

- Embedding model name for local vLLM
- Rerank model name for local vLLM
- Remote LLM details if your main LLM is still external

**What gets written**

- `.env`
- `docker-compose.final.yml` with the selected local services

**What to do next**

```bash
docker compose -f docker-compose.final.yml up -d
```

This starts the generated Docker-based LightRAG stack together with the selected local services.

## Scenario 3: Add Storage After The Base Setup

Use this when you already have `.env` from `make env-base` and now want to switch from default local-file storage to database-backed storage.

**Command**

```bash
make env-storage
```

**Prerequisite**

- `.env` must already exist

**What the wizard asks**

- KV storage backend
- Vector storage backend
- Graph storage backend
- Doc-status storage backend
- For each required database, whether it should run locally via Docker
- For each required database, the needed connection details such as host, URI, port, user, password, database name, or device type

**Important rule**

- If you choose `MongoVectorDBStorage` for vector storage, the wizard does not offer the bundled local Docker MongoDB service. You must provide a MongoDB deployment that supports Atlas Search / Vector Search.

**What gets written**

- `.env`
- `docker-compose.final.yml` if you selected wizard-managed storage services

**What to do next**

- If you selected Docker-managed storage services:

```bash
docker compose -f docker-compose.final.yml up -d
```

- If you pointed LightRAG at external databases, make sure those services are reachable before starting LightRAG.

## Scenario 4: Harden A Deployment With Auth And SSL

Use this when you already have `.env` and need to prepare the server for shared or external use.

**Commands**

```bash
make env-server
make env-security-check
```

**Prerequisite**

- `.env` must already exist

**What `env-server` asks**

- Server host and port
- WebUI title and description
- Summary language
- Whether to configure authentication and API key settings
- Auth accounts, JWT secret, token lifetime, API key, and whitelist paths
- Whether to enable SSL/TLS
- SSL certificate file path and SSL key file path

**What gets written**

- `.env`
- `docker-compose.final.yml` may be updated if your current setup already uses wizard-managed Docker services

**What to do next**

- Run `make env-security-check`
- If the stack uses Docker, recreate the LightRAG service with your compose file
- If the stack runs on the host, restart `lightrag-server`

For broader deployment guidance, see [DockerDeployment.md](/Users/ydh/mycode/ai/paper-RAG/docs/DockerDeployment.md).

## Validate, Audit, And Backup

These commands do not walk you through a full setup flow, but they are part of normal operations.

### Validate The Current Configuration

```bash
make env-validate
```

Use this when you want to confirm that the current `.env` is internally consistent. It reports problems such as missing required values, malformed auth settings, invalid URIs, invalid ports, or missing SSL files.

### Audit Security Before Exposure

```bash
make env-security-check
```

Use this before exposing LightRAG beyond localhost. It reports risky setups such as missing authentication, weak or missing JWT secrets, unsafe whitelist settings, or unresolved sensitive placeholders.

### Create A Standalone Backup

```bash
make env-backup
```

Use this when you want a manual backup without running any setup flow.

## Outputs And What They Mean

### `.env`

The wizard writes `.env` in the repository root. This file becomes the current runtime configuration produced by the latest wizard run.

In practice, this means:

- rerunning the wizard updates `.env`
- existing values are reused as defaults on later runs
- you should treat `.env` as the active configuration for the workflow you most recently configured
- before `env-base`, `env-storage`, or `env-server` writes `.env`, the wizard automatically creates a timestamped backup of the existing file when one is present

### `docker-compose.final.yml`

The wizard creates or updates `docker-compose.final.yml` only when you choose wizard-managed Docker services or when an existing wizard-generated compose setup needs to stay aligned with new server settings.

When one of the setup flows is about to replace or remove an existing generated compose file, it automatically creates a timestamped backup first.

Use this file when starting the generated Docker stack:

```bash
docker compose -f docker-compose.final.yml up -d
```

The base `docker-compose.yml` remains the general project compose file. The generated `docker-compose.final.yml` is the wizard-managed output.

## Troubleshooting And Advanced Notes

- If `make env-storage` or `make env-server` says `.env` is missing, run `make env-base` first.
- You do not need to run `make env-backup` before rerunning `env-base`, `env-storage`, or `env-server`; those flows already back up the existing `.env`, and they also back up the generated compose file before changing it.
- If you need to fully rebuild wizard-managed compose services from the current bundled templates, use `make env-base-rewrite` or `make env-storage-rewrite`.
- If you switch between host-oriented and Docker-oriented workflows, rerun the relevant setup step instead of trying to manually merge old settings.
- If the generated stack includes local Milvus, make sure `MINIO_ACCESS_KEY_ID` and `MINIO_SECRET_ACCESS_KEY` are available before running `docker compose -f docker-compose.final.yml up -d`.
- For Docker deployment details beyond the interactive wizard, see [DockerDeployment.md](/Users/ydh/mycode/ai/paper-RAG/docs/DockerDeployment.md).

## Typical Command Sequences

### Remote models, local server

```bash
make env-base
lightrag-server
```

### Remote LLM, local embedding and rerank in Docker

```bash
make env-base
docker compose -f docker-compose.final.yml up -d
```

### Add storage after the base setup

```bash
make env-base
make env-storage
docker compose -f docker-compose.final.yml up -d
```

### Add security and SSL before exposure

```bash
make env-base
make env-storage
make env-server
make env-security-check
docker compose -f docker-compose.final.yml up -d
```
