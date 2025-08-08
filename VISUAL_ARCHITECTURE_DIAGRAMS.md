# LightRAG Visual Architecture Diagrams

## System Architecture Overview

```mermaid
graph TB
    subgraph "Client Layer"
        WEB[Web UI<br/>React/TypeScript]
        MCP[MCP Server<br/>Claude CLI]
        API_CLIENT[Direct API<br/>REST/HTTP]
    end

    subgraph "API Service Layer"
        FASTAPI[FastAPI Server<br/>Authentication & Routing]
        AUTH[Auth Middleware<br/>JWT & Rate Limiting]
        MIDDLEWARE[Security Middleware<br/>CORS & Headers]
    end

    subgraph "Core Processing Layer"
        LIGHTRAG[LightRAG Core<br/>Async Processing Engine]
        PIPELINE[Document Pipeline<br/>Chunking & Extraction]
        QUERY_ENGINE[Query Engine<br/>Multi-Mode Retrieval]
    end

    subgraph "Storage Layer"
        KV[KV Storage<br/>Chunks & Cache]
        VECTOR[Vector Storage<br/>Embeddings]
        GRAPH[Graph Storage<br/>Knowledge Graph]
        STATUS[Document Status<br/>Pipeline Tracking]
    end

    subgraph "External Services"
        LLM[LLM Providers<br/>OpenAI, xAI, Ollama]
        EMBED[Embedding Models<br/>BGE, Sentence-T]
        DOCLING[Docling Service<br/>Document Processing]
    end

    subgraph "Infrastructure"
        POSTGRES[(PostgreSQL<br/>Primary Database)]
        REDIS[(Redis<br/>Cache & Queue)]
        MONITORING[Monitoring Stack<br/>Prometheus/Grafana]
    end

    WEB --> FASTAPI
    MCP --> FASTAPI
    API_CLIENT --> FASTAPI

    FASTAPI --> AUTH
    AUTH --> MIDDLEWARE
    MIDDLEWARE --> LIGHTRAG

    LIGHTRAG --> PIPELINE
    LIGHTRAG --> QUERY_ENGINE
    PIPELINE --> LLM
    PIPELINE --> EMBED
    PIPELINE --> DOCLING

    LIGHTRAG --> KV
    LIGHTRAG --> VECTOR
    LIGHTRAG --> GRAPH
    LIGHTRAG --> STATUS

    KV --> POSTGRES
    VECTOR --> POSTGRES
    GRAPH --> POSTGRES
    STATUS --> POSTGRES

    FASTAPI --> REDIS
    LIGHTRAG --> MONITORING
```

## Document Processing Data Flow

```mermaid
sequenceDiagram
    participant User
    participant WebUI as Web UI
    participant API as FastAPI Server
    participant Core as LightRAG Core
    participant Pipeline as Processing Pipeline
    participant LLM as LLM Service
    participant Storage as Storage Backends

    User->>WebUI: Upload Document
    WebUI->>API: POST /documents/upload
    API->>API: Authentication & Rate Limiting
    API->>Core: rag.ainsert(content)

    Core->>Pipeline: Initialize Processing
    Pipeline->>Pipeline: Text Chunking
    Pipeline->>LLM: Extract Entities & Relations
    LLM-->>Pipeline: Structured Knowledge

    Pipeline->>Storage: Store Chunks (KV)
    Pipeline->>Storage: Store Embeddings (Vector)
    Pipeline->>Storage: Store Graph (Graph)
    Pipeline->>Storage: Update Status (DocStatus)

    Storage-->>Core: Confirm Storage
    Core-->>API: Processing Complete
    API-->>WebUI: Success Response
    WebUI-->>User: Document Processed
```

## Query Processing Flow

```mermaid
flowchart TD
    START[User Query] --> AUTH{Authentication}
    AUTH -->|Valid| MODE{Query Mode Selection}
    AUTH -->|Invalid| ERROR[Authentication Error]

    MODE --> LOCAL[Local Mode<br/>Entity-focused]
    MODE --> GLOBAL[Global Mode<br/>Graph-based]
    MODE --> HYBRID[Hybrid Mode<br/>Combined]
    MODE --> MIX[Mix Mode<br/>Vector + Graph]
    MODE --> NAIVE[Naive Mode<br/>Vector only]

    LOCAL --> ENTITY_SEARCH[Entity Vector Search]
    ENTITY_SEARCH --> CHUNK_RETRIEVAL[Related Chunk Retrieval]

    GLOBAL --> GRAPH_QUERY[Graph Traversal]
    GRAPH_QUERY --> RELATIONSHIP_EXPANSION[Relationship Expansion]

    HYBRID --> ENTITY_SEARCH
    HYBRID --> GRAPH_QUERY

    MIX --> VECTOR_SEARCH[Vector Similarity]
    MIX --> GRAPH_CONTEXT[Graph Context]

    NAIVE --> VECTOR_SEARCH

    CHUNK_RETRIEVAL --> CONTEXT_ASSEMBLY[Context Assembly]
    RELATIONSHIP_EXPANSION --> CONTEXT_ASSEMBLY
    VECTOR_SEARCH --> CONTEXT_ASSEMBLY
    GRAPH_CONTEXT --> CONTEXT_ASSEMBLY

    CONTEXT_ASSEMBLY --> TOKEN_MANAGEMENT[Token Budget Management]
    TOKEN_MANAGEMENT --> LLM_GENERATION[LLM Response Generation]
    LLM_GENERATION --> RESPONSE[Final Response]
```

## Storage Architecture Diagram

```mermaid
graph LR
    subgraph "Storage Types"
        KV[KV Storage<br/>Document Chunks<br/>LLM Cache]
        VECTOR[Vector Storage<br/>Entity Embeddings<br/>Similarity Search]
        GRAPH[Graph Storage<br/>Entity Relationships<br/>Knowledge Graph]
        STATUS[Document Status<br/>Pipeline Tracking<br/>Processing State]
    end

    subgraph "Backend Implementations"
        subgraph "KV Backends"
            JSON_KV[JsonKVStorage]
            PG_KV[PGKVStorage]
            REDIS_KV[RedisKVStorage]
            MONGO_KV[MongoKVStorage]
        end

        subgraph "Vector Backends"
            NANO[NanoVectorDB]
            PG_VEC[PGVectorStorage]
            MILVUS[MilvusStorage]
            QDRANT[QdrantStorage]
        end

        subgraph "Graph Backends"
            NETWORKX[NetworkXStorage]
            NEO4J[Neo4JStorage]
            PG_GRAPH[PGGraphStorage]
            MEMGRAPH[MemgraphStorage]
        end
    end

    KV --> JSON_KV
    KV --> PG_KV
    KV --> REDIS_KV
    KV --> MONGO_KV

    VECTOR --> NANO
    VECTOR --> PG_VEC
    VECTOR --> MILVUS
    VECTOR --> QDRANT

    GRAPH --> NETWORKX
    GRAPH --> NEO4J
    GRAPH --> PG_GRAPH
    GRAPH --> MEMGRAPH

    STATUS --> JSON_KV
    STATUS --> PG_KV
    STATUS --> REDIS_KV
    STATUS --> MONGO_KV
```

## Multi-Component Integration

```mermaid
graph TB
    subgraph "Development Environment"
        DEV_UI[Web UI Dev Server<br/>Vite + React]
        DEV_API[API Server<br/>Development Mode]
        DEV_MCP[MCP Server<br/>Local Instance]
    end

    subgraph "Production Environment"
        NGINX[Nginx Reverse Proxy<br/>Load Balancer]
        PROD_UI[Web UI<br/>Production Build]
        PROD_API[FastAPI Server<br/>Gunicorn Workers]
        PROD_MCP[MCP Server<br/>Production Mode]

        subgraph "Container Orchestration"
            DOCKER[Docker Compose<br/>Service Orchestration]
            HEALTH[Health Checks<br/>Service Dependencies]
        end

        subgraph "Data Services"
            POSTGRES[(PostgreSQL<br/>Primary Database)]
            REDIS[(Redis<br/>Cache & Sessions)]
            DOCLING_SVC[Docling Service<br/>Document Processing]
        end

        subgraph "Monitoring Stack"
            PROMETHEUS[Prometheus<br/>Metrics Collection]
            GRAFANA[Grafana<br/>Dashboards]
            JAEGER[Jaeger<br/>Distributed Tracing]
        end
    end

    DEV_UI --> DEV_API
    DEV_API --> DEV_MCP

    NGINX --> PROD_UI
    NGINX --> PROD_API
    PROD_API --> PROD_MCP

    DOCKER --> PROD_UI
    DOCKER --> PROD_API
    DOCKER --> PROD_MCP
    DOCKER --> POSTGRES
    DOCKER --> REDIS
    DOCKER --> DOCLING_SVC

    HEALTH --> POSTGRES
    HEALTH --> REDIS
    HEALTH --> DOCLING_SVC

    PROD_API --> POSTGRES
    PROD_API --> REDIS
    PROD_API --> DOCLING_SVC

    PROMETHEUS --> PROD_API
    GRAFANA --> PROMETHEUS
    JAEGER --> PROD_API
```

## Authentication and Security Flow

```mermaid
sequenceDiagram
    participant Client
    participant Nginx as Nginx Proxy
    participant API as FastAPI Server
    participant Auth as Auth Middleware
    participant RateLimit as Rate Limiter
    participant Core as LightRAG Core
    participant Audit as Audit Logger

    Client->>Nginx: HTTPS Request
    Nginx->>API: Forward Request
    API->>Auth: Validate JWT Token

    alt Valid Token
        Auth->>RateLimit: Check Rate Limits
        alt Within Limits
            RateLimit->>Core: Process Request
            Core-->>RateLimit: Response
            RateLimit-->>Auth: Response
            Auth-->>API: Response
            API->>Audit: Log Request/Response
            API-->>Nginx: Response
            Nginx-->>Client: Response
        else Rate Limited
            RateLimit-->>Auth: Rate Limit Error
            Auth-->>API: 429 Too Many Requests
            API->>Audit: Log Rate Limit Event
            API-->>Nginx: Error Response
            Nginx-->>Client: Error Response
        end
    else Invalid Token
        Auth-->>API: 401 Unauthorized
        API->>Audit: Log Auth Failure
        API-->>Nginx: Error Response
        Nginx-->>Client: Error Response
    end
```

## MCP Server Claude Integration

```mermaid
graph LR
    subgraph "Claude CLI Environment"
        CLAUDE[Claude CLI<br/>User Interface]
        MCP_CLIENT[MCP Client<br/>Protocol Handler]
    end

    subgraph "LightRAG MCP Server"
        MCP_SERVER[MCP Server<br/>Protocol Implementation]

        subgraph "Tools (11)"
            QUERY_TOOL[lightrag_query<br/>Document Queries]
            INSERT_TOOL[lightrag_insert_file<br/>Document Upload]
            LIST_TOOL[lightrag_list_documents<br/>Document Management]
            GRAPH_TOOL[lightrag_get_graph<br/>Graph Exploration]
            SEARCH_TOOL[lightrag_search_entities<br/>Entity Search]
            DELETE_TOOL[lightrag_delete_document<br/>Document Removal]
            STATUS_TOOL[lightrag_get_status<br/>Processing Status]
            HEALTH_TOOL[lightrag_health_check<br/>System Health]
            CONFIG_TOOL[lightrag_get_config<br/>Configuration]
            STATS_TOOL[lightrag_get_stats<br/>System Statistics]
            CLEAR_TOOL[lightrag_clear_storage<br/>Data Management]
        end

        subgraph "Resources (3)"
            SYS_CONFIG[System Configuration<br/>lightrag://system/config]
            HEALTH_RES[Health Status<br/>lightrag://system/health]
            API_DOCS[API Documentation<br/>lightrag://api/docs]
        end
    end

    subgraph "Backend Services"
        API_SERVICE[LightRAG API<br/>HTTP Interface]
        DIRECT_CLIENT[Direct Client<br/>Python Interface]
    end

    CLAUDE --> MCP_CLIENT
    MCP_CLIENT <--> MCP_SERVER
    MCP_SERVER --> QUERY_TOOL
    MCP_SERVER --> INSERT_TOOL
    MCP_SERVER --> LIST_TOOL
    MCP_SERVER --> GRAPH_TOOL
    MCP_SERVER --> SEARCH_TOOL
    MCP_SERVER --> DELETE_TOOL
    MCP_SERVER --> STATUS_TOOL
    MCP_SERVER --> HEALTH_TOOL
    MCP_SERVER --> CONFIG_TOOL
    MCP_SERVER --> STATS_TOOL
    MCP_SERVER --> CLEAR_TOOL

    MCP_SERVER --> SYS_CONFIG
    MCP_SERVER --> HEALTH_RES
    MCP_SERVER --> API_DOCS

    QUERY_TOOL --> API_SERVICE
    QUERY_TOOL --> DIRECT_CLIENT
    INSERT_TOOL --> API_SERVICE
    LIST_TOOL --> API_SERVICE
    GRAPH_TOOL --> API_SERVICE
```

## Production Deployment Architecture

```mermaid
graph TB
    subgraph "Load Balancer Layer"
        LB[Load Balancer<br/>Nginx/HAProxy]
    end

    subgraph "Application Layer"
        subgraph "Container Cluster"
            APP1[LightRAG Instance 1<br/>Gunicorn Workers]
            APP2[LightRAG Instance 2<br/>Gunicorn Workers]
            APP3[LightRAG Instance N<br/>Gunicorn Workers]
        end

        subgraph "Support Services"
            WEBUI[Web UI<br/>Static Assets]
            MCP_CLUSTER[MCP Server Cluster<br/>Claude Integration]
            DOCLING_CLUSTER[Docling Service Cluster<br/>Document Processing]
        end
    end

    subgraph "Data Layer"
        subgraph "Primary Storage"
            PG_PRIMARY[(PostgreSQL Primary<br/>Read/Write)]
            PG_REPLICA[(PostgreSQL Replica<br/>Read Only)]
        end

        subgraph "Caching Layer"
            REDIS_CLUSTER[(Redis Cluster<br/>Cache & Sessions)]
        end

        subgraph "Vector Storage"
            MILVUS_CLUSTER[(Milvus Cluster<br/>Vector Search)]
            QDRANT_CLUSTER[(Qdrant Cluster<br/>Vector Search)]
        end
    end

    subgraph "Infrastructure Layer"
        subgraph "Monitoring"
            PROMETHEUS[Prometheus<br/>Metrics Collection]
            GRAFANA[Grafana<br/>Visualization]
            JAEGER[Jaeger<br/>Tracing]
            LOKI[Loki<br/>Log Aggregation]
        end

        subgraph "Security"
            VAULT[HashiCorp Vault<br/>Secret Management]
            CERT_MANAGER[Cert Manager<br/>TLS Certificates]
        end

        subgraph "Backup & Recovery"
            BACKUP_SYSTEM[Backup System<br/>Database & Storage]
            RESTORE_SYSTEM[Restore System<br/>Point-in-Time Recovery]
        end
    end

    LB --> APP1
    LB --> APP2
    LB --> APP3
    LB --> WEBUI

    APP1 --> PG_PRIMARY
    APP1 --> PG_REPLICA
    APP1 --> REDIS_CLUSTER
    APP1 --> MILVUS_CLUSTER
    APP1 --> DOCLING_CLUSTER

    APP2 --> PG_PRIMARY
    APP2 --> PG_REPLICA
    APP2 --> REDIS_CLUSTER
    APP2 --> QDRANT_CLUSTER
    APP2 --> DOCLING_CLUSTER

    MCP_CLUSTER --> APP1
    MCP_CLUSTER --> APP2
    MCP_CLUSTER --> APP3

    PROMETHEUS --> APP1
    PROMETHEUS --> APP2
    PROMETHEUS --> APP3
    PROMETHEUS --> PG_PRIMARY
    PROMETHEUS --> REDIS_CLUSTER

    GRAFANA --> PROMETHEUS
    JAEGER --> APP1
    JAEGER --> APP2
    LOKI --> APP1
    LOKI --> APP2

    BACKUP_SYSTEM --> PG_PRIMARY
    BACKUP_SYSTEM --> REDIS_CLUSTER
    BACKUP_SYSTEM --> MILVUS_CLUSTER
```

## Network Communication Diagram

```ascii
┌─────────────────┐    HTTPS/443     ┌─────────────────┐
│   Client Apps   │◄──────────────────►│  Load Balancer  │
│  Web UI, CLI    │                   │   Nginx/HAProxy │
└─────────────────┘                   └─────────────────┘
                                               │
                                      HTTP/8080│
                           ┌───────────────────┼───────────────────┐
                           │                   ▼                   │
                 ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
                 │ LightRAG App 1  │ │ LightRAG App 2  │ │ LightRAG App N  │
                 │   Port 9621     │ │   Port 9622     │ │   Port 962N     │
                 └─────────────────┘ └─────────────────┘ └─────────────────┘
                           │                   │                   │
                           └───────────────────┼───────────────────┘
                                               │
                                      Internal │ Network
                           ┌───────────────────┼───────────────────┐
                           │                   ▼                   │
                 ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
                 │   PostgreSQL    │ │     Redis       │ │  Vector Store   │
                 │   Port 5432     │ │   Port 6379     │ │  Various Ports  │
                 └─────────────────┘ └─────────────────┘ └─────────────────┘
                           │                   │                   │
                           └───────────────────┼───────────────────┘
                                               │
                                   Monitoring  │ Stack
                           ┌───────────────────┼───────────────────┐
                           │                   ▼                   │
                 ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
                 │   Prometheus    │ │    Grafana      │ │     Jaeger      │
                 │   Port 9090     │ │   Port 3000     │ │   Port 16686    │
                 └─────────────────┘ └─────────────────┘ └─────────────────┘
```

## Data Persistence and Backup Flow

```mermaid
graph LR
    subgraph "Application Data"
        DOCS[Document Content]
        CHUNKS[Text Chunks]
        ENTITIES[Knowledge Graph]
        VECTORS[Embeddings]
        CACHE[LLM Cache]
        STATUS[Processing Status]
    end

    subgraph "Storage Systems"
        POSTGRES[(PostgreSQL<br/>Primary Storage)]
        REDIS[(Redis<br/>Cache Layer)]
        FILES[File System<br/>Working Directory]
    end

    subgraph "Backup Systems"
        PG_BACKUP[PostgreSQL Backups<br/>pg_dump + WAL]
        REDIS_BACKUP[Redis Backups<br/>RDB + AOF]
        FILE_BACKUP[File System Backups<br/>tar.gz Archives]
    end

    subgraph "Recovery Systems"
        PG_RESTORE[PostgreSQL Recovery<br/>Point-in-Time]
        REDIS_RESTORE[Redis Recovery<br/>RDB/AOF Restore]
        FILE_RESTORE[File System Recovery<br/>Archive Extraction]
    end

    DOCS --> POSTGRES
    CHUNKS --> POSTGRES
    ENTITIES --> POSTGRES
    VECTORS --> POSTGRES
    CACHE --> REDIS
    STATUS --> POSTGRES

    POSTGRES --> PG_BACKUP
    REDIS --> REDIS_BACKUP
    FILES --> FILE_BACKUP

    PG_BACKUP --> PG_RESTORE
    REDIS_BACKUP --> REDIS_RESTORE
    FILE_BACKUP --> FILE_RESTORE

    PG_RESTORE --> POSTGRES
    REDIS_RESTORE --> REDIS
    FILE_RESTORE --> FILES
```

This visual architecture documentation provides comprehensive diagrams covering all aspects of the LightRAG system architecture, from high-level component interactions to detailed deployment patterns and data flows.
