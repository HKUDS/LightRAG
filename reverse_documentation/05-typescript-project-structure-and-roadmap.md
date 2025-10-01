# TypeScript Project Structure and Migration Roadmap

## Table of Contents
1. [Recommended Project Structure](#recommended-project-structure)
2. [Module Organization](#module-organization)
3. [Configuration Files](#configuration-files)
4. [Build and Development Workflow](#build-and-development-workflow)
5. [Testing Strategy](#testing-strategy)
6. [Phase-by-Phase Migration Roadmap](#phase-by-phase-migration-roadmap)
7. [Deployment and CI/CD](#deployment-and-cicd)

## Recommended Project Structure

### Directory Layout

```
lightrag-ts/
├── src/
│   ├── core/
│   │   ├── LightRAG.ts              # Main LightRAG class
│   │   ├── Pipeline.ts              # Document processing pipeline
│   │   ├── QueryEngine.ts           # Query execution engine
│   │   └── index.ts
│   ├── storage/
│   │   ├── base/
│   │   │   ├── BaseKVStorage.ts     # KV storage interface
│   │   │   ├── BaseVectorStorage.ts # Vector storage interface
│   │   │   ├── BaseGraphStorage.ts  # Graph storage interface
│   │   │   ├── DocStatusStorage.ts  # Status storage interface
│   │   │   └── index.ts
│   │   ├── implementations/
│   │   │   ├── kv/
│   │   │   │   ├── JsonKVStorage.ts
│   │   │   │   ├── PostgresKVStorage.ts
│   │   │   │   ├── MongoKVStorage.ts
│   │   │   │   └── RedisKVStorage.ts
│   │   │   ├── vector/
│   │   │   │   ├── NanoVectorStorage.ts
│   │   │   │   ├── PostgresVectorStorage.ts
│   │   │   │   ├── QdrantVectorStorage.ts
│   │   │   │   └── MilvusVectorStorage.ts
│   │   │   ├── graph/
│   │   │   │   ├── GraphologyStorage.ts
│   │   │   │   ├── Neo4jStorage.ts
│   │   │   │   ├── PostgresGraphStorage.ts
│   │   │   │   └── MemgraphStorage.ts
│   │   │   └── status/
│   │   │       ├── JsonDocStatusStorage.ts
│   │   │       ├── PostgresDocStatusStorage.ts
│   │   │       └── MongoDocStatusStorage.ts
│   │   ├── StorageFactory.ts        # Factory for creating storage instances
│   │   └── index.ts
│   ├── llm/
│   │   ├── base/
│   │   │   ├── LLMProvider.ts       # LLM provider interface
│   │   │   ├── EmbeddingProvider.ts # Embedding interface
│   │   │   └── index.ts
│   │   ├── providers/
│   │   │   ├── OpenAIProvider.ts
│   │   │   ├── AnthropicProvider.ts
│   │   │   ├── OllamaProvider.ts
│   │   │   ├── BedrockProvider.ts
│   │   │   └── HuggingFaceProvider.ts
│   │   ├── LLMFactory.ts
│   │   └── index.ts
│   ├── operations/
│   │   ├── chunking.ts              # Text chunking logic
│   │   ├── extraction.ts            # Entity/relation extraction
│   │   ├── merging.ts               # Graph merging algorithms
│   │   ├── retrieval.ts             # Retrieval strategies
│   │   └── index.ts
│   ├── api/
│   │   ├── server.ts                # Fastify server setup
│   │   ├── routes/
│   │   │   ├── query.ts
│   │   │   ├── documents.ts
│   │   │   ├── graph.ts
│   │   │   ├── status.ts
│   │   │   └── index.ts
│   │   ├── middleware/
│   │   │   ├── auth.ts
│   │   │   ├── validation.ts
│   │   │   ├── errorHandler.ts
│   │   │   └── index.ts
│   │   ├── schemas/
│   │   │   ├── query.schema.ts
│   │   │   ├── document.schema.ts
│   │   │   └── index.ts
│   │   └── index.ts
│   ├── types/
│   │   ├── models.ts                # Core data models
│   │   ├── config.ts                # Configuration types
│   │   ├── storage.ts               # Storage-related types
│   │   ├── query.ts                 # Query-related types
│   │   └── index.ts
│   ├── utils/
│   │   ├── tokenizer.ts             # Tokenization utilities
│   │   ├── hashing.ts               # MD5, SHA utilities
│   │   ├── cache.ts                 # Caching utilities
│   │   ├── retry.ts                 # Retry logic
│   │   ├── concurrency.ts           # Rate limiting, semaphores
│   │   ├── logger.ts                # Logging setup
│   │   └── index.ts
│   ├── prompts/
│   │   ├── extraction.ts            # Entity extraction prompts
│   │   ├── keywords.ts              # Keyword extraction prompts
│   │   ├── summary.ts               # Summary prompts
│   │   └── index.ts
│   └── index.ts                     # Main entry point
├── tests/
│   ├── unit/
│   │   ├── core/
│   │   ├── storage/
│   │   ├── llm/
│   │   ├── operations/
│   │   └── utils/
│   ├── integration/
│   │   ├── storage/
│   │   ├── llm/
│   │   └── api/
│   ├── e2e/
│   │   └── scenarios/
│   ├── fixtures/
│   │   └── test-data.ts
│   └── setup.ts
├── examples/
│   ├── basic-usage.ts
│   ├── custom-storage.ts
│   ├── streaming-query.ts
│   └── batch-processing.ts
├── docs/
│   ├── api/
│   ├── guides/
│   └── migration/
├── scripts/
│   ├── build.ts
│   ├── migrate-data.ts
│   └── generate-types.ts
├── .github/
│   └── workflows/
│       ├── ci.yml
│       ├── release.yml
│       └── deploy.yml
├── package.json
├── tsconfig.json
├── tsconfig.build.json
├── vitest.config.ts
├── .env.example
├── .eslintrc.js
├── .prettierrc
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Module Organization

### Core Module (`src/core/`)

The core module contains the main LightRAG orchestration logic.

**LightRAG.ts** - Main class:
```typescript
import { LightRAGConfig } from '../types/config';
import { StorageFactory } from '../storage/StorageFactory';
import { LLMFactory } from '../llm/LLMFactory';
import { Pipeline } from './Pipeline';
import { QueryEngine } from './QueryEngine';

export class LightRAG {
  private config: LightRAGConfig;
  private kvStorage: BaseKVStorage;
  private vectorStorage: BaseVectorStorage;
  private graphStorage: BaseGraphStorage;
  private statusStorage: DocStatusStorage;
  private llmProvider: LLMProvider;
  private embeddingProvider: EmbeddingProvider;
  private pipeline: Pipeline;
  private queryEngine: QueryEngine;

  constructor(config: LightRAGConfig) {
    this.config = config;
    this.initializeStorages();
    this.initializeProviders();
    this.pipeline = new Pipeline(this);
    this.queryEngine = new QueryEngine(this);
  }

  async initialize(): Promise<void> {
    await Promise.all([
      this.kvStorage.initialize(),
      this.vectorStorage.initialize(),
      this.graphStorage.initialize(),
      this.statusStorage.initialize(),
    ]);
  }

  async insert(documents: string | string[]): Promise<string[]> {
    return this.pipeline.processDocuments(documents);
  }

  async query(query: string, params?: QueryParam): Promise<QueryResult> {
    return this.queryEngine.execute(query, params);
  }

  async close(): Promise<void> {
    await Promise.all([
      this.kvStorage.finalize(),
      this.vectorStorage.finalize(),
      this.graphStorage.finalize(),
      this.statusStorage.finalize(),
    ]);
  }
}
```

### Storage Module (`src/storage/`)

Storage implementations follow a factory pattern for easy swapping.

**StorageFactory.ts**:
```typescript
import { LightRAGConfig } from '../types/config';
import { BaseKVStorage, BaseVectorStorage, BaseGraphStorage } from './base';

export class StorageFactory {
  static createKVStorage(config: LightRAGConfig): BaseKVStorage {
    const { kv_storage } = config;
    
    switch (kv_storage) {
      case 'JsonKVStorage':
        return new JsonKVStorage(config);
      case 'PostgresKVStorage':
        return new PostgresKVStorage(config);
      case 'MongoKVStorage':
        return new MongoKVStorage(config);
      case 'RedisKVStorage':
        return new RedisKVStorage(config);
      default:
        throw new Error(`Unknown KV storage: ${kv_storage}`);
    }
  }

  static createVectorStorage(config: LightRAGConfig): BaseVectorStorage {
    // Similar pattern for vector storage
  }

  static createGraphStorage(config: LightRAGConfig): BaseGraphStorage {
    // Similar pattern for graph storage
  }
}
```

### LLM Module (`src/llm/`)

LLM providers implement a common interface for consistent usage.

**base/LLMProvider.ts**:
```typescript
export interface LLMProvider {
  name: string;
  
  chat(
    messages: ConversationMessage[],
    options?: LLMOptions
  ): Promise<string>;
  
  streamChat(
    messages: ConversationMessage[],
    options?: LLMOptions
  ): AsyncIterableIterator<string>;
  
  generateEmbeddings(
    texts: string[],
    options?: EmbeddingOptions
  ): Promise<number[][]>;
}

export interface LLMOptions {
  temperature?: number;
  max_tokens?: number;
  top_p?: number;
  model?: string;
  stop?: string[];
}
```

**providers/OpenAIProvider.ts**:
```typescript
import OpenAI from 'openai';
import { LLMProvider, LLMOptions } from '../base/LLMProvider';

export class OpenAIProvider implements LLMProvider {
  name = 'openai';
  private client: OpenAI;

  constructor(apiKey: string) {
    this.client = new OpenAI({ apiKey });
  }

  async chat(
    messages: ConversationMessage[],
    options: LLMOptions = {}
  ): Promise<string> {
    const completion = await this.client.chat.completions.create({
      model: options.model || 'gpt-4o-mini',
      messages,
      temperature: options.temperature ?? 0.7,
      max_tokens: options.max_tokens,
    });
    
    return completion.choices[0].message.content || '';
  }

  async *streamChat(
    messages: ConversationMessage[],
    options: LLMOptions = {}
  ): AsyncIterableIterator<string> {
    const stream = await this.client.chat.completions.create({
      model: options.model || 'gpt-4o-mini',
      messages,
      stream: true,
      temperature: options.temperature ?? 0.7,
    });

    for await (const chunk of stream) {
      const content = chunk.choices[0]?.delta?.content;
      if (content) yield content;
    }
  }

  async generateEmbeddings(texts: string[]): Promise<number[][]> {
    const response = await this.client.embeddings.create({
      model: 'text-embedding-3-small',
      input: texts,
    });
    
    return response.data.map(item => item.embedding);
  }
}
```

### API Module (`src/api/`)

RESTful API using Fastify.

**server.ts**:
```typescript
import Fastify from 'fastify';
import cors from '@fastify/cors';
import helmet from '@fastify/helmet';
import jwt from '@fastify/jwt';
import swagger from '@fastify/swagger';
import swaggerUi from '@fastify/swagger-ui';
import { queryRoutes } from './routes/query';
import { documentRoutes } from './routes/documents';
import { errorHandler } from './middleware/errorHandler';

export async function createServer(rag: LightRAG) {
  const app = Fastify({
    logger: {
      level: process.env.LOG_LEVEL || 'info',
    },
  });

  // Security
  await app.register(helmet);
  await app.register(cors, {
    origin: process.env.CORS_ORIGIN || '*',
  });

  // JWT
  await app.register(jwt, {
    secret: process.env.JWT_SECRET!,
  });

  // OpenAPI docs
  await app.register(swagger, {
    openapi: {
      info: {
        title: 'LightRAG API',
        version: '1.0.0',
      },
      servers: [
        { url: 'http://localhost:9621' },
      ],
    },
  });
  
  await app.register(swaggerUi, {
    routePrefix: '/docs',
  });

  // Routes
  await app.register(queryRoutes, { rag });
  await app.register(documentRoutes, { rag });

  // Error handling
  app.setErrorHandler(errorHandler);

  return app;
}
```

## Configuration Files

### package.json

```json
{
  "name": "lightrag-ts",
  "version": "1.0.0",
  "description": "TypeScript implementation of LightRAG",
  "main": "dist/index.js",
  "types": "dist/index.d.ts",
  "engines": {
    "node": ">=18.0.0"
  },
  "scripts": {
    "dev": "tsx watch src/index.ts",
    "build": "tsup",
    "test": "vitest",
    "test:unit": "vitest run --testPathPattern=tests/unit",
    "test:integration": "vitest run --testPathPattern=tests/integration",
    "test:e2e": "vitest run --testPathPattern=tests/e2e",
    "test:coverage": "vitest run --coverage",
    "lint": "eslint src --ext .ts",
    "lint:fix": "eslint src --ext .ts --fix",
    "format": "prettier --write \"src/**/*.ts\"",
    "typecheck": "tsc --noEmit",
    "start": "node dist/index.js",
    "start:api": "node dist/api/server.js"
  },
  "dependencies": {
    "@anthropic-ai/sdk": "^0.17.0",
    "@dqbd/tiktoken": "^1.0.7",
    "@fastify/cors": "^8.5.0",
    "@fastify/helmet": "^11.1.0",
    "@fastify/jwt": "^7.2.0",
    "@fastify/swagger": "^8.13.0",
    "@fastify/swagger-ui": "^2.1.0",
    "@qdrant/js-client-rest": "^1.8.0",
    "@zilliz/milvus2-sdk-node": "^2.3.0",
    "axios": "^1.6.0",
    "bcrypt": "^5.1.0",
    "bottleneck": "^2.19.0",
    "date-fns": "^3.0.0",
    "dotenv": "^16.0.0",
    "fastify": "^4.25.0",
    "graphology": "^0.25.0",
    "ioredis": "^5.3.0",
    "json-repair": "^0.2.0",
    "jsonwebtoken": "^9.0.2",
    "mongodb": "^6.3.0",
    "neo4j-driver": "^5.15.0",
    "ollama": "^0.5.0",
    "openai": "^4.28.0",
    "p-limit": "^5.0.0",
    "p-queue": "^8.0.0",
    "p-retry": "^6.0.0",
    "p-timeout": "^6.0.0",
    "pg": "^8.11.0",
    "pgvector": "^0.1.0",
    "pino": "^8.17.0",
    "pino-pretty": "^10.3.0",
    "zod": "^3.22.0"
  },
  "devDependencies": {
    "@types/bcrypt": "^5.0.0",
    "@types/jsonwebtoken": "^9.0.0",
    "@types/node": "^20.10.0",
    "@types/pg": "^8.10.0",
    "@typescript-eslint/eslint-plugin": "^6.15.0",
    "@typescript-eslint/parser": "^6.15.0",
    "@vitest/coverage-v8": "^1.1.0",
    "eslint": "^8.56.0",
    "eslint-config-prettier": "^9.1.0",
    "eslint-plugin-import": "^2.29.0",
    "prettier": "^3.1.0",
    "tsup": "^8.0.0",
    "tsx": "^4.7.0",
    "typescript": "^5.3.0",
    "vitest": "^1.1.0"
  }
}
```

### tsconfig.json

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "lib": ["ES2022"],
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "allowJs": false,
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true,
    "outDir": "./dist",
    "rootDir": "./src",
    "removeComments": true,
    "esModuleInterop": true,
    "forceConsistentCasingInFileNames": true,
    "strict": true,
    "noImplicitAny": true,
    "strictNullChecks": true,
    "strictFunctionTypes": true,
    "strictBindCallApply": true,
    "strictPropertyInitialization": true,
    "noImplicitThis": true,
    "alwaysStrict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noImplicitReturns": true,
    "noFallthroughCasesInSwitch": true,
    "skipLibCheck": true,
    "paths": {
      "@/*": ["./src/*"],
      "@core/*": ["./src/core/*"],
      "@storage/*": ["./src/storage/*"],
      "@llm/*": ["./src/llm/*"],
      "@utils/*": ["./src/utils/*"]
    }
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist", "tests"]
}
```

### vitest.config.ts

```typescript
import { defineConfig } from 'vitest/config';
import path from 'path';

export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      exclude: [
        'node_modules/',
        'dist/',
        'tests/',
        '**/*.spec.ts',
        '**/*.test.ts',
      ],
    },
    setupFiles: ['./tests/setup.ts'],
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      '@core': path.resolve(__dirname, './src/core'),
      '@storage': path.resolve(__dirname, './src/storage'),
      '@llm': path.resolve(__dirname, './src/llm'),
      '@utils': path.resolve(__dirname, './src/utils'),
    },
  },
});
```

### .eslintrc.js

```javascript
module.exports = {
  parser: '@typescript-eslint/parser',
  extends: [
    'eslint:recommended',
    'plugin:@typescript-eslint/recommended',
    'plugin:import/recommended',
    'plugin:import/typescript',
    'prettier',
  ],
  plugins: ['@typescript-eslint', 'import'],
  rules: {
    '@typescript-eslint/no-explicit-any': 'warn',
    '@typescript-eslint/no-unused-vars': ['error', { 
      argsIgnorePattern: '^_',
      varsIgnorePattern: '^_' 
    }],
    'import/order': ['error', {
      'groups': [
        'builtin',
        'external',
        'internal',
        'parent',
        'sibling',
        'index'
      ],
      'newlines-between': 'always',
      'alphabetize': { order: 'asc' }
    }],
  },
};
```

## Build and Development Workflow

### Development Mode

```bash
# Install dependencies
pnpm install

# Start development server with hot reload
pnpm dev

# Run tests in watch mode
pnpm test

# Type checking
pnpm typecheck
```

### Build for Production

```bash
# Build optimized bundle
pnpm build

# Run production build
pnpm start
```

### Build Configuration (tsup.config.ts)

```typescript
import { defineConfig } from 'tsup';

export default defineConfig({
  entry: ['src/index.ts', 'src/api/server.ts'],
  format: ['esm'],
  dts: true,
  splitting: false,
  sourcemap: true,
  clean: true,
  minify: true,
  treeshake: true,
  target: 'es2022',
  outDir: 'dist',
});
```

## Testing Strategy

### Unit Tests

Test individual functions and classes in isolation.

**Example: Storage unit test**
```typescript
// tests/unit/storage/JsonKVStorage.test.ts
import { describe, it, expect, beforeEach } from 'vitest';
import { JsonKVStorage } from '@storage/implementations/kv/JsonKVStorage';

describe('JsonKVStorage', () => {
  let storage: JsonKVStorage;

  beforeEach(() => {
    storage = new JsonKVStorage({
      working_dir: './test-storage',
      workspace: 'test',
    });
  });

  it('should store and retrieve values', async () => {
    await storage.upsert({ key1: { value: 'test' } });
    const result = await storage.get_by_id('key1');
    expect(result).toEqual({ value: 'test' });
  });

  it('should return null for non-existent keys', async () => {
    const result = await storage.get_by_id('nonexistent');
    expect(result).toBeNull();
  });
});
```

### Integration Tests

Test interactions between multiple components.

**Example: LLM + Storage integration**
```typescript
// tests/integration/extraction.test.ts
import { describe, it, expect } from 'vitest';
import { LightRAG } from '@core/LightRAG';

describe('Entity Extraction Integration', () => {
  it('should extract and store entities', async () => {
    const rag = new LightRAG({
      llm_provider: 'openai',
      openai_api_key: process.env.OPENAI_API_KEY,
      kv_storage: 'JsonKVStorage',
    });

    await rag.initialize();
    
    const docId = await rag.insert(
      'John Smith works at OpenAI in San Francisco.'
    );

    const entities = await rag.graphStorage.get_all_labels();
    expect(entities).toContain('John Smith');
    expect(entities).toContain('OpenAI');
    
    await rag.close();
  });
});
```

### End-to-End Tests

Test complete workflows from API to storage.

**Example: Complete query flow**
```typescript
// tests/e2e/query-flow.test.ts
import { describe, it, expect } from 'vitest';
import { createServer } from '@/api/server';
import { LightRAG } from '@core/LightRAG';

describe('Complete Query Flow', () => {
  it('should handle document upload and query', async () => {
    const rag = new LightRAG(config);
    await rag.initialize();
    
    const app = await createServer(rag);
    
    // Upload document
    const uploadResponse = await app.inject({
      method: 'POST',
      url: '/documents',
      payload: {
        content: 'Test document about AI.',
      },
    });
    
    expect(uploadResponse.statusCode).toBe(200);
    
    // Query
    const queryResponse = await app.inject({
      method: 'POST',
      url: '/query',
      payload: {
        query: 'What is AI?',
        mode: 'mix',
      },
    });
    
    expect(queryResponse.statusCode).toBe(200);
    const result = JSON.parse(queryResponse.body);
    expect(result.response).toBeDefined();
    
    await app.close();
    await rag.close();
  });
});
```

### Test Coverage Goals

- **Unit Tests**: >80% code coverage
- **Integration Tests**: Cover all component interactions
- **E2E Tests**: Cover critical user flows
- **Performance Tests**: Benchmark indexing and query latency

## Phase-by-Phase Migration Roadmap

### Phase 1: Foundation (Weeks 1-2)

**Goal**: Establish project structure and core abstractions

**Tasks**:
1. Set up TypeScript project with build configuration
2. Configure linting, formatting, and testing
3. Define all TypeScript types and interfaces
4. Implement base storage classes (abstract interfaces)
5. Set up logger and basic utilities
6. Create tokenizer wrapper for tiktoken

**Deliverables**:
- Complete project structure
- Type definitions matching Python data models
- Abstract storage interfaces
- Basic utility functions
- Test infrastructure

**Success Criteria**:
- Project builds without errors
- All types properly defined
- Tests can run (even if no implementations yet)

### Phase 2: Storage Layer (Weeks 3-5)

**Goal**: Implement storage backends

**Week 3: PostgreSQL Storage**
- PostgresKVStorage (with pooling)
- PostgresVectorStorage (with pgvector)
- PostgresGraphStorage
- PostgresDocStatusStorage
- Connection management and error handling

**Week 4: Alternative Storage**
- JsonKVStorage (file-based)
- GraphologyStorage (NetworkX equivalent)
- NanoVectorStorage (in-memory)
- JsonDocStatusStorage

**Week 5: Additional Storage & Testing**
- MongoKVStorage and MongoDocStatusStorage
- RedisKVStorage
- QdrantVectorStorage
- Comprehensive storage tests

**Deliverables**:
- 4+ complete storage implementations
- Storage factory pattern
- 100+ storage tests
- Performance benchmarks

**Success Criteria**:
- All storage tests pass
- Can switch storage backends via config
- Performance comparable to Python

### Phase 3: LLM Integration (Week 6)

**Goal**: Implement LLM and embedding providers

**Tasks**:
1. Implement OpenAIProvider (chat + embeddings)
2. Implement AnthropicProvider
3. Implement OllamaProvider
4. Add retry logic and rate limiting
5. Implement streaming support
6. Add embedding batch processing

**Deliverables**:
- 3+ LLM provider implementations
- LLM factory pattern
- Streaming support
- Comprehensive error handling

**Success Criteria**:
- All providers work with streaming
- Rate limiting prevents API errors
- Retry logic handles transient failures

### Phase 4: Core Engine - Part 1 (Weeks 7-8)

**Goal**: Implement document processing pipeline

**Week 7: Chunking & Extraction**
- Text chunking with overlap
- Entity extraction with LLM
- Relationship extraction
- Prompt template system

**Week 8: Graph Merging & Indexing**
- Entity deduplication and merging
- Relationship consolidation
- Vector embedding generation
- Pipeline status tracking

**Deliverables**:
- Complete chunking implementation
- Entity/relation extraction working
- Graph merging algorithms
- Vector indexing

**Success Criteria**:
- Documents successfully indexed
- Entities and relations extracted
- Graph properly merged
- Status tracking works

### Phase 5: Core Engine - Part 2 (Weeks 9-10)

**Goal**: Implement query engine

**Week 9: Retrieval Strategies**
- Keyword extraction
- Local mode (entity-centric)
- Global mode (relationship-centric)
- Hybrid mode (combined)

**Week 10: Query Modes & Context**
- Mix mode (KG + chunks)
- Naive mode (vector only)
- Bypass mode (direct LLM)
- Token budget management
- Reranking support

**Deliverables**:
- All 6 query modes working
- Context building with token budgets
- Reranking integration
- Query result formatting

**Success Criteria**:
- All query modes produce correct results
- Token budgets respected
- Comparable results to Python version

### Phase 6: API Layer (Week 11)

**Goal**: Build REST API

**Tasks**:
1. Set up Fastify server
2. Implement query endpoints
3. Implement document management endpoints
4. Add authentication (JWT)
5. Add OpenAPI documentation
6. Implement streaming responses

**Deliverables**:
- Complete REST API
- Authentication working
- OpenAPI docs auto-generated
- Streaming query support

**Success Criteria**:
- All endpoints functional
- API compatible with existing WebUI
- Authentication secure
- Documentation complete

### Phase 7: Testing & Optimization (Week 12)

**Goal**: Comprehensive testing and optimization

**Tasks**:
1. Complete unit test coverage
2. Integration test suite
3. E2E test scenarios
4. Performance benchmarking
5. Memory profiling
6. Load testing

**Deliverables**:
- >80% test coverage
- Performance benchmarks
- Load test results
- Optimization recommendations

**Success Criteria**:
- All tests pass
- Performance meets targets
- No memory leaks
- Handles expected load

### Phase 8: Production Hardening (Weeks 13-14)

**Goal**: Production readiness

**Tasks**:
1. Add monitoring and observability
2. Implement health checks
3. Add graceful shutdown
4. Create Docker images
5. Write deployment docs
6. Set up CI/CD pipelines

**Deliverables**:
- Docker images
- Kubernetes manifests
- Deployment documentation
- CI/CD pipelines
- Monitoring dashboards

**Success Criteria**:
- Deploys successfully
- Monitoring works
- Zero-downtime updates possible
- Documentation complete

## Deployment and CI/CD

### Dockerfile

```dockerfile
# Build stage
FROM node:20-alpine AS builder

WORKDIR /app

# Copy package files
COPY package.json pnpm-lock.yaml ./
RUN npm install -g pnpm && pnpm install --frozen-lockfile

# Copy source
COPY . .

# Build
RUN pnpm build

# Production stage
FROM node:20-alpine

WORKDIR /app

# Install production dependencies only
COPY package.json pnpm-lock.yaml ./
RUN npm install -g pnpm && pnpm install --prod --frozen-lockfile

# Copy built files
COPY --from=builder /app/dist ./dist

# Create data directories
RUN mkdir -p /app/data/rag_storage /app/data/inputs

# Environment
ENV NODE_ENV=production
ENV WORKING_DIR=/app/data/rag_storage
ENV INPUT_DIR=/app/data/inputs

EXPOSE 9621

CMD ["node", "dist/api/server.js"]
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  lightrag:
    build: .
    ports:
      - "9621:9621"
    environment:
      - NODE_ENV=production
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - PG_HOST=postgres
      - PG_DATABASE=lightrag
      - REDIS_HOST=redis
    depends_on:
      - postgres
      - redis
    volumes:
      - ./data:/app/data

  postgres:
    image: ankane/pgvector:latest
    environment:
      - POSTGRES_DB=lightrag
      - POSTGRES_USER=lightrag
      - POSTGRES_PASSWORD=password
    ports:
      - "5432:5432"
    volumes:
      - postgres-data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data

volumes:
  postgres-data:
  redis-data:
```

### GitHub Actions CI/CD

**.github/workflows/ci.yml**:
```yaml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: ankane/pgvector:latest
        env:
          POSTGRES_DB: lightrag_test
          POSTGRES_USER: test
          POSTGRES_PASSWORD: test
        ports:
          - 5432:5432

      redis:
        image: redis:7-alpine
        ports:
          - 6379:6379

    steps:
      - uses: actions/checkout@v4
      
      - uses: pnpm/action-setup@v2
        with:
          version: 8
      
      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: 'pnpm'
      
      - run: pnpm install --frozen-lockfile
      
      - run: pnpm lint
      
      - run: pnpm typecheck
      
      - run: pnpm test:coverage
      
      - uses: codecov/codecov-action@v3
        with:
          files: ./coverage/coverage-final.json

  build:
    runs-on: ubuntu-latest
    needs: test
    
    steps:
      - uses: actions/checkout@v4
      
      - uses: pnpm/action-setup@v2
        with:
          version: 8
      
      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: 'pnpm'
      
      - run: pnpm install --frozen-lockfile
      
      - run: pnpm build
      
      - uses: actions/upload-artifact@v3
        with:
          name: dist
          path: dist/
```

## Summary

This comprehensive project structure and roadmap provides:

1. **Complete directory organization** following TypeScript best practices
2. **Module structure** with clear separation of concerns
3. **Configuration files** optimized for TypeScript development
4. **Testing strategy** with unit, integration, and E2E tests
5. **14-week migration roadmap** with clear phases and deliverables
6. **CI/CD setup** for automated testing and deployment
7. **Docker configuration** for containerized deployment

The structure is designed to be maintainable, scalable, and production-ready while following TypeScript and Node.js ecosystem best practices. Each phase builds on previous work, allowing for incremental delivery and testing throughout the migration process.
