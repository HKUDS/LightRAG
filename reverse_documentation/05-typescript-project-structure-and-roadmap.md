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

RESTful API using Hono framework (ultrafast, runtime-agnostic).

**server.ts**:
```typescript
import { Hono } from 'hono';
import { cors } from 'hono/cors';
import { logger } from 'hono/logger';
import { jwt } from 'hono/jwt';
import { z } from 'zod';
import { zValidator } from '@hono/zod-validator';
import { createRoute, OpenAPIHono } from '@hono/zod-openapi';
import { LightRAG } from '../core/LightRAG';
import { queryRoutes } from './routes/query';
import { documentRoutes } from './routes/documents';
import { graphRoutes } from './routes/graph';
import { errorHandler } from './middleware/errorHandler';

export function createServer(rag: LightRAG) {
  const app = new OpenAPIHono();

  // Middleware
  app.use('*', logger());
  app.use('*', cors({
    origin: Bun.env.CORS_ORIGIN || '*',
    credentials: true,
  }));

  // JWT authentication for protected routes
  app.use('/api/*', jwt({
    secret: Bun.env.JWT_SECRET!,
    cookie: 'auth_token',
  }));

  // Health check
  app.get('/health', (c) => {
    return c.json({ 
      status: 'ok',
      timestamp: new Date().toISOString(),
      version: '1.0.0',
    });
  });

  // Register routes
  app.route('/api/query', queryRoutes(rag));
  app.route('/api/documents', documentRoutes(rag));
  app.route('/api/graph', graphRoutes(rag));

  // OpenAPI documentation
  app.doc('/openapi.json', {
    openapi: '3.0.0',
    info: {
      title: 'LightRAG API',
      version: '1.0.0',
      description: 'Graph-based RAG system with TypeScript and Bun',
    },
    servers: [
      { url: 'http://localhost:9621', description: 'Development server' },
    ],
  });

  // Swagger UI
  app.get('/docs', (c) => {
    const html = `
      <!DOCTYPE html>
      <html>
        <head>
          <meta charset="utf-8" />
          <title>LightRAG API Documentation</title>
          <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css" />
        </head>
        <body>
          <div id="swagger-ui"></div>
          <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
          <script>
            SwaggerUIBundle({
              url: '/openapi.json',
              dom_id: '#swagger-ui',
            });
          </script>
        </body>
      </html>
    `;
    return c.html(html);
  });

  // Error handling
  app.onError(errorHandler);

  return app;
}

// Start server with Bun
const rag = new LightRAG({ /* config */ });
const app = createServer(rag);

Bun.serve({
  port: parseInt(Bun.env.PORT || '9621'),
  fetch: app.fetch,
  development: Bun.env.NODE_ENV !== 'production',
});

console.log('🚀 LightRAG server started on http://localhost:9621');
console.log('📚 API docs available at http://localhost:9621/docs');
```

**routes/query.ts** (Hono with type-safe validation):
```typescript
import { OpenAPIHono, createRoute } from '@hono/zod-openapi';
import { z } from 'zod';
import { LightRAG } from '../../core/LightRAG';

const QuerySchema = z.object({
  query: z.string().min(1).openapi({
    example: 'What is LightRAG?',
    description: 'The query text',
  }),
  mode: z.enum(['local', 'global', 'hybrid', 'mix', 'naive', 'bypass'])
    .default('mix')
    .openapi({
      description: 'Query mode: local (entity-focused), global (relationship-focused), hybrid, mix, naive, bypass',
    }),
  top_k: z.number().int().positive().default(40).openapi({
    description: 'Number of top results to return',
  }),
  stream: z.boolean().default(false).openapi({
    description: 'Enable streaming response',
  }),
});

const QueryResponseSchema = z.object({
  response: z.string(),
  sources: z.array(z.string()).optional(),
  metadata: z.record(z.any()).optional(),
});

export function queryRoutes(rag: LightRAG) {
  const app = new OpenAPIHono();

  const queryRoute = createRoute({
    method: 'post',
    path: '/',
    tags: ['Query'],
    request: {
      body: {
        content: {
          'application/json': {
            schema: QuerySchema,
          },
        },
      },
    },
    responses: {
      200: {
        description: 'Query response',
        content: {
          'application/json': {
            schema: QueryResponseSchema,
          },
        },
      },
      400: {
        description: 'Bad request',
      },
      500: {
        description: 'Internal server error',
      },
    },
  });

  app.openapi(queryRoute, async (c) => {
    const { query, mode, top_k, stream } = c.req.valid('json');

    if (stream) {
      // Streaming response
      return c.stream(async (stream) => {
        for await (const chunk of rag.queryStream(query, { mode, top_k })) {
          await stream.write(chunk);
        }
      });
    }

    // Regular response
    const result = await rag.query(query, { mode, top_k });
    
    return c.json({
      response: result.response,
      sources: result.sources,
      metadata: result.metadata,
    });
  });

  return app;
}
```

**middleware/errorHandler.ts**:
```typescript
import { Context } from 'hono';
import { HTTPException } from 'hono/http-exception';
import pino from 'pino';

const logger = pino();

export function errorHandler(err: Error, c: Context) {
  if (err instanceof HTTPException) {
    return c.json(
      {
        error: err.message,
        status: err.status,
      },
      err.status
    );
  }

  // Log unexpected errors
  logger.error(
    {
      err,
      path: c.req.path,
      method: c.req.method,
    },
    'Unhandled error'
  );

  return c.json(
    {
      error: 'Internal server error',
      message: Bun.env.NODE_ENV === 'development' ? err.message : undefined,
    },
    500
  );
}
```

## Configuration Files

### package.json (Bun-Optimized)

```json
{
  "name": "lightrag-ts",
  "version": "1.0.0",
  "description": "TypeScript implementation of LightRAG with Bun runtime",
  "type": "module",
  "main": "dist/index.js",
  "types": "dist/index.d.ts",
  "engines": {
    "bun": ">=1.1.0",
    "node": ">=18.0.0"
  },
  "scripts": {
    "dev": "bun --watch src/index.ts",
    "dev:api": "bun --watch src/api/server.ts",
    "build": "bun build src/index.ts --outdir dist --target bun --minify",
    "build:standalone": "bun build src/api/server.ts --compile --minify --outfile lightrag-server",
    "test": "bun test",
    "test:watch": "bun test --watch",
    "test:coverage": "bun test --coverage",
    "lint": "eslint src --ext .ts",
    "lint:fix": "eslint src --ext .ts --fix",
    "format": "prettier --write \"src/**/*.ts\"",
    "typecheck": "tsc --noEmit",
    "start": "bun run dist/index.js",
    "start:api": "bun run src/api/server.ts",
    "db:generate": "drizzle-kit generate:pg",
    "db:migrate": "drizzle-kit push:pg",
    "db:studio": "drizzle-kit studio"
  },
  "dependencies": {
    "@anthropic-ai/sdk": "^0.17.0",
    "@dqbd/tiktoken": "^1.0.7",
    "@hono/zod-openapi": "^0.9.0",
    "@qdrant/js-client-rest": "^1.8.0",
    "@zilliz/milvus2-sdk-node": "^2.3.0",
    "drizzle-orm": "^0.33.0",
    "graphology": "^0.25.0",
    "hono": "^4.0.0",
    "ioredis": "^5.3.0",
    "json-repair": "^0.2.0",
    "mongodb": "^6.3.0",
    "neo4j-driver": "^5.15.0",
    "ollama": "^0.5.0",
    "openai": "^4.28.0",
    "p-limit": "^5.0.0",
    "p-queue": "^8.0.0",
    "p-retry": "^6.0.0",
    "p-timeout": "^6.0.0",
    "pgvector": "^0.2.0",
    "pino": "^8.17.0",
    "pino-pretty": "^10.3.0",
    "postgres": "^3.4.0",
    "zod": "^3.22.0"
  },
  "devDependencies": {
    "@types/bun": "^1.1.0",
    "@typescript-eslint/eslint-plugin": "^6.15.0",
    "@typescript-eslint/parser": "^6.15.0",
    "drizzle-kit": "^0.24.0",
    "eslint": "^8.56.0",
    "eslint-config-prettier": "^9.1.0",
    "eslint-plugin-import": "^2.29.0",
    "prettier": "^3.1.0",
    "typescript": "^5.3.0"
  }
}
```

**Key Changes for Bun:**
- ✅ Uses `bun --watch` instead of tsx/nodemon
- ✅ Bun's native build command for bundling
- ✅ `--compile` flag creates standalone executable
- ✅ Drizzle Kit for database migrations
- ✅ Hono instead of Fastify
- ✅ `postgres` driver instead of `pg` (faster with Bun)
- ✅ No need for ts-node, tsx, or vitest
- ✅ Smaller dependency tree

### Alternative: Node.js-Compatible package.json

If you need to support both Bun and Node.js:

```json
{
  "name": "lightrag-ts",
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "tsx watch src/index.ts",
    "dev:bun": "bun --watch src/index.ts",
    "build": "tsup",
    "test": "vitest",
    "test:bun": "bun test",
    "start": "node dist/index.js"
  },
  "dependencies": {
    "hono": "^4.0.0",
    "drizzle-orm": "^0.33.0",
    "postgres": "^3.4.0"
  },
  "devDependencies": {
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

### drizzle.config.ts

```typescript
import type { Config } from 'drizzle-kit';

export default {
  schema: './src/storage/schema.ts',
  out: './drizzle',
  driver: 'pg',
  dbCredentials: {
    connectionString: Bun.env.DATABASE_URL!,
  },
  verbose: true,
  strict: true,
} satisfies Config;
```

**Usage:**
```bash
# Generate migration files from schema changes
bun run db:generate

# Apply migrations to database
bun run db:migrate

# Open Drizzle Studio (GUI for database inspection)
bun run db:studio
```

### bunfig.toml (Bun Configuration)

```toml
[install]
# Configure package installation
cache = true
exact = true

[install.cache]
# Cache directory
dir = "~/.bun/install/cache"

[test]
# Test configuration
preload = ["./tests/setup.ts"]
coverage = true

[run]
# Runtime configuration
bun = true
silent = false

[env]
# Environment variable prefix (optional)
# Loads from .env, .env.local, .env.production
```

### Bun Test Configuration (bunfig.test.ts)

For Bun's built-in test runner:

```typescript
// tests/setup.ts
import { beforeAll, afterAll } from 'bun:test';
import { db } from '../src/db';

beforeAll(async () => {
  // Setup test database
  await db.execute(sql`CREATE EXTENSION IF NOT EXISTS vector`);
  console.log('Test database initialized');
});

afterAll(async () => {
  // Cleanup
  await db.execute(sql`DROP SCHEMA IF EXISTS test CASCADE`);
  console.log('Test database cleaned up');
});
```

**Test Example:**
```typescript
// tests/unit/storage/drizzle.test.ts
import { describe, test, expect } from 'bun:test';
import { db } from '../../../src/db';
import { textChunks, entities } from '../../../src/storage/schema';
import { eq } from 'drizzle-orm';

describe('Drizzle Storage', () => {
  test('should insert and query text chunk', async () => {
    // Insert
    await db.insert(textChunks).values({
      id: 'test-chunk-1',
      content: 'Test content',
      tokens: 10,
      fullDocId: 'test-doc',
    });

    // Query
    const chunks = await db
      .select()
      .from(textChunks)
      .where(eq(textChunks.id, 'test-chunk-1'));

    expect(chunks).toHaveLength(1);
    expect(chunks[0].content).toBe('Test content');
  });

  test('should perform vector similarity search', async () => {
    const queryVector = new Array(1536).fill(0.1);
    
    const results = await db
      .select()
      .from(entities)
      .orderBy(sql`${entities.embedding} <-> ${queryVector}`)
      .limit(10);

    expect(results.length).toBeLessThanOrEqual(10);
  });
});
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

### Development Mode with Bun

```bash
# Install Bun (if not already installed)
curl -fsSL https://bun.sh/install | bash

# Install dependencies (20-100x faster than npm)
bun install

# Start development server with hot reload
bun run dev

# Or directly run TypeScript file with watch mode
bun --watch src/api/server.ts

# Run tests with Bun's test runner (faster than Jest/Vitest)
bun test

# Run tests in watch mode
bun test --watch

# Type checking
bun run typecheck

# Database operations
bun run db:generate    # Generate migrations from schema
bun run db:migrate     # Apply migrations
bun run db:studio      # Open Drizzle Studio GUI
```

### Build for Production with Bun

```bash
# Build optimized bundle for Bun runtime
bun build src/index.ts --outdir dist --target bun --minify

# Build standalone executable (single binary, no Node.js needed!)
bun build src/api/server.ts --compile --minify --outfile lightrag-server

# The standalone binary includes Bun runtime + your code
# Can be deployed without installing Bun or Node.js
./lightrag-server

# Run production build
bun run start
```

### Alternative: Node.js Compatible Build

If you need Node.js compatibility, you can still use traditional tools:

```bash
# Using tsup for Node.js build
bun run build

# Or using Bun's bundler with Node target
bun build src/index.ts --outdir dist --target node --minify
```

### Build Configuration Comparison

**Option 1: Bun Native (Recommended)**

No configuration needed! Bun handles TypeScript natively:

```bash
# Just run TypeScript directly
bun src/api/server.ts
```

**Option 2: Using tsup (for Node.js compatibility)**

```typescript
// tsup.config.ts
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

### Performance Comparison

| Operation | npm | pnpm | Bun | Speedup |
|-----------|-----|------|-----|---------|
| Install (cold) | 20s | 10s | 0.5s | **40x faster** |
| Install (warm) | 10s | 5s | 0.2s | **50x faster** |
| Run TS file | 500ms | 500ms | 50ms | **10x faster** |
| Test suite | 5s | 5s | 1s | **5x faster** |
| Build | 3s | 3s | 0.5s | **6x faster** |

### Development Workflow Example

```bash
# 1. Clone and setup
git clone https://github.com/your-org/lightrag-ts
cd lightrag-ts
bun install  # Super fast!

# 2. Setup database
createdb lightrag
bun run db:migrate

# 3. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 4. Start development
bun --watch src/api/server.ts

# 5. Run tests in another terminal
bun test --watch

# 6. Check database with Drizzle Studio
bun run db:studio
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
