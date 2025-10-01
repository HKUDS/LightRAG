# Quick Start Guide: LightRAG TypeScript Implementation

## TL;DR - Get Started in 5 Minutes

```bash
# 1. Install Bun
curl -fsSL https://bun.sh/install | bash

# 2. Create project
bun init lightrag-ts
cd lightrag-ts

# 3. Install dependencies
bun add hono drizzle-orm postgres @hono/zod-openapi zod pgvector graphology \
  openai @anthropic-ai/sdk p-limit p-queue p-retry pino

bun add -d drizzle-kit @types/bun typescript

# 4. Setup database
createdb lightrag
psql lightrag -c "CREATE EXTENSION IF NOT EXISTS vector;"

# 5. Create schema (see full example below)
# 6. Run migrations
bun run db:generate && bun run db:migrate

# 7. Start coding!
bun --watch src/api/server.ts
```

## Recommended Technology Stack

### Core Stack
```typescript
{
  "runtime": "Bun 1.1+",           // 3x faster than Node.js
  "framework": "Hono 4.0+",        // Ultrafast web framework
  "orm": "Drizzle ORM 0.33+",      // Type-safe SQL queries
  "database": "PostgreSQL + pgvector",
  "validation": "Zod 3.22+",
  "graph": "graphology 0.25+",
  "llm": "openai 4.28+",
  "logger": "pino 8.17+",
  "testing": "bun:test"            // Built-in
}
```

### Performance Gains
- **HTTP**: 50k req/s (vs 15k with Node.js + Express)
- **Install**: 0.5s (vs 20s with npm)
- **Cold Start**: 10ms (vs 100ms with Node.js)
- **Memory**: 30% less than Node.js

## Project Structure

```
lightrag-ts/
├── src/
│   ├── api/                    # Hono API routes
│   │   ├── server.ts          # Main server
│   │   ├── routes/
│   │   │   ├── query.ts       # Query endpoints
│   │   │   ├── documents.ts   # Document management
│   │   │   └── graph.ts       # Graph visualization
│   │   └── middleware/
│   │       ├── auth.ts        # JWT authentication
│   │       └── errorHandler.ts
│   ├── core/
│   │   ├── LightRAG.ts        # Main class
│   │   ├── Pipeline.ts        # Document processing
│   │   └── QueryEngine.ts     # Query execution
│   ├── storage/
│   │   ├── schema.ts          # Drizzle schema
│   │   ├── db.ts              # Database connection
│   │   └── implementations/
│   ├── llm/
│   │   ├── providers/
│   │   │   ├── openai.ts
│   │   │   ├── anthropic.ts
│   │   │   └── ollama.ts
│   │   └── base.ts
│   ├── operations/
│   │   ├── chunking.ts        # Text chunking
│   │   ├── extraction.ts      # Entity extraction
│   │   └── retrieval.ts       # Retrieval strategies
│   └── utils/
│       ├── tokenizer.ts
│       ├── logger.ts
│       └── retry.ts
├── tests/
├── drizzle/                    # Migrations
├── package.json
├── drizzle.config.ts
├── tsconfig.json
└── bunfig.toml
```

## Essential Code Snippets

### 1. Drizzle Schema (src/storage/schema.ts)

```typescript
import { pgTable, text, varchar, timestamp, integer, vector, jsonb, serial } from 'drizzle-orm/pg-core';

// Text chunks
export const textChunks = pgTable('text_chunks', {
  id: varchar('id', { length: 255 }).primaryKey(),
  content: text('content').notNull(),
  tokens: integer('tokens').notNull(),
  fullDocId: varchar('full_doc_id', { length: 255 }).notNull(),
  metadata: jsonb('metadata'),
  createdAt: timestamp('created_at').defaultNow().notNull(),
});

// Entities
export const entities = pgTable('entities', {
  id: serial('id').primaryKey(),
  name: varchar('name', { length: 500 }).notNull().unique(),
  type: varchar('type', { length: 100 }),
  description: text('description'),
  embedding: vector('embedding', { dimensions: 1536 }),
  sourceIds: jsonb('source_ids').$type<string[]>(),
  createdAt: timestamp('created_at').defaultNow().notNull(),
});

// Relationships
export const relationships = pgTable('relationships', {
  id: serial('id').primaryKey(),
  sourceEntity: varchar('source_entity', { length: 500 }).notNull(),
  targetEntity: varchar('target_entity', { length: 500 }).notNull(),
  relationshipType: varchar('relationship_type', { length: 200 }),
  description: text('description'),
  weight: integer('weight').default(1),
  embedding: vector('embedding', { dimensions: 1536 }),
  createdAt: timestamp('created_at').defaultNow().notNull(),
});
```

### 2. Database Connection (src/storage/db.ts)

```typescript
import { drizzle } from 'drizzle-orm/postgres-js';
import postgres from 'postgres';
import * as schema from './schema';

const client = postgres(Bun.env.DATABASE_URL!, { max: 20 });
export const db = drizzle(client, { schema });
```

### 3. Hono API Server (src/api/server.ts)

```typescript
import { Hono } from 'hono';
import { cors } from 'hono/cors';
import { logger } from 'hono/logger';
import { jwt } from 'hono/jwt';
import { OpenAPIHono, createRoute } from '@hono/zod-openapi';
import { z } from 'zod';

const app = new OpenAPIHono();

app.use('*', logger());
app.use('*', cors());
app.use('/api/*', jwt({ secret: Bun.env.JWT_SECRET! }));

// Query route
const QuerySchema = z.object({
  query: z.string().min(1),
  mode: z.enum(['local', 'global', 'hybrid', 'mix', 'naive', 'bypass']).default('mix'),
  top_k: z.number().int().positive().default(40),
});

const queryRoute = createRoute({
  method: 'post',
  path: '/api/query',
  request: { body: { content: { 'application/json': { schema: QuerySchema } } } },
  responses: {
    200: { description: 'Query response' },
  },
});

app.openapi(queryRoute, async (c) => {
  const { query, mode, top_k } = c.req.valid('json');
  const result = await rag.query(query, { mode, top_k });
  return c.json(result);
});

// Start server
Bun.serve({
  port: 9621,
  fetch: app.fetch,
});
```

### 4. Type-Safe Database Queries

```typescript
import { db } from './db';
import { textChunks, entities, relationships } from './schema';
import { eq, sql } from 'drizzle-orm';

// Insert chunk
await db.insert(textChunks).values({
  id: chunkId,
  content: text,
  tokens: tokenCount,
  fullDocId: docId,
});

// Vector similarity search
const similar = await db
  .select()
  .from(entities)
  .orderBy(sql`${entities.embedding} <-> ${queryVector}`)
  .limit(10);

// Complex join
const entityWithRelations = await db.query.entities.findFirst({
  where: eq(entities.name, 'John Doe'),
  with: {
    outgoingRelations: { with: { target: true } },
  },
});
```

### 5. LLM Integration

```typescript
import OpenAI from 'openai';

export class OpenAIProvider {
  private client: OpenAI;

  constructor(apiKey: string) {
    this.client = new OpenAI({ apiKey });
  }

  async chat(messages: any[], options: any): Promise<string> {
    const response = await this.client.chat.completions.create({
      model: options.model || 'gpt-4-turbo',
      messages,
      temperature: options.temperature || 0.7,
      max_tokens: options.max_tokens || 1000,
    });
    return response.choices[0].message.content || '';
  }

  async *streamChat(messages: any[], options: any) {
    const stream = await this.client.chat.completions.create({
      model: options.model || 'gpt-4-turbo',
      messages,
      stream: true,
    });

    for await (const chunk of stream) {
      yield chunk.choices[0]?.delta?.content || '';
    }
  }

  async embeddings(texts: string[]): Promise<number[][]> {
    const response = await this.client.embeddings.create({
      model: 'text-embedding-3-small',
      input: texts,
    });
    return response.data.map(d => d.embedding);
  }
}
```

## Configuration Files

### package.json

```json
{
  "name": "lightrag-ts",
  "type": "module",
  "scripts": {
    "dev": "bun --watch src/api/server.ts",
    "build": "bun build src/index.ts --outdir dist --target bun --minify",
    "build:standalone": "bun build src/api/server.ts --compile --outfile lightrag-server",
    "test": "bun test",
    "db:generate": "drizzle-kit generate:pg",
    "db:migrate": "drizzle-kit push:pg",
    "db:studio": "drizzle-kit studio"
  },
  "dependencies": {
    "@anthropic-ai/sdk": "^0.17.0",
    "@dqbd/tiktoken": "^1.0.7",
    "@hono/zod-openapi": "^0.9.0",
    "drizzle-orm": "^0.33.0",
    "graphology": "^0.25.0",
    "hono": "^4.0.0",
    "openai": "^4.28.0",
    "p-limit": "^5.0.0",
    "p-queue": "^8.0.0",
    "p-retry": "^6.0.0",
    "pgvector": "^0.2.0",
    "pino": "^8.17.0",
    "postgres": "^3.4.0",
    "zod": "^3.22.0"
  },
  "devDependencies": {
    "@types/bun": "^1.1.0",
    "drizzle-kit": "^0.24.0",
    "typescript": "^5.3.0"
  }
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
} satisfies Config;
```

### .env

```bash
DATABASE_URL=postgresql://user:password@localhost:5432/lightrag
OPENAI_API_KEY=sk-...
JWT_SECRET=your-secret-key
PORT=9621
NODE_ENV=development
```

## Development Workflow

```bash
# 1. Start database
docker run -d --name postgres \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=lightrag \
  -p 5432:5432 \
  pgvector/pgvector:pg16

# 2. Create schema and migrate
bun run db:generate
bun run db:migrate

# 3. Start dev server with hot reload
bun --watch src/api/server.ts

# 4. Run tests in watch mode (separate terminal)
bun test --watch

# 5. Open Drizzle Studio (separate terminal)
bun run db:studio
```

## Testing

```typescript
// tests/unit/storage.test.ts
import { describe, test, expect } from 'bun:test';
import { db } from '../src/storage/db';
import { textChunks } from '../src/storage/schema';
import { eq } from 'drizzle-orm';

describe('Storage', () => {
  test('should insert and query chunk', async () => {
    await db.insert(textChunks).values({
      id: 'test-1',
      content: 'Test content',
      tokens: 10,
      fullDocId: 'doc-1',
    });

    const chunks = await db
      .select()
      .from(textChunks)
      .where(eq(textChunks.id, 'test-1'));

    expect(chunks).toHaveLength(1);
    expect(chunks[0].content).toBe('Test content');
  });
});
```

## Production Deployment

### Option 1: Standalone Binary

```bash
# Build single executable (includes Bun runtime)
bun build src/api/server.ts --compile --minify --outfile lightrag-server

# Deploy binary (no dependencies needed)
./lightrag-server
```

### Option 2: Docker

```dockerfile
FROM oven/bun:1

WORKDIR /app
COPY package.json bun.lockb ./
RUN bun install --frozen-lockfile

COPY . .
RUN bun run db:migrate

EXPOSE 9621
CMD ["bun", "run", "src/api/server.ts"]
```

### Option 3: Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: lightrag
spec:
  replicas: 3
  selector:
    matchLabels:
      app: lightrag
  template:
    metadata:
      labels:
        app: lightrag
    spec:
      containers:
      - name: lightrag
        image: lightrag:latest
        ports:
        - containerPort: 9621
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: lightrag-secrets
              key: database-url
```

## Next Steps

1. **Read full documentation**:
   - [Executive Summary](./01-executive-summary.md) - System overview
   - [Architecture](./02-architecture-documentation.md) - Detailed design
   - [Data Models](./03-data-models-and-schemas.md) - Type system
   - [Dependencies](./04-dependency-migration-guide.md) - Package mapping
   - [Project Structure](./05-typescript-project-structure-and-roadmap.md) - Complete setup

2. **Implement core features**:
   - Document processing pipeline
   - Entity extraction
   - Graph construction
   - Query engine
   - API endpoints

3. **Add tests**:
   - Unit tests for core logic
   - Integration tests for storage
   - E2E tests for API

4. **Optimize performance**:
   - Connection pooling
   - Query optimization
   - Caching strategies

5. **Deploy to production**:
   - Set up monitoring
   - Configure logging
   - Add health checks

## Resources

- **Bun**: https://bun.sh
- **Hono**: https://hono.dev
- **Drizzle ORM**: https://orm.drizzle.team
- **pgvector**: https://github.com/pgvector/pgvector
- **graphology**: https://graphology.github.io
- **Zod**: https://zod.dev

## Support

For issues or questions:
1. Check the full documentation suite
2. Review code examples in this guide
3. Consult the original Python implementation
4. Open an issue on GitHub
