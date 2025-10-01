# Dependency Migration Guide: Python to TypeScript/Bun

## Table of Contents
1. [Runtime and Build Tools](#runtime-and-build-tools)
2. [Core Dependencies Mapping](#core-dependencies-mapping)
3. [Storage Driver Dependencies](#storage-driver-dependencies)
4. [LLM and Embedding Dependencies](#llm-and-embedding-dependencies)
5. [API and Web Framework Dependencies](#api-and-web-framework-dependencies)
6. [Utility and Helper Dependencies](#utility-and-helper-dependencies)
7. [Migration Complexity Assessment](#migration-complexity-assessment)

## Runtime and Build Tools

### Bun vs Node.js Comparison

| Aspect | Python | Node.js | Bun | Recommendation |
|--------|--------|---------|-----|----------------|
| **Runtime** | CPython | V8 Engine | JavaScriptCore | **Bun** (3x faster I/O) |
| **Package Manager** | pip/poetry | npm/pnpm/yarn | Built-in | **Bun** (20-100x faster install) |
| **TypeScript** | N/A (needs transpiler) | Needs ts-node/tsx | Built-in native | **Bun** (no config needed) |
| **Bundler** | N/A | webpack/esbuild/vite | Built-in | **Bun** (3-5x faster) |
| **Test Runner** | pytest/unittest | jest/vitest | Built-in | **Bun** (faster, simpler) |
| **Watch Mode** | N/A | nodemon/tsx | Built-in | **Bun** (integrated) |
| **HTTP Performance** | ~10k req/s | ~15k req/s | ~50k req/s | **Bun** (3-4x faster) |
| **Startup Time** | Fast | ~50-100ms | ~5-10ms | **Bun** (10x faster cold start) |
| **Memory Usage** | Medium | High | Lower | **Bun** (30% less memory) |
| **Ecosystem** | Mature | Very Mature | Growing (90%+ compatible) | **Bun** for new projects |

### Bun-Specific Features

**Built-in APIs:**
```typescript
// Bun's native file I/O (faster than Node.js fs)
const file = Bun.file("./data.json");
const text = await file.text();
const json = await file.json();

// Bun's native HTTP server
Bun.serve({
  port: 3000,
  fetch(req) {
    return new Response("Hello World");
  },
});

// Bun's native crypto (faster than Node.js crypto)
const hash = Bun.hash("md5", "content");
const password = await Bun.password.hash("secret");

// Bun's native SQLite support
import { Database } from "bun:sqlite";
const db = new Database("mydb.sqlite");

// Bun's native environment variables
const apiKey = Bun.env.OPENAI_API_KEY;
```

**Installation Command:**
```bash
# Install Bun
curl -fsSL https://bun.sh/install | bash

# Initialize project
bun init

# Install dependencies (20-100x faster than npm)
bun install

# Run TypeScript directly (no compilation needed)
bun run index.ts

# Run with watch mode
bun --watch run index.ts

# Build for production
bun build --compile --minify src/index.ts --outfile lightrag

# Test
bun test
```

### Migration Complexity: Low to Medium
- **Node.js → Bun**: Very Low (95%+ compatible, drop-in replacement for most packages)
- **Python → Bun**: Low-Medium (similar to Node.js migration, but with better performance)
- **Key Benefit**: Bun eliminates need for separate build tools, test runners, and transpilers

## Core Dependencies Mapping

### Python Standard Library → Node.js/TypeScript

| Python Package | Purpose | TypeScript/Node.js Equivalent | Migration Notes |
|----------------|---------|-------------------------------|-----------------|
| `asyncio` | Async/await runtime | Native Node.js (async/await) | Direct support, different patterns (see below) |
| `dataclasses` | Data class definitions | TypeScript classes/interfaces | Use `class` with constructor or `interface` |
| `typing` | Type hints | Native TypeScript types | Superior type system in TS |
| `functools` | Function tools (partial, lru_cache) | `lodash/partial`, custom decorators | `lodash.partial` or arrow functions |
| `collections` | Counter, defaultdict, deque | Native Map/Set, or `collections-js` | Map/Set cover most cases |
| `json` | JSON parsing | Native `JSON` | Direct support |
| `hashlib` | MD5, SHA hashing | `crypto` (built-in) | `crypto.createHash('md5')` |
| `os` | OS operations | `fs`, `path` (built-in) | Similar APIs |
| `time` | Time operations | Native `Date`, `performance.now()` | Different epoch (JS uses ms) |
| `datetime` | Date/time handling | Native `Date` or `date-fns` | `date-fns` recommended for manipulation |
| `configparser` | INI file parsing | `ini` npm package | Direct equivalent |
| `warnings` | Warning system | `console.warn()` or custom | Simpler in Node.js |
| `traceback` | Stack traces | Native Error stack | `error.stack` property |

### Core Python Packages → npm Packages

| Python Package | Purpose | npm Package | Version | Migration Complexity |
|----------------|---------|-------------|---------|---------------------|
| `aiohttp` | Async HTTP client | `axios` or `undici` | ^1.6.0 or ^6.0.0 | Low - similar APIs |
| `json_repair` | Fix malformed JSON | `json-repair` | ^0.2.0 | Low - direct port exists |
| `numpy` | Numerical arrays | `@tensorflow/tfjs` or `ndarray` | ^4.0.0 or ^1.0.0 | Medium - different paradigm |
| `pandas` | Data manipulation | `danfojs` | ^1.1.0 | Medium - similar but less mature |
| `pydantic` | Data validation | `zod` or `class-validator` | ^3.22.0 or ^0.14.0 | Low - excellent TS support |
| `python-dotenv` | Environment variables | `dotenv` | ^16.0.0 | Low - identical functionality |
| `tenacity` | Retry logic | `async-retry` or `p-retry` | ^3.0.0 or ^6.0.0 | Low - similar patterns |
| `tiktoken` | OpenAI tokenizer | `@dqbd/tiktoken` or `js-tiktoken` | ^1.0.0 or ^1.0.0 | Medium - WASM-based port |
| `pypinyin` | Chinese pinyin | `pinyin` npm package | ^3.0.0 | Low - direct equivalent |

### Async/Await Pattern Differences

**Python asyncio patterns:**
```python
import asyncio

# Semaphore
semaphore = asyncio.Semaphore(4)
async with semaphore:
    await do_work()

# Gather with error handling
results = await asyncio.gather(*tasks, return_exceptions=True)

# Task cancellation
task.cancel()
await task  # Raises CancelledError

# Wait with timeout
try:
    result = await asyncio.wait_for(coro, timeout=30)
except asyncio.TimeoutError:
    pass
```

**TypeScript/Node.js equivalents:**
```typescript
import pLimit from 'p-limit';
import pTimeout from 'p-timeout';

// Semaphore using p-limit
const limit = pLimit(4);
await limit(() => doWork());

// Promise.allSettled for error handling
const results = await Promise.allSettled(promises);
results.forEach(result => {
  if (result.status === 'fulfilled') {
    console.log(result.value);
  } else {
    console.error(result.reason);
  }
});

// AbortController for cancellation
const controller = new AbortController();
fetch(url, { signal: controller.signal });
controller.abort();

// Timeout using p-timeout
try {
  const result = await pTimeout(promise, { milliseconds: 30000 });
} catch (error) {
  if (error.name === 'TimeoutError') {
    // Handle timeout
  }
}
```

**Recommended npm packages for async patterns:**
- `p-limit` (^5.0.0): Rate limiting / semaphore
- `p-queue` (^8.0.0): Priority queue for async tasks
- `p-retry` (^6.0.0): Retry with exponential backoff
- `p-timeout` (^6.0.0): Timeout for promises
- `bottleneck` (^2.19.0): Advanced rate limiting

## Storage Driver Dependencies

### PostgreSQL with Drizzle ORM (Recommended)

| Python Package | npm Package | Version | Notes |
|----------------|-------------|---------|-------|
| `asyncpg` | `drizzle-orm` | ^0.33.0 | Type-safe ORM, recommended for TypeScript |
| | `postgres` | ^3.4.0 | Fast PostgreSQL client for Drizzle (works with Bun) |
| | `pg` | ^8.11.0 | Traditional client (Node.js) |
| | `drizzle-kit` | ^0.24.0 | Migration tool |
| | `pgvector` | ^0.2.0 | Vector extension support |

**Migration complexity**: Low  
**Recommendation**: Use Drizzle ORM for type-safe queries, automatic migrations, and excellent TypeScript support.

#### Why Drizzle ORM?
- ✅ **Type-safe** queries with full inference
- ✅ **SQL-like** syntax (familiar to SQL developers)
- ✅ **Zero runtime overhead** (direct SQL generation)
- ✅ **Auto-generated** migrations from schema changes
- ✅ **Bun and Node.js** compatible
- ✅ **Lightweight** (40KB gzipped vs Prisma's 5MB)

#### Schema Definition with Drizzle

```typescript
// schema.ts - Define your database schema
import { pgTable, text, varchar, timestamp, integer, vector, jsonb, pgEnum, serial, boolean } from 'drizzle-orm/pg-core';
import { relations } from 'drizzle-orm';

// Enums
export const docStatusEnum = pgEnum('doc_status', ['pending', 'processing', 'completed', 'failed']);
export const queryModeEnum = pgEnum('query_mode', ['local', 'global', 'hybrid', 'mix', 'naive', 'bypass']);

// Text chunks table (KV storage)
export const textChunks = pgTable('text_chunks', {
  id: varchar('id', { length: 255 }).primaryKey(),
  content: text('content').notNull(),
  tokens: integer('tokens').notNull(),
  fullDocId: varchar('full_doc_id', { length: 255 }).notNull(),
  metadata: jsonb('metadata'),
  createdAt: timestamp('created_at').defaultNow().notNull(),
});

// Document status table
export const documents = pgTable('documents', {
  id: varchar('id', { length: 255 }).primaryKey(),
  filename: text('filename'),
  status: docStatusEnum('status').default('pending').notNull(),
  chunkCount: integer('chunk_count').default(0),
  entityCount: integer('entity_count').default(0),
  relationCount: integer('relation_count').default(0),
  processedAt: timestamp('processed_at'),
  errorMessage: text('error_message'),
  createdAt: timestamp('created_at').defaultNow().notNull(),
  updatedAt: timestamp('updated_at').defaultNow().notNull(),
});

// Entities table (graph nodes)
export const entities = pgTable('entities', {
  id: serial('id').primaryKey(),
  name: varchar('name', { length: 500 }).notNull().unique(),
  type: varchar('type', { length: 100 }),
  description: text('description'),
  sourceIds: jsonb('source_ids').$type<string[]>(),
  embedding: vector('embedding', { dimensions: 1536 }), // For pgvector
  degree: integer('degree').default(0),
  createdAt: timestamp('created_at').defaultNow().notNull(),
});

// Relationships table (graph edges)
export const relationships = pgTable('relationships', {
  id: serial('id').primaryKey(),
  sourceEntity: varchar('source_entity', { length: 500 }).notNull(),
  targetEntity: varchar('target_entity', { length: 500 }).notNull(),
  relationshipType: varchar('relationship_type', { length: 200 }),
  description: text('description'),
  weight: integer('weight').default(1),
  sourceIds: jsonb('source_ids').$type<string[]>(),
  embedding: vector('embedding', { dimensions: 1536 }),
  createdAt: timestamp('created_at').defaultNow().notNull(),
});

// Define relations
export const entityRelations = relations(entities, ({ many }) => ({
  outgoingRelations: many(relationships, { relationName: 'source' }),
  incomingRelations: many(relationships, { relationName: 'target' }),
}));

export const relationshipRelations = relations(relationships, ({ one }) => ({
  source: one(entities, {
    fields: [relationships.sourceEntity],
    references: [entities.name],
    relationName: 'source',
  }),
  target: one(entities, {
    fields: [relationships.targetEntity],
    references: [entities.name],
    relationName: 'target',
  }),
}));
```

#### Database Connection with Drizzle

```typescript
// db.ts - Setup database connection
import { drizzle } from 'drizzle-orm/postgres-js';
import postgres from 'postgres';
import * as schema from './schema';

// For Bun or Node.js
const connectionString = process.env.DATABASE_URL!;

// Create connection pool
const client = postgres(connectionString, {
  max: 20, // Connection pool size
  idle_timeout: 30,
  connect_timeout: 10,
});

// Create Drizzle instance
export const db = drizzle(client, { schema });

// For migrations
import { migrate } from 'drizzle-orm/postgres-js/migrator';

export async function runMigrations() {
  await migrate(db, { migrationsFolder: './drizzle' });
  console.log('Migrations completed');
}
```

#### Type-Safe Queries with Drizzle

```typescript
import { db } from './db';
import { textChunks, entities, relationships, documents } from './schema';
import { eq, and, or, sql, desc, asc } from 'drizzle-orm';

// Insert text chunk
await db.insert(textChunks).values({
  id: chunkId,
  content: chunkContent,
  tokens: tokenCount,
  fullDocId: docId,
  metadata: { page: 1, section: 'intro' },
});

// Query text chunks
const chunks = await db
  .select()
  .from(textChunks)
  .where(eq(textChunks.fullDocId, docId))
  .orderBy(desc(textChunks.createdAt));

// Insert entity with type inference
await db.insert(entities).values({
  name: 'John Doe',
  type: 'Person',
  description: 'CEO of Company X',
  sourceIds: ['doc1', 'doc2'],
});

// Complex join query - get entity with relationships
const entityWithRelations = await db.query.entities.findFirst({
  where: eq(entities.name, 'John Doe'),
  with: {
    outgoingRelations: {
      with: {
        target: true,
      },
    },
  },
});

// Vector similarity search with pgvector
const similarEntities = await db
  .select()
  .from(entities)
  .orderBy(sql`${entities.embedding} <-> ${queryVector}`)
  .limit(10);

// Transaction example
await db.transaction(async (tx) => {
  // Insert entity
  const [entity] = await tx.insert(entities).values({
    name: 'Jane Smith',
    type: 'Person',
  }).returning();
  
  // Insert relationship
  await tx.insert(relationships).values({
    sourceEntity: 'John Doe',
    targetEntity: entity.name,
    relationshipType: 'KNOWS',
  });
});

// Batch insert for performance
await db.insert(textChunks).values([
  { id: '1', content: 'chunk1', tokens: 100, fullDocId: 'doc1' },
  { id: '2', content: 'chunk2', tokens: 150, fullDocId: 'doc1' },
  { id: '3', content: 'chunk3', tokens: 120, fullDocId: 'doc1' },
]);

// Update document status
await db
  .update(documents)
  .set({ 
    status: 'completed',
    processedAt: new Date(),
    chunkCount: 10,
  })
  .where(eq(documents.id, docId));

// Delete old data
await db
  .delete(textChunks)
  .where(
    and(
      eq(textChunks.fullDocId, docId),
      sql`${textChunks.createdAt} < NOW() - INTERVAL '30 days'`
    )
  );
```

#### Drizzle Kit - Migrations

```typescript
// drizzle.config.ts
import type { Config } from 'drizzle-kit';

export default {
  schema: './src/schema.ts',
  out: './drizzle',
  driver: 'pg',
  dbCredentials: {
    connectionString: process.env.DATABASE_URL!,
  },
} satisfies Config;
```

**Migration Commands:**
```bash
# Generate migration from schema changes
bun drizzle-kit generate:pg

# Apply migrations
bun drizzle-kit push:pg

# View current migrations
bun drizzle-kit up:pg

# Drop all tables (be careful!)
bun drizzle-kit drop
```

#### Alternative: Plain SQL with postgres.js (for complex queries)

```typescript
import postgres from 'postgres';

const sql = postgres(connectionString);

// Complex graph traversal query
const result = await sql`
  WITH RECURSIVE entity_network AS (
    SELECT id, name, type, 1 as depth
    FROM entities
    WHERE name = ${startEntity}
    
    UNION ALL
    
    SELECT e.id, e.name, e.type, en.depth + 1
    FROM entities e
    JOIN relationships r ON r.target_entity = e.name
    JOIN entity_network en ON r.source_entity = en.name
    WHERE en.depth < ${maxDepth}
  )
  SELECT DISTINCT * FROM entity_network;
`;
```

**Performance Benefits:**
- Drizzle generates optimal SQL (no ORM overhead)
- Connection pooling built-in
- Prepared statements for security
- Type-safe at compile time, fast at runtime

### PostgreSQL pgvector Extension

| Python Package | npm Package | Version | Notes |
|----------------|-------------|---------|-------|
| `pgvector` | `pgvector` | ^0.1.0 | Official Node.js client |

**Migration complexity**: Low  
**Implementation**: Use `pgvector` npm package with `pg`:

```typescript
import pgvector from 'pgvector/pg';

// Register pgvector type
await pgvector.registerType(pool);

// Insert vector
await pool.query(
  'INSERT INTO items (embedding) VALUES ($1)',
  [pgvector.toSql([1.0, 2.0, 3.0])]
);

// Query by similarity
const result = await pool.query(
  'SELECT * FROM items ORDER BY embedding <-> $1 LIMIT 10',
  [pgvector.toSql(queryVector)]
);
```

### MongoDB

| Python Package | npm Package | Version | Notes |
|----------------|-------------|---------|-------|
| `motor` | `mongodb` | ^6.3.0 | Official MongoDB driver |
| | `mongoose` | ^8.0.0 | Optional: ODM for schemas |

**Migration complexity**: Low  
**Recommendation**: Use official `mongodb` driver. Add `mongoose` if you need schema validation.

```typescript
import { MongoClient } from 'mongodb';

const client = new MongoClient(process.env.MONGODB_URI!, {
  maxPoolSize: 50,
  minPoolSize: 10,
  serverSelectionTimeoutMS: 5000,
});

await client.connect();
const db = client.db('lightrag');
const collection = db.collection('entities');
```

### Redis

| Python Package | npm Package | Version | Notes |
|----------------|-------------|---------|-------|
| `redis-py` | `ioredis` | ^5.3.0 | Better TypeScript support than `redis` |
| | `redis` | ^4.6.0 | Official client, good but less TS-friendly |

**Migration complexity**: Low  
**Recommendation**: Use `ioredis` for better TypeScript experience and cluster support.

```typescript
import Redis from 'ioredis';

const redis = new Redis({
  host: process.env.REDIS_HOST,
  port: parseInt(process.env.REDIS_PORT || '6379'),
  password: process.env.REDIS_PASSWORD,
  db: 0,
  maxRetriesPerRequest: 3,
  enableReadyCheck: true,
  lazyConnect: false,
});

// Cluster support
const cluster = new Redis.Cluster([
  { host: 'node1', port: 6379 },
  { host: 'node2', port: 6379 },
]);
```

### Neo4j

| Python Package | npm Package | Version | Notes |
|----------------|-------------|---------|-------|
| `neo4j` | `neo4j-driver` | ^5.15.0 | Official driver with TypeScript support |

**Migration complexity**: Low  
**Recommendation**: Use official driver with TypeScript type definitions.

```typescript
import neo4j from 'neo4j-driver';

const driver = neo4j.driver(
  process.env.NEO4J_URI!,
  neo4j.auth.basic(
    process.env.NEO4J_USER!,
    process.env.NEO4J_PASSWORD!
  ),
  {
    maxConnectionPoolSize: 50,
    connectionTimeout: 30000,
  }
);

const session = driver.session({ database: 'neo4j' });
try {
  const result = await session.run(
    'MATCH (n:Entity {name: $name}) RETURN n',
    { name: 'John' }
  );
} finally {
  await session.close();
}
```

### Memgraph

| Python Package | npm Package | Version | Notes |
|----------------|-------------|---------|-------|
| `neo4j` | `neo4j-driver` | ^5.15.0 | Compatible with Neo4j driver |

**Migration complexity**: Low  
**Note**: Memgraph uses the Neo4j protocol, so the same driver works.

### NetworkX (Graph Library)

| Python Package | npm Package | Version | Notes |
|----------------|-------------|---------|-------|
| `networkx` | `graphology` | ^0.25.0 | Modern graph library for JS |
| | `cytoscape` | ^3.28.0 | Alternative with visualization |

**Migration complexity**: Medium  
**Recommendation**: Use `graphology` - most feature-complete and maintained.

```typescript
import Graph from 'graphology';

const graph = new Graph({ type: 'undirected' });

// Add nodes and edges
graph.addNode('A', { name: 'Node A', type: 'entity' });
graph.addNode('B', { name: 'Node B', type: 'entity' });
graph.addEdge('A', 'B', { weight: 0.5, description: 'related to' });

// Query
const degree = graph.degree('A');
const neighbors = graph.neighbors('A');
const hasEdge = graph.hasEdge('A', 'B');

// Serialization
import { serialize, deserialize } from 'graphology-utils';
const json = serialize(graph);
const newGraph = deserialize(json);
```

### FAISS (Vector Search)

| Python Package | npm Package | Version | Notes |
|----------------|-------------|---------|-------|
| `faiss-cpu` | **No direct port** | N/A | Need alternative approach |

**Migration complexity**: High  
**Alternatives**:
1. **hnswlib-node** (^2.0.0): HNSW algorithm implementation
2. **vectra** (^0.4.0): Simple vector database for Node.js
3. Use cloud services: Pinecone, Weaviate, Qdrant

```typescript
// Option 1: hnswlib-node (closest to FAISS)
import { HierarchicalNSW } from 'hnswlib-node';

const index = new HierarchicalNSW('cosine', 1536);
index.initIndex(10000); // Max elements
index.addItems(vectors, labels);

const results = index.searchKnn(queryVector, 10);

// Option 2: vectra (simpler, file-based)
import { LocalIndex } from 'vectra';

const index = new LocalIndex('./vectors');
await index.createIndex();
await index.insertItem({
  id: 'item1',
  vector: [0.1, 0.2, ...],
  metadata: { text: 'content' }
});

const results = await index.queryItems(queryVector, 10);
```

### Milvus

| Python Package | npm Package | Version | Notes |
|----------------|-------------|---------|-------|
| `pymilvus` | `@zilliz/milvus2-sdk-node` | ^2.3.0 | Official client |

**Migration complexity**: Low  
**Recommendation**: Use official SDK.

```typescript
import { MilvusClient } from '@zilliz/milvus2-sdk-node';

const client = new MilvusClient({
  address: process.env.MILVUS_ADDRESS!,
  username: process.env.MILVUS_USER,
  password: process.env.MILVUS_PASSWORD,
});

// Create collection
await client.createCollection({
  collection_name: 'entities',
  fields: [
    { name: 'id', data_type: DataType.VarChar, is_primary_key: true },
    { name: 'vector', data_type: DataType.FloatVector, dim: 1536 },
  ],
});

// Insert vectors
await client.insert({
  collection_name: 'entities',
  data: [{ id: '1', vector: [...] }],
});

// Search
const results = await client.search({
  collection_name: 'entities',
  vector: queryVector,
  limit: 10,
});
```

### Qdrant

| Python Package | npm Package | Version | Notes |
|----------------|-------------|---------|-------|
| `qdrant-client` | `@qdrant/js-client-rest` | ^1.8.0 | Official REST client |

**Migration complexity**: Low  
**Recommendation**: Use official client.

```typescript
import { QdrantClient } from '@qdrant/js-client-rest';

const client = new QdrantClient({
  url: process.env.QDRANT_URL!,
  apiKey: process.env.QDRANT_API_KEY,
});

// Create collection
await client.createCollection('entities', {
  vectors: { size: 1536, distance: 'Cosine' },
});

// Upsert vectors
await client.upsert('entities', {
  points: [
    {
      id: 1,
      vector: [...],
      payload: { text: 'content' },
    },
  ],
});

// Search
const results = await client.search('entities', {
  vector: queryVector,
  limit: 10,
});
```

## LLM and Embedding Dependencies

### OpenAI

| Python Package | npm Package | Version | Notes |
|----------------|-------------|---------|-------|
| `openai` | `openai` | ^4.28.0 | Official SDK with TypeScript |

**Migration complexity**: Low  
**Recommendation**: Use official SDK - excellent TypeScript support.

```typescript
import OpenAI from 'openai';

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  maxRetries: 3,
  timeout: 60000,
});

// Chat completion
const completion = await openai.chat.completions.create({
  model: 'gpt-4o-mini',
  messages: [{ role: 'user', content: 'Hello!' }],
  temperature: 0.7,
});

// Streaming
const stream = await openai.chat.completions.create({
  model: 'gpt-4o-mini',
  messages: [{ role: 'user', content: 'Hello!' }],
  stream: true,
});

for await (const chunk of stream) {
  process.stdout.write(chunk.choices[0]?.delta?.content || '');
}

// Embeddings
const embedding = await openai.embeddings.create({
  model: 'text-embedding-3-small',
  input: 'Text to embed',
});
```

### Anthropic

| Python Package | npm Package | Version | Notes |
|----------------|-------------|---------|-------|
| `anthropic` | `@anthropic-ai/sdk` | ^0.17.0 | Official SDK |

**Migration complexity**: Low

```typescript
import Anthropic from '@anthropic-ai/sdk';

const client = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
});

const message = await client.messages.create({
  model: 'claude-3-opus-20240229',
  max_tokens: 1024,
  messages: [{ role: 'user', content: 'Hello!' }],
});

// Streaming
const stream = await client.messages.stream({
  model: 'claude-3-opus-20240229',
  max_tokens: 1024,
  messages: [{ role: 'user', content: 'Hello!' }],
});

for await (const chunk of stream) {
  process.stdout.write(chunk.delta.text || '');
}
```

### Ollama

| Python Package | npm Package | Version | Notes |
|----------------|-------------|---------|-------|
| `ollama` | `ollama` | ^0.5.0 | Official Node.js library |

**Migration complexity**: Low

```typescript
import ollama from 'ollama';

const response = await ollama.chat({
  model: 'llama2',
  messages: [{ role: 'user', content: 'Hello!' }],
});

// Streaming
const stream = await ollama.chat({
  model: 'llama2',
  messages: [{ role: 'user', content: 'Hello!' }],
  stream: true,
});

for await (const chunk of stream) {
  process.stdout.write(chunk.message.content);
}

// Embeddings
const embedding = await ollama.embeddings({
  model: 'llama2',
  prompt: 'Text to embed',
});
```

### Hugging Face

| Python Package | npm Package | Version | Notes |
|----------------|-------------|---------|-------|
| `transformers` | `@xenova/transformers` | ^2.12.0 | Transformers.js (ONNX-based) |
| `sentence-transformers` | **Use Inference API** | N/A | No direct equivalent |

**Migration complexity**: Medium to High  
**Recommendation**: Use Hugging Face Inference API for server-side, or Transformers.js for client-side.

```typescript
// Option 1: Hugging Face Inference API (recommended for server)
import { HfInference } from '@huggingface/inference';

const hf = new HfInference(process.env.HF_TOKEN);

const result = await hf.textGeneration({
  model: 'meta-llama/Llama-2-7b-chat-hf',
  inputs: 'The answer to the universe is',
});

// Option 2: Transformers.js (ONNX models, client-side or server)
import { pipeline } from '@xenova/transformers';

const embedder = await pipeline('feature-extraction', 
  'Xenova/all-MiniLM-L6-v2');
const embeddings = await embedder('Text to embed', {
  pooling: 'mean',
  normalize: true,
});
```

### Tokenization (tiktoken)

| Python Package | npm Package | Version | Notes |
|----------------|-------------|---------|-------|
| `tiktoken` | `@dqbd/tiktoken` | ^1.0.7 | WASM-based port |
| | `js-tiktoken` | ^1.0.10 | Alternative pure JS |

**Migration complexity**: Medium  
**Recommendation**: Use `@dqbd/tiktoken` for best compatibility.

```typescript
import { encoding_for_model } from '@dqbd/tiktoken';

// Initialize encoder for specific model
const encoder = encoding_for_model('gpt-4o-mini');

// Encode text to tokens
const tokens = encoder.encode('Hello, world!');
console.log(`Token count: ${tokens.length}`);

// Decode tokens back to text
const text = encoder.decode(tokens);

// Don't forget to free resources
encoder.free();

// Alternative: js-tiktoken (pure JS, no WASM)
import { encodingForModel } from 'js-tiktoken';
const enc = encodingForModel('gpt-4o-mini');
const tokenCount = enc.encode('Hello, world!').length;
```

## API and Web Framework Dependencies

### FastAPI → Hono (Recommended) or Fastify

#### Option 1: Hono (Recommended for Bun)

| Python Package | npm Package | Version | Notes |
|----------------|-------------|---------|-------|
| `fastapi` | `hono` | ^4.0.0 | Ultrafast, runtime-agnostic, TypeScript-first |
| | `@hono/zod-openapi` | ^0.9.0 | OpenAPI with Zod validation |
| | `zod` | ^3.22.0 | Type-safe validation |

**Migration complexity**: Low  
**Recommendation**: Use Hono for best performance (3-4x faster than Express, works on Bun/Node/Deno/Cloudflare Workers).

**Hono Example:**
```typescript
import { Hono } from 'hono';
import { cors } from 'hono/cors';
import { jwt } from 'hono/jwt';
import { logger } from 'hono/logger';
import { z } from 'zod';
import { zValidator } from '@hono/zod-validator';
import { createRoute, OpenAPIHono } from '@hono/zod-openapi';

// Type-safe OpenAPI routes
const app = new OpenAPIHono();

// Middleware
app.use('*', logger());
app.use('*', cors());
app.use('/api/*', jwt({ secret: process.env.JWT_SECRET! }));

// Query schema
const QuerySchema = z.object({
  query: z.string().min(1).openapi({
    example: 'What is LightRAG?',
  }),
  mode: z.enum(['local', 'global', 'hybrid', 'mix', 'naive', 'bypass']).default('mix'),
  top_k: z.number().positive().default(40),
  stream: z.boolean().default(false),
});

// Type-safe route with OpenAPI
const queryRoute = createRoute({
  method: 'post',
  path: '/api/query',
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
          schema: z.object({
            response: z.string(),
            sources: z.array(z.string()),
          }),
        },
      },
    },
  },
});

app.openapi(queryRoute, async (c) => {
  const { query, mode, top_k, stream } = c.req.valid('json');
  
  // Process query
  const response = await processQuery(query, mode, top_k, stream);
  
  return c.json(response);
});

// OpenAPI documentation
app.doc('/openapi.json', {
  openapi: '3.0.0',
  info: {
    title: 'LightRAG API',
    version: '1.0.0',
  },
});

// Swagger UI (optional, can use external tool)
app.get('/docs', (c) => {
  return c.html(swaggerUIHtml('/openapi.json'));
});

// Start server (Bun)
export default app;

// Or use Bun.serve
Bun.serve({
  port: 9621,
  fetch: app.fetch,
});
```

#### Option 2: Fastify (Alternative for Node.js)

| Python Package | npm Package | Version | Notes |
|----------------|-------------|---------|-------|
| `fastapi` | `fastify` | ^4.25.0 | Fast, low overhead, TypeScript-friendly |
| | `@fastify/swagger` | ^8.13.0 | OpenAPI documentation |
| | `@fastify/swagger-ui` | ^2.1.0 | Swagger UI |
| | `@fastify/cors` | ^8.5.0 | CORS support |
| | `@fastify/jwt` | ^7.2.0 | JWT authentication |

**Migration complexity**: Low  
**Use Case**: When you need Node.js-specific features or existing Fastify ecosystem.

**Fastify Example:**
```typescript
import Fastify from 'fastify';
import fastifySwagger from '@fastify/swagger';
import fastifySwaggerUi from '@fastify/swagger-ui';
import fastifyJwt from '@fastify/jwt';

const app = Fastify({ logger: true });

// Swagger/OpenAPI
await app.register(fastifySwagger, {
  openapi: {
    info: { title: 'LightRAG API', version: '1.0.0' },
  },
});
await app.register(fastifySwaggerUi, {
  routePrefix: '/docs',
});

// JWT
await app.register(fastifyJwt, {
  secret: process.env.JWT_SECRET!,
});

// Route with schema
app.post('/query', {
  schema: {
    body: {
      type: 'object',
      required: ['query'],
      properties: {
        query: { type: 'string' },
        mode: { type: 'string', enum: ['local', 'global', 'hybrid'] },
      },
    },
  },
}, async (request, reply) => {
  return { response: 'Answer' };
});

await app.listen({ port: 9621, host: '0.0.0.0' });
```

**Performance Comparison (req/s):**
- FastAPI (Python): ~10,000
- Express.js: ~15,000
- Fastify: ~30,000
- **Hono on Bun: ~50,000** ⚡

### Pydantic → TypeScript Validation

| Python Package | npm Package | Version | Notes |
|----------------|-------------|---------|-------|
| `pydantic` | `zod` | ^3.22.0 | Runtime validation, best TS integration |
| | `class-validator` | ^0.14.0 | Decorator-based validation |
| | `joi` | ^17.12.0 | Traditional validation library |

**Migration complexity**: Low  
**Recommendation**: Use `zod` for runtime validation with excellent TypeScript inference.

```typescript
import { z } from 'zod';

// Define schema
const QuerySchema = z.object({
  query: z.string().min(3),
  mode: z.enum(['local', 'global', 'hybrid', 'mix', 'naive', 'bypass']).default('mix'),
  top_k: z.number().positive().default(40),
  stream: z.boolean().default(false),
});

// Infer TypeScript type from schema
type QueryInput = z.infer<typeof QuerySchema>;

// Validate at runtime
function handleQuery(data: unknown) {
  const validated = QuerySchema.parse(data); // Throws on invalid
  // validated is now typed as QueryInput
  return validated;
}

// Safe parse (returns result object)
const result = QuerySchema.safeParse(data);
if (result.success) {
  console.log(result.data);
} else {
  console.error(result.error);
}
```

### Uvicorn → Node.js Server

| Python Package | npm Package | Version | Notes |
|----------------|-------------|---------|-------|
| `uvicorn` | Native Node.js | N/A | Node.js has built-in HTTP server |

**Migration complexity**: None  
**Note**: Fastify/Express handle the server internally. For clustering:

```typescript
import cluster from 'cluster';
import os from 'os';

if (cluster.isPrimary) {
  const numCPUs = os.cpus().length;
  for (let i = 0; i < numCPUs; i++) {
    cluster.fork();
  }
} else {
  // Start Fastify server
  await app.listen({ port: 9621 });
}
```

## Utility and Helper Dependencies

### Text Processing and Utilities

| Python Package | npm Package | Version | Migration Complexity |
|----------------|-------------|---------|---------------------|
| `pypinyin` | `pinyin` | ^3.0.0 | Low - direct equivalent |
| `xlsxwriter` | `xlsx` or `exceljs` | ^0.18.0 or ^4.3.0 | Low - good alternatives |
| `pypdf2` | `pdf-parse` | ^1.1.1 | Medium - different API |
| `python-docx` | `docx` | ^8.5.0 | Low - similar API |
| `python-pptx` | `pptxgenjs` | ^3.12.0 | Medium - different focus |
| `openpyxl` | `xlsx` | ^0.18.0 | Low - handles both read/write |

### Logging and Monitoring

| Python Package | npm Package | Version | Notes |
|----------------|-------------|---------|-------|
| `logging` | `pino` | ^8.17.0 | Fast, structured logging |
| | `winston` | ^3.11.0 | Feature-rich alternative |
| `psutil` | `systeminformation` | ^5.21.0 | System monitoring |

```typescript
import pino from 'pino';

const logger = pino({
  level: process.env.LOG_LEVEL || 'info',
  transport: {
    target: 'pino-pretty',
    options: { colorize: true },
  },
});

logger.info({ user: 'John' }, 'User logged in');
logger.error({ err: error }, 'Operation failed');
```

### Authentication

| Python Package | npm Package | Version | Notes |
|----------------|-------------|---------|-------|
| `PyJWT` | `jsonwebtoken` | ^9.0.2 | JWT creation and verification |
| `python-jose` | `jose` | ^5.2.0 | More complete JWT/JWE/JWS library |
| `passlib` | `bcrypt` | ^5.1.0 | Password hashing |

```typescript
import jwt from 'jsonwebtoken';
import bcrypt from 'bcrypt';

// JWT
const token = jwt.sign(
  { user_id: 123 },
  process.env.JWT_SECRET!,
  { expiresIn: '24h' }
);

const decoded = jwt.verify(token, process.env.JWT_SECRET!);

// Password hashing
const hash = await bcrypt.hash('password', 10);
const isValid = await bcrypt.compare('password', hash);
```

## Migration Complexity Assessment

### Complexity Levels Defined

**Low Complexity** (1-2 days per component):
- Direct npm equivalent exists with similar API
- Well-documented TypeScript support
- Minimal code changes required
- Examples: PostgreSQL, MongoDB, Redis, OpenAI, Anthropic

**Medium Complexity** (3-5 days per component):
- npm equivalent exists but with different API patterns
- Requires adapter layer or wrapper
- Good TypeScript support but needs configuration
- Examples: NetworkX → graphology, tiktoken, Hugging Face

**High Complexity** (1-2 weeks per component):
- No direct npm equivalent
- Requires custom implementation or major architectural changes
- May need cloud service integration
- Examples: FAISS → alternatives, sentence-transformers

### Overall Migration Complexity by Category

| Category | Complexity | Estimated Effort | Risk Level |
|----------|-----------|------------------|-----------|
| Core Storage (PostgreSQL, MongoDB, Redis) | Low | 1 week | Low |
| Graph Storage (Neo4j, NetworkX) | Medium | 1-2 weeks | Medium |
| Vector Storage (FAISS alternatives) | Medium-High | 2 weeks | Medium |
| LLM Integration (OpenAI, Anthropic, Ollama) | Low | 3 days | Low |
| Tokenization (tiktoken) | Medium | 3 days | Medium |
| API Framework (FastAPI → Fastify) | Low | 1 week | Low |
| Validation (Pydantic → Zod) | Low | 3 days | Low |
| Authentication & Security | Low | 3 days | Low |
| File Processing | Medium | 1 week | Medium |
| Utilities & Helpers | Low | 3 days | Low |

### Version Compatibility Matrix

| Dependency | Recommended Version | Minimum Node.js | Notes |
|-----------|-------------------|----------------|-------|
| Node.js | 20 LTS | 18.0.0 | Use LTS for production |
| TypeScript | 5.3.0+ | N/A | For latest type features |
| pg | ^8.11.0 | 14.0.0 | PostgreSQL client |
| mongodb | ^6.3.0 | 16.0.0 | MongoDB driver |
| ioredis | ^5.3.0 | 14.0.0 | Redis client |
| neo4j-driver | ^5.15.0 | 14.0.0 | Neo4j client |
| graphology | ^0.25.0 | 14.0.0 | Graph library |
| openai | ^4.28.0 | 18.0.0 | OpenAI SDK |
| fastify | ^4.25.0 | 18.0.0 | Web framework |
| zod | ^3.22.0 | 16.0.0 | Validation |
| pino | ^8.17.0 | 14.0.0 | Logging |
| @dqbd/tiktoken | ^1.0.7 | 14.0.0 | Tokenization |

### Migration Strategy Recommendations

**Phase 1 - Core Infrastructure** (Weeks 1-2):
- Set up TypeScript project structure
- Migrate storage abstractions and interfaces
- Implement PostgreSQL storage (reference implementation)
- Set up testing framework

**Phase 2 - Storage Implementations** (Weeks 3-4):
- Migrate MongoDB, Redis implementations
- Implement NetworkX → graphology migration
- Set up vector storage alternatives (Qdrant or Milvus)
- Implement document status storage

**Phase 3 - LLM Integration** (Week 5):
- Migrate OpenAI integration
- Add Anthropic and Ollama support
- Implement tiktoken for tokenization
- Add retry and rate limiting logic

**Phase 4 - Core Logic** (Weeks 6-8):
- Migrate chunking logic
- Implement entity extraction
- Implement graph merging
- Set up pipeline processing

**Phase 5 - Query Engine** (Weeks 8-9):
- Implement query modes
- Add context building
- Integrate reranking
- Add streaming support

**Phase 6 - API Layer** (Week 10):
- Build Fastify server
- Implement all endpoints
- Add authentication
- Set up OpenAPI docs

**Phase 7 - Testing & Polish** (Weeks 11-12):
- Comprehensive testing
- Performance optimization
- Documentation
- Deployment setup

## Summary

This dependency migration guide provides:

1. **Complete mapping** of Python packages to Node.js/npm equivalents
2. **Code examples** showing API differences and migration patterns
3. **Complexity assessment** for each dependency category
4. **Version recommendations** with compatibility notes
5. **Migration strategy** with phased approach
6. **Risk assessment** for high-complexity migrations

The migration is very feasible with most dependencies having good or excellent npm equivalents. The main challenges are:
- FAISS vector search (use Qdrant, Milvus, or hnswlib-node)
- NetworkX graph library (use graphology)
- Sentence transformers (use Hugging Face Inference API)

With proper planning and the recommended npm packages, a production-ready TypeScript implementation can be achieved in 12-14 weeks with a small team.
