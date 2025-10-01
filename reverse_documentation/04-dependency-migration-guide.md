# Dependency Migration Guide: Python to TypeScript/Node.js

## Table of Contents
1. [Core Dependencies Mapping](#core-dependencies-mapping)
2. [Storage Driver Dependencies](#storage-driver-dependencies)
3. [LLM and Embedding Dependencies](#llm-and-embedding-dependencies)
4. [API and Web Framework Dependencies](#api-and-web-framework-dependencies)
5. [Utility and Helper Dependencies](#utility-and-helper-dependencies)
6. [Migration Complexity Assessment](#migration-complexity-assessment)

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

### PostgreSQL

| Python Package | npm Package | Version | Notes |
|----------------|-------------|---------|-------|
| `asyncpg` | `pg` | ^8.11.0 | Most popular, excellent TypeScript support |
| | `drizzle-orm` | ^0.29.0 | Optional: Type-safe query builder |
| | `@neondatabase/serverless` | ^0.9.0 | For serverless environments |

**Migration complexity**: Low  
**Recommendation**: Use `pg` with connection pooling. Consider `drizzle-orm` for type-safe queries.

```typescript
// PostgreSQL connection with pg
import { Pool } from 'pg';

const pool = new Pool({
  host: process.env.PG_HOST,
  port: parseInt(process.env.PG_PORT || '5432'),
  database: process.env.PG_DATABASE,
  user: process.env.PG_USER,
  password: process.env.PG_PASSWORD,
  max: 20,  // Connection pool size
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 2000,
});

// With Drizzle ORM for type safety
import { drizzle } from 'drizzle-orm/node-postgres';
const db = drizzle(pool);
```

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

### FastAPI → Node.js Framework

| Python Package | npm Package | Version | Notes |
|----------------|-------------|---------|-------|
| `fastapi` | `fastify` | ^4.25.0 | Fast, low overhead, TypeScript-friendly |
| | `@fastify/swagger` | ^8.13.0 | OpenAPI documentation |
| | `@fastify/swagger-ui` | ^2.1.0 | Swagger UI |
| | `@fastify/cors` | ^8.5.0 | CORS support |
| | `@fastify/jwt` | ^7.2.0 | JWT authentication |

**Alternative**: Express.js (^4.18.0) - More familiar but slower

**Migration complexity**: Low  
**Recommendation**: Use Fastify for similar performance to FastAPI.

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
