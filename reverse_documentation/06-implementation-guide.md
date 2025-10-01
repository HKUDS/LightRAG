# Implementation Guide: Key Components and Patterns

## Table of Contents
1. [Storage Implementation Patterns](#storage-implementation-patterns)
2. [LLM Integration Implementation](#llm-integration-implementation)
3. [Document Processing Pipeline](#document-processing-pipeline)
4. [Query Engine Implementation](#query-engine-implementation)
5. [Error Handling and Resilience](#error-handling-and-resilience)
6. [Performance Optimization](#performance-optimization)
7. [Testing Patterns](#testing-patterns)

## Storage Implementation Patterns

### PostgreSQL KV Storage Implementation

A complete example showing connection pooling, error handling, and workspace isolation.

```typescript
import { Pool, PoolClient } from 'pg';
import { BaseKVStorage } from '../base/BaseKVStorage';
import { EmbeddingFunc } from '@/types/models';

export class PostgresKVStorage extends BaseKVStorage {
  private pool: Pool;
  private namespace: string;
  private tableName: string;

  constructor(config: StorageConfig) {
    super(config.namespace, config.workspace, config.global_config);
    
    // Create connection pool
    this.pool = new Pool({
      host: process.env.PG_HOST,
      port: parseInt(process.env.PG_PORT || '5432'),
      database: process.env.PG_DATABASE,
      user: process.env.PG_USER,
      password: process.env.PG_PASSWORD,
      max: 20,  // Max connections in pool
      idleTimeoutMillis: 30000,
      connectionTimeoutMillis: 2000,
    });

    // Workspace-based namespace
    this.namespace = config.workspace 
      ? `${config.workspace}_${config.namespace}`
      : config.namespace;
      
    this.tableName = `kv_${this.namespace}`;
  }

  async initialize(): Promise<void> {
    const client = await this.pool.connect();
    try {
      // Create table if not exists
      await client.query(`
        CREATE TABLE IF NOT EXISTS ${this.tableName} (
          id TEXT PRIMARY KEY,
          data JSONB NOT NULL,
          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
          updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
      `);
      
      // Create index on JSONB data
      await client.query(`
        CREATE INDEX IF NOT EXISTS idx_${this.tableName}_data 
        ON ${this.tableName} USING GIN (data)
      `);
      
      logger.info(`PostgresKVStorage initialized: ${this.tableName}`);
    } finally {
      client.release();
    }
  }

  async get_by_id(id: string): Promise<Record<string, any> | null> {
    try {
      const result = await this.pool.query(
        `SELECT data FROM ${this.tableName} WHERE id = $1`,
        [id]
      );
      
      return result.rows[0]?.data || null;
    } catch (error) {
      logger.error({ error, id }, 'Failed to get by id');
      throw new StorageError('Failed to retrieve data', { cause: error });
    }
  }

  async get_by_ids(ids: string[]): Promise<Record<string, any>[]> {
    if (ids.length === 0) return [];
    
    try {
      const result = await this.pool.query(
        `SELECT data FROM ${this.tableName} WHERE id = ANY($1)`,
        [ids]
      );
      
      return result.rows.map(row => row.data);
    } catch (error) {
      logger.error({ error, ids }, 'Failed to get by ids');
      throw new StorageError('Failed to retrieve data', { cause: error });
    }
  }

  async filter_keys(keys: Set<string>): Promise<Set<string>> {
    if (keys.size === 0) return new Set();
    
    const keyArray = Array.from(keys);
    
    try {
      const result = await this.pool.query(
        `SELECT id FROM ${this.tableName} WHERE id = ANY($1)`,
        [keyArray]
      );
      
      const existingKeys = new Set(result.rows.map(row => row.id));
      return new Set(keyArray.filter(key => !existingKeys.has(key)));
    } catch (error) {
      logger.error({ error }, 'Failed to filter keys');
      throw new StorageError('Failed to filter keys', { cause: error });
    }
  }

  async upsert(data: Record<string, Record<string, any>>): Promise<void> {
    if (Object.keys(data).length === 0) return;
    
    const client = await this.pool.connect();
    try {
      await client.query('BEGIN');
      
      // Batch upsert using unnest
      const ids = Object.keys(data);
      const values = ids.map(id => data[id]);
      
      await client.query(`
        INSERT INTO ${this.tableName} (id, data, updated_at)
        SELECT * FROM unnest($1::text[], $2::jsonb[])
        ON CONFLICT (id) 
        DO UPDATE SET 
          data = EXCLUDED.data,
          updated_at = CURRENT_TIMESTAMP
      `, [ids, values]);
      
      await client.query('COMMIT');
      
      logger.debug(`Upserted ${ids.length} entries to ${this.tableName}`);
    } catch (error) {
      await client.query('ROLLBACK');
      logger.error({ error }, 'Failed to upsert data');
      throw new StorageError('Failed to upsert data', { cause: error });
    } finally {
      client.release();
    }
  }

  async delete(ids: string[]): Promise<void> {
    if (ids.length === 0) return;
    
    try {
      await this.pool.query(
        `DELETE FROM ${this.tableName} WHERE id = ANY($1)`,
        [ids]
      );
      
      logger.debug(`Deleted ${ids.length} entries from ${this.tableName}`);
    } catch (error) {
      logger.error({ error, ids }, 'Failed to delete');
      throw new StorageError('Failed to delete data', { cause: error });
    }
  }

  async index_done_callback(): Promise<void> {
    // For PostgreSQL, operations are immediately persisted
    // This callback is primarily for in-memory storages
    logger.debug(`Index done callback for ${this.tableName}`);
  }

  async drop(): Promise<{ status: string; message: string }> {
    try {
      await this.pool.query(`DROP TABLE IF EXISTS ${this.tableName}`);
      return { status: 'success', message: 'data dropped' };
    } catch (error) {
      logger.error({ error }, 'Failed to drop table');
      return { status: 'error', message: String(error) };
    }
  }

  async finalize(): Promise<void> {
    await this.pool.end();
    logger.info(`PostgresKVStorage finalized: ${this.tableName}`);
  }
}
```

### Vector Storage with pgvector

Example showing vector similarity search with PostgreSQL's pgvector extension.

```typescript
import { Pool } from 'pg';
import pgvector from 'pgvector/pg';
import { BaseVectorStorage } from '../base/BaseVectorStorage';

export class PostgresVectorStorage extends BaseVectorStorage {
  private pool: Pool;
  private tableName: string;
  private dimension: number = 1536; // OpenAI text-embedding-3-small

  async initialize(): Promise<void> {
    const client = await this.pool.connect();
    try {
      // Register pgvector type
      await pgvector.registerType(client);
      
      // Create extension
      await client.query('CREATE EXTENSION IF NOT EXISTS vector');
      
      // Create table
      await client.query(`
        CREATE TABLE IF NOT EXISTS ${this.tableName} (
          id TEXT PRIMARY KEY,
          vector vector(${this.dimension}),
          metadata JSONB,
          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
      `);
      
      // Create index for vector similarity search
      await client.query(`
        CREATE INDEX IF NOT EXISTS idx_${this.tableName}_vector
        ON ${this.tableName} 
        USING ivfflat (vector vector_cosine_ops)
        WITH (lists = 100)
      `);
    } finally {
      client.release();
    }
  }

  async query(
    query: string,
    top_k: number,
    query_embedding?: number[]
  ): Promise<Array<Record<string, any>>> {
    // Generate embedding if not provided
    const embedding = query_embedding || 
      (await this.embedding_func([query]))[0];
    
    try {
      // Cosine similarity search
      const result = await this.pool.query(`
        SELECT 
          id,
          metadata,
          1 - (vector <=> $1::vector) AS similarity
        FROM ${this.tableName}
        WHERE 1 - (vector <=> $1::vector) > $2
        ORDER BY vector <=> $1::vector
        LIMIT $3
      `, [
        pgvector.toSql(embedding),
        this.cosine_better_than_threshold,
        top_k
      ]);
      
      return result.rows.map(row => ({
        id: row.id,
        ...row.metadata,
        similarity: row.similarity,
      }));
    } catch (error) {
      logger.error({ error }, 'Vector query failed');
      throw new StorageError('Vector query failed', { cause: error });
    }
  }

  async upsert(data: Record<string, Record<string, any>>): Promise<void> {
    const entries = Object.entries(data);
    if (entries.length === 0) return;
    
    // Generate embeddings for new entries
    const textsToEmbed: string[] = [];
    const idsToEmbed: string[] = [];
    
    for (const [id, entry] of entries) {
      if (!entry.vector && entry.content) {
        textsToEmbed.push(entry.content);
        idsToEmbed.push(id);
      }
    }
    
    let embeddings: number[][] = [];
    if (textsToEmbed.length > 0) {
      embeddings = await this.embedding_func(textsToEmbed);
    }
    
    // Map embeddings back to entries
    idsToEmbed.forEach((id, idx) => {
      data[id].vector = embeddings[idx];
    });
    
    const client = await this.pool.connect();
    try {
      await client.query('BEGIN');
      
      for (const [id, entry] of entries) {
        const { vector, ...metadata } = entry;
        
        await client.query(`
          INSERT INTO ${this.tableName} (id, vector, metadata)
          VALUES ($1, $2, $3)
          ON CONFLICT (id)
          DO UPDATE SET
            vector = EXCLUDED.vector,
            metadata = EXCLUDED.metadata
        `, [
          id,
          pgvector.toSql(vector),
          JSON.stringify(metadata)
        ]);
      }
      
      await client.query('COMMIT');
    } catch (error) {
      await client.query('ROLLBACK');
      throw new StorageError('Upsert failed', { cause: error });
    } finally {
      client.release();
    }
  }
}
```

### Graph Storage with Graphology

NetworkX alternative using graphology for in-memory graph operations.

```typescript
import Graph from 'graphology';
import { Attributes } from 'graphology-types';
import { BaseGraphStorage } from '../base/BaseGraphStorage';
import fs from 'fs/promises';
import path from 'path';

export class GraphologyStorage extends BaseGraphStorage {
  private graph: Graph;
  private filePath: string;

  constructor(config: StorageConfig) {
    super(config.namespace, config.workspace, config.global_config);
    this.graph = new Graph({ type: 'undirected' });
    this.filePath = path.join(
      config.working_dir,
      config.workspace || '',
      `${config.namespace}_graph.json`
    );
  }

  async initialize(): Promise<void> {
    // Load existing graph from disk if available
    try {
      const data = await fs.readFile(this.filePath, 'utf-8');
      const serialized = JSON.parse(data);
      this.graph.import(serialized);
      logger.info(`Loaded graph from ${this.filePath}`);
    } catch (error) {
      // File doesn't exist, start with empty graph
      logger.info(`Starting with empty graph at ${this.filePath}`);
    }
  }

  async has_node(node_id: string): Promise<boolean> {
    return this.graph.hasNode(node_id);
  }

  async has_edge(source_node_id: string, target_node_id: string): Promise<boolean> {
    return this.graph.hasEdge(source_node_id, target_node_id);
  }

  async node_degree(node_id: string): Promise<number> {
    if (!this.graph.hasNode(node_id)) return 0;
    return this.graph.degree(node_id);
  }

  async edge_degree(src_id: string, tgt_id: string): Promise<number> {
    const srcDegree = await this.node_degree(src_id);
    const tgtDegree = await this.node_degree(tgt_id);
    return srcDegree + tgtDegree;
  }

  async get_node(node_id: string): Promise<Record<string, string> | null> {
    if (!this.graph.hasNode(node_id)) return null;
    return this.graph.getNodeAttributes(node_id);
  }

  async get_edge(
    source_node_id: string,
    target_node_id: string
  ): Promise<Record<string, string> | null> {
    if (!this.graph.hasEdge(source_node_id, target_node_id)) return null;
    return this.graph.getEdgeAttributes(source_node_id, target_node_id);
  }

  async get_node_edges(source_node_id: string): Promise<Array<[string, string]> | null> {
    if (!this.graph.hasNode(source_node_id)) return null;
    
    const edges: Array<[string, string]> = [];
    this.graph.forEachEdge(source_node_id, (edge, attributes, source, target) => {
      edges.push([source, target]);
    });
    
    return edges;
  }

  async upsert_node(node_id: string, node_data: Record<string, string>): Promise<void> {
    if (this.graph.hasNode(node_id)) {
      this.graph.mergeNodeAttributes(node_id, node_data);
    } else {
      this.graph.addNode(node_id, node_data);
    }
  }

  async upsert_edge(
    source_node_id: string,
    target_node_id: string,
    edge_data: Record<string, string>
  ): Promise<void> {
    // Ensure nodes exist
    if (!this.graph.hasNode(source_node_id)) {
      this.graph.addNode(source_node_id, {});
    }
    if (!this.graph.hasNode(target_node_id)) {
      this.graph.addNode(target_node_id, {});
    }
    
    // Add or update edge (undirected)
    if (this.graph.hasEdge(source_node_id, target_node_id)) {
      this.graph.mergeEdgeAttributes(source_node_id, target_node_id, edge_data);
    } else {
      this.graph.addEdge(source_node_id, target_node_id, edge_data);
    }
  }

  async get_all_labels(): Promise<string[]> {
    return this.graph.nodes();
  }

  async get_knowledge_graph(
    node_label: string,
    max_depth: number = 3,
    max_nodes: number = 1000
  ): Promise<KnowledgeGraph> {
    const nodes: KnowledgeGraphNode[] = [];
    const edges: KnowledgeGraphEdge[] = [];
    const visited = new Set<string>();
    const queue: Array<{ id: string; depth: number }> = [{ id: node_label, depth: 0 }];
    
    while (queue.length > 0 && visited.size < max_nodes) {
      const { id, depth } = queue.shift()!;
      
      if (visited.has(id) || depth > max_depth) continue;
      visited.add(id);
      
      // Add node
      const nodeAttrs = this.graph.getNodeAttributes(id);
      nodes.push({
        id,
        labels: [nodeAttrs.entity_type || 'Entity'],
        properties: nodeAttrs,
      });
      
      // Add edges and queue neighbors
      if (depth < max_depth) {
        this.graph.forEachEdge(id, (edge, edgeAttrs, source, target) => {
          const neighbor = source === id ? target : source;
          
          if (!visited.has(neighbor)) {
            queue.push({ id: neighbor, depth: depth + 1 });
          }
          
          // Add edge if both nodes will be included
          if (visited.has(neighbor) || visited.size < max_nodes) {
            edges.push({
              id: edge,
              source,
              target,
              type: edgeAttrs.keywords || 'related',
              properties: edgeAttrs,
            });
          }
        });
      }
    }
    
    return {
      nodes,
      edges,
      is_truncated: visited.size >= max_nodes,
    };
  }

  async index_done_callback(): Promise<void> {
    // Persist graph to disk
    try {
      const serialized = this.graph.export();
      await fs.mkdir(path.dirname(this.filePath), { recursive: true });
      await fs.writeFile(
        this.filePath,
        JSON.stringify(serialized, null, 2),
        'utf-8'
      );
      logger.info(`Persisted graph to ${this.filePath}`);
    } catch (error) {
      logger.error({ error }, 'Failed to persist graph');
      throw new StorageError('Failed to persist graph', { cause: error });
    }
  }

  async drop(): Promise<{ status: string; message: string }> {
    try {
      this.graph.clear();
      await fs.unlink(this.filePath);
      return { status: 'success', message: 'data dropped' };
    } catch (error) {
      return { status: 'error', message: String(error) };
    }
  }
}
```

## LLM Integration Implementation

### OpenAI Provider with Retry Logic

```typescript
import OpenAI from 'openai';
import pRetry from 'p-retry';
import { LLMProvider } from '../base/LLMProvider';

export class OpenAIProvider implements LLMProvider {
  name = 'openai';
  private client: OpenAI;
  private defaultModel: string;

  constructor(apiKey: string, model: string = 'gpt-4o-mini') {
    this.client = new OpenAI({
      apiKey,
      maxRetries: 0, // We handle retries ourselves
      timeout: 180000, // 3 minutes
    });
    this.defaultModel = model;
  }

  async chat(
    messages: ConversationMessage[],
    options: LLMOptions = {}
  ): Promise<string> {
    return pRetry(
      async () => {
        const completion = await this.client.chat.completions.create({
          model: options.model || this.defaultModel,
          messages,
          temperature: options.temperature ?? 0.7,
          max_tokens: options.max_tokens,
          top_p: options.top_p,
          stop: options.stop,
        });
        
        const content = completion.choices[0].message.content;
        if (!content) {
          throw new Error('Empty response from LLM');
        }
        
        return content;
      },
      {
        retries: 3,
        onFailedAttempt: (error) => {
          logger.warn({
            attempt: error.attemptNumber,
            retriesLeft: error.retriesLeft,
            error: error.message,
          }, 'LLM call failed, retrying...');
        },
        shouldRetry: (error) => {
          // Retry on rate limits and temporary errors
          return error instanceof OpenAI.APIError && 
                 (error.status === 429 || error.status >= 500);
        },
      }
    );
  }

  async *streamChat(
    messages: ConversationMessage[],
    options: LLMOptions = {}
  ): AsyncIterableIterator<string> {
    const stream = await this.client.chat.completions.create({
      model: options.model || this.defaultModel,
      messages,
      stream: true,
      temperature: options.temperature ?? 0.7,
      max_tokens: options.max_tokens,
    });

    for await (const chunk of stream) {
      const content = chunk.choices[0]?.delta?.content;
      if (content) yield content;
    }
  }

  async generateEmbeddings(
    texts: string[],
    batchSize: number = 100
  ): Promise<number[][]> {
    const allEmbeddings: number[][] = [];
    
    // Process in batches
    for (let i = 0; i < texts.length; i += batchSize) {
      const batch = texts.slice(i, i + batchSize);
      
      const response = await pRetry(
        () => this.client.embeddings.create({
          model: 'text-embedding-3-small',
          input: batch,
        }),
        { retries: 3 }
      );
      
      allEmbeddings.push(...response.data.map(item => item.embedding));
    }
    
    return allEmbeddings;
  }
}
```

## Document Processing Pipeline

### Chunking Implementation

```typescript
import { Tiktoken, encoding_for_model } from '@dqbd/tiktoken';

export class TextChunker {
  private encoder: Tiktoken;
  
  constructor(modelName: string = 'gpt-4o-mini') {
    this.encoder = encoding_for_model(modelName);
  }

  chunk(
    content: string,
    chunkSize: number = 1200,
    overlap: number = 100
  ): TextChunkSchema[] {
    const tokens = this.encoder.encode(content);
    const chunks: TextChunkSchema[] = [];
    
    let chunkIndex = 0;
    for (let i = 0; i < tokens.length; i += chunkSize - overlap) {
      const chunkTokens = tokens.slice(i, i + chunkSize);
      const chunkContent = this.encoder.decode(chunkTokens);
      
      chunks.push({
        tokens: chunkTokens.length,
        content: chunkContent,
        full_doc_id: '', // Set by caller
        chunk_order_index: chunkIndex++,
      });
    }
    
    return chunks;
  }

  dispose() {
    this.encoder.free();
  }
}
```

### Entity Extraction

```typescript
import { LLMProvider } from '@/llm/base/LLMProvider';
import { EXTRACTION_PROMPT } from '@/prompts/extraction';

export async function extractEntities(
  chunk: string,
  llmProvider: LLMProvider
): Promise<{ entities: EntityData[]; relationships: RelationshipData[] }> {
  const prompt = EXTRACTION_PROMPT
    .replace('{input_text}', chunk)
    .replace('{entity_types}', DEFAULT_ENTITY_TYPES.join(', '))
    .replace('{language}', 'English');

  const response = await llmProvider.chat([
    { role: 'system', content: prompt },
    { role: 'user', content: 'Extract entities and relationships.' },
  ]);

  return parseExtractionResponse(response);
}

function parseExtractionResponse(response: string): {
  entities: EntityData[];
  relationships: RelationshipData[];
} {
  const entities: EntityData[] = [];
  const relationships: RelationshipData[] = [];
  
  const lines = response.split('\n');
  
  for (const line of lines) {
    if (line.startsWith('entity<|#|>')) {
      const parts = line.split('<|#|>');
      if (parts.length >= 4) {
        entities.push({
          entity_name: parts[1].trim(),
          entity_type: parts[2].trim(),
          description: parts[3].trim(),
          source_id: '',
          file_path: '',
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        });
      }
    } else if (line.startsWith('relation<|#|>')) {
      const parts = line.split('<|#|>');
      if (parts.length >= 5) {
        relationships.push({
          src_id: parts[1].trim(),
          tgt_id: parts[2].trim(),
          keywords: parts[3].trim(),
          description: parts[4].trim(),
          weight: 1.0,
          source_id: '',
          file_path: '',
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        });
      }
    }
  }
  
  return { entities, relationships };
}
```

## Error Handling and Resilience

### Custom Error Classes

```typescript
export class LightRAGError extends Error {
  constructor(message: string, public cause?: unknown) {
    super(message);
    this.name = 'LightRAGError';
  }
}

export class StorageError extends LightRAGError {
  constructor(message: string, cause?: unknown) {
    super(message, cause);
    this.name = 'StorageError';
  }
}

export class LLMError extends LightRAGError {
  constructor(message: string, cause?: unknown) {
    super(message, cause);
    this.name = 'LLMError';
  }
}

export class ValidationError extends LightRAGError {
  constructor(message: string, cause?: unknown) {
    super(message, cause);
    this.name = 'ValidationError';
  }
}
```

### Circuit Breaker Pattern

```typescript
import { CircuitBreaker } from './circuitBreaker';

export class ResilientLLMProvider {
  private provider: LLMProvider;
  private breaker: CircuitBreaker;

  constructor(provider: LLMProvider) {
    this.provider = provider;
    this.breaker = new CircuitBreaker({
      failureThreshold: 5,
      resetTimeout: 60000, // 1 minute
    });
  }

  async chat(
    messages: ConversationMessage[],
    options?: LLMOptions
  ): Promise<string> {
    return this.breaker.execute(() => 
      this.provider.chat(messages, options)
    );
  }
}

class CircuitBreaker {
  private state: 'CLOSED' | 'OPEN' | 'HALF_OPEN' = 'CLOSED';
  private failureCount = 0;
  private lastFailureTime?: number;

  constructor(private config: {
    failureThreshold: number;
    resetTimeout: number;
  }) {}

  async execute<T>(fn: () => Promise<T>): Promise<T> {
    if (this.state === 'OPEN') {
      const now = Date.now();
      if (now - this.lastFailureTime! > this.config.resetTimeout) {
        this.state = 'HALF_OPEN';
      } else {
        throw new Error('Circuit breaker is OPEN');
      }
    }

    try {
      const result = await fn();
      this.onSuccess();
      return result;
    } catch (error) {
      this.onFailure();
      throw error;
    }
  }

  private onSuccess() {
    this.failureCount = 0;
    this.state = 'CLOSED';
  }

  private onFailure() {
    this.failureCount++;
    this.lastFailureTime = Date.now();
    
    if (this.failureCount >= this.config.failureThreshold) {
      this.state = 'OPEN';
    }
  }
}
```

## Performance Optimization

### Connection Pooling

```typescript
export class ConnectionPoolManager {
  private pools = new Map<string, Pool>();

  getPool(config: DatabaseConfig): Pool {
    const key = `${config.host}:${config.port}/${config.database}`;
    
    if (!this.pools.has(key)) {
      this.pools.set(key, new Pool({
        ...config,
        max: 20,
        idleTimeoutMillis: 30000,
        connectionTimeoutMillis: 2000,
      }));
    }
    
    return this.pools.get(key)!;
  }

  async closeAll(): Promise<void> {
    await Promise.all(
      Array.from(this.pools.values()).map(pool => pool.end())
    );
    this.pools.clear();
  }
}
```

### Batch Processing with Concurrency Control

```typescript
import pLimit from 'p-limit';

export async function processBatch<T, R>(
  items: T[],
  processor: (item: T) => Promise<R>,
  concurrency: number = 4
): Promise<R[]> {
  const limit = pLimit(concurrency);
  
  return Promise.all(
    items.map(item => limit(() => processor(item)))
  );
}

// Usage in document processing
async function processDocuments(documents: string[]): Promise<string[]> {
  return processBatch(
    documents,
    async (doc) => {
      const docId = computeMD5(doc);
      await extractAndIndex(doc, docId);
      return docId;
    },
    4 // Max 4 concurrent document processing
  );
}
```

## Testing Patterns

### Mock Storage for Testing

```typescript
export class MockKVStorage extends BaseKVStorage {
  private data = new Map<string, Record<string, any>>();

  async get_by_id(id: string): Promise<Record<string, any> | null> {
    return this.data.get(id) || null;
  }

  async upsert(data: Record<string, Record<string, any>>): Promise<void> {
    for (const [id, value] of Object.entries(data)) {
      this.data.set(id, value);
    }
  }

  async delete(ids: string[]): Promise<void> {
    ids.forEach(id => this.data.delete(id));
  }

  clear() {
    this.data.clear();
  }
}
```

### Integration Test Helper

```typescript
export class TestHelper {
  static async setupTestEnvironment(): Promise<LightRAG> {
    const config: LightRAGConfig = {
      working_dir: './test-storage',
      workspace: `test-${Date.now()}`,
      kv_storage: 'MockKVStorage',
      vector_storage: 'MockVectorStorage',
      graph_storage: 'MockGraphStorage',
      llm_provider: 'mock',
    };

    const rag = new LightRAG(config);
    await rag.initialize();
    return rag;
  }

  static async cleanupTestEnvironment(rag: LightRAG): Promise<void> {
    await rag.drop();
    await rag.close();
  }
}

// Usage
describe('Document Processing', () => {
  let rag: LightRAG;

  beforeEach(async () => {
    rag = await TestHelper.setupTestEnvironment();
  });

  afterEach(async () => {
    await TestHelper.cleanupTestEnvironment(rag);
  });

  it('should process document', async () => {
    const docId = await rag.insert('Test document');
    expect(docId).toBeDefined();
  });
});
```

## Summary

This implementation guide provides:

1. **Complete storage implementations** with PostgreSQL, pgvector, and graphology
2. **LLM integration patterns** with retry logic and circuit breakers
3. **Document processing examples** for chunking and extraction
4. **Error handling patterns** with custom error classes
5. **Performance optimization** techniques for production
6. **Testing patterns** for unit and integration tests

These patterns form the foundation for a robust, production-ready TypeScript implementation of LightRAG. Each pattern can be adapted and extended based on specific requirements while maintaining the core architectural principles.
