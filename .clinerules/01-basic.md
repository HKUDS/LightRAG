# LightRAG Project Intelligence (.clinerules)

## Project Overview
LightRAG is a mature, production-ready Retrieval-Augmented Generation (RAG) system with comprehensive knowledge graph capabilities. The system has evolved from experimental to production-ready status with extensive functionality across all major components.

## Current System State (August 15, 2025)
- **Status**: Production Ready - Stable and Mature
- **Configuration**: Gemini 2.5 Flash + BAAI/bge-m3 embeddings via custom endpoints
- **Storage**: Default in-memory with file persistence (JsonKVStorage, NetworkXStorage, NanoVectorDBStorage)
- **Language**: Chinese for summaries
- **Workspace**: `space1` for data isolation
- **Authentication**: JWT-based with admin/user accounts

## Critical Implementation Patterns

### 1. Embedding Format Compatibility (CRITICAL)
**Pattern**: Always handle both base64 and raw array embedding formats
**Location**: `lightrag/llm/openai.py` - `openai_embed` function
**Issue**: Custom OpenAI-compatible endpoints return embeddings as raw arrays, not base64 strings
**Solution**:
```python
np.array(dp.embedding, dtype=np.float32) if isinstance(dp.embedding, list)
else np.frombuffer(base64.b64decode(dp.embedding), dtype=np.float32)
```
**Impact**: Document processing fails completely without this dual format support

### 2. Async Pattern Consistency (CRITICAL)
**Pattern**: Always await coroutines before calling methods on the result
**Common Error**: `coroutine.method()` instead of `(await coroutine).method()`
**Locations**: MongoDB implementations, Neo4j operations
**Example**: `await self._data.list_indexes()` then `await cursor.to_list()`

### 3. Storage Layer Data Compatibility (CRITICAL)
**Pattern**: Always filter deprecated/incompatible fields during deserialization
**Common Fields to Remove**: `content`, `_id` (MongoDB), database-specific fields
**Implementation**: `data.pop('field_name', None)` before creating dataclass objects
**Locations**: All storage implementations (JSON, Redis, MongoDB, PostgreSQL)

### 4. Lock Key Generation (CRITICAL)
**Pattern**: Always sort relationship pairs for consistent lock keys
**Implementation**: `sorted_key_parts = sorted([src, tgt])` then `f"{sorted_key_parts[0]}-{sorted_key_parts[1]}"`
**Impact**: Prevents deadlocks in concurrent relationship processing

### 5. Event Loop Management (CRITICAL)
**Pattern**: Handle event loop mismatches during shutdown gracefully
**Implementation**: Timeout + specific RuntimeError handling for "attached to a different loop"
**Location**: Neo4j storage finalization
**Impact**: Prevents application shutdown failures

### 6. Async Generator Lock Management (CRITICAL)
**Pattern**: Never hold locks across async generator yields - create snapshots instead
**Issue**: Holding locks while yielding causes deadlock when consumers need the same lock
**Location**: `lightrag/tools/migrate_llm_cache.py` - `stream_default_caches_json`
**Solution**: Create snapshot of data while holding lock, release lock, then iterate over snapshot
```python
# WRONG - Deadlock prone:
async with storage._storage_lock:
    for key, value in storage._data.items():
        batch[key] = value
        if len(batch) >= batch_size:
            yield batch  # Lock still held!

# CORRECT - Snapshot approach:
async with storage._storage_lock:
    matching_items = [(k, v) for k, v in storage._data.items() if condition]
# Lock released here
for key, value in matching_items:
    batch[key] = value
    if len(batch) >= batch_size:
        yield batch  # No lock held
```
**Impact**: Prevents deadlocks in Json→Json migrations and similar scenarios where source/target share locks
**Applicable To**: Any async generator that needs to access shared resources while yielding

## Architecture Patterns

### 1. Dependency Injection
**Pattern**: Pass configuration through object constructors, not direct imports
**Example**: OllamaAPI receives configuration through LightRAG object
**Benefit**: Better testability and modularity

### 2. Memory Bank Documentation
**Pattern**: Maintain comprehensive memory bank for development continuity
**Structure**: Core files (projectbrief.md, activeContext.md, progress.md, etc.)
**Purpose**: Essential for context preservation across development sessions

### 3. Configuration Management
**Pattern**: Centralize defaults in constants.py, use environment variables for runtime config
**Implementation**: Default values in constants, override via .env file
**Benefit**: Consistent configuration across components

## Development Workflow Patterns

### 1. Frontend Development (CRITICAL)
**Package Manager**: **ALWAYS USE BUN** - Never use npm or yarn unless Bun is unavailable
**Commands**:
- `bun install` - Install dependencies
- `bun run dev` - Start development server
- `bun run build` - Build for production
- `bun run lint` - Run linting
- `bun test` - Run tests
- `bun run preview` - Preview production build

**Pattern**: All frontend operations must use Bun commands
**Fallback**: Only use npm/yarn if Bun installation fails
**Testing**: Use `bun test` for all frontend testing

### 2. Bug Fix Approach
1. **Identify root cause** - Don't just fix symptoms
2. **Implement robust solution** - Handle edge cases and format variations
3. **Maintain backward compatibility** - Preserve existing functionality
4. **Add comprehensive error handling** - Graceful degradation
5. **Document the fix** - Update memory bank with technical details

### 3. Feature Implementation
1. **Follow existing patterns** - Maintain architectural consistency
2. **Use dependency injection** - Avoid direct imports between modules
3. **Implement comprehensive error handling** - Handle all failure modes
4. **Add proper logging** - Debug and warning messages
5. **Update documentation** - Memory bank and code comments
6. **Comment Language** - Use English for comments and documentation

### 4. Performance Optimization
1. **Profile before optimizing** - Identify actual bottlenecks
2. **Maintain algorithmic correctness** - Don't sacrifice functionality for speed
3. **Use appropriate data structures** - Match structure to access patterns
4. **Implement caching strategically** - Cache expensive operations
5. **Monitor memory usage** - Prevent memory leaks

## Technology Stack Intelligence

### 1. LLM Integration
- **Primary**: Gemini 2.5 Flash via custom endpoint
- **Embedding**: BAAI/bge-m3 via custom endpoint
- **Reranking**: BAAI/bge-reranker-v2-m3
- **Pattern**: Always handle multiple provider formats

### 2. Storage Backends
- **Default**: In-memory with file persistence
- **Production Options**: PostgreSQL, MongoDB, Redis, Neo4j
- **Pattern**: Abstract storage interface with multiple implementations

### 3. API Architecture
- **Framework**: FastAPI with Gunicorn for production
- **Authentication**: JWT-based with role support
- **Compatibility**: Ollama-compatible endpoints for easy integration

### 4. Frontend
- **Framework**: React with TypeScript
- **Package Manager**: **BUN (REQUIRED)** - Always use Bun for all frontend operations
- **Build Tool**: Vite with Bun runtime
- **Visualization**: Sigma.js for graph rendering
- **State Management**: React hooks with context
- **Internationalization**: i18next for multi-language support

## Common Pitfalls and Solutions

### 1. Embedding Format Issues
**Pitfall**: Assuming all endpoints return base64-encoded embeddings
**Solution**: Always check format and handle both base64 and raw arrays

### 2. Async/Await Patterns
**Pitfall**: Calling methods on coroutines instead of awaited results
**Solution**: Always await coroutines before accessing their methods

### 3. Data Model Evolution
**Pitfall**: Breaking changes when removing fields from dataclasses
**Solution**: Filter deprecated fields during deserialization, don't break storage

### 4. Concurrency Issues
**Pitfall**: Inconsistent lock key generation causing deadlocks
**Solution**: Always sort keys for deterministic lock ordering

### 5. Event Loop Management
**Pitfall**: Event loop mismatches during shutdown
**Solution**: Implement timeout and specific error handling for loop issues

## Performance Considerations

### 1. Query Context Building
- **Algorithm**: Linear gradient weighted polling for fair resource allocation
- **Optimization**: Round-robin merging to eliminate mode bias
- **Pattern**: Smart chunk selection based on cross-entity occurrence

### 2. Graph Operations
- **Optimization**: Batch operations where possible
- **Pattern**: Use appropriate indexing for large datasets
- **Consideration**: Memory usage with large graphs

### 3. LLM Request Management
- **Pattern**: Priority-based queue for request ordering
- **Optimization**: Connection pooling and retry mechanisms
- **Consideration**: Rate limiting and cost management

## Security Patterns

### 1. Authentication
- **Implementation**: JWT tokens with role-based access
- **Pattern**: Stateless authentication with configurable expiration
- **Security**: Proper token validation and refresh mechanisms

### 2. API Security
- **Pattern**: Input validation and sanitization
- **Implementation**: FastAPI dependency injection for auth
- **Consideration**: Rate limiting and abuse prevention

## Maintenance Guidelines

### 1. Memory Bank Updates
- **Trigger**: After significant changes or bug fixes
- **Pattern**: Update activeContext.md and progress.md
- **Purpose**: Maintain development continuity

### 2. Configuration Management
- **Pattern**: Environment-based configuration with sensible defaults
- **Implementation**: .env files with example templates
- **Consideration**: Security for production deployments

### 3. Error Handling
- **Pattern**: Comprehensive logging with appropriate levels
- **Implementation**: Graceful degradation where possible
- **Consideration**: User-friendly error messages

## Project Evolution Notes

The project has evolved from experimental to production-ready status. Key milestones:
- **Early 2025**: Basic RAG implementation
- **Mid 2025**: Multiple storage backends and LLM providers
- **July 2025**: Major query optimization and algorithm improvements
- **August 2025**: Production-ready stable state

The system now supports enterprise-level deployments with comprehensive functionality across all components.
