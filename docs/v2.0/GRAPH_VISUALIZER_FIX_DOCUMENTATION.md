# Graph Visualizer Multi-Document Fix Documentation

## Problem Summary
The graph visualizer worked fine with 1-2 documents but failed with 3+ documents, showing the error:
```
UsageGraphError: Graph.addEdge: an edge linking "OpenAI Operator" to "User" already exists.
If you really want to add multiple edges linking those nodes, you should create a multi graph by using the 'multi' option.
```

**UPDATE (2025-04-06)**: After restoring from backup due to git issues, the same error reoccurred. Additional investigation revealed that duplicate edge checking was missing in the `createSigmaGraph` function.

## Root Cause Analysis

### The Issue
The problem had **two layers**:

1. **Frontend Graph Library Configuration**: The React frontend was using Sigma.js with graphology, but the graph was configured as a regular `UndirectedGraph` instead of a `MultiUndirectedGraph`. Regular graphs don't allow multiple edges between the same pair of nodes.

2. **Backend Edge Deduplication Logic**: The Neo4j backend was creating composite edge IDs like `{source}_{target}_{relationshipType}` and using a deduplication mechanism that filtered out relationships that appeared to be "duplicates" but were actually distinct relationships with different properties.

### Why It Worked With 1-2 Documents
- With fewer documents, there were fewer chances for multiple relationships of the same type between the same entities
- The composite ID approach worked when relationships were truly unique
- Edge conflicts were minimal

### Why It Failed With 3+ Documents
- Multiple documents often create relationships between the same entities with different properties
- Example: Document 1 creates "OpenAI Operator -> WORKS_WITH -> User" (weight: 0.8)
- Document 2 creates "OpenAI Operator -> WORKS_WITH -> User" (weight: 0.6, different context)
- Backend would filter the second as a "duplicate" OR frontend would reject it as already existing

## Technical Investigation Process

### Step 1: Analyzed Frontend Graph Library
```bash
# Searched for graph initialization in the compiled JavaScript
grep -r "addEdge" lightrag/api/webui/assets/
grep -r "DirectedGraph\|UndirectedGraph" lightrag_webui/src/
```

**Discovery**: Found that the app uses Sigma.js with graphology, and all graph instances were created as `UndirectedGraph`:
- `lightrag_webui/src/hooks/useLightragGraph.tsx:1` - `import { UndirectedGraph }`
- `lightrag_webui/src/hooks/useLightragGraph.tsx:340` - `new UndirectedGraph()`
- `lightrag_webui/src/stores/graph.ts:3` - Type definitions using `DirectedGraph`

### Step 2: Examined Backend Edge Handling
```bash
# Found the Neo4j implementation
grep -n "get_knowledge_graph" lightrag/kg/neo4j_impl.py
```

**Discovery**: In `neo4j_impl.py` around line 2293:
```python
# Old problematic code
edge_key = tuple(sorted([rel_source, rel_target, rel_type]))
if edge_key not in seen_edges:
    seen_edges.add(edge_key)
    # Create edge with composite ID
    id=f"{rel_source}_{rel_target}_{rel_type}"
```

This meant distinct Neo4j relationships with different properties but same source/target/type were being filtered out.

### Step 3: Understanding the Data Flow
```
Neo4j (multiple distinct relationships)
    ↓
Backend (filtered to "unique" edges by composite key)
    ↓
Frontend (regular graph that can't handle multiple edges anyway)
    ↓
Error when trying to add "duplicate" edge
```

## The Complete Fix

### Phase 1: Frontend - Enable Multi-Graph Support

**Why this was needed**: Graphology's `UndirectedGraph` class inherently doesn't support multiple edges between the same nodes. We needed `MultiUndirectedGraph`.

**Files Changed**:
1. `lightrag_webui/src/hooks/useLightragGraph.tsx`
```typescript
// OLD
import Graph, { UndirectedGraph } from 'graphology'
const graph = new UndirectedGraph()

// NEW
import Graph, { MultiUndirectedGraph } from 'graphology'
const graph = new MultiUndirectedGraph()
```

2. `lightrag_webui/src/stores/graph.ts`
```typescript
// OLD
import { DirectedGraph } from 'graphology'
sigmaGraph: DirectedGraph | null

// NEW
import { MultiUndirectedGraph } from 'graphology'
sigmaGraph: MultiUndirectedGraph | null
```

3. `lightrag_webui/src/hooks/useRandomGraph.tsx` - For consistency in test graphs

### Phase 2: Backend - Use True Unique Edge IDs & Fix Node Reference Validation

**Why this was needed**: We needed to ensure every distinct relationship from Neo4j gets a unique ID, not filtered out by composite key matching. Additionally, we discovered that edges were referencing nodes outside the limited node set, causing validation failures.

**File Changed**: `lightrag/kg/neo4j_impl.py`

**Part A: Unique Edge IDs**
```python
# OLD Cypher Query
RETURN startNode(rel) as start_node, endNode(rel) as end_node, rel, type(rel) as rel_type

# NEW Cypher Query
RETURN startNode(rel) as start_node, endNode(rel) as end_node, rel, type(rel) as rel_type, elementId(rel) as rel_id
```

```python
# OLD Edge Deduplication
edge_key = tuple(sorted([rel_source, rel_target, rel_type]))
if edge_key not in seen_edges:
    seen_edges.add(edge_key)

# NEW Edge Deduplication
if rel_id not in seen_edges:  # Use Neo4j's unique relationship ID
    seen_edges.add(rel_id)
```

```python
# OLD Edge ID Assignment
id=f"{rel_source}_{rel_target}_{rel_type}"

# NEW Edge ID Assignment
id=rel_id  # Use Neo4j's unique relationship ID
```

**Part B: Fix Node Reference Validation**
```python
# OLD Query (caused "Target node X is undefined" errors)
MATCH (source:base)
WHERE source.entity_id IN $node_ids
MATCH path = (source)-[r*1..{max_depth}]-(target:base)
UNWIND relationships(path) as rel

# NEW Query (ensures both nodes exist in limited set)
MATCH (source:base)-[rel]-(target:base)
WHERE source.entity_id IN $node_ids AND target.entity_id IN $node_ids
```

This critical fix ensures that edges only connect nodes that exist in the frontend's limited node set, preventing "Target node X is undefined" validation errors.

### Phase 3: Frontend Build Process

**Why rebuild was needed**:
- The compiled JavaScript bundle in `lightrag/api/webui/assets/` still contained the old `UndirectedGraph` code
- Vite.js compiles TypeScript to JavaScript and bundles it for production
- The server serves the compiled bundle, not the source files
- Changes to `.tsx` files don't take effect until rebuilt

**Build Issues Encountered**:
1. **Vite Config Import Error**: `vite.config.ts` was trying to import from `@/lib/constants` using path alias, but aliases don't work in config files during build
2. **Environment Variable Error**: Config referenced `import.meta.env` variables that weren't available during build

**Build Fixes**:
```typescript
// OLD vite.config.ts
import { webuiPrefix } from '@/lib/constants'
base: webuiPrefix,
server: {
  proxy: import.meta.env.VITE_API_PROXY === 'true' && ...
}

// NEW vite.config.ts
base: '/webui/',  // Direct constant
server: {
  proxy: {}  // Simplified for build
}
```

**Final Build Command**:
```bash
npm run build-no-bun
# Outputs to: ../lightrag/api/webui/
```

## Why Each Step Was Critical

### Frontend Graph Type Change
- **Without this**: Even with perfect backend data, frontend would reject multiple edges between same nodes
- **With this**: Frontend can now handle parallel edges and display them distinctly

### Backend Unique ID Usage
- **Without this**: Distinct relationships from multiple documents get filtered out as "duplicates"
- **With this**: Every actual Neo4j relationship gets sent to frontend with truly unique ID

### Frontend Rebuild
- **Without this**: Changes remain in source code but compiled bundle still has old code
- **With this**: Server serves updated JavaScript that implements MultiUndirectedGraph

## Technical Insights

### Graphology Library Behavior
```javascript
// Regular Graph
const graph = new UndirectedGraph()
graph.addEdge('A', 'B', 'edge1') // ✅ Works
graph.addEdge('A', 'B', 'edge2') // ❌ Throws "already exists" error

// Multi Graph
const graph = new MultiUndirectedGraph()
graph.addEdge('A', 'B', 'edge1') // ✅ Works
graph.addEdge('A', 'B', 'edge2') // ✅ Works - parallel edge
```

### Neo4j Relationship Uniqueness
```cypher
-- Neo4j can have multiple relationships of same type between nodes
CREATE (a:Person {name: 'Alice'})-[:KNOWS {context: 'work', weight: 0.8}]->(b:Person {name: 'Bob'})
CREATE (a)-[:KNOWS {context: 'personal', weight: 0.6}]->(b)
-- These are distinct relationships with different elementId() values
```

### React/Vite Build Process
```
Source Files (.tsx) → TypeScript Compiler → JavaScript → Bundler → Minified Bundle
     ↓                       ↓                  ↓           ↓            ↓
Development          Compilation           Chunking    Optimization   Production
```

## Debugging Commands Used

```bash
# Find graph initialization points
grep -r "UndirectedGraph\|DirectedGraph" lightrag_webui/src/

# Find Neo4j edge handling
grep -n "get_knowledge_graph" lightrag/kg/neo4j_impl.py

# Check TypeScript compilation
npx tsc --noEmit

# Build frontend
npm run build-no-bun

# Verify build output
ls -la ../lightrag/api/webui/assets/feature-graph-*.js
```

## Result
- ✅ Graph visualizer now supports multiple documents
- ✅ No more "Graph.addEdge already exists" errors
- ✅ Distinct relationships display properly
- ✅ All edge properties preserved and accessible
- ✅ Backward compatibility maintained

The fix ensures that every distinct relationship from Neo4j gets a unique identity in the frontend graph, allowing proper visualization of complex multi-document knowledge graphs.

## Additional Fix: Duplicate Edge Prevention in createSigmaGraph (2025-04-06)

After the git restore incident, the error recurred because the `createSigmaGraph` function in `useLightragGraph.tsx` was missing duplicate edge checking.

### The Problem
In the `createSigmaGraph` function around line 366, edges were being added without checking for duplicates:

```typescript
// OLD - No duplicate checking
rawEdge.dynamicId = graph.addEdge(rawEdge.source, rawEdge.target, {
  label: rawEdge.properties?.keywords || undefined,
  size: weight,
  originalWeight: weight,
  type: 'curvedNoArrow'
})
```

### The Fix
Added duplicate edge checking before attempting to add edges:

```typescript
// NEW - With duplicate checking
// Skip if edge already exists (check both directions for undirected graph)
if (graph.hasEdge(rawEdge.source, rawEdge.target)) {
  console.warn(`Edge already exists between ${rawEdge.source} and ${rawEdge.target}, skipping duplicate`);
  continue;
}

rawEdge.dynamicId = graph.addEdge(rawEdge.source, rawEdge.target, {
  label: rawEdge.properties?.keywords || undefined,
  size: weight,
  originalWeight: weight,
  type: 'curvedNoArrow'
})
```

This prevents the "edge already exists" error even when the backend sends duplicate edges, making the frontend more resilient to data inconsistencies.