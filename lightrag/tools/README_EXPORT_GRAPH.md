# Streaming Graph Export Tool

`lightrag-export-graph` writes a workspace's knowledge graph (entities and
relations) to files. It reads the graph in batches and writes each batch as
soon as it is read, so memory does not grow with the graph and the number of
storage calls grows with `entities / batch_size`, not with `entities²`.

It reads **graph storage only**. Vectors are not exported (rebuild them with
`lightrag-rebuild-vdb`), and neither are documents, chunks or the LLM cache.

## Usage

```bash
# jsonl (default) into ./graph_export/graph.jsonl
lightrag-export-graph

# CSV for spreadsheets or neo4j-admin import
lightrag-export-graph --format csv --output ./exports/run1

# GraphML for Gephi, yEd or NetworkX, from a named workspace
lightrag-export-graph --format graphml --workspace space1
```

The tool reads the same `.env` as the server: `LIGHTRAG_GRAPH_STORAGE`
selects the backend (or pass `--graph-storage`), its usual connection
variables (`NEO4J_URI`, `POSTGRES_*`, `MONGO_URI`, ...) locate it, and
`WORKSPACE` selects the workspace unless `--workspace` names one exactly
(`--workspace ""` is the default workspace). `NetworkXStorage` reads the
GraphML file under `WORKING_DIR`.

| Option | Default | Meaning |
|---|---|---|
| `--format` | `jsonl` | `jsonl`, `csv` or `graphml` |
| `--output` | `./graph_export` | output directory |
| `--workspace` | `WORKSPACE` | exact workspace name |
| `--graph-storage` | `LIGHTRAG_GRAPH_STORAGE` | graph storage class |
| `--batch-size` | `500` | labels or edges per storage call |
| `--overwrite` | off | replace existing output files |

The summary is printed as JSON on stdout. Exit code `0` means the files were
written; `1` means the graph storage could not be initialized; `2` means the
export was refused (existing output, or a backend without batch iteration).

## Formats

### `jsonl`: `graph.jsonl`

One JSON object per line. The first line is a header, then one line per node,
then one per edge. Every stored property is kept, so this is the format meant
for backups and a later re-import.

```json
{"type": "header", "format": "lightrag-graph", "version": 1, "graph_storage": "Neo4JStorage", "workspace": "base", "exported_at": "2026-11-27T10:00:00+00:00"}
{"type": "node", "id": "Alice", "properties": {"entity_id": "Alice", "entity_type": "person", "description": "...", "source_id": "chunk-1", "file_path": "a.txt", "created_at": 1764000000}}
{"type": "edge", "source": "Alice", "target": "Bob", "properties": {"weight": 1.0, "description": "...", "keywords": "...", "source_id": "chunk-1", "file_path": "a.txt", "created_at": 1764000000}}
```

A consumer must check `format` and `version` before reading the rest.

### `csv`: `nodes.csv` and `edges.csv`

Fixed columns, then `extra`:

- `nodes.csv`: `entity_id, entity_type, description, source_id, file_path, created_at, extra`
- `edges.csv`: `source, target, weight, description, keywords, source_id, file_path, created_at, extra`

`extra` holds every other stored property as a JSON object (empty when there
are none). A stored `entity_id` property that differs from the node's label is
kept in `extra`. Non-string values in the fixed columns are written as text;
use `jsonl` when types matter.

### `graphml`: `graph.graphml`

An undirected GraphML graph with the same fixed attributes as the CSV columns
plus `extra`, written in one pass. The header object is stored in the graph's
`<desc>`. `weight` is typed `double`; a stored weight that is not a number
moves to `extra`. XML 1.0 cannot carry some control characters (for example
U+0007); they are replaced with U+FFFD, and the summary counts them under
`graphml_replaced_xml_chars`. Use `jsonl` for a lossless copy.

## How it reads

1. `iter_labels(batch_size)` is drained into a private temporary SQLite file
   (`tempfile.gettempdir()`, removed when the run ends), and its cursor is
   closed.
2. For each batch of spooled labels, `get_nodes_batch(labels)` reads the
   node properties.
3. `iter_edges(batch_size)` streams the edges, which carry their properties.

Spooling the labels first matters on pooled backends: reading nodes while the
label cursor is still open needs a second connection, and with a pool of one
(`NEO4J_MAX_CONNECTION_POOL_SIZE=1`) that read times out. The spool holds only
entity names and stays on disk.

A backend that does not implement `iter_labels` / `iter_edges` is refused.
The tool never falls back to `get_all_nodes()` / `get_all_edges()`, which
load the whole graph. All built-in graph backends implement both.

Initializing a storage is not always read-only: some backends create their
indexes or schema on first connect, as they do when the server starts. The
export itself issues reads only.

## Consistency

Nodes and edges are read in separate passes with no snapshot across them.
**Stop writers (the server's pipeline, admin edits) for a consistent export.**
If the graph changes during the run, the export still completes, and the
summary sets `"consistent": false` and reports:

| Field | Meaning |
|---|---|
| `vanished_nodes` | listed by `iter_labels` but gone when its details were read |
| `dangling_edges` | an edge whose endpoint is not among the exported nodes |

Up to 20 examples of each are listed. Two other counts describe the stored
data rather than a concurrent change, and do not make an export inconsistent:

| Field | Meaning |
|---|---|
| `duplicate_labels_skipped` | a label the backend listed more than once; exported once |
| `duplicate_edge_pairs_kept` | a second row for the same undirected pair (`A->B` beside `B->A`); written as stored |

## Output files

Files are written under a `.partial` name and renamed only when the whole
export succeeded, so an interrupted or failed run never leaves a file that
looks complete and never replaces an earlier export. An existing output file
is refused unless `--overwrite` is given.

## Memory

The tool holds one batch at a time. On Neo4j and Memgraph the Python heap
during the node pass can still climb to a few MB before it drops: the driver's
node objects form reference cycles that only the cyclic garbage collector
frees, so the peak depends on when it runs rather than on `batch_size` alone.
Measured on a 21,256-entity, 24,733-relation Neo4j graph: the label and edge
passes stay flat below 1 MB; the node pass peaks at 8 to 9 MB.

## Compared with `aexport_data()`

`lightrag.utils.aexport_data()` (and `LightRAG.export_data`) builds the whole
export in memory and finds relations by probing `has_edge` for every pair of
entities: 225.9 million calls for 21,256 entities, about 1.6 days on a local
Neo4j. This tool exported the same graph in about 5 seconds, start-up
included, with about 140 storage calls at the default batch size.
