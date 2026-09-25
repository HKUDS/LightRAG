# Offline Storage Clear Tool

`lightrag-clear-storage` drops every data storage of **one** LightRAG
workspace without a running server. It does exactly what the WebUI *Clear*
button (`DELETE /documents`) does, in the same order, and it never touches the
LLM response cache.

## When do I need this?

The server refuses to start a workspace whose vector index is empty,
corrupt, written in another embedding space, or recorded under another
embedding baseline while the graph or `text_chunks` still hold data. Every
such refusal names two ways out: `lightrag-rebuild-vdb`, which re-embeds every
record from the authoritative sources and is the right recovery when the data
matters, and this tool.

When the data does **not** matter — a test corpus, a workspace being
decommissioned, a deployment you only want to wipe and start over — a rebuild
is wasted embedding cost and time, and it is wasted twice: the WebUI's own
*Clear* button lives behind the server that refuses to start. This tool is the
way out: it clears the workspace offline, which also clears the refusal.

Everything below applies to one workspace per run, the one named by
`WORKSPACE` in your `.env` (empty for the default workspace).

## Usage

```bash
# Stop the LightRAG Server first!
lightrag-clear-storage
# or
python -m lightrag.tools.clear_storage
```

The tool takes no command-line options. Server flags such as `--workspace`
or `--input-dir` are refused, not ignored: the embedding function is built
through the server's own argument parser, which would honor them, while the
workspace and directories the tool clears come from the environment — so a
flag could select one embedding configuration and clear another workspace.
Put the values in `.env` instead. A backend named under the wrong storage
category (for example `LIGHTRAG_GRAPH_STORAGE=JsonKVStorage`) is refused
before anything opens, exactly as the server refuses it.

The tool reads the same `.env` / environment configuration as the server
(`LIGHTRAG_KV_STORAGE`, `LIGHTRAG_VECTOR_STORAGE`, `LIGHTRAG_GRAPH_STORAGE`,
`LIGHTRAG_DOC_STATUS_STORAGE`, `LIGHTRAG_CONFIG_STORAGE`, `WORKSPACE`,
`WORKING_DIR`, `INPUT_DIR`, `EMBEDDING_MODEL`, `EMBEDDING_DIM`, backend
connection settings) — through the server's own argument parser, so it sees
exactly what the server sees, its normalization included: the parser rewrites
every character of `WORKSPACE` outside `[A-Za-z0-9_]` to `_` (a server
configured with `WORKSPACE=customer-prod` stores under `customer_prod`, and so
does this tool) and makes the directories absolute. It never embeds anything and needs no reachable
embedding service, but it does need the `api` extra (`pip install
"lightrag-hku[api]"`): the embedding function is built through the server's
own factory, exactly as `lightrag-rebuild-vdb` builds it, because Qdrant,
PostgreSQL and Milvus name their vector containers after the embedding model
and dimension and the server derives an omitted `EMBEDDING_DIM` from the
provider's default. A tool that guessed the dimension would open — and drop —
a container the server never wrote to, then delete the configuration records
while the real vectors survived. Without the `api` extra the tool refuses to
run rather than guess.

The run, in order:

1. Asks whether the server has been shut down.
2. Opens every storage. A file-backed configuration storage claims the
   configuration directory for the run, so a server still holding it makes the
   tool refuse before anything else.
3. Shows what is about to be deleted, then stops for confirmation:
   - the number of documents in each `doc_status` state, and their sum as
     the total;
   - the ten most recently updated documents (updated time, status, file
     path, id) — a listing read, so a page that comes back empty while the
     strict counts say documents exist is read as the backend's swallowed
     failure, not as "no documents";
   - whether `text_chunks` holds any data, and the same for each other KV
     namespace it drops (`full_docs`, `full_entities`, `full_relations`,
     `entity_chunks`, `relation_chunks`) -- these can hold rows while
     doc-status and chunks are empty, after an interrupted write;
   - whether the knowledge graph holds entities, and relations;
   - whether each vector storage is empty or has vectors; one that refused
     to attach (a corrupt snapshot, a container in another vector space) was
     never read and shows as `UNREADABLE`;
   - the recorded embedding baselines;
   - the top-level files of this workspace's input directory
     (`INPUT_DIR/<workspace>`, or `INPUT_DIR` for the default workspace).
4. Requires the exact phrase `Delete All`. Anything else exits with nothing
   changed.
5. Drops the eleven data storages, deletes the workspace's configuration
   records, and deletes the top-level input files.

A value in step 3 that cannot be read is shown as `UNREADABLE`, never as
zero, because zero is exactly what makes an operator clear the wrong workspace.
Whether an unreadable value stops the run depends on the backend's kind — see
*Errors: what stops the run and what does not* below.

The summary uses three colours and nothing else: **yellow** for a storage that
holds something (`has data`, `has vectors`, `has entities`, a non-zero document
count or total, a non-zero input file count), **red** for `UNREADABLE`, and
plain text for everything that holds nothing. Most of the screen is normally
zeros and `EMPTY`, so the yellow lines are exactly the ones worth stopping at
before typing the phrase. `UNREADABLE` deliberately does not share that colour:
unknown is not the same as present.

The text chunk store is only asked whether it holds a row, through the same
strict first-page read the startup gate uses (`iter_rows`, which raises on a
backend failure) — not `BaseKVStorage.is_empty()`, which on the server
backends catches its transport errors and answers "empty", so an outage would
show as an empty store and let the confirmation drop every healthy sibling
around it. An exact row count would take a backend-specific query per store
for a number that changes nothing about whether to confirm. The status counts
are one strict query per status.

The knowledge graph is read the same way, with the gate's own probes:
`graph_has_nodes` (`get_popular_labels(limit=1)`, abstract on
`BaseGraphStorage`, so every backend answers it) and `graph_has_edges`
(`iter_edges`, bounded, and fail-closed on a backend that never implemented
it — the line then reads `has entities` and the relation half is simply not
reported). Both raise on a backend failure, so the kind rule applies to a
Neo4j outage exactly as it does to a Redis one. The one backend that answers
`empty` without raising is OpenSearch, for a *confirmed* missing graph index —
the truth at this point, since `initialize()` has just recreated both of its
indices; a transport failure still raises. The graph is the most
expensive thing the run drops and the one a rebuild treats as authoritative,
so it is shown rather than left to be inferred from the document counts.

## What is deleted, and what is not

Deleted, as `/documents/clear` deletes them:

| Storage | Content |
|---|---|
| `text_chunks`, `full_docs` | chunk and document text |
| `full_entities`, `full_relations`, `entity_chunks`, `relation_chunks` | per-document write-ahead anchors and chunk tracking |
| `entities_vdb`, `relationships_vdb`, `chunks_vdb` | the three vector indexes |
| `chunk_entity_relation_graph` | the knowledge graph |
| `doc_status` | document processing status |
| configuration records | the workspace's three embedding baselines, last, and only when every drop above succeeded |
| input directory | top-level files of this workspace's upload directory only: `INPUT_DIR/<workspace>` for a named workspace, `INPUT_DIR` itself for the default one, exactly as the server resolves it |

Preserved:

- **The LLM response cache.** Extraction results are the expensive part of
  ingestion, and re-adding the same documents reuses them. To clear the cache
  too, start the server after this tool finishes and run *Clear* from the WebUI
  with the *clear LLM cache* option checked.
- **The `__parsed__` directory** under the input directory, and every other
  subdirectory there: pre-parsed artifacts let a re-added file skip parsing.
- **Every other workspace.** Backend-specific `*_WORKSPACE` variables outrank
  `WORKSPACE` inside the storage layer; the summary names every such override
  in effect, and the workspace a storage resolved to where its backend reports
  it, so check them before you type the phrase. The names a backend
  substitutes for the default workspace (`default` on PostgreSQL, `_` on
  Redis, `base` on Neo4j and Memgraph) are not reported as overrides.

## Errors: what stops the run and what does not

The rule is the backend's **kind**, decided from whether it needs a connection
setting (`STORAGE_ENV_REQUIREMENTS`): a backend that does is a server, one
that does not is file-backed. An unknown backend counts as a server.

| Failure | Server backend (PostgreSQL, Redis, Mongo, Milvus, Qdrant, Neo4j, Memgraph, OpenSearch) | File-backed storage (JSON, NetworkX, Nano, Faiss) |
|---|---|---|
| cannot be opened at startup | **refuses the run**, nothing dropped | shown; summary reads `UNREADABLE`; `drop()` still attempted |
| a summary read fails | **refuses the run**, nothing dropped | that value reads `UNREADABLE`; the run goes on |
| a vector storage refuses to attach (foreign embedding space, corrupt Nano/Faiss snapshot) | dropped anyway — both are typed data-level refusals and `drop()` is servable | same |
| the storage lacks a read capability (no key enumeration, no strict count) | that value reads `UNREADABLE`; the run goes on | same |
| the configuration storage cannot be opened or a baseline row cannot be fetched | **refuses the run** | **refuses the run** (the records are deleted last; a store that cannot take that step is not a clean clear) |
| a baseline record is fetched but does not parse — at either depth: the record is not a mapping at all, or it is one whose `value` is not | shown `UNREADABLE`; the record is deleted by key like any other | same |
| `drop()` fails on one storage | the others are still dropped; the configuration records stay; exit non-zero | same |

Both depths of a damaged baseline record parse as damage, and the distinction
that decides it is *fetched* versus *not fetched*, never *readable* versus
*not readable*: the store answered, so it is serving, and
`delete_workspace_configuration` removes the record by key without ever
reading its value. A configuration file someone hand-edited into an invalid
shape is a reason to reach for this tool, not a reason for it to refuse.

The reasoning: a server backend that cannot be reached will not serve the
drop either, and clearing the storages that did answer would leave the
unreachable one populated — a partial clear nobody asked for, and the server's
next start would find surviving data behind deleted records. A corrupt local
file, by contrast, *is* the data the operator is about to delete; refusing to
delete it because it cannot be read would send them to `rm`.

One file-backed storage cannot be dropped through the tool when its file is
corrupt: `NetworkXStorage` parses its GraphML in the constructor, so there is
no instance to call `drop()` on. The tool reports that storage as a failed
drop, keeps the configuration records, exits non-zero, and names the working
directory; remove `graph_chunk_entity_relation.graphml` from the workspace's
directory by hand and re-run.

## Important notes

- **Stop the server first, and know what does not enforce it.** Dropping
  storages under a live pipeline tears them down out from under the writer and
  loses data, on every backend: every `drop()` requires its caller to hold the
  pipeline's destructive reservation, and that reservation lives in one
  process's shared memory, so no separate process can take it. What this tool
  substitutes is partial by design, and the difference is the configuration
  storage. **File-backed** (the default `JsonKVStorage`): the tool claims
  `config_dir` for the run (`lightrag/kg/working_dir_lock.py`), so a server
  still holding it makes the tool refuse before it opens anything.
  **Server-backed** (PostgreSQL, MongoDB, OpenSearch): nothing is claimed —
  `uses_working_dir` is false — and the only thing between a live server and a
  cleared workspace is the "Has the LightRAG Server been shut down?" prompt.
  That is the same posture as `lightrag-rebuild-vdb`, and it is an accepted
  residue rather than an oversight: a cross-process claim would have to be
  taken on something every backend shares, which the storage layer does not
  have. Recovery if it happens anyway: stop every writer, re-run this tool
  (the drops are idempotent), then restart the server.
- **Opening the storages runs the server's one-time migrations, before the
  summary.** The tool initializes every storage exactly as the server does,
  and on Qdrant, PostgreSQL and Milvus that includes migrating a legacy
  (unsuffixed) vector container into the current model-named one when only
  the legacy one exists — a copy that can take a while on a large workspace,
  and that happens before anything is shown or confirmed. This is accepted,
  as `lightrag-rebuild-vdb` accepts the same thing: the migration moves data
  into the very container the clear then drops, loses nothing, and leaves
  exactly the state a server start would have produced; a run cancelled at
  the prompt keeps that state, which is why the cancellation says no
  workspace data was *deleted*. Avoiding it would need a non-mutating attach
  path that no backend offers, or dropping without attaching, which not
  every backend can serve.
- **A partial drop keeps the configuration records.** If any storage fails to
  drop, the workspace's embedding baselines stay on record, the tool exits
  non-zero, and the summary says which drop failed. Fix the cause and re-run;
  the drops are idempotent. Records deleted while data survived would let the
  next start adopt a wrong baseline over it, which is the one residue this
  ordering exists to prevent (*Workspace drop* in
  `docs/design/ConfigurationStorageContract.md`).
- **A vector storage that refuses to attach is dropped anyway.** The two typed
  refusals — vectors written in another embedding space, or a corrupt local
  snapshot — are the states this tool exists to clear, and `drop()` is
  servable while refused. Nothing is backed up first: the operator is typing
  the phrase that deletes it. An outage or a bad credential on a server
  backend refuses the run untouched (see the table above).
- **Named-container backends clear the container the current configuration
  names.** Qdrant, PostgreSQL and Milvus derive the collection / table from the
  embedding model and dimension, resolved through the server's factory. Run
  the tool with the `.env` the server uses; a container named by a previous
  embedding configuration is left orphaned, exactly as a rebuild leaves it.
- **The coverage gate is never consulted.** The tool drives the storages
  directly, like `lightrag-rebuild-vdb`, so the refusal that keeps the server
  from starting does not keep this tool from running.
- **Exit status.** Zero on a clean clear and on a deliberate cancellation
  (either prompt answered with anything but the expected phrase). Non-zero
  when any storage drop, the configuration delete or an input file delete
  failed, or when the workspace could not be read.
