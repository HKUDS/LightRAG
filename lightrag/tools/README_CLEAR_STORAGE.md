# Offline Storage Clear Tool

`lightrag-clear-storage` drops every data storage of **one** LightRAG
workspace without a running server. It does exactly what the WebUI *Clear*
button (`DELETE /documents`) does, in the same order, and it never touches the
LLM response cache.

## When do I need this?

The server refuses to start a workspace whose vector index is empty or
unreadable while the graph or `text_chunks` still hold data. The message names
`lightrag-rebuild-vdb`, which re-embeds every record from the authoritative
sources. That is the right recovery when the data matters.

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

The tool reads the same `.env` / environment configuration as the server
(`LIGHTRAG_KV_STORAGE`, `LIGHTRAG_VECTOR_STORAGE`, `LIGHTRAG_GRAPH_STORAGE`,
`LIGHTRAG_DOC_STATUS_STORAGE`, `LIGHTRAG_CONFIG_STORAGE`, `WORKSPACE`,
`WORKING_DIR`, `INPUT_DIR`, `EMBEDDING_MODEL`, `EMBEDDING_DIM`, backend
connection settings). It never embeds anything, so it does not need the `api`
extra or a reachable embedding service; `EMBEDDING_MODEL` / `EMBEDDING_DIM`
still matter because Qdrant and PostgreSQL name their vector containers after
them (see *Important notes*).

The run, in order:

1. Asks whether the server has been shut down.
2. Opens every storage. A file-backed configuration storage claims the
   configuration directory for the run, so a server still holding it makes the
   tool refuse before anything else.
3. Shows what is about to be deleted, then stops for confirmation:
   - the number of documents in each `doc_status` state, and the total;
   - the ten most recently updated documents (updated time, status, file
     path, id);
   - the number of text chunks in `text_chunks`;
   - whether each vector storage is empty, has vectors, or refused to attach;
   - the recorded embedding baselines;
   - the top-level files of the input directory.
4. Requires the exact phrase `Delete All`. Anything else exits with nothing
   changed.
5. Drops the eleven data storages, deletes the workspace's configuration
   records, and deletes the top-level input files.

Every read in step 3 is fail-loud: a backend that cannot answer aborts the run
with nothing deleted. A count that cannot be read is never shown as zero,
because zero is exactly what makes an operator clear the wrong workspace.

The chunk count enumerates every `text_chunks` key, and the status counts are
one strict query per status, so on a very large workspace the summary takes a
moment. That is the same read `lightrag-rebuild-vdb` performs before a
rebuild, and a small fraction of what the rebuild itself would cost.

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
| input directory | top-level files only |

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
  it, so check them before you type the phrase.

## Important notes

- **Stop the server first.** Dropping storages under a live pipeline tears
  them down out from under the writer and loses data, on every backend.
- **A partial drop keeps the configuration records.** If any storage fails to
  drop, the workspace's embedding baselines stay on record, the tool exits
  non-zero, and the summary says which drop failed. Fix the cause and re-run;
  the drops are idempotent. Records deleted while data survived would let the
  next start adopt a wrong baseline over it, which is the one residue this
  ordering exists to prevent (*Workspace drop* in
  `docs/design/ConfigurationStorage.md`).
- **A vector storage that refuses to attach is dropped anyway.** The two typed
  refusals — vectors written in another embedding space, or a corrupt local
  snapshot — are the states this tool exists to clear, and `drop()` is
  servable while refused. Nothing is backed up first: the operator is typing
  the phrase that deletes it. Any other initialization failure (an outage, a
  bad credential) aborts the run untouched.
- **Named-container backends clear the container the current configuration
  names.** Qdrant and PostgreSQL derive the collection / table from
  `EMBEDDING_MODEL` and `EMBEDDING_DIM`. Run the tool with the `.env` the server
  uses; a container named by a previous embedding configuration is left
  orphaned, exactly as a rebuild leaves it.
- **The coverage gate is never consulted.** The tool drives the storages
  directly, like `lightrag-rebuild-vdb`, so the refusal that keeps the server
  from starting does not keep this tool from running.
- **Exit status.** Zero on a clean clear and on a deliberate cancellation
  (either prompt answered with anything but the expected phrase). Non-zero
  when any storage drop, the configuration delete or an input file delete
  failed, or when the workspace could not be read.
