# Third-party entity-name normalizers

Install a package exposing a synchronous `normalize(name: str) -> str` callable
under the `lightrag.entity_name_normalizers` entry-point group. For example:

```toml
[project.entry-points."lightrag.entity_name_normalizers"]
acme = "acme_names:normalize"
```

```python
# acme_names.py
def normalize(name: str) -> str:
    return name.lower()
```

Select its registered name in the same environment as LightRAG:

```dotenv
CUSTOM_ENTITY_NAME_NORMALIZER=acme
```

Names must match `[a-z0-9][a-z0-9_-]{0,63}`. Import paths are not configuration
values. Unlike chunker registration functions, each entry point directly exposes
the name-only callable. No entity type or document context is supplied, so entity
names and the relation endpoints referencing them receive the same mapping.
There are no language-specific rules in core.

Selection is process-wide, including SDK instances and graph API/custom-KG
operations. Configure it before constructing `LightRAG`. The Server resolves it
before storage construction, including Gunicorn preload; SDK construction also
validates it. Module-level helpers resolve on first use. Restart all workers after
changing the setting or installing/updating packages. Unset preserves built-in
behavior and does not discover or import providers. Merely installing a package
does not activate it. Only the selected implementation is loaded.

Unknown names, duplicate selected names (with both distribution origins), import
failures, non-callables, async functions and failures of a startup `"LightRAG"`
probe abort startup with the selected name in the error. Startup logs the selected
entry point and distribution version. A probe cannot establish correctness for
every possible input; input-dependent failures fail the document too.

## Key contract

- Pure and deterministic: no network, clock, mutable state, or external decisions.
- Idempotent: `normalize(normalize(x)) == normalize(x)`. Stored names pass through
  normalization again during graph operations and rebuilds after document deletion.
- Return a string within 256 characters and 512 UTF-8 bytes. Each invocation checks
  the type, limits and idempotence. Oversized output raises an error naming the
  plugin; it is never truncated again. Exceptions fail processing rather than
  falling back to another spelling or silently dropping an extraction record.
- An empty string keeps the input name and logs a warning.
- Return canonical identifiers that remain stable under LightRAG's built-in
  normalization as well. Descriptions, types and keywords are not plugin inputs.
- Merge-only, including across plugin/data versions: adding an alias can merge
  names; removing a mapping can split them and requires a reindex. The core cannot
  prove purity or merge-only behavior across versions.

On extraction, the order is built-in normalization, character/byte cap, final
space pass, then plugin. `normalize_entity_name(..., final_space_pass=False)`
deliberately defers the plugin too: callers must cap and finalize the identifier.
The default `normalize_entity_name(name)` applies it at the end for callers that
do not cap, including graph operations and custom KG insertion. Text and JSON
extraction, relation endpoints and cached extraction replay share this policy.

Changing, upgrading or removing a plugin on an existing index is a normalizer
change. It does not migrate stored names or automatically reprocess documents.
Review [Entity name normalization changes](design/PurgeRecoveryContract.md#entity-name-normalization-changes)
and reindex when a change can split previously equal names. Keep package and alias
data versions reproducible across every worker serving an index. Startup identity
logging is diagnostic; there is no persisted identity gate or automatic migration.
