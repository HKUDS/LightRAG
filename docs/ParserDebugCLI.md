# Parser CLI Debugger Guide

This tool is used to locally debug any parsing engine in LightRAG's registry (the built-in `native` / `legacy` / `mineru` / `docling` / `paddleocr_vl`, plus third-party engines registered via the `lightrag.parsers` entry point — see `docs/ThirdPartyParser.md`). It drives the same registry dispatch path as the pipeline worker (`get_parser(engine).parse(...)`) for a **single file** and outputs the parsing artifacts (sidecar and raw cache) into a **flat directory layout**. Compared with the production ingestion directory, the only differences are:

- **No `__parsed__/` intermediate layer**: artifacts land directly under the specified parent directory for easy inspection;
- **The source file is not archived**: the source file stays at its original location (the production path moves the source file to `<INPUT_DIR>/__parsed__/`);
- **Raw cache validity only checks directory existence**: any non-empty `mineru` / `docling` / `paddleocr_vl` raw directory is considered valid, skipping `_manifest.json` validation.

The rest of the flow (IR construction, sidecar writing, `full_docs` synchronization logic) is identical to production ingestion, making it convenient for troubleshooting parsing-stage issues.

## Command Format

```bash
python -m lightrag.parser.cli <input_file> \
    --engine <engine> \
    [-o <sidecar_parent_dir>] \
    [--doc-id <doc-id>] \
    [--force-reparse] \
    [--preview N]
```

| Argument | Description |
|---|---|
| `input_file` | Path to the source file to parse (positional argument, required). The file must actually exist. |
| `--engine` | Required; choices come from the registry: built-in `native` (local structured parsing) / `legacy` (plain-text extraction, no sidecar) / `mineru` (PDF/Office documents, calls MinerU service) / `docling` (PDF/Office documents, calls docling-serve) / `paddleocr_vl` (PDF/images, calls PaddleOCR-VL service), plus any registered third-party engine. |
| `-o / --sidecar-parent-dir` | Parent directory of the sidecar and raw directories. Defaults to the directory containing the source file. |
| `--doc-id` | Custom document ID. Defaults to `doc-<md5(absolute path of source file)>` (stable across multiple runs on the same file). |
| `--force-reparse` | Effective only for external-service engines (`mineru` / `docling` / `paddleocr_vl` and third-party engines subclassing `ExternalParserBase`): clears the raw directory and forces re-download and re-parse. By default, a non-empty raw directory is reused. |
| `--preview N` | After parsing completes, prints a preview of the first N blocks (headings + content snippets). Default 5; `0` disables it. For engines without a sidecar (e.g. `legacy`), prints the first 400 characters of the extracted text instead. |

## Output Directory Layout

Taking input `./inputs/workspace/sample.pdf` + the default sidecar parent directory (i.e., `./inputs/workspace/`) as an example:

```
./inputs/workspace/
├── sample.pdf                       # original file, untouched
├── sample.pdf.parsed/               # ← sidecar output
│   ├── sample.blocks.jsonl          # JSONL: first line is meta, each subsequent line is a block
│   ├── sample.blocks.assets/        # image/media assets extracted by native (if any)
│   ├── sample.tables.json           # table sidecar (if IR contains tables)
│   ├── sample.drawings.json         # drawing/image sidecar (if IR contains drawings)
│   └── sample.equations.json        # equation sidecar (if IR contains equations)
└── sample.pdf.<engine>_raw/         # ← raw cache for mineru / docling / paddleocr_vl (native has no such directory)
    ├── _manifest.json               # written by the engine download flow; not read by CLI cache validation
    └── <bundle files>               # engine-specific raw artifacts (content_list.json / *.json / assets, etc.)
```

The `native` engine does not produce a raw directory (parsing is local, with no external service involved).

## Typical Use Cases

### A. Locally parse a `.docx` (zero network dependency)

```bash
python -m lightrag.parser.cli ./inputs/workspace/sample.docx --engine native
# Output: ./inputs/workspace/sample.docx.parsed/  (contains blocks.jsonl + assets)
```

### B. Parse a PDF with MinerU (raw will be downloaded on first run)

```bash
# First run: download raw bundle + generate sidecar
python -m lightrag.parser.cli ./inputs/workspace/sample.pdf --engine mineru
# Second run (no changes): raw directory non-empty → reused directly → only regenerate sidecar, fast
python -m lightrag.parser.cli ./inputs/workspace/sample.pdf --engine mineru
# The log will show: [mineru] raw cache hit doc_id=...
```

### C. Parse a PDF with Docling + reuse an existing raw directory

```bash
# Existing ./inputs/workspace/sample.pdf.docling_raw/ (contains docling's JSON output, etc.)
python -m lightrag.parser.cli ./inputs/workspace/sample.pdf --engine docling
# The CLI does not check the manifest; as long as the raw directory is non-empty, the docling-serve call is skipped
```

> Note: this is the equivalent replacement for the "rebuild sidecar from an existing raw directory" scenario that used to live in the legacy `python -m lightrag.parser.external.docling` debug entry point — just place the raw directory at the agreed location (`<sidecar_parent>/<source>.docling_raw/`) to trigger the cache-hit branch.

### D. Output to a custom directory

```bash
python -m lightrag.parser.cli ./inputs/workspace/sample.docx \
    --engine native -o /tmp/debug_sidecar
# Output: /tmp/debug_sidecar/sample.docx.parsed/
# The source file ./inputs/workspace/sample.docx is not moved
```

### E. Force re-parse (clear raw and re-download)

```bash
python -m lightrag.parser.cli ./inputs/workspace/sample.pdf \
    --engine docling --force-reparse
# raw directory is cleared → docling-serve is called again to download → sidecar regenerated
```

## Environment Variables

The `mineru` / `docling` / `paddleocr_vl` engines call external services when the **cache misses** (first parse or `--force-reparse`); the required environment variables are identical to production ingestion:

- **MinerU**: `MINERU_API_MODE` (`local` / `official`), `MINERU_API_TOKEN`, `MINERU_LOCAL_ENDPOINT` or `MINERU_OFFICIAL_ENDPOINT`, optional `MINERU_ENGINE_VERSION` / `MINERU_MODEL_VERSION` / `MINERU_POLL_INTERVAL_SECONDS` / `MINERU_MAX_POLLS`.
- **Docling**: `DOCLING_ENDPOINT`, optional `DOCLING_ENGINE_VERSION` / `DOCLING_DO_OCR` / `DOCLING_FORCE_OCR` / `DOCLING_OCR_ENGINE` / `DOCLING_OCR_PRESET` / `DOCLING_OCR_LANG` / `DOCLING_DO_FORMULA_ENRICHMENT` / `DOCLING_POLL_INTERVAL_SECONDS` / `DOCLING_MAX_POLLS`.
- **PaddleOCR-VL**: `PADDLEOCR_VL_API_MODE` (`official` / `local`), `PADDLEOCR_VL_API_TOKEN` or `PADDLEOCR_VL_LOCAL_ENDPOINT`, optional `PADDLEOCR_VL_OFFICIAL_ENDPOINT` / `PADDLEOCR_VL_POLL_INTERVAL_SECONDS` / `PADDLEOCR_VL_MAX_POLLS` and request-parameter environment variables.

See [FileProcessingPipeline.md](./FileProcessingPipeline.md) for details.

When the **cache is hit** (the raw directory already exists and is non-empty, and `--force-reparse` is not passed), no external service environment variables are needed — this can be used to offline-reproduce parsing output.

## Common Troubleshooting

| Symptom | Action |
|---|---|
| `error: input file does not exist: ...` | Check the `input_file` path; it must be an existing file (not a raw directory). |
| Raw directory exists but sidecar content is still stale | The default behavior is to **reuse** raw and regenerate sidecar. If the raw itself is outdated or has been replaced, add `--force-reparse` to clear and re-download. |
| MinerU reports `MINERU_API_TOKEN` missing / Docling fails to connect to `DOCLING_ENDPOINT` / PaddleOCR-VL reports a missing endpoint or token | A cache miss triggered an external service call — verify the corresponding environment variables; or confirm whether the raw directory is non-empty (no service needed when the cache hits). |
| Source file is unexpectedly moved | Should not happen: the CLI has mocked the archive function. If reproducible, please file an issue (a new archive call site may have been added in the pipeline). |
| docling reports `produced zero blocks` | The main JSON content in docling raw is unparseable or empty. Check whether the `*.json` files in the raw directory are valid. |

## Equivalence with the Production Parsing Path

This CLI drives the same registry dispatch path as the pipeline parse worker — `get_parser(engine).parse(ParseContext(rag, ...))` (with `rag` being the lightweight stand-in in `lightrag/parser/debug.py`), so:

- The sidecar fields, naming, and content format are identical to production ingestion;
- The IR builders, `write_sidecar` calls, and `_persist_parsed_full_docs` behavior are identical;
- All three differences are implemented via `monkey-patch` inside the CLI — **no production code is modified**:
  1. `parsed_artifact_dir_for` → returns the flat path (no `__parsed__/`);
  2. the resolved parser instance's `is_bundle_valid` → "raw is valid if non-empty" (external-service engines only);
  3. `archive_docx_source_after_full_docs_sync` → no-op, source file preserved.

Results can be cross-validated against golden fixtures under `tests/parser/docx/golden/native_docx/` (the CLI does not freeze timestamps; just exclude time fields such as `created_at` when comparing).
