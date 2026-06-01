# File Processing Pipeline Specification

Starting from version v1.5.0 (currently on the dev branch), LightRAG's file processing pipeline has received a major upgrade:

* Supports multiple file content extraction engines: legacy, native, mineru, docling
* Supports multiple text chunking methods: Fix, Recursive, Vector, Paragraph
* Supports disabling entity-relation extraction for individual files

LightRAG Server introduces an intermediate file-processing format: `LightRAG Document`. This format supports multimodal data such as tables and images, and also includes the document's section/paragraph metadata, which is convenient for content traceability later.

This document is organized from the perspective of **LightRAG Server** deployment and use: the quick-start configuration that can be applied directly is given first, followed by configuration syntax for content extraction and chunking, storage / directory layout, deduplication, concurrency, and resume rules. Developers who call the `LightRAG` class directly via Python should jump to [Chapter 8: Python SDK Invocation](#8-python-sdk-invocation).

## 1. Quick Start

### Keep the legacy file-processing behavior

All files are processed using the legacy document parsing and chunking strategy. Either leave `LIGHTRAG_PARSER` unconfigured, or set it to the following value:

```bash
LIGHTRAG_PARSER=*:legacy-F
```

### Recommended starting file-processing behavior

No reliance on external document parsing services or on `VLM` vision models. Use the new built-in `Native` engine to parse `docx` documents with table (t) and equation (e) modality analysis enabled, paired with the `P` chunking strategy; other documents use the legacy content extractor paired with the more effective `R` chunking strategy.

```bash
LIGHTRAG_PARSER=*:native-teP,*:legacy-R
```

### Enable multimodal processing capability

Enabling multimodal processing requires the `MinerU` file parsing service and a `VLM` vision recognition model. Use `Native` to parse `docx` files; use `MinerU` to parse `pdf`, `office`, and various image files. All of the above files have image (i), table (t), and equation (e) modality analysis enabled and are paired with the `P` chunking strategy. Other documents fall back to the legacy content extractor paired with the `R` chunking strategy.

```bash
LIGHTRAG_PARSER=*:native-iteP,*:mineru-iteP,*:legacy-R
VLM_PROCESS_ENABLE=true
VLM_LLM_MODEL=kimi-k2.6
MINERU_API_MODE=local
MINERU_LOCAL_ENDPOINT=http://localhost:8000
```

> `P` is LightRAG's native chunking strategy; see [Paragraph Semantic Chunking](ParagraphSemanticChunking.md) for details. For VLM configuration, see [Role-based LLM/VLM Configuration Guide](RoleSpecificLLMConfiguration.md).

## 2. Content Extraction and Processing Option Configuration

LightRAG's file processing configuration is composed of two parts: the content extraction engine determines how the original file is parsed, and the processing options determine whether multimodal analysis is performed after parsing, which chunking method to use, and whether to build a knowledge graph. Typically, the environment variable `LIGHTRAG_PARSER` is first used to set default rules by file extension, and then a `[hint]` in the filename overrides individual files. Engine and options can be written in the same configuration fragment, for example `docx:native-iet` or `report.[native-R!].docx`.

For backward compatibility, when the configuration is not modified, the upgraded file content extraction behavior remains the original `legacy` behavior. To enable the new content processing engines, configure as described in this section.

### 2.1 Configuration Syntax Overview

The complete configuration model is as follows:

```text
LIGHTRAG_PARSER=ext:engine-options,ext:engine,*:legacy-R
filename.[ENGINE].ext
filename.[ENGINE-OPTIONS].ext
filename.[-OPTIONS].ext
```

- `LIGHTRAG_PARSER` is the default rule table, matched by file extension, e.g., `pdf:mineru`, `docx:native-iet`.
- The `[hint]` in a filename is a single-file override rule, e.g., `paper.[mineru].pdf`, `memo.[native-R!].docx`.
- `ENGINE` is the content extraction engine: `legacy`, `native`, `mineru`, or `docling`.
- `OPTIONS` is a string combination of processing options, e.g., `iet`, `R!`, `P`. The options are ultimately written into `process_options` and read by subsequent pipeline stages.
- The hyphen in `ENGINE-OPTIONS` is only used to separate the engine from the options; it is not part of the options themselves.
- When only processing options are specified, it must be written as `[-OPTIONS]`, e.g., `[-!]`. `[abc]` without a hyphen is strictly interpreted as an engine name and will raise an error; it will not fall back to being interpreted as options.

Common combination examples:

```bash
LIGHTRAG_PARSER=pdf:mineru-R,docx:native-ietP,*:legacy-R
MINERU_API_MODE=local
MINERU_LOCAL_ENDPOINT=http://localhost:8000
DOCLING_ENDPOINT=http://localhost:5001
```

```text
my-proposal.[native-iet].docx   # Use the native engine, enable drawing/table/equation analysis
my-memo.[native-R!].docx        # Use the native engine, recursive semantic chunking, disable knowledge graph construction
my-proposal.[-!].docx           # Use the default engine, only disable knowledge graph construction
my-proposal.[mineru].docx       # Use the MinerU engine, all processing options default
```

### 2.2 Default Rules: `LIGHTRAG_PARSER`

`LIGHTRAG_PARSER` is used to configure the default content extraction engine for different file extensions; default processing options for the rule can also be appended after the engine:

```text
ext:engine,ext:engine,*:legacy
ext:engine;ext:engine;*:legacy
ext:engine-options
```

- The left side matches the file extension, not the full filename; write `pdf:mineru`, not `*.pdf:mineru`.
- Rules can be separated by either a comma `,` or a semicolon `;`.
- Rules are checked left to right; priority rules go in front, with the wildcard rule typically at the end.
- The `-options` suffix after the engine serves as the default `process_options` for files matched by this rule. For example, `LIGHTRAG_PARSER=docx:native-iet` means all `.docx` files default to the `native` engine with image, table, and equation analysis enabled.

### 2.3 Single-File Override: filename hints

Square brackets in the filename can be used to temporarily specify how a single file is processed:

```text
paper.[mineru-R].pdf
slides.[docling].pptx
memo.[native-P].docx
notes.[-R].md
```

The content inside the square brackets supports three forms:

```text
[ENGINE]              # Specify only the engine; processing options use the default or what LIGHTRAG_PARSER provides
[ENGINE-OPTIONS]      # Specify both engine and processing options
[-OPTIONS]            # Specify only processing options; the engine still follows LIGHTRAG_PARSER / default rules
```

When parsing the hint, content without a hyphen must match an engine name exactly (`mineru` / `native` / `docling` / `legacy`); when there is content before a hyphen, the part before the hyphen is the engine and the part after is the options; when starting with a hyphen, it specifies only options. The legacy `[OPTIONS]` syntax is no longer valid; for example, `[iet]` must now be written as `[-iet]`.

### 2.4 Content Extraction Engines

| Engine | Description | Supported file formats (extensions) |
| --- | --- | --- |
| `legacy` | Legacy extraction; content is centrally extracted before joining the pipeline | `txt` `md` `mdx` `pdf` `docx` `pptx` `xlsx` `rtf` `odt` `tex` `epub` `html` `htm` `csv` `json` `xml` `yaml` `yml` `log` `conf` `ini` `properties` `sql` `bat` `sh` `c` `h` `cpp` `hpp` `py` `java` `js` `ts` `swift` `go` `rb` `php` `css` `scss` `less` |
| `native` | Built-in intelligent structured content extractor | `docx` |
| `mineru` | External MinerU content extraction engine | `pdf` `doc` `docx` `ppt` `pptx` `xls` `xlsx` `png` `jpg` `jpeg` `jp2` `webp` `gif` `bmp` |
| `docling` | External Docling content extraction engine | `pdf` `docx` `pptx` `xlsx` `md` `html` `xhtml` `png` `jpg` `jpeg` `tiff` `webp` `bmp` |

`mineru` and `docling` are external content extraction engines; before enabling related rules, the services must be running first, and the corresponding endpoint/token must be configured in LightRAG.

LightRAG caches the parsing results of the `mineru` and `docling` engines locally. Re-uploading the same file usually does not trigger the engine to re-parse the document. To delete the parse cache, you must click the "also delete file" option in the delete-file dialog of the document management interface. Modifying the endpoint addresses and effective extraction parameters of the `mineru` / `docling` engines will also invalidate the cache, causing the engine to re-parse the file content on the next upload of the same file.

#### MinerU Configuration and Local Deployment

The MinerU client supports two modes; choose one:

- `local`: self-hosted MinerU service (the official Docker Compose deployment is recommended); LightRAG calls the local container via HTTP.
- `official`: directly connects to the MinerU official precise API v4; you need to apply for a token at [mineru.net](https://mineru.net).

**Local deployment (Docker Compose)**

Clone the official [opendatalab/MinerU](https://github.com/opendatalab/MinerU) repository to your local machine, enter the docker deployment directory inside the repository, and first build the image:

```bash
docker compose -f compose.yaml build
```

Then start the API service (`--profile api` is required to enable the HTTP API container; the default listening port is 8000):

```bash
docker compose -f compose.yaml --profile api up -d
```

For image build details, GPU driver setup, model weight locations, etc., refer to the official README: <https://github.com/opendatalab/MinerU>.

**LightRAG-side env configuration**

Local mode (self-hosted mineru-api):

```bash
MINERU_API_MODE=local
MINERU_LOCAL_ENDPOINT=http://localhost:8000
```

Official mode (MinerU cloud API):

```bash
MINERU_API_MODE=official
MINERU_API_TOKEN=<your_token>
# MINERU_OFFICIAL_ENDPOINT=https://mineru.net   # Default value, usually no need to change
```

For the remaining advanced switches (`MINERU_MODEL_VERSION`, `MINERU_LANGUAGE`, `MINERU_ENABLE_TABLE` / `MINERU_ENABLE_FORMULA`, `MINERU_PAGE_RANGES`, `MINERU_LOCAL_BACKEND` / `MINERU_LOCAL_PARSE_METHOD`, `MINERU_POLL_INTERVAL_SECONDS` / `MINERU_MAX_POLLS`, `MINERU_ENGINE_VERSION`, `LIGHTRAG_FORCE_REPARSE_MINERU`, etc.), refer to the MinerU section of the `env.example` template at the repository root. Note that `MINERU_PAGE_RANGES` has different semantics in the two modes: `official` supports a complete list (e.g., `1-3,5,7-9`), while `local` only supports a single page (`3`) or a simple range (`1-10`); it does not accept comma-separated lists.

#### Docling Configuration

The `docling` content extraction engine requires an external [docling-serve](https://github.com/DS4SD/docling-serve) service (v1 async API). Minimal configuration:

```bash
DOCLING_ENDPOINT=http://localhost:5001
```

`DOCLING_ENDPOINT` is just the base URL (**without** `/v1/convert/file/async`). Currently LightRAG uses Docling's standard pipeline to process files. Users can control the behavior of the Docling pipeline through the following environment variables:

| Env | Default | Meaning |
| --- | --- | --- |
| `DOCLING_DO_OCR` | `true` | OCR master switch |
| `DOCLING_FORCE_OCR` | `true` | Force OCR per page (mandatory for scanned documents; enabling it for non-scanned documents usually also helps improve layout recognition quality) |
| `DOCLING_OCR_ENGINE` | `auto` | OCR engine selection (not recommended to change) |
| `DOCLING_OCR_PRESET` | `auto` | OCR engine preset (not recommended to change) |
| `DOCLING_OCR_LANG` | (empty) | Set per OCR engine requirements (not recommended to change) |
| `DOCLING_DO_FORMULA_ENRICHMENT` | `false` | Whether to recognize equations in the document and output them in LaTeX format; before enabling, ensure that Docling has downloaded the equation recognition model on the backend (see explanation below) |

When `DOCLING_OCR_ENGINE` / `DOCLING_OCR_PRESET` are not configured, they are equivalent to `auto`; when `DOCLING_OCR_LANG` is not configured, no language list is passed to docling-serve, and the OCR engine uses its own default. The parse cache signature is computed from these effective parameters, so "not configured" and "explicitly set to the default value" do not invalidate the cache.

Two polling-budget envs (docling-serve uses server-side long-poll; the client does not sleep extra):

| Env | Default | Meaning |
| --- | --- | --- |
| `DOCLING_POLL_INTERVAL_SECONDS` | `5` | Poll interval for awaiting parse results |
| `DOCLING_MAX_POLLS` | `240` | Maximum poll iterations; raises `TimeoutError` when exceeded;<br />default wait time ≈ 5 × 240 (about 20 minutes) |

Three bundle-cache envs:

| Env | Default | Meaning |
| --- | --- | --- |
| `DOCLING_ENGINE_VERSION` | (empty) | Docling engine version; version changes invalidate the parse cache |
| `LIGHTRAG_FORCE_REPARSE_DOCLING` | `false` | When set to `true`/`1`, the parse cache is not used |
| `DOCLING_BBOX_ATTRIBUTES` | `{"origin":"LEFTBOTTOM"}` | Default coordinate system for Docling layout |

**Prerequisites for `DOCLING_DO_FORMULA_ENRICHMENT`**: the docling-serve side must have the code-formula model weights ready. The adapter is dual-track compatible — when enabled, the `text` field is LaTeX; when disabled, or when missing weights cause `text == orig`, it falls back to plain text and does not write `equations.json`. Therefore the default of `false` is conservative; turn it on only after confirming the model is ready on the deployment side.

#### Docling Local Deployment (enabling LaTeX equation recognition)

The following uses a Docker-based docling-serve deployment as an example, giving the complete steps from image download to model mounting. After deployment completes, write `DOCLING_DO_FORMULA_ENRICHMENT=true` into LightRAG's `.env` to enable LaTeX equation recognition.

> **Important**: the steps below are based on an environment where the GPU supports CUDA 13. If your GPU is older and does not support CUDA 13, replace the image name `docling-serve-cu130:main` in the command and compose file with the tag corresponding to your CUDA version. For the list of available images, see [docling-serve Packages](https://github.com/orgs/docling-project/packages?repo_name=docling-serve).

**1. Pull the image**

```bash
docker pull ghcr.io/docling-project/docling-serve-cu130:main
```

**2. Download models**

```bash
# Create the docling working directory
mkdir docling
cd docling

# Create the model mount directory
mkdir models

# Copy the existing models inside the container into the models directory
docker run --rm -it \
  -v "$(pwd)/models:/opt/app-root/src/models" \
  ghcr.io/docling-project/docling-serve-cu130:main \
  cp -r /opt/app-root/src/.cache/docling/models /opt/app-root/src/

# Download the equation recognition model
docker run --rm \
  -v "$(pwd)/models:/opt/app-root/src/models" \
  -e DOCLING_SERVE_ARTIFACTS_PATH="/opt/app-root/src/models" \
  ghcr.io/docling-project/docling-serve-cu130:main \
  docling-tools models download-hf-repo docling-project/CodeFormulaV2 -o models
```

**3. Create `docker-compose.yaml`**

Create `docker-compose.yaml` in the `docling` directory from the previous step, with the following contents:

```yaml
services:
  docling-serve:
    image: ghcr.io/docling-project/docling-serve-cu130:main
    container_name: docling-serve
    ports:
      - "5001:5001"
    environment:
      DOCLING_SERVE_ENABLE_UI: "true"
      NVIDIA_VISIBLE_DEVICES: "all"
      DOCLING_SERVE_ARTIFACTS_PATH: "/opt/app-root/src/models"
    # deploy:  # This section is for compatibility with Swarm
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: all
    #           capabilities: [gpu]
    runtime: nvidia
    restart: always
    volumes:
      - ./models:/opt/app-root/src/models
```

Then execute `docker compose up -d` in that directory to start the service. After the container is ready, set the following in LightRAG's `.env`:

```bash
DOCLING_ENDPOINT=http://localhost:5001
DOCLING_DO_FORMULA_ENRICHMENT=true
```

This enables LightRAG to recognize equations in documents via the local docling-serve and output them in LaTeX form.

### 2.5 File Processing Options

Processing options control the behavior of a single file with respect to multimodal analysis, knowledge graph construction, and text chunking. All options are optional; defaults are shown in the table below. At most one chunking method (F/R/V/P) is specified per file; the other options can be combined arbitrarily.

| Option | Type | Default | Meaning |
| --- | --- | --- | --- |
| `i` | Multimodal | Off | Enable image analysis (VLM) |
| `t` | Multimodal | Off | Enable table analysis (VLM) |
| `e` | Multimodal | Off | Enable equation analysis (VLM) |
| `!` | Pipeline | Off | Disable entity/relation extraction; do not build the knowledge graph (only the chunks vector index is kept; naive / mix retrieval still works) |
| `F` | Chunking | Default | Fix / fixed-length chunking: legacy method, splits mechanically by fixed token length or by separator (no chunk overlap when splitting by separator) |
| `R` | Chunking | - | Recursive / recursive character chunking (RecursiveCharacterTextSplitter@LangChain): takes a list of separators (default `["\n\n","\n","。","！","？","；","，"," ",""]`, ordered from strongest to weakest semantic boundary). Splits by paragraph (double newline) first; if a chunk is still over the token limit, falls back stepwise to single newline → Chinese sentence-ending punctuation (`。！？`) → Chinese mid-sentence punctuation (`；，`) → space → per-character split. **The default cascade includes Chinese punctuation**, letting Chinese / mixed Chinese-English documents split at semantic boundaries. English `.?!` is deliberately excluded (literal matching would mis-split `0.95` / `e.g.`). |
| `V` | Chunking | - | Vector / semantic vector chunking (SemanticChunker@LangChain): first splits text into sentences (the default sentence splitting regex recognizes both English `.?!` and Chinese `。？！`, allowing correct sentence splitting in Chinese / mixed Chinese-English documents), computes embeddings of adjacent sentences, then finds semantic breakpoints based on the specified threshold strategy (e.g., percentile, standard_deviation, or interquartile) for splitting. `SemanticChunker` itself has no chunk size cap — any semantic chunk that exceeds `chunk_token_size` is automatically split again by R before persistence (preserving V's non-overlap semantics). This chunking strategy never produces overlapping chunks. |
| `P` | Chunking | - | Paragraph / paragraph semantic chunking (native); splits by heading first and strictly avoids mixing content from the bottom of the previous heading with content from the next heading, which would break semantics. Suited for chunking documents that can accurately identify headings with a clear heading structure. When the body under the same heading is too long and falls back to R, overlap can be preserved according to `CHUNK_P_OVERLAP_SIZE`; bridging text between adjacent large tables can also be repeated into the surrounding table chunks within that budget. This chunking method can only be applied to `lightrag` content stored in the sidecar directory. If `lightrag` content does not exist, it degrades to chunking with `R`. This chunking method produces far fewer overlapping chunks than the `R` or `F` strategies. |

> The global multimodal switch `addon_params["enable_multimodal_pipeline"]` is deprecated; the related behavior is now uniformly controlled by the file-level `i/t/e` options. See [Appendix A](#appendix-a-notes-on-upgrading-from-legacy).

#### Option effective stages

Different characters of processing options take effect at different stages of the pipeline:

| Option | Stage | Description |
| :-: | --- | --- |
| i/t/e | Analyzing (multimodal analysis) | Determines whether VLM summarization analysis is invoked on the images / tables / equations in the sidecar. **The extraction stage is unaffected**: the content extraction engine outputs `drawings.json` / `tables.json` / `equations.json` sidecar files based on what the document actually contains. As a result, simply tweaking the `i`/`t`/`e` options to trigger "re-analysis" can complete VLM later without re-parsing the original file. |
| ! | Extraction (entity-relation extraction) | Skips entity/relation extraction and graph writing; chunks are still written to the vector store to retain naive / mix retrieval capabilities. |
| F/R/V/P | Chunking (text chunking) | Determines which chunking strategy to use; does not affect the output of the parsing stage. |

> Modality availability is signaled solely by "whether the sidecar file exists"; the content extraction engine does not need to declare its capabilities in meta. If a given document contains no images/tables/equations, the corresponding sidecar is not written; even if the user has enabled `i/t/e`, the corresponding modality is silently skipped, but `analyze_multimodal` logs an INFO-level line for that document (`[analyze_multimodal] sidecar e:equations empty: doc—id ...`), making it easy to diagnose "why didn't the VLM run". This is not an error.

### 2.6 Validation, Priority, and Fallback

- `LIGHTRAG_PARSER` is strictly validated at startup: unknown content extraction engines, malformed extension syntax, explicitly using an unsupported extension, external engines missing endpoint, and illegal characters in processing options all cause startup to fail.
- **When a wildcard rule matches a certain extension**, the engine must pass two usability checks (see `parser_routing._engine_is_usable`): (a) the engine's capability table supports that extension; (b) if it is an external engine (`mineru` / `docling`), the corresponding endpoint/token environment variable is configured. If either check fails, the rule is skipped and the next rule is matched. For example, in `*:mineru;html:docling`: MinerU does not support the `html` extension (condition a fails), so `html` continues to match `docling`; if `MINERU_API_MODE=local` but `MINERU_LOCAL_ENDPOINT` is not set, all PDFs also skip `*:mineru` and fall to the next rule (condition b fails). This behavior applies to both `LIGHTRAG_PARSER` rule matching and filename hint engine selection.
- Filename hints have higher priority than `LIGHTRAG_PARSER`. If the engine specified in a hint does not support that extension, the system falls back to the default rules to continue selecting an available engine.
- If the filename hint provides a non-empty options string, the hint takes precedence; otherwise the default options of the matching item in `LIGHTRAG_PARSER` are used; if neither is provided, all defaults are used.
- If no rule is available, the file content extraction falls back to `legacy`; if `legacy` also does not support the file extension, an error entry is added to the system and the uploaded file remains in the `INPUT` directory.
- At most one of F/R/V/P may appear; repeating the same option has effect only once but does not raise an error.
- Case-sensitive: the chunking options F/R/V/P must be uppercase; other options i/t/e must be lowercase.
- If illegal characters appear inside the square brackets, the entire hint is invalidated, the engine follows the default rules, and the options fall back to `LIGHTRAG_PARSER` defaults or all defaults; a warning is also logged.
- `P` is only effective for structured `LightRAG Document` results extracted by `native`; for the `legacy` path or unstructured output, it automatically degrades to `R` and logs a warning.

## 3. Chunker Parameter Configuration (chunk_options)

### 3.1 Responsibilities of process_options vs chunk_options

`process_options` selects **which** chunking strategy (F/R/V/P), while `chunk_options` decides **which parameters** that chunker uses. The two responsibilities are orthogonal: the former is a single-character selector, the latter is a structured dictionary.

```
env vars                                                  (read once at startup)
   │
   ▼
addon_params["chunker"]                                   (LightRAG instance field, filled by env with legacy fallback)
   │
   ▼  resolve_chunk_options(addon_params, split_by_character=…, split_by_character_only=…)
   │
full_docs[doc_id]["chunk_options"]                       (frozen at enqueue time, an independent snapshot per file)
   │
   ▼
chunker(tokenizer, content, chunk_token_size, **strategy_kwargs)   (dispatched by selector during chunking)
```

- **env vars** are loaded into `addon_params["chunker"]` during the `LightRAG.__init__` stage (strategy-specific env is read by `default_chunker_config()`, then `_apply_chunk_size_overlay` fills in legacy env as a fallback).
- **`addon_params["chunker"]`** is an `ObservableAddonParams` field; for Server deployments, you only need env / restart for the new values to take effect. To change it at runtime within the Python process (without restarting) and to do per-file overrides, see [Chapter 8: Python SDK Invocation](#8-python-sdk-invocation).
- **`full_docs.chunk_options`** is frozen at `apipeline_enqueue_documents` enqueue time: by default it is assembled by `resolve_chunk_options(self.addon_params, ...)` on the spot; if the caller passes a `chunk_options` argument, it is persisted as-is (SDK usage, see §8.4).
- **The chunker invocation** takes the corresponding sub-dictionary from `full_docs.chunk_options` and dispatches to F/R/V/P by the `process_options.chunking` selector.

### 3.2 Environment Variables

All variables in the table below are read into `addon_params["chunker"]` once when `LightRAG` is instantiated: strategy-specific env is read by `default_chunker_config()`, while legacy env (`CHUNK_SIZE` / `CHUNK_OVERLAP_SIZE`) is filled in by `_apply_chunk_size_overlay` into slots that neither strategy env nor legacy constructor fields filled. After modifying env, the service must be restarted (or a new `LightRAG` instance created) for it to take effect; documents already enqueued hold the frozen snapshot and are unaffected.

| Variable | Default | Type | Scope |
|---|---|---|---|
| `CHUNK_SIZE` | `1200` | int | Legacy top-level `chunk_token_size` fallback; lower priority than strategy-specific env and the SDK path setting of `addon_params["chunker"]["chunk_token_size"]` |
| `CHUNK_OVERLAP_SIZE` | `100` | int | Legacy overlap fallback; filled when a strategy has neither a specific env (`CHUNK_F_OVERLAP_SIZE` / `CHUNK_R_OVERLAP_SIZE` / `CHUNK_P_OVERLAP_SIZE`) nor the SDK path's `LightRAG(chunk_overlap_token_size=…)` |
| `CHUNK_F_SIZE` | unset | int | F strategy-specific `chunk_token_size`; higher than the top-level legacy fallback (`CHUNK_SIZE` and the SDK path's `LightRAG(chunk_token_size=…)`). When unset, F inherits the top-level resolved value. |
| `CHUNK_F_OVERLAP_SIZE` | unset | int | F strategy-specific overlap; higher than the legacy constructor field and `CHUNK_OVERLAP_SIZE` |
| `CHUNK_F_SPLIT_BY_CHARACTER` | (unset = `null`) | str? | F pre-split separator; `null` / empty string = split by token window only |
| `CHUNK_F_SPLIT_BY_CHARACTER_ONLY` | `false` | bool | F strict mode: no secondary token split; raise error when oversized |
| `CHUNK_R_SIZE` | unset | int | R strategy-specific `chunk_token_size`; higher than top-level legacy fallback (`CHUNK_SIZE` and the SDK path's `LightRAG(chunk_token_size=…)`). When unset, R inherits the top-level resolved value. |
| `CHUNK_R_OVERLAP_SIZE` | unset | int | R strategy-specific overlap; higher than the legacy constructor field and `CHUNK_OVERLAP_SIZE` |
| `CHUNK_R_SEPARATORS` | `["\n\n","\n","。","！","？","；","，"," ",""]` | JSON array string | R separator cascade, ordered from strongest to weakest semantic boundary. The default includes Chinese sentence-ending (`。！？`) and mid-sentence (`；，`) punctuation, letting Chinese / mixed Chinese-English documents split at semantic boundaries. English `.?!` is deliberately excluded (literal matching would mis-split numbers and abbreviations). |
| `CHUNK_V_SIZE` | unset | int | V strategy-specific `chunk_token_size` (hard cap, automatically re-split through R when exceeded); higher than the top-level legacy fallback. When unset, V inherits the top-level resolved value. |
| `CHUNK_V_BREAKPOINT_THRESHOLD_TYPE` | `percentile` | str | V threshold type; can be `percentile` / `standard_deviation` / `interquartile` / `gradient` |
| `CHUNK_V_BREAKPOINT_THRESHOLD_AMOUNT` | (unset = `null`) | float? | V threshold magnitude; `null` lets LangChain pick the default by type (e.g., percentile=95) |
| `CHUNK_V_BUFFER_SIZE` | `1` | int | V sentence buffer window; the number of adjacent sentences to merge during distance computation |
| `CHUNK_V_SENTENCE_SPLIT_REGEX` | `(?<=[.?!])\s+\|(?<=[。？！])` | str | V's sentence splitting regex, fed to LangChain's `SemanticChunker`. The default recognizes both English `.?!` (requiring trailing whitespace to avoid mis-splitting `0.95`) and Chinese `。？！` (no whitespace required, fitting Chinese continuous writing). The env value is the raw regex string; no JSON quoting needed. |
| `CHUNK_P_SIZE` | `2000` (`DEFAULT_CHUNK_P_SIZE`) | int | P strategy-specific `chunk_token_size`. Unlike R/V, P does NOT inherit the top-level `CHUNK_SIZE` / `LightRAG(chunk_token_size=…)` when unset — paragraph-semantic merging needs more headroom than the global default to keep related paragraphs together, so the slot always carries `DEFAULT_CHUNK_P_SIZE` (2000) instead. |
| `CHUNK_P_OVERLAP_SIZE` | unset | int | P strategy-specific overlap; higher than the legacy constructor field and `CHUNK_OVERLAP_SIZE`. Used for text overlap when long body text within the same JSONL content line falls back to R, and as the per-side budget for bridging text copied into the adjacent large-table chunks. |

P's internal ratio constants are algorithmic scales and are automatically derived in proportion to `chunk_token_size`. P always uses an independent `chunk_token_size` decoupled from the global chain — even when `CHUNK_P_SIZE` is unset, P falls back to `DEFAULT_CHUNK_P_SIZE` (2000) rather than the global `CHUNK_SIZE`, because paragraph-semantic merging needs more headroom than the global default to keep related paragraphs together. Use `CHUNK_P_SIZE` to override that default per deployment. `CHUNK_P_OVERLAP_SIZE` only affects P's internal plain-text fallback and table bridging context; it does not let table row-level slices overlap each other. `CHUNK_F_SIZE` / `CHUNK_R_SIZE` / `CHUNK_V_SIZE` work differently — when unset they DO fall back to the top-level `chunk_token_size` (F is the default global window, R prefers a smaller target to better split sentences, while V — as an advisory ceiling — typically wants to be enlarged to reduce over-splitting).

### 3.3 Priority Chain

The final value of each chunking slot is resolved by a specificity-ordered chain (high → low):

1. **`addon_params["chunker"]` explicit value** — field values explicitly written at construction time or set at runtime via the SDK path (see §8.3). Server-only deployments usually don't hit this tier. Most direct; wins everything.
2. **Strategy-specific env** — `CHUNK_F_SIZE` / `CHUNK_R_SIZE` / `CHUNK_V_SIZE` (per-strategy `chunk_token_size`), `CHUNK_F_OVERLAP_SIZE` / `CHUNK_R_OVERLAP_SIZE` / `CHUNK_P_OVERLAP_SIZE` (overlap), `CHUNK_P_SIZE` (P-specific). When the corresponding size env is unset, F/R/V inherit the top-level `chunk_token_size`. Filled only when the slot is not already occupied by ①.
3. **Legacy constructor fields** — `LightRAG(chunk_token_size=…, chunk_overlap_token_size=…)`; only effective on the SDK path, see §8.2. Strategy-agnostic, "coarse-grained default", fills only the slots still empty.
4. **Legacy env** — `CHUNK_SIZE` / `CHUNK_OVERLAP_SIZE`. Final fallback.

Example: `CHUNK_R_OVERLAP_SIZE=42` + `LightRAG(chunk_overlap_token_size=2)` → R sub-dictionary `chunk_overlap_token_size=42` (strategy env wins), F / P sub-dictionary `chunk_overlap_token_size=2` (no F / P-specific env; the legacy constructor field is filled in).

**Special case for P's `chunk_token_size`**: the P `chunk_token_size` slot does NOT walk the full four-tier chain. When ① is not explicitly provided, it resolves directly via `CHUNK_P_SIZE` env > `DEFAULT_CHUNK_P_SIZE` (2000), **skipping** ③ legacy constructor field `LightRAG(chunk_token_size=…)` and ④ legacy env `CHUNK_SIZE`. See the `CHUNK_P_SIZE` row in §3.2 for the rationale.

Three layers of semantic guarantee:

1. **Reproducibility**: change env, restart — old documents still chunk by the snapshot from the moment they were enqueued; results unchanged.
2. **Resume consistency**: resume branch B (content already extracted, redo chunking by current `process_options`) also reads `full_docs.chunk_options`, preventing env drift from breaking consistency.
3. **Per-file personalization**: callers can pass different `chunk_options` for each file (typical usage: a management UI configures separators or V threshold individually for a certain file). These are the input semantics on the SDK path; see §8.4.

### 3.4 Field Structure

`addon_params["chunker"]` (instance field) keeps the sub-dictionaries of all four strategies as the runtime baseline; `full_docs[doc_id]["chunk_options"]` is a **slim snapshot** — at enqueue time, only the strategy sub-dictionary selected by `process_options` is kept (default F), and the parameters of other strategies are discarded, because the processing stage will not read them. When re-parsing, `process_options` and `chunk_options` are rewritten together, avoiding residue of old-strategy parameters.

**`addon_params["chunker"]` full baseline** (modifiable at runtime via SDK, affecting subsequent enqueues):

```jsonc
{
  "chunk_token_size": 1200,                                   // common token cap
  "fixed_token": {                                            // F-specific
    "chunk_token_size": 1200,                                 // optional; when omitted, inherits the top-level chunk_token_size (seedable via CHUNK_F_SIZE)
    "chunk_overlap_token_size": 100,
    "split_by_character": null,
    "split_by_character_only": false
  },
  "recursive_character": {                                    // R-specific
    "chunk_token_size": 1200,                                 // optional; when omitted, inherits the top-level chunk_token_size
    "chunk_overlap_token_size": 100,
    "separators": ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]   // default cascade includes Chinese punctuation
  },
  "semantic_vector": {                                        // V-specific
    "chunk_token_size": 1200,                                 // optional hard cap; re-split through R when exceeded
    "breakpoint_threshold_type": "percentile",                // percentile | standard_deviation | interquartile | gradient
    "breakpoint_threshold_amount": null,                      // null = LangChain default
    "buffer_size": 1,
    "sentence_split_regex": "(?<=[.?!])\\s+|(?<=[。？！])"      // default regex handles both English and Chinese sentence-ending punctuation
  },
  "paragraph_semantic": {                                     // P-specific
    "chunk_token_size": 2000,                                 // when omitted, resolves from CHUNK_P_SIZE or DEFAULT_CHUNK_P_SIZE (2000);
                                                              // does NOT inherit the common chunk_token_size
    "chunk_overlap_token_size": 100                           // when omitted, inherits the legacy overlap resolution chain
  }
}
```

**`full_docs[doc_id]["chunk_options"]` slim snapshot** (projected by selector; example below is for `process_options="R"`):

```jsonc
{
  "chunk_token_size": 1200,                                   // common token cap (kept as a top-level fallback)
  "recursive_character": {                                    // the only retained strategy sub-dictionary
    "chunk_overlap_token_size": 100,
    "separators": ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
  }
}
```

selector → sub-dictionary mapping: F → `fixed_token`, R → `recursive_character`, V → `semantic_vector`, P → `paragraph_semantic`; without a selector, F is the default. Each sub-dictionary corresponds one-to-one with the keyword-only parameters of the corresponding chunker function; when adding new parameters, no dispatcher change is needed, just add a kwarg to the chunker function.

### 3.5 Backward Compatibility for Missing Fields

Old documents at enqueue time don't yet have the `chunk_options` field; during chunking, the dispatcher calls `resolve_chunk_options(self.addon_params, process_options=…)` per the current `process_options` to fall back to a slim snapshot. After upgrading, it is recommended to run a reprocess once to give old documents a slim `chunk_options` snapshot (aligned with the current `process_options`).

## 4. Storage and Directory Layout

### 4.1 `full_docs` Fields

File enqueue and extraction results are written into `full_docs`:

| Field | Description |
| --- | --- |
| `file_path` | Basename of the filename (without directory), **preserves the original name provided by the user (including the square-bracket hint)**, e.g., `abc.[native-iet].docx` is written as-is. When no valid source is provided, it is saved as `unknown_source`. The filename hint is not stripped, so the management UI can directly show the user's original naming intent. |
| `canonical_basename` | The canonicalized basename with the processing hint stripped (e.g., `abc.docx`). Filename deduplication uses this field as the index key, ensuring `abc.docx` and `abc.[native-iet].docx` are treated as the same logical document. |
| `source_path` | The original path provided at enqueue time (written only when it contains a directory separator or is an absolute path), used by the `native` / `mineru` / `docling` parsers to locate the actual file. |
| `parse_format` | Content format: `pending_parse`, `raw`, `lightrag`. |
| `content` | When `raw`, holds the extracted text; when `pending_parse`, it is an empty string; when `lightrag`, holds the **complete merged text** starting with `{{LRdoc}}` (concatenated body segments of all `type=="content"` lines in `.blocks.jsonl`). During chunking, `parse_native` strips the prefix and hands it to the chunking_func, going through exactly the same code path as `raw`. |
| `content_hash` | MD5 of the content, used for cross-filename deduplication. For `parse_format=raw`, takes the hash of text after `sanitize_text_for_encoding`; for `parse_format=lightrag`, takes the hash of the `*.blocks.jsonl` file; for `parse_format=pending_parse`, not written, filled in after extraction completes. |
| `lightrag_document_path` | When `parse_format=lightrag`, saves the path to the structured LightRAG Document; new records prefer to save the path relative to `INPUT_DIR`, e.g., `__parsed__/report.docx.parsed/report.blocks.jsonl`. Note that the subdirectories and the blocks filename in the path both use the canonicalized basename (without hint). |
| `parse_engine` | The engine that actually completed extraction: `legacy`, `native`, `mineru`, `docling`. For files awaiting extraction, can also temporarily store the target engine. |
| `process_options` | The original processing options string recorded at enqueue time (without engine name and the separator `-`), e.g., `"iet"`, `"R!"`, `""`. Downstream stages take this field as the authoritative source for deciding whether to enable image / table / equation analysis (`i/t/e`), whether to disable knowledge graph construction (`!`), and the chunking method (`F/R/V/P`). An empty string is equivalent to all defaults. |
| `chunk_options` | The **frozen** snapshot of chunker parameters at enqueue time (slim dictionary: only the strategy sub-dictionary selected by `process_options` is retained, others discarded). Passed in by the SDK-path caller or assembled by `resolve_chunk_options(self.addon_params, process_options=…)` from instance fields (containing env defaults) as a fallback (see §3.1). `process_options` chooses which chunking strategy (F/R/V/P); `chunk_options` decides which parameters that chunker uses. The downstream `process_single_document` reads strategy-specific kwargs from this field before chunking; persistence guarantees that old documents behave reproducibly across env changes, resumes, and restarts. Rewritten together with `process_options` when re-parsing. |

`pending_parse` indicates the file has been enqueued but extraction is not yet complete. After successful extraction, it is rewritten to `raw` or `lightrag`, and `content_hash` is filled in. On extraction failure, `pending_parse` and the empty `content` are kept, making subsequent troubleshooting and retry easier.

> The original `file_path` (with hint), `canonical_basename`, and `content_hash` are also synchronized into `doc_status`, serving as the deduplication index sources for `get_doc_by_file_basename` / `get_doc_by_content_hash`. `get_doc_by_file_basename` internally canonicalizes the input through `canonicalize_parser_hinted_basename` before comparing against `canonical_basename`, so `abc.docx` and `abc.[native-iet].docx` always hit the same document.
> `process_options` is also mirrored into `doc_status.metadata["process_options"]`, making it convenient for the management UI to directly display the current file's processing policy.

### 4.2 `__parsed__` Directory Structure

`__parsed__` is the archival and analysis-result directory next to the input directory. It both stores already-processed original documents and the `LightRAG Document` (lightrag format) files and image assets produced by structured parsing.

- Original file archival: after `legacy` local extraction succeeds and enqueueing finishes, the original file is moved into the sibling `__parsed__` directory; `native` / `mineru` / `docling` keep the original file first for the pipeline to parse, and only move it to `__parsed__` after successful parsing and writing to `full_docs`. **When archived, the original filename (including `[hint]`) is preserved**, e.g., `report.[native-iet].docx` is archived as `__parsed__/report.[native-iet].docx`, making it easy to trace the user's original name and processing options.
- Analysis result directory: structured parsing results are written into a subdirectory named with the **canonicalized filename** (with `[hint]` removed) plus the `.parsed` suffix, avoiding name conflicts with the archived original file and ensuring that the same logical document continues to point to the same directory when the filename hint or processing options change. For example, the analysis results of `report.docx`, `report.[native].docx`, and `report.[native-iet].docx` are all written into `__parsed__/report.docx.parsed/`.
- Analysis result files: the LightRAG Document blocks file and sidecars are named with the canonicalized filename stem, e.g., `__parsed__/report.docx.parsed/report.blocks.jsonl`; the same directory may also contain `report.tables.json`, `report.drawings.json`, `report.equations.json`, and the `report.blocks.assets/` image asset directory. **Whether a sidecar is generated is determined by the document content**: the parser only writes the corresponding file when the document actually contains tables / images / equations. This is the only signal of modality availability — the engine does not need to declare capabilities in meta. The `i`/`t`/`e` options only determine whether the next stage invokes the VLM for summarization analysis on already-existing sidecars.
- When parsing fails, the original file is not moved, making it easy to fix the configuration and re-process.
- When `/documents/scan` encounters a file with the same name that is already `PROCESSED`, the input file is treated as already processed and moved to `__parsed__`, not enqueued as a new document.
- When `/documents/scan` finds multiple files that share the same canonicalized name in the same scan, it prefers the file with a supported engine hint to respect the user's engine selection; if no variant has a hint, it processes the first file in sorted order. Other variants emit warnings and are moved to `__parsed__`, avoiding files in the same batch overwriting each other. For example, if both `abc.docx` and `abc.[native].docx` exist, only `abc.[native].docx` is processed.
- When duplicate content hashes are found during scanning or parsing, the input file is likewise moved to `__parsed__`; this `doc_status` entry is kept as `FAILED duplicate` for tracking.
- File moves only act on the current input file and do not overwrite or move existing document source files. If a file with the same name already exists at the destination, the system automatically appends `_001`, `_002`, etc., e.g., `report.pdf` is archived as `report_001.pdf`, `report_002.pdf`. If the analysis result directory name is already taken by a regular file, a number is also appended, e.g., `report.docx.parsed_001/`.

### 4.3 MinerU Raw Artifacts Directory `<base>.mineru_raw/`

The `mineru` engine writes the complete artifacts returned by the MinerU service (`content_list.json` + optional `full.md` / `middle.json` / `layout.pdf` / `images/`, etc.) into the `__parsed__/<canonical filename>.mineru_raw/` directory during parsing, and writes `_manifest.json` as the integrity validation file.

Design goals:

- **Avoid duplicate uploads**. When parsing the same file again, the source file's content hash + size is first validated against `_manifest.json`; on hit, the MinerU service call is skipped and the local `content_list.json` is fed directly through adapter → SidecarWriter.
- **Preserve diagnostic information**. When MinerU parses incorrectly or downstream sidecar fields are abnormal, you can go straight to `*.mineru_raw/` to compare the original content_list and image assets.
- **Support object traceability**. The `drawings.json` / `tables.json` / `equations.json` generated by MinerU save `content_list.json#/N` in `self_ref`, used for looking up the corresponding MinerU original object and its `page_idx` / `bbox`, etc.
- **De-hint uploaded filenames**. When the source filename contains processing hints like `[mineru-...]` / `[-iet]`, the MinerU API is called with the canonicalized filename (hint removed), to avoid hint-bearing filenames inside the raw bundle returned by MinerU.

Lifecycle:

| Operation | Behavior |
|---|---|
| First parse | Download all artifacts → atomically write `_manifest.json`. |
| Re-parse (cache hit) | Do not call the MinerU service; do not rewrite artifacts; rerun adapter+Writer to regenerate sidecar (for adapter upgrade scenarios). |
| Re-parse (cache miss) | Clear all files in the directory, then re-download and write manifest. |
| `DELETE /documents` with `delete_file=True` | `*.parsed/`, `*.mineru_raw/`, and the original file are all deleted together. |
| `DELETE /documents` with `delete_file=False` | All artifacts are preserved; only doc_status and KG data are deleted. |
| `clear_documents` / a full sweep of `__parsed__` | Naturally cleared together. |
| scan cycle | Does not actively GC orphan `*.mineru_raw/` (only cleared on explicit deletion by the user, to avoid accidentally removing the debug site). |

Force re-parse (bypass cache): set `LIGHTRAG_FORCE_REPARSE_MINERU=true`.

Concurrency safety: LightRAG mandates `canonical_basename` uniqueness within the same workspace (HTTP 409 on upload/enqueue), and combined with the pipeline's serialization per document, `*.mineru_raw/` has no concurrent write conflicts and needs no extra locks.

`_manifest.json` invalidation conditions (any triggers a cache miss):

- Source file size or sha256 does not match manifest;
- `MINERU_ENGINE_VERSION` environment variable and the `engine_version` recorded in manifest are both non-empty but inconsistent;
- Current `MINERU_API_MODE` and the `api_mode` recorded in manifest are both non-empty but inconsistent;
- Endpoint for the current mode (`MINERU_OFFICIAL_ENDPOINT` / `MINERU_LOCAL_ENDPOINT`) and the `endpoint_signature` recorded in manifest are both non-empty but inconsistent;
- `content_list.json` size or sha256 does not match manifest;
- Size of any recorded non-critical file (images, `middle.json`, etc.) does not match manifest.

> About the "either side empty → skip" semantics of `engine_version` / `endpoint_signature`: when the field was empty at manifest-write time (e.g., `MINERU_ENGINE_VERSION` was not configured at first parse), or when the current environment variable is not set, the check is skipped for that item. If the version env was not set at first parse, setting it later does not automatically invalidate the historical cache — this scenario requires manually setting `LIGHTRAG_FORCE_REPARSE_MINERU=true` to trigger re-parsing.

### 4.4 Docling Raw Artifacts Directory `<base>.docling_raw/`

The `docling` engine extracts the zip artifact returned by docling-serve (DoclingDocument JSON, Markdown, and referenced images) into the `__parsed__/<canonical filename>.docling_raw/` directory during parsing, and writes `_manifest.json` as the integrity validation file. On a subsequent parse, the IR builder reads the `.json` file in that directory and feeds it to `DoclingIRBuilder`, no longer calling docling-serve.

Directory layout:

```text
__parsed__/<base>.docling_raw/
├── _manifest.json
├── <base>.json        # DoclingDocument JSON (contains pages[].image base64)
├── <base>.md          # Markdown form, for human inspection
└── artifacts/
    └── image_*.png    # image assets referenced by pictures[*].image.uri
```

Design goals:

- **Avoid duplicate uploads/conversions**. When parsing the same file again, the source file's hash + size is first validated against `_manifest.json`; on hit, the upload / poll / download against docling-serve is skipped, and the local `.json` is fed directly through DoclingIRBuilder → SidecarWriter.
- **Preserve diagnostic information**. When docling-serve parses incorrectly or downstream sidecar fields are abnormal, you can go straight to `*.docling_raw/` to compare the original DoclingDocument JSON, Markdown, and `artifacts/` images.

Lifecycle:

| Operation | Behavior |
|---|---|
| First parse | `POST /v1/convert/file/async` upload → long-poll `/v1/status/poll/{task_id}?wait=N` → `GET /v1/result/{task_id}` download zip → safe extraction (rejecting absolute paths and `..`) → atomically write `_manifest.json`. |
| Re-parse (cache hit) | Do not call docling-serve; do not rewrite artifacts; rerun adapter+Writer to regenerate sidecar (for adapter upgrade scenarios). |
| Re-parse (cache miss) | Clear all files in the directory, then re-upload / download / write manifest. |
| `DELETE /documents` with `delete_file=True` | `*.parsed/`, `*.docling_raw/`, and the original file are all deleted together. |
| `DELETE /documents` with `delete_file=False` | All artifacts are preserved; only doc_status and KG data are deleted. |
| `clear_documents` / a full sweep of `__parsed__` | Naturally cleared together. |
| scan cycle | Does not actively GC orphan `*.docling_raw/` (only cleared on explicit deletion by the user, to avoid accidentally removing the debug site). |

Force re-parse (bypass cache): set `LIGHTRAG_FORCE_REPARSE_DOCLING=true`.

Concurrency safety: identical to the MinerU path — LightRAG mandates `canonical_basename` uniqueness within the same workspace (HTTP 409 on upload / enqueue), and combined with the pipeline's serialization per document, `*.docling_raw/` has no concurrent write conflicts and needs no extra locks.

`_manifest.json` invalidation conditions (any triggers a cache miss):

- Source file size or sha256 does not match manifest;
- `DOCLING_ENDPOINT` does not match the `endpoint_signature` recorded in manifest;
- `DOCLING_ENGINE_VERSION` is set and does not match the `engine_version` recorded in manifest;
- `options_signature` does not match — any OCR / equation / pipeline field change triggers it, covering:
  - Tunable env: `DOCLING_DO_OCR` / `DOCLING_FORCE_OCR` / `DOCLING_OCR_ENGINE` / `DOCLING_OCR_PRESET` / `DOCLING_OCR_LANG` / `DOCLING_DO_FORMULA_ENRICHMENT`;
  - Hard-coded constants: `pipeline` / `target_type` / `to_formats` / `image_export_mode` (written into the signature to prevent old bundles from being mistakenly reused if these values change in the future);
- Main JSON missing, size, or sha256 does not match;
- Any image in `artifacts/` missing or size mismatch;
- `LIGHTRAG_FORCE_REPARSE_DOCLING=true`.

> The "either side empty → skip" semantics of `engine_version` / `endpoint_signature` is the same as MinerU §4.3: when the field was empty at manifest-write time (first parse without `DOCLING_ENGINE_VERSION` configured) or when the current environment variable is not set, the check is skipped for that item; adding the version number later does not automatically invalidate the historical cache; `LIGHTRAG_FORCE_REPARSE_DOCLING=true` is needed to trigger.

## 5. Document Duplicate Detection Rules

File upload, file-parse enqueue, and the text APIs check duplicates against two gates: "filename + content hash". Hitting either is considered a duplicate, and a `FAILED` record is written without overwriting the existing `full_docs`. `/documents/scan` directory scanning uses the same set of indexes, but in order to facilitate automatic retry of unfinished files, it has separate archive and re-process rules for duplicate filenames.

### 5.1 Filename (basename) Deduplication

- The granularity of the check is basename, excluding directory path and workspace path. For example, `/data/a.pdf`, `inputs/a.pdf`, and `a.pdf` are all considered the same filename `a.pdf`.
- Filename deduplication uses `canonical_basename` as the index: the supported-engine processing hint at the end of the filename is stripped before comparison, so `abc.docx`, `abc.[native].docx`, and `abc.[native-iet].docx` are considered the same name. Unsupported hints are not stripped; e.g., `abc.[draft].docx` is still treated by its original filename.
- For ordinary upload, text APIs, and core enqueue APIs, as long as a file with the same name already exists in `doc_status` — whether that record is currently `PENDING`, `PARSING`, `ANALYZING`, `PROCESSING`, `FAILED`, or `PROCESSED` — the same-name file is considered a duplicate.
- For `/documents/scan` directory scan:
  - If multiple files in the same scan share the same canonicalized name, the file with a supported engine hint is processed first; if no variant has a hint, the first file after sorting is processed, and the rest are archived to `__parsed__` and skipped.
  - If the same-name record is already `PROCESSED`, the file just scanned is treated as already processed; the system emits a warning, moves the input file to the sibling `__parsed__` directory, and skips enqueueing.
  - If the same-name record is not `PROCESSED`, the scanned file is **not** skipped simply because of the same name, but **also** does not re-extract / overwrite the existing record. The specific path depends on the form of the existing record (consistent with the classification rules listed below in the "Why is scan still the exclusive writer" section):
    - Same name non-PROCESSED with `full_docs` present → **resume path**: doc_status is preserved as-is, the source file remains in `INPUT/`, and the processing loop picks it up by status query (no re-extract, no overwrite of existing status).
    - Same name `FAILED` with `full_docs` missing → recognized as an extraction-error stub written by `apipeline_enqueue_error_documents`: scan deletes the stub and **enqueues the current file as a new file**. This is the only sub-branch that re-extracts; the purpose is to make "fix the source file, scan again" automatically take effect.
- For ordinary upload and core enqueue APIs, a file with the same name — even if its content has changed — must have its old document record deleted before re-upload or re-enqueue; the two automatic recoveries above only apply to the directory-scan path.
- The text APIs must provide a valid `file_source`, and duplicates are checked by the basename of `file_source`; lacking a valid `file_source` returns 400 directly.
- When the SDK path calls `insert` / `ainsert` / `apipeline_enqueue_documents` without `file_paths`, that is allowed; related behavior is detailed in §8.4. Such documents without a source have `file_path` saved as `unknown_source`.
- Empty strings, `no-file-path`, and `unknown_source` are all considered unknown sources; they do not block new source-less text from being enqueued, nor do they deduplicate each other as same-named files.

The storage backend provides basename direct lookup via `get_doc_by_file_basename`, internally comparing against the `canonical_basename` field (the input parameter is first canonicalized through `canonicalize_parser_hinted_basename`). `JsonDocStatusStorage` already implements an in-memory traversal; other backends currently fall back to the default implementation (scanning all states and comparing `canonical_basename`), to be augmented with native indexes in subsequent PRs.

### 5.2 Content Hash Deduplication

- Documents with different filenames but identical extracted content are also considered duplicates. The hash here is the content hash of the final text or LightRAG Document obtained by the configured extraction engine; it is not the hash of the original file bytes.
- `full_docs` and `doc_status` write or fill in the `content_hash` field according to the content format:
  - `parse_format=raw`: the MD5 of the text after `sanitize_text_for_encoding`.
  - `parse_format=lightrag`: the MD5 of the `*.blocks.jsonl` file parsed out of `lightrag_document_path`. Relative paths are resolved against `INPUT_DIR`.
  - `parse_format=pending_parse`: no hash is written yet; it is filled in by subsequent steps after parsing actually completes (to avoid mistakenly judging by empty content).
- The `legacy` path deduplicates content hashes after locally extracting text and during enqueue; on hit, this record is written as `FAILED duplicate`, and no new `full_docs`, chunks, or graph data are generated.
- The `native` / `mineru` / `docling` paths first enqueue with `pending_parse`; after parsing completes and `content_hash` is filled in, if another document already has the same hash, this record is stopped before entering analysis, chunking, entity extraction, and graph writing.
- Duplicate records are marked as `filename` or `content_hash` in `metadata.duplicate_kind` for diagnosis. Content-hash duplicates also record `metadata.is_duplicate=true`, `metadata.original_doc_id`, and `metadata.original_track_id`; duplicates discovered only after parsing also have the temporarily-written `full_docs` deleted.
- Related warnings minimize repetitive noise: when scanning discovers a same-name file already `PROCESSED`, a log and pipeline status are written; duplicates at the enqueue stage use the LightRAG layer's `Duplicate document detected (...)` log; content duplicates only discovered after parsing use `Duplicate content skipped after parsing` and write a pipeline status. Scan archiving does not emit the extra `[File Extraction]Duplicate skipped`.
- The storage backend provides hash direct lookup via `get_doc_by_content_hash`; the naming convention is the same as `get_doc_by_file_basename`.

> Within an enqueue batch (the same `apipeline_enqueue_documents` call), basename and content_hash dedup are also performed; on hit, subsequent entries are written as `FAILED` directly and marked with `existing_status=batch_duplicate`. Basename dedup only applies to valid filenames; `unknown_source`, `no-file-path`, and empty sources only participate in content-hash dedup.
>
> **Cross-call concurrent dedup** is also guaranteed by the workspace-level serialization lock (see [§6.7 enqueue serialization lock (preventing concurrent dedup leakage)](#67-enqueue-serialization-lock-preventing-concurrent-dedup-leakage)): two concurrent enqueues of identical content with different filenames will not both leak past the `content_hash` check.

## 6. Pipeline Concurrency and Reentry Constraints

To prevent `scan` / `upload` / `insert` from overwriting `doc_status` / `full_docs` records of an in-flight pipeline, all write entry points coordinate via the `pipeline_status` shared dictionary. The `pipeline_status_lock` per workspace ensures that all transitions in the table below are completed atomically within the lock.

### 6.1 `pipeline_status` Fields

| Field | Semantics |
| --- | --- |
| `busy` | Generic pipeline-busy flag. Both the processing loop and destructive jobs (clear/delete) set it. **`busy=True` (processing loop) alone does not block enqueue** — the loop pulls a `doc_status` snapshot per batch and checks `request_pending` between batches for any newly arrived work. |
| `destructive_busy` | A destructive subset of `busy`: `/documents/clear` or `/documents/{doc_id}` (delete) is dropping storages / removing source files. Both reservation and the enqueue last-line guard reject — a concurrent enqueue would write to storage being torn down, and accepted documents would be silently lost. The processing loop does not set this field. |
| `scanning` | The `/documents/scan` background task is running (entire lifecycle: classification stage + processing stage). Only the `/scan` endpoint uses it to reject overlapping scans; it does **not** itself block upload/insert. |
| `scanning_exclusive` | An exclusive subset of `scanning`: True only during scan's **classification phase** — run_scanning_process is reading doc_status to classify (already processed / resume / delete stub / archive) and cannot interleave with concurrent writers. Both reservation and the enqueue last-line guard reject. After classification, the flag is cleared immediately, and concurrent uploads are allowed once scan enters the processing phase. |
| `pending_enqueues` | The number of upload/insert calls that have passed `_reserve_enqueue_slot` but whose bg task has not completed. Used only by the scan endpoint — to decide whether to take the exclusive lock. The bg task releases the slot in `finally`. |
| `request_pending` | A signal nudging the running processing loop to scan another round. Enqueue sets it after writing to `doc_status` when `busy=True`; the processing loop checks it after each batch and re-pulls the snapshot. |

### 6.2 Entry Point Behavior

| Entry point | Condition | Behavior |
| --- | --- | --- |
| `/documents/upload` / `/documents/text` / `/documents/texts` | `scanning_exclusive=True` or `destructive_busy=True` | Throw HTTP 409; do not write file, do not call enqueue |
| Same as above | Otherwise (including pure `busy=True`, scan-processing-phase `scanning=True` but `scanning_exclusive=False`) | Within the lock: `pending_enqueues++` reserves a slot → strict name precheck → save file → schedule bg task; the bg task releases the slot in `finally` |
| `/documents/scan` | `busy=True` or `scanning=True` or `pending_enqueues>0` | Emit a warning and immediately return `scanning_skipped_pipeline_busy`; do not schedule a background task |
| Same as above | All idle | Within the lock, set `scanning=True` then schedule; the task clears the flag in `finally` upon completion |
| `/documents/clear` / `/documents/delete_document` | `busy=True` or `scanning=True` or `pending_enqueues>0` | The endpoint synchronously returns `status="busy"` and does not schedule a background task |
| Same as above | All idle | The endpoint **synchronously** within the lock sets `busy=True` + `destructive_busy=True` (before `delete_document` returns `deletion_started`), and the bg task's finally clears both flags |
| `apipeline_enqueue_documents` internal (last-line guard) | `scanning_exclusive=True` and `from_scan=False`, or `destructive_busy=True` | Throw `RuntimeError("Cannot enqueue while scan is classifying / clearing or deleting")` |
| Same as above | Anything else (including pure `busy=True`, scan processing phase) | Enqueue normally; after writing `doc_status`, if `busy=True`, automatically nudge `request_pending=True` |

`from_scan=True` is a bypass for scan's own background-task enqueue: scan already holds the `scanning` flag, so it must be allowed to enqueue the files it has scanned.

### 6.3 Why `busy` no longer blocks enqueue

In the old version, `busy=True` always rejected any new enqueue, on the reasoning that "modifying `doc_status` would interleave with the pipeline worker thread." However, in practice:

1. **Write order guarantees consistency**: `apipeline_enqueue_documents` always upserts `full_docs` first, then upserts `doc_status`. The consistency check at the start of the processing loop only deletes "orphan `doc_status` rows that have no corresponding `full_docs`" — a state that cannot occur with concurrent enqueue.
2. **Batch-level snapshots**: each processing-loop batch pulls a `get_docs_by_statuses` snapshot once; newly written `PENDING` rows don't disturb the current batch, and the next round re-pulls the snapshot via `request_pending` to see the new work.
3. **`request_pending` is designed for this**: the old version already had the `request_pending` field — it was designed for "new work arrives while running" — but was gated by busy.

With this mechanism enabled in the new contract, **users can continue to upload new documents during long batch processing**, and the bg task, after writing `doc_status`, will be automatically picked up by the running loop.

### 6.4 Why scan is still the exclusive writer

scan not only enqueues the new files it finds, but also reads `doc_status` to decide what to do with each file:

- Same-name `PROCESSED` row → archive source file, skip enqueue.
- Same-name non-PROCESSED with `full_docs` present → resume path; the source file **stays in `INPUT/`**, not archived (the pending-parse parser may still need it); the processing loop picks it up by status query.
- Same-name `FAILED` with `full_docs` missing → recognized as an extraction-error stub previously written by `apipeline_enqueue_error_documents` (consistency check preserves such rows for human review); scan automatically deletes that stub and enqueues the current file as a new file, so that "fix the source file, scan again" takes effect directly.

These "read–decide–write" combinations cannot interleave with other writers; otherwise classification decisions would be based on a stale view. So scan must be exclusive, and the scan endpoint will reject when any of `busy` / `scanning` / `pending_enqueues>0` is present.

### 6.5 Strict name precheck (upload path)

After upload passes the reservation but before saving the file, a two-pass check is required:

1. **INPUT directory scan**: canonicalize the basename to be saved via `canonicalize_parser_hinted_basename`, traverse the INPUT directory for any existing same-canonical variant (with hint / without hint); 409 on hit.
2. **doc_status check**: call `get_existing_doc_by_file_basename` with the canonicalized basename; 409 on hit.

Both pass → save the file → schedule the bg task → bg task calls `apipeline_enqueue_documents` to write the store + calls `apipeline_process_enqueue_documents` to trigger processing.

> The old version once allowed upload to silently write a FAILED duplicate entry when a same-name record existed; the new rule is fail-fast, leaving no duplicate traces in doc_status. To replace a same-name document, call the `/documents/{doc_id}` delete API first.

### 6.6 Coordination of Multiple Concurrent Reservations

When two uploads arrive simultaneously (scan cannot acquire exclusivity at this time):

1. A `_reserve_enqueue_slot` → `pending_enqueues=1`, write file, schedule bg task A, return success.
2. B `_reserve_enqueue_slot` → `pending_enqueues=2`, write file, schedule bg task B, return success.
3. bg task A `apipeline_enqueue_documents` → writes `doc_status` → calls `apipeline_process_enqueue_documents` → sets `busy=True` to process A's document.
4. bg task B `apipeline_enqueue_documents` → sees `scanning=False`, writes normally; after writing, sees `busy=True`, automatically sets `request_pending=True`.
5. bg task B calls `apipeline_process_enqueue_documents` → sees `busy=True`, sets `request_pending=True` and returns immediately.
6. A's processing loop finishes the current batch, sees `request_pending=True`, re-pulls the snapshot, and picks up B's `PENDING` row.
7. After all is complete: `busy=False`, `pending_enqueues=0`.

No bg task will be falsely rejected due to busy — because enqueue no longer checks busy; the processing loop will not process the same batch repeatedly — because `request_pending` only takes effect between batches and is cleared before each re-pull.

### 6.7 enqueue Serialization Lock (Preventing Concurrent Dedup Leakage)

Inside `apipeline_enqueue_documents`, "read doc_status to dedupe → write `full_docs` / `doc_status`" runs serially under the workspace-level `enqueue_serialize` lock. Reason: now that concurrent enqueue is allowed during the busy/scan-processing phases, two enqueues with identical content but different filenames (typical scenario: a scan-processing-phase enqueue and an upload arriving together) would, without the lock, race as follows —

1. A reads `doc_status` to check `content_hash`: miss.
2. B reads `doc_status` to check `content_hash`: still miss (A hasn't upserted yet).
3. A upserts `full_docs` + `doc_status`.
4. B upserts `full_docs` + `doc_status`.

Result: both `PENDING` rows with the same `content_hash` enter the downstream pipeline, and the row that should have been identified as `duplicate_kind=content_hash` was **not** identified.

With the serialization lock, the second enqueue's dedup read is guaranteed to see the row already upserted by the first, taking the normal "no new unique document" early-return path and writing this run as a `duplicate_kind=content_hash` FAILED row. The lock only covers:

- `filter_keys` (exclude existing by doc_id)
- Filename / content hash dedup reads
- Upsert of duplicate FAILED rows
- `full_docs.upsert` + `doc_status.upsert`

The lock does **not** cover the `request_pending` nudge (outside the lock; only briefly takes `pipeline_status_lock`), and does **not** block the `get_docs_by_statuses` read of the processing loop (which goes through `doc_status`'s own concurrent reads — a KV-level atomic with the enqueue writes, not contending for the same lock). Lock order: `enqueue_serialize → pipeline_status_lock`; no deadlock path.

### 6.8 Pipeline Concurrency Parameters

The locks around `pipeline_status` solve the correctness problem of "who can write"; this section's set of parameters solves the throughput problem of "how many workers run concurrently". The pipeline is divided into 3 stages, each with an independently tunable worker pool:

```
          ┌─ q_native  ──► [native parser  × N1] ─┐
PENDING ─►├─ q_mineru  ──► [mineru parser  × N2] ─┼─► q_analyze ─►[analyzer × N4] ─► q_process ─►[processor × N5]
          └─ q_docling ──► [docling parser × N3] ─┘
```

At enqueue time, `resolve_stored_document_parser_engine` puts each document into the corresponding parse queue based on its `parser_engine` (from `LIGHTRAG_PARSER` defaults or the filename hint); the three parse queues are **completely non-blocking** with respect to each other — mineru saturation does not slow down docling or native. After parsing, they enter `q_analyze` (multimodal analysis) uniformly, and then enter `q_process` (entity/relation extraction + ingest).

| Environment variable | Default | Effect | Tuning advice |
| --- | --- | --- | --- |
| `MAX_PARALLEL_PARSE_NATIVE` | `5` | N1: number of concurrent workers for native parsing (docx / pdf / txt and other pure local processing) | Pure CPU, low memory usage; can be raised to CPU core count |
| `MAX_PARALLEL_PARSE_MINERU` | `2` | N2: number of concurrent workers for MinerU parsing | MinerU has significant GPU/CPU usage; **the default of 2 is a modest amount of parallelism**. Lower to 1 when resources are tight; with local deployment and ample VRAM, you can set 2–3; when going through MinerU's official cloud service, you can raise it appropriately (subject to cloud quotas). |
| `MAX_PARALLEL_PARSE_DOCLING` | `2` | N3: number of concurrent workers for Docling parsing | Docling is similarly resource-sensitive; **the default of 2 is a modest amount of parallelism**. Lower to 1 when resources are tight; with local deployment and ample CPU/GPU, you can set 2–3. |
| `MAX_PARALLEL_ANALYZE` | `5` | N4: number of concurrent workers for multimodal analysis (VLM image / table description) | Directly consumes the VLM quota. Recommended ≤ VLM service concurrency cap. |
| `MAX_PARALLEL_INSERT` | `3` | N5: number of concurrent documents at the entity / relation extraction + ingest stage | Recommended `MAX_ASYNC / 3`, in the range 2–10. This stage triggers multiple LLM calls per document; setting it too high will hit LLM rate limits. This value also serves as the `asyncio.Semaphore` for an additional constraint (worker count and semaphore value are the same). |
| `QUEUE_SIZE_DEFAULT` | `100` | Bounded queue capacity between the parse / analyze stages | Generally no need to tune. For very large batches (thousands or more), can be raised to avoid backpressure at the enqueue side; lower it when memory is tight. |
| `QUEUE_SIZE_INSERT` | `4` | Queue capacity between the analyze → process stage | The process stage is the slowest and most memory-hungry in the pipeline; the queue is deliberately small to provide backpressure to upstream and prevent memory bloat. |

**Several key points:**

1. **Parsing stage is isolated per engine**, so when mixing native/mineru/docling, you don't have to worry about a slow engine dragging another down.
2. **mineru / docling default to 2**: both have high resource usage, so the default keeps parallelism modest. Lower to 1 when resources are tight (OOM / VRAM contention / failure retry); with multi-GPU or a dedicated parser server, you can raise them manually.
3. **`MAX_PARALLEL_INSERT` doubles as worker pool size and semaphore cap**: the pipeline creates a `Semaphore(max_parallel_insert)`, and each process worker also takes the semaphore before extraction and ingest. So even if you manually raise the worker count, the actual concurrency cap is still bounded by this value — just tune it directly.
4. **Queue size and backpressure**: the small default `QUEUE_SIZE_INSERT=4` is intentional — the process stage is slow and memory-hungry; when the queue fills, analyze blocks, and backpressure reaches the parse stage, preventing thousands of parsing results from piling up in memory at once.
5. **How changes take effect**: all parameters are passed in via `.env` (or environment variables), read once at `LightRAG` construction; restart the service after changing them.

**Typical tuning scenarios:**

- Large batch of PDFs + local MinerU on a single GPU: `MAX_PARALLEL_PARSE_MINERU=2`, `MAX_PARALLEL_ANALYZE=5`, `MAX_PARALLEL_INSERT=2` (defaults are fine; lower MINERU to 1 if VRAM is tight).
- Large batch of PDFs + MinerU cloud service: `MAX_PARALLEL_PARSE_MINERU=3~5` (depending on cloud quota), others at defaults.
- Pure docx / txt (only native): `MAX_PARALLEL_PARSE_NATIVE=10`; `MAX_PARALLEL_INSERT` derived from `MAX_ASYNC/3`.
- Heavy LLM rate-limiting: first lower `MAX_PARALLEL_INSERT` (the process stage makes multiple LLM calls per document), then lower `MAX_PARALLEL_ANALYZE` (VLM is a separate quota).

## 7. Pipeline Resume Rules at Startup

Each time `apipeline_process_enqueue_documents` starts up, it pulls all documents in `PARSING` / `ANALYZING` / `PROCESSING` / `PENDING` / `FAILED` to continue processing. The resume path **branches by "whether content has been extracted"**, ensuring that any document, regardless of its previous progress, has an idempotent result when resumed under the current `process_options`.

The resume rule only applies to documents whose `doc_id` already exists in `doc_status`. New files joining the queue require the file dedup logic in "Concurrency and Reentry Constraints", to avoid new files squeezing out the records of files whose content has already been successfully extracted.

### 7.1 Determining "Content Has Been Extracted"

Read `full_docs[doc_id]`:

| `parse_format` | Verdict |
| --- | --- |
| `lightrag` and `lightrag_document_path` file exists | ✅ extracted |
| `raw` and `content` is non-empty | ✅ extracted |
| Other (including `pending_parse`, missing record) | ❌ not extracted |

### 7.2 Branch A: Not Extracted

Go through the full pipeline (`parse_native` / `parse_mineru` / `parse_docling` → `analyze_multimodal` → chunking → entity extraction), with each stage's behavior determined by `full_docs.process_options`. This is the normal flow of a "first-time enqueue".

### 7.3 Branch B: Already Extracted

**Always skip parsing** (do not call `parse_*` again), restart from the ANALYZING stage, clear old chunks / entities, and redo per the current `process_options`:

| Sub-step | Behavior |
| --- | --- |
| Engine comparison | If the engine implied by `process_options` ≠ `full_docs.parse_engine`, **only warn**, do not re-parse. The extracted content is an immutable fact; re-running a different engine would produce inconsistency. To switch engines, delete the whole document and re-upload it. |
| Old chunks / entities / relations cleanup | Read `status_doc.chunks_list` to collect old chunk id set, call `_purge_doc_chunks_and_kg(doc_id, chunk_ids)`: delete chunk rows from `chunks_vdb` / `text_chunks`; reverse-lookup affected entities / relations by `entity_chunks` / `relation_chunks`, directly remove entries that have lost all sources from the graph and vector store, and call `rebuild_knowledge_from_chunks` to rebuild with the remaining chunks for entries still contributed by other documents; finally delete the index rows of this doc in `full_entities` / `full_relations`. After purge completes, `status_doc.chunks_list = []` / `chunks_count = 0` are reset to avoid the subsequent state-machine upsert writing back old IDs. |
| `analyze_multimodal` | For enabled modalities, every run recomputes the sidecar item analysis and overwrites the existing `llm_analyze_result`. The LLM analysis cache still applies: a cache hit reuses the previous provider response, so semantic fields usually stay the same and only runtime fields such as `analyze_time` are rewritten. Cache misses, for example after changing the model or prompt, can produce different saved content. |
| Re-chunk | Pick the strategy by the new `process_options.chunking`, with parameters read from `full_docs.chunk_options` (the enqueue snapshot; not overwritten by resume; env changes do not affect old documents that still chunk by the parameters from the moment of enqueue). The LightRAG Document path uses paragraph_semantic when `process_options=P`, otherwise dispatches to F/R/V by selector. |
| Entity extraction / KG-skip | Determined by the new `process_options.skip_kg` |

> This rule guarantees: when users change `i/t/e` and re-upload the same-named document (delete the old doc first, then upload the file with the new hint), multimodal analysis is incrementally filled in; when changing `F/R/V/P`, chunks and graph are rebuilt; when changing `!`, KG construction is stopped or restored. Engine changes are considered a "major change", uniformly handled by delete + re-upload, not implicitly happening on the resume path.

## 8. Python SDK Invocation

This chapter targets developers who **directly import the `LightRAG` class** for integration, covering runtime APIs, constructor parameters, and removed legacy interfaces that Server deployments don't use. Server users usually don't need to read this chapter.

### 8.1 Audience

```python
from lightrag import LightRAG
rag = LightRAG(working_dir="./rag_storage", ...)
await rag.initialize_storages()
await rag.ainsert("text", file_paths="doc.pdf")
```

The following behaviors of this invocation style differ from the Server path: you can change `addon_params["chunker"]` without restarting the process, you can pass per-file `chunk_options` into `apipeline_enqueue_documents`, and you can dynamically override the F strategy's pre-split parameters in an `ainsert` call.

### 8.2 LightRAG Constructor Parameters

`LightRAG(chunk_token_size=…, chunk_overlap_token_size=…)` is **tier 3** in §3.3's priority chain: "legacy constructor field". Strategy-agnostic and coarse-grained default, fills only slots still empty:

- Lower priority than `addon_params["chunker"]` explicit values (§8.3) and strategy-specific env (§3.2).
- Higher priority than the legacy env `CHUNK_SIZE` / `CHUNK_OVERLAP_SIZE`.
- The instance fields `self.chunk_token_size` / `self.chunk_overlap_token_size` are always back-filled to `int` after `__post_init__`, so legacy paths still reading these two fields (e.g., the `chunk_opts.get("chunk_token_size") or self.chunk_token_size` fallback in `pipeline.py`) continue to work.

### 8.3 Modifying `addon_params["chunker"]` at Runtime

`addon_params["chunker"]` is an `ObservableAddonParams` field; it can be **modified at runtime**:

```python
rag.addon_params["chunker"]["recursive_character"]["separators"] = ["##", "\n", " "]
```

After modification, **subsequent enqueues** get the new defaults; already-enqueued documents keep the snapshot from their enqueue moment (see the three layers of semantic guarantee in §3.3). This is tier 1 of §3.3's priority chain: "`addon_params["chunker"]` explicit value", winning everything.

Server deployments do not have this capability — after changing env, the service must be restarted for it to take effect.

### 8.4 `apipeline_enqueue_documents(chunk_options=…)`

`apipeline_enqueue_documents` accepts an optional `chunk_options` argument. When the caller passes a `dict` / `list[dict]`, it is projected by the current document's `process_options` into a slim snapshot (keeping only the corresponding strategy sub-dictionary + top-level `chunk_token_size`) before being persisted to `full_docs[doc_id]["chunk_options"]`; when not passed, `resolve_chunk_options(self.addon_params, process_options=…)` assembles one on the spot. Callers can safely pass the full dictionary — the other strategies' sub-dictionaries will be discarded by the dispatcher and won't pollute the store.

Typical usage:

```python
await rag.apipeline_enqueue_documents(
    input=["text A", "text B"],
    file_paths=["a.[native-R].txt", "b.txt"],
    process_options=["R", ""],
    chunk_options=[
        {"chunk_token_size": 800, "recursive_character": {"separators": ["\n\n", "\n"]}},
        {"chunk_token_size": 1500},
    ],
)
```

Typical scenarios for per-file personalization: a management UI configures separators or V threshold individually for a certain file; in the future, upload APIs may also accept overrides in form / hint.

**Compatibility for not passing `file_paths`**: the core APIs `insert` / `ainsert` / `apipeline_enqueue_documents` still support invocations without `file_paths`; the `file_path` of such documents is saved as `unknown_source`, does not participate in filename dedup, and the document ID continues to be generated from text content.

For `apipeline_enqueue_documents`'s own concurrency constraints (last-line guard, `from_scan=True` bypass), see the entry-point behavior table in §6.2.

### 8.5 `ainsert(split_by_character=…, split_by_character_only=…)`

`LightRAG.ainsert(split_by_character=…, split_by_character_only=…)` runtime parameters are overridden into `chunk_options.fixed_token` by `resolve_chunk_options` at enqueue time:

- A non-`None` `split_by_character` overrides the env default;
- `split_by_character_only=True` overrides (`False` is the signature default, indistinguishable from "not specified", so the env default wins).

Only effective for the F strategy; other strategies' sub-dictionaries are unaffected.

### 8.6 Removed SDK Parameter: `reprocess_existing_non_processed`

The legacy `apipeline_enqueue_documents` behavior of `reprocess_existing_non_processed=True` would directly delete non-PROCESSED old records and rebuild them during scan, which conflicts with the rules in §5 / §6; it has been entirely removed. Replacement paths:

- Automatic resume: scan handles same-named files per the classification rules in §6.4 (archive / resume / delete stub then re-enqueue), uniformly picked up by the resume rules in §7 inside the processing loop.
- Forced refresh: first call `/documents/{doc_id}` to delete the old document, then upload the same-named new file.
