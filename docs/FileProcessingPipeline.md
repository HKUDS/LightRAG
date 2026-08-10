# File Processing Pipeline Specification

Starting from version v1.5.0 (currently on the dev branch), LightRAG's file processing pipeline has received a major upgrade:

* Supports multiple file content extraction engines: legacy, native, mineru, docling
* Supports multiple text chunking methods: Fix, Recursive, Vector, Paragraph
* Supports disabling entity-relation extraction for individual files

LightRAG Server introduces an intermediate file-processing format: `LightRAG Document`. This format supports multimodal data such as tables and images, and also includes the document's section/paragraph metadata, which is convenient for content traceability later.

This document is organized from the perspective of **LightRAG Server** deployment and use: the quick-start configuration that can be applied directly is given first, followed by configuration syntax for content extraction and chunking, storage / directory layout, deduplication, concurrency, and resume rules. Developers who call the `LightRAG` class directly via Python should jump to [Chapter 11: Python SDK Invocation](#11-python-sdk-invocation).

## Table of Contents

- [1. Quick Start](#1-quick-start)
- [2. Processing Options and Configuration Syntax](#2-processing-options-and-configuration-syntax)
- [3. File Parsing Engines](#3-file-parsing-engines)
- [4. Multimodal Analysis (VLM)](#4-multimodal-analysis-vlm)
- [5. Chunker Parameter Configuration (chunk_options)](#5-chunker-parameter-configuration-chunk_options)
- [6. Storage and Directory Layout](#6-storage-and-directory-layout)
- [7. Same-name and Duplicate Documents](#7-same-name-and-duplicate-documents)
- [8. Operating the Pipeline: Uploading While It Runs, Stopping, Retrying](#8-operating-the-pipeline-uploading-while-it-runs-stopping-retrying)
- [9. Pipeline Resume Rules at Startup](#9-pipeline-resume-rules-at-startup)
- [10. Troubleshooting](#10-troubleshooting)
- [11. Python SDK Invocation](#11-python-sdk-invocation)
- [Appendix A. Notes on Upgrading from Legacy](#appendix-a-notes-on-upgrading-from-legacy)
- [Appendix B. Environment Variable Quick Reference](#appendix-b-environment-variable-quick-reference)
## 1. Quick Start

### 1.1 Keep the legacy file-processing behavior

All files are processed using the legacy document parsing and chunking strategy. Either leave `LIGHTRAG_PARSER` unconfigured, or set it to the following value:

```bash
LIGHTRAG_PARSER=*:legacy-F
```

### 1.2 Recommended starting file-processing behavior

No reliance on external document parsing services or on `VLM` vision models. Use the new built-in `Native` engine to parse `docx` documents with table (t) and equation (e) modality analysis enabled, paired with the `P` chunking strategy; other documents use the legacy content extractor paired with the more effective `R` chunking strategy.

```bash
LIGHTRAG_PARSER=*:native-teP,*:legacy-R
```

### 1.3 Enable multimodal processing capability

Enabling multimodal processing requires the `MinerU` file parsing service and a `VLM` vision recognition model. Use `Native` to parse `docx` files; use `MinerU` to parse `pdf`, `office`, and various image files. All of the above files have image (i), table (t), and equation (e) modality analysis enabled and are paired with the `P` chunking strategy. Other documents fall back to the legacy content extractor paired with the `R` chunking strategy.

```bash
LIGHTRAG_PARSER=*:native-iteP,*:mineru-iteP,*:legacy-R
VLM_PROCESS_ENABLE=true
VLM_LLM_MODEL=kimi-k2.6
MINERU_API_MODE=local
MINERU_LOCAL_ENDPOINT=http://localhost:8000
```

> `P` is LightRAG's native chunking strategy; see [Paragraph Semantic Chunking](ParagraphSemanticChunking.md) for details. For VLM configuration, see [Role-based LLM/VLM Configuration Guide](RoleSpecificLLMConfiguration.md).


### 1.4 Reading a preset

Each comma-separated item of `LIGHTRAG_PARSER` is one routing rule, matched against the file extension from left to right; the first usable rule wins. Taking `*:native-iteP,*:legacy-R` apart:

```
*  :  native  -  iteP
│     │          │
│     │          └─ processing options: i/t/e enable image, table and equation
│     │             analysis; P selects the paragraph-semantic chunker
│     └─ content extraction engine
└─ extension the rule applies to; `*` matches any
```

So the first rule says "parse anything with `native`, analyze all three modalities, chunk with `P`", and the second says "anything the first rule cannot handle falls back to `legacy` with `R` chunking". A rule is skipped when the engine does not support that extension, or when an external engine has no endpoint configured — which is exactly how `*:native-iteP` ends up handling only `docx` / `md` / `textpack` while everything else drops through to `legacy`.

The full grammar is in §2.3, the option letters in §2.1, and the engines in §3.

### 1.5 First run: ingest one document and verify it worked

A dependency-free smoke test — one `.docx`, no MinerU, no docling, no VLM:

```bash
LIGHTRAG_PARSER=*:native-teP,*:legacy-R
```

**1. Check the routing before spending a parse.** A malformed `LIGHTRAG_PARSER` already fails startup (§2.7). Once the server is up, `GET /documents/supported_file_types` returns the extension allowlist and the engine-to-suffix mapping **as the server actually resolved them**, which answers "did my rule take effect" without ingesting anything.

**2. Get the file in.** Either `POST /documents/upload` (the file is stored under `INPUT_DIR` and the response carries a `track_id`), or drop files into `INPUT_DIR` and call `POST /documents/scan`. To try different settings on a single file without touching `.env`, rename it with a hint: `report.[native-teP].docx` (§2.5).

**3. Watch it.** `GET /documents/track_status/{track_id}` for that upload, `GET /documents/pipeline_status` for the live log, `GET /documents/status_counts` for the batch view. The status ladder is `PENDING → PARSING → ANALYZING → PROCESSING → PROCESSED`.

**4. Find the artifacts.** Everything lands under `INPUT_DIR/__parsed__/` (§6.2). For `report.[native-teP].docx`:

```
__parsed__/report.[native-teP].docx   # archived original, hint preserved
__parsed__/report.docx.parsed/        # canonical name, hint stripped
    report.blocks.jsonl               # first line is meta, the rest are blocks
    report.tables.json                # only if the document has tables
    report.equations.json             # only if the document has equations
    report.blocks.assets/             # exported images
```

**5. Verify the parse**, in this order:

1. The first line of `*.blocks.jsonl` is a `meta` record.
2. Content lines carry a sensible heading structure — this is what `P` splits on.
3. The sidecars you expected exist. A missing one means the document has no such content, or the engine does not emit it (`legacy` emits none at all — §3.2).
4. Each sidecar item you enabled analysis for carries `"llm_analyze_result": {"status": "success"}`. A `skipped` or `failure` here is the answer to "why is my image/table missing from the knowledge graph" (§4.3).

**6. Verify chunking and ingest.** `chunks_count` in `GET /documents/paginated` shows the chunk count; the doc-status record's `metadata.parse_engine` and `metadata.process_options` show **which engine and options actually ran**, which is the fastest way to catch a hint that was silently invalidated (§2.7).

**If it failed:** the source file **stays in `INPUT_DIR`**, so you can fix the configuration and scan again. `error_msg` on the doc-status record is the diagnosis; §10 maps the common ones. Note that both the engine and the processing options are frozen at enqueue time — changing `LIGHTRAG_PARSER` or a hint does not affect an existing record, so re-running with different settings means deleting the document and uploading it again (§9.3). To reproduce a parse offline without the server, use `python -m lightrag.parser.cli` ([ParserDebugCLI.md](./ParserDebugCLI.md)).

## 2. Processing Options and Configuration Syntax

LightRAG's file processing configuration is composed of two parts: the content extraction engine determines how the original file is parsed, and the processing options determine whether multimodal analysis is performed after parsing, which chunking method to use, and whether to build a knowledge graph. Typically, the environment variable `LIGHTRAG_PARSER` is first used to set default rules by file extension, and then a `[hint]` in the filename overrides individual files. Engine and options can be written in the same configuration fragment, for example `docx:native-iet` or `report.[native-R!].docx`.

For backward compatibility, when the configuration is not modified, the upgraded file content extraction behavior remains the original `legacy` behavior. To enable the new content processing engines, configure as described in this chapter; each engine's own capabilities and settings are in §3.

### 2.1 File Processing Options

Processing options control, on a per-file basis, the behavior with respect to multimodal analysis, knowledge graph construction, and text chunking. They can be set as per-rule defaults in `LIGHTRAG_PARSER` (see [§2.4](#24-default-rules-lightrag_parser)) or overridden for an individual file via a filename hint (see [§2.5](#25-single-file-override-filename-hints)). All options are optional; defaults are shown in the table below. At most one chunking method (F/R/V/P) is specified per file; the other options can be combined arbitrarily.

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

### 2.2 Option Effective Stages

Different characters of processing options take effect at different stages of the pipeline:

| Option | Stage | Description |
| :-: | --- | --- |
| i/t/e | Analyzing (multimodal analysis) | Determines whether VLM summarization analysis is invoked on the images / tables / equations in the sidecar. **The extraction stage is unaffected**: the content extraction engine outputs `drawings.json` / `tables.json` / `equations.json` sidecar files based on what the document actually contains. As a result, simply tweaking the `i`/`t`/`e` options to trigger "re-analysis" can complete VLM later without re-parsing the original file. |
| ! | Extraction (entity-relation extraction) | Skips entity/relation extraction and graph writing; chunks are still written to the vector store to retain naive / mix retrieval capabilities. |
| F/R/V/P | Chunking (text chunking) | Determines which chunking strategy to use; does not affect the output of the parsing stage. |

> Modality availability is signaled solely by "whether the sidecar file exists"; the content extraction engine does not need to declare its capabilities in meta. If a given document contains no images/tables/equations, the corresponding sidecar is not written; even if the user has enabled `i/t/e`, the corresponding modality is silently skipped, but `analyze_multimodal` logs an INFO-level line for that document (`[analyze_multimodal] sidecar e:equations empty: doc—id ...`), making it easy to diagnose "why didn't the VLM run". This is not an error.

### 2.3 Configuration Syntax Overview

The complete configuration model is as follows:

```text
LIGHTRAG_PARSER=ext:engine-options,ext:engine,*:legacy-R
filename.[ENGINE].ext
filename.[ENGINE-OPTIONS].ext
filename.[-OPTIONS].ext
```

- `LIGHTRAG_PARSER` is the default rule table, matched by file extension, e.g., `pdf:mineru`, `docx:native-iet`.
- The `[hint]` in a filename is a single-file override rule, e.g., `paper.[mineru].pdf`, `memo.[native-R!].docx`.
- `ENGINE` is the content extraction engine. Which one you pick decides what the parse stage can produce at all:

  | Engine | What it is | Reach for it when |
  | --- | --- | --- |
  | `legacy` | plain-text extraction, no sidecars | you want the pre-upgrade behavior, or the document is plain text anyway |
  | `native` | built-in structured extractor, fully local, no external service | `docx` / `md` / `textpack`, and you want `P` chunking or modality analysis without deploying anything |
  | `mineru` | external MinerU service | PDFs, scanned documents, and office / image formats that need layout and OCR |
  | `docling` | external docling-serve | an alternative to MinerU; the only built-in path to LaTeX equations |

  Only `native`, `mineru` and `docling` write sidecars, so only they can serve the `i` / `t` / `e` options and the `P` chunker. Per-engine capabilities, configuration and deployment are in [§3](#3-file-parsing-engines).
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

### 2.4 Default Rules: `LIGHTRAG_PARSER`

`LIGHTRAG_PARSER` is used to configure the default content extraction engine for different file extensions; default processing options for the rule can also be appended after the engine:

```text
ext:engine,ext:engine,*:legacy
ext:engine;ext:engine;*:legacy
ext:engine-options
```

- The left side matches the file extension, not the full filename; write `pdf:mineru`, not `*.pdf:mineru`.
- Rules are separated by a semicolon `;` (recommended) or a comma `,`.
- Rules are checked left to right; priority rules go in front, with the wildcard rule typically at the end.
- The `-options` suffix after the engine serves as the default `process_options` for files matched by this rule. For example, `LIGHTRAG_PARSER=docx:native-iet` means all `.docx` files default to the `native` engine with image, table, and equation analysis enabled.

### 2.5 Single-File Override: filename hints

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

### 2.6 Attaching chunk parameters

A chunk-strategy selector (`F` / `R` / `V` / `P`) — in a `LIGHTRAG_PARSER` rule or a filename hint — may carry per-strategy chunking parameters in parentheses. Inside the parentheses a comma **only** separates parameters; rule splitting is parenthesis-aware, so this comma is never mistaken for a rule separator (both `;` and `,` remain valid rule separators, but `;` is recommended).

```text
notes.[-R(chunk_ts=800,chunk_ol=80)].md                            # filename hint
LIGHTRAG_PARSER=pdf:legacy-R(chunk_ts=800,chunk_ol=80);*:legacy-R  # rule
```

Currently supported parameters (canonical name / short alias):

| Parameter | Alias | Strategies | Type | Meaning |
| --- | --- | --- | --- | --- |
| `chunk_token_size` | `chunk_ts` | F / R / V / P | int (≥ 1) | Per-strategy chunk size |
| `chunk_overlap_token_size` | `chunk_ol` | F / R / P | int (≥ 0) | Overlap between chunks (V has no overlap) |
| `drop_references` | `drop_rf` | P | bool | Drop matching reference blocks before chunking, e.g. `paper.[-P(drop_rf=true)].pdf`. As a boolean it may be written bare: `paper.[-P(drop_rf)].pdf` means `drop_rf=true` |

- `process_options` stays a pure selector string; each parameter is applied to that strategy's `chunk_options` (see §5) while the strategy's other env-derived parameters are kept. Aliases are normalized to their canonical name internally.
- Merge priority: the selector still follows "a non-empty filename-hint options string wholesale-overrides the rule options"; parameters overlay **per strategy** — rule parameters first, then filename-hint parameters (filename wins on a shared key).
- Validation is strict both at startup (`LIGHTRAG_PARSER`) and at upload (filename hint): an unknown parameter, a wrong type, an out-of-range value, or a parameter on a strategy that does not support it (e.g. `chunk_ol` on `V`) all raise a friendly error.

> `drop_references` detection knobs `CHUNK_P_REFERENCES_TAIL_N` (default `0`: scan all content blocks; a positive value scans only the last N) / `CHUNK_P_REFERENCES_HEADINGS` (pipe-separated, default `References\|Bibliography\|参考文献`) are env-only and read live at run time. Global default can be set via env var `CHUNK_P_DROP_REFERENCES`.

### 2.7 Validation, Priority, and Fallback

- `LIGHTRAG_PARSER` is strictly validated at startup: unknown content extraction engines, malformed extension syntax, explicitly using an unsupported extension, external engines missing endpoint, and illegal characters in processing options all cause startup to fail.
- **When a wildcard rule matches a certain extension**, the engine must pass two usability checks (see `parser_routing._engine_is_usable`): (a) the engine's capability table supports that extension; (b) if it is an external engine (`mineru` / `docling`), the corresponding endpoint/token environment variable is configured. If either check fails, the rule is skipped and the next rule is matched. For example, in `*:mineru;html:docling`: MinerU does not support the `html` extension (condition a fails), so `html` continues to match `docling`; if `MINERU_API_MODE=local` but `MINERU_LOCAL_ENDPOINT` is not set, all PDFs also skip `*:mineru` and fall to the next rule (condition b fails). This behavior applies to both `LIGHTRAG_PARSER` rule matching and filename hint engine selection.
- Filename hints have higher priority than `LIGHTRAG_PARSER`. If the engine specified in a hint does not support that extension, the system falls back to the default rules to continue selecting an available engine.
- If the filename hint provides a non-empty options string, the hint takes precedence; otherwise the default options of the matching item in `LIGHTRAG_PARSER` are used; if neither is provided, all defaults are used.
- If no rule is available, the file content extraction falls back to `legacy`; if `legacy` also does not support the file extension, an error entry is added to the system and the uploaded file remains in the `INPUT` directory.
- At most one of F/R/V/P may appear; repeating the same option has effect only once but does not raise an error.
- Case-sensitive: the chunking options F/R/V/P must be uppercase; other options i/t/e must be lowercase.
- If illegal characters appear inside the square brackets, the entire hint is invalidated, the engine follows the default rules, and the options fall back to `LIGHTRAG_PARSER` defaults or all defaults; a warning is also logged.
- `P` is only effective for structured `LightRAG Document` results extracted by `native`; for the `legacy` path or unstructured output, it automatically degrades to `R` and logs a warning.

## 3. File Parsing Engines

### 3.1 Engine Capability Matrix

| Engine | Description | Supported file formats (extensions) |
| --- | --- | --- |
| `legacy` | Legacy extraction; content is centrally extracted before joining the pipeline | `txt` `md` `mdx` `pdf` `docx` `pptx` `xlsx` `rtf` `odt` `tex` `epub` `html` `htm` `csv` `json` `xml` `yaml` `yml` `log` `conf` `ini` `properties` `sql` `bat` `sh` `c` `h` `cpp` `hpp` `py` `java` `js` `ts` `swift` `go` `rb` `php` `css` `scss` `less` |
| `native` | Built-in intelligent structured content extractor | `docx` `md` `textpack` |
| `mineru` | External MinerU content extraction engine | `pdf` `docx` `pptx` `xlsx` `png` `jpg` `jpeg` `jp2` `webp` `gif` `bmp` (extensible, see `MINERU_ADDITIONAL_SUFFIXES`) |
| `docling` | External Docling content extraction engine | `pdf` `docx` `pptx` `xlsx` `md` `html` `xhtml` `png` `jpg` `jpeg` `tiff` `webp` `bmp` (extensible, see `DOCLING_ADDITIONAL_SUFFIXES`) |

`mineru` and `docling` are external content extraction engines; before enabling related rules, the services must be running first, and the corresponding endpoint/token must be configured in LightRAG.

For both external engines the row above is the **baseline** set — what the engine handles out of the box. Their remaining input formats (legacy Office `doc` / `xls` / `ppt`, and for docling also ODF, EPUB, AsciiDoc, LaTeX, CSV, …) depend on components installed on the *service* side rather than in LightRAG — legacy Office conversion requires LibreOffice there — and for MinerU also on the active `MINERU_API_MODE`. They are therefore not advertised globally: declare what your own deployment can actually handle with `MINERU_ADDITIONAL_SUFFIXES` / `DOCLING_ADDITIONAL_SUFFIXES` (see each engine's section below and the MinerU / Docling blocks in [env.example](https://github.com/HKUDS/LightRAG/blob/main/env.example)), then route those suffixes with a `LIGHTRAG_PARSER` rule or a per-file hint — declaring a suffix alone does not make a bare `x.doc` uploadable.

LightRAG caches the parsing results of the `mineru` and `docling` engines locally. Re-uploading the same file usually does not trigger the engine to re-parse the document. To delete the parse cache, you must click the "also delete file" option in the delete-file dialog of the document management interface. Modifying the endpoint addresses and effective extraction parameters of the `mineru` / `docling` engines will also invalidate the cache, causing the engine to re-parse the file content on the next upload of the same file.

### 3.2 Using the legacy Content Extractor

`legacy` is the fallback engine for every extension except `.textpack`, which the routing layer always sends to `native`. It is also what you get with `LIGHTRAG_PARSER` unset. It extracts plain text only, which has four consequences worth knowing before you route anything to it:

- **It never writes sidecars.** Its output is `parse_format=raw`, so there is no `drawings.json` / `tables.json` / `equations.json` for the analyze stage to read. The `i` / `t` / `e` options therefore do nothing on a legacy-parsed document, and `P` degrades to `R` because there is no heading structure to split on (§2.7).
- **It has no raw cache directory**, so `LIGHTRAG_FORCE_REPARSE_*` does not apply to it (§3.7).
- **Its only configuration knob is `PDF_DECRYPT_PASSWORD`**, the password used to open protected PDFs.
- **A scanned PDF with no text layer fails hard** rather than producing an empty document. Route such files to `mineru` or `docling`, which can OCR them.

### 3.3 Using the Native File Parsing Engine

`native` is LightRAG's built-in structured content extractor that runs **fully locally**: it does not depend on external services such as MinerU / Docling, the extraction stage never calls a VLM, and it works out of the box with no deployment. Its runtime dependencies are only `python-docx` + `defusedxml` (required); the markdown path additionally relies on the **optional** `cairosvg` for SVG rasterization (when missing, the SVG is skipped with a warning and the rest of the content is unaffected). Enabling the opt-in `smart_heading` engine parameter for docx additionally requires the pinned `zh_core_web_sm` / `en_core_web_sm` spaCy models (the `spacy` runtime ships with the `api` extra; install the models with `lightrag-download-cache --spacy-install` — the main Docker image already bundles them); deployments that never enable it need no models, and the smart_heading path also calls the EXTRACT-role LLM during parsing. Setting the `DOCX_SMART_HEADING=true` env var enables smart_heading by default for `.docx` files that resolve to the native engine — an explicit `native(smart_heading=false)` rule/hint opts a file back out — and makes the server verify the spaCy models at startup (fail fast instead of failing on the first parse); the same startup check triggers when a `LIGHTRAG_PARSER` rule carries `native(smart_heading=true)` (or its flag shorthand `native(smart_heading)`). The default applies at upload time only: already-ingested documents keep their persisted engine parameters on re-parse.

Supported extensions: `docx` / `md` / `textpack`. How to enable:

- `docx` and `md` still default to `legacy`; select native explicitly, e.g. a default rule `LIGHTRAG_PARSER=docx:native` / `LIGHTRAG_PARSER=md:native`, or a filename hint `report.[native-iet].docx` / `notes.[native].md` (syntax in [§2.4](#24-default-rules-lightrag_parser) / [§2.5](#25-single-file-override-filename-hints)).
- `textpack` is a native-exclusive extension and is routed to native automatically without a hint/rule.

#### docx Extraction Capabilities

`native` parses OOXML directly and recognizes the following structures, writing them to the corresponding sidecars (whether a sidecar is produced depends on the document's actual content; see [§6.2](#62-__parsed__-directory-structure)):

| Element | Extraction behavior | Sidecar |
| --- | --- | --- |
| Heading levels | Heading 1–9 (inferred from `pPr/outlineLvl` or the style inheritance chain), feeding the `P` chunking strategy's heading-based splitting | `blocks.jsonl` |
| Paragraphs | Includes hyperlink text and list auto-numbering; tracked changes keep only the final text (deletions removed) | `blocks.jsonl` |
| Tables | 2D structure, auto-expanding merged cells (colspan/rowspan) and extracting cross-page repeated headers | `tables.json` |
| Images / drawings | Embedded images exported to a resource directory, with placeholders left in the body | `drawings.json` + `<base>.blocks.assets/` |
| Equations | OMML → LaTeX, distinguishing block-level vs inline | `equations.json` |

Image export details:

- Embedded images are exported to a `<base>.blocks.assets/` directory beside `blocks.jsonl`, supporting `png` `jpeg` `gif` `bmp` `tiff` `webp` `emf` `wmf`.
- **SVG images**: when Word saves an SVG it stores both the vector `.svg` and a PNG raster fallback; native docx writes that **PNG fallback** (reading `<a:blip>`'s `r:embed`, which points at the PNG) and does not export the SVG vector original. For downstream VLM consumption PNG is usually sufficient, with no further rasterization needed. (Note this differs from the md path's "SVG rasterized via cairosvg" below: docx simply takes the PNG Word already generated.)
- **VML / OLE objects** (legacy Word images, Visio diagrams, equation-editor previews, etc.): their rendered preview is exported via `v:imagedata`, commonly EMF/WMF, landing in the same assets directory; if the relationship is marked as an external link (`TargetMode="External"`), only the URL is recorded and no bytes are exported. **Note: EMF/WMF (and the previews of OLE objects such as Visio) can currently only be "extracted to disk" and cannot enter multimodal analysis** — the downstream VLM image analysis accepts only the raster formats `png` / `jpg` / `jpeg` / `gif` / `webp`, and other formats (EMF/WMF/SVG, etc.) are silently skipped (marked `skipped`; no error, and the rest of the document is unaffected). The exception is **equations**: they are stored as LaTeX text rather than images and are analyzed by the text (EXTRACT) role rather than the VLM, so they are processed normally.

#### docx Paragraph Provenance (paraId) Notice

native docx collects the `w14:paraId` written by Word 2013+ as a paragraph-level provenance anchor. If a document was produced by LibreOffice / WPS / older Word, or its internal docx XML was edited by hand, some paragraphs will lack paraId, and a one-time notice is logged:

```text
[parse_native] <filename>: N paragraphs lack paraId; Re-saving file in Word 2013+ to regenerate ids.
```

The affected blocks' `positions` degrade to `[{"type": "paraid", "range": null}]`. This is only a notice and **does not affect parsing success**; if you need precise paragraph provenance, follow the hint and "Save As .docx" in Word 2013+ to regenerate the ids.

#### md / textpack Extraction Capabilities

Beyond `docx`, the `native` engine also supports Markdown:

- `md`: splits by heading (ATX `#`), recognizes native pipe tables (with header), HTML `<table>` (with `<thead>`, preserving colspan/rowspan), block-level equations (a paragraph starting with `$$` and ending with `$$`; inline `$...$` is not recognized), and embedded images (base64 data URLs). Content inside fenced code blocks (```` ``` ````) is kept verbatim and not interpreted. As with `docx`, `md` still defaults to `legacy`; select native via `LIGHTRAG_PARSER=md:native` or a filename `[native]` hint.
- `textpack`: a TextBundle-format zip package (markdown body plus a resource directory, conventionally `assets/`; the export format of Bear / Ulysses, etc.). Only `native` supports this extension, so it is routed to native automatically without a hint/rule.
  - **Package structure requirements** (the body is located by extension, not a fixed `text.markdown` name, so you can pack it with any zip tool):
    - The body file may have any name, as long as its extension is `.md` or `.markdown`.
    - If the package contains a `*.textbundle` subdirectory, **at most one is allowed** (more than one is an error), and the body is **looked up only inside that `.textbundle` subdirectory** (md files in the root are ignored).
    - If the package contains **no** `*.textbundle` subdirectory, the body is **looked up only in the package root**.
    - The lookup directory must contain **exactly one** `.md` / `.markdown` file: zero or more than one is an error.
    - The directory holding the body is the "bundle root" (`bundle_root`) used for asset resolution.
  - File-reference images embedded by relative path are resolved relative to the bundle root and **may live in any subdirectory** (not only `assets/`); directory traversal is forbidden (`..`, absolute paths, or references escaping the bundle root are skipped with a warning), and the resolved bytes must pass an image magic-byte check or they are skipped. Relative-path images in a standalone `.md` (not a textpack) are not resolved (skipped with a warning).
- SVG images (base64 / textpack file / downloaded) are rasterized to PNG via cairosvg before being written to the sidecar; if cairosvg is unavailable or rendering fails, the image is skipped (with a warning). **System dependency**: `cairosvg` is a cffi binding — `pip install cairosvg` (via the `api` extra) always succeeds, but rasterization only actually works if the native `libcairo` shared library is *also* installed on the host (`sudo apt-get install libcairo2` on Debian/Ubuntu, `sudo dnf install cairo` on RHEL/Fedora, `brew install cairo` on macOS, or the GTK3 runtime on Windows) — `pip`/`uv` cannot install system libraries, so this step is never automatic outside the official Docker image. The server probes rasterization at startup and logs a warning if `libcairo` is missing, so the gap is visible before it surfaces as a per-document warning.
- External URL images (`![](http://...)`) are **downloaded and embedded by default** (`NATIVE_MD_IMAGE_DOWNLOAD_ENABLED` defaults to `true`); a drawing is always emitted (the fetched asset on success, or an external-link fallback on failure). Downloading allows only globally-routable public IPs (both DNS-resolved IPs and every redirect target are checked, and the socket dials the validated IP directly to defeat DNS rebinding; any ambient `HTTP(S)_PROXY` is ignored); private / loopback / link-local / reserved / CGNAT (`100.64.0.0/10`) ranges are all rejected. To allow specific internal ranges, configure a CIDR allowlist via `NATIVE_MD_IMAGE_ALLOWED_NON_PUBLIC_CIDRS`. Set the flag to `false` to instead drop external images entirely (no drawing emitted, so a document whose only images are external links produces no `drawings.json`).
  - Downloading is additionally bounded **per document**, not just per image: total retained image bytes (`NATIVE_MD_IMAGE_MAX_TOTAL_BYTES`), remote fetch attempts including redirect hops (`NATIVE_MD_IMAGE_MAX_REQUESTS`), and a wall clock across all downloads (`NATIVE_MD_IMAGE_DOWNLOAD_TOTAL_TIMEOUT`). Over-budget remote images degrade to external links with a parse warning rather than failing the document, and each request has a real wall-clock deadline (`NATIVE_MD_IMAGE_DOWNLOAD_TIMEOUT`) that a slow-trickling peer cannot reset. A download in flight is interrupted by `POST /documents/cancel_pipeline`.

#### Environment Variables

All of native's `NATIVE_*` environment variables and the `.native_raw/` cache directory **apply only to external-image downloading in the markdown / textpack engine**; **the docx path reads no `NATIVE_*` variable**. The two most common:

- `LIGHTRAG_FORCE_REPARSE_NATIVE` (default `false`): discard the `.native_raw/` cache and re-download external images over the network.
- `NATIVE_MD_IMAGE_DOWNLOAD_ENABLED` (default `true`): the master switch for external-image downloading; set to `false` to drop all external images.

The remaining download / size / budget / SSRF variables (`NATIVE_MD_IMAGE_DOWNLOAD_TIMEOUT` / `NATIVE_MD_IMAGE_DOWNLOAD_REQUIRED` / `NATIVE_MD_IMAGE_MAX_BYTES` / `NATIVE_MD_IMAGE_MAX_SVG_PIXELS` / `NATIVE_MD_IMAGE_MAX_TOTAL_BYTES` / `NATIVE_MD_IMAGE_MAX_REQUESTS` / `NATIVE_MD_IMAGE_DOWNLOAD_TOTAL_TIMEOUT` / `NATIVE_MD_IMAGE_ALLOWED_NON_PUBLIC_CIDRS`) — their meanings and defaults are listed in [env.example](https://github.com/HKUDS/LightRAG/blob/main/env.example) at the repository root.

Downloaded external images are cached in `<file>.native_raw/` (beside `.parsed/`, analogous to `.mineru_raw`/`.docling_raw`), reused directly when re-parsing the same unchanged file instead of going back over the network; the cache is invalidated when the source content or the size / SVG-pixel / CIDR options above change. When the document is deleted (with "also delete file" checked in the delete dialog), this cache directory is removed together with `.parsed/`.

### 3.4 Using the MinerU File Parsing Engine

The LightRAG document processing pipeline supports MinerU as a document parser and offers two MinerU access modes:

- `official` mode: uses MinerU's cloud API v4 service. You need to register an account at the [MinerU official website](https://mineru.net/) and create an API-KEY first. Then add the following configuration to LightRAG's `.env` file:

```bash
MINERU_API_MODE=official
MINERU_API_TOKEN=<your_token>
# MINERU_OFFICIAL_ENDPOINT=https://mineru.net   # Default value, usually no need to change
```

* `local` mode: uses a locally deployed MinerU service, see [ParserServiceDeployment.md §1](./ParserServiceDeployment.md#1-local-deployment-of-the-mineru-service). After the local MinerU service is started, add the following configuration to LightRAG's `.env` file:

```bash
MINERU_API_MODE=local
MINERU_LOCAL_ENDPOINT=http://<your_mineru_local_server_ip>:8000
```

Shared parameters, honoured in both modes:

| Env | Default | Meaning |
| --- | --- | --- |
| `MINERU_LANGUAGE` | `ch` | OCR / parsing language |
| `MINERU_ENABLE_TABLE` | `true` | Table recognition. **This decides whether `tables.json` is written at all** — one stage earlier than the `t` option of §2.1, which only decides whether an existing sidecar is analyzed |
| `MINERU_ENABLE_FORMULA` | `true` | Formula recognition; same relationship to the `e` option |
| `MINERU_PAGE_RANGES` | (empty) | Page range. `official` forwards it verbatim and accepts `1-3,5,7-9`; `local` accepts only a single page or one simple range, and a comma list is rejected at startup |
| `MINERU_ADDITIONAL_SUFFIXES` | (empty) | Extensions this deployment can handle beyond the baseline in §3.1 — see the note there on why declaring one is not enough on its own |

`local` mode only:

| Env | Default | Meaning |
| --- | --- | --- |
| `MINERU_LOCAL_ENDPOINT` | `http://127.0.0.1:8000` | mineru-api / mineru-router base URL |
| `MINERU_LOCAL_BACKEND` | `hybrid-auto-engine` | Which backend handles the parse: `hybrid-auto-engine` (pipeline + VLM, needs a GPU and the matching inference engine), `pipeline` (CPU-friendly, no VLM step), or `vlm-auto-engine` |
| `MINERU_LOCAL_PARSE_METHOD` | `auto` | `auto` / `txt` / `ocr` for the pipeline component. **Pure VLM backends ignore it** — the model handles layout and OCR natively |
| `MINERU_LOCAL_IMAGE_ANALYSIS` | `false` | MinerU's *own* VLM pass for captions and footnotes — see §4.7, it is not related to the `i` option. The `pipeline` backend silently drops the flag |
| `MINERU_LOCAL_START_PAGE_ID` | `0` | First page to parse |
| `MINERU_LOCAL_END_PAGE_ID` | `99999` | Last page to parse |

`official` mode only:

| Env | Default | Meaning |
| --- | --- | --- |
| `MINERU_API_TOKEN` | — | API key from the MinerU website; required |
| `MINERU_OFFICIAL_ENDPOINT` | `https://mineru.net` | Service endpoint |
| `MINERU_MODEL_VERSION` | `vlm` | Model version requested from the cloud service |
| `MINERU_IS_OCR` | `false` | Cloud-side OCR toggle |

Polling budget and cache:

| Env | Default | Meaning |
| --- | --- | --- |
| `MINERU_POLL_INTERVAL_SECONDS` | `2` | Interval between task-status polls |
| `MINERU_MAX_POLLS` | `600` | Maximum polls before giving up; the default budget is about 20 minutes |
| `MINERU_ENGINE_VERSION` | (empty) | Recorded in the raw-bundle manifest; a mismatch invalidates the cache. Empty skips the check (§6.3) |
| `LIGHTRAG_FORCE_REPARSE_MINERU` | `false` | Bypass the raw cache and re-upload on every parse (§3.7) |
| `MINERU_BBOX_ATTRIBUTES` | `{"origin":"LEFTTOP","max":1000}` | Coordinate system recorded in the sidecar meta. Note the default differs from `DOCLING_BBOX_ATTRIBUTES` |

> **Local deployment of the MinerU service** (Docker image build, vLLM preload, title-level correction) is described in [ParserServiceDeployment.md §1](./ParserServiceDeployment.md#1-local-deployment-of-the-mineru-service).

### 3.5 Using the Docling File Parsing Engine

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
| `DOCLING_DO_FORMULA_ENRICHMENT` | `false` | Whether to recognize equations in the document and output them in LaTeX format; before enabling, ensure that Docling has downloaded the equation recognition model on the backend (see [ParserServiceDeployment.md §2](./ParserServiceDeployment.md#2-local-deployment-of-docling-serve-latex-equation-recognition)) |

When `DOCLING_OCR_ENGINE` / `DOCLING_OCR_PRESET` are not configured, they are equivalent to `auto`; when `DOCLING_OCR_LANG` is not configured, no language list is passed to docling-serve, and the OCR engine uses its own default. The parse cache signature is computed from these effective parameters, so "not configured" and "explicitly set to the default value" do not invalidate the cache.

Optional input formats (1 env):

| Env | Default | Meaning |
| --- | --- | --- |
| `DOCLING_ADDITIONAL_SUFFIXES` | (empty) | Comma-separated suffixes this deployment's docling-serve can handle on top of the baseline set in §3.1, e.g. `doc,ppt,xls`. Docling's legacy Office support requires LibreOffice on the docling-serve side, so these formats are opted in per deployment instead of being advertised globally |

Notes on `DOCLING_ADDITIONAL_SUFFIXES`:

- Bare lowercase suffixes separated by `,`; a leading dot and surrounding whitespace are tolerated (` .DOC ` = `doc`). Anything else (`*.doc` written out of glob habit, or a `;`-separated list) is rejected at startup rather than silently ignored.
- It only makes the suffix **routable to docling**; it does not by itself make a bare `x.doc` uploadable. Pair it with a routing rule (`LIGHTRAG_PARSER=doc:docling`) or a per-file hint (`x.[docling].doc`) — otherwise such files still fall through to the default `legacy` engine and are rejected as an unsupported suffix. Conversely, a rule like `doc:docling` without this env fails startup validation, since `doc` is not among docling's capabilities.
- Read live from the environment, so it takes effect whether it comes from the parent shell or from `.env`.

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

> **Local deployment of docling-serve**, including downloading the equation-recognition model required by `DOCLING_DO_FORMULA_ENRICHMENT`, is described in [ParserServiceDeployment.md §2](./ParserServiceDeployment.md#2-local-deployment-of-docling-serve-latex-equation-recognition).

### 3.6 Attaching engine parameters

Parameters may also be attached to the **engine token** to override an external engine's per-file behaviour. They are encoded into the persisted `parse_engine` field and feed both the engine request and its raw-bundle cache signature (so changing a parameter forces a re-parse rather than reusing a stale bundle).

```text
paper.[mineru(page_range=1-3,language=en,local_parse_method=ocr)].pdf   # filename hint
scan.[docling(force_ocr=true)].pdf
report.[native(smart_heading)].docx                                      # bare boolean flag
LIGHTRAG_PARSER=pdf:mineru(language=en);*:legacy-R                       # rule
```

Currently supported engine parameters (canonical / alias):

| Engine | Parameter | Alias | Type | Notes |
| --- | --- | --- | --- | --- |
| `mineru` | `page_range` | `pr` | list | One or more page ranges; **see the list note below** |
| `mineru` | `language` | — | str | OCR / model language (e.g. `en`, `ch`) |
| `mineru` | `local_parse_method` | `local_pm` | enum | `auto` / `txt` / `ocr` (local mode) |
| `docling` | `force_ocr` | `ocr` | bool | `true` / `false`; may be written bare: `docling(ocr)` means `docling(force_ocr=true)` |
| `native` | `smart_heading` | — | bool | Opt-in docx smart heading discovery (see [§3.3](#33-using-the-native-file-parsing-engine)); may be written bare (`native(smart_heading)` means `=true`); the markdown path warns and ignores it |

- **`page_range` may contain multiple page segments — write one `page_range=...` item per segment.** Inside `(...)` a comma only separates parameters, so a multi-segment list should be written as `page_range=1-3,page_range=5,page_range=7-9`, not as the env-var single-string form `MINERU_PAGE_RANGES="1-3,5,7-9"`. A **multi-segment** `page_range` requires `MINERU_API_MODE=official`; `local` mode accepts only a single page/range (for example, `page_range=1-3`).
- **`local_parse_method` is local-only.** It only affects the local MinerU request, so it is **rejected** under `MINERU_API_MODE=official` (the official API neither sends it nor folds it into the cache key — accepting it would silently do nothing).
- **A boolean engine parameter may be written bare as a flag** to keep rules and filenames short: `native(smart_heading)` means `native(smart_heading=true)`, and `docling(ocr)` means `docling(force_ocr=true)`. Only booleans may be bare (`mineru(language)` is a friendly error), and the persisted `parse_engine` is always re-encoded in the canonical `key=value` form (`native(smart_heading=true)`), so the shorthand never changes a cache signature. To turn a boolean off you still write it explicitly (`native(smart_heading=false)`).
- Engine parameters are only accepted by engines that declare them (`mineru` / `docling` / `native`); attaching a parameter to `legacy`, or an unknown parameter to any engine, is a friendly error. Validation runs at startup (`LIGHTRAG_PARSER`) and at upload.
- Merge priority: engine parameters resolve for the **final engine** — a rule's engine parameters are dropped when a filename hint selects a different (usable) engine.
- `parse_engine` is stored in hint syntax (e.g. `mineru(page_range=1-3)`) and shown in `doc_status` metadata so you can see the parse parameters a document used.

### 3.7 Parse Cache and Forced Re-parse

The `native`, `mineru` and `docling` engines each keep a raw-artifact cache next to the parsed output, so re-parsing an unchanged file does not repeat the expensive work — a network round trip to an external service, or an image download for markdown. Layout and invalidation rules are in §6.3; what matters when configuring is:

| Engine | Cache directory | What it holds | Force re-parse |
| --- | --- | --- | --- |
| `native` | `<base>.native_raw/` | external images downloaded by the markdown / textpack path | `LIGHTRAG_FORCE_REPARSE_NATIVE=true` |
| `mineru` | `<base>.mineru_raw/` | the artifact bundle returned by the MinerU service | `LIGHTRAG_FORCE_REPARSE_MINERU=true` |
| `docling` | `<base>.docling_raw/` | the artifact bundle returned by docling-serve | `LIGHTRAG_FORCE_REPARSE_DOCLING=true` |
| `legacy` | — | no cache | not applicable |

Each bundle records the engine version and the effective parameter signature in its manifest, so changing the endpoint, an extraction parameter, or `MINERU_ENGINE_VERSION` / `DOCLING_ENGINE_VERSION` invalidates the cache on its own. The force flags exist for the case the manifest cannot detect: the *service* changed while its version string did not. Deleting a document with "also delete file" removes the cache directory along with `.parsed/`.

## 4. Multimodal Analysis (VLM)

### 4.1 What this stage does

Parsing writes sidecars; the ANALYZING stage reads them, sends each item to a model, and writes the result back into the item as `llm_analyze_result`; the PROCESS stage then builds multimodal chunks from those sidecars. Two consequences follow from that ordering:

- **Analysis is re-runnable without re-parsing.** The extraction stage is unaffected by `i` / `t` / `e` — the engine writes whatever the document contains — so enabling a modality later completes the VLM work without touching the original file (§9.3).
- **A modality with no sidecar is silently a no-op**, logged at INFO level. That is not an error; it means the document has no such content, or the engine does not emit it.

### 4.2 What each option actually requires

`VLM_PROCESS_ENABLE` is often read as a master switch over all three modalities. It is not — it gates images only:

| Option | Analyzed by | Requires `VLM_PROCESS_ENABLE` | Sidecar read |
| --- | --- | --- | --- |
| `i` images | VLM role | **Yes** | `*.drawings.json` |
| `t` tables | EXTRACT role | No | `*.tables.json` |
| `e` equations | EXTRACT role | No | `*.equations.json` |

This is why the recommended preset `*:native-teP` works with no VLM configured at all. Whether a sidecar exists in the first place also depends on the engine: `MINERU_ENABLE_TABLE` / `MINERU_ENABLE_FORMULA` for MinerU, `DOCLING_DO_FORMULA_ENRICHMENT` for docling, and `legacy` emits none.

### 4.3 Failure versus skip

Everything in this section applies **only to a document whose `process_options` include `i`**. A document without `i` never enters image analysis at all, so it processes normally no matter how many images it carries and no matter how `VLM_PROCESS_ENABLE` is set.

For a document that does have `i`, each image runs a filter chain, and **the VLM gate sits in the middle of it**. Everything before the gate is a skip that leaves the document healthy; the gate itself fails the whole document:

| Order | Condition | Outcome |
| :-: | --- | --- |
| 1 | image file not found | item `skipped`, document continues |
| 2 | not a supported raster format (emf / wmf / svg …) | item `skipped`, document continues |
| 3 | either side smaller than `VLM_MIN_IMAGE_PIXEL` (64) | item `skipped`, document continues |
| 4 | **VLM gate**: `VLM_PROCESS_ENABLE=false` or no VLM role | **document FAILED** — `error_msg` reads "VLM analysis required but VLM role is not available" |
| 5 | image file is empty | document FAILED |
| 6 | larger than `VLM_MAX_IMAGE_BYTES` (5 MB) | item `skipped` — note this check is **after** the gate, so an oversized image does not escape step 4 |

For the text modalities: a **table** item with empty content is recorded as `skipped` with a warning and never reaches format validation; only a table item that *has* content fails the document when its `format` is missing or invalid, which indicates a corrupt sidecar. An **equation** item with empty content fails the document directly.

### 4.4 Token budgets

| Variable | Default | Meaning |
| --- | --- | --- |
| `MAX_EXTRACT_INPUT_TOKENS` | `20480` | total input budget for one extraction / analysis prompt |
| `SURROUNDING_LEADING_MAX_TOKENS` | `2000` | per-half cap on the text injected *before* the item |
| `SURROUNDING_TRAILING_MAX_TOKENS` | `2000` | per-half cap on the text injected *after* the item |
| `MM_EXTRACT_CONTENT_MIN_TOKENS` | `100` | floor reserved for the item's own content |

The surrounding budgets are subtracted from the total, so setting them too high starves the item itself. When that happens the server warns at startup and names the variable to raise; either raise `MAX_EXTRACT_INPUT_TOKENS` or lower the surrounding pair.

### 4.5 Image limits and throughput

`VLM_MAX_IMAGE_BYTES` (default 5 MB) and `VLM_MIN_IMAGE_PIXEL` (default 64) bound what is worth sending: the lower bound exists to skip icons and separator rules rather than pay for a VLM call on them. Analyze-stage concurrency is `MAX_PARALLEL_ANALYZE` (§8.6), which is independent of the parse and insert stages.

### 4.6 Model configuration lives elsewhere

`VLM_PROCESS_ENABLE` is the pipeline switch and belongs here. *Which* model serves the VLM role — `VLM_LLM_MODEL`, `VLM_LLM_BINDING`, `VLM_LLM_BINDING_HOST`, `VLM_LLM_BINDING_API_KEY`, `VLM_MAX_ASYNC_LLM`, `VLM_LLM_TIMEOUT` — plus the list of vision-capable providers is owned by [Role-based LLM/VLM Configuration Guide](RoleSpecificLLMConfiguration.md) and is deliberately not repeated here.

### 4.7 Not the same thing: `MINERU_LOCAL_IMAGE_ANALYSIS`

MinerU has a VLM pass of its own, enabled by `MINERU_LOCAL_IMAGE_ANALYSIS`. It runs **at parse time on the MinerU host**, improves captions and footnotes inside MinerU's own output, costs GPU on that machine, and is ignored by the `pipeline` backend. It is unrelated to the `i` option and to `VLM_PROCESS_ENABLE`.

### 4.8 A safe order to switch things on

1. Start with `te`. It needs only the EXTRACT role — no new infrastructure, no new failure mode.
2. Confirm the sidecar items carry `llm_analyze_result.status == "success"`.
3. Then add `i` **together with** `VLM_PROCESS_ENABLE=true` and a vision-capable binding.

Doing step 3 in two halves is what produces the failure in §4.3 step 4. And because processing options are frozen at enqueue, a document that already failed that way cannot be rescued by editing configuration and retrying: `/documents/reprocess_failed` re-runs it with the same `i`. Either make the VLM available and retry, or delete the document and upload it again without `i` (§8.3).

## 5. Chunker Parameter Configuration (chunk_options)

### 5.1 Responsibilities of process_options vs chunk_options

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
- **`addon_params["chunker"]`** is an `ObservableAddonParams` field; for Server deployments, you only need env / restart for the new values to take effect. To change it at runtime within the Python process (without restarting) and to do per-file overrides, see [Chapter 11: Python SDK Invocation](#11-python-sdk-invocation).
- **`full_docs.chunk_options`** is frozen at `apipeline_enqueue_documents` enqueue time: by default it is assembled by `resolve_chunk_options(self.addon_params, ...)` on the spot; if the caller passes a `chunk_options` argument, it is persisted as-is (SDK usage, see §11.4).
- **The chunker invocation** takes the corresponding sub-dictionary from `full_docs.chunk_options` and dispatches to F/R/V/P by the `process_options.chunking` selector.

### 5.2 Environment Variables

All variables below are read into `addon_params["chunker"]` once when `LightRAG` is instantiated: strategy-specific env is read by `default_chunker_config()`, while legacy env (`CHUNK_SIZE` / `CHUNK_OVERLAP_SIZE`) is filled in by `_apply_chunk_size_overlay` into slots that neither strategy env nor legacy constructor fields filled. After modifying env, the service must be restarted (or a new `LightRAG` instance created) for it to take effect; documents already enqueued hold the frozen snapshot and are unaffected.

They are grouped by the strategy they configure. A strategy-specific variable always outranks the global fallback for the same slot.

#### Global fallbacks

- **`CHUNK_SIZE`** — `1200`, int.
  Top-level `chunk_token_size` fallback. Ranks below strategy-specific env and below `addon_params["chunker"]["chunk_token_size"]` set on the SDK path.
- **`CHUNK_OVERLAP_SIZE`** — `100`, int.
  Overlap fallback. Fills a slot only when the strategy has neither its own env (`CHUNK_F_OVERLAP_SIZE` / `CHUNK_R_OVERLAP_SIZE` / `CHUNK_P_OVERLAP_SIZE`) nor the SDK path's `LightRAG(chunk_overlap_token_size=…)`.

#### F — fixed length

- **`CHUNK_F_SIZE`** — unset, int.
  F's own `chunk_token_size`, outranking the top-level fallback (`CHUNK_SIZE` and the SDK path's `LightRAG(chunk_token_size=…)`). When unset, F inherits the top-level resolved value.
- **`CHUNK_F_OVERLAP_SIZE`** — unset, int.
  F's own overlap, outranking the legacy constructor field and `CHUNK_OVERLAP_SIZE`.
- **`CHUNK_F_SPLIT_BY_CHARACTER`** — unset (`null`), str.
  Pre-split separator. `null` or an empty string means split by token window only.
- **`CHUNK_F_SPLIT_BY_CHARACTER_ONLY`** — `false`, bool.
  Strict mode: do not re-split by token, raise on an oversized segment instead.

#### R — recursive character

- **`CHUNK_R_SIZE`** — unset, int.
  R's own `chunk_token_size`, outranking the top-level fallback. When unset, R inherits the top-level resolved value.
- **`CHUNK_R_OVERLAP_SIZE`** — unset, int.
  R's own overlap, outranking the legacy constructor field and `CHUNK_OVERLAP_SIZE`.
- **`CHUNK_R_SEPARATORS`** — JSON array string. Default:

  ```json
  ["\n\n","\n","。","！","？","；","，"," ",""]
  ```

  The separator cascade, ordered from the strongest semantic boundary to the weakest. The default includes Chinese sentence-ending (`。！？`) and mid-sentence (`；，`) punctuation, so Chinese and mixed Chinese-English documents split at semantic boundaries. English `.?!` is deliberately excluded — matching it literally would cut through numbers and abbreviations.

  **Bounded to 64 entries of at most 256 characters each.** The splitter descends one level per remaining separator and re-scans the whole text at every level, so an oversized cascade costs `O(len(separators) × len(text))` while splitting nothing extra. The two limits do not behave the same way, and the difference is visible in the output: an entry longer than 256 characters is **dropped**, not shortened — a lone 300-character separator disappears rather than matching its first 256 — while a list longer than 64 entries is **truncated** to 64, keeping the trailing char-level `""` sentinel when the original had one. If nothing survives, the fallback is the consumer's own and is *not* this variable's default: `chunking_by_recursive_character` takes its documented `separators=None` path, which is the splitter's four-entry English cascade `["\n\n", "\n", " ", ""]` rather than the nine-entry `DEFAULT_R_SEPARATORS`, while `load_chunk_separators` falls back to `DEFAULT_R_SEPARATORS` minus the sentinel.

  For non-HTTP configuration, correction is reported once when the value is cached: `CHUNK_R_SEPARATORS` at configuration load, and a supplied or replaced `addon_params['chunker']` immediately (or at the first enqueue for a compatible nested in-place mutation). The normalized value is then stored for subsequent documents — in place, so a reference to the nested `recursive_character` dict obtained through the documented runtime-mutation idiom keeps steering later documents. Direct SDK calls and snapshots persisted by older releases retain their stored value and are bounded silently at execution — including the all-dropped fallback to `separators=None`, which is reachable only from a direct SDK call — so one historical bad value cannot create a warning per document.

  One correction is not a bound at all: an `addon_params['chunker']` whose `separators` is neither a list/tuple nor `None` has the key **removed** and is reported separately. A bare `str` is why — it satisfies `Sequence[str]`, so bounding it would iterate characters and rewrite one typo into a cascade of 64 single characters that reads as deliberate ever after. Dropping the key routes the chunker to its documented `separators=None` path instead. An **HTTP request body is rejected instead**: the request model raises on either limit, so `/documents/text` and friends answer **422** and nothing is dropped, truncated or fallen back to. Same numeric bound everywhere, deliberately different response to breaching it — the caller of an HTTP request is present to read the error, an environment variable's author is not.

#### V — semantic vector

- **`CHUNK_V_SIZE`** — unset, int.
  V's own `chunk_token_size`. It is a hard cap: anything over it is re-split through R. Outranks the top-level fallback; when unset, V inherits the top-level resolved value.
- **`CHUNK_V_BREAKPOINT_THRESHOLD_TYPE`** — `percentile`, str.
  One of `percentile` / `standard_deviation` / `interquartile` / `gradient`.
- **`CHUNK_V_BREAKPOINT_THRESHOLD_AMOUNT`** — unset (`null`), float.
  Threshold magnitude. `null` lets LangChain pick the per-type default (percentile uses 95).
- **`CHUNK_V_BUFFER_SIZE`** — `1`, int.
  How many adjacent sentences are merged when computing distances.
- **`CHUNK_V_SENTENCE_SPLIT_REGEX`** — str. Default:

  ```text
  (?<=[.?!])\s+|(?<=[。？！])
  ```

  The sentence split fed to LangChain's `SemanticChunker`. The default recognizes English `.?!` (requiring trailing whitespace, so `0.95` survives) and Chinese `。？！` (no whitespace required, matching continuous Chinese text). The env value is the raw regex — no JSON quoting.

#### P — paragraph semantic

- **`CHUNK_P_SIZE`** — `2000` (`DEFAULT_CHUNK_P_SIZE`), int.
  P's own `chunk_token_size`, and the one slot that does **not** inherit. When unset, P does not fall back to the top-level `CHUNK_SIZE` / `LightRAG(chunk_token_size=…)`; the slot always carries `DEFAULT_CHUNK_P_SIZE` instead, because paragraph-semantic merging needs more headroom than the global default to keep related paragraphs together. Override it here when a deployment needs a different ceiling. P's internal ratio constants are algorithmic scales derived in proportion to whatever this resolves to.
- **`CHUNK_P_OVERLAP_SIZE`** — unset, int.
  P's own overlap, outranking the legacy constructor field and `CHUNK_OVERLAP_SIZE`. It covers two things: text overlap when long body text inside one JSONL content line falls back to R, and the per-side budget for bridging text copied into the chunks around a large table. It does **not** make table row-level slices overlap each other.

> `CHUNK_F_SIZE` / `CHUNK_R_SIZE` / `CHUNK_V_SIZE` behave the opposite way from `CHUNK_P_SIZE`: left unset they **do** inherit the top-level `chunk_token_size`. That is usually what you want — F is the default global window, R prefers a smaller target so it can split at sentence level, and V, being an advisory ceiling, is usually raised rather than lowered to avoid over-splitting.

### 5.3 Priority Chain

The final value of each chunking slot is resolved by a specificity-ordered chain (high → low):

1. **`addon_params["chunker"]` explicit value** — field values explicitly written at construction time or set at runtime via the SDK path (see §11.3). Server-only deployments usually don't hit this tier. Most direct; wins everything.
2. **Strategy-specific env** — `CHUNK_F_SIZE` / `CHUNK_R_SIZE` / `CHUNK_V_SIZE` (per-strategy `chunk_token_size`), `CHUNK_F_OVERLAP_SIZE` / `CHUNK_R_OVERLAP_SIZE` / `CHUNK_P_OVERLAP_SIZE` (overlap), `CHUNK_P_SIZE` (P-specific). When the corresponding size env is unset, F/R/V inherit the top-level `chunk_token_size`. Filled only when the slot is not already occupied by ①.
3. **Legacy constructor fields** — `LightRAG(chunk_token_size=…, chunk_overlap_token_size=…)`; only effective on the SDK path, see §11.2. Strategy-agnostic, "coarse-grained default", fills only the slots still empty.
4. **Legacy env** — `CHUNK_SIZE` / `CHUNK_OVERLAP_SIZE`. Final fallback.

Example: `CHUNK_R_OVERLAP_SIZE=42` + `LightRAG(chunk_overlap_token_size=2)` → R sub-dictionary `chunk_overlap_token_size=42` (strategy env wins), F / P sub-dictionary `chunk_overlap_token_size=2` (no F / P-specific env; the legacy constructor field is filled in).

**Special case for P's `chunk_token_size`**: the P `chunk_token_size` slot does NOT walk the full four-tier chain. When ① is not explicitly provided, it resolves directly via `CHUNK_P_SIZE` env > `DEFAULT_CHUNK_P_SIZE` (2000), **skipping** ③ legacy constructor field `LightRAG(chunk_token_size=…)` and ④ legacy env `CHUNK_SIZE`. See the `CHUNK_P_SIZE` row in §5.2 for the rationale.

Three layers of semantic guarantee:

1. **Reproducibility**: change env, restart — old documents still chunk by the snapshot from the moment they were enqueued; results unchanged.
2. **Resume consistency**: resume branch B (content already extracted, redo chunking by current `process_options`) also reads `full_docs.chunk_options`, preventing env drift from breaking consistency.
3. **Per-file personalization**: callers can pass different `chunk_options` for each file (typical usage: a management UI configures separators or V threshold individually for a certain file). These are the input semantics on the SDK path; see §11.4.

### 5.4 Field Structure

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
                                                              // env/SDK only (CHUNK_V_SENTENCE_SPLIT_REGEX); the REST `chunking.params`
                                                              // object rejects this key with 422 — see GHSA-32jh-39m7-8x84 (ReDoS)
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

### 5.5 Backward Compatibility for Missing Fields

Old documents at enqueue time don't yet have the `chunk_options` field; during chunking, the dispatcher calls `resolve_chunk_options(self.addon_params, process_options=…)` per the current `process_options` to fall back to a slim snapshot. After upgrading, it is recommended to run a reprocess once to give old documents a slim `chunk_options` snapshot (aligned with the current `process_options`).

## 6. Storage and Directory Layout

### 6.1 `full_docs` Fields

File enqueue and extraction results are written into `full_docs`:

| Field | Description |
| --- | --- |
| `file_path` | Basename of the filename (without directory), **preserves the original name provided by the user (including the square-bracket hint)**, e.g., `abc.[native-iet].docx` is written as-is. When no valid source is provided, it is saved as `unknown_source`. The filename hint is not stripped, so the management UI can directly show the user's original naming intent. |
| `canonical_basename` | The canonicalized basename with the processing hint stripped (e.g., `abc.docx`). Filename deduplication uses this field as the index key, ensuring `abc.docx` and `abc.[native-iet].docx` are treated as the same logical document. |
| `source_path` | The original path provided at enqueue time (written only when it contains a directory separator or is an absolute path), used by the `native` / `mineru` / `docling` parsers to locate the actual file. |
| `parse_format` | Content format: `pending_parse`, `raw`, `lightrag`. |
| `content` | When `raw`, holds the extracted text; when `pending_parse`, it is an empty string; when `lightrag`, holds the **complete merged text** starting with `{{LRdoc}}` (concatenated body segments of all `type=="content"` lines in `.blocks.jsonl`). At the parse stage, the reuse handler (`ReuseParser`) strips the prefix and hands it to the chunking_func, going through exactly the same code path as `raw`. |
| `content_hash` | MD5 of the content, used for cross-filename deduplication. For `parse_format=raw`, takes the hash of text after `sanitize_text_for_encoding`; for `parse_format=lightrag`, takes the hash of the `*.blocks.jsonl` file; for `parse_format=pending_parse`, not written, filled in after extraction completes. |
| `lightrag_document_path` | When `parse_format=lightrag`, saves the path to the structured LightRAG Document; new records prefer to save the path relative to `INPUT_DIR`, e.g., `__parsed__/report.docx.parsed/report.blocks.jsonl`. Note that the subdirectories and the blocks filename in the path both use the canonicalized basename (without hint). |
| `parse_engine` | The engine that actually completed extraction: `legacy`, `native`, `mineru`, `docling`. For files awaiting extraction, can also temporarily store the target engine. |
| `process_options` | The original processing options string recorded at enqueue time (without engine name and the separator `-`), e.g., `"iet"`, `"R!"`, `""`. Downstream stages take this field as the authoritative source for deciding whether to enable image / table / equation analysis (`i/t/e`), whether to disable knowledge graph construction (`!`), and the chunking method (`F/R/V/P`). An empty string is equivalent to all defaults. |
| `chunk_options` | The **frozen** snapshot of chunker parameters at enqueue time (slim dictionary: only the strategy sub-dictionary selected by `process_options` is retained, others discarded). Passed in by the SDK-path caller or assembled by `resolve_chunk_options(self.addon_params, process_options=…)` from instance fields (containing env defaults) as a fallback (see §5.1). `process_options` chooses which chunking strategy (F/R/V/P); `chunk_options` decides which parameters that chunker uses. The downstream `process_single_document` reads strategy-specific kwargs from this field before chunking; persistence guarantees that old documents behave reproducibly across env changes, resumes, and restarts. Rewritten together with `process_options` when re-parsing. |

`pending_parse` indicates the file has been enqueued but extraction is not yet complete. After successful extraction, it is rewritten to `raw` or `lightrag`, and `content_hash` is filled in. On extraction failure, `pending_parse` and the empty `content` are kept, making subsequent troubleshooting and retry easier.

> The original `file_path` (with hint), `canonical_basename`, and `content_hash` are also synchronized into `doc_status`, serving as the deduplication index sources for `get_doc_by_file_basename` / `get_doc_by_content_hash`. `get_doc_by_file_basename` internally canonicalizes the input through `canonicalize_parser_hinted_basename` before comparing against `canonical_basename`, so `abc.docx` and `abc.[native-iet].docx` always hit the same document.
> `process_options` is also mirrored into `doc_status.metadata["process_options"]`, making it convenient for the management UI to directly display the current file's processing policy.

### 6.2 `__parsed__` Directory Structure

`__parsed__` is the archival and analysis-result directory next to the input directory. It both stores already-processed original documents and the `LightRAG Document` (lightrag format) files and image assets produced by structured parsing.

- Original file archival: after `legacy` local extraction succeeds and enqueueing finishes, the original file is moved into the sibling `__parsed__` directory; `native` / `mineru` / `docling` keep the original file first for the pipeline to parse, and only move it to `__parsed__` after successful parsing and writing to `full_docs`. **When archived, the original filename (including `[hint]`) is preserved**, e.g., `report.[native-iet].docx` is archived as `__parsed__/report.[native-iet].docx`, making it easy to trace the user's original name and processing options.
- Analysis result directory: structured parsing results are written into a subdirectory named with the **canonicalized filename** (with `[hint]` removed) plus the `.parsed` suffix, avoiding name conflicts with the archived original file and ensuring that the same logical document continues to point to the same directory when the filename hint or processing options change. For example, the analysis results of `report.docx`, `report.[native].docx`, and `report.[native-iet].docx` are all written into `__parsed__/report.docx.parsed/`.
- Analysis result files: the LightRAG Document blocks file and sidecars are named with the canonicalized filename stem, e.g., `__parsed__/report.docx.parsed/report.blocks.jsonl`; the same directory may also contain `report.tables.json`, `report.drawings.json`, `report.equations.json`, and the `report.blocks.assets/` image asset directory. **Whether a sidecar is generated is determined by the document content**: the parser only writes the corresponding file when the document actually contains tables / images / equations. This is the only signal of modality availability — the engine does not need to declare capabilities in meta. The `i`/`t`/`e` options only determine whether the next stage invokes the VLM for summarization analysis on already-existing sidecars.
- When parsing fails, the original file is not moved, making it easy to fix the configuration and re-process.
- When `/documents/scan` encounters a file with the same name that is already `PROCESSED`, the input file is treated as already processed and moved to `__parsed__`, not enqueued as a new document.
- When `/documents/scan` finds multiple files that share the same canonicalized name in the same scan, **the file that claims the canonical source key first wins** (i.e. whichever the streaming directory walk reaches first). A scan-wide UNIQUE claim in the disposable disk spool archives every later variant with a warning before any candidate is persisted. For example, if both `abc.docx` and `abc.[native].docx` exist, only the one reached first is processed. A hint selects the engine and no longer grants scheduling priority.
- When duplicate content hashes are found during scanning or parsing, the input file is likewise moved to `__parsed__`; this `doc_status` entry is kept as `FAILED duplicate` for tracking.
- File moves only act on the current input file and do not overwrite or move existing document source files. If a file with the same name already exists at the destination, the system automatically appends `_001`, `_002`, etc., e.g., `report.pdf` is archived as `report_001.pdf`, `report_002.pdf`. If the analysis result directory name is already taken by a regular file, a number is also appended, e.g., `report.docx.parsed_001/`.

### 6.3 Raw Artifact Bundles (`<base>.*_raw/`)

Three engines keep a raw-artifact bundle beside `<canonical filename>.parsed/`, so that re-parsing an unchanged file does not repeat the one expensive step that engine performs. All three follow the same design: the bundle directory holds the raw artifacts plus a `_manifest.json` that doubles as the atomic success marker and the cache key.

| Bundle | Written by | Expensive step it avoids | Contents |
| --- | --- | --- | --- |
| `<base>.native_raw/` | `native` (markdown / textpack path) | downloading and rasterizing external `http(s)` images | one file per cached image, named `sha256(url)[:16]`, holding the post-rasterization bytes |
| `<base>.mineru_raw/` | `mineru` | the upload / poll / download round trip to the MinerU service | `content_list.json`, plus optional `full.md` / `middle.json` / `layout.pdf` / `images/` |
| `<base>.docling_raw/` | `docling` | the upload / poll / download round trip to docling-serve | `<base>.json` (DoclingDocument, with `pages[].image` base64), `<base>.md` for human inspection, and `artifacts/image_*.png` referenced by `pictures[*].image.uri` |

The bundle is a **sibling** of `.parsed/`, not a child, so it survives the directory wipe performed before every re-extraction.

Design goals:

- **Avoid repeating the expensive step.** On re-parse the source file's hash and size are validated against `_manifest.json` first; on a hit the network work is skipped entirely and the stored artifacts are fed straight through the adapter and the sidecar writer.
- **Preserve diagnostic material.** When an engine parses something incorrectly, or a sidecar field looks wrong, the bundle is the place to compare against what the engine actually returned.
- **Support object traceability** (external engines). MinerU-derived `drawings.json` / `tables.json` / `equations.json` record `content_list.json#/N` in `self_ref`, which resolves back to the original MinerU object and its `page_idx` / `bbox`.
- **De-hint the uploaded filename** (MinerU). When the source filename carries processing hints such as `[mineru-...]` / `[-iet]`, the MinerU API is called with the canonicalized name, so no hint-bearing filename ends up inside the returned bundle.

Docling bundle layout:

```text
__parsed__/<base>.docling_raw/
├── _manifest.json
├── <base>.json        # DoclingDocument JSON (contains pages[].image base64)
├── <base>.md          # Markdown form, for human inspection
└── artifacts/
    └── image_*.png    # image assets referenced by pictures[*].image.uri
```

Lifecycle, identical for all three bundles:

| Operation | Behavior |
|---|---|
| First parse | Fetch the artifacts, then atomically write `_manifest.json`. For docling that is `POST /v1/convert/file/async` → long-poll `/v1/status/poll/{task_id}?wait=N` → `GET /v1/result/{task_id}` → safe extraction of the zip, rejecting absolute paths and `..`. |
| Re-parse (cache hit) | Do not call the external service; do not rewrite artifacts; rerun adapter + writer to regenerate the sidecar (this is what makes an adapter upgrade cheap). |
| Re-parse (cache miss) | Clear the directory, then fetch and write the manifest again. |
| `DELETE /documents` with `delete_file=True` | `*.parsed/`, the raw bundle, and the original file are all removed together. |
| `DELETE /documents` with `delete_file=False` | All artifacts are preserved; only doc_status and KG data are deleted. |
| `clear_documents` / a full sweep of `__parsed__` | Naturally cleared together. |
| scan cycle | Does **not** GC orphaned bundles — they are removed only on an explicit user deletion, so a debugging site is never swept away by accident. |

Force re-parse (bypass the cache entirely): `LIGHTRAG_FORCE_REPARSE_NATIVE` / `LIGHTRAG_FORCE_REPARSE_MINERU` / `LIGHTRAG_FORCE_REPARSE_DOCLING` (§3.7).

Concurrency safety: LightRAG mandates `canonical_basename` uniqueness within a workspace (HTTP 409 on upload / enqueue), and the pipeline serializes work per document, so no bundle is ever written concurrently and no extra lock is needed.

#### Manifest invalidation

A mismatch on any of the following is a cache miss. Shared by all three bundles:

- source file size or sha256 does not match the manifest;
- the recorded artifacts are missing, or their size / sha256 does not match.

Engine-specific conditions on top of that:

| Engine | Additional invalidation conditions |
| --- | --- |
| `native` | the download-options signature changed — the size / SVG-pixel / CIDR options of §3.3 |
| `mineru` | `MINERU_ENGINE_VERSION` differs from the recorded `engine_version`; `MINERU_API_MODE` differs from the recorded `api_mode`; the endpoint for the current mode (`MINERU_OFFICIAL_ENDPOINT` / `MINERU_LOCAL_ENDPOINT`) differs from the recorded `endpoint_signature`; `content_list.json` size or sha256 mismatch; a recorded non-critical file (images, `middle.json`, …) has a different size |
| `docling` | `DOCLING_ENDPOINT` differs from the recorded `endpoint_signature`; `DOCLING_ENGINE_VERSION` is set and differs from the recorded `engine_version`; `options_signature` differs — covering the tunable env (`DOCLING_DO_OCR` / `DOCLING_FORCE_OCR` / `DOCLING_OCR_ENGINE` / `DOCLING_OCR_PRESET` / `DOCLING_OCR_LANG` / `DOCLING_DO_FORMULA_ENRICHMENT`) **and** the hard-coded constants `pipeline` / `target_type` / `to_formats` / `image_export_mode`, which are written into the signature so that changing them later cannot silently reuse an old bundle; main JSON missing or size / sha256 mismatch; any `artifacts/` image missing or size mismatch |

> **"Either side empty means skip"** applies to `engine_version` and `endpoint_signature` on both external engines. If the field was empty when the manifest was written (for example `MINERU_ENGINE_VERSION` was unset at first parse), or if the current environment variable is unset, that particular check is skipped. Consequently, setting a version variable *after* a bundle already exists does not retroactively invalidate it — use the matching `LIGHTRAG_FORCE_REPARSE_*` flag for that.

## 7. Same-name and Duplicate Documents

Uploads, directory scans and the text endpoints all deduplicate. Ingesting the same content twice wastes LLM quota and creates duplicate entities in the knowledge graph, so LightRAG checks two rules — filename and content: hitting either marks the attempt a duplicate, records it as `FAILED`, and **never overwrites the existing document**. Every check completes **before** chunking, entity extraction and any graph write — but that is not the same as "before anything is stored": `native` / `mineru` / `docling` must persist a pending-parse record and then the parse result before a content hash exists at all, so that path briefly holds a record that is subsequently rolled back (see §7.1, §7.2).

### 7.1 The Two Dedup Rules

**Rule one: duplicate filename.** Only the filename itself is compared (no directory, no workspace path), so `/data/a.pdf`, `inputs/a.pdf` and `a.pdf` are the same name. Any recognized `[engine-options]` hint is stripped first, so `abc.docx`, `abc.[native].docx` and `abc.[native-iet].docx` all count as the same name; an unrecognized hint is not stripped, so `abc.[draft].docx` is still a different name. If `doc_status` already holds a record under that name, the attempt is a duplicate **whatever state that record is in — `PENDING`, `PARSING`, `ANALYZING`, `PROCESSING`, `FAILED` or `PROCESSED`**.

**Rule two: duplicate content.** What is compared is the **extracted text**, not the raw file bytes. Rename the file, or convert it to another format — as long as the extraction produces the same content it is still a duplicate. The value depends on the content format:

| `parse_format` | The content hash comes from |
| --- | --- |
| `raw` | MD5 of the text after encoding sanitization |
| `lightrag` | MD5 of the sidecar's `*.blocks.jsonl` file (relative paths resolve against `INPUT_DIR`) |
| `pending_parse` | Not computed yet; filled in once parsing actually completes (so empty content cannot cause a false match) |

That also decides *when* the content check happens: `legacy` extracts text locally and can check at enqueue time; `native` / `mineru` / `docling` can only check once parsing has really finished, and at that point the attempt **stops before anything is built** — no chunking, no entity extraction, no graph writes — and the `full_docs` row written for this attempt is deleted.

Two additional rules:

- **The text endpoints** (`/documents/text`, `/documents/texts`) must supply a valid `file_source`, whose basename is used for the name check; without one they return 400.
- **Sourceless documents** (SDK `insert` / `ainsert` without `file_paths`, recorded as `file_path=unknown_source`) take no part in name dedup — `unknown_source` is only a placeholder, so two sourceless documents never collide on it — but they **are still deduplicated by content**: inserting the same text again is still a duplicate. The empty string and `no-file-path` behave the same way.

Neither documents within one enqueue batch nor two concurrently issued enqueues can both slip past the check: the later one is always recognized as a duplicate and recorded as `FAILED`.

### 7.2 What You Will See

| Symptom | What it means |
| --- | --- |
| Upload returns 409 `Document storage already contains '<name>' (Status: …)` | `doc_status` already holds a record under that name (any status, including `FAILED`) |
| Upload returns 409 `Input directory already contains a file with the same canonical basename …` | No record exists, but `INPUT_DIR` still holds a file with that name |
| An extra `FAILED` row appears with `error_msg` `File name already exists. Original doc_id: …, Status: …` | A batch enqueue or a scan hit the filename rule |
| `FAILED` with `error_msg` `Identical content already exists under another filename. Original doc_id: …` | The content rule was hit |
| `FAILED` with `error_msg` `N existing documents already share this file name …` | A legacy "one filename, several primary records" conflict that has to be repaired by doc id first |

The upload path is **fail-fast**: the first two cases return 409 outright, writing no file and leaving no trace in `doc_status`. The other three do leave a record in the document list, but **the records have different shapes** — do not go looking only for the `dup-` prefix:

- **A duplicate caught at enqueue time** (same name, or a content hash that already exists on the `legacy` path): a new `FAILED` record with a doc id prefixed `dup-` is created, and the original document is untouched. Its `metadata` carries `duplicate_kind` (`filename` / `content_hash`), `original_doc_id` and `original_track_id`, naming the copy that was kept.
- **A content duplicate found only after parsing** (`native` / `mineru` / `docling` have no content hash at enqueue time): **no** `dup-*` record is created. Instead this document's own `doc-*` record is flipped to `FAILED` in place with `metadata.is_duplicate=true` and `duplicate_kind=content_hash`, the `full_docs` row written for this attempt is deleted, and the source file is archived into `__parsed__`.
- **`filename_conflict`**: several primary records already share the filename and the system **deliberately does not pick a primary for you**, so `original_doc_id` on this record does not name "the copy that was kept" — repair the conflict first, per [§7.3](#73-what-to-do-about-it).

### 7.3 What To Do About It

- **To replace a same-named document with new content**: delete it first (`POST /documents/delete_document`, or the delete dialog in the WebUI document list), then upload. An upload never overwrites an existing record.
- **To clear out duplicate records**: both the `dup-*` rows and an original row flipped to `FAILED` after parsing are **inert records** — they carry no content, and both `/documents/scan` and `/documents/reprocess_failed` skip them, so they neither re-run nor fail again. The only way to remove one is an explicit delete; leaving it in place does not affect retrieval.
- **To re-run the same file with a different engine or different options**: dedup is not the obstacle, the frozen configuration is — see [§8.3](#83-retrying-after-a-failure) and [§9.3](#93-branch-b-already-extracted); this case requires delete + re-upload.
- **`filename_conflict` (one name, several records)**: list the conflicts with `GET /documents/source_conflicts` and repair them with `POST /documents/source_conflicts/repair`, naming the `primary_doc_id` to keep (dry-run by default; submit again with the returned `candidate_count` / `fingerprint`, and a changed candidate set returns 409), or run `python -m lightrag.tools.source_conflict_repair` offline. Scan again once repaired.

### 7.4 What a Directory Scan Does Differently

`/documents/scan` applies the same dedup rules, but it faces both sides at once — the files on disk and the records in storage — so it handles same-named files with extra automatic steps, precisely so that "fix the configuration, fix the source file, scan again" works directly instead of requiring a manual delete for every file:

| The record in storage | What the scan does |
| --- | --- |
| No record under that name | Enqueued as a new document; the whole batch is ordered by file modification time, oldest first |
| Already `PROCESSED` | Not reprocessed; the source file is archived into `__parsed__` and a warning is logged |
| `FAILED`, and content is confirmed to have never been extracted | The failure record is deleted and the file goes through as a brand-new document — this is what makes "fix the source file and scan again" work |
| Interrupted mid-flight (`PENDING` / `PARSING` / `ANALYZING` / `PROCESSING`) | Both the record and the source file are kept as-is; the pipeline resumes them without re-extracting |
| A different physical file under the same name / a record with unknown origin / one name with several records | Neither enqueued nor deleted, only archived or reported — these need a human decision, see [§7.3](#73-what-to-do-about-it) |

Two operational notes:

- **If a scan is interrupted (cancelled, crashed, restarted) nothing at all is enqueued.** Discovery and enqueue are two separate steps, the source files never leave `INPUT_DIR`, and the next scan simply rediscovers them. The one thing that does not come back is a failure record already deleted per row three above — the file re-runs as a new document, but the record kept for human review is gone.
- **`SCAN_SPOOL_DIR` must point at real, writable local disk.** The candidate list built during a scan is stored under `SCAN_SPOOL_DIR`, falling back to `WORKING_DIR/scan_spool`, both further split per workspace. Do not point it at tmpfs (on many Linux hosts `/tmp` is RAM-backed); set it explicitly when `WORKING_DIR` itself is a network volume. If the directory is unusable the scan **fails outright and enqueues nothing**, and the error names the variable; input files are untouched, so nothing is lost by fixing the configuration and scanning again.

## 8. Operating the Pipeline: Uploading While It Runs, Stopping, Retrying

This chapter answers three runtime questions: whether you can keep uploading once the pipeline is running, how to stop it, and what to do after a document fails. Concurrency tuning and request admission limits close the chapter.

### 8.1 What You Can Do While Processing Is Running

| What you want to do | Processing documents | Scan classification phase | Clearing / deleting | Manual retry draining |
| --- | :-: | :-: | :-: | :-: |
| Upload a file / insert text | ✅ allowed | ❌ 409 | ❌ 409 | ❌ 409 |
| `/documents/scan` | ⛔ refused | ⛔ refused | ⛔ refused | ⛔ refused |
| Delete / clear documents | ⛔ refused | ⛔ refused | ⛔ refused | ⛔ refused |
| Query / retrieval | ✅ not refused | ✅ not refused | ⚠️ not refused, results not guaranteed | ✅ not refused |

Key points:

- **You can keep uploading during a long batch — no need to wait.** This is the most commonly misread part: document processing itself does **not** block uploads. Once the new document's record is written, the running pipeline picks it up on its own — possibly folded into the current batch, possibly at the batch boundary — and neither case needs anything from you.
- **Scan, delete and clear need an idle pipeline.** A scan reads storage while deciding what to do with each file on disk ([§7.4](#74-what-a-directory-scan-does-differently)), and delete/clear tear storage down directly; neither can interleave with concurrent writes. When refused, `/documents/scan` returns HTTP 200 with `status="scanning_skipped_pipeline_busy"` and delete/clear return `status="busy"` — neither is an error, just retry once things are idle.
- **Queries are not subject to pipeline admission, but results are not guaranteed during a clear/delete.** The query endpoints do not check pipeline state, so they are never refused with a 409 the way an upload is; but `/documents/clear` concurrently drops a dozen storages via `asyncio.gather` — text chunks, entity / relation / chunk vectors, the knowledge graph, `full_docs` and `doc_status` — while a query holds no consistent snapshot and does not wait for it to finish, so a query issued mid-clear may see empty results, partial results, or a storage error. "The request is accepted" and "the result is consistent" are different guarantees: wait for a destructive operation to finish before querying.
- **When an upload is refused, the message identifies the cause** — three distinct 409s, one per state:
  - `Document scan is classifying files. …` — a scan is in its classification phase.
  - `Pipeline is clearing or deleting documents. …` — a clear or delete is running.
  - `A retry of failed documents is draining the pipeline. …` — a manual retry is draining the pipeline ([§8.3](#83-retrying-after-a-failure)).
- An upload returning **413 / 429** has nothing to do with pipeline busyness; those are request admission limits, see [§8.7](#87-admission-and-request-limits). For **503**, see [§8.5](#85-everything-returns-503-the-recovery_required-fence).

### 8.2 Stopping a Running Pipeline

```text
POST /documents/cancel_pipeline      # no request body
```

The WebUI entry point is the **Cancel** button inside the "Pipeline Status" dialog on the document management page, clickable only while the pipeline is running and no cancellation has been requested yet. The endpoint returns `{"status": "cancellation_requested" | "not_busy", "message": …}`; `not_busy` means nothing is running and there is nothing to cancel.

**A 200 does not mean it has stopped.** All it does is set a cancellation-requested flag. Cancellation is **cooperative**: it takes effect between stages, at batch boundaries, and at the 0.5-second poll inside multimodal analysis — it **never interrupts an LLM call already in flight**. So expect to wait for the current step to finish; `cancellation_requested` in `GET /documents/pipeline_status` confirms the request was received.

Once it stops, documents end up in one of these states:

| Where the document was | State after the stop |
| --- | --- |
| Already finished | `PROCESSED`, kept, unaffected |
| Already taken into the batch (parsing / analyzing / extracting, or queued) | `FAILED`, with `error_msg` like `User cancelled during parse: <filename>` |
| Not yet taken into the batch | Stays `PENDING`, resumes on the next trigger |

A few follow-ups:

- **Completed work is not wasted.** The LLM cache and any multimodal items that analyzed successfully are flushed to disk, so a later retry hits the cache instead of paying again.
- ⚠️ **Documents marked `FAILED` are not retried automatically** — you have to recover them with one of the explicit retries in [§8.3](#83-retrying-after-a-failure).
- **Uploads keep working during a cancellation.** Newly uploaded documents are picked up by the next run, once this one has exited.
- **When it does not apply**: a scan's classification phase does not count as "pipeline busy", so the call returns `not_busy` there, and there is currently no endpoint to cancel a scan job — just let it finish. Delete/clear jobs **can** be cancelled, per document: what was already deleted stays deleted, and the rest is reported as not deleted in the response.

**Restarting or killing the server** is the other way to stop, with different consequences:

- In-flight documents are left at `PARSING` / `ANALYZING` / `PROCESSING`; they are **not** written as `FAILED`.
- Those interrupted states are **automatically reset to `PENDING` and re-run** the next time the pipeline starts, with no manual step.
- But **starting the server does not start a run by itself**: it takes an upload, or a call to `POST /documents/scan`, to trigger one. So documents sitting at `PROCESSING` after a restart are not cause for alarm — just trigger a run.

### 8.3 Retrying After a Failure

There are three ways to recover a failed document, widest coverage first:

| Method | What it does | When to use it |
| --- | --- | --- |
| `POST /documents/scan` (the WebUI "Scan/Retry" button) | First resets every recoverable `FAILED` back to `PENDING`, then scans `INPUT_DIR` for new files; it can also handle failure records whose content was never extracted (delete the record and re-run the file as a new document) | **First choice**, widest coverage |
| `POST /documents/reprocess_failed` | Retries `FAILED` records from storage only, with no directory discovery; but a document that failed *during parsing* is **re-parsed**, which still reads the source file its record points at | When you do not want to trigger a full directory scan |
| Delete + re-upload | Starts over completely | You need a different engine or different processing options, or neither of the above can recover it |

Four rules you have to know:

1. **Each retry request grants each document exactly one attempt.** Fail again and it stays `FAILED` until the next explicit retry — it never spins.
2. **Automatic resume does not touch `FAILED`.** The run triggered by a new upload only recovers interrupted states (`PENDING` / `PARSING` / `ANALYZING` / `PROCESSING`); `FAILED` needs one of the two explicit entry points above.
3. **Whether it re-parses depends on whether content was extracted successfully.** A document that already has content is not re-parsed: the retry restarts from the multimodal analysis stage, first purging the chunks and graph contributions written by the previous run (details in [§9.3](#93-branch-b-already-extracted)) — which is why "fix the VLM configuration / wait out the rate limit, then retry" works. A document that **failed during parsing** (only a `pending_parse` placeholder in `full_docs`) is reset to `PENDING` and re-enters the parse stage, reading the source file in `INPUT_DIR` again and calling the engine again — so retrying after fixing MinerU / Docling works, but such a retry fails again if the source file has been deleted.
4. **A retry does not change the engine or the processing options.** `parse_engine`, `process_options` and `chunk_options` are frozen into the record at enqueue time; editing `.env` or a filename hint only affects new uploads.

Which failures a retry fixes, and which need a delete + re-upload:

| Failure cause | Retry in place? |
| --- | --- |
| Transient LLM / VLM / storage / network failure, or a rate limit | ✅ just retry |
| Cancelled by the user (`User cancelled during …`) | ✅ just retry |
| VLM not configured (`VLM analysis required but VLM role is not available`) | ⚠️ retry works once the VLM is configured; to drop image analysis instead (remove `i`), delete and re-upload |
| External parser service down / wrong endpoint | ⚠️ retry after fixing the configuration; a raw-bundle cache hit avoids calling the external service again (§6.3) |
| You want a different parser engine, chunking strategy, or `i/t/e/!` | ❌ delete and re-upload ([§9.3](#93-branch-b-already-extracted)) |
| Bad filename hint or invalid chunk parameters (rows prefixed `[File Extraction]` that never produced content) | ❌ `reprocess_failed` skips them; fix the filename and use `/documents/scan`, or delete the record and re-upload |
| A scanned PDF routed to `legacy`, which extracts no text | ❌ route the suffix to `mineru` / `docling` first, then delete and re-upload ([§3.2](#32-using-the-legacy-content-extractor)) |
| A `dup-*` duplicate record | ❌ inert record; a retry will not touch it, just delete it ([§7.3](#73-what-to-do-about-it)) |
| Deleting the document returns 409 (missing recovery anchor) | ❌ retrying unchanged is refused again; run `audit_kg_integrity(..., apply=True)` first |

Two more things about `/documents/reprocess_failed`. It first freezes ingestion and drains the pipeline to idle — during which uploads return 409 and `/documents/scan` is refused — then rewrites `FAILED` back to `PENDING` with no worker running, and finally resumes normal processing; with many documents this takes a while. It can return 429 (too many unacknowledged retry requests, capped by `MAX_UNACKED_MANUAL_RETRIES`) or 503 (the fence is up, or a clear/delete is running); a drain that cannot reach idle raises the fence, see [§8.5](#85-everything-returns-503-the-recovery_required-fence).

### 8.4 Deleting a Document: What the Two Checkboxes Remove

Both checkboxes in the delete dialog default to unchecked; they decide whether the on-disk artifacts go away too:

- **Neither checked**: only storage state is removed — chunks, vectors, graph contributions, `doc_status`, `full_docs`. The source file on disk, the archived copy and `.parsed/` sidecar under `__parsed__`, and the external engines' raw artifact bundles are **all kept**.
- **"Also delete uploaded files"** (API parameter `delete_file=true`): additionally removes the source file in `INPUT_DIR`, the archived copy and `<base>.parsed/` sidecar under `__parsed__`, and the `<base>.mineru_raw/` / `<base>.docling_raw/` / `<base>.native_raw/` raw artifact bundles. **If you want to re-run with a different engine, or to make an external engine genuinely re-parse, you must check this**, otherwise the re-upload hits the cache and gets the old result back ([§3.7](#37-parse-cache-and-forced-re-parse), §6.3).
- **"Also delete extracted LLM cache"**: additionally clears that document's extraction-stage LLM cache, so a re-upload really re-runs the LLM instead of hitting the cache. Check it when you want to verify the effect of a new model or new prompt.

### 8.5 Everything Returns 503: the `recovery_required` Fence

Some failures leave a workspace in a state where continuing would only be guesswork. The pipeline does not guess; it raises the `recovery_required` fence, after which **every** write operation (upload / text / scan / manual retry / delete / clear) returns **HTTP 503** until an operator lifts it explicitly. Three situations raise it — of which 1 and 3 depend on cross-process dead-owner detection and therefore only occur on **Linux with multi-worker Gunicorn** (single-process Uvicorn loses its coordination state with the process):

1. **A worker died in the middle of `custom_chunks` / `delete` / `clear`.** These may be half-committed, so they cannot simply be re-run. (A dead `processing` / `scan` owner is re-runnable, is reclaimed silently, and raises no fence.)
2. **A manual retry's drain cannot reach idle.** Two ways it gets stuck: the same documents keep coming back with no change in state (re-checking could only spin), and documents the drain can never advance at all — rows holding an **unfinished custom-chunk operation**, which only `/documents/scan`'s rollback can resolve. In both cases the reset does not run, the retry request stays unacknowledged (its one attempt is still owed), and the fence message carries a bounded sample of the blocking document ids. `recovery_kind` distinguishes them: `manual_drain_stalled` and `manual_drain_blocked`.
3. **Whether an owner is alive cannot be determined.** Reclaiming an owner's hold requires proving its process is really dead; a holder record without a process identity can never prove it, so the pipeline does not reclaim on a guess and the fence provides the exit instead.

`GET /documents/pipeline_status` reports `recovery_required` (boolean), `recovery_kind` (coarse reason) and `recovery_message` (the same text as the 503, with a bounded sample of blocking documents for some reasons).

Lifting the fence:

```text
POST /documents/recovery/force_reset
```

This is an **unsafe manual override** — it repairs nothing. Besides the fence it also **cancels the workspace's queued manual retry requests**, and that is required rather than incidental: while a request is queued `/documents/scan` refuses to run (a scan runs its own exclusive `FAILED` reset and may not jump the queue), so clearing only the fence would leave the recovery path just as blocked. The response reports `cancelled_manual_retries`. No document is lost — failed documents stay `FAILED` and are handled by the next retry request or by the scan's own reset. Because both halves are required, the call is **all-or-nothing**: if the queued requests cannot be cancelled the endpoint returns **503** and the fence stays up, so you can retry to complete the recovery rather than have the API report a recovery that did not happen.

Recovery order:

| Reason | Action |
|---|---|
| `manual_drain_blocked` | `POST /documents/recovery/force_reset`, then `POST /documents/scan` — the scan rolls back the unfinished operations **and** runs the `FAILED` reset itself, so no separate retry call is needed. |
| `manual_drain_stalled` | `POST /documents/recovery/force_reset`, then investigate the documents named in `recovery_message` — the fence cannot say why they are stuck. Re-issue `POST /documents/reprocess_failed` once they are resolved. |
| A worker died during `custom_chunks` / `delete` / `clear` | Do not reach for `force_reset` — follow "Repairing half-committed storage" below. |

**Can a restart just fix it?** There is exactly one test — **is the blocker in memory or in storage?** A restart clears only runtime coordination state (the fence and owner records in `pipeline_status`, the queued manual retry requests in the ingress; none of it is persisted). Anything written to `doc_status` / `full_docs` / the storages survives untouched.

| Cause | Does a restart fix it? | Why |
| --- | :-: | --- |
| Owner liveness undeterminable | ✅ Yes | The stuck reservation record lives in cross-process shared state, so restarting the process group removes it, and no storage was ever touched — this is the cleanest fix |
| `manual_drain_stalled` | ⚠️ Usually | The fence and the queued requests go away with the restart; if the active rows that "keep coming back unchanged" are dead-process `PROCESSING` / `PARSING` / `ANALYZING` orphans, they are automatically reset to `PENDING` after the restart and re-run on the next trigger, so the blocker disappears. But a restart diagnoses nothing: if they stall on the same state for another reason, the next `/documents/reprocess_failed` stalls again |
| `manual_drain_blocked` | ❌ No | The blocker is the unfinished custom-chunk journal in `doc_status.metadata`, which **is persisted** and survives the restart intact. A restart only clears the fence and the queued requests (still necessary — a queued request makes `/scan` refuse its own reservation); the actual rollback has to come from `POST /documents/scan` |
| A worker died during `custom_chunks` / `delete` / `clear` | ❌ No | What is half-committed is the storage itself — see below |

A restart is therefore a "gentler `force_reset`": both clear the fence and the queued requests and neither repairs anything, except that a restart additionally resets interrupted documents to `PENDING` (which is exactly why a stall often heals itself), at the cost of downtime. Prefer a restart when you can take one; use `force_reset` when you cannot and have confirmed storage was never touched (causes 2 and 3).

#### Repairing Half-committed Storage: the Two Offline Tools vs `force_reset`

The fence itself is **not persisted** (it lives in `pipeline_status`), so **restarting the service clears it along with the queued requests**. That is what decides where `force_reset` belongs:

- **Use it for causes 2 and 3.** In both, the reset never ran and storage was not touched at all — only scheduling is stuck, and stopping the service for that is not worth it.
- **Avoid it for cause 1.** It clears a flag and repairs nothing; once cleared, every write (including concurrent uploads) resumes immediately against possibly half-committed storage. Both offline tools below already require a stopped service — and stopping it removes the fence anyway, so `force_reset` never needs to appear.

What each offline tool can and cannot repair:

| Tool | Source of truth and what it repairs | What it cannot repair | Cost |
| --- | --- | --- | --- |
| `python -m lightrag.tools.kg_integrity_repair` | Walks graph → chunk `source_id` → `text_chunks` → `full_doc_id` to rebuild missing or unusable `full_entities` / `full_relations` anchor rows; can also write empty anchor rows for a document that genuinely owns nothing | Graph↔VDB drift, `doc_status` / `full_docs` themselves, custom-chunk journals, the fence itself | Never calls the LLM or embedder — effectively free |
| `lightrag-rebuild-vdb` | Drops and rebuilds `entities_vdb` / `relationships_vdb` from the graph and `chunks_vdb` from `text_chunks`; also clears reverse orphans (present in the vector store, absent from the graph), which incremental repair cannot do | Anchor rows, `doc_status`, the correctness of the graph itself, the fence itself | A full re-embed — real money |

Two ordering traps; getting them backwards turns repairable data into unrepairable data:

- **`kg_integrity_repair` must run before any further deletion.** It can only rebuild anchors from **surviving** chunk provenance; if a half-finished delete already removed the `text_chunks` rows, those contributions become irrecoverable orphans that the tool can only report and will never modify on its own. There is no reason not to run report mode (without `--apply`) first.
- **`lightrag-rebuild-vdb` must run last.** It treats the graph and `text_chunks` as the truth: graph objects a half-finished delete should have removed will be faithfully re-embedded back into the vector store. It guarantees that vectors are **consistent** with the graph, not that the graph is **correct**. So settle the graph side first, then consider rebuilding vectors.

The full order for cause 1 is therefore:

1. **Stop the service** (the fence disappears with it — do not reach for `force_reset`).
2. Run `python -m lightrag.tools.kg_integrity_repair --verbose` for the report first: anchor gaps, plus the irrecoverable orphans that are already beyond saving. Run `--apply` only if there are gaps.
3. Start the service and re-run the operation that did not finish — a whole-document purge is journaled and resumes after the phases it already completed; `clear` is simply re-run; a `custom_chunks` rollback is triggered by `POST /documents/scan`.
4. Once things are stable, if you suspect graph↔VDB drift, stop the service again and run `lightrag-rebuild-vdb`, using its read-only consistency check first to decide whether the embedding cost is worth paying.

Both tools must run with the service stopped and the workspace idle (concurrent writes make them read a moving target), and `lightrag-rebuild-vdb` must use the same `.env` as the server or the rebuilt vectors land in a different embedding space. See [README_KG_INTEGRITY_REPAIR.md](../lightrag/tools/README_KG_INTEGRITY_REPAIR.md) and [README_REBUILD_VDB.md](../lightrag/tools/README_REBUILD_VDB.md).

### 8.6 Pipeline Concurrency Parameters

The previous sections are about the correctness question of "who may write"; this set of parameters answers the throughput question of "how many workers run at once". The pipeline has 3 stages, and each stage's worker pool is sized independently:

```
          ┌─ parse_queues["native"]  ─► [native pool  × N1] ─┐   ← legacy shares this pool
PENDING ─►├─ parse_queues["mineru"]  ─► [mineru pool  × N2] ─┼─► q_analyze ─►[analyzer × N4] ─► q_process ─►[processor × N5]
          ├─ parse_queues["docling"] ─► [docling pool × N3] ─┤
          └─ parse_queues[<3rd-party>] ─► [custom pool]   ──┘   ← created dynamically per ParserSpec.queue_group
```

Parse queues are **created dynamically from the registry's `ParserSpec.queue_group`** (one registry snapshot per batch): built-in native/mineru/docling each get a group, legacy shares the native pool (local, no network), and third-party engines can declare their own group and concurrency (see [ThirdPartyParser.md](./ThirdPartyParser.md)). At enqueue time each document's parser engine (from the `LIGHTRAG_PARSER` default or a filename hint) decides which parse queue it lands in; the parse queues **never block one another** — a saturated mineru queue does not slow docling or native down. After parsing, everything converges on `q_analyze` (multimodal analysis) and then `q_process` (entity/relation extraction + ingestion).

| Environment variable | Default | Role | Tuning advice |
| --- | --- | --- | --- |
| `MAX_PARALLEL_PARSE_NATIVE` | `5` | N1: concurrent workers for native parsing (docx / pdf / txt, all local) | Pure CPU with a low memory footprint; scale with core count |
| `MAX_PARALLEL_PARSE_MINERU` | `2` | N2: concurrent workers for MinerU parsing | MinerU is GPU/CPU heavy, so **2 is a moderate default**. Drop to 1 when resources are tight; 2-3 for a local deployment with enough VRAM; higher against MinerU's official cloud service (subject to its quota) |
| `MAX_PARALLEL_PARSE_DOCLING` | `2` | N3: concurrent workers for Docling parsing | Docling is equally resource-sensitive, so **2 is a moderate default**. Drop to 1 when resources are tight; 2-3 for a local deployment with enough CPU/GPU |
| `MAX_PARALLEL_ANALYZE` | `5` | N4: concurrent workers for multimodal analysis (VLM image / table descriptions) | Consumes VLM quota directly. Keep ≤ the VLM service's concurrency limit |
| `MAX_PARALLEL_INSERT` | `3` | N5: concurrent documents in the entity/relation extraction + ingestion stage | `MAX_ASYNC_LLM / 3` is a good rule of thumb, in the 2~10 range. Each document triggers many LLM calls here, so too high hits LLM rate limits. The same value also backs an `asyncio.Semaphore` as a second constraint (worker count equals the semaphore value) |
| `QUEUE_SIZE_PARSE` | `20` | Input queue length for parse (native/MinerU/Docling) | Rarely needs tuning. The queue holds only lightweight doc_ids (large document bodies are stripped before analyze), and it just bounds how many documents the pipeline pre-dispatches to parse workers |
| `QUEUE_SIZE_ANALYZE` | `100` | Bounded capacity of the analyze queue (parse → analyze) | Rarely needs tuning. Raise it slightly for very large batches (tens of thousands) to avoid back-pressure at the enqueue side; lower it when memory is tight |
| `QUEUE_SIZE_INSERT` | `4` | Queue capacity between the analyze and process stages | process is the slowest and most memory-hungry stage, so this queue is deliberately small, giving upstream back-pressure |

**A few key points:**

1. **The parse stage is isolated per engine**, so mixing native/mineru/docling never lets one slow engine drag another down.
2. **mineru / docling default to 2**: both are resource-heavy, so the default stays moderate. Drop to 1 when resources are tight (avoiding OOM / VRAM contention / failure retries); raise it by hand if you have multiple GPUs or a dedicated parsing server.
3. **`MAX_PARALLEL_INSERT` is both pool size and semaphore ceiling**: the pipeline creates `Semaphore(max_parallel_insert)` and every process worker takes it before extracting and ingesting. So even if you raise the worker count by hand, this value still caps real concurrency — just tune it directly.
4. **Queue size and back-pressure**: the small `QUEUE_SIZE_INSERT=4` default is deliberate — process is slow and memory-hungry, so a full queue blocks the analyze stage and back-pressures parse, instead of piling tens of thousands of parse results into memory at once.
5. **How changes take effect**: every parameter comes from `.env` (or the environment) and is read once when the `LightRAG` instance is constructed; restart the service after changing one.
6. **Chunking does not scale with concurrency**: chunking runs in a dedicated single-worker thread pool so it does not block the event loop, and its concurrency does not grow with `MAX_PARALLEL_INSERT` — raising that will not make chunking faster. A custom `chunking_func` still runs on the event loop (its contract allows touching the running loop), so CPU-heavy implementations should call `asyncio.to_thread` themselves.

**Typical tuning scenarios:**

- Many PDFs + local MinerU on a single GPU: `MAX_PARALLEL_PARSE_MINERU=2`, `MAX_PARALLEL_ANALYZE=5`, `MAX_PARALLEL_INSERT=3` (the defaults; drop MINERU to 1 when VRAM is tight).
- Many PDFs + MinerU's cloud service: `MAX_PARALLEL_PARSE_MINERU=3~5` (per your cloud quota), everything else default.
- Pure docx / txt (native only): `MAX_PARALLEL_PARSE_NATIVE=10`, with `MAX_PARALLEL_INSERT` derived from `MAX_ASYNC_LLM/3`.
- Visible LLM rate limiting: lower `MAX_PARALLEL_INSERT` first (the process stage makes many LLM calls per document), then `MAX_PARALLEL_ANALYZE` (VLM has its own quota).

### 8.7 Admission and Request Limits

Before a document reaches any of the mechanisms above, the server may refuse it outright. These variables decide whether an upload is rejected or queued:

| Variable | Default | How it refuses |
| --- | --- | --- |
| `MAX_UPLOAD_SIZE` | `104857600` (100 MB) | `413` — a single uploaded file is too large |
| `MAX_REQUEST_BODY_BYTES` | `1048576` (1 MiB) | `413` — the raw request body is too large. It applies to **all** routes, in tiers: ordinary routes take this value, `/documents/text` and `/documents/texts` take a built-in 50 MiB **when the variable is unset**, and `/documents/upload` derives its limit from `MAX_UPLOAD_SIZE` + 1 MiB. Setting any positive value explicitly (including the 1 MiB default) applies it uniformly to every non-upload route. `0` disables all of them |
| `MAX_TEXTS_PER_REQUEST` | `0` (off) | `413` — too many texts in a single `/documents/texts` call |
| `MAX_PENDING_DOCUMENTS` | `0` (off) | `429` — too many documents already in PENDING / PARSING / ANALYZING / PROCESSING |

Their full semantics (including how `MAX_UPLOAD_SIZE` relates to a reverse proxy's own body limit) are in [LightRAG Server](./LightRAG-API-Server.md).

Three more knobs belong to the pipeline itself rather than to request admission:

| Variable | Default | Meaning |
| --- | --- | --- |
| `PIPELINE_SCHEDULING_PAGE_SIZE` | `500` | Keyset page size for the doc_status backlog scan; `0` disables paging |
| `PIPELINE_REQUIRE_STRICT_STORAGE_READS` | `false` | Refuse to start when the doc_status backend cannot serve strict reads. This directly decides whether the "delete the failure record and re-run the file as a new document" row in [§7.4](#74-what-a-directory-scan-does-differently) can act at all — without a reliable point read, nothing is deleted |
| `MAX_UNACKED_MANUAL_RETRIES` | `64` | Per-workspace ceiling on published-but-unacknowledged manual retry requests ([§8.3](#83-retrying-after-a-failure)) |

## 9. Pipeline Resume Rules at Startup

Each time `apipeline_process_enqueue_documents` starts up, it pulls all documents in `PARSING` / `ANALYZING` / `PROCESSING` / `PENDING` / `FAILED` to continue processing. The resume path **branches by "whether content has been extracted"**, ensuring that any document, regardless of its previous progress, has an idempotent result when resumed under the current `process_options`.

The resume rule only applies to documents whose `doc_id` already exists in `doc_status`. New files joining the queue require the file dedup logic in §7, to avoid new files squeezing out the records of files whose content has already been successfully extracted.

### 9.1 Determining "Content Has Been Extracted"

Read `full_docs[doc_id]`:

| `parse_format` | Verdict |
| --- | --- |
| `lightrag` and `lightrag_document_path` file exists | ✅ extracted |
| `raw` and `content` is non-empty | ✅ extracted |
| Other (including `pending_parse`, missing record) | ❌ not extracted |

### 9.2 Branch A: Not Extracted

Go through the full pipeline (registry-dispatched parsing `get_parser(engine).parse(...)` → `analyze_multimodal` → chunking → entity extraction), with each stage's behavior determined by `full_docs.process_options`. This is the normal flow of a "first-time enqueue".

### 9.3 Branch B: Already Extracted

**Always skip parsing** (do not call `parse_*` again), restart from the ANALYZING stage, clear old chunks / entities, and redo per the current `process_options`:

| Sub-step | Behavior |
| --- | --- |
| Engine comparison | If the engine implied by `process_options` ≠ `full_docs.parse_engine`, **only warn**, do not re-parse. The extracted content is an immutable fact; re-running a different engine would produce inconsistency. To switch engines, delete the whole document and re-upload it. |
| Old chunks / entities / relations cleanup | Read `status_doc.chunks_list` to collect old chunk id set, call `_purge_doc_chunks_and_kg(doc_id, chunk_ids)`: delete chunk rows from `chunks_vdb` / `text_chunks`; reverse-lookup affected entities / relations by `entity_chunks` / `relation_chunks`, directly remove entries that have lost all sources from the graph and vector store, and call `rebuild_knowledge_from_chunks` to rebuild with the remaining chunks for entries still contributed by other documents; finally delete the index rows of this doc in `full_entities` / `full_relations`. After purge completes, `status_doc.chunks_list = []` / `chunks_count = 0` are reset to avoid the subsequent state-machine upsert writing back old IDs. |
| `analyze_multimodal` | For enabled modalities, every run recomputes the sidecar item analysis and overwrites the existing `llm_analyze_result`. The LLM analysis cache still applies: a cache hit reuses the previous provider response, so semantic fields usually stay the same and only runtime fields such as `analyze_time` are rewritten. Cache misses, for example after changing the model or prompt, can produce different saved content. |
| Re-chunk | Pick the strategy by the new `process_options.chunking`, with parameters read from `full_docs.chunk_options` (the enqueue snapshot; not overwritten by resume; env changes do not affect old documents that still chunk by the parameters from the moment of enqueue). The LightRAG Document path uses paragraph_semantic when `process_options=P`, otherwise dispatches to F/R/V by selector. |
| Entity extraction / KG-skip | Determined by the new `process_options.skip_kg` |

> This rule guarantees: when users change `i/t/e` and re-upload the same-named document (delete the old doc first, then upload the file with the new hint), multimodal analysis is incrementally filled in; when changing `F/R/V/P`, chunks and graph are rebuilt; when changing `!`, KG construction is stopped or restored. Engine changes are considered a "major change", uniformly handled by delete + re-upload, not implicitly happening on the resume path.

## 10. Troubleshooting

| Symptom | Cause | What to do |
| --- | --- | --- |
| Upload rejected as an unsupported type | The extension is not in the live allowlist. Declaring `MINERU_ADDITIONAL_SUFFIXES` / `DOCLING_ADDITIONAL_SUFFIXES` alone does not make a bare `x.doc` uploadable (§3.1) | Add a routing rule or a per-file hint; confirm with `GET /documents/supported_file_types` |
| Server refuses to start after editing `LIGHTRAG_PARSER` | Startup validation is strict: unknown engine, malformed syntax, external engine without an endpoint, or a bad engine / chunk parameter (§2.6, §2.7, §3.6) | Read the startup error — it names the offending rule |
| A rule seems ignored; everything is parsed by `legacy` | The engine failed a usability check and the rule was skipped: either it does not support that extension, or its endpoint / token is unset (§2.7) | Configure the endpoint, or route the extension to an engine that supports it |
| A filename hint is ignored and a warning is logged | Illegal characters inside the brackets, or the options-only form written without its leading hyphen (§2.5) | Use `[-OPTIONS]`, e.g. `report.[-teP].docx` |
| `i` / `t` / `e` enabled but nothing was analyzed | The sidecar does not exist — the document has no such content, or the engine does not emit it (`MINERU_ENABLE_TABLE` / `MINERU_ENABLE_FORMULA`, `DOCLING_DO_FORMULA_ENRICHMENT`; `legacy` emits none) | Look for the INFO line naming the empty sidecar, and inspect `*.parsed/` (§4.2) |
| Document FAILED with "VLM analysis required but VLM role is not available" | `i` is enabled and an image passed the pre-filters while no VLM is configured (§4.3) | Set `VLM_PROCESS_ENABLE=true` with a vision-capable binding, **or** delete the document and re-upload it with `te` — retrying will not drop `i` |
| Some images are analyzed, others silently are not | Non-raster format, below `VLM_MIN_IMAGE_PIXEL`, or above `VLM_MAX_IMAGE_BYTES` (§4.3) | Read the item's `llm_analyze_result` message; raise the byte cap if the image is legitimate |
| Multimodal items fail on a token budget | The surrounding budgets leave too little for the item itself (§4.4) | Raise `MAX_EXTRACT_INPUT_TOKENS`, or lower `SURROUNDING_LEADING_MAX_TOKENS` / `SURROUNDING_TRAILING_MAX_TOKENS` |
| `P` was requested but the chunks look like `R` output | No structured `LightRAG Document` was produced, so `P` degraded (§2.7, §3.2) | Route the file to `native` / `mineru` / `docling` instead of `legacy` |
| Headings are wrong in a `.docx`, so `P` splits badly | The document's Word outline is unreliable | Try `native(smart_heading=true)` (§3.3) and iterate with the parser CLI |
| Log notes that paragraphs lack `paraId` | Produced by LibreOffice / WPS / older Word (§3.3) | Informational. Re-save in Word 2013+ only if you need paragraph-level provenance |
| Document is FAILED as a duplicate | Basename dedup (canonicalized, hint stripped) or content-hash dedup (§7.1, §7.2) | Delete the existing document first, or rename the new file; duplicate records (`dup-*`, or an original row flipped to FAILED after parsing) can only be removed explicitly (§7.3) |
| Upload returns `409` | A document with the same canonical basename already exists in the input directory or in doc-status (§7.2) | Delete it via `POST /documents/delete_document`, then upload again |
| Upload returns `413` or `429` | An admission limit was hit (§8.7) | See the limits table for which one |
| Everything returns `503` | The `recovery_required` fence is up (§8.5) | Follow the recovery order documented in that section |
| `/documents/scan` returns `scanning_skipped_pipeline_busy` | The pipeline is busy or scanning, uploads are in flight, or a manual retry is queued (§8.1) | Wait for idle; `POST /documents/reprocess_failed` is the single-call recovery for a stuck retry request |
| Cancel was clicked and a batch of documents turned `FAILED` | Documents already taken into the batch are marked FAILED on cancellation (§8.2) | Retry with `POST /documents/scan` or `POST /documents/reprocess_failed`; finished documents are unaffected |
| Cancel was clicked but the pipeline is still running | Cancellation is cooperative and never interrupts an LLM call already in flight (§8.2) | Wait for the current stage to end; `cancellation_requested` in `GET /documents/pipeline_status` confirms the request arrived |
| Documents stuck at `PROCESSING` / `PARSING` after a restart | Interrupted states recover automatically, but starting the server does not itself trigger a run (§8.2) | Call `POST /documents/scan` once, or upload any file to trigger one |
| Changed the engine or the options, but the output is unchanged | Both the engine and `process_options` are frozen into the doc-status record at enqueue. Automatic resume and `/documents/reprocess_failed` reuse the stored values; editing `LIGHTRAG_PARSER` or a hint affects new uploads only (§8.3, §9.3) | Delete the document (with "also delete file") and upload it again |
| An external engine keeps returning stale output after fixing its service config | The raw-artifact bundle cache was hit (§6.3) | Set the matching `LIGHTRAG_FORCE_REPARSE_*` flag (§3.7), or delete the document with "also delete file" |
| A scanned PDF fails with "extracted no usable text" | `legacy` cannot read a PDF with no text layer (§3.2) | Route it to `mineru` or `docling` with OCR enabled |
| MinerU rejects a multi-segment page range | Multi-segment ranges require `official` mode; `local` accepts a single page or one simple range (§3.6) | Use a single range under `local`, or switch modes |
| Startup fails asking for spaCy models | `DOCX_SMART_HEADING=true`, or a rule carrying `native(smart_heading=true)`, triggers a fail-fast model check (§3.3) | Run `lightrag-download-cache --spacy-install`, or use the main Docker image, which bundles them |

## 11. Python SDK Invocation

This chapter targets developers who **directly import the `LightRAG` class** for integration, covering runtime APIs, constructor parameters, and removed legacy interfaces that Server deployments don't use. Server users usually don't need to read this chapter.

### 11.1 Audience

```python
from lightrag import LightRAG
rag = LightRAG(working_dir="./rag_storage", ...)
await rag.initialize_storages()
await rag.ainsert("text", file_paths="doc.pdf")
```

The following behaviors of this invocation style differ from the Server path: you can change `addon_params["chunker"]` without restarting the process, you can pass per-file `chunk_options` into `apipeline_enqueue_documents`, and you can dynamically override the F strategy's pre-split parameters in an `ainsert` call.

### 11.2 LightRAG Constructor Parameters

`LightRAG(chunk_token_size=…, chunk_overlap_token_size=…)` is **tier 3** in §5.3's priority chain: "legacy constructor field". Strategy-agnostic and coarse-grained default, fills only slots still empty:

- Lower priority than `addon_params["chunker"]` explicit values (§11.3) and strategy-specific env (§5.2).
- Higher priority than the legacy env `CHUNK_SIZE` / `CHUNK_OVERLAP_SIZE`.
- The instance fields `self.chunk_token_size` / `self.chunk_overlap_token_size` are always back-filled to `int` after `__post_init__`, so legacy paths still reading these two fields (e.g., the `chunk_opts.get("chunk_token_size") or self.chunk_token_size` fallback in `pipeline.py`) continue to work.

### 11.3 Modifying `addon_params["chunker"]` at Runtime

`addon_params["chunker"]` is an `ObservableAddonParams` field; it can be **modified at runtime**:

```python
rag.addon_params["chunker"]["recursive_character"]["separators"] = ["##", "\n", " "]
```

After modification, **subsequent enqueues** get the new defaults; already-enqueued documents keep the snapshot from their enqueue moment (see the three layers of semantic guarantee in §5.3). This is tier 1 of §5.3's priority chain: "`addon_params["chunker"]` explicit value", winning everything.

Server deployments do not have this capability — after changing env, the service must be restarted for it to take effect.

### 11.4 `apipeline_enqueue_documents(chunk_options=…)`

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

For `apipeline_enqueue_documents`'s own concurrency constraints, see the state/action table in §8.1.

### 11.5 `ainsert(split_by_character=…, split_by_character_only=…)`

`LightRAG.ainsert(split_by_character=…, split_by_character_only=…)` runtime parameters are overridden into `chunk_options.fixed_token` by `resolve_chunk_options` at enqueue time:

- A non-`None` `split_by_character` overrides the env default;
- `split_by_character_only=True` overrides (`False` is the signature default, indistinguishable from "not specified", so the env default wins).

Only effective for the F strategy; other strategies' sub-dictionaries are unaffected.

### 11.6 Removed SDK Parameter: `reprocess_existing_non_processed`

The legacy `apipeline_enqueue_documents` behavior of `reprocess_existing_non_processed=True` would directly delete non-PROCESSED old records and rebuild them during scan, which conflicts with the rules in §7 / §8; it has been entirely removed. Replacement paths:

- Automatic resume: scan handles same-named files per the rules in §7.4 (archive / resume / delete the record then re-enqueue), uniformly picked up by the resume rules in §9 inside the processing loop.
- Forced refresh: first call `/documents/delete_document` to delete the old document, then upload the same-named new file.

## Appendix A. Notes on Upgrading from Legacy

### A.1 The global multimodal switch has been replaced by per-file options

`addon_params["enable_multimodal_pipeline"]` is deprecated. Multimodal analysis is now selected per document through the `i` / `t` / `e` processing options (§2.1), set either as a rule default in `LIGHTRAG_PARSER` (§2.4) or via a filename hint (§2.5). There is no global "analyze everything" flag any more, because the modality set a document actually contains is decided by the parsing engine and reported through sidecar existence, not by configuration.

Migration: replace the old global switch with the corresponding letters on your routing rules — for example `LIGHTRAG_PARSER=*:native-iteP,*:legacy-R`. Note that `VLM_PROCESS_ENABLE` is a separate, orthogonal switch: it gates **image** analysis only (tables and equations are analyzed by the `EXTRACT` role), and with `i` enabled but no VLM available an image that passes the pre-filters fails the document rather than being skipped.

Documents ingested before the upgrade keep the `process_options` frozen into their `doc_status` record at enqueue time; changing rules or hints does not retroactively alter them. To re-process an existing document under new options, delete it and upload it again (§9.3).

### A.2 Behavior defaults are unchanged unless you opt in

With `LIGHTRAG_PARSER` unset, every extension still routes to the `legacy` content extractor with the legacy chunking behavior, exactly as before the upgrade. The new engines and chunking strategies are opt-in; see §1 for three ready-to-use presets.

### A.3 Removed SDK parameter

`apipeline_enqueue_documents(reprocess_existing_non_processed=...)` has been removed; see §11.6 for the replacement paths.

## Appendix B. Environment Variable Quick Reference

Where to find each family of file-processing variables. This is an index, not a duplicate: the authoritative per-variable comments live in [env.example](https://github.com/HKUDS/LightRAG/blob/main/env.example).

| Family | Variables | Documented in |
| --- | --- | --- |
| Routing | `LIGHTRAG_PARSER` | §2.3, §2.4, §2.5 |
| Chunking | `CHUNK_SIZE`, `CHUNK_OVERLAP_SIZE`, `CHUNK_{F,R,V,P}_*` | §5.2 |
| Multimodal | `VLM_PROCESS_ENABLE`, `VLM_MAX_IMAGE_BYTES`, `VLM_MIN_IMAGE_PIXEL`, `MAX_EXTRACT_INPUT_TOKENS`, `SURROUNDING_*_MAX_TOKENS`, `MM_EXTRACT_CONTENT_MIN_TOKENS` | §4 |
| VLM / role models | `VLM_LLM_*`, `VLM_MAX_ASYNC_LLM` | [RoleSpecificLLMConfiguration.md](RoleSpecificLLMConfiguration.md) |
| legacy engine | `PDF_DECRYPT_PASSWORD` | §3.2 |
| native engine | `NATIVE_MD_IMAGE_*` | §3.3 |
| native docx smart_heading | `DOCX_SMART_HEADING`, `DOCX_SMART_*` tuning | §3.3 and the smart_heading block of `env.example` |
| MinerU | `MINERU_*` | §3.4 |
| Docling | `DOCLING_*` | §3.5 |
| Parse cache | `LIGHTRAG_FORCE_REPARSE_{NATIVE,MINERU,DOCLING}`, `{MINERU,DOCLING}_ENGINE_VERSION` | §3.7, §6.3 |
| Directories | `INPUT_DIR`, `WORKING_DIR`, `SCAN_SPOOL_DIR` | §6, §7.4 |
| Concurrency | `MAX_PARALLEL_*`, `QUEUE_SIZE_*` | §8.6 |
| Admission and limits | `MAX_UPLOAD_SIZE`, `MAX_REQUEST_BODY_BYTES`, `MAX_TEXTS_PER_REQUEST`, `MAX_PENDING_DOCUMENTS`, `PIPELINE_*`, `MAX_UNACKED_MANUAL_RETRIES`, `SCAN_ENQUEUE_BATCH_SIZE` | §8.7 |
| Query-time (not chunking) | `ENABLE_CONTENT_HEADINGS` — appends each chunk's heading path when assembling the answer context; it does not change chunk boundaries or stored chunk text | [LightRAG Server](./LightRAG-API-Server.md) |
| Offline / tokenizer | `TIKTOKEN_CACHE_DIR` | [OfflineDeployment.md](./OfflineDeployment.md) |
