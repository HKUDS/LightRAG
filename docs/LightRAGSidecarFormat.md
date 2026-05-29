# LightRAG Sidecar File Format Specification

This document describes the **LightRAG Sidecar** file format that content parsing engines output. When LightRAG uses multimodal-capable content parsing engines such as native/mineru/docling to extract file content, it splits "body text + multimodal objects + parsing metadata" into a `*.parsed/` directory. Each JSON / JSONL file in that directory is collectively called a **sidecar** file. Sidecars are the only reliable source of truth for the subsequent pipeline (multimodal analysis → multimodal chunk construction → entity extraction → cache cleanup on document deletion). The sidecar format is LightRAG's built-in universal file interchange format; new multimodal content extraction engines must follow this format. The purpose of publicly documenting the **LightRAG Sidecar** format is to make it convenient for community developers to write their own content parsing engines.

## 1. Overview

| Concern | File | Contents | Notes |
|---|---|---|---|
| Main file | `<doc>.blocks.jsonl` | Stores block body | Concatenating the `content` fields of all blocks reconstructs the complete original text |
| Drawing objects | `<doc>.drawings.json` | Drawing objects extracted from the file | Sent to a VLM for analysis; analysis results are written back |
| Table objects | `<doc>.tables.json` | Table objects extracted from the file | Sent to an LLM for analysis; analysis results are written back |
| Equation objects | `<doc>.equations.json` | Equation objects extracted from the file | Sent to an LLM for analysis; analysis results are written back |
| Original image assets | `<doc>.blocks.assets/` | Original image files extracted from the document | Sent to a VLM for image analysis |

Design intent of sidecars:

- During the parsing stage, the content extraction engine (native/mineru/docling) is **only** responsible for generating "objective" fields such as `blockid / heading / content / surrounding`;
- During the multimodal analysis stage (`analyze_multimodal`), the analysis result dict `llm_analyze_result` is written by LightRAG and may be appended or overwritten; parsers should not pre-populate it.

## 2. Directory Layout

```
inputs/space1/__parsed__/<canonical filename>.parsed/
├── <canonical filename>.blocks.jsonl        body block sequence + document-level meta (first line)
├── <canonical filename>.drawings.json       drawing sidecar (dict container, key = drawing id)
├── <canonical filename>.tables.json         table sidecar
├── <canonical filename>.equations.json      equation sidecar
└── <canonical filename>.blocks.assets/      original asset directory (image files referenced by drawings.json live here)
    ├── image1.wmf
    ├── image2.wmf
    ├── image3.wmf
    ├── image4.png
    ├── image5.png
    ├── image6.png
    └── image7.emf
```

## 3. blocks.jsonl

`blocks.jsonl` is JSON serialized line by line. The **first line has `type="meta"`**; every subsequent line is a content block with `type="content"`.

### 3.1 meta line example

```json
{
  "type": "meta",
  "format": "lightrag",
  "version": "1.0",
  "document_name": "m012-manual.docx",
  "document_format": "docx",
  "document_hash": "sha256:4840...3f9543d9db0822d2d59",
  "table_file": true,
  "equation_file": true,
  "drawing_file": true,
  "asset_dir": true,
  "split_option": { "fixlevel": 0 },
  "blocks": 39,
  "doc_id": "doc-f1bee60173d067d88595c00e7d9b0ce5",
  "parse_engine": "native",
  "parse_time": "2026-05-13T18:42:25.943490+00:00",
  "doc_title": "m012-manual"
}
```

| Field | Type | Description |
|---|---|---|
| `type` | `"meta"` | Line type, fixed value, sanity check |
| `format` | `"lightrag"` | Sidecar major version family identifier |
| `version` | `str` | Sidecar schema version |
| `document_name` | `str` | Canonical filename (with extension, without processing hints) |
| `document_format` | `str` | File format (currently expressed as the file extension) |
| `document_hash` | `"sha256:<hex>"` | Sidecar body fingerprint, defined as `SHA-256(merged_text)`, where `merged_text` is the concatenation of all non-empty content lines' `content` fields joined by `"\n\n"`. Used by external consumers to quickly determine whether two `.parsed/` directories share the same source (without line-by-line body comparison), and serves as a self-describing content checksum for the sidecar file. Note: the LightRAG ingestion pipeline itself does not read this field; cross-document deduplication is handled separately by `doc_status.content_hash`. |
| `table_file` / `equation_file` / `drawing_file` | `bool` | Whether the corresponding sidecar files exist (when true, the corresponding file must exist) |
| `asset_dir` | `bool` | Whether the `blocks.assets` asset directory exists |
| `split_option` | `object` | Chunking parameters used during file extraction. This field is reserved for the extraction engine itself to record and use |
| `blocks` | `int` | Number of content lines (excluding meta) |
| `doc_id` | `"doc-<md5>"` | Global document ID. Sidecar item IDs (`im-/tb-/eq-`) use the hash portion of `doc_id` with the `doc-` prefix removed, in order to shorten the placeholder tags embedded in body text. |
| `parse_engine` | `str` | Parsing engine `native/mineru/docling/legacy` |
| `parse_time` | `str` | Parse completion time; format: ISO-8601 UTC |
| `doc_title` | `str` | Document title (usually the first H1); optional |
| `doc_summary` | `str` | Document summary; optional |
| `doc_attributes` | `object` | Document extended attributes object; optional |
| `bbox_attributes` | `object` | Global bbox position attributes; see [§8](#8-positions) |

> LightRAG requires that filenames (`document_name`) be unique within the same workspace (knowledge base).

### 3.2 content line

Each content line is the minimum addressable unit of an original document "block" and contains at least:

```json
{
  "type": "content",
  "blockid": "462c6364584a7ba4bdae6853f85ac429",
  "format": "plain_text",
  "content": "1 Product Purpose and Functions\nThe MI012 module is used to support the oxygen-supply and anti-gravity control function of the oxygen-supply and anti-gravity regulator...",
  "heading": "1 Product Purpose and Functions",
  "parent_headings": [],
  "level": 1,
  "session_type": "body",
  "table_slice": "none",
  "positions": [
    {
      "type": "paraid",
      "range": ["5EA4577A", "6555DDCB"]
    }
  ]
}
```

| Field | Meaning |
|---|---|
| `type` | `"content"` |
| `blockid` | Globally unique Block ID |
| `format` | Content form, currently fixed to `"plain_text"` |
| `content` | Text content; **equations and images appear as placeholder tags here, tables appear as JSON or HTML wrapped in table tags** (see §3.3). The heading line is rendered with a markdown `#` prefix (plus a space) matching `level`: level 1 → one `#`, level 2 → two `#`, …, capped at 6 (a level ≥ 7 heading still renders `######`). If the source heading text already begins with a markdown prefix (1–6 `#` followed by a space), it is kept verbatim and not prefixed again. Note the `heading` field itself stays clean (no `#`). |
| `heading` | The top-most-level heading of the section containing this content. When `heading` is real, it should also appear at the beginning of `content`. **Every recognized heading starts its own block**: if a heading is immediately followed by body text, that body is merged into the same block (content = heading + body); if a heading has no following body (e.g., it is immediately followed by a heading at the next level), it still becomes a standalone block whose content is just the heading text. This ensures that concatenating the `content` fields of all blocks still reconstructs the complete original text without overlap. |
| `parent_headings` | String array: the top-down list of ancestor headings, excluding the current `heading` |
| `level` | Integer: the level of `heading` in the document outline (`1` = H1 / first-level heading; `0` means no heading) |
| `session_type` | The region the block belongs to: `body` `preface` `TOC` `references` `appendix` |
| `table_slice` | Optional reserved field; indicates whether the block contains only a slice of a table. The current analysis engines do not split long tables, so this field is fixed to `"none"` (meaning the table will not be sliced) |
| `table_header` | Optional reserved field; when the current block is a table slice, this holds the recognized table header. Currently unused. |
| `positions` | Array of `position` objects: identifies the layout position of the text block; when the text block comes from multiple positions in the layout, multiple `position` objects appear. See [§8](#8-positions) |

> - blockid computation: `md5(doc_id + ":" + block_index + ":" + heading + ":" + content)`. Chunks produced by chunking strategies record the blockid for tracing the chunk back to its location in the sidecar.
> - The chunking strategies `F` / `R` / `V` that ignore document section structure operate on the concatenated `content` fields. Therefore, concatenating the `content` fields of all blocks must form the complete document content — no content missing, no content overlapping.

### 3.3 Inline placeholder tags inside content

To let the P chunking strategy split body text without breaking multimodal objects, three XML-style placeholder tags are used inside `content`:

| Tag | Meaning | Tag attributes |
|---|---|---|
| `<table id="tb-…" format="json">…</table>` | Table placeholder; the body is the raw table JSON / HTML | `id` points to the corresponding item in `tables.json`; `format` ∈ `json` / `html` |
| `<drawing id="im-…" format="png" path="…" src="…" caption="…" />` | Self-closing drawing placeholder | `id` points to `drawings.json`; `path` is relative to the `*.parsed/` directory; `src` is the reference name in the original document |
| `<equation id="eq-…" format="latex" caption="…">…</equation>` | Equation placeholder | Inline equations also use `<equation format="latex">`, but **without** `id`, and are not written to the sidecar; only block equations (occupying one or more entire lines) carry an `id` |

When the text is fed to the LLM during entity/relation extraction, internal attributes such as `id / path / src` are stripped, but key attributes (`format / caption`) are preserved. The goal is to avoid extracting entities that are invisible in the article and injecting too much noise into the extraction results.

### 3.4 Correspondence between blockid and chunk sidecar.refs

When a sidecar file exists, the chunking strategies attach `sidecar = {"type": "block", "id": <primary source blockid>, "refs": [{"type": "block", "id": <blockid>}, …]}` to each output chunk, where:

- Unmerged chunk → `sidecar.refs` has only one element, equal to the `blockid` of the blocks.jsonl line the chunk came from;
- Chunk merged in Stage D → `refs` preserves the order of all source `blockid`s (deduplicated);
- Sub-chunks after hard fallback split → share the parent chunk's `sidecar`.

This linkage is the basis for document-level traceability (chunk ↔ block ↔ original paragraph paraId).

## 4. drawings.json

The top level is a dict container of the form `{"version": "1.0", "drawings": { <id>: <item>, … }}`, **keyed by the `id` field** for lookup by id. Each item looks like:

```json
{
  "id": "im-f1bee60173d067d88595c00e7d9b0ce5-0004",
  "blockid": "2f52b70839d13a936d97955916820147",
  "heading": "2.3 Structural Dimensions and Weight",
  "format": "png",
  "path": "m012-manual.blocks.assets/image4.png",
  "src": "",
  "caption": "",
  "footnotes": [],
  "extras": {
    "ocr_texts": "First OCR paragraph inside the image\n\nSecond OCR paragraph inside the image",
    "ocr_texts_count": 2
  },
  "surrounding": {
    "leading": "2.3 Structural Dimensions and Weight\nDimensional and weight requirements are as follows:\na) Outer dimensions length: <drawing …",
    "trailing": "\nFigure 1  Outer dimension schematic\nb) Weight does not exceed 0.85 kg.\nc) Test result: measured circuit noise Vpp=1.526 mV…"
  },
  "llm_analyze_result": {
    "name": "Product outer-dimension engineering drawing",
    "type": "Illustration",
    "description": "This drawing is a schematic of the product's outer dimensions, presenting three views of an electronic device or power module design…",
    "analyze_time": 1778697752,
    "status": "success",
    "message": ""
  },
  "llm_cache_list": [
    "default:analysis:fcf4c4f88227ee1c1bf0ed4394039e37"
  ]
}
```

| Field | Description |
|---|---|
| `id` | Form `im-<doc_hash>-<NNNN>` (`doc_hash` is the 32-character md5 portion of `doc_id` with the `doc-` prefix removed) |
| `blockid` | Points to the content line that produced this drawing |
| `heading` | The section heading the drawing belongs to |
| `format` | Original extension (no dot): `png` / `jpeg` / `gif` / `webp` / `wmf` / `emf` / … |
| `path` | Resource path relative to the `*.parsed/` directory; **always** points to a file inside `*.blocks.assets/` |
| `src` | The reference alias of the drawing in the original document (empty in most cases) |
| `caption` | Visible caption (the parser may leave it empty) |
| `footnotes` | List of footnote strings |
| `surrounding` | Context object: see [§7](#7-surrounding) |
| `self_ref` | String, optional; an object reference from the original parsing engine output (e.g., Docling JSON Pointer `#/pictures/3`, or MinerU `content_list.json#/23`), used to look up the original object in the parsing artifacts (page position, original structure, etc.) when tracing back. Not output by `native` and other engines that do not provide this field. |
| `extras` | Object, optional; engine-specific bypass fields (such as OCR text contained inside the image, etc.). Not part of spec validation; downstream consumers should not rely on specific keys. |
| `llm_analyze_result` | Modal analysis result object: see [§9](#9-llm_analyze_result) (will later be injected into the multimodal text block) |
| `llm_cache_list` | LLM cache list for modal analysis (will later be injected into the multimodal text block) |

Common drawing-specific keys inside `extras`:

| Key | Description |
|---|---|
| `ocr_texts` | String, optional; OCR text inside the drawing object, with multiple paragraphs concatenated by blank lines (`\n\n`). Only written when the parsing engine explicitly attaches OCR text under this drawing's children; caption / footnote do not enter this field. |
| `ocr_texts_count` | Integer, optional; number of non-empty OCR paragraphs written into `ocr_texts`. |

**Only raster formats supported by drawings (png / jpeg / gif / webp) enter VLM analysis**; other formats (wmf / emf / svg, etc.) get `llm_analyze_result.status="skipped"`, no multimodal chunk is generated downstream, and document processing continues. Images larger than the size specified by the environment variable `VLM_MAX_IMAGE_BYTES` likewise will not enter VLM analysis.

> Information such as image size and DPI is uniformly placed in the `extras` object; do not introduce undeclared fields (like `image` / `img_path`, etc.) at the item top level. tables / equations follow the same `extras` convention. `self_ref` is a top-level optional field declared by the spec and does not belong to `extras`.

## 5. tables.json

The top level is a dict container of the form `{"version": "1.0", "tables": { <id>: <item>, ... }}`, **keyed by the `id` field** for lookup by id. Each item looks like:

```json
{
  "id": "tb-f1bee60173d067d88595c00e7d9b0ce5-0007",
  "blockid": "3f33897b5e105d254addc655f1efbf8c",
  "heading": "2.4.4 Temperature-Humidity-Altitude (run with the system)",
  "dimension": [16, 8],
  "format": "json",
  "content": "[[\"Step\", \"Temperature (°C)\", \"Altitude (m)\", \"Relative humidity\", \"Time (min)\", \"Auxiliary cooling\", \"System power\", \"Functional/performance check\"],…",
  "caption": "",
  "footnotes": [],
  "table_header": "[[\"Step\", \"Temperature (°C)\", \"Altitude (m)\", \"Relative humidity\", \"Time (min)\", \"Auxiliary cooling\", \"System power\", \"Functional/performance check\"]]"
  "surrounding": {
    "leading": "2.4.4 Temperature-Humidity-Altitude (run with the system)\nThe product shall withstand the combined temperature, humidity, and altitude environment during mission execution…",
    "trailing": "\nNote: the above steps are repeated for 10 cycles. a) Finished product and accessories reach thermal stability or 240 min, whichever is longer; b) Finished product and accessories reach thermal stability or 120 min, whichever is longer.…"
  },
  "llm_analyze_result": {
    "name": "Document management metadata table",
    "description": "This is a document management information table used to record basic metadata and version control information for a technical document …",
    "analyze_time": 1778697759,
    "status": "success",
    "message": ""
  },
  "llm_cache_list": [
    "default:analysis:b316aacd40fdca0cb56430870bb89a62"
  ]
}
```

The `blockid` / `heading` / `surrounding` / `llm_analyze_result` fields of tables.json have the same meaning as in drawings.json. Different or newly added fields are described below:

| Field | Description |
|---|---|
| `id` | Form `tb-<doc_hash>-<NNNN>` (`doc_hash` is the 32-character md5 portion of `doc_id` with the `doc-` prefix removed) |
| `dimension` | Integer array: `[num_rows, num_cols]`, including header rows |
| `format` | `"json"` (2D array) or `"html"` (payload `<table>…</table>` fragment including the opening and closing tags) |
| `content` | String: the table body, structured according to `format`; this is the string actually used by the downstream multimodal chunk. |
| `table_header` | String, optional; the recognized row(s) treated as the table header |
| `self_ref` | Optional; object reference from the original parsing engine output (e.g., Docling JSON Pointer `#/tables/2`, or MinerU `content_list.json#/31`), used to look up the original artifact when tracing back |

During the modal analysis stage, when the length of the `content` field exceeds the LLM's context window, the table content is mechanically truncated before being fed to the model.

## 6. equations.json

The top level is a dict container of the form `{"version": "1.0", "equations": { <id>: <item>, ... }}`, **keyed by the `id` field** for lookup by id. Each item looks like:

```json
{
  "id": "eq-f1bee60173d067d88595c00e7d9b0ce5-0001",
  "blockid": "2f52b70839d13a936d97955916820147",
  "heading": "2.3 Structural Dimensions and Weight",
  "format": "latex",
  "content": "C=2∗\\frac{P∗T}{\\left( {V}_{H}^{2}−{V}_{L}^{2} \\right)∗η}",
  "caption": "",
  "footnotes": [],
  "surrounding": {
    "leading": "2.3 Structural Dimensions and Weight\nDimensional and weight requirements are as follows:\n …",
    "trailing": "\nwhere P is the power maintained during power abnormalities 28 W, T is the desired energy-storage time, V<sub>H</sub> is before capacitor discharge…"
  },
  "llm_analyze_result": {
    "name": "Capacitor energy-storage time calculation formula",
    "description": "This formula calculates the capacitor energy storage value required to maintain normal system operation during power abnormality …",
    "analyze_time": 1778697783,
    "status": "success",
    "message": "",
    "equation": "C=2\\cdot\\frac{P\\cdot T}{(V_{H}^{2}-V_{L}^{2})\\cdot\\eta}"
  },
  "llm_cache_list": [
    "default:analysis:fcf4c4f88227ee1c1bf0ed4394039e37"
  ]
}
```

The `blockid` / `heading` / `surrounding` / `llm_analyze_result` fields of equations.json have the same meaning as in drawings.json. Different or newly added fields are described below:

| Field | Description |
|---|---|
| `id` | Form `eq-<doc_hash>-<NNNN>` (`doc_hash` is the 32-character md5 portion of `doc_id` with the `doc-` prefix removed) |
| `format` | Fixed to `"latex"` |
| `content` | String: the **raw** LaTeX (possibly containing Unicode operators, outer `\[ \]`); does not include the leading/trailing `$` delimiters; read directly by the modal analysis stage |
| `self_ref` | Optional; object reference from the original parsing engine output (e.g., Docling JSON Pointer `#/texts/15`, or MinerU `content_list.json#/45`), used to look up the original artifact when tracing back |
| `llm_analyze_result.equation` | String: the **canonicalized** LaTeX equation output by the LLM (outer `$ / \[ \] / equation` environment, Unicode converted to LaTeX, no leading/trailing `$` delimiters); this is the string actually used by the downstream multimodal chunk. |

During the modal analysis stage, when the length of the `content` field exceeds the LLM's context window, the content is mechanically truncated before being fed to the model. Inline equations (those continuous with the body, as `<equation format="latex">…</equation>`) **are not** saved to equations.json; they remain only in the blocks text without an `id`. The goal is to avoid injecting too much noise into the extraction results.

## 7. surrounding

`surrounding.leading` and `surrounding.trailing` are the analyzable context windows of a sidecar item; their purpose is to provide contextual information about the paragraph containing the image, table, or equation, improving the quality of multimodal analysis. **The surrounding content is automatically injected by LightRAG during the analysis stage; it does not need to be actively written into the sidecar by the document parsing engine.** The generation logic of the surrounding content is as follows:

- Taken from the text of the content line with the same `blockid`, split at the position of the multimodal placeholder tag;
- The token limit on each side is controlled by the environment variables `SURROUNDING_LEADING_MAX_TOKENS` / `SURROUNDING_TRAILING_MAX_TOKENS` (default `2000`, can be tuned independently); truncated by tokenizer, preferring to retain sentences close to the target;
- The text preserves placeholder tags of **other multimodal objects on the same line**, allowing the model to perceive context such as "after Figure 1 there is also Equation 1"; but internal parser identifiers (`id` / `path` / `src` / `refid`) have been stripped by `strip_internal_multimodal_markup_for_extraction` — consistent with chunk content cleanup before entity extraction, to avoid noise entering the VLM/LLM prompt. Specific cleanup rules:
  - `<drawing id="im-…" path="…" src="…" caption="Fig 1" />` → `<drawing caption="Fig 1" />`; **drawings without a caption are removed entirely** (the tag carries no model-visible information anymore);
  - `<table id="tb-…" format="json" caption="…">rows</table>` → `<table format="json" caption="…">rows</table>`;
  - `<equation id="eq-…" format="latex">body</equation>` → `<equation format="latex">body</equation>`;
  - `<cite type="table" refid="tb-…">Table 1</cite>` → `<cite type="table">Table 1</cite>`; `<cite type="equation" refid="eq-…">Equation 2</cite>` → `<cite type="equation">Equation 2</cite>`. Only the `refid` attribute is removed; the `<cite type="…">…</cite>` wrapper is preserved — letting the VLM/LLM recognize "this is a reference to another table/equation" rather than ordinary text, while hiding the parser-internal id that the LLM cannot see.
    - Exception: surrounding of the `tables.json` type first goes through `remove_table_tags` before stripping, removing all `<cite type="table">` blocks entirely (when analyzing the target table, we don't want to be distracted by dangling references to other tables);
- Cleanup happens **before** token-budget truncation: the token count is computed on "what the LLM actually sees", and truncation does not land inside an uncleaned `id="…"` attribute, avoiding broken tag structure;
- When the target object itself sits at the start / end of the block, the corresponding side is `""` instead of `"n/a"` (when assembling the prompt, the empty string is later displayed as `n/a`);
- `enrich_sidecars_with_surrounding` is idempotent: each `analyze_multimodal` entry point recomputes and overwrites `surrounding`, so after changing `SURROUNDING_LEADING_MAX_TOKENS` / `SURROUNDING_TRAILING_MAX_TOKENS` there is no need to manually clean the sidecar — just re-run multimodal analysis and `surrounding` will be rewritten under the new budget.

## 8. positions

`positions` is an array of objects that identifies which piece of text in the file the `blockid` content comes from, allowing the original content to be located and displayed in the source file during content traceability. When the content of a `blockid` is composed of several columns from the layout, multiple `position` objects appear, with each `position` object corresponding to one layout box or column. To accommodate different document formats' content positioning approaches, the system supports the following types of `position` object.

`position` objects have multiple types, and the `type` field determines its type:

* paraid

Applicable to docx-format files; locates content by `paragraph id` (paraid). The `range` field specifies the start and end `paragraph id`s; `charspan` is an optional field specifying that the content starts at character m and ends at character n of the paragraph. When `charspan` is not provided, the `blockid` covers the entire content of the start and end paragraphs. Example:

```
"positions": [
{
    "type": "paraid",
    "range": ["5EA4577A", "6555DDCB"]
    "charspan": [10,999]
}]
```

* bbox

Applicable to PDF-like files; identifies the original position of the content via a rectangle on the page. bbox supports the following fields:

```
origin: Which position the rectangle coordinates are relative to on the page (optional, defaults to LEFTTOP; another option is LEFTBOTTOM)
max: Maximum length and width of the page layout; coordinates are normalized by this value for accurate position display (optional; empty means coordinates are computed by the image's pixel grid)
anchor: Page number, as a string, supporting non-Arabic page numbers such as Roman numerals
range: Rectangle coordinate array [h1, w1, h2, w2], e.g., [174, 155, 818, 333]
charspan: Content starts at character m and ends at character n of the anchored paragraph (optional)
```

The `bbox_attributes` field of the `meta` line in `blocks.jsonl` holds global bbox settings, avoiding repeating the same content in every `content` line's `positions` object. A typical `positions` object example:

```
"positions": [
{
    "type": "bbox",
    "anchor": "ii"
    "range": [174, 155, 818, 333]
    "charspan": [10, 999]
}]
```

* heading

Applicable to Markdown-like files; locates content by heading. `anchor` is the starting heading (for handling duplicated headings, refer to the Markdown anchor specification); `charspan` is an optional field specifying that the content starts at character m and ends at character n of the paragraph. When `charspan` is not provided, the `blockid` covers the entire content of the start and end paragraphs.

```
"positions": [
{
    "type": "heading",
    "anchor": "ii"
    "range": [174, 155, 818, 333]
    "charspan": [10, 999]
}]
```

* absolute

Applicable to text-like files; locates content by absolute character position. `charspan` specifies that the content starts at character m and ends at character n.

```
"positions": [
{
    "charspan": [10, 999]
}]
```

## 9. `llm_analyze_result`

| `status` | Trigger scenario | Field description |
|---|---|---|
| `success` | The model returns valid JSON and all required fields are present | Drawing: `name / type / description`; Table: `name / description`; Equation: `name / description / equation` |
| `skipped` | Multimodal analysis was deliberately skipped: image format unsupported, pixels < `VLM_MIN_IMAGE_PIXEL` (default 32 px), larger than `VLM_MAX_IMAGE_BYTES` (default 5 MB), or VLM not enabled | `message` records the skip reason |
| `failure` | Required fields missing, JSON still invalid after repair, the VLM/EXTRACT role is not configured while the corresponding modality is enabled, or the model invocation throws an exception | `message` records the diagnostic |

Additional notes:

- `analyze_time` is epoch seconds and is present for every status;
- `message` is **always an empty string** when `status="success"`, making filtering convenient;
- Items for enabled modalities are recomputed on each `analyze_multimodal` run, and the current run overwrites any prior `llm_analyze_result` (`success`, `skipped`, or `failure`). This allows operators to fix VLM/EXTRACT configuration and retry without manually clearing stale sidecar results. LLM calls still use the analysis cache: if the cache key matches, the provider is not called and semantic fields usually remain the same, though runtime fields such as `analyze_time` are rewritten. A cache miss, for example after changing the effective role model/binding/host, prompt inputs, or image metadata, can produce different saved content.

Drawing `type` is constrained to a 12-value enum (see [`IMAGE_TYPE_ENUM`](../lightrag/prompt_multimodal.py): `Photo / Illustration / Screenshot / Icon / Chart / Table / Infographic / Flowchart / Chat Log / Wireframe / Texture / Other`); values returned by the model outside the enum are normalized to `Other` rather than failing.
