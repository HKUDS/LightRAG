## RAG-Anything Parser Alignment Notes

This document summarizes the companion changes made on the `RAG-Anything` side to better align its parser output with the LightRAG multimodal pipeline introduced in this PR.

These notes are provided as reviewer context. The code changes described below live in the `RAG-Anything` repository, mainly in `raganything/parser.py`, rather than in this LightRAG pull request.

## Why This Alignment Was Needed

The LightRAG-side pipeline in this PR expects parser output to preserve heading structure, normalize multimodal block types consistently, and expose enough table metadata to generate correct LightRAG document sidecars.

Without the parser-side alignment, several downstream issues appear more easily:

- section headings may be lost before LightRAG sidecar generation
- table dimensions can degrade to `[0, 0]`
- table content may be harder to serialize into stable sidecar payloads
- parser output shape may drift between Docling variants

## RAG-Anything Changes

### 1. Add safe helper functions for parser normalization

Two small helpers were added:

- `_to_int(value, default=0)`
- `_grid_to_rows(grid)`

Their purpose is to make parser output more defensive and consistent when Docling returns numeric fields or table cell structures in slightly different formats.

### 2. Normalize text labels before branching

Docling text blocks are now normalized through:

- `label = str(block.get("label", "")).strip().lower()`

This avoids relying on a raw case-sensitive label and makes formula / title / section-header detection more stable.

### 3. Preserve section heading structure explicitly

For Docling text blocks, `section_header` and `title` are now emitted as dedicated structured blocks:

- `type: "section_header"` or `type: "title"`
- `text`
- `level`
- `page_idx`

This is important because the LightRAG-side conversion logic uses heading information to:

- propagate `heading`
- build `parent_headings`
- keep multimodal sidecars attached to the correct section context

### 4. Preserve label and level on normal text blocks

For non-heading text blocks, the parser now also retains:

- `label`
- `level`

This gives LightRAG more context when converting parser output into LightRAG document blocks and helps preserve document structure more faithfully.

### 5. Improve table normalization for Docling output

Table parsing was expanded to support both:

- dict-style table payloads with `grid`, `num_rows`, `num_cols`
- legacy list-style table payloads

The parser now derives and exposes:

- `table_body`
- `rows`
- `num_rows`
- `num_cols`

This is the key alignment needed for LightRAG-side table sidecar generation, especially to avoid empty dimensions and to keep table content serializable in a stable form.

## Practical Impact on This PR

These RAG-Anything parser changes are the external counterpart of the LightRAG work in this PR:

- LightRAG now converts structured parser output into LightRAG document artifacts
- multimodal sidecars depend on parser-side heading and table metadata
- heading propagation and table dimension fixes are more reliable when the parser emits normalized structure upstream

In short, the LightRAG code in this PR can run independently, but the best end-to-end behavior for Docling/RAG-Anything-driven multimodal ingestion depends on this parser alignment on the `RAG-Anything` side as well.

## Scope Note

This document is intentionally limited to parser-alignment notes for `RAG-Anything`.

It does not describe the entity disambiguation experiment, which is explicitly excluded from this PR.
