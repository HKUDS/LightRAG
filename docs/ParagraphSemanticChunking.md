# Paragraph Semantic Chunking Strategy

## 1. Use Cases and Strategy Selection

### 1.1 What the P Strategy Solves

Paragraph Semantic Chunking (hereafter the **P strategy**) targets documents with clear sectional structure such as DOCX. Its core goal is to **align chunk boundaries with the document's native semantic boundaries** (headings, paragraphs, table rows) as much as possible, rather than determining cut points solely from token-length counting.

The P strategy is mainly designed to address the following four categories of problems:

1. **Table context fragmentation**: When a large table is split, its head and tail slices easily become detached from the preceding description, following explanation, or intermediate bridging text, making them impossible to understand independently during recall.
2. **Insufficient utilization of hierarchical information**: Methods that only look at neighboring paragraphs cannot leverage parent heading paths or relationships between sibling clauses.
3. **Imbalanced sizes of fine-grained sections**: Regulations, standards, contracts, etc., often contain many fine-grained clauses of 100–300 tokens. Without merging, chunks become too short and semantically thin; merging by adjacent length alone causes cross-topic pollution.
4. **Long-chunk re-splitting breaks structure**: When sections are excessively long, ordinary character splitting ignores table row boundaries and heading hierarchy.

The P strategy is effective only for the `.blocks.jsonl` structured artifacts produced by the `native` extraction engine; for unstructured inputs, it automatically falls back to the R strategy (see §8).

### 1.2 Comparison of P / R / V Strategies

| Dimension | R Strategy (Recursive) | V Strategy (SemanticVector) | P Strategy (ParagraphSemantic) |
|---|---|---|---|
| Splitting basis | Cascading character separators (paragraph → newline → Chinese punctuation → whitespace → character) + token budget | Sentence-level embedding distance thresholds (percentile / standard deviation / IQR / gradient) to locate semantic breaks | DOCX outline level with `parent_headings` + table row boundaries + anchors + hierarchy-aware merging |
| Chunk size control | `chunk_token_size` hard cap | `chunk_token_size` is merely an advisory ceiling; when exceeded, secondary splitting via R | `target_max` hard cap + `target_ideal` soft target + table threshold + tail-absorption threshold working in concert |
| Table handling | Table-unaware; may cut in the middle of a table | Table-unaware | Tables smaller than `table_max` are kept intact; large tables are sliced by JSON row array / HTML `<tr>` row boundaries and re-wrapped as valid `<table>` |
| Table context | Relies on incidental window coverage | Relies on embedding distance | First slice glues to preceding description, last slice glues to following explanation; bidirectional overlap of bridging text between consecutive large tables |
| Inter-chunk overlap | Global `chunk_overlap_token_size` | No overlap | No overlap across section boundaries; within the same section, long body falls back to R with overlap by `CHUNK_P_OVERLAP_SIZE`; bridging text between consecutive large tables may enter both the preceding and following table chunks |
| Heading metadata | Usually none | Usually none | Inherits or promotes heading; appends `[part n]` suffix after splitting; preserves `parent_headings` and `level` |
| Embedding compute cost | None | High (must compute embedding per sentence) | None |
| Input requirements | Any text | Any text + Embedding model | Must have a `.blocks.jsonl` sidecar (i.e., result of the `native` engine); otherwise falls back to R |

### 1.3 How to Choose

| Scenario | Recommended | Rationale |
|---|---|---|
| DOCX with clear sectional hierarchy, large tables, fine-grained clauses | **P** | Fully leverages heading hierarchy and table row boundaries; chunk boundaries best match semantics; avoids cross-topic pollution |
| Documents dominated by prose / commentary / long body without clear sectional structure | **V** | Splitting by semantic similarity forms natural boundaries at topic shifts, more stable than character splitting |
| Inputs are plain text, Markdown, code, logs, or you want minimum compute overhead | **R** | No embedding overhead; cascading separators are stable enough for mixed Chinese-English text |
| General configuration (uncertain about file types) | **R** | P automatically falls back to R when no sidecar is present; V also falls back to R when no Embedding model is available |
| Documents with chaotic heading styles and many pseudo-headings in body | **R** or **V** | P depends on the native parser correctly identifying headings; messy headings cause basic chunk boundaries to shift |
| Single-line giant tables or unparsable tables | Any | All three strategies eventually fall back to character-level splitting; P still retains the advantage of table context gluing |

### 1.4 Costs of the P Strategy

- Must be paired with the `native` engine: explicitly declared in `LIGHTRAG_PARSER`, e.g., `docx:native-P`; otherwise, even if `P` is written, it falls back to R due to the missing `.blocks.jsonl`.
- DOCX only: other formats have no `.blocks.jsonl` artifact.
- Many algorithmic paths and thresholds: debugging requires first verifying the input sidecar, then inspecting the outputs of each stage.

## 2. Overview of How It Works

The P strategy takes as input the `.blocks.jsonl` produced by the native parser in `fixlevel=0` mode. **Each `type == "content"` line is treated as one heading-level basic chunk**, then table slicing, long-chunk splitting, and hierarchical merging are performed on top:

```text
DOCX
  ↓  native parser (fixlevel=0)
.blocks.jsonl + sidecars (.tables.json / .equations.json / .drawings.json / .blocks.assets/)
  ↓  TableRowSplit: slice oversized tables along row boundaries and assign first/middle/last roles
  ↓  TableBridge: bidirectional overlap of bridging text between consecutive large tables
  ↓  AnchorSplit: anchor-driven re-splitting of long text chunks
  ↓  PartLabeling: [part n] line-level provenance numbering (per original content line, hence before cross-line merging)
  ↓  HeadingGlue: glue body-less heading blocks forward into their strictly-deeper child
  ↓  LevelMerge: hierarchy-aware two-phase merging
Final chunk list
```

**Key invariants of the P strategy**:

1. **No overlap across section boundaries**: Text between different `.blocks.jsonl` content lines is never copied into the other chunk, avoiding "misattribution".
2. **Long body within a section may overlap**: Multiple slices from within the same content line may keep R-style overlap controlled by `chunk_overlap_token_size`, reducing mid-sentence cuts in long bodies.
3. **Bridging text between tables may overlap bidirectionally**: The only cross-paragraph copying scenario, specifically serving context preservation for consecutive large tables.
4. **Table rows do not overlap each other**: Row-level slicing itself is non-overlapping, different from R's overlap concept.

## 3. Input and Output

### 3.1 Input

`chunking_by_paragraph_semantic()` receives the following inputs:

| Parameter | Source | Description |
|---|---|---|
| `content` | `full_docs[doc_id].content` | Concatenated merged text, used for fallback when sidecar is missing |
| `blocks_path` | `full_docs[doc_id].lightrag_document_path` | Path to `.blocks.jsonl`, the primary input for the P strategy |
| `chunk_token_size` | `chunk_options.chunk_token_size` / `CHUNK_P_SIZE` | Target hard cap N; defaults to `2000` |
| `chunk_overlap_token_size` | `CHUNK_P_OVERLAP_SIZE` / `chunk_overlap_token_size` | Upper bound for long-body fallback overlap within the same content line and for the table bridging budget; defaults to `100` |
| `tokenizer` | The tokenizer already parsed by LightRAG | Basis for all token counting and text overlap truncation |

The P strategy **does not accept** `split_by_character` / `split_by_character_only`, because the normal path is driven by heading and paragraph structure.

### 3.2 `.blocks.jsonl` Convention

The P strategy only processes `type == "content"` lines. Each content line typically contains:

- `content`: The body text under the heading, possibly including ordinary paragraphs, `<table ... />` tags, `<equation ... />` formulas, `<drawing ... />` graphics.
- `heading`: The current heading.
- `parent_headings`: The chain of parent headings.
- `level`: Heading level (1–9, corresponding to the original outline levels 0–8).
- `positions`: Original paragraph positioning (used for traceability).

The native parser's `fixlevel=0` mode guarantees that "the body under a heading becomes one basic chunk" without performing token-threshold splitting during parsing. Tables are inserted into `content` while staying intact.

### 3.3 Output

The final output is an ordered list of chunks, where each element is:

```python
{
    "tokens": int,                    # Actual token count (re-measured after merging)
    "content": str,                   # Chunk text (may contain <table> tags)
    "chunk_order_index": int,         # Chunk ordering index
    "heading": str,                   # Suffix [part n] appended after splitting
    "parent_headings": list[str],     # Parent heading chain; no suffix appended
    "level": int,                     # Heading level
}
```

Internally, the implementation also temporarily uses fields such as `paragraphs`, `table_chunk_role`, `uuid`, `uuid_end`, `type` to assist splitting and merging, but **these do not appear in the final output**.

### 3.4 `[part n]` Suffix Rules

- When the same original `.blocks.jsonl` content line is split into multiple slices, the `heading` field of every slice gets `[part 1]`, `[part 2]` … appended.
- Content lines that are not split keep the original heading unchanged.
- `parent_headings` does not get any suffix.
- Numbering is **reset independently within each original content line**.
- The legacy `[表格片段N]` ("table fragment N") suffix is uniformly replaced by `[part n]`.

## 4. Key Thresholds

P strategy thresholds are not fixed constants; they are dynamically derived from `chunk_token_size` (denoted N):

| Name | Formula | Value when N = 2000 | Technical meaning |
|---|---|---:|---|
| `target_max` | N | 2000 | Hard upper bound for text chunks |
| `target_ideal` | 0.75 × N | 1500 | Ideal target for text chunks; chunks at or above this value stop participating in ordinary peer merging |
| `table_max` | 0.625 × N | 1250 | Threshold that triggers table slicing |
| `table_ideal` | 0.375 × N | 750 | Ideal size for a table slice |
| `table_min_last` | 0.32 × `table_max` | 400 | Last-slice swallow-back threshold (if the last slice is smaller and can be merged, swallow it back into the previous slice) |
| `small_tail_threshold` | 0.125 × N | 250 | Threshold for tail fragment absorption |
| `max_anchor_candidate_length` | Fixed | 100 chars | Upper bound on paragraph length for candidate anchors in long-chunk splitting |

Proportional constraint relationships: `table_max < target_ideal < target_max`, `table_ideal < table_max`. These ratios originate from empirical values in the audit mode (`large chunk 8000, small table 5000, ideal table 3000, table tail 1600`) and are now proportionally scaled by `chunk_token_size`.

## 5. HeadingBlocks: Heading-Level Basic Chunks

Heading recognition is performed by the native parser; **the P chunker itself does not scan the docx body nor judge heading styles**.

In `fixlevel=0` mode, the native parser:

1. Reads `styles.xml`, builds a style inheritance chain via `<w:basedOn>`, and traces back the effective `<w:outlineLvl>`.
2. Iterates over the paragraphs of `document.xml`, resolving outline levels along the inheritance chain; original outline levels 0–8 are mapped to internal `level` 1–9.
3. Maintains `current_heading_stack`, clearing old headings no shallower than the current level when a new heading is encountered, and computing `parent_headings`.
4. Extracts tables, formulas, and drawings into single-line tags (`<table id="..." format="json">...</table>` etc.) and writes them to the corresponding sidecars.
5. All recognizable headings trigger a basic chunk boundary; **no** token-threshold splitting is performed.

The P chunker directly reads `.blocks.jsonl`, treating each content line as an independent unit of processing for subsequent TableRowSplit/AnchorSplit. This implies that `[part n]` numbering is reset independently per original content line.

## 6. TableRowSplit: Row-Boundary Slicing for Oversized Tables

TableRowSplit only processes tables whose token count exceeds `table_max`. Its goal is **not merely to split the table** but to preserve table boundary context based on row-boundary-priority splitting.

### 6.1 Row-Boundary-Priority Slicing

- `format="json"`: Slice by the top-level JSON row array.
- `format="html"`: Slice by `<tr>...</tr>` rows.
- Tables not explicitly tagged but sniffable as JSON / HTML are handled by the same rules.

Before slicing, the `<table {attrs}></table>` wrapper token cost is pre-deducted so that each re-wrapped slice stays under `table_max` as much as possible. Each slice is re-wrapped as a valid `<table>` tag for ease of downstream parsing.

### 6.2 Row-Level Recursive Re-Slicing

If a row subset, after re-wrapping, still exceeds `table_max`, further subdivision is performed within that row subset. **Only when slicing has converged to a single row that itself exceeds the limit does it degrade to character-level splitting**. This mechanism keeps as much valid table structure as possible for table content expressible by row boundaries.

### 6.3 Last-Slice Swallow-Back

If the token count of the last table slice falls below `table_min_last` and the result of merging with the previous slice does not exceed `table_max`, the last slice is swallowed back into the previous slice, reducing useless short table chunks.

### 6.4 Table Slice Roles and Physical Gluing

Each table slice is assigned an internal field `table_chunk_role`, and gluing to surrounding paragraphs is decided by role:

| Role | Meaning | Gluing strategy |
|---|---|---|
| `first` | First slice of the original table | Appended to the tail of the current accumulating chunk so that the table's **preceding description** enters the same chunk as the first slice |
| `middle` | Middle slice of the original table | Output independently to avoid merging with unrelated body |
| `last` | Last slice of the original table | Used as the starting point of a new accumulating chunk so that the **following explanation** is automatically appended after the last slice |
| `none` | Non-table slice or untouched intact table | Treated as ordinary text chunks |

`table_chunk_role` is an internal field that does not survive in the final output, **but in LevelMerge it continues to serve as a merging constraint** (see §9.1).

## 7. TableBridge: Bidirectional Overlap of Bridging Text Between Consecutive Large Tables

When the pattern "large table A, short bridging text, large table B" appears in the same original content line and both tables are split, the bridging text is distributed bidirectionally according to a context budget:

1. Encode the bridging text into tokens.
2. Compute the left budget `prev_budget = min(chunk_overlap_token_size, target_max - current token count of the left last slice)`.
3. Compute the right budget `next_budget = min(chunk_overlap_token_size, target_max - current token count of the right first slice)`.
4. **If the bridging text length does not exceed either budget**: Both the left and right table boundary chunks contain the **complete bridging text**.
5. **If the bridging text is longer**: The prefix enters the left last-slice chunk, the suffix enters the right first-slice chunk; the middle portion that exceeds both budgets becomes an independent ordinary text chunk.

Each one-sided budget is additionally capped at `chunk_token_size / 2` to prevent the bridging text from dominating an entire chunk.

The difference from ordinary adjacent chunk overlap:

- Ordinary overlap copies characters or tokens by forward/backward order, regardless of boundary type.
- The B.1 mechanism is triggered by table slice roles, treating bridging text as both the post-text context of the left table and the pre-text context of the right table, avoiding the bridging description being assigned to only one side or being split off and hard to recall.

## 8. AnchorSplit: Anchor-Driven Re-Splitting of Long Text Chunks

AnchorSplit processes content chunks that still exceed `target_max` after TableRowSplit.

### 8.1 Short-Paragraph Anchors

Restore content into paragraphs, then select paragraphs that satisfy all of the following as candidate anchors:

- The paragraph is not a table (does not start with `<table`).
- The paragraph text length does not exceed `max_anchor_candidate_length` (100 chars).
- The paragraph is not the first paragraph of the chunk (to avoid non-convergent recursion).

### 8.2 Balanced Anchor Selection

Based on the target sub-chunk count, ideal split positions are computed, and the anchor closest to each ideal position is chosen from candidates. The chosen anchor is **promoted to the new `heading` of the following sub-chunk**, while the original heading is written into that sub-chunk's `parent_headings`.

### 8.3 No-Anchor Fallback

If no qualifying anchor exists:

1. **Table first**: If oversized tables still exist within the chunk, prioritize TableRowSplit's row-boundary slicing.
2. **Greedy packing**: Greedily pack the remaining text by paragraph, approaching `target_max`.
3. **Recursive character splitting**: A single excessively long ordinary text paragraph falls back to the R strategy (`chunking_by_recursive_character`), using `chunk_overlap_token_size` to keep continuity between adjacent text slices.

The no-anchor fallback path guarantees the algorithm **does not discard content** and tries to respect the user-configured chunk size cap.

## 8.5 HeadingGlue: Gluing Body-Less Heading Blocks

Some sections carry only a heading and no body of their own (heading-only), for example:

```
## 2.3   Structural dimensions and weight .....   (level 2, has body)
## 2.4   Environmental adaptability metrics       (level 2, heading-only, no body)
### 2.4.1   Overview                               (level 3, has body)
```

If this goes straight into LevelMerge, `## 2.4` becomes an independent same-level small chunk and gets **absorbed backward into the tail of the previous peer chunk `## 2.3`** — either via Phase A peer merging or via batched tail absorption — separating this parent heading from its actual child content `### 2.4.1`.

A pre-pass (`_glue_heading_only_blocks`) is therefore inserted before LevelMerge. A block is heading-only when its `content` consists solely of heading lines (detected via `^#{1,6} +`). Gluing is **forward only**:

- **Trigger**: the heading-only block's immediately following block is **strictly deeper** (greater `level`) and its `table_chunk_role` is `none` or `first`. A `first` slice is the **first slice of a split table** — when a section's body is an oversized table, TableRowSplit's first emitted block for that row has role `first`. The block immediately after a heading-only row can only be the next row's first emitted block, so its role is necessarily `none` or `first` (`middle`/`last` only occur inside the same row's table).
- **Keep the `first` role when gluing into it**: after gluing `## 2.4` into a `first` slice, the merged block **stays `first`** (the `## 2.4` heading is exactly the preceding context a `first` slice is meant to carry). LevelMerge then cannot pull it backward into `## 2.3` (a `first` slice cannot be absorbed backward), so the table-boundary protection survives; the `none` case behaves exactly as before.
- **Action**: glue the heading forward into that child, **keeping the parent heading's identity** (`heading` / `level` / `parent_headings` from the shallower parent). So `## 2.4` bonds with `### 2.4.1`; the heading path stays anchored at `2.4`, and since the child's `parent_headings` already contains 2.4, no hierarchy is lost. A chain of bare ancestor headings (`# 2` → `## 2.4` → `### 2.4.1`) keeps folding down, retaining the **shallowest** identity, until the first child with a body.
- **No backward gluing**: a heading-only block whose next block is **not** deeper (a shallower/sibling heading, or end of list) is left untouched for LevelMerge. It is deliberately **not** pulled backward into a deeper previous block (e.g. into `### 2.3.9`): absorbing a shallower `## 2.4` heading into a deeper L3 chunk would invert the hierarchy (deep-absorbs-shallow) and demote the heading's level. Such an orphan heading is simply left for LevelMerge's normal handling.
- **Hard-cap preserved**: the child came out of AnchorSplit within `target_max`, but prepending the parent heading line(s) can tip the bonded block over the cap. Since nothing downstream re-splits an oversized chunk (LevelMerge only refuses to grow it further), an over-cap bonded block is re-split here: the leading heading lines are **peeled off**, the body is split at the **full `target_max`** (so later body-only chunks keep the full budget), and the heading prefix is glued back onto the first body piece. Only if that first piece is too large to also hold the prefix is it (and it alone) re-split with a reduced cap — so a large prefix never over-fragments the whole child section. This keeps the heading with real content — never sliced off as a heading-only orphan (which LevelMerge would re-absorb backward), and every emitted piece still honours `target_max`. (Degenerate case: when the prefix alone fills the cap — a very long title, or a tiny `chunk_token_size` — it cannot be kept whole, so the whole block is split directly and the oversized heading line is character-split; the cap wins over heading-intactness.)
- **No extra backfill into the previous block**: because `keep="left"` preserves the parent's `level`, a forward-bonded group enters LevelMerge as an ordinary small chunk (not pinned independent). Whether it merges back into the previous chunk `2.3` follows LevelMerge's existing rules entirely — peer merging when `2.3` is still below `target_ideal`, or tail absorption when the group is below `small_tail_threshold` (which can pull it even into an already-saturated `2.3`), both bounded by `target_max` on the re-measured join. This pre-pass only guarantees the heading is never detached FROM its content; letting `2.3 + 2.4 + 2.4.1` share one chunk when sizes allow is the intended anti-fragmentation behaviour.

> Boundary ambiguity: a body line that genuinely begins with `#` + space would be misclassified as a heading line — this is the same accepted heuristic ambiguity documented in `lightrag/parser/_markdown.py`, and is extremely unlikely in real corpora.

## 9. LevelMerge: Hierarchy-Aware Two-Phase Merging

LevelMerge resolves the tension between "chunks too small" and "cross-topic pollution" in fine-grained section scenarios. The core idea is to **process from deeper levels to shallower levels**, first merging small chunks at the same level, then allowing shallow chunks to absorb deep chunks, while introducing size constraints, table slice role constraints, and heading path constraints.

### 9.1 Merging Constraints (every merge must satisfy)

1. **Size constraint**: The actual text token count after merging does not exceed `target_max`; chunks that have reached `target_ideal` in principle do not continue to participate in ordinary peer merging.
2. **Role constraint**: `middle` table slices are locked as independent; `first` and `last` participate in merging directionally to prevent table boundary context from being incorrectly swallowed.
3. **Level constraint**: Peer merging happens between equal `level`; cross-level absorption only allows shallow absorbing deep, **disallowing deep absorbing shallow in reverse**.
4. **Parent heading path consistency constraint**: Adjacent chunks have identical `parent_headings`, or are within a contiguous range constrained by the same parent heading path. This is key to avoiding cross-topic pollution.

### 9.2 Phase A: Peer Merging

For adjacent chunks at the current level, if both are below `target_ideal` and satisfy the above constraints, merge them into one chunk.

Directional rules of table slice roles:

| Chunk role | Can forward-absorb next chunk | Can be absorbed by previous chunk |
|---|:-:|:-:|
| `none` | Yes | Yes |
| `first` | Yes | No |
| `middle` | No | No |
| `last` | No | Yes |

### 9.3 Batched Tail Absorption

If a chunk that has reached `target_ideal` is followed by a string of peer small chunks, and the total token count of that string is below `small_tail_threshold` and the actual merged token count does not exceed `target_max`, then **absorb that string in one shot**. Stop when encountering a `middle` table slice.

### 9.4 Phase B: Cross-Level Absorption

For small chunks still unsaturated after Phase A, attempt cross-level merging, but only allow shallow absorbing deep:

- When the current chunk is shallower than the next, the current chunk may forward-absorb the next.
- When the current chunk is deeper than the previous, the previous shallower chunk may absorb the current.
- Reverse merging is forbidden.
- In the cross-level phase, the `last` role is allowed to forward-absorb; `middle` still does not participate in merging.

### 9.5 Post-Merge Actual Token Re-Measurement

Because merging inserts newline connectors, chunk-by-chunk token summation may underestimate the merged result. **Before committing each merge, the actual concatenated text must be re-tokenized**, and the merge is committed only after confirming it does not exceed `target_max`.

After merging, the main chunk's `heading` is retained. If multiple part slices are merged, the final heading keeps the part suffix of the main chunk, **never** additionally concatenating multiple part tags.

## 10. Fallback and Degradation Paths

The P strategy has multiple layers of fallback protection:

| Trigger | Degradation behavior |
|---|---|
| `blocks_path` missing, unreadable, or no valid content line | Fall back entirely to `chunking_by_recursive_character()`, passing in the parsed `chunk_overlap_token_size` |
| TableRowSplit cannot identify the JSON / HTML structure of a table | That table uses the R strategy's character splitting |
| TableRowSplit finds a single-row table itself exceeding `table_max` | That single row uses the R strategy's character splitting |
| AnchorSplit finds a long chunk with no qualifying short-paragraph anchor | Table first → greedy packing → fall back to R character splitting if a single paragraph is too long |

**Important**: After the overall fallback, capabilities such as heading hierarchy, table roles, and bidirectional bridging-text overlap are no longer available; however, it still ensures the document produces retrieval chunks and is not silently dropped due to a missing structured sidecar.

## 11. Configuration

| Configuration | Default | Description |
|---|---|---|
| `CHUNK_P_SIZE` | `2000` (when unset, uses `DEFAULT_CHUNK_P_SIZE`; does **not** fall back to `CHUNK_SIZE`) | P-specific `chunk_token_size`; paragraph semantic merging requires a higher cap than the global default, hence an independent default rather than falling back to `CHUNK_SIZE` |
| `CHUNK_P_OVERLAP_SIZE` | Unset (falls back to `CHUNK_OVERLAP_SIZE`) | P-specific overlap; only affects long-body fallback within the same content line and the table bridging budget. **Does not** cause table row-level slices to overlap |
| `CHUNK_OVERLAP_SIZE` / `LightRAG(chunk_overlap_token_size=…)` | `100` | Global fallback when no P-specific overlap is set |

For configuration syntax, the priority chain, and runtime overrides via `addon_params["chunker"]`, see [FileProcessingConfiguration-zh.md](FileProcessingConfiguration-zh.md) §3.

A typical `LIGHTRAG_PARSER` setup that enables P:

```bash
LIGHTRAG_PARSER=docx:native-P,*:legacy-R
CHUNK_P_SIZE=2000
CHUNK_P_OVERLAP_SIZE=100
```

Or override per single file:

```text
my-proposal.[native-P].docx
```

## 12. Validating Chunking Results

### 12.1 Check Whether the Sidecar Was Generated

Confirm whether the native parser successfully produced `.blocks.jsonl`:

```bash
ls -l INPUT/__parsed__/<doc>.docx.parsed/<doc>.blocks.jsonl
```

If the file is missing or empty, the P strategy falls back to R entirely and gains none of P's benefits. Common causes:

- `LIGHTRAG_PARSER=docx:native-...` was not configured.
- Parsing failed (see error entries in `pipeline_status`).
- The document is not a DOCX (other formats do not support P).

### 12.2 Inspect the Contents of blocks.jsonl

Each line is a JSON; filter `type == "content"` and inspect whether heading / level / parent_headings match expectations:

```bash
jq -c 'select(.type=="content") | {level, heading, parent_headings}' \
   INPUT/__parsed__/<doc>.docx.parsed/<doc>.blocks.jsonl | head
```

If most headings are empty or levels are abnormal, the native parser did not correctly recognize heading styles — in which case P's hierarchical merging and anchor promotion will both fail.

### 12.3 Inspect the Final Chunks

View chunk metadata in the `text_chunks` storage:

```bash
jq '.[] | {heading, level, tokens, parent_headings}' \
   rag_storage/kv_store_text_chunks.json | head -30
```

You should observe:

- Headings of chunks around large tables typically correspond to `[part 1]` / `[part n]` (indicating TableRowSplit splitting occurred).
- Fine-grained clauses are merged into chunks close to `target_ideal` (indicating LevelMerge took effect).
- `parent_headings` jumps at boundaries between different sections and stays stable within the same section.

### 12.4 Chunk Size Distribution Check

Ideal distribution: most chunks fall in the range `[target_ideal, target_max]` (i.e., approximately 1500–2000 tokens when N=2000); chunks noticeably smaller are usually `middle` table slices (locked as independent) or tail chunks at section boundaries.

If many tail chunks below `small_tail_threshold` appear, possible causes include:

- The parent heading path consistency constraint is too strict (adjacent small chunks with different `parent_headings` cannot merge).
- Many `middle` table slices pile up (the table itself is very large).

## 13. Troubleshooting

### 13.1 P Did Not Take Effect; Output Matches R

Investigate in this order:

1. Does `full_docs[doc_id].process_options` contain `P`?
2. Is `full_docs[doc_id].parse_format` equal to `lightrag`? If `raw`, it is on the legacy path and P automatically falls back to R.
3. Does the `.blocks.jsonl` pointed to by `lightrag_document_path` exist and is it non-empty?
4. Are there `paragraph_semantic ... fallback to recursive_character` messages in the logs?

### 13.2 Tables Are Scattered; Preceding and Following Explanations Are Detached

- Check whether the table is truly recognized as `<table format="json">` or `<table format="html">` (see `.blocks.jsonl`). Tables with unrecognized format can only undergo character splitting and cannot trigger TableRowSplit's role mechanism.
- Check whether the table's token count actually exceeds `table_max`. Tables below the threshold remain intact and never trigger first/middle/last slicing.
- For consecutive large tables, confirm whether the bridging text between the two tables resides in the **same content line** — bridging across content lines does not participate in B.1 bidirectional overlap.

### 13.3 Fine-Grained Clauses Are Not Merged

- Check whether the `parent_headings` of adjacent clauses are identical: the parent heading path consistency constraint prevents cross-topic merging.
- Check whether `level` is the same: peer merging requires equal `level`; cross-level absorption only allows shallow absorbing deep.
- Check whether a `middle` table slice is inserted in the middle: this blocks batched tail absorption.

### 13.4 A Single Chunk Exceeds `target_max`

Normally, LevelMerge's actual token re-measurement rejects oversized merges, but oversized chunks may still occur in the following scenarios:

- A single-row table itself exceeds `target_max` with no anchor to split on; eventually it goes through R character splitting but a single chunk still exceeds the limit.
- `enforce_chunk_token_limit_before_embedding` performs a final hard cut before embedding; downstream will not actually embed an oversized chunk into the vector store.

### 13.5 Abnormal `[part n]` Suffixes

- Multiple slices come from the same original content line, but only one `[part 1]` is seen: check whether they were merged in LevelMerge — after merging, the main chunk's part suffix is retained and multiple part tags are not concatenated.
- Legacy `[表格片段N]` suffix appears: this indicates data output by an older chunker; the new version standardizes on `[part n]`, and re-chunking is required.

### 13.6 Log Keywords

P-strategy-related log keywords (for `grep`-based troubleshooting):

- `paragraph_semantic` — module entry
- `fallback to recursive_character` — overall or single-paragraph degradation
- `table_chunk_role` — table role-related
- `bridge` — TableBridge bridging text handling
- `anchor` — AnchorSplit anchor selection
