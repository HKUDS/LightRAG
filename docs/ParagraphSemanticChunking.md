# Paragraph Semantic Chunking Strategy

## 1. Use Cases and Strategy Selection

### 1.1 What the P Strategy Solves

Paragraph Semantic Chunking (hereafter the **P strategy**) targets documents with a clear sectional structure such as DOCX and PDF. Its core goal: **align chunk boundaries with the document's native semantic boundaries** (headings, paragraphs, table rows) as much as possible, rather than deciding split points purely by token-length counting.

The P strategy mainly addresses these four problems:

1. **Table context fracture**: after a large table is split, the head/tail slices easily detach from their leading explanation, trailing commentary, or intervening bridge text, becoming impossible to understand on their own at recall time.
2. **Underused hierarchy information**: methods that look only at adjacent paragraphs cannot exploit the parent-heading path or the relationships between same-level clauses.
3. **Imbalanced fine-grained section sizes**: regulations, standards, and contracts often contain many 100–300 token fine-grained clauses; leaving them unmerged yields chunks that are too short and semantically thin, while merging purely by adjacent length causes cross-topic pollution.
4. **Long-block re-splitting breaks structure**: when a section is too long, ordinary character splitting ignores table-row boundaries and heading levels.

The P strategy is valid for the structured output of **any parser that can produce a `.blocks.jsonl` sidecar** (`native` / `mineru` / `docling`) — all three persist an identically-structured `.blocks.jsonl` (carrying `heading` / `level` / `parent_headings`) through the shared `write_sidecar()`. Only the `legacy` engine produces no sidecar; input without a sidecar (the legacy path or a parse failure) automatically degrades to the R strategy (see §6).

### 1.2 Comparison of the P / R / V Strategies

| Dimension | R Strategy (Recursive) | V Strategy (SemanticVector) | P Strategy (ParagraphSemantic) |
|---|---|---|---|
| Split basis | Cascaded character separators (paragraph → newline → Chinese punctuation → space → character) + token budget | Sentence-level embedding-distance thresholds (percentile / standard deviation / interquartile range / gradient) to find semantic gaps | Heading outline level and `parent_headings` + table-row boundaries + anchors + hierarchy-aware merging |
| Chunk-size control | `chunk_token_size` hard cap | `chunk_token_size` is only an advisory ceiling; over-limit chunks are re-split via R | `target_max` hard cap + `target_ideal` soft target + table thresholds + tail-absorption threshold acting in concert |
| Table handling | Table-unaware; may cut in the middle of a table | Table-unaware | Tables under `table_max` stay whole; large tables are sliced along JSON row arrays / HTML `<tr>` row boundaries and re-wrapped as legal `<table>` |
| Table context | Relies on a window happening to cover it | Relies on embedding distance | First slice glues the leading explanation, last slice glues the trailing commentary, bridge text between consecutive large tables overlaps bidirectionally |
| Inter-chunk overlap | Global `chunk_overlap_token_size` | No overlap occurs | Section boundaries never overlap; long body text within one section that falls back to R overlaps by `CHUNK_P_OVERLAP_SIZE`; bridge text between consecutive large tables can enter both the preceding and following table chunks |
| heading metadata | Usually none | Usually none | Inherits or promotes heading; appends a `[part n]` suffix after splitting; preserves `parent_headings` and `level` |
| Embedding compute cost | None | High (must embed every sentence) | None |
| Required input | Any text | Any text + an embedding model | Must have a `.blocks.jsonl` sidecar (produced by any of `native` / `mineru` / `docling`), otherwise degrades to R |

### 1.3 How to Choose

| Scenario | Recommended | Reason |
|---|---|---|
| Clear section hierarchy (the content-parsing engine must be able to generate a sidecar file) | **P** | Fully exploits heading levels and table-row boundaries; chunk boundaries hug semantics most closely; avoids cross-topic pollution |
| Document is mostly prose / commentary / long-form body with no clear sectional structure | **V** | Splitting by semantic similarity forms natural boundaries at topic-shift points, more stable than character splitting |
| Input is plain text, Markdown, code, or logs, or you want the lowest compute cost | **R** | No embedding cost; cascaded separators are robust enough for mixed Chinese/English text |
| General configuration (file type uncertain) | **R** | P auto-degrades to R when there is no sidecar; V auto-degrades to R when there is no embedding model |
| Documents with messy heading styles or many pseudo-headings in the body | **R** or **V** | P relies on the parser correctly identifying headings; messy headings shift the basic-block boundaries |
| Single-row huge tables or unparseable tables | Any | All three strategies ultimately fall back to character level; P still keeps its table-context-gluing advantage |

## 2. Design Goals and Core Invariants

Every rule of the P strategy serves one goal: **align chunk boundaries with the document's native semantic boundaries, and make each chunk understandable on its own at recall time**. It decomposes this goal into concrete rules for three scenarios (tables, long blocks, fine-grained sections), expanded one by one in §3. No matter how the rules combine, the following four **overlap invariants** always hold — they delimit "where text duplication is allowed, and where it is never allowed":

1. **Section boundaries never overlap**: text between different `.blocks.jsonl` content lines is never copied into each other's chunks, avoiding mis-attribution.
2. **Long body text within a section may overlap**: multiple fragments split from one content line may keep R-style overlap by `chunk_overlap_token_size`, reducing mid-body cuts.
3. **Bridge text between tables may overlap bidirectionally**: the only cross-paragraph duplication scenario, dedicated to preserving context for consecutive large tables.
4. **Table rows never overlap each other**: row-level slicing is itself non-overlapping, distinct from R's overlap concept.

### 2.1 Rule-to-Effect Overview

The table below maps each rule of §3 to the effect it achieves and the internal stage that implements it (stage names double as the cross-reference identifiers in code comments, log keywords, and debugging — see §7.6):

| Chunking rule | Effect achieved | Implementing stage | See |
|---|---|---|---|
| Heading-level basic chunks | Chunk boundaries align with the document's native structure, not token counts | HeadingBlocks | §3.1 |
| Table integrity + row-boundary slicing | Tables are not cut mid-cell; slices remain legal `<table>` | TableRowSplit | §3.2 |
| Table context gluing (roles + bidirectional bridge overlap) | A table's leading explanation, trailing commentary, and bridge text never detach from the table | TableRowSplit / TableBridge | §3.3 |
| Header recovery (re-attach the header to middle/last slices at split time) | A split table's `middle`/`last` slices keep their column names when recalled alone, without ever exceeding the cap | HeaderRecovery | §3.3.3 |
| Anchor-driven long-block re-splitting | Over-long sections are split at semantic points, preserving heading levels | AnchorSplit | §3.4 |
| Body-less heading gluing | A parent heading is never separated from its child content | HeadingGlue | §3.5 |
| Hierarchy-aware merging | Fine-grained clauses are gathered toward the ideal size without cross-topic pollution | LevelMerge | §3.6 |
| Overlap rules | Sufficient recall context, yet section/table boundaries are never mis-attributed | Throughout | §3.7 |
| Size-threshold coordination | Most chunks land in `[target_ideal, target_max]` | Throughout | §3.8 |

### 2.2 Processing Pipeline Overview

The rules above chain into a pipeline that takes `.blocks.jsonl` as input (`fixlevel=0` mode, **each `type == "content"` line is treated as one heading-level basic block**):

```text
DOCX / PDF / PPTX / …
  ↓  native(docx, fixlevel=0) / mineru / docling parser —— emit basic blocks by heading, no token splitting
.blocks.jsonl + sidecar (.tables.json / .equations.json / .drawings.json / .blocks.assets/)
  ↓  TableRowSplit: slice oversized tables along row boundaries and assign first/middle/last roles   → §3.2
  ↓  HeaderRecovery: budget + re-inject the repeating header into middle/last slices during the split → §3.3.3
  ↓  TableBridge: bidirectional overlap of bridge text between consecutive large tables              → §3.3
  ↓  AnchorSplit: anchor-driven re-splitting of long text chunks                                     → §3.4
  ↓  PartLabeling: [part n] line-level provenance numbering (numbered per original content line, hence before cross-line merging)
  ↓  HeadingGlue: glue body-less heading blocks forward into their strictly-deeper child             → §3.5
  ↓  LevelMerge: hierarchy-aware two-phase merging                                                   → §3.6
Final chunk list
```

## 3. Chunking Rules and Effects

### 3.1 Heading-Level Basic Chunks — Aligning Boundaries with Native Semantics 〔HeadingBlocks〕

**Rule**: each `type == "content"` line of `.blocks.jsonl` is a basic block, i.e. "the body under one heading as one block". Heading identification is performed entirely by the **parser**; **the P chunker itself never scans the document body or judges heading styles**, and it does no token-threshold splitting at parse time.

**Effect**: a chunk's initial boundaries fall naturally on the document outline structure (at heading transitions) rather than at arbitrary token positions; every later stage works on top of this semantically aligned basis.

The three sidecar-producing engines all carve out basic blocks by heading, each obtaining `heading` / `level` / `parent_headings`:

- **native (docx, `fixlevel=0`)**: reads `styles.xml`, builds the style-inheritance chain via `<w:basedOn>` to recover the effective `<w:outlineLvl>`; walks the `document.xml` paragraphs resolving the outline level along the chain, mapping original outline levels 0–8 to internal `level` 1–9; maintains a `current_heading_stack`, clearing old headings no shallower than the current level and computing `parent_headings` on each new heading.
- **mineru**: detects headings by an item's `text_level > 0` or `label` being `title` / `section_header`, using a heading_stack to maintain the parent chain.
- **docling**: `label="title"` → level 1, `label="section_header"` → `item.level + 1` (default level 2), likewise maintaining the parent chain.

All three ultimately produce a unified `IRBlock` (carrying `heading` / `level` / `parent_headings`), persisted by `write_sidecar()` into an identically-structured `.blocks.jsonl`; tables, equations, and drawings are extracted as single-line tags (`<table id="..." format="json">...</table>` etc.) written to the corresponding sidecar. Every recognizable heading triggers a basic-block boundary, with **no** token-threshold splitting.

The P chunker reads `.blocks.jsonl` directly, treating each content line as an independent processing unit for the subsequent TableRowSplit/AnchorSplit — which also means `[part n]` numbering is **reset independently** per original content line (see §3.4 and §4.4).

### 3.2 Table Integrity and Row-Boundary Slicing — Never Cut a Table Mid-Cell 〔TableRowSplit〕

**Rule**: a table whose token count does not exceed `table_max` **stays whole**; only a table exceeding `table_max` is sliced, and it is **sliced along row boundaries first** — the whole table degrades to character-level splitting only when a slice has collapsed to a single row that still cannot be expressed within the limit.

**Effect**: a table is never cut in the "middle of a cell"; every slice is re-wrapped as a legal `<table>` tag, so downstream parsing and LLM reading can interpret it as a table rather than as broken markup fragments.

#### 3.2.1 Row-Boundary-First Slicing

- `format="json"`: slice along the top-level JSON row array.
- `format="html"`: slice along `<tr>...</tr>` rows.
- Tables not explicitly tagged but whose content can be sniffed as JSON / HTML are handled by the same rules.

Before slicing, the `<table {attrs}></table>` wrapper token overhead is debited so that re-wrapped slices stay within `table_max` as much as possible. Each slice is re-wrapped as a legal `<table>` tag for easy downstream parsing.

#### 3.2.2 Row-Level Recursive Re-Slicing

If a row subset still exceeds `table_max` after re-wrapping, it is subdivided further within that row subset. **When a slice has converged to a single row that cannot be kept both `≤ target_max` and header-complete (the row's content itself exceeds the cap, or it fits but leaves no room for the header it would need), the whole table degrades to an R recursive character split of the original `<table>` text (whose body still carries the header), and a `logger.warning` is logged** — the header content survives as plain text along with the original table text and is never silently dropped, nor is a "some `<table>` slices + some orphaned character fragments" mixed output produced. A slice that needs no injected header and whose single row fits `target_max` is still kept whole as legal `<table>` markup. This mechanism keeps table content expressible by row boundaries in legal table structure as much as possible.

#### 3.2.3 Last-Slice Swallow-Back

If a table's last slice has a token count below `table_min_last` and merging it with the previous slice does not exceed `table_max`, the last slice is swallowed back into the previous slice, reducing useless short table chunks.

### 3.3 Table Context Gluing — Leading/Trailing Explanations, Bridges, and Headers Stay Attached 〔TableRowSplit / TableBridge / HeaderRecovery〕

**Rule**: a sliced table glues to surrounding paragraphs differently by "first/middle/last" role; short bridge text between two consecutive large tables is distributed **bidirectionally** to the table chunks on both sides by budget; middle/last slices that lose the header row get the table's repeating header re-injected into their own `<table>` **during the split** (the header's tokens are budgeted out of each slice's cap before splitting).

**Effect**: a table's **leading explanation** enters the first-slice chunk, its **trailing commentary** enters the last-slice chunk, and **bridge text** serves as both the left table's following context and the right table's preceding context — any table slice carries enough context to be understood on its own at recall, with no "table here, explanation in another chunk" fracture. A split table's middle/last slices, even though detached from the first slice that carries the header, get the header row re-injected back at the top of their own `<table>`, so they remain interpretable per-column when recalled alone.

#### 3.3.1 Table Slice Roles and Physical Gluing

Each table slice is given an internal field `table_chunk_role`, and its role determines how it glues to surrounding paragraphs:

| Role | Meaning | Gluing strategy |
|---|---|---|
| `first` | The first slice of the original table | Appended to the tail of the current accumulation block, so the table's **leading explanation** enters the same chunk as the first slice |
| `middle` | A middle slice of the original table | Emitted standalone, avoiding merger with unrelated body text |
| `last` | The last slice of the original table | Starts a fresh accumulation block, so the **trailing commentary** is automatically appended after the last slice |
| `none` | A non-table slice or an unsplit whole table | Handled as an ordinary text chunk |

`table_chunk_role` is an internal field that does not survive into the final output, **but it continues to serve as a merging constraint in LevelMerge** (see §3.6.1).

#### 3.3.2 Bidirectional Bridge-Text Overlap Between Consecutive Large Tables 〔TableBridge〕

When the pattern "large table A, short bridge text, large table B" occurs within the same original content line and both tables are split, the bridge text is distributed bidirectionally by context budget:

1. Encode the bridge text into tokens.
2. Compute the left budget `prev_budget = min(chunk_overlap_token_size, target_max - current token count of the left last slice)`.
3. Compute the right budget `next_budget = min(chunk_overlap_token_size, target_max - current token count of the right first slice)`.
4. **If the bridge text fits within both side budgets**: both the left and right table boundary chunks contain the **complete bridge text**.
5. **If the bridge text is longer**: the prefix enters the left last-slice chunk, the suffix enters the right first-slice chunk; the middle segment exceeding both budgets becomes a standalone ordinary text chunk. This middle chunk **keeps `chunk_overlap_token_size` of R-style overlap with each side**: extending left to re-include the tail of the prefix that went into the left table chunk, and right to include the head of the suffix that went into the right table chunk. Because each side's prefix/suffix is itself ≤ the overlap budget, the overlap span covers the entire prefix and suffix, so **the middle chunk in effect carries the complete bridge text** (the bridge is therefore never fragmented; only its head/tail are **additionally** copied into the neighbouring table chunks). The overlap indices always stay within the bridge tokens, so **`<table>` content is never copied into the middle chunk**.

A single side's budget is further capped at no more than `chunk_token_size / 2`, so bridge text can never dominate the whole chunk.

How this differs from ordinary adjacent chunk overlap:

- Ordinary overlap copies characters or tokens in sequence, regardless of boundary type.
- The TableBridge mechanism is triggered by table-slice roles, making the bridge text serve simultaneously as the left table's following context and the right table's preceding context, so a bridging explanation is not attributed to only one side's table nor scattered into a separate chunk that is hard to recall.

#### 3.3.3 Header Recovery for Middle/Last Slices 〔HeaderRecovery〕

After a large table is sliced along row boundaries, the header row stays only in the **first slice**; `middle` / `last` slices thus lose the column names and cannot tell each column's meaning when recalled on their own. To fix this, **during TableRowSplit** the header row is re-injected into the non-first slices' own `<table>`, so every slice becomes a complete header-bearing table.

1. **Header source**: at parse time each table's "cross-page repeating header" is written into the sibling `.tables.json` (entry field `table_header`, a JSON 2-D array string; **only tables that genuinely carry a repeating header have this field**). P traces back to the matching table entry via the `id` preserved on the to-be-split `<table>` tag and takes its `table_header`.
2. **Budgeted reserve, injected at split time**: the header's token cost is reserved out of each slice's body cap **before** splitting (alongside the `<table {attrs}></table>` wrapper overhead). `_split_table_text` splits against that reduced budget, then prepends the header into each non-first slice — a `format="json"` slice prepends the header rows to its row array, a `format="html"` slice emits them as a leading `<thead>` (`<th>` cells, HTML-escaped). The slice keeps its original `attrs` (including the leading `id`). Because the room was reserved, **a slice plus its header still stays `≤ target_max`**, so the hard cap is enforced naturally by every downstream stage — there is no late backfill that can overflow the cap. The first slice keeps its own real header row and is not injected again. If a slice has converged to a single row that can no longer hold both the row content and the header within `target_max` (see §3.2.2), **the whole table degrades to an R recursive character split (header included) and a warning is logged** — never leaving an orphaned header-less slice.
3. **Never fabricate a header** — none of the following are injected: the source table has no `table_header` field in `.tables.json` (no repeating header), `.tables.json` is missing/unreadable, the slice has degraded to a character-level non-`<table>` fragment (no `id` to trace), or the table was not actually split into multiple pieces.

> Because the header enters the slice at split time, a split table's slices are **completely frozen against LevelMerge — never re-merged with each other** (see §3.6.1); otherwise re-merging two slices of one table would duplicate the header mid-body. The recovered header **enters `content`** and counts toward the chunk's token total (headers are typically tiny); it is **not** stored on `heading`.

### 3.4 Anchor-Driven Long-Block Re-Splitting — Cut at Semantic Points, Keep Headings 〔AnchorSplit〕

**Rule**: for content blocks that still exceed `target_max` after TableRowSplit, split in a balanced way at "short-paragraph anchors" first, promoting the chosen anchor to the new heading of the sub-block; when no qualifying anchor exists, fall back through a three-tier "table first → greedy packing → character splitting".

**Effect**: an over-long section is not hard-cut at an arbitrary token position but cut at **natural semantic points** like short subheadings/transition sentences, with sub-blocks inheriting a readable heading and parent-heading path; meanwhile the algorithm **never drops content** and respects the user-configured chunk-size cap as much as possible.

#### 3.4.1 Short-Paragraph Anchors

Recover the content into paragraphs and choose paragraphs satisfying all of the following as candidate anchors:

- The paragraph is not a table (does not start with `<table`).
- The paragraph text length does not exceed `max_anchor_candidate_length` (100 characters).
- The paragraph is not the block's first paragraph (so recursion can converge).

#### 3.4.2 Balanced Anchor Selection

Compute the ideal split positions from the target number of sub-blocks, and from the candidate anchors choose the one nearest the ideal position. The chosen anchor is **promoted to the new `heading`** of the following sub-block, and the original heading is written into that sub-block's `parent_headings`.

#### 3.4.3 No-Anchor Fallback

If no qualifying anchor exists:

1. **Table first**: if an over-limit table still exists within the block, invoke TableRowSplit's row-boundary slicing first.
2. **Greedy packing**: pack the remaining text by paragraph greedily up to near `target_max`.
3. **Recursive character splitting**: a single over-long ordinary text paragraph degrades to the R strategy (`chunking_by_recursive_character`), using `chunk_overlap_token_size` to keep adjacent text fragments continuous.

The no-anchor fallback path guarantees the algorithm **does not discard content** and respects the user-configured chunk-size cap as much as possible.

### 3.5 Body-Less Heading Gluing — A Parent Heading Never Separates From Its Child 〔HeadingGlue〕

**Rule**: when a block is heading-only (only a heading, no body of its own) and the immediately following block is **strictly deeper**, glue it **forward** into that deeper child block while preserving the shallower **parent-heading** identity; all other cases are left as-is for LevelMerge.

**Effect**: a parent heading like `## 2.4` (no body) is never sliced off as a lone chunk and then absorbed backward by LevelMerge into the previous peer chunk `## 2.3`, becoming separated from its actual child content `### 2.4.1` — the heading always travels with its child content, with no loss of heading-path levels.

Some sections have only a heading and no body of their own (heading-only), e.g.:

```
## 2.3   Structural dimensions and weight .....   (level 2, has body)
## 2.4   Environmental adaptability metrics        (level 2, heading-only, no body)
### 2.4.1   Overview                               (level 3, has body)
```

If this went straight into LevelMerge, `## 2.4` would become an independent same-level small block and, via Phase A peer merging or batched tail absorption, be **absorbed backward into the tail of the previous peer block `## 2.3`**, separating this parent heading from its actual child content `### 2.4.1`.

A pre-pass (`_glue_heading_only_blocks`) is therefore inserted before LevelMerge. When the current block is heading-only (`content` consists solely of heading lines, detected by `^#{1,6} +`), it **glues forward only**:

- **Trigger**: the immediately following block is **strictly deeper** (greater `level`) and its `table_chunk_role` is `none` or `first`. A `first` slice is "the first slice of a split large table" — when a subsection's body is an oversized table, TableRowSplit's first emitted block has role `first`; the block right after a heading-only line can only be the next line's first emitted block, so its role must be `none` or `first` (`middle`/`last` only occur inside the same line's table).
- **Keep the `first` role when gluing into a `first` slice**: after gluing `## 2.4` into a `first` slice, the merged block **stays `first`** (the `## 2.4` heading is exactly the preceding context a `first` slice should carry). LevelMerge then will not absorb it backward into `## 2.3` (a `first` slice cannot be absorbed backward), preserving the table-boundary protection; the `none` sub-block behaves exactly as before.
- **Action**: glue forward into that sub-block, preserving the **parent-heading** identity (`heading` / `level` / `parent_headings` taken from the shallower parent block). That is, `## 2.4` and `### 2.4.1` are bonded into one block, the heading path still centered on `2.4` — sub-block 2.4.1's `parent_headings` already contains 2.4, so hierarchy info is lossless. A chained heading (`# 2` → `## 2.4` → `### 2.4.1`) collapses along the chain, keeping the **shallowest** identity, until the first sub-block with body content is reached.
- **No backward gluing**: when the next block is **not** deeper (a shallower/sibling heading, or end of list), the heading-only block is left as-is for LevelMerge. It is **not** glued backward into a deeper previous block (e.g. `### 2.3.9`) — absorbing the shallower `## 2.4` heading into a deeper L3 block would invert the hierarchy (deep-absorbs-shallow) and demote the heading's level. Such an orphan heading is handed directly to LevelMerge's normal handling.
- **Hard cap preserved**: the sub-block came out of AnchorSplit within `target_max`, but prepending the parent heading line(s) can tip it over the cap. Since nothing downstream re-splits an over-limit block (LevelMerge only prevents it from growing further), an over-cap bonded block is re-split here: **first peel off the leading heading line(s)**, split the body at the **full `target_max`** (so later prefix-free body pieces keep the full budget), then glue the heading prefix back onto the **first body piece**. Only when the first body piece is too large to also hold the prefix is it alone re-split with a reduced cap — so a large prefix does not over-fragment the whole subsection. This way the heading always travels with real body content and is never sliced off as a heading-only orphan (which LevelMerge would otherwise absorb backward), and every emitted piece is still ≤ `target_max`. (Degenerate case: when the prefix alone fills the cap — a very long title, or a tiny `chunk_token_size` — it cannot be kept whole, so the whole block is split directly and the oversized heading line is character-split; here the cap wins over heading integrity.)
- **No extra backfill into the previous block**: because `keep="left"` preserves the parent's `level`, the bonded whole is just an ordinary small block (not pinned as independent). Whether it merges back into the previous block `2.3` follows LevelMerge's existing rules entirely — peer merging when `2.3` is still < `target_ideal`, or tail absorption when the whole is below `small_tail_threshold` (which can pull it even into an already-saturated previous block), both bounded by the re-measured real token ≤ `target_max`. This pre-pass only guarantees the heading is **never detached from its child content**; it does not lock the whole as an independent block — so letting `2.3 + 2.4 + 2.4.1` share one chunk when size allows is exactly the intended anti-fragmentation behaviour.

> Boundary ambiguity: a body line that genuinely begins with `#␠` would be misjudged as a heading line — this is the same heuristic ambiguity already documented and accepted in `lightrag/parser/_markdown.py`, with very low probability in real corpora.

### 3.6 Hierarchy-Aware Merging — Gather Fine-Grained Clauses to the Ideal Size Without Cross-Topic Pollution 〔LevelMerge〕

**Rule**: **process from deeper levels to shallower levels** — first merge same-level small blocks (Phase A), then batch-absorb the tail, finally allow shallow blocks to absorb deep blocks (Phase B); every merge must simultaneously satisfy four constraints: size, table role, level, and parent-heading path.

**Effect**: many 100–300 token fine-grained clauses are merged toward near `target_ideal` (chunks are no longer too short and semantically thin), while **never lumping together adjacent small blocks that belong to different topics / different parent sections** — curing both "chunks too small" and "cross-topic pollution".

#### 3.6.1 Merging Constraints (every merge must satisfy)

1. **Size constraint**: the merged real text token count does not exceed `target_max`; a block that has reached `target_ideal` in principle no longer participates in ordinary same-level merging.
2. **Role constraint (slice freeze)**: every split-table slice — `first` / `middle` / `last` — is **locked standalone and never participates in any merge** (it neither absorbs forward, nor is absorbed backward, nor joins batched tail absorption). Reason: the repeating header is injected into each slice at TableRowSplit time, so re-merging two slices of one table would duplicate the header mid-body (§3.3.3). A table's boundary explanations were already glued into the first/last slices during the split, so the freeze does not lose context gluing — it only gives up the post-hoc consolidation of small first/last blocks with unrelated neighbours. Only `none` (an ordinary block / an unsplit whole table) may merge.
3. **Level constraint**: same-level merging happens between equal `level`s; cross-level absorption allows only shallow-absorbs-deep, **forbidding deep from absorbing shallow in reverse**.
4. **Parent-heading-path consistency constraint**: the key to avoiding cross-topic pollution, with strict semantics by merge direction —
   - **Same-level merging (Phase A / tail absorption)**: the two blocks' `parent_headings` must be **exactly equal** (true siblings). Blocks with the same `level` but different parent chains (e.g. `2.4.1` and `2.5.1`) may not merge.
   - **Cross-level absorption (Phase B, shallow-absorbs-deep)**: the deep block must be a **descendant** of the shallow one — the shallow block's full heading path (`parent_headings` + its own `heading`, with any `[part n]` stripped) must be a prefix of the deep block's `parent_headings`. A shallow block absorbing a deep block from a different branch is forbidden.
   - Blocks with empty `parent_headings` (preamble / non-hierarchical input) are treated as path-compatible and allowed (no hierarchy to pollute).

#### 3.6.2 Phase A: Peer Merging

For adjacent blocks at the current level, when **the current block** is below `target_ideal`, the merged real token count is ≤ `target_max`, and the constraints above are satisfied, merge them into one block (the absorbed neighbour need not be below `target_ideal`; a backward merge additionally requires the previous block to be < `target_ideal`).

Directional rules by table-slice role (all split-table slices are frozen; only `none` may merge):

| Block role | Can absorb the next block forward | Can be absorbed by the previous block |
|---|:-:|:-:|
| `none` | Yes | Yes |
| `first` | No | No |
| `middle` | No | No |
| `last` | No | No |

#### 3.6.3 Batched Tail Absorption

If an **ordinary (`none`)** block that has reached `target_ideal` is immediately followed by a run of same-level small blocks whose total token count is below `small_tail_threshold` and whose merged real token count does not exceed `target_max`, then **absorb that run in one shot**. Stop on encountering **any split-table slice** (`first` / `middle` / `last`), or when the parent-heading path diverges; a split-table slice never initiates tail absorption either.

#### 3.6.4 Phase B: Cross-Level Absorption

For small blocks still unsaturated after Phase A, attempt cross-level merging, but allow only shallow-absorbs-deep:

- When the current block is shallower than the next block, the current block may absorb the next block forward.
- When the current block is deeper than the previous block, the previous shallower block may absorb the current block.
- Merging in the reverse direction is forbidden.
- Split-table slices (`first` / `middle` / `last`) are likewise frozen in the cross-level stage and do not participate; only `none` blocks take part in cross-level absorption.

#### 3.6.5 Post-Merge Real-Token Re-Measurement

Because merging inserts a newline joiner, summing per-block token counts may underestimate the merged result. **Before committing every merge, recompute the token count on the joined real text** and confirm it does not exceed `target_max` before committing.

After merging, the main block's `heading` is kept. If multiple part fragments are merged, the final heading keeps the main block's part suffix and does **not** additionally concatenate multiple part tags.

### 3.7 Overlap-Rule Summary — Where It Overlaps, Where It Never Does

**Rule + effect**: the P strategy draws a precise boundary on "text duplication (overlap)", ensuring sufficient recall context while ruling out cross-section/cross-table mis-attribution. The overlap behaviours scattered across stages are gathered here:

| Scenario | Overlaps? | Budget / mechanism | Effect served |
|---|---|---|---|
| Different `.blocks.jsonl` content lines (section boundaries) | **Never overlaps** | —— | Clear section boundaries, no mis-attribution |
| Long body text within one content line falling back to R | May overlap | `chunk_overlap_token_size` | Keeps semantic continuity at a mid-body cut |
| Bridge text between consecutive large tables | Bidirectional overlap | `min(overlap, …, target_max/2)` per side | Bridge explanation serves as context for both the left and right tables |
| Standalone middle chunk of a long bridge | Overlaps each side | `chunk_overlap_token_size` (kept within bridge tokens, never includes `<table>`) | The middle reads continuously with the neighbouring table chunks |
| Between table row-level slices | **Never overlaps** | —— | Row slices are non-overlapping, avoiding duplicate rows |

### 3.8 Size-Threshold Coordination — Most Chunks Land in [ideal, max]

**Rule**: the P strategy's thresholds are not fixed constants but derived dynamically from `chunk_token_size` (denoted N); multiple thresholds act in concert to control the size of text chunks and table slices.

**Effect**: under the ideal distribution, most chunks land in the `[target_ideal, target_max]` interval (about 1500–2000 tokens when N=2000); noticeably small chunks are usually just standalone-locked `middle` table slices or section-boundary tail blocks.

| Name | Formula | Value at N = 2000 | Technical meaning |
|---|---|---:|---|
| `target_max` | N | 2000 | Text-chunk hard cap |
| `target_ideal` | 0.75 × N | 1500 | Text-chunk ideal target; once reached, stops participating in ordinary same-level merging |
| `table_max` | 0.625 × N | 1250 | Table slicing trigger threshold |
| `table_ideal` | 0.375 × N | 750 | Table slice ideal size |
| `table_min_last` | 0.32 × `table_max` | 400 | Table last-slice swallow-back threshold (below this and mergeable → swallowed back into the previous slice) |
| `small_tail_threshold` | 0.125 × N | 250 | Tail-fragment absorption threshold |
| `max_anchor_candidate_length` | Fixed | 100 chars | Upper bound on candidate anchor-paragraph length for long-block splitting |

Proportional constraints: `table_max < target_ideal < target_max`, `table_ideal < table_max`. These ratios come from audit-mode empirical values (`large block 8000, small table 5000, ideal table 3000, table tail block 1600`) and are now scaled proportionally by `chunk_token_size`.

## 4. Input and Output

### 4.1 Input

`chunking_by_paragraph_semantic()` accepts the following inputs:

| Parameter | Source | Description |
|---|---|---|
| `content` | `full_docs[doc_id].content` | The concatenated merged text, used for degradation when the sidecar is missing |
| `blocks_path` | `full_docs[doc_id].lightrag_document_path` | The `.blocks.jsonl` path, the P strategy's main input |
| `.tables.json` (implicit) | Derived from `blocks_path` (`<base>.blocks.jsonl` → `<base>.tables.json`) | The header source for HeaderRecovery (§3.3.3); silently skipped when missing |
| `chunk_token_size` | `chunk_options.chunk_token_size` / `CHUNK_P_SIZE` | The target hard cap N, default `2000` |
| `chunk_overlap_token_size` | `CHUNK_P_OVERLAP_SIZE` / `chunk_overlap_token_size` | The cap on long-body fallback within one content line and on the table bridge budget, default `100` |
| `tokenizer` | The tokenizer already resolved by LightRAG | The basis for all token counting and text-overlap extraction |

The P strategy **does not accept** `split_by_character` / `split_by_character_only`, because the normal path is driven by heading and paragraph structure.

### 4.2 `.blocks.jsonl` Convention

The P strategy only processes `type == "content"` lines. Each content line typically contains:

- `content`: the body text under that heading, possibly containing ordinary paragraphs, `<table ... />` tags, `<equation ... />` formulas, `<drawing ... />` graphics.
- `heading`: the current heading.
- `parent_headings`: the parent-heading chain.
- `level`: the heading level (1–9, corresponding to original outline levels 0–8).
- `positions`: the original paragraph positions (for traceability).
- `blockid`: a stable identifier of this content line (optional). When present, it is carried into the final chunk's `sidecar` field, letting the multimodal pipeline and document deletion trace back by source block; when absent (raw / legacy input), the output contains no `sidecar`.

The parser guarantees "the body under one heading as one basic block" (native via `fixlevel=0` mode, mineru / docling via their respective IR builders), with no token-threshold splitting at parse time. Tables stay whole, inserted into `content`.

### 4.3 Output

The final output is an ordered chunk list, each element:

```python
{
    "tokens": int,                    # Real token count (re-measured after merging)
    "content": str,                   # Chunk text (may contain <table> tags)
    "chunk_order_index": int,         # Chunk order index
    "heading": {                      # Heading metadata (nested dict, not flat fields)
        "level": int,                 # Heading level
        "heading": str,               # Gets a [part n] suffix after splitting
        "parent_headings": list[str], # Parent-heading chain, no suffix appended
    },
    # Optional: present only when the input .blocks.jsonl line carries a blockid,
    # for the multimodal pipeline and document deletion to trace back by source block.
    "sidecar": {
        "type": "block",
        "id": str,                    # Main-block blockid (refs[0])
        "refs": [{"type": "block", "id": str}, ...],  # All source blockids, deduplicated
    },
}
```

Note: `level` and `parent_headings` are now folded into the nested `heading` dict and are no longer provided at the top level; the `[part n]` suffix lands on `heading["heading"]`. A middle/last table slice's recovered header does not enter `heading`; it is prepended back into the slice's own `<table>` inside the chunk `content` (§3.3.3).

Internally the implementation also uses temporary fields like `paragraphs`, `content`, `table_chunk_role`, `blockids` to aid splitting and merging, but they do **not** enter the final output under those names (`blockids` is materialized as `sidecar` after conversion).

### 4.4 `[part n]` Suffix Rules

- When one original `.blocks.jsonl` content line is split into multiple fragments, every fragment's `heading` field gets `[part 1]`, `[part 2]`, …
- A content line that was not split keeps its original heading.
- `parent_headings` gets no suffix.
- The numbering is **reset independently** within each original content line (because PartLabeling numbers before cross-line merging, see §2.2).
- The legacy `[表格片段N]` suffix has been unified under `[part n]`.

## 5. Configuration

| Config | Default | Description |
|---|---|---|
| `CHUNK_P_SIZE` | `2000` (uses `DEFAULT_CHUNK_P_SIZE` when unset; does **not** inherit `CHUNK_SIZE`) | P-specific `chunk_token_size`; paragraph-semantic merging needs a larger cap than the global default, hence an independent default rather than falling back to `CHUNK_SIZE` |
| `CHUNK_P_OVERLAP_SIZE` | Unset (inherits `CHUNK_OVERLAP_SIZE`) | P-specific overlap; affects only long-body fallback within one content line and the table bridge budget, and does **not** make table row-level slices overlap each other |
| `CHUNK_OVERLAP_SIZE` / `LightRAG(chunk_overlap_token_size=…)` | `100` | Global fallback when no P-specific overlap is set |

For config syntax, the precedence chain, runtime overrides via `addon_params["chunker"]`, etc., see [FileProcessingConfiguration-zh.md](FileProcessingConfiguration-zh.md) §3.

`P` is a chunking option orthogonal to the engine (`suffix:engine-options`) and can combine with any sidecar-producing engine. A typical `LIGHTRAG_PARSER` setup enabling P:

```bash
# docx uses native, pdf uses mineru, other supported formats use docling, all with P; unsupported formats fall back to legacy-R
LIGHTRAG_PARSER=docx:native-teP,pdf:mineru-iteP,*:docling-iteP,*:legacy-R
CHUNK_P_SIZE=2000
CHUNK_P_OVERLAP_SIZE=100
```

(The option flags `i`/`t`/`e` mean image/table/formula analysis respectively, and `P` is the chunking strategy, combinable as needed.) Or override per file:

```text
my-proposal.[native-P].docx
paper.[mineru-P].pdf
```

## 6. Fallback Protection — Never Drop Content

**Rule + effect**: the P strategy has multi-layer fallback protection; whenever a structural capability fails it retreats to character-level splitting, **guaranteeing the document still produces retrieval chunks and is not silently dropped because a structured sidecar is missing**.

| Trigger | Degradation behaviour |
|---|---|
| `blocks_path` missing, unreadable, or with no valid content lines | Degrade wholesale to `chunking_by_recursive_character()`, passing the resolved `chunk_overlap_token_size` |
| TableRowSplit cannot identify a table's JSON / HTML structure | That table uses R-strategy character splitting |
| TableRowSplit finds a single row that cannot be kept within `target_max` alongside its header (the row content exceeds the cap, or it fits but the header would push it over) | **The whole table (header included) degrades to R-strategy character splitting and a `logger.warning` is logged**; the header content survives as plain text along with the original table text |
| AnchorSplit finds a long block with no qualifying short-paragraph anchor | Table first → greedy packing → degrade to R character splitting if a single paragraph is too long |
| HeaderRecovery finds `.tables.json` missing/unreadable, or the source table has no `table_header` | Skip header injection (that table has no repeating header to begin with; does not affect the rest of chunking) |

**Important**: after a wholesale fallback there is no longer heading hierarchy, table roles, or bidirectional bridge-text overlap; but it guarantees the document still produces retrieval chunks.

## 7. Validating Effects and Debugging

### 7.1 Check Whether the Sidecar Was Generated

Confirm the parser successfully produced `.blocks.jsonl`:

```bash
ls -l INPUT/__parsed__/<doc>.<ext>.parsed/<doc>.blocks.jsonl
```

If the file is missing or empty, the P strategy degrades wholesale to R and gains none of P's benefits. Common causes:

- No sidecar-producing engine was configured for that format (e.g. `LIGHTRAG_PARSER=docx:native-...` / `pdf:mineru-...` / `*:docling-...`), so it actually took the `legacy` path.
- Parse failure (check the `pipeline_status` error entries).
- The format is not supported by the chosen engine (e.g. native supports only docx; switch to mineru / docling to cover more formats).

### 7.2 Check the blocks.jsonl Content

One JSON per line; after filtering `type == "content"`, inspect whether heading / level / parent_headings match expectations:

```bash
jq -c 'select(.type=="content") | {level, heading, parent_headings}' \
   INPUT/__parsed__/<doc>.<ext>.parsed/<doc>.blocks.jsonl | head
```

If heading is mostly empty or level is abnormal, the parser did not correctly identify headings — in which case the P strategy's hierarchy merging and anchor promotion both fail.

### 7.3 Check Whether the Final Chunks Achieve the Expected Effects

Inspect the chunk metadata in the `text_chunks` store:

```bash
jq '.[] | {heading, level, tokens, parent_headings}' \
   rag_storage/kv_store_text_chunks.json | head -30
```

You should observe the following signs of "rules taking effect":

- The heading of chunks around a large table usually corresponds to `[part 1]` / `[part n]` (§3.2 table slicing happened).
- Fine-grained clauses are merged into chunks near `target_ideal` (§3.6 hierarchy merging took effect).
- `parent_headings` jumps at section transitions and stays stable within a section (§3.1 / §3.6 parent-path constraint).
- Most chunks land in the `[target_ideal, target_max]` interval (§3.8); noticeably small chunks are usually `middle` table slices (locked standalone) or tail blocks right at a section boundary.

If many tail blocks below `small_tail_threshold` appear, it may be:

- The parent-heading-path consistency constraint being too strict (adjacent small blocks with different `parent_headings` cannot merge, §3.6.1).
- A pile-up of `middle` table slices (the table itself is very large).

### 7.4 Common Troubleshooting

#### 7.4.1 P Did Not Take Effect; Output Matches R

Check in this order:

1. Does `full_docs[doc_id].process_options` include `P`?
2. Is `full_docs[doc_id].parse_format` equal to `lightrag`? If it is `raw`, the legacy path was taken and P auto-degrades to R.
3. Does the `.blocks.jsonl` pointed to by `lightrag_document_path` exist and is it non-empty?
4. Are there `paragraph_semantic ... fallback to recursive_character` lines in the log?

#### 7.4.2 Table Scattered, Leading/Trailing Explanation Separated (§3.2 / §3.3 not in effect)

- Check whether the table was actually identified as `<table format="json">` or `<table format="html">` (look at `.blocks.jsonl`). A table of unrecognized format can only go through character splitting and cannot start TableRowSplit's role mechanism.
- Check whether the table's token count actually exceeds `table_max`. A table below the threshold stays whole and does not trigger first/middle/last slicing.
- For consecutive large tables, confirm the bridge text between them is within the **same content line** — a bridge across content lines does not participate in TableBridge bidirectional overlap.

#### 7.4.3 Fine-Grained Clauses Not Merged (§3.6 not in effect)

- Check whether adjacent clauses have consistent `parent_headings`: the parent-heading-path consistency constraint blocks cross-topic merging.
- Check whether `level` is consistent: same-level merging requires equal `level`, and cross-level absorption allows only shallow-absorbs-deep.
- Check whether a `middle` table slice is inserted in between: it blocks batched tail absorption.

#### 7.4.4 A Single Chunk Exceeding `target_max` Appears

Normally LevelMerge's real-token re-measurement rejects over-limit merges, but over-limit chunks can still appear in these scenarios:

- A single-row table itself exceeds `target_max`, with no anchor to split on, ultimately going through R character splitting but a single chunk still exceeds the limit.
- `enforce_chunk_token_limit_before_embedding` does a final hard split before embedding, so downstream never actually embeds an over-limit chunk into the vector store.

#### 7.4.5 `[part n]` Suffix Anomalies (§3.4 / §4.4)

- One original content line was split into multiple pieces but only one `[part 1]` is seen: check whether they were merged in LevelMerge — after merging the main block's part suffix is kept and not concatenated.
- A legacy `[表格片段N]` suffix appears: this means data output by an old chunker version; the new version unifies on `[part n]`, so re-chunk.

### 7.5 Log Keywords

P-strategy-related log keywords (for `grep` troubleshooting):

- `paragraph_semantic` — module entry
- `fallback to recursive_character` — wholesale or single-paragraph degradation
- `table_chunk_role` — table-role related (§3.3)
- `bridge` — TableBridge bridge-text handling (§3.3.2)
- `table_header` / `tables.json` — HeaderRecovery header recovery (§3.3.3)
- `anchor` — AnchorSplit anchor selection (§3.4)

### 7.6 Stage Name ↔ Rule Mapping

The following **stage names** are used as cross-reference identifiers in code comments, docstrings, logs, and tests. The "Former name" column gives the old letter scheme (which may still appear in historical commits / issues / PR discussions):

| Stage name | Former name | Corresponding rule | Section |
|---|---|---|---|
| `HeadingBlocks` | Stage A | Heading-level basic chunks | §3.1 |
| `TableRowSplit` | Stage B | Table integrity and row-boundary slicing | §3.2 |
| `HeaderRecovery` | Stage B.2 | Re-attach the header to middle/last slices during the split | §3.3.3 |
| `TableBridge` | Stage B.1 | Bidirectional bridge-text overlap between consecutive large tables | §3.3.2 |
| `AnchorSplit` | Stage C | Anchor-driven long-block re-splitting | §3.4 |
| `PartLabeling` | Stage C.1 | `[part n]` line-level provenance numbering | §4.4 |
| `HeadingGlue` | Stage D pre-pass | Body-less heading gluing | §3.5 |
| `LevelMerge` | Stage D | Hierarchy-aware two-phase merging | §3.6 |
