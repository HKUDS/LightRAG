# LaTeX Escape Repair Contract

Read this before touching `_WS_LATEX_RESIDUES`, `_WS_LATEX_SUSPECT_PATTERN`,
`_WS_LATEX_MATH_PATTERN`, `_repair_ws_latex_in_dollar_math` or
`repair_vlm_json_escape_damage` in `lightrag/utils.py`.

An LLM writing LaTeX inside a JSON string routinely emits a single backslash.
`"\frac"` is *valid* JSON meaning form feed + `rac`, so every JSON parser —
`json_repair` included — silently decodes it and the command is destroyed
before any code of ours sees the value. This document is the reasoning behind
the repair; the docstrings in the module state the rules and point here.

## Damage model

Only the five decodable escape letters are at risk. An invalid escape such as
`\alpha` is preserved verbatim by `json_repair`, so it never needs repair.

| Emitted | Decodes to | Repaired |
| --- | --- | --- |
| `\f` + letter | form feed + letter | always |
| `\b` + letter | backspace + letter | always |
| `\t` / `\r` / `\n` + residue | tab / CR / LF + residue | only inside math |

Form feed and backspace followed by a letter have no legitimate use in
LLM-generated prose, so restoring the backslash is unconditional. Tab, CR and
LF are ordinary whitespace, so a residue after one is only *evidence*, and the
strength of that evidence is what the rest of this document is about.

Isolated control characters — not followed by a letter — are left for
downstream sanitization to drop rather than guessed at.

## Residue whitelist and its two boundaries

A residue is the tail of a LaTeX command whose remainder collides with no
English word: `au`, `heta`, `imes`, `ext`, `ilde`, `herefore`, `riangle` after
a tab; `ho`, `ight`, `angle`, `ceil` after a CR; `abla`, `otin` after a LF.
`eq`, `o` and `exists` are deliberately absent — "eq." abbreviations and the
words "o"/"exists" would false-positive.

The same whitelist is compiled twice, and the difference is the trailing
guard:

- **`_WS_LATEX_SUSPECT_PATTERN`** (`(?:\b|(?=[^\x00-\x7F]))`) is the prose
  detector, used only to warn. The word boundary keeps `col<tab>ext_id` and
  `<tab>au2` — plausible values in tab-separated data — out of the log. The
  second alternative exists because `\b` alone reported *nothing* when a word
  character followed, and Python's `re` counts a CJK ideograph as a word
  character: `阈值<tab>au为0.5`, the shape Chinese corpora actually produce,
  was neither repaired (no math span) nor warned about. A Latin fragment
  pressed against a non-ASCII character with no space between them is
  essentially only produced by this damage, so the false-positive risk is low
  and the ceiling on being wrong is one WARNING line — this path never
  rewrites text. ASCII word characters stay excluded deliberately.
- **`_WS_LATEX_MATH_PATTERN`** (`(?![A-Za-z])`) is used inside a confirmed
  math span. There, `_`, `{`, `^` and digits are the most common characters to
  follow a command, and the "the residue might be an English word" argument
  does not apply. Requiring `\b` there left `$<tab>au_i$` neither repaired nor
  warned about — silent, because the prose pattern refuses it too.

The in-math pattern is a strict superset of the prose pattern, which is why a
repaired span can never re-trigger the prose warning afterwards.

Accepted gap: the prose detector is still silent when an **ASCII** word
character follows the residue — `col<tab>ext_id` and `<tab>au2` are damage as
often as they are data, and in tab-separated values the data reading is common
enough that the warning would cost more than it buys. Such a residue is
neither repaired (no math span) nor warned about. This is a detector-side
decision, tracked separately from the pairing rules below.

## Delimiter policy

Dollar delimiters are ambiguous by construction. Prose contains stray and
currency dollars, and even Pandoc reads `$5 ... $x$` as one span, so **no
delimiter rule is correct in both directions**. Pairing is therefore only the
*necessary* half of the decision — the sufficient half is the content gate in
the next section. The scanner is deliberately biased:

> Never rewrite text a Pandoc-style parser would not call math, and accept the
> misses that follow.

Rewriting prose is corruption that reaches storage silently. Skipping a repair
leaves the damage that was already there and, in most shapes, still logs the
prose warning. The two are not symmetric, and the bias follows that asymmetry.

The rules, all applied left to right:

0. Markdown code is verbatim and is never scanned: a fenced block (closed, or
   running to the end of the text when the model never closed it) and a closed
   inline code span are copied through, and no span may cross one. Code quotes
   dollars for its own reasons — `echo "$HOME" ... "$PATH"` pairs exactly as
   neatly as a formula — and Pandoc does not parse math inside code either. An
   unclosed single backtick protects nothing, which is also CommonMark's
   reading and keeps one stray backtick from suppressing the rest of the text.
1. A backslash-escaped `$` is not a delimiter.
2. `$$` opens a display span, closed by the next unescaped `$$`. Display math
   is routinely formatted across lines, so no flanking rule applies to it.
3. A single `$` opens only when the next character is not whitespace —
   Pandoc's rule — **with one exception: the damage itself**. JSON decoding
   puts the tab/CR/LF immediately after the opener, so a residue match at that
   position is evidence of a command, not of whitespace. Without the exception
   the scanner would reject `$<tab>au$`, the very shape it exists to repair.
4. Only the *very next* unescaped delimiter may close, and for an inline span
   it must itself be a valid closer: no whitespace before it, no digit after
   it. A span that has to reach over another dollar is not one span; it is a
   stray dollar plus a real span. The digit clause is what keeps two currency
   amounts (`$5 - $10`) from pairing.
5. A delimiter that cannot pair is emitted as an ordinary character and the
   scan continues — a stray dollar must not cost every later formula its
   repair. A failed `$$` is skipped **whole**: letting its second dollar open
   an inline span pairs it with the next single `$` and rewrites the prose in
   between (`Unclosed $$x and<tab>ext$ suffix`).

## Content gate

Pairing says *where* a span is; it cannot say whether the span is math. Code
pairs dollars for reasons of its own — `echo "$HOME" ... "$PATH"` satisfies
every delimiter rule — and so the span's own body is tested before anything
inside it is rewritten:

- a body containing a **double quote or a backtick** is code. Both are
  ordinary in shell and effectively absent from LaTeX math.
- an **inline** span whose body crosses a line break, or runs past
  `_MAX_INLINE_MATH_CHARS` (200), is not a formula. Display math is exempt
  from both: `$$...$$` is routinely long and multi-line.

  The line-break test is stated by **character class and exemption**, never by
  example, because both halves have been wrong once. A line break is any `\r`
  or `\n` — looking only at `\n` let CR-separated code through while its LF
  twin was refused. A break the residue pattern matches is **not** a break: a
  decoded `\nabla` *is* a newline followed by `abla`, a decoded `\rho` *is* a
  carriage return followed by `ho`, so a blanket veto rejects exactly the
  damage the gate exists to let through. Both halves cover the whole residue
  whitelist and every line ending; a fix phrased for one of them is the bug.

A semicolon is deliberately **not** a marker: `$p(x; \theta)$` is ordinary
notation in this corpus, and vetoing it would cost more than it saves.

The region pass is **heuristic and secondary**, and completeness with respect
to CommonMark is explicitly **not** a goal. Where it reads a construct wrongly
it over-protects, and over-protection costs a repair, never a rewrite -- the
same direction as every other trade here. Fix a deviation when it is cheap and
its direction is clear (a closing fence's trailing text, a closer longer than
its opener, CRLF line endings, a backtick in a backtick fence's info string
have each been fixed on those grounds); do not grow it toward a Markdown
parser. The gate is what has to be right.

Two of those fixes went the *other* direction — they were cases where the
region pass under-protected, which is the direction that costs a rewrite:

- **Block structure before inlines.** Fences are matched in their own pass,
  before inline spans, because CommonMark settles block structure first and an
  inline span can never cross a fence boundary. A single alternation cannot
  express that: its inline branch wins by *position*, not by branch order, so
  a stray backtick anywhere earlier in the text paired with one inside a
  fenced block, swallowed the opening fence, and handed the rest of that
  block's code to the scanner. A **blank line** is the same boundary by the
  same argument — a code span is an inline inside one leaf block — and cost
  the same thing: a stray backtick one paragraph earlier ate a real span's
  opener, leaving that span's closer unmatched and its code exposed. Inline
  spans are therefore searched one paragraph at a time. The rule is a blank
  line, not any line break: CommonMark lets a code span cross an ordinary one,
  and that is what keeps wrapped shell inside a real span protected. The blank
  line must tolerate `\r`, or CRLF text keeps the exposure.

  Other block boundaries — adjacent list items, an ATX heading with no blank
  line under it, a block quote — are *not* handled, and enumerating them is
  the blacklist this document rejects everywhere else. Fences and blank lines
  are kept because generated Markdown produces them at scale; the rest is left
  to the gate.
- **Opener escape parity.** ``\` `` is a literal backtick and opens nothing;
  ``\\` `` is a literal backslash followed by a real opener, so the rule is
  parity over the backslash run, not a one-character lookbehind. Treating an
  escaped backtick as an opener closed the region on the *next* run — a real
  code span's opener — which left that span's own closer as an unclosed single
  backtick, protecting nothing.

This is the general defense, and the reason the Markdown code regions above
are a *secondary* one. Identifying the constructs that contain code — fenced,
indented, inline, block-quoted, HTML — is a blacklist that never closes; each
new construct is another way for the same code to reach the scanner. The gate
tests what the span contains, so it does not grow with Markdown's grammar. The
region pass is kept because it is cheap and still right for code whose content
does look like math (a fence quoting `$x^2$` verbatim).

Every rewritten span is logged with its content, not just counted. Five review
rounds of "silently rewrites X" is what a count-only log buys.

## Accepted misses

Each keeps the prose warning. All but the last are misses rather than rewrites.

| Shape | Outcome | Why it is accepted |
| --- | --- | --- |
| `$ x $` (padded inline span) | not math | Pandoc does not read it as math either. Recognizing it required a fallback that scanned past the span's own closer, which is how prose between two spans got rewritten. |
| `价格$5，公式$<tab>au$为` (stray dollar against math, no spaces) | first pair wins, repair missed | Only a content heuristic could tell this from a real span, and the CJK shape below shows what such symmetry costs. |
| `$$<tab>au$` (display open, inline close) | not math | Malformed either way; Pandoc finds no display closer. Consistent with rule 5. |
| `$\text{"x"}$` (a double quote inside real math) | not math | The gate's price. A quote in a formula is rarer than a quote in code, and the gate cannot have both. |
| `$a +<newline><tab>au$` (inline span across a line break) | not math | Same trade: an inline formula split across lines is rarer than a shell command that is. Display math is exempt. |
| `A=$X; <tab>ext=1; B=$Y` (one line, no quotes, no backticks) | paired as math, **and rewritten** | The one rewrite left in this table. Nothing in the content separates it from `$x ... $y$`. The gate narrows the exposure from all code to this shape; it does not close it. |

## Rejected alternatives

Both were implemented and measured. Do not reopen either without addressing
the counterexample. A candidate pairing rule is only as good as the corpus it
was run against: the shapes that decided these three rounds — currency amounts
beside math, CJK prose with no spaces around the delimiters, display math
formatted across lines, padded spans, escaped dollars, a stray `$$` — are
pinned in `tests/llm/test_vlm_json_escape_repair.py`. Run a fourth design
against all of them before proposing it; each of the two below passed the
cases its author had in mind and failed a shape they had not thought to try.

**Enumerating the Markdown constructs that hold code.** Five review rounds
walked this path — fenced blocks, then their closing-fence and backtick-run
edge cases, then inline spans, then indented blocks — and each round found
another construct or another boundary. The list does not converge (HTML
blocks, block-quoted fences, tables and `$` in URLs are all still out there),
and every entry defends against the *container* rather than against the thing
that actually matters, which is that the span's content is code. The content
gate replaced this direction; the region pass that survives is a cheap
secondary defense, not the argument.

**Closer escape parity.** Apply the opener's backslash-parity rule to the
closing run as well, for symmetry. CommonMark gives backslash escapes no
effect inside a code span, so ``\` `` *closes* it and the text after is
prose. Honouring the escape there leaves the span unclosed, and an unclosed
run protects nothing — the span's own code would go to the scanner. The
symmetry is the bug: the two runs sit on opposite sides of the escape rule.

**Soft closer fallback.** Accept a whitespace-preceded `$` as a closer when no
stricter candidate remains, so that `$ x $` still pairs. The fallback scans
past the span's own closer and on to the *next* formula's:
`$ x $ then<tab>ext column $y$` rewrote the ordinary prose between the two
spans to a literal `\text`. That is the corruption this whole document is
organized against.

**Suspect-anchored nearest-dollar pairing.** Drop the left-to-right scan; for
each residue, take the nearest unescaped `$` on either side and validate them
as opener and closer. Simpler, and it fixes the currency cases, but:

- It has no parity. The *closer* of one span and the *opener* of the next look
  exactly like a pair when no whitespace separates them from the prose between
  — which is the normal shape of Chinese text. `成本为$a$，领域<tab>au为$b$`
  rewrote the prose.
- Treating `$$` as two single dollars loses display math whose content starts
  on the next line (`$$\n<tab>au^2\n$$`), because the newline after the opener
  is not itself a residue. LLMs format display math this way constantly.

## Where the repair is applied

`repair_vlm_json_escape_damage` runs on string values parsed out of LLM JSON,
and `repair_vlm_json_escape_damage_nested` walks a parsed structure. The
callers are multimodal analysis objects and `_process_json_extraction_result`
(initial extraction, gleaning, and rebuild all parse through it). The
prompts — `entity_extraction_json_system_prompt` and the three multimodal
templates — already require double-escaped backslashes; this repair is the
safety net for models that do not honor that, not a substitute for it.
