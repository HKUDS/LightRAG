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

- **`_WS_LATEX_SUSPECT_PATTERN`** (`\b`) is the prose detector, used only to
  warn. The word boundary keeps `col<tab>ext_id` and `<tab>au2` — plausible
  values in tab-separated data — out of the log.
- **`_WS_LATEX_MATH_PATTERN`** (`(?![A-Za-z])`) is used inside a confirmed
  math span. There, `_`, `{`, `^` and digits are the most common characters to
  follow a command, and the "the residue might be an English word" argument
  does not apply. Requiring `\b` there left `$<tab>au_i$` neither repaired nor
  warned about — silent, because the prose pattern refuses it too.

The in-math pattern is a strict superset of the prose pattern, which is why a
repaired span can never re-trigger the prose warning afterwards.

Accepted gap: the prose detector is silent whenever a word character follows
the residue, and a CJK ideograph is a word character, so `阈值<tab>au为0.5` in
ordinary Chinese prose is neither repaired (no math span) nor warned about.
This is a detector-side decision, tracked separately from the pairing rules
below.

## Delimiter policy

Dollar delimiters are ambiguous by construction. Prose contains stray and
currency dollars, and even Pandoc reads `$5 ... $x$` as one span, so **no
delimiter rule is correct in both directions**. The scanner is therefore
deliberately biased:

> Never rewrite text a Pandoc-style parser would not call math, and accept the
> misses that follow.

Rewriting prose is corruption that reaches storage silently. Skipping a repair
leaves the damage that was already there and, in most shapes, still logs the
prose warning. The two are not symmetric, and the bias follows that asymmetry.

The rules, all applied left to right:

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

## Accepted misses

Each is a miss, never a rewrite, and each keeps the prose warning.

| Shape | Outcome | Why it is accepted |
| --- | --- | --- |
| `$ x $` (padded inline span) | not math | Pandoc does not read it as math either. Recognizing it required a fallback that scanned past the span's own closer, which is how prose between two spans got rewritten. |
| `价格$5，公式$<tab>au$为` (stray dollar against math, no spaces) | first pair wins, repair missed | Only a content heuristic could tell this from a real span, and the CJK shape below shows what such symmetry costs. |
| `$$<tab>au$` (display open, inline close) | not math | Malformed either way; Pandoc finds no display closer. Consistent with rule 5. |

## Rejected alternatives

Both were implemented and measured. Do not reopen either without addressing
the counterexample.

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
