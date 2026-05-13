"""Multimodal analysis prompts for LightRAG.

These templates are consumed by ``LightRAG.analyze_multimodal`` to produce
modality-specific analysis JSON written into each sidecar item's
``llm_analyze_result``.

Each template accepts the same variable set so the caller can format them
uniformly:

- ``language``  : target language for ``name`` / ``description`` outputs.
- ``content``   : modality body (table JSON/HTML, equation LaTeX, etc.).
                  Images pass an empty string and rely on ``image_inputs``.
- ``captions``  : caption text or ``"n/a"``.
- ``footnotes`` : joined footnotes string or ``"n/a"``.
- ``leading``   : surrounding leading context or ``"n/a"``.
- ``trailing``  : surrounding trailing context or ``"n/a"``.
- ``item_id``   : sidecar item identifier (for diagnostics, not required by
                  every template).
- ``file_path`` : source document path (diagnostics only).

The output schema differs by modality:

- Image    : ``{"name": str, "type": str, "description": str}``
- Table    : ``{"name": str, "description": str}``
- Equation : ``{"name": str, "equation": str, "description": str}``

Image ``type`` is restricted to :data:`IMAGE_TYPE_ENUM`; values outside the
enum are folded into :data:`IMAGE_TYPE_FALLBACK` by the caller.
"""

from __future__ import annotations


IMAGE_TYPE_ENUM: tuple[str, ...] = (
    "Photo",
    "Illustration",
    "Screenshot",
    "Icon",
    "Chart",
    "Table",
    "Infographic",
    "Flowchart",
    "Chat Log",
    "Wireframe",
    "Texture",
    "Other",
)

IMAGE_TYPE_FALLBACK = "Other"


MULTIMODAL_PROMPTS: dict[str, str] = {}


MULTIMODAL_PROMPTS[
    "image_analysis"
] = """You are an expert image analyzer. Analyze the provided image and return a single JSON object describing its content.

================ INSTRUCTIONS ================

1. CONTENT RECOGNITION
   Examine the image carefully and identify:
   - The primary subject(s), scene, or composition.
   - Salient visual elements (objects, people, text overlays, diagrams, charts, screenshots, etc.).
   - Spatial layout when meaningful (e.g. left/right, foreground/background, panels of a figure).
   - Any visible text — quote it verbatim when short; summarize when long.
   - Color, style, or visual cues only when they materially aid interpretation.

2. USE OF ADDITIONAL CONTEXT
   The Additional Context section provides surrounding information that may help disambiguate the image's role in its source document:
   - Captions      : caption attached to the image ("n/a" = none)
   - Footnotes     : footnote attached to the image ("n/a" = none)
   - Leading Text  : text appearing immediately BEFORE the image ("n/a" = none)
   - Trailing Text : text appearing immediately AFTER the image ("n/a" = none)

   Rules:
   - Use context to disambiguate abbreviations, units, named entities, and the image's purpose.
   - The IMAGE ITSELF takes priority when it conflicts with context — describe what is visible.
   - Only mention a relationship between the image and Leading/Trailing Text if it is clearly supported. If uncertain, omit it.
   - Captions, footnotes, leading text and trailing text must NOT be used to invent visual content not present in the image.

3. NAMING (`name`)
   - Produce a concise, distinctive name (3–8 words, snake_case preferred).
   - It should convey what the image depicts, not just "image".
   - Good examples: `crispr_cas9_workflow_diagram`, `q4_revenue_bar_chart`, `paris_eiffel_tower_photo`.
   - Bad examples: `image`, `figure`, `picture_1`.

4. TYPE (`type`)
   - Pick exactly one value from this fixed list (verbatim, case-sensitive):
     Photo, Illustration, Screenshot, Icon, Chart, Table, Infographic, Flowchart, Chat Log, Wireframe, Texture, Other
   - Choose the single best fit. Use `Other` when no listed type clearly applies.

5. DESCRIPTION (`description`, ≤ 500 words, natural prose — not bullets)
   Cover the following where applicable:
   - What the image depicts overall and what question/claim it visually supports.
   - The primary subject(s), their attributes, and any meaningful relationships between them.
   - Quantitative findings if the image is a chart/diagram (cite specific values when visible).
   - Visible text content that carries meaning (labels, annotations, axis titles).
   - Use specific proper nouns rather than pronouns whenever possible.
   - If the image clearly supports the surrounding context(eading or trailing text), briefly note that relationship at the end. Otherwise omit.

6. OUTPUT RULES
   - Return ONE valid JSON object only.
   - No surrounding markdown, no code fences, no preamble, no explanation.
   - All string values must be properly escaped JSON strings (escape `"` as `\\"`, newlines as `\\n`).
   - The output values for the JSON fields `name` and `description` must be written in `{language}`.

================ ADDITIONAL CONTEXT ================
- Captions: {captions}

- Footnotes: {footnotes}

- Leading Text:
```
{leading}
```

- Trailing Text:
```
{trailing}
```

================ OUTPUT FORMAT ================
{{
  "name": "<concise distinctive name>",
  "type": "<one value from the fixed type list>",
  "description": "<interpretive description, ≤500 words>"
}}

Output:
"""


MULTIMODAL_PROMPTS[
    "table_analysis"
] = """You are an expert table analyzer. The provided content contains table content in JSON or HTML format. Analyze it and return a single JSON object describing its structure and content.

================ INSTRUCTIONS ================

1. CONTENT RECOGNITION
   Read the table carefully and identify:
   - Overall structure: number of rows and columns, presence of merged cells, multi-level headers, row groupings, or totals/subtotals rows.
   - Column headers and (if present) row headers — capture their exact wording.
   - Units of measurement (%, $, ms, kg, etc.) and any scale indicators ("in millions", "×1000").
   - Key data points: maxima, minima, outliers, notable values, totals.
   - Patterns and trends across rows or columns (growth, decline, correlation, ranking).
   - Empty cells, "—", "N/A", or other null markers — preserve them as-is, do NOT fabricate values.
   - Footnote markers inside cells (e.g. "*", "†", "[1]") and what they refer to.

2. USE OF ADDITIONAL CONTEXT
   The Additional Context section provides surrounding information to help you understand the table's role in its source document:
   - Captions      : the table's caption ("n/a" = none)
   - Footnotes     : footnote attached to the table ("n/a" = none)
   - Leading Text  : text appearing immediately BEFORE the table ("n/a" = none)
   - Trailing Text : text appearing immediately AFTER the table ("n/a" = none)

   Rules:
   - Use context to disambiguate column meanings, units, abbreviations, and entity names.
   - TABLE CONTENT TAKES PRIORITY over context when they conflict. Describe what you actually see; note the discrepancy only if it is material.
   - Only mention a relationship between the table and Leading/Trailing Text if it is clearly supported. If uncertain, omit it.
   - Captions, footnotes, leading text and trailing text may only be used for disambiguation purposes and must not be used to infer or fabricate content not present in TABLE CONTENT.
   - NEVER invent rows, columns, values, units, or entities that are not visible.

3. NAMING (`name`)
   - Produce a concise, distinctive name (3–8 words, snake_case preferred).
   - It should convey what the table is about, not just "table".
   - Good examples: `q4_2024_revenue_by_region`, `model_benchmark_accuracy_latency`, `patient_demographics_baseline`.
   - Bad examples: `table`, `data_table`, `results`.

4. DESCRIPTION (`description`, ≤ 500 words, natural prose — not bullets)
   Cover the following where applicable:
   - What the table is about and what question it answers.
   - What the rows represent and what the columns represent (the "shape" of the data).
   - Units, time range, and scope of the data.
   - The most important patterns, trends, comparisons, or outliers — cite specific values from the table to support each observation (e.g. "revenue grew from $1.2M in Q1 to $3.8M in Q4").
   - Any totals, subtotals, averages, or computed columns and what they reveal.
   - Use specific proper nouns (entity names, column names) instead of pronouns.
   - If the table clearly illustrates or supports the surrounding context(eading or trailing text), briefly note that relationship at the end. Otherwise omit.
   - Do not restate the table cell by cell or row by row; focus on interpretation.

5. OUTPUT RULES
   - Return ONE valid JSON object only.
   - No surrounding markdown, no code fences, no preamble, no explanation.
   - All string values must be properly escaped JSON strings (escape `"` as `\\"`, newlines as `\\n`).
   - The output values for the JSON fields `name` and `description` must be written in `{language}`.

================ TABLE CONTENT ================
```
{content}
```

================ ADDITIONAL CONTEXT ================
- Captions: {captions}

- Footnotes: {footnotes}

- Leading Text:
```
{leading}
```

- Trailing Text:
```
{trailing}
```

================ OUTPUT FORMAT ================
{{
  "name": "<concise distinctive name>",
  "description": "<interpretive description of the table, ≤500 words>"
}}

Output:
"""


MULTIMODAL_PROMPTS[
    "equation_analysis"
] = """You are an expert analyzer of mathematical and chemical equations. The input is a TEXT-form equation written in LaTeX or Markdown. Analyze it and return a single JSON object describing its meaning and role.

================ INSTRUCTIONS ================

1. CONTENT RECOGNITION
   Read the equation carefully and identify:
   - The type of expression: definition, identity, equation to solve, inequality, differential / integral equation, recurrence, chemical reaction, balance equation, etc.
   - The mathematical or chemical meaning of the expression as a whole.
   - The variables, constants, operators, and functions that appear, and what each likely denotes given the surrounding context.
   - The application domain (e.g. classical mechanics, probability, thermodynamics, organic chemistry, machine learning loss function) inferred from context.
   - Any physical, statistical, or theoretical significance.
   - Whether the expression matches a well-known named formula (e.g. Bayes' theorem, Schrödinger equation, softmax, Michaelis–Menten). Name it explicitly when you are confident; do NOT guess.

2. USE OF ADDITIONAL CONTEXT
   The Additional Context section provides surrounding information to help you understand the equation's role in its source document:
   - Captions      : the equation's caption or label ("n/a" = none)
   - Footnotes     : footnote attached to the equation ("n/a" = none)
   - Leading Text  : text appearing immediately BEFORE the equation ("n/a" = none)
   - Trailing Text : text appearing immediately AFTER the equation ("n/a" = none)

   Rules:
   - Use context to determine variable meanings, units, and the domain of discussion.
   - THE EQUATION ITSELF TAKES PRIORITY over context if they conflict; note the discrepancy if material.
   - Only mention a relationship between the equation and Leading/Trailing Text if it is clearly supported. If uncertain, omit it.
   - Captions, footnotes, leading text and trailing text may only be used for disambiguation purposes and must not be used to infer or fabricate content not present in EQUATION BODY.
   - NEVER invent variables, terms, or interpretations that are not justified by either the equation or the context.

3. NAMING (`name`)
   - Produce a concise, distinctive name (3–8 words, snake_case preferred).
   - It should convey what the equation IS or DOES, not just "equation".
   - Good examples:
       `bayes_theorem_posterior`
       `softmax_cross_entropy_loss`
       `ideal_gas_law`
       `michaelis_menten_rate`
       `combustion_of_methane`
       `quadratic_formula_roots`
   - Bad examples:
       `equation`, `formula`, `math`, `the_equation`, `eq_1`

4. NORMALIZED EQUATION (`equation`)
   - Output the math-mode BODY ONLY. Do NOT wrap in any delimiter or environment: no `$...$`, no `$$...$$`, no `\\(...\\)`, no `\\[...\\]`, no `\\begin{{equation}}...\\end{{equation}}`.
   - Strip those outer wrappers if present in the input.
   - KEEP semantic inner environments such as `aligned`, `cases`, `pmatrix`, `bmatrix`, `array`, `split` — they are part of the equation's structure, not delimiters.
   - If the input uses `\\begin{{align}}` or `\\begin{{align*}}`, convert to `\\begin{{aligned}}`.
   - Strip equation numbering (`\\tag{{...}}`, automatic numbers from `align`/`equation`).
   - Preserve all symbols, subscripts, superscripts, and operators faithfully. Do NOT simplify or rename variables.
   - Convert Markdown / plain-text / Unicode math to standard LaTeX (`x^2` → `x^{{2}}`, `sqrt(a)` → `\\sqrt{{a}}`, `≤` → `\\leq`, `α` → `\\alpha`).
   - For chemical equations, use `mhchem`: `\\ce{{2H2 + O2 -> 2H2O}}`.
   - If multiple independent equations appear together, join them with `\\\\` inside a single `\\begin{{aligned}}...\\end{{aligned}}` and note the grouping in `description`.

5. DESCRIPTION (`description`, ≤ 300 words, natural prose — not bullets)
   Cover the following where applicable:
   - What the equation expresses and what problem it addresses.
   - Its role in the surrounding text (e.g. defines a quantity, states a constraint, derives a result, models a phenomenon).
   - The named formula it corresponds to, if any, and where it is commonly used.
   - Briefly clarify only those symbols whose meaning is non-obvious or domain-specific, OR whose meaning is fixed by the Leading/Trailing Text. Do NOT enumerate every symbol mechanically.
   - Use specific proper nouns (variable names, entity names) instead of pronouns.
   - If the equation clearly illustrates or supports the surrounding context(eading or trailing text), briefly note that relationship at the end. Otherwise omit.

6. OUTPUT RULES
   - Return ONE valid JSON object only.
   - No surrounding markdown, no code fences, no preamble, no explanation.
   - All string values must be properly escaped JSON strings (escape `"` as `\\"`, escape backslashes as `\\\\`, newlines as `\\n`).
   - LaTeX backslashes inside the `equation` string must be double-escaped (e.g. `\\frac{{a}}{{b}}` is written as `"\\\\frac{{a}}{{b}}"` in the JSON).
   - If the input uses `\\begin{{align}}` or `\\begin{{align*}}`, convert to `\\begin{{aligned}}` in the output (since the outer display wrapper is stripped).
   - The output values for the JSON fields `name` and `description` must be written in `{language}`.

================ EQUATION BODY ================
```
{content}
```

================ ADDITIONAL CONTEXT ================
- Captions: {captions}

- Footnotes: {footnotes}

- Leading Text:
```
{leading}
```

- Trailing Text:
```
{trailing}
```

================ OUTPUT FORMAT ================
{{
  "name": "<concise distinctive name>",
  "equation": "<normalized LaTeX, math-mode body only>",
  "description": "<interpretive description, ≤300 words>"
}}

Output:
"""


__all__ = [
    "IMAGE_TYPE_ENUM",
    "IMAGE_TYPE_FALLBACK",
    "MULTIMODAL_PROMPTS",
]
