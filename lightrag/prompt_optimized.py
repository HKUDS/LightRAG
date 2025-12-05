"""
Optimized prompts for LightRAG entity extraction.

Contains two variants:
- PROMPTS_OPTIMIZED: Aggressive optimization (55% token savings, sparser graphs)
- PROMPTS_BALANCED: Moderate optimization (40% token savings, richer graphs)
"""

from __future__ import annotations

from typing import Any

PROMPTS_OPTIMIZED: dict[str, Any] = {}
PROMPTS_BALANCED: dict[str, Any] = {}

PROMPTS_OPTIMIZED['DEFAULT_TUPLE_DELIMITER'] = '<|#|>'
PROMPTS_OPTIMIZED['DEFAULT_COMPLETION_DELIMITER'] = '<|COMPLETE|>'

# =============================================================================
# OPTIMIZED: Entity Extraction System Prompt
# Original: 1,375 tokens | Target: ~850 tokens (~38% reduction)
# Changes:
#   - Consolidated format instructions (removed 3x repetition)
#   - Removed delimiter usage section (examples demonstrate it)
#   - Merged naming conventions into single statement
#   - Tightened language throughout
# =============================================================================

PROMPTS_OPTIMIZED['entity_extraction_system_prompt'] = """---Role---
You are a Knowledge Graph Specialist extracting entities and relationships from text.

---Output Format---
Output raw lines only—NO markdown, NO headers, NO backticks around lines.

Entity format (4 fields per line):
entity{tuple_delimiter}name{tuple_delimiter}type{tuple_delimiter}description

Relation format (5 fields per line):
relation{tuple_delimiter}source{tuple_delimiter}target{tuple_delimiter}keywords{tuple_delimiter}description

Use Title Case for entity names. Separate multiple keywords with commas (not {tuple_delimiter}).
Output all entities first, then relationships. End with {completion_delimiter}.

---Entity Extraction---
- Extract clearly defined, meaningful entities
- Types: `{entity_types}` (use `Other` if none fit)
- Description: concise summary based only on text content

---Relationship Extraction---
Extract meaningful relationships between entities:
- **Direct:** explicit interactions, actions, connections
- **Comparative:** entities grouped, ranked, or compared together
- **Hierarchical:** part-of, member-of, type-of connections
- **Causal:** explicit cause-effect relationships
- **Categorical:** entities sharing explicit group membership

For N-ary relationships (3+ entities), decompose into binary pairs.
Relationships are undirected; avoid duplicates.
Do NOT invent speculative connections—only extract what text supports.

---Guidelines---
- Write in third person; avoid pronouns like "this article", "I", "you"
- Output in `{language}`. Keep proper nouns in original language.
- Prioritize most significant relationships first.

---Examples---
{examples}

---Input---
Entity_types: [{entity_types}]
Text:
```
{input_text}
```
"""

# =============================================================================
# OPTIMIZED: Entity Extraction Examples
# Original: 2,204 tokens (4 examples) | Target: ~1,300 tokens (2-3 examples)
# Changes:
#   - Kept Example 1 (narrative): demonstrates character extraction
#   - Condensed Example 2 (financial): demonstrates factual/numeric extraction
#   - Kept Example 4 (legal): short, demonstrates org relationships
#   - Removed Example 3 (medical): redundant with Example 4 pattern
# =============================================================================

PROMPTS_OPTIMIZED['entity_extraction_examples'] = [
    # Example 1: Narrative text with characters and abstract concepts
    """<Input Text>
```
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.

Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. "If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us."

Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.
```

<Output>
entity{tuple_delimiter}Alex{tuple_delimiter}person{tuple_delimiter}Alex experiences frustration and observes dynamics among other characters.
entity{tuple_delimiter}Taylor{tuple_delimiter}person{tuple_delimiter}Taylor shows authoritarian certainty but displays reverence toward a device.
entity{tuple_delimiter}Jordan{tuple_delimiter}person{tuple_delimiter}Jordan shares commitment to discovery and has a significant interaction with Taylor.
entity{tuple_delimiter}Cruz{tuple_delimiter}person{tuple_delimiter}Cruz holds a vision of control and order that others rebel against.
entity{tuple_delimiter}The Device{tuple_delimiter}equipment{tuple_delimiter}Central object with game-changing potential, revered by Taylor.
relation{tuple_delimiter}Alex{tuple_delimiter}Jordan{tuple_delimiter}shared goals, rebellion{tuple_delimiter}Alex and Jordan share commitment to discovery, contrasting with Cruz's vision.
relation{tuple_delimiter}Taylor{tuple_delimiter}Jordan{tuple_delimiter}conflict resolution, mutual respect{tuple_delimiter}Taylor and Jordan reach an uneasy truce regarding the device.
relation{tuple_delimiter}Jordan{tuple_delimiter}Cruz{tuple_delimiter}ideological conflict{tuple_delimiter}Jordan's discovery commitment rebels against Cruz's control vision.
relation{tuple_delimiter}Taylor{tuple_delimiter}The Device{tuple_delimiter}reverence, significance{tuple_delimiter}Taylor shows reverence toward the device's potential impact.
{completion_delimiter}

""",
    # Example 2: Factual/financial text with metrics and organizations
    """<Input Text>
```
Stock markets faced a sharp downturn as tech giants saw significant declines, with the global tech index dropping 3.4%. Nexon Technologies saw its stock plummet 7.8% after lower-than-expected earnings. In contrast, Omega Energy posted a 2.1% gain driven by rising oil prices.

Gold futures rose 1.5% to $2,080/oz as investors sought safe-haven assets. Crude oil climbed to $87.60/barrel. The Federal Reserve's upcoming policy announcement is expected to influence market stability.
```

<Output>
entity{tuple_delimiter}Global Tech Index{tuple_delimiter}category{tuple_delimiter}Tracks major tech stocks; dropped 3.4% today.
entity{tuple_delimiter}Nexon Technologies{tuple_delimiter}organization{tuple_delimiter}Tech company whose stock fell 7.8% after disappointing earnings.
entity{tuple_delimiter}Omega Energy{tuple_delimiter}organization{tuple_delimiter}Energy company that gained 2.1% due to rising oil prices.
entity{tuple_delimiter}Gold Futures{tuple_delimiter}product{tuple_delimiter}Rose 1.5% to $2,080/oz as safe-haven investment.
entity{tuple_delimiter}Crude Oil{tuple_delimiter}product{tuple_delimiter}Climbed to $87.60/barrel on supply constraints and demand.
entity{tuple_delimiter}Federal Reserve{tuple_delimiter}organization{tuple_delimiter}Central bank whose policy announcement may impact markets.
relation{tuple_delimiter}Nexon Technologies{tuple_delimiter}Global Tech Index{tuple_delimiter}component, decline{tuple_delimiter}Nexon's decline contributed to the tech index drop.
relation{tuple_delimiter}Omega Energy{tuple_delimiter}Crude Oil{tuple_delimiter}price correlation{tuple_delimiter}Omega's gain driven by rising crude oil prices.
relation{tuple_delimiter}Gold Futures{tuple_delimiter}Global Tech Index{tuple_delimiter}inverse correlation, safe-haven{tuple_delimiter}Gold rose as tech stocks fell, reflecting flight to safety.
relation{tuple_delimiter}Federal Reserve{tuple_delimiter}Global Tech Index{tuple_delimiter}policy impact{tuple_delimiter}Fed policy expectations influence market stability.
{completion_delimiter}

""",
    # Example 3: Short legal/regulatory text
    """<Input Text>
```
The merger between Acme Corp and Beta Industries requires Federal Trade Commission approval. Legal counsel advised the deal may face antitrust scrutiny due to market concentration concerns.
```

<Output>
entity{tuple_delimiter}Acme Corp{tuple_delimiter}organization{tuple_delimiter}Company proposing merger with Beta Industries.
entity{tuple_delimiter}Beta Industries{tuple_delimiter}organization{tuple_delimiter}Company proposing merger with Acme Corp.
entity{tuple_delimiter}Federal Trade Commission{tuple_delimiter}organization{tuple_delimiter}Regulatory body that must approve the merger.
entity{tuple_delimiter}Antitrust Scrutiny{tuple_delimiter}concept{tuple_delimiter}Regulatory review for market concentration concerns.
relation{tuple_delimiter}Acme Corp{tuple_delimiter}Beta Industries{tuple_delimiter}merger{tuple_delimiter}Companies are parties to a proposed merger.
relation{tuple_delimiter}Federal Trade Commission{tuple_delimiter}Acme Corp{tuple_delimiter}regulatory approval{tuple_delimiter}FTC must approve the merger.
relation{tuple_delimiter}Antitrust Scrutiny{tuple_delimiter}Federal Trade Commission{tuple_delimiter}regulatory process{tuple_delimiter}FTC conducts antitrust review to assess market impact.
{completion_delimiter}

""",
]

# =============================================================================
# OPTIMIZED: User Prompt
# Original: 174 tokens | Target: ~100 tokens
# =============================================================================

PROMPTS_OPTIMIZED['entity_extraction_user_prompt'] = """---Task---
Extract entities and relationships from the text above.

Follow the system prompt format exactly. Output only the extraction list—no explanations.
End with `{completion_delimiter}`. Output in {language}; keep proper nouns in original language.

<Output>
"""

# =============================================================================
# OPTIMIZED: Continue Extraction Prompt
# Original: 499 tokens | Target: ~250 tokens
# =============================================================================

PROMPTS_OPTIMIZED['entity_continue_extraction_user_prompt'] = """---Task---
Review your extraction for missed or incorrectly formatted entities/relationships.

**Focus on:**
1. Orphan entities (no relationships)—check if text groups, compares, or relates them
2. Missed relationships from lists, rankings, or shared contexts
3. Formatting errors (wrong field count, missing delimiter)

**Rules:**
- Do NOT re-output correctly extracted items
- Only output new or corrected items
- End with `{completion_delimiter}`
- Output in {language}

<Output>
"""


# =============================================================================
# COMPARISON HELPER
# =============================================================================

def compare_token_counts():
    """Compare token counts between original and optimized prompts."""
    import tiktoken
    from lightrag.prompt import PROMPTS

    enc = tiktoken.encoding_for_model("gpt-4")

    print("=== Token Comparison: Original vs Optimized ===\n")

    comparisons = [
        ('entity_extraction_system_prompt', 'System Prompt'),
        ('entity_extraction_user_prompt', 'User Prompt'),
        ('entity_continue_extraction_user_prompt', 'Continue Prompt'),
    ]

    total_orig = 0
    total_opt = 0

    for key, name in comparisons:
        orig_tokens = len(enc.encode(PROMPTS[key]))
        opt_tokens = len(enc.encode(PROMPTS_OPTIMIZED[key]))
        savings = orig_tokens - opt_tokens
        pct = (savings / orig_tokens) * 100

        total_orig += orig_tokens
        total_opt += opt_tokens

        print(f"{name}:")
        print(f"  Original:  {orig_tokens:,} tokens")
        print(f"  Optimized: {opt_tokens:,} tokens")
        print(f"  Savings:   {savings:,} tokens ({pct:.1f}%)\n")

    # Examples
    orig_examples = '\n'.join(PROMPTS['entity_extraction_examples'])
    opt_examples = '\n'.join(PROMPTS_OPTIMIZED['entity_extraction_examples'])

    orig_ex_tokens = len(enc.encode(orig_examples))
    opt_ex_tokens = len(enc.encode(opt_examples))
    ex_savings = orig_ex_tokens - opt_ex_tokens
    ex_pct = (ex_savings / orig_ex_tokens) * 100

    total_orig += orig_ex_tokens
    total_opt += opt_ex_tokens

    print(f"Examples:")
    print(f"  Original:  {orig_ex_tokens:,} tokens (4 examples)")
    print(f"  Optimized: {opt_ex_tokens:,} tokens (3 examples)")
    print(f"  Savings:   {ex_savings:,} tokens ({ex_pct:.1f}%)\n")

    total_savings = total_orig - total_opt
    total_pct = (total_savings / total_orig) * 100

    print("=" * 50)
    print(f"TOTAL:")
    print(f"  Original:  {total_orig:,} tokens")
    print(f"  Optimized: {total_opt:,} tokens")
    print(f"  Savings:   {total_savings:,} tokens ({total_pct:.1f}%)")


# =============================================================================
# BALANCED: Entity Extraction Prompts
# Target: ~40% token savings while maintaining rich extraction
# Key difference: Explicitly encourages conceptual/abstract entities
# =============================================================================

PROMPTS_BALANCED['DEFAULT_TUPLE_DELIMITER'] = '<|#|>'
PROMPTS_BALANCED['DEFAULT_COMPLETION_DELIMITER'] = '<|COMPLETE|>'

PROMPTS_BALANCED['entity_extraction_system_prompt'] = """---Role---
You are a Knowledge Graph Specialist extracting entities and relationships from text.

---Output Format---
Output raw lines only—NO markdown, NO headers, NO backticks.

Entity: entity{tuple_delimiter}name{tuple_delimiter}type{tuple_delimiter}description
Relation: relation{tuple_delimiter}source{tuple_delimiter}target{tuple_delimiter}keywords{tuple_delimiter}description

Use Title Case for names. Separate keywords with commas. Output entities first, then relations. End with {completion_delimiter}.

---Entity Extraction---
Extract BOTH concrete and abstract entities:
- **Concrete:** Named people, organizations, places, products, dates
- **Abstract:** Concepts, events, categories, processes mentioned in text (e.g., "market selloff", "merger", "pandemic")

Types: `{entity_types}` (use `Other` if none fit)

---Relationship Extraction---
Extract meaningful relationships:
- **Direct:** explicit interactions, actions, connections
- **Categorical:** entities sharing group membership or classification
- **Causal:** cause-effect relationships
- **Hierarchical:** part-of, member-of, type-of

Create intermediate concept entities when they help connect related items (e.g., "Vaccines" connecting Pfizer/Moderna/AstraZeneca).

For N-ary relationships, decompose into binary pairs. Avoid duplicates.

---Guidelines---
- Third person only; no pronouns like "this article", "I", "you"
- Output in `{language}`. Keep proper nouns in original language.

---Examples---
{examples}

---Input---
Entity_types: [{entity_types}]
Text:
```
{input_text}
```
"""

PROMPTS_BALANCED['entity_extraction_examples'] = [
    # Example 1: Shows abstract concept extraction (Market Selloff as hub)
    """<Input Text>
```
Stock markets faced a sharp downturn as tech giants saw significant declines, with the global tech index dropping 3.4%. Nexon Technologies saw its stock plummet 7.8% after lower-than-expected earnings. In contrast, Omega Energy posted a 2.1% gain driven by rising oil prices.

Gold futures rose 1.5% to $2,080/oz as investors sought safe-haven assets. The Federal Reserve's upcoming policy announcement is expected to influence market stability.
```

<Output>
entity{tuple_delimiter}Market Selloff{tuple_delimiter}event{tuple_delimiter}Significant decline in stock values due to investor concerns.
entity{tuple_delimiter}Global Tech Index{tuple_delimiter}category{tuple_delimiter}Tracks major tech stocks; dropped 3.4% today.
entity{tuple_delimiter}Nexon Technologies{tuple_delimiter}organization{tuple_delimiter}Tech company whose stock fell 7.8% after disappointing earnings.
entity{tuple_delimiter}Omega Energy{tuple_delimiter}organization{tuple_delimiter}Energy company that gained 2.1% due to rising oil prices.
entity{tuple_delimiter}Gold Futures{tuple_delimiter}product{tuple_delimiter}Rose 1.5% to $2,080/oz as safe-haven investment.
entity{tuple_delimiter}Federal Reserve{tuple_delimiter}organization{tuple_delimiter}Central bank whose policy may impact markets.
relation{tuple_delimiter}Global Tech Index{tuple_delimiter}Market Selloff{tuple_delimiter}market decline{tuple_delimiter}Tech index drop is part of broader selloff.
relation{tuple_delimiter}Nexon Technologies{tuple_delimiter}Market Selloff{tuple_delimiter}tech decline{tuple_delimiter}Nexon among hardest hit in selloff.
relation{tuple_delimiter}Omega Energy{tuple_delimiter}Market Selloff{tuple_delimiter}contrast, resilience{tuple_delimiter}Omega gained while broader market sold off.
relation{tuple_delimiter}Gold Futures{tuple_delimiter}Market Selloff{tuple_delimiter}safe-haven{tuple_delimiter}Gold rose as investors fled stocks.
relation{tuple_delimiter}Federal Reserve{tuple_delimiter}Market Selloff{tuple_delimiter}policy impact{tuple_delimiter}Fed policy expectations contributed to volatility.
{completion_delimiter}

""",
    # Example 2: Shows intermediate entity (Vaccines) connecting multiple orgs
    """<Input Text>
```
COVID-19 vaccines developed by Pfizer, Moderna, and AstraZeneca have shown high efficacy in preventing severe illness. The World Health Organization recommends vaccination for all eligible adults.
```

<Output>
entity{tuple_delimiter}COVID-19{tuple_delimiter}concept{tuple_delimiter}Disease that vaccines are designed to prevent.
entity{tuple_delimiter}Vaccines{tuple_delimiter}product{tuple_delimiter}Medical products developed to prevent COVID-19.
entity{tuple_delimiter}Pfizer{tuple_delimiter}organization{tuple_delimiter}Pharmaceutical company that developed a COVID-19 vaccine.
entity{tuple_delimiter}Moderna{tuple_delimiter}organization{tuple_delimiter}Pharmaceutical company that developed a COVID-19 vaccine.
entity{tuple_delimiter}AstraZeneca{tuple_delimiter}organization{tuple_delimiter}Pharmaceutical company that developed a COVID-19 vaccine.
entity{tuple_delimiter}World Health Organization{tuple_delimiter}organization{tuple_delimiter}Global health body recommending vaccination.
relation{tuple_delimiter}Pfizer{tuple_delimiter}Vaccines{tuple_delimiter}development{tuple_delimiter}Pfizer developed a COVID-19 vaccine.
relation{tuple_delimiter}Moderna{tuple_delimiter}Vaccines{tuple_delimiter}development{tuple_delimiter}Moderna developed a COVID-19 vaccine.
relation{tuple_delimiter}AstraZeneca{tuple_delimiter}Vaccines{tuple_delimiter}development{tuple_delimiter}AstraZeneca developed a COVID-19 vaccine.
relation{tuple_delimiter}Vaccines{tuple_delimiter}COVID-19{tuple_delimiter}prevention{tuple_delimiter}Vaccines prevent severe COVID-19 illness.
relation{tuple_delimiter}World Health Organization{tuple_delimiter}Vaccines{tuple_delimiter}recommendation{tuple_delimiter}WHO recommends vaccination for adults.
{completion_delimiter}

""",
    # Example 3: Short legal example
    """<Input Text>
```
The merger between Acme Corp and Beta Industries requires Federal Trade Commission approval due to antitrust concerns.
```

<Output>
entity{tuple_delimiter}Merger{tuple_delimiter}event{tuple_delimiter}Proposed business combination between Acme Corp and Beta Industries.
entity{tuple_delimiter}Acme Corp{tuple_delimiter}organization{tuple_delimiter}Company involved in proposed merger.
entity{tuple_delimiter}Beta Industries{tuple_delimiter}organization{tuple_delimiter}Company involved in proposed merger.
entity{tuple_delimiter}Federal Trade Commission{tuple_delimiter}organization{tuple_delimiter}Regulatory body that must approve the merger.
relation{tuple_delimiter}Acme Corp{tuple_delimiter}Merger{tuple_delimiter}party to{tuple_delimiter}Acme Corp is party to the merger.
relation{tuple_delimiter}Beta Industries{tuple_delimiter}Merger{tuple_delimiter}party to{tuple_delimiter}Beta Industries is party to the merger.
relation{tuple_delimiter}Federal Trade Commission{tuple_delimiter}Merger{tuple_delimiter}regulatory approval{tuple_delimiter}FTC must approve the merger.
{completion_delimiter}

""",
]

PROMPTS_BALANCED['entity_extraction_user_prompt'] = """---Task---
Extract entities and relationships from the text. Include both concrete entities AND abstract concepts/events.

Follow format exactly. Output only extractions—no explanations. End with `{completion_delimiter}`.
Output in {language}; keep proper nouns in original language.

<Output>
"""

PROMPTS_BALANCED['entity_continue_extraction_user_prompt'] = """---Task---
Review extraction for missed entities/relationships.

Check for:
1. Abstract concepts that could serve as hubs (events, categories, processes)
2. Orphan entities that need connections
3. Formatting errors

Only output NEW or CORRECTED items. End with `{completion_delimiter}`. Output in {language}.

<Output>
"""


if __name__ == "__main__":
    compare_token_counts()
