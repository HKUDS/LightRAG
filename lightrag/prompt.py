from __future__ import annotations

from typing import Any

PROMPTS: dict[str, Any] = {}

# All delimiters must be formatted as "<|TOKEN|>" style markers (e.g., "<|#|>" or "<|COMPLETE|>")
PROMPTS['DEFAULT_TUPLE_DELIMITER'] = '<|#|>'
PROMPTS['DEFAULT_COMPLETION_DELIMITER'] = '<|COMPLETE|>'

PROMPTS['entity_extraction_system_prompt'] = """---Role---
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

PROMPTS['entity_extraction_user_prompt'] = """---Task---
Extract entities and relationships from the text. Include both concrete entities AND abstract concepts/events.

Follow format exactly. Output only extractions—no explanations. End with `{completion_delimiter}`.
Output in {language}; keep proper nouns in original language.

<Output>
"""

PROMPTS['entity_continue_extraction_user_prompt'] = """---Task---
Review extraction for missed entities/relationships.

Check for:
1. Abstract concepts that could serve as hubs (events, categories, processes)
2. Orphan entities that need connections
3. Formatting errors

Only output NEW or CORRECTED items. End with `{completion_delimiter}`. Output in {language}.

<Output>
"""

PROMPTS['entity_extraction_examples'] = [
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
    # Example 3: Short legal example with hub entity (Merger)
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

PROMPTS['summarize_entity_descriptions'] = """---Task---
Merge multiple descriptions of "{description_name}" ({description_type}) into one comprehensive summary.

Rules:
- Plain text output only, no formatting or extra text
- Include ALL key facts from every description
- Third person, mention entity name at start
- Max {summary_length} tokens
- Output in {language}; keep proper nouns in original language
- If descriptions conflict: reconcile or note uncertainty

Descriptions:
```
{description_list}
```

Output:"""

PROMPTS['fail_response'] = "Sorry, I'm not able to provide an answer to that question.[no-context]"

# Default RAG response prompt - cite-ready (no LLM-generated citations)
# Citations are added by post-processing. This gives cleaner, more accurate results.
PROMPTS[
    'rag_response'
] = """You're helping someone understand a topic. Write naturally, like explaining to a curious friend.

Focus on directly answering the question asked. Include only information relevant to the query.

STYLE RULES:
- Flowing paragraphs, NOT bullets or numbered lists
- Connect sentences with transitions (however, this means, for example)
- Combine related facts into sentences rather than listing separately
- Vary sentence length - mix short and long

GOOD EXAMPLE:
"Machine learning is a branch of AI that enables computers to learn from data without explicit programming. The field includes several approaches: supervised learning uses labeled data, while unsupervised learning finds hidden patterns. Deep learning, using multi-layer neural networks, has proven especially effective for image recognition and language processing."

BAD EXAMPLE:
"- Machine learning: branch of AI
- Learns from data
- Types: supervised, unsupervised
- Deep learning uses neural networks"

Answer using ONLY the context below. Prefer information from the context over general knowledge.
Do NOT include [1], [2] citations - they're added automatically.

{user_prompt}

Context:
{context_data}
"""

# Strict mode suffix - append when response_type="strict"
PROMPTS['rag_response_strict_suffix'] = """
STRICT GROUNDING:
- NEVER state specific numbers/dates unless they appear EXACTLY in context
- If information isn't in context, say "not specified in available information"
- Entity summaries for overview, Source Excerpts for precision
"""

# Default naive RAG response prompt - cite-ready (no LLM-generated citations)
PROMPTS['naive_rag_response'] = """---Task---
Answer the query using ONLY the provided context.

Rules:
- NO citation markers ([1], [2]) - added automatically
- NO References section - added automatically
- Each factual claim as distinct, traceable sentence
- If not in context, say so clearly
- Match query language; use Markdown formatting
- Response type: {response_type}

{user_prompt}

---Context---
{content_data}
"""

PROMPTS['kg_query_context'] = """
## Entity Summaries (use for definitions and general facts)

```json
{entities_str}
```

## Relationships (use to explain connections between concepts)

```json
{relations_str}
```

## Source Excerpts (use for specific facts, numbers, quotes)

```json
{text_chunks_str}
```

## References
{reference_list_str}

"""

PROMPTS['naive_query_context'] = """
Document Chunks (Each entry includes a reference_id that refers to the `Reference Document List`):

```json
{text_chunks_str}
```

Reference Document List (Each entry starts with a [reference_id] that corresponds to entries in the Document Chunks):

```
{reference_list_str}
```

"""

PROMPTS['keywords_extraction'] = """---Task---
Extract keywords from the query for RAG retrieval.

Output valid JSON (no markdown):
{{"high_level_keywords": [...], "low_level_keywords": [...]}}

Guidelines:
- high_level: Topic categories, question types, abstract themes
- low_level: Specific terms from the query (entities, technical terms, key concepts)
- Extract at least 1 keyword per category for meaningful queries
- Only return empty lists for nonsensical input (e.g., "asdfgh", "hello")

---Examples---
{examples}

---Query---
{query}

Output:"""

PROMPTS['keywords_extraction_examples'] = [
    """Query: "What is the capital of France?"
Output: {{"high_level_keywords": ["Geography", "Capital city"], "low_level_keywords": ["France"]}}
""",
    """Query: "Why does inflation affect interest rates?"
Output: {{"high_level_keywords": ["Economics", "Cause-effect"], "low_level_keywords": ["inflation", "interest rates"]}}
""",
    """Query: "How does Python compare to JavaScript for web development?"
Output: {{"high_level_keywords": ["Programming languages", "Comparison"], "low_level_keywords": ["Python", "JavaScript"]}}
""",
]

PROMPTS['orphan_connection_validation'] = """---Task---
Evaluate if a meaningful relationship exists between two entities.

Orphan: {orphan_name} ({orphan_type}) - {orphan_description}
Candidate: {candidate_name} ({candidate_type}) - {candidate_description}
Similarity: {similarity_score}

Valid relationship types:
- Direct: One uses/creates/owns the other
- Industry: Both operate in same sector (finance, tech, healthcare)
- Competitive: Direct competitors or alternatives
- Temporal: Versions, successors, or historical connections
- Dependency: One relies on/runs on the other

Confidence levels (use these exact labels):
- HIGH: Direct/explicit relationship (Django is Python framework, iOS is Apple product)
- MEDIUM: Strong implicit or industry relationship (Netflix runs on AWS, Bitcoin and Visa both in payments)
- LOW: Very weak, tenuous connection
- NONE: No logical relationship

Output valid JSON:
{{"should_connect": bool, "confidence": "HIGH"|"MEDIUM"|"LOW"|"NONE", "relationship_type": str|null, "relationship_keywords": str|null, "relationship_description": str|null, "reasoning": str}}

Rules:
- HIGH/MEDIUM: should_connect=true (same industry = MEDIUM)
- LOW/NONE: should_connect=false
- High similarity alone is NOT sufficient
- Explain the specific relationship in reasoning

Example: Python↔Django
{{"should_connect": true, "confidence": "HIGH", "relationship_type": "direct", "relationship_keywords": "framework, built-with", "relationship_description": "Django is a web framework written in Python", "reasoning": "Direct explicit relationship - Django is implemented in Python"}}

Example: Mozart↔Docker
{{"should_connect": false, "confidence": "NONE", "relationship_type": null, "relationship_keywords": null, "relationship_description": null, "reasoning": "No logical connection between classical composer and container technology"}}

Output:"""
