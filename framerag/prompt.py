"""All LLM prompts for the FrameRAG pipeline."""
from __future__ import annotations

PROMPTS: dict[str, str] = {}

# ─────────────────────────────────────────────────────────────────────────────
# Call 1 — Entity Extraction (extended from LightRAG)
# ─────────────────────────────────────────────────────────────────────────────

PROMPTS["entity_extraction"] = """---Role---
You are a Knowledge Graph specialist. Extract all named entities from the text.

---Instructions---
For EACH entity output a JSON object with:
  - entity_name     : canonical name (title-case)
  - entity_type     : one of [PERSON, ORG, LOCATION, ARTIFACT, CONCEPT, EVENT, OTHER]
  - entity_description : concise description based ONLY on the text
  - entity_aliases  : list of other surface forms used in the text (e.g. pronouns, abbreviations)
  - entity_salience : HIGH | MEDIUM | LOW (importance to the text)

Rules:
- Output ONLY a JSON array. No prose, no markdown fences.
- Proper nouns keep original language.
- Write descriptions in third person, no pronouns like "it" or "they" at the start.
- If no entities found, output [].

---Text---
{chunk_text}

---Output (JSON array)---"""

# ─────────────────────────────────────────────────────────────────────────────
# Call 2 — Event Detection + Frame Induction + Role Assignment (combined)
# ─────────────────────────────────────────────────────────────────────────────

PROMPTS["event_frame_role"] = """---Role---
You are an expert in Frame Semantics (Fillmore 1985) and event extraction.

---Background: Frame Semantics---
A "frame" is a schematic representation of a situation type. Each frame has:
  - Lexical Unit (LU): word that evokes the frame (e.g. "acquire.v")
  - Frame Definition: what situation type this represents
  - Core Frame Elements (FEs): obligatory, definitionally central roles
  - Non-Core FEs: peripheral roles (Time, Place, Manner, Degree, ...)

---Existing Frames in Database (prefer reusing these)---
{frame_db_hints}

---Entities already extracted from this chunk---
{entity_list}

---Task---
Given the chunk text below:
1. Identify ALL events. An event MUST:
   - Have a clear TRIGGER WORD (verb, event-denoting noun, nominalization)
   - Involve at least ONE entity from the entity list above

2. For each event, determine its frame:
   - If trigger matches a Lexical Unit of an existing frame above → REUSE it (set is_new_frame=false)
   - Otherwise CREATE a new frame following FrameNet conventions (set is_new_frame=true)

3. Assign Frame Element roles to entities AND extract non-core FE values (time, price, location, etc.)

3b. IMPORTANT — also capture NEGATION and EXEMPTION events: if the text explicitly states that
    an entity did NOT participate in an event, was UNAFFECTED, was EXEMPT, or a situation did
    NOT occur (e.g. "X was unaffected by Y", "unlike X who did not...", "X was spared from..."),
    include this as a separate event entry with:
    - "trigger": the negating/exempting expression (e.g. "unaffected", "spared", "did not melt")
    - "is_negation": true   ← add this extra field
    - frame_name prefixed with "NOT_" (e.g. "NOT_Affect", "NOT_Transform")
    - core FE "Excluded_Entity" → the entity that is NOT affected
    - noncore FE "Reference_Event" → brief description of what they are excluded from

---Output Format---
Output ONLY a JSON array. Each element:
{{
  "trigger": "<exact trigger word from text>",
  "trigger_lemma": "<base form>",
  "trigger_pos": "VERB|NOUN|ADJ",
  "event_span": "<full clause/sentence>",
  "event_description": "<paraphrase: 'TRIGGER_LEMMA event where [participants] [key details]'>",
  "participant_entity_names": ["<name from entity list>", ...],

  "frame": {{
    "frame_name": "<PascalCase name, e.g. Commerce_buy>",
    "is_new_frame": true,
    "lexical_unit": "<trigger.POS, e.g. acquire.v>",
    "frame_definition": "<ONLY if is_new_frame=true: 1-2 sentence definition>",
    "core_fes": [                    // ONLY if is_new_frame=true
      {{"fe_name": "Buyer", "fe_definition": "...", "semantic_type": "Human"}},
      ...
    ],
    "noncore_fes": [                 // ONLY if is_new_frame=true
      {{"fe_name": "Time", "fe_definition": "...", "semantic_type": "Time"}},
      ...
    ]
  }},

  "role_assignments": {{
    "core": [
      {{"fe_name": "Buyer",  "filler_text": "Apple", "filler_type": "ENTITY", "is_missing": false}},
      {{"fe_name": "Seller", "filler_text": null,     "filler_type": "ENTITY", "is_missing": true}}
    ],
    "noncore": [
      {{"fe_name": "Price", "filler_text": "$3 billion", "filler_type": "VALUE", "is_missing": false}},
      {{"fe_name": "Time",  "filler_text": "2014",       "filler_type": "VALUE", "is_missing": false}}
    ]
  }}
}}

Output [] if no valid events found.

---Chunk Text---
{chunk_text}

---Output (JSON array)---"""

# ─────────────────────────────────────────────────────────────────────────────
# Call 3 — Causal / Temporal Edge Extraction (optional)
# ─────────────────────────────────────────────────────────────────────────────

PROMPTS["causal_temporal"] = """---Role---
You are an expert in event causality and temporal reasoning.

---Task---
Given the events below extracted from the same document section, identify CAUSAL and TEMPORAL
relations between them. Only output relations with explicit textual evidence.

Relation types:
  CAUSES   : Event A directly causes Event B
  PRECEDES : Event A happens before Event B (temporal, not necessarily causal)
  ENABLES  : Event A creates conditions that make Event B possible

---Events---
{event_list}

---Output Format---
Output ONLY a JSON array. Each element:
{{
  "source_event_id": "<event_id of A>",
  "relation_type": "CAUSES|PRECEDES|ENABLES",
  "target_event_id": "<event_id of B>",
  "confidence": 0.0-1.0,
  "evidence_span": "<exact text fragment that supports this relation>"
}}

Output [] if no clear relations exist.

---Output (JSON array)---"""

# ─────────────────────────────────────────────────────────────────────────────
# Call 4 — Query Processing (seed extraction)
# ─────────────────────────────────────────────────────────────────────────────

PROMPTS["query_processing"] = """---Role---
You are a query analysis expert for a Frame-Semantic knowledge graph retrieval system.

---Task---
Analyze the query and extract signals for graph-based retrieval.

---Output Format---
Output ONLY a JSON object:
{{
  "entity_hints": ["<entity name or description implied by query>", ...],
  "event_hints": ["<action/event type implied>", ...],
  "frame_hints": "<primary FrameNet-style frame name or description, e.g. Commerce_buy>",
  "fe_focus": ["<FE name most critical to answering the query>", ...],
  "temporal_hints": ["<year, date, or period if query specifies time>", ...]
}}

Examples:
  Query "Who acquired Beats Electronics?" →
    entity_hints: ["Beats Electronics"],
    event_hints: ["acquire", "buy", "purchase"],
    frame_hints: "Commerce_buy",
    fe_focus: ["Buyer"],
    temporal_hints: []

  Query "What did Apple buy in 2014?" →
    entity_hints: ["Apple"],
    event_hints: ["buy", "acquire"],
    frame_hints: "Commerce_buy",
    fe_focus: ["Goods"],
    temporal_hints: ["2014"]

---Query---
{query}

---Output (JSON object)---"""

# ─────────────────────────────────────────────────────────────────────────────
# Entity Coreference Verification (LLM borderline cases)
# ─────────────────────────────────────────────────────────────────────────────

PROMPTS["entity_coref_verify"] = """---Task---
Determine whether Entity A and Entity B refer to the same real-world entity.

---Entity A---
Name: {name_a}
Description: {desc_a}

---Entity B---
Name: {name_b}
Description: {desc_b}

---Output---
Answer SAME or DIFFERENT, followed by one concise reason.
Format: "<SAME|DIFFERENT> - <reason>"
Example: "SAME - both refer to Apple Inc., the technology company"
"""

# ─────────────────────────────────────────────────────────────────────────────
# Entity Extraction Gleaning (Call 1b — catch missed entities)
# ─────────────────────────────────────────────────────────────────────────────

PROMPTS["entity_extraction_glean"] = """---Task---
Review the text again for any entities you may have MISSED in your initial extraction.

---Already Extracted---
{existing_entities}

---Original Text---
{chunk_text}

---Instructions---
Output ONLY entities NOT already in the list above (new ones only).
Same JSON format as before. If none found, output [].

---Output (JSON array)---"""

# ─────────────────────────────────────────────────────────────────────────────
# Frame Relation Expansion (query-time — broaden retrieval scope)
# ─────────────────────────────────────────────────────────────────────────────

PROMPTS["frame_relation_expand"] = """---Task---
Given a query and its primary FrameNet-style frame, identify 3-5 RELATED frames that would
contain complementary information needed to fully answer the query.

Consider frames that:
  - Share participants with the primary frame
  - Precede or follow the primary frame temporally
  - Are causally linked (cause, enable, or result from the primary frame)

---Query---
{query}

---Primary Frame---
{primary_frame}

---Output---
JSON array of frame name strings only. Example:
["Commerce_buy", "Transfer", "Ownership"]

Output [] if no clearly related frames exist."""

# ─────────────────────────────────────────────────────────────────────────────
# Event Coreference Verification (LLM borderline cases)
# ─────────────────────────────────────────────────────────────────────────────

PROMPTS["event_coref_verify"] = """---Task---
Determine whether Event A and Event B describe the SAME real-world occurrence.

Two events corefer only if they:
  1. Are triggered by the same (or semantically equivalent) word
  2. Evoke the same frame
  3. Involve the same or highly overlapping participants in compatible roles

---Event A---
Trigger: {trigger_a}
Frame: {frame_a}
Description: {desc_a}
Participants: {participants_a}

---Event B---
Trigger: {trigger_b}
Frame: {frame_b}
Description: {desc_b}
Participants: {participants_b}

---Output---
Answer SAME or DIFFERENT, followed by one concise reason.
Format: "<SAME|DIFFERENT> - <reason>"
Example: "SAME - both describe Apple's acquisition of Beats Electronics in 2014"
"""

# ─────────────────────────────────────────────────────────────────────────────
# Answer Generation (Call 5)
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Entity Description Merge (cross-document accumulation)
# ─────────────────────────────────────────────────────────────────────────────

PROMPTS["entity_description_merge"] = """---Task---
Merge the following descriptions of the same entity into a single concise description.
Preserve all unique facts. Remove redundancies. Write in third person.
Keep the result to 3-5 sentences maximum.

---Entity Name---
{entity_name}

---Descriptions to Merge---
{descriptions}

---Merged Description---"""

# ─────────────────────────────────────────────────────────────────────────────
# Answer Generation (Call 5)
# ─────────────────────────────────────────────────────────────────────────────

PROMPTS["answer_generation"] = """---Role---
You are a precise question-answering assistant. Answer using ONLY the retrieved context below.

---Structured Facts (Frame Instances)---
{structured_facts}

---Supporting Text Passages---
{text_passages}

---Question---
{query}

---Instructions---
1. First, identify which frame instances and passages are most directly relevant to the question.
2. If the question asks about multiple entities, groups, or time periods, address each one explicitly.
3. Pay special attention to NOT_* frames — these encode what did NOT happen or who was NOT affected.
4. If a key fact is absent from the retrieved context, state "Not found in retrieved context" rather
   than guessing.
5. Structure your answer:
   [Direct answer to the question]
   [Key supporting evidence from facts/passages — be specific]
   [Any important exceptions or caveats, especially from NOT_* frames]
6. Keep the answer focused and no longer than needed. Avoid re-stating the question.

---Answer---"""
