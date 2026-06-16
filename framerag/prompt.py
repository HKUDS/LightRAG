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
# Call 2 (Step A) — Event Detection ONLY (no frames)
# ─────────────────────────────────────────────────────────────────────────────

PROMPTS["event_extraction"] = """---Role---
You are an expert in event extraction from narrative text.

---Entities already extracted from this chunk---
{entity_list}

---Task---
Read the chunk text and identify ALL events. An event is something that HAPPENS or a
state that holds — an action, occurrence, change, perception, communication, or mental
event. Each event MUST have a clear trigger word (a verb, an event-denoting noun, or a
nominalization such as "arrival", "discovery", "murder").

For EACH event output a JSON object with:
  - event_span        : the exact trigger phrase / clause from the text
  - description       : 1-2 sentence plain-language description of what happened
  - participant_names : list of entity names (from the entity list above) involved in the event
  - temporal_marker   : optional time/ordering phrase ("the next morning", "in 1887", "after dinner") or "" if none
  - is_negation       : true if the text says the event did NOT happen / an entity was NOT
                        affected / was exempt / spared / unaffected; otherwise false

Rules:
- Capture NEGATION/EXEMPTION events too (e.g. "X was unaffected", "Y did not arrive",
  "Z was spared"). Set is_negation=true for these.
- For participant_names, use the EXACT surface form from the entity list above whenever
  the participant appears there (this lets the system link the event to the right entity
  node). If a clearly-named participant is missing from the list, include its name anyway —
  but match an existing entity name verbatim if at all possible.
- Resolve pronouns (he/she/they/it) to the named entity they refer to, using the entity
  list and surrounding context — do NOT output bare pronouns as participants.
- Output ONLY a JSON array. No prose, no markdown fences.
- If no events found, output [].

---Chunk Text---
{chunk_text}

---Output (JSON array)---"""

# ─────────────────────────────────────────────────────────────────────────────
# Call 2 (Step B) — Frame Annotation for ONE event
# ─────────────────────────────────────────────────────────────────────────────

PROMPTS["frame_annotation"] = """---Role---
You are an expert in Frame Semantics (Fillmore 1985).

---Background---
A "frame" is a schematic representation of a situation type. Each frame has a name
(FrameNet-style PascalCase_Underscore, e.g. "Commerce_buy", "Reveal_secret",
"Intentional_act"), a definition, and Frame Elements (FEs) — the participant/circumstance
roles. Core FEs are definitionally central (e.g. Buyer, Goods); non-core FEs are peripheral
(Time, Place, Manner, Purpose, Degree).

---Existing Frames in Database (PREFER reusing one of these if it fits)---
{frame_db_hints}

---Entities available in this chunk---
{entity_list}

---The Event to Annotate---
Trigger / span : {event_span}
Description    : {event_description}
Participants   : {participant_names}
Negation       : {is_negation}

---Source Context (excerpt around event)---
{chunk_text}

---Task---
1. Choose the single best-matching frame for this event.
   - If one of the existing database frames fits, REUSE its exact name (set is_new_frame=false).
   - Otherwise create a new FrameNet-style frame name (set is_new_frame=true) and give a
     1-sentence definition.
   - If the event is a negation/exemption, prefix the frame name with "NOT_"
     (e.g. "NOT_Affect") and use core FE "Excluded_Entity".
2. Identify the CORE Frame Elements and which entity name or text phrase fills each.
3. Identify any NON-CORE FEs that are present (Time, Place, Manner, Purpose, ...).
4. For each FE filler, set filler_type to "ENTITY" if it is one of the available entities,
   else "VALUE" (dates, prices, locations-as-values, manner phrases). If a core FE has no
   filler in the text, set is_missing=true.

---Output Format---
Output ONLY a JSON object (no markdown fences):
{{
  "frame_name": "<PascalCase_Underscore>",
  "is_new_frame": true,
  "frame_definition": "<1 sentence; required when is_new_frame=true>",
  "lexical_unit": "<trigger.POS, e.g. reveal.v>",
  "core_elements": [
    {{"fe_name": "Speaker", "filler_text": "Holmes", "filler_type": "ENTITY", "is_missing": false}},
    {{"fe_name": "Topic",   "filler_text": "the cipher", "filler_type": "VALUE", "is_missing": false}}
  ],
  "noncore_elements": [
    {{"fe_name": "Time", "filler_text": "that evening", "filler_type": "VALUE", "is_missing": false}}
  ]
}}

---Output (JSON object)---"""

# ─────────────────────────────────────────────────────────────────────────────
# Call 3 — Causal / Temporal Edge Extraction (optional)
# ─────────────────────────────────────────────────────────────────────────────

PROMPTS["causal_temporal"] = """---Role---
You are an expert in event causality and temporal reasoning.

---Task---
Below is a passage of text followed by the events extracted from it (with stable IDs).
Identify CAUSAL and TEMPORAL relations between these events, grounded in the passage.

Relation types:
  CAUSES   : Event A directly causes Event B
  PRECEDES : Event A happens before Event B (temporal ordering, not necessarily causal)
  ENABLES  : Event A creates conditions that make Event B possible

Guidance:
- Use the passage to judge ordering and causality. Narrative order, connectives
  ("because", "so", "after", "then", "as a result", "which led to"), and plain
  chronological sequence are all valid evidence.
- It is normal for a passage to contain several relations — when two events are described
  in sequence, a PRECEDES relation almost always holds. Be willing to assert PRECEDES for
  clearly ordered events even without an explicit connective.
- Copy the event_id values EXACTLY as given (do not invent or shorten them).

---Passage---
{chunk_text}

---Events (use these exact IDs)---
{event_list}

---Output Format---
Output ONLY a JSON array (no markdown fences). Each element:
{{
  "source_event_id": "<exact event_id of A>",
  "relation_type": "CAUSES|PRECEDES|ENABLES",
  "target_event_id": "<exact event_id of B>",
  "confidence": 0.0-1.0,
  "evidence_span": "<short text fragment or 'narrative sequence' if ordering-based>"
}}

Output [] only if the events are genuinely unrelated.

---Output (JSON array)---"""

# ─────────────────────────────────────────────────────────────────────────────
# Call 4 — Query Processing (seed extraction)
# ─────────────────────────────────────────────────────────────────────────────

PROMPTS["query_processing"] = """---Role---
You are a query analysis expert for a Frame-Semantic knowledge graph retrieval system.
The queries are about narrative fiction (detective stories, novels) — character actions,
motivations, relationships, and events.

---Task---
Analyze the query and extract signals for graph-based retrieval.

---Output Format---
Output ONLY a JSON object:
{{
  "entity_hints": ["<character names or entity names implied by query>", ...],
  "event_hints": ["<action/event type implied — use verbs>", ...],
  "frame_hints": "<primary FrameNet-style frame name, e.g. Scrutiny, Communication_tell, Motion_travel>",
  "fe_focus": ["<FE name most critical to answering the query>", ...],
  "temporal_hints": ["<time expression if query specifies time, else empty>", ...]
}}

Examples (narrative fiction):
  Query "What did Holmes discover about the suspect?" →
    entity_hints: ["Holmes", "suspect"],
    event_hints: ["discover", "find", "investigate"],
    frame_hints: "Scrutiny",
    fe_focus: ["Ground", "Phenomenon"],
    temporal_hints: []

  Query "Why did Watson accompany Holmes to Baker Street?" →
    entity_hints: ["Watson", "Holmes", "Baker Street"],
    event_hints: ["accompany", "travel", "go"],
    frame_hints: "Accompaniment",
    fe_focus: ["Co-participant", "Purpose"],
    temporal_hints: []

  Query "Who did Irene Adler outwit in the story?" →
    entity_hints: ["Irene Adler"],
    event_hints: ["outwit", "deceive", "trick"],
    frame_hints: "Deception",
    fe_focus: ["Victim"],
    temporal_hints: []

  Query "How did the murderer escape after the crime?" →
    entity_hints: ["murderer"],
    event_hints: ["escape", "flee", "run"],
    frame_hints: "Escaping",
    fe_focus: ["Agent", "Source"],
    temporal_hints: []

---Query---
{query}

---Output (JSON object)---"""

# ─────────────────────────────────────────────────────────────────────────────
# Call 4b — LLM Frame Selection (query-time, uses actual frame DB)
# ─────────────────────────────────────────────────────────────────────────────

PROMPTS["frame_selection"] = """---Role---
You are a semantic frame retrieval expert for narrative text question answering.

---Task---
Given a query about a narrative, select the most relevant semantic frames from the list below.
These frames describe event/action types that occurred in the story.

---Available Frames---
{frame_list}

---Query---
{query}

---Instructions---
Select up to 6 frame names that describe events or situations most directly relevant to answering
the query. Focus on the ACTION or EVENT type the query is asking about, not just keywords.

Output ONLY a JSON array of frame names from the list above.
Example: ["Cognition_find", "Communication_tell", "Motion_flee"]

---Output (JSON array)---"""

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
You are a precise question-answering assistant for narrative text. Answer using the retrieved context below.

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
4. Always provide the best answer you can from the available context:
   - If the context fully supports the answer: answer directly and cite evidence.
   - If the context partially supports the answer: answer with what you know, note what is uncertain.
   - Only say "Not found in retrieved context" if the context contains absolutely no relevant
     information whatsoever — not even indirect or partial evidence.
5. Structure your answer:
   [Direct answer to the question]
   [Key supporting evidence from facts/passages — be specific]
   [Any important exceptions or caveats, especially from NOT_* frames]
6. Keep the answer focused and no longer than needed. Avoid re-stating the question.

---Answer---"""
