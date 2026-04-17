from __future__ import annotations
from typing import Any


PROMPTS: dict[str, Any] = {}

# All delimiters must be formatted as "<|UPPER_CASE_STRING|>"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|#|>"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

# Default entity type guidance injected into extraction prompts via {entity_types_guidance}.
# Users can override this by passing entity_types_guidance in addon_params, or by
# replacing the full prompt template string in PROMPTS.
PROMPTS[
    "default_entity_types_guidance"
] = """Classify each entity using one of the following types. If no type fits, use `Other`.

- Person: Human individuals, real or fictional
- Creature: Non-human living beings (animals, mythical beings, etc.)
- Organization: Companies, institutions, government bodies, groups
- Location: Geographic places (cities, countries, buildings, regions)
- Event: Occurrences, incidents, ceremonies, meetings
- Concept: Abstract ideas, theories, principles, beliefs
- Method: Procedures, techniques, algorithms, workflows
- Content: Creative or informational works (books, articles, films, reports)
- Data: Quantitative or structured information (statistics, datasets, measurements)
- Artifact: Physical or digital objects created by humans (tools, software, devices)
- NaturalObject: Natural non-living objects (minerals, celestial bodies, chemical compounds)"""

PROMPTS["entity_extraction_system_prompt"] = """---Role---
You are a Knowledge Graph Specialist responsible for extracting entities and relationships from the `---Input Text---` section of user prompt.

---Instructions---
1. **Entity Extraction & Output:**
  -  **Identification:** Identify clearly defined and meaningful entities in the in the `---Input Text---` section of user prompt.
  -  **Entity Details:** For each identified entity, extract the following information:
    - `entity_name`: The name of the entity. If the entity name is case-insensitive, capitalize the first letter of each significant word (title case). Ensure **consistent naming** across the entire extraction process.
    - `entity_type`: Categorize the entity using the type guidance provided in the `---Entity Types---` section below. If none of the provided entity types apply, classify it as `Other`.
    - `entity_description`: Provide a concise yet comprehensive description of the entity's attributes and activities, based *solely* on the information present in the input text.

2. **Relationship Extraction & Output:**
  - **Identification:** Identify direct, clearly stated, and meaningful relationships between previously extracted entities.
  - **N-ary Relationship Decomposition:** If a single statement describes a relationship involving more than two entities (an N-ary relationship), decompose it into multiple binary (two-entity) relationship pairs for separate description.
    - **Example:** For "Alice, Bob, and Carol collaborated on Project X," extract binary relationships such as "Alice collaborated with Project X," "Bob collaborated with Project X," and "Carol collaborated with Project X," or "Alice collaborated with Bob," based on the most reasonable binary interpretations.
  - **Relationship Details:** For each binary relationship, extract the following fields:
    - `source_entity`: The name of the source entity. Ensure **consistent naming** with entity extraction. Capitalize the first letter of each significant word (title case) if the name is case-insensitive.
    - `target_entity`: The name of the target entity. Ensure **consistent naming** with entity extraction. Capitalize the first letter of each significant word (title case) if the name is case-insensitive.
    - `relationship_keywords`: One or more high-level keywords summarizing the overarching nature, concepts, or themes of the relationship. Multiple keywords within this field must be separated by a comma `,`. **DO NOT use `{tuple_delimiter}` for separating multiple keywords within this field.**
    - `relationship_description`: A concise explanation of the nature of the relationship between the source and target entities, providing a clear rationale for their connection.

3. **Output Format:**
  - **Entity:**
     - Output exactly 3 fields for each entity on a single line, prefixed with the literal string "entity" and delimited by {tuple_delimiter}.
     - Format: `entity{tuple_delimiter}entity_name{tuple_delimiter}entity_type{tuple_delimiter}entity_description`
  - **Relationship:**
    - Output exactly 4 fields for each relationship on a single line, prefixed with the literal string "relation" and delimited by {tuple_delimiter}.
    - Format: `relation{tuple_delimiter}source_entity{tuple_delimiter}target_entity{tuple_delimiter}relationship_keywords{tuple_delimiter}relationship_description`

4. **Delimiter Usage Protocol:**
  - The `{tuple_delimiter}` is a complete, atomic marker and **must not be filled with content**. It serves strictly as a field separator.
  - **Incorrect Example:** `entity{tuple_delimiter}Tokyo<|location|>Tokyo is the capital of Japan.`
  - **Correct Example:** `entity{tuple_delimiter}Tokyo{tuple_delimiter}location{tuple_delimiter}Tokyo is the capital of Japan.`

5. **Relationship Direction & Duplication:**
  - Treat all relationships as **undirected** unless explicitly stated otherwise. Swapping the source and target entities for an undirected relationship does not constitute a new relationship.
  - Avoid outputting duplicate relationships.

6. **Output Order & Prioritization:**
   - Output all extracted entities first, followed by all extracted relationships.
   - Within the list of relationships, prioritize and output those relationships that are **most significant** to the core meaning of the input text first.

7. **Context & Objectivity:**
  - Ensure all entity names and descriptions are written in the **third person**.
  - Explicitly name the subject or object; **avoid using pronouns** such as `this article`, `this paper`, `our company`, `I`, `you`, and `he/she`.

8. **Language & Proper Nouns:**
  - The entire output (entity names, keywords, and descriptions) must be written in `{language}`.
  - Proper nouns (e.g., personal names, place names, organization names) should be retained in their original language if a proper, widely accepted translation is not available or would cause ambiguity.

9. **Completion Signal:** Output the literal string `{completion_delimiter}` only after all entities and relationships, following all criteria, have been completely extracted and outputted.

---Entity Types---
{entity_types_guidance}

---Examples---
{examples}
"""

PROMPTS["entity_extraction_user_prompt"] = """---Task---
Extract entities and relationships from the `---Input Text---` session below.

---Instructions---
1. **Strict Adherence to Format:** Strictly adhere to all format requirements for entity and relationship lists, including output order, field delimiters, and proper noun handling, as specified in the system prompt.
2. **Output Content Only:** Output *only* the extracted list of entities and relationships. Do not include any introductory or concluding remarks, explanations, or additional text before or after the list.
3. **Completion Signal:** Output `{completion_delimiter}` as the final line after all relevant entities and relationships have been extracted and presented.
4. **Output Language:** Ensure the output language is {language}. Proper nouns (e.g., personal names, place names, organization names) must be kept in their original language and not translated.

---Input Text---
```
{input_text}
```

---Output---
"""

PROMPTS["entity_continue_extraction_user_prompt"] = """---Task---
Based on the last extraction task, identify and extract any missed or incorrectly formatted entities and relationships from the input text.

---Instructions---
1. **Strict Adherence to System Format:** Strictly adhere to all format requirements for entity and relationship lists, including output order, field delimiters, and proper noun handling, as specified in the system instructions.
2. **Focus on Corrections/Additions:**
  - **Do NOT** re-output entities and relationships that were **correctly and fully** extracted in the last task.
  - If an entity or relationship was **missed** in the last task, extract and output it now according to the system format.
  - If an entity or relationship was **truncated, had missing fields, or was otherwise incorrectly formatted** in the last task, re-output the *corrected and complete* version in the specified format.
3. **Output Content Only:** Output *only* the extracted list of entities and relationships. Do not include any introductory or concluding remarks, explanations, or additional text before or after the list.
4. **Completion Signal:** Output `{completion_delimiter}` as the final line after all relevant missing or corrected entities and relationships have been extracted and presented.
5. **Output Language:** Ensure the output language is {language}. Proper nouns (e.g., personal names, place names, organization names) must be kept in their original language and not translated.

---Output---
"""

PROMPTS["entity_extraction_examples"] = [
    """---Entity Types---
- Person: Human individuals, real or fictional
- Artifact: Physical or digital objects created by humans (tools, software, devices)
- Concept: Abstract ideas, theories, principles, beliefs

---Input Text---
```
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.

Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. "If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us."

The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.

It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths
```

---Output---
entity{tuple_delimiter}Alex{tuple_delimiter}Person{tuple_delimiter}Alex is a character who experiences frustration and is observant of the dynamics among other characters.
entity{tuple_delimiter}Taylor{tuple_delimiter}Person{tuple_delimiter}Taylor is portrayed with authoritarian certainty and shows a moment of reverence towards a device, indicating a change in perspective.
entity{tuple_delimiter}Jordan{tuple_delimiter}Person{tuple_delimiter}Jordan shares a commitment to discovery and has a significant interaction with Taylor regarding a device.
entity{tuple_delimiter}Cruz{tuple_delimiter}Person{tuple_delimiter}Cruz is associated with a vision of control and order, influencing the dynamics among other characters.
entity{tuple_delimiter}The Device{tuple_delimiter}Artifact{tuple_delimiter}The Device is central to the story, with potential game-changing implications, and is revered by Taylor.
entity{tuple_delimiter}Discovery{tuple_delimiter}Concept{tuple_delimiter}Discovery represents the shared intellectual pursuit that unites Jordan and Alex in opposition to Cruz's controlling worldview.
relation{tuple_delimiter}Alex{tuple_delimiter}Taylor{tuple_delimiter}power dynamics, observation{tuple_delimiter}Alex observes Taylor's authoritarian behavior and notes changes in Taylor's attitude toward the device.
relation{tuple_delimiter}Alex{tuple_delimiter}Jordan{tuple_delimiter}shared goals, rebellion{tuple_delimiter}Alex and Jordan share a commitment to discovery, which contrasts with Cruz's vision.)
relation{tuple_delimiter}Taylor{tuple_delimiter}Jordan{tuple_delimiter}conflict resolution, mutual respect{tuple_delimiter}Taylor and Jordan interact directly regarding the device, leading to a moment of mutual respect and an uneasy truce.
relation{tuple_delimiter}Jordan{tuple_delimiter}Cruz{tuple_delimiter}ideological conflict, rebellion{tuple_delimiter}Jordan's commitment to discovery is in rebellion against Cruz's vision of control and order.
relation{tuple_delimiter}Taylor{tuple_delimiter}The Device{tuple_delimiter}reverence, technological significance{tuple_delimiter}Taylor shows reverence towards the device, indicating its importance and potential impact.
{completion_delimiter}

""",
    """---Entity Types---
- Person: Human individuals, real or fictional
- Location: Geographic places (cities, countries, buildings, regions)
- Creature: Non-human living beings (animals, mythical beings, etc.)
- Method: Procedures, techniques, algorithms, workflows
- Organization: Companies, institutions, government bodies, groups
- Content: Creative or informational works (books, articles, films, reports)
- NaturalObject: Natural non-living objects (minerals, celestial bodies, chemical compounds)

---Input Text---
```
Dr. Elena Vasquez led a field expedition to the Borneo rainforest to document the population decline of the Bornean orangutan. Using transect sampling — a method where researchers walk predetermined line paths and record every animal sighting within a fixed distance — her team estimated that fewer than 1,500 individuals remained in the surveyed region.

The expedition was funded by the Global Wildlife Conservation Institute and produced a landmark report titled "Primate Decline in Insular Southeast Asia." Vasquez attributed the collapse primarily to peat-soil destruction caused by palm oil plantation expansion, which had converted over 40% of the surveyed forest area within a decade.
```

---Output---
entity{tuple_delimiter}Dr. Elena Vasquez{tuple_delimiter}Person{tuple_delimiter}Dr. Elena Vasquez is a field researcher who led an expedition to document orangutan population decline in Borneo.
entity{tuple_delimiter}Borneo Rainforest{tuple_delimiter}Location{tuple_delimiter}The Borneo rainforest is the field site of the expedition and the primary habitat of the Bornean orangutan.
entity{tuple_delimiter}Bornean Orangutan{tuple_delimiter}Creature{tuple_delimiter}The Bornean orangutan is a primate species whose population was found to have declined to fewer than 1,500 individuals in the surveyed region.
entity{tuple_delimiter}Transect Sampling{tuple_delimiter}Method{tuple_delimiter}Transect sampling is a wildlife survey technique where researchers walk predetermined paths and record animal sightings within a fixed lateral distance.
entity{tuple_delimiter}Global Wildlife Conservation Institute{tuple_delimiter}Organization{tuple_delimiter}The Global Wildlife Conservation Institute funded the expedition led by Dr. Vasquez.
entity{tuple_delimiter}Primate Decline in Insular Southeast Asia{tuple_delimiter}Content{tuple_delimiter}A landmark research report produced by Vasquez's expedition documenting primate population decline in the region.
entity{tuple_delimiter}Peat Soil{tuple_delimiter}NaturalObject{tuple_delimiter}Peat soil is a natural substrate in the Borneo rainforest that has been destroyed by palm oil plantation expansion.
relation{tuple_delimiter}Dr. Elena Vasquez{tuple_delimiter}Bornean Orangutan{tuple_delimiter}field research, population survey{tuple_delimiter}Dr. Vasquez led the expedition that documented the population decline of the Bornean orangutan.
relation{tuple_delimiter}Dr. Elena Vasquez{tuple_delimiter}Transect Sampling{tuple_delimiter}methodology, research application{tuple_delimiter}Dr. Vasquez's team used transect sampling to estimate the orangutan population.
relation{tuple_delimiter}Global Wildlife Conservation Institute{tuple_delimiter}Dr. Elena Vasquez{tuple_delimiter}funding, research support{tuple_delimiter}The institute funded the expedition led by Dr. Vasquez.
relation{tuple_delimiter}Dr. Elena Vasquez{tuple_delimiter}Primate Decline in Insular Southeast Asia{tuple_delimiter}authorship, research output{tuple_delimiter}Dr. Vasquez's expedition produced the landmark report on primate decline.
relation{tuple_delimiter}Peat Soil{tuple_delimiter}Borneo Rainforest{tuple_delimiter}habitat composition, ecological destruction{tuple_delimiter}Peat soil destruction in the Borneo rainforest was caused by palm oil plantation expansion and is a primary driver of orangutan decline.
{completion_delimiter}

""",
    """---Entity Types---
- Content: Creative or informational works (books, articles, films, reports)
- Artifact: Physical or digital objects created by humans (tools, software, devices)
- Person: Human individuals, real or fictional
- Organization: Companies, institutions, government bodies, groups
- Method: Procedures, techniques, algorithms, workflows
- Data: Quantitative or structured information (statistics, datasets, measurements)
- Concept: Abstract ideas, theories, principles, beliefs

---Input Text---
```
The 2023 edition of "Advances in Neural Architecture Search" synthesized findings from over 200 peer-reviewed papers and introduced a new benchmarking framework called NASBench-360, designed to evaluate search algorithms across diverse task domains. The publication was co-authored by Dr. Priya Nair and Dr. Luca Ferretti of the DeepSystems Research Lab.

NASBench-360 measures three key metrics: search efficiency (time-to-solution), model accuracy on held-out test sets, and computational cost in GPU-hours. Early results showed that evolutionary search algorithms outperformed gradient-based methods by 12% on accuracy while consuming 30% fewer GPU-hours on vision tasks.
```

---Output---
entity{tuple_delimiter}Advances in Neural Architecture Search{tuple_delimiter}Content{tuple_delimiter}A 2023 publication that synthesizes findings from over 200 papers and introduces the NASBench-360 benchmarking framework.
entity{tuple_delimiter}NASBench-360{tuple_delimiter}Artifact{tuple_delimiter}NASBench-360 is a benchmarking framework introduced to evaluate neural architecture search algorithms across diverse task domains.
entity{tuple_delimiter}Dr. Priya Nair{tuple_delimiter}Person{tuple_delimiter}Dr. Priya Nair is a co-author of the publication and a researcher at the DeepSystems Research Lab.
entity{tuple_delimiter}Dr. Luca Ferretti{tuple_delimiter}Person{tuple_delimiter}Dr. Luca Ferretti is a co-author of the publication and a researcher at the DeepSystems Research Lab.
entity{tuple_delimiter}DeepSystems Research Lab{tuple_delimiter}Organization{tuple_delimiter}The DeepSystems Research Lab is the institution where the co-authors of the publication are affiliated.
entity{tuple_delimiter}Evolutionary Search{tuple_delimiter}Method{tuple_delimiter}Evolutionary search is a class of neural architecture search algorithms that outperformed gradient-based methods in the NASBench-360 evaluation.
entity{tuple_delimiter}Gradient-Based Search{tuple_delimiter}Method{tuple_delimiter}Gradient-based search is a class of neural architecture search algorithms that was benchmarked against evolutionary search in NASBench-360.
entity{tuple_delimiter}GPU-Hours{tuple_delimiter}Data{tuple_delimiter}GPU-hours is a metric used in NASBench-360 to measure the computational cost of neural architecture search algorithms.
entity{tuple_delimiter}Neural Architecture Search{tuple_delimiter}Concept{tuple_delimiter}Neural architecture search is the automated process of designing optimal neural network architectures, the central topic of the publication.
relation{tuple_delimiter}Dr. Priya Nair{tuple_delimiter}Advances in Neural Architecture Search{tuple_delimiter}authorship{tuple_delimiter}Dr. Priya Nair co-authored the publication.
relation{tuple_delimiter}Dr. Luca Ferretti{tuple_delimiter}Advances in Neural Architecture Search{tuple_delimiter}authorship{tuple_delimiter}Dr. Luca Ferretti co-authored the publication.
relation{tuple_delimiter}Advances in Neural Architecture Search{tuple_delimiter}NASBench-360{tuple_delimiter}introduces, benchmarking{tuple_delimiter}The publication introduced the NASBench-360 framework.
relation{tuple_delimiter}Evolutionary Search{tuple_delimiter}Gradient-Based Search{tuple_delimiter}performance comparison{tuple_delimiter}Evolutionary search outperformed gradient-based methods by 12% on accuracy and used 30% fewer GPU-hours on vision tasks.
relation{tuple_delimiter}NASBench-360{tuple_delimiter}GPU-Hours{tuple_delimiter}evaluation metric{tuple_delimiter}NASBench-360 uses GPU-hours as one of three key metrics to measure computational cost.
{completion_delimiter}

""",
]

###############################################################################
# JSON Structured Output Prompts for Entity Extraction
# Used when entity_extraction_use_json is enabled for higher extraction quality
###############################################################################

PROMPTS["entity_extraction_json_system_prompt"] = """---Role---
You are a Knowledge Graph Specialist responsible for extracting entities and relationships from the `---Input Text---` session of user prompt.

---Instructions---
1. **Entity Extraction:**
  - **Identification:** Identify clearly defined and meaningful entities in the `---Input Text---` session of user prompt.
  - **Entity Details:** For each identified entity, extract the following information:
    - `name`: The name of the entity. If the entity name is case-insensitive, capitalize the first letter of each significant word (title case). Ensure **consistent naming** across the entire extraction process.
    - `type`: Categorize the entity using the type guidance provided in the `---Entity Types---` section below. If none of the provided entity types apply, classify it as `Other`.
    - `description`: Provide a concise yet comprehensive description of the entity's attributes and activities, based *solely* on the information present in the input text.

2. **Relationship Extraction:**
  - **Identification:** Identify direct, clearly stated, and meaningful relationships between previously extracted entities.
  - **N-ary Relationship Decomposition:** If a single statement describes a relationship involving more than two entities (an N-ary relationship), decompose it into multiple binary (two-entity) relationship pairs for separate description.
    - Example: For "Alice, Bob, and Carol collaborated on Project X," extract binary relationships such as "Alice collaborated with Project X," "Bob collaborated with Project X," and "Carol collaborated with Project X," or "Alice collaborated with Bob," based on the most reasonable binary interpretations.
  - **Relationship Details:** For each binary relationship, extract the following fields:
    - `source`: The name of the source entity. Ensure **consistent naming** with entity extraction. Capitalize the first letter of each significant word (title case) if the name is case-insensitive.
    - `target`: The name of the target entity. Ensure **consistent naming** with entity extraction. Capitalize the first letter of each significant word (title case) if the name is case-insensitive.
    - `keywords`: One or more high-level keywords summarizing the overarching nature, concepts, or themes of the relationship, separated by commas.
    - `description`: A concise explanation of the nature of the relationship between the source and target entities, providing a clear rationale for their connection.

3. **Relationship Direction & Duplication:**
  - Treat all relationships as **undirected** unless explicitly stated otherwise. Swapping the source and target entities for an undirected relationship does not constitute a new relationship.
  - Avoid outputting duplicate relationships.

4. **Prioritization:**
  - Within the list of relationships, prioritize and output those relationships that are **most significant** to the core meaning of the input text first.

5. **Context & Objectivity:**
  - Ensure all entity names and descriptions are written in the **third person**.
  - Explicitly name the subject or object; **avoid using pronouns** such as `this article`, `this paper`, `our company`, `I`, `you`, and `he/she`.

6. **Language & Proper Nouns:**
  - The entire output (entity names, keywords, and descriptions) must be written in `{language}`.
  - Proper nouns (e.g., personal names, place names, organization names) should be retained in their original language if a proper, widely accepted translation is not available or would cause ambiguity.

---Entity Types---
{entity_types_guidance}

---Examples---
{examples}
"""

PROMPTS["entity_extraction_json_user_prompt"] = """---Task---
Extract entities and relationships from the `---Input Text---` session below.

---Instructions---
1. **Strict Adherence to JSON Format:** Your output MUST be a valid JSON object with `entities` and `relationships` arrays. Do not include any introductory or concluding remarks, explanations, markdown code fences, or any other text before or after the JSON.
2. **Output Language:** Ensure the output language is {language}. Proper nouns (e.g., personal names, place names, organization names) must be kept in their original language and not translated.

---Entity Types---
{entity_types_guidance}

---Input Text---
```
{input_text}
```

---Output---
"""

PROMPTS["entity_continue_extraction_json_user_prompt"] = """---Task---
Based on the last extraction task, identify and extract any **missed or incorrectly described** entities and relationships from the `---Input Text---` session.

---Instructions---
1. **Focus on Corrections/Additions:**
  - **Do NOT** re-output entities and relationships that were **correctly and fully** extracted in the last task.
  - If an entity or relationship was **missed** in the last task, extract and output it now.
  - If an entity or relationship was **incorrectly described** in the last task, re-output the *corrected and complete* version.
2. **Strict Adherence to JSON Format:** Your output MUST be a valid JSON object with `entities` and `relationships` arrays. Do not include any introductory or concluding remarks, explanations, markdown code fences, or any other text before or after the JSON.
3. **Output Language:** Ensure the output language is {language}. Proper nouns (e.g., personal names, place names, organization names) must be kept in their original language and not translated.
4. **If nothing was missed or needs correction**, output: `{{"entities": [], "relationships": []}}`

---Output---
"""

PROMPTS["entity_extraction_json_examples"] = [
    """---Entity Types---
- Person: Human individuals, real or fictional
- Artifact: Physical or digital objects created by humans (tools, software, devices)
- Concept: Abstract ideas, theories, principles, beliefs

---Input Text---
```
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.

Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. "If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us."

The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.

It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths
```

---Output---
{
  "entities": [
    {"name": "Alex", "type": "Person", "description": "Alex is a character who experiences frustration and is observant of the dynamics among other characters."},
    {"name": "Taylor", "type": "Person", "description": "Taylor is portrayed with authoritarian certainty and shows a moment of reverence towards a device, indicating a change in perspective."},
    {"name": "Jordan", "type": "Person", "description": "Jordan shares a commitment to discovery and has a significant interaction with Taylor regarding a device."},
    {"name": "Cruz", "type": "Person", "description": "Cruz is associated with a vision of control and order, influencing the dynamics among other characters."},
    {"name": "The Device", "type": "Artifact", "description": "The Device is central to the story, with potential game-changing implications, and is revered by Taylor."},
    {"name": "Discovery", "type": "Concept", "description": "Discovery represents the shared intellectual pursuit that unites Jordan and Alex in opposition to Cruz's controlling worldview."}
  ],
  "relationships": [
    {"source": "Alex", "target": "Taylor", "keywords": "power dynamics, observation", "description": "Alex observes Taylor's authoritarian behavior and notes changes in Taylor's attitude toward the device."},
    {"source": "Alex", "target": "Jordan", "keywords": "shared goals, rebellion", "description": "Alex and Jordan share a commitment to discovery, which contrasts with Cruz's vision."},
    {"source": "Taylor", "target": "Jordan", "keywords": "conflict resolution, mutual respect", "description": "Taylor and Jordan interact directly regarding the device, leading to a moment of mutual respect and an uneasy truce."},
    {"source": "Jordan", "target": "Cruz", "keywords": "ideological conflict, rebellion", "description": "Jordan's commitment to discovery is in rebellion against Cruz's vision of control and order."},
    {"source": "Taylor", "target": "The Device", "keywords": "reverence, technological significance", "description": "Taylor shows reverence towards the device, indicating its importance and potential impact."}
  ]
}

""",
    """---Entity Types---
- Person: Human individuals, real or fictional
- Location: Geographic places (cities, countries, buildings, regions)
- Creature: Non-human living beings (animals, mythical beings, etc.)
- Method: Procedures, techniques, algorithms, workflows
- Organization: Companies, institutions, government bodies, groups
- Content: Creative or informational works (books, articles, films, reports)
- NaturalObject: Natural non-living objects (minerals, celestial bodies, chemical compounds)

---Input Text---
```
Dr. Elena Vasquez led a field expedition to the Borneo rainforest to document the population decline of the Bornean orangutan. Using transect sampling — a method where researchers walk predetermined line paths and record every animal sighting within a fixed distance — her team estimated that fewer than 1,500 individuals remained in the surveyed region.

The expedition was funded by the Global Wildlife Conservation Institute and produced a landmark report titled "Primate Decline in Insular Southeast Asia." Vasquez attributed the collapse primarily to peat-soil destruction caused by palm oil plantation expansion, which had converted over 40% of the surveyed forest area within a decade.
```

---Output---
{
  "entities": [
    {"name": "Dr. Elena Vasquez", "type": "Person", "description": "Dr. Elena Vasquez is a field researcher who led an expedition to document orangutan population decline in Borneo."},
    {"name": "Borneo Rainforest", "type": "Location", "description": "The Borneo rainforest is the field site of the expedition and the primary habitat of the Bornean orangutan."},
    {"name": "Bornean Orangutan", "type": "Creature", "description": "The Bornean orangutan is a primate species whose population was found to have declined to fewer than 1,500 individuals in the surveyed region."},
    {"name": "Transect Sampling", "type": "Method", "description": "Transect sampling is a wildlife survey technique where researchers walk predetermined paths and record animal sightings within a fixed lateral distance."},
    {"name": "Global Wildlife Conservation Institute", "type": "Organization", "description": "The Global Wildlife Conservation Institute funded the expedition led by Dr. Vasquez."},
    {"name": "Primate Decline in Insular Southeast Asia", "type": "Content", "description": "A landmark research report produced by Vasquez's expedition documenting primate population decline in the region."},
    {"name": "Peat Soil", "type": "NaturalObject", "description": "Peat soil is a natural substrate in the Borneo rainforest that has been destroyed by palm oil plantation expansion."}
  ],
  "relationships": [
    {"source": "Dr. Elena Vasquez", "target": "Bornean Orangutan", "keywords": "field research, population survey", "description": "Dr. Vasquez led the expedition that documented the population decline of the Bornean orangutan."},
    {"source": "Dr. Elena Vasquez", "target": "Transect Sampling", "keywords": "methodology, research application", "description": "Dr. Vasquez's team used transect sampling to estimate the orangutan population."},
    {"source": "Global Wildlife Conservation Institute", "target": "Dr. Elena Vasquez", "keywords": "funding, research support", "description": "The institute funded the expedition led by Dr. Vasquez."},
    {"source": "Dr. Elena Vasquez", "target": "Primate Decline in Insular Southeast Asia", "keywords": "authorship, research output", "description": "Dr. Vasquez's expedition produced the landmark report on primate decline."},
    {"source": "Peat Soil", "target": "Borneo Rainforest", "keywords": "habitat composition, ecological destruction", "description": "Peat soil destruction in the Borneo rainforest was caused by palm oil plantation expansion and is a primary driver of orangutan decline."}
  ]
}

""",
    """---Entity Types---
- Content: Creative or informational works (books, articles, films, reports)
- Artifact: Physical or digital objects created by humans (tools, software, devices)
- Person: Human individuals, real or fictional
- Organization: Companies, institutions, government bodies, groups
- Method: Procedures, techniques, algorithms, workflows
- Data: Quantitative or structured information (statistics, datasets, measurements)
- Concept: Abstract ideas, theories, principles, beliefs

---Input Text---
```
The 2023 edition of "Advances in Neural Architecture Search" synthesized findings from over 200 peer-reviewed papers and introduced a new benchmarking framework called NASBench-360, designed to evaluate search algorithms across diverse task domains. The publication was co-authored by Dr. Priya Nair and Dr. Luca Ferretti of the DeepSystems Research Lab.

NASBench-360 measures three key metrics: search efficiency (time-to-solution), model accuracy on held-out test sets, and computational cost in GPU-hours. Early results showed that evolutionary search algorithms outperformed gradient-based methods by 12% on accuracy while consuming 30% fewer GPU-hours on vision tasks.
```

---Output---
{
  "entities": [
    {"name": "Advances in Neural Architecture Search", "type": "Content", "description": "A 2023 publication that synthesizes findings from over 200 papers and introduces the NASBench-360 benchmarking framework."},
    {"name": "NASBench-360", "type": "Artifact", "description": "NASBench-360 is a benchmarking framework introduced to evaluate neural architecture search algorithms across diverse task domains."},
    {"name": "Dr. Priya Nair", "type": "Person", "description": "Dr. Priya Nair is a co-author of the publication and a researcher at the DeepSystems Research Lab."},
    {"name": "Dr. Luca Ferretti", "type": "Person", "description": "Dr. Luca Ferretti is a co-author of the publication and a researcher at the DeepSystems Research Lab."},
    {"name": "DeepSystems Research Lab", "type": "Organization", "description": "The DeepSystems Research Lab is the institution where the co-authors of the publication are affiliated."},
    {"name": "Evolutionary Search", "type": "Method", "description": "Evolutionary search is a class of neural architecture search algorithms that outperformed gradient-based methods in the NASBench-360 evaluation."},
    {"name": "Gradient-Based Search", "type": "Method", "description": "Gradient-based search is a class of neural architecture search algorithms that was benchmarked against evolutionary search in NASBench-360."},
    {"name": "GPU-Hours", "type": "Data", "description": "GPU-hours is a metric used in NASBench-360 to measure the computational cost of neural architecture search algorithms."},
    {"name": "Neural Architecture Search", "type": "Concept", "description": "Neural architecture search is the automated process of designing optimal neural network architectures, the central topic of the publication."}
  ],
  "relationships": [
    {"source": "Dr. Priya Nair", "target": "Advances in Neural Architecture Search", "keywords": "authorship", "description": "Dr. Priya Nair co-authored the publication."},
    {"source": "Dr. Luca Ferretti", "target": "Advances in Neural Architecture Search", "keywords": "authorship", "description": "Dr. Luca Ferretti co-authored the publication."},
    {"source": "Advances in Neural Architecture Search", "target": "NASBench-360", "keywords": "introduces, benchmarking", "description": "The publication introduced the NASBench-360 framework."},
    {"source": "Evolutionary Search", "target": "Gradient-Based Search", "keywords": "performance comparison", "description": "Evolutionary search outperformed gradient-based methods by 12% on accuracy and used 30% fewer GPU-hours on vision tasks."},
    {"source": "NASBench-360", "target": "GPU-Hours", "keywords": "evaluation metric", "description": "NASBench-360 uses GPU-hours as one of three key metrics to measure computational cost."}
  ]
}

""",
]

PROMPTS["summarize_entity_descriptions"] = """---Role---
You are a Knowledge Graph Specialist, proficient in data curation and synthesis.

---Task---
Your task is to synthesize a list of descriptions of a given entity or relation into a single, comprehensive, and cohesive summary.

---Instructions---
1. Input Format: The description list is provided in JSON format. Each JSON object (representing a single description) appears on a new line within the `Description List` section.
2. Output Format: The merged description will be returned as plain text, presented in multiple paragraphs, without any additional formatting or extraneous comments before or after the summary.
3. Comprehensiveness: The summary must integrate all key information from *every* provided description. Do not omit any important facts or details.
4. Context: Ensure the summary is written from an objective, third-person perspective; explicitly mention the name of the entity or relation for full clarity and context.
5. Context & Objectivity:
  - Write the summary from an objective, third-person perspective.
  - Explicitly mention the full name of the entity or relation at the beginning of the summary to ensure immediate clarity and context.
6. Conflict Handling:
  - In cases of conflicting or inconsistent descriptions, first determine if these conflicts arise from multiple, distinct entities or relationships that share the same name.
  - If distinct entities/relations are identified, summarize each one *separately* within the overall output.
  - If conflicts within a single entity/relation (e.g., historical discrepancies) exist, attempt to reconcile them or present both viewpoints with noted uncertainty.
7. Length Constraint:The summary's total length must not exceed {summary_length} tokens, while still maintaining depth and completeness.
8. Language: The entire output must be written in {language}. Proper nouns (e.g., personal names, place names, organization names) may in their original language if proper translation is not available.
  - The entire output must be written in {language}.
  - Proper nouns (e.g., personal names, place names, organization names) should be retained in their original language if a proper, widely accepted translation is not available or would cause ambiguity.

---Input---
{description_type} Name: {description_name}

Description List:

```
{description_list}
```

---Output---
"""

PROMPTS["fail_response"] = (
    "Sorry, I'm not able to provide an answer to that question.[no-context]"
)

PROMPTS["rag_response"] = """---Role---

You are an expert AI assistant specializing in synthesizing information from a provided knowledge base. Your primary function is to answer user queries accurately by ONLY using the information within the provided **Context**.

---Goal---

Generate a comprehensive, well-structured answer to the user query.
The answer must integrate relevant facts from the Knowledge Graph and Document Chunks found in the **Context**.
Consider the conversation history if provided to maintain conversational flow and avoid repeating information.

---Instructions---

1. Step-by-Step Instruction:
  - Carefully determine the user's query intent in the context of the conversation history to fully understand the user's information need.
  - Scrutinize both `Knowledge Graph Data` and `Document Chunks` in the **Context**. Identify and extract all pieces of information that are directly relevant to answering the user query.
  - Weave the extracted facts into a coherent and logical response. Your own knowledge must ONLY be used to formulate fluent sentences and connect ideas, NOT to introduce any external information.
  - Track the reference_id of the document chunk which directly support the facts presented in the response. Correlate reference_id with the entries in the `Reference Document List` to generate the appropriate citations.
  - Generate a references section at the end of the response. Each reference document must directly support the facts presented in the response.
  - Do not generate anything after the reference section.

2. Content & Grounding:
  - Strictly adhere to the provided context from the **Context**; DO NOT invent, assume, or infer any information not explicitly stated.
  - If the answer cannot be found in the **Context**, state that you do not have enough information to answer. Do not attempt to guess.

3. Formatting & Language:
  - The response MUST be in the same language as the user query.
  - The response MUST utilize Markdown formatting for enhanced clarity and structure (e.g., headings, bold text, bullet points).
  - The response should be presented in {response_type}.

4. References Section Format:
  - The References section should be under heading: `### References`
  - Reference list entries should adhere to the format: `* [n] Document Title`. Do not include a caret (`^`) after opening square bracket (`[`).
  - The Document Title in the citation must retain its original language.
  - Output each citation on an individual line
  - Provide maximum of 5 most relevant citations.
  - Do not generate footnotes section or any comment, summary, or explanation after the references.

5. Reference Section Example:
```
### References

- [1] Document Title One
- [2] Document Title Two
- [3] Document Title Three
```

6. Additional Instructions: {user_prompt}


---Context---

{context_data}
"""

PROMPTS["naive_rag_response"] = """---Role---

You are an expert AI assistant specializing in synthesizing information from a provided knowledge base. Your primary function is to answer user queries accurately by ONLY using the information within the provided **Context**.

---Goal---

Generate a comprehensive, well-structured answer to the user query.
The answer must integrate relevant facts from the Document Chunks found in the **Context**.
Consider the conversation history if provided to maintain conversational flow and avoid repeating information.

---Instructions---

1. Step-by-Step Instruction:
  - Carefully determine the user's query intent in the context of the conversation history to fully understand the user's information need.
  - Scrutinize `Document Chunks` in the **Context**. Identify and extract all pieces of information that are directly relevant to answering the user query.
  - Weave the extracted facts into a coherent and logical response. Your own knowledge must ONLY be used to formulate fluent sentences and connect ideas, NOT to introduce any external information.
  - Track the reference_id of the document chunk which directly support the facts presented in the response. Correlate reference_id with the entries in the `Reference Document List` to generate the appropriate citations.
  - Generate a **References** section at the end of the response. Each reference document must directly support the facts presented in the response.
  - Do not generate anything after the reference section.

2. Content & Grounding:
  - Strictly adhere to the provided context from the **Context**; DO NOT invent, assume, or infer any information not explicitly stated.
  - If the answer cannot be found in the **Context**, state that you do not have enough information to answer. Do not attempt to guess.

3. Formatting & Language:
  - The response MUST be in the same language as the user query.
  - The response MUST utilize Markdown formatting for enhanced clarity and structure (e.g., headings, bold text, bullet points).
  - The response should be presented in {response_type}.

4. References Section Format:
  - The References section should be under heading: `### References`
  - Reference list entries should adhere to the format: `* [n] Document Title`. Do not include a caret (`^`) after opening square bracket (`[`).
  - The Document Title in the citation must retain its original language.
  - Output each citation on an individual line
  - Provide maximum of 5 most relevant citations.
  - Do not generate footnotes section or any comment, summary, or explanation after the references.

5. Reference Section Example:
```
### References

- [1] Document Title One
- [2] Document Title Two
- [3] Document Title Three
```

6. Additional Instructions: {user_prompt}


---Context---

{content_data}
"""

PROMPTS["kg_query_context"] = """
Knowledge Graph Data (Entity):

```json
{entities_str}
```

Knowledge Graph Data (Relationship):

```json
{relations_str}
```

Document Chunks (Each entry has a reference_id refer to the `Reference Document List`):

```json
{text_chunks_str}
```

Reference Document List (Each entry starts with a [reference_id] that corresponds to entries in the Document Chunks):

```
{reference_list_str}
```

"""

PROMPTS["naive_query_context"] = """
Document Chunks (Each entry has a reference_id refer to the `Reference Document List`):

```json
{text_chunks_str}
```

Reference Document List (Each entry starts with a [reference_id] that corresponds to entries in the Document Chunks):

```
{reference_list_str}
```

"""

PROMPTS["keywords_extraction"] = """---Role---
You are an expert keyword extractor, specializing in analyzing user queries for a Retrieval-Augmented Generation (RAG) system. Your purpose is to identify both high-level and low-level keywords in the user's query that will be used for effective document retrieval.

---Goal---
Given a user query, your task is to extract two distinct types of keywords:
1. **high_level_keywords**: for overarching concepts or themes, capturing user's core intent, the subject area, or the type of question being asked.
2. **low_level_keywords**: for specific entities or details, identifying the specific entities, proper nouns, technical jargon, product names, or concrete items.

---Instructions & Constraints---
1. **Output Format**: Your output MUST be a valid JSON object and nothing else. Do not include any explanatory text, markdown code fences (like ```json), or any other text before or after the JSON. It will be parsed directly by a JSON parser.
2. **Source of Truth**: All keywords must be explicitly derived from the user query, with both high-level and low-level keyword categories are required to contain content.
3. **Concise & Meaningful**: Keywords should be concise words or meaningful phrases. Prioritize multi-word phrases when they represent a single concept. For example, from "latest financial report of Apple Inc.", you should extract "latest financial report" and "Apple Inc." rather than "latest", "financial", "report", and "Apple".
4. **Handle Edge Cases**: For queries that are too simple, vague, or nonsensical (e.g., "hello", "ok", "asdfghjkl"), you must return a JSON object with empty lists for both keyword types.
5. **Language**: All extracted keywords MUST be in {language}. Proper nouns (e.g., personal names, place names, organization names) should be kept in their original language.

---Examples---
{examples}

---Real Data---
User Query: {query}

---Output---
Output:"""

PROMPTS["keywords_extraction_examples"] = [
    """Example 1:

Query: "How does international trade influence global economic stability?"

Output:
{
  "high_level_keywords": ["International trade", "Global economic stability", "Economic impact"],
  "low_level_keywords": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]
}

""",
    """Example 2:

Query: "What are the environmental consequences of deforestation on biodiversity?"

Output:
{
  "high_level_keywords": ["Environmental consequences", "Deforestation", "Biodiversity loss"],
  "low_level_keywords": ["Species extinction", "Habitat destruction", "Carbon emissions", "Rainforest", "Ecosystem"]
}

""",
    """Example 3:

Query: "What is the role of education in reducing poverty?"

Output:
{
  "high_level_keywords": ["Education", "Poverty reduction", "Socioeconomic development"],
  "low_level_keywords": ["School access", "Literacy rates", "Job training", "Income inequality"]
}

""",
]
