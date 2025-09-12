from __future__ import annotations
from typing import Any


PROMPTS: dict[str, Any] = {}

PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

PROMPTS["DEFAULT_USER_PROMPT"] = "n/a"

# --- INICIO DE LA MODIFICACIÓN PRINCIPAL PARA DAGs ---

PROMPTS["entity_extraction_system_prompt"] = """---Role---
You are an expert Epidemiological Knowledge Graph Specialist. Your task is to extract causal and hierarchical relationships from epidemiological texts to build a Directed Acyclic Graph (DAG).

---Instructions---
1. **Entity Extraction:** Identify clearly defined epidemiological entities and extract the following:
   - entity_name: Consistent name of the entity.
   - entity_type: Categorize using these types: {entity_types}; suitable types include `Exposure`, `Outcome`, `Risk Factor`, `Protective Factor`, `Biomarker`, `Population`, `Intervention`. If none fit, use `Other`.
   - entity_description: A comprehensive description of the entity based on the text.
2. **Entity Output Format:** (entity{tuple_delimiter}entity_name{tuple_delimiter}entity_type{tuple_delimiter}entity_description)

3. **Directed Relationship Extraction:** Identify direct, meaningful, and causal relationships between extracted entities. Relationships MUST have a clear direction from a source to a target. Extract the following:
   - source_entity: The name of the entity that is the cause or predecessor.
   - target_entity: The name of the entity that is the effect or successor.
   - relationship_type: Classify the relationship using a specific causal or hierarchical term. Examples: `CAUSES_INCREASED_RISK_OF`, `LEADS_TO`, `IS_A_PREDICTOR_FOR`, `PREVENTS`, `IS_A_SUBTYPE_OF`. This field is mandatory.
   - relationship_description: Explain the nature of the directed relationship, providing a clear rationale and citing evidence from the text.
4. **Relationship Output Format:** (relationship{tuple_delimiter}source_entity{tuple_delimiter}target_entity{tuple_delimiter}relationship_type{tuple_delimiter}relationship_description)

5. **Acyclicity Principle:** Your extracted relationships must logically form a DAG. Do not extract relationships that would imply a logical cycle (e.g., A causes B, and B causes A).

6. **Prioritization:** Prioritize foundational causal links (e.g., direct exposure-to-outcome relationships) before secondary or associative links.

7. **Clarity and Explicitness:** Avoid pronouns. Explicitly name all subjects and objects in descriptions.

8. **Language:** Output all text in {language}.
9. **Delimiter:** Use `{record_delimiter}` as the list delimiter and output `{completion_delimiter}` upon completion.

---Examples---
{examples}

---Real Data to be Processed---
<Input>
Entity_types: [{entity_types}]
Text:
```
{input_text}
```
"""

"""

PROMPTS["entity_extraction_user_prompt"] = """---Task---
Extract entities and directed causal relationships from the input text to be Processed.

---Instructions---
1. Output entities and relationships, prioritized by their relevance and causal significance.
2. Output `{completion_delimiter}` when all entities and relationships are extracted.
3. Ensure the output language is {language}.

<Output>
"""

PROMPTS["entity_continue_extraction_user_prompt"] = """---Task---
Identify any missed entities or directed causal relationships from the input text of the last extraction task.

---Instructions---
1. Output the entities and relationships in the same format as the previous extraction task.
2. Do not include entities and relations that have been correctly extracted in the last extraction task.
3. If the entity or relation output was truncated or had missing fields in the last extraction task, please re-output it in the correct format.
4. Output `{completion_delimiter}` when all entities and relationships are extracted.
5. Ensure the output language is {language}.

<Output>
"""

PROMPTS["entity_extraction_examples"] = [
    """<Input Text>
```
A cohort study published in the Lancet found that long-term exposure to PM2.5 air pollution is a significant risk factor for the development of hypertension in urban populations. The mechanism is believed to involve systemic inflammation, 
which is a known predictor for cardiovascular diseases. Hypertension, in turn, is a primary cause of ischemic stroke.
```

<Output>
(entity{tuple_delimiter}PM2.5 Exposure{tuple_delimiter}Exposure{tuple_delimiter}Long-term exposure to PM2.5 air pollution, identified as a risk factor for hypertension.){record_delimiter}
(entity{tuple_delimiter}Hypertension{tuple_delimiter}Outcome{tuple_delimiter}A medical condition (high blood pressure) that can be caused by PM2.5 exposure and can lead to ischemic stroke.){record_delimiter}
(entity{tuple_delimiter}Systemic Inflammation{tuple_delimiter}Biomarker{tuple_delimiter}A biological response believed to be the mechanism linking PM2.5 exposure to hypertension.){record_delimiter}
(entity{tuple_delimiter}Ischemic Stroke{tuple_delimiter}Outcome{tuple_delimiter}A type of stroke that is primarily caused by hypertension.){record_delimiter}
(entity{tuple_delimiter}Urban Populations{tuple_delimiter}Population{tuple_delimiter}The demographic group studied in the cohort study, affected by PM2.5 and hypertension.){record_delimiter}
(relationship{tuple_delimiter}PM2.5 Exposure{tuple_delimiter}Hypertension{tuple_delimiter}IS_A_RISK_FACTOR_FOR{tuple_delimiter}The text states that long-term exposure to PM2.5 is a significant risk factor for the development of hypertension.){record_delimiter}
(relationship{tuple_delimiter}PM2.5 Exposure{tuple_delimiter}Systemic Inflammation{tuple_delimiter}LEADS_TO{tuple_delimiter}The text suggests that the mechanism linking PM2.5 exposure and hypertension involves systemic inflammation.){record_delimiter}
(relationship{tuple_delimiter}Systemic Inflammation{tuple_delimiter}Hypertension{tuple_delimiter}IS_A_PREDICTOR_FOR{tuple_delimiter}Systemic inflammation is mentioned as a predictor for cardiovascular diseases, with hypertension being a related outcome.){record_delimiter}
(relationship{tuple_delimiter}Hypertension{tuple_delimiter}Ischemic Stroke{tuple_delimiter}IS_A_PRIMARY_CAUSE_OF{tuple_delimiter}The text explicitly states that hypertension is a primary cause of ischemic stroke.){record_delimiter}
{completion_delimiter}
""",
    # --- LOS EJEMPLOS ORIGINALES PUEDEN PERMANECER AQUÍ PARA ROBUSTEZ, O PUEDEN SER REEMPLAZADOS POR MÁS EJEMPLOS DEL DOMINIO MÉDICO ---
    """<Input Text>
```
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken 
rebellion against Cruz's narrowing vision of control and order
```

<Output>
(entity{tuple_delimiter}Alex{tuple_delimiter}person{tuple_delimiter}Alex is a character who experiences frustration and is observant of the dynamics among other characters.){record_delimiter}
(entity{tuple_delimiter}Taylor{tuple_delimiter}person{tuple_delimiter}Taylor is portrayed with authoritarian certainty and shows a moment of reverence towards a device, indicating a change in perspective.){record_delimiter}
(entity{tuple_delimiter}Jordan{tuple_delimiter}person{tuple_delimiter}Jordan shares a commitment to discovery and has a significant interaction with Taylor regarding a device.){record_delimiter}
(entity{tuple_delimiter}Cruz{tuple_delimiter}person{tuple_delimiter}Cruz is associated with a vision of control and order, influencing the dynamics among other characters.){record_delimiter}
(relationship{tuple_delimiter}Taylor{tuple_delimiter}Alex{tuple_delimiter}HAS_AUTHORITY_OVER{tuple_delimiter}Alex observes Taylor's authoritarian behavior.){record_delimiter}
(relationship{tuple_delimiter}Jordan{tuple_delimiter}Cruz{tuple_delimiter}IS_IN_REBELLION_AGAINST{tuple_delimiter}Jordan's commitment to discovery is in rebellion against Cruz's vision of control and order.){record_delimiter}
{completion_delimiter}
""",
]

# --- FIN DE LA MODIFICACIÓN PRINCIPAL ---


PROMPTS["summarize_entity_descriptions"] = """---Role---
You are a Knowledge Graph Specialist responsible for data curation and synthesis.

---Task---
Your task is to synthesize a list of descriptions of a given entity or relation into a single, comprehensive, and cohesive summary.

---Instructions---
1. **Comprehensiveness:** The summary must integrate key information from all provided descriptions. Do not omit important facts.
2. **Context:** The summary must explicitly mention the name of the entity or relation for full context.
3. **Conflict:** In case of conflicting or inconsistent descriptions, determine if they originate from multiple, distinct entities or relationships that share the same name. If so, summarize each entity or relationship separately and then consolidate all summaries.
4. **Style:** The output must be written from an objective, third-person perspective.
5. **Length:** Maintain depth and completeness while ensuring the summary's length not exceed {summary_length} tokens.
6. **Language:** The entire output must be written in {language}.

---Data---
{description_type} Name: {description_name}
Description List:
{description_list}

---Output---
"""

PROMPTS["fail_response"] = (
    "Sorry, I'm not able to provide an answer to that question.[no-context]"
)

PROMPTS["rag_response"] = """---Role---

You are a helpful assistant responding to user query about Knowledge Graph and Document Chunks provided in JSON format below.


---Goal---

Generate a concise response based on Knowledge Base and follow Response Rules, considering both current query and the conversation history if provided. Summarize all information in the provided Knowledge Base, and incorporating general knowledge relevant to the Knowledge Base. Do not include information not provided by Knowledge Base.

---Conversation History---
{history}

---Knowledge Graph and Document Chunks---
{context_data}

---Response Guidelines---
**1. Content & Adherence:**
- Strictly adhere to the provided context from the Knowledge Base. Do not invent, assume, or include any information not present in the source data.
- If the answer cannot be found in the provided context, state that you do not have enough information to answer.
- Ensure the response maintains continuity with the conversation history.

**2. Formatting & Language:**
- Format the response using markdown with appropriate section headings.
- The response language must in the same language as the user's question.
- Target format and length: {response_type}

**3. Citations / References:**
- At the end of the response, under a "References" section, each citation must clearly indicate its origin (KG or DC).
- The maximum number of citations is 5, including both KG and DC.
- Use the following formats for citations:
  - For a Knowledge Graph Entity: `[KG] <entity_name>`
  - For a Knowledge Graph Relationship: `[KG] <source_entity_name> -> <target_entity_name>`
  - For a Document Chunk: `[DC] <file_path_or_document_name>`

---USER CONTEXT---
- Additional user prompt: {user_prompt}

---Response---
"""

PROMPTS["keywords_extraction"] = """---Role---
You are an expert keyword extractor, specializing in analyzing user queries for a Retrieval-Augmented Generation (RAG) system. Your purpose is to identify both high-level and low-level keywords in the user's query that will be used for effective document retrieval.

---Goal---
Given a user query, your task is to extract two distinct types of keywords:
1. **high_level_keywords**: for overarching concepts or themes, capturing user's core intent, the subject area, or the type of question being asked.
2. **low_level_keywords**: for specific entities or details, identifying the specific entities, proper nouns, technical jargon, product names, or concrete items.

---Instructions & Constraints---
1. **Output Format**: Your output MUST be a valid JSON object and nothing else. Do not include any explanatory text, markdown code fences (like ```json), or any other text before or after the JSON. It will be parsed directly by a JSON parser.
2. **Source of Truth**: All keywords must be explicitly derived from the user query, with both high-level and low-level keyword categories required to contain content.
3. **Concise & Meaningful**: Keywords should be concise words or meaningful phrases. Prioritize multi-word phrases when they represent a single concept. For example, from "latest financial report of Apple Inc.", you should extract "latest financial report" and "Apple Inc." rather than "latest", "financial", "report", and "Apple".
4. **Handle Edge Cases**: For queries that are too simple, vague, or nonsensical (e.g., "hello", "ok", "asdfghjkl"), you must return a JSON object with empty lists for both keyword types.

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

PROMPTS["naive_rag_response"] = """---Role---

You are a helpful assistant responding to user query about Document Chunks provided provided in JSON format below.

---Goal---

Generate a concise response based on Document Chunks and follow Response Rules, considering both the conversation history and the current query. Summarize all information in the provided Document Chunks, and incorporating general knowledge relevant to the Document Chunks. Do not include information not provided by Document Chunks.

---Conversation History---
{history}

---Document Chunks(DC)---
{content_data}

---RESPONSE GUIDELINES---
**1. Content & Adherence:**
- Strictly adhere to the provided context from the Knowledge Base. Do not invent, assume, or include any information not present in the source data.
- If the answer cannot be found in the provided context, state that you do not have enough information to answer.
- Ensure the response maintains continuity with the conversation history.

**2. Formatting & Language:**
- Format the response using markdown with appropriate section headings.
- The response language must match the user's question language.
- Target format and length: {response_type}

**3. Citations / References:**
- At the end of the response, under a "References" section, cite a maximum of 5 most relevant sources used.
- Use the following formats for citations: `[DC] <file_path_or_document_name>`

---USER CONTEXT---
- Additional user prompt: {user_prompt}

---Response---
Output:"""
