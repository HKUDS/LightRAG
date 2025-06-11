from __future__ import annotations
from typing import Any

GRAPH_FIELD_SEP = "<SEP>"

PROMPTS: dict[str, Any] = {}

PROMPTS["DEFAULT_LANGUAGE"] = "English"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

# Generic entity types covering various domains
PROMPTS["DEFAULT_ENTITY_TYPES"] = [
    "organization",
    "person",
    "location",
    "event",
    "product",
    "technology",
    "concept",
]

PROMPTS["DEFAULT_USER_PROMPT"] = "n/a"

PROMPTS["entity_extraction"] = """---Goal---
Given a text document and a list of entity types, identify the most important and relevant entities of those types from the text and the most significant relationships among the identified entities.

**Key Principles**:
- Focus on the most important and frequently mentioned items only
- Extract entities and relationships that are central to understanding the document
- Prioritize quality over quantity in your extractions

**Entity Priority (extract in this order)**:
1. Named entities (people, organizations, locations) that play key roles
2. Specific products, technologies, or systems mentioned prominently
3. Important events with clear identities or significance
4. Key concepts that are central to understanding the text

**Relationship Priority (extract only the most essential)**:
1. Direct relationships that are explicitly stated in the text
2. Clear connections or interactions between entities
3. Specific actions or dependencies described
4. Important associations that are clearly documented

**Quality over Quantity**: Better to extract fewer high-quality, clearly documented relationships than many inferred or weak ones

When analyzing the text, pay special attention to:
- Pre-existing entities and relationships that may already be documented
- Additional entities and relationships that emerge from the analysis
- Key actors, systems, and their interactions
- Important events and their participants
- Central concepts and how they relate to each other

Use {language} as output language.

---Steps---
1. Identify key entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, use same language as input text. If English, capitalized the name.
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. From the entities identified in step 1, identify pairs of (source_entity, target_entity) that have **explicit, meaningful relationships** documented in the text.
**Strict Relationship Criteria**:
- Must describe an actual interaction or connection stated in the text
- Must use concrete, specific relationship types
- Avoid abstract relationship types unless explicitly stated
- Skip relationships between similar/synonymous concepts
- Each relationship must answer: "What specific action or connection is described?"
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_type: Choose the most appropriate relationship type from the following list. If none fit exactly, choose "related":
  {relationship_types}

  **IMPORTANT: For multi-word relationship types, use underscores to separate words (e.g., 'created_by', 'located_in', 'reports_to'). Do not concatenate words without separators.**

  Examples of specific relationship types to prefer:
  {relationship_examples}
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
- relationship_keywords: one or more high-level key words that summarize the overarching nature of the relationship, focusing on concepts or themes rather than specific details

Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_type>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. Return output in {language} as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

4. When finished, output {completion_delimiter}

---Note on Output Format---
IMPORTANT: Only output entities and relationships in the format specified above. Do not include content summaries, keywords, or any other types of records in your output.

---Note on Pre-existing Data---
If the document contains pre-existing entities and relationships (e.g., from structured analysis), include them in your output while also identifying any additional entities and relationships not already captured.

######################
---Examples---
######################
{examples}

#############################
---Real Data---
######################
Entity_types: [{entity_types}]
Relationship_types: {relationship_types}
Text:
{input_text}
######################
Output:

"""

PROMPTS["entity_extraction_examples"] = [
    """Example 1:

Entity_types: [person, technology, concept, organization, location]
Text:
```
Dr. Sarah Chen presented her groundbreaking research on quantum computing at the MIT Technology Conference. Her team at Quantum Dynamics Inc. has developed a new algorithm that significantly reduces error rates in quantum calculations. The breakthrough was achieved through collaboration with researchers from the Tokyo Institute of Technology.

The algorithm, named QStable, addresses one of the fundamental challenges in quantum computing: maintaining coherence in quantum states. During her presentation, Dr. Chen demonstrated how QStable could revolutionize fields from cryptography to drug discovery.
```

Output:
("entity"{tuple_delimiter}"Dr. Sarah Chen"{tuple_delimiter}"person"{tuple_delimiter}"Dr. Sarah Chen is a researcher who presented groundbreaking quantum computing research at the MIT Technology Conference."){record_delimiter}
("entity"{tuple_delimiter}"MIT Technology Conference"{tuple_delimiter}"event"{tuple_delimiter}"A technology conference where Dr. Sarah Chen presented her quantum computing research."){record_delimiter}
("entity"{tuple_delimiter}"Quantum Dynamics Inc."{tuple_delimiter}"organization"{tuple_delimiter}"The company where Dr. Sarah Chen's team works on quantum computing research."){record_delimiter}
("entity"{tuple_delimiter}"Tokyo Institute of Technology"{tuple_delimiter}"organization"{tuple_delimiter}"An academic institution that collaborated on the quantum computing breakthrough."){record_delimiter}
("entity"{tuple_delimiter}"QStable"{tuple_delimiter}"technology"{tuple_delimiter}"A new algorithm that reduces error rates in quantum calculations by maintaining coherence in quantum states."){record_delimiter}
("entity"{tuple_delimiter}"Quantum Computing"{tuple_delimiter}"concept"{tuple_delimiter}"An advanced computing paradigm that uses quantum states for calculations."){record_delimiter}
("entity"{tuple_delimiter}"Cryptography"{tuple_delimiter}"concept"{tuple_delimiter}"A field that could be revolutionized by the QStable algorithm."){record_delimiter}
("entity"{tuple_delimiter}"Drug Discovery"{tuple_delimiter}"concept"{tuple_delimiter}"Another field that could benefit from the QStable algorithm's capabilities."){record_delimiter}
("relationship"{tuple_delimiter}"Dr. Sarah Chen"{tuple_delimiter}"MIT Technology Conference"{tuple_delimiter}"Dr. Sarah Chen presented her research at the MIT Technology Conference."{tuple_delimiter}"presents_at"{tuple_delimiter}"conference presentation, research dissemination"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Dr. Sarah Chen"{tuple_delimiter}"Quantum Dynamics Inc."{tuple_delimiter}"Dr. Sarah Chen leads a research team at Quantum Dynamics Inc."{tuple_delimiter}"works_at"{tuple_delimiter}"employment, research leadership"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"Quantum Dynamics Inc."{tuple_delimiter}"Tokyo Institute of Technology"{tuple_delimiter}"The two organizations collaborated on the quantum computing breakthrough."{tuple_delimiter}"collaborates_with"{tuple_delimiter}"research collaboration, partnership"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Dr. Sarah Chen"{tuple_delimiter}"QStable"{tuple_delimiter}"Dr. Sarah Chen's team developed the QStable algorithm."{tuple_delimiter}"develops"{tuple_delimiter}"algorithm development, innovation"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"QStable"{tuple_delimiter}"Quantum Computing"{tuple_delimiter}"QStable addresses fundamental challenges in quantum computing."{tuple_delimiter}"advances"{tuple_delimiter}"technological advancement, problem solving"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"QStable"{tuple_delimiter}"Cryptography"{tuple_delimiter}"QStable could revolutionize the field of cryptography."{tuple_delimiter}"impacts"{tuple_delimiter}"potential application, field transformation"{tuple_delimiter}7){completion_delimiter}
#############################
""",
    """Example 2:

Entity_types: [organization, person, location, event, product, concept]
Text:
```
The Global Climate Summit in Geneva brought together leaders from over 150 nations to address rising sea levels. President Maria Rodriguez of Costa Verde announced a $2 billion Green Infrastructure Initiative, partnering with the World Bank and environmental organizations like EcoFuture Foundation.

The initiative will focus on sustainable urban development and renewable energy projects. John Clarke, CEO of EcoFuture Foundation, praised the commitment but emphasized the need for immediate action. The summit concluded with the Geneva Accord, setting ambitious targets for carbon reduction by 2035.
```

Output:
("entity"{tuple_delimiter}"Global Climate Summit"{tuple_delimiter}"event"{tuple_delimiter}"An international summit in Geneva addressing climate change with participation from over 150 nations."){record_delimiter}
("entity"{tuple_delimiter}"Geneva"{tuple_delimiter}"location"{tuple_delimiter}"The city hosting the Global Climate Summit."){record_delimiter}
("entity"{tuple_delimiter}"President Maria Rodriguez"{tuple_delimiter}"person"{tuple_delimiter}"President of Costa Verde who announced the Green Infrastructure Initiative."){record_delimiter}
("entity"{tuple_delimiter}"Costa Verde"{tuple_delimiter}"location"{tuple_delimiter}"The country led by President Maria Rodriguez."){record_delimiter}
("entity"{tuple_delimiter}"Green Infrastructure Initiative"{tuple_delimiter}"product"{tuple_delimiter}"A $2 billion initiative for sustainable urban development and renewable energy projects."){record_delimiter}
("entity"{tuple_delimiter}"World Bank"{tuple_delimiter}"organization"{tuple_delimiter}"International financial institution partnering in the Green Infrastructure Initiative."){record_delimiter}
("entity"{tuple_delimiter}"EcoFuture Foundation"{tuple_delimiter}"organization"{tuple_delimiter}"Environmental organization involved in the climate initiative."){record_delimiter}
("entity"{tuple_delimiter}"John Clarke"{tuple_delimiter}"person"{tuple_delimiter}"CEO of EcoFuture Foundation who commented on the climate commitments."){record_delimiter}
("entity"{tuple_delimiter}"Geneva Accord"{tuple_delimiter}"product"{tuple_delimiter}"An agreement setting carbon reduction targets for 2035."){record_delimiter}
("entity"{tuple_delimiter}"Sustainable Urban Development"{tuple_delimiter}"concept"{tuple_delimiter}"A focus area of the Green Infrastructure Initiative."){record_delimiter}
("relationship"{tuple_delimiter}"Global Climate Summit"{tuple_delimiter}"Geneva"{tuple_delimiter}"The Global Climate Summit was held in Geneva."{tuple_delimiter}"located_in"{tuple_delimiter}"event location, hosting"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"President Maria Rodriguez"{tuple_delimiter}"Costa Verde"{tuple_delimiter}"Maria Rodriguez is the President of Costa Verde."{tuple_delimiter}"leads"{tuple_delimiter}"political leadership, governance"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"President Maria Rodriguez"{tuple_delimiter}"Green Infrastructure Initiative"{tuple_delimiter}"President Rodriguez announced the Green Infrastructure Initiative."{tuple_delimiter}"announces"{tuple_delimiter}"policy announcement, initiative launch"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Green Infrastructure Initiative"{tuple_delimiter}"World Bank"{tuple_delimiter}"The World Bank is partnering in the Green Infrastructure Initiative."{tuple_delimiter}"partners_with"{tuple_delimiter}"financial partnership, collaboration"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"John Clarke"{tuple_delimiter}"EcoFuture Foundation"{tuple_delimiter}"John Clarke serves as CEO of EcoFuture Foundation."{tuple_delimiter}"leads"{tuple_delimiter}"executive leadership, management"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"Global Climate Summit"{tuple_delimiter}"Geneva Accord"{tuple_delimiter}"The summit concluded with the Geneva Accord agreement."{tuple_delimiter}"produces"{tuple_delimiter}"agreement creation, summit outcome"{tuple_delimiter}9){completion_delimiter}
#############################
""",
    """Example 3:

Entity_types: [organization, product, technology, person, concept]
Text:
```
TechCorp unveiled its revolutionary AI assistant, ARIA, at the annual Consumer Electronics Show. The product integrates natural language processing with advanced machine learning to provide personalized user experiences. CEO Amanda Foster highlighted ARIA's potential to transform how people interact with technology in their daily lives.
```

Output:
("entity"{tuple_delimiter}"TechCorp"{tuple_delimiter}"organization"{tuple_delimiter}"A technology company that developed and unveiled the ARIA AI assistant."){record_delimiter}
("entity"{tuple_delimiter}"ARIA"{tuple_delimiter}"product"{tuple_delimiter}"A revolutionary AI assistant that integrates natural language processing with machine learning."){record_delimiter}
("entity"{tuple_delimiter}"Consumer Electronics Show"{tuple_delimiter}"event"{tuple_delimiter}"An annual technology exhibition where ARIA was unveiled."){record_delimiter}
("entity"{tuple_delimiter}"Amanda Foster"{tuple_delimiter}"person"{tuple_delimiter}"CEO of TechCorp who presented ARIA's capabilities."){record_delimiter}
("entity"{tuple_delimiter}"Natural Language Processing"{tuple_delimiter}"technology"{tuple_delimiter}"A technology integrated into ARIA for understanding human language."){record_delimiter}
("entity"{tuple_delimiter}"Machine Learning"{tuple_delimiter}"technology"{tuple_delimiter}"Advanced technology used in ARIA for personalized experiences."){record_delimiter}
("relationship"{tuple_delimiter}"TechCorp"{tuple_delimiter}"ARIA"{tuple_delimiter}"TechCorp developed and unveiled the ARIA AI assistant."{tuple_delimiter}"develops"{tuple_delimiter}"product development, innovation"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"TechCorp"{tuple_delimiter}"Consumer Electronics Show"{tuple_delimiter}"TechCorp unveiled ARIA at the Consumer Electronics Show."{tuple_delimiter}"presents_at"{tuple_delimiter}"product launch, exhibition"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Amanda Foster"{tuple_delimiter}"TechCorp"{tuple_delimiter}"Amanda Foster is the CEO of TechCorp."{tuple_delimiter}"leads"{tuple_delimiter}"executive leadership, company management"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"ARIA"{tuple_delimiter}"Natural Language Processing"{tuple_delimiter}"ARIA integrates natural language processing technology."{tuple_delimiter}"integrates"{tuple_delimiter}"technology integration, capability"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"ARIA"{tuple_delimiter}"Machine Learning"{tuple_delimiter}"ARIA uses advanced machine learning for personalization."{tuple_delimiter}"utilizes"{tuple_delimiter}"technology application, functionality"{tuple_delimiter}9){completion_delimiter}
#############################
""",
]

PROMPTS[
    "summarize_entity_descriptions"
] = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or two entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we have the full context.

For all entity types, ensure the summary includes:
- Primary purpose, role, or function
- Key attributes or characteristics
- Relationships or connections with other entities
- Significance or impact within the context

Use {language} as output language.

#######
---Data---
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""

PROMPTS["entity_continue_extraction"] = """
Some additional entities and relationships may have been missed in the last extraction.

Focus on capturing ONLY concrete items that were overlooked:
- Named people, organizations, or locations that were missed
- Specific products, technologies, or systems with clear names
- Important events or initiatives that were missed
- Key concepts that were extensively discussed but not captured

**Apply the same strict criteria as before - focus on concrete, important entities and relationships**

---Remember Steps---

1. Identify key entities that were missed. For each identified entity, extract the following information:
- entity_name: Name of the entity, use same language as input text. If English, capitalized the name.
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. From the entities identified in step 1, identify pairs of (source_entity, target_entity) that have **explicit, meaningful relationships** documented in the text.
**Apply the same strict criteria as before - only extract relationships that are clearly stated, not inferred**
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_type: Choose the most appropriate relationship type from the following list. If none fit exactly, choose "related":
  {relationship_types}

  **IMPORTANT: For multi-word relationship types, use underscores to separate words (e.g., 'created_by', 'located_in', 'reports_to'). Do not concatenate words without separators.**
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
- relationship_keywords: one or more high-level key words that summarize the overarching nature of the relationship, focusing on concepts or themes rather than specific details
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_type>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. Return output in {language} as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

4. When finished, output {completion_delimiter}

---Output---

Add them below using the same format:
""".strip()

PROMPTS["entity_if_loop_extraction"] = """
---Goal---

It appears some entities may have still been missed. Check for:
- Additional people, organizations, or locations not yet captured
- Products, technologies, or systems mentioned in passing
- Events or initiatives that were referenced
- Concepts that were discussed but not explicitly captured

---Output---

Answer ONLY by `YES` OR `NO` if there are still entities that need to be added.
""".strip()

PROMPTS["fail_response"] = (
    "Sorry, I'm not able to provide an answer to that question.[no-context]"
)

PROMPTS["rag_response"] = """---Role---

You are a helpful assistant responding to queries using the Knowledge Graph and Document Chunks provided in JSON format below.

---Goal---

Generate a concise response based on Knowledge Base and follow Response Rules, considering both the conversation history and the current query. Summarize all information in the provided Knowledge Base, and incorporating general knowledge relevant to the Knowledge Base. Do not include information not provided by Knowledge Base.

When handling relationships with timestamps:
1. Each relationship has a "created_at" timestamp indicating when we acquired this knowledge
2. When encountering conflicting relationships, consider both the semantic content and the timestamp
3. Don't automatically prefer the most recently created relationships - use judgment based on the context
4. For time-specific queries, prioritize temporal information in the content before considering creation timestamps

When answering queries:
- Provide comprehensive insights based on the knowledge graph
- Highlight connections and relationships between entities
- Include relevant context from document chunks when available

---Conversation History---
{history}

---Knowledge Graph and Document Chunks---
{context_data}

---Response Rules---

- Target format and length: {response_type}
- Use markdown formatting with appropriate section headings
- Please respond in the same language as the user's question.
- Ensure the response maintains continuity with the conversation history.
- List up to 5 most important reference sources at the end under "References" section. Clearly indicating whether each source is from Knowledge Graph (KG) or Document Chunks (DC), and include the file path if available, in the following format: [KG/DC] file_path
- If you don't know the answer, just say so.
- Do not make anything up. Do not include information not provided by the Knowledge Base.
- Additional user prompt: {user_prompt}

Response:"""

PROMPTS["keywords_extraction"] = """---Role---

You are a helpful assistant tasked with identifying both high-level and low-level keywords in the user's query and conversation history.

---Goal---

Given the query and conversation history, list both high-level and low-level keywords. High-level keywords focus on overarching concepts or themes, while low-level keywords focus on specific entities, details, or concrete terms.

When extracting keywords, ensure you capture:
- Named entities (people, organizations, locations)
- Specific products, technologies, or systems
- Key concepts and themes
- Actions or relationships mentioned
- Any domain-specific terminology

---Instructions---

- Consider both the current query and relevant conversation history when extracting keywords
- Output the keywords in JSON format, it will be parsed by a JSON parser, do not add any extra content in output
- The JSON should have two keys:
  - "high_level_keywords" for overarching concepts or themes
  - "low_level_keywords" for specific entities or details

######################
---Examples---
######################
{examples}

#############################
---Real Data---
######################
Conversation History:
{history}

Current Query: {query}
######################
The `Output` should be human text, not unicode characters. Keep the same language as `Query`.
Output:

"""

PROMPTS["keywords_extraction_examples"] = [
    """Example 1:

Query: "How does the new climate initiative impact renewable energy development in coastal regions?"
################
Output:
{
  "high_level_keywords": ["Climate initiative", "Renewable energy development", "Environmental impact", "Regional development"],
  "low_level_keywords": ["coastal regions", "renewable energy", "climate", "development projects", "environmental policy"]
}
#############################""",
    """Example 2:

Query: "What role did Dr. Chen play in the quantum computing breakthrough at MIT?"
################
Output:
{
  "high_level_keywords": ["Quantum computing breakthrough", "Research leadership", "Academic achievement", "Technological innovation"],
  "low_level_keywords": ["Dr. Chen", "MIT", "quantum computing", "research role", "breakthrough", "technology development"]
}
#############################""",
    """Example 3:

Query: "Which organizations are partnering with TechCorp on the AI assistant project?"
################
Output:
{
  "high_level_keywords": ["Organizational partnerships", "AI development", "Technology collaboration", "Project cooperation"],
  "low_level_keywords": ["TechCorp", "AI assistant", "partner organizations", "project", "collaboration", "technology partners"]
}
#############################""",
]

PROMPTS["naive_rag_response"] = """---Role---

You are a helpful assistant responding to queries using Document Chunks provided in JSON format below.

---Goal---

Generate a concise response based on Document Chunks and follow Response Rules, considering both the conversation history and the current query. Summarize all information in the provided Document Chunks, and incorporating general knowledge relevant to the Document Chunks. Do not include information not provided by Document Chunks.

When handling content with timestamps:
1. Each piece of content has a "created_at" timestamp indicating when we acquired this knowledge
2. When encountering conflicting information, consider both the content and the timestamp
3. Don't automatically prefer the most recent content - use judgment based on the context
4. For time-specific queries, prioritize temporal information in the content before considering creation timestamps

When analyzing content:
- Focus on extracting key information and insights
- Identify patterns and connections across chunks
- Provide relevant context when available

---Conversation History---
{history}

---Document Chunks(DC)---
{content_data}

---Response Rules---

- Target format and length: {response_type}
- Use markdown formatting with appropriate section headings
- Please respond in the same language as the user's question.
- Ensure the response maintains continuity with the conversation history.
- List up to 5 most important reference sources at the end under "References" section. Clearly indicating each source from Document Chunks(DC), and include the file path if available, in the following format: [DC] file_path
- If you don't know the answer, just say so.
- Do not include information not provided by the Document Chunks.
- Additional user prompt: {user_prompt}

Response:"""

# TODO: deprecated
PROMPTS[
    "similarity_check"
] = """Please analyze the similarity between these two questions:

Question 1: {original_prompt}
Question 2: {cached_prompt}

Please evaluate whether these two questions are semantically similar, and whether the answer to Question 2 can be used to answer Question 1, provide a similarity score between 0 and 1 directly.

Similarity score criteria:
0: Completely unrelated or answer cannot be reused, including but not limited to:
   - The questions have different topics
   - The entities or subjects mentioned are different
   - The context or domain is different
   - The specific details or requirements are different
   - The time periods or locations are different
   - The relationships or interactions described are different
   - The intent or purpose of the questions is different
1: Identical and answer can be directly reused
0.5: Partially related and answer needs modification to be used
Return only a number between 0-1, without any additional content.
"""

PROMPTS["relationship_post_processing"] = """---Goal---
You are analyzing extracted entities and relationships from a document to improve accuracy and remove noise.

---Context---
DOCUMENT CONTENT:
{document_text}

EXTRACTED ENTITIES:
{entities_list}

EXTRACTED RELATIONSHIPS:
{relationships_list}

---Task---
Review and filter these relationships for accuracy and relevance. Your goal is to achieve high relationship accuracy by keeping only the most valuable and well-supported relationships.

**VALIDATION CRITERIA:**
1. **Document Evidence**: Relationship must be explicitly supported by document content
2. **Specificity**: Avoid abstract/generic relationships without clear evidence
3. **Practical Value**: Keep relationships that provide meaningful insights about the document
4. **Avoid Noise**: Remove relationships based on coincidental mentions or weak associations
5. **Preserve Relationship Types**: KEEP the original rel_type (e.g., "uses", "located_in", "creates") unless it is factually incorrect. Do NOT change specific relationship types to generic "related"

**FILTERING GUIDELINES:**
✅ KEEP relationships that are:
- Explicitly stated connections or interactions
- Clear organizational or hierarchical relationships
- Specific actions or events described in the text
- Well-documented partnerships or collaborations

❌ REMOVE relationships that are:
- Too abstract or conceptual without evidence
- Based on weak associations or assumptions
- Redundant or duplicate meanings
- Generic connections without specific evidence

**QUALITY SCORING** (1-10 scale):
- 9-10: Explicitly stated, high practical value
- 7-8: Well-supported, clear evidence
- 5-6: Moderately supported, some evidence
- 3-4: Weak evidence, questionable value
- 1-2: No clear evidence, likely noise

**CONFIDENCE THRESHOLD**: Keep relationships scoring 6 or higher.

---Output Format---
Respond with valid JSON only:

```json
{{
  "validated_relationships": [
    {{
      "src_id": "entity1",
      "tgt_id": "entity2",
      "rel_type": "specific_type",
      "description": "clear description of the relationship",
      "quality_score": 8,
      "evidence": "specific reference or quote from document supporting this relationship",
      "weight": 0.9,
      "source_id": "original_chunk_id"
    }}
  ],
  "removed_relationships": [
    {{
      "src_id": "entity1",
      "tgt_id": "entity2",
      "rel_type": "original_type",
      "reason": "specific reason for removal (e.g., 'too abstract', 'no document evidence', 'weak association')"
    }}
  ],
  "processing_summary": {{
    "total_input": 150,
    "validated": 85,
    "removed": 65,
    "accuracy_improvement": "Removed abstract relationships and unsupported associations",
    "average_quality_score": 7.2
  }}
}}
```

**CRITICAL INSTRUCTION**: You MUST preserve the exact original relationship type (rel_type) from the input relationships. Do NOT convert specific types like "works_at", "located_in", "develops", "partners_with", "leads", etc. to generic "related". The relationship types carry important semantic meaning that must be maintained.

Examples of what to preserve:
- "Dr. Chen -[\"presents_at\"]-> MIT Conference" → Keep "presents_at"
- "TechCorp -[\"develops\"]-> ARIA" → Keep "develops"
- "World Bank -[\"partners_with\"]-> Green Initiative" → Keep "partners_with"
- "Summit -[\"located_in\"]-> Geneva" → Keep "located_in"

Only change rel_type if it's factually incorrect based on the document evidence.

Focus on relationships that a domain expert reviewing this document would find genuinely useful and accurate.
"""
