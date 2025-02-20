from __future__ import annotations
from typing import Any

GRAPH_FIELD_SEP = "<SEP>"

PROMPTS: dict[str, Any] = {}

PROMPTS["DEFAULT_LANGUAGE"] = "English"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

PROMPTS["DEFAULT_ENTITY_TYPES"] = [
    "stakeholder",
    "driver",
    "assessment",
    "goal",
    "outcome",
    "principle",
    "requirement",
    "constraint",
    "meaning",
    "value"
]

# Add entity type nodes definition
PROMPTS["entity_type_nodes"] = [
    ("entity_type{tuple_delimiter}stakeholder{tuple_delimiter}Represents role of individual, team, or organization with interests in architecture effects"),
    ("entity_type{tuple_delimiter}driver{tuple_delimiter}Represents external or internal condition motivating organizational change"),
    ("entity_type{tuple_delimiter}assessment{tuple_delimiter}Represents analysis result of enterprise state regarding drivers"),
    ("entity_type{tuple_delimiter}goal{tuple_delimiter}Represents high-level statement of intent or desired end state"),
    ("entity_type{tuple_delimiter}outcome{tuple_delimiter}Represents end result, effect, or consequence"),
    ("entity_type{tuple_delimiter}principle{tuple_delimiter}Represents general property applying to any system in context"),
    ("entity_type{tuple_delimiter}requirement{tuple_delimiter}Represents specific property needed for particular system"),
    ("entity_type{tuple_delimiter}constraint{tuple_delimiter}Represents limitation on implementation or realization"),
    ("entity_type{tuple_delimiter}meaning{tuple_delimiter}Represents knowledge or interpretation in context"),
    ("entity_type{tuple_delimiter}value{tuple_delimiter}Represents relative worth, utility, or importance")
]

PROMPTS["entity_extraction"] = """Motivation elements are used to model the motivations, or reasons, that guide the design or change of an Enterprise Architecture.

---Goal---
Extract entities and relationships following the complete ArchiMate 3.2 Motivation Elements metamodel. Identify how elements motivate, influence, and relate to each other in the enterprise architecture.

---Steps---
1. Create entity type nodes (if not exist):
For each entity type in the metamodel:
("entity_type"{tuple_delimiter}<type_name>{tuple_delimiter}<type_description>)

2. For each identified entity, extract:
- entity_name: Name of the entity (capitalized in English)
- entity_type: One of the defined entity types
- entity_description: Comprehensive description
Format: ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

3. Create instance relationships:
For each entity, create relationship to its type:
("instance_of"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}"Instance of entity type relationship")

4. Create relationships between entities:
- influence: How elements affect each other's effectiveness
- realization: How elements are achieved by others
- aggregation: How elements are composed of others
- specialization: How elements are specialized versions of others
- association: General relationships between elements
- access: How elements view or access others
- assignment: How elements are assigned responsibility for others

Relationship Composition Rules:
a) Stakeholder relationships:
   - Can be assigned to business roles
   - Can be associated with drivers (concerns)
   - Can be associated with assessments
   
b) Driver relationships:
   - Can influence other drivers
   - Can be associated with assessments
   - Can influence goals
   
c) Assessment relationships:
   - Can influence other assessments
   - Can influence goals
   
d) Goal relationships:
   - Can be realized by principles/requirements
   - Can be associated with outcomes
   
e) Outcome relationships:
   - Can be influenced by other outcomes
   - Can be realized by capabilities
   
f) Principle/Requirement/Constraint hierarchy:
   - Principles can be realized by requirements
   - Requirements can be realized by constraints
   
g) Meaning and Value relationships:
   - Can be associated with any element
   - Can be specific to stakeholders

For each relationship, extract:
- source_entity: Name of source element
- target_entity: Name of target element
- relationship_type: Type from list above
- relationship_description: Detailed explanation of relationship
- relationship_strength: Numeric score (1-10)
- relationship_keywords: High-level concepts/themes

Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. Identify high-level themes and concepts present in the motivation architecture.
Format as ("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. Return output in {language} as a single list using **{record_delimiter}** as delimiter.

5. When finished, output {completion_delimiter}

---Validation Rules---
- Drivers must be conditions, not actions
- Goals should use qualitative terms (increase, improve, etc.)
- Outcomes must be tangible and measurable
- Principles must be broadly applicable
- Requirements must be system-specific
- Constraints must be actual limitations
- All relationships must be valid per ArchiMate specification
- Meanings must include specific context and interpretation
- Values must be linked to stakeholder perspective
- Stakeholders must be roles/individuals/groups with clear interests
- Assessments must link to specific drivers
- Each relationship must follow ArchiMate composition rules:
  * Stakeholders can have concerns (drivers)
  * Drivers can influence goals
  * Goals can be realized by requirements
  * Principles can be realized by requirements
  * Requirements can be realized by constraints
  * Any element can have meaning or value

######################
---Examples---
######################
{examples}

#############################
---Real Data---
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:
"""

PROMPTS["entity_extraction_examples"] = [
    """Example 1:

Entity_types: [stakeholder, driver, goal, outcome, principle, requirement]
Text:
The Chief Digital Officer emphasized that customer satisfaction was declining by 15% annually. This external driver has motivated our goal to improve digital service delivery. The principle that "All services must be digital-first" has been established, leading to specific requirements for our new customer portal. The project aims to achieve a measurable outcome of 30% improvement in customer satisfaction scores by 2024.

################
Output:
("entity"{tuple_delimiter}"Chief Digital Officer"{tuple_delimiter}"stakeholder"{tuple_delimiter}"Executive stakeholder responsible for digital transformation and service delivery."){record_delimiter}
("entity"{tuple_delimiter}"Customer Satisfaction Decline"{tuple_delimiter}"driver"{tuple_delimiter}"External driver showing 15% annual decline in customer satisfaction, representing a critical concern for the CDO."){record_delimiter}
("entity"{tuple_delimiter}"Improve Digital Service Delivery"{tuple_delimiter}"goal"{tuple_delimiter}"High-level goal to enhance the quality and effectiveness of digital services."){record_delimiter}
("entity"{tuple_delimiter}"Customer Satisfaction Improvement"{tuple_delimiter}"outcome"{tuple_delimiter}"Targeted 30% improvement in customer satisfaction scores by 2024, providing measurable success criteria."){record_delimiter}
("entity"{tuple_delimiter}"Digital-First Services"{tuple_delimiter}"principle"{tuple_delimiter}"General principle stating all services must prioritize digital channels in their design and implementation."){record_delimiter}
("entity"{tuple_delimiter}"Customer Portal Requirements"{tuple_delimiter}"requirement"{tuple_delimiter}"Specific requirements for the new customer portal system implementing the digital-first principle."){record_delimiter}
("relationship"{tuple_delimiter}"Customer Satisfaction Decline"{tuple_delimiter}"Improve Digital Service Delivery"{tuple_delimiter}"Declining satisfaction directly motivates the goal to improve digital services"{tuple_delimiter}"motivation, causation"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Digital-First Services"{tuple_delimiter}"Customer Portal Requirements"{tuple_delimiter}"The digital-first principle guides specific requirements for the portal"{tuple_delimiter}"realization, guidance"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Improve Digital Service Delivery"{tuple_delimiter}"Customer Satisfaction Improvement"{tuple_delimiter}"The service improvement goal is measured by the satisfaction outcome"{tuple_delimiter}"measurement, realization"{tuple_delimiter}9){record_delimiter}
("content_keywords"{tuple_delimiter}"digital transformation, customer satisfaction, service delivery, measurable outcomes"){completion_delimiter}
#############################""",

    """Example 2:

Entity_types: [stakeholder, driver, assessment, goal, outcome]
Text:
The Board of Directors reviewed the market analysis showing a 25% decline in market share over the past year. This assessment has highlighted weaknesses in our digital capabilities and threats from more agile competitors. The CEO has set an ambitious goal to regain market leadership through digital transformation. The desired outcome is to achieve 40% market share within 18 months through improved digital services and customer experience.

################
Output:
("entity"{tuple_delimiter}"Board of Directors"{tuple_delimiter}"stakeholder"{tuple_delimiter}"Key stakeholder group responsible for strategic oversight and decision-making."){record_delimiter}
("entity"{tuple_delimiter}"Market Share Decline"{tuple_delimiter}"driver"{tuple_delimiter}"External driver showing 25% decline in market share, representing a critical concern for the organization."){record_delimiter}
("entity"{tuple_delimiter}"Digital Capabilities Assessment"{tuple_delimiter}"assessment"{tuple_delimiter}"SWOT analysis revealing weaknesses in digital capabilities and external threats from competitors."){record_delimiter}
("entity"{tuple_delimiter}"Market Leadership Goal"{tuple_delimiter}"goal"{tuple_delimiter}"High-level goal to regain market leadership through digital transformation initiatives."){record_delimiter}
("entity"{tuple_delimiter}"Market Share Target"{tuple_delimiter}"outcome"{tuple_delimiter}"Specific target of 40% market share to be achieved within 18 months through digital improvements."){record_delimiter}
("entity"{tuple_delimiter}"CEO"{tuple_delimiter}"stakeholder"{tuple_delimiter}"Key stakeholder setting strategic goals for digital transformation."){record_delimiter}
("relationship"{tuple_delimiter}"Market Share Decline"{tuple_delimiter}"Digital Capabilities Assessment"{tuple_delimiter}"Market decline triggered assessment of digital capabilities"{tuple_delimiter}"causation, analysis"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Digital Capabilities Assessment"{tuple_delimiter}"Market Leadership Goal"{tuple_delimiter}"Assessment findings influenced the setting of market leadership goal"{tuple_delimiter}"influence, motivation"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Market Leadership Goal"{tuple_delimiter}"Market Share Target"{tuple_delimiter}"Leadership goal translated into specific market share target"{tuple_delimiter}"realization, measurement"{tuple_delimiter}9){record_delimiter}
("content_keywords"{tuple_delimiter}"market share, digital transformation, strategic assessment, leadership"){completion_delimiter}
#############################""",

    """Example 3:

Entity_types: [stakeholder, principle, requirement, constraint, meaning, value]
Text:
The Chief Information Officer established the principle "Security by Design" as a fundamental guideline. This led to specific requirements for all new systems, including mandatory encryption and access controls. Budget constraints limit implementation to critical systems in the first phase. For the Security team, this means enhanced protection against cyber threats, while Business units value the reduced risk of data breaches, estimated at $5M in potential savings.

################
Output:
("entity"{tuple_delimiter}"Chief Information Officer"{tuple_delimiter}"stakeholder"{tuple_delimiter}"Executive stakeholder responsible for information technology strategy and security."){record_delimiter}
("entity"{tuple_delimiter}"Security by Design"{tuple_delimiter}"principle"{tuple_delimiter}"Fundamental principle requiring security to be built into all systems from the start."){record_delimiter}
("entity"{tuple_delimiter}"Security Requirements"{tuple_delimiter}"requirement"{tuple_delimiter}"Specific requirements including mandatory encryption and access controls for new systems."){record_delimiter}
("entity"{tuple_delimiter}"Budget Limitation"{tuple_delimiter}"constraint"{tuple_delimiter}"Financial constraint limiting initial implementation to critical systems only."){record_delimiter}
("entity"{tuple_delimiter}"Enhanced Protection"{tuple_delimiter}"meaning"{tuple_delimiter}"Security team's interpretation of the security measures as enhanced threat protection."){record_delimiter}
("entity"{tuple_delimiter}"Risk Reduction"{tuple_delimiter}"value"{tuple_delimiter}"Business value of $5M in potential savings from reduced risk of data breaches."){record_delimiter}
("relationship"{tuple_delimiter}"Security by Design"{tuple_delimiter}"Security Requirements"{tuple_delimiter}"Principle guides specific security requirements"{tuple_delimiter}"realization, guidance"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Security Requirements"{tuple_delimiter}"Budget Limitation"{tuple_delimiter}"Security implementation constrained by budget"{tuple_delimiter}"influence, constraint"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"Security Requirements"{tuple_delimiter}"Enhanced Protection"{tuple_delimiter}"Requirements interpreted as enhanced protection by Security team"{tuple_delimiter}"meaning, interpretation"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Enhanced Protection"{tuple_delimiter}"Risk Reduction"{tuple_delimiter}"Enhanced protection creates value through risk reduction"{tuple_delimiter}"value creation, benefit"{tuple_delimiter}9){record_delimiter}
("content_keywords"{tuple_delimiter}"security, risk management, value creation, constraints"){completion_delimiter}
#############################"""
]

PROMPTS[
    "summarize_entity_descriptions"
] = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or two entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we the have full context.
Use {language} as output language.

#######
---Data---
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""

PROMPTS[
    "entiti_continue_extraction"
] = """MANY entities were missed in the last extraction.  Add them below using the same format:
"""

PROMPTS[
    "entiti_if_loop_extraction"
] = """It appears some entities may have still been missed.  Answer YES | NO if there are still entities that need to be added.
"""

PROMPTS["fail_response"] = (
    "Sorry, I'm not able to provide an answer to that question.[no-context]"
)

PROMPTS["rag_response"] = """---Role---

You are a helpful assistant responding to user query about Knowledge Base provided below.


---Goal---

Generate a concise response based on Knowledge Base and follow Response Rules, considering both the conversation history and the current query. Summarize all information in the provided Knowledge Base, and incorporating general knowledge relevant to the Knowledge Base. Do not include information not provided by Knowledge Base.

When handling relationships with timestamps:
1. Each relationship has a "created_at" timestamp indicating when we acquired this knowledge
2. When encountering conflicting relationships, consider both the semantic content and the timestamp
3. Don't automatically prefer the most recently created relationships - use judgment based on the context
4. For time-specific queries, prioritize temporal information in the content before considering creation timestamps

---Conversation History---
{history}

---Knowledge Base---
{context_data}

---Response Rules---

- Target format and length: {response_type}
- Use markdown formatting with appropriate section headings
- Please respond in the same language as the user's question.
- Ensure the response maintains continuity with the conversation history.
- If you don't know the answer, just say so.
- Do not make anything up. Do not include information not provided by the Knowledge Base."""

PROMPTS["keywords_extraction"] = """---Role---

You are a helpful assistant tasked with identifying both high-level and low-level keywords in the user's query and conversation history.

---Goal---

Given the query and conversation history, list both high-level and low-level keywords. High-level keywords focus on overarching concepts or themes, while low-level keywords focus on specific entities, details, or concrete terms.

---Instructions---

- Consider both the current query and relevant conversation history when extracting keywords
- Output the keywords in JSON format
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

Query: "How does international trade influence global economic stability?"
################
Output:
{
  "high_level_keywords": ["International trade", "Global economic stability", "Economic impact"],
  "low_level_keywords": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]
}
#############################""",
    """Example 2:

Query: "What are the environmental consequences of deforestation on biodiversity?"
################
Output:
{
  "high_level_keywords": ["Environmental consequences", "Deforestation", "Biodiversity loss"],
  "low_level_keywords": ["Species extinction", "Habitat destruction", "Carbon emissions", "Rainforest", "Ecosystem"]
}
#############################""",
    """Example 3:

Query: "What is the role of education in reducing poverty?"
################
Output:
{
  "high_level_keywords": ["Education", "Poverty reduction", "Socioeconomic development"],
  "low_level_keywords": ["School access", "Literacy rates", "Job training", "Income inequality"]
}
#############################""",
]


PROMPTS["naive_rag_response"] = """---Role---

You are a helpful assistant responding to user query about Document Chunks provided below.

---Goal---

Generate a concise response based on Document Chunks and follow Response Rules, considering both the conversation history and the current query. Summarize all information in the provided Document Chunks, and incorporating general knowledge relevant to the Document Chunks. Do not include information not provided by Document Chunks.

When handling content with timestamps:
1. Each piece of content has a "created_at" timestamp indicating when we acquired this knowledge
2. When encountering conflicting information, consider both the content and the timestamp
3. Don't automatically prefer the most recent content - use judgment based on the context
4. For time-specific queries, prioritize temporal information in the content before considering creation timestamps

---Conversation History---
{history}

---Document Chunks---
{content_data}

---Response Rules---

- Target format and length: {response_type}
- Use markdown formatting with appropriate section headings
- Please respond in the same language as the user's question.
- Ensure the response maintains continuity with the conversation history.
- If you don't know the answer, just say so.
- Do not include information not provided by the Document Chunks."""


PROMPTS[
    "similarity_check"
] = """Please analyze the similarity between these two questions:

Question 1: {original_prompt}
Question 2: {cached_prompt}

Please evaluate whether these two questions are semantically similar, and whether the answer to Question 2 can be used to answer Question 1, provide a similarity score between 0 and 1 directly.

Similarity score criteria:
0: Completely unrelated or answer cannot be reused, including but not limited to:
   - The questions have different topics
   - The locations mentioned in the questions are different
   - The times mentioned in the questions are different
   - The specific individuals mentioned in the questions are different
   - The specific events mentioned in the questions are different
   - The background information in the questions is different
   - The key conditions in the questions are different
1: Identical and answer can be directly reused
0.5: Partially related and answer needs modification to be used
Return only a number between 0-1, without any additional content.
"""

PROMPTS["mix_rag_response"] = """---Role---

You are a helpful assistant responding to user query about Data Sources provided below.


---Goal---

Generate a concise response based on Data Sources and follow Response Rules, considering both the conversation history and the current query. Data sources contain two parts: Knowledge Graph(KG) and Document Chunks(DC). Summarize all information in the provided Data Sources, and incorporating general knowledge relevant to the Data Sources. Do not include information not provided by Data Sources.

When handling information with timestamps:
1. Each piece of information (both relationships and content) has a "created_at" timestamp indicating when we acquired this knowledge
2. When encountering conflicting information, consider both the content/relationship and the timestamp
3. Don't automatically prefer the most recent information - use judgment based on the context
4. For time-specific queries, prioritize temporal information in the content before considering creation timestamps

---Conversation History---
{history}

---Data Sources---

1. From Knowledge Graph(KG):
{kg_context}

2. From Document Chunks(DC):
{vector_context}

---Response Rules---

- Target format and length: {response_type}
- Use markdown formatting with appropriate section headings
- Please respond in the same language as the user's question.
- Ensure the response maintains continuity with the conversation history.
- Organize answer in sesctions focusing on one main point or aspect of the answer
- Use clear and descriptive section titles that reflect the content
- List up to 5 most important reference sources at the end under "References" sesction. Clearly indicating whether each source is from Knowledge Graph (KG) or Vector Data (DC), in the following format: [KG/DC] Source content
- If you don't know the answer, just say so. Do not make anything up.
- Do not include information not provided by the Data Sources."""
