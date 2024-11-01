GRAPH_FIELD_SEP = "<SEP>"

PROMPTS = {}

PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

PROMPTS["DEFAULT_ENTITY_TYPES"] = ["organization", "person", "geo", "event"]

PROMPTS["DEFAULT_ENTITY_TYPES"] = ["disease", "biomarker", "treatment", "test", "organization", "person"]

PROMPTS["entity_extraction"] = """-Goal-
Given a health-related text document and a list of entity types, identify all entities of those types from the text and any relationships among the identified entities.

-Steps-
1. Identify all entities. For each identified entity, extract the following details:
   - entity_name: Name of the entity, capitalized
   - entity_type: One of the following types: [{entity_types}]
   - entity_description: Detailed description of the entity’s attributes, clinical significance, and health-related context
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clinically related*.
   For each related entity pair, extract the following details:
   - source_entity: Name of the source entity, as identified in step 1
   - target_entity: Name of the target entity, as identified in step 1
   - relationship_description: Explanation of the clinical relationship or interaction
   - relationship_strength: A numeric score indicating the strength of the clinical relationship (e.g., 1-10, where 10 is strongest)
   - relationship_keywords: One or more high-level keywords summarizing the relationship (e.g., "risk factor," "diagnostic marker," "therapeutic target")
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. Identify high-level keywords that summarize the main health concepts, conditions, or treatments covered in the document.
Format content-level keywords as ("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. Return output in English as a single list of all entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

5. When finished, output {completion_delimiter}

######################
-Examples-
######################
Example 1:

Entity_types: [disease, treatment, biomarker, test, organization]
Text:
The patient was diagnosed with Type 2 diabetes, a chronic disease marked by high blood sugar levels. HbA1c testing is commonly used to monitor blood glucose control in these patients. Metformin is the first-line treatment for managing diabetes, often combined with lifestyle modifications like diet and exercise.

################
Output:
("entity"{tuple_delimiter}"Type 2 Diabetes"{tuple_delimiter}"disease"{tuple_delimiter}"A chronic disease characterized by high blood sugar levels, requiring ongoing management through medication and lifestyle adjustments."){record_delimiter}
("entity"{tuple_delimiter}"HbA1c Test"{tuple_delimiter}"test"{tuple_delimiter}"A blood test used to measure average blood glucose levels over three months, important for monitoring diabetes management."){record_delimiter}
("entity"{tuple_delimiter}"Metformin"{tuple_delimiter}"treatment"{tuple_delimiter}"A common first-line medication used to control blood sugar in Type 2 diabetes, often used alongside lifestyle modifications."){record_delimiter}
("relationship"{tuple_delimiter}"Type 2 Diabetes"{tuple_delimiter}"HbA1c Test"{tuple_delimiter}"The HbA1c test is used to monitor blood glucose control in patients with Type 2 diabetes."{tuple_delimiter}"monitoring, diagnostic"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Type 2 Diabetes"{tuple_delimiter}"Metformin"{tuple_delimiter}"Metformin is a first-line treatment option for managing Type 2 diabetes."{tuple_delimiter}"therapeutic, medication"{tuple_delimiter}8){record_delimiter}
("content_keywords"{tuple_delimiter}"diabetes, glucose control, first-line treatment, chronic disease"){completion_delimiter}

######################
-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:
"""

PROMPTS[
    "summarize_entity_descriptions"
] = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or two entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we the have full context.

#######
-Data-
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

PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."

PROMPTS["rag_response"] = """---Role---

You are a helpful assistant responding to questions about data in the tables provided.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}

---Data tables---

{context_data}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""

PROMPTS["keywords_extraction"] = """---Role---

You are a helpful assistant tasked with identifying both high-level and low-level keywords in the user's query.

---Goal---

Given the query, list both high-level and low-level keywords. High-level keywords focus on overarching concepts or themes, while low-level keywords focus on specific entities, details, or concrete terms.

---Instructions---

- Output the keywords in JSON format.
- The JSON should have two keys:
  - "high_level_keywords" for overarching concepts or themes.
  - "low_level_keywords" for specific entities or details.

######################
-Examples-
######################
Example 1:

Query: "How does international trade influence global economic stability?"
################
Output:
{{
  "high_level_keywords": ["International trade", "Global economic stability", "Economic impact"],
  "low_level_keywords": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]
}}
#############################
Example 2:

Query: "What are the environmental consequences of deforestation on biodiversity?"
################
Output:
{{
  "high_level_keywords": ["Environmental consequences", "Deforestation", "Biodiversity loss"],
  "low_level_keywords": ["Species extinction", "Habitat destruction", "Carbon emissions", "Rainforest", "Ecosystem"]
}}
#############################
Example 3:

Query: "What is the role of education in reducing poverty?"
################
Output:
{{
  "high_level_keywords": ["Education", "Poverty reduction", "Socioeconomic development"],
  "low_level_keywords": ["School access", "Literacy rates", "Job training", "Income inequality"]
}}
#############################
-Real Data-
######################
Query: {query}
######################
Output:

"""

PROMPTS["naive_rag_response"] = """You're a helpful assistant
Below are the knowledge you know:
{content_data}
---
If you don't know the answer or if the provided knowledge do not contain sufficient information to provide an answer, just say so. Do not make anything up.
Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.
---Target response length and format---
{response_type}
"""
