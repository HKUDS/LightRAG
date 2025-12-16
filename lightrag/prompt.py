from __future__ import annotations
from typing import Any


PROMPTS: dict[str, Any] = {}

# All delimiters must be formatted as "<|UPPER_CASE_STRING|>"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|#|>"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

PROMPTS["entity_extraction_system_prompt"] = """---Role---
You are a Business Knowledge Graph Specialist responsible for extracting business entities and their relationships from company profile data.

---Context---
You will be analyzing structured company profile information that includes business details, leadership, locations, services, partnerships, and industry focus. Your goal is to extract meaningful business entities and their interconnections to build a knowledge graph.

---Instructions---
1.  **Entity Extraction & Output:**
    *   **Focus on Business-Relevant Entities:** Extract entities that are business-significant and can form meaningful connections. Focus on:
        - **Companies:** Main company, partner companies, clients, suppliers
        - **People:** Leadership (CEOs, managing directors, founders), key team members
        - **Technologies:** Specific technologies, platforms, tools, or systems mentioned
        - **Locations:** Headquarters, office locations, operational regions
        - **Services/Products:** Core services offered or products sold
        - **Industries:** Industry sectors, business domains, market segments
        - **Certifications/Standards:** NACE codes, ISO certifications, industry standards

    *   **Entity Naming Rules:**
        - Use full official names for companies (e.g., "AMR Automation GmbH", not just "AMR")
        - Use full names for people (e.g., "Ing. Lobato Jimenez Alvaro")
        - Use standardized technology names (e.g., "SCADA", "PLC Programming")
        - For locations, use "City, Country" format (e.g., "Gralla, Austria")
        - Ensure **consistent naming** - use exact same name if entity appears multiple times

    *   **Entity Details:** For each entity, extract:
        *   `entity_name`: Official, consistent name of the entity
        *   `entity_type`: One of: {entity_types}. If none apply, use `Other`.
        *   `entity_description`: Brief description of the entity's role, capabilities, or relevance in the business context

    *   **Output Format - Entities:**
        *   Format: `entity{tuple_delimiter}entity_name{tuple_delimiter}entity_type{tuple_delimiter}entity_description`

2.  **Relationship Extraction & Output:**
    *   **Focus on Business Relationships:** Only extract relationships that represent real business connections using standardized relationship types.

    *   **Standardized Relationship Keywords:** Use ONE of these standardized relationship types:
        **Organizational Structure:**
        - `is_ceo_of` - CEO/Chief Executive Officer relationship
        - `is_cto_of` - CTO/Chief Technology Officer relationship
        - `is_cfo_of` - CFO/Chief Financial Officer relationship
        - `is_founder_of` - Founder relationship
        - `is_co_founder_of` - Co-founder relationship
        - `works_for` - Employment relationship
        - `leads` - Leadership/management relationship
        - `manages` - Direct management relationship

        **Location & Geography:**
        - `headquartered_at` - Company headquarters location
        - `has_office_at` - Office or branch location
        - `operates_in` - Operational region/area
        - `located_in` - General location relationship

        **Business Relationships:**
        - `partners_with` - Business partnership
        - `is_client_of` - Client relationship
        - `supplies_to` - Supplier relationship
        - `acquired_by` / `acquired` - Acquisition relationship
        - `competes_with` - Competitive relationship
        - `collaborates_with` - Collaboration relationship

        **Technology & Products:**
        - `uses` - Uses technology/tool/platform
        - `develops` - Develops technology/product
        - `implements` - Implements technology/solution
        - `specializes_in` - Specialization in technology/domain
        - `provides` - Provides technology/service/product

        **Industry & Services:**
        - `serves_industry` - Serves specific industry sector
        - `offers_service` - Offers specific service
        - `offers_product` - Offers specific product
        - `targets_market` - Target market segment

        **Certifications & Standards:**
        - `certified_in` - Has certification or standard
        - `complies_with` - Compliance with standard/regulation

    *   **Skip Weak Relationships:** Do NOT extract relationships that are:
        - Merely mentioned without clear connection
        - Generic or trivial (e.g., "company uses computers")
        - Not verifiable from the text

    *   **Relationship Details:**
        *   `source_entity`: Source entity name (must match an extracted entity exactly)
        *   `target_entity`: Target entity name (must match an extracted entity exactly)
        *   `relationship_keywords`: ONE standardized keyword from the list above (e.g., "is_ceo_of", "headquartered_at", "provides")
        *   `relationship_description`: Clear explanation of the business connection

    *   **Output Format - Relationships:**
        *   Format: `relation{tuple_delimiter}source_entity{tuple_delimiter}target_entity{tuple_delimiter}relationship_keywords{tuple_delimiter}relationship_description`

3.  **Delimiter Usage Protocol:**
    *   The `{tuple_delimiter}` serves strictly as a field separator
    *   Do not include `{tuple_delimiter}` within field content
    *   **Correct Example:** `entity{tuple_delimiter}Tokyo{tuple_delimiter}location{tuple_delimiter}Tokyo is the capital of Japan.`

4.  **Quality Guidelines:**
    *   **Only extract entities that appear explicitly in the text** - do not infer or assume
    *   **Only create relationships between entities you extracted** - both source and target must exist
    *   **Prioritize quality over quantity** - better to have fewer, accurate relationships than many weak ones
    *   **No self-referential relationships** - a company cannot have a relationship with itself
    *   Treat relationships as **undirected** unless direction is explicitly stated

5.  **Output Order:**
    *   Output all entities first
    *   Then output all relationships
    *   Prioritize most significant relationships first

6.  **Language & Style:**
    *   The entire output (entity names, keywords, and descriptions) must be written in `{language}`.
    *   Third-person perspective only
    *   No pronouns (avoid "this company", "our", "their")
    *   Proper nouns (e.g., personal names, place names, organization names) should be retained in their original language if a proper, widely accepted translation is not available or would cause ambiguity.

7.  **Completion Signal:**
    *   Output `{completion_delimiter}` as the final line after all entities and relationships

---Examples---
{examples}
"""

PROMPTS["entity_extraction_user_prompt"] = """---Task---
Extract entities and relationships from the input text in Data to be Processed below.

---Instructions---
1.  **Strict Adherence to Format:** Strictly adhere to all format requirements for entity and relationship lists, including output order, field delimiters, and proper noun handling, as specified in the system prompt.
2.  **Output Content Only:** Output *only* the extracted list of entities and relationships. Do not include any introductory or concluding remarks, explanations, or additional text before or after the list.
3.  **Completion Signal:** Output `{completion_delimiter}` as the final line after all relevant entities and relationships have been extracted and presented.
4.  **Output Language:** Ensure the output language is {language}. Proper nouns (e.g., personal names, place names, organization names) must be kept in their original language and not translated.

---Data to be Processed---
<Entity_types>
[{entity_types}]

<Input Text>
```
{input_text}
```

<Output>
"""

PROMPTS["entity_continue_extraction_user_prompt"] = """---Task---
Based on the last extraction task, identify and extract any **missed or incorrectly formatted** entities and relationships from the input text.

---Instructions---
1.  **Strict Adherence to System Format:** Strictly adhere to all format requirements for entity and relationship lists, including output order, field delimiters, and proper noun handling, as specified in the system instructions.
2.  **Focus on Corrections/Additions:**
    *   **Do NOT** re-output entities and relationships that were **correctly and fully** extracted in the last task.
    *   If an entity or relationship was **missed** in the last task, extract and output it now according to the system format.
    *   If an entity or relationship was **truncated, had missing fields, or was otherwise incorrectly formatted** in the last task, re-output the *corrected and complete* version in the specified format.
3.  **Output Format - Entities:** Output a total of 4 fields for each entity, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `entity`.
4.  **Output Format - Relationships:** Output a total of 5 fields for each relationship, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `relation`.
5.  **Output Content Only:** Output *only* the extracted list of entities and relationships. Do not include any introductory or concluding remarks, explanations, or additional text before or after the list.
6.  **Completion Signal:** Output `{completion_delimiter}` as the final line after all relevant missing or corrected entities and relationships have been extracted and presented.
7.  **Output Language:** Ensure the output language is {language}. Proper nouns (e.g., personal names, place names, organization names) must be kept in their original language and not translated.

<Output>
"""

PROMPTS["entity_extraction_examples"] = [
    """<Entity_types>
["Company","Person","Technology","Location","Service","Product","Industry","Partnership","Certification","Project"]

<Input Text>
```
### Name
TechFlow Solutions GmbH

### Summary
TechFlow Solutions GmbH is a German software company headquartered in Munich, Bavaria. The firm specializes in industrial automation software, IoT platforms, and cloud-based monitoring systems. The company serves the manufacturing and energy sectors across Europe.

### Leadership
Dr. Maria Schmidt, CEO and Co-Founder
Thomas Mueller, CTO and Co-Founder

### Headquarters
Leopoldstrasse 45, 80802 Munich, Bavaria, Germany

### Partners
- SAP AG: Technology partnership for ERP integration
- Siemens AG: Joint development of IoT solutions
- BASF SE: Long-term client for manufacturing automation

### Portfolio
- Industrial IoT Platform: Cloud-based monitoring and analytics
- SCADA Systems: Process control and automation
- Predictive Maintenance: AI-driven maintenance solutions

### Industries Served
Manufacturing, Energy, Process Industries

### Technologies
IoT, Cloud Computing, Machine Learning, SCADA, OPC UA
```

<Output>
entity{tuple_delimiter}TechFlow Solutions GmbH{tuple_delimiter}company{tuple_delimiter}TechFlow Solutions GmbH is a German software company specializing in industrial automation software, IoT platforms, and cloud-based monitoring systems.
entity{tuple_delimiter}Dr. Maria Schmidt{tuple_delimiter}person{tuple_delimiter}Dr. Maria Schmidt is the CEO and Co-Founder of TechFlow Solutions GmbH.
entity{tuple_delimiter}Thomas Mueller{tuple_delimiter}person{tuple_delimiter}Thomas Mueller is the CTO and Co-Founder of TechFlow Solutions GmbH.
entity{tuple_delimiter}Munich, Germany{tuple_delimiter}location{tuple_delimiter}Munich, Bavaria, Germany is the headquarters location of TechFlow Solutions GmbH.
entity{tuple_delimiter}SAP AG{tuple_delimiter}company{tuple_delimiter}SAP AG is a technology partner of TechFlow Solutions GmbH for ERP integration.
entity{tuple_delimiter}Siemens AG{tuple_delimiter}company{tuple_delimiter}Siemens AG is a partner company collaborating with TechFlow Solutions on IoT solution development.
entity{tuple_delimiter}BASF SE{tuple_delimiter}company{tuple_delimiter}BASF SE is a long-term client of TechFlow Solutions for manufacturing automation.
entity{tuple_delimiter}Industrial IoT Platform{tuple_delimiter}product{tuple_delimiter}Industrial IoT Platform is a cloud-based monitoring and analytics solution offered by TechFlow Solutions.
entity{tuple_delimiter}SCADA Systems{tuple_delimiter}technology{tuple_delimiter}SCADA Systems are process control and automation technologies provided by TechFlow Solutions.
entity{tuple_delimiter}Predictive Maintenance{tuple_delimiter}service{tuple_delimiter}Predictive Maintenance is an AI-driven maintenance solution service offered by TechFlow Solutions.
entity{tuple_delimiter}Manufacturing{tuple_delimiter}industry{tuple_delimiter}Manufacturing is a key industry sector served by TechFlow Solutions.
entity{tuple_delimiter}Energy{tuple_delimiter}industry{tuple_delimiter}Energy is a key industry sector served by TechFlow Solutions.
relation{tuple_delimiter}Dr. Maria Schmidt{tuple_delimiter}TechFlow Solutions GmbH{tuple_delimiter}is_ceo_of{tuple_delimiter}Dr. Maria Schmidt is the CEO and Co-Founder of TechFlow Solutions GmbH.
relation{tuple_delimiter}Thomas Mueller{tuple_delimiter}TechFlow Solutions GmbH{tuple_delimiter}is_cto_of{tuple_delimiter}Thomas Mueller is the CTO and Co-Founder of TechFlow Solutions GmbH.
relation{tuple_delimiter}Dr. Maria Schmidt{tuple_delimiter}TechFlow Solutions GmbH{tuple_delimiter}is_co_founder_of{tuple_delimiter}Dr. Maria Schmidt is a co-founder of TechFlow Solutions GmbH.
relation{tuple_delimiter}Thomas Mueller{tuple_delimiter}TechFlow Solutions GmbH{tuple_delimiter}is_co_founder_of{tuple_delimiter}Thomas Mueller is a co-founder of TechFlow Solutions GmbH.
relation{tuple_delimiter}TechFlow Solutions GmbH{tuple_delimiter}Munich, Germany{tuple_delimiter}headquartered_at{tuple_delimiter}TechFlow Solutions GmbH is headquartered in Munich, Bavaria, Germany.
relation{tuple_delimiter}TechFlow Solutions GmbH{tuple_delimiter}SAP AG{tuple_delimiter}partners_with{tuple_delimiter}TechFlow Solutions has a technology partnership with SAP AG for ERP integration.
relation{tuple_delimiter}TechFlow Solutions GmbH{tuple_delimiter}Siemens AG{tuple_delimiter}collaborates_with{tuple_delimiter}TechFlow Solutions collaborates with Siemens AG on joint development of IoT solutions.
relation{tuple_delimiter}BASF SE{tuple_delimiter}TechFlow Solutions GmbH{tuple_delimiter}is_client_of{tuple_delimiter}BASF SE is a long-term client of TechFlow Solutions for manufacturing automation solutions.
relation{tuple_delimiter}TechFlow Solutions GmbH{tuple_delimiter}Industrial IoT Platform{tuple_delimiter}offers_product{tuple_delimiter}TechFlow Solutions offers Industrial IoT Platform as a cloud-based monitoring and analytics solution.
relation{tuple_delimiter}TechFlow Solutions GmbH{tuple_delimiter}SCADA Systems{tuple_delimiter}provides{tuple_delimiter}TechFlow Solutions provides SCADA Systems for process control and automation.
relation{tuple_delimiter}TechFlow Solutions GmbH{tuple_delimiter}Predictive Maintenance{tuple_delimiter}offers_service{tuple_delimiter}TechFlow Solutions offers Predictive Maintenance as an AI-driven service.
relation{tuple_delimiter}TechFlow Solutions GmbH{tuple_delimiter}Manufacturing{tuple_delimiter}serves_industry{tuple_delimiter}TechFlow Solutions serves the Manufacturing industry sector.
relation{tuple_delimiter}TechFlow Solutions GmbH{tuple_delimiter}Energy{tuple_delimiter}serves_industry{tuple_delimiter}TechFlow Solutions serves the Energy industry sector.
{completion_delimiter}

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
