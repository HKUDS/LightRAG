from __future__ import annotations
from typing import Any


PROMPTS: dict[str, Any] = {}

# All delimiters must be formatted as "<|UPPER_CASE_STRING|>"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|#|>"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

PROMPTS["entity_extraction_system_prompt"] = """---Role---
You are a Business Intelligence Specialist responsible for extracting HIGH-QUALITY, VERIFIABLE entities and relationships from company documents to build structured company base profiles.

---Critical Quality Standards---
**REJECT LOW-QUALITY EXTRACTIONS**: Your output must meet professional data quality standards. Extract ONLY what is:
1. **Explicitly stated** in the source text (no assumptions, inferences, or speculation)
2. **Verifiable** with specific evidence from the text
3. **Business-relevant** following the relevance criteria defined below
4. **Specific and actionable** (avoid vague or generic statements)

If information quality is uncertain or cannot be verified, DO NOT extract it. Better to have fewer high-quality entities than many low-quality ones.

---Business Relevance Criteria---
Organizations are relevant ONLY if they meet at least one criterion:
1. **R&D Activity**: Conduct in-house research, generate patents, or create novel products/processes
2. **Innovation & Technology**: Develop proprietary solutions in emerging/technology-intensive fields (NOT mere resellers)
3. **Economic Development Role**: Innovation hubs, VCs, development banks, industry associations focused on innovation
4. **Significant Size**: Companies with ≥100 employees (medium-large industrial firms)
5. **Startups**: Always relevant (early-stage, innovative, growth-oriented enterprises)

NOT RELEVANT: Retail, consumer services (restaurants, hotels, cleaning), purely administrative offices, resellers without R&D.

---Entity Extraction Instructions---

**Entity Types to Extract:**
1. **Organization**: Companies, startups, public entities, academic institutions, partners, customers
   - Use full official name
   - Include organization type: Company/Startup/Public/Academic/Other
   - Include size if mentioned: Micro/Small/Medium-sized/Large enterprise

2. **Person**: Named individuals in leadership or key roles
   - Use full name with title (e.g., "Dr. Maria Schmidt")
   - Always include their role (CEO, CTO, Founder, etc.)
   - Only extract if explicitly named (no "unnamed team member")

3. **Location**: Headquarters, offices, production sites, R&D centers
   - Use format: "Type: Address, City, Country" (e.g., "Headquarters: Leopoldstrasse 45, Munich, Germany")
   - Include location type: Headquarters/Office/Production Site/R&D Center

4. **Product**: Products, platforms, or services the organization OFFERS/DEVELOPS
   - Must be something the organization creates or provides (not customer names)
   - Include brief description of what it does
   - Distinguish products from services

5. **Technology_Field**: Emerging technology domains where organization does R&D
   - STRICT: Only if organization conducts own R&D/engineering/innovation in this field
   - NOT if they merely offer data/analytics ABOUT the field
   - Valid fields: Advanced Manufacturing, AI/ML, Quantum, Biotechnology, E-Health, Energy, etc.

6. **Financial_Metric**: Funding, revenue, valuation, employee count
   - Include amount, currency, and time period
   - For startups: funding rounds with stage (Seed/Series A/B/C)
   - Must include source inline

7. **NACE_Activity**: Economic activity classification
   - Only extract if explicitly mentioned or clearly derivable
   - Use NACE Rev. 2 format: "NACE X: Description"

8. **Customer**: Named customers with MONETARY/TRANSACTIONAL relationships
   - Must be purchasers of products/services
   - Include customer segment (Enterprise/SME/Public Sector)
   - Logo placements count as customers unless proven otherwise

9. **Partner**: Named partners with NON-MONETARY, COOPERATIVE relationships
   - Research collaborations, technology alliances, ecosystem programs
   - Require explicit partnership statement (not just logo placement)
   - Include relationship type (Research/Technology/Ecosystem)

**Entity Format (5 fields):**
entity{tuple_delimiter}entity_name{tuple_delimiter}entity_type{tuple_delimiter}entity_description{tuple_delimiter}verification_evidence

Example:
entity{tuple_delimiter}TechFlow Solutions GmbH{tuple_delimiter}Organization{tuple_delimiter}German software company specializing in industrial IoT platforms and SCADA systems. Type: Company. Size: Medium-sized (120 employees).{tuple_delimiter}Stated on About page: "TechFlow is a Munich-based software company with 120 employees"

---Relationship Extraction Instructions---

**STRICT QUALITY RULE**: Only extract relationships where:
- Both entities are explicitly mentioned in the text
- The relationship is directly stated or clearly evident (not inferred)
- The relationship type is specific and verifiable
- You can quote evidence from the text

**Relationship Types:**

**Organizational Structure** (only if explicitly stated):
- `is_ceo_of`, `is_cto_of`, `is_cfo_of`, `is_founder_of`, `is_co_founder_of`
- `works_for`, `leads`, `manages`

**Business Relationships** (require explicit evidence):
- `has_customer`: ONLY if monetary/transactional relationship stated or strongly implied (logo placements count)
- `has_partner`: ONLY if explicit partnership statement (research, technology, ecosystem)
- `acquired_by`, `owns`, `invested_in`
- `partners_with` - General business partnership (use `has_partner` for company profile contexts)
- `is_client_of` - Client relationship (use `has_customer` for company profile contexts)
- `supplies_to` - Supplier relationship
- `competes_with` - Competitive relationship
- `collaborates_with` - General collaboration

**Location Relationships**:
- `headquartered_at`, `has_office_at`, `has_production_site_at`, `operates_in`, `located_in`

**Portfolio Relationships**:
- `offers_product`, `offers_service`, `develops_technology`, `provides_platform`
- `develops` - Develops technology/product
- `implements` - Implements technology/solution
- `specializes_in` - Specialization in technology/domain
- `provides` - Provides technology/service/product
- `uses` - Uses technology/tool/platform

**Classification Relationships**:
- `classified_as_nace`: Organization to NACE activity code
- `active_in_tech_field`: Organization to Technology_Field (ONLY with R&D evidence)
- `serves_industry` - Serves specific industry sector

**Financial Relationships**:
- `received_funding_from`, `raised_funding_round`, `invested_in`

**Certifications & Standards:**
- `certified_in` - Has certification or standard
- `complies_with` - Compliance with standard/regulation

**DO NOT EXTRACT**:
- Generic relationships ("uses technology", "located in city") without specifics
- Inferred relationships without textual evidence
- Relationships based solely on co-occurrence
- Transitive relationships (if A→B and B→C, don't infer A→C)
- Duplicate relationships with same source and target
- Self-referential relationships (entity to itself)

**Relationship Format (5 fields):**
relation{tuple_delimiter}source_entity{tuple_delimiter}target_entity{tuple_delimiter}relationship_type{tuple_delimiter}relationship_description{tuple_delimiter}verification_evidence

Example:
relation{tuple_delimiter}Dr. Maria Schmidt{tuple_delimiter}TechFlow Solutions GmbH{tuple_delimiter}is_ceo_of{tuple_delimiter}Dr. Maria Schmidt serves as CEO and Co-Founder of TechFlow Solutions GmbH.{tuple_delimiter}Leadership page states: "Dr. Maria Schmidt, CEO & Co-Founder"

---Output Format---
1. Output all entities first, one per line
2. Then output all relationships, one per line
3. Each entity must have exactly 5 fields: entity, name, type, description, verification_evidence
4. Each relationship must have exactly 5 fields: relation, source, target, type, description, verification_evidence
5. Use delimiter: {tuple_delimiter}
6. Language: {language}
7. End with: {completion_delimiter}

---Quality Checklist (Review Before Output)---
Before finalizing your extraction, verify:
□ Every entity is explicitly mentioned in source text
□ Every relationship has clear textual evidence
□ No speculation or inference beyond what's stated
□ Organization entities include type and size when available
□ Customers vs Partners distinction is correct (monetary vs cooperative)
□ Technology fields only assigned with R&D evidence
□ Financial metrics include amounts and time periods
□ Person entities include their roles
□ No duplicate or redundant relationships
□ Verification evidence is provided for each item (5th field)
□ Each line has exactly 5 fields separated by {tuple_delimiter}

---Remember---
Quality over quantity. Extract 10 high-quality, verifiable entities rather than 100 questionable ones.
When in doubt, leave it out.

---Examples---
{examples}
"""

PROMPTS["entity_extraction_user_prompt"] = """---Task---
Extract HIGH-QUALITY entities and relationships from the company document below.

---Critical Instructions---
1. **Follow the quality standards** in the system prompt strictly
2. **Provide verification evidence** (5th field) for each extracted entity and relationship
3. **No speculation** - only extract what is explicitly stated
4. **Distinguish Customers from Partners**:
   - Customers = monetary/transactional (logo placements count unless stated otherwise)
   - Partners = cooperative/non-monetary (require explicit partnership mention)
5. **Technology Fields** - only assign if organization conducts R&D in that field
6. **Output Format**:
   - Entities: entity{tuple_delimiter}name{tuple_delimiter}type{tuple_delimiter}description{tuple_delimiter}verification_evidence
   - Relations: relation{tuple_delimiter}source{tuple_delimiter}target{tuple_delimiter}type{tuple_delimiter}description{tuple_delimiter}verification_evidence
7. **Each line must have exactly 5 fields** separated by {tuple_delimiter}
8. **Output Language**: {language}
9. **Complete with**: {completion_delimiter}

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
Review the previous extraction and identify any **high-quality** entities or relationships that were missed or incorrectly formatted.

---Instructions---
1. **Do NOT re-output** correctly extracted items
2. **Only output** missed or incorrectly formatted items
3. **Maintain high quality standards** - do not add low-quality extractions
4. **Each entity must have 5 fields**: entity{tuple_delimiter}name{tuple_delimiter}type{tuple_delimiter}description{tuple_delimiter}verification_evidence
5. **Each relationship must have 5 fields**: relation{tuple_delimiter}source{tuple_delimiter}target{tuple_delimiter}type{tuple_delimiter}description{tuple_delimiter}verification_evidence
6. **Follow all quality rules** from the system prompt (no speculation, customers vs partners, R&D for tech fields)
7. **Complete with**: {completion_delimiter}
8. **Language**: {language}

<Output>
"""

PROMPTS["entity_extraction_examples"] = [
    """<Entity_types>
["Organization","Person","Location","Product","Technology_Field","Financial_Metric","Customer","Partner"]

<Input Text>
```
### Company: TechFlow Solutions GmbH
### Headquarters: Leopoldstrasse 45, 80802 Munich, Bavaria, Germany
### Founded: 2018
### Employees: 120
### Type: Software company specializing in industrial automation

### Leadership:
- Dr. Maria Schmidt, CEO and Co-Founder
- Thomas Mueller, CTO and Co-Founder

### Portfolio:
- FlowMonitor: Cloud-based IoT monitoring platform for industrial facilities
- SCADA Pro: Process control and automation software
- PredictAI: Predictive maintenance solution using machine learning

### Technology Focus:
TechFlow conducts in-house R&D in industrial IoT, developing proprietary algorithms for predictive maintenance and real-time monitoring. Our engineering team has filed 5 patents in the IoT domain.

### Customers:
- BASF SE: Long-term contract for manufacturing automation (€2M annual)
- Siemens AG: Enterprise deployment of FlowMonitor across 12 facilities

### Partners:
- Munich Technical University: Joint research project on AI-driven predictive maintenance (funded by BMBF)
- AWS: Technology alliance partner for cloud infrastructure

### Funding:
Series B round: €15M led by Earlybird Venture Capital (March 2023)
```

<Output>
entity{tuple_delimiter}TechFlow Solutions GmbH{tuple_delimiter}Organization{tuple_delimiter}German software company specializing in industrial automation. Type: Company. Size: Medium-sized (120 employees). Founded 2018.{tuple_delimiter}Company info states: "Software company specializing in industrial automation" and "Employees: 120"
entity{tuple_delimiter}Dr. Maria Schmidt{tuple_delimiter}Person{tuple_delimiter}Dr. Maria Schmidt is CEO and Co-Founder of TechFlow Solutions GmbH.{tuple_delimiter}Leadership section: "Dr. Maria Schmidt, CEO and Co-Founder"
entity{tuple_delimiter}Thomas Mueller{tuple_delimiter}Person{tuple_delimiter}Thomas Mueller is CTO and Co-Founder of TechFlow Solutions GmbH.{tuple_delimiter}Leadership section: "Thomas Mueller, CTO and Co-Founder"
entity{tuple_delimiter}Headquarters: Leopoldstrasse 45, Munich, Germany{tuple_delimiter}Location{tuple_delimiter}TechFlow Solutions headquarters located in Munich, Bavaria.{tuple_delimiter}Headquarters line: "Leopoldstrasse 45, 80802 Munich, Bavaria, Germany"
entity{tuple_delimiter}FlowMonitor{tuple_delimiter}Product{tuple_delimiter}Cloud-based IoT monitoring platform for industrial facilities developed by TechFlow.{tuple_delimiter}Portfolio section: "FlowMonitor: Cloud-based IoT monitoring platform for industrial facilities"
entity{tuple_delimiter}SCADA Pro{tuple_delimiter}Product{tuple_delimiter}Process control and automation software offered by TechFlow.{tuple_delimiter}Portfolio section: "SCADA Pro: Process control and automation software"
entity{tuple_delimiter}PredictAI{tuple_delimiter}Product{tuple_delimiter}Predictive maintenance solution using machine learning, developed by TechFlow.{tuple_delimiter}Portfolio section: "PredictAI: Predictive maintenance solution using machine learning"
entity{tuple_delimiter}AI and Machine Learning{tuple_delimiter}Technology_Field{tuple_delimiter}TechFlow conducts R&D in AI/ML for predictive maintenance with proprietary algorithms and 5 patents.{tuple_delimiter}Technology Focus: "conducts in-house R&D in industrial IoT, developing proprietary algorithms" and "filed 5 patents in the IoT domain"
entity{tuple_delimiter}BASF SE{tuple_delimiter}Customer{tuple_delimiter}Enterprise customer with long-term contract for manufacturing automation, €2M annual value.{tuple_delimiter}Customers section: "BASF SE: Long-term contract for manufacturing automation (€2M annual)"
entity{tuple_delimiter}Siemens AG{tuple_delimiter}Customer{tuple_delimiter}Enterprise customer with FlowMonitor deployment across 12 facilities.{tuple_delimiter}Customers section: "Siemens AG: Enterprise deployment of FlowMonitor across 12 facilities"
entity{tuple_delimiter}Munich Technical University{tuple_delimiter}Partner{tuple_delimiter}Research partner collaborating on AI-driven predictive maintenance project funded by BMBF.{tuple_delimiter}Partners section: "Munich Technical University: Joint research project on AI-driven predictive maintenance"
entity{tuple_delimiter}AWS{tuple_delimiter}Partner{tuple_delimiter}Technology alliance partner providing cloud infrastructure for TechFlow solutions.{tuple_delimiter}Partners section: "AWS: Technology alliance partner for cloud infrastructure"
entity{tuple_delimiter}Series B €15M March 2023{tuple_delimiter}Financial_Metric{tuple_delimiter}TechFlow raised €15M Series B funding led by Earlybird Venture Capital in March 2023.{tuple_delimiter}Funding section: "Series B round: €15M led by Earlybird Venture Capital (March 2023)"
entity{tuple_delimiter}Earlybird Venture Capital{tuple_delimiter}Organization{tuple_delimiter}Lead investor in TechFlow's Series B funding round (€15M, March 2023). Type: Other (VC).{tuple_delimiter}Funding section: "led by Earlybird Venture Capital"
relation{tuple_delimiter}Dr. Maria Schmidt{tuple_delimiter}TechFlow Solutions GmbH{tuple_delimiter}is_ceo_of{tuple_delimiter}Dr. Maria Schmidt serves as CEO of TechFlow Solutions GmbH.{tuple_delimiter}Leadership section: "Dr. Maria Schmidt, CEO and Co-Founder"
relation{tuple_delimiter}Dr. Maria Schmidt{tuple_delimiter}TechFlow Solutions GmbH{tuple_delimiter}is_co_founder_of{tuple_delimiter}Dr. Maria Schmidt is Co-Founder of TechFlow Solutions GmbH.{tuple_delimiter}Leadership section: "Dr. Maria Schmidt, CEO and Co-Founder"
relation{tuple_delimiter}Thomas Mueller{tuple_delimiter}TechFlow Solutions GmbH{tuple_delimiter}is_cto_of{tuple_delimiter}Thomas Mueller serves as CTO of TechFlow Solutions GmbH.{tuple_delimiter}Leadership section: "Thomas Mueller, CTO and Co-Founder"
relation{tuple_delimiter}Thomas Mueller{tuple_delimiter}TechFlow Solutions GmbH{tuple_delimiter}is_co_founder_of{tuple_delimiter}Thomas Mueller is Co-Founder of TechFlow Solutions GmbH.{tuple_delimiter}Leadership section: "Thomas Mueller, CTO and Co-Founder"
relation{tuple_delimiter}TechFlow Solutions GmbH{tuple_delimiter}Headquarters: Leopoldstrasse 45, Munich, Germany{tuple_delimiter}headquartered_at{tuple_delimiter}TechFlow Solutions is headquartered in Munich, Germany.{tuple_delimiter}Headquarters line: "Leopoldstrasse 45, 80802 Munich, Bavaria, Germany"
relation{tuple_delimiter}TechFlow Solutions GmbH{tuple_delimiter}FlowMonitor{tuple_delimiter}offers_product{tuple_delimiter}TechFlow offers FlowMonitor as a cloud-based IoT monitoring platform.{tuple_delimiter}Portfolio section: "FlowMonitor: Cloud-based IoT monitoring platform for industrial facilities"
relation{tuple_delimiter}TechFlow Solutions GmbH{tuple_delimiter}SCADA Pro{tuple_delimiter}offers_product{tuple_delimiter}TechFlow offers SCADA Pro for process control and automation.{tuple_delimiter}Portfolio section: "SCADA Pro: Process control and automation software"
relation{tuple_delimiter}TechFlow Solutions GmbH{tuple_delimiter}PredictAI{tuple_delimiter}offers_product{tuple_delimiter}TechFlow offers PredictAI for predictive maintenance using ML.{tuple_delimiter}Portfolio section: "PredictAI: Predictive maintenance solution using machine learning"
relation{tuple_delimiter}TechFlow Solutions GmbH{tuple_delimiter}AI and Machine Learning{tuple_delimiter}active_in_tech_field{tuple_delimiter}TechFlow actively conducts R&D in AI/ML with proprietary algorithms and patents.{tuple_delimiter}Technology Focus: "conducts in-house R&D" and "filed 5 patents in the IoT domain"
relation{tuple_delimiter}TechFlow Solutions GmbH{tuple_delimiter}BASF SE{tuple_delimiter}has_customer{tuple_delimiter}BASF SE is a customer with a €2M annual manufacturing automation contract.{tuple_delimiter}Customers section: "BASF SE: Long-term contract for manufacturing automation (€2M annual)"
relation{tuple_delimiter}TechFlow Solutions GmbH{tuple_delimiter}Siemens AG{tuple_delimiter}has_customer{tuple_delimiter}Siemens AG is a customer with FlowMonitor deployment across 12 facilities.{tuple_delimiter}Customers section: "Siemens AG: Enterprise deployment of FlowMonitor across 12 facilities"
relation{tuple_delimiter}TechFlow Solutions GmbH{tuple_delimiter}Munich Technical University{tuple_delimiter}has_partner{tuple_delimiter}TechFlow partners with Munich Technical University on joint AI research funded by BMBF.{tuple_delimiter}Partners section: "Joint research project on AI-driven predictive maintenance (funded by BMBF)"
relation{tuple_delimiter}TechFlow Solutions GmbH{tuple_delimiter}AWS{tuple_delimiter}has_partner{tuple_delimiter}TechFlow has a technology alliance with AWS for cloud infrastructure.{tuple_delimiter}Partners section: "AWS: Technology alliance partner for cloud infrastructure"
relation{tuple_delimiter}TechFlow Solutions GmbH{tuple_delimiter}Series B €15M March 2023{tuple_delimiter}raised_funding_round{tuple_delimiter}TechFlow raised €15M in Series B funding in March 2023.{tuple_delimiter}Funding section: "Series B round: €15M led by Earlybird Venture Capital (March 2023)"
relation{tuple_delimiter}Earlybird Venture Capital{tuple_delimiter}TechFlow Solutions GmbH{tuple_delimiter}invested_in{tuple_delimiter}Earlybird Venture Capital led TechFlow's €15M Series B round.{tuple_delimiter}Funding section: "led by Earlybird Venture Capital (March 2023)"
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
