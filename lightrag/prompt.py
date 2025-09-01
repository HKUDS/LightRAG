from __future__ import annotations
from typing import Any


PROMPTS: dict[str, Any] = {}

PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

PROMPTS["DEFAULT_USER_PROMPT"] = "n/a"

PROMPTS["entity_extraction"] = """---Goal---
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.
Use {language} as output language.

---Steps---
1. Recognizing definitively conceptualized entities in text. For each identified entity, extract the following information:
- entity_name: Name of the entity, use same language as input text. If English, capitalized the name
- entity_type: One of the following types: [{entity_types}]. If the entity doesn't clearly fit any category, classify it as "Other".
- entity_description: Provide a comprehensive description of the entity's attributes and activities based on the information present in the input text. Do not add external knowledge.

2. Format each entity as:
("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

3. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are directly and clearly related based on the text. Unsubstantiated relationships must be excluded from the output.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_keywords: one or more high-level key words that summarize the overarching nature of the relationship, focusing on concepts or themes rather than specific details
- relationship_description: Explain the nature of the relationship between the source and target entities, providing a clear rationale for their connection

4. Format each relationship as:
("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_description>)

5. Use `{tuple_delimiter}` as field delimiter. Use `{record_delimiter}` as the list delimiter. Ensure no spaces are added around the delimiters.

6. When finished, output `{completion_delimiter}`

7. Return identified entities and relationships in {language}.

---Quality Guidelines---
- Only extract entities that are clearly defined and meaningful in the context
- Avoid over-interpretation; stick to what is explicitly stated in the text
- Include specific numerical data in entity name when relevant
- Ensure entity names are consistent throughout the extraction

---Examples---
{examples}

---Real Data---
Entity_types: [{entity_types}]
Text:
```
{input_text}
```

---Output---
Output:
"""

PROMPTS["entity_extraction_examples"] = [
    """------Example 1------

Entity_types: [organization,person,equiment,product,technology,location,event,category]
Text:
```
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.

Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. "If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us."

The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.

It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths
```

Output:
(entity{tuple_delimiter}Alex{tuple_delimiter}person{tuple_delimiter}Alex is a character who experiences frustration and is observant of the dynamics among other characters.){record_delimiter}
(entity{tuple_delimiter}Taylor{tuple_delimiter}person{tuple_delimiter}Taylor is portrayed with authoritarian certainty and shows a moment of reverence towards a device, indicating a change in perspective.){record_delimiter}
(entity{tuple_delimiter}Jordan{tuple_delimiter}person{tuple_delimiter}Jordan shares a commitment to discovery and has a significant interaction with Taylor regarding a device.){record_delimiter}
(entity{tuple_delimiter}Cruz{tuple_delimiter}person{tuple_delimiter}Cruz is associated with a vision of control and order, influencing the dynamics among other characters.){record_delimiter}
(entity{tuple_delimiter}The Device{tuple_delimiter}equiment{tuple_delimiter}The Device is central to the story, with potential game-changing implications, and is revered by Taylor.){record_delimiter}
(relationship{tuple_delimiter}Alex{tuple_delimiter}Taylor{tuple_delimiter}power dynamics, observation{tuple_delimiter}Alex observes Taylor's authoritarian behavior and notes changes in Taylor's attitude toward the device.){record_delimiter}
(relationship{tuple_delimiter}Alex{tuple_delimiter}Jordan{tuple_delimiter}shared goals, rebellion{tuple_delimiter}Alex and Jordan share a commitment to discovery, which contrasts with Cruz's vision.){record_delimiter}
(relationship{tuple_delimiter}Taylor{tuple_delimiter}Jordan{tuple_delimiter}conflict resolution, mutual respect{tuple_delimiter}Taylor and Jordan interact directly regarding the device, leading to a moment of mutual respect and an uneasy truce.){record_delimiter}
(relationship{tuple_delimiter}Jordan{tuple_delimiter}Cruz{tuple_delimiter}ideological conflict, rebellion{tuple_delimiter}Jordan's commitment to discovery is in rebellion against Cruz's vision of control and order.){record_delimiter}
(relationship{tuple_delimiter}Taylor{tuple_delimiter}The Device{tuple_delimiter}reverence, technological significance{tuple_delimiter}Taylor shows reverence towards the device, indicating its importance and potential impact.){record_delimiter}
{completion_delimiter}

""",
    """------Example 2------

Entity_types: [organization,person,equiment,product,technology,location,event,category]
Text:
```
Stock markets faced a sharp downturn today as tech giants saw significant declines, with the Global Tech Index dropping by 3.4% in midday trading. Analysts attribute the selloff to investor concerns over rising interest rates and regulatory uncertainty.

Among the hardest hit, Nexon Technologies saw its stock plummet by 7.8% after reporting lower-than-expected quarterly earnings. In contrast, Omega Energy posted a modest 2.1% gain, driven by rising oil prices.

Meanwhile, commodity markets reflected a mixed sentiment. Gold futures rose by 1.5%, reaching $2,080 per ounce, as investors sought safe-haven assets. Crude oil prices continued their rally, climbing to $87.60 per barrel, supported by supply constraints and strong demand.

Financial experts are closely watching the Federal Reserve's next move, as speculation grows over potential rate hikes. The upcoming policy announcement is expected to influence investor confidence and overall market stability.
```

Output:
(entity{tuple_delimiter}Global Tech Index{tuple_delimiter}category{tuple_delimiter}The Global Tech Index tracks the performance of major technology stocks and experienced a 3.4% decline today.){record_delimiter}
(entity{tuple_delimiter}Nexon Technologies{tuple_delimiter}organization{tuple_delimiter}Nexon Technologies is a tech company that saw its stock decline by 7.8% after disappointing earnings.){record_delimiter}
(entity{tuple_delimiter}Omega Energy{tuple_delimiter}organization{tuple_delimiter}Omega Energy is an energy company that gained 2.1% in stock value due to rising oil prices.){record_delimiter}
(entity{tuple_delimiter}Gold Futures{tuple_delimiter}product{tuple_delimiter}Gold futures rose by 1.5%, indicating increased investor interest in safe-haven assets.){record_delimiter}
(entity{tuple_delimiter}Crude Oil{tuple_delimiter}product{tuple_delimiter}Crude oil prices rose to $87.60 per barrel due to supply constraints and strong demand.){record_delimiter}
(entity{tuple_delimiter}Market Selloff{tuple_delimiter}category{tuple_delimiter}Market selloff refers to the significant decline in stock values due to investor concerns over interest rates and regulations.){record_delimiter}
(entity{tuple_delimiter}Federal Reserve Policy Announcement{tuple_delimiter}category{tuple_delimiter}The Federal Reserve's upcoming policy announcement is expected to impact investor confidence and market stability.){record_delimiter}
(entity{tuple_delimiter}3.4% Decline{tuple_delimiter}category{tuple_delimiter}The Global Tech Index experienced a 3.4% decline in midday trading.){record_delimiter}
(relationship{tuple_delimiter}Global Tech Index{tuple_delimiter}Market Selloff{tuple_delimiter}market performance, investor sentiment{tuple_delimiter}The decline in the Global Tech Index is part of the broader market selloff driven by investor concerns.){record_delimiter}
(relationship{tuple_delimiter}Nexon Technologies{tuple_delimiter}Global Tech Index{tuple_delimiter}company impact, index movement{tuple_delimiter}Nexon Technologies' stock decline contributed to the overall drop in the Global Tech Index.){record_delimiter}
(relationship{tuple_delimiter}Gold Futures{tuple_delimiter}Market Selloff{tuple_delimiter}market reaction, safe-haven investment{tuple_delimiter}Gold prices rose as investors sought safe-haven assets during the market selloff.){record_delimiter}
(relationship{tuple_delimiter}Federal Reserve Policy Announcement{tuple_delimiter}Market Selloff{tuple_delimiter}interest rate impact, financial regulation{tuple_delimiter}Speculation over Federal Reserve policy changes contributed to market volatility and investor selloff.){record_delimiter}
{completion_delimiter}

""",
    """------Example 3------

Entity_types: [organization,person,equiment,product,technology,location,event,category]
Text:
```
At the World Athletics Championship in Tokyo, Noah Carter broke the 100m sprint record using cutting-edge carbon-fiber spikes.
```

Output:
(entity{tuple_delimiter}World Athletics Championship{tuple_delimiter}event{tuple_delimiter}The World Athletics Championship is a global sports competition featuring top athletes in track and field.){record_delimiter}
(entity{tuple_delimiter}Tokyo{tuple_delimiter}location{tuple_delimiter}Tokyo is the host city of the World Athletics Championship.){record_delimiter}
(entity{tuple_delimiter}Noah Carter{tuple_delimiter}person{tuple_delimiter}Noah Carter is a sprinter who set a new record in the 100m sprint at the World Athletics Championship.){record_delimiter}
(entity{tuple_delimiter}100m Sprint Record{tuple_delimiter}category{tuple_delimiter}The 100m sprint record is a benchmark in athletics, recently broken by Noah Carter.){record_delimiter}
(entity{tuple_delimiter}Carbon-Fiber Spikes{tuple_delimiter}equipment{tuple_delimiter}Carbon-fiber spikes are advanced sprinting shoes that provide enhanced speed and traction.){record_delimiter}
(entity{tuple_delimiter}World Athletics Federation{tuple_delimiter}organization{tuple_delimiter}The World Athletics Federation is the governing body overseeing the World Athletics Championship and record validations.){record_delimiter}
(relationship{tuple_delimiter}World Athletics Championship{tuple_delimiter}Tokyo{tuple_delimiter}event location, international competition{tuple_delimiter}The World Athletics Championship is being hosted in Tokyo.){record_delimiter}
(relationship{tuple_delimiter}Noah Carter{tuple_delimiter}100m Sprint Record{tuple_delimiter}athlete achievement, record-breaking{tuple_delimiter}Noah Carter set a new 100m sprint record at the championship.){record_delimiter}
(relationship{tuple_delimiter}Noah Carter{tuple_delimiter}Carbon-Fiber Spikes{tuple_delimiter}athletic equipment, performance boost{tuple_delimiter}Noah Carter used carbon-fiber spikes to enhance performance during the race.){record_delimiter}
(relationship{tuple_delimiter}Noah Carter{tuple_delimiter}World Athletics Championship{tuple_delimiter}athlete participation, competition{tuple_delimiter}Noah Carter is competing at the World Athletics Championship.){record_delimiter}
{completion_delimiter}

""",
    """------Example 4------

Entity_types: [organization,person,equiment,product,technology,location,event,category]
Text:
```
在北京举行的人工智能大会上，腾讯公司的首席技术官张伟发布了最新的大语言模型"腾讯智言"，该模型在自然语言处理方面取得了重大突破。

```

Output:
(entity{tuple_delimiter}人工智能大会{tuple_delimiter}event{tuple_delimiter}人工智能大会是在北京举行的技术会议，专注于人工智能领域的最新发展。){record_delimiter}
(entity{tuple_delimiter}北京{tuple_delimiter}location{tuple_delimiter}北京是人工智能大会的举办城市。){record_delimiter}
(entity{tuple_delimiter}腾讯公司{tuple_delimiter}organization{tuple_delimiter}腾讯公司是参与人工智能大会的科技企业，发布了新的语言模型产品。){record_delimiter}
(entity{tuple_delimiter}张伟{tuple_delimiter}person{tuple_delimiter}张伟是腾讯公司的首席技术官，在大会上发布了新产品。){record_delimiter}
(entity{tuple_delimiter}腾讯智言{tuple_delimiter}product{tuple_delimiter}腾讯智言是腾讯公司发布的大语言模型产品，在自然语言处理方面有重大突破。){record_delimiter}
(entity{tuple_delimiter}自然语言处理技术{tuple_delimiter}technology{tuple_delimiter}自然语言处理技术是腾讯智言模型取得重大突破的技术领域。){record_delimiter}
(relationship{tuple_delimiter}人工智能大会{tuple_delimiter}北京{tuple_delimiter}会议地点, 举办关系{tuple_delimiter}人工智能大会在北京举行。){record_delimiter}
(relationship{tuple_delimiter}张伟{tuple_delimiter}腾讯公司{tuple_delimiter}雇佣关系, 高管职位{tuple_delimiter}张伟担任腾讯公司的首席技术官。){record_delimiter}
(relationship{tuple_delimiter}张伟{tuple_delimiter}腾讯智言{tuple_delimiter}产品发布, 技术展示{tuple_delimiter}张伟在大会上发布了腾讯智言大语言模型。){record_delimiter}
(relationship{tuple_delimiter}腾讯智言{tuple_delimiter}自然语言处理技术{tuple_delimiter}技术应用, 突破创新{tuple_delimiter}腾讯智言在自然语言处理技术方面取得了重大突破。){record_delimiter}
{completion_delimiter}

""",
]

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
Output:"""

PROMPTS["entity_continue_extraction"] = """
MANY entities and relationships were missed in the last extraction. Please find only the missing entities and relationships from previous text. Do not include entities and relations that have been previously extracted. :\n

---Remember Steps---
1. Recognizing definitively conceptualized entities in text. For each identified entity, extract the following information:
- entity_name: Name of the entity, use same language as input text. If English, capitalized the name
- entity_type: One of the following types: [{entity_types}]. If the entity doesn't clearly fit any category, classify it as "Other".
- entity_description: Provide a comprehensive description of the entity's attributes and activities based on the information present in the input text. Do not add external knowledge.

2. Format each entity as:
("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

3. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are directly and clearly related based on the text. Unsubstantiated relationships must be excluded from the output.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_keywords: one or more high-level key words that summarize the overarching nature of the relationship, focusing on concepts or themes rather than specific details
- relationship_description: Explain the nature of the relationship between the source and target entities, providing a clear rationale for their connection

4. Format each relationship as:
("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_description>)

5. Use `{tuple_delimiter}` as field delimiter. Use `{record_delimiter}` as the list delimiter. Ensure no spaces are added around the delimiters.

6. When finished, output `{completion_delimiter}`

7. Return identified entities and relationships in {language}.

---Output---
Output:
"""

PROMPTS["entity_if_loop_extraction"] = """
---Goal---'

It appears some entities may have still been missed.

---Output---
Output:
""".strip()

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
  - For a Knowledge Graph Relationship: `[KG] <entity1_name> - <entity2_name>`
  - For a Document Chunk: `[DC] <file_path_or_document_name>`

---USER CONTEXT---
- Additional user prompt: {user_prompt}

---Response---
Output:"""

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
