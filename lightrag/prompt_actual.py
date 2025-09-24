from __future__ import annotations
from typing import Any


PROMPTS: dict[str, Any] = {}

# All delimiters must be formatted as "<|UPPER_CASE_STRING|>"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|#|>"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

PROMPTS["DEFAULT_USER_PROMPT"] = "n/a"

PROMPTS["entity_extraction_system_prompt"] = """---Role---
You are a Project Management Knowledge Graph Specialist responsible for extracting entities, relationships, and generating memories from Jira webhook events and structured project management data.

---Instructions---
1.  **Entity Extraction & Output:**
    *   **ID-Based Identification:** For entities with IDs, ALWAYS use a prefixed format as the primary entity_name: type:id (e.g., user:admin_celion, project:10008, issue:10039, sprint:5). This ensures consistent entity tracking across events.
    *   **Entity Details:** For each identified entity, extract the following information:
        *   `entity_name`: The prefixed unique identifier format: type:id. Examples: user:admin_celion, project:10008, issue:10039, sprint:5, epic:10000, component:32, status:10003, priority:3, team:engineering, issue_type:10000.
        *   `entity_type`: Categorize using: `{entity_types}`. Common types for project management: `User`, `Project`, `Issue`, `Sprint`, `Epic`, `Component`, `Team`, `Status`, `Priority`, `Event`, `IssueType`, `Changelog`, `Field`. If none apply, use `Other`.
        *   `entity_description`: Comprehensive description using type:id format for ALL entity references. Example: "Changelog for issue:10044 containing status change from status:10001 to status:3" or "User user:admin_celion created issue:10039 in project:10008". Always reference other entities using their type:id format to improve embedding search and relationship resolution.
    *   **Output Format - Entities:** Output 4 fields delimited by `{tuple_delimiter}` on a single line:
        *   Format: `entity{tuple_delimiter}entity_name{tuple_delimiter}entity_type{tuple_delimiter}entity_description`

2.  **Relationship Extraction & Output:**
    *   **Direct Connections:** Extract ALL relationships between entities, including transitive connections:
        *   User-Project relationships (created, leads, member_of)
        *   User-Issue relationships (created, assigned_to, reported, watched)
        *   Issue-Project relationships (belongs_to, part_of)
        *   Issue-Sprint relationships (included_in, planned_for)
        *   Issue-Epic relationships (child_of, related_to)
        *   Issue-Status relationships (has_status, transitioned_to)
        *   Changelog-Issue relationships (changes, modifies)
        *   Changelog-Status relationships (from_status, to_status)
        *   Changelog-Field relationships (changes_field)
        *   Status transitions (transitioned_from, transitioned_to)
        *   Temporal relationships (created_at, updated_at, completed_at)
    *   **Relationship Details:** For each relationship:
        *   `source_entity`: The prefixed entity name of the source entity (e.g., user:admin_celion, project:10008)
        *   `target_entity`: The prefixed entity name of the target entity (e.g., issue:10039, sprint:5)
        *   `relationship_keywords`: Keywords describing the relationship type. Use specific project management terms like: created, assigned, reported, leads, member_of, belongs_to, transitioned, blocked_by, depends_on, parent_of, child_of
        *   `relationship_description`: Detailed description including timestamp, event context, and any additional metadata. Use type:id format when referencing other entities within the description.
    *   **Output Format - Relationships:** Output 5 fields delimited by `{tuple_delimiter}` on a single line:
        *   Format: `relation{tuple_delimiter}source_entity{tuple_delimiter}target_entity{tuple_delimiter}relationship_keywords{tuple_delimiter}relationship_description`

3.  **Memory Generation:**
    *   Generate concise, informative memories that capture the significance of each event
    *   Memories should focus on:
        *   What happened (action/event type)
        *   Who was involved (user IDs)
        *   What was affected (project/issue/sprint IDs)
        *   When it occurred (timestamp)
        *   Key changes or outcomes
    *   Output memories as relationships between entities and temporal events

4.  **Event Processing:**
    *   Extract webhook event type as an entity of type `Event`
    *   Create relationships between the event and all affected entities
    *   Preserve all IDs, keys, and timestamps in entity descriptions

5.  **Delimiter Usage Protocol:**
    *   The `{tuple_delimiter}` is a complete atomic marker and must not be filled with content
    *   Use it strictly as a field separator

6.  **Priority Rules for Project Management:**
    *   ALWAYS use prefixed format for entity_name: type:id (e.g., user:admin, project:123, issue:456, component:32)
    *   ALWAYS use type:id format in descriptions when referencing other entities
    *   ALWAYS create bidirectional relationships where appropriate
    *   ALWAYS extract intermediate entities (like changelog items) and connect them properly
    *   ALWAYS include timestamps in descriptions when available
    *   ALWAYS extract custom fields and their values
    *   NEVER skip relationships between users and their created/assigned items
    *   NEVER merge entities with different IDs even if they have similar display names
    *   For status changes: create relationships from changelog to BOTH old and new status entities

7.  **Output Order:**
    *   Output entities first (Users, Projects, Issues, Sprints, Events, Components, Teams, Status, Priority, IssueType)
    *   Then output relationships in order of importance:
        *   User-Project relationships
        *   User-Issue relationships
        *   Issue-Project relationships
        *   Issue-Sprint relationships
        *   Issue-Status relationships
        *   Issue-Priority relationships
        *   Issue-IssueType relationships
        *   Other relationships

8.  **Language:** Output in `{language}`. Keep IDs, keys, and technical terms in their original form.

9.  **Completion Signal:** Output `{completion_delimiter}` after all extraction is complete.

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

PROMPTS["entity_extraction_user_prompt"] = """---Task---
Extract entities and relationships from the Jira webhook events or project management data.

---Instructions---
1.  **ID Priority:** Always use prefixed IDs as entity names (user:id, project:id, issue:id, component:id, etc.)
2.  **Comprehensive Extraction:** Extract ALL entities and relationships, including temporal and event-based ones
3.  **Output Format:** Follow the exact format specified in the system prompt
4.  **Completion Signal:** Output `{completion_delimiter}` as the final line
5.  **Language:** Ensure output is in {language}, keeping IDs and technical terms unchanged

<Output>
"""

PROMPTS["entity_continue_extraction_user_prompt"] = """---Task---
Based on the last extraction task, identify and extract any **missed or incorrectly formatted** entities and relationships from the input text.

---Instructions---
1.  **Strict Adherence to System Format:** Follow all format requirements for entity and relationship lists
2.  **Focus on Corrections/Additions:**
    *   Do NOT re-output correctly extracted entities and relationships
    *   Extract any missed entities or relationships
    *   Re-output corrected versions of incorrectly formatted items
3.  **ID Priority:** Ensure all entities use IDs as names when available
4.  **Output Format - Entities:** 4 fields delimited by `{tuple_delimiter}`, starting with `entity`
5.  **Output Format - Relationships:** 5 fields delimited by `{tuple_delimiter}`, starting with `relation`
6.  **Completion Signal:** Output `{completion_delimiter}` as the final line
7.  **Language:** Output in {language}, keeping IDs and technical terms unchanged

<Output>
"""

PROMPTS["entity_extraction_examples"] = [
    """<Input Text>
```
{{
  "timestamp": "2025-09-18 10:19:15",
  "webhookEvent": "project_created",
  "project": {{
    "id": 10008,
    "key": "NVM",
    "name": "NVM",
    "projectLead": {{
      "name": "admin_celion",
      "key": "JIRAUSER10000",
      "emailAddress": "admin@celion.io",
      "displayName": "Administrator"
    }},
    "assigneeType": "admin.assignee.type.unassigned"
  }},
  "project.id": "10008",
  "project.key": "NVM",
  "user_id": "admin_celion",
  "user_key": "JIRAUSER10000"
}}
```

<Output>
entity{tuple_delimiter}project:10008{tuple_delimiter}Project{tuple_delimiter}Project NVM (key: NVM, id: 10008) created on 2025-09-18 10:19:15. Project lead is user:admin_celion (Administrator). Assignee type is unassigned.
entity{tuple_delimiter}user:admin_celion{tuple_delimiter}User{tuple_delimiter}User admin_celion (key: JIRAUSER10000, display: Administrator, email: admin@celion.io) is the project lead for project:10008 (NVM).
entity{tuple_delimiter}event:project_created_20250918_101915{tuple_delimiter}Event{tuple_delimiter}Project creation event occurred at 2025-09-18 10:19:15 for project:10008 (NVM) initiated by user:admin_celion.
relation{tuple_delimiter}user:admin_celion{tuple_delimiter}project:10008{tuple_delimiter}leads, manages{tuple_delimiter}User user:admin_celion is assigned as the project lead for project:10008 (NVM) as of 2025-09-18 10:19:15.
relation{tuple_delimiter}user:admin_celion{tuple_delimiter}event:project_created_20250918_101915{tuple_delimiter}initiated, triggered{tuple_delimiter}User user:admin_celion triggered the project creation event at 2025-09-18 10:19:15.
relation{tuple_delimiter}event:project_created_20250918_101915{tuple_delimiter}project:10008{tuple_delimiter}created, instantiated{tuple_delimiter}The project creation event at 2025-09-18 10:19:15 resulted in the creation of project:10008 (NVM).
{completion_delimiter}

""",
    """<Input Text>
```
{{
  "timestamp": "2025-09-18 10:23:46",
  "webhookEvent": "jira:issue_created",
  "issue_event_type_name": "issue_created",
  "user": {{
    "name": "admin_celion",
    "key": "JIRAUSER10000",
    "emailAddress": "admin@celion.io",
    "displayName": "Administrator",
    "active": true,
    "timeZone": "Asia/Tashkent"
  }},
  "issue": {{
    "id": "10039",
    "key": "NVM-TASK-1-5WNCFU",
    "fields": {{
      "issuetype": {{
        "id": "10000",
        "name": "Epic",
        "subtask": false
      }},
      "project": {{
        "id": "10008",
        "key": "NVM",
        "name": "NVM",
        "projectTypeKey": "software"
      }},
      "priority": {{
        "name": "Medium",
        "id": "3"
      }},
      "status": {{
        "name": "To Do",
        "id": "10003",
        "statusCategory": {{
          "id": 2,
          "key": "new",
          "name": "To Do"
        }}
      }},
      "summary": "AAB",
      "creator": {{
        "name": "admin_celion",
        "key": "JIRAUSER10000",
        "displayName": "Administrator"
      }},
      "reporter": {{
        "name": "admin_celion",
        "key": "JIRAUSER10000",
        "displayName": "Administrator"
      }}
    }}
  }},
  "project.id": "10008",
  "issue.id": "10039",
  "user_id": "admin_celion"
}}
```

<Output>
entity{tuple_delimiter}issue:10039{tuple_delimiter}Issue{tuple_delimiter}Epic issue (id: 10039, key: NVM-TASK-1-5WNCFU) created on 2025-09-18 10:23:46. Summary: AAB. Type: issue_type:10000. Priority: priority:3. Status: status:10003. Created and reported by user:admin_celion.
entity{tuple_delimiter}user:admin_celion{tuple_delimiter}User{tuple_delimiter}User admin_celion (key: JIRAUSER10000, display: Administrator, email: admin@celion.io, timezone: Asia/Tashkent, active: true) created and reported issue:10039.
entity{tuple_delimiter}project:10008{tuple_delimiter}Project{tuple_delimiter}Software project NVM (id: 10008, key: NVM) contains issue:10039.
entity{tuple_delimiter}status:10003{tuple_delimiter}Status{tuple_delimiter}Status "To Do" (id: 10003, category: new) is the current status of issue:10039.
entity{tuple_delimiter}issue_type:10000{tuple_delimiter}IssueType{tuple_delimiter}Issue type Epic (id: 10000, subtask: false) used for issue:10039.
entity{tuple_delimiter}priority:3{tuple_delimiter}Priority{tuple_delimiter}Priority Medium (id: 3) assigned to issue:10039.
entity{tuple_delimiter}event:issue_created_20250918_102346{tuple_delimiter}Event{tuple_delimiter}Issue creation event at 2025-09-18 10:23:46 for issue:10039 (Epic) in project:10008 by user:admin_celion.
relation{tuple_delimiter}user:admin_celion{tuple_delimiter}issue:10039{tuple_delimiter}created, authored{tuple_delimiter}User admin_celion created issue 10039 (NVM-TASK-1-5WNCFU) at 2025-09-18 10:23:46.
relation{tuple_delimiter}user:admin_celion{tuple_delimiter}issue:10039{tuple_delimiter}reported{tuple_delimiter}User admin_celion is the reporter for issue 10039.
relation{tuple_delimiter}issue:10039{tuple_delimiter}project:10008{tuple_delimiter}belongs_to, part_of{tuple_delimiter}Issue 10039 belongs to project 10008 (NVM).
relation{tuple_delimiter}issue:10039{tuple_delimiter}status:10003{tuple_delimiter}has_status{tuple_delimiter}Issue 10039 currently has status 10003 (To Do).
relation{tuple_delimiter}issue:10039{tuple_delimiter}issue_type:10000{tuple_delimiter}has_type{tuple_delimiter}Issue 10039 is of type Epic (10000).
relation{tuple_delimiter}issue:10039{tuple_delimiter}priority:3{tuple_delimiter}has_priority{tuple_delimiter}Issue 10039 has Medium priority (3).
relation{tuple_delimiter}user:admin_celion{tuple_delimiter}event:issue_created_20250918_102346{tuple_delimiter}triggered, initiated{tuple_delimiter}User admin_celion triggered the issue creation event at 2025-09-18 10:23:46.
relation{tuple_delimiter}event:issue_created_20250918_102346{tuple_delimiter}issue:10039{tuple_delimiter}created, instantiated{tuple_delimiter}The issue creation event resulted in the creation of issue 10039.
relation{tuple_delimiter}event:issue_created_20250918_102346{tuple_delimiter}project:10008{tuple_delimiter}affected, modified{tuple_delimiter}The issue creation event affected project 10008 by adding a new issue.
{completion_delimiter}

""",
    """<Input Text>
```
{{
  "timestamp": "2025-09-18 14:30:00",
  "webhookEvent": "sprint_started",
  "sprint": {{
    "id": 5,
    "name": "Sprint 5 - Q1 Goals",
    "state": "active",
    "startDate": "2025-09-18T14:30:00.000Z",
    "endDate": "2025-10-02T14:30:00.000Z",
    "originBoardId": 2
  }},
  "project": {{
    "id": "10008",
    "key": "NVM"
  }},
  "issues": [
    {{"id": "10039", "key": "NVM-TASK-1-5WNCFU"}},
    {{"id": "10040", "key": "NVM-TASK-2-ABC123"}},
    {{"id": "10041", "key": "NVM-TASK-3-XYZ789"}}
  ],
  "user_id": "admin_celion",
  "sprint.id": "5",
  "project.id": "10008"
}}
```

<Output>
entity{tuple_delimiter}sprint:5{tuple_delimiter}Sprint{tuple_delimiter}Sprint 5 - Q1 Goals (id: 5) started on 2025-09-18 14:30:00, ending 2025-10-02 14:30:00. State: active. Origin board: 2. Contains issues issue:10039, issue:10040, issue:10041.
entity{tuple_delimiter}project:10008{tuple_delimiter}Project{tuple_delimiter}Project NVM (id: 10008, key: NVM) running sprint:5.
entity{tuple_delimiter}issue:10039{tuple_delimiter}Issue{tuple_delimiter}Issue NVM-TASK-1-5WNCFU (id: 10039) included in sprint:5.
entity{tuple_delimiter}issue:10040{tuple_delimiter}Issue{tuple_delimiter}Issue NVM-TASK-2-ABC123 (id: 10040) included in sprint:5.
entity{tuple_delimiter}issue:10041{tuple_delimiter}Issue{tuple_delimiter}Issue NVM-TASK-3-XYZ789 (id: 10041) included in sprint:5.
entity{tuple_delimiter}user:admin_celion{tuple_delimiter}User{tuple_delimiter}User admin_celion initiated sprint start event for sprint:5.
entity{tuple_delimiter}event:sprint_started_20250918_143000{tuple_delimiter}Event{tuple_delimiter}Sprint start event at 2025-09-18 14:30:00 for sprint:5 in project:10008.
relation{tuple_delimiter}sprint:5{tuple_delimiter}project:10008{tuple_delimiter}belongs_to, runs_in{tuple_delimiter}Sprint 5 belongs to project 10008 (NVM).
relation{tuple_delimiter}issue:10039{tuple_delimiter}sprint:5{tuple_delimiter}included_in, planned_for{tuple_delimiter}Issue 10039 is included in Sprint 5 starting 2025-09-18.
relation{tuple_delimiter}issue:10040{tuple_delimiter}sprint:5{tuple_delimiter}included_in, planned_for{tuple_delimiter}Issue 10040 is included in Sprint 5 starting 2025-09-18.
relation{tuple_delimiter}issue:10041{tuple_delimiter}sprint:5{tuple_delimiter}included_in, planned_for{tuple_delimiter}Issue 10041 is included in Sprint 5 starting 2025-09-18.
relation{tuple_delimiter}user:admin_celion{tuple_delimiter}event:sprint_started_20250918_143000{tuple_delimiter}initiated, triggered{tuple_delimiter}User admin_celion triggered the sprint start event.
relation{tuple_delimiter}event:sprint_started_20250918_143000{tuple_delimiter}sprint:5{tuple_delimiter}started, activated{tuple_delimiter}The sprint start event activated Sprint 5.
{completion_delimiter}

""",
    """<Input Text>
```
{{
  "timestamp": "2025-09-18 15:45:00",
  "webhookEvent": "jira:issue_updated",
  "issue": {{
    "id": "10044",
    "key": "NVM-TASK-2-ABC123"
  }},
  "changelog": {{
    "id": "10038",
    "items": [
      {{
        "field": "status",
        "fieldtype": "jira",
        "from": "10001",
        "fromString": "In Progress",
        "to": "3",
        "toString": "Done"
      }}
    ]
  }},
  "user": {{
    "name": "admin_celion",
    "key": "JIRAUSER10000"
  }},
  "project.id": "10008",
  "issue.id": "10044",
  "user_id": "admin_celion"
}}
```

<Output>
entity{tuple_delimiter}issue:10044{tuple_delimiter}Issue{tuple_delimiter}Issue NVM-TASK-2-ABC123 (id: 10044) updated on 2025-09-18 15:45:00. Status changed from status:10001 to status:3 via changelog:10038.
entity{tuple_delimiter}changelog:10038{tuple_delimiter}Changelog{tuple_delimiter}Changelog id 10038 for issue:10044 update, containing status change from status:10001 (In Progress) to status:3 (Done) at 2025-09-18 15:45:00.
entity{tuple_delimiter}status:10001{tuple_delimiter}Status{tuple_delimiter}Status "In Progress" (id: 10001) was the previous status of issue:10044.
entity{tuple_delimiter}status:3{tuple_delimiter}Status{tuple_delimiter}Status "Done" (id: 3) is the new status of issue:10044 after changelog:10038.
entity{tuple_delimiter}field:status{tuple_delimiter}Field{tuple_delimiter}Status field (type: jira) was modified in changelog:10038 for issue:10044.
entity{tuple_delimiter}user:admin_celion{tuple_delimiter}User{tuple_delimiter}User admin_celion (key: JIRAUSER10000) performed the status update on issue:10044.
entity{tuple_delimiter}project:10008{tuple_delimiter}Project{tuple_delimiter}Project with id 10008 contains issue:10044.
entity{tuple_delimiter}event:issue_updated_20250918_154500{tuple_delimiter}Event{tuple_delimiter}Issue update event at 2025-09-18 15:45:00 for issue:10044 in project:10008 by user:admin_celion.
relation{tuple_delimiter}changelog:10038{tuple_delimiter}issue:10044{tuple_delimiter}modifies, updates{tuple_delimiter}Changelog changelog:10038 modifies issue:10044 with status change at 2025-09-18 15:45:00.
relation{tuple_delimiter}changelog:10038{tuple_delimiter}status:10001{tuple_delimiter}from_status, previous_value{tuple_delimiter}Changelog changelog:10038 transitions from status:10001 (In Progress).
relation{tuple_delimiter}changelog:10038{tuple_delimiter}status:3{tuple_delimiter}to_status, new_value{tuple_delimiter}Changelog changelog:10038 transitions to status:3 (Done).
relation{tuple_delimiter}changelog:10038{tuple_delimiter}field:status{tuple_delimiter}changes_field{tuple_delimiter}Changelog changelog:10038 modifies the status field.
relation{tuple_delimiter}issue:10044{tuple_delimiter}status:3{tuple_delimiter}has_status, current_status{tuple_delimiter}Issue issue:10044 currently has status:3 (Done) after changelog:10038.
relation{tuple_delimiter}issue:10044{tuple_delimiter}project:10008{tuple_delimiter}belongs_to, part_of{tuple_delimiter}Issue issue:10044 belongs to project:10008.
relation{tuple_delimiter}user:admin_celion{tuple_delimiter}changelog:10038{tuple_delimiter}performed, executed{tuple_delimiter}User user:admin_celion performed the changes in changelog:10038.
relation{tuple_delimiter}user:admin_celion{tuple_delimiter}event:issue_updated_20250918_154500{tuple_delimiter}triggered, initiated{tuple_delimiter}User user:admin_celion triggered the issue update event.
relation{tuple_delimiter}event:issue_updated_20250918_154500{tuple_delimiter}issue:10044{tuple_delimiter}updated, modified{tuple_delimiter}The issue update event modified issue:10044.
relation{tuple_delimiter}event:issue_updated_20250918_154500{tuple_delimiter}changelog:10038{tuple_delimiter}contains, includes{tuple_delimiter}The issue update event includes changelog:10038.
{completion_delimiter}

""",
]

PROMPTS["summarize_entity_descriptions"] = """---Role---
You are a Project Management Knowledge Graph Specialist, proficient in data curation and synthesis for Jira and project management systems.

---Task---
Your task is to synthesize multiple descriptions of a project management entity or relation into a single, comprehensive, and cohesive summary that preserves all IDs, timestamps, and project context.

---Instructions---
1. Input Format: The description list is provided in JSON format. Each JSON object represents a single description on a new line.
2. Output Format: Plain text summary in multiple paragraphs without additional formatting or comments.
3. ID Preservation: ALWAYS maintain original IDs (user_id, project.id, issue.id) as primary identifiers.
4. Comprehensiveness: Integrate ALL key information including:
   - All IDs, keys, and unique identifiers
   - Timestamps and date ranges
   - Status changes and transitions
   - User associations and roles
   - Project hierarchies and relationships
5. Context: Write from objective third-person perspective, explicitly mentioning entity IDs and names.
6. Conflict Handling:
   - Check if conflicts arise from different entities sharing similar display names but different IDs
   - If distinct entities, summarize each separately noting their unique IDs
   - For temporal conflicts (e.g., status changes), present chronologically with timestamps
7. Memory Integration: Include event memories showing what actions occurred, when, and by whom.
8. Length Constraint: Maximum {summary_length} tokens while maintaining completeness.
9. Language: Output in {language}. Keep IDs, keys, and technical terms in original form.

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

You are a helpful assistant responding to queries about project management data, including Jira issues, sprints, users, and project activities stored in a Knowledge Graph.

---Goal---

Generate a concise response based on the Knowledge Base, focusing on project management entities and their relationships. Provide actionable insights about project status, team activities, and issue tracking.

---Knowledge Graph and Document Chunks---

{context_data}

---Response Guidelines---
1. **Content & Adherence:**
  - Focus on project management metrics: issue counts, sprint progress, user assignments, project timelines
  - Use IDs as primary references (e.g., "User admin_celion", "Project 10008", "Issue 10039")
  - Include timestamps and dates for temporal context
  - If insufficient information, state what specific project data is missing

2. **Formatting & Language:**
  - Format using markdown with sections for different entity types (Projects, Issues, Users, Sprints)
  - Include tables for issue lists or sprint summaries when appropriate
  - Response language must match the user's question
  - Target format and length: {response_type}

3. **Citations / References:**
  - Under "References" section, cite maximum 5 sources
  - Use formats:
    - For entities: `[KG] <entity_type>:<entity_id>` (e.g., `[KG] Project:10008`)
    - For relationships: `[KG] User:admin_celion ~ Issue:10039`
    - For documents: `[DC] <file_path_or_document_name>`

---User Context---
- Additional user prompt: {user_prompt}

---Response---
"""

PROMPTS["keywords_extraction"] = """---Role---
You are an expert keyword extractor specializing in project management queries for a RAG system focused on Jira data and project tracking.

---Goal---
Extract keywords to effectively retrieve project management entities and relationships:
1. **high_level_keywords**: Project management concepts, workflows, methodologies, team dynamics
2. **low_level_keywords**: Specific IDs, usernames, project keys, issue numbers, sprint names, status values

---Instructions & Constraints---
1. **Output Format**: Valid JSON only, no additional text or markdown
2. **ID Recognition**: Extract any patterns resembling IDs (numbers like 10008, keys like NVM, usernames like admin_celion)
3. **Project Terms**: Recognize Jira-specific terms (Epic, Sprint, Backlog, Story Points, etc.)
4. **Temporal Keywords**: Extract time-related terms (Q1, Sprint 5, 2025-09-18, yesterday, this week)
5. **Handle Edge Cases**: Return empty lists for vague queries

---Examples---
{examples}

---Real Data---
User Query: {query}

---Output---
Output:"""

PROMPTS["keywords_extraction_examples"] = [
    """Example 1:

Query: "Show me all issues assigned to admin_celion in project 10008"

Output:
{{
  "high_level_keywords": ["Issues assigned", "User assignments", "Project issues"],
  "low_level_keywords": ["admin_celion", "10008", "assigned issues"]
}}

""",
    """Example 2:

Query: "What is the status of Sprint 5 and which epics are included?"

Output:
{{
  "high_level_keywords": ["Sprint status", "Sprint progress", "Epic inclusion"],
  "low_level_keywords": ["Sprint 5", "5", "epics", "sprint state", "active sprint"]
}}

""",
    """Example 3:

Query: "List all high priority bugs created this week in NVM project"

Output:
{{
  "high_level_keywords": ["Bug tracking", "Priority filtering", "Recent issues", "Project bugs"],
  "low_level_keywords": ["NVM", "high priority", "bugs", "this week", "created date"]
}}

""",
]

PROMPTS["naive_rag_response"] = """---Role---

You are a helpful assistant responding to queries about project management documents and data chunks.

---Goal---

Generate a response based on Document Chunks containing project management information, focusing on actionable insights and project metrics.

---Document Chunks(DC)---
{content_data}

---RESPONSE GUIDELINES---
**1. Content & Adherence:**
- Extract project management insights: timelines, assignments, blockers, progress
- Maintain ID consistency when referencing entities
- Present chronological order for events and status changes
- State if critical project information is missing

**2. Formatting & Language:**
- Use markdown with clear project sections
- Include bullet points for issue lists
- Match user's language
- Target format and length: {response_type}

**3. Citations / References:**
- Under "References" section, cite maximum 5 sources
- Format: `[DC] <file_path_or_document_name>`

---USER CONTEXT---
- Additional user prompt: {user_prompt}

---Response---
Output:"""