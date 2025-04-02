Okay, let's refine the process based on your observation that `demo3_merged.json` already provides the structural information for `Document` and `Section`. This is a very good point and allows us to simplify the LLM's task and improve the reliability of the structural part of the knowledge graph.

Here’s how we can modify the configuration and the script:

**1. Improvements to `config_simple.yaml`**

The core idea is to tell the LLM *not* to extract `Document`, `Section`, `HAS_SECTION`, and `HAS_PARENT_SECTION`. These will be derived directly from the `demo3_merged.json` structure. However, we still keep them conceptually in the schema mappings for Cypher generation.

```yaml
# config.yaml (Simplified - Structure derived from JSON)
# Schema and Normalization for Corporate Regulations KG

# LightRAG 知识图谱提取配置文件

# 图谱模式定义
schema:
  # Entity types: Focused on actors and core content categorization (Structure derived from JSON)
  entity_types_llm: # Renamed to clarify these are for LLM extraction
    # Key Actors
    - Organization
    - Role
    # Core Content & Categorization
    - Statement       # (What rule/req/def/goal?)
    - Topic           # (What predefined category/type of regulation?)

  # --- Keep conceptual types for Cypher mapping ---
  all_entity_types: # All types that will exist in the graph
    - Document
    - Section
    - Organization
    - Role
    - Statement
    - Topic

  # Mapping to Cypher labels (Includes Document and Section)
  entity_type_map_cypher:
    Document: Document
    Section: Section
    Organization: Organization
    Role: Role
    Statement: Statement
    Topic: Topic

  # Relation types: Focused on semantic links and containment (Structure derived from JSON)
  relation_types_llm: # Renamed to clarify these are for LLM extraction
    # 语义关系
    # 职责和归属
    - RESPONSIBLE_FOR # (Organization/Role -> Statement/Topic)
    - BELONGS_TO      # (Organization/Role -> Organization/Role)
    # 内容关系
    - HAS_PURPOSE     # (Statement -> Statement[type=Goal])
    - REFERENCES      # (Statement/Document/Section -> Section) # LLM might still find references *within* text
    - APPLIES_TO      # (Statement -> Organization/Role)
    # 通用关系
    - RELATED_TO      # (Entity -> Entity)
    # --- MENTIONS and CONTAINS might be partially LLM, partially structural ---
    # MENTIONS        # (Section -> Topic) - LLM can find this semantic link
    # CONTAINS        # (Section -> Statement/Organization/Role/Topic) - LLM finds *what* is contained, script links it to the section

  # --- Keep conceptual types for Cypher mapping ---
  all_relation_types: # All types that will exist in the graph
    # Structural (Derived from JSON)
    - HAS_SECTION     # (Document -> Section)
    - HAS_PARENT_SECTION # (Section -> Section)
    # Containment (linking structure to semantics)
    - CONTAINS        # (Section -> Statement/Organization/Role/Topic) - Added explicitly
    # Semantic (Extracted by LLM or potentially inferred)
    - RESPONSIBLE_FOR
    - BELONGS_TO
    - HAS_PURPOSE
    - REFERENCES
    - MENTIONS # Keep MENTIONS here if LLM should extract Section->Topic links
    - APPLIES_TO
    - HAS_VERSION # Keep if needed, LLM *might* find version refs in text
    - RELATED_TO

  # Mapping to Cypher relationship types (Includes structural ones)
  relation_type_map_cypher:
    HAS_SECTION: HAS_SECTION
    HAS_PARENT_SECTION: HAS_PARENT_SECTION
    CONTAINS: CONTAINS
    RESPONSIBLE_FOR: RESPONSIBLE_FOR
    BELONGS_TO: BELONGS_TO
    HAS_PURPOSE: HAS_PURPOSE
    REFERENCES: REFERENCES
    MENTIONS: MENTIONS # Map if used
    APPLIES_TO: APPLIES_TO
    HAS_VERSION: HAS_VERSION
    RELATED_TO: RELATED_TO

normalization:
  # (保持不变, 但可以添加 Section 标题的规范化，如去除页码)
  canonical_map:
    # ... (existing mappings) ...
    # Optional: Add Section title normalization here if needed beyond basic cleaning
    # Example: "1 总则 (p.1)": "1 总则" # Though script might handle this directly

# Prompt 模板配置 (Revised - LLM focuses on semantic content)
prompts:
  entity_extraction:
    definitions: |
      实体类型定义 (请在返回结果中使用括号中的英文类型名称):
      # --- Document 和 Section 的定义已移除，因为它们将从输入结构中获取 ---
      - Topic (主题/规定类型)：指管理规定所属的**预定义分类或类型**，用于标识规则内容的性质。请识别文本中明确提及的、代表这些分类的术语。例如：'旅客候车'、'乘车人身安全'、'劳动安全'、'服务质量'、'工作流程'、'考核标准'等。 如果文本中没有明确提及这些分类术语，则不提取Topic。
      - Role (角色)：指代具有特定职责的职位或岗位，如"客运部主任"、"值班负责人"。注意与组织区分。
      - Organization (组织)：涉及的单位、部门或公司，如"中国铁路广州局集团有限公司"、"集团公司客运部"、"信息技术所"、"各车务站段"。
      - Statement (陈述)：规则、要求、定义、标准、目标或条件的明确说明，如"本预案适用于集团公司及所属各客运单位相关工作"、"各单位应明确职责分工"、"应保持站台清洁"。这是具体的规定内容。

    template: |
      请从以下文本中提取定义的实体类型。专注于识别 **组织、角色、具体规定陈述、以及预定义的规定类型(Topic)**。尽可能完整提取，确保不遗漏文本中的关键信息。

      {definitions}

      **要提取的实体类型列表**: {entity_types} # Excludes Document, Section

      文档信息 (仅供参考，不要提取为实体):
      文档标题: {document_title}
      当前章节: {current_heading}
      章节路径: {section_path}
      父章节内容摘要: {parent_section_summary} # Optional context

      文本：
      """
      {content}
      """

      【重要】：必须使用英文实体类型！返回的JSON中，实体类型必须为英文，且仅包含以下类型: {entity_types_english}

      请以严格的 JSON 格式输出，包含一个名为 "entities" 的列表，其中每个对象包含 "name" (实体名称) 和 "type" (实体类型)。确保实体名称是文本中实际出现的词语。

      注意事项：
      1. **不要提取文档标题或章节标题作为实体。**
      2. 确保每个实体完整识别。
      3. 识别实体时考虑缩写和全称。
      4. 保留相同实体的不同表述。
      5. 确保每个实体都准确分配了正确的英文实体类型。
      6. **严格使用以下英文类型名**: {entity_types_english}
      7. 类型区分指南：
         - **Topic与Statement的区别**：Topic是规定内容的**分类标签**（如"服务质量"），Statement是**具体的规定本身**（如"应使用规范用语"）。
         - Organization与Role的区别：Organization是部门/单位，Role是岗位/职责。
      8. 利用文档上下文信息辅助判断。

      例如:
      {
        "entities": [
          {"name": "集团公司客运部", "type": "Organization"},
          # {"name": "《服务质量管理办法》", "type": "Document"}, # REMOVED
          {"name": "应使用规范用语，保持微笑服务", "type": "Statement"},
          {"name": "服务质量", "type": "Topic"},
          {"name": "站务员", "type": "Role"}
          # {"name": "3.1 基本要求", "type": "Section"} # REMOVED
        ]
      }

  relation_extraction:
    definitions: |
      关系类型定义 (请在返回结果中使用括号中的英文关系类型):
      # --- HAS_SECTION, HAS_PARENT_SECTION, and potentially CONTAINS (if fully structural) definitions removed ---
      # 语义关系
      - BELONGS_TO (隶属关系): 指明组织或角色的归属。 (Organization/Role -> Organization/Role)
      - RESPONSIBLE_FOR (责任): 明确组织或角色对某项具体规定(Statement)或某一类规定(Topic)的职责。 (Organization/Role -> Statement/Topic)
      - REFERENCES (引用): 明确引用某章节。 (Statement/Document/Section -> Section) # LLM might find textual references
      - MENTIONS (提及): 章节中明确提及某规定类型(Topic)。 (Section -> Topic) # LLM finds semantic mention
      - APPLIES_TO (适用于): 规则(Statement)适用的组织或角色。 (Statement -> Organization/Role)
      - HAS_PURPOSE (目的): 陈述的目的。 (Statement -> Statement)
      # 通用关系
      - RELATED_TO (关联): 连接实体间的其他关联。 (Entity -> Entity)

    template: |
      请从以下文本中，根据预定义的实体列表（**不包括文档和章节实体**），提取这些实体之间符合定义的关系类型。请专注于在文本段落中**直接陈述**的语义关系。

      {definitions}

      **要提取的关系类型列表**: {relation_types} # Excludes HAS_SECTION, HAS_PARENT_SECTION
      预定义的实体类型列表 (用于关系端点): {all_entity_types_english} # LLM can reference Struct types

      文档信息 (仅供参考):
      文档标题: {document_title}
      当前章节: {current_heading}
      章节路径: {section_path}

      文本：
      """
      {content}
      """

      文本中已识别的 **语义实体** 列表 (用于构建关系):
      """
      {entities_json} # Should contain only Organization, Role, Statement, Topic extracted earlier
      """

      【重要】：必须使用英文实体类型和关系类型！

      请以严格的 JSON 格式输出，包含一个名为 "relations" 的列表，其中每个对象必须包含 "source", "source_type", "target", "target_type", "type" 字段。

      注意事项:
      1. 关系必须连接上面 **语义实体列表** 中的实体，或者连接语义实体到已知的**章节(Section)** 或 **文档(Document)**（如果文本明确引用）。
      2. source_type 和 target_type 必须是以下英文类型之一: {all_entity_types_english}。
      3. 关系类型必须是预定义的英文类型之一: {relation_types_english}
      4. **不要提取 Document -> Section 或 Section -> Section 的结构关系。**
      5. `CONTAINS` 关系将由脚本自动处理（连接章节和其包含的语义实体），**不要** 让 LLM 提取 `CONTAINS`。
      6. 关系类型使用指南：
         - RESPONSIBLE_FOR：描述对具体规则(Statement)或规则类型(Topic)的责任。
         - MENTIONS：用于连接章节(Section)和其明确提及的规定类型(Topic)。
         - REFERENCES: 用于文本中明确提到的对其他章节的引用 (e.g., "详见章节 3.1").

      例如 (基于上面修改的实体提取例子):
      {
        "relations": [
          {
            "source": "集团公司客运部",
            "source_type": "Organization",
            "target": "服务质量",
            "target_type": "Topic",
            "type": "RESPONSIBLE_FOR"
          },
          {
            "source": "应使用规范用语，保持微笑服务",
            "source_type": "Statement",
            "target": "站务员",
            "target_type": "Role",
            "type": "APPLIES_TO"
           }
          # { # Section -> Topic MENTION might still be extracted by LLM if relevant
          #   "source": "3.1 基本要求", # Assuming this context is implicitly known or passed
          #   "source_type": "Section",
          #   "target": "服务质量",
          #   "target_type": "Topic",
          #   "type": "MENTIONS"
          # },
          # --- CONTAINS relationship is NOT extracted by LLM ---
        ]
      }

# 图数据库配置 (保持不变)
database:
  enable_uniqueness_constraints: true
  enable_relation_metadata: true
```

**Key Changes in Config:**

1.  **Entity/Relation Types for LLM:** Renamed lists to `entity_types_llm` and `relation_types_llm` to make it clear these are what the LLM should *extract*. Removed `Document`, `Section`, `HAS_SECTION`, `HAS_PARENT_SECTION` from these lists.
2.  **All Entity/Relation Types:** Introduced `all_entity_types` and `all_relation_types` to list *all* types that will eventually exist in the graph (including structural ones).
3.  **Mappings:** Kept `Document` and `Section` in `entity_type_map_cypher` and structural relations in `relation_type_map_cypher` as they are needed for Cypher generation. Added `CONTAINS` explicitly.
4.  **Prompts:**
    *   Removed `Document` and `Section` definitions and examples from the LLM's task.
    *   Instructed the LLM *not* to extract these structural elements.
    *   Clarified that the LLM should focus on semantic entities (`Organization`, `Role`, `Statement`, `Topic`) and semantic relations.
    *   Explicitly stated that `CONTAINS` will be handled by the script.
    *   Updated placeholders (`{entity_types}`, `{relation_types}`) to reflect the reduced scope for the LLM. Added `{entity_types_english}` and `{relation_types_english}` for clarity in instructions.
    *   Used `{all_entity_types_english}` in the relation prompt's type list to allow LLM to potentially link semantic entities *to* structurally known Documents/Sections if referenced in text.

**2. Modifications to `entity_extract.py`**

The script needs a major refactoring to implement the two-phase approach:

```python
import json
import os
import logging
import time
import requests
import yaml
import asyncio
import httpx
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import re
import argparse

# --- (Keep existing configuration loading and basic setup) ---
# ... (LLM_API_KEY, LLM_API_HOST, etc.) ...
# ... (Logging setup) ...

# Global variables loaded from config
ENTITY_TYPES_LLM = None
ALL_ENTITY_TYPES = None
ENTITY_TYPE_MAP_CYPHER = None
RELATION_TYPES_LLM = None
ALL_RELATION_TYPES = None
RELATION_TYPE_MAP_CYPHER = None
CANONICAL_MAP = None
PROMPT_TEMPLATES = None

def load_config_and_setup_globals(config_path: str):
    """Loads config and sets up global variables."""
    global ENTITY_TYPES_LLM, ALL_ENTITY_TYPES, ENTITY_TYPE_MAP_CYPHER
    global RELATION_TYPES_LLM, ALL_RELATION_TYPES, RELATION_TYPE_MAP_CYPHER
    global CANONICAL_MAP, PROMPT_TEMPLATES

    CONFIG = load_config(config_path) # Use your existing robust load_config

    # Load LLM-specific types
    ENTITY_TYPES_LLM = CONFIG['schema']['entity_types_llm']
    RELATION_TYPES_LLM = CONFIG['schema']['relation_types_llm']

    # Load all conceptual types
    ALL_ENTITY_TYPES = CONFIG['schema']['all_entity_types']
    ALL_RELATION_TYPES = CONFIG['schema']['all_relation_types']

    # Load mappings and normalization
    ENTITY_TYPE_MAP_CYPHER = CONFIG['schema']['entity_type_map_cypher']
    RELATION_TYPE_MAP_CYPHER = CONFIG['schema']['relation_type_map_cypher']
    CANONICAL_MAP = CONFIG['normalization']['canonical_map']
    PROMPT_TEMPLATES = CONFIG.get('prompts', {})

    # Validation (ensure necessary keys exist)
    if not all([ENTITY_TYPES_LLM, ALL_ENTITY_TYPES, ENTITY_TYPE_MAP_CYPHER,
                RELATION_TYPES_LLM, ALL_RELATION_TYPES, RELATION_TYPE_MAP_CYPHER,
                CANONICAL_MAP, PROMPT_TEMPLATES]):
        raise ValueError("Configuration is missing required keys after loading.")

    logging.info(f"Loaded {len(ENTITY_TYPES_LLM)} entity types for LLM extraction.")
    logging.info(f"Loaded {len(RELATION_TYPES_LLM)} relation types for LLM extraction.")
    logging.info(f"Total conceptual entity types: {len(ALL_ENTITY_TYPES)}")
    logging.info(f"Total conceptual relation types: {len(ALL_RELATION_TYPES)}")

    return CONFIG

# --- (Keep Helper Functions: load_json_data, normalize_entity_name, normalize_entity_type, escape_cypher_string, parse_llm_response) ---
# Note: normalize_entity_type might need slight adjustment if it relied on ENTITY_TYPES only. It should ideally check against ALL_ENTITY_TYPES or the keys/values of ENTITY_TYPE_MAP_CYPHER.

# --- Modified Prompt Creation ---

def clean_heading(heading: str) -> str:
    """Removes page numbers and potentially other noise from headings."""
    if not heading:
        return "Unknown Section"
    # Remove (p.XX)
    cleaned = re.sub(r'\s*\(p\.\d+\)\s*$', '', heading).strip()
    # Optional: Apply canonical map normalization if defined for headings
    cleaned = CANONICAL_MAP.get(cleaned, cleaned)
    return cleaned

def create_entity_prompt(chunk_content: str, context: Dict[str, Any]) -> str:
    """Creates entity extraction prompt focusing on semantic types."""
    entity_config = PROMPT_TEMPLATES.get('entity_extraction', {})
    definitions = entity_config.get('definitions', '')
    template = entity_config.get('template', '')

    if not template:
        logging.error("Entity extraction template not found in config.")
        return "Error: Prompt template missing."

    # Get English names for the prompt instructions
    entity_types_english = [ENTITY_TYPE_MAP_CYPHER.get(t, t) for t in ENTITY_TYPES_LLM]

    try:
        prompt = template.format(
            definitions=definitions,
            entity_types=', '.join(ENTITY_TYPES_LLM), # Types LLM should extract
            entity_types_english=', '.join(entity_types_english),
            content=chunk_content,
            document_title=context.get('document_title', 'N/A'),
            current_heading=context.get('current_heading', 'N/A'),
            section_path=context.get('section_path', 'N/A'),
            parent_section_summary=context.get('parent_section_summary', '') # Optional
        )
    except KeyError as e:
        logging.error(f"Missing key in entity prompt template: {e}")
        # Fallback or safer formatting
        prompt = template # Simplified if format fails catastrophically
        prompt = prompt.replace("{definitions}", definitions)
        prompt = prompt.replace("{entity_types}", ', '.join(ENTITY_TYPES_LLM))
        prompt = prompt.replace("{entity_types_english}", ', '.join(entity_types_english))
        prompt = prompt.replace("{content}", chunk_content)
        # Replace context placeholders safely
        prompt = prompt.replace("{document_title}", context.get('document_title', 'N/A'))
        prompt = prompt.replace("{current_heading}", context.get('current_heading', 'N/A'))
        prompt = prompt.replace("{section_path}", context.get('section_path', 'N/A'))
        prompt = prompt.replace("{parent_section_summary}", context.get('parent_section_summary', ''))

    return prompt

def create_relation_prompt(chunk_content: str, entities_json: str, context: Dict[str, Any]) -> str:
    """Creates relation extraction prompt focusing on semantic relations."""
    relation_config = PROMPT_TEMPLATES.get('relation_extraction', {})
    definitions = relation_config.get('definitions', '')
    template = relation_config.get('template', '')

    if not template:
        logging.error("Relation extraction template not found in config.")
        return "Error: Prompt template missing."

    # Get English names for prompt instructions
    relation_types_english = [RELATION_TYPE_MAP_CYPHER.get(t, t) for t in RELATION_TYPES_LLM]
    all_entity_types_english = [ENTITY_TYPE_MAP_CYPHER.get(t, t) for t in ALL_ENTITY_TYPES]


    try:
        prompt = template.format(
            definitions=definitions,
            relation_types=', '.join(RELATION_TYPES_LLM), # Types LLM should extract
            relation_types_english=', '.join(relation_types_english),
            all_entity_types_english=', '.join(all_entity_types_english), # All possible endpoint types
            content=chunk_content,
            entities_json=entities_json, # Only semantic entities here
            document_title=context.get('document_title', 'N/A'),
            current_heading=context.get('current_heading', 'N/A'),
            section_path=context.get('section_path', 'N/A')
        )
    except KeyError as e:
        logging.error(f"Missing key in relation prompt template: {e}")
        # Fallback or safer formatting
        prompt = template # Simplified
        prompt = prompt.replace("{definitions}", definitions)
        prompt = prompt.replace("{relation_types}", ', '.join(RELATION_TYPES_LLM))
        prompt = prompt.replace("{relation_types_english}", ', '.join(relation_types_english))
        prompt = prompt.replace("{all_entity_types_english}", ', '.join(all_entity_types_english))
        prompt = prompt.replace("{content}", chunk_content)
        prompt = prompt.replace("{entities_json}", entities_json)
        # Replace context placeholders safely
        prompt = prompt.replace("{document_title}", context.get('document_title', 'N/A'))
        prompt = prompt.replace("{current_heading}", context.get('current_heading', 'N/A'))
        prompt = prompt.replace("{section_path}", context.get('section_path', 'N/A'))

    return prompt


# --- (Keep LLM Interaction: LLMTask, call_llm_async, call_llm, RateLimiter, process_tasks) ---
# Note: call_llm_async might need adjustment in the system prompt if it was very specific before.

# --- (Keep Cypher Generation: generate_cypher_statements) ---
# Note: generate_cypher_statements should already work if it receives the combined entity/relation sets correctly. Ensure it uses ALL_ENTITY_TYPES and ALL_RELATION_TYPES mappings if needed.

# --- *** NEW/REVISED Main Processing Logic *** ---

async def process_document_structure(data: List[Dict[str, Any]]) -> Tuple[
    Dict[str, Dict[str, Any]], # documents_map {doc_id: {name: title, type: 'Document'}}
    Dict[str, Dict[str, Any]], # sections_map {chunk_id: {name: heading, type: 'Section', doc_id: doc_id}}
    Set[Tuple[str, str, str, str]] # structural_relations {(source_id, target_id, type, context_id)} - using IDs first
]:
    """
    Parses the JSON structure to extract Document and Section entities and their relationships.
    Uses IDs internally for relationships, to be resolved to names later.
    """
    documents_map: Dict[str, Dict[str, Any]] = {}
    sections_map: Dict[str, Dict[str, Any]] = {}
    structural_relations: Set[Tuple[str, str, str, str]] = set() # (source_id, target_id, type, chunk_id for context)

    # --- Heuristic for Document Title ---
    # Attempt to find the main document title. Assume it might be part of the first chunk's content or filename.
    doc_titles: Dict[str, str] = {}
    if data:
        first_chunk = data[0]
        doc_id = first_chunk.get("full_doc_id", "unknown_doc_0")
        title = "Unknown Document"
        # Try extracting from file path
        if "file_path" in first_chunk:
             # Basic extraction from filename, might need refinement
            title = Path(first_chunk["file_path"]).stem.split('-')[-1].replace('_', ' ')
            # Apply normalization if needed
            title = normalize_entity_name(title) # Use normalization
        # Fallback: try first line of first chunk content if title seems generic
        if title == "Unknown Document" and "content" in first_chunk:
             first_line = first_chunk["content"].split('\n')[0].strip()
             if first_line and len(first_line) > 5: # Avoid short/generic lines
                 title = normalize_entity_name(first_line)

        doc_titles[doc_id] = title
        # Note: This assumes one document per JSON file. If multiple docs exist, logic needs adjustment.

    # --- Process Chunks for Sections and Relationships ---
    for chunk in data:
        chunk_id = chunk.get("chunk_id")
        doc_id = chunk.get("full_doc_id")
        heading = chunk.get("heading")
        parent_id = chunk.get("parent_id") # Might be chunk_id of parent

        if not chunk_id or not doc_id or not heading:
            logging.warning(f"Skipping chunk due to missing key info: {chunk.get('chunk_id', 'N/A')}")
            continue

        # Create/Update Document Entity
        doc_title = doc_titles.get(doc_id, f"Unknown Document {doc_id}")
        if doc_id not in documents_map:
            documents_map[doc_id] = {
                "name": doc_title,
                "type": "Document", # Use the conceptual type name
                # Add other relevant metadata if available, e.g., file_path
                "file_path": chunk.get("file_path")
            }
            logging.debug(f"Identified Document: ID={doc_id}, Title={doc_title}")

        # Create Section Entity
        section_name = clean_heading(heading)
        section_name = normalize_entity_name(section_name) # Normalize further if needed
        if chunk_id not in sections_map:
            sections_map[chunk_id] = {
                "name": section_name,
                "type": "Section", # Use the conceptual type name
                "doc_id": doc_id, # Link back to document
                "original_heading": heading # Keep original for context if needed
            }
            logging.debug(f"Identified Section: ID={chunk_id}, Name={section_name}, DocID={doc_id}")

            # Create HAS_SECTION relationship (Document -> Section)
            # Using IDs here for simplicity, resolve to names before Cypher gen
            structural_relations.add((doc_id, chunk_id, "HAS_SECTION", chunk_id))

            # Create HAS_PARENT_SECTION relationship (Section -> Section)
            if parent_id and parent_id != chunk_id: # Ensure parent is different
                 # Check if parent_id is a known section chunk_id
                 # Note: Assumes parent_id refers to another chunk_id in the same doc
                 # If parent_id is something else, this logic needs adjustment
                 # We add the relation assuming parent_id will be in sections_map
                 structural_relations.add((chunk_id, parent_id, "HAS_PARENT_SECTION", chunk_id))
                 logging.debug(f"Identified Parent Relation: Child={chunk_id}, Parent={parent_id}")
        else:
             # Update section info if needed (e.g., if seen again with more details)
             pass

    return documents_map, sections_map, structural_relations


async def extract_semantic_info(data: List[Dict[str, Any]],
                                sections_map: Dict[str, Dict[str, Any]],
                                documents_map: Dict[str, Dict[str, Any]]) -> Tuple[
                                    Set[Tuple[str, str, str]], # semantic_entities {(name, type, chunk_id)}
                                    Set[Tuple[str, str, str, str]] # semantic_relations {(source, target, type, chunk_id)}
                                ]:
    """
    Uses LLM to extract semantic entities and relations from chunk content.
    """
    semantic_entities: Set[Tuple[str, str, str]] = set()  # (name, type, chunk_id)
    semantic_relations: Set[Tuple[str, str, str, str]] = set() # (source, target, type, chunk_id)
    processed_chunk_entities: Dict[str, List[Dict[str, str]]] = {} # Store entities per chunk for relation prompt

    # --- Phase 2a: Entity Extraction ---
    entity_tasks = []
    for i, chunk in enumerate(data):
        chunk_id = chunk.get("chunk_id")
        content = chunk.get("content")
        doc_id = chunk.get("full_doc_id")

        if not chunk_id or not content or chunk_id not in sections_map:
            logging.warning(f"Skipping entity extraction for chunk {chunk_id or i}: Missing data or not found in structure.")
            continue

        # Prepare context for the prompt
        section_info = sections_map[chunk_id]
        doc_info = documents_map.get(doc_id, {})
        context = {
            "document_title": doc_info.get("name", "N/A"),
            "current_heading": section_info.get("name", "N/A"),
            "section_path": "N/A", # TODO: Build section path if needed
            "parent_section_summary": "" # TODO: Add parent summary if needed
        }

        entity_prompt = create_entity_prompt(content, context)
        entity_tasks.append(LLMTask(
            chunk_id=chunk_id,
            prompt_type="entity",
            content=content,
            prompt=entity_prompt
        ))

    logging.info(f"Starting LLM entity extraction for {len(entity_tasks)} chunks...")
    processed_entity_tasks = await process_tasks(entity_tasks)
    logging.info("Finished LLM entity extraction.")

    # Process entity results
    for task in processed_entity_tasks:
        if task.result and 'entities' in task.result:
            chunk_entities = []
            for entity in task.result['entities']:
                raw_name = entity.get('name')
                raw_type = entity.get('type') # This should be one of ENTITY_TYPES_LLM
                # Validate type against LLM-specific types
                if raw_name and raw_type and raw_type in [ENTITY_TYPE_MAP_CYPHER.get(t, t) for t in ENTITY_TYPES_LLM]:
                    normalized_name = normalize_entity_name(raw_name)
                    # Find the original conceptual type name before mapping for consistency
                    conceptual_type = raw_type # Default if mapping not found back
                    for cn_type, en_type in ENTITY_TYPE_MAP_CYPHER.items():
                         if en_type == raw_type and cn_type in ENTITY_TYPES_LLM:
                              conceptual_type = cn_type
                              break

                    semantic_entities.add((normalized_name, conceptual_type, task.chunk_id))
                    chunk_entities.append({"name": normalized_name, "type": raw_type}) # Use normalized name and LLM type for relation prompt
                else:
                    logging.warning(f"LLM returned invalid entity type '{raw_type}' or missing name in chunk {task.chunk_id}: {entity}")
            processed_chunk_entities[task.chunk_id] = chunk_entities
        elif task.result is None:
             logging.error(f"Entity extraction failed for chunk {task.chunk_id} (no result)")
        else:
             logging.warning(f"No 'entities' key found in result for chunk {task.chunk_id}")
             processed_chunk_entities[task.chunk_id] = []


    # --- Phase 2b: Relation Extraction ---
    relation_tasks = []
    for i, chunk in enumerate(data):
        chunk_id = chunk.get("chunk_id")
        content = chunk.get("content")
        doc_id = chunk.get("full_doc_id")

        if not chunk_id or not content or chunk_id not in sections_map:
            logging.warning(f"Skipping relation extraction for chunk {chunk_id or i}: Missing data.")
            continue

        # Get semantic entities extracted for this chunk
        chunk_semantic_entities = processed_chunk_entities.get(chunk_id, [])
        if not chunk_semantic_entities:
            logging.debug(f"Skipping relation extraction for chunk {chunk_id}: No semantic entities found.")
            continue

        entities_json_str = json.dumps({"entities": chunk_semantic_entities}, ensure_ascii=False, indent=2)

        # Prepare context
        section_info = sections_map[chunk_id]
        doc_info = documents_map.get(doc_id, {})
        context = {
            "document_title": doc_info.get("name", "N/A"),
            "current_heading": section_info.get("name", "N/A"),
            "section_path": "N/A", # TODO: Build section path
        }

        relation_prompt = create_relation_prompt(content, entities_json_str, context)
        relation_tasks.append(LLMTask(
            chunk_id=chunk_id,
            prompt_type="relation",
            content=content, # Content might not be needed if prompt template doesn't use it here
            prompt=relation_prompt
        ))

    logging.info(f"Starting LLM relation extraction for {len(relation_tasks)} chunks...")
    processed_relation_tasks = await process_tasks(relation_tasks)
    logging.info("Finished LLM relation extraction.")

    # Process relation results
    for task in processed_relation_tasks:
        if task.result and 'relations' in task.result:
            for relation in task.result['relations']:
                raw_source = relation.get('source')
                raw_target = relation.get('target')
                raw_type = relation.get('type') # Should be one of RELATION_TYPES_LLM mapped names
                source_type = relation.get('source_type') # Optional, helps validation
                target_type = relation.get('target_type') # Optional, helps validation

                # Find the original conceptual relation type name
                conceptual_rel_type = raw_type
                for cn_type, en_type in RELATION_TYPE_MAP_CYPHER.items():
                     if en_type == raw_type and cn_type in RELATION_TYPES_LLM:
                           conceptual_rel_type = cn_type
                           break

                # Validate relation type against LLM-specific types
                if raw_source and raw_target and raw_type and conceptual_rel_type in RELATION_TYPES_LLM:
                    normalized_source = normalize_entity_name(raw_source)
                    normalized_target = normalize_entity_name(raw_target)
                    semantic_relations.add((normalized_source, normalized_target, conceptual_rel_type, task.chunk_id))
                    # Optional: Add source/target entities if LLM provided types and they weren't caught before
                    # Be careful not to add Document/Section types here unless intended (e.g., REFERENCES)
                    if source_type and source_type in [ENTITY_TYPE_MAP_CYPHER.get(t,t) for t in ALL_ENTITY_TYPES]:
                        conceptual_src_type = source_type
                        for cn_type, en_type in ENTITY_TYPE_MAP_CYPHER.items():
                             if en_type == source_type: conceptual_src_type = cn_type; break
                        # Add only if it's a semantic type or an allowed structural type (Doc/Sec)
                        if conceptual_src_type in ENTITY_TYPES_LLM or conceptual_src_type in ["Document", "Section"]:
                            semantic_entities.add((normalized_source, conceptual_src_type, task.chunk_id))

                    if target_type and target_type in [ENTITY_TYPE_MAP_CYPHER.get(t,t) for t in ALL_ENTITY_TYPES]:
                        conceptual_tgt_type = target_type
                        for cn_type, en_type in ENTITY_TYPE_MAP_CYPHER.items():
                             if en_type == target_type: conceptual_tgt_type = cn_type; break
                        if conceptual_tgt_type in ENTITY_TYPES_LLM or conceptual_tgt_type in ["Document", "Section"]:
                            semantic_entities.add((normalized_target, conceptual_tgt_type, task.chunk_id))

                else:
                    logging.warning(f"LLM returned invalid relation type '{raw_type}' or missing source/target in chunk {task.chunk_id}: {relation}")
        elif task.result is None:
            logging.error(f"Relation extraction failed for chunk {task.chunk_id} (no result)")
        # else: No relations found is normal

    return semantic_entities, semantic_relations


async def main(input_json_path: str, config_path: str, output_cypher_path: str):
    """Main processing pipeline."""
    try:
        # 0. Load Config
        logging.info(f"Loading configuration from: {config_path}")
        load_config_and_setup_globals(config_path)

        # 1. Load Input Data
        logging.info(f"Loading input data from: {input_json_path}")
        data = load_json_data(input_json_path)
        if not data:
            logging.error("Failed to load input data. Exiting.")
            return

        # --- Phase 1: Process Structure ---
        logging.info("Phase 1: Processing document structure from JSON...")
        start_time = time.time()
        documents_map, sections_map, structural_relations_by_id = await process_document_structure(data)
        logging.info(f"Structure processing completed in {time.time() - start_time:.2f} seconds.")
        logging.info(f"Identified {len(documents_map)} documents and {len(sections_map)} sections.")

        # --- Prepare Structural Entities/Relations with Names ---
        structural_entities: Set[Tuple[str, str, str]] = set() # (name, type, id) - use doc_id/chunk_id as context id
        for doc_id, doc_info in documents_map.items():
            structural_entities.add((doc_info['name'], doc_info['type'], doc_id))
        for sec_id, sec_info in sections_map.items():
            structural_entities.add((sec_info['name'], sec_info['type'], sec_id))

        structural_relations: Set[Tuple[str, str, str, str]] = set() # (source_name, target_name, type, context_id)
        for src_id, tgt_id, rel_type, context_id in structural_relations_by_id:
            source_info = documents_map.get(src_id) or sections_map.get(src_id)
            target_info = documents_map.get(tgt_id) or sections_map.get(tgt_id)
            if source_info and target_info:
                structural_relations.add((source_info['name'], target_info['name'], rel_type, context_id))
            else:
                 logging.warning(f"Could not resolve IDs for structural relation: {src_id} -[{rel_type}]-> {tgt_id}")


        # --- Phase 2: Extract Semantic Info via LLM ---
        logging.info("Phase 2: Extracting semantic information using LLM...")
        start_time = time.time()
        semantic_entities, semantic_relations_llm = await extract_semantic_info(data, sections_map, documents_map)
        logging.info(f"Semantic extraction completed in {time.time() - start_time:.2f} seconds.")
        logging.info(f"Extracted {len(semantic_entities)} semantic entities and {len(semantic_relations_llm)} semantic relations via LLM.")

        # --- Phase 3: Create CONTAINS relations (Section -> Semantic Entity) ---
        containment_relations: Set[Tuple[str, str, str, str]] = set()
        # Add section names from semantic_entities that might have been missed if only referenced in relations
        # Also ensures all semantic entities have a type known for Cypher generation
        final_semantic_entities = set()
        all_known_entities_map = {name: type for name, type, _ in structural_entities}
        for name, type, chunk_id in semantic_entities:
             all_known_entities_map[name] = type # Update map with semantic types
             final_semantic_entities.add((name, type, chunk_id))

        for entity_name, entity_type, chunk_id in final_semantic_entities:
            # Link Section to the semantic entities found within its chunk
            if chunk_id in sections_map:
                section_name = sections_map[chunk_id]['name']
                # Check if the entity is NOT a Section itself to avoid Section-CONTAINS->Section
                if entity_type != "Section":
                     containment_relations.add((section_name, entity_name, "CONTAINS", chunk_id))
            else:
                 logging.warning(f"Chunk ID '{chunk_id}' from semantic entity '{entity_name}' not found in sections map.")


        # --- Combine all entities and relations ---
        all_entities = structural_entities.union(final_semantic_entities)
        all_relations = structural_relations.union(semantic_relations_llm).union(containment_relations)

        logging.info(f"Total unique entities: {len(all_entities)}")
        logging.info(f"Total unique relations: {len(all_relations)}")
        logging.debug("Sample final entities:")
        for i, item in enumerate(list(all_entities)[:5]): logging.debug(item)
        logging.debug("Sample final relations:")
        for i, item in enumerate(list(all_relations)[:5]): logging.debug(item)


        # --- Phase 4: Generate Cypher ---
        logging.info("Phase 4: Generating Cypher statements...")
        # Update generate_cypher_statements if needed to handle the (name, type, context_id) format for entities
        # and (source_name, target_name, type, context_id) for relations
        cypher_statements = generate_cypher_statements(all_entities, all_relations) # Pass combined sets

        # --- Phase 5: Save Output ---
        logging.info(f"Saving Cypher statements to: {output_cypher_path}")
        output_dir = Path(output_cypher_path).parent
        output_dir.mkdir(parents=True, exist_ok=True) # Create output directory if it doesn't exist
        with open(output_cypher_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(cypher_statements))

        logging.info("Processing complete.")

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
    except ValueError as e:
        logging.error(f"Configuration or data error: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract entities and relations from structured JSON.")
    parser.add_argument("input_json", help="Path to the input demo3_merged.json file.")
    parser.add_argument("config", help="Path to the config_simple.yaml file.")
    parser.add_argument("output_cypher", help="Path to save the generated Cypher script.")
    args = parser.parse_args()

    # Basic check for environment variables needed for LLM
    if not LLM_API_KEY or not LLM_API_HOST or not LLM_MODEL:
         logging.warning("One or more LLM environment variables (API_KEY, HOST, MODEL) are not set. LLM calls will fail.")
         # Decide if you want to exit or continue (maybe for structural parsing only)
         # exit(1) # Uncomment to exit if LLM config is missing

    # Run the main async function
    asyncio.run(main(args.input_json, args.config, args.output_cypher))
```

**Key Changes in `entity_extract.py`:**

1.  **Configuration Loading:** Updated `load_config_and_setup_globals` to load the new config structure (`entity_types_llm`, `all_entity_types`, etc.).
2.  **Two-Phase Processing:** The `main` function orchestrates the new workflow:
    *   **`process_document_structure`:** New async function to parse `demo3_merged.json`. It identifies `Document` and `Section` entities and `HAS_SECTION`, `HAS_PARENT_SECTION` relations *without* LLM calls. It uses maps and sets with IDs internally, then resolves to names. Includes a basic heuristic for guessing the document title.
    *   **`extract_semantic_info`:** Revised async function.
        *   It now only creates LLM tasks for semantic entities (`ENTITY_TYPES_LLM`) and relations (`RELATION_TYPES_LLM`).
        *   Prompts are generated using the updated `create_entity_prompt` and `create_relation_prompt` which exclude structural elements from the LLM's task but provide context.
        *   It processes LLM results, focusing only on the requested semantic types.
    *   **Containment Relations:** After semantic extraction, the script explicitly creates `CONTAINS` relationships linking each `Section` (from phase 1) to the semantic entities extracted from its corresponding chunk's content (from phase 2).
    *   **Combining Results:** Entities and relations from both phases (structural and semantic) are combined into final sets (`all_entities`, `all_relations`).
    *   **Cypher Generation:** The existing `generate_cypher_statements` function is called with the final combined sets. Ensure this function correctly handles all entity/relation types defined in `all_entity_types` / `all_relation_types` and their mappings.

**Benefits of this Approach:**

1.  **Accuracy:** Document/Section structure is derived directly from the reliable JSON, eliminating LLM errors for this part.
2.  **Efficiency:** Reduces the number of LLM calls and the complexity of the prompts, potentially saving costs and time.
3.  **Focus:** Allows the LLM to concentrate on the more nuanced task of extracting semantic entities (Organizations, Roles, Statements, Topics) and their relationships within the context of a specific section.
4.  **Clarity:** Separates structural parsing from semantic interpretation.

Remember to test thoroughly after implementing these changes. You might need minor adjustments in normalization or Cypher generation depending on the exact output format needed.