import json
import os
import logging
import time # For potential rate limiting
import requests  # 用于HTTP请求，调用LLM API
from typing import List, Dict, Set, Tuple, Optional, Any

# --- Configuration ---

# 1. LLM Configuration
# 从环境变量获取API密钥，需要在运行前设置
# SiliconFlow API密钥可在账户设置中获取
# 可以通过执行 export LLM_BINDING_API_KEY="your_api_key_here" 设置环境变量
LLM_API_KEY = os.environ.get("LLM_BINDING_API_KEY") or os.environ.get("SILICONFLOW_API_KEY")
if not LLM_API_KEY:
    logging.error("环境变量 LLM_BINDING_API_KEY 或 SILICONFLOW_API_KEY 未设置，API调用将会失败")
    logging.error("请执行: export LLM_BINDING_API_KEY='your_api_key_here' 设置环境变量")

# 如果没有API密钥或需要测试，可以设置以下变量为True来启用测试模式
# 测试模式使用预设的响应，不会调用实际的API
TEST_MODE = os.environ.get("TEST_MODE", "").lower() in ["true", "1", "yes"] or not LLM_API_KEY
if TEST_MODE:
    logging.warning("测试模式已启用，将使用预设响应而非实际API调用")

# SiliconFlow API配置
LLM_API_HOST = os.environ.get("LLM_BINDING_HOST", "https://api.siliconflow.cn/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-14B-Instruct")

# 2. File Path
# 从环境变量获取输入目录或使用默认值
INPUT_DIR = "."
INPUT_JSON_PATH = os.environ.get("INPUT_JSON_PATH", os.path.join(INPUT_DIR, "demo3.json"))
OUTPUT_CYPHER_PATH = os.environ.get("OUTPUT_CYPHER_PATH", "output_graph.cypher")

# 3. Knowledge Graph Schema (Based on 改进建议.md)
ENTITY_TYPES = [
    "文档", "章节", "主题", "关键词", "人员", "角色",
    "组织", "时间", "事件", "法规"
]

# Map Chinese types to English for potentially better compatibility or preference
ENTITY_TYPE_MAP_CYPHER = {
    "文档": "Document",
    "章节": "Section",
    "主题": "Topic",
    "关键词": "Keyword",
    "人员": "Person",
    "角色": "Role",
    "组织": "Organization",
    "时间": "Time",
    "事件": "Event",
    "法规": "Regulation"
}


RELATION_TYPES = [
    "隶属关系", "版本关系", "引用", "依据", "责任",
    "审批", "时间", "生效", "关联"
]

# Map Chinese types to English (often uppercase) for Cypher relationships
RELATION_TYPE_MAP_CYPHER = {
    "隶属关系": "BELONGS_TO",
    "版本关系": "HAS_VERSION",
    "引用": "REFERENCES",
    "依据": "BASED_ON",
    "责任": "RESPONSIBLE_FOR",
    "审批": "APPROVED_BY",
    "时间": "OCCURS_AT", # Consider if Time should be a node instead
    "生效": "EFFECTIVE_FROM", # Consider if Time should be a node instead
    "关联": "RELATED_TO"
}


# 4. Entity Normalization Map (Example - expand as needed)
CANONICAL_MAP = {
    "客运部": "集团公司客运部",
    "信息技术所": "集团公司信息技术所",
    "科信部": "集团公司科信部",
    "财务部": "集团公司财务部",
    "计统部": "集团公司计统部",
    "电务部": "集团公司电务部",
    "供电部": "集团公司供电部",
    "宣传部": "集团公司宣传部",
    "调度所": "集团公司调度所",
    "集团公司应急领导小组办公室": "集团公司应急领导小组办公室", # Already specific
    "集团公司应急领导小组": "集团公司应急领导小组", # Already specific
    "国铁集团应急领导小组办公室": "国铁集团应急领导小组办公室",
    "国铁集团应急领导小组": "国铁集团应急领导小组",
    "国铁集团客运部": "国铁集团客运部",
    "12306科创中心": "12306科创中心",
    "广铁集团": "中国铁路广州局集团有限公司",
    "集团公司": "中国铁路广州局集团有限公司", # Be careful with context, might need more specific rules
    "本预案": "《广州局集团公司客票发售和预订系统（含互联网售票部分）应急预案》", # Main document title
    "《铁路客票发售和预订系统(含互联网售票部分)应急预案》": "《铁路客票发售和预订系统(含互联网售票部分)应急预案》（铁办客〔2021〕92号）", # Add identifier if possible
    "《广州局集团公司网络安全事件应急预案》": "《广州局集团公司网络安全事件应急预案》（广铁科信〔2019〕105号）",
    "《广州局集团公司信息系统故障应急处置和调查处理办法》": "《广州局集团公司信息系统故障应急处置和调查处理办法》（广铁科信发〔2022〕76号）",
    "客票系统": "客票发售和预订系统", # Normalize system name
}

# 5. Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Helper Functions ---

def load_json_data(file_path: str) -> Optional[List[Dict[str, Any]]]:
    """Loads data from a JSON file."""
    # [ Function code remains the same ]
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logging.info(f"Successfully loaded data from {file_path}")
        return data
    except FileNotFoundError:
        logging.error(f"Error: Input file not found at {file_path}")
        return None
    except json.JSONDecodeError:
        logging.error(f"Error: Could not decode JSON from {file_path}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during JSON loading: {e}")
        return None


def normalize_entity_name(raw_name: str) -> str:
    """Normalizes entity names using the CANONICAL_MAP."""
    # [ Function code remains the same ]
    if not isinstance(raw_name, str):
        logging.warning(f"Attempted to normalize non-string value: {raw_name}. Returning as is.")
        return str(raw_name)
    cleaned_name = raw_name.strip().replace('\n', ' ')
    return CANONICAL_MAP.get(cleaned_name, cleaned_name)

def escape_cypher_string(value: str) -> str:
    """Escapes single quotes and backslashes for Cypher strings."""
    if not isinstance(value, str):
        return str(value) # Return as string if not already
    return value.replace('\\', '\\\\').replace("'", "\\'")

def create_entity_prompt(chunk_content: str) -> str:
    """Creates the prompt for entity extraction."""
    # [ Function code remains the same ]
    definitions = """
    实体类型定义:
    - 文档：管理规定的文件名称，如《应急预案》。
    - 章节：文档中的具体章节标题，如"1 总则"。
    - 主题：文档或章节的核心议题，如"应急组织机构"。
    - 关键词：文本中重要的名词或术语，如"客票系统"、"应急响应"、"电子客票"。
    - 人员：具体的人名（此文档中可能较少）。
    - 角色：指代具有特定职责的职位或岗位，如"客运部主任"、"售票员"。
    - 组织：涉及的单位、部门或公司，如"中国铁路广州局集团有限公司"、"集团公司客运部"、"信息技术所"、"各车务站段"。
    - 时间：具体的日期、时间点或时间段，如"2021年"、"4小时及以上"、"每年3月"。
    - 事件：文档中描述的具体活动或状况，如"系统突发事件"、"启动应急预案"、"应急演练"、"售票故障"。
    - 法规：引用的其他法规或文件名称及其编号，如"《铁路客票发售和预订系统(含互联网售票部分)应急预案》（铁办客〔2021〕92号）"。
    """
    prompt = f"""
请从以下文本中提取定义的实体类型。

{definitions}

预定义的实体类型列表: {', '.join(ENTITY_TYPES)}

文本：
\"\"\"
{chunk_content}
\"\"\"

请以严格的 JSON 格式输出，包含一个名为 "entities" 的列表，其中每个对象包含 "name" (实体名称) 和 "type" (实体类型)。确保实体名称是文本中实际出现的词语。
例如:
{{
  "entities": [
    {{"name": "集团公司客运部", "type": "组织"}},
    {{"name": "售票故障", "type": "事件"}},
    {{"name": "《铁路客票发售和预订系统(含互联网售票部分)应急预案》（铁办客〔2021〕92号）", "type": "法规"}}
  ]
}}
"""
    return prompt

def create_relation_prompt(chunk_content: str) -> str:
    """Creates the prompt for relation extraction (within the chunk)."""
    # [ Function code remains the same ]
    definitions = """
    关系类型定义 (请仅提取文本段落内明确描述的关系):
    - 隶属关系 (BelongsTo): 通常是结构化的，此提示词主要关注文本内描述，如"办公室设在客运部"。(结构化部分将后处理)
    - 版本关系 (HasVersion): 指明文档的版本信息或与其他版本的关系 (如"修订版"、"废止旧版")。
    - 引用 (References): 一个实体提到了另一个实体或文件，如"详见附件5"。
    - 依据 (BasedOn): 指出制定某文件或采取某行动所依据的法规或原则，如"根据...制定本预案"。
    - 责任 (ResponsibleFor): 指明某个角色或组织负责某项任务或职责，如"客运部负责协调"。
    - 审批 (ApprovedBy): 指出某事项需要经过哪个组织或角色批准，如"经...同意后"。
    - 时间 (OccursAt): 事件发生的时间，或规定适用的时间点/段，如"事件影响4小时"、"每年3月开展演练"。
    - 生效 (EffectiveFrom): 规定或文件的生效日期，如"自发布之日起实施"。
    - 关联 (RelatedTo): 实体间的其他关联，如"与...不一致时，以此为准"。
    """
    prompt = f"""
请从以下文本中提取实体之间的关系。请专注于在文本段落中**直接陈述**的关系。

{definitions}

预定义的关系类型列表: {', '.join(RELATION_TYPES)}

文本：
\"\"\"
{chunk_content}
\"\"\"

请以严格的 JSON 格式输出，包含一个名为 "relations" 的列表，其中每个对象包含 "source" (源实体名称), "target" (目标实体名称), 和 "type" (关系类型)。确保实体名称是文本中实际出现的词语。
例如:
{{
  "relations": [
    {{"source": "集团公司应急领导小组办公室", "target": "集团公司客运部", "type": "隶属关系"}},
    {{"source": "本预案", "target": "《铁路客票发售和预订系统(含互联网售票部分)应急预案》", "type": "依据"}},
    {{"source": "客运部", "target": "协调各相关部门", "type": "责任"}}
  ]
}}
"""
    return prompt


# --- LLM Interaction (Placeholder) ---

def call_llm(prompt: str) -> Optional[str]:
    """
    调用LLM API进行推理
    在测试模式下返回预设响应，否则调用实际API
    """
    logging.info(f"--- 发送Prompt到LLM (长度: {len(prompt)}) ---")
    
    # 测试模式: 使用预设响应而非调用API
    if TEST_MODE:
        logging.warning("使用测试模式的预设响应")
        if "提取定义的实体类型" in prompt:
            logging.warning("返回预设的实体提取响应")
            if "集团公司成立客票系统应急领导小组" in prompt:
                return json.dumps({ 
                    "entities": [ 
                        {"name": "集团公司", "type": "组织"}, 
                        {"name": "客票系统应急领导小组", "type": "组织"},
                        {"name": "集团公司应急领导小组", "type": "组织"}, 
                        {"name": "科信部", "type": "组织"}, 
                        {"name": "客运部", "type": "组织"},
                        {"name": "各车务站段", "type": "组织"}
                    ] 
                })
            return json.dumps({"entities": []})
        elif "提取实体之间的关系" in prompt:
            logging.warning("返回预设的关系提取响应")
            if "集团公司成立客票系统应急领导小组" in prompt:
                return json.dumps({ 
                    "relations": [ 
                        {"source": "集团公司应急领导小组办公室", "target": "集团公司客运部", "type": "隶属关系"},
                        {"source": "各车务站段", "target": "集团公司", "type": "隶属关系"} 
                    ] 
                })
            return json.dumps({"relations": []})
        else: 
            return None
    
    # 实际模式: 检查API密钥并调用API
    if not LLM_API_KEY:
        logging.error("LLM API密钥未设置，无法调用API")
        logging.error("请设置环境变量 LLM_BINDING_API_KEY 或 SILICONFLOW_API_KEY")
        return None
        
    try:
        # 使用SiliconFlow API (OpenAI兼容接口)调用Qwen模型
        # 参考OpenAI API文档: https://platform.openai.com/docs/api-reference/chat
        api_url = f"{LLM_API_HOST}/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {LLM_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # OpenAI兼容接口的参数
        payload = {
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": "你是一个实体关系提取助手，善于从文本中提取结构化信息并以JSON格式输出。"},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 2000,
            "temperature": 0.1,  # 设置较低的温度以获得确定性结果
            "response_format": {"type": "text"}
        }
        
        logging.info("--- 等待LLM响应 ---")
        response = requests.post(api_url, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 200:
            response_json = response.json()
            # 解析OpenAI API返回的JSON结果
            llm_response = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
            logging.info(f"--- 成功接收LLM响应 (长度: {len(llm_response)}) ---")
            return llm_response
        elif response.status_code == 401:
            logging.error("API认证失败: 无效的API密钥。请检查API密钥是否正确设置。")
            return None
        else:
            logging.error(f"API调用失败: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        logging.error(f"API请求错误: {e}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"JSON解析错误: {e}")
        return None
    except Exception as e:
        logging.error(f"调用LLM时发生未知错误: {e}")
        return None


def parse_llm_response(response_text: Optional[str]) -> Optional[Dict[str, List[Dict[str, str]]]]:
    """Safely parses the LLM's JSON response."""
    # [ Function code remains the same ]
    if not response_text: return None
    try:
        if response_text.strip().startswith("```json"):
            response_text = response_text.strip()[7:-3].strip()
        elif response_text.strip().startswith("`"):
             response_text = response_text.strip()[1:-1].strip()
        parsed_data = json.loads(response_text)
        if isinstance(parsed_data, dict) and \
           (('entities' in parsed_data and isinstance(parsed_data['entities'], list)) or \
            ('relations' in parsed_data and isinstance(parsed_data['relations'], list))):
            return parsed_data
        else:
            logging.warning(f"LLM response is valid JSON but not the expected structure: {response_text}")
            return None
    except json.JSONDecodeError as e:
        logging.error(f"Failed to decode LLM JSON response: {e}\nResponse text:\n{response_text}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during LLM response parsing: {e}\nResponse text:\n{response_text}")
        return None

# --- Cypher Generation ---

def generate_cypher_statements(entities: Set[Tuple[str, str]], relations: Set[Tuple[str, str, str]]) -> List[str]:
    """Generates Memgraph/Neo4j Cypher MERGE statements."""
    cypher_statements = []

    # Add constraint for uniqueness if desired (recommended)
    # You might want to run these manually once or ensure they exist
    # for entity_type_cn in ENTITY_TYPES:
    #     entity_type_cypher = ENTITY_TYPE_MAP_CYPHER.get(entity_type_cn, entity_type_cn) # Use mapped or original
    #     cypher_statements.append(f"CREATE CONSTRAINT ON (n:{entity_type_cypher}) ASSERT n.name IS UNIQUE;")

    cypher_statements.append("\n// --- Entity Creation ---")
    sorted_entities = sorted(list(entities))
    for name, type_cn in sorted_entities:
        if not name: # Skip empty names
            continue
        entity_type_cypher = ENTITY_TYPE_MAP_CYPHER.get(type_cn, type_cn.replace(" ", "_")) # Map or sanitize
        escaped_name = escape_cypher_string(name)
        # Use MERGE to avoid duplicates
        cypher_statements.append(f"MERGE (:`{entity_type_cypher}` {{name: '{escaped_name}'}});")

    cypher_statements.append("\n// --- Relationship Creation ---")
    sorted_relations = sorted(list(relations))
    for source, target, type_cn in sorted_relations:
        if not source or not target: # Skip if source or target is missing
            continue
        relation_type_cypher = RELATION_TYPE_MAP_CYPHER.get(type_cn, type_cn.upper().replace(" ", "_")) # Map or sanitize
        escaped_source = escape_cypher_string(source)
        escaped_target = escape_cypher_string(target)

        # Find the types of source and target from the entities set for better matching
        source_type_cn = next((t for n, t in sorted_entities if n == source), None)
        target_type_cn = next((t for n, t in sorted_entities if n == target), None)

        if source_type_cn and target_type_cn:
            source_type_cypher = ENTITY_TYPE_MAP_CYPHER.get(source_type_cn, source_type_cn.replace(" ", "_"))
            target_type_cypher = ENTITY_TYPE_MAP_CYPHER.get(target_type_cn, target_type_cn.replace(" ", "_"))
            # Use MERGE for relationships too
            cypher_statements.append(
                f"MATCH (a:`{source_type_cypher}` {{name: '{escaped_source}'}}), (b:`{target_type_cypher}` {{name: '{escaped_target}'}}) "
                f"MERGE (a)-[:`{relation_type_cypher}`]->(b);"
            )
        else:
             # Fallback if type not found (less safe, might match wrong nodes if names overlap across types)
             logging.warning(f"Could not determine types for relationship: ({source})-[{type_cn}]->({target}). Using generic MATCH.")
             cypher_statements.append(
                 f"MATCH (a {{name: '{escaped_source}'}}), (b {{name: '{escaped_target}'}}) "
                 f"MERGE (a)-[:`{relation_type_cypher}`]->(b);"
             )

    return cypher_statements


# --- Main Processing Logic ---

def main():
    """Main function to orchestrate the KG extraction process."""
    # 验证输入文件是否存在
    if not os.path.exists(INPUT_JSON_PATH):
        logging.error(f"输入文件 {INPUT_JSON_PATH} 不存在，请确保文件路径正确")
        return
        
    data = load_json_data(INPUT_JSON_PATH)
    if not data:
        return

    entities_set: Set[Tuple[str, str]] = set()
    relations_set: Set[Tuple[str, str, str]] = set()
    chunk_map: Dict[str, Dict[str, Any]] = {chunk['chunk_id']: chunk for chunk in data}

    # --- Phase 1: LLM Extraction and Normalization ---
    # [ This loop remains the same as before ]
    for i, chunk in enumerate(data):
        chunk_id = chunk.get("chunk_id", f"unknown_{i}")
        content = chunk.get("content")
        heading = chunk.get("heading", "")
        if not content: continue
        logging.info(f"Processing chunk {i+1}/{len(data)}: ID {chunk_id} (Heading: {heading[:50]}...)")

        # Extract Entities
        entity_prompt = create_entity_prompt(content)
        entity_response_text = call_llm(entity_prompt)
        parsed_entities = parse_llm_response(entity_response_text)
        if parsed_entities and 'entities' in parsed_entities:
            for entity in parsed_entities['entities']:
                raw_name = entity.get('name')
                raw_type = entity.get('type')
                if raw_name and raw_type and raw_type in ENTITY_TYPES:
                    normalized_name = normalize_entity_name(raw_name)
                    entities_set.add((normalized_name, raw_type))
                else: logging.warning(f"Invalid entity format or type in chunk {chunk_id}: {entity}")

        # Extract Relations
        relation_prompt = create_relation_prompt(content)
        relation_response_text = call_llm(relation_prompt)
        parsed_relations = parse_llm_response(relation_response_text)
        if parsed_relations and 'relations' in parsed_relations:
            for relation in parsed_relations['relations']:
                raw_source = relation.get('source')
                raw_target = relation.get('target')
                raw_type = relation.get('type')
                if raw_source and raw_target and raw_type and raw_type in RELATION_TYPES:
                    normalized_source = normalize_entity_name(raw_source)
                    normalized_target = normalize_entity_name(raw_target)
                    relations_set.add((normalized_source, normalized_target, raw_type))
                else: logging.warning(f"Invalid relation format or type in chunk {chunk_id}: {relation}")

    logging.info(f"Finished LLM extraction. Found {len(entities_set)} unique entities and {len(relations_set)} unique relations so far.")

    # --- Phase 2: Add Structural Relations (BelongsTo) ---
    # [ This loop remains the same as before ]
    logging.info("Adding structural 'BelongsTo' relations...")
    structural_relations_added = 0
    for chunk in data:
        chunk_id = chunk.get("chunk_id")
        parent_id = chunk.get("parent_id")
        heading = chunk.get("heading")
        if parent_id and chunk_id and heading:
            parent_chunk = chunk_map.get(parent_id)
            if parent_chunk and parent_chunk.get("heading"):
                child_entity_name = normalize_entity_name(heading)
                parent_entity_name = normalize_entity_name(parent_chunk["heading"])
                if child_entity_name and parent_entity_name: # Ensure names are not empty
                    child_type = "章节"
                    parent_type = "章节" if parent_chunk.get("parent_id") else "文档"
                    entities_set.add((child_entity_name, child_type))
                    entities_set.add((parent_entity_name, parent_type))
                    relation_tuple = (child_entity_name, parent_entity_name, "隶属关系")
                    if relation_tuple not in relations_set:
                        relations_set.add(relation_tuple)
                        structural_relations_added += 1
    logging.info(f"Added {structural_relations_added} structural 'BelongsTo' relations.")


    # --- Output Results ---

    # 1. Console Output (Original)
    print("\n--- Final Extracted Entities (Console) ---")
    sorted_entities = sorted(list(entities_set))
    for name, type in sorted_entities:
        print(f"- ({type}) {name}")
    print(f"\nTotal Unique Entities: {len(sorted_entities)}")

    print("\n--- Final Extracted Relations (Console) ---")
    sorted_relations = sorted(list(relations_set))
    for source, target, type in sorted_relations:
        print(f"- {source} --[{type}]--> {target}")
    print(f"\nTotal Unique Relations: {len(sorted_relations)}")

    # 2. Cypher Statement Generation and Output
    print(f"\n--- Generating Cypher Statements (Memgraph/Neo4j) ---")
    cypher_statements = generate_cypher_statements(entities_set, relations_set)

    # Option A: Print to console
    # print("\n".join(cypher_statements))

    # Option B: Save to file
    try:
        with open(OUTPUT_CYPHER_PATH, 'w', encoding='utf-8') as f:
            f.write(";\n".join(cypher_statements) + ";\n") # Add semicolon after each statement
        print(f"\nCypher statements saved to: {OUTPUT_CYPHER_PATH}")
    except IOError as e:
        print(f"\nError writing Cypher statements to file {OUTPUT_CYPHER_PATH}: {e}")
        print("\nCypher Statements:\n")
        print(";\n".join(cypher_statements) + ";\n") # Print to console as fallback


if __name__ == "__main__":
    main()

# README:
# 
# 本脚本用于从文本中提取实体和关系，构建知识图谱
# 
# 使用说明:
# 1. 设置API密钥(二选一):
#    export LLM_BINDING_API_KEY="your_api_key_here"
#    export SILICONFLOW_API_KEY="your_api_key_here"
#
# 2. 设置API主机(可选):
#    export LLM_BINDING_HOST="https://api.siliconflow.cn/v1"
#
# 3. 设置模型(可选):
#    export LLM_MODEL="Qwen/Qwen2.5-14B-Instruct"
#
# 4. 或使用测试模式，不需要API密钥:
#    export TEST_MODE=true
#
# 5. 设置输入/输出路径(可选):
#    export INPUT_DIR="/app/data/inputs/passenger"
#    export INPUT_JSON_PATH="/app/data/inputs/passenger/demo3.json"
#    export OUTPUT_CYPHER_PATH="output_graph.cypher"
#
# 6. 准备输入文件(JSON格式，默认为demo3.json)，确保文件格式正确
#
# 7. 运行脚本:
#    python entity_extract.py
#
# 8. 查看输出结果:
#    - 控制台将显示提取的实体和关系
#    - Cypher语句将被保存到output_graph.cypher
#
# 可自定义的配置:
# - ENTITY_TYPES: 实体类型列表
# - RELATION_TYPES: 关系类型列表
# - CANONICAL_MAP: 实体名称标准化映射
