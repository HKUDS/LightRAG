"""
提示词构建模块

负责构建用于实体和关系提取的提示词
"""

from typing import Dict, Any
from graph_tools.models import Config

def create_entity_prompt(chunk_content: str, context: Dict[str, Any], config: Config) -> str:
    """
    创建实体提取的提示词
    
    Args:
        chunk_content: 要处理的文本内容
        context: 上下文信息，包含文档标题、当前章节等
        config: 配置对象
        
    Returns:
        str: 用于实体提取的完整提示词
    """
    # 从配置中获取模板
    template = config.prompt_templates.get('entity_extraction', {}).get('template', '')
    definitions = config.prompt_templates.get('entity_extraction', {}).get('definitions', '')
    
    # 获取实体类型列表
    entity_types = config.entity_types_llm
    entity_types_english = [config.entity_type_map_cypher.get(t, t) for t in entity_types]
    
    # 获取文档上下文信息
    document_title = context.get('document_title', "未知文档")
    current_heading = context.get('current_heading', "未知章节")
    section_path = context.get('section_path', "")
    parent_section_summary = context.get('parent_section_summary', "")
    
    # 强化对实体类型的约束
    strict_entity_type_guidance = """
【特别强调】：
1. 必须严格使用以下英文实体类型，不得使用其他类型：{entity_types_english}
2. 请勿将关系类型（如RESPONSIBLE_FOR, APPLIES_TO, HAS_PURPOSE等）错误地用作实体类型
3. 请确保每个提取的实体都分配了正确的实体类型
"""
    
    # 先使用安全的手工构建的提示词，完全避开模板格式化的问题
    prompt = f"""请从以下文本中提取定义的实体类型。专注于识别组织、角色、具体规定陈述、以及预定义的规定类型(Topic)。

{definitions}

要提取的实体类型列表: {', '.join(entity_types)}

文档信息 (仅供参考，不要提取为实体):
文档标题: {document_title}
当前章节: {current_heading}
章节路径: {section_path}
父章节内容摘要: {parent_section_summary}

文本：
\"\"\"
{chunk_content}
\"\"\"

【重要】：必须使用英文实体类型！返回的JSON中，实体类型必须为英文，且仅包含以下类型: {', '.join(entity_types_english)}

请以严格的JSON格式输出，包含一个名为"entities"的列表，其中每个对象包含"name"(实体名称)和"type"(实体类型)。确保实体名称是文本中实际出现的词语。

注意事项：
1. 不要提取文档标题或章节标题作为实体。
2. 确保每个实体完整识别。
3. 识别实体时考虑缩写和全称。
4. 保留相同实体的不同表述。
5. 确保每个实体都准确分配了正确的英文实体类型。
6. 严格使用以下英文类型名: {', '.join(entity_types_english)}
7. 类型区分指南：
   - Topic与Statement的区别：Topic是规定内容的分类标签（如"服务质量"），Statement是具体的规定本身（如"应使用规范用语"）。
   - Organization与Role的区别：Organization是部门/单位，Role是岗位/职责。
   - 文档名称、预案名称：带有《》括号的文件名不是Organization，如"《广州局集团公司网络安全事件应急预案》"不是Organization。
   - 信息系统：各种信息系统如"客票发售和预订系统"、"12306系统"等不是Organization，应考虑作为Statement或Topic提取（如果符合这些类型的定义）。
8. 利用文档上下文信息辅助判断。

{strict_entity_type_guidance.format(entity_types_english=', '.join(entity_types_english))}

JSON输出示例：
{{
  "entities": [
    {{"name": "集团公司客运部", "type": "Organization"}},
    {{"name": "应使用规范用语，保持微笑服务", "type": "Statement"}},
    {{"name": "服务质量", "type": "Topic"}},
    {{"name": "站务员", "type": "Role"}}
  ]
}}"""
    
    return prompt

def create_relation_prompt(chunk_content: str, entities_json: str, context: Dict[str, Any], config: Config) -> str:
    """
    创建关系提取的提示词
    
    Args:
        chunk_content: 要处理的文本内容
        entities_json: 已提取实体的JSON字符串
        context: 上下文信息，包含文档标题、当前章节等
        config: 配置对象
        
    Returns:
        str: 用于关系提取的完整提示词
    """
    # 从配置中获取模板
    template = config.prompt_templates.get('relation_extraction', {}).get('template', '')
    definitions = config.prompt_templates.get('relation_extraction', {}).get('definitions', '')
    
    # 获取关系类型列表
    relation_types = config.relation_types_llm
    relation_types_english = [config.relation_type_map_cypher.get(t, t) for t in relation_types]
    
    # 获取所有实体类型（包括Document和Section）
    all_entity_types = config.all_entity_types
    all_entity_types_english = [config.entity_type_map_cypher.get(t, t) for t in all_entity_types]
    
    # 获取文档上下文信息
    document_title = context.get('document_title', "未知文档")
    current_heading = context.get('current_heading', "未知章节")
    section_path = context.get('section_path', "")
    
    # 强化对关系类型的约束
    strict_relation_type_guidance = """
【特别强调】：
1. 必须严格使用以下英文关系类型，不得使用其他类型：{relation_types_english}
2. source_type和target_type必须严格使用以下英文实体类型：{all_entity_types_english}
3. 请勿将实体类型（如Organization, Statement, Topic等）错误地用作关系类型
4. 确保每个关系的source和target都是文本中实际提取的实体
"""
    
    # 直接使用安全的手工构建的提示词，完全避开模板格式化的问题
    prompt = f"""请从以下文本中提取定义的关系类型。根据预定义的实体列表提取这些实体之间符合定义的关系类型。请专注于在文本段落中直接陈述的语义关系。

{definitions}

要提取的关系类型列表: {', '.join(relation_types)}
预定义的实体类型列表 (用于关系端点): {', '.join(all_entity_types_english)}

文档信息 (仅供参考):
文档标题: {document_title}
当前章节: {current_heading}
章节路径: {section_path}

文本：
\"\"\"
{chunk_content}
\"\"\"

文本中已识别的语义实体列表 (用于构建关系):
\"\"\"
{entities_json}
\"\"\"

【重要】：必须使用英文实体类型和关系类型！

请以严格的JSON格式输出，包含一个名为"relations"的列表，其中每个对象必须包含"source", "source_type", "target", "target_type", "type"字段。

注意事项:
1. 关系必须连接上面语义实体列表中的实体，或者连接语义实体到已知的章节(Section)或文档(Document)（如果文本明确引用）。
2. source_type和target_type必须是以下英文类型之一: {', '.join(all_entity_types_english)}
3. 关系类型必须是预定义的英文类型之一: {', '.join(relation_types_english)}
4. 不要提取Document -> Section或Section -> Section的结构关系。
5. CONTAINS关系将由脚本自动处理（连接章节和其包含的语义实体），不要让LLM提取CONTAINS。
6. 关系类型使用指南：
   - RESPONSIBLE_FOR：描述对具体规则(Statement)或规则类型(Topic)的责任。
   - MENTIONS：用于连接章节(Section)和其明确提及的规定类型(Topic)。
   - REFERENCES: 用于文本中明确提到的对其他章节或文档的引用 (e.g., "详见章节 3.1")。不要用于连接非文档/章节的实体。例如，系统名称、业务流程等非章节/文档的引用应使用RELATED_TO关系。
   - RELATED_TO: 用于表达其他关系类型无法清晰表达的联系，包括提及系统、业务流程等非文档/章节的情况。

{strict_relation_type_guidance.format(
    relation_types_english=', '.join(relation_types_english),
    all_entity_types_english=', '.join(all_entity_types_english)
)}

JSON输出示例：
{{
  "relations": [
    {{
      "source": "集团公司客运部",
      "source_type": "Organization",
      "target": "服务质量",
      "target_type": "Topic",
      "type": "RESPONSIBLE_FOR"
    }},
    {{
      "source": "应使用规范用语，保持微笑服务",
      "source_type": "Statement",
      "target": "站务员",
      "target_type": "Role",
      "type": "APPLIES_TO"
    }}
  ]
}}"""
    
    return prompt 