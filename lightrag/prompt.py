from __future__ import annotations
from typing import Any


PROMPTS: dict[str, Any] = {}

PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

PROMPTS["DEFAULT_USER_PROMPT"] = "n/a"

PROMPTS["entity_extraction"] = """---任务---
对于给定的文本和实体类型列表，提取所有实体及其关系，然后以下述指定语言和格式返回。

---指令---
1. 识别文本中明确概念化的实体。对于每个识别的实体，提取以下信息：
  - entity_name: 实体名称，使用与输入文本相同的语言。如果是英文，请将名称首字母大写
  - entity_type: 使用提供的`Entity_types`列表对实体进行分类。如果无法确定合适的类别，请将其分类为`Other`
  - entity_description: 根据输入文本中的信息，提供实体属性和活动的全面描述。为确保清晰和准确，所有描述必须将代词和指代词（如"这个文档"、"我们公司"、"我"、"你"、"他/她"）替换为它们所代表的具体名词
2. 将每个实体格式化为：(entity{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)
3. 从识别的实体中，识别所有直接且明确相关的(source_entity, target_entity)对，并提取以下信息：
  - source_entity: 源实体名称
  - target_entity: 目标实体名称
  - relationship_keywords: 一个或多个高级关键词，总结关系的总体性质，重点关注概念或主题而非具体细节
  - relationship_description: 解释源实体和目标实体之间关系的性质，为它们的连接提供清晰的理由
4. 将每个关系格式化为：(relationship{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_description>)
5. 使用`{tuple_delimiter}`作为字段分隔符。使用`{record_delimiter}`作为实体或关系列表分隔符。
6. 当所有实体和关系都提取完成时，输出`{completion_delimiter}`。
7. 确保输出语言为{language}。
8. 必须使用半角字符作为分隔符，例如半角竖线"|"而不是全角竖线"｜"。
9. 一定不要出现符号顺序颠倒的情况。

---质量指南---
- 只提取在上下文中明确定义且有意义的实体和关系
- 避免过度解释；坚持文本中明确陈述的内容
- 对于所有输出内容，明确命名主体或客体，而不是使用代词
- 在相关时在实体名称中包含具体的数值数据
- 确保实体名称在整个提取过程中保持一致

---示例---
{examples}

---输入---
Entity_types: [{entity_types}]
Text:
```
{input_text}
```

---输出---
"""

PROMPTS["entity_continue_extraction"] = """---任务---
识别上次提取任务中遗漏的任何实体或关系。

---指令---
1. 以与之前提取任务相同的格式输出实体和关系。
2. 不要包含之前已经提取过的实体和关系。
3. 如果实体不能明确归类到提供的`Entity_types`中的任何一个，请将其分类为"Other"。
4. 确保输出语言为{language}。
5. 必须使用半角字符作为分隔符，例如半角竖线"|"而不是全角竖线"｜"。
6. 一定不要出现符号顺序颠倒的情况。

---输出---
"""

PROMPTS["entity_extraction_examples"] = [
   """[示例 1]

---输入---
实体类型: ["产品编码", "工序", "维修完成时间", "故障现象", "不良现象说明", "问题定位", "原因分类", "原因子类", "原因代码", "维修说明", "不良单板编码", "元件位号"]
文本:
```
故障数据块 #2

记录 7:
  产品编码: M0007
  工序: T2
  维修完成时间: 2025/7/4 11:03:24
  故障现象: SMT-AOI和测试段不良使用-勿删
  不良现象说明: [175：测试设置（检测第四通道18mA）|Min:17960|Max:18040|Mea:17951]
  问题定位: 研发设计类问题
  原因分类: 产品设计缺陷
  原因子类: 其他产品设计缺陷
  原因代码: D70130
  维修说明: 电流检测电路设计缺陷，更换R133
  不良单板编码: DB0009
  元件位号: R133
```
---输出---
(entity{tuple_delimiter}M0007{tuple_delimiter}产品编码{tuple_delimiter}“M0007” 是在 “T2” 工序出现故障产品的产品编码。){record_delimiter}
(entity{tuple_delimiter}T2{tuple_delimiter}工序{tuple_delimiter}“T2” 是编码为 “M0007” 的产品在制造过程中出现问题的工序。){record_delimiter}
(entity{tuple_delimiter}2025/7/4 11:03:24{tuple_delimiter}维修完成时间{tuple_delimiter}“2025年7月4日11时03分24秒” 是产品 “M0007” 在 “T2” 工序出现 “SMT-AOI和测试段不良使用-勿删” 故障现象后的维修完成时间。){record_delimiter}
(entity{tuple_delimiter}SMT-AOI和测试段不良使用-勿删{tuple_delimiter}故障现象{tuple_delimiter}“SMT-AOI和测试段不良使用-勿删” 是产品 “M0007” 在工序 “T2” 出现的故障现象。){record_delimiter}
(entity{tuple_delimiter}[175：测试设置（检测第四通道18mA）|Min:17960|Max:18040|Mea:17951]{tuple_delimiter}不良现象说明{tuple_delimiter}“[175：测试设置（检测第四通道18mA）|Min:17960|Max:18040|Mea:17951]” 是故障现象 “SMT-AOI和测试段不良使用-勿删” 的不良现象说明。){record_delimiter}
(entity{tuple_delimiter}研发设计类问题{tuple_delimiter}问题定位{tuple_delimiter}“研发设计类问题” 是原因分类 “产品设计缺陷” 对应的问题定位。){record_delimiter}
(entity{tuple_delimiter}产品设计缺陷{tuple_delimiter}原因分类{tuple_delimiter}“产品设计缺陷” 是原因子类 “其他产品设计缺陷” 对应的原因分类。){record_delimiter}
(entity{tuple_delimiter}其他产品设计缺陷{tuple_delimiter}原因子类{tuple_delimiter}“其他产品设计缺陷” 是造成不良现象 “[175：测试设置（检测第四通道18mA）|Min:17960|Max:18040|Mea:17951]” 的具体原因。){record_delimiter}
(entity{tuple_delimiter}D70130{tuple_delimiter}原因代码{tuple_delimiter}“D70130” 是原因子类 “其他产品设计缺陷” 对应的唯一原因代码。){record_delimiter}
(entity{tuple_delimiter}电流检测电路设计缺陷，更换R133{tuple_delimiter}维修说明{tuple_delimiter}“电流检测电路设计缺陷，更换R133” 是原因子类 “其他产品设计缺陷” 的维修说明。){record_delimiter}
(entity{tuple_delimiter}R133{tuple_delimiter}元件位号{tuple_delimiter}“R133” 是维修说明 “电流检测电路设计缺陷，更换R133” 用于标识更换的原件位号。){record_delimiter}
(relationship{tuple_delimiter}[175：测试设置（检测第四通道18mA）|Min:17960|Max:18040|Mea:17951]{tuple_delimiter}M0007{tuple_delimiter}不良现象{tuple_delimiter}不良现象说明 “[175：测试设置（检测第四通道18mA）|Min:17960|Max:18040|Mea:17951]” 在产品编码为 “M0007” 的产品中出现。){record_delimiter}
(relationship{tuple_delimiter}[175：测试设置（检测第四通道18mA）|Min:17960|Max:18040|Mea:17951]{tuple_delimiter}T2{tuple_delimiter}生产工序{tuple_delimiter}不良现象说明 “[175：测试设置（检测第四通道18mA）|Min:17960|Max:18040|Mea:17951]” 出现在生产工序 “T2” 中。){record_delimiter}
(relationship{tuple_delimiter}[175：测试设置（检测第四通道18mA）|Min:17960|Max:18040|Mea:17951]{tuple_delimiter}2025/7/4 11:03:24{tuple_delimiter}维修时间{tuple_delimiter}不良现象说明 “[175：测试设置（检测第四通道18mA）|Min:17960|Max:18040|Mea:17951]” 对应的维修完成时间为 “2025年7月4日11时03分24秒”。){record_delimiter}
(relationship{tuple_delimiter}[175：测试设置（检测第四通道18mA）|Min:17960|Max:18040|Mea:17951]{tuple_delimiter}SMT-AOI和测试段不良使用-勿删{tuple_delimiter}故障表现{tuple_delimiter}故障现象 “SMT-AOI和测试段不良使用-勿删” 的不良现象说明为 “[175：测试设置（检测第四通道18mA）|Min:17960|Max:18040|Mea:17951]”。){record_delimiter}
(relationship{tuple_delimiter}[175：测试设置（检测第四通道18mA）|Min:17960|Max:18040|Mea:17951]{tuple_delimiter}研发设计类问题{tuple_delimiter}问题定位{tuple_delimiter}不良现象说明 “[175：测试设置（检测第四通道18mA）|Min:17960|Max:18040|Mea:17951]” 对应的问题定位为 “研发设计类问题”。){record_delimiter}
(relationship{tuple_delimiter}[175：测试设置（检测第四通道18mA）|Min:17960|Max:18040|Mea:17951]{tuple_delimiter}产品设计缺陷{tuple_delimiter}原因分类{tuple_delimiter}不良现象说明 “[175：测试设置（检测第四通道18mA）|Min:17960|Max:18040|Mea:17951]” 所属原因分类为 “产品设计缺陷”。){record_delimiter}
(relationship{tuple_delimiter}[175：测试设置（检测第四通道18mA）|Min:17960|Max:18040|Mea:17951]{tuple_delimiter}其他产品设计缺陷{tuple_delimiter}原因子类{tuple_delimiter}不良现象说明 “[175：测试设置（检测第四通道18mA）|Min:17960|Max:18040|Mea:17951]” 所属原因子类为 “其他产品设计缺陷”。){record_delimiter}
(relationship{tuple_delimiter}[175：测试设置（检测第四通道18mA）|Min:17960|Max:18040|Mea:17951]{tuple_delimiter}电流检测电路设计缺陷，更换R133{tuple_delimiter}维修措施{tuple_delimiter}不良现象说明 “[175：测试设置（检测第四通道18mA）|Min:17960|Max:18040|Mea:17951]” 对应的维修措施为 “电流检测电路设计缺陷，更换 R133”。){record_delimiter}
(relationship{tuple_delimiter}其他产品设计缺陷{tuple_delimiter}产品设计缺陷{tuple_delimiter}原因分类{tuple_delimiter}原因子类 “其他产品设计缺陷” 对应的原因分类为 “产品设计缺陷”。){record_delimiter}
(relationship{tuple_delimiter}其他产品设计缺陷{tuple_delimiter}D70130{tuple_delimiter}原因代码{tuple_delimiter}原因子类 “其他产品设计缺陷” 对应的原因代码为 “D70130”。){record_delimiter}
(relationship{tuple_delimiter}其他产品设计缺陷{tuple_delimiter}电流检测电路设计缺陷，更换R133{tuple_delimiter}维修措施{tuple_delimiter}原因子类 “其他产品设计缺陷” 对应的维修措施为 “电流检测电路设计缺陷，更换 R133”。){record_delimiter}
(relationship{tuple_delimiter}其他产品设计缺陷{tuple_delimiter}R133{tuple_delimiter}维修部件{tuple_delimiter}原因子类 “其他产品设计缺陷” 相关的维修更换部件为元件 “R133”。){record_delimiter}
(relationship{tuple_delimiter}其他产品设计缺陷{tuple_delimiter}DB0009{tuple_delimiter}不良单板{tuple_delimiter}原因子类 “其他产品设计缺陷” 对应的不良单板编码为 “DB0009”。){record_delimiter}
(relationship{tuple_delimiter}M0007{tuple_delimiter}T2{tuple_delimiter}生产流程{tuple_delimiter}产品编码 “M0007” 在生产过程中经过工序 “T2” 的时候出现故障。){record_delimiter}
(relationship{tuple_delimiter}M0007{tuple_delimiter}2025/7/4 11:03:24{tuple_delimiter}维修时间{tuple_delimiter}产品编码 “M0007” 于 “2025 年 7 月 4 日 11 时 03 分 24 秒” 完成维修。){record_delimiter}
(relationship{tuple_delimiter}T2{tuple_delimiter}SMT-AOI和测试段不良使用-勿删{tuple_delimiter}工序故障模式{tuple_delimiter}生产工序 “T2” 的典型故障现象有 “SMT-AOI和测试段不良使用-勿删”。){record_delimiter}
(relationship{tuple_delimiter}电流检测电路设计缺陷，更换R133{tuple_delimiter}R133{tuple_delimiter}更换部件{tuple_delimiter}维修措施 “电流检测电路设计缺陷，更换 R133” 涉及更换元件 “R133”。){record_delimiter}
(relationship{tuple_delimiter}产品设计缺陷{tuple_delimiter}其他产品设计缺陷{tuple_delimiter}类别层级{tuple_delimiter}缺陷分类 “产品设计缺陷” 下的子类包含 “其他产品设计缺陷”。){record_delimiter}
(relationship{tuple_delimiter}研发设计类问题{tuple_delimiter}产品设计缺陷{tuple_delimiter}定位与分类{tuple_delimiter}问题定位 “研发设计类问题” 通常归因为 “产品设计缺陷”。){record_delimiter}
{completion_delimiter}

""",
    """[示例 2]

---输入---
实体类型: [organization,person,location,event,technology,equiment,product,Document,category]
文本:
```
在北京举行的人工智能大会上，腾讯公司的首席技术官张伟发布了最新的大语言模型"腾讯智言"，该模型在自然语言处理方面取得了重大突破。
```

---输出---
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

PROMPTS["summarize_entity_descriptions"] = """---角色---
您是负责数据整理和综合的知识图谱专家。

---任务---
您的任务是将给定实体或关系的描述列表综合为一个单一、全面且连贯的摘要。

---指令---
1. **全面性：** 摘要必须整合所有提供描述中的关键信息。不要遗漏重要事实。
2. **上下文：** 摘要必须明确提及实体或关系的名称以提供完整上下文。
3. **冲突：** 如果描述存在冲突或不一致，请确定它们是否来自共享相同名称的多个不同实体或关系。如果是，请分别总结每个实体或关系，然后合并所有摘要。
4. **风格：** 输出必须以客观的第三人称视角编写。
5. **长度：** 在保持深度和完整性的同时，确保摘要长度不超过{summary_length}个标记。
6. **语言：** 整个输出必须使用{language}编写。

---数据---
{description_type}名称：{description_name}
描述列表：
{description_list}

---输出---
"""

PROMPTS["fail_response"] = (
    "抱歉，我无法回答该问题。[no-context]"
)

PROMPTS["rag_response"] = """---角色---

你是一名有帮助的助手，负责回答关于知识图谱和以下以 JSON 格式提供的文档块的用户问题。

---目标---

基于知识库生成简洁的回复，并遵循回复规则，结合当前问题以及（如有）对话历史。总结知识库中提供的所有信息，并结合与知识库相关的一般知识。不要包含知识库中没有提供的信息。

---对话历史---
{history}

---知识图谱和文档块---
{context_data}

---回复指南---
**1. 内容与遵循：**
- 严格遵循知识库提供的上下文。不要虚构、推测或包含来源数据中不存在的信息。
- 如果在提供的上下文中找不到答案，应明确说明没有足够的信息回答。
- 确保回复与对话历史保持连贯性。

**2. 格式与语言：**
- 使用 Markdown 格式化回复，并添加适当的章节标题。
- 各个部分的标题要加粗，除了主要内容部分，其余不重要内容字体颜色可以淡一些。
- 可以添加一些生动的Unicode表情符号，让回复更加生动。
- 回复语言必须与用户提问的语言保持一致。
- 目标格式和长度：{response_type}

**3. 引用 / 参考文献：**
- 在回复末尾添加 "参考文献" 部分，每条引用必须清楚标明其来源（KG 或 DC）。
- 引用的最大数量为 5 条，包括 KG 和 DC。
- 按以下格式添加引用：
  - 知识图谱实体：`[KG] <entity_name>`
  - 知识图谱关系：`[KG] <entity1_name> - <entity2_name>`
  - 文档块：`[DC] <file_path_or_document_name>`

---用户上下文---
- 用户附加提示：{user_prompt}

---回复---
"""

PROMPTS["keywords_extraction"] = """---角色---
你是一名擅长分析用户查询的关键词提取专家，专注于为检索增强生成（RAG）系统提取高层和低层关键词。你的目标是识别用户查询中的核心概念和具体细节，以便进行有效的文档检索。

---目标---
给定一个用户查询，你的任务是提取两类不同的关键词：
1. **high_level_keywords**：体现整体概念或主题，捕捉用户的核心意图、主题领域或问题类型。
2. **low_level_keywords**：具体实体或细节，包括特定名词、专有名词、技术术语、产品名称或具体项目。

---说明与约束---
1. **输出格式**：你的输出必须是一个有效的 JSON 对象，且不能包含任何解释性文字、Markdown 代码块（例如 ```json）、或其他多余内容。输出将直接被 JSON 解析器解析。
2. **真实性来源**：所有关键词必须明确来源于用户查询，且高层和低层关键词类别都必须有内容。
3. **简洁且有意义**：关键词应为简短的词语或有意义的短语，优先提取代表单一概念的多词短语。例如，从“Apple Inc. 最新财报”中，你应提取“最新财报”和“Apple Inc.”，而不是“最新”、“财报”、“Apple”。
4. **处理边缘情况**：对于过于简单、模糊或无意义的查询（如 "hello"、"ok"、"asdfghjkl"），必须返回一个两个类别都为空列表的 JSON 对象。

---示例---
{examples}

---真实数据---
用户查询：{query}

---输出---
输出："""

PROMPTS["keywords_extraction_examples"] = [
    """示例 1:

查询: "国际贸易如何影响全球经济稳定？"

输出:
{
  "high_level_keywords": ["国际贸易", "全球经济稳定", "经济影响"],
  "low_level_keywords": ["贸易协定", "关税", "货币兑换", "进口", "出口"]
}

""",
    """示例 2:

查询: "森林砍伐对生物多样性的环境影响是什么？"

输出:
{
  "high_level_keywords": ["环境影响", "森林砍伐", "生物多样性丧失"],
  "low_level_keywords": ["物种灭绝", "栖息地破坏", "碳排放", "雨林", "生态系统"]
}

""",
    """示例 3:

查询: "教育在减少贫困中起到什么作用？"

输出:
{
  "high_level_keywords": ["教育", "减贫", "社会经济发展"],
  "low_level_keywords": ["入学机会", "识字率", "职业培训", "收入不平等"]
}

""",
]

PROMPTS["naive_rag_response"] = """---角色---

你是一名有帮助的助手，负责回答关于以下以 JSON 格式提供的文档块的用户问题。

---目标---

基于文档块生成简洁的回复，并遵循回复规则，结合对话历史和当前问题。总结所提供的文档块中的全部信息，并结合与文档块相关的一般知识。不要包含文档块中未提供的信息。

---对话历史---
{history}

---文档块（DC）---
{content_data}

---回复指南---
**1. 内容与遵循：**
- 严格遵循知识库提供的上下文。不要虚构、推测或包含来源数据中不存在的信息。
- 如果在提供的上下文中找不到答案，应明确说明没有足够的信息回答。
- 确保回复与对话历史保持连贯性。

**2. 格式与语言：**
- 使用 Markdown 格式化回复，并添加适当的章节标题。
- 各个部分的标题要加粗，除了主要内容部分，其余不重要内容字体颜色可以淡一些。
- 可以添加一些生动的Unicode表情符号，让回复更加生动。
- 回复语言必须与用户提问的语言一致。
- 目标格式和长度：{response_type}

**3. 引用 / 参考文献：**
- 在回复末尾添加“参考文献”部分，引用最多 10 个最相关的来源。
- 按以下格式添加引用：`[DC] <file_path_or_document_name>`

---用户上下文---
- 用户附加提示：{user_prompt}

---回复---
输出："""