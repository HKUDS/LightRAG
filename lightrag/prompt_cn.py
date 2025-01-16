GRAPH_FIELD_SEP = "<SEP>"

PROMPTS = {}

PROMPTS["DEFAULT_LANGUAGE"] = "中文"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

PROMPTS["DEFAULT_ENTITY_TYPES"] = ["标准元数据", "毒素", "食品", "检测", "限量", "特殊食品", "食品加工", "食品来源", "食品状态",
                                   "食品成分", "容器/包装", "处理方法", "生产工艺", "法规属性", "计量单位", "文档结构",
                                   "食品饮料子", "地域/来源", "营养特征", "加工技术"]

PROMPTS["entity_extraction"] = """-目的-
给定可能与此活动相关的Markdown或Html文本和一个实体类型列表，从文本中识别出这些类型的所有实体以及所识别实体之间的所有关系，不要遗漏。如果实体内容中存在、号，将、前后内容拆分成不同的实体。
如果识别出的实体不在给出的实体类型列表中，可定义新的实体类型。
提取的内容中html上下标记保留，图片标记保留，图片标记中链接地址保留原文，不要改变英文的大小写。
不要编造待处理数据中不存在的链接。
不要编造待处理数据中不存在的latex公式。
按要求的格式输出，不要任何说明与解释。
使用{language}作为输出语言。
-步骤-
1.识别所有实体。对于每个已识别的实体，提取以下信息：
-entity_name：实体的名称，使用与输入文本相同的语言。如果是英文，则保留原来的格式。
-entity_type：实体类型。保留实体类型名称，不需要举例说明内容。
-entity_description：对实体的属性和活动进行的全面描述。
将每个实体格式化为 ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>){record_delimiter}

2.从步骤1中识别的实体中，识别彼此*明显相关*的所有对（source_entity、target_entity）。如果有步骤1中识别的实体无*明显相关*的实体，那么建立与"{extend_entity_sn}"实体的“标准与内容"关系。
对于每对相关实体，提取以下信息：
-source_entity：源实体的名称，如步骤1中所标识的。
-target_entity：目标实体的名称，如步骤1中所标识的。
-relationship_description：解释为什么你认为源实体和目标实体是相互关联的。
-relationship_strength：一个整数分数，表示源实体和目标实体之间关系的强度，0-9之间，9最高，0最低。
-relationship_keywords：一个或多个高级关键词，总结关系的总体性质，侧重于概念或主题而非具体细节，关键字之间用、分隔。
将每个关系格式化为 ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>){record_delimiter}

3.识别能够概括整个文本主要概念、主题或话题的高级关键词。这些关键词应捕捉文档中呈现的总体思想。关键词中的html上下标记保留。
将内容级关键字格式化为 ("content_keywords"{tuple_delimiter}<high_level_keywords>)

4.以{language}返回输出，作为步骤1和2中标识的所有实体和关系的单个列表。

5.完成后，输出{completion_delimiter}

-实体类型列表-
entity_types: [{entity_types}]
#############################
-范例-
######################
{examples}

#############################
-待处理的数据-
######################
文本内容: {input_text}
######################
输出:
"""

PROMPTS["entity_extraction_examples"] = [
    """范例1:
文本内容: 
中华人民共和国国家标准
GB2761-2017
食品安全国家标准
食品中真菌毒素限量
2017-03-17发布
2017-09-17实施
中华人民共和国国家卫生和计划生育委员会 发布
# 前言
GB2761-2017代替GB2761-2011《食品安全国家标准食品中真菌毒素限量》。
GB2761-2017与GB2761-2011相比,主要变化如下:
--增加了葡萄酒和咖啡中赭曲霉毒素A限量要求;
--更新了检验方法标准号;
食品安全国家标准食品中真菌毒素限量
################
输出:
("entity"{tuple_delimiter}GB2761-2017{tuple_delimiter}标准编号{tuple_delimiter}食品安全国家标准食品中真菌毒素限量的标准编号){record_delimiter}
("entity"{tuple_delimiter}食品安全国家标准{tuple_delimiter}标准名称{tuple_delimiter}规范食品中真菌毒素限量的国家标准){record_delimiter}
("entity"{tuple_delimiter}食品中真菌毒素限量{tuple_delimiter}标准类别{tuple_delimiter}食品安全国家标准的具体类别){record_delimiter}
("entity"{tuple_delimiter}2017-03-17{tuple_delimiter}发布日期{tuple_delimiter}标准发布的具体日期){record_delimiter}
("entity"{tuple_delimiter}2017-09-17{tuple_delimiter}实施日期{tuple_delimiter}标准正式实施的日期){record_delimiter}
("entity"{tuple_delimiter}中华人民共和国国家卫生和计划生育委员会{tuple_delimiter}发布机构{tuple_delimiter}负责发布该标准的政府机构之一){record_delimiter}
("entity"{tuple_delimiter}GB2761-2011{tuple_delimiter}代替标准{tuple_delimiter}被GB2761-2017替代的旧标准){record_delimiter}
("entity"{tuple_delimiter}葡萄酒{tuple_delimiter}食品名称{tuple_delimiter}新增赭曲霉毒素A限量要求的食品){record_delimiter}
("entity"{tuple_delimiter}咖啡{tuple_delimiter}食品名称{tuple_delimiter}新增赭曲霉毒素A限量要求的食品){record_delimiter}
("entity"{tuple_delimiter}赭曲霉毒素A{tuple_delimiter}元素{tuple_delimiter}在葡萄酒和咖啡中新增限量要求的真菌毒素){record_delimiter}
("entity"{tuple_delimiter}检验方法标准号{tuple_delimiter}检验方法标准号{tuple_delimiter}更新的检验方法相关标准号){record_delimiter}
("relationship"{tuple_delimiter}GB2761-2017{tuple_delimiter}食品安全国家标准{tuple_delimiter}国家标准文件描述了食品中真菌毒素限量的具体标准{tuple_delimiter}标准、规范、内容{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}GB2761-2017{tuple_delimiter}2017-03-17{tuple_delimiter}标准的发布日期{tuple_delimiter}发布、时间{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}GB2761-2017{tuple_delimiter}2017-09-17{tuple_delimiter}标准的实施日期{tuple_delimiter}实施、时间{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}中华人民共和国国家卫生和计划生育委员会{tuple_delimiter}GB2761-2017{tuple_delimiter}作为标准发布机构之一{tuple_delimiter}发布者、标准{tuple_delimiter}6){record_delimiter}
("relationship"{tuple_delimiter}GB2761-2017{tuple_delimiter}GB2761-2011{tuple_delimiter}新标准替代旧标准{tuple_delimiter}代替、更新{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}葡萄酒{tuple_delimiter}赭曲霉毒素A{tuple_delimiter}新增限量要求的食品类型和元素{tuple_delimiter}限量、增加{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}咖啡{tuple_delimiter}赭曲霉毒素A{tuple_delimiter}新增限量要求的食品类型和元素{tuple_delimiter}限量、增加{tuple_delimiter}7){record_delimiter}
("content_keywords"{tuple_delimiter}食品安全、真菌毒素、国家标准、限量要求、营养食品、检验方法、发布日期、实施日期、发布机构){completion_delimiter}
######################""",
    """范例2:
文本内容: 
GB2761-2017 食品中真菌毒素限量
# 1 范围
GB2761-2017规定了食品中黄曲霉毒素B<sub>1</sub>、黄曲霉毒素M<sub>1</sub>、脱氧雪腐镰刀菌烯醇、展青霉素、赭曲霉毒素A及玉米赤霉烯酮的限量指标。
# 2 术语和定义
## 2.1 真菌毒素
真菌在生长繁殖过程中产生的次生有毒代谢产物。
## 2.2 可食用部分
食品原料经过机械手段(如谷物碾磨、水果剥皮、坚果去壳、肉去骨、鱼去刺、贝去壳等)去除非食用部分后,所得到的用于食用的部分。
注1:非食用部分的去除不可采用任何非机械手段(如粗制植物油精炼过程)。
注2:用相同的食品原料生产不同产品时,可食用部分的量依生产工艺不同而异。如用麦类加工麦片和全麦粉时,可食用部分按100%计算;加工小麦粉时,可食用部分按出粉率折算。
## 2.3 限量
真菌毒素在食品原料和(或)食品成品可食用部分中允许的最大含量水平。
# 3 应用原则
## 3.1 无论是否制定真菌毒素限量,食品生产和加工者均应采取控制措施,使食品中真菌毒素的含量达到最低水平。
## 3.2 GB2761-2017列出了可能对公众健康构成较大风险的真菌毒素,制定限量值的食品是对消费者膳食暴露量产生较大影响的食品。
## 3.3 食品类别(名称)说明(附录A)用于界定真菌毒素限量的适用范围,仅适用于GB2761-2017。当某种真菌毒素限量应用于某一食品类别(名称)时,则该食品类别(名称)内的所有类别食品均适用,有特别规定的除外。
## 3.4 食品中真菌毒素限量以食品通常的可食用部分计算,有特别规定的除外。
#############
输出:
("entity"{tuple_delimiter}GB2761-2017{tuple_delimiter}标准号{tuple_delimiter}食品中真菌毒素限量的国家标准编号，2017年版本){record_delimiter}
("entity"{tuple_delimiter}食品中真菌毒素限量{tuple_delimiter}标准名称{tuple_delimiter}规定食品中真菌毒素最大允许含量的国家标准){record_delimiter}
("entity"{tuple_delimiter}黄曲霉毒素B<sub>1</sub>{tuple_delimiter}元素{tuple_delimiter}一种由黄曲霉菌产生的有毒代谢产物){record_delimiter}
("entity"{tuple_delimiter}黄曲霉毒素M<sub>1</sub>{tuple_delimiter}元素{tuple_delimiter}黄曲霉毒素的一种变体，常见于奶制品){record_delimiter}
("entity"{tuple_delimiter}脱氧雪腐镰刀菌烯醇{tuple_delimiter}化合物{tuple_delimiter}一种由镰刀菌产生的真菌毒素){record_delimiter}
("entity"{tuple_delimiter}展青霉素{tuple_delimiter}化合物{tuple_delimiter}一种由青霉菌产生的真菌毒素){record_delimiter}
("entity"{tuple_delimiter}赭曲霉毒素A{tuple_delimiter}化合物{tuple_delimiter}一种由曲霉菌产生的霉菌毒素){record_delimiter}
("entity"{tuple_delimiter}玉米赤霉烯酮{tuple_delimiter}化合物{tuple_delimiter}一种由赤霉菌在玉米上产生的真菌毒素){record_delimiter}
("entity"{tuple_delimiter}真菌毒素{tuple_delimiter}术语{tuple_delimiter}真菌在生长繁殖过程中产生的次生有毒代谢产物){record_delimiter}
("entity"{tuple_delimiter}可食用部分{tuple_delimiter}术语{tuple_delimiter}食品原料经过机械手段去除非食用部分后，用于食用的部分){record_delimiter}
("entity"{tuple_delimiter}限量{tuple_delimiter}术语{tuple_delimiter}真菌毒素在食品原料和(或)食品成品可食用部分中允许的最大含量水平){record_delimiter}
("entity"{tuple_delimiter}3.1{tuple_delimiter}应用原则{tuple_delimiter}食品生产和加工者应采取控制措施，使食品中真菌毒素含量达到最低水平){record_delimiter}
("entity"{tuple_delimiter}3.2{tuple_delimiter}应用原则{tuple_delimiter}标准列出了可能对公众健康构成较大风险的真菌毒素，并制定限量值的食品){record_delimiter}
("entity"{tuple_delimiter}3.3{tuple_delimiter}应用原则{tuple_delimiter}食品类别说明用于界定真菌毒素限量的适用范围，适用于标准中的所有食品类别){record_delimiter}
("entity"{tuple_delimiter}3.4{tuple_delimiter}应用原则{tuple_delimiter}食品中真菌毒素限量以食品通常的可食用部分计算，特别规定除外){record_delimiter}
("relationship"{tuple_delimiter}GB2761-2017{tuple_delimiter}黄曲霉毒素B<sub>1</sub>{tuple_delimiter}标准规定了该毒素在食品中的最大允许限量{tuple_delimiter}规定{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}GB2761-2017{tuple_delimiter}黄曲霉毒素M<sub>1</sub>{tuple_delimiter}标准规定了该毒素在食品中的最大允许限量{tuple_delimiter}规定{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}GB2761-2017{tuple_delimiter}脱氧雪腐镰刀菌烯醇{tuple_delimiter}标准规定了该毒素在食品中的最大允许限量{tuple_delimiter}规定{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}GB2761-2017{tuple_delimiter}展青霉素{tuple_delimiter}标准规定了该毒素在食品中的最大允许限量{tuple_delimiter}规定{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}GB2761-2017{tuple_delimiter}赭曲霉毒素A{tuple_delimiter}标准规定了该毒素在食品中的最大允许限量{tuple_delimiter}规定{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}GB2761-2017{tuple_delimiter}玉米赤霉烯酮{tuple_delimiter}标准规定了该毒素在食品中的最大允许限量{tuple_delimiter}规定{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}GB2761-2017{tuple_delimiter}可食用部分{tuple_delimiter}标准规定了食品原料经过机械手段(如谷物碾磨、水果剥皮、坚果去壳、肉去骨、鱼去刺、贝去壳等)去除非食用部分后,所得到的用于食用的部分{tuple_delimiter}规定{tuple_delimiter}6){record_delimiter}
("relationship"{tuple_delimiter}GB2761-2017{tuple_delimiter}真菌毒素{tuple_delimiter}标准规定了真菌毒素在食品原料和(或)食品成品可食用部分中的最大允许含量。{tuple_delimiter}含量限制{tuple_delimiter}4){record_delimiter}
("relationship"{tuple_delimiter}GB2761-2017{tuple_delimiter}应用原则{tuple_delimiter}标准规定了应用原则{tuple_delimiter}适用{tuple_delimiter}5){record_delimiter}
("content_keywords"{tuple_delimiter}真菌毒素、食品安全、限量标准、毒素类型、可食用部分、应用原则、控制措施){completion_delimiter}
######################""",
    """范例3:
文本内容: 
GB2761-2017 食品中真菌毒素限量
# 4 指标要求
## 4.1 黄曲霉毒素B<sub>1</sub>
### 4.1.1 食品中黄曲霉毒素B<sub>1</sub>限量指标见表1.。
表1. 食品中黄曲霉毒素B1限量指标
<table><tr><td colspan="2">食品类别(名称)</td><td rowspan="2">限量μg/kg</td></tr><tr><td >一级类别</td><td >二级类别</td></tr><tr><td >乳及乳制品</td><td >/</td><td >0.5</td></tr><tr><td rowspan="7">特殊膳食用食品</td><td >婴幼儿配方食品--婴儿配方食品<sup>b</sup></td><td >0.5(以粉状产品计)</td></tr><tr><td >婴幼儿配方食品--较大婴儿和幼儿配方食品<sup>b</sup></td><td >0.5(以粉状产品计)</td></tr><tr><td >婴幼儿配方食品--特殊医学用途婴儿配方食品</td><td >0.5(以粉状产品计)</td></tr><tr><td >特殊医学用途配方食品<sup>b</sup>(特殊医学用途婴儿配方食品涉及的品种除外)</td><td >0.5(以固态产品计)</td></tr><tr><td >辅食营养补充品<sup>c</sup></td><td >0.5</td></tr><tr><td >运动营养食品<sup>b</sup></td><td >0.5</td></tr><tr><td >孕妇及乳母营养补充食品<sup>c</sup></td><td >0.5</td></tr><tr><td colspan="3"><sup>a</sup>乳粉按生乳折算。<br><sup>b</sup>以乳类及乳蛋白制品为主要原料的产品。<br><sup>c</sup>只限于含乳类的产品。</td></tr></table>
### 4.1.2 检验方法:按GB5009.22规定的方法测定。
#############
输出:
("entity"{tuple_delimiter}GB2761-2017{tuple_delimiter}标准号{tuple_delimiter}食品中真菌毒素限量的国家标准编号，2017年版本){record_delimiter}
("entity"{tuple_delimiter}食品中真菌毒素限量{tuple_delimiter}标准名称{tuple_delimiter}规定食品中真菌毒素最大允许含量的国家标准){record_delimiter}
("entity"{tuple_delimiter}黄曲霉毒素B<sub>1</sub>{tuple_delimiter}指标{tuple_delimiter}食品中黄曲霉毒素B<sub>1</sub>的最大允许含量水平){record_delimiter}
("entity"{tuple_delimiter}食品类别(名称){tuple_delimiter}范围{tuple_delimiter}列出不同食品类别的黄曲霉毒素B<sub>1</sub>限量指标){record_delimiter}
("entity"{tuple_delimiter}乳及乳制品{tuple_delimiter}食品类别{tuple_delimiter}包含乳制品在内的食品类别){record_delimiter}
("entity"{tuple_delimiter}特殊膳食用食品{tuple_delimiter}食品类别{tuple_delimiter}专为特定人群设计的膳食食品){record_delimiter}
("entity"{tuple_delimiter}婴幼儿配方食品--婴儿配方食品{tuple_delimiter}食品类别{tuple_delimiter}为婴儿设计的配方食品){record_delimiter}
("entity"{tuple_delimiter}婴幼儿配方食品--较大婴儿和幼儿配方食品{tuple_delimiter}食品类别{tuple_delimiter}为较大婴儿和幼儿设计的配方食品){record_delimiter}
("entity"{tuple_delimiter}婴幼儿配方食品--特殊医学用途婴儿配方食品{tuple_delimiter}食品类别{tuple_delimiter}特殊医学用途婴儿配方食品){record_delimiter}
("entity"{tuple_delimiter}特殊医学用途配方食品{tuple_delimiter}食品类别{tuple_delimiter}为特定医疗需求人群设计的食品){record_delimiter}
("entity"{tuple_delimiter}辅食营养补充品{tuple_delimiter}食品类别{tuple_delimiter}用于补充婴幼儿营养的食品产品){record_delimiter}
("entity"{tuple_delimiter}运动营养食品{tuple_delimiter}食品类别{tuple_delimiter}专门为运动员和运动爱好者设计的营养食品){record_delimiter}
("entity"{tuple_delimiter}孕妇及乳母营养补充食品{tuple_delimiter}食品类别{tuple_delimiter}为孕妇和哺乳期妇女设计的营养补充食品){record_delimiter}
("entity"{tuple_delimiter}GB5009.22{tuple_delimiter}检验方法标准号{tuple_delimiter}描述检测食品中黄曲霉毒素B<sub>1</sub>的方法标准号){record_delimiter}
("entity"{tuple_delimiter}0.5{tuple_delimiter}数值{tuple_delimiter}黄曲霉毒素B1的限量指标值){record_delimiter}
("entity"{tuple_delimiter}μg/kg{tuple_delimiter}计量单位{tuple_delimiter}黄曲霉毒素B1限量的计量单位){record_delimiter}
("relationship"{tuple_delimiter}黄曲霉毒素B<sub>1</sub>{tuple_delimiter}乳及乳制品{tuple_delimiter}标准规定乳及乳制品中黄曲霉毒素B<sub>1</sub>的限量为0.5μg/kg{tuple_delimiter}规定{tuple_delimiter}6){record_delimiter}
("relationship"{tuple_delimiter}黄曲霉毒素B<sub>1</sub>{tuple_delimiter}特殊膳食用食品{tuple_delimiter}标准规定特殊膳食用食品中黄曲霉毒素B<sub>1</sub>的限量为0.5μg/kg{tuple_delimiter}规定{tuple_delimiter}6){record_delimiter}
("relationship"{tuple_delimiter}黄曲霉毒素B<sub>1</sub>{tuple_delimiter}婴幼儿配方食品--婴儿配方食品{tuple_delimiter}婴儿配方食品中的黄曲霉毒素B<sub>1</sub>限量为0.5μg/kg{tuple_delimiter}规定{tuple_delimiter}6){record_delimiter}
("relationship"{tuple_delimiter}黄曲霉毒素B<sub>1</sub>{tuple_delimiter}婴幼儿配方食品--较大婴儿和幼儿配方食品{tuple_delimiter}较大婴儿和幼儿配方食品中的黄曲霉毒素B<sub>1</sub>限量为0.5μg/kg{tuple_delimiter}规定{tuple_delimiter}6){record_delimiter}
("relationship"{tuple_delimiter}黄曲霉毒素B<sub>1</sub>{tuple_delimiter}婴幼儿配方食品--特殊医学用途婴儿配方食品{tuple_delimiter}特殊医学用途婴儿配方食品中的黄曲霉毒素B<sub>1</sub>限量为0.5μg/kg{tuple_delimiter}规定{tuple_delimiter}6){record_delimiter}
("relationship"{tuple_delimiter}黄曲霉毒素B<sub>1</sub>{tuple_delimiter}特殊医学用途配方食品{tuple_delimiter}特殊医学用途配方食品中的黄曲霉毒素B<sub>1</sub>限量为0.5μg/kg{tuple_delimiter}规定{tuple_delimiter}6){record_delimiter}
("relationship"{tuple_delimiter}黄曲霉毒素B<sub>1</sub>{tuple_delimiter}辅食营养补充品{tuple_delimiter}辅食营养补充品中的黄曲霉毒素B<sub>1</sub>限量为0.5μg/kg{tuple_delimiter}规定{tuple_delimiter}6){record_delimiter}
("relationship"{tuple_delimiter}黄曲霉毒素B<sub>1</sub>{tuple_delimiter}运动营养食品{tuple_delimiter}运动营养食品中的黄曲霉毒素B<sub>1</sub>限量为0.5μg/kg{tuple_delimiter}规定{tuple_delimiter}6){record_delimiter}
("relationship"{tuple_delimiter}黄曲霉毒素B<sub>1</sub>{tuple_delimiter}孕妇及乳母营养补充食品{tuple_delimiter}孕妇及乳母营养补充食品中的黄曲霉毒素B<sub>1</sub>限量为0.5μg/kg{tuple_delimiter}规定{tuple_delimiter}6){record_delimiter}
("relationship"{tuple_delimiter}黄曲霉毒素B<sub>1</sub>{tuple_delimiter}GB5009.22{tuple_delimiter}检验方法标准号GB5009.22用于测定黄曲霉毒素B<sub>1</sub>含量{tuple_delimiter}测定{tuple_delimiter}7){record_delimiter}
("content_keywords"{tuple_delimiter}黄曲霉毒素B<sub>1</sub>、食品类别、营养食品、限量标准、检验方法){completion_delimiter}
######################""",
]

PROMPTS[
    "summarize_entity_descriptions"
] = """你是一个负责生成综合摘要的助手，按要求处理提供的数据：给定一个或两个实体，以及一系列与这些实体或实体组相关的描述。请将所有这些描述合并成一个综合描述，确保包含从所有描述中收集到的信息。
如果提供的描述存在矛盾之处，请解决这些矛盾，并提供一个单一且连贯的摘要。
请确保以第三人称撰写，并包含实体名称，以便我们了解完整的上下文。
描述中html上下标记保留，图片标记保留，图片标记中链接地址保留原文，不要改变英文的大小写。
不要编造待处理数据中不存在的链接。
不要编造待处理数据中不存在的latex公式。
按要求的格式输出，不要任何说明与解释。
使用{language}作为输出语言。
#######
-待处理数据-
实体: {entity_name}
相关描述: {description_list}
#######
输出:
"""

PROMPTS[
    "entiti_continue_extraction"
] = """在上次提取中遗漏了许多实体。请使用相同的格式将它们添加在下面：
"""

PROMPTS[
    "entiti_if_loop_extraction"
] = """似乎仍有一些实体可能被遗漏了。如果仍有需要添加的实体，请回答“YES”；如果没有，请回答“NO”。回答不要包含“YES”、“NO”之外的任何内容。
"""

PROMPTS["fail_response"] = "非常抱歉，我目前训练完成的标准知识库中暂时没有包含您问题中的内容，您可以尝试问其它问题或者等待标准知识库进一步丰富。"

PROMPTS["rag_response"] = """---角色---
你是一个负责依据所提供数据来源回答问题的助手。请用与用户问题相同的语言回答。
---目的---
生成一个符合目标长度和格式的回复，以回答用户的问题。应根据重要性和关联程度的高低结合所提供数据来源中所有信息进行回复。
如果回复内容需要包含所提供数据来源中的链接，直接用原始链接地址，不要改变英文的大小写。
如果你不知道答案，明确表示不知道。不要编造任何内容。
不要直接提供latex公式，如果所提供数据来源中存在符合要求的图片链接，输出相关公式图片链接。
不要编造任何所提供数据来源中不存在的链接。
不要包含所提供数据来源中没有支持性证据的信息。
---目标回复长度和格式---
{response_type}
---数据来源---
{context_data}
---响应要求---
  -没有明确要求的话，不要添加任何解释和评论，直接输出符合要求的简洁回答。
  -有明确要求进行解释和评论的话，根据回复的长度和格式要求，在回复中添加适当的章节和评论。使用带有适当章节标题的markdown格式对回复进行排版。
  -如果要求以markdown标题形式回复，并且标题中存在前言或附录，将前言部分放在最前面，附录放在最后面。
"""

PROMPTS["keywords_extraction"] = """---角色---
你是一个负责识别用户查询中的高级和低级关键词的助手。
使用{language}作为输出语言。
---目的---
根据查询，列出高级和低级关键词。高级关键词关注总体概念或主题，而低级关键词关注具体实体、细节或具体术语。
---指令---
- 以JSON格式输出关键词。
- JSON应包含两个键：
  - "high_level_keywords" 高级关键词，用于表示总体概念或主题。
  - "low_level_keywords" 低级关键词，用于表示具体实体或细节。
#############################
-范例-
######################
{examples}

#############################
-待处理的数据-
######################
查询: {query}
######################
“输出”应该是人类文本，而不是unicode字符。保持与“查询”相同的语言。
输出:
"""

PROMPTS["keywords_extraction_examples"] = [
    """范例1:
查询: "国际贸易如何影响全球经济稳定？"
################
输出:
{{
  "high_level_keywords": ["国际贸易", "影响", "全球经济", "经济稳定"],
  "low_level_keywords": ["经济增长", "国内生产总值（GDP）", "就业率", "技术创新", "产业升级", "资源配置", "市场竞争力", "企业盈利能力", "市场规模"]
}}
######################""",
    """范例2:
查询: "森林砍伐对生物多样性的环境影响是什么？"
################
输出:
{{
  "high_level_keywords": ["森林砍伐", "生物多样性", "环境影响"],
  "low_level_keywords": ["栖息地丧失", "物种灭绝", "生态系统破坏", "土壤侵蚀", "气候变化", "碳排放", "水源保护", "植被覆盖"]
}}
######################""",
    """范例3:
查询: "教育在减少贫困中的作用是什么？"
################
输出:
{{
  "high_level_keywords": ["教育", "减少贫困", "作用"],
  "low_level_keywords": ["就业机会", "收入水平", "技能提升", "社会流动性", "健康意识", "儿童福利", "社区发展", "经济贡献"]
}}
#############################""",
]

PROMPTS["naive_rag_response"] = """---角色---
你是一个负责依据所提供数据来源回答问题的助手。请用与用户问题相同的语言回答。
---目的---
生成一个符合目标长度和格式的回复，以回答用户的问题，应根据重要性和关联程度的高低结合所提供数据来源中所有信息进行回复。
如果回复内容需要包含所提供数据来源中的链接，直接用原始链接地址，不要改变链接中英文的大小写。
如果你不知道答案，明确表示不知道，不要编造任何内容。
不要直接提供latex公式，如果所提供数据来源中存在符合要求的图片链接，输出相关公式图片链接。
不要编造任何所提供数据来源中不存在的链接。
不要包含所提供数据来源中没有支持性证据的信息。
---目标回复长度和格式---
{response_type}
---数据来源---
{content_data}
---响应要求---
  -没有明确要求的话，不要添加任何解释和评论，直接输出符合要求的简洁回答。
  -有明确要求进行解释和评论的话，根据回复的长度和格式要求，在回复中添加适当的章节和评论。使用带有适当章节标题的markdown格式对回复进行排版。
  -如果要求以markdown标题形式回复，并且标题中存在前言或附录，将前言部分放在最前面，附录放在最后面。
"""

PROMPTS[
    "similarity_check"
] = """请分析这两个问题之间的相似性：

问题1:{original_prompt}
问题2:{cached_prompt}

请评估以下两点，并直接提供0到1之间的相似性得分：
1.这两个问题在语义上是否相似；
2.“问题2”的答案是否可用于回答“问题1”。
相似性评分标准：
0:完全无关或答案不能重复使用，包括但不限于：
  -这些问题有不同的主题。
  -问题中提到的地点不同。
  -问题中提到的时间不同。
  -问题中提到的具体个人不同。
  -问题中提到的具体事件不同。
  -问题中的背景信息不同。
  -问题中的关键条件不同。
  -问题中提到的标准名称、标准编号、标准号不同。
1:完全相同，答案可以直接重复使用
0.5:部分相关，需要修改答案才能使用
仅返回0-1之间的数字，不包含任何其他内容。
"""

PROMPTS["mix_rag_response"] = """---角色---
你是一个负责根据所提供数据来源回答问题的助手。请用与用户问题相同的语言回答。
---目的---
生成一个符合目标长度和格式的回复，以回答用户的问题。应根据重要性和关联程度的高低结合所提供数据来源中所有信息进行回复。
如果回复内容需要使用所提供数据来源中的链接，直接用原始链接地址，不要改变链接中英文的大小写。
如果你不知道答案，明确表示不知道。不要编造任何东西。
不要直接提供latex公式，如果所提供数据来源中存在符合要求的图片链接，输出相关公式图片链接。
不要编造任何所提供数据来源中不存在的链接。
不要包含所提供数据来源中没有支持性证据的信息。
处理带有时间戳的信息时：
  1.每条信息（实体、关系和内容）都有一个“created_at”时间戳，表示我们何时获得了这些知识
  2.遇到冲突信息时，请同时考虑内容/实体/关系和时间戳
  3.不要自动选择最新的信息——根据上下文进行判断
  4.对于特定时间的查询，在考虑创建时间戳之前，优先考虑内容中的时间信息
---数据来源---
  1.知识图谱数据：
｛kg_context｝
  2.向量数据：
｛vector_context｝
---响应要求---
  -目标格式和长度：｛response_type｝
  -没有明确要求的话，不要添加任何解释和评论，直接输出符合要求的简洁回答。
  -有明确要求进行解释和评论的话，根据回复的长度和格式要求，在回复中添加适当的章节和评论。使用带有适当章节标题的markdown格式对回复进行排版。
  -每一段都应在相关章节标题下
  -每个部分都应该关注答案的一个要点或方面
  -使用清晰、描述性的章节标题来反映内容
  -如果要求以markdown标题形式回复，并且标题中存在前言或附录，将前言部分放在最前面，附录放在最后面。
"""
#   -在“参考文献”下的末尾列出最多5个最重要关联度最高的参考来源，明确指出每个来源是来自知识图谱还是向量数据
# 格式：[知识图谱/向量数据]源内容
# """