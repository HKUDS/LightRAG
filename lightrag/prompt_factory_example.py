from lightrag.prompt_factory import PromptFactory
from lightrag.prompt import PROMPTS

def main():
    # 初始化工厂类
    factory = PromptFactory(default_prompts=PROMPTS)
    
    # 示例1: 配置法律领域的提示
    legal_config = {
        "language": "中文",
        "entity_types": "法律案例, 法规, 当事人, 法院, 裁决",
        "tuple_delimiter": "<|>",
        "record_delimiter": "##",
        "completion_delimiter": "<|COMPLETE|>"
    }
    factory.register_domain("legal", legal_config)
    
    # 示例2: 配置医疗领域的提示
    medical_config = {
        "language": "中文",
        "entity_types": "疾病, 药物, 症状, 治疗方案, 医院, 医生",
        "tuple_delimiter": "<|>",
        "record_delimiter": "##",
        "completion_delimiter": "<|COMPLETE|>"
    }
    factory.register_domain("medical", medical_config)
    
    # 动态修改医疗领域的实体类型
    factory.customize_entity_types(
        ["疾病", "药物", "症状", "治疗方案", "医院", "医生", "患者", "副作用"],
        domain="medical"
    )
    
    # 示例3: 配置金融领域的提示
    finance_config = {
        "language": "英文",
        "entity_types": "company, stock, market, index, investment, trend",
        "tuple_delimiter": "<|>",
        "record_delimiter": "##",
        "completion_delimiter": "<|COMPLETE|>"
    }
    factory.register_domain("finance", finance_config)
    
    # 获取特定领域的提示
    legal_prompt = factory.get_prompt(
        "entity_extraction", 
        domain="legal",
        input_text="最高人民法院于2023年5月10日对王某与李某合同纠纷一案做出了终审判决，维持原判，驳回上诉。",
        examples="法律示例：合同纠纷、刑事案件等"
    )
    print("法律领域提示示例:")
    print(legal_prompt)
    print("\n" + "-" * 50 + "\n")
    
    # 获取医疗领域的提示
    medical_prompt = factory.get_prompt(
        "entity_extraction", 
        domain="medical",
        input_text="患者张先生，男，45岁，因持续高烧38.5℃、咳嗽、胸痛三天，来院就诊。经CT检查发现右肺炎症，给予阿奇霉素抗感染治疗。",
        examples="医疗示例：高烧、咳嗽、胸痛等典型症状及诊断"
    )
    print("医疗领域提示示例:")
    print(medical_prompt)
    print("\n" + "-" * 50 + "\n")
    
    # 自定义新的提示模板
    factory.add_prompt(
        "finance_analysis",
        """---Goal---
分析以下金融市场数据，识别主要趋势和投资机会。
使用{language}作为输出语言。

---Market Data---
{market_data}

---Output Format---
趋势分析:
1. 
2. 
3. 

投资建议:
1. 
2. 
3. 

风险提示:
1. 
2. 

当分析完成时，输出{completion_delimiter}
"""
    )
    
    # 使用自定义提示模板
    finance_prompt = factory.get_prompt(
        "finance_analysis",
        domain="finance",
        market_data="上证指数收于3,215.58点，下跌0.78%；恒生指数收于18,341.29点，上涨1.25%；美元兑人民币汇率为6.9821，上涨0.15%，",
        examples="金融示例：股票趋势、投资组合分析等"
    )
    print("自定义金融分析提示示例:")
    print(finance_prompt)
    print("\n" + "-" * 50 + "\n")
    
    # 导出领域配置为JSON
    medical_config_json = factory.export_domain_config_json("medical")
    print("医疗领域配置JSON:")
    print(medical_config_json)
    
    # 创建新领域并导入配置
    factory.import_domain_config_json("healthcare", medical_config_json)
    
    # 验证导入是否成功
    healthcare_config = factory.get_domain_config("healthcare")
    print("\n导入的医疗保健领域配置:")
    print(healthcare_config)


if __name__ == "__main__":
    main() 