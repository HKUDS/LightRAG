# 使用Domain和PromptFactory功能

LightRAG现在支持通过domain属性来配置不同领域的prompt，方便用户根据不同的应用场景自定义提示模板和参数。

## 基本概念

- **Domain（领域）**：表示LightRAG实例所处理的特定领域，如医疗、法律、金融等
- **PromptFactory（提示工厂）**：负责根据领域配置生成适合该领域的提示模板

## 快速开始

### 1. 初始化时指定领域

可以在初始化LightRAG实例时直接指定domain：

```python
rag = LightRAG(
    working_dir="./my_rag_cache",
    # 其他参数...
    domain="medical",  # 指定领域
)
```

### 2. 注册领域配置

需要为每个领域注册特定的配置：

```python
# 注册医疗领域配置
medical_config = {
    "language": "中文",
    "entity_types": ["疾病", "药物", "症状", "治疗方案", "医院", "医生"],
    "tuple_delimiter": "<|>",
    "record_delimiter": "##",
    "completion_delimiter": "<|COMPLETE|>"
}
rag.register_domain("medical", medical_config)
```

### 3. 切换领域

可以在运行时切换LightRAG实例的领域：

```python
# 切换到法律领域
rag.set_domain("legal")
```

## 领域配置参数

领域配置可以包含以下参数：

- `language`：生成提示的语言，例如"中文"、"English"等
- `entity_types`：实体类型列表，根据领域特点定义
- `tuple_delimiter`：元组分隔符
- `record_delimiter`：记录分隔符
- `completion_delimiter`：完成分隔符
- 其他自定义参数...

## 使用领域专属提示

您可以直接使用`get_prompt`方法获取特定领域的提示：

```python
# 获取实体提取提示
entity_prompt = rag.get_prompt(
    "entity_extraction", 
    input_text="待处理的文本",
    examples="示例文本"
)
```

## 通过addon_params设置领域

除了使用domain参数，也可以通过addon_params设置领域：

```python
rag = LightRAG(
    # 其他参数...
    addon_params={
        "language": "中文",
        "domain": "medical",
        "entity_types": ["疾病", "药物", "症状"]
    }
)
```

## 完整示例

```python
import asyncio
from lightrag import LightRAG, QueryParam

async def main():
    # 初始化LightRAG实例
    rag = LightRAG(
        working_dir="./domain_example",
        # 其他参数...
        domain="medical",
    )
    
    # 注册医疗领域配置
    medical_config = {
        "language": "中文",
        "entity_types": ["疾病", "药物", "症状", "治疗方案", "医院", "医生"]
    }
    rag.register_domain("medical", medical_config)
    
    # 注册法律领域配置
    legal_config = {
        "language": "中文",
        "entity_types": ["法律案例", "法规", "当事人", "法院", "裁决"]
    }
    rag.register_domain("legal", legal_config)
    
    # 初始化存储
    await rag.initialize_storages()
    
    # 医疗领域处理
    await rag.ainsert("患者因高烧38度来院就诊...")
    
    result = await rag.aquery(
        "患者的主要症状是什么？",
        param=QueryParam(mode="mix")
    )
    print(result)
    
    # 切换到法律领域
    rag.set_domain("legal")
    
    # 法律领域处理
    await rag.ainsert("最高人民法院对王某与李某合同纠纷一案做出了终审判决...")
    
    result = await rag.aquery(
        "这个案件的判决结果是什么？",
        param=QueryParam(mode="mix")
    )
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

有关更详细的示例，请参阅`examples/domain_prompt_example.py`。 