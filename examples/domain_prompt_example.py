import asyncio
import os
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from lightrag.llm.openai import openai_embed, openai_model_complete

# Configure working directory
WORKING_DIR = "./domain_example"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# Initialize embedding function
async def embedding_func(texts):
    return await openai_embed(texts, model="text-embedding-3-small")

# Initialize LLM function
async def llm_model_func(text, **kwargs):
    return await openai_model_complete(text, model="gpt-4o-mini", **kwargs)

# Get embedding dimension
async def get_embedding_dim():
    test_embedding = await embedding_func(["test"])
    return len(test_embedding[0])

async def initialize_rag():
    # Detect embedding dimension
    embedding_dimension = await get_embedding_dim()
    print(f"Detected embedding dimension: {embedding_dimension}")

    # Initialize LightRAG with domain
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dimension,
            max_token_size=8192,
            func=embedding_func,
        ),
        # 初始化时可以设置domain
        domain="medical",
    )
    
    # 注册医疗领域配置
    medical_config = {
        "language": "中文",
        "entity_types": ["疾病", "药物", "症状", "治疗方案", "医院", "医生", "患者", "副作用"],
        "tuple_delimiter": "<|>",
        "record_delimiter": "##",
        "completion_delimiter": "<|COMPLETE|>"
    }
    rag.register_domain("medical", medical_config)
    
    # 注册法律领域配置
    legal_config = {
        "language": "中文",
        "entity_types": ["法律案例", "法规", "当事人", "法院", "裁决", "法条", "罪名"],
        "tuple_delimiter": "<|>",
        "record_delimiter": "##",
        "completion_delimiter": "<|COMPLETE|>"
    }
    rag.register_domain("legal", legal_config)
    
    # 初始化存储
    await rag.initialize_storages()
    
    return rag

async def demo_medical_domain(rag):
    print("\n=== 医疗领域示例 ===")
    
    # 插入医疗文本
    medical_text = """
    患者张某，男性，45岁，因持续高烧38.5℃、咳嗽、胸痛三天，来院就诊。
    经CT检查发现右肺炎症，给予阿奇霉素抗感染治疗。三天后症状明显好转，
    体温恢复正常，咳嗽减轻。医生建议患者继续服药一周，并定期复查。
    """
    
    await rag.ainsert(medical_text)
    
    # 医疗领域查询
    query = "这位患者的主要症状是什么？治疗效果如何？"
    result = await rag.aquery(
        query,
        param=QueryParam(mode="mix", response_type="单段详细描述"),
    )
    
    print(f"查询: {query}")
    print(f"回答: {result}")

async def demo_legal_domain(rag):
    print("\n=== 法律领域示例 ===")
    
    # 切换到法律领域
    rag.set_domain("legal")
    
    # 插入法律文本
    legal_text = """
    2023年5月10日，最高人民法院对王某与李某合同纠纷一案做出了终审判决。
    本案中，王某与李某于2022年1月签订了房屋买卖合同，王某支付定金后，
    李某违约拒绝交付房屋。一审法院判决李某返还定金并赔偿损失，李某不服上诉。
    最高人民法院认为，原判决认定事实清楚，适用法律正确，驳回上诉，维持原判。
    """
    
    await rag.ainsert(legal_text)
    
    # 法律领域查询
    query = "这个案件的基本情况是什么？最终判决结果如何？"
    result = await rag.aquery(
        query,
        param=QueryParam(mode="mix", response_type="要点分析"),
    )
    
    print(f"查询: {query}")
    print(f"回答: {result}")

async def main():
    rag = await initialize_rag()
    
    # 演示医疗领域
    await demo_medical_domain(rag)
    
    # 演示法律领域
    await demo_legal_domain(rag)
    
    # 完成后关闭存储
    await rag.finalize_storages()

if __name__ == "__main__":
    asyncio.run(main()) 