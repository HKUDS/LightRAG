import asyncio
import os
import sys
import nest_asyncio
from dotenv import load_dotenv

# 1. 基礎設定
nest_asyncio.apply()
load_dotenv()

try:
    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc
    from lightrag.llm.openai import openai_embed
    from openai import AsyncAzureOpenAI
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)

# 環境變數
sf_api_key = os.getenv("EMBEDDING_BINDING_API_KEY")
sf_base_url = os.getenv("EMBEDDING_BINDING_HOST")
sf_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
sf_dim = int(os.getenv("EMBEDDING_DIM", 1024))
azure_api_key = os.getenv("LLM_BINDING_API_KEY")
azure_endpoint = os.getenv("LLM_BINDING_HOST")
azure_version = os.getenv("AZURE_OPENAI_API_VERSION")
azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
WORKING_DIR = "./data/rag_storage"

# 模型函數
def create_qa_prompt_from_template(prompt, system_prompt=None, history_messages=[]):
    messages = []
    if system_prompt: messages.append({"role": "system", "content": system_prompt})
    if history_messages: messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    return messages

async def azure_llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    client = AsyncAzureOpenAI(api_key=azure_api_key, azure_endpoint=azure_endpoint, api_version=azure_version)
    kwargs.pop("hashing_kv", None)
    kwargs.pop("keyword_extraction", None)
    messages = create_qa_prompt_from_template(prompt, system_prompt=system_prompt, history_messages=history_messages)
    response = await client.chat.completions.create(model=azure_deployment, messages=messages, **kwargs)
    if response.choices: return response.choices[0].message.content
    return ""

async def siliconflow_embed_func(texts):
    truncated_texts = [t[:4000] for t in texts] 
    return await openai_embed(truncated_texts, model=sf_model, api_key=sf_api_key, base_url=sf_base_url)

# ==========================================
#         核心修復邏輯
# ==========================================

async def resurrect_and_delete(rag, entity_name):
    """
    先創建(復活)實體，讓 Graph 知道它的存在，然後再徹底刪除。
    這樣可以確保 Vector DB 中的殘留數據被正確清除。
    """
    print(f"   🔄 正在處理: '{entity_name}'")
    
    # 步驟 1: 復活 (Resurrect)
    # 我們隨便給個描述即可，目的是為了產生 ID 並寫入 Graph
    try:
        await rag.acreate_entity(entity_name, {
            "description": "Temporary entity for deletion fix",
            "entity_type": "unknown"
        })
        # print(f"      -> 已復活 (Re-created)")
    except Exception as e:
        print(f"      -> 復活時遇到小問題 (可忽略): {e}")

    # 步驟 2: 處決 (Delete)
    # 現在 Graph 裡有這個人了，delete 函數就會乖乖去 Vector DB 刪除對應的資料
    try:
        await rag.adelete_by_entity(entity_name)
        print(f"      ✅ 已徹底刪除 (Deleted via API)")
    except Exception as e:
        print(f"      ❌ 刪除失敗: {e}")

async def main():
    print(f"🚀 初始化 LightRAG...")
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=azure_llm_func,
        embedding_func=EmbeddingFunc(embedding_dim=sf_dim, max_token_size=8192, func=siliconflow_embed_func)
    )
    await rag.initialize_storages()

    print("\n🛠️ [修復模式] 開始清理殭屍數據...")
    print("邏輯: 重新建立實體 -> 觸發完整刪除流程")

    # 1. 清理實體清單
    zombie_list = [
        "Iron Man", "Jarvis", "Stark Industries", # Test 1
        "Alice", "Bob", "量子計算",               # Test 2
        "Google", "Gmail", "Google Mail" # Test 3
    ]

    for name in zombie_list:
        await resurrect_and_delete(rag, name)

    # 2. 清理 Document ID
    print("\n📄 [修復文檔] 清理 manual-1...")
    try:
        # 這裡我們嘗試直接刪除，如果失敗則需要類似的邏輯(先 insert 再 delete)，但通常 doc id 比較少出錯
        await rag.adelete_by_doc_id("manual-1")
        print("   ✅ 文檔 'manual-1' 清理指令已發送")
    except Exception as e:
        print(f"   ⚠️ 文檔清理訊息: {e}")

    print("\n✨ 所有操作已完成！請檢查你的 JSON 檔案，現在應該乾淨了。")

if __name__ == "__main__":
    asyncio.run(main())