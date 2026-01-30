import asyncio
import os
import sys
import time
import nest_asyncio
from dotenv import load_dotenv

# [重要] 解決 Event Loop 衝突
nest_asyncio.apply()

# 1. 讀取 .env
load_dotenv()

# --- 輔助函數 ---
def create_qa_prompt_from_template(prompt, system_prompt=None, history_messages=[]):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    return messages

# 2. 導入套件
try:
    from raganything import RAGAnything, RAGAnythingConfig
    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc
    from lightrag.llm.openai import openai_embed
    from openai import AsyncAzureOpenAI
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)

# --- 環境變數設定 ---
sf_api_key = os.getenv("EMBEDDING_BINDING_API_KEY")
sf_base_url = os.getenv("EMBEDDING_BINDING_HOST")
sf_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
sf_dim = int(os.getenv("EMBEDDING_DIM", 1024))

azure_api_key = os.getenv("LLM_BINDING_API_KEY")
azure_endpoint = os.getenv("LLM_BINDING_HOST")
azure_version = os.getenv("AZURE_OPENAI_API_VERSION")
azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# --- 設定路徑 ---
WORKING_DIR = "./data/rag_storage"

# --- 模型連接函數 ---
async def azure_llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    client = AsyncAzureOpenAI(api_key=azure_api_key, azure_endpoint=azure_endpoint, api_version=azure_version)
    # [修正] 過濾掉 LightRAG 內部傳入但 Azure API 不支援的參數
    kwargs.pop("hashing_kv", None)
    kwargs.pop("keyword_extraction", None)
    kwargs.pop("enable_cot", None)
    messages = create_qa_prompt_from_template(prompt, system_prompt=system_prompt, history_messages=history_messages)
    response = await client.chat.completions.create(model=azure_deployment, messages=messages, **kwargs)
    if response.choices: return response.choices[0].message.content
    return ""

async def siliconflow_embed_func(texts):
    # 這裡可以加一個 print 來監控 Embedding 是否正在運作
    # print(f"DEBUG: Embedding {len(texts)} texts...") 
    truncated_texts = [t[:4000] for t in texts] 
    return await openai_embed(truncated_texts, model=sf_model, api_key=sf_api_key, base_url=sf_base_url)

# --- 初始化系統 ---
async def initialize_system():
    print(f"🚀 初始化 LightRAG (用於功能測試)...")
    lightrag_instance = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=azure_llm_func,
        # [修正] 恢復使用原始的異步 embedding 函數
        embedding_func=EmbeddingFunc(embedding_dim=sf_dim, max_token_size=8192, func=siliconflow_embed_func)
    )
    await lightrag_instance.initialize_storages()
    
    rag = RAGAnything(
        lightrag=lightrag_instance,
        vision_model_func=None,
        config=RAGAnythingConfig(working_dir=WORKING_DIR)
    )
    return rag, lightrag_instance

# ==========================================
#         核心測試功能區域
# ==========================================

async def test_manual_crud(rag_core: LightRAG):
    print("\n🧪 [測試 1] 手動創建 (Create)")
    
    # 1. 創建 Iron Man
    print("   -> 正在嘗試創建實體 'Iron Man'...")
    try:
        await rag_core.acreate_entity("Iron Man", {
            "description": "Iron Man (Tony Stark) 是一位超級英雄。",
            "entity_type": "person"
        })
        print("      ✅ 創建成功")
    except ValueError as e:
        print(f"      ⚠️ 跳過 (可能已存在): {e}")

    # 2. 創建 Jarvis
    print("   -> 正在嘗試創建實體 'Jarvis'...")
    try:
        await rag_core.acreate_entity("Jarvis", {
            "description": "Jarvis 是 AI 助手。",
            "entity_type": "product"
        })
        print("      ✅ 創建成功")
    except ValueError as e:
        print(f"      ⚠️ 跳過 (可能已存在): {e}")

    # 3. [新增] 創建 Stark Industries
    print("   -> 正在嘗試創建實體 'Stark Industries' (Type: industry)...")
    try:
        await rag_core.acreate_entity("Stark Industries", {
            "description": "Stark Industries 是一間大型科技與軍工企業，由 Tony Stark 經營。",
            "entity_type": "industry"
        })
        print("      ✅ 創建成功 (New Type: industry)")
    except ValueError as e:
        print(f"      ⚠️ 跳過 (可能已存在): {e}")

    # 4. 建立關係
    print("   -> 建立關係: Iron Man <-> Jarvis")
    try:
        await rag_core.acreate_relation("Iron Man", "Jarvis", {
            "description": "Iron Man 使用 Jarvis。",
            "keywords": "使用",
            "weight": 2.0
        })
        print("      ✅ 關係建立成功")
    except Exception as e:
        print(f"      ⚠️ 警告: {e}")

    # 5. [新增] 建立關係
    print("   -> 建立關係: Iron Man <-> Stark Industries")
    try:
        await rag_core.acreate_relation("Iron Man", "Stark Industries", {
            "description": "Iron Man (Tony Stark) 擁有並經營 Stark Industries。",
            "keywords": "擁有 經營 CEO",
            "weight": 2.0
        })
        print("      ✅ 關係建立成功")
    except Exception as e:
        print(f"      ⚠️ 警告: {e}")
    
    print("✅ [測試 1] 完成！")

async def test_custom_kg_insert(rag_core: LightRAG):
    print("\n🧪 [測試 2] 插入自定義 Knowledge Graph")
    
    custom_kg = {
        "chunks": [{"content": "Manual Data Source", "source_id": "manual-1"}],
        "entities": [
            {"entity_name": "Alice", "entity_type": "person", "description": "Alice 是一位量子物理學家。", "source_id": "manual-1"},
            {"entity_name": "Bob", "entity_type": "person", "description": "Bob 是一位數學家。", "source_id": "manual-1"},
            {"entity_name": "量子計算", "entity_type": "technology", "description": "量子力學計算技術。", "source_id": "manual-1"}
        ],
        "relationships": [
            {"src_id": "Alice", "tgt_id": "Bob", "description": "研究夥伴。", "keywords": "合作", "weight": 1.0, "source_id": "manual-1"},
            {"src_id": "Alice", "tgt_id": "量子計算", "description": "研究領域。", "keywords": "研究", "weight": 1.0, "source_id": "manual-1"}
        ]
    }
    
    print("   -> 正在插入 JSON 數據... (這可能需要 30-60 秒，請耐心等待)")
    start_time = time.time()
    
    # [修正] 直接調用異步版本的 ainsert_custom_kg，避免同步/異步混合調用導致的死鎖或類型錯誤。
    await rag_core.ainsert_custom_kg(custom_kg)
    end_time = time.time()
    duration = end_time - start_time
    print(f"✅ [測試 2] 完成！耗時: {duration:.2f} 秒")

async def test_edit_updates(rag_core: LightRAG):
    print("\n🧪 [測試 3] 編輯與更新實體 (Edit & Rename)")
    
    # 0. 準備數據
    print("   -> [準備] 正在創建初始實體 Google 和 Gmail...")
    try:
        await rag_core.acreate_entity("Google", {"description": "一家公司", "entity_type": "company"})
        await rag_core.acreate_entity("Gmail", {"description": "一個產品", "entity_type": "product"})
        await rag_core.acreate_relation("Google", "Gmail", {"description": "Google 擁有 Gmail"})
    except ValueError:
        pass 

    # 1. 編輯 Google
    print("   -> 正在更新 'Google' 的描述...")
    updated_entity = await rag_core.aedit_entity("Google", {
        "description": "Google是Alphabet Inc.的子公司，成立于1998年。",
        "entity_type": "tech_company"
    })
    print(f"      ✅ Google 更新完成")

    # 2. 重命名 Gmail -> Google Mail
    print("   -> 正在將 'Gmail' 重命名為 'Google Mail'...")
    renamed_entity = await rag_core.aedit_entity("Gmail", {
        "entity_name": "Google Mail", 
        "description": "Google Mail（前身为Gmail）是一项电子邮件服务。"
    })
    print(f"      ✅ 重命名完成！")

    # 3. 編輯關係
    print("   -> 正在更新關係 (Google <-> Google Mail)...")
    await rag_core.aedit_relation("Google", "Google Mail", {
        "description": "Google创建并维护Google Mail服务。",
        "keywords": "创建 维护 电子邮件服务",
        "weight": 3.0
    })
    print("      ✅ 關係更新完成！")
    
    print("✅ [測試 3] 編輯功能測試完成！")

async def verify_query(rag_wrapper: RAGAnything):
    print("\n❓ [驗證] 查詢測試結果...")
    # 測試多個問題
    questions = [
        "Google Mail 是什麼？",
        "Alice 和 Bob 是什麼關係？",
        "Stark Industries 是什麼類型的機構？它和 Iron Man 有什麼關係？" 
    ]
    
    for q in questions:
        print(f"\n👉 問題: {q}")
        try:
            result = await rag_wrapper.query_with_multimodal(q, mode="hybrid")
            print(f"🤖 AI 回答: {result[:150]}...") 
        except Exception as e:
            print(f"查詢失敗: {e}")

async def main():
    rag_wrapper, rag_core = await initialize_system()
    
    await test_manual_crud(rag_core)
    await test_custom_kg_insert(rag_core)
    await test_edit_updates(rag_core)
    await verify_query(rag_wrapper)

if __name__ == "__main__":
    asyncio.run(main())