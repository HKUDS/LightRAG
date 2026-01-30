import os
import asyncio
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from lightrag.llm.ollama import ollama_embed
from lightrag.llm.openai import azure_openai_complete 
from lightrag.base import DocStatus  # 引入 DocStatus 枚舉
from functools import partial
from dotenv import load_dotenv

load_dotenv()

# === 設定 ===
WORKING_DIR = "./data/rag_storage"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBEDDING_MODEL = "bge-m3"
EMBEDDING_DIM = 1024

# === 🎯 定義你要刪除的文件關鍵字 ===
# 只要文件路徑 (file_path) 包含這些字，就會被刪除
TARGET_FILES_TO_DELETE = [
    "0001_2024",
    "0775_2024",
    "1038_2024",
    "1366_2024",
    "Global Market Insights",
    "HSBC_Report_Source_A",
    "SFC",
    "Strategic Investment Partners",
    "HSBC_Report_Source_B",
    # "SFC_Report_2023.pdf",   # 例子：完整文件名
    # "Draft_v1",              # 例子：文件名的一部分
    # "Old_Data_Folder"        # 例子：某個文件夾下的所有文件
]

async def find_ids_by_filename(rag, targets):
    """
    遍歷所有狀態的文檔，尋找匹配文件名的 ID
    """
    print("🔍 正在掃描數據庫中的文檔...")
    
    found_ids = set()
    
    # 我們需要檢查所有可能的狀態
    statuses_to_check = [
        DocStatus.PROCESSED, 
        DocStatus.FAILED, 
        DocStatus.PENDING, 
        DocStatus.PROCESSING
    ]
    
    total_scanned = 0
    
    for status in statuses_to_check:
        # 獲取該狀態下的所有文檔
        docs_dict = await rag.doc_status.get_docs_by_status(status)
        
        for doc_id, doc_obj in docs_dict.items():
            total_scanned += 1
            # 獲取文件路徑 (兼容字典或對象訪問)
            file_path = getattr(doc_obj, "file_path", "") or doc_obj.get("file_path", "")
            
            # 檢查是否包含我們要刪除的關鍵字
            for target in targets:
                if target in file_path:
                    print(f"   🎯 找到目標: {file_path} (ID: {doc_id}) [Status: {status}]")
                    found_ids.add(doc_id)
                    break # 找到一個關鍵字匹配就夠了
                    
    print(f"📊 掃描完成: 共檢查 {total_scanned} 個文檔，找到 {len(found_ids)} 個待刪除。")
    return list(found_ids)

async def main():
    print(f"🚀 初始化 LightRAG 以進行清理...")
    
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=azure_openai_complete, 
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=8192,
            func=partial(
                ollama_embed, 
                host=OLLAMA_HOST,
                embed_model=EMBEDDING_MODEL
            )
        )
    )

    print("⚙️ 正在初始化 Storage...")
    await rag.initialize_storages()

    # 1. 根據文件名查找 ID
    if not TARGET_FILES_TO_DELETE:
        print("⚠️ 沒有設定要刪除的文件名 (TARGET_FILES_TO_DELETE 為空)。")
        return

    ids_to_delete = await find_ids_by_filename(rag, TARGET_FILES_TO_DELETE)

    if not ids_to_delete:
        print("✅ 沒有發現需要刪除的文件，系統乾淨。")
        return

    # 2. 用戶確認 (防止誤刪)
    confirm = input(f"⚠️ 即將刪除 {len(ids_to_delete)} 個文檔 (包含相關的 Chunks 和 Graph Nodes)。確定嗎? (y/n): ")
    if confirm.lower() != 'y':
        print("❌ 操作已取消。")
        return

    # 3. 執行刪除
    print(f"🗑️ 開始刪除...")
    
    for i, doc_id in enumerate(ids_to_delete):
        try:
            print(f" [{i+1}/{len(ids_to_delete)}] 正在刪除 ID: {doc_id} ...")
            
            # 調用 LightRAG 的刪除接口
            await rag.adelete_by_doc_id(doc_id)
            
            print(f"   ✅ 刪除成功")
        except Exception as e:
            # 忽略 "Not found" 錯誤，因為可能已經被清理過
            if "not found" in str(e).lower():
                print(f"   ⚠️ 文檔已不存在 (視為成功)")
            else:
                print(f"   ❌ 刪除失敗: {str(e)}")

    print("🏁 所有指定文件清理完成！")

if __name__ == "__main__":
    asyncio.run(main())