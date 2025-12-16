import sys
import os
import json
import glob
import time
import requests
from loguru import logger
from dotenv import load_dotenv

# === 1. 設定 Logging ===
LOG_DIR = "./logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

log_file = os.path.join(LOG_DIR, f"step1_api_run_{time.strftime('%Y%m%d_%H%M%S')}.log")
logger.remove() 
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>", level="INFO")
logger.add(log_file, rotation="10 MB", level="DEBUG", encoding="utf-8")
logger.info(f"📝 Log 檔案已建立: {log_file}")

# 讀取 .env
load_dotenv()

# === API 設定 ===
API_TOKEN = os.getenv("MINERU_API_TOKEN")
UPLOAD_APPLY_URL = "https://mineru.net/api/v4/file-urls/batch"
QUERY_BASE_URL = "https://mineru.net/api/v4/extract/task"
# =================

def process_single_file_via_api(input_path, step1_std_base_dir):
    """
    處理單個檔案：
    1. 如果是 TXT/MD -> 本地處理 (Bypass API)
    2. 如果是 PDF -> 呼叫 Mineru API (含斷線重連機制)
    """
    filename = os.path.basename(input_path)
    file_stem = os.path.splitext(filename)[0]
    ext = os.path.splitext(filename)[1].lower()
    
    # 建立專屬資料夾
    current_file_std_dir = os.path.join(step1_std_base_dir, file_stem)
    if not os.path.exists(current_file_std_dir):
        os.makedirs(current_file_std_dir)
    
    final_json_path = os.path.join(current_file_std_dir, "intermediate_result.json")

    logger.info("-" * 40)
    logger.info(f"🚀 [Start] 處理: {filename}")

    # === [策略 A] 純文字檔 Bypass (省錢/省時間) ===
    if ext in ['.txt', '.md']:
        logger.info(f"📄 純文字檔 ({ext}) -> 跳過 API，直接轉換...")
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            mock_data = [{
                "type": "text",
                "text": content,
                "page_idx": 0,
                "bbox": [0, 0, 0, 0], 
                "img_path": ""
            }]
            
            with open(final_json_path, "w", encoding="utf-8") as f:
                json.dump(mock_data, f, ensure_ascii=False, indent=2)
                
            logger.success(f"✅ [Text Mode] 轉換成功！")
            return True
        except Exception as e:
            logger.error(f"❌ 讀取文字檔失敗: {e}")
            return False

    # === [策略 B] PDF/其他 -> 使用 Mineru API ===
    if not API_TOKEN:
        logger.error("❌ 找不到 MINERU_API_TOKEN，請檢查 .env")
        return False

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_TOKEN}"
    }

    try:
        # 1. 申請上傳連結
        logger.info("📡 1. 申請 API 上傳連結...")
        apply_data = {
            "files": [{"name": filename, "data_id": f"local_{int(time.time())}"}],
            "model_version": "vlm"
        }
        
        # 加入 timeout 防止卡死
        res = requests.post(UPLOAD_APPLY_URL, headers=headers, json=apply_data, timeout=30)
        res.raise_for_status()
        res_json = res.json()
        
        if res_json.get("code") != 0:
            logger.error(f"❌ API 申請失敗: {res_json.get('msg')}")
            return False

        batch_id = res_json["data"]["batch_id"]
        upload_url = res_json["data"]["file_urls"][0]
        logger.info(f"   Batch ID: {batch_id}")

        # 2. 上傳檔案
        logger.info("📤 2. 上傳檔案至 Mineru...")
        with open(input_path, 'rb') as f:
            # 上傳通常比較久，timeout 設長一點 (例如 300秒)
            upload_res = requests.put(upload_url, data=f, timeout=300)
            if upload_res.status_code != 200:
                logger.error(f"❌ 上傳失敗 (Code {upload_res.status_code})")
                return False

        # 3. 輪詢狀態 (🔥 加入斷線重連機制 🔥)
        logger.info("⏳ 3. 等待伺服器解析...")
        query_url = f"{QUERY_BASE_URL}/{batch_id}"
        
        network_retry_count = 0
        MAX_RETRIES = 20  # 最多容許連續失敗 20 次 (約 10 分鐘斷網容忍)
        
        while True:
            try:
                # 查詢狀態 (timeout=30)
                status_res = requests.get(query_url, headers=headers, timeout=30)
                status_res.raise_for_status()
                
                # 成功連線，重置計數器
                network_retry_count = 0
                
                status_data = status_res.json().get("data", {})
                state = status_data.get("state")
                
                if state == "done":
                    break
                elif state == "failed":
                    logger.error("❌ API 解析任務回報失敗 (State: failed)")
                    return False
                else:
                    # 正在處理中，正常等待 5 秒
                    time.sleep(5)

            except (requests.exceptions.RequestException, Exception) as e:
                network_retry_count += 1
                logger.warning(f"⚠️ 網絡連線不穩 ({network_retry_count}/{MAX_RETRIES}): {e}")
                
                if network_retry_count > MAX_RETRIES:
                    logger.error("❌ 連續多次連線失敗，放棄此檔案。")
                    return False
                
                # 失敗後休息 30 秒再重試，給網絡恢復時間
                logger.info("🔄 網絡異常，30秒後嘗試重連...")
                time.sleep(30)

        # 4. 下載結果
        logger.info("⬇️ 4. 下載解析結果...")
        result_links = status_data.get("links", [])
        content_json_url = None
        
        for link in result_links:
            if link.get("file_name", "").endswith("content_list.json"):
                content_json_url = link.get("url")
                break
        
        if not content_json_url and result_links:
             content_json_url = result_links[0].get("url")

        if content_json_url:
            # 下載也要加 timeout 和簡單重試
            for _ in range(3):
                try:
                    content_res = requests.get(content_json_url, timeout=60)
                    content_res.raise_for_status()
                    extracted_data = content_res.json()
                    
                    with open(final_json_path, "w", encoding="utf-8") as f:
                        json.dump(extracted_data, f, ensure_ascii=False, indent=2)

                    logger.success(f"✅ [API Mode] 解析成功！")
                    logger.success(f"💾 儲存於: {final_json_path}")
                    return True
                except Exception as dl_err:
                    logger.warning(f"⚠️ 下載失敗，重試中: {dl_err}")
                    time.sleep(5)
            
            logger.error("❌ 下載結果失敗 (重試 3 次後)")
            return False
        else:
            logger.error("❌ 找不到結果下載鏈接")
            return False

    except Exception as e:
        logger.exception(f"⚠️ API 處理發生未預期錯誤: {e}")
        return False

def main():
    # === 設定區 ===
    input_dir = "./data/input/__enqueued__"
    
    # 這是 Step 2 讀取的目錄結構
    step1_std_base_dir = "./data/input/step1_output" 
    
    FORCE_RERUN = False 
    EXCLUDE_FILES = ["sfc.pdf", "sfc_report.pdf"] 
    # ============

    if not os.path.exists(input_dir):
        logger.error(f"❌ 找不到輸入資料夾: {input_dir}")
        return

    if not os.path.exists(step1_std_base_dir):
        os.makedirs(step1_std_base_dir)

    # 掃描所有檔案
    all_entries = glob.glob(os.path.join(input_dir, "*"))
    files = []
    
    logger.info(f"🔍 正在掃描資料夾: {input_dir}")
    
    for entry in all_entries:
        if os.path.isfile(entry):
            filename = os.path.basename(entry)
            if not filename.startswith("."):
                files.append(entry)

    if not files:
        logger.warning(f"📂 資料夾內無檔案")
        return

    logger.info(f"📦 發現 {len(files)} 個檔案...")
    
    success_count = 0
    fail_count = 0
    skipped_count = 0

    for i, file_path in enumerate(files):
        logger.info(f"\n[進度: {i+1}/{len(files)}]")
        
        filename = os.path.basename(file_path)
        file_stem = os.path.splitext(filename)[0]

        # 排除檢查
        if filename.lower() in EXCLUDE_FILES:
            logger.info(f"🛑 跳過排除名單: {filename}")
            skipped_count += 1
            continue
        
        # 斷點續傳檢查
        expected_output = os.path.join(step1_std_base_dir, file_stem, "intermediate_result.json")
        if not FORCE_RERUN and os.path.exists(expected_output):
            logger.info(f"⏭️ 檔案已存在，跳過: {filename}")
            skipped_count += 1
            success_count += 1
            continue
        
        # 呼叫處理函式
        if process_single_file_via_api(file_path, step1_std_base_dir):
            success_count += 1
        else:
            fail_count += 1

    logger.info("\n" + "="*60)
    logger.info(f"🏁 作業完成！成功: {success_count} | 跳過: {skipped_count} | 失敗: {fail_count}")

if __name__ == "__main__":
    main()