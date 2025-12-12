import sys
import os
import subprocess
import json
import glob
import time
from loguru import logger

# === 1. 設定 Logging (統一風格) ===
LOG_DIR = "./logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# 設定 Log 檔案名
log_file = os.path.join(LOG_DIR, f"step1_run_{time.strftime('%Y%m%d_%H%M%S')}.log")

# 重置 Logger 設定
logger.remove() 

# Handler 1: Console (螢幕輸出 - 簡潔)
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>", level="INFO")

# Handler 2: File (檔案紀錄 - 詳細)
logger.add(log_file, rotation="10 MB", level="DEBUG", encoding="utf-8")

logger.info(f"📝 Log 檔案已建立: {log_file}")
# ==========================================

def process_single_file(input_path, output_base_dir, step1_std_dir, config):
    filename = os.path.basename(input_path)
    file_stem = os.path.splitext(filename)[0]
    
    logger.info("="*60)
    logger.info(f"🚀 [Start] 正在處理: {filename}")
    logger.info(f"📍 路徑: {input_path}")
    logger.info("="*60)

    start_time = time.time()

    # 組合指令
    cmd = [
        "uv", "run", "mineru",
        "-p", input_path,
        "-o", output_base_dir,
        "-m", "auto",
        "-b", config["use_backend"],
        "-d", config["use_device"]
    ]

    logger.info(f"🔧 執行指令: {' '.join(cmd)}")

    try:
        # 🌟 使用 Popen 即時抓取 Mineru 的輸出並轉發給 loguru 🌟
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, # 將 stderr 合併到 stdout
            text=True,
            encoding='utf-8',
            errors='replace'
        )

        # 逐行讀取 Mineru 的輸出
        for line in process.stdout:
            line = line.strip()
            if line:
                # 這裡用 info 級別，如果你覺得螢幕太花，可以改 logger.debug
                logger.info(f"   [Mineru] {line}")

        # 等待指令結束
        return_code = process.wait()

        if return_code != 0:
            logger.error(f"❌ Mineru 執行失敗，Return Code: {return_code}")
            return False # 回傳失敗狀態

        # --- 後續 JSON 處理邏輯 (只有成功才跑) ---
        # Mineru 的輸出路徑結構有時會變，這裡保留你的雙重檢查邏輯
        possible_paths = [
            os.path.join(output_base_dir, file_stem, config["use_backend"], f"{file_stem}_content_list.json"),
            os.path.join(output_base_dir, file_stem, f"{file_stem}_content_list.json"),
        ]
        
        target_json = None
        for p in possible_paths:
            if os.path.exists(p):
                target_json = p
                break
        
        if target_json:
            final_json_name = f"intermediate_result.json" # 統一命名方便 Step 2 讀取
            # 如果你想保留原檔名，可以用: f"{file_stem}.json"
            
            final_json_path = os.path.join(step1_std_dir, final_json_name)
            
            try:
                with open(target_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # 這裡可以加入額外 meta data
                # if isinstance(data, list):
                #     for block in data:
                #         block['original_filename'] = filename

                with open(final_json_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                
                duration = time.time() - start_time
                logger.success(f"✅ 解析成功！耗時: {duration:.2f} 秒")
                logger.success(f"💾 中間檔已儲存: {final_json_path}")
                return True

            except Exception as json_err:
                logger.error(f"⚠️ JSON 讀寫錯誤: {json_err}")
                return False
        else:
            logger.error(f"❌ Mineru 雖然跑完 (Code 0)，但找不到輸出的 JSON 檔案。")
            return False

    except Exception as e:
        logger.exception(f"⚠️ 發生未預期錯誤: {e}") # exception 會印出 Traceback
        return False

def main():
    # === 設定區 ===
    input_dir = "./data/inputs"
    output_dir = "./data/output/step1_vlm_output"
    step1_std_dir = "./data/input/step1_output" # 確保這裡跟 Step 2 的 input 對得上
    
    config = {
        "use_backend": "vlm-transformers",
        "use_device": "cpu"
    }
    # ============

    if not os.path.exists(input_dir):
        logger.error(f"❌ 找不到輸入資料夾: {input_dir}")
        return

    if not os.path.exists(step1_std_dir):
        os.makedirs(step1_std_dir)

    files = glob.glob(os.path.join(input_dir, "*.pdf"))
    if not files:
        logger.warning(f"📂 資料夾 {input_dir} 內找不到任何 .pdf 檔案")
        return

    logger.info(f"📦 發現 {len(files)} 個檔案，準備開始批次處理...")
    logger.info(f"⚙️ 設定: {json.dumps(config)}")

    success_count = 0
    fail_count = 0

    for i, file_path in enumerate(files):
        logger.info(f"\n⏳ [總進度: {i+1}/{len(files)}]")
        
        # 執行單個檔案處理
        if process_single_file(file_path, output_dir, step1_std_dir, config):
            success_count += 1
        else:
            fail_count += 1

    logger.info("\n" + "="*60)
    logger.info(f"🏁 所有作業完成！")
    logger.info(f"📊 統計: 總數 {len(files)} | 成功 {success_count} | 失敗 {fail_count}")
    logger.info(f"📝 詳細 Log 請查看: {LOG_DIR}")

if __name__ == "__main__":
    main()