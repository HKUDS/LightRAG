import sys
import os
import subprocess
import json
import glob
import time
import torch  # 🔥 新增 torch 用於偵測 GPU
from loguru import logger

# === 1. 設定 Logging (統一風格) ===
LOG_DIR = "./logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# 設定 Log 檔案名
log_file = os.path.join(LOG_DIR, f"step1_mineru_gpu_{time.strftime('%Y%m%d_%H%M%S')}.log")

# 重置 Logger 設定
logger.remove() 

# Handler 1: Console (螢幕輸出 - 簡潔)
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>", level="INFO")

# Handler 2: File (檔案紀錄 - 詳細)
logger.add(log_file, rotation="10 MB", level="DEBUG", encoding="utf-8")

logger.info(f"📝 Log 檔案已建立: {log_file}")
# ==========================================

def get_device_config():
    """
    🔥 自動偵測最佳執行裝置
    """
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.success(f"🎮 偵測到 GPU: {gpu_name} ({vram_gb:.2f} GB VRAM)")
        logger.info("🚀 Mineru 將以 CUDA 模式執行 (極速模式)")
        return "cuda"
    else:
        logger.warning("⚠️ 未偵測到 GPU，將退回 CPU 模式 (速度較慢)")
        return "cpu"

def process_single_file(input_path, output_base_dir, step1_std_base_dir, config):
    filename = os.path.basename(input_path)
    file_stem = os.path.splitext(filename)[0]
    ext = os.path.splitext(filename)[1].lower()
    
    # 建立專屬資料夾
    current_file_std_dir = os.path.join(step1_std_base_dir, file_stem)
    
    if not os.path.exists(current_file_std_dir):
        os.makedirs(current_file_std_dir)
        logger.info(f"📂 已建立專屬目錄: {current_file_std_dir}")

    logger.info("="*60)
    logger.info(f"🚀 [Start] 正在處理: {filename}")
    logger.info(f"📍 路徑: {input_path}")
    logger.info("="*60)

    start_time = time.time()

    # === [策略 A] 針對純文字檔 (.txt / .md) 的特殊處理 (Bypass Mineru) ===
    if ext in ['.txt', '.md']:
        logger.info(f"📄 檢測到純文字檔 ({ext})，跳過 Mineru，直接轉換格式...")
        
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # 構造 Step 2 需要的 JSON 格式
            mock_data = [{
                "type": "text",
                "text": content,
                "page_idx": 0,
                "bbox": [0, 0, 0, 0], 
                "img_path": ""
            }]
            
            final_json_path = os.path.join(current_file_std_dir, "intermediate_result.json")
            
            with open(final_json_path, "w", encoding="utf-8") as f:
                json.dump(mock_data, f, ensure_ascii=False, indent=2)
                
            duration = time.time() - start_time
            logger.success(f"✅ 轉換成功！(Text Bypass Mode) 耗時: {duration:.2f} 秒")
            return True
            
        except Exception as e:
            logger.error(f"❌ 讀取文字檔失敗: {e}")
            return False

    # === [策略 B] 針對 PDF，呼叫 Mineru (GPU Mode) ===
    # 組合指令
    cmd = [
        "uv", "run", "mineru",
        "-p", input_path,
        "-o", output_base_dir,
        "-m", "auto",                 # 模型選擇
        "-b", config["use_backend"],  # Backend (vlm-transformers 支援 GPU)
        "-d", config["use_device"]    # 🔥 關鍵：這裡是 "cuda"
    ]

    logger.info(f"🔧 執行 Mineru 指令: {' '.join(cmd)}")

    try:
        # 使用 Popen 即時抓取 Mineru 的輸出
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, 
            text=True,
            encoding='utf-8',
            errors='replace'
        )

        for line in process.stdout:
            line = line.strip()
            if line:
                # 過濾掉一些無用的進度條，只顯示關鍵訊息
                if "Processing" in line or "Loading" in line or "Error" in line:
                    logger.info(f"   [Mineru] {line}")

        return_code = process.wait()

        if return_code != 0:
            logger.error(f"❌ Mineru 執行失敗，Return Code: {return_code}")
            return False 

        # --- 後續 JSON 處理邏輯 (Mineru 成功後) ---
        # Mineru 的輸出結構通常是: output_dir/filename/model_name/filename_content_list.json
        possible_paths = [
            os.path.join(output_base_dir, file_stem, config["use_backend"], f"{file_stem}_content_list.json"),
            os.path.join(output_base_dir, file_stem, "auto", f"{file_stem}_content_list.json"),
            os.path.join(output_base_dir, file_stem, f"{file_stem}_content_list.json"),
        ]
        
        target_json = None
        for p in possible_paths:
            if os.path.exists(p):
                target_json = p
                break
        
        if target_json:
            final_json_name = "intermediate_result.json"
            final_json_path = os.path.join(current_file_std_dir, final_json_name)
            
            try:
                with open(target_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                with open(final_json_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                
                duration = time.time() - start_time
                logger.success(f"✅ 解析成功！耗時: {duration:.2f} 秒")
                logger.success(f"💾 中間檔已儲存於: {final_json_path}")
                return True

            except Exception as json_err:
                logger.error(f"⚠️ JSON 讀寫錯誤: {json_err}")
                return False
        else:
            logger.error(f"❌ Mineru 雖然跑完 (Code 0)，但找不到輸出的 JSON 檔案。請檢查: {output_base_dir}")
            return False

    except Exception as e:
        logger.exception(f"⚠️ 發生未預期錯誤: {e}") 
        return False

def main():
    # === 設定區 ===
    input_dir = "./data/input/__enqueued__"
    
    # 這是 Mineru 的原始輸出區 (會包含各種模型子資料夾)
    output_dir = "./data/output/step1_vlm_output"
    
    # 這是整理後的乾淨輸出區 (給 Step 2 用)
    step1_std_base_dir = "./data/input/step1_output" 
    
    # 🛑 排除名單 (全部轉小寫比對)
    EXCLUDE_FILES = ["sfc.pdf", "sfc_report.pdf"] 
    
    FORCE_RERUN = False 

    # 🔥 GPU 設定偵測
    device_type = get_device_config()
    
    config = {
        # 新版 Mineru 建議使用 'pipeline'，它會根據 -d cuda 自動調用 GPU
        "use_backend": "pipeline", 
        "use_device": device_type
    }
    # ============

    if not os.path.exists(input_dir):
        logger.error(f"❌ 找不到輸入資料夾: {input_dir}")
        return

    if not os.path.exists(step1_std_base_dir):
        os.makedirs(step1_std_base_dir)

    # 掃描檔案
    all_entries = glob.glob(os.path.join(input_dir, "*"))
    files = []
    
    logger.info(f"🔍 正在掃描所有檔案...")
    for entry in all_entries:
        if os.path.isfile(entry):
            filename = os.path.basename(entry)
            if not filename.startswith("."):
                files.append(entry)

    if not files:
        logger.warning(f"📂 資料夾 {input_dir} 內找不到任何檔案")
        return

    logger.info(f"📦 發現 {len(files)} 個檔案，準備開始批次處理...")
    
    success_count = 0
    fail_count = 0
    skipped_count = 0

    for i, file_path in enumerate(files):
        logger.info(f"\n⏳ [總進度: {i+1}/{len(files)}]")
        
        filename = os.path.basename(file_path)
        file_stem = os.path.splitext(filename)[0]

        # === 🛑 檢查是否在排除名單 ===
        if filename.lower() in EXCLUDE_FILES:
            logger.info(f"🛑 檢測到排除檔案，明確跳過: {filename}")
            skipped_count += 1
            continue
        
        # --- 🔄 檢查是否已處理 (Resume Logic) ---
        expected_output_path = os.path.join(step1_std_base_dir, file_stem, "intermediate_result.json")
        
        if not FORCE_RERUN and os.path.exists(expected_output_path):
            logger.info(f"⏭️  檢測到檔案已存在，跳過處理: {filename}")
            skipped_count += 1
            success_count += 1 
            continue
        
        # 執行單個檔案處理
        if process_single_file(file_path, output_dir, step1_std_base_dir, config):
            success_count += 1
        else:
            fail_count += 1

    logger.info("\n" + "="*60)
    logger.info(f"🏁 所有作業完成！")
    logger.info(f"📊 統計: 總數 {len(files)} | ✅ 完成/跳過 {success_count} | ⏭️ 跳過 {skipped_count} | ❌ 失敗 {fail_count}")
    logger.info(f"📝 詳細 Log 請查看: {LOG_DIR}")

if __name__ == "__main__":
    main()