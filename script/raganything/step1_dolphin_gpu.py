import sys
import os
import json
import glob
import time
import math
import gc
import torch
from pdf2image import convert_from_path, pdfinfo_from_path
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from loguru import logger

# ==========================================
# 🔥 [使用者設定區] 請根據你的環境修改這裡
# ==========================================

# 1. Poppler 路徑 (Windows 必填)
# 如果你已經加到系統環境變數，可以設為 None
POPPLER_BIN_PATH = r"C:\Users\sammi_hung\LightRAG\poppler-25.12.0\Library\bin"

# 2. 輸入/輸出路徑
INPUT_DIR = "./data/input/__enqueued__"
OUTPUT_DIR_BASE = "./data/input/step1_output"

# 3. 模型設定
MODEL_ID = "ByteDance/Dolphin-v2"

# 4. 效能調優 (針對 RTX 4060 8GB)
# 12000-14000 是 8GB VRAM 的安全區間
# 14000 tokens ≈ 274萬像素 (例如 1400x1900)
MAX_VISUAL_TOKENS = 14000 

# 每幾頁存檔一次 (防止當機資料全失)
SAVE_INTERVAL = 5 

# ==========================================

# === 0. 環境準備 ===
# 將 Poppler 加入 PATH
if POPPLER_BIN_PATH and os.path.exists(POPPLER_BIN_PATH):
    if POPPLER_BIN_PATH not in os.environ["PATH"]:
        os.environ["PATH"] += os.pathsep + POPPLER_BIN_PATH
        print(f"🔧 已將 Poppler 加入 PATH: {POPPLER_BIN_PATH}")

# 設定 Logging
LOG_DIR = "./logs"
if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
log_file = os.path.join(LOG_DIR, f"step1_dolphin_gpu_{time.strftime('%Y%m%d_%H%M%S')}.log")

logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>", level="INFO")
logger.add(log_file, rotation="10 MB", level="DEBUG", encoding="utf-8")
logger.info(f"📝 Log 檔案已建立: {log_file}")

# === 1. 輔助函式 ===

def resize_to_token_limit(image, max_tokens=14000):
    """
    智慧縮圖：確保圖片不會產生過多 Token 導致 OOM
    Qwen2.5-VL: 1 token ≈ 14x14 pixels = 196 pixels
    """
    w, h = image.size
    total_pixels = w * h
    current_tokens = total_pixels / 196
    
    if current_tokens > max_tokens:
        scale = math.sqrt(max_tokens / current_tokens)
        new_w = int(w * scale)
        new_h = int(h * scale)
        logger.debug(f"📉 圖片壓縮: {w}x{h} -> {new_w}x{new_h} (Tokens: {int(current_tokens)} -> ~{max_tokens})")
        return image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    return image

def load_model():
    """
    載入模型 (4-bit 量化 + GPU 加速)
    """
    logger.info("="*60)
    logger.info(f"📥 正在載入模型: {MODEL_ID} (4-bit GPU Mode)...")
    
    if not torch.cuda.is_available():
        logger.critical("❌ 檢測不到 GPU！請確認 CUDA 是否安裝正確。")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f"🎮 GPU 偵測: {gpu_name} | VRAM: {vram_gb:.2f} GB")

    try:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        logger.success("✅ 模型載入完成！")
        return model, processor
    except Exception as e:
        logger.critical(f"❌ 模型載入失敗: {e}")
        sys.exit(1)

def process_single_file(input_path, output_base, model, processor):
    filename = os.path.basename(input_path)
    file_stem = os.path.splitext(filename)[0]
    
    # 建立輸出目錄
    current_out_dir = os.path.join(output_base, file_stem)
    if not os.path.exists(current_out_dir): os.makedirs(current_out_dir)
    final_json = os.path.join(current_out_dir, "intermediate_result.json")
    images_dir = os.path.join(current_out_dir, "images")
    if not os.path.exists(images_dir): os.makedirs(images_dir)

    logger.info("-" * 40)
    logger.info(f"🚀 [Start] 處理檔案: {filename}")
    
    start_time = time.time()

    # 1. 獲取總頁數 (不讀取圖片，只讀 Metadata，極快且省 RAM)
    try:
        info = pdfinfo_from_path(input_path, poppler_path=POPPLER_BIN_PATH)
        total_pages = info["Pages"]
        logger.info(f"📄 PDF 總頁數: {total_pages}")
    except Exception as e:
        logger.error(f"❌ 無法讀取 PDF 資訊 (可能是路徑錯誤或檔案損毀): {e}")
        return False

    parsed_results = []
    
    # 如果有舊的進度，可以在這裡載入 (Optional)
    # ...

    # 2. 逐頁處理 (Page-by-Page)
    for i in range(1, total_pages + 1):
        page_start = time.time()
        logger.info(f"   🔄 正在處理 Page {i}/{total_pages}...")

        try:
            # 🔥 關鍵優化：只載入「這一頁」
            # dpi=200 保證原始細節，後續再用 resize_to_token_limit 縮小
            page_images = convert_from_path(
                input_path, 
                dpi=200, 
                first_page=i, 
                last_page=i, 
                poppler_path=POPPLER_BIN_PATH
            )
            
            if not page_images:
                logger.warning(f"⚠️ Page {i} 讀取空白，跳過。")
                continue
            
            image = page_images[0]
            
            # 儲存原圖 (方便 Debug)
            img_filename = f"page_{i-1}.jpg" # 0-based index for saving
            image.save(os.path.join(images_dir, img_filename))

            # 智慧縮放 (保護 VRAM)
            image = resize_to_token_limit(image, max_tokens=MAX_VISUAL_TOKENS)

            # 建構 Prompt
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Read the text in the image word by word and transcribe it into Markdown format. Represent tables using Markdown syntax."}
                ]
            }]

            # 準備 Tensor
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = processor(
                text=[text], 
                images=image_inputs, 
                videos=video_inputs, 
                padding=True, 
                return_tensors="pt"
            ).to(model.device)

            # 推理 Generation
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=2048, # 如果還爆，可降至 1024
                    do_sample=False
                )

            # 解碼 Decoding
            gen_ids_trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
            md_text = processor.batch_decode(gen_ids_trimmed, skip_special_tokens=True)[0]

            # 存入結果列表 (注意 page_idx 轉為 0-based)
            parsed_results.append({
                "type": "text",
                "text": md_text,
                "page_idx": i - 1,
                "img_path": f"images/{img_filename}",
                "bbox": [0, 0, image.width, image.height]
            })

            dur = time.time() - page_start
            logger.info(f"     ✅ 完成 ({dur:.2f}s) | VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")

            # 🔥 定期存檔 (Incremental Save)
            if i % SAVE_INTERVAL == 0 or i == total_pages:
                with open(final_json, "w", encoding="utf-8") as f:
                    json.dump(parsed_results, f, ensure_ascii=False, indent=2)
                logger.debug(f"💾 進度已儲存 (Page {i})")

        except torch.cuda.OutOfMemoryError:
            logger.error(f"❌ Page {i} OOM (顯存不足)！跳過此頁。建議降低 MAX_VISUAL_TOKENS。")
            torch.cuda.empty_cache()
        except Exception as e:
            logger.exception(f"❌ Page {i} 發生錯誤: {e}")
        finally:
            # 🔥 極致清理：確保每一頁處理完都釋放資源
            if 'inputs' in locals(): del inputs
            if 'generated_ids' in locals(): del generated_ids
            if 'image' in locals(): del image
            if 'page_images' in locals(): del page_images
            torch.cuda.empty_cache()
            gc.collect()

    logger.success(f"🎉 檔案處理完成！總耗時: {time.time() - start_time:.2f}s")
    return True

def main():
    if not os.path.exists(INPUT_DIR):
        logger.error(f"❌ 找不到輸入目錄: {INPUT_DIR}")
        return
    if not os.path.exists(OUTPUT_DIR_BASE):
        os.makedirs(OUTPUT_DIR_BASE)

    # 載入模型 (只做一次)
    model, processor = load_model()

    # 掃描檔案
    all_files = glob.glob(os.path.join(INPUT_DIR, "*"))
    files = [f for f in all_files if os.path.isfile(f) and not os.path.basename(f).startswith(".")]
    
    logger.info(f"📦 發現 {len(files)} 個檔案，準備開始...")

    for idx, file_path in enumerate(files):
        logger.info(f"\n[{idx+1}/{len(files)}] ----------------------------------------")
        
        # 排除非 PDF (暫時只處理 PDF，如果是圖片可自行修改)
        if not file_path.lower().endswith(".pdf"):
            logger.warning(f"⏭️ 跳過非 PDF 檔案: {file_path}")
            continue
            
        process_single_file(file_path, OUTPUT_DIR_BASE, model, processor)

    logger.success("\n🏁 所有任務執行完畢！")

if __name__ == "__main__":
    main()