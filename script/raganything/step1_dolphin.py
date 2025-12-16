import sys
import os
import json
import glob
import time
import math
import io
import psutil
import torch
import pymupdf  # 🔥 取代 pdf2image (官方 Dolphin 使用這個)
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from loguru import logger

# ==========================================
# 🔥 [使用者設定區]
# ==========================================

# 1. 路徑設定
INPUT_DIR = "./data/input/__enqueued__"
OUTPUT_DIR_BASE = "./data/input/step1_output"

# 2. 模型設定
MODEL_ID = "ByteDance/Dolphin-v2"

# 3. 畫質設定 (PyMuPDF Zoom Factor)
# 標準 PDF 是 72 DPI。設定 zoom=4.16 大約等於 300 DPI (高品質印刷標準)
# 這樣能確保小字體也能被精確識別
PDF_ZOOM = 300 / 72  # ~4.166

# 4. Token 限制 (保護 RAM)
# 25000 tokens ≈ 490萬像素。
# 配合 CPU 模式，這個設定能吃下 A4 全頁高畫質細節。
MAX_VISUAL_TOKENS = 25000 

# 每幾頁存檔一次
SAVE_INTERVAL = 1

# ==========================================

# 設定 Logging
LOG_DIR = "./logs"
if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
log_file = os.path.join(LOG_DIR, f"step1_dolphin_official_{time.strftime('%Y%m%d_%H%M%S')}.log")

logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>", level="INFO")
logger.add(log_file, rotation="10 MB", level="DEBUG", encoding="utf-8")
logger.info(f"📝 Log 檔案已建立: {log_file}")

# === 輔助函式 ===

def get_ram_usage():
    """取得目前系統 RAM 使用量 (GB)"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)

def resize_to_token_limit(image, max_tokens=25000):
    """
    智慧縮圖：防止超大圖片導致推理過慢
    """
    w, h = image.size
    total_pixels = w * h
    current_tokens = total_pixels / 196  # Qwen2.5-VL patch size 14x14
    
    if current_tokens > max_tokens:
        scale = math.sqrt(max_tokens / current_tokens)
        new_w = int(w * scale)
        new_h = int(h * scale)
        logger.info(f"📉 [Resize] 圖片過大: {w}x{h} -> {new_w}x{new_h} (Tokens: {int(current_tokens)} -> ~{max_tokens})")
        return image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    return image

def load_model():
    """
    載入模型 (CPU High-Quality Mode)
    """
    logger.info("="*60)
    logger.info(f"📥 正在載入模型: {MODEL_ID} (CPU + PyMuPDF)...")
    logger.info("⚠️ 注意：CPU 推理速度較慢，請耐心等待。")
    
    try:
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        # 使用 float32 確保 CPU 上的最佳相容性與畫質
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32, 
            device_map="cpu",          
            trust_remote_code=True
        )
        logger.success(f"✅ 模型載入完成！目前 RAM 使用: {get_ram_usage():.2f} GB")
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

    # === 🔥 [核心修改] 使用 PyMuPDF 讀取 PDF ===
    try:
        doc = pymupdf.open(input_path)
        total_pages = len(doc)
        logger.info(f"📄 PDF 總頁數: {total_pages} (Engine: PyMuPDF)")
    except Exception as e:
        logger.error(f"❌ 無法讀取 PDF: {e}")
        return False

    parsed_results = []
    
    # 逐頁處理
    for i in range(total_pages):
        # 注意: PyMuPDF 頁碼從 0 開始
        page_start = time.time()
        logger.info(f"   🔄 正在處理 Page {i+1}/{total_pages} ...")

        try:
            # 1. 渲染頁面 (Render Page)
            page = doc[i]
            # 設定縮放矩陣 (控制 DPI)
            mat = pymupdf.Matrix(PDF_ZOOM, PDF_ZOOM)
            pix = page.get_pixmap(matrix=mat, alpha=False) # alpha=False 移除透明通道，轉為 RGB
            
            # 2. 轉換為 PIL Image
            # 方法參考官方 utils.py: 使用 tobytes("png") 再用 PIL 開啟，最穩健
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))
            
            # 儲存原圖備份
            img_filename = f"page_{i}.jpg"
            image.save(os.path.join(images_dir, img_filename))

            # 3. 智慧縮放 (Token Limit)
            image = resize_to_token_limit(image, max_tokens=MAX_VISUAL_TOKENS)

            # 4. 建構 Prompt
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Read the text in the image word by word and transcribe it into Markdown format. Represent tables using Markdown syntax."}
                ]
            }]

            # 5. 推理 (Inference)
            text_inputs = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = processor(
                text=[text_inputs], 
                images=image_inputs, 
                videos=video_inputs, 
                padding=True, 
                return_tensors="pt"
            ).to("cpu")

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=4096, 
                    do_sample=False
                )

            gen_ids_trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
            md_text = processor.batch_decode(gen_ids_trimmed, skip_special_tokens=True)[0]

            # 6. 儲存結果
            parsed_results.append({
                "type": "text",
                "text": md_text,
                "page_idx": i,
                "img_path": f"images/{img_filename}",
                "bbox": [0, 0, image.width, image.height]
            })

            dur = time.time() - page_start
            logger.info(f"     ✅ 完成 (耗時: {dur:.2f}s) | RAM: {get_ram_usage():.2f} GB")

            # 定期寫入硬碟
            if (i + 1) % SAVE_INTERVAL == 0 or (i + 1) == total_pages:
                with open(final_json, "w", encoding="utf-8") as f:
                    json.dump(parsed_results, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.exception(f"❌ Page {i+1} 發生錯誤: {e}")
        
        finally:
            # 清理記憶體
            if 'inputs' in locals(): del inputs
            if 'generated_ids' in locals(): del generated_ids
            if 'image' in locals(): del image
            if 'pix' in locals(): del pix  # 清理 PyMuPDF 物件
            import gc; gc.collect()

    logger.success(f"🎉 檔案處理完成！總耗時: {time.time() - start_time:.2f}s")
    doc.close()
    return True

def main():
    if not os.path.exists(INPUT_DIR):
        logger.error(f"❌ 找不到輸入目錄: {INPUT_DIR}")
        return
    if not os.path.exists(OUTPUT_DIR_BASE):
        os.makedirs(OUTPUT_DIR_BASE)

    # 載入模型
    model, processor = load_model()

    # 掃描檔案
    all_files = glob.glob(os.path.join(INPUT_DIR, "*"))
    files = [f for f in all_files if os.path.isfile(f) and not os.path.basename(f).startswith(".")]
    
    logger.info(f"📦 發現 {len(files)} 個檔案，準備開始...")

    for idx, file_path in enumerate(files):
        logger.info(f"\n[{idx+1}/{len(files)}] ----------------------------------------")
        
        if not file_path.lower().endswith(".pdf"):
            logger.warning(f"⏭️ 跳過非 PDF 檔案: {file_path}")
            continue
            
        process_single_file(file_path, OUTPUT_DIR_BASE, model, processor)

    logger.success("\n🏁 所有任務執行完畢！")

if __name__ == "__main__":
    main()