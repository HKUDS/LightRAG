import sys
import os
import json
import glob
import time
import math
import gc
import io
import torch
import pymupdf  # 🔥 官方使用的 PDF 引擎 (無需 Poppler)
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
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

# 3. 畫質與效能設定 (RTX 4060 8GB 專用)
# 設定讀取時的 DPI (300 DPI = 高畫質，PyMuPDF 速度很快，這沒問題)
RENDER_DPI = 300 

# Token 限制 (保護 VRAM 不爆)
# 14000 tokens ≈ 274萬像素 (足夠看清絕大多數 A4 文件)
MAX_VISUAL_TOKENS = 14000 

# 存檔頻率 (每幾頁存一次)
SAVE_INTERVAL = 5 

# ==========================================

# === 0. 環境準備 ===
# 設定 Logging
LOG_DIR = "./logs"
if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
log_file = os.path.join(LOG_DIR, f"step1_official_gpu_{time.strftime('%Y%m%d_%H%M%S')}.log")

logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>", level="INFO")
logger.add(log_file, rotation="10 MB", level="DEBUG", encoding="utf-8")
logger.info(f"📝 Log 檔案已建立: {log_file}")

# === 1. 核心函式 ===

def resize_to_token_limit(image, max_tokens=14000):
    """
    智慧縮圖：確保圖片不會產生過多 Token 導致 OOM
    """
    w, h = image.size
    total_pixels = w * h
    current_tokens = total_pixels / 196 # Qwen2.5-VL: 1 token ≈ 14x14 pixels
    
    if current_tokens > max_tokens:
        scale = math.sqrt(max_tokens / current_tokens)
        new_w = int(w * scale)
        new_h = int(h * scale)
        logger.debug(f"📉 圖片壓縮: {w}x{h} -> {new_w}x{new_h} (Tokens: {int(current_tokens)} -> ~{max_tokens})")
        return image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    return image

def render_pdf_page(doc, page_index, dpi=200):
    """
    使用 PyMuPDF 渲染單頁 PDF 為 PIL Image
    """
    try:
        page = doc[page_index]
        # 計算縮放比例 (72 dpi 是 PDF 標準解析度)
        zoom = dpi / 72.0
        mat = pymupdf.Matrix(zoom, zoom)
        
        # 獲取像素圖
        pix = page.get_pixmap(matrix=mat, alpha=False)
        
        # 轉換為 PIL Image
        img_data = pix.tobytes("png")
        return Image.open(io.BytesIO(img_data))
    except Exception as e:
        logger.error(f"❌ PyMuPDF 渲染失敗 (Page {page_index}): {e}")
        return None

def load_model():
    """
    載入模型 (4-bit 量化 + GPU 加速)
    """
    logger.info("="*60)
    logger.info(f"📥 正在載入模型: {MODEL_ID} (4-bit Official Mode)...")
    
    if not torch.cuda.is_available():
        logger.critical("❌ 檢測不到 GPU！請確認 CUDA 是否安裝正確。")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    logger.info(f"🎮 GPU 偵測: {gpu_name}")

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

    # 1. 開啟 PDF (PyMuPDF)
    try:
        doc = pymupdf.open(input_path)
        total_pages = len(doc)
        logger.info(f"📄 PDF 總頁數: {total_pages}")
    except Exception as e:
        logger.error(f"❌ 無法開啟 PDF: {e}")
        return False

    parsed_results = []
    
    # 2. 逐頁處理
    for i in range(total_pages):
        page_start = time.time()
        logger.info(f"   🔄 正在處理 Page {i+1}/{total_pages}...")

        try:
            # 🔥 渲染：使用官方庫 PyMuPDF
            image = render_pdf_page(doc, i, dpi=RENDER_DPI)
            
            if image is None:
                logger.warning(f"⚠️ Page {i+1} 渲染失敗，跳過。")
                continue
            
            # 儲存原圖 (Debug 用)
            img_filename = f"page_{i}.jpg"
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

            # 準備輸入 Tensor
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = processor(
                text=[text], 
                images=image_inputs, 
                videos=video_inputs, 
                padding=True, 
                return_tensors="pt"
            ).to(model.device)

            # 推理 (GPU)
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    do_sample=False
                )

            # 解碼
            gen_ids_trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
            md_text = processor.batch_decode(gen_ids_trimmed, skip_special_tokens=True)[0]

            # 收集結果
            parsed_results.append({
                "type": "text",
                "text": md_text,
                "page_idx": i,
                "img_path": f"images/{img_filename}",
                "bbox": [0, 0, image.width, image.height]
            })

            dur = time.time() - page_start
            vram_usage = torch.cuda.memory_allocated()/1e9
            logger.info(f"     ✅ 完成 ({dur:.2f}s) | VRAM: {vram_usage:.2f}GB")

            # 🔥 定期存檔
            if (i + 1) % SAVE_INTERVAL == 0 or (i + 1) == total_pages:
                with open(final_json, "w", encoding="utf-8") as f:
                    json.dump(parsed_results, f, ensure_ascii=False, indent=2)
                logger.debug(f"💾 進度已儲存 (Page {i+1})")

        except torch.cuda.OutOfMemoryError:
            logger.error(f"❌ Page {i+1} OOM (顯存不足)！跳過此頁。")
            torch.cuda.empty_cache()
        except Exception as e:
            logger.exception(f"❌ Page {i+1} 發生錯誤: {e}")
        finally:
            # 清理顯存
            if 'inputs' in locals(): del inputs
            if 'generated_ids' in locals(): del generated_ids
            if 'image' in locals(): del image
            torch.cuda.empty_cache()
            gc.collect()

    doc.close()
    logger.success(f"🎉 檔案處理完成！總耗時: {time.time() - start_time:.2f}s")
    return True

def main():
    if not os.path.exists(INPUT_DIR):
        logger.error(f"❌ 找不到輸入目錄: {INPUT_DIR}")
        return
    if not os.path.exists(OUTPUT_DIR_BASE):
        os.makedirs(OUTPUT_DIR_BASE)

    # 檢查是否安裝了 PyMuPDF
    try:
        import pymupdf
    except ImportError:
        logger.error("❌ 缺少 PyMuPDF！請執行: `uv pip install pymupdf`")
        return

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