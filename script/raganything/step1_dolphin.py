# @title 3. 執行 AI 解析 (Dolphin-v2 | 限制 16k Tokens)
import sys
import os
import json
import glob
import time
import math
import torch
from pdf2image import convert_from_path
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from loguru import logger

# === 設定 ===
INPUT_DIR = "./data/input/__enqueued__"
OUTPUT_DIR_BASE = "./data/input/step1_output"
MODEL_ID = "ByteDance/Dolphin-v2"

# === 🔥 關鍵設定：Token 限制 ===
# 16000 Tokens 約等於 313萬像素 (例如 1500x2000)
# 這能有效防止 OOM，同時保持比 150 DPI 更好的畫質
MAX_VISUAL_TOKENS = 16000 

# === Logger ===
logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>", level="INFO")

def resize_to_token_limit(image, max_tokens=16000):
    """
    根據目標 Token 數量自動縮放圖片
    原理: Qwen2.5-VL 使用 14x14 patch，1 token ≈ 196 pixels
    """
    w, h = image.size
    total_pixels = w * h
    
    # 計算目前的視覺 Token 估算值
    current_tokens = total_pixels / (14 * 14)
    
    # 如果超過限制，進行縮放
    if current_tokens > max_tokens:
        scale = math.sqrt(max_tokens / current_tokens)
        new_w = int(w * scale)
        new_h = int(h * scale)
        logger.info(f"📉 壓縮圖片: {w}x{h} ({int(current_tokens)} tokens) -> {new_w}x{new_h} (~{max_tokens} tokens)")
        return image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    return image

def load_model():
    logger.info(f"📥 正在載入模型: {MODEL_ID} (4-bit Mode)...")
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
    
    current_out_dir = os.path.join(output_base, file_stem)
    if not os.path.exists(current_out_dir): os.makedirs(current_out_dir)
    final_json = os.path.join(current_out_dir, "intermediate_result.json")

    logger.info(f"🚀 處理: {filename}")
    
    try:
        # 先用較高 DPI (200) 讀取，然後用程式碼精準壓縮到 16k tokens
        # 這樣比直接設低 DPI (如 150) 畫質更好，因為是 Downsampling
        images = convert_from_path(input_path, dpi=200)
    except Exception as e:
        logger.error(f"❌ 轉圖失敗: {e}")
        return

    parsed_results = []
    
    for i, image in enumerate(images):
        # 🔥 套用 Token 限制
        image = resize_to_token_limit(image, max_tokens=MAX_VISUAL_TOKENS)
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Extract all text and layout from this document into Markdown format."}
            ]
        }]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        ).to(model.device)

        try:
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    do_sample=False
                )
            
            gen_ids_trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
            md_text = processor.batch_decode(gen_ids_trimmed, skip_special_tokens=True)[0]
            
            parsed_results.append({
                "type": "text", "text": md_text, "page_idx": i,
                "img_path": "", "bbox": [0,0,image.width, image.height]
            })
            
            logger.info(f"   ↳ Page {i+1} 完成")
            
        except torch.cuda.OutOfMemoryError:
            logger.error(f"❌ Page {i+1} OOM (爆顯存)！嘗試清理快取...")
            torch.cuda.empty_cache()
            continue
            
        del inputs, generated_ids, image
        torch.cuda.empty_cache()

    with open(final_json, "w", encoding="utf-8") as f:
        json.dump(parsed_results, f, ensure_ascii=False, indent=2)
    logger.success(f"💾 儲存成功: {final_json}")

# === 主流程 ===
files_list = glob.glob(os.path.join(INPUT_DIR, "*"))
if not files_list:
    logger.warning("📂 沒有檔案！請先執行 Step 2 上傳檔案。")
else:
    model, processor = load_model()
    for f in files_list:
        process_single_file(f, OUTPUT_DIR_BASE, model, processor)
    logger.info("🎉 全部完成！請執行 Step 4 下載結果。")