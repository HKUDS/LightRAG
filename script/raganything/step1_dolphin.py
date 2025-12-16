import sys
import os
import json
import glob
import time
import math
import gc
import io
import re
import numpy as np
import torch
import pymupdf  # 無需 Poppler
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from loguru import logger

# ==========================================
# 🔥 [使用者設定區]
# ==========================================

INPUT_DIR = "./data/input/__enqueued__"
OUTPUT_DIR_BASE = "./data/input/step1_output"
MODEL_ID = "ByteDance/Dolphin-v2"

# 渲染 PDF 的畫質 (300 DPI 為佳，保證小字清晰)
RENDER_DPI = 300 

# 存檔頻率 (每幾頁存一次)
SAVE_INTERVAL = 1

# ==========================================
# 🛠️ 核心工具函式 (Dolphin 官方邏輯)
# ==========================================

def smart_resize(height, width, factor=28, min_pixels=784, max_pixels=2560000):
    """Dolphin 官方的圖片縮放邏輯 (用於坐標換算)"""
    if max(height, width) / min(height, width) > 200:
        resize_factor = max(height, width) // min_pixels
        if resize_factor > 1:
            height = height // resize_factor
            width = width // resize_factor
            return height, width
            
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
        
    return h_bar, w_bar

def resize_img(image, max_size=1600, min_size=28):
    """Dolphin 預處理縮放 (用於推論輸入)"""
    width, height = image.size
    if max(width, height) < max_size and min(width, height) >= 28:
        return image
    
    if max(width, height) > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        image = image.resize((new_width, new_height))
        width, height = image.size
    
    if min(width, height) < 28:
        if width < height:
            new_width = min_size
            new_height = int(height * (min_size / width))
        else:
            new_height = min_size
            new_width = int(width * (min_size / height))
        image = image.resize((new_width, new_height))

    return image

def process_coordinates(coords, pil_image):
    """將模型輸出的歸一化坐標轉回原圖坐標"""
    original_w, original_h = pil_image.size
    resized_pil = resize_img(pil_image)
    resized_image = np.array(resized_pil)
    resized_h, resized_w = resized_image.shape[:2]
    resized_h, resized_w = smart_resize(resized_h, resized_w, factor=28, min_pixels=784, max_pixels=2560000)

    w_ratio, h_ratio = original_w / resized_w, original_h / resized_h
    x1 = int(coords[0] * w_ratio)
    y1 = int(coords[1] * h_ratio)
    x2 = int(coords[2] * w_ratio)
    y2 = int(coords[3] * h_ratio)

    x1 = max(0, min(x1, original_w - 1))
    y1 = max(0, min(y1, original_h - 1))
    x2 = max(x1 + 1, min(x2, original_w))
    y2 = max(y1 + 1, min(y2, original_h))
    return x1, y1, x2, y2

def extract_labels_from_string(text):
    """從輸出字串提取標籤"""
    all_matches = re.findall(r'\[([^\]]+)\]', text)
    labels = []
    for match in all_matches:
        if not re.match(r'^\d+,\d+,\d+,\d+$', match):
            labels.append(match)
    return labels

def parse_layout_string(bbox_str):
    """解析模型輸出的 Layout 字串"""
    parsed_results = []
    if not bbox_str: return []
    
    segments = bbox_str.split('[PAIR_SEP]')
    new_segments = []
    for seg in segments:
        new_segments.extend(seg.split('[RELATION_SEP]'))
    segments = new_segments
    
    for segment in segments:
        segment = segment.strip()
        if not segment: continue
        
        coord_pattern = r'\[(\d*\.?\d+),(\d*\.?\d+),(\d*\.?\d+),(\d*\.?\d+)\]'
        coord_match = re.search(coord_pattern, segment)
        label_matches = extract_labels_from_string(segment)
        
        if coord_match and label_matches:
            coords = [float(coord_match.group(i)) for i in range(1, 5)]
            label = label_matches[0].strip()
            parsed_results.append((coords, label, label_matches[1:]))
            
    return parsed_results

# ==========================================
# ⚙️ 核心處理邏輯
# ==========================================

# Setup Logging
logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>", level="INFO")

def load_model():
    logger.info("="*60)
    
    # 1. 自動偵測裝置
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"📥 正在載入模型: {MODEL_ID} (GPU 4-bit Mode)...")
    else:
        device = "cpu"
        logger.info(f"📥 正在載入模型: {MODEL_ID} (CPU High-Quality Mode)...")
        logger.warning("⚠️ 注意：CPU 模式推論速度較慢，且需要約 12GB+ RAM。")

    try:
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        
        if device == "cuda":
            # === GPU 模式：使用 4-bit 量化省顯存 ===
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                MODEL_ID,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            # === CPU 模式：使用 Float32 確保相容性 ===
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float32, 
                device_map="cpu",
                trust_remote_code=True
            )

        logger.success(f"✅ 模型載入完成！(Device: {model.device})")
        return model, processor

    except Exception as e:
        logger.critical(f"❌ 模型載入失敗: {e}")
        sys.exit(1)

def run_inference(model, processor, image, prompt):
    """通用的推論函式 (自動處理 Device)"""
    # 預處理圖片
    image = resize_img(image)
    
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt}
        ]
    }]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text], 
        images=image_inputs, 
        videos=video_inputs, 
        padding=True, 
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False
        )
    
    gen_ids_trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(gen_ids_trimmed, skip_special_tokens=True)[0]
    
    # Clean memory
    del inputs, generated_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return output_text

def process_single_file(input_path, output_base, model, processor):
    filename = os.path.basename(input_path)
    file_stem = os.path.splitext(filename)[0]
    
    # 建立目錄
    current_out_dir = os.path.join(output_base, file_stem)
    if not os.path.exists(current_out_dir): os.makedirs(current_out_dir)
    final_json = os.path.join(current_out_dir, "intermediate_result.json")
    
    # 分別存放 crop 出來的圖片
    images_dir = os.path.join(current_out_dir, "images")
    if not os.path.exists(images_dir): os.makedirs(images_dir)

    logger.info("-" * 40)
    logger.info(f"🚀 處理檔案: {filename}")
    
    try:
        doc = pymupdf.open(input_path)
        total_pages = len(doc)
    except Exception as e:
        logger.error(f"❌ 無法開啟 PDF: {e}")
        return

    parsed_data = []

    for i in range(total_pages):
        logger.info(f"   📄 Page {i+1}/{total_pages} 分析 Layout 中...")
        
        # 1. Render Page (High Res)
        try:
            page = doc[i]
            pix = page.get_pixmap(dpi=RENDER_DPI)
            img_data = pix.tobytes("png")
            pil_image = Image.open(io.BytesIO(img_data)).convert("RGB")
        except Exception as e:
            logger.error(f"❌ Page {i+1} 渲染失敗: {e}")
            continue

        # 2. Stage 1: Layout Parsing
        try:
            layout_text = run_inference(model, processor, pil_image, "Parse the reading order of this document.")
            layout_items = parse_layout_string(layout_text)
            
            # 如果 Layout 解析失敗或回傳空，退回到 "distorted_page" (整頁當一個 Text 處理)
            if not layout_items:
                logger.warning(f"      ⚠️ 無法解析 Layout，退回整頁 OCR 模式")
                layout_items = [([0,0,0,0], 'distorted_page', [])] # 假 Layout
        except Exception as e:
            logger.error(f"❌ Layout 推理錯誤: {e}")
            continue

        logger.info(f"      🔍 偵測到 {len(layout_items)} 個元素，開始提取...")
        
        page_reading_order = 0
        
        # 3. Stage 2: Element Extraction & Routing
        for bbox, label, tags in layout_items:
            # 處理坐標
            if label == 'distorted_page':
                x1, y1, x2, y2 = 0, 0, pil_image.width, pil_image.height
                pil_crop = pil_image
            else:
                x1, y1, x2, y2 = process_coordinates(bbox, pil_image)
                # 安全邊界檢查
                if x2 <= x1 or y2 <= y1: continue
                pil_crop = pil_image.crop((x1, y1, x2, y2))
            
            # 忽略過小的碎片
            if pil_crop.width < 10 or pil_crop.height < 10: continue
            
            element_data = {
                "page_idx": i,
                "bbox": [x1, y1, x2, y2],
                "reading_order": page_reading_order,
                "label": label
            }
            
            # === 🔥 關鍵路由 (Routing Logic) ===
            
            # Case A: 圖片 (Image/Figure) -> 只存圖，不 OCR
            if label == "fig":
                img_filename = f"p{i+1}_{page_reading_order:03d}_fig.jpg"
                save_path = os.path.join(images_dir, img_filename)
                pil_crop.save(save_path)
                
                element_data["type"] = "image"
                element_data["content"] = f"![Figure]({img_filename})" # Markdown 格式
                element_data["img_path"] = f"images/{img_filename}"
                logger.debug(f"         🖼️ 圖片 (Saved): {img_filename}")

            # Case B: 表格 (Table) -> 使用表格專用 Prompt
            elif label == "tab":
                element_data["type"] = "table"
                md_table = run_inference(model, processor, pil_crop, "Parse the table in the image.")
                element_data["content"] = md_table
                logger.debug(f"         📊 表格 (Parsed)")
                
                # 順便存個表格截圖備份
                tab_filename = f"p{i+1}_{page_reading_order:03d}_tab.jpg"
                pil_crop.save(os.path.join(images_dir, tab_filename))
                element_data["img_path"] = f"images/{tab_filename}"

            # Case C: 文字/標題 (Text/Title) -> 使用文字 Prompt
            else:
                # 包含: text, section_header, title, list, code 等
                element_data["type"] = "text"
                ocr_text = run_inference(model, processor, pil_crop, "Read text in the image.")
                element_data["content"] = ocr_text
                # 文字通常不存圖，省空間
                element_data["img_path"] = ""

            parsed_data.append(element_data)
            page_reading_order += 1
            
        # 釋放記憶體
        del pil_image
        gc.collect()

        # Incremental Save
        if (i + 1) % SAVE_INTERVAL == 0:
            with open(final_json, "w", encoding="utf-8") as f:
                json.dump(parsed_data, f, ensure_ascii=False, indent=2)

    # Final Save
    with open(final_json, "w", encoding="utf-8") as f:
        json.dump(parsed_data, f, ensure_ascii=False, indent=2)
    
    logger.success(f"🎉 檔案 {filename} 完成！")

def main():
    if not os.path.exists(INPUT_DIR):
        logger.error(f"❌ 找不到輸入目錄: {INPUT_DIR}")
        return
    if not os.path.exists(OUTPUT_DIR_BASE):
        os.makedirs(OUTPUT_DIR_BASE)

    # 檢查 PyMuPDF
    try:
        import pymupdf
    except ImportError:
        logger.error("❌ 缺少 PyMuPDF！請執行: `uv pip install pymupdf`")
        return

    model, processor = load_model()

    all_files = glob.glob(os.path.join(INPUT_DIR, "*"))
    files = [f for f in all_files if os.path.isfile(f) and not os.path.basename(f).startswith(".")]
    
    logger.info(f"📦 發現 {len(files)} 個檔案...")

    for file_path in files:
        if not file_path.lower().endswith(".pdf"):
            continue
        process_single_file(file_path, OUTPUT_DIR_BASE, model, processor)

if __name__ == "__main__":
    main()