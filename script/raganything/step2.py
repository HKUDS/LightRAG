import os
import json
import time
import base64
import sys
from loguru import logger
from dotenv import load_dotenv  

# 強制載入 .env 檔案
load_dotenv()

# === 1. 設定區 (Configuration) ===
# SiliconFlow API Key
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
MODEL_NAME = "thudm/glm-4.1v-9b-thinking" 

# 設定 Log 目錄
LOG_DIR = "./logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# 設定 Loguru
log_file = os.path.join(LOG_DIR, f"step2_resume_{time.strftime('%Y%m%d_%H%M%S')}.log")
logger.remove() 
logger.add(sys.stderr, level="INFO") 
logger.add(log_file, rotation="10 MB", level="DEBUG", encoding="utf-8")
logger.info(f"📝 Log 檔案已建立: {log_file}")

# ============

HAS_AI = False
ai_client = None

# 初始化 OpenAI Client
try:
    from openai import OpenAI
    if SILICONFLOW_API_KEY and "你的_SILICONFLOW_KEY" not in SILICONFLOW_API_KEY:
        ai_client = OpenAI(
            api_key=SILICONFLOW_API_KEY,
            base_url="https://api.siliconflow.cn/v1"
        )
        HAS_AI = True
        logger.info(f"✅ 已啟用 SiliconFlow AI ({MODEL_NAME})")
    else:
        logger.warning("⚠️ 未填寫 SILICONFLOW_API_KEY，將跳過 AI 描述功能")
except ImportError:
    logger.error("⚠️ 缺少 openai 套件，請執行: pip install openai")

def encode_image(image_path):
    """將圖片轉為 Base64"""
    if not os.path.exists(image_path): 
        logger.error(f"❌ 找不到圖片: {image_path}")
        return None
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def call_vision_llm(img_path, mode="table", context_text=""):
    if not HAS_AI: return None
    
    base64_image = encode_image(img_path)
    if not base64_image: return None

    try:
        context_instruction = ""
        if context_text:
            context_instruction = f"""
            \n[Context Info]:
            The text surrounding this image/table is:
            " ... {context_text} ... "
            Instruction: Use this context to infer the title, subject, time period, or data units.
            """

        if mode == "table":
            system_prompt = "You are an expert OCR engine. Transcribe the table in the image into a clean Markdown table."
            user_msg = f"{context_instruction}\nTask: Output ONLY the markdown table content. Handle merged cells. No explanations."
        else: # Image / Chart
            system_prompt = "You are a helpful assistant describing images for a RAG system."
            user_msg = f"{context_instruction}\nTask: Provide a detailed description of this image. Extract key data points, trends, and the title."

        response = ai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": user_msg},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ],
            temperature=0.1, max_tokens=4096, stream=False
        )
        content = response.choices[0].message.content.strip()
        
        if "</think>" in content:
            content = content.split("</think>")[-1].strip()
            
        return content

    except Exception as e:
        logger.error(f"❌ SiliconFlow API Error on {img_path}: {e}")
        return None

def main():
    # 路徑設定
    input_json = "./data/input/step1_output/intermediate_result.json"
    output_dir = "./data/output/step2_output_granular"
    output_path = os.path.join(output_dir, "granular_content.json")
    
    if not os.path.exists(input_json):
        input_json_fallback = "./data/input/step1_output/intermediate_result.json"
        if os.path.exists(input_json_fallback):
            input_json = input_json_fallback
            output_dir = "./data/output/step2_output_granular"
            output_path = os.path.join(output_dir, "granular_content.json")
        else:
            logger.critical(f"❌ 找不到輸入檔案: {input_json}")
            return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # === 🔄 斷點續傳邏輯 (Resume Logic) ===
    processed_blocks = []
    processed_ids = set() # 用來記錄邊啲已經做過

    if os.path.exists(output_path):
        logger.info(f"📂 發現舊有檔案: {output_path}，嘗試讀取以進行續傳...")
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                if isinstance(existing_data, list):
                    processed_blocks = existing_data
                    # 建立 ID Set (Page + Bbox String)
                    for block in processed_blocks:
                        pid = f"{block['page']}_{str(block['bbox'])}"
                        processed_ids.add(pid)
                    logger.success(f"✅ 成功載入 {len(processed_blocks)} 個已處理區塊，將會跳過這些。")
        except Exception as e:
            logger.warning(f"⚠️ 讀取舊檔失敗 ({e})，將會重新開始。")
            processed_blocks = []

    logger.info(f"🚀 [Step 2] 啟動處理... 讀取: {input_json}")
    
    with open(input_json, "r", encoding="utf-8") as f:
        content_list = json.load(f)
    
    base_dir = os.path.dirname(input_json)
    stats = {"text": 0, "table": 0, "image": 0, "ai_processed": 0, "skipped": 0}

    for idx, item in enumerate(content_list):
        item_type = item.get('type')
        page_idx = item.get('page_idx', 0)
        
        # 處理 Bbox: 轉為 List [x, y, w, h] (你原本的修正)
        raw_bbox = item.get('bbox') or item.get('rect')
        bbox = [int(b) for b in raw_bbox] if raw_bbox else None
        
        # 構造唯一 ID
        current_id = f"{page_idx + 1}_{str(bbox)}"

        # 🛑 檢查是否已處理 (Skip Check)
        if current_id in processed_ids:
            stats["skipped"] += 1
            if idx % 100 == 0: logger.info(f"⏭️ 跳過已處理區塊 (進度: {idx}/{len(content_list)})")
            continue

        # === 處理新區塊 ===
        rel_path = item.get('img_path', '')
        abs_img_path = os.path.join(base_dir, rel_path) if rel_path else None
        
        block_data = {
            "type": item_type,
            "page": page_idx + 1,
            "bbox": bbox,
            "content": "",
            "original_img": rel_path
        }

        # Context Logic
        context_text = ""
        if idx > 0 and content_list[idx-1].get('type') == 'text':
            prev_text = content_list[idx-1].get('text', '').strip()
            context_text += f"Preceding Text: {prev_text[-500:]}\n"
        if idx < len(content_list) - 1 and content_list[idx+1].get('type') == 'text':
            next_text = content_list[idx+1].get('text', '').strip()
            context_text += f"Following Text: {next_text[:200]}"

        # --- [A] Text ---
        if item_type == 'text':
            text = item.get('text', '').strip()
            if text:
                block_data["content"] = text
                processed_blocks.append(block_data)
                processed_ids.add(current_id)
                stats["text"] += 1

        # --- [B] Table ---
        elif item_type == 'table':
            logger.info(f"🔍 [Table] P{page_idx+1} 處理中...")
            content = item.get('table_body', '')
            if HAS_AI and abs_img_path:
                ai_content = call_vision_llm(abs_img_path, mode="table", context_text=context_text)
                if ai_content:
                    content = ai_content
                    stats["ai_processed"] += 1
            
            caption = "".join(item.get('table_caption', []))
            if caption: content = f"**Table Caption:** {caption}\n\n{content}"
            block_data["content"] = content
            processed_blocks.append(block_data)
            processed_ids.add(current_id)
            stats["table"] += 1

        # --- [C] Image ---
        elif item_type == 'image':
            logger.info(f"🖼️ [Image] P{page_idx+1} 處理中...")
            caption = "".join(item.get('image_caption', []))
            if HAS_AI and abs_img_path:
                ai_desc = call_vision_llm(abs_img_path, mode="caption", context_text=context_text)
                if ai_desc:
                    caption = f"{caption}\n**Image Description:** {ai_desc}".strip()
                    stats["ai_processed"] += 1
            
            block_data["content"] = caption
            processed_blocks.append(block_data)
            processed_ids.add(current_id)
            stats["image"] += 1
            
        # 💾 自動存檔 (Auto-Save): 每 5 個新 Item 就儲存一次
        newly_processed = stats["text"] + stats["table"] + stats["image"]
        if newly_processed > 0 and newly_processed % 5 == 0:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(processed_blocks, f, ensure_ascii=False, indent=2)

        if idx > 0 and idx % 50 == 0:
            logger.info(f"⏳ 進度: {idx}/{len(content_list)} (New: {newly_processed}, Skipped: {stats['skipped']})...")

    # 最後儲存
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_blocks, f, ensure_ascii=False, indent=2)

    logger.success("=" * 40)
    logger.success(f"🎉 處理完成！總區塊數: {len(processed_blocks)}")
    logger.info(f"📊 統計: New AI Calls={stats['ai_processed']} | Skipped={stats['skipped']}")
    logger.success(f"💾 檔案已儲存: {output_path}")
    logger.success("=" * 40)

if __name__ == "__main__":
    main()