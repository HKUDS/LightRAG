import os
import json
import time
import base64
import sys
import glob
from loguru import logger
from dotenv import load_dotenv  

# 強制載入 .env 檔案
load_dotenv()

# === 1. 設定區 (Configuration) ===
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
MODEL_NAME = "thudm/glm-4.1v-9b-thinking" 

# 🔥 [新增] 黑名單：填入不想處理的檔案名稱 (資料夾名稱/file_stem)
# 例如: ["SFC", "Another_Doc", "Old_Report"]
SKIP_FILES = [
    "SFC", 
    "Example_Doc_To_Skip"
]

# 設定 Log 目錄
LOG_DIR = "./logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# 設定 Loguru
log_file = os.path.join(LOG_DIR, f"step2_split_run_{time.strftime('%Y%m%d_%H%M%S')}.log")
logger.remove() 
logger.add(sys.stderr, level="INFO") 
logger.add(log_file, rotation="10 MB", level="DEBUG", encoding="utf-8")
logger.info(f"📝 Log 檔案已建立: {log_file}")

# ============

HAS_AI = False
ai_client = None

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
    logger.error("⚠️ 缺少 openai 套件")

def encode_image(image_path):
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
        else: 
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
    # === 設定路徑 ===
    step1_base_dir = "./data/input/step1_output"
    output_base_dir = "./data/output/step2_output_granular"

    if not os.path.exists(step1_base_dir):
        logger.error(f"❌ 找不到 Step 1 輸出目錄: {step1_base_dir}")
        return

    # 掃描所有 input json
    all_json_files = glob.glob(os.path.join(step1_base_dir, "*", "intermediate_result.json"))
    
    if not all_json_files:
        logger.error(f"❌ 在 {step1_base_dir} 找不到任何 intermediate_result.json")
        return
        
    logger.info(f"📦 發現 {len(all_json_files)} 個檔案待處理...")

    # === 逐個檔案處理 (Per-File Loop) ===
    for i, json_file_path in enumerate(all_json_files):
        # 1. 準備路徑和資料夾
        file_stem = os.path.basename(os.path.dirname(json_file_path))
        
        # === 🚫 Blacklist Check (新增檢查邏輯) ===
        if file_stem in SKIP_FILES:
            logger.warning(f"🚫 [{i+1}/{len(all_json_files)}] 跳過黑名單檔案: {file_stem}")
            continue
        # ========================================

        current_base_dir = os.path.dirname(json_file_path)
        
        # 建立專屬輸出目錄: output/doc_name/granular_content.json
        current_output_dir = os.path.join(output_base_dir, file_stem)
        current_output_path = os.path.join(current_output_dir, "granular_content.json")
        
        if not os.path.exists(current_output_dir):
            os.makedirs(current_output_dir)

        logger.info(f"\n🚀 [{i+1}/{len(all_json_files)}] 正在處理: {file_stem}")
        logger.info(f"   📂 輸出位置: {current_output_path}")

        # 2. 針對「這個檔案」的斷點續傳邏輯
        processed_blocks = [] # 這次執行要產生的完整列表
        existing_map = {}     # 用來快速查找舊資料
        
        if os.path.exists(current_output_path):
            try:
                with open(current_output_path, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                    if isinstance(existing_data, list):
                        # 建立 ID -> Block 的對照表
                        for block in existing_data:
                            if "unique_id" in block:
                                existing_map[block["unique_id"]] = block
                        logger.info(f"   🔄 載入舊進度: {len(existing_map)} 筆資料，將檢查內容完整性...")
            except Exception as e:
                logger.warning(f"   ⚠️ 讀取舊檔失敗，將重新開始: {e}")
                existing_map = {}

        # 3. 讀取輸入與處理
        with open(json_file_path, "r", encoding="utf-8") as f:
            content_list = json.load(f)

        stats = {"text": 0, "table": 0, "image": 0, "ai_processed": 0, "skipped": 0}

        for idx, item in enumerate(content_list):
            item_type = item.get('type')
            page_idx = item.get('page_idx', 0)
            
            raw_bbox = item.get('bbox') or item.get('rect')
            bbox = [int(b) for b in raw_bbox] if raw_bbox else None
            
            # 生成 ID
            current_id = f"{file_stem}_{page_idx}_{str(bbox)}"

            # === 💡 Skip Logic ===
            old_block = existing_map.get(current_id)
            should_skip = False

            if old_block:
                old_content = old_block.get("content", "").strip()
                
                # A: Text -> Skip
                if item_type == 'text':
                    should_skip = True
                
                # B: Image/Table -> Check Content Length
                elif item_type in ['table', 'image']:
                    if len(old_content) > 5:
                        should_skip = True
                    else:
                        logger.info(f"   ⚠️ 發現舊資料但內容為空，將重新執行 AI: P{page_idx+1} {item_type}")

            if should_skip:
                processed_blocks.append(old_block) # 直接使用舊的區塊
                stats["skipped"] += 1
                continue
            # =====================

            # 如果不 Skip，就準備進行處理
            
            # 準備路徑
            rel_path = item.get('img_path', '')
            abs_img_path = None
            if rel_path:
                abs_img_path = os.path.join(current_base_dir, rel_path)

            block_data = {
                "type": item_type,
                "page": page_idx + 1,
                "bbox": bbox,
                "content": "",
                "original_img": rel_path,
                "source_file": file_stem,
                "unique_id": current_id 
            }

            # Context
            context_text = ""
            if idx > 0 and content_list[idx-1].get('type') == 'text':
                context_text += f"Pre: {content_list[idx-1].get('text', '')[-200:]}\n"

            # --- Process ---
            if item_type == 'text':
                text = item.get('text', '').strip()
                if text:
                    block_data["content"] = text
                    processed_blocks.append(block_data)
                    stats["text"] += 1

            elif item_type == 'table':
                logger.info(f"   🔍 [Table] P{page_idx+1}")
                content = item.get('table_body', '')
                if HAS_AI and abs_img_path and os.path.exists(abs_img_path):
                    ai_content = call_vision_llm(abs_img_path, mode="table", context_text=context_text)
                    if ai_content:
                        content = ai_content
                        stats["ai_processed"] += 1
                
                caption = "".join(item.get('table_caption', []))
                if caption: content = f"**Table Caption:** {caption}\n\n{content}"
                block_data["content"] = content
                processed_blocks.append(block_data)
                stats["table"] += 1

            elif item_type == 'image':
                logger.info(f"   🖼️ [Image] P{page_idx+1}")
                caption = "".join(item.get('image_caption', []))
                if HAS_AI and abs_img_path and os.path.exists(abs_img_path):
                    ai_desc = call_vision_llm(abs_img_path, mode="caption", context_text=context_text)
                    if ai_desc:
                        caption = f"{caption}\n**Image Description:** {ai_desc}".strip()
                        stats["ai_processed"] += 1
                
                block_data["content"] = caption
                processed_blocks.append(block_data)
                stats["image"] += 1

            # 💾 Per-File Auto-Save
            new_processed_count = stats["text"] + stats["table"] + stats["image"]
            if new_processed_count > 0 and new_processed_count % 5 == 0:
                 with open(current_output_path, "w", encoding="utf-8") as f:
                    json.dump(processed_blocks, f, ensure_ascii=False, indent=2)

        # 4. 完成該檔案，最終儲存
        with open(current_output_path, "w", encoding="utf-8") as f:
            json.dump(processed_blocks, f, ensure_ascii=False, indent=2)

        logger.success(f"✅ 文件 {file_stem} 處理完成")
        logger.info(f"   📊 統計: New={len(processed_blocks)-stats['skipped']} | Skipped (Reused)={stats['skipped']} | AI Calls={stats['ai_processed']}")

    logger.success("=" * 40)
    logger.success(f"🎉 所有檔案處理完畢！")

if __name__ == "__main__":
    main()