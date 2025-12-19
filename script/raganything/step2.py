import os
import json
import time
import base64
import sys
import glob
from loguru import logger
from dotenv import load_dotenv  

# Force load .env file
load_dotenv()

# === 1. Configuration ===
# Azure OpenAI Settings
AZURE_OPENAI_API_KEY = os.getenv("LLM_BINDING_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("LLM_BINDING_HOST")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o") # Must support Vision

# Blacklist
SKIP_FILES = []

# Log Directory
LOG_DIR = "./logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

log_file = os.path.join(LOG_DIR, f"step2_azure_{time.strftime('%Y%m%d_%H%M%S')}.log")
logger.remove() 
logger.add(sys.stderr, level="INFO") 
logger.add(log_file, rotation="10 MB", level="DEBUG", encoding="utf-8")
logger.info(f"📝 Log file created: {log_file}")

# ============

HAS_AI = False
ai_client = None

try:
    from openai import AzureOpenAI
    
    if AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT:
        ai_client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )
        HAS_AI = True
        logger.info(f"✅ Azure OpenAI Enabled (Deployment: {AZURE_DEPLOYMENT_NAME})")
    else:
        logger.warning("⚠️ AZURE_OPENAI_API_KEY or ENDPOINT missing. AI features disabled.")
except ImportError:
    logger.error("⚠️ Missing 'openai' package. Run: pip install openai")

def encode_image(image_path):
    if not os.path.exists(image_path): 
        logger.error(f"❌ Image not found: {image_path}")
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
            model=AZURE_DEPLOYMENT_NAME,
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
        
        # Azure sometimes adds markdown code blocks, strip them
        if content.startswith("```"):
            content = content.replace("```markdown", "").replace("```", "").strip()
            
        return content

    except Exception as e:
        logger.error(f"❌ Azure OpenAI Error on {img_path}: {e}")
        return None

# 🔥 Safe Content Extractor (Handles Dolphin/Mineru differences)
def get_safe_content(item):
    """
    Try to fetch content from different keys
    """
    # Priority: text (Mineru) -> content (Dolphin) -> table_body
    candidates = [
        item.get("text"),
        item.get("content"),
        item.get("table_body"),
    ]
    
    # Handle list captions
    caption = item.get("image_caption") or item.get("table_caption")
    if isinstance(caption, list):
        caption = "".join(caption)
    if caption:
        candidates.append(caption)

    # Return first non-empty value
    for c in candidates:
        if c and str(c).strip():
            return str(c).strip()
    
    return ""

def main():
    step1_base_dir = "./data/input/step1_output"
    output_base_dir = "./data/output/step2_output_granular"

    if not os.path.exists(step1_base_dir):
        logger.error(f"❌ Step 1 output directory not found: {step1_base_dir}")
        return

    all_json_files = glob.glob(os.path.join(step1_base_dir, "*", "intermediate_result.json"))
    
    if not all_json_files:
        logger.error(f"❌ No intermediate_result.json found in {step1_base_dir}")
        return
        
    logger.info(f"📦 Found {len(all_json_files)} files to process...")

    for i, json_file_path in enumerate(all_json_files):
        file_stem = os.path.basename(os.path.dirname(json_file_path))
        
        if file_stem in SKIP_FILES:
            continue

        current_base_dir = os.path.dirname(json_file_path)
        current_output_dir = os.path.join(output_base_dir, file_stem)
        current_output_path = os.path.join(current_output_dir, "granular_content.json")
        
        if not os.path.exists(current_output_dir):
            os.makedirs(current_output_dir)

        logger.info(f"\n🚀 [{i+1}/{len(all_json_files)}] Processing: {file_stem}")

        processed_blocks = []
        existing_map = {}
        
        # Resume logic
        if os.path.exists(current_output_path):
            try:
                with open(current_output_path, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                    if isinstance(existing_data, list):
                        for block in existing_data:
                            if "unique_id" in block:
                                existing_map[block["unique_id"]] = block
            except: pass

        with open(json_file_path, "r", encoding="utf-8") as f:
            content_list = json.load(f)

        stats = {"text": 0, "table": 0, "image": 0, "ai_processed": 0, "skipped": 0}

        for idx, item in enumerate(content_list):
            item_type = item.get('type', 'text') # Default to text
            page_idx = item.get('page_idx', 0)
            
            raw_bbox = item.get('bbox') or item.get('rect')
            bbox = [int(b) for b in raw_bbox] if raw_bbox else None
            current_id = f"{file_stem}_{page_idx}_{str(bbox)}"

            # === Skip Logic ===
            old_block = existing_map.get(current_id)
            if old_block:
                old_content = old_block.get("content", "").strip()
                if len(old_content) > 5:
                    processed_blocks.append(old_block)
                    stats["skipped"] += 1
                    continue
            # ==================

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
                "unique_id": current_id,
                "label": item.get('label', '')
            }

            # Prepare Context
            context_text = ""
            if idx > 0:
                prev_text = get_safe_content(content_list[idx-1])
                if prev_text: context_text += f"Pre: {prev_text[-200:]}\n"

            # === Core Logic ===
            
            # 1. Table
            if item_type in ['table', 'tabular']:
                content = get_safe_content(item)
                if HAS_AI and abs_img_path and os.path.exists(abs_img_path):
                    logger.info(f"   🔍 Azure AI Table: P{page_idx+1}")
                    ai_content = call_vision_llm(abs_img_path, mode="table", context_text=context_text)
                    if ai_content:
                        content = ai_content
                        stats["ai_processed"] += 1
                
                block_data["content"] = content
                processed_blocks.append(block_data)
                stats["table"] += 1

            # 2. Image
            elif item_type in ['image', 'figure', 'fig']:
                content = get_safe_content(item)
                if HAS_AI and abs_img_path and os.path.exists(abs_img_path):
                    logger.info(f"   🖼️ Azure AI Caption: P{page_idx+1}")
                    ai_desc = call_vision_llm(abs_img_path, mode="caption", context_text=context_text)
                    if ai_desc:
                        content = f"{content}\n**Image Description:** {ai_desc}".strip()
                        stats["ai_processed"] += 1
                
                block_data["content"] = content
                processed_blocks.append(block_data)
                stats["image"] += 1

            # 3. Text (Universal Fallback)
            else:
                text_content = get_safe_content(item)
                
                if text_content:
                    block_data["content"] = text_content
                    if block_data.get('label') in ['title', 'section_header', 'header']:
                        block_data["content"] = f"# {text_content}" 
                    
                    processed_blocks.append(block_data)
                    stats["text"] += 1

            # Auto-save
            if (stats["text"] + stats["table"] + stats["image"]) % 20 == 0:
                 with open(current_output_path, "w", encoding="utf-8") as f:
                    json.dump(processed_blocks, f, ensure_ascii=False, indent=2)

        # Final Save
        with open(current_output_path, "w", encoding="utf-8") as f:
            json.dump(processed_blocks, f, ensure_ascii=False, indent=2)

        logger.success(f"✅ Completed {file_stem}")
        logger.info(f"   📊 Stats: Text={stats['text']} | Table={stats['table']} | Image={stats['image']} | Skipped={stats['skipped']}")

    logger.success("=" * 40)
    logger.success(f"🎉 All files processed!")

if __name__ == "__main__":
    main()