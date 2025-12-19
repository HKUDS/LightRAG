import os
import re
import time
import base64
import sys
import glob
from loguru import logger
from dotenv import load_dotenv
from openai import AzureOpenAI

# 強制載入 .env
load_dotenv()

# === 1. 設定區 (Configuration) ===
# Azure OpenAI 設定
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o") 

# 輸入與輸出路徑
INPUT_BASE_DIR = "./data/output/step1_vlm_output"
OUTPUT_SUFFIX = "_enriched" # 處理後的檔案會加上這個後綴，例如 doc_enriched.md

# Log 設定
LOG_DIR = "./logs"
if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add(os.path.join(LOG_DIR, f"markdown_enrich_{time.strftime('%Y%m%d_%H%M%S')}.log"), rotation="10 MB", encoding="utf-8")

# ============

HAS_AI = False
ai_client = None

try:
    if AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT:
        ai_client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )
        HAS_AI = True
        logger.info(f"✅ Azure OpenAI 已啟用 (Deployment: {AZURE_DEPLOYMENT_NAME})")
    else:
        logger.warning("⚠️ 未設定 Azure API Key 或 Endpoint，AI 功能將跳過")
except ImportError:
    logger.error("⚠️ 缺少 openai 套件，請執行: pip install openai")

def encode_image(image_path):
    if not os.path.exists(image_path):
        return None
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def generate_image_description(img_path, context_text=""):
    """呼叫 Azure OpenAI 生成圖片描述"""
    if not HAS_AI: return None
    
    base64_image = encode_image(img_path)
    if not base64_image: 
        logger.warning(f"   ⚠️ 找不到圖片或無法讀取: {img_path}")
        return None

    try:
        system_prompt = "You are a helpful assistant assisting in document digitization. Your task is to provide a concise but descriptive summary of the image provided."
        user_msg = "Describe this image in detail. If it is a chart, summarize the key trends. If it is a diagram, explain its components. Output plain text only."
        
        # 如果有上下文，可以加強 Prompt
        if context_text:
            user_msg += f"\n\nContext surrounding this image:\n{context_text[:500]}"

        response = ai_client.chat.completions.create(
            model=AZURE_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": user_msg},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ],
            temperature=0.3, 
            max_tokens=1024
        )
        content = response.choices[0].message.content.strip()
        
        # 清理可能出現的 markdown code block
        if content.startswith("```"):
            content = content.replace("```markdown", "").replace("```", "").strip()
            
        return content

    except Exception as e:
        logger.error(f"❌ Azure API Error on {os.path.basename(img_path)}: {e}")
        return None

def process_single_markdown(md_file_path):
    """處理單個 Markdown 檔案"""
    file_dir = os.path.dirname(md_file_path)
    file_name = os.path.basename(md_file_path)
    file_stem = os.path.splitext(file_name)[0]
    
    # 定義輸出檔案路徑
    output_path = os.path.join(file_dir, f"{file_stem}{OUTPUT_SUFFIX}.md")
    
    # 如果已經處理過，可以選擇跳過 (這裡設為 True 則跳過)
    if os.path.exists(output_path):
        logger.info(f"⏭️ 檔案已存在，跳過: {output_path}")
        return

    logger.info(f"📖 讀取檔案: {md_file_path}")
    
    with open(md_file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Regex 尋找 Markdown 圖片語法: ![alt text](image_path)
    # capture group 1: alt text
    # capture group 2: image path
    img_pattern = re.compile(r'!\[(.*?)\]\((.*?)\)')
    
    # 找出所有圖片連結
    matches = list(img_pattern.finditer(content))
    
    if not matches:
        logger.info("   ℹ️ 此文件沒有發現圖片")
        return

    logger.info(f"   🔍 發現 {len(matches)} 張圖片，開始生成描述...")
    
    # 我們使用「替換」的方式，為了避免替換後 offset 跑掉，我們從後面往前處理，或者重建字串
    # 這裡使用重建字串的方式比較安全
    
    new_content = content
    # 統計用
    processed_count = 0
    
    # 為了避免多次替換導致混亂，我們建立一個替換清單
    replacements = {}

    for match in matches:
        alt_text = match.group(1)
        rel_img_path = match.group(2)
        full_match_str = match.group(0)
        
        # 組合圖片的絕對路徑 (Mineru 的圖片路徑通常是相對的)
        abs_img_path = os.path.join(file_dir, rel_img_path)
        
        # 取得圖片周圍的上下文 (前後 200 字)
        start_idx = match.start()
        context_text = content[max(0, start_idx-200) : min(len(content), start_idx+200)]

        # 呼叫 AI 生成描述
        logger.info(f"   🖼️ 正在分析圖片: {rel_img_path}")
        description = generate_image_description(abs_img_path, context_text)
        
        if description:
            # 構建新的 Markdown 區塊
            # 格式:
            # ![alt](path)
            # > **AI Description:** ...
            new_block = f"{full_match_str}\n\n> **AI Image Description:** {description}\n"
            replacements[full_match_str] = new_block
            processed_count += 1
            
            # 休息一下避免 Rate Limit
            time.sleep(1) 
        else:
            replacements[full_match_str] = full_match_str # 沒描述就保持原樣

    # 執行替換 (一次性替換所有圖片)
    # 注意：如果有多個相同的圖片標籤，這裡會全部替換
    for old_str, new_str in replacements.items():
        new_content = new_content.replace(old_str, new_str)

    # 寫入新檔案
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    logger.success(f"✅ 完成！已處理 {processed_count}/{len(matches)} 張圖片")
    logger.info(f"   💾 儲存至: {output_path}")

def main():
    # 1. 搜尋所有 Markdown 檔案
    # 路徑模式: data/output/step1_vlm_output/<folder>/vlm/<file>.md
    # 使用遞迴搜尋
    search_pattern = os.path.join(INPUT_BASE_DIR, "**", "*.md")
    all_md_files = glob.glob(search_pattern, recursive=True)
    
    # 過濾掉已經是 _enriched 的檔案，避免重複處理
    target_files = [f for f in all_md_files if OUTPUT_SUFFIX not in f and "README" not in f]

    if not target_files:
        logger.error(f"❌ 在 {INPUT_BASE_DIR} 找不到任何 Markdown (.md) 檔案")
        return

    logger.info(f"📦 總共發現 {len(target_files)} 個 Markdown 檔案待處理")

    for i, md_path in enumerate(target_files):
        logger.info(f"\n🚀 [{i+1}/{len(target_files)}] 處理文件...")
        process_single_markdown(md_path)

    logger.success("\n🎉 所有 Markdown 檔案處理完畢！")

if __name__ == "__main__":
    main()