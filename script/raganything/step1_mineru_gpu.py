import sys
import os
import subprocess
import json
import glob
import time
import shutil
import torch
from loguru import logger

# === 1. Enhanced Logging Setup ===
LOG_DIR = "./logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

log_file = os.path.join(LOG_DIR, f"step1_mineru_verbose_{time.strftime('%Y%m%d_%H%M%S')}.log")
logger.remove() 

# Console Handler: Show INFO and above to screen
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>", level="INFO")

# File Handler: Capture DEBUG and above (everything) to file
logger.add(log_file, rotation="10 MB", level="DEBUG", encoding="utf-8")

logger.info(f"📝 Log file created: {log_file}")

# ==========================================

def get_device_config():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.success(f"🎮 GPU Detected: {gpu_name} ({vram_gb:.2f} GB VRAM)")
        logger.info("🚀 Running Mineru in CUDA mode (High Speed)")
        return "cuda"
    else:
        logger.warning("⚠️ No GPU detected. Falling back to CPU mode (Slower)")
        return "cpu"

def run_command_with_live_output(cmd):
    """
    Executes a shell command and streams the output (stdout/stderr) in real-time.
    """
    logger.debug(f"Executing command: {' '.join(cmd)}")
    
    try:
        # Popen allows us to read the stream line by line
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, # Merge stderr into stdout
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1 # Line buffered
        )

        # Read and log line by line
        for line in process.stdout:
            line = line.strip()
            if line:
                # Log to file as DEBUG, print to console with a prefix
                logger.debug(f"[Mineru Internal] {line}")
                # Optional: Filter what shows on console to avoid spam, or show everything
                print(f"   [Mineru] {line}") 

        return_code = process.wait()
        return return_code

    except Exception as e:
        logger.exception(f"❌ Failed to run command: {e}")
        return -1

def process_single_file(input_path, output_base_dir, step1_std_base_dir, config):
    filename = os.path.basename(input_path)
    file_stem = os.path.splitext(filename)[0]
    ext = os.path.splitext(filename)[1].lower()
    
    current_file_std_dir = os.path.join(step1_std_base_dir, file_stem)
    if not os.path.exists(current_file_std_dir):
        os.makedirs(current_file_std_dir)
        logger.info(f"📂 Created directory: {current_file_std_dir}")

    logger.info("="*60)
    logger.info(f"🚀 [Start] Processing: {filename}")
    
    start_time = time.time()

    # === [Strategy A] Text File Bypass ===
    if ext in ['.txt', '.md']:
        logger.info(f"📄 Detected text file ({ext}). Bypassing Mineru...")
        try:
            with open(input_path, "r", encoding="utf-8") as f: content = f.read()
            mock_data = [{"type": "text", "text": content, "page_idx": 0, "bbox": [0,0,0,0], "img_path": ""}]
            final_json_path = os.path.join(current_file_std_dir, "intermediate_result.json")
            with open(final_json_path, "w", encoding="utf-8") as f:
                json.dump(mock_data, f, ensure_ascii=False, indent=2)
            logger.success(f"✅ Text conversion successful!")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to read text file: {e}")
            return False

    # === [Strategy B] Mineru PDF Processing ===
    
    # 1. Define Expected Paths
    # Note: 'vlm' is the output folder name for the vlm-transformers model
    expected_paths = [
        os.path.join(output_base_dir, file_stem, "vlm", f"{file_stem}_content_list.json"),
        os.path.join(output_base_dir, file_stem, config["use_backend"], f"{file_stem}_content_list.json"),
        os.path.join(output_base_dir, file_stem, "auto", f"{file_stem}_content_list.json"),
        os.path.join(output_base_dir, file_stem, f"{file_stem}_content_list.json"),
    ]

    target_json = None
    
    # 2. Smart Check: Does the output ALREADY exist?
    logger.info("🔍 Checking for existing Mineru output...")
    for p in expected_paths:
        if os.path.exists(p):
            target_json = p
            break
            
    run_mineru = True
    if target_json:
        logger.success(f"✨ Found existing output at: {target_json}")
        logger.info("⏩ Skipping AI inference. Proceeding to file copy...")
        run_mineru = False 
    else:
        logger.info("❌ No existing output found. Preparing to run Mineru...")
    
    # 3. Execute Mineru (if needed)
    if run_mineru:
        cmd = [
            "uv", "run", "mineru",
            "-p", input_path,
            "-o", output_base_dir,
            "-m", "auto",
            "-b", config["use_backend"],
            "-d", config["use_device"]
        ]
        
        logger.info(f"🔧 Launching Mineru...")
        return_code = run_command_with_live_output(cmd)
        
        if return_code != 0:
            logger.error(f"❌ Mineru process failed with return code: {return_code}")
            # We don't return False immediately, just in case partial output was generated
        else:
            logger.success("✅ Mineru process finished successfully.")

        # Re-check for files after execution
        logger.info("🔍 Verifying output files...")
        for p in expected_paths:
            if os.path.exists(p):
                target_json = p
                logger.info(f"   -> Found: {p}")
                break

    # 4. Copy/Move Files
    if target_json:
        final_json_name = "intermediate_result.json"
        final_json_path = os.path.join(current_file_std_dir, final_json_name)
        
        try:
            # A. Copy JSON
            logger.info(f"📋 Copying JSON to {final_json_path}...")
            shutil.copy2(target_json, final_json_path)
            
            # B. Copy Images
            mineru_img_dir = os.path.join(os.path.dirname(target_json), "images")
            my_img_dir = os.path.join(current_file_std_dir, "images")
            
            if os.path.exists(mineru_img_dir):
                logger.info(f"🖼️ Copying images from {mineru_img_dir}...")
                if os.path.exists(my_img_dir): 
                    shutil.rmtree(my_img_dir)
                shutil.copytree(mineru_img_dir, my_img_dir)
                
                img_count = len(os.listdir(my_img_dir))
                logger.success(f"✅ Copied {img_count} images.")
            else:
                logger.warning(f"⚠️ No 'images' folder found at source (Normal for text-only PDFs).")

            duration = time.time() - start_time
            logger.success(f"🎉 Process Complete! Total time: {duration:.2f}s")
            return True

        except Exception as json_err:
            logger.error(f"⚠️ File copy error: {json_err}")
            return False
    else:
        logger.error(f"❌ Critical: Could not find Mineru output JSON after execution.")
        logger.debug(f"   Checked paths: {expected_paths}")
        return False

def main():
    # === Configuration ===
    input_dir = "./data/input/__enqueued__"
    output_dir = "./data/output/step1_vlm_output"
    step1_std_base_dir = "./data/input/step1_output" 
    
    EXCLUDE_FILES = ["sfc.pdf", "sfc_report.pdf"] 
    
    FORCE_RERUN = False # Set to True to ignore existing final output and force re-process

    # 🔥 GPU Config
    device_type = get_device_config()
    config = {
        "use_backend": "vlm-transformers", 
        "use_device": device_type
    }

    # === Validation ===
    if not os.path.exists(input_dir):
        logger.error(f"❌ Input directory not found: {input_dir}")
        return

    if not os.path.exists(step1_std_base_dir):
        os.makedirs(step1_std_base_dir)

    # === File Scanning ===
    all_entries = glob.glob(os.path.join(input_dir, "*"))
    files = [f for f in all_entries if os.path.isfile(f) and not os.path.basename(f).startswith(".")]

    if not files:
        logger.warning(f"📂 No files found in {input_dir}")
        return

    logger.info(f"📦 Found {len(files)} files. Starting batch process...")
    
    success_count = 0
    fail_count = 0
    skipped_count = 0

    # === Batch Processing ===
    for i, file_path in enumerate(files):
        filename = os.path.basename(file_path)
        file_stem = os.path.splitext(filename)[0]
        
        logger.info(f"\n⏳ [Progress: {i+1}/{len(files)}] Processing: {filename}")

        # 1. Exclude List Check
        if filename.lower() in EXCLUDE_FILES:
            logger.info(f"🛑 Skipped (In Exclude List): {filename}")
            skipped_count += 1
            continue
        
        # 2. Final Result Exists Check
        # This checks if the *final destination* file exists.
        # If FORCE_RERUN is False, we skip the whole file.
        expected_output_path = os.path.join(step1_std_base_dir, file_stem, "intermediate_result.json")
        
        if not FORCE_RERUN and os.path.exists(expected_output_path):
            logger.info(f"⏭️  File already processed (Result exists). Skipping.")
            skipped_count += 1
            success_count += 1 
            continue
        
        # 3. Process File
        if process_single_file(file_path, output_dir, step1_std_base_dir, config):
            success_count += 1
        else:
            fail_count += 1

    # === Summary ===
    logger.info("\n" + "="*60)
    logger.info(f"🏁 Batch Processing Finished!")
    logger.info(f"📊 Stats: Total {len(files)} | ✅ Success {success_count} | ⏭️ Skipped {skipped_count} | ❌ Failed {fail_count}")
    logger.info(f"📝 Full logs available at: {LOG_DIR}")

if __name__ == "__main__":
    main()