#!/bin/bash
set -e

# ==============================================================================
# 1. 變數與路徑設定
# ==============================================================================
MODEL_DIR="${MINERU_MODEL_DIR:-/app/data/mineru_models}"
REPO_ID="${MINERU_REPO_ID:-opendatalab/PDF-Extract-Kit}"

CONFIG_FILE_ROOT="/root/magic-pdf.json"
CONFIG_FILE_APP="/app/magic-pdf.json"
CONFIG_FILE_DATA="/app/data/magic-pdf.json"

echo "🚀 [MinerU-Init] 初始化環境..."

# ==============================================================================
# 2. 智能 GPU 偵測
# ==============================================================================
if [ -z "$MINERU_DEVICE_MODE" ]; then
    echo "🔍 [MinerU-Init] 未設定運行模式，正在自動偵測 GPU..."
    if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
        export DEVICE_MODE="cuda"
    else
        export DEVICE_MODE="cpu"
    fi
    echo "💡 [MinerU-Init] 自動偵測結果: $DEVICE_MODE"
else
    export DEVICE_MODE="$MINERU_DEVICE_MODE"
    echo "⚙️ [MinerU-Init] 使用環境變數設定: $DEVICE_MODE"
fi

# ==============================================================================
# 3. 檢查並下載模型 (保留這個重要功能！)
# ==============================================================================
# 檢查具體的 YOLO 權重文件是否存在，而不僅僅是資料夾
if [ ! -f "$MODEL_DIR/models/MFD/weights.pt" ]; then
    echo "⚠️ [MinerU-Init] 未偵測到模型，準備自動下載..."
    mkdir -p "$MODEL_DIR"
    python -c "
import os
from huggingface_hub import snapshot_download
try:
    print('⬇️ 開始下載模型 (約 5GB+)...')
    snapshot_download(repo_id='$REPO_ID', local_dir='$MODEL_DIR', resume_download=True)
    print('✅ 模型下載完成！')
except Exception as e:
    print(f'❌ 下載失敗: {e}')
    exit(1)
"
else
    echo "✅ [MinerU-Init] 模型已存在，跳過下載。"
fi

# ==============================================================================
# 4. 生成 Config
# ==============================================================================
echo "⚙️ [MinerU-Init] 生成設定檔內容..."
CONFIG_CONTENT=$(cat <<EOF
{
  "bucket_info": { "bucket-name-1": ["ak", "sk", "endpoint"], "bucket-name-2": ["ak", "sk", "endpoint"] },
  "models-dir": "$MODEL_DIR/models",
  "device-mode": "$DEVICE_MODE",
  "layout-config": { "model": "doclayout_yolo" },
  "formula-config": { "mfd_model": "doclayout_yolo", "mfr_model": "unimernet_small" },
  "table-config": { "model": "rapid_table", "model_dir": "$MODEL_DIR/models/Table/RapidTable" }
}
EOF
)
echo "$CONFIG_CONTENT" > "$CONFIG_FILE_ROOT"
echo "$CONFIG_CONTENT" > "$CONFIG_FILE_APP"
echo "$CONFIG_CONTENT" > "$CONFIG_FILE_DATA"
echo "✅ [MinerU-Init] 設定檔已寫入: /root, /app, /app/data"

# ❌ [已刪除] 原本在這裡的 "Section 5: Hotfix" 已經不需要了，因為我們在 Git 源碼修好了

# ==============================================================================
# 6. 啟動 LightRAG Server
# ==============================================================================
echo "✨ [LightRAG] 啟動主程式..."
exec python -m lightrag.api.lightrag_server