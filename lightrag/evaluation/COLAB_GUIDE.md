# Hướng dẫn chạy đánh giá trên Google Colab

Copy từng cell dưới đây vào Colab theo thứ tự. Đọc kỹ phần **"Bạn cần điền"** trước khi chạy.

> **Lưu ý về phạm vi đánh giá:** File này đánh giá frame-semantic extraction (do chúng ta thêm vào) bằng RAGAS metrics.
> Đây **không phải** reproduce đúng paper LightRAG gốc (paper dùng UltraDomain dataset + LLM-as-judge win rate).

---

## CELL 1 — Cài đặt thư viện

```python
# Cài tất cả dependencies cần thiết
!cd /content && git clone https://github.com/HKUDS/LightRAG.git 2>/dev/null || echo "Đã clone rồi"
%cd /content/LightRAG
!pip install -q -e ".[api,test]"

# frame-semantic-transformer cần protobuf < 4, cài riêng sau cùng
!pip install -q ragas datasets langchain-openai httpx python-dotenv
!pip install -q "protobuf<4.0.0" frame-semantic-transformer

# Kiểm tra
import importlib
for pkg in ["frame_semantic_transformer", "ragas", "httpx"]:
    try:
        importlib.import_module(pkg)
        print(f"✅ {pkg}")
    except ImportError:
        print(f"❌ {pkg} — chưa cài được")
```

---

## CELL 2 — Tạo file .env (BẠN CẦN ĐIỀN API KEY)

```python
# ============================================================
# ĐIỀN CÁC GIÁ TRỊ CỦA BẠN VÀO ĐÂY
# ============================================================
LLM_API_KEY      = "sk-..."        # OpenAI API key (cho LightRAG server)
EVAL_API_KEY     = "sk-..."        # OpenAI API key (cho RAGAS, có thể dùng cùng key)
EMBEDDING_MODEL  = "text-embedding-3-large"   # hoặc text-embedding-3-small
LLM_MODEL        = "gpt-4o-mini"              # model LightRAG dùng để index
# ============================================================

# Tự động xác định EMBEDDING_DIM theo model — BẮT BUỘC phải đúng
DIM_MAP = {
    "text-embedding-3-small":  1536,
    "text-embedding-ada-002":  1536,
    "text-embedding-3-large":  3072,
}
EMBEDDING_DIM = DIM_MAP.get(EMBEDDING_MODEL, 1536)
print(f"EMBEDDING_DIM tự động: {EMBEDDING_DIM}  (model: {EMBEDDING_MODEL})")

env_content = f"""# LightRAG server config — tự sinh bởi COLAB_GUIDE
LLM_BINDING=openai
LLM_BINDING_HOST=https://api.openai.com/v1
LLM_BINDING_API_KEY={LLM_API_KEY}
LLM_MODEL={LLM_MODEL}

EMBEDDING_BINDING=openai
EMBEDDING_BINDING_HOST=https://api.openai.com/v1
EMBEDDING_BINDING_API_KEY={LLM_API_KEY}
EMBEDDING_MODEL={EMBEDDING_MODEL}
EMBEDDING_DIM={EMBEDDING_DIM}

EVAL_LLM_BINDING_API_KEY={EVAL_API_KEY}
OPENAI_API_KEY={EVAL_API_KEY}

LIGHTRAG_FRAME_EXTRACTION_MODE=full
WORKING_DIR=./rag_storage
MAX_ASYNC=4
MAX_TOKENS=32768
"""

with open("/content/LightRAG/.env", "w") as f:
    f.write(env_content)

print("Đã tạo .env. Kiểm tra:")
with open("/content/LightRAG/.env") as f:
    for line in f:
        if "API_KEY" not in line:
            print(" ", line.strip())
        else:
            key_name = line.split("=")[0]
            print(f"  {key_name}=****")
```

---

## CELL 3 — Xóa storage cũ (PHẢI chạy mỗi lần đổi mode hoặc có lỗi)

```python
import shutil, os

# Storage nằm tại /content/LightRAG/rag_storage (KHÔNG phải /content/rag_storage)
storage_path = "/content/LightRAG/rag_storage"
shutil.rmtree(storage_path, ignore_errors=True)
os.makedirs(storage_path, exist_ok=True)
print(f"✅ Storage đã được xóa sạch: {storage_path}")
```

---

## CELL 4 — Khởi động LightRAG server

```python
import subprocess, time, os, sys, httpx

os.chdir("/content/LightRAG")

# Kill server cũ
os.system("pkill -f lightrag_server 2>/dev/null; pkill -f 'lightrag.api' 2>/dev/null; sleep 2")

proc = subprocess.Popen(
    [sys.executable, "-c", "from lightrag.api.lightrag_server import main; main()"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    cwd="/content/LightRAG",
    env={**os.environ, "PYTHONUNBUFFERED": "1"},
)

# Đọc log và tìm thông báo ready (timeout 3 phút — server cần thời gian load model)
print("Đang chờ server khởi động (tối đa 3 phút)...")
started = False
READY_KEYWORDS = ["Uvicorn running", "Application startup complete", "started server", "listening"]

for i in range(180):
    time.sleep(1)
    line = proc.stdout.readline().decode("utf-8", errors="ignore").strip()
    if line:
        print(f"[{i:02d}s] {line}")
    if proc.poll() is not None:
        remaining = proc.stdout.read().decode("utf-8", errors="ignore")
        print("❌ Server thoát sớm!\n", remaining[-3000:])
        break
    if any(kw in line for kw in READY_KEYWORDS):
        print("\n✅ Server sẵn sàng!")
        started = True
        break

# Fallback: kể cả không thấy keyword, thử kết nối trực tiếp
if not started:
    print("\nKhông thấy keyword 'ready', thử kết nối trực tiếp...")
    for attempt in range(10):
        time.sleep(3)
        try:
            r = httpx.get("http://localhost:9621/health", timeout=5)
            if r.status_code == 200:
                print(f"✅ Server đang chạy (kết nối thành công sau {i + attempt*3}s)")
                started = True
                break
        except Exception:
            print(f"  [{attempt+1}/10] Chưa kết nối được, thử lại...")

if not started:
    print("❌ Server không khởi động được. Kiểm tra .env và thử lại.")
```

---

## CELL 5 — Xác nhận server và EMBEDDING_DIM

```python
import httpx, os

# Kiểm tra server
try:
    r = httpx.get("http://localhost:9621/health", timeout=5)
    print("✅ Server OK:", r.json())
except Exception as e:
    print("❌ Không kết nối được:", e)
    raise SystemExit("Chạy lại Cell 4 trước")

# Xác nhận EMBEDDING_DIM thực tế (phòng ngừa mismatch)
from openai import OpenAI
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY"))
model = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")

# Đọc từ .env nếu chưa có trong os.environ
if not client.api_key:
    from dotenv import dotenv_values
    cfg = dotenv_values("/content/LightRAG/.env")
    client = OpenAI(api_key=cfg.get("EMBEDDING_BINDING_API_KEY"))
    model = cfg.get("EMBEDDING_MODEL", "text-embedding-3-small")

resp = client.embeddings.create(model=model, input=["test"])
actual_dim = len(resp.data[0].embedding)
print(f"✅ Embedding model '{model}' trả về {actual_dim} chiều")

# Đọc EMBEDDING_DIM từ .env để so sánh
from dotenv import dotenv_values
cfg = dotenv_values("/content/LightRAG/.env")
configured_dim = int(cfg.get("EMBEDDING_DIM", 0))
if configured_dim != actual_dim:
    print(f"⚠️  MISMATCH! .env có EMBEDDING_DIM={configured_dim} nhưng model trả về {actual_dim}")
    print(f"   Sửa .env thành EMBEDDING_DIM={actual_dim} rồi chạy lại Cell 3+4")
else:
    print(f"✅ EMBEDDING_DIM={configured_dim} khớp với thực tế — OK")
```

---

## CELL 6 — Chạy đánh giá đầy đủ (frame-semantic mode=full)

```python
import subprocess, os

os.makedirs("/content/LightRAG/lightrag/evaluation/results", exist_ok=True)

result = subprocess.run(
    [
        "python", "lightrag/evaluation/run_full_eval.py",
        "--output", "lightrag/evaluation/results/eval_full_mode.json",
    ],
    cwd="/content/LightRAG",
    capture_output=True, text=True,
    timeout=900,    # 15 phút tối đa
)

print(result.stderr)
if result.stdout:
    print("STDOUT:", result.stdout)
```

---

## CELL 7 — Chạy mode=none (LLM baseline, để so sánh)

```python
import subprocess, os, shutil, sys, time, httpx

# Xóa storage để index lại từ đầu với mode khác
shutil.rmtree("/content/LightRAG/rag_storage", ignore_errors=True)
os.makedirs("/content/LightRAG/rag_storage", exist_ok=True)

# Restart server với mode=none
os.system("pkill -f lightrag_server 2>/dev/null; pkill -f 'lightrag.api' 2>/dev/null; sleep 2")

env_none = {
    **os.environ,
    "LIGHTRAG_FRAME_EXTRACTION_MODE": "none",
    "PYTHONUNBUFFERED": "1",
}

proc2 = subprocess.Popen(
    [sys.executable, "-c", "from lightrag.api.lightrag_server import main; main()"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    cwd="/content/LightRAG", env=env_none,
)

print("Đang chờ server (mode=none) khởi động...")
started2 = False
for i in range(180):
    time.sleep(1)
    line = proc2.stdout.readline().decode("utf-8", errors="ignore").strip()
    if line: print(f"[{i:02d}s] {line}")
    if any(kw in line for kw in ["Uvicorn running", "Application startup complete"]):
        print("✅ Server (mode=none) sẵn sàng!")
        started2 = True
        break

if not started2:
    for attempt in range(10):
        time.sleep(3)
        try:
            httpx.get("http://localhost:9621/health", timeout=5)
            print("✅ Server kết nối được")
            started2 = True
            break
        except Exception:
            pass

if not started2:
    print("❌ Server không start được")
else:
    # Chạy eval mode=none
    result2 = subprocess.run(
        [
            "python", "lightrag/evaluation/run_full_eval.py",
            "--output", "lightrag/evaluation/results/eval_none_mode.json",
        ],
        cwd="/content/LightRAG",
        capture_output=True, text=True,
        timeout=900,
        env=env_none,
    )
    print(result2.stderr[-5000:])
```

---

## CELL 8 — So sánh kết quả 2 mode

```python
import json

def load_ragas(path):
    try:
        with open(path) as f:
            data = json.load(f)
        return data.get("ragas_eval", {}).get("average_metrics", {})
    except FileNotFoundError:
        print(f"❌ Không tìm thấy: {path}")
        return {}

r_full = load_ragas("/content/LightRAG/lightrag/evaluation/results/eval_full_mode.json")
r_none = load_ragas("/content/LightRAG/lightrag/evaluation/results/eval_none_mode.json")

if not r_full or not r_none:
    print("Thiếu kết quả. Chạy Cell 6 và Cell 7 trước.")
else:
    metrics = ["faithfulness", "answer_relevance", "context_recall", "context_precision", "ragas_score"]

    print(f"\n{'Metric':<22}  {'Frame-full':>12}  {'LLM-none':>12}  {'Delta':>10}  Winner")
    print("-" * 75)
    frame_wins = 0
    for m in metrics:
        f = r_full.get(m, 0)
        n = r_none.get(m, 0)
        delta = f - n
        if delta > 0.01:
            winner = "Frame+ ✅"
            frame_wins += 1
        elif delta < -0.01:
            winner = "LLM+   ⬆"
        else:
            winner = "Hòa   ="
        print(f"{m:<22}  {f:>12.4f}  {n:>12.4f}  {delta:>+10.4f}  {winner}")

    print(f"\nFrame-semantic thắng: {frame_wins}/{len(metrics)} metrics")
```

---

## Troubleshooting

### RAGAS toàn 0.0
1. **EMBEDDING_DIM sai** — Cell 5 sẽ phát hiện và báo lỗi ngay
2. **Storage chưa xóa** — chạy Cell 3 trước khi restart server
3. **Documents failed** — xem dòng "Trang thai:" trong output Cell 6; nếu thấy `failed > 0` thì quay lại fix

### Server không start được
- Kiểm tra `.env` có đủ `LLM_BINDING_API_KEY` không
- Thử tăng timeout lên 240s trong Cell 4
- Xem dòng `❌` trong log để tìm lỗi cụ thể

### `frame-semantic-transformer` không cài được
```python
# Chạy riêng, theo thứ tự nghiêm ngặt
!pip uninstall -y protobuf 2>/dev/null
!pip install "protobuf<4.0.0"
!pip install frame-semantic-transformer
```

### Protobuf conflict (frame-semantic vs google-genai)
frame-semantic-transformer cần `protobuf<4` nhưng một số thư viện Google cần `protobuf>=4`.
Nếu bị conflict, chỉ có thể chạy trong môi trường riêng không cài google-cloud packages.

---

## Chi phí ước tính (1 lần chạy đầy đủ 2 mode)

| Thành phần | Chi phí ước tính |
|-----------|-----------------|
| Indexing 5 docs × 2 modes (LLM + embedding) | ~$0.05–0.15 |
| RAGAS 6 queries × 2 modes | ~$0.02–0.05 |
| **Tổng** | **~$0.10–0.20** |

> Dùng `text-embedding-3-small` (EMBEDDING_DIM=1536) để tiết kiệm hơn `text-embedding-3-large`.

---

## Phạm vi evaluation so với paper gốc

| | File này (COLAB_GUIDE) | Paper LightRAG gốc |
|--|--|--|
| **Mục đích** | Đánh giá frame-semantic extraction | Benchmark RAG framework tổng thể |
| **Dataset** | 6 câu hỏi mẫu (tự tạo) | UltraDomain: Agriculture, CS, Legal, Mixed (~500+ câu) |
| **Metrics** | RAGAS 4 metrics | LLM-as-judge: Comprehensiveness, Diversity, Empowerment |
| **So sánh** | frame-full vs frame-none | LightRAG vs NaiveRAG vs RQ-RAG vs HyDE vs GraphRAG |
| **Output** | Điểm số tuyệt đối (0–1) | Win rate (%) |
| **Chi phí** | ~$0.10–0.20 | ~$50–200+ (ước tính) |
