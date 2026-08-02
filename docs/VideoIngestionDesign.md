# Đặc tả ingest video nội bộ cho LightRAG

Tài liệu này là specification cho việc bổ sung ingest video trực tiếp vào
LightRAG. Mục tiêu là biến một video thành dữ liệu có thể truy hồi theo nội
dung lời nói, cảnh nhìn thấy, thực thể/quan hệ và **timecode**, trong khi vẫn
đi qua pipeline chuẩn của LightRAG:

```text
upload/scan -> parse -> multimodal analysis -> chunk -> embedding -> KG -> query
```

Toàn bộ điều phối, tiền xử lý và lưu artifact chạy trong container LightRAG.
Chỉ các tác vụ cần GPU gọi các model đã được serving riêng bằng OpenAI-compatible
API. Không tạo video-processing service, không mount source video vào các
container model và không tạo Docker network dùng chung mới.

## 1. Phạm vi và quyết định thiết kế

### 1.1. Mục tiêu của phiên bản đầu

- Nhận các video `mp4`, `mov`, `mkv`, `webm`, `avi`, `m4v` qua API upload và
  thư mục scan hiện có.
- Trích transcript có timestamp bằng Qwen3-ASR.
- Phát hiện các scene, tạo contact sheet từ các frame đại diện và phân tích
  hình ảnh bằng Qwen3.5 VLM.
- Lưu transcript, scene, asset ảnh và timecode trong sidecar chuẩn LightRAG.
- Dùng pipeline sẵn có để chunk, embedding, entity/relation extraction và KG.
- Khi truy vấn, câu trả lời và reference có thể chỉ ra khoảng thời gian
  `HH:MM:SS.mmm -> HH:MM:SS.mmm` của video gốc.

### 1.2. Ngoài phạm vi của phiên bản đầu

- Không tách FFmpeg, scene detection, contact-sheet generation hay manifest
  thành microservice.
- Không dùng MinerU để parse video. MinerU tiếp tục phục vụ PDF, Office và ảnh
  theo cấu hình hiện có.
- Không dùng TTS trong ingest. TTS chỉ là năng lực output audio ở một tính năng
  khác trong tương lai.
- Không thêm speaker diarization, nhận diện âm thanh không phải lời nói hay
  OCR engine riêng. Qwen3.5 thực hiện OCR/visual description của contact sheet.
- Không tạo index vector ảnh/video riêng trong phiên bản đầu. Đây là Phase 2
  sau khi pipeline cơ bản được đo chất lượng và tải thực tế.

### 1.3. Các service model sử dụng

| Vai trò | Model/service hiện có | Cổng từ container LightRAG | Dùng cho video |
| --- | --- | --- | --- |
| Generation / VLM | Qwen3.5-35B-A3B | `host.docker.internal:8010/v1` | Phân tích contact sheet, entity/KG, query answer |
| Embedding | Qwen3-VL-Embedding-8B | `host.docker.internal:8011/v1` | Embed chunk giàu transcript + visual description |
| ASR | Qwen3-ASR-1.7B | `host.docker.internal:8012/v1` | Transcript theo đoạn thời gian |
| TTS | Qwen3-TTS | `host.docker.internal:8013/v1` | Không dùng trong ingest |
| MinerU | Service hiện có | Docker network hiện hữu | Không dùng trong parser `video` |

`host.docker.internal` đã được khai báo bằng `extra_hosts` trong compose của
LightRAG. Vì vậy parser gọi các model qua host gateway giống cấu hình LLM và
embedding hiện tại; tuyệt đối không dùng `localhost` từ bên trong container
LightRAG.

Qwen3-VL-Embedding-8B hỗ trợ semantic space chung cho text, image và video.
Ở phiên bản đầu, LightRAG dùng model này để embed phần text đã được làm giàu
bởi ASR và VLM. Điều này giữ nguyên chuẩn vector storage hiện hữu. Khả năng
embed trực tiếp contact sheet/video sẽ được khai thác ở Phase 2.

## 2. Kiến trúc thực thi

```text
                 CPU trong LightRAG container
┌─────────────────────────────────────────────────────────────────────┐
│  VideoParser                                                        │
│                                                                     │
│  ffprobe -> audio normalize -> ASR segmentation -> transcript      │
│       │                                                             │
│       └-> shot detection -> frames -> contact sheet -> sidecar     │
└───────────────────────┬─────────────────────────────────────────────┘
                        │ OpenAI-compatible HTTP, qua host gateway
          ┌─────────────┴─────────────┐
          ▼                           ▼
 Qwen3-ASR :8012              Qwen3.5 VLM :8010
 transcript                  scene visual analysis
          │                           │
          └─────────────┬─────────────┘
                        ▼
      LightRAG Document sidecar + timeline manifest
                        ▼
       P chunking -> Qwen3-VL embedding :8011 -> KG/vector DB
```

Hai nguyên tắc cần giữ khi code:

1. VideoParser là **built-in parser** của LightRAG, không phải một HTTP service
   độc lập. Việc tách các helper thành module Python chỉ để code dễ kiểm thử,
   không thay đổi ranh giới triển khai.
2. Model-serving service chỉ nhận byte audio hoặc ảnh được chuẩn hoá trong
   request. Service model không cần truy cập file gốc, bind mount, volume hay
   knowledge về storage của LightRAG.

## 3. Điểm tích hợp vào mã nguồn

### 3.1. Parser registry và worker queue

Thêm engine name `video` vào `lightrag.constants`, sau đó thêm `ParserSpec`
vào `lightrag/parser/registry.py`:

```python
ParserSpec(
    engine_name="video",
    impl="lightrag.parser.video.parser:VideoParser",
    suffixes=frozenset({"mp4", "mov", "mkv", "webm", "avi", "m4v"}),
    queue_group="video",
    endpoint_configured=_video_asr_endpoint_configured,
    endpoint_requirement=lambda: "VIDEO_ASR_BINDING_HOST",
)
```

`queue_group="video"` là bắt buộc: pipeline đã tạo queue riêng theo
`ParserSpec.queue_group`. Nhờ vậy video không chiếm worker của `native`,
`mineru` hoặc `docling`. Vì `video` là queue built-in, worker count được đọc
bởi field `LightRAG.max_parallel_parse_video` từ `MAX_PARALLEL_PARSE_VIDEO`;
service LightRAG phải restart sau khi đổi giá trị.

Video được xem là khả dụng để route chỉ khi `VIDEO_ASR_BINDING_HOST` được cấu
hình. Video không có audio vẫn được xử lý thành visual-only document, nhưng
host ASR vẫn là điều kiện đăng ký engine để tránh một deployment vô tình nhận
video rồi tạo transcript rỗng hàng loạt.

### 3.2. Cấu trúc module nội bộ

```text
lightrag/parser/video/
├── __init__.py
├── parser.py       # VideoParser và parse transaction
├── media.py        # ffprobe, FFmpeg, audio/frame extraction
├── scene.py        # chọn scene và contact-sheet planning
├── asr.py          # VideoAsrClient, request/response contract, retry
├── manifest.py     # dataclass + JSON serialisation của timeline manifest
└── ir_builder.py   # scene/transcript -> IRDoc, IRBlock, IRDrawing
```

Đây là các module cùng process. Không module nào được import khi registry chỉ
đang kiểm tra suffix/capability; registry giữ `impl` dưới dạng string để lazy
import đúng theo pattern parser hiện có.

`VideoParser` kế thừa `NativeParserBase`. CPU work, FFmpeg và HTTP ASR đồng bộ
chạy trong worker thread của `NativeParserBase.extract`; do đó event loop của
API không bị block. `VideoAsrClient` dùng một synchronous `httpx.Client` có
timeout/retry giới hạn và chỉ gọi tuần tự trong một video. Không gọi nhiều ASR
segment song song vì service hiện được cấu hình `ASR_MAX_NUM_SEQS=1`.

## 4. Cấu hình triển khai

### 4.1. Docker image LightRAG

Thêm các system package vào final stage của `Dockerfile`:

```dockerfile
ffmpeg
fonts-dejavu-core
```

`ffmpeg` cung cấp cả `ffmpeg` và `ffprobe`. Font DejaVu dùng để ghi timecode
vào contact sheet một cách xác định. Không chạy FFmpeg trong container model.

Dependency Python runtime:

- `Pillow>=10.0` dùng để ghép contact sheet;
- `httpx` đã có trong dependency, dùng cho ASR client.

Scene cut hiện dùng FFmpeg `select=gt(scene,threshold)` và fallback về các cửa
sổ đều nhau khi codec không hỗ trợ phân tích; cách này không kéo OpenCV vào
image LightRAG.

### 4.2. `.env` của LightRAG

Giữ nguyên block LLM/VLM/embedding nội bộ hiện tại. Thêm block sau, thay
placeholder bằng secret thực tế; không ghi secret vào repository hay tài liệu:

```dotenv
# ============================================================
# Video ingestion (built-in parser)
# ============================================================
VIDEO_ASR_BINDING_HOST=http://host.docker.internal:8012/v1
VIDEO_ASR_BINDING_API_KEY=<QWEN_ASR_API_KEY>
VIDEO_ASR_MODEL=qwen3-asr-1.7b
VIDEO_ASR_LANGUAGE=
VIDEO_ASR_TIMEOUT=180

# Admission control and source limits
MAX_PARALLEL_PARSE_VIDEO=1
MAX_UPLOAD_SIZE=4294967296
VIDEO_MAX_DURATION_SECONDS=21600

# Audio segmentation (seconds)
VIDEO_ASR_SEGMENT_SECONDS=30
VIDEO_FFMPEG_TIMEOUT=180
VIDEO_FFPROBE_TIMEOUT=60

# Visual timeline and contact sheets
VIDEO_SCENE_THRESHOLD=0.30
VIDEO_MIN_SCENE_SECONDS=2
VIDEO_MAX_SCENE_SECONDS=45
VIDEO_MAX_SCENES=128
VIDEO_FRAMES_PER_SCENE=4
VIDEO_CONTACT_SHEET_CELL_WIDTH=384
VIDEO_CONTACT_SHEET_JPEG_QUALITY=82
```

`MAX_UPLOAD_SIZE` chỉ là ví dụ 4 GiB. Chọn một giới hạn phù hợp với disk và
reverse proxy thực tế; không đặt `0`/unlimited trên production nếu không có
quota ở tầng khác. `VIDEO_MAX_DURATION_SECONDS` giới hạn thời lượng decode và
chi phí GPU, không thay thế giới hạn dung lượng upload.

Thêm các route video **trước** wildcard `*:legacy-R` trong `LIGHTRAG_PARSER`
hiện có. Không được thay thế các rule PDF/Office/Image đang hoạt động:

```dotenv
LIGHTRAG_PARSER=mp4:video-iteP;mov:video-iteP;mkv:video-iteP;webm:video-iteP;avi:video-iteP;m4v:video-iteP;...;*:legacy-R
```

`i` yêu cầu pipeline phân tích ảnh sidecar bằng VLM. `P` giữ scene là đơn vị
ngữ nghĩa và tránh trộn hai time range kề nhau thành một chunk tùy tiện.

VLM đã dùng các biến đang tồn tại:

```dotenv
VLM_PROCESS_ENABLE=true
VLM_LLM_BINDING=openai
VLM_LLM_BINDING_HOST=http://host.docker.internal:8010/v1
VLM_LLM_MODEL=qwen3.5-35b-a3b
```

Embedding tiếp tục dùng port `8011` và `EMBEDDING_DIM=1536` hiện tại. Không
đổi dimension trong feature video: thay đổi dimension sẽ yêu cầu migration/
rebuild vector index và re-ingest dữ liệu cũ.

### 4.3. Qwen-serving

Không cần thay đổi Docker network hoặc `docker-compose.yml` của Qwen-serving
cho phiên bản đầu. Điều kiện vận hành duy nhất là ASR đã chạy:

```bash
docker compose --profile speech up -d qwen-asr
```

Không cần khởi động TTS chỉ để ingest video. Các giá trị serving hiện tại là
phù hợp với admission control này:

- Qwen3-ASR nhận tối đa một request đồng thời;
- Qwen3.5 nhận tối đa hai request đồng thời; `MAX_PARALLEL_ANALYZE=2` hiện có
  là trần, không cần tăng khi thêm video;
- Qwen3-VL-Embedding tiếp tục batch text chunk như các document khác.

## 5. Thuật toán `VideoParser`

### 5.1. Preflight

1. Dùng `ffprobe -show_format -show_streams -of json` để đọc duration, codec,
   width/height, rotation, FPS và audio streams.
2. Từ chối video không có video stream, không đọc được duration, duration vượt
   `VIDEO_MAX_DURATION_SECONDS` hoặc có metadata không nhất quán.
3. Video không có audio stream là hợp lệ. Parser phát warning
   `no_audio_stream` và tiếp tục visual-only.
4. Không decode toàn bộ video vào RAM. Mọi frame/audio phải được ghi ra
   temporary directory, xử lý streaming và xoá sau khi sidecar hoàn tất.

### 5.2. Audio và ASR

Chuẩn hoá stream audio bằng FFmpeg thành WAV mono 16 kHz. `silencedetect` tạo
ứng viên boundary; thuật toán sau đó chọn boundary gần target 45 giây nhất,
nhưng không segment nào dài quá 60 giây. Nếu không có khoảng lặng phù hợp,
hard cut tại giới hạn tối đa. Segment lân cận overlap 0.5 giây để giảm mất từ
ở boundary.

`VideoAsrClient` có contract nội bộ sau:

```python
transcribe(
    wav_bytes: bytes,
    *,
    language: str = "",
) -> str
```

Khoảng thời gian của segment được parser gắn vào `AudioSegment`/manifest;
client chỉ trả về text ASR. Client là nơi duy nhất biết payload
audio chính xác của OpenAI-compatible vLLM API. Trước khi merge implementation,
phải có contract test dùng sample WAV nhỏ với service Qwen3-ASR thật; không để
payload API rải trong parser.

ASR là best-effort có kiểm soát: nếu endpoint chưa cấu hình hoặc một segment
lỗi, parser vẫn giữ scene/contact sheet và ghi warning theo scene trong
`parse_warnings`/manifest. Muốn vận hành quality-first, cần để
`VIDEO_ASR_BINDING_HOST` cấu hình đúng và coi warning ASR là lỗi vận hành cần
điều tra; parser không làm mất toàn bộ bằng chứng hình ảnh chỉ vì ASR tạm lỗi.

Transcript được làm sạch khoảng trắng, deduplicate phần overlap ở boundary
theo text similarity đơn giản, nhưng không tự bịa punctuation hoặc timestamp
cấp word. Time range của ASR segment là nguồn sự thật của transcript trong
phiên bản đầu.

### 5.3. Lập timeline scene

Scene timeline phải nắm cả thay đổi hình ảnh lẫn thời lượng. Thuật toán:

1. Dùng FFmpeg scene filter để tìm shot cut; nếu codec không hỗ trợ phân tích
   thì dùng uniform scene windows.
2. Bỏ candidate cách nhau dưới `VIDEO_MIN_SCENE_SECONDS` để tránh flash/cut
   noise.
3. Nếu một shot dài hơn `VIDEO_MAX_SCENE_SECONDS`, chia thêm các mốc đều nhau.
   Điều này quan trọng với slide, màn hình demo và camera tĩnh.
4. Luôn giữ mốc đầu, mốc cuối và các cut có độ thay đổi cao.
5. Nếu số scene vượt `VIDEO_MAX_SCENES`, phân bổ budget theo toàn bộ duration:
   giữ cut quan trọng trước, rồi lấy uniform samples ở các khoảng còn trống.
   Manifest phải ghi rõ `scene_budget_truncated=true` và số scene bỏ qua.

Mỗi scene có một time range đóng/mở `[start_ms, end_ms)`. Không dùng frame
number làm định danh lâu dài vì VFR video có thể làm frame number mơ hồ.

### 5.4. Contact sheet

Mỗi scene chọn ba timestamp: đầu có margin, giữa, cuối có margin. Frame được
decode bởi FFmpeg, resize từng frame theo `VIDEO_CONTACT_SHEET_CELL_WIDTH`
và ghép thành một ảnh JPEG. Contact sheet phải có:

- nhãn `Scene N`;
- `start -> end` của scene;
- timestamp của từng ô frame;
- kích thước/quality bị giới hạn để không vượt `VLM_MAX_IMAGE_BYTES`.

Một contact sheet tương ứng một `IRDrawing`. Gửi một ảnh này cho VLM tốt hơn
gửi ba request frame lẻ: model thấy thay đổi ngắn theo thời gian, chi phí VLM
vẫn là một call/scene và sidecar vẫn tương thích với cơ chế image hiện tại.

### 5.5. Nối audio và visual thành scene block

Với mỗi scene, lấy tất cả transcript segment giao với scene range. Nếu đoạn
ASR giao một phần, cắt theo **time range** ở cấp segment và giữ nguyên text;
không cố tạo word timestamp giả. Nội dung block có format ổn định:

```md
# Video: <document title>

## Scene 0014 — 00:12:34.500 -> 00:12:51.200
Speech transcript:
<transcript overlapping the scene, or "[No spoken audio detected]">

Visual evidence:
{{IMG:scene-0014}}
```

Mỗi scene phải là heading cấp 1 độc lập trong `IRDoc` để paragraph semantic
chunker không gộp hai time range không liên quan. Metadata tổng quan video là
một block riêng ở đầu document: duration, streams, language hint và warnings;
không sao chép metadata kỹ thuật vào mọi scene.

## 6. Sidecar và traceability

### 6.1. Artifact layout

Sau khi parse thành công, directory có dạng:

```text
data/inputs/<workspace>/__parsed__/<video>.parsed/
├── <video>.blocks.jsonl
├── <video>.drawings.json
├── <video>.blocks.assets/
│   ├── scene-0001.jpg
│   └── scene-0002.jpg
├── <video>.video_manifest.json
└── <video>.transcript.json
```

`blocks.jsonl`, `drawings.json` và `blocks.assets` được tạo qua
`write_sidecar()` và là sidecar chuẩn. Hai file JSON còn lại là artifact video
phục vụ traceability/debug; downstream LightRAG không được phụ thuộc vào việc
parse thủ công hai file này để index.

Source video chỉ được archive sau khi `full_docs` đã được persist thành công,
theo đúng transaction của `NativeParserBase`.

### 6.2. `drawings.json` extension an toàn

Không thay schema top-level của `IRDrawing`. Dùng `extras` đã có sẵn:

```json
{
  "media_type": "video_contact_sheet",
  "scene_id": "scene-0014",
  "start_ms": 754500,
  "end_ms": 771200,
  "frame_timestamps_ms": [755000, 762850, 770700],
  "manifest_ref": "<video>.video_manifest.json#/scenes/13"
}
```

`caption` của drawing chứa scene/timecode ở dạng người đọc được. `self_ref`
trỏ đến JSON Pointer trong manifest. Không thêm `IRPosition.type` mới ở phiên
bản đầu; timecode luôn hiện trong heading, caption, `extras` và manifest, nên
không phụ thuộc schema position hiện có.

### 6.3. Timeline manifest

`<video>.video_manifest.json` tối thiểu gồm:

```json
{
  "version": "1.0",
  "parser": {"name": "video", "version": "<implementation version>"},
  "source": {"name": "demo.mp4", "size_bytes": 0, "duration_ms": 0},
  "media": {"video_stream": {}, "audio_stream": {}},
  "config": {"scene_max_seconds": 20, "asr_target_seconds": 45},
  "transcript_segments": [],
  "scenes": [],
  "warnings": []
}
```

Không lưu API key, request Authorization header, full raw provider response
hay absolute path host trong manifest. Manifest cho phép tái hiện lý do một
scene được chọn, đối chiếu transcript và debug lỗi mà không cần reprocess
video.

## 7. VLM analysis cho video scene

Pipeline hiện có sẽ đọc `drawings.json` khi file route chứa option `i` và
`VLM_PROCESS_ENABLE=true`. Cần bổ sung `video_frame_analysis` vào
`lightrag/prompt_multimodal.py` và trong `analyze_multimodal` chọn prompt này
khi `item.extras.media_type == "video_contact_sheet"`.

Prompt mới vẫn trả về cùng contract `llm_analyze_result` hiện có:

```json
{
  "name": "short factual scene name",
  "type": "Scene | Screen | Chart | Demonstration | Interview | Other",
  "description": "factual, retrieval-oriented description"
}
```

Yêu cầu prompt:

- ghi rõ timecode scene trong description;
- mô tả đối tượng, hành động, sự thay đổi giữa các frame, bảng/biểu đồ/màn hình;
- OCR visible text quan trọng, giữ nguyên chính tả nếu model tự tin;
- dùng transcript chỉ như context và phân biệt “seen” với “heard”;
- không suy đoán tên người, nội dung ngoài khung hình hoặc quan hệ không có
  evidence;
- trả về bằng `SUMMARY_LANGUAGE` và JSON hợp lệ.

Không gọi VLM trực tiếp trong `VideoParser`. Parser chỉ tạo sidecar objective;
VLM analysis ở stage tiêu chuẩn giúp retry, LLM cache, queue control và việc
re-analyze bằng option `i` hoạt động giống tất cả document khác.

## 8. Indexing, truy hồi và hiển thị evidence

### 8.1. Phiên bản đầu: textual multimodal retrieval

Sau khi VLM ghi `llm_analyze_result`, content scene gồm đồng thời:

- transcript ASR;
- heading/caption/timecode;
- description và OCR của VLM;
- metadata source/scenes cần thiết.

Chunk này được Qwen3-VL-Embedding-8B embed như text bình thường. Entity và
relation extraction cũng nhìn thấy transcript cùng visual description, vì vậy
KG có thể liên kết speech với người, sản phẩm, biểu đồ, thao tác hoặc sự kiện
trong scene.

Query `mix`/`naive` không cần một API mới để hoạt động. Câu trả lời phải ưu
tiên cite timecode xuất hiện trong context. `query_enrichment` nên đưa contact
sheet của scene được recall vào `content_blocks` loại `image`, caption có
timecode. Giai đoạn UI player có thể dùng chính `manifest_ref` để deep-link
tới video gốc và seek đến `start_ms`.

### 8.2. Rerank

Khi chưa cấu hình reranker, client phải gửi `enable_rerank=false` hoặc set:

```dotenv
RERANK_BY_DEFAULT=False
```

để không còn warning “rerank is enabled but no rerank model is configured”.
Đây không làm ingest thất bại nhưng làm kết quả query khó đánh giá.

Sau khi có benchmark, có thể thêm Qwen3-VL-Reranker-2B như một model-serving
service độc lập và dùng nó để rerank text scene. Việc này là follow-up riêng:
phải đo VRAM khi ASR/TTS/embedding/generation cùng chạy và phải contract-test
endpoint `/v1/rerank` tương thích binding `cohere` của LightRAG. Không thêm
model này trong implementation đầu chỉ vì feature video.

### 8.3. Phase 2: direct visual vector recall

Phase 2 dùng chính Qwen3-VL-Embedding-8B để tạo một vector cho contact sheet
(và có thể mixed input: transcript ngắn + contact sheet). Trước khi code phải
contract-test chính version vLLM đang serving với request multimodal
`/v1/embeddings`; không giả định format từ một version vLLM khác.

Index mới phải là bảng/repository riêng `video_scene_vectors`, có workspace,
`doc_id`, `scene_id`, `start_ms`, `end_ms`, asset ref và vector. Không nhét
vector ảnh vào collection chunk hiện có vì schema/lifecycle khác nhau.

Ở query-time, chạy text chunk recall và visual scene recall song song, sau đó
fuse bằng reciprocal-rank fusion (RRF) trước khi hydrate chunk/context. RRF
tránh phải giả định cosine score của text vector và image vector có cùng scale.
Phase này không đổi model service và không cần Docker network mới.

## 9. Concurrency, tài nguyên và failure semantics

| Stage | Resource | Concurrency mặc định | Quy tắc |
| --- | --- | ---: | --- |
| ffprobe / FFmpeg / scene detection | CPU + disk | 1 video | Chạy trong `video` queue riêng |
| ASR | Qwen3-ASR GPU | 1 request | Segment tuần tự trong từng parser worker |
| VLM scene analysis | Qwen3.5 GPU | `MAX_PARALLEL_ANALYZE=2` | Không tăng chỉ vì video |
| Entity/KG/query LLM | Qwen3.5 GPU | `MAX_ASYNC_LLM=2` | Chia chung generation service với VLM |
| Embedding | Qwen3-VL-Embedding GPU | Config hiện có | Chỉ embed text scene ở Phase 1 |

Quy tắc failure:

- FFprobe không đọc được, không có video stream, zero contact sheet hoặc source
  vượt duration limit: document `FAILED`.
- Không có audio stream: `PROCESSED` với warning, visual-only.
- ASR lỗi: giữ document ở trạng thái parse được, ghi warning theo scene và tiếp
  tục index phần visual; lỗi media/FFmpeg hoặc không tạo được contact sheet mới
  làm document `FAILED`.
- Một frame không extract được nhưng scene vẫn có contact sheet hợp lệ: tiếp tục
  và ghi warning; nếu không còn contact sheet nào thì `FAILED`.
- VLM failure do pipeline hiện có quản lý và ghi `llm_analyze_result` failure;
  transcript vẫn có thể tiếp tục qua indexing. Retry/re-analyze dùng cơ chế
  multimodal chuẩn, không re-run ASR trừ khi document được parse lại.

Khi thực hiện delete document với `delete_file=true`, source đã archive,
sidecar, contact sheet, manifest và transcript artifact phải bị xoá cùng nhau
theo lifecycle `__parsed__` hiện tại.

## 10. Kế hoạch code theo commit nhỏ

1. **Foundation**: constants, registry `video`, config parser, supported suffix
   tests và Dockerfile có FFmpeg/font.
2. **Media core**: ffprobe, duration guard, FFmpeg audio/frame extraction,
   scene planning/contact sheet unit tests.
3. **ASR adapter**: `VideoAsrClient`, contract test với Qwen3-ASR, retry và
   transcript manifest.
4. **Sidecar parser**: `VideoParser`, IR builder, assets/extras/manifest,
   parser debug CLI test.
5. **VLM integration**: prompt riêng cho contact sheet, multimodal analysis
   test, E2E ingest video ngắn.
6. **Query evidence**: trả image/timecode trong content blocks và kiểm thử
   reference.
7. **Benchmark/rollout**: kiểm thử tải, quality set, sau đó mới quyết định
   reranker hoặc Phase 2 visual vector index.

Không gộp Phase 2 hoặc reranker vào các commit 1–6. Điều đó giữ thay đổi
storage/query có rủi ro cao độc lập với chức năng ingest cốt lõi.

## 11. Kiểm thử và tiêu chí nghiệm thu

### 11.1. Unit test

- routing chấp nhận sáu suffix video, từ chối suffix không thuộc engine;
- `MAX_PARALLEL_PARSE_VIDEO` tạo queue riêng với đúng worker count;
- ffprobe parsing, rotation, no-audio, duration-limit, invalid codec metadata;
- scene de-dup, min/max duration, scene cap và deterministic budget allocation;
- contact sheet có đúng frame timestamp, kích thước và byte limit;
- ASR payload/timeout/fallback, segment ordering và degraded warning;
- sidecar có `IRDrawing.extras`, heading timecode, manifest ref và source asset;
- lỗi ở một video không làm document cùng batch bị fail.

### 11.2. Integration test

Tạo fixture video nhỏ có speech, scene cut, slide/OCR và một fixture silent
video. Dùng mock HTTP cho unit/integration CI; test thật với Qwen services chỉ
được đánh dấu integration và chạy chủ động.

Acceptance tối thiểu cho test thật:

1. Upload `mp4` được nhận, status cuối là `PROCESSED`.
2. `*.blocks.jsonl`, `*.drawings.json`, manifest, transcript và contact sheet
   tồn tại trong `__parsed__`.
3. Transcript có các segment time range tăng dần; visual-only fixture có warning
   đúng thay vì fail.
4. VLM result có description cho ít nhất một scene.
5. Query theo câu nói và query theo nội dung hình ảnh đều recall đúng scene;
   reference chứa video filename và timecode.
6. Hai video upload đồng thời không tạo hơn một ASR request đang chạy.

### 11.3. Quality benchmark trước production

Xây một tập kiểm thử nhỏ, đa dạng: lecture/slide, screen recording, interview,
demo thao tác, video không lời và video tiếng Việt/Anh. Mỗi câu hỏi có đáp án
`video + start_ms/end_ms`. Theo dõi ít nhất:

- scene timecode Recall@K;
- transcript answer Recall@K;
- visual/OCR answer Recall@K;
- tỷ lệ scene có hallucination rõ ràng;
- thời gian ingest theo phút video, số ASR segment, số VLM call, GPU queue time.

Chỉ tăng `VIDEO_MAX_SCENES`, giảm `VIDEO_SCENE_MAX_SECONDS` hoặc thêm reranker
sau khi đo các metric này; không tinh chỉnh mù theo một video mẫu.

## 12. Vận hành và bảo mật

- API key chỉ nằm trong `.env` runtime/secret store, không đưa vào docs, log,
  manifest, sidecar hoặc test fixture.
- `host.docker.internal` là routing nội bộ theo host gateway, không phải lý do
  để expose API key hoặc port ra Internet. Firewall/NSG của host phải hạn chế
  các cổng serving nếu có khả năng truy cập ngoài server.
- Video và contact sheet có thể chứa dữ liệu nhạy cảm. Áp dụng retention,
  backup encryption và quyền đọc `data/inputs`/`__parsed__` tương đương source
  video gốc.
- Log chỉ ghi `doc_id`, duration, scene/segment count, model id và lỗi đã được
  redaction; không log base64 audio/image hay transcript đầy đủ ở INFO level.

## 13. Tài liệu liên quan

- [File Processing Pipeline](./FileProcessingPipeline.md)
- [LightRAG Sidecar Format](./LightRAGSidecarFormat.md)
- [Paragraph Semantic Chunking](./ParagraphSemanticChunking.md)
- [Role-Specific LLM/VLM Configuration](./RoleSpecificLLMConfiguration.md)
- [Qwen3-VL-Embedding](https://github.com/QwenLM/Qwen3-VL-Embedding)
