# 解析服务本地部署

MinerU 与 Docling 都是**外部服务**：LightRAG 通过 HTTP 与它们通信，不会在进程内运行它们的模型。只有当你把文件路由到 `mineru` 或 `docling` 引擎、并且希望自行搭建该服务（而不是使用现成的服务端点）时，才需要本文档。

本文档内容全部是这两个上游项目的容器侧配置，其中没有任何 LightRAG 环境变量。LightRAG 侧的配置（哪个扩展名交给哪个引擎、服务端点、凭据、引擎参数）请见 [FileProcessingPipeline-zh.md](./FileProcessingPipeline-zh.md)。

## 1. 本地部署 MinerU 服务


从 Github官方仓库   [opendatalab/MinerU](https://github.com/opendatalab/MinerU) 把 Dockerfile 和 compose.yaml 拷贝到本地。这两个文件应该在仓库的 docker 目录可以找到。针对中国供应商的特殊显卡需要选择相应的 Dockerfile 。

准备好上诉两个文件后通过以下命令构建 docker 镜像:

```bash
docker build --tag mineru:latest .
```

镜像构建好之后通过以下命令启动 API 服务（参数 `--profile api` 标识仅启动MinerU的 API 服务，服务默认监听 8000 端口）：

```bash
docker compose -f compose.yaml --profile api up -d
```

镜像构建细节、GPU 驱动准备、模型权重位置等请参考官方 README：<https://github.com/opendatalab/MinerU>。

**进阶配置：开启 vLLM 预加载与标题层级修正（可选）**

在基础部署之上，建议为本地 MinerU 额外开启两项 MinerU **服务端**功能。这两项都改的是 MinerU 容器侧配置（容器内 `mineru.json` 与官方 `compose.yaml`），不涉及 LightRAG 的 env 变量；其中标题层级修正还需要一个可用的 LLM API。

- **vLLM 启动预加载**：让容器启动时就把 VLM 模型加载进显存，避免首个解析请求承担模型加载延迟。
- **标题层级修正（`title_aided`）**：MinerU 借助一个外部 LLM 修正解析输出的标题层级，提升结构化产物质量。这对依赖标题结构的 [P（段落语义）分块策略](./FileProcessingPipeline-zh.md#21-文件处理选项)尤其有帮助；`P分块策略` 优先按标题分割，标题层级越准确，分块语义越好。

**步骤1：导出并修改 `mineru-lightrag.json`**

从官方镜像中把 `/root/mineru.json` 拷到宿主机当前目录的 `mineru-lightrag.json`（用固定容器名 `temp_mineru`，无需运行容器）：

```bash
docker create --name temp_mineru mineru:latest
docker cp temp_mineru:/root/mineru.json ./mineru-lightrag.json
docker rm temp_mineru
```

然后修改 `mineru-lightrag.json` 中的 `llm-aided-config.title_aided`：填入 `api_key`，并把 `enable` 改为 `true`：

```json
"llm-aided-config": {
    "title_aided": {
        "api_key": "your_api_key",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen3.5-plus",
        "enable_thinking": false,
        "enable": true
    }
}
```

> `api_key` / `base_url` / `model` 需替换为用户自己可用的 LLM 服务（示例使用阿里云 DashScope 的 OpenAI 兼容接口）。

**步骤2：修改官方 `compose.yaml` 的 `api` profile 服务（`mineru-api`）**

在 `mineru-api` 服务上做三处改动：`environment` 增加 `MINERU_TOOLS_CONFIG_JSON`（让 MinerU 读改过的配置而非镜像内置 `mineru.json`），`volumes` 把宿主机 `mineru-lightrag.json` 挂进容器，`command` 追加 `--enable-vlm-preload true` 开启 vLLM 预加载。改好后的完整 `mineru-api` profile 如下（以 `# <-- 新增` 标注三处增量）：

```yaml
  mineru-api:
    image: mineru:latest
    container_name: mineru-api
    restart: always
    profiles: ["api"]
    ports:
      - 8000:8000
    environment:
      MINERU_MODEL_SOURCE: local
      MINERU_TOOLS_CONFIG_JSON: /root/mineru-lightrag.json   # <-- Added
    volumes:
      - ./mineru-lightrag.json:/root/mineru-lightrag.json    # <-- Added
    entrypoint: mineru-api
    command:
      --host 0.0.0.0
      --port 8000
      --allow-public-http-client
      --gpu-memory-utilization 0.45                          # Reserved 10GB is fine, preventing OOM errors
      --enable-vlm-preload true                              # <-- Added
    ulimits:
      memlock: -1
      stack: 67108864
    ipc: host
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["0"]
              capabilities: [gpu]
```

> 示范中请按实际显卡情况调整 `gpu-memory-utilization` ；`environment` / `volumes` / `command` 三处为本次新增项，其余保持官方原样。

**步骤3：重启生效**

改完后重新启动 API 服务让改动生效：

```bash
docker compose -f compose.yaml --profile api up -d
```

## 2. 本地部署 docling-serve（启用 LaTeX 公式识别）


下面以 Docker 部署 docling-serve 为例，给出从镜像下载到模型挂载的完整步骤，部署完成后将 `DOCLING_DO_FORMULA_ENRICHMENT=true` 写入 LightRAG 的 `.env` 即可启用 LaTeX 公式识别。

> **重要提示**：以下步骤基于显卡支持 CUDA 13 的环境。如果显卡较老旧、不支持 CUDA 13，需要把命令与 compose 文件中的镜像名 `docling-serve-cu130:main` 替换为对应 CUDA 版本的标签。可选镜像列表参见 [docling-serve Packages](https://github.com/orgs/docling-project/packages?repo_name=docling-serve)。

**1. 下载镜像**

```bash
docker pull ghcr.io/docling-project/docling-serve-cu130:main
```

**2. 下载模型**

```bash
# 创建 docling 工作目录
mkdir docling
cd docling

# 创建模型挂载目录
mkdir models

# 把容器内的原有模型拷贝到 models 目录
docker run --rm -it \
  -v "$(pwd)/models:/opt/app-root/src/models" \
  ghcr.io/docling-project/docling-serve-cu130:main \
  cp -r /opt/app-root/src/.cache/docling/models /opt/app-root/src/

# 下载公式识别模型
docker run --rm \
  -v "$(pwd)/models:/opt/app-root/src/models" \
  -e DOCLING_SERVE_ARTIFACTS_PATH="/opt/app-root/src/models" \
  ghcr.io/docling-project/docling-serve-cu130:main \
  docling-tools models download-hf-repo docling-project/CodeFormulaV2 -o models
```

**3. 创建 `docker-compose.yaml` 文件**

在上一步的 `docling` 目录下创建 `docker-compose.yaml`，内容如下：

```yaml
services:
  docling-serve:
    image: ghcr.io/docling-project/docling-serve-cu130:main
    container_name: docling-serve
    ports:
      - "5001:5001"
    environment:
      DOCLING_SERVE_ENABLE_UI: "true"
      NVIDIA_VISIBLE_DEVICES: "all"
      DOCLING_SERVE_ARTIFACTS_PATH: "/opt/app-root/src/models"
    # deploy:  # This section is for compatibility with Swarm
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: all
    #           capabilities: [gpu]
    runtime: nvidia
    restart: always
    volumes:
      - ./models:/opt/app-root/src/models
```

随后在该目录执行 `docker compose up -d` 启动服务。容器就绪后，在 LightRAG 的 `.env` 中设置：

```bash
DOCLING_ENDPOINT=http://localhost:5001
DOCLING_DO_FORMULA_ENRICHMENT=true
```

即可让 LightRAG 通过本地 docling-serve 识别文档中的公式并以 LaTeX 形式输出。
