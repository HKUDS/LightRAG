# Parser Service Deployment

MinerU and Docling are **external services**: LightRAG talks to them over HTTP and never runs their models in-process. You need this document only if you route files to the `mineru` or `docling` engine and want to host that service yourself instead of using a hosted endpoint.

Everything here is container-side configuration for those upstream projects. Nothing on this page is a LightRAG environment variable — for the LightRAG side (which engine handles which extension, endpoints, credentials, per-engine options) see [FileProcessingPipeline.md](./FileProcessingPipeline.md).

## 1. Local Deployment of the MinerU Service


Copy `Dockerfile` and `compose.yaml` from the official GitHub repository [opendatalab/MinerU](https://github.com/opendatalab/MinerU) to your local machine. Both files can be found in the repository's `docker` directory. For special GPUs from Chinese vendors, you need to choose the corresponding `Dockerfile`.

After preparing the two files above, build the Docker image with the following command:

```bash
docker build --tag mineru:latest .
```

Once the image is built, start the API service with the following command (the `--profile api` parameter indicates starting only MinerU's API service; the service listens on port 8000 by default):

```bash
docker compose -f compose.yaml --profile api up -d
```

For image build details, GPU driver setup, model weight locations, etc., refer to the official README: <https://github.com/opendatalab/MinerU>.

**Advanced configuration: enabling vLLM preload and title-level correction (optional)**

On top of the basic deployment, it is recommended to additionally enable two MinerU **server-side** features for your local MinerU. Both modify MinerU container-side configuration (the in-container `mineru.json` and the official `compose.yaml`), and do not involve any LightRAG env variable; title-level correction additionally requires an available LLM API.

- **vLLM startup preload**: loads the VLM model into GPU memory at container startup, avoiding the model-loading latency on the first parse request.
- **Title-level correction (`title_aided`)**: MinerU uses an external LLM to correct the title hierarchy of the parsed output, improving the quality of the structured artifacts. This is especially helpful for the [P (paragraph semantic) chunking strategy](./FileProcessingPipeline.md#21-file-processing-options), which depends on the title structure; the `P` chunking strategy splits by titles first, so the more accurate the title hierarchy, the better the chunking semantics.

**Step 1: Export and modify `mineru-lightrag.json`**

Copy `/root/mineru.json` from the official image to `mineru-lightrag.json` in the host's current directory (using the fixed container name `temp_mineru`, without running the container):

```bash
docker create --name temp_mineru mineru:latest
docker cp temp_mineru:/root/mineru.json ./mineru-lightrag.json
docker rm temp_mineru
```

Then modify `llm-aided-config.title_aided` in `mineru-lightrag.json`: fill in `api_key` and change `enable` to `true`:

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

> `api_key` / `base_url` / `model` should be replaced with an LLM service available to you (the example uses Alibaba Cloud DashScope's OpenAI-compatible endpoint).

**Step 2: Modify the `api` profile service (`mineru-api`) in the official `compose.yaml`**

Make three changes to the `mineru-api` service: add `MINERU_TOOLS_CONFIG_JSON` to `environment` (so MinerU reads the modified config instead of the image's built-in `mineru.json`), mount the host's `mineru-lightrag.json` into the container via `volumes`, and append `--enable-vlm-preload true` to `command` to enable vLLM preload. The complete `mineru-api` profile after modification is as follows (the three increments are marked with `# <-- added`):

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
      MINERU_TOOLS_CONFIG_JSON: /root/mineru-lightrag.json   # <-- added
    volumes:
      - ./mineru-lightrag.json:/root/mineru-lightrag.json    # <-- added
    entrypoint: mineru-api
    command:
      --host 0.0.0.0
      --port 8000
      --allow-public-http-client
      --gpu-memory-utilization 0.45         #
      --enable-vlm-preload true             # <-- added
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
              device_ids: ["0"]  # For multiple GPUs: ["0", "1"]
              capabilities: [gpu]
```

> In the example, adjust `gpu-memory-utilization` according to your actual GPU setup. The three items `environment` / `volumes` / `command` are the additions for this change; keep everything else as in the official file.

**Step 3: Restart to take effect**

After making the changes, restart the API service for them to take effect:

```bash
docker compose -f compose.yaml --profile api up -d
```

## 2. Local Deployment of docling-serve (LaTeX equation recognition)


The following uses a Docker-based docling-serve deployment as an example, giving the complete steps from image download to model mounting. After deployment completes, write `DOCLING_DO_FORMULA_ENRICHMENT=true` into LightRAG's `.env` to enable LaTeX equation recognition.

> **Important**: the steps below are based on an environment where the GPU supports CUDA 13. If your GPU is older and does not support CUDA 13, replace the image name `docling-serve-cu130:main` in the command and compose file with the tag corresponding to your CUDA version. For the list of available images, see [docling-serve Packages](https://github.com/orgs/docling-project/packages?repo_name=docling-serve).

**1. Pull the image**

```bash
docker pull ghcr.io/docling-project/docling-serve-cu130:main
```

**2. Download models**

```bash
# Create the docling working directory
mkdir docling
cd docling

# Create the model mount directory
mkdir models

# Copy the existing models inside the container into the models directory
docker run --rm -it \
  -v "$(pwd)/models:/opt/app-root/src/models" \
  ghcr.io/docling-project/docling-serve-cu130:main \
  cp -r /opt/app-root/src/.cache/docling/models /opt/app-root/src/

# Download the equation recognition model
docker run --rm \
  -v "$(pwd)/models:/opt/app-root/src/models" \
  -e DOCLING_SERVE_ARTIFACTS_PATH="/opt/app-root/src/models" \
  ghcr.io/docling-project/docling-serve-cu130:main \
  docling-tools models download-hf-repo docling-project/CodeFormulaV2 -o models
```

**3. Create `docker-compose.yaml`**

Create `docker-compose.yaml` in the `docling` directory from the previous step, with the following contents:

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

Then execute `docker compose up -d` in that directory to start the service. After the container is ready, set the following in LightRAG's `.env`:

```bash
DOCLING_ENDPOINT=http://localhost:5001
DOCLING_DO_FORMULA_ENRICHMENT=true
```

This enables LightRAG to recognize equations in documents via the local docling-serve and output them in LaTeX form.
