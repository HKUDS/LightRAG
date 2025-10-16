# LightRAG Docker Deployment

A lightweight Knowledge Graph Retrieval-Augmented Generation system with multiple LLM backend support.

## 🚀 Preparation

### Clone the repository:

```bash
# Linux/MacOS
git clone https://github.com/HKUDS/LightRAG.git
cd LightRAG
```
```powershell
# Windows PowerShell
git clone https://github.com/HKUDS/LightRAG.git
cd LightRAG
```

### Configure your environment:

```bash
# Linux/MacOS
cp .env.example .env
# Edit .env with your preferred configuration
```
```powershell
# Windows PowerShell
Copy-Item .env.example .env
# Edit .env with your preferred configuration
```

LightRAG can be configured using environment variables in the `.env` file:

**Server Configuration**

- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 9621)

**LLM Configuration**

- `LLM_BINDING`: LLM backend to use (lollms/ollama/openai)
- `LLM_BINDING_HOST`: LLM server host URL
- `LLM_MODEL`: Model name to use

**Embedding Configuration**

- `EMBEDDING_BINDING`: Embedding backend (lollms/ollama/openai)
- `EMBEDDING_BINDING_HOST`: Embedding server host URL
- `EMBEDDING_MODEL`: Embedding model name

**RAG Configuration**

- `MAX_ASYNC`: Maximum async operations
- `MAX_TOKENS`: Maximum token size
- `EMBEDDING_DIM`: Embedding dimensions

## 🐳 Docker Deployment

Docker instructions work the same on all platforms with Docker Desktop installed.

### Start LightRAG  server:

```bash
docker-compose up -d
```

LightRAG Server uses the following paths for data storage:

```
data/
├── rag_storage/    # RAG data persistence
└── inputs/         # Input documents
```

### Updates

To update the Docker container:
```bash
docker-compose pull
docker-compose down
docker-compose up
```

### Offline docker deployment

LightRAG provide an docker image can be deployment in offline environments where internet access is limited or unavailable. All you need to do is modify `docker-compose.yml`: change image tag from `latest` to `offline`.

> The offline Docker image does not include `docling` because its large size made integration into LightRAG internally impractical. This approach has been deprecated, and `docling` will be provided as a standalone service moving forward.

## 📦 Build Multi-Architecture Docker Images

### Prerequisites

Before building multi-architecture images, ensure you have:

- Docker 20.10+ with Buildx support
- Sufficient disk space (20GB+ recommended for offline image)
- Registry access credentials (if pushing images)

### 1. Setup Buildx Builder

Create and configure a multi-architecture builder:

```bash
# Create a new buildx builder instance
docker buildx create --name multiarch-builder --use

# Start and verify the builder
docker buildx inspect --bootstrap

# Verify supported platforms
docker buildx inspect multiarch-builder
```

You should see support for `linux/amd64` and `linux/arm64` in the output.

### 2. Registry Authentication

#### For GitHub Container Registry (ghcr.io)

**Option 1: Using Personal Access Token**

1. Create a GitHub Personal Access Token:
   - Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
   - Generate new token with `write:packages` and `read:packages` permissions
   - Copy the token

2. Login to registry:
   ```bash
   echo "YOUR_GITHUB_TOKEN" | docker login ghcr.io -u YOUR_GITHUB_USERNAME --password-stdin
   ```

**Option 2: Using GitHub CLI**

```bash
gh auth token | docker login ghcr.io -u YOUR_GITHUB_USERNAME --password-stdin
```

#### For Docker Hub

```bash
docker login
# Enter your Docker Hub username and password
```

#### For Other Registries

```bash
docker login your-registry.example.com
# Enter your credentials
```

### 3. Build Commands

#### A. Local Build (No Push)

Build multi-architecture images locally without pushing to registry:

**Normal image:**
```bash
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --file Dockerfile \
  --tag ghcr.io/hkuds/lightrag:latest \
  --load \
  .
```

**Offline image:**
```bash
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --file Dockerfile.offline \
  --tag ghcr.io/hkuds/lightrag:offline \
  --load \
  .
```

> **Note**: `--load` loads the image to local Docker, but only supports single platform. For multi-platform, use `--push` instead.

#### B. Build and Push to Registry

Build and directly push to container registry:

**Normal image:**
```bash
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --file Dockerfile \
  --tag ghcr.io/hkuds/lightrag:latest \
  --push \
  .
```

**Offline image:**
```bash
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --file Dockerfile.offline \
  --tag ghcr.io/hkuds/lightrag:offline \
  --push \
  .
```

#### C. Build with Multiple Tags

Add version tags alongside latest:

```bash
# Get version from git tag
VERSION=$(git describe --tags --abbrev=0 2>/dev/null || echo "v1.0.0")

# Build with multiple tags
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --file Dockerfile.offline \
  --tag ghcr.io/hkuds/lightrag:offline \
  --tag ghcr.io/hkuds/lightrag:${VERSION}-offline \
  --tag ghcr.io/hkuds/lightrag:offline-latest \
  --push \
  .
```

### 4. Verify Built Images

After building, verify the multi-architecture manifest:

```bash
# Inspect image manifest
docker buildx imagetools inspect ghcr.io/hkuds/lightrag:offline

# Expected output shows multiple platforms:
# Name:      ghcr.io/hkuds/lightrag:offline
# MediaType: application/vnd.docker.distribution.manifest.list.v2+json
# Platforms: linux/amd64, linux/arm64
```

Pull and test specific architectures:

```bash
# Pull AMD64 version
docker pull --platform linux/amd64 ghcr.io/hkuds/lightrag:offline

# Pull ARM64 version
docker pull --platform linux/arm64 ghcr.io/hkuds/lightrag:offline

# Test run
docker run --rm -p 9621:9621 ghcr.io/hkuds/lightrag:offline
```

### 5. Troubleshooting

#### Build Time is Very Slow

**Cause**: Building ARM64 on AMD64 (or vice versa) requires QEMU emulation, which is slower.

**Solutions**:
- Use remote cache (`--cache-from/--cache-to`) for faster subsequent builds
- Build on native architecture when possible
- Be patient - initial multi-arch builds take 30-60 minutes

#### "No space left on device" Error

**Cause**: Insufficient disk space for build layers and cache.

**Solutions**:
```bash
# Clean up Docker system
docker system prune -a

# Clean up buildx cache
docker buildx prune

# Check disk space
df -h
```

#### "failed to solve: failed to push" Error

**Cause**: Not logged into the registry or insufficient permissions.

**Solutions**:
1. Verify you're logged in: `docker login ghcr.io`
2. Check you have push permissions to the repository
3. Verify the image name matches your repository path

#### Builder Not Found

**Cause**: Buildx builder not created or not set as current.

**Solutions**:
```bash
# List builders
docker buildx ls

# Create and use new builder
docker buildx create --name multiarch-builder --use

# Or switch to existing builder
docker buildx use multiarch-builder
```

### 6. Cleanup

Remove builder when done:

```bash
# Switch back to default builder
docker buildx use default

# Remove multiarch builder
docker buildx rm multiarch-builder

# Prune build cache
docker buildx prune
```

### 7. Best Practices

1. **Use specific tags**: Avoid only using `latest`, include version tags
2. **Verify platforms**: Always check the manifest after pushing
4. **Monitor resources**: Ensure sufficient disk space before building
5. **Test both architectures**: Pull and test each platform variant
6. **Use .dockerignore**: Exclude unnecessary files to speed up build context transfer

### 9. Build Script Example

For convenience, create a build script `build-and-push.sh`:

```bash
#!/bin/bash
set -e

# Configuration
IMAGE_NAME="ghcr.io/hkuds/lightrag"
DOCKERFILE="Dockerfile.offline"
TAG="offline"

# Get version
VERSION=$(git describe --tags --abbrev=0 2>/dev/null || echo "dev")

echo "Building ${IMAGE_NAME}:${TAG} (version: ${VERSION})"

# Create builder if not exists
if ! docker buildx inspect multiarch-builder &>/dev/null; then
    echo "Creating buildx builder..."
    docker buildx create --name multiarch-builder --use
    docker buildx inspect --bootstrap
else
    docker buildx use multiarch-builder
fi

# Build and push
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --file ${DOCKERFILE} \
  --tag ${IMAGE_NAME}:${TAG} \
  --tag ${IMAGE_NAME}:${VERSION}-${TAG} \
  --push \
  .

echo "✓ Build complete!"
echo "Image pushed: ${IMAGE_NAME}:${TAG}"
echo "Version tag: ${IMAGE_NAME}:${VERSION}-${TAG}"

# Verify
docker buildx imagetools inspect ${IMAGE_NAME}:${TAG}
```

Make it executable and run:

```bash
chmod +x build-and-push.sh
./build-and-push.sh
```
