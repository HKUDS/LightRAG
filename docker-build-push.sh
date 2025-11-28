#!/bin/bash
set -e

# Configuration
IMAGE_NAME="ghcr.io/clssck/lightrag"
DOCKERFILE="Dockerfile"
TAG="latest"

# Get version from git tags
VERSION=$(git describe --tags --abbrev=0 2>/dev/null || echo "dev")

echo "=================================="
echo "  Multi-Architecture Docker Build"
echo "=================================="
echo "Image: ${IMAGE_NAME}:${TAG}"
echo "Version: ${VERSION}"
echo "Platforms: linux/amd64, linux/arm64"
echo "=================================="
echo ""

# Check Docker login status (skip if CR_PAT is set for CI/CD)
if [ -z "$CR_PAT" ]; then
    if ! docker info 2>/dev/null | grep -q "Username"; then
        echo "⚠️  Warning: Not logged in to Docker registry"
        echo "Please login first: docker login ghcr.io"
        echo "Or set CR_PAT environment variable for automated login"
        echo ""
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
else
    echo "Using CR_PAT environment variable for authentication"
fi

# Check if buildx builder exists, create if not
if ! docker buildx ls | grep -q "desktop-linux"; then
    echo "Creating buildx builder..."
    docker buildx create --name desktop-linux --use
    docker buildx inspect --bootstrap
else
    echo "Using existing buildx builder: desktop-linux"
    docker buildx use desktop-linux
fi

echo ""
echo "Building and pushing multi-architecture image..."
echo ""

# Build and push
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --file ${DOCKERFILE} \
  --tag ${IMAGE_NAME}:${TAG} \
  --tag ${IMAGE_NAME}:${VERSION} \
  --push \
  .

echo ""
echo "✓ Build and push complete!"
echo ""
echo "Images pushed:"
echo "  - ${IMAGE_NAME}:${TAG}"
echo "  - ${IMAGE_NAME}:${VERSION}"
echo ""
echo "Verifying multi-architecture manifest..."
echo ""

# Verify
docker buildx imagetools inspect ${IMAGE_NAME}:${TAG}

echo ""
echo "✓ Verification complete!"
echo ""
echo "Pull with: docker pull ${IMAGE_NAME}:${TAG}"
