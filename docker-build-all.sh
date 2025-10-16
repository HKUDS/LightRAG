#!/bin/bash
set -e

# Configuration
IMAGE_NAME="ghcr.io/hkuds/lightrag"
DOCKERFILE="Dockerfile"
TAG="latest"

# Get version
VERSION=$(git describe --tags --abbrev=0 2>/dev/null || echo "dev")

echo "Building ${IMAGE_NAME}:${TAG} (version: ${VERSION})"

# Build and push
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --file ${DOCKERFILE} \
  --tag ${IMAGE_NAME}:${TAG} \
  --tag ${IMAGE_NAME}:${VERSION} \
  --load \
  .

echo "✓ Build complete!"
echo "Image pushed: ${IMAGE_NAME}:${TAG}"
echo "Version tag: ${IMAGE_NAME}:${VERSION}"

# Verify
docker buildx imagetools inspect ${IMAGE_NAME}:${TAG}
