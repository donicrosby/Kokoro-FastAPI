#!/bin/bash
set -e

# Get version from argument or use default
VERSION=${1:-"latest"}

# Build CPU, GPU, ROCm, and OpenVINO images using docker buildx bake
# Use group: docker buildx bake cpu gpu  (default), or add rocm-amd64 openvino-amd64
echo "Building images (cpu, gpu, rocm-amd64, openvino-amd64)..."
VERSION=$VERSION docker buildx bake --push

echo "Build complete!"
echo "Created images with version: $VERSION"
