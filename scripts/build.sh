#!/usr/bin/env bash
set -euo pipefail

# Build llama.cpp with HauhauCS FastMTP patch on AMD ROCm/HIP
COMMIT="4df29be4f4c3673f428170fda944a5b19f743bb8"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH_FILE="${ROOT_DIR}/patches/HauhauCS-FastMTP-llama.cpp.patch"
BUILD_DIR="${ROOT_DIR}/build/llama.cpp"

if [[ ! -f "${PATCH_FILE}" ]]; then
    echo "Error: Patch file not found at ${PATCH_FILE}" >&2
    exit 1
fi

mkdir -p "${ROOT_DIR}/build"
if [[ ! -d "${BUILD_DIR}/.git" ]]; then
    echo "Cloning llama.cpp at commit ${COMMIT}..."
    git clone https://github.com/ggerganov/llama.cpp "${BUILD_DIR}"
    cd "${BUILD_DIR}"
    git checkout "${COMMIT}"
    echo "Applying FastMTP patch..."
    git apply --check "${PATCH_FILE}"
    git apply "${PATCH_FILE}"
else
    cd "${BUILD_DIR}"
fi

echo "Configuring CMake with HIP for gfx1100 (RX 7900 XTX / XT)..."
cmake -B build -G Ninja \
    -DGGML_HIP=ON \
    -DAMDGPU_TARGETS=gfx1100 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=/opt/rocm/bin/hipcc \
    -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc

echo "Compiling llama-server and llama-cli..."
ninja -C build llama-server llama-cli

echo "Build complete. Executables located in: ${BUILD_DIR}/build/bin/"
