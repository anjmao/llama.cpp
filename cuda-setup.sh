#!/usr/bin/env bash
set -euo pipefail

# 1. Install build tools AND OpenSSL development headers
if ! command -v cmake &>/dev/null || ! command -v g++ &>/dev/null || ! dpkg -l | grep -q libssl-dev; then
    echo "==> Installing build tools and OpenSSL headers..."
    apt update && apt install -y build-essential cmake pkg-config wget libssl-dev
else
    echo "==> Build dependencies already installed."
fi

# 2. Check for CUDA Toolkit (nvcc) and install via APT if missing
if ! command -v nvcc &>/dev/null && [ ! -x "/usr/local/cuda/bin/nvcc" ]; then
    echo "==> Setting up NVIDIA APT repository..."
    
    if [ ! -f /usr/share/keyrings/cuda-archive-keyring.gpg ]; then
        wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
        dpkg -i cuda-keyring_1.1-1_all.deb
        rm -f cuda-keyring_1.1-1_all.deb
    fi

    echo "==> Installing CUDA Toolkit 12-8..."
    apt update
    apt install -y cuda-toolkit-12-8
else
    echo "==> CUDA Toolkit already installed, skipping."
fi

# 3. Environment path setup
export PATH="/usr/local/cuda/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

# 4. Clean old CMake build cache to wipe out stale BoringSSL/LibreSSL references
if [ -d "build/CMakeCache.txt" ] || [ -f "build/CMakeCache.txt" ]; then
    echo "==> Wiping stale CMake cache..."
    rm -rf build
fi

# 5. Configure llama.cpp
# Target Ada Lovelace (89-real) for GCP L4 GPU
echo "==> Configuring llama.cpp build..."
cmake -B build \
    -DCMAKE_BUILD_TYPE=Debug \
    -DGGML_CUDA=ON \
    -DLLAMA_OPENSSL=ON \
    -DCMAKE_CUDA_ARCHITECTURES=89-real

# 6. Build llama.cpp
echo "==> Building llama.cpp..."
cmake --build build --config Debug -j "$(nproc)"

echo "==> Build complete."

echo "==> Starting llama-server..."
./build/bin/llama-server -hf unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M --host 0.0.0.0