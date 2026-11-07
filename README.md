# qwen-fastmtp-rocm

Reproducible recipe and build configuration for running Qwen 3.8 27B with HauhauCS FastMTP speculative decoding on AMD Radeon RX 7900 XTX (24 GB) under ROCm.

## Background

On an AMD Radeon RX 7900 XTX (24 GB VRAM, 960 GB/s memory bandwidth), pure autoregressive generation with a 27B model at Q4 (~17 GB) is memory-bandwidth bound to ~34 to 40 tokens/s:

$$\text{Throughput} \le \frac{960\text{ GB/s}}{17\text{ GB}} \approx 56.4\text{ tokens/s (theoretical ceiling)}$$

In practice, single-token forward evaluation runs at 29 to 33 ms per step (~34 tokens/s).

Reaching 70 to 110 tokens/s decode speed requires multi-token speculative decoding. HauhauCS FastMTP provides an 862 MB Multi-Token Prediction (MTP) draft sidecar (`FastMTP-32K.gguf`) tailored for Qwen 3.8 27B. Because FastMTP truncates its output projection head to 32,768 tokens (down from Qwen's native 248,320 vocabulary), stock `llama.cpp` refuses to load the draft tensor with a shape mismatch:

```
check_tensor_dims: tensor 'output.weight' has wrong shape; expected 5120, 248320, got 5120, 32768
```

The patch in `patches/HauhauCS-FastMTP-llama.cpp.patch` adds `d2t` draft-vocabulary trimming and logit scattering to `src/models/qwen35.cpp`, enabling speculative drafting at depth 3 to 5 with 80% to 92% acceptance.

## Measured Benchmarks

Measured on a single AMD Radeon RX 7900 XTX (Navi 31, 24 GB, `gfx1100`, ROCm 7.2):

| Metric | Raw Baseline (No MTP) | FastMTP (Depth 3) | Sustained Boost (Full PPT1) |
| :--- | :--- | :--- | :--- |
| Prompt Evaluation (pp512) | 1,034.5 t/s | 1,034.5 t/s | 1,034.5 t/s |
| Code / Document Decode | 34.3 t/s | 66.4 to 70.3 t/s | 107 to 110 t/s |
| Reasoning Decode (Thinking) | 34.3 t/s | 48.0 t/s | ~75 t/s |
| Time to First Token (TTFT) | ~0.50 s | 0.16 s | 0.16 s |
| Draft Acceptance Rate | N/A | 81.8% to 89.2% | ~90% |
| VRAM Usage | 16.7 GB | 22.6 GB / 24.5 GB | 22.6 GB / 24.5 GB |

Draft acceptance reaches 89.2% on structured code and repetitive text blocks, producing 3.68 accepted tokens per forward verification step.

## Prerequisites

- Linux (x86_64, kernel 6.x+)
- AMD ROCm SDK 6.x or 7.x (`hipcc`, `rocblas`, `hipblas`)
- CMake 3.28+, Ninja, Python 3.10+
- AMD Radeon RX 7900 XTX or 7900 XT (Navi 31)
- LACT or `rocm-smi` configured for performance mode

## Quick Start

### 1. Download Model and Draft Sidecar

Obtain the weights from Hugging Face (`HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF`):

```bash
mkdir -p models
cd models

# Target model (Q4_K_P or Q4_K_M)
curl -L -O https://huggingface.co/HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF/resolve/main/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf

# FastMTP 32K draft sidecar
curl -L -O https://huggingface.co/HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF/resolve/main/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-FastMTP-32K.gguf
cd ..
```

### 2. Build the Patched Engine

Run the build script:

```bash
./scripts/build.sh
```

Or manually:

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
git checkout 4df29be4f4c3673f428170fda944a5b19f743bb8
git apply ../patches/HauhauCS-FastMTP-llama.cpp.patch

cmake -B build -G Ninja \
    -DGGML_HIP=ON \
    -DAMDGPU_TARGETS=gfx1100 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=/opt/rocm/bin/hipcc \
    -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc

ninja -C build llama-server llama-cli
```

### 3. GPU Power Setup (LACT / rocm-smi)

Ensure the GPU power ceiling and clocks are unlocked:

```bash
# Using rocm-smi
rocm-smi -d 0 --setperflevel high

# Or via LACT config (/etc/lact/config.yaml):
# performance_level: manual
# power_cap: 402.0
# min_memory_clock: 1300
# max_memory_clock: 1300
# power_profile_mode_index: 5 (Compute)
```

### 4. Run the Server

```bash
./build/llama.cpp/build/bin/llama-server \
  --model models/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf \
  --spec-draft-model models/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-FastMTP-32K.gguf \
  --spec-draft-ngl all \
  --spec-type draft-mtp \
  --spec-draft-n-max 3 \
  --spec-draft-p-min 0 \
  --ctx-size 32768 \
  --parallel 1 \
  --batch-size 2048 \
  --ubatch-size 512 \
  --n-gpu-layers all \
  --split-mode none \
  --flash-attn on \
  --no-mmap \
  --temp 0.2 \
  --top-k 20 \
  --top-p 0.95 \
  --jinja \
  --host 127.0.0.1 \
  --port 8080
```

### 5. Benchmark

Run the client script to assert streaming throughput and TTFT:

```bash
python3 scripts/bench.py --endpoint http://127.0.0.1:8080
```

## Systemd Service

A user unit template is provided in `systemd/qwen-uncensored.service`. Install and start:

```bash
cp systemd/qwen-uncensored.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now qwen-uncensored
systemctl --user status qwen-uncensored
```

## Repository Structure

```
.
├── patches/
│   └── HauhauCS-FastMTP-llama.cpp.patch   # d2t vocab trim and logit scatter
├── scripts/
│   ├── bench.py                           # Streaming TTFT and decode throughput tester
│   └── build.sh                           # Clone, patch, and build helper
├── systemd/
│   └── qwen-uncensored.service            # Systemd user service unit
└── README.md
```

## License

MIT. The patch applies to llama.cpp (MIT). Qwen 3.8 weights are governed by the Apache 2.0 license.
