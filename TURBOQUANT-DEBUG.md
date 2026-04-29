# TurboQuant GPU Debug Guide

## What We've Done

### Commit History (branch: feature/turboquant-kv-cache)

```
92107f1 docs: add systemd service setup and library discovery notes
aefcdeb perf: wire CUDA WHT kernel and optimize turbo KQ dot products
7a487a4 chore: rename binary to ollama-tq, consolidate docs, update gitignore
17e29b6 fix: correct CUDA flash attention for TurboQuant KV cache types
6aecb28 docs: add GPU debug guide for turbo4 SIGABRT diagnosis
2225e1f feat: add CUDA turbo flash attention and set-rows WHT kernels
7f10cc2 feat: add CPU turbo implementation and CGo bindings
ba0a120 fix: reuse deprecated type slots 36-38 for TurboQuant (keeps COUNT=40)
cdac18f feat: TurboQuant KV cache integration (Go scaffolding)
87288ce New models (#15861)  ← upstream Ollama main
```

### Root Causes Found and Fixed

1. **Deprecated type_traits override (SIGFPE)**: ggml.c had hardcoded `[36]`, `[37]`, `[38]` entries for deprecated IQ4_NL types with `blck_size=0`. C designated initializer rule: last entry wins. Our turbo entries at the same indices were silently overridden → division by zero in `ggml_blck_size()`. **Fixed** by removing deprecated entries.

2. **GGML_TYPE_COUNT=45 (SIGSEGV)**: Increasing TYPE_COUNT from 40 to 45 changed array sizes throughout GGML, causing ABI mismatches between the Go binary (CGo-linked) and the CUDA shared library (dynamically loaded). **Fixed** by reusing deprecated slots 36-38 so COUNT stays at 40.

3. **CGo symbol leaking (SIGSEGV)**: The Go binary exported GGML symbols in its dynamic symbol table, conflicting with the CUDA library's copies. **Fixed** with `-buildmode=pie` and `hide-ggml.ver` version script.

4. **Double CUDA backend loading (SIGFPE)**: Both `build/lib/ollama/libggml-cuda.so` and `build/lib/ollama/cuda_v12/libggml-cuda.so` were loaded, causing duplicate initialization. **Fixed** by deleting the flat copy.

5. **cuda_v12 subdirectory layout**: Ollama's GPU discovery uses `filepath.Glob(path, "*", "*ggml-*")` which requires backend libraries in a subdirectory. **Fixed** by creating `cuda_v12/` with the CUDA and base libraries.

6. **CUDA architecture 120 for RTX 5090**: Missing SM 120 (Blackwell) architecture caused silent GPU discovery failure. **Fixed** by adding to CMAKE_CUDA_ARCHITECTURES.

7. **CUDA flash attention SIGABRT for turbo KV types**: The FA kernel selector dispatched turbo KV types to MMA/WMMA/tile kernels that lack turbo support, triggering `GGML_ABORT`. **Fixed** by forcing VEC kernel for turbo types in `fattn.cu`; returns NONE when the vector kernel is unavailable so the scheduler falls back.

8. **Garbled GPU output (Q/K element mismatch)**: The turbo KQ dot products in `fattn-common.cuh` used an interleaved K access pattern where threads within a sub-group accessed adjacent K elements but read mismatched Q values from thread-local registers. **Fixed** by restructuring all three turbo dot products to mirror the F16 pattern: each thread loads contiguous K elements at offset `(tid%nthreads)*cpy_ne`, matching Q_reg layout.

9. **5x speed regression (35 vs 175 t/s)**: `GGML_OP_TURBO_WHT` (Walsh-Hadamard Transform) had no CUDA backend dispatch — the existing `turbo-wht.cu` kernel was present but unwired. Every attention layer fell back to CPU for Q rotation + inverse output rotation, causing 56 PCIe round-trips per token. **Fixed** by adding the op dispatch and `supports_op` entry in `ggml-cuda.cu`.

10. **Turbo KQ dot product __constant__ memory serialization**: Centroid table lookups from `__constant__` memory serialized across warps (up to 16 serialized reads for turbo4's 16-entry table). **Fixed** by copying centroids to register arrays, bulk-loading packed `qs[]` bytes as uint32_t/uint16_t, and pre-scaling centroids by block norm.

### Current State

| Test | Result |
|------|--------|
| f16 GPU (RTX 4090) | **Working** — 175 t/s gen, 3791 t/s prompt |
| turbo4 GPU (RTX 4090) | **Working** — 163 t/s gen, 3297 t/s prompt (93% of f16) |
| turbo4 KV cache savings | 3.7x smaller (59.5 MiB vs 224 MiB for qwen2.5:7b @ 4K ctx) |
| turbo4 GPU layers (qwen3.6:27b @ 65K) | 64/65 on GPU (vs 56/65 with f16) |
| Go FA regression tests | 2/2 PASS |

## Build Commands

```bash
cd /home/yannik/Work/ollama-build

# 1. CMAKE
rm -rf build
cmake --preset "CUDA 12" -B build \
  -DCMAKE_DISABLE_FIND_PACKAGE_Vulkan=TRUE \
  -DCMAKE_CUDA_ARCHITECTURES="89;120"
cmake --build build -j$(nproc)

# 2. Library layout (CRITICAL)
mkdir -p build/lib/ollama/cuda_v12
cp build/lib/ollama/libggml-cuda.so build/lib/ollama/cuda_v12/
cp build/lib/ollama/libggml-base.so.0.0.0 build/lib/ollama/cuda_v12/libggml-base.so
rm build/lib/ollama/libggml-cuda.so  # prevent double backend loading

# 3. Go binary (CRITICAL flags)
go clean -cache  # ALWAYS clean before build — stale CGo cache causes crashes
CGO_LDFLAGS="-L$(pwd)/build/lib/ollama" \
  go build -trimpath -buildmode=pie \
  -ldflags='-extldflags "-Wl,--version-script=hide-ggml.ver"' \
  -o ollama-tq .

# 4. Install binary + library symlink
cp ollama-tq ~/.local/bin/ollama-tq
# Ollama discovers backends at <exe_dir>/../lib/ollama/ — symlink to build output:
ln -sfn "$(pwd)/build/lib/ollama" ~/.local/lib/ollama

# 5. Test (ad-hoc, uses a separate port)
pkill -f 'ollama-tq'  # kill ALL previous instances
CUDA_VISIBLE_DEVICES=0 OLLAMA_HOST=localhost:9999 \
  OLLAMA_KV_CACHE_TYPE=turbo4 OLLAMA_FLASH_ATTENTION=1 \
  OLLAMA_NEW_ENGINE=1 OLLAMA_CONTEXT_LENGTH=4096 \
  ~/.local/bin/ollama-tq serve
```

## Systemd Service

The `ollama-tq` systemd service runs TurboQuant on port 8001:

```bash
# Service file: /etc/systemd/system/ollama-tq.service
# Launch script: ~/.local/bin/ollama-serve-tq
# Binary: ~/.local/bin/ollama-tq
# Libraries: ~/.local/lib/ollama → ~/Work/ollama-build/build/lib/ollama (symlink)

sudo systemctl start ollama-tq       # start
sudo systemctl status ollama-tq      # check
journalctl -u ollama-tq -f           # logs
OLLAMA_HOST=localhost:8001 ollama ps  # verify GPU offloading
```

After rebuilding, install and restart:
```bash
cp ollama-tq ~/.local/bin/ollama-tq
sudo systemctl restart ollama-tq
```

## Files Reference

| File | Purpose |
|------|---------|
| `ml/backend/ggml/ggml/include/ggml.h` | Type enums (36-38), TURBO_WHT op |
| `ml/backend/ggml/ggml/src/ggml.c` | Type traits, ggml_turbo_wht() |
| `ml/backend/ggml/ggml/src/ggml-turbo-quant.c` | CPU WHT + PolarQuant |
| `ml/backend/ggml/ggml/src/ggml-common.h` | Block struct definitions |
| `ml/backend/ggml/ggml/src/ggml-cpu/ggml-cpu.c` | CPU type traits + vec_dot |
| `ml/backend/ggml/ggml/src/ggml-cuda/fattn.cu` | FA kernel selector (turbo → VEC only) |
| `ml/backend/ggml/ggml/src/ggml-cuda/fattn-common.cuh` | Turbo KQ dot products + V dequant |
| `ml/backend/ggml/ggml/src/ggml-cuda/fattn-vec.cuh` | Turbo constexprs + extern declarations |
| `ml/backend/ggml/ggml/src/ggml-cuda/turbo-wht.cu` | CUDA WHT kernel (butterfly transform) |
| `ml/backend/ggml/ggml/src/ggml-cuda/set-rows.cu` | GPU WHT quantization kernel |
| `ml/backend/ggml/ggml/src/ggml-cuda/ggml-cuda.cu` | Op dispatch + supports_op (incl. TURBO_WHT) |
| `ml/backend/ggml/ggml/src/ggml-cuda/convert.cu` | Turbo dequant CUDA kernels |
| `ml/nn/attention.go` | WHT rotation hooks |
| `ml/backend/ggml/flash_attention_turbo_test.go` | Go regression tests for FA selector |
| `hide-ggml.ver` | Linker version script |
