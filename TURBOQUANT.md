# TurboQuant Ollama Integration

## What This Is

A surgical integration of [TurboQuant](https://github.com/TheTom/llama-cpp-turboquant) KV cache compression (arXiv 2504.19874) into Ollama, using the `feature/turboquant-kv-cache` branch of the fork.

TurboQuant compresses the KV cache during inference using Walsh-Hadamard Transform (WHT) rotation + PolarQuant quantization, reducing KV cache memory by 3.8-7.5x.

## Current State

This commit contains the **Go-layer scaffolding** for TurboQuant. The C/CUDA layer changes need to be re-applied (they were reverted during a bisect investigation — see "GPU Investigation" below).

### What's committed (Go layer — 4 files, 61 lines)

| File | Change |
|------|--------|
| `ml/backend.go` | Added `DTypeTurbo2`, `DTypeTurbo3`, `DTypeTurbo4` to DType enum |
| `ml/nn/attention.go` | WHT rotation hooks: forward WHT on Q before attention, inverse WHT on output. Detection via `key.DType()` — works with any cache wrapper (Causal, HybridCache, etc.) |
| `runner/ollamarunner/cache.go` | Added turbo2/3/4 to `kvCacheTypeFromStr()` |
| `fs/ggml/ggml.go` | `SupportsKVCacheType()` with head_dim validation, `kvCacheBytesPerElement()` with turbo byte ratios |

### What needs re-applying (C/CUDA layer — not in this commit)

The following changes were developed and tested but reverted during bisect. They need to be re-applied from the TurboQuant fork at `/tmp/turboquant-fork`:

**GGML core (from fork → Ollama):**
- `ggml.h` — add TURBO2_0/3_0/4_0 type enums (IDs 42-44), bump COUNT to 45, add GGML_OP_TURBO_WHT, add ggml_turbo_wht() declaration
- `ggml.c` — type_traits entries for turbo2/3/4, ggml_turbo_wht() implementation, op name table entry, op count assertion (96)
- `ggml-common.h` — block_turbo2_0/3_0/4_0 structs, QK defines, block_tq3_1s/tq4_1s
- `ggml-quants.h` — quantize/dequantize function declarations
- `ggml-turbo-quant.c` — **new file** (1026 lines): CPU WHT + PolarQuant quantize/dequant
- `CMakeLists.txt` — add ggml-turbo-quant.c to sources

**CPU backend:**
- `ggml-cpu.c` — `#include "ggml-quants.h"`, vec_dot stubs for turbo2/3/4, type_traits_cpu entries, GGML_OP_TURBO_WHT compute forward

**CUDA backend (copy from fork):**
- `turbo-quant.cuh` — device dequant functions, WHT sign arrays, centroid tables (453 lines)
- `turbo-innerq.cuh` + `turbo-innerq.cu` — per-channel equalization (34+32 lines)
- 18x `fattn-vec-instance-*turbo*.cu` — flash attention template instances (~8 lines each)

**CUDA backend (surgical patches):**
- `fattn-common.cuh` — turbo KQ dot product functions, turbo V dequant functions, dispatcher updates
- `fattn-vec.cuh` — K/V_is_unquantized constexprs for turbo types, extern declarations
- `fattn.cu` — FATTN_VEC_CASES_ALL_D dispatch entries, mixed KV type support, dimension validation
- `set-rows.cu` — replace with fork's version (adds 929-line GPU WHT quantization kernel)
- `ggml-cuda.cu` — TURBO_WHT dispatch, SET_ROWS supports_op for turbo types
- `convert.cu` — turbo dequant CUDA kernels, registration in ggml_get_to_fp32_cuda
- `dequantize.cuh` — turbo dequant device function declarations
- `CMakeLists.txt` — turbo template glob, turbo-innerq.cu source

**Go CGo bindings:**
- `ml/backend/ggml/ggml.go` — DType↔C.GGML_TYPE mapping, TurboWHT() tensor method
- `llama/llama.go` — turbo2/3/4 in kvCacheTypeFromStr (for llamarunner path)

## GPU Investigation Results

### Key finding: GPU crashes are a pre-existing Ollama dev-build issue

Through binary search bisection, we proved that **even vanilla Ollama (zero turbo changes) crashes on GPU** when built with `go build` from source:

```
# Vanilla Ollama (no turbo patches), go build, GPU:
offloaded 29/29 layers to GPU
SIGSEGV: segmentation violation   ← crashes during model weight loading
```

```
# System-installed Ollama (same version 0.20.7), GPU:  
offloaded 29/29 layers to GPU
"Four."   ← works perfectly
```

**Root cause:** The system release binary packages a **1.79 GB monolithic** `libggml-cuda.so` (GGML core statically linked into the CUDA library). Dev builds produce a **468 MB CUDA library + 798 KB thin base library**. The thin base library doesn't contain the full GGML implementation, causing symbol resolution conflicts between the Go binary (CGo-linked GGML) and the dynamically loaded CUDA library.

### What works

| Configuration | Status | Speed |
|--------------|--------|-------|
| CPU, f16 KV cache | Works | ~296 t/s (qwen2.5:7b) |
| CPU, turbo4 KV cache | Works | ~35 t/s (qwen2.5:7b) |
| GPU, f16 KV cache (release build) | Works | ~222 t/s |
| GPU, any KV cache (dev build) | SIGSEGV | N/A |

### Speed analysis

The 8.5x CPU slowdown (turbo4 vs f16) is caused by the `GGML_OP_TURBO_WHT` falling back to CPU even when the model runs on GPU. Each attention layer requires 4 PCIe transfers (GPU→CPU→GPU for Q rotation + output inverse rotation). A CUDA WHT kernel exists in the fork (`turbo-wht.cu`, 189 lines) but could not be tested due to the dev-build GPU crash.

### How to get GPU working

Either:
1. **Build a release binary** using the official Dockerfile/CI pipeline that produces a monolithic CUDA library
2. **Modify the cmake to link ggml-base statically into ggml-cuda** (the `GGML_BACKEND_SHARED` flag doesn't work — Ollama's CMakeLists.txt overrides it)
3. **Use the system Ollama binary with turbo patches** — would require patching the installed binary

## Architecture

### WHT rotation flow

```
KV cache write (SetRows):
  K/V → forward WHT → PolarQuant → store as turbo block

Attention (nn/attention.go):
  1. cache.Get() → turbo K, turbo V, mask
  2. Detect turbo: key.DType() ∈ {Turbo2, Turbo3, Turbo4}
  3. Q → forward WHT (GGML_OP_TURBO_WHT, direction=0)
  4. FlashAttention(Q_rot, K_turbo, V_turbo)  
  5. output → inverse WHT (GGML_OP_TURBO_WHT, direction=1)
```

Detection is cache-wrapper-agnostic — works with Causal, HybridCache (qwen3next), WrapperCache.

### Comparison with ollama-tq

| | This build | ollama-tq |
|---|---|---|
| WHT rotation | Go-level (nn/attention.go) | C-level (tqp_set_default_rotation) |
| Type naming | 3 fixed (turbo2/3/4) | 9 flexible (TQ4P_D*/TQP_D*_B*) |
| Flash attention | 18 CUDA templates | 0 (modular) |
| Source | TheTom/llama-cpp-turboquant fork | Custom ggml-tq-paper.c |
| Self-contained | Yes (no ollama-tq dependencies) | N/A |

## Usage (after re-applying C/CUDA changes)

```bash
# Build
cmake --preset "CUDA 12" -B build \
  -DCMAKE_DISABLE_FIND_PACKAGE_Vulkan=TRUE \
  -DCMAKE_CUDA_ARCHITECTURES="89;120"
cmake --build build -j$(nproc)
mkdir -p build/lib/ollama/cuda_v12
cp build/lib/ollama/libggml-cuda.so build/lib/ollama/cuda_v12/
cp build/lib/ollama/libggml-base.so.0.0.0 build/lib/ollama/cuda_v12/libggml-base.so
CGO_LDFLAGS="-L$(pwd)/build/lib/ollama" go build -o ollama-tq-test .

# Run (CPU inference — GPU blocked by dev-build issue)
OLLAMA_KV_CACHE_TYPE=turbo4 \
OLLAMA_FLASH_ATTENTION=1 \
OLLAMA_NEW_ENGINE=1 \
./ollama-tq-test serve
```

## Rollback

```bash
git checkout -- .
git clean -fd
rm -rf build ollama-tq-test cuda_v12 libggml-*.so
```

## GPU Build Fix (CRITICAL)

The dev-build GPU crash is caused by CGo GGML symbols leaking into the dynamic symbol table, conflicting with the dynamically loaded CUDA library. The fix:

```bash
CGO_LDFLAGS="-L$(pwd)/build/lib/ollama" \
go build -trimpath -buildmode=pie \
  -ldflags='-extldflags "-Wl,--version-script=hide-ggml.ver"' \
  -o ollama-tq-test .
```

The `hide-ggml.ver` version script:
```
{
  local:
    ggml_*;
    quantize_*;
    dequantize_*;
    turbo_*;
};
```

This matches how the official Ollama release binary is built — PIE executable with zero exported GGML symbols. Without this, the CUDA shared library resolves `ggml_*` functions to the Go binary's CGo-linked copies instead of `libggml-base.so`, causing SIGSEGV during model weight loading.

**Verified:** f16 at 218.8 t/s on RTX 4090 with this fix (vs SIGSEGV without it).
