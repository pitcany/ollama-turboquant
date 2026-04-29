# TurboQuant GPU Debug Guide

## What We've Done

### Commit History (branch: feature/turboquant-kv-cache)

```
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

### Current State

| Test | Result |
|------|--------|
| f16 GPU (RTX 4090) | **Working** — 207.8 t/s |
| turbo4 GPU (RTX 4090) | **SIGABRT** during first inference |
| turbo4 CPU | **Working** — ~35 t/s, correct output |
| turbo4 CPU quality | Correct: "Four", "2,3,5,7,11", Pythagorean theorem |

### Current GPU Crash

The turbo4 GPU crash happens during the first forward pass after successful model loading:

```
load request: KvCacheType:turbo4, GPULayers:29, offloaded 29/29 layers to GPU
kv cache device=CUDA0 size=59.5 MiB
llama runner started in 0.80 seconds
SIGABRT: abort
```

The model loads, weights transfer to GPU, KV cache allocates as turbo4 on GPU. Then the first inference triggers `GGML_ABORT("fatal error")` somewhere in the CUDA flash attention dispatch.

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

## Diagnosis Steps for the GPU SIGABRT

### Step 1: Identify the exact abort location

Add `fprintf(stderr, ...)` debug prints to the CUDA flash attention dispatch chain. The abort is `GGML_ABORT("fatal error")` which exists in multiple places in `fattn.cu`.

**File**: `ml/backend/ggml/ggml/src/ggml-cuda/fattn.cu`

Add before each `GGML_ABORT("fatal error")`:
```c
fprintf(stderr, "TURBO DEBUG: abort at %s:%d, K->type=%d, V->type=%d, Q->ne[0]=%lld, K->ne[1]=%lld\n",
    __FILE__, __LINE__, K->type, V->type, (long long)Q->ne[0], (long long)K->ne[1]);
```

There are multiple `GGML_ABORT` calls in fattn.cu:
- Line ~136: in `ggml_cuda_flash_attn_ext_mma` (MMA kernel dispatch)
- Line ~229: in `ggml_cuda_flash_attn_ext_vec` (VEC kernel dispatch)
- Others in tile/wmma dispatchers

This tells us WHICH dispatcher is aborting and with what tensor dimensions.

### Step 2: Check kernel selection

**File**: `ml/backend/ggml/ggml/src/ggml-cuda/fattn.cu`

In `ggml_cuda_get_best_fattn_kernel()` (~line 232), add:
```c
fprintf(stderr, "TURBO DEBUG: best_fattn K->type=%d V->type=%d Q->ne[0]=%lld K->ne[1]=%lld\n",
    K->type, V->type, (long long)Q->ne[0], (long long)K->ne[1]);
```

And at the return statement:
```c
fprintf(stderr, "TURBO DEBUG: selected kernel=%d\n", result);
```

This tells us which kernel type was selected (VEC=100, TILE=200, MMA=300, etc.).

### Step 3: Check can_use_vector_kernel

The vector kernel requires `K->ne[1] % FATTN_KQ_STRIDE == 0` where `FATTN_KQ_STRIDE=256`. If the KV cache has fewer than 256 entries, the vector kernel can't be used and the code falls to MMA/WMMA/tile kernels, which don't have turbo support.

Add after `can_use_vector_kernel` calculation:
```c
fprintf(stderr, "TURBO DEBUG: can_use_vec=%d K_ne1=%lld stride=%d\n",
    can_use_vector_kernel, (long long)K->ne[1], FATTN_KQ_STRIDE);
```

### Step 4: Rebuild and test

```bash
cmake --build build -j$(nproc)
# copy libs, go clean, go build (see build commands above)
CUDA_VISIBLE_DEVICES=0 OLLAMA_HOST=localhost:9999 \
  OLLAMA_KV_CACHE_TYPE=turbo4 OLLAMA_FLASH_ATTENTION=1 \
  OLLAMA_NEW_ENGINE=1 OLLAMA_CONTEXT_LENGTH=4096 \
  ./ollama-tq serve 2>&1 | grep "TURBO DEBUG"
```

### Step 5: Apply the fix based on findings

**If can_use_vector_kernel is false**: Force vector kernel for turbo types:
```c
// In ggml_cuda_get_best_fattn_kernel, after can_use_vector_kernel is computed:
const bool is_turbo = K->type == GGML_TYPE_TURBO2_0 || K->type == GGML_TYPE_TURBO3_0 || K->type == GGML_TYPE_TURBO4_0;
if (is_turbo) {
    if (can_use_vector_kernel) {
        return BEST_FATTN_KERNEL_VEC;
    }
    return BEST_FATTN_KERNEL_NONE;  // fall back to CPU attention
}
```

**If the abort is in MMA/WMMA dispatcher**: The kernel selector chose a non-VEC kernel that doesn't have turbo support. Fix by adding turbo check to force VEC or return NONE.

**If K->ne[1] is the issue**: The KV cache stride doesn't meet FATTN_KQ_STRIDE requirements. This is a legitimate limitation — turbo FA templates only support D=64/128/256 and K_ne1 % 256 == 0.

## Structural Comparison with ollama-tq

The existing working ollama-tq build at `~/.local/src/ollama-tq/ollama/` handles this differently:
- Uses C-level WHT rotation (not Go attention hooks)
- No flash attention template instances (uses modular kernels)
- Uses `tqp_set_default_rotation()` at global init time
- Different type naming (TQ4P_D128 etc.)

Our approach uses CUDA flash attention templates (same as the TurboQuant fork), which requires the VEC kernel path. The fork's flash attention dispatch probably has the same constraint but may handle it differently.

**To check**: `grep -n "FATTN_KQ_STRIDE\|can_use_vector" /tmp/turboquant-fork/ggml/src/ggml-cuda/fattn.cu`

## Files Reference

| File | Purpose |
|------|---------|
| `ml/backend/ggml/ggml/include/ggml.h` | Type enums (36-38), TURBO_WHT op |
| `ml/backend/ggml/ggml/src/ggml.c` | Type traits, ggml_turbo_wht() |
| `ml/backend/ggml/ggml/src/ggml-turbo-quant.c` | CPU WHT + PolarQuant |
| `ml/backend/ggml/ggml/src/ggml-common.h` | Block struct definitions |
| `ml/backend/ggml/ggml/src/ggml-cpu/ggml-cpu.c` | CPU type traits + vec_dot |
| `ml/backend/ggml/ggml/src/ggml-cuda/fattn.cu` | **Flash attention dispatch (ABORT here)** |
| `ml/backend/ggml/ggml/src/ggml-cuda/fattn-common.cuh` | Turbo KQ dot + V dequant |
| `ml/backend/ggml/ggml/src/ggml-cuda/fattn-vec.cuh` | Turbo constexprs + externs |
| `ml/backend/ggml/ggml/src/ggml-cuda/set-rows.cu` | GPU WHT quantization kernel |
| `ml/backend/ggml/ggml/src/ggml-cuda/ggml-cuda.cu` | Op dispatch + supports_op |
| `ml/backend/ggml/ggml/src/ggml-cuda/convert.cu` | Turbo dequant CUDA kernels |
| `ml/nn/attention.go` | WHT rotation hooks |
| `hide-ggml.ver` | Linker version script |
| `TURBOQUANT.md` | Main documentation |
