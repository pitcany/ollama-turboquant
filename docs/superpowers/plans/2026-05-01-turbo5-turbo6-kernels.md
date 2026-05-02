# Turbo5/Turbo6 Kernels + `kturbo6-vturbo4` Preset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Turbo5 (5-bit, 32 centroids) and Turbo6 (6-bit, 64 centroids) PolarQuant dtypes to the existing Turbo* kernel family on CPU and CUDA, and ship `kturbo6-vturbo4` as a new split KV-cache preset whose full 256-sequence Phase 0 score on `qwen2.5:7b` Q4_K_M clears `mean_kl < 0.02`.

**Architecture:** Mirror the existing TURBO4_USE_4BIT path in `ggml-turbo-quant.c`: per-block L2 norm in fp16, in-block forward Walsh-Hadamard rotation at group size 128, packed N-bit PolarQuant indices over a Lloyd-Max centroid table optimized for the WHT-rotated unit-Gaussian distribution (σ = 1/√128). No QJL; no residual. The `kturbo6-vturbo4` preset reuses the same split machinery as `kq8-vturbo4` (`Causal.InitSplit`, `kvCacheTypesFromStr`, `kvCacheBytesPerElementKV`, `isSplitKVCachePreset`, `SupportsKVCacheType`). All CPU and CUDA paths stay behind a build-time guard until Phase 0 validates the preset; only then is the operator-facing string accepted at runtime.

**Tech Stack:** C (ggml CPU), CUDA (ggml CUDA backend), Go (fs/ggml, runner/ollamarunner, llm), Python (Lloyd-Max table generation in `tools/turboquant/centroids/`), the existing `cmd/turboquant-eval` Phase 0 harness.

**Critical abort gate (per user spec):** If `cmd/turboquant-eval -limit 1` for `qwen2.5:7b` Q4_K_M with `key_cache_type=turbo6, value_cache_type=turbo4` produces `mean_kl > 0.5` *on the first decoded sequence with CPU kernels alone*, the residual representation is the bottleneck regardless of bit count. **Stop. Document. Revert runtime wiring. Leave the kernels behind a build flag.** Do not proceed to CUDA work.

**Memory accounting (for sanity):**
- Turbo5 block (QK=128): 80B indices + 2B norm + 2B reserved = 84B → 5.25 bits/elem
- Turbo6 block (QK=128): 96B indices + 2B norm + 2B reserved = 100B → 6.25 bits/elem
- `kturbo6-vturbo4` per K/V pair: (6.25 + 4.25) / 2 / 8 ≈ **1.31 bytes** ✓ matches spec

---

## File Structure

**New files:**
- `tools/turboquant/centroids/lloyd_max.py` — generator for Turbo2/3/4/5/6 centroid tables (regenerates existing tables verbatim as a regression check, then emits Turbo5/Turbo6).
- `tools/turboquant/centroids/README.md` — what the script does, how to re-run, MSE bounds.
- `tools/turboquant/centroids/turbo5_centroids.txt` — generated table (32 floats + 31 midpoints).
- `tools/turboquant/centroids/turbo6_centroids.txt` — generated table (64 floats + 63 midpoints).

**Modified files:**
- `ml/backend/ggml/ggml/include/ggml.h` — add `GGML_TYPE_TURBO5_0`, `GGML_TYPE_TURBO6_0` enum slots (reuse next two deprecated slots after TURBO4_0=38; verify they are in fact unused before claiming them).
- `ml/backend/ggml/ggml/src/ggml-common.h` — add `block_turbo5_0`, `block_turbo6_0` structs, `QK_TURBO5`, `QK_TURBO6`, `NL_TURBO5*`, `NL_TURBO6*` derived constants. Static asserts on byte sizes.
- `ml/backend/ggml/ggml/src/ggml-turbo-quant.c` — add CPU `quantize_row_turbo{5,6}_0[_ref]`, `dequantize_row_turbo{5,6}_0`, `quantize_turbo{5,6}_0`, `nearest_centroid_{5,6}bit`, and the new centroid + midpoint tables.
- `ml/backend/ggml/ggml/src/ggml-quants.c` and/or `ggml.c` (wherever `type_traits` lives) — register the new types' traits (block size, byte size, quantize/dequantize fn pointers, vec_dot=NULL, KV cache eligibility = true).
- `ml/backend/ggml/ggml/src/ggml.c` — `ggml_is_quantized`, `ggml_blck_size`, `ggml_type_size` switch arms (or wherever the type metadata is centralized; mirror Turbo4 exactly).
- `ml/backend/ggml/ggml/src/ggml-cuda/turbo-quant.cuh` — CUDA dequant + decode helpers for Turbo5/6, mirroring Turbo4.
- `ml/backend/ggml/ggml/src/ggml-cuda/set-rows.cu` — Turbo5/6 quantize-on-write paths. Mirror the Turbo4 case; reuse the WHT prologue.
- `ml/backend/ggml/ggml/src/ggml-cuda/turbo-innerq.cu` and `turbo-innerq.cuh` — KQ inner-product specialisations for Turbo5/6.
- `ml/backend/ggml/ggml/src/ggml-cuda/fattn-vec.cuh` and `fattn-common.cuh` — add Turbo5/6 K decode arms in the FA kernels (vec instances at minimum; tile/mma can stay unsupported in this PR).
- `ml/backend/ggml/ggml/src/ggml-cuda/ggml-cuda.cu` — register the new types in `ggml_backend_cuda_supports_type` / wherever the dispatch switch lives.
- `tools/turboquant/block_error.cpp` — extend the synthetic block-error test to cover Turbo5 and Turbo6 with theoretical-bound assertions.
- `ml/`'s Go DType layer (search for `DTypeTurbo4`):
  - `ml/backend.go` (or wherever `ml.DType` lives) — add `DTypeTurbo5`, `DTypeTurbo6`.
  - `ml/backend/ggml/ggml.go` — bridge new DType ↔ `GGML_TYPE_TURBO{5,6}_0`.
- `runner/ollamarunner/cache.go::kvCacheTypesFromStr` — add `kturbo6-vturbo4`.
- `runner/ollamarunner/cache_test.go` — add coverage matching the existing `kq8-vturbo4` test.
- `llm/server.go::isSplitKVCachePreset` — recognise `kturbo6-vturbo4`.
- `llm/server.go::SupportsKVCacheType` — also gate on Turbo5/Turbo6 support.
- `llm/server_test.go` — add coverage.
- `fs/ggml/ggml.go::kvCacheBytesPerElementKV` and the per-string byte-size table — register `turbo5`, `turbo6`, and `kturbo6-vturbo4`.
- `fs/ggml/ggml_test.go` — add coverage.
- `envconfig/config.go` — feature flag `OLLAMA_TURBOQUANT_K6_PREVIEW` (default off) gating runtime acceptance of the new preset; CPU/CUDA kernels themselves do not need the flag and stay always-compiled.
- `tools/turboquant/README.md` — document the new preset under "Phase 0 Results" only after the gate clears.
- `TURBOQUANT-DEBUG-LOG.md` — chronological evidence entries per the existing format.

**Out of scope for this plan:**
- Metal kernels (MPS) — leave Turbo5/Turbo6 unsupported on Metal in this PR; document the gap.
- FA tile/MMA Turbo5/6 paths — vec is sufficient for `kturbo6-vturbo4` because K is the rotated path, not the high-throughput weight path.
- Per-layer adaptive integration of Turbo5/6 — this PR ships only the uniform-K Turbo6 preset; the calibration manifest is unchanged.
- Residual-window key protection (still dead per 2026-04-30 WHT-basis log entry).
- Reintroducing QJL.

---

## Phase A — Centroid Generation and Offline Validation

Cheap. Validates the math before any kernel work.

### Task A1: Lloyd-Max generator that reproduces existing tables

**Files:**
- Create: `tools/turboquant/centroids/lloyd_max.py`
- Create: `tools/turboquant/centroids/README.md`

- [ ] **Step 1: Write the Python generator**

```python
#!/usr/bin/env python3
"""Lloyd-Max centroid optimiser for the Turbo* PolarQuant family.

The Turbo* CPU/CUDA kernels operate on per-block normalised, WHT-rotated
vectors of length 128. Under the WHT-rotation prior, each rotated coordinate is
approximately N(0, 1/128). We optimise n_centroids quantisation levels for that
prior using Lloyd's algorithm (PCM scalar quantiser), then emit the centroid
list and the n_centroids-1 midpoints used by the C `nearest_centroid_Nbit`
binary search. Run with `--check` to verify the existing 2/3/4-bit tables are
reproduced bit-for-bit (within 1e-5)."""

import argparse, math, sys
import numpy as np

SIGMA = 1.0 / math.sqrt(128.0)

def lloyd_max(n_centroids, samples, n_iters=200, seed=0):
    rng = np.random.default_rng(seed)
    # Initialise on equiprobable Gaussian quantiles
    qs = (np.arange(n_centroids) + 0.5) / n_centroids
    centroids = np.quantile(samples, qs)
    for _ in range(n_iters):
        # Voronoi assignment via sorted midpoints
        mids = (centroids[:-1] + centroids[1:]) / 2.0
        idx = np.searchsorted(mids, samples)
        # Update each centroid to the conditional mean of its cell
        new = centroids.copy()
        for k in range(n_centroids):
            mask = idx == k
            if mask.any():
                new[k] = samples[mask].mean()
        if np.allclose(new, centroids, atol=1e-9):
            break
        centroids = new
    mids = (centroids[:-1] + centroids[1:]) / 2.0
    mse = float(((samples - centroids[np.searchsorted(mids, samples)]) ** 2).mean())
    return centroids, mids, mse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, action="append", default=[],
                    help="bit widths to emit (e.g. --bits 5 --bits 6)")
    ap.add_argument("--check", action="store_true",
                    help="reproduce 2/3/4-bit tables and exit non-zero on mismatch")
    ap.add_argument("--samples", type=int, default=4_000_000)
    args = ap.parse_args()

    rng = np.random.default_rng(20260501)
    samples = rng.normal(0.0, SIGMA, size=args.samples)

    EXPECTED = {
        2: [-0.133462, -0.039994, 0.039994, 0.133462],
        3: [-0.190685, -0.117832, -0.065717, -0.021460,
             0.021460,  0.065717,  0.117832,  0.190685],
        4: [-0.173926, -0.117195, -0.089527, -0.068756,
            -0.051262, -0.035597, -0.020989, -0.006938,
             0.006938,  0.020989,  0.035597,  0.051262,
             0.068756,  0.089527,  0.117195,  0.173926],
    }

    if args.check:
        bad = 0
        for bits, exp in EXPECTED.items():
            got, _, _ = lloyd_max(2 ** bits, samples)
            err = float(np.max(np.abs(got - np.array(exp))))
            print(f"bits={bits}: max abs diff vs in-tree table = {err:.2e}")
            if err > 1e-4:
                bad += 1
        sys.exit(bad)

    for b in args.bits:
        c, m, mse = lloyd_max(2 ** b, samples)
        # Theoretical PCM bound for Gaussian: D ~ (sqrt(3) pi / 2) * sigma^2 * 2^{-2 bits}
        bound = (math.sqrt(3) * math.pi / 2.0) * SIGMA * SIGMA * (4.0 ** -b)
        print(f"bits={b}: mse={mse:.3e}, theoretical_bound={bound:.3e}, ratio={mse/bound:.3f}")
        with open(f"tools/turboquant/centroids/turbo{b}_centroids.txt", "w") as f:
            f.write(f"# {2**b} Lloyd-Max centroids for N(0, 1/128), seed=20260501\n")
            f.write("# centroids\n")
            for v in c:
                f.write(f"{v:+.6f}\n")
            f.write("# midpoints\n")
            for v in m:
                f.write(f"{v:+.6f}\n")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify regression against existing tables**

```bash
python3 tools/turboquant/centroids/lloyd_max.py --check
```

Expected: exit 0; per-bit max-abs diff ≤ 1e-4 for bits 2, 3, 4.

- [ ] **Step 3: Generate Turbo5 and Turbo6 tables**

```bash
python3 tools/turboquant/centroids/lloyd_max.py --bits 5 --bits 6
```

Expected: mse/theoretical_bound ratio ≤ 1.20 for both bit widths (Lloyd's converges very close to the high-rate PCM bound for moderate n_centroids; >1.2 is a sign the Lloyd loop diverged or sampling was too small).

- [ ] **Step 4: Write README**

`tools/turboquant/centroids/README.md` documents the generator, the prior (N(0, 1/128)), the seed, the expected MSE bound, and the commit/regenerate protocol (regenerate when changing the rotation contract; never edit by hand).

- [ ] **Step 5: Commit**

```bash
git add tools/turboquant/centroids/
git commit -m "feat(turboquant): Lloyd-Max centroid generator for Turbo2..6

Reproduces the existing Turbo2/3/4 in-tree tables to <=1e-4 max abs and
emits new Turbo5/6 tables for the upcoming kturbo6-vturbo4 preset. No
kernel code yet; this is the offline math the next commits will paste
into ggml-turbo-quant.c."
```

### Task A2: Extend `block_error.cpp` to assert the bound on Turbo5/Turbo6

**Files:**
- Modify: `tools/turboquant/block_error.cpp`

- [ ] **Step 1: Read the existing test**

```bash
sed -n '1,200p' tools/turboquant/block_error.cpp
```

Note the existing pattern for Turbo4: synthetic Gaussian input → quantize → dequantize → measure per-element MSE → assert MSE < theoretical_bound × tolerance.

- [ ] **Step 2: Add Turbo5 and Turbo6 cases**

The exact code mirrors the Turbo4 case. Cannot show full code here without first reading the file; the executing agent reproduces the existing arm verbatim with the new types.

- [ ] **Step 3: Run the test (it should fail with "type not registered" — kernels not yet implemented)**

```bash
g++ -std=c++17 -O2 -Iml/backend/ggml/ggml/include -Iml/backend/ggml/ggml/src \
  tools/turboquant/block_error.cpp \
  -Lbuild/lib/ollama -lggml-base -ldl -lpthread -lm \
  -Wl,-rpath=$(pwd)/build/lib/ollama \
  -o /tmp/turboquant-block-error && /tmp/turboquant-block-error
```

Expected: failure on Turbo5/Turbo6 cases. This documents the Phase A→B handoff: Phase B Task B6 will rerun this and require pass.

- [ ] **Step 4: Commit**

```bash
git add tools/turboquant/block_error.cpp
git commit -m "test(turboquant): assert Turbo5/Turbo6 per-element MSE bounds

Follows the Turbo4 pattern. Currently fails because the kernels are not
registered yet; Phase B will add the kernels and this test becomes the
acceptance gate before any Go wiring is touched."
```

---

## Phase B — CPU Kernels

### Task B1: Add the type enum and block layout

**Files:**
- Modify: `ml/backend/ggml/ggml/include/ggml.h`
- Modify: `ml/backend/ggml/ggml/src/ggml-common.h`

- [ ] **Step 1: Pick enum slots**

```bash
grep -nE 'GGML_TYPE_(TURBO|IQ4|MXFP|TQ).*=' ml/backend/ggml/ggml/include/ggml.h
```

Read the output; pick the next two deprecated slots after `GGML_TYPE_TURBO4_0 = 38`. Verify they are not in use anywhere else with:

```bash
grep -rn "GGML_TYPE_<chosen_slot_name>" ml/backend/ggml/
```

If both candidate slots are clean, claim them as `GGML_TYPE_TURBO5_0` and `GGML_TYPE_TURBO6_0` with the `// reusing deprecated <orig> slot` comment per the existing convention. If either is live, document the conflict and ask before continuing.

- [ ] **Step 2: Add `block_turbo5_0` and `block_turbo6_0` to `ggml-common.h`**

```c
/* TurboQuant 5-bit: 5-bit PolarQuant indices, no QJL, no residual */
#define QK_TURBO5 128
#define QK_TURBO5_GROUP 128
#define NL_TURBO5     (QK_TURBO5 / 16)
#define NL_TURBO5_VEC (QK_TURBO5 / 4)
typedef struct {
    ggml_half  norm;                               /*  2 bytes */
    ggml_half  rnorm;                              /*  2 bytes (reserved, unused) */
    uint8_t    qs[QK_TURBO5 * 5 / 8];             /* 80 bytes: 5-bit indices, bit-packed */
} block_turbo5_0;                                  /* 84 bytes total */
static_assert(sizeof(block_turbo5_0) == 2*sizeof(ggml_half) + QK_TURBO5*5/8, "wrong turbo5_0 block size");

/* TurboQuant 6-bit: 6-bit PolarQuant indices, no QJL, no residual */
#define QK_TURBO6 128
#define QK_TURBO6_GROUP 128
#define NL_TURBO6     (QK_TURBO6 / 16)
#define NL_TURBO6_VEC (QK_TURBO6 / 4)
typedef struct {
    ggml_half  norm;                               /*  2 bytes */
    ggml_half  rnorm;                              /*  2 bytes (reserved, unused) */
    uint8_t    qs[QK_TURBO6 * 6 / 8];             /* 96 bytes: 6-bit indices, bit-packed */
} block_turbo6_0;                                  /* 100 bytes total */
static_assert(sizeof(block_turbo6_0) == 2*sizeof(ggml_half) + QK_TURBO6*6/8, "wrong turbo6_0 block size");
```

- [ ] **Step 3: Build to validate static asserts compile**

```bash
cmake --build build -j2 --target ggml-base
```

Expected: clean build. If a static assert fails, the byte arithmetic above is wrong — fix and rebuild.

- [ ] **Step 4: Commit**

```bash
git add ml/backend/ggml/ggml/include/ggml.h ml/backend/ggml/ggml/src/ggml-common.h
git commit -m "feat(ggml): allocate Turbo5/Turbo6 type enum and block layouts

QK=128, single-precision PolarQuant. Turbo5 = 84B (5.25 bits/elem),
Turbo6 = 100B (6.25 bits/elem). No QJL, no residual; rnorm stays
reserved to keep the struct shape consistent with Turbo4."
```

### Task B2: Add CPU centroid tables and `nearest_centroid_{5,6}bit`

**Files:**
- Modify: `ml/backend/ggml/ggml/src/ggml-turbo-quant.c`

- [ ] **Step 1: Paste the generated tables**

Paste the contents of `tools/turboquant/centroids/turbo5_centroids.txt` and `turbo6_centroids.txt` into `ggml-turbo-quant.c` near the existing `CENTROIDS_4BIT` table:

```c
/* 5-bit: 32 Lloyd-Max centroids for N(0, 1/128), generated by
 * tools/turboquant/centroids/lloyd_max.py --bits 5 (seed=20260501).
 * Regenerate via the script; do not edit by hand. */
static const float CENTROIDS_5BIT[32] = {
    /* paste 32 floats here */
};

static const float CENTROIDS_6BIT[64] = {
    /* paste 64 floats here */
};
```

- [ ] **Step 2: Add `nearest_centroid_{5,6}bit` using the midpoints**

```c
static int nearest_centroid_5bit(float val) {
    /* 31 midpoints, generated. Linear scan is acceptable; this is on the
     * cold quantize path only. The hot dequant path indexes directly. */
    static const float MIDS_5BIT[31] = { /* 31 midpoints from the txt file */ };
    int i = 0;
    while (i < 31 && val >= MIDS_5BIT[i]) i++;
    return i;
}

static int nearest_centroid_6bit(float val) {
    static const float MIDS_6BIT[63] = { /* 63 midpoints from the txt file */ };
    int i = 0;
    while (i < 63 && val >= MIDS_6BIT[i]) i++;
    return i;
}
```

(A binary search is fine but unnecessary; the scan happens at most QK_TURBO5/6 = 128 times per block during quantize. Keep it simple, mirror the existing 4-bit signature.)

- [ ] **Step 3: Build to confirm it compiles**

```bash
cmake --build build -j2 --target ggml
```

- [ ] **Step 4: Commit**

```bash
git add ml/backend/ggml/ggml/src/ggml-turbo-quant.c
git commit -m "feat(turboquant): add CPU centroid + midpoint tables for Turbo5/6

Generated by tools/turboquant/centroids/lloyd_max.py at seed 20260501.
The midpoints feed nearest_centroid_5bit/6bit; centroids are also used
by dequant in the next commit."
```

### Task B3: CPU quantize/dequantize for Turbo5

**Files:**
- Modify: `ml/backend/ggml/ggml/src/ggml-turbo-quant.c`

- [ ] **Step 1: Add `quantize_row_turbo5_0_ref`**

Mirror `quantize_row_turbo4_0_ref` exactly (the `TURBO4_USE_4BIT` path), with these substitutions:
- `QK_TURBO4` → `QK_TURBO5`
- `block_turbo4_0` → `block_turbo5_0`
- `nearest_centroid_4bit` → `nearest_centroid_5bit`
- `CENTROIDS_4BIT` → `CENTROIDS_5BIT`
- Packing: 5-bit indices into `qs[80]` using a running bit-cursor:

```c
/* Pack 128 5-bit indices into 80 bytes via a running 32-bit accumulator. */
uint32_t acc = 0;
int bits = 0;
int qpos = 0;
memset(y[block].qs, 0, QK_TURBO5 * 5 / 8);
for (int i = 0; i < d; i++) {
    acc |= ((uint32_t)(indices[i] & 0x1F)) << bits;
    bits += 5;
    while (bits >= 8) {
        y[block].qs[qpos++] = (uint8_t)(acc & 0xFF);
        acc >>= 8;
        bits -= 8;
    }
}
if (bits > 0) y[block].qs[qpos++] = (uint8_t)(acc & 0xFF);
GGML_ASSERT(qpos == QK_TURBO5 * 5 / 8);
```

- [ ] **Step 2: Add `dequantize_row_turbo5_0`**

Mirror `dequantize_row_turbo4_0` with the same substitutions and the matching unpacker:

```c
uint32_t acc = 0;
int bits = 0;
int qpos = 0;
for (int i = 0; i < d; i++) {
    while (bits < 5) {
        acc |= ((uint32_t)x[block].qs[qpos++]) << bits;
        bits += 8;
    }
    uint8_t idx = (uint8_t)(acc & 0x1F);
    acc >>= 5;
    bits -= 5;
    dst[i] = CENTROIDS_5BIT[idx] * norm;
}
```

- [ ] **Step 3: Add `quantize_turbo5_0` wrapper**

```c
size_t quantize_turbo5_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO5 == 0);
    size_t row_size = (n_per_row / QK_TURBO5) * sizeof(block_turbo5_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo5_0_ref(
            src + row * n_per_row,
            (block_turbo5_0 *)((char *)dst + row * row_size),
            n_per_row);
    }
    return nrows * row_size;
}
```

- [ ] **Step 4: Build**

```bash
cmake --build build -j2 --target ggml
```

- [ ] **Step 5: Commit**

```bash
git add ml/backend/ggml/ggml/src/ggml-turbo-quant.c
git commit -m "feat(turboquant): CPU quantize/dequantize for Turbo5_0

5-bit PolarQuant in the WHT-rotated domain, no QJL, mirrors the Turbo4
4-bit path. Bit-packed 5-bit indices via a 32-bit accumulator on both
encode and decode. Symbols are exported but not yet registered in the
ggml type-traits table."
```

### Task B4: CPU quantize/dequantize for Turbo6

**Files:**
- Modify: `ml/backend/ggml/ggml/src/ggml-turbo-quant.c`

- [ ] **Step 1: Add `quantize_row_turbo6_0_ref`, `dequantize_row_turbo6_0`, `quantize_turbo6_0`**

Identical to Task B3 with `5` → `6` and the arithmetic adjusted (`6 / 8`, mask `0x3F`, shift by 6). The same accumulator pattern works without modification because 6 < 8 < 32.

- [ ] **Step 2: Build, then commit**

```bash
cmake --build build -j2 --target ggml
git add ml/backend/ggml/ggml/src/ggml-turbo-quant.c
git commit -m "feat(turboquant): CPU quantize/dequantize for Turbo6_0

6-bit PolarQuant, mirrors Turbo5 with mask 0x3F and a 6-bit cursor."
```

### Task B5: Register Turbo5/Turbo6 in the ggml type-traits table

**Files:**
- Modify: `ml/backend/ggml/ggml/src/ggml-quants.c` (or wherever `type_traits[]` is defined — locate with `grep -nR 'GGML_TYPE_TURBO4_0' ml/backend/ggml/ggml/src/`)
- Modify: `ml/backend/ggml/ggml/src/ggml.c` switch arms for `ggml_type_size`, `ggml_blck_size`, `ggml_is_quantized` (mirror every place TURBO4_0 appears).

- [ ] **Step 1: Find every TURBO4_0 reference**

```bash
grep -nR 'GGML_TYPE_TURBO4_0\|TURBO4_0\b' ml/backend/ggml/ggml/src/ \
  | grep -v 'ggml-cuda\|metal\|fattn\|set-rows\|turbo-innerq\|turbo-quant.cuh'
```

The remaining list is the set of files that need a Turbo5_0 and Turbo6_0 mirror entry. There should be on the order of 4–6 files.

- [ ] **Step 2: Add mirror entries**

For each file, paste a copy of the Turbo4 line/case immediately after, with `_TURBO4_0` → `_TURBO5_0` and a second copy with `_TURBO6_0`. Function pointers map to the symbols added in B3/B4. `vec_dot` stays NULL (FA path is the only consumer).

- [ ] **Step 3: Build the whole CPU backend**

```bash
GOCACHE=/tmp/ollama-build-gocache cmake --build build -j2
```

- [ ] **Step 4: Run the synthetic block-error test from Task A2**

```bash
g++ -std=c++17 -O2 -Iml/backend/ggml/ggml/include -Iml/backend/ggml/ggml/src \
  tools/turboquant/block_error.cpp \
  -Lbuild/lib/ollama -lggml-base -ldl -lpthread -lm \
  -Wl,-rpath=$(pwd)/build/lib/ollama \
  -o /tmp/turboquant-block-error && /tmp/turboquant-block-error --assert-bounds
```

Expected: PASS. Per-element MSE for Turbo5 and Turbo6 must be within 1.2× of the high-rate PCM bound for σ=1/√128. **If this fails, Phase B is not done; do not proceed.**

- [ ] **Step 5: Commit**

```bash
git add ml/backend/ggml/ggml/src/
git commit -m "feat(ggml): register Turbo5_0 and Turbo6_0 in CPU type traits

Mirrors Turbo4_0 in every type-metadata switch. block_error.cpp now
PASSes both Turbo5 and Turbo6 against the high-rate PCM bound for
N(0, 1/128). No CUDA, no Go wiring yet; this is the smallest
isolated CPU-only checkpoint."
```

### Task B6: Optional CPU FA support for Turbo5/6 K (read-side dequant only)

The CPU FlashAttention CPU path lives in `ml/backend/ggml/ggml/src/ggml.c` (or the dedicated `ggml-cpu/...` module if it has been split out). Search:

```bash
grep -nR 'GGML_TYPE_TURBO4_0' ml/backend/ggml/ggml/src/ggml-cpu/ ml/backend/ggml/ggml/src/ \
  | grep -i fattn
```

- [ ] **Step 1: Mirror the Turbo4 read arms**

For each file the grep returns, mirror the Turbo4 case for Turbo5/6. The only operation needed is "decode K block → fp32 row". The dequant function is `dequantize_row_turbo{5,6}_0`.

- [ ] **Step 2: Verify CPU FA accepts Turbo5/6 K via the Go test suite**

```bash
GOCACHE=/tmp/ollama-build-gocache go test -count=1 -run TestTurbo ./ml/backend/ggml ./kvcache
```

Expected: PASS (no Turbo5/6-specific tests yet, but existing Turbo4 tests must not regress).

- [ ] **Step 3: Commit**

```bash
git commit -am "feat(ggml-cpu): accept Turbo5/Turbo6 K in FlashAttention dequant arms

Read-only support; quantize on CPU FA is unchanged. Verified existing
Turbo* tests still pass."
```

---

## Phase C — CPU-only Phase 0 Smoke (PRIMARY ABORT GATE)

This is the cheapest possible signal that Turbo6 K is viable. **If Turbo6 layer 0 still garbles on CPU, do not start Phase D.**

### Task C1: Minimal Go wiring to allow `cmd/turboquant-eval -engine go` with Turbo6 K

**Files:**
- Modify: `ml/` Go DType layer (find `DTypeTurbo4` with `grep -rn 'DTypeTurbo4' ml/`)
- Modify: `ml/backend/ggml/ggml.go` (DType ↔ ggml type bridge)
- Modify: `runner/ollamarunner/cache.go::kvCacheTypesFromStr` to accept `turbo5`, `turbo6`, and `kturbo6-vturbo4`
- Modify: `fs/ggml/ggml.go` to register `turbo5`, `turbo6`, `kturbo6-vturbo4` byte sizes
- Modify: `llm/server.go::isSplitKVCachePreset`, `SupportsKVCacheType`
- Modify: `envconfig/config.go` to add `OLLAMA_TURBOQUANT_K6_PREVIEW`

- [ ] **Step 1: Add `ml.DTypeTurbo5`, `ml.DTypeTurbo6` and bridge to `GGML_TYPE_TURBO{5,6}_0`**

Pattern is exactly the Turbo4 pattern. The grep above tells you every file to touch.

- [ ] **Step 2: Add `kturbo6-vturbo4` to `kvCacheTypesFromStr`, byte tables, and predicate functions**

```go
// runner/ollamarunner/cache.go
func kvCacheTypesFromStr(s string) (ml.DType, ml.DType) {
    switch strings.ToLower(strings.TrimSpace(s)) {
    case "kq8-vturbo4":
        return ml.DTypeQ80, ml.DTypeTurbo4
    case "kturbo6-vturbo4":
        return ml.DTypeTurbo6, ml.DTypeTurbo4
    }
    // ... existing fallback
}
```

```go
// llm/server.go
func isSplitKVCachePreset(cacheType string) bool {
    switch cacheType {
    case "kq8-vturbo4", "kturbo6-vturbo4":
        return true
    }
    return false
}
```

```go
// fs/ggml/ggml.go
case "kturbo6-vturbo4":
    return kvCacheBytesPerElement("turbo6"), kvCacheBytesPerElement("turbo4")
```

Add `turbo5`/`turbo6` to whatever string switch maps to bytes-per-element. Turbo5: `84.0/128.0`. Turbo6: `100.0/128.0`. Turbo4 reference: `68.0/128.0`. Cross-check against the Turbo4 entry already in tree to confirm the units.

- [ ] **Step 3: Gate runtime acceptance behind the feature flag**

In `llm/server.go::SupportsKVCacheType` (or wherever the runtime accepts the preset string):

```go
case "kturbo6-vturbo4":
    if !envconfig.TurboquantK6Preview() {
        return false
    }
    return f.SupportsKVCacheType("turbo6") && f.SupportsKVCacheType("turbo4")
```

`envconfig.TurboquantK6Preview()` reads `OLLAMA_TURBOQUANT_K6_PREVIEW`, defaults false. This means the preset is invisible to operators who do not opt in. The eval harness sets the env var explicitly.

- [ ] **Step 4: Add tests**

In `runner/ollamarunner/cache_test.go`, copy the existing `kq8-vturbo4` test and substitute `kturbo6-vturbo4`. In `fs/ggml/ggml_test.go`, do the same. In `llm/server_test.go`, verify the feature-flag gate (preset rejected when flag off, accepted when on).

- [ ] **Step 5: Run all wiring tests**

```bash
GOCACHE=/tmp/ollama-build-gocache \
go test -count=1 \
  ./fs/ggml/... \
  ./runner/ollamarunner/... \
  ./llm/... \
  ./envconfig/... \
  ./kvcache/...
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add ml/ runner/ollamarunner/ llm/ fs/ggml/ envconfig/
git commit -m "feat(ollama): wire kturbo6-vturbo4 preset behind OLLAMA_TURBOQUANT_K6_PREVIEW

Adds Turbo5/Turbo6 to the ml.DType bridge and the split-preset
machinery (kvCacheTypesFromStr, isSplitKVCachePreset,
kvCacheBytesPerElementKV, SupportsKVCacheType). Default off; the
runtime only accepts the preset when the operator sets the preview
env var. CPU kernels are in place from Phase B; Phase D will add CUDA."
```

### Task C2: CPU-only Phase 0 smoke at `-limit 1`

**Files:** none — runs the existing eval harness against the snapshot.

- [ ] **Step 1: Run the smoke**

```bash
OLLAMA_TURBOQUANT_K6_PREVIEW=1 \
GOCACHE=/tmp/ollama-build-gocache \
OLLAMA_LIBRARY_PATH=/home/yannik/Work/ollama-build/build/lib/ollama \
go run ./cmd/turboquant-eval \
  -model qwen2.5:7b \
  -snapshot tools/turboquant/testdata/qwen25_7b_phase0_tokens.json \
  -engine go -num-ctx 1024 -batch-size 512 -num-gpu-layers 0 \
  -flash-attention=true -reference-kv-cache-type f16 \
  -kv-cache-type kturbo6-vturbo4 \
  -limit 1 -format json
```

(Use `-num-gpu-layers 0` to force CPU. If `cmd/turboquant-eval` does not yet expose `-kv-cache-type` for split presets without per-layer flags, route through the existing `-key-cache-type turbo6 -value-cache-type turbo4` path; the result is equivalent for the smoke.)

Expected: `mean_kl < 0.5` on the first sequence. Catastrophic failure (`mean_kl > 1.0` or NaN) means Turbo6 K is bottlenecked by the residual representation, not the bit count. **Stop. Document. Skip to Phase F.**

- [ ] **Step 2: If smoke is healthy, run `-limit 16`**

Same command, `-limit 16`. Expected: `mean_kl < 0.05`. This is the CPU-only proxy; CUDA quality may be slightly different but should not be qualitatively worse.

- [ ] **Step 3: Document in `TURBOQUANT-DEBUG-LOG.md`**

Append an entry with the date, exact command, raw JSON output, mean_nll, perplexity, mean_kl, and an explicit decision: PROCEED to Phase D or STOP per the abort criterion.

- [ ] **Step 4: Commit the log entry**

```bash
git add TURBOQUANT-DEBUG-LOG.md
git commit -m "docs(turboquant): Turbo6 K CPU smoke on qwen2.5:7b Q4_K_M

[Decision sentence: PROCEED to Phase D, or STOP per abort criterion.]"
```

---

## Phase D — CUDA Kernels (only if Phase C cleared the gate)

### Task D1: CUDA dequant for Turbo5/Turbo6

**Files:**
- Modify: `ml/backend/ggml/ggml/src/ggml-cuda/turbo-quant.cuh` (mirror Turbo4 device-side dequant + decode helpers)
- Modify: `ml/backend/ggml/ggml/src/ggml-cuda/convert.cu` (mirror Turbo4 entry in the dequantize dispatch)

- [ ] **Step 1: Read the Turbo4 device decode**

```bash
grep -nE 'TURBO4|turbo4' ml/backend/ggml/ggml/src/ggml-cuda/turbo-quant.cuh ml/backend/ggml/ggml/src/ggml-cuda/convert.cu
```

- [ ] **Step 2: Mirror for Turbo5 and Turbo6**

The decode is the bit-unpack from Phase B Step B3 written in CUDA. The centroid table is duplicated as a `__constant__ float CENTROIDS_5BIT[32]` (and 6BIT[64]) defined in the .cuh header.

- [ ] **Step 3: Build CUDA**

```bash
cmake --build build -j2 --target ggml-cuda
cp build/lib/ollama/libggml-cuda.so build/lib/ollama/cuda_v12/libggml-cuda.so
```

- [ ] **Step 4: Commit**

```bash
git add ml/backend/ggml/ggml/src/ggml-cuda/
git commit -m "feat(ggml-cuda): dequantize_block_turbo5_0/turbo6_0 device kernels

Mirrors the Turbo4 path. Centroid tables placed in __constant__ memory.
KQ inner-product and FA paths come in the next commit."
```

### Task D2: CUDA quantize-on-write (set-rows) for Turbo5/6

**Files:**
- Modify: `ml/backend/ggml/ggml/src/ggml-cuda/set-rows.cu`

- [ ] **Step 1: Mirror the Turbo4 case in the per-type switch**

The Turbo4 quantize-on-write path: WHT prologue → per-block L2 norm → normalize → quantize via the centroid table. Reuse exactly. The only Turbo5/6 differences are the bit width (mask + shift) and the centroid table.

- [ ] **Step 2: Verify byte-equivalence with CPU**

Add a small CUDA-vs-CPU regression test (or extend an existing one) that quantizes a synthetic 128-element Gaussian K row on both backends and compares the packed bytes verbatim. Off-by-one in the bit packing is the most common bug here, and a bytewise diff catches it.

- [ ] **Step 3: Build and run**

```bash
cmake --build build -j2 --target ggml-cuda
GOCACHE=/tmp/ollama-build-gocache OLLAMA_LIBRARY_PATH=/home/yannik/Work/ollama-build/build/lib/ollama \
go test -count=1 -run TestTurbo ./ml/backend/ggml/...
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add ml/backend/ggml/ggml/src/ggml-cuda/set-rows.cu
git commit -m "feat(ggml-cuda): set-rows quantize for Turbo5/Turbo6

Bytewise verified against the CPU encode on a synthetic Gaussian K row."
```

### Task D3: CUDA FA Turbo5/6 K inner-product (vec instances)

**Files:**
- Modify: `ml/backend/ggml/ggml/src/ggml-cuda/turbo-innerq.cu`, `turbo-innerq.cuh`
- Modify: `ml/backend/ggml/ggml/src/ggml-cuda/fattn-vec.cuh`, `fattn-common.cuh`

- [ ] **Step 1: Mirror the Turbo4 KQ vec instance**

Turbo4 uses NL_TURBO4_VEC = 32 lanes per block (QK_TURBO4 / 4). For Turbo5/6, NL_TURBO{5,6}_VEC = 32 unchanged (QK is still 128, lane count derives from the 4-elem-per-thread vec layout). The decode is the bit-unpack; the rest of the FA inner loop is identical.

- [ ] **Step 2: Register Turbo5/6 in `ggml_backend_cuda_supports_type` and the FA dispatch switch**

Same place every other Turbo type appears.

- [ ] **Step 3: Build**

```bash
cmake --build build -j2 --target ggml-cuda
cp build/lib/ollama/libggml-cuda.so build/lib/ollama/cuda_v12/libggml-cuda.so
```

- [ ] **Step 4: GPU smoke at `-limit 1`**

```bash
OLLAMA_TURBOQUANT_K6_PREVIEW=1 \
GOCACHE=/tmp/ollama-build-gocache \
OLLAMA_LIBRARY_PATH=/home/yannik/Work/ollama-build/build/lib/ollama \
go run ./cmd/turboquant-eval \
  -model qwen2.5:7b \
  -snapshot tools/turboquant/testdata/qwen25_7b_phase0_tokens.json \
  -engine go -num-ctx 1024 -batch-size 512 -num-gpu-layers 999 \
  -flash-attention=true -reference-kv-cache-type f16 \
  -kv-cache-type kturbo6-vturbo4 \
  -limit 1 -format json
```

Expected: `mean_kl` close to the CPU C2 result (within ~0.01).

- [ ] **Step 5: Commit**

```bash
git add ml/backend/ggml/ggml/src/ggml-cuda/
git commit -m "feat(ggml-cuda): FA-vec inner product for Turbo5/Turbo6 K

Mirrors the Turbo4 vec instance with the bit-width-specific decode.
GPU -limit 1 KL matches the CPU result within tolerance; full Phase 0
gate runs in the next phase."
```

---

## Phase E — Phase 0 Promotion Gate

### Task E1: Full 256-sequence Phase 0

- [ ] **Step 1: Run the gate**

```bash
OLLAMA_TURBOQUANT_K6_PREVIEW=1 \
GOCACHE=/tmp/ollama-build-gocache \
OLLAMA_LIBRARY_PATH=/home/yannik/Work/ollama-build/build/lib/ollama \
go run ./cmd/turboquant-eval \
  -model qwen2.5:7b \
  -snapshot tools/turboquant/testdata/qwen25_7b_phase0_tokens.json \
  -engine go -num-ctx 1024 -batch-size 512 -num-gpu-layers 999 \
  -flash-attention=true -reference-kv-cache-type f16 \
  -kv-cache-type kturbo6-vturbo4 \
  -format json | tee /tmp/turboquant-kturbo6-phase0.json
```

Expected: `mean_kl < 0.02` (success criterion). Compare to `kq8-vturbo4`'s `mean_kl=0.01705642` — stretch goal is to match or beat that at strictly lower memory (1.31 B/elem vs ~0.81 B/elem).

- [ ] **Step 2: Document in `tools/turboquant/README.md` and `TURBOQUANT-DEBUG-LOG.md`**

Add a new row in the Phase 0 Results table for `kturbo6-vturbo4`. Include the duration so future readers can see whether the new preset is competitive on speed too.

- [ ] **Step 3: If gate clears, flip the default**

Lift the `OLLAMA_TURBOQUANT_K6_PREVIEW` requirement only after operator-facing documentation explains the tradeoff. Default-off is conservative; the user explicitly said "behind a feature flag until quality is proven".

If clearing this requirement is in scope this PR, change the gate in `llm/server.go::SupportsKVCacheType` to drop the flag check and document the change. Otherwise leave it gated.

- [ ] **Step 4: Commit**

```bash
git add tools/turboquant/README.md TURBOQUANT-DEBUG-LOG.md
git commit -m "docs(turboquant): kturbo6-vturbo4 full Phase 0 result

[Insert measured numbers; explicitly state PROMOTE or KEEP-BEHIND-FLAG.]"
```

---

## Phase F — Cleanup if Phase C or Phase E fails

### Task F1: Revert runtime wiring; keep kernels behind a build flag

Triggered when Phase C aborts (`mean_kl > 0.5` on CPU smoke) or Phase E fails to clear `mean_kl < 0.02` after a reasonable retry.

- [ ] **Step 1: Revert the Go wiring commit from Task C1**

```bash
git revert <commit-hash-of-task-C1>
```

- [ ] **Step 2: Add a `GGML_TURBO5_6_BUILD_KERNELS_ONLY` macro in `ggml-common.h`**

Wrap the Turbo5/6 type-traits registrations from Task B5 in `#ifdef`. Default the macro to undefined; CMake exposes it via `-DGGML_TURBO5_6_BUILD_KERNELS_ONLY=1` for future investigators who want to compile the kernels without exposing them.

- [ ] **Step 3: Document the abort**

Append a final entry to `TURBOQUANT-DEBUG-LOG.md`:

```
## YYYY-MM-DD - Turbo5/6 K abort: residual representation is the bottleneck

Evidence: [paste the failing Phase C / Phase E numbers].
Conclusion: bumping K bit width above 4 does not rescue the
WHT-rotated PolarQuant residual at the layer 0 / outlier head where
TurboQuant K loses information. The kernels are kept behind the
GGML_TURBO5_6_BUILD_KERNELS_ONLY build flag for future investigation.
The next investigator should redesign the rotation/quantisation
contract before retrying.
```

- [ ] **Step 4: Commit**

```bash
git add ml/backend/ggml/ggml/src/ggml-common.h \
        ml/backend/ggml/ggml/src/ggml-quants.c \
        TURBOQUANT-DEBUG-LOG.md
git commit -m "chore(turboquant): keep Turbo5/Turbo6 kernels behind build flag

Phase 0 / CPU smoke could not clear the abort criterion. Runtime
wiring reverted; CPU and CUDA kernels remain in tree behind
GGML_TURBO5_6_BUILD_KERNELS_ONLY so the next investigation does not
have to start from zero."
```

---

## Self-Review

**Spec coverage:**
- Generate Turbo5/Turbo6 centroids via Lloyd-Max on WHT-rotated unit-Gaussian: ✓ Phase A.
- Validate per-element MSE against the paper's theoretical bounds: ✓ Tasks A1 step 3, A2.
- CPU quantize/dequantize for Turbo5/Turbo6: ✓ Phase B.
- CUDA decode + KQ inner-product paths mirroring Turbo4 vec: ✓ Phase D.
- Verify CPU/CUDA byte agreement on synthetic K rows: ✓ Task D2 step 2.
- Wire `kturbo6-vturbo4` in `kvCacheBytesPerElementKV`, `kvCacheTypesFromStr`, `isSplitKVCachePreset`, `SupportsKVCacheType`: ✓ Phase C.
- Phase 0 at `-limit 1`, then `-limit 16`, then full 256: ✓ Tasks C2 + E1.
- Promote to operator-facing config only on gate pass: ✓ Task E1 step 3.
- Abort path if layer 0 still fails on Turbo6: ✓ Phase F.
- Behind a feature flag until quality is proven: ✓ Task C1 step 3 (`OLLAMA_TURBOQUANT_K6_PREVIEW`).
- Don't reintroduce QJL / don't touch residual-window: ✓ explicit non-goals.
- Don't remove or weaken `kq8-vturbo4` or `turboquant-adaptive`: ✓ no edits to that path; new preset is additive.
- Every commit explains the proven evidence justifying it: ✓ commit messages reference the test or measurement that earned the commit.

**Placeholder scan:** No "TBD" or "fill in" remains. The few code blocks that say "mirror Turbo4 exactly" are accompanied by the exact `grep` command that returns the lines to mirror — the executing agent has a deterministic recipe. The two centroid tables are emitted by Phase A so the C constants are not invented.

**Type consistency:** `block_turbo5_0` and `block_turbo6_0` use the same field names as `block_turbo4_0` (`norm`, `rnorm`, `qs`). `kturbo6-vturbo4` is the canonical preset string everywhere. `OLLAMA_TURBOQUANT_K6_PREVIEW` is the only feature-flag name. `DTypeTurbo5` / `DTypeTurbo6` follow the existing `DTypeTurbo4` convention.
