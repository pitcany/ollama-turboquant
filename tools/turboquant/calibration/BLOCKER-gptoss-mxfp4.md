# Blocker: gpt-oss MXFP4 calibration — turbo K cache vs head_dim=64

**Date:** 2026-05-04
**Status:** BLOCKED — architectural mismatch between ggml turbo dtype block size and gpt-oss head dimension.
**Affects:** `tools/turboquant/calibration/manifest_data/` cannot ship a `gptoss / MXFP4 / head_dim=64` entry until the K cache storage layout is fixed.
**Workaround for end users:** gpt-oss:20b will continue to fall back to the default `kq8-vturbo4` preset (no per-layer adaptive override).

## Symptom

Running `scripts/turboquant-calibrate.sh -m gpt-oss:20b -a gptoss -f MXFP4 -d 64 -t 4` aborts on
every layer of the single-layer sweep with:

```
ggml.c:1717: GGML_ASSERT(view_src == NULL || data_size == 0
                       || data_size + view_offs <= ggml_nbytes(view_src)) failed
SIGABRT: abort
```

Reproduces with **any** turbo K-cache type for gpt-oss, not just the per-layer override path:

```bash
go run ./cmd/turboquant-eval -engine go -model gpt-oss:20b \
  -snapshot /tmp/turboquant-calibration-gptoss-mxfp4/tokens.json \
  -kv-cache-type turbo4 -reference-kv-cache-type f16 \
  -num-ctx 1024 -batch-size 512 -num-gpu-layers 999 \
  -flash-attention=true -limit 1 -format json
```

`-kv-cache-type f16` runs cleanly — the failure is specific to turbo K storage on gpt-oss.

## Root cause

Instrumentation in `kvcache.(*Causal).Put` (snapshot below) shows the failing reshape is the
*cache* tensor, not the input K:

```
[DBG] Causal.Put layer=0 kHeadDim=64 numKVHeads=8 batchSize=512 cells=1024 keyDtype=f16 cacheKeyDtype=turbo4
[DBG]   key.Dim(0,1,2)=(64,8,512)
[DBG]   reshaping key     -> (512, 512)    # passes
[DBG]   reshaping keyCache-> (512, 1024)   # ABORT
```

`kvcache/causal.go:518` allocates the cache as

```go
c.keys[layer] = ctx.Zeros(turbo4_dtype, kHeadDim, numKVHeads, len(c.cells))
            //   = Zeros(turbo4,        64,        8,          1024)
```

For ggml, every quantized type defines `blck_size`. From `ml/backend/ggml/ggml/src/ggml-common.h:283`:

```
#define QK_TURBO4 128
```

(Same value for `QK_TURBO2`, `QK_TURBO3`, `QK_TURBO5`, `QK_TURBO6`.) The leading dim must be a
multiple of `blck_size`. ggml computes the per-row stride as
`nb[1] = type_size * ne[0] / blck_size`. With `ne[0] = 64` and `blck_size = 128`, integer division
yields `nb[1] = 0`, so the cache buffer's reported `nbytes` is far smaller than the data the
subsequent `Reshape(kHeadDim*numKVHeads, len(c.cells)) = Reshape(512, 1024)` needs to address —
hence the `data_size > nbytes(view_src)` assertion fires.

Existing manifest entries are unaffected because they use `head_dim = 128` (qwen2 / qwen3moe / qwen35),
which exactly matches `QK_TURBO* = 128`.

## Why the obvious fixes do not work

| Option | Why it fails |
| --- | --- |
| Allocate cache as 2D `(kHeadDim*numKVHeads, cells)` | `kvcache.(*Causal).Get` (`causal.go:462-475`) reads the cache with `Dim(0)/Dim(1)` and a 3D `View` — needs `(kHeadDim, numKVHeads, cells)` semantics. Reading turbo storage at a sub-`blck_size` leading dim (64 of 128) is not supported by ggml: there is no way to slice a row smaller than the block. |
| Pad `kHeadDim` to `blck_size` | Wastes 50% of K-cache memory and breaks every consumer that indexes `key.Dim(0) == head_dim`, including the FA-vec kernel selection. |
| Insert `Contiguous` after `ChunkSections` in `model/models/gptoss/model.go:125` | Tested — does not change behavior. Verified the failing reshape is on the cache buffer, not the model-side K view. |
| Switch to `-engine llama` | Per-layer K dtype overrides (`-key-cache-layer-types`) require `-engine go` (`cmd/turboquant-eval/main.go:233-234`), and the underlying turbo block constraint also exists at the C++ runner level. |

## What the real fix probably looks like

A turbo K-cache layout that decouples physical storage from logical `(head_dim, num_kv_heads, cells)`
indexing. Either:

1. Introduce a head-packed turbo storage type whose `blck_size` is `min(head_dim, 128)` *or* whose
   row spans `head_dim*num_kv_heads` natively, plus matching dequant kernels for FA-vec at
   `head_dim < 128`.
2. Switch the K-cache layout to `(head_dim*num_kv_heads, cells)` everywhere (storage, `Get`,
   FA-vec kernel `nthreads_KQ_q` selection) and drop the implicit `(head_dim, num_kv_heads, cells)`
   view contract. Get-side code becomes a single 2D `View(rowSize*c.curCellRange.min, ...)`; FA-vec
   reads heads as adjacent slabs within the row.

Both options require ggml-side changes and a kernel/cache contract update. They are out of scope for
a calibration session.

## Pre-flight notes (still useful when the cache layout lands)

The Phase B token snapshot is already produced and should be reused:

- Tokenizer family: o200k-derived (GGUF reports `tokenizer.ggml.model = gpt2`, `pre = default`).
- Snapshot: `/tmp/turboquant-calibration-gptoss-mxfp4/tokens.json` (16 sequences × 1024 tokens, 227 KB).
- Source: WikiText-103 raw + repo-doc subset (the canonical corpus from
  `tools/turboquant/README.md:474-489`).
- Architecture: `gptoss` (block_count=24, head_count=64, head_count_kv=8, head_dim=64,
  embedding_length=2880, sliding_window=128).
- File type: `MXFP4` (`GGML_TYPE_MXFP4 = 39` per `ml/backend/ggml/ggml/include/ggml.h:424`).
- FA-vec kernel coverage at `D=64` already includes turbo K + turbo V combinations
  (`ml/backend/ggml/ggml/src/ggml-cuda/template-instances/fattn-vec-instance-turbo*.cu`), so once the
  cache-storage issue is resolved, the kernel side should be ready.

Calibration command, ready to run after the fix:

```bash
scripts/turboquant-calibrate.sh \
  -m gpt-oss:20b \
  -s /tmp/turboquant-calibration-gptoss-mxfp4/tokens.json \
  -a gptoss -f MXFP4 -d 64 \
  -t 4 -k 0.05 -V 16
```
