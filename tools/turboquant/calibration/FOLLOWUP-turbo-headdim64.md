# Follow-up: head-packed turbo K/V cache for head_dim < 128

**Date opened:** 2026-05-03
**Owner:** unassigned
**Predecessor:** [BLOCKER-gptoss-mxfp4.md](./BLOCKER-gptoss-mxfp4.md)
**Goal:** Land a turbo K/V cache layout that works on models with `head_dim < 128`,
restoring access to per-layer adaptive K-dtype overrides on `gpt-oss`-class models
(`head_dim=64`) and any future architecture in the same regime.

## Why this is multi-PR

The current turbo dtypes (`turbo2..turbo6`) all have `blck_size = 128`, and that
constant is structural — the WHT/Hadamard rotation that gives turbo its accuracy
operates on a 128-element block (`QK_TURBOn_GROUP = 128`). The kvcache.Causal
allocation pattern is `(head_dim, num_kv_heads, cells)`, so the leading dim is
`head_dim`. When `head_dim < 128`, ggml's per-row stride `nb[1] = type_size *
ne[0] / blck_size` integer-divides to zero and the buffer is too small for any
subsequent reshape or view. There is no way to slice a quantized tensor at
sub-`blck_size` boundaries, so options that try to keep the existing layout and
"just" reshape don't work.

The only path that preserves turbo's accuracy guarantees is a parallel turbo
family with `blck_size = 64` and a 64×64 WHT, plus matching FA-vec kernel
instances and a manifest schema that keys on head_dim. That's a multi-PR
project, not a calibration-session task.

## Scope (rough breakdown)

Each line is one PR's worth of work. Order matters — later PRs depend on
earlier ones being in main.

1. **PR-1 — `turbo4_64` reference quantize/dequantize on CPU**
   Add a new ggml type ID (e.g. `GGML_TYPE_TURBO4_0_64 = 60`), its
   `block_turbo4_0_64` packed struct (`blck_size = QK_TURBO_64 = 64`, half the
   payload of turbo4_0 per block), and the CPU `quantize_row_turbo4_0_64_ref`
   and `dequantize_row_turbo4_0_64` paths. Wire into `ggml.c` type-traits
   table and the CPU dispatcher. Add `static_assert(QK_TURBO_64 == 64, ...)`.
   Tests: round-trip 1024-element vectors, compare WHT-derived MSE against
   turbo4_0 at the same compression ratio.

2. **PR-2 — `turbo4_64` CUDA quantize/dequantize**
   Port the row-pack kernels to `ggml-cuda/turbo-quant.cu` (or a new
   `turbo-quant-64.cu`). This is mostly mechanical — same WHT structure,
   smaller matrix.

3. **PR-3 — FA-vec template instances at D=64 for `turbo4_64`**
   Add `fattn-vec-instance-turbo4_0_64-turbo4_0_64.cu` etc. Update
   `fattn-vec.cuh` if the dispatcher needs to know the new type. Verify the
   instance is picked up at `D=64`.

4. **PR-4 — Repeat PRs 1-3 for `turbo2_64`, `turbo3_64`**
   `turbo5_64` and `turbo6_64` are nice-to-have; only do them if calibration
   shows they outperform `turbo4_64` on real workloads.

5. **PR-5 — Manifest + runtime resolution**
   Extend `ManifestEntry` so the resolver can pick `turbo4_64` for
   `head_dim=64` and `turbo4` for `head_dim=128`. Update
   `kvCacheTypesFromStr` so the runtime accepts `turbo4_64` as a cache type.
   Loosen `fs/ggml/ggml.go:SupportsKVCacheType` to allow `turbo*_64` when
   `head_dim%64 == 0`. Tests in `runner/ollamarunner/cache_test.go`.

6. **PR-6 — Calibration tooling + first artifact**
   Drop the `head_dim < 128` gate in `scripts/turboquant-calibrate.sh` for
   the `_64` variant. Re-run `gpt-oss:20b` calibration using
   `/tmp/turboquant-calibration-gptoss-mxfp4/tokens.json` (the snapshot is
   tokenizer-stable). Bundle the artifact under `manifest_data/`. Update
   `BLOCKER-gptoss-mxfp4.md` to CLOSED.

## Out of scope here

- Changing the existing `turbo*` dtype semantics. They stay as-is; the
  `_64` variants are additive.
- New compression algorithms. PolarQuant + WHT structure carries over
  unchanged at the smaller block size.
- Migration tooling — there is nothing to migrate; calibrations are keyed
  on `(arch, file_type, head_dim)` and the manifest can hold both 128 and 64
  variants side by side.

## Pre-flight notes (still useful when PR-1 lands)

The Phase B token snapshot is already produced and should be reused:

- Tokenizer family: o200k-derived (GGUF reports `tokenizer.ggml.model = gpt2`,
  `pre = default`).
- Snapshot: `/tmp/turboquant-calibration-gptoss-mxfp4/tokens.json` (16
  sequences × 1024 tokens, 227 KB).
- Architecture: `gptoss` (block_count=24, head_count=64, head_count_kv=8,
  head_dim=64, embedding_length=2880, sliding_window=128).
- File type: `MXFP4` (`GGML_TYPE_MXFP4 = 39`).
- FA-vec kernel coverage at `D=64` already includes turbo K + turbo V
  combinations for `turbo*_0`; the new `turbo*_0_64` variants need their
  own instances (PR-3).

Calibration command, ready to run after PR-5:

```bash
scripts/turboquant-calibrate.sh \
  -m gpt-oss:20b \
  -s /tmp/turboquant-calibration-gptoss-mxfp4/tokens.json \
  -a gptoss -f MXFP4 -d 64 \
  -t 4 -k 0.05 -V 16
```
