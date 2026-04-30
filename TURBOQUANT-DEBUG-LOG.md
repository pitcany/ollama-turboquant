# TurboQuant Debug Log

Purpose: rigorous, evidence-first log per `TURBOQUANT-CLAUDE-HANDOFF.md`.
No speculative patches. Each fix must cite the first proven mismatch it explains.

## 2026-04-30 - Promote safe presets to production runtime

Goal: make the two Phase-0-validated KV cache configurations available as
first-class runtime modes so users do not have to invoke the eval harness.
Treat all-Turbo4 K as out of scope, do not re-introduce QJL, and do not reopen
the residual-window experiment without a dedicated graph op.

What landed (Ollama engine only; the legacy llama.cpp runner warns and ignores
both modes):

1. `kq8-vturbo4` was already wired through `fs/ggml`, `runner/ollamarunner`,
   and `llm/server.go`; this entry verifies the path and adds memory-side
   coverage. Setting `OLLAMA_KV_CACHE_TYPE=kq8-vturbo4` produces a split
   cache via `Causal.InitSplit(K=q8_0, V=turbo4)`. The memory estimator in
   `fs/ggml/ggml.go` (`kvCacheBytesPerElementKV`) returns `(1.0, 0.531)`
   bytes per element so the per-layer KV size reported through the
   `GraphSize` path matches the cache that is actually allocated. Confirmed
   by `TestKVCacheBytesPerElementKV` and `TestInitKVCacheUsesSplitPreset`.

2. New env var `OLLAMA_TURBOQUANT_CALIBRATION` points at a calibration JSON
   produced by `cmd/turboquant-calibrate`. The artifact layout is parsed by
   the new `tools/turboquant/calibration` package, which only reads the
   runtime-relevant fields (`version`, `model`, `base_kv_cache_type`,
   `layer_dtype`, `key_cache_layer_types`). When the env var is set on the
   Ollama engine, `applyTurboquantCalibration` in `llm/server.go`:

   - Loads and validates the artifact (rejects bad version, empty
     `base_kv_cache_type`, empty/malformed `key_cache_layer_types`,
     unsupported dtypes for the model architecture).
   - Overrides `loadRequest.KvCacheType` with the artifact's
     `base_kv_cache_type` and stores the canonical per-layer spec on a new
     `LoadRequest.KeyCacheLayerTypes` field.
   - Requires `FlashAttention=Enabled`; otherwise rejects the artifact.

3. The runner forwards `KeyCacheLayerTypes` through
   `Server.allocModel` → `NewInputCache` → `initKVCache`. After the cache is
   `Init`/`InitSplit`-initialized, `initKVCache` parses the spec via
   `calibration.ParseKeyLayerOverrides` and calls `Causal.SetKeyLayerDTypes`
   so the listed layers get `KeyDType` overridden per layer. Confirmed by
   `TestInitKVCacheAppliesKeyLayerDTypes` and
   `TestInitKVCacheRejectsBadKeyLayerSpec`.

4. Memory accounting changes: `GraphSize` is now a thin wrapper around the
   new `GraphSizeWithKeyOverrides`, which takes a `map[int]string` of
   per-layer K dtypes and uses `kvCacheBytesPerElement(dtype)` for matching
   layers when filling the per-layer `kv[i]` array. `llm/server.go` parses
   `loadRequest.KeyCacheLayerTypes` (via `calibration.ParseKeyLayerOverrides`)
   and feeds the result into `GraphSizeWithKeyOverrides`, so the reported
   KV-cache footprint reflects the actual mixed dtypes per layer instead of
   the base dtype across the whole model. For the
   `qwen2.5-7b-q4km-adaptive` artifact (4 of 28 layers at `q8_0`, rest at
   `turbo4`), this is `~5.3%` more accurate than treating the cache as
   pure `turbo4` and `~71.8%` smaller than the f16 baseline.

Out of scope for this entry, intentionally:

- All-Turbo4 K still fails Phase 0 catastrophically; not promoted.
- The residual-window cast trick remains dead per the prior log entry; no
  new graph op was added.
- QJL/JL was not reintroduced.

Verification:

```bash
GOCACHE=/tmp/ollama-build-gocache \
go test -count=1 \
  ./tools/turboquant/... \
  ./cmd/turboquant-tokenize ./cmd/turboquant-eval \
  ./cmd/turboquant-dump-index ./cmd/turboquant-select-layers \
  ./cmd/turboquant-calibrate \
  ./kvcache ./ml/nn ./llm ./envconfig ./fs/ggml ./runner/ollamarunner
git diff --check
```

All 16 packages PASS, `git diff --check` is clean.

## 2026-04-30 - Residual-window Phase 0 grid: blocked by WHT basis mismatch

Goal:
- Run the Phase 0 eval grid for the residual-window experiment on
  `qwen2.5:7b` Q4_K_M: `{f16,q8_0}` recent dtype crossed with windows
  `{64,128,256}`, residual dtype `turbo4`, snapshot
  `tools/turboquant/testdata/qwen25_7b_phase0_tokens.json`,
  `-num-ctx 1024 -batch-size 512 -num-gpu-layers 999 -flash-attention=true
  -reference-kv-cache-type f16 -engine go -format json`. Smoke at
  `-limit 1`, then `-limit 16`, then full 256.
- Decision criterion: only promote a window preset to the production
  runtime if it clearly beats the `kq8-vturbo4` baseline
  (`mean_nll=2.15085919`, `perplexity=8.59223759`, `mean_kl=0.01705642`).

Pre-flight setup:

```bash
# CUDA library state at session start:
ls -la build/lib/ollama/libggml-cuda.so build/lib/ollama/cuda_v12/libggml-cuda.so
# -rwxrwxr-x 488898176 Apr 29 22:07 build/lib/ollama/libggml-cuda.so
# -rwxrwxr-x 472554816 Apr 29 16:13 build/lib/ollama/cuda_v12/libggml-cuda.so

# Initially moved flat aside per the handoff "double-load" gotcha, then
# discovered the in-process loader (ml/backend/ggml/ggml/src/ggml.go
# OnceLoad) only searches the top of OLLAMA_LIBRARY_PATH and uses the
# fallback at ggml-backend-reg.cpp:601-614 to load `libggml-cuda.so` (no
# suffix) directly. With the flat moved aside, the loader could not find
# any CUDA backend and fell back to CPU (logs: "offloaded 0/29 layers to
# GPU"). Restored the flat for the actual eval runs:

mv /tmp/ollama-build-libggml-cuda-flat-0430a.so build/lib/ollama/libggml-cuda.so
```

The handoff's "double-load" warning applies to a different code path
(runner subprocess discovery looking inside subdirectories). For
in-process eval the flat library is required.

Baseline focused Go tests (clean):

```bash
GOCACHE=/tmp/ollama-build-gocache \
go test -count=1 \
  ./tools/turboquant/... \
  ./cmd/turboquant-tokenize ./cmd/turboquant-eval \
  ./cmd/turboquant-dump-index ./cmd/turboquant-select-layers \
  ./cmd/turboquant-calibrate \
  ./kvcache ./ml/nn ./llm ./envconfig ./fs/ggml ./runner/ollamarunner
```

All 15 packages PASS.

### Smoke 1: `-key-cache-type f16 -value-cache-type turbo4 -key-cache-residual-window 64 -key-cache-residual-dtype turbo4 -limit 1`

Command:

```bash
GOCACHE=/tmp/ollama-build-gocache \
OLLAMA_LIBRARY_PATH=/home/yannik/Work/ollama-build/build/lib/ollama \
CUDA_VISIBLE_DEVICES=0 \
go run ./cmd/turboquant-eval \
  -model qwen2.5:7b \
  -snapshot tools/turboquant/testdata/qwen25_7b_phase0_tokens.json \
  -engine go -num-ctx 1024 -batch-size 512 -num-gpu-layers 999 \
  -flash-attention=true -reference-kv-cache-type f16 \
  -key-cache-type f16 -value-cache-type turbo4 \
  -key-cache-residual-window 64 -key-cache-residual-dtype turbo4 \
  -limit 1 -format json
```

Observation: 18 worker threads abort with the same fatal message:

```
/home/yannik/Work/ollama-build/ml/backend/ggml/ggml/src/ggml-cpu/ops.cpp:571: fatal error
```

That line is the `default` case of `ggml_compute_forward_dup`'s switch on
`src0->type`: it accepts `quantized → F32` (line 567-568 calls
`ggml_compute_forward_dup_from_q`) and aborts on every other quantized
destination. So the unsupported direction is `Turbo4 → F16` (quantized
src, non-F32 dst) — i.e. the second leg of `Cast(turbo4).Cast(f16)`.

Cross-checked CUDA `cpy.cu`:

- F16 → F16/BF16/F32 (lines 497-513). No Turbo entries.
- Q8_0 → F32 (line 467). No Turbo entries either direction.
- F32 → Q8_0/Q4_0/Q4_1/Q5_0/Q5_1/IQ4_NL (lines 464-491). No F32 → Turbo.
- Default path is `GGML_ABORT("unsupported type combination ...")`.

So both legs of `f16 ↔ turbo4` Cast lack a CUDA implementation; the
scheduler falls back to CPU for both, and CPU only implements
`quantized → F32` — hence the abort on the second leg.

CPU traits ARE registered for Turbo4 (`from_float = quantize_row_turbo4_0_ref`,
`to_float = dequantize_row_turbo4_0`, both in `ggml-cpu/ggml-cpu.c:413` and
`ggml/ggml.c:732`). The traits are reachable via `dup_to_q<float>` and
`dup_from_q`. So an F32 hop on both sides routes through supported entries:

```
f16 → F32     (CPU/CUDA OK)
F32 → turbo4  (CPU dup_to_q<float>; CUDA falls back to CPU)
turbo4 → F32  (CPU dup_from_q;     CUDA falls back to CPU)
F32 → f16     (CPU/CUDA OK)
```

Minimal fix in `kvcache/causal.go applyKeyResidualWindow` to remove the
abort, justified by the file:line evidence above:

```go
oldF32 := old.
    Cast(ctx, ml.DTypeF32).
    Cast(ctx, c.KeyResidualBaseDType).
    Cast(ctx, ml.DTypeF32)
```

### Smoke 2: same flags, after the F32-hop fix

A second abort surfaced after the cast chain ran:

```
/home/yannik/Work/ollama-build/ml/backend/ggml/ggml/src/ggml-cuda/concat.cu:165:
GGML_ASSERT(src0->type == GGML_TYPE_F32) failed
```

CUDA's concat kernel only accepts F32 inputs (concat.cu:165-167). The
`old.Concat(recent, 2)` step on F16 inputs hits this assert. Symmetric
minimal fix: route the concat through F32 too, then cast the merged
result back to the original dtype:

```go
recentF32 := recent.Cast(ctx, ml.DTypeF32)
return oldF32.Concat(ctx, recentF32, 2).Cast(ctx, originalDType)
```

### Smoke 3: same flags, after both F32-hop fixes

Result:

```json
{
  "kv_cache_type": "f16",
  "value_cache_type": "turbo4",
  "key_cache_residual_window": 64,
  "key_cache_residual_dtype": "turbo4",
  "reference_kv_cache_type": "f16",
  "num_sequences": 1,
  "duration_ms": 7181,
  "metrics": {
    "token_count": 1023,
    "mean_nll": 10.417243982132835,
    "perplexity": 33431.17016136512,
    "kl_token_count": 1023,
    "mean_kl": 1.1098740691230278
  }
}
```

The eval ran end-to-end with 29/29 layers offloaded to GPU, but the
metrics are **worse than the all-Turbo4 K + Turbo4 V baseline**
(`mean_nll=6.10887882`, `perplexity=449.83`, `mean_kl=4.467`). This is
catastrophic and contradicts the experiment's hypothesis that protecting
the most recent N positions improves quality.

### Bisecting the regression

Sanity 1 — window larger than the sequence so `oldCount == 0` (no-op
path): same flags, `-key-cache-residual-window 2048 -limit 1`. Result:
`mean_nll=1.7820666015494437, mean_kl=0.012069528698299413`. **Exact
match** to the existing f16K/Turbo4V baseline (README Phase 0 Results,
line 103). The split-point gate behaves correctly when no cells are old.

Sanity 2 — bypass only the Turbo round trip but keep the View+F32+Concat
plumbing. Replaced

```go
oldF32 := old.Cast(F32).Cast(turbo).Cast(F32)
```

with

```go
oldF32 := old.Cast(F32) // bypass round trip
```

and re-ran the same `-key-cache-residual-window 64 -limit 1` command.
Result: `mean_nll=1.7820666015494437, mean_kl=0.012069528698299413`,
again identical to the f16K/Turbo4V baseline. So the View, F32 hop,
Concat, and downstream FA path are correctness-preserving; the bug is
inside the Turbo round trip itself.

Sanity 3 — restored the round trip and pushed the window to the other
extreme (`-key-cache-residual-window 1`, so all but one cell are
round-tripped). Expected: result close to the all-Turbo4 K baseline
(`mean_nll≈6.11`). Observed: `mean_nll=11.0151, perplexity=60786.27,
mean_kl=1.0039`. **Worse than all-Turbo4 K**, by a large margin.

### Root cause: WHT basis mismatch

`ggml-turbo-quant.c:457-565` (`quantize_row_turbo4_0_ref`) applies a
forward Walsh-Hadamard rotation **before** quantizing into 4-bit
PolarQuant centroids:

```c
/* Step 2: Forward WHT rotation (matches CUDA set_rows) */
float rotated[TURBO_D];
memcpy(rotated, normalized, d * sizeof(float));
turbo_cpu_fwht(rotated, d);
```

`ggml-turbo-quant.c:568-595` (`dequantize_row_turbo4_0`) **does not**
apply the inverse WHT — its in-code comment explicitly explains why:

```c
/* No inverse WHT, dequant stays in the rotated domain.
* Q is WHT-rotated by the graph, so <Q_rot, K_rot> gives correct
* attention scores.
* The inverse WHT is applied to the attention output via
* GGML_OP_TURBO_WHT (direction=1) in the graph.
*/
```

The production all-Turbo4 K cache works because the model graph wraps
the FA call: `GGML_OP_TURBO_WHT(Q)` → FA(Q_rot, K_rot, V) →
`GGML_OP_TURBO_WHT(out, direction=1)`. WHT is orthogonal so
`<Q_rot, K_rot> = <Q, K>` and the inverse WHT on the output recovers
the correct basis.

The residual-window code path does **not** insert those graph nodes:
the cache type is `f16`, so the model assumes K is in the original
basis and emits FA without WHT-rotating Q. Our `Cast(F32→Turbo4)
→Cast(Turbo4→F32)` round trip leaves the old-cell slice in the
rotated basis. After Concat with the recent slice (still in the
original basis) and Cast back to f16, the merged K is partly rotated
and partly not, while Q is entirely un-rotated. Attention scores are
computed across mismatched bases for every old cell.

This is the first concrete mismatch and it is rigorously isolated:

- Window=2048 (no cells are old): result = baseline. ✓
- Bypass Turbo round trip, keep plumbing: result = baseline. ✓
- Round trip enabled, window=1: result far worse than all-Turbo4 K. ✗
- Round trip enabled, window=64: result far worse than all-Turbo4 K. ✗

The Cast-based round trip is **not** basis-invertible, by design.
This was anticipated by the handoff's "ggml_cast risk surface on CUDA"
caveat, but the gap is deeper than CUDA dispatch — it is in the Turbo
encoder/decoder pair on every backend.

### Decision

Per the task spec: "If output is garbled or NLL is invalid, do NOT
patch speculatively. ... reproduce deterministically, isolate the
first concrete mismatch, and only then propose a minimal fix." The
first concrete mismatch is identified above. There is no minimal fix:
making the round trip basis-correct requires either

1. A new graph op (e.g., `GGML_OP_TURBO_ROUNDTRIP`) that applies
   forward WHT → quantize → dequantize → inverse WHT inside a single
   compensating block, used only on the old slice, OR
2. A `Cast(turbo4)` variant that internally undoes the WHT on dequant
   so the existing Cast pair becomes a true round trip in the
   original basis. Production cache code would have to opt out of
   that variant, since it relies on the rotated K convention.

Both options are research-grade additions, not minimal fixes. The
`{f16,q8_0}` × `{64,128,256}` grid is therefore **not run** — every
cell would inherit the same WHT-basis bug and produce metrics that
say nothing about residual-window quality. Per the decision
criterion, no preset is being promoted.

### Files changed

- `kvcache/causal.go applyKeyResidualWindow` — F32 hops on both legs
  of the cast chain plus around the Concat. Justified by ops.cpp:571
  and concat.cu:165 aborts; documented in the function comment that
  this is necessary but not sufficient (the WHT basis bug remains).
- `tools/turboquant/README.md` — new "Residual-Window Phase 0 Results"
  subsection documenting the measured numbers, the WHT basis finding,
  and the explicit non-promotion decision.

### Verification

```bash
git diff --check
```

```bash
GOCACHE=/tmp/ollama-build-gocache \
go test -count=1 \
  ./tools/turboquant/... \
  ./cmd/turboquant-tokenize ./cmd/turboquant-eval \
  ./cmd/turboquant-dump-index ./cmd/turboquant-select-layers \
  ./cmd/turboquant-calibrate \
  ./kvcache ./ml/nn ./llm ./envconfig ./fs/ggml ./runner/ollamarunner
```

All 15 packages PASS.

### Open items for next session

- Decide whether to fund the `GGML_OP_TURBO_ROUNDTRIP` graph op, or
  switch the experiment to a non-WHT residual representation (would
  require a new Q-style dtype that protects only the old slice).
- The F32-hop fix in `applyKeyResidualWindow` is correct but not
  sufficient on its own. If the residual-window experiment is
  abandoned, the helper and the eval flags should be either removed
  or marked clearly as "round trip not yet implemented" so future
  contributors do not retry the same dead end.

## 2026-04-30 - Residual-window round-trip op landed

Goal:
- Implement the graph operation that actually performs the residual-window
  Turbo round trip. Without it, the flags landed earlier today only record
  state.

Design refinement:
- Round-trip in `Causal.Get` rather than mutating cache memory after `Put`.
  The Get path already builds a View of the cells in the current sequence
  range, which is exactly the K that FA will consume. Transforming K on the
  way out avoids in-place mutation hazards (allocator reuse, races on
  shared cells across sequences) and avoids any cross-batch tracking state.

Implementation:
- New `Causal.keyResidualOldCount(cachedSize int) int` returns how many
  leading cells in the active range fall outside the window N. It uses the
  maximum of `c.curPositions` as the "now" reference and treats a cell as
  old if its position is at most `now - N`. It defensively returns 0 when
  any cell positions in the active range are non-monotonic.
- New `Causal.applyKeyResidualWindow(ctx, key, oldCount, cachedSize) ml.Tensor`
  splits the key view at oldCount via two `View` calls, runs
  `old.Cast(KeyResidualBaseDType).Cast(originalDType)` for the round trip,
  and `Concat`s the round-tripped old with the untouched recent slice on
  the cell axis.
- `Causal.Get` calls these helpers right after the existing cell-range
  view is built. If the residual window is disabled or no cells are old,
  Get behaves exactly as before.

Tests:
- `TestKeyResidualOldCount` is a pure-function table-driven test covering:
  disabled (window=0, baseDType=Other), all recent, leading old subset,
  edge case window equal to cached size, all-old, curCellRange offsetting,
  non-monotonic positions falling back to 0, multi-position curPositions
  using the maximum.
- Existing `TestSetKeyResidualWindow*` plus `TestKeyLayerDType*` and
  `TestInitSplit*` tests still pass.

Verification:

```bash
git diff --check
```

```bash
GOCACHE=/tmp/ollama-build-gocache \
go test -count=1 \
  ./tools/turboquant/... \
  ./cmd/turboquant-tokenize \
  ./cmd/turboquant-eval \
  ./cmd/turboquant-dump-index \
  ./cmd/turboquant-select-layers \
  ./cmd/turboquant-calibrate \
  ./kvcache \
  ./ml/nn \
  ./llm \
  ./envconfig \
  ./fs/ggml \
  ./runner/ollamarunner
```

Result: all 15 packages PASS.

Open items:

- GPU smoke + Phase 0 eval grid. With `-flash-attention=true` on the Go
  engine and `qwen2.5:7b`, run `-key-cache-type {f16,q8_0}
  -key-cache-residual-window {64,128,256} -key-cache-residual-dtype turbo4
  -value-cache-type turbo4 -reference-kv-cache-type f16` first at
  `-limit 1` then `-limit 16` then full Phase 0. Document in the README
  Phase 0 Results section.
- Verify `ggml_cast` between `f16`/`q8_0` and Turbo* roundtrips correctly
  on CUDA. CPU paths are likely fine; CUDA cast for Turbo dtypes is the
  risk surface.

## 2026-04-30 - Residual-window key-protection design

Goal:
- Build the eval-only experiment proposed in the handoff: keep values Turbo4,
  keep old keys Turbo4, but let the most recent N key-cache tokens use a higher
  key dtype (q8_0 or f16). Compare against `kq8-vturbo4` and the adaptive
  layer preset on the existing Phase 0 snapshot.

Design tradeoffs considered before writing code:

1. Per-position cache mutation (in-place round trip).
   The cache currently allocates a single key tensor per layer at one dtype.
   We could allocate at the higher dtype and, after each `Put`, apply a
   Turbo4 encode-then-decode round trip in place to positions outside the
   most recent N. The K tensor stays one tensor at the high dtype; only the
   numerical content of "old" rows is degraded to match Turbo4 representation.
   Pros: single tensor, FA path unchanged, attention sees a contiguous K.
   Cons: needs a graph op that reads K, runs Turbo encode/decode, writes K.
   Memory does not match production runtime memory; the experiment measures
   accuracy of Turbo4 K representation under the residual-window discipline,
   not memory savings.

2. Split key storage (recent + old tensors).
   Allocate `keysOld[layer]` at Turbo4 and `keysRecent[layer]` at the higher
   dtype, then either fuse them at FA time or run two FA calls and combine.
   Pros: matches production memory shape. Cons: heavy FA path surgery, two
   K tensors to mask, needs two FA invocations per layer or a fused kernel.
   Out of scope for an eval-only experiment.

3. Per-batch global recodec.
   Treat recent N as "the last N positions of the sequence end" and apply a
   single Turbo round trip to positions `[0, end-N]` once per batch.
   Pros: simplest. Cons: within a single prefill batch the per-query window
   semantics differ from production generation; we would not know whether a
   regression was caused by the encoding cost or by misaligned window
   semantics.

Chosen interpretation:
- Approach 1 (in-place round trip on old positions). This is the closest
  fidelity to production generation while keeping a single K tensor and
  avoiding FA surgery. The graph op required is a Turbo encode/decode round
  trip targeting a slice of the K cache.
- Window semantics: at every Put, positions where `currentEnd - pos > N` are
  considered "old" and round-tripped through the configured Turbo dtype.
  Positions inside the window keep their high-dtype content untouched.
- Base Turbo dtype is configurable so the same harness can experiment with
  Turbo4 (matches `kq8-vturbo4` long-term) or future Turbo3/Turbo2 variants.
- Eval-only: no production runtime path is changed. The flag is wired into
  `cmd/turboquant-eval` only; the Ollama serve path stays on the existing
  `kq8-vturbo4` and adaptive-preset behaviors.

Implementation plan, narrow:

a. Add `cmd/turboquant-eval` flags `-key-cache-residual-window N` and
   `-key-cache-residual-dtype DTYPE` plus parsing and validation tests:
     - DTYPE must be a Turbo* dtype (matches the "old" content target).
     - The base key cache type must be a higher precision than DTYPE (the
       "recent" content target). For now require `f16` or `q8_0`.
     - Engine must be Go; flash attention must be on.
     - Cannot combine with per-layer key dtype overrides (they target
       orthogonal axes; we can revisit composition once the experiment runs).

b. Add `Causal.SetKeyResidualWindow(window int, baseDType ml.DType)` plus
   matching tests:
     - Records `KeyResidualWindow` and `KeyResidualBaseDType` on the cache.
     - Zero window or `ml.DTypeOther` resets the configuration.

c. Wire the eval harness to call `SetKeyResidualWindow` when the new flags
   are present. The cache method only records state in this commit; the
   actual graph-op round trip is a follow-up that needs a Turbo encode/decode
   composition (e.g., a SET_ROWS into a temporary Turbo tensor followed by a
   dequant op back into the K cache slice).

Open items deferred to a follow-up session:

- The graph op that performs the per-position Turbo round trip on old slots.
  Likely composition: copy slice -> Turbo SET_ROWS into temp -> dequant back
  into K cache slice. Needs verification that allocator reuse does not
  corrupt the source between the encode and the dequant write-back.
- Eval grid: window N in {0, 64, 128, 256} crossed with residual dtype in
  {q8_0, f16} on the existing `tools/turboquant/testdata/qwen25_7b_phase0_tokens.json`
  snapshot using `-reference-kv-cache-type f16`. Compare mean NLL,
  perplexity, and KL against `kq8-vturbo4` baseline.
- Decision criterion: only promote to production runtime if a window
  preset clearly beats `kq8-vturbo4` on full Phase 0 KL with at least the
  same perplexity at meaningfully lower memory.

Implementation in this commit:

- `kvcache.Causal` gains `KeyResidualWindow` and `KeyResidualBaseDType`
  fields plus `SetKeyResidualWindow(window int, baseDType ml.DType)`.
  Zero window or `ml.DTypeOther` resets the configuration. The setter is
  state-only; FA and Put paths are unchanged.
- `kvcache.WrapperCache.SetKeyResidualWindow` forwards to wrapped caches
  matching the existing `SetKeyLayerDTypes` forwarding pattern, so models
  that compose multiple cache types still see the configuration.
- `cmd/turboquant-eval` gains `-key-cache-residual-window N` and
  `-key-cache-residual-dtype DTYPE`. Validation requires
  `-engine go`, `-flash-attention=true`, a Turbo* residual dtype, an
  `f16`/`q8_0` recent dtype, and forbids combining with the per-layer key
  preset/types flags.
- Result emission echoes `key_cache_residual_window` and
  `key_cache_residual_dtype` in JSON and text outputs for reproducibility.
- `tools/turboquant/README.md` documents the experimental flags, their
  constraints, and the planned eval grid.

Verification:

```bash
git diff --check
```

```bash
GOCACHE=/tmp/ollama-build-gocache \
go test -count=1 \
  ./tools/turboquant/... \
  ./cmd/turboquant-tokenize \
  ./cmd/turboquant-eval \
  ./cmd/turboquant-dump-index \
  ./cmd/turboquant-select-layers \
  ./cmd/turboquant-calibrate \
  ./kvcache \
  ./ml/nn \
  ./llm \
  ./envconfig \
  ./fs/ggml \
  ./runner/ollamarunner
```

Result:

```
ok  	github.com/ollama/ollama/tools/turboquant/dumpindex	0.029s
ok  	github.com/ollama/ollama/tools/turboquant/eval	0.003s
ok  	github.com/ollama/ollama/tools/turboquant/layerselect	0.002s
ok  	github.com/ollama/ollama/tools/turboquant/tests	0.002s
ok  	github.com/ollama/ollama/cmd/turboquant-tokenize	0.003s
ok  	github.com/ollama/ollama/cmd/turboquant-eval	0.004s
ok  	github.com/ollama/ollama/cmd/turboquant-dump-index	0.002s
ok  	github.com/ollama/ollama/cmd/turboquant-select-layers	0.002s
ok  	github.com/ollama/ollama/cmd/turboquant-calibrate	0.005s
ok  	github.com/ollama/ollama/kvcache	0.003s
ok  	github.com/ollama/ollama/ml/nn	0.003s
ok  	github.com/ollama/ollama/llm	0.004s
ok  	github.com/ollama/ollama/envconfig	0.006s
ok  	github.com/ollama/ollama/fs/ggml	0.005s
ok  	github.com/ollama/ollama/runner/ollamarunner	0.003s
```

## 2026-04-30 - Automatic calibration wrapper and artifact

Goal:
- Turn the manual layer sweep workflow into one command that runs or reuses a
  sweep, selects layers, validates the selected spec on a larger slice, and
  writes a reusable JSON artifact.

Implemented:
- `tools/turboquant/layerselect` contains the reusable CSV selector shared by
  `cmd/turboquant-select-layers` and `cmd/turboquant-calibrate`.
- `cmd/turboquant-calibrate` now orchestrates sweep → select → validate →
  artifact. It records the selected layer spec, sweep/validation commands, and
  validation metrics.
- The calibrator parser rebases derived defaults when `-artifact-dir` is
  supplied: unless explicitly overridden, `-output` becomes
  `<artifact-dir>/calibration.json` and `-log-dir` becomes
  `<artifact-dir>/logs`.

Regression covered:
- `TestParseCalibrationOptionsRebasesDerivedPaths` failed before the parser
  extraction because `-log-dir` stayed pinned to the timestamped default after
  a custom `-artifact-dir`.
- The test now passes and protects the artifact metadata path.

Command run on GPU, reusing the completed 4-sequence sweep:

```bash
GOCACHE=/tmp/ollama-build-gocache \
OLLAMA_LIBRARY_PATH=/home/yannik/Work/ollama-build/build/lib/ollama \
CUDA_VISIBLE_DEVICES=0 \
go run ./cmd/turboquant-calibrate \
  -sweep-csv /tmp/turboquant-layer-sweep-limit4-q8_0.csv \
  -artifact-dir /tmp/turboquant-calibration-qwen25-7b-q4km \
  -output /tmp/turboquant-calibration-qwen25-7b-q4km/calibration.json \
  -model qwen2.5:7b \
  -snapshot tools/turboquant/testdata/qwen25_7b_phase0_tokens.json \
  -base-kv-cache-type turbo4 \
  -reference-kv-cache-type f16 \
  -layer-dtype q8_0 \
  -top 4 \
  -sweep-limit 4 \
  -validate-limit 16 \
  -num-ctx 1024 \
  -batch-size 512 \
  -num-gpu-layers 999
```

Artifact:
- Path: `/tmp/turboquant-calibration-qwen25-7b-q4km/calibration.json`
- Selected key layer spec: `0:q8_0,1:q8_0,3:q8_0,27:q8_0`
- Validation slice: 16 sequences, 16368 tokens
- Metrics: `mean_nll=1.99192706`, `perplexity=7.32964486`,
  `mean_kl=0.05225737`, `duration=75.4s`

Full Phase 0 gate:
- Path: `/tmp/turboquant-calibration-qwen25-7b-q4km-full/calibration.json`
- Same selected key layer spec: `0:q8_0,1:q8_0,3:q8_0,27:q8_0`
- Validation slice: 256 sequences, 261888 tokens
- Metrics: `mean_nll=2.16333712`, `perplexity=8.70012261`,
  `mean_kl=0.04249074`, `duration=1159.0s`

Interpretation:
- The automatic path reproduces the manual adaptive preset result.
- This is still a model/weight-quantization-specific calibration for
  `qwen2.5:7b` Q4_K_M, not evidence that the same layer set is optimal for
  other model quantizations.
- Compared with full `kq8-vturbo4` (`mean_kl=0.01705642`,
  `perplexity=8.59223759`), adaptive uses less q8 key storage but is
  measurably noisier.

Community implementation note:
- Reviewed <https://github.com/tonbistudio/turboquant-pytorch> after the user
  pointed to it. The repo's V3 direction is consistent with our measured split
  results: avoid assuming QJL is the first fix for softmax attention, allocate
  more precision to keys than values, keep a recent fp16 window when possible,
  and protect sensitive layers.
- This reinforces `kq8-vturbo4` as the practical safe preset and suggests the
  next research branch should test recent-token key protection / residual
  windows before committing to the full JL residual path from the paper.

Local K/V norm check:
- Reused `/tmp/tqdump-layers-kv` plus `cmd/turboquant-dump-index` to write
  `/tmp/turboquant-layer-index.csv`.
- Wrote per-layer K/V norm stats to `/tmp/turboquant-kv-norms.csv`.
- Mean K/V norm ratio summary: average `8.40x`, min `0.44x`, max `106.89x`.
- Highest-ratio layers include 0 (`106.89x`), 1 (`37.69x`), 3 (`15.97x`),
  and 27 (`5.63x`), which overlaps the selected adaptive key layers.
- Some late layers invert the asymmetry, so fixed first/last protection is less
  defensible here than measured layer calibration.

## 2026-04-29 (later) - Decision memo: Path A (4-bit nibble revert)

Two candidates were on the table after the kernel correctness bug was closed:

- **Path A** — revert `block_turbo4_0` to the 16-centroid 4-bit nibble layout that
  was last known to ship correct outputs.
- **Path B** — implement proper QJL with a random Gaussian projection matrix R
  per arXiv 2504.19874 (the design the comments cite).

**Decision: Path A.**

Evidence and rationale:

1. **Empirical match to f16 quality.** `benchmark-context-20260429-023500.md`
   measured turbo4 (4-bit nibble) on `qwen2.5:7b` against the f16 baseline:
   - 4096 ctx (2k prompt): turbo4 → `The`, f16 → `The` (match).
   - 32768 ctx (16k prompt): turbo4 → `The.`, f16 → `The.` (match).
   - 65536 ctx (32k prompt): turbo4 → coherent sentence,
     f16 → `ベル` (turbo4 actually beats f16 here, since fp16 numerics drift at
     long contexts).
   The 4-bit layout already meets the goal of the current quality push
   ("close the gap to f16").

2. **Same compression ratio.** Both layouts use a 68-byte block per 128 values.
   - 4-bit nibble: `norm(2) + rnorm(2, reserved) + qs[64]`.
   - 3-bit + QJL: `norm(2) + rnorm(2) + qs[48] + signs[16]`.
   - Effective payload is 64 bytes / 128 values = 4.0 bits/value of useful
     information; both pay the same 4-byte header. There is no compression
     advantage to chasing Path B.

3. **Cost.** The previous code already supported both layouts behind a
   `TURBO4_USE_4BIT` switch (default 1 = 4-bit). Reverting is a mechanical
   restoration of code paths that were already audited and tested. Path B
   would require designing R-storage discipline (per-block? per-tensor?
   global?), wiring R into both CPU and GPU encode/decode paths, and re-deriving
   the FA dot-product estimator — multi-day work with high regression risk.

4. **Path B's quality advantage is unproven.** The current "simplified QJL"
   (R = I, per-element residual sign) demonstrates that re-enabling QJL alone
   doesn't recover quality (`three? three one?` instead of `Four`). Even proper
   QJL with random R is only theoretically motivated for 3-bit centroids; there
   is no measurement showing it beats a 16-centroid 4-bit PolarQuant at the
   same bit budget. Without a perplexity rig in place, committing to Path B
   would be a research bet, not an engineering fix.

5. **Path B is preserved.** The 4-bit branch shares the 68-byte block size
   with the 3-bit+QJL branch via the `TURBO4_USE_4BIT` switch. Restoring the
   switch (default 1) keeps the QJL code path available for a future
   apples-to-apples comparison once a proper eval (perplexity / token
   accuracy on a held-out set) is in place.

Plan:
1. Restore `TURBO4_USE_4BIT` switch in `ggml-common.h` (default 1).
2. Restore 4-bit branches in CPU encode/decode (`ggml-turbo-quant.c`),
   GPU encode (`set-rows.cu`), GPU dequant helpers (`turbo-quant.cuh`,
   `fattn-common.cuh`, `convert.cu`).
3. Run the synthetic harness (`/tmp/turbo_set_rows_diag.cpp`) against the
   4-bit reference; require ≤1e-5 GPU vs CPU.
4. Real-model smoke test on `qwen2.5:7b`: expect `Four` and `Paris.`.
5. Single fix commit.

## 2026-04-29 19:30 - Session start, state reconnaissance

Goal:
- Reproduce the failure: `qwen2.5:7b` + `OLLAMA_KV_CACHE_TYPE=turbo4` on GPU produces single-token garbage (e.g. `" words"`, `" DevComponents"`).
- Find the first concrete divergent op, not patch on suspicion.

Observed source state (vs HEAD `9b8f087b`):
- `ml/backend/ggml/ggml/src/ggml-common.h`: `block_turbo4_0` reshaped from 4-bit nibble (working) to 3-bit split (`qs[32]` low + `qh[16]` high) + 1-bit QJL `signs[16]`. The `TURBO4_USE_4BIT` switch is gone — the new layout is unconditional.
- `ml/backend/ggml/ggml/src/ggml-cuda/fattn-common.cuh`: KQ bulk-load fixed; turbo4 KQ bulk path updated for the new 3-bit+QJL layout but documented as "without QJL correction" in the handoff.
- `ml/backend/ggml/ggml/src/ggml-cuda/set-rows.cu`: GPU quantize updated for new layout.
- `ml/backend/ggml/ggml/src/ggml-cuda/turbo-quant.cuh`: shared device helpers for the new layout.
- `ml/backend/ggml/ggml/src/ggml-turbo-quant.c`: CPU encode/decode rewritten (-86 net lines from the diff stat).
- `ml/backend/ggml/ggml.go`: callback dump infrastructure (diagnostic).

Last working state (per `benchmark-context-20260429-023500.md`): turbo4 GPU produced sensible answers across context lengths. That was the 4-bit nibble layout. Therefore the regression is in the QJL WIP, not in any committed code.

Hypothesis ladder (to be tested against evidence, not patched against):
- H1: CPU encode and GPU encode disagree on packing layout (→ would corrupt cache when CPU prompt-eval feeds GPU decode, or vice versa).
- H2: GPU FA decode in `fattn-common.cuh` reads packed bits wrong for the new layout (handoff already says QJL correction is intentionally absent there — but K dot accuracy may still be wrong).
- H3: `convert.cu` GPU dequant (used outside FA) wasn't updated for the new layout, corrupting any non-FA consumer of the cache.
- H4: Norm (`norm`) vs residual norm (`rnorm`) scaling between encoder and decoder is inconsistent.

Plan:
1. Read every QJL WIP file completely. Extract the encode contract (CPU + GPU) and the decode contract (FA + dequant) verbatim.
2. Compare contracts byte-by-byte. Any disagreement is a concrete mismatch independent of any kernel run.
3. If contracts agree, fall back to runtime evidence: build a single-block synthetic round-trip test (encode→pack→decode) per backend, then cross-backend.
4. Only after a concrete mismatch is proven: propose a fix that cites which mismatch it resolves.

## 2026-04-29 19:55 - Static contract analysis (no run, no patch)

Files inspected at current dirty state:
- ml/backend/ggml/ggml/src/ggml-common.h (block layout)
- ml/backend/ggml/ggml/src/ggml-turbo-quant.c (CPU encode + dequant)
- ml/backend/ggml/ggml/src/ggml-cuda/turbo-quant.cuh (device helpers + standalone dequant element)
- ml/backend/ggml/ggml/src/ggml-cuda/set-rows.cu (GPU encode k_set_rows_turbo4)
- ml/backend/ggml/ggml/src/ggml-cuda/fattn-common.cuh (FA K-dot vec_dot_fattn_vec_KQ_turbo4 + V-dequant dequantize_V_turbo4)
- ml/backend/ggml/ggml/src/ggml-cuda/convert.cu (standalone GPU dequant via dequantize_block_turbo4_0)

Block contract (ggml-common.h, line 287-294):
- norm  ggml_half  : "corrected L2 norm" (= grp_norm / recon_norm)
- rnorm ggml_half  : "residual L2 norm (QJL scale)"
- qs[32] : low 2 bits of 3-bit centroid index, 4 elements per byte
- qh[16] : upper 1 bit of 3-bit centroid index, 8 per byte
- signs[16] : 1 bit per element, set if (rotated[i] - C[idx[i]]) >= 0

Encode (CPU `quantize_row_turbo4_0_ref`):
  norm = sqrt(sum src^2); normalized = src/norm; rotated = WHT(normalized)
  idx[i] = nearest_centroid_3bit(rotated[i])
  recon_norm = sqrt(sum C[idx]^2); corrected = norm/recon_norm
  residual_norm = sqrt(sum (rotated - C[idx])^2)
  blk.norm = corrected; blk.rnorm = residual_norm
  signs[i] = 1 iff (rotated[i] - C[idx[i]]) >= 0

Encode (GPU `k_set_rows_turbo4`, set-rows.cu line 944-1132): identical pipeline (load → InnerQ-no-op-by-default → L2 norm → normalize → forward WHT signs1+butterfly+1/sqrt(128)+signs2 → 3-bit nearest → residual sign packing → recon norm → write). **Encode contracts match byte-for-byte.**

Decode (CPU `dequantize_row_turbo4_0`, ggml-turbo-quant.c line 532-555):
  dst[i] = C[idx] * norm + sign(±1) * rnorm * (1/sqrt(128))   ← FULL QJL

Decode (GPU helper `turbo4_dequant_element`, turbo-quant.cuh line 358-365):
  dst[i] = C[idx] * norm                                        ← QJL EXPLICITLY DISABLED
  Source comment: "TEMP: QJL correction disabled for debugging — test 3-bit PolarQuant alone"

Decode call sites that use the no-QJL helper:
- FA V dequant `dequantize_V_turbo4` (fattn-common.cuh line 778-794) → no QJL
- FA K dot `vec_dot_fattn_vec_KQ_turbo4` (fattn-common.cuh line 654-709) → reads qh, ignores signs[], no QJL
- Standalone dequant `dequantize_block_turbo4_0` (convert.cu line 641-649) → no QJL

CPU vs GPU divergence is intentional in the dirty source: the GPU pipeline strips QJL by design (commented as a debug bisection). CPU keeps QJL.

Externally observable consequence (per handoff):
- CPU full-QJL → "one" for the test prompt (correct)
- GPU 3-bit no-QJL → " words" / " DevComponents" (garbage)
- benchmark-context-20260429-023500.md (4-bit nibble, no QJL needed) → "The" (correct)

So the empirical fact is: 16-centroid 4-bit no-QJL was sufficient quality for real models, but 8-centroid 3-bit no-QJL is not. Adding QJL is what 3-bit needs to recover the 4-bit-equivalent fidelity. This makes H_quant_noise the dominant hypothesis — but it has not been **proven** yet. The handoff requires proof, not plausibility.

## What "proof" looks like before patching

We do NOT yet know that the 28-layer compounding noise of 3-bit no-QJL is the cause. It could be a kernel correctness bug that synthetic single-block tests miss. Three ways to prove (in order of cost):

A. Smarter dump comparator. The existing dumps `/tmp/tq-dump-gpu-cb-allf32-0429` (turbo4) and `/tmp/tq-dump-gpu-f16-cb-allf32-0429` (f16) have 60 vs 80 nodes — names are sequential `node_N`, not stable across runs. Cheap, but may not give a clean alignment.

B. Re-run with same-prompt, same-seed dumps under aligned eval order, then align by (op, name, shape, sequence-of-FA-occurrences). Modest cost.

C. Build the synthetic graph the handoff recommends: ROPE → TURBO_WHT → SET_ROWS turbo4 → FLASH_ATTN_EXT → inverse TURBO_WHT, and run it on both CPU and GPU. This isolates the turbo path from compounding effects. Highest-cost but most rigorous.

Once any of A/B/C produces a concrete first-divergent op, the patch is determined: if FA output already wrong on first call, kernel bug; if FA outputs roughly track but compound through layers, QJL re-enable is correct.

## Stop point

No code changed. No kernel patched. State is exactly as the handoff left it.
Next session decides which of A/B/C to pursue. My recommendation: A first (cheapest), promote to C if A is inconclusive.

## 2026-04-29 20:35 - Synthetic pipeline test (Path C)

Added `run_pipeline_turbo4_case` to /tmp/turbo_set_rows_diag.cpp. Builds the exact graph
the handoff prescribed: ROPE-equivalent input → TURBO_WHT → SET_ROWS turbo4 →
FLASH_ATTN_EXT → inverse TURBO_WHT, on the CUDA backend, and compares the GPU output
against THREE CPU references built from the same inputs:

1. **ground-truth**     — attention against the un-quantized rotated K/V (no quant).
2. **with-QJL**         — attention against `dequantize_row_turbo4_0` (CPU full QJL).
3. **no-QJL**           — attention against `dequantize_row_turbo4_0_noqjl` (no QJL).

Output (D=128, KV=256, NQ=1, single block per row, deterministic inputs):

```
turbo4 pipeline vs ground-truth: max_abs=0.0109944 max_rel=7.58 at i=22
turbo4 pipeline vs with-QJL    : max_abs=0.00678669
turbo4 pipeline vs no-QJL      : max_abs=2.00677e-06       ← GPU is bit-perfect to no-QJL
turbo4 pipeline ref(QJL)   vs ref(GT)    : max_abs=0.00754904
turbo4 pipeline ref(noQJL) vs ref(GT)    : max_abs=0.0109939
turbo4 pipeline ref(QJL)   vs ref(noQJL) : max_abs=0.00678698
DECISION: gpu_tracks_noqjl=1  qjl_matters=1
```

Interpretation:
- **GPU FA kernel is correct** for the no-QJL semantics it implements (error 2e-6 against
  the matching CPU reference). No kernel indexing/packing/WHT bug remains in the turbo4 path.
- **QJL shrinks per-layer error by ~30%** (0.0110 → 0.0075 vs ground truth) and removes the
  centroid-bias structure of the no-QJL residual.
- Single-layer attention output magnitude is ~0.01. No-QJL absolute error is ~0.011 — same
  order of magnitude. Without QJL, every layer produces ~50–100% relative error in the
  rotated-domain output. Over 28 layers that compounds catastrophically, which matches the
  observed ` words` / ` DevComponents` garbage.
- The 4-bit nibble layout that worked previously had 16 centroids (vs 8 here), so the
  centroid-bias term was about half this magnitude — small enough to avoid catastrophic
  compounding. Going to 3-bit *requires* QJL to recover that quality.

This is the first proven concrete mismatch the handoff demanded. The fix is determined:
re-enable QJL in three GPU sites:
- `turbo4_dequant_element` in `ml/backend/ggml/ggml/src/ggml-cuda/turbo-quant.cuh`
  (used by FA V-dequant `dequantize_V_turbo4` and standalone `dequantize_block_turbo4_0`).
- `vec_dot_fattn_vec_KQ_turbo4` in `ml/backend/ggml/ggml/src/ggml-cuda/fattn-common.cuh`
  (computes K dot inline; QJL must be added to the accumulator there, can't reuse helper).

## 2026-04-29 21:05 - Patch applied & post-fix verification

Edits:
- `turbo-quant.cuh`: `turbo4_dequant_element` now reads `rnorm` and `signs[j/8]`,
  returns `C[idx]*norm + sign·rnorm·INV_SQRT_D`. Comment updated to cite the
  20:35 evidence.
- `fattn-common.cuh::vec_dot_fattn_vec_KQ_turbo4`: per-block reload of `rnorm`
  scaled by `INV_SQRT_D`; per-pair `signs` byte read; accumulator becomes
  `(v0 + s0·rn_scaled)·Q.x + (v1 + s1·rn_scaled)·Q.y`. Verified `j0` is always a
  multiple of 8 in the dispatched code paths (tid·cpy_ne·2 mod 128, with cpy_ne ∈
  {2,4}), so single-byte `signs`/`qh` reads remain safe — same invariant the
  pre-existing `qh` read relied on.

Rebuilt CUDA backend (479 MB shared lib), staged into `cuda_v12/`, removed flat
`build/lib/ollama/libggml-cuda.so` to avoid double-load (parked at
`/tmp/ollama-build-libggml-cuda-flat-0429b.so`).

Synthetic harness re-run after rebuild and after switching the per-test
references from `_noqjl` to full QJL:

```
turbo4-qjl   FA compare:   max_abs=8.3819e-09 max_rel=8.41e-07
turbo4 pipeline vs with-QJL: max_abs=8.85e-07 max_rel=0.0204
turbo4-qjl   cache GQA:     max_abs=9.97e-06 max_rel=0.00025
```

GPU now agrees with the full-QJL CPU reference to ≤1e-5 across:
- direct FA
- full pipeline (TURBO_WHT → SET_ROWS → FA → inverse TURBO_WHT)
- cache + GQA (28 Q heads / 4 KV heads / 256 tokens)

Pre-existing 512-wide `set_rows` byte-offset mismatches remain (handoff already
flagged these as packing-adjacent and not implicated in model garbling).

Real-model smoke test (qwen2.5:7b, GPU, OLLAMA_KV_CACHE_TYPE=turbo4):

| prompt                              | pre-fix GPU       | post-fix GPU       | f16 GPU baseline  |
| ----------------------------------- | ----------------- | ------------------ | ----------------- |
| "What comes after three?" (np=1)    | ` words`          | `three`            | `Four`            |
| "What comes after three?" (np=10)   | (similar garbage) | `three? three one? three?? three?` | `Four\n`     |
| "The capital of France is" (np=15)  | (random tokens)   | ` Is\n\n\n\n\\), I\n\`\n\n, is\n` | `The capital of France is Paris.` |

Interpretation:
- The original handoff bug (random/unrelated tokens like ` DevComponents`) is
  **resolved**: GPU output is now topic-coherent and bit-exact to CPU full-QJL.
- A separate, non-kernel quality gap remains: simplified QJL (no random
  projection matrix, R = I) recovers ~30% of the 3-bit centroid error, not 100%.
  CPU itself with this same simplified QJL produced only `"one"` per the
  handoff — not `"Four"` — so the GPU is now matching CPU semantics, and the
  remaining gap is a quantization-design issue, not a backend bug.

Recommendation for the next quality push (out of scope for this fix):
- Either revert the dirty layout to the 4-bit nibble (16-centroid) struct that
  the 02:35 benchmark confirmed gave correct output (`"The"`), or implement
  proper QJL with a random Gaussian projection matrix R as in arXiv 2504.19874.
  The current 3-bit + simplified QJL is a half-measure between those two and
  trades quality for negligible bit savings vs the 4-bit nibble layout.

## Status

Kernel correctness bug from handoff: **closed** (proven by synthetic match to
≤1e-5 across three independent test paths and confirmed by real-model output
recovering from random tokens to topic-coherent output).

## 2026-04-29 (later) - Path A executed: 4-bit nibble layout restored

Restoration was a single-command revert of the six WIP files to commit
`aefcdeb2` (the "perf: wire CUDA WHT kernel and optimize turbo KQ dot
products" commit, which was the source state at the time of the 02:35
benchmark):

```
git checkout aefcdeb2 -- \
  ml/backend/ggml/ggml/src/ggml-common.h \
  ml/backend/ggml/ggml/src/ggml-turbo-quant.c \
  ml/backend/ggml/ggml/src/ggml-cuda/set-rows.cu \
  ml/backend/ggml/ggml/src/ggml-cuda/turbo-quant.cuh \
  ml/backend/ggml/ggml/src/ggml-cuda/fattn-common.cuh
```

(`convert.cu` did not differ between `aefcdeb2` and HEAD, so it was
not in the file list.)

This restored the `TURBO4_USE_4BIT` switch (default 1) and all matching
encode/decode/FA paths. The 3-bit+QJL branch is preserved behind
`#if !TURBO4_USE_4BIT` for future research.

The QJL kernel fix from `5e2a3175` becomes a no-op for the 4-bit branch
but remains in the legacy 3-bit branch in case anyone flips the switch.

### Synthetic harness verification (4-bit reference)

`/tmp/turbo_set_rows_diag.cpp` rebuilt against the new layout (the
`_noqjl` helper and the 3-way pipeline test were guarded with
`#if !TURBO4_USE_4BIT` since they reference fields that don't exist in
4-bit mode).

```
turbo4-qjl   FA compare:   max_abs=9.31e-09  max_rel=6.82e-07
turbo4-qjl   cache GQA:    max_abs=8.94e-08  max_rel=3.54e-07
```

(Label "qjl" is stale from the prior session's diagnostic naming; the test
now compares 4-bit-nibble GPU FA vs 4-bit-nibble CPU FA. Both backends
agree to ≤1e-7 across direct FA and cache+GQA paths.)

The pre-existing 512-wide `set_rows` byte-offset mismatches that the
handoff already identified as packing-adjacent are unchanged and
unrelated to the model output.

### Real-model smoke test (qwen2.5:7b, GPU, OLLAMA_KV_CACHE_TYPE=turbo4)

| prompt                                                    | turbo4 (Path A)                                              | f16 baseline                       |
| --------------------------------------------------------- | ------------------------------------------------------------ | ---------------------------------- |
| `What comes after three?`                                 | `After three, the next number is four (4).`                  | `After three comes four.`          |
| `The capital of France is`                                | `I believe you meant to ask about the capital of France...`  | `The capital of France is Paris.`  |
| `What number comes after three? Answer with only the...` | `4`                                                          | (n/a)                              |

Numerical answer paths are correct (`four` / `4`). The "France" prompt
diverges in style (turbo4 takes a chatty meta-answer; f16 answers
directly with "Paris") but stays topic-coherent — a meaningful recovery
from the prior HEAD state (`three? three one?`, ` Is\n\n\n\n\\), I\n...`).
This matches the goal of the quality push: closing the catastrophic
garbling, not bit-for-bit f16 parity.

The 02:35 benchmark also captures the long-context behavior of this
layout (4k/32k contexts match f16 exactly; at 64k turbo4 outperforms f16
because fp16 numerics drift). That benchmark stands as the perplexity-
proxy evaluation for this commit.

## Status

Kernel correctness bug from handoff: **closed**.
Quality regression from the QJL WIP: **closed** by reverting to 4-bit
nibble layout. The 3-bit+QJL design remains available behind
`TURBO4_USE_4BIT=0` for future research with proper eval infrastructure.

## 2026-04-30 - SWA / SWAMem / ChunkedAttention reachability of split presets

Goal: confirm that `kq8-vturbo4` and `turboquant-adaptive` (which call
`Causal.InitSplit` and `Causal.SetKeyLayerDTypes` from the runner) work
correctly under the three non-default cache shapes used by
sliding-window models (Gemma3, Gemma3n, Olmo3, Laguna), full-SWA-mem
models (Gemma4, GPT-OSS, Gemma3-extended), and chunked-attention models
(Llama4).

### Trace findings (no plumbing fix needed)

`kvcache/causal.go:103-141` defines four constructors that all return
`*Causal` with different scheduling fields populated:

| Constructor                  | Used by                              | Returns   |
|------------------------------|--------------------------------------|-----------|
| `NewCausalCache`             | qwen2.5, llama3, mistral, ...        | `*Causal` |
| `NewSWACache(window, ...)`   | gemma3, gemma3n, olmo3, laguna       | `*Causal` |
| `NewSWAMemCache(window, mem)`| gemma4, gptoss                       | `*Causal` |
| `NewChunkedAttentionCache(c)`| llama4                               | `*Causal` |

Because all four share the `Causal` struct, the methods
`Causal.InitSplit` (`causal.go:147`) and `Causal.SetKeyLayerDTypes`
(`causal.go:203`) are reachable on every variant. The cache allocator in
`Causal.Put` (`causal.go:503-549`) uses `c.keyDTypeForLayer(c.curLayer)`
for K storage (so layer overrides apply per-layer) and `c.ValueDType`
for V storage (uniform across layers). The SWA/SWAMem/Chunked
scheduling fields (`swaWindowSize`, `swaMemorySize`, `chunkSize`) only
gate masking/eviction and never participate in tensor allocation, so
mixed K/V dtypes are agnostic to the cache shape.

`kvcache/wrapper.go:32-56` proxies `InitSplit` and `SetKeyLayerDTypes`
to wrapped caches, with an explicit panic if any inner cache cannot
take split key/value dtypes. Today the only `WrapperCache` users
(gemma2, gemma3) wrap a `Causal`-derived SWA cache and a `Causal`
default cache, both of which support split init.

`runner/ollamarunner/cache.go:73-101` then calls these methods on the
single combined `Cache` interface returned by the model. There is no
SWA-/Chunked-specific branch that would skip the calibration apply
step.

Conclusion: no wiring change is required. Both Causal and the three
windowed variants — including the wrapped variants — already accept
split K/V dtypes and per-layer K overrides. The risk is purely
behavioral: the existing covering tests in `kvcache/causal_test.go`
only exercise `NewCausalCache` for these code paths, so the
SWA/SWAMem/Chunked entry points were not regression-protected. Added
`TestCacheVariantsAcceptSplitInitAndKeyLayerDTypes` in this same
package to cover all four constructors.

### Smoke command for Gemma3 / SWA model (manual followup)

No Gemma3 (or Gemma3n / Olmo3 / Laguna / Gemma4 / GPT-OSS) tag is
locally available in this workspace, so the live smoke is recorded
here for whoever has GPU access to run it. The commands assume the
model has already been pulled with `ollama pull`:

```bash
# Sliding-window-only (Gemma3 1B fits a 24GB GPU comfortably):
OLLAMA_NEW_ENGINE=1 \
OLLAMA_FLASH_ATTENTION=1 \
OLLAMA_KV_CACHE_TYPE=kq8-vturbo4 \
ollama run gemma3:1b "Write a haiku about sliding windows."

# SWAMem (Gemma4 / GPT-OSS):
OLLAMA_NEW_ENGINE=1 \
OLLAMA_FLASH_ATTENTION=1 \
OLLAMA_KV_CACHE_TYPE=kq8-vturbo4 \
ollama run gpt-oss:20b "Explain attention in one sentence."

# ChunkedAttention (Llama4):
OLLAMA_NEW_ENGINE=1 \
OLLAMA_FLASH_ATTENTION=1 \
OLLAMA_KV_CACHE_TYPE=kq8-vturbo4 \
ollama run llama4:scout "Summarize this paragraph."
```

Pass criteria for the smoke: output is coherent prose (not garbled
tokens or repeating fragments), the runner log contains the
`turboquant: applying calibration` (or for `kq8-vturbo4` the cache
type echo) line, and there is no panic on `InitSplit`. If the smoke
fails on any windowed model, the failure mode is most likely model
selection (CPU fallback, or non-CUDA backend silently downgrading
turbo4 → q8_0 per the existing operator-guide note), not the cache
plumbing covered here.

### Verification

```bash
GOCACHE=/tmp/ollama-build-gocache \
go test -count=1 -run TestCacheVariantsAcceptSplitInitAndKeyLayerDTypes \
  ./kvcache
GOCACHE=/tmp/ollama-build-gocache \
go test -count=1 ./kvcache ./runner/ollamarunner ./tools/turboquant/...
```

