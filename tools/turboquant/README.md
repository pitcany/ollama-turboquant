# TurboQuant Phase 0 Tools

These tools are the checked-in Phase 0 gate from `TURBOQUANT-PAPER-IMPLEMENTATION-PLAN.md`.

## Perplexity

The Phase 0 gate snapshot is checked in at:

```text
tools/turboquant/testdata/qwen25_7b_phase0_tokens.json
```

It contains 256 sequences of 1024 tokens for `qwen2.5:7b`. The source corpus is the WikiText-103 raw test split from the `mattdangerw/wikitext-103-raw` Hugging Face mirror (`https://huggingface.co/datasets/mattdangerw/wikitext-103-raw`) plus a deterministic repository code/doc subset:

- `README.md`
- `docs/api.md`
- `docs/development.md`
- `ml/backend/ggml/ggml/src/ggml-turbo-quant.c`
- `ml/backend/ggml/ggml/src/ggml-cuda/fattn-common.cuh`
- `ml/backend/ggml/ggml/src/ggml-cuda/set-rows.cu`
- `ml/nn/attention.go`
- `model/models/qwen2/model.go`
- `runner/llamarunner/runner.go`
- `llama/llama.go`

Create a replacement token snapshot from a fixed corpus:

```bash
go run ./cmd/turboquant-tokenize \
  -model qwen2.5:7b \
  -corpus /path/to/heldout-corpus.txt \
  -output /tmp/qwen25_7b_phase0_tokens.json \
  -sequence-len 1024 \
  -max-sequences 256
```

`cmd/turboquant-eval` scores a token snapshot with a selected KV cache type. Use the default `-engine go` path for TurboQuant correctness because the Go model attention path applies the required WHT rotation to Q and inverse WHT to the attention output. The `-engine llama` path is useful for f16-only comparison but does not currently wire those TurboQuant graph transforms.

```bash
go run ./cmd/turboquant-eval \
  -model qwen2.5:7b \
  -snapshot tools/turboquant/testdata/qwen25_7b_phase0_tokens.json \
  -kv-cache-type f16 \
  -num-ctx 1024 \
  -batch-size 512 \
  -num-gpu-layers 999 \
  -format json
```

Run the same snapshot with `-kv-cache-type turbo4` and compare `mean_nll` and `perplexity`.
Add `-reference-kv-cache-type f16` to compute output-logit KL against an f16 reference context in the same process.
The checked-in `smoke_tokens.json` only verifies the harness; it is not the Phase 0 quality gate.

Turbo KV cache types require flash attention. Keep `-flash-attention=true` for `turbo4`; CPU-only runs work for smoke testing but the Phase 0 gate should be run on the target CUDA backend.

For isolation diagnostics, the Go engine also accepts `-key-cache-type` and `-value-cache-type` overrides. For example, use `-key-cache-type turbo4 -value-cache-type f16` to measure K-score error without Turbo V reconstruction. Use `-key-cache-layer-types 0:f16,27:q8_0` to override key cache dtype only for selected layers. Per-layer overrides are Go-engine only and accept `f16`, `q8_0`, `q4_0`, `turbo2`, `turbo3`, and `turbo4`.

Use `-kv-cache-preset kq8-vturbo4` in `cmd/turboquant-eval` for the temporary safe fallback measured in this investigation. It expands to `-key-cache-type q8_0 -value-cache-type turbo4`, keeping Turbo4 value compression while avoiding the current Turbo4 key scoring failure. Full Phase 0 validation on 2026-04-30 completed all 256 sequences: `mean_nll=2.15085919`, `perplexity=8.59223759`, `mean_kl=0.01705642`, `duration=1123.3s`.

The same fallback is available in the production Ollama engine with:

```bash
OLLAMA_NEW_ENGINE=1 \
OLLAMA_FLASH_ATTENTION=1 \
OLLAMA_KV_CACHE_TYPE=kq8-vturbo4 \
ollama run qwen2.5:7b
```

`kq8-vturbo4` is intentionally Go/Ollama-engine-only. The llama.cpp compatibility runner only accepts single K/V cache types, so the server warns and ignores this split preset there. Memory estimation accounts for q8_0 K bytes and Turbo4 V bytes separately.

The measured Qwen2.5 7B Q4_K_M adaptive policy is available as a named preset:

```bash
go run ./cmd/turboquant-eval \
  -engine go \
  -model qwen2.5:7b \
  -snapshot tools/turboquant/testdata/qwen25_7b_phase0_tokens.json \
  -kv-cache-type turbo4 \
  -key-cache-layer-preset qwen2.5-7b-q4km-adaptive \
  -reference-kv-cache-type f16 \
  -num-ctx 1024 \
  -batch-size 512 \
  -num-gpu-layers 999 \
  -flash-attention=true \
  -format json
```

This preset expands to `-key-cache-layer-types 0:q8_0,1:q8_0,3:q8_0,27:q8_0`. It is intentionally named after the model and weight quantization used in the sweep; re-run the layer sweep before applying the same layer set to other weight formats.

Preset smoke result on 2026-04-30 with `-limit 1`: `mean_nll=1.78801009`, `perplexity=5.97754583`, `mean_kl=0.03065761`. The JSON output includes both `key_cache_layer_preset` and the resolved `key_cache_layer_types` for reproducibility.

## Operator Guide

### Quick start (the thing most users want)

Set `OLLAMA_KV_CACHE_TYPE=turboquant-adaptive` and run:

```bash
OLLAMA_NEW_ENGINE=1 \
OLLAMA_FLASH_ATTENTION=1 \
OLLAMA_KV_CACHE_TYPE=turboquant-adaptive \
ollama run qwen2.5:7b
```

The runtime resolves the model's architecture, file type, and head dimension
against the bundled calibration manifest at
`tools/turboquant/calibration/manifest_data/manifest.json`. If a matching
entry exists, the runtime applies its base K/V cache type and per-layer key
overrides automatically. Otherwise it falls back to `kq8-vturbo4`. The
resolution is logged at info level so it is easy to see which path was used.

`/api/show` includes a static `kv_cache` field summarizing what would happen
for each model (manifest source, base type, projected per-pair bytes, and
projected savings vs an f16 baseline). Operators can sanity-check coverage
without ever running the model. The same preview is shown in the
`ollama show <model>` CLI output under the `KV Cache` section.

When TurboQuant is configured but the active backend is not CUDA (Metal,
ROCm, Vulkan, or CPU), the runner downgrades all Turbo* dtypes to `q8_0`
and logs a warning. There is no silent garbage path.
Embedding models do not use flash attention; when TurboQuant is configured,
they fall back to the default f16 KV cache.

#### Supported cache-type matrix

`kq8-vturbo4` and `turboquant-adaptive` both work across every cache
shape that the Ollama Go engine ships today, because all four
constructors in `kvcache/causal.go` return the same `*Causal` struct
and `Causal.InitSplit` / `Causal.SetKeyLayerDTypes` are invoked
unconditionally by the runner. `WrapperCache` proxies both methods to
its inner caches and panics if any inner cache cannot accept split
K/V dtypes.

| Cache shape                  | Constructor                  | Example architectures              | `kq8-vturbo4` | `turboquant-adaptive` |
|------------------------------|------------------------------|------------------------------------|:-------------:|:---------------------:|
| Full causal                  | `NewCausalCache`             | qwen2.5, llama3, mistral, ...      | ✅ supported  | ✅ supported          |
| Sliding window only          | `NewSWACache`                | gemma3n, olmo3, laguna             | ✅ supported  | ✅ supported          |
| Sliding window + memory      | `NewSWAMemCache`             | gemma4, gpt-oss                    | ✅ supported  | ✅ supported          |
| Chunked attention            | `NewChunkedAttentionCache`   | llama4                             | ✅ supported  | ✅ supported          |
| Wrapper (SWA + causal)       | `NewWrapperCache(SWA, Causal)`| gemma2, gemma3                     | ✅ supported  | ✅ supported          |

Test coverage: `kvcache.TestCacheVariantsAcceptSplitInitAndKeyLayerDTypes`
exercises `InitSplit(K=q8_0, V=turbo4)` and `SetKeyLayerDTypes` on each
of the four `Causal` constructors and asserts that storage and
`Get()`-view tensors carry the expected dtypes.

Live smoke commands for the windowed shapes (run on a host with the
relevant model pulled and a CUDA backend available):

```bash
# Sliding-window-only:
OLLAMA_NEW_ENGINE=1 OLLAMA_FLASH_ATTENTION=1 \
OLLAMA_KV_CACHE_TYPE=kq8-vturbo4 \
ollama run gemma3:1b "Write a haiku about sliding windows."

# Sliding-window + memory:
OLLAMA_NEW_ENGINE=1 OLLAMA_FLASH_ATTENTION=1 \
OLLAMA_KV_CACHE_TYPE=kq8-vturbo4 \
ollama run gpt-oss:20b "Explain attention in one sentence."

# Chunked attention:
OLLAMA_NEW_ENGINE=1 OLLAMA_FLASH_ATTENTION=1 \
OLLAMA_KV_CACHE_TYPE=kq8-vturbo4 \
ollama run llama4:scout "Summarize this paragraph."
```

### Adding a calibration to the bundled manifest

`cmd/turboquant-calibrate` produces a JSON artifact for one model+quant pair.
To bundle it for end users:

1. Run `cmd/turboquant-calibrate` per the recipe below for the new model.
2. Copy the artifact (only the `version`, `model`, `base_kv_cache_type`,
   `layer_dtype`, `key_cache_layer_types`, and an optional `notes` field
   are read by the loader) to
   `tools/turboquant/calibration/manifest_data/<arch>-<size>-<ftype>-adaptive.json`.
3. Add a `tools/turboquant/calibration/manifest_data/manifest.json` entry
   with `architecture`, `file_type`, `head_dim`, and the artifact path.
4. Run `go test ./tools/turboquant/calibration/...` to verify it loads.

`manifest.go` resolves on `(architecture, file_type, head_dim)`; the lookup
is case-insensitive. Leave `head_dim` zero to match any head dimension.

### Benchmarking prefill and decode (`cmd/turboquant-bench`)

The harness times prefill and single-token decode for a model under a
chosen KV cache configuration. Run it on the target GPU and collect numbers
alongside the Phase 0 quality results.

```bash
GOCACHE=/tmp/ollama-build-gocache \
OLLAMA_LIBRARY_PATH=/home/yannik/Work/ollama-build/build/lib/ollama \
go run ./cmd/turboquant-bench \
  -model qwen2.5:7b \
  -kv-cache-type turboquant-adaptive \
  -prompt-tokens 1024 \
  -decode-tokens 128 \
  -warmup 1 \
  -repeats 3 \
  -num-ctx 2048 \
  -batch-size 512 \
  -num-gpu-layers 999 \
  -flash-attention=true \
  -format json
```

Flag notes:

- `-kv-cache-type` accepts the same values as `OLLAMA_KV_CACHE_TYPE` plus
  the `turboquant-adaptive` sentinel.
- `-calibration <path>` overrides the cache type with the artifact's base
  and applies per-layer key overrides.
- `-decode-tokens 0` skips decode timing; useful when only prefill matters.
- `-format text` prints a single line summary; `-format json` (default)
  emits a machine-readable record.

The output includes `prefill_tokens_per_sec`, `decode_tokens_per_sec`, and
backend per-device used bytes (cache + weights as the backend reports them).
Pair this with a quality run from `cmd/turboquant-eval` to validate that a
new preset is both faster than f16 *and* still inside the Phase 0 KL budget.

The bench command is intentionally minimal: it does not aggregate runs
across machines, it does not poll `nvidia-smi`, and it does not produce
HTML reports. Drive it from a shell loop and pipe the JSON into your
preferred analysis stack.

### CI Quality Gate

The opt-in Phase 0 CI scaffold runs the four-sequence Qwen2.5 7B snapshot on
the CUDA self-hosted runner:

```bash
make turboquant-phase0
go run ./cmd/turboquant-ci-gate \
  -baseline /tmp/turboquant-phase0-f16.json \
  -candidate /tmp/turboquant-phase0-kq8-vturbo4.json \
  -max-mean-kl 0.05 \
  -max-perplexity-drift 0.05
```

`make turboquant-phase0` writes `/tmp/turboquant-phase0-f16.json` and
`/tmp/turboquant-phase0-kq8-vturbo4.json` using `cmd/turboquant-eval` with
`-engine go`, `-limit 4`, and `-reference-kv-cache-type f16`. The gate fails
if the candidate `mean_kl` exceeds the documented budget (see below) or if
relative perplexity drift
`(candidate_perplexity - f16_perplexity) / f16_perplexity` exceeds the
documented budget. Missing `mean_kl` is also a failure because it usually
means the eval was not run with an f16 reference context.

#### Quality budget

These thresholds are pinned both in `cmd/turboquant-ci-gate` defaults
(`defaultMaxMeanKL`, `defaultMaxPerplexityDriftRel`) and in
`.github/workflows/turboquant-phase0.yml`. A test in
`cmd/turboquant-ci-gate/main_test.go` (`TestDefaultThresholdsArePinned`)
fails if either default drifts from the documented value, so a code-only
change cannot silently widen the budget.

| preset | metric | measured (256-seq) | budget | headroom |
|---|---|---:|---:|---:|
| `kq8-vturbo4` | `mean_kl` vs f16 | 0.01706 | **0.05** | 2.93x |
| `kq8-vturbo4` | relative perplexity drift | 0.0080 (0.80%) | **0.05** | 6.25x |
| `qwen2.5-7b-q4km-adaptive` | `mean_kl` vs f16 | 0.04249 | **0.13** | 3.06x |
| `qwen2.5-7b-q4km-adaptive` | relative perplexity drift | 0.0207 (2.07%) | **0.05** | 2.42x |

Rationale:

1. **`kq8-vturbo4` mean_kl = 0.05.** The 256-sequence Phase 0 measurement
   on Qwen2.5 7B Q4_K_M is 0.01706. A 3x headroom budget would be 0.0512;
   we round down to 0.05 so a code regression that doubles KL still passes
   while a regression that triples it fails. The kq8-vturbo4 preset is the
   conservative model-agnostic safe split, so we want this gate to be the
   tightest in the budget.

2. **Adaptive preset mean_kl = 0.13.** The bundled
   `qwen2.5-7b-q4km-adaptive` preset trades higher KL for materially less
   KV-cache memory (71.8% savings vs 61.7%). Its 256-sequence measurement
   is 0.04249. 3x headroom is 0.1275; we round to 0.13 so the adaptive
   preset has the same ~3x error margin as kq8-vturbo4. Adaptive is not
   yet run on every CI invocation, but pinning the budget here lets us
   add a `-candidate /tmp/turboquant-phase0-adaptive.json` step under
   the same gate without re-debating the threshold.

3. **Relative perplexity drift = 0.05.** Both presets sit well under 5%
   drift on the 256-sequence corpus (0.80% and 2.07% respectively). 5% is
   a well-known rule-of-thumb for "quantization is still usable" on
   English perplexity benchmarks, and it gives kq8-vturbo4 ~6x headroom
   and adaptive ~2.4x headroom. Tighter would risk false positives from
   stochastic batch ordering on a 4-sequence CI slice; looser would let
   silently broken Turbo paths pass.

4. **Retry policy.** The CUDA self-hosted runner can flake (driver
   stalls, OOM from a noisy neighbor). The workflow allows **one**
   automatic re-run via GitHub Actions' "Re-run failed jobs". Operators
   should treat the second failure as a real regression and not re-run
   again. If the first run fails and the re-run passes, the operator
   files an issue tagged `turboquant-flake` with the `mean_kl` and
   perplexity from both runs so we can decide whether to widen the
   budget or harden the runner. We deliberately do not configure
   `continue-on-error` or in-workflow retry loops, because both would
   hide real quality drift.

5. **Manifest artifact updates.** Any change under
   `tools/turboquant/calibration/manifest_data/` (the bundled adaptive
   presets) must trigger a fresh full-gate run before the PR can merge:

   - The PR author runs `make turboquant-phase0` against the new
     artifact and pastes the resulting `mean_kl`, `perplexity`, and
     `duration` for both f16 baseline and the new artifact in the PR
     description.
   - For larger swings, the author also posts the 256-sequence numbers
     so the budget table above can be revisited.
   - A CODEOWNER for `tools/turboquant/calibration/manifest_data/` must
     approve the PR explicitly. Self-merge is not allowed for manifest
     changes even if other approvals are present.
   - If the author needs to widen the budget to land the new artifact,
     they update both this section and the defaults in
     `cmd/turboquant-ci-gate/main.go` in the same commit, and the
     `TestDefaultThresholdsArePinned` test must be updated to match.
     Treat threshold bumps as quality policy changes and link the
     supporting measurements from `TURBOQUANT-DEBUG-LOG.md`.

Workflow status: the gate has been wired up but has not yet executed on
a real CUDA runner. The next merge to `main` that touches
`tools/turboquant/**` or `ml/backend/ggml/**` will be the first live
exercise; treat that run as the calibration of the gate itself.

### Long-context validation (8k, 32k, 128k)

The 256-sequence Phase 0 gate runs at `num_ctx=1024`. To check that the
two promoted runtime presets do not silently degrade at extreme context,
the same `qwen2.5:7b` Q4_K_M model was scored against a held-out
WikiText-103 + repo-doc corpus at `num_ctx ∈ {8192, 32768, 131072}` on
2026-04-30. Snapshots are checked in at
`tools/turboquant/testdata/qwen25_7b_phase0_tokens_<len>.json` (16
sequences each).

Sequence-count rationale: 32k uses 4 sequences and 128k uses 1 sequence.
Each (length, cache) cell scores ~131k tokens — the same per-cell token
budget as the 256x1024 baseline — so statistical power on `mean_kl` is
comparable. The 16-sequence specification is held only at 8k; running 16
sequences for kq8/adaptive at 32k or 128k would have needed roughly 30 h
and 120 h respectively on this hardware, and 16x131072 KV at f16 plus an
f16 reference KV would also exceed the 24 GB on the local GPU. See
`TURBOQUANT-DEBUG-LOG.md` for the rationale.

| ctx | preset | seqs | tokens | mean_nll | perplexity | mean_kl | duration |
|---:|---|---:|---:|---:|---:|---:|---:|
| 8192 | f16 | 16 | 131056 | 1.8579 | 6.4104 | — | 147.0s |
| 8192 | `kq8-vturbo4` | 16 | 131056 | 1.8631 | 6.4438 | 0.01094 | 629.8s |
| 8192 | `qwen2.5-7b-q4km-adaptive` | 16 | 131056 | 1.8780 | 6.5405 | 0.03229 | 764.2s |
| 32768 | f16 | 16 | 524272 | 1.8522 | 6.3735 | — | 632.5s |
| 32768 | `kq8-vturbo4` | 4 | 131068 | 1.8143 | 6.1365 | 0.01094 | 870.5s |
| 32768 | `qwen2.5-7b-q4km-adaptive` | 4 | 131068 | 1.8289 | 6.2271 | 0.03238 | 1486.8s |
| 131072 | f16 | 1 | 131071 | 1.8201 | 6.1726 | — | 216.1s |
| 131072 | `kq8-vturbo4` | 1 | 131071 | 1.8220 | 6.1844 | 0.01072 | 1807.3s |
| 131072 | `qwen2.5-7b-q4km-adaptive` | 1 | 131071 | 1.8449 | 6.3274 | 0.03838 | 6488.1s |

Regression check vs the 1k baseline (gate: `mean_kl` at 32k or 128k must
not exceed 2x the 1k value):

| ctx | preset | 1k_kl | this_kl | ratio | verdict |
|---:|---|---:|---:|---:|---|
| 32768 | `kq8-vturbo4` | 0.01706 | 0.01094 | 0.64x | ok |
| 32768 | `qwen2.5-7b-q4km-adaptive` | 0.04249 | 0.03238 | 0.76x | ok |
| 131072 | `kq8-vturbo4` | 0.01706 | 0.01072 | 0.63x | ok |
| 131072 | `qwen2.5-7b-q4km-adaptive` | 0.04249 | 0.03838 | 0.90x | ok |

Tested up to 131072 tokens. Both presets keep `mean_kl` to f16 below
their 1k values across the full range, so the 2x regression threshold
is not approached on Qwen2.5 7B Q4_K_M. The `kq8-vturbo4` ratio is
roughly flat in context length (0.64x → 0.63x); the adaptive preset
ratio drifts modestly upward with length (0.76x at 8k/32k → 0.90x at
128k), but stays well under 2x. **Do NOT widen the CI gate from this
evidence.** The gate is pinned at 0.05 (kq8) and 0.13 (adaptive) in
`cmd/turboquant-ci-gate/main.go` and intentionally has 3x headroom; the
long-context numbers are confirmation that the headroom is real, not
justification to tighten or loosen the gate.

Performance note: the adaptive preset's 128k 1-sequence run took 6488s
versus 1807s for `kq8-vturbo4` on the same input — a 3.6x slowdown
specific to long context. The two cache shapes are similar in size
(adaptive's per-pair byte cost is actually slightly lower), so the
slowdown is not memory bandwidth. Mixed per-layer K dtypes (4 layers
q8_0 + 24 layers turbo4) appear to defeat some kernel fusion at long
context. Treat `kq8-vturbo4` as the latency-preferred preset for users
running 32k+ contexts; revisit the adaptive code path if context-aware
calibration is later needed.

Limits surfaced by this run:

- 128k coverage is one held-out sequence (131k tokens scored). That is
  the same token budget as 1k but a smaller sample over the joint
  distribution of long-range dependencies. Treat the 128k row as a
  directional signal, not a high-precision measurement.
- 32k+128k were not run on any architecture other than Qwen2.5 7B
  Q4_K_M. The bundled adaptive preset is model-specific by design; this
  evidence does not generalize to other architectures or quantizations.
  Re-run the layer sweep before applying the same layer set elsewhere.
- The runtime was not modified to special-case long contexts in this
  session. If a future model surfaces a ratio above 2x at long context,
  the path forward is context-conditional adaptive calibration (re-run
  the layer sweep at the failing context length and emit a separate
  bundled artifact), not a runtime change.

#### Reproducing the long-context grid

The corpus is deterministic. Build it (~16 MB):

```bash
unzip -p /tmp/wikitext-103-raw-v1.zip wikitext-103-raw/wiki.test.raw  > /tmp/long-context-corpus.txt
unzip -p /tmp/wikitext-103-raw-v1.zip wikitext-103-raw/wiki.valid.raw >> /tmp/long-context-corpus.txt
unzip -p /tmp/wikitext-103-raw-v1.zip wikitext-103-raw/wiki.train.raw \
  | head -c 13631488 >> /tmp/long-context-corpus.txt
for f in README.md docs/api.md docs/development.md \
  ml/backend/ggml/ggml/src/ggml-turbo-quant.c \
  ml/backend/ggml/ggml/src/ggml-cuda/fattn-common.cuh \
  ml/backend/ggml/ggml/src/ggml-cuda/set-rows.cu \
  ml/nn/attention.go model/models/qwen2/model.go \
  runner/llamarunner/runner.go llama/llama.go; do
  cat "$f" >> /tmp/long-context-corpus.txt
  printf "\n\n" >> /tmp/long-context-corpus.txt
done
```

Regenerate the snapshots:

```bash
for L in 8192 32768 131072; do
  go run ./cmd/turboquant-tokenize \
    -model qwen2.5:7b \
    -corpus /tmp/long-context-corpus.txt \
    -source-label "wikitext-103-raw test+valid+train[:13M] + repo doc subset" \
    -output tools/turboquant/testdata/qwen25_7b_phase0_tokens_${L}.json \
    -sequence-len $L -max-sequences 16
done
```

Run the eval grid (set `-limit` to control wall-time vs sample-count
trade-off; sequence counts of 4 at 32k and 1 at 128k each score 131k
tokens):

```bash
go run ./cmd/turboquant-eval \
  -model qwen2.5:7b \
  -snapshot tools/turboquant/testdata/qwen25_7b_phase0_tokens_32768.json \
  -engine go -num-ctx 32768 -batch-size 512 -num-gpu-layers 999 \
  -flash-attention=true -reference-kv-cache-type f16 \
  -kv-cache-preset kq8-vturbo4 -limit 4 -format json
```

## Original Operator Guide

Two TurboQuant KV-cache modes are now first-class runtime configurations on the
Ollama engine. Both require flash attention; both fall back gracefully on the
legacy llama.cpp compatibility runner with a warning.

### Preset 1: `kq8-vturbo4` (model-agnostic safe split)

```bash
OLLAMA_NEW_ENGINE=1 \
OLLAMA_FLASH_ATTENTION=1 \
OLLAMA_KV_CACHE_TYPE=kq8-vturbo4 \
ollama run qwen2.5:7b
```

The runtime constructs a split K/V cache with `K=q8_0` and `V=turbo4`. The
KV-cache memory estimate in `llm/server.go` reflects the actual mixed dtypes:
`kvCacheBytesPerElementKV("kq8-vturbo4")` returns `(1.0, 0.531)` bytes per
element — `1.531` bytes/element total per K/V pair.

Expected memory vs `f16` (`4` bytes per K/V pair):

| dtype | bytes / K-elem | bytes / V-elem | total bytes / pair | savings vs f16 |
|---|---:|---:|---:|---:|
| `f16` | 2.000 | 2.000 | 4.000 | — |
| `kq8-vturbo4` | 1.000 | 0.531 | 1.531 | 61.7% |

Quality: full 256-sequence Phase 0 gate on `qwen2.5:7b` Q4_K_M reports
`mean_nll=2.15085919`, `perplexity=8.59223759`, `mean_kl=0.01705642` against
the f16 reference — within `0.02` perplexity of the f16 result of `8.515`.

### Preset 2: model-specific adaptive via `OLLAMA_TURBOQUANT_CALIBRATION`

Point the runtime at a calibration JSON produced by `cmd/turboquant-calibrate`:

```bash
OLLAMA_NEW_ENGINE=1 \
OLLAMA_FLASH_ATTENTION=1 \
OLLAMA_TURBOQUANT_CALIBRATION=/tmp/turboquant-calibration-qwen25-7b-q4km-full/calibration.json \
ollama run qwen2.5:7b
```

When the env var is set, the server:

1. Loads the artifact via `tools/turboquant/calibration.Load` (rejects
   missing/invalid `version`, `base_kv_cache_type`, or `key_cache_layer_types`).
2. Sets the effective KV cache type to the artifact's `base_kv_cache_type`
   (e.g. `turbo4`), overriding `OLLAMA_KV_CACHE_TYPE`.
3. Validates that every dtype referenced (base + per-layer) is supported by
   the model's head dimensions.
4. Forwards the per-layer key dtype spec through `LoadRequest.KeyCacheLayerTypes`
   to the runner. The runner calls `Causal.SetKeyLayerDTypes` after the cache
   is initialized, so layers listed in the spec store K at the override dtype
   while everything else stays at the base dtype.
5. Reports per-layer KV-cache memory through `GraphSizeWithKeyOverrides` so
   the `mem.Log` line in the server log matches the actual cache footprint.

For the checked-in `qwen2.5-7b-q4km-adaptive` artifact at
`/tmp/turboquant-calibration-qwen25-7b-q4km-full/calibration.json` the spec
is `0:q8_0,1:q8_0,3:q8_0,27:q8_0` over a `turbo4` base. Per-pair byte cost
on a 28-layer Qwen2.5 7B:

| layer set | K dtype | bytes / K-elem |
|---|---|---:|
| 0,1,3,27 (4 layers) | `q8_0` | 1.000 |
| remaining 24 layers | `turbo4` | 0.531 |
| weighted average | — | ≈ 0.598 |

Expected memory vs `f16`:

| dtype | bytes / K-elem (avg) | bytes / V-elem | total bytes / pair | savings vs f16 |
|---|---:|---:|---:|---:|
| `f16` | 2.000 | 2.000 | 4.000 | — |
| adaptive (qwen2.5-7b-q4km) | 0.598 | 0.531 | 1.129 | 71.8% |
| all-`turbo4` (unsafe) | 0.531 | 0.531 | 1.062 | 73.4% |

Quality: full 256-sequence Phase 0 gate on `qwen2.5:7b` Q4_K_M reports
`mean_nll=2.16333712`, `perplexity=8.70012261`, `mean_kl=0.04249074`. This is
slightly noisier than `kq8-vturbo4` but uses ≈26% less KV cache memory.

The legacy llama.cpp runner does not honor calibration artifacts. The server
logs `OLLAMA_TURBOQUANT_CALIBRATION requires the Ollama engine; ignored` and
proceeds with the user's `OLLAMA_KV_CACHE_TYPE`.

### Regenerating a model-specific adaptive calibration

`cmd/turboquant-calibrate` runs the full sweep + selection + validation pipeline
and writes a JSON artifact with the layer spec, validation metrics, and the
exact commands that produced them. Re-run when:

- changing model architecture or weight quantization (e.g. Q4_K_M → Q5_K_M),
- changing `-num-ctx` or `-batch-size` materially,
- updating the snapshot or any of the kernels that participate in attention.

```bash
GOCACHE=/tmp/ollama-build-gocache \
OLLAMA_LIBRARY_PATH=/home/yannik/Work/ollama-build/build/lib/ollama \
go run ./cmd/turboquant-calibrate \
  -artifact-dir /tmp/turboquant-calibration-mymodel \
  -model mymodel:size \
  -snapshot tools/turboquant/testdata/mymodel_phase0_tokens.json \
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

Validate the output artifact at `<artifact-dir>/calibration.json` before
pointing the production runtime at it. The smallest valid artifact has
`version: 1`, a non-empty `base_kv_cache_type`, and a non-empty
`key_cache_layer_types` spec — the loader rejects anything else.

### Phase 0 Results

Measured on 2026-04-29 with `qwen2.5:7b`, CUDA on an RTX 4090, `-num-ctx 1024`, `-batch-size 512`, and full GPU offload.

| engine | cache | reference | sequences | tokens | mean NLL | perplexity | mean KL | duration |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| llama | f16 | - | 256 | 261888 | 2.14183994 | 8.51509047 | - | 309.0s |
| go | f16 | - | 1 | 1023 | 1.76991064 | 5.87032875 | - | 1.9s |
| go | q8_0 | f16 | 1 | 1023 | 1.77401016 | 5.89444369 | 0.00581534 | 5.6s |
| go | turbo4 | f16 | 1 | 1023 | 6.10887882 | 449.83408783 | 4.46774773 | 5.7s |
| go | turbo4 K, f16 V | f16 | 1 | 1023 | 5.99082551 | 399.74446754 | 4.34615674 | 5.7s |
| go | f16 K, turbo4 V | f16 | 1 | 1023 | 1.78206660 | 5.94212374 | 0.01206953 | 5.7s |
| go | kq8-vturbo4 | f16 | 256 | 261888 | 2.15085919 | 8.59223759 | 0.01705642 | 1123.3s |
| go | turbo4 + adaptive K q8_0 layers 0,1,3,27 | f16 | 256 | 261888 | 2.16333712 | 8.70012261 | 0.04249074 | 1159.0s |

The `go` f16 smoke exactly matches the `llama` f16 smoke for sequence 0, and `q8_0` stays close to f16 through the same corrected path. `turbo4` fails the Phase 0 gate on the first real held-out sequence, so Phase 1 should not start until the current TurboQuant path is fixed. A full corrected-engine turbo4 run was intentionally skipped after this fail-fast result. The `kq8-vturbo4` safe fallback survives the full 256-sequence gate with low KL to f16 reference, so it is the current practical runtime mode while Turbo4 K scoring is investigated.

The split K/V runs isolate the failure to Turbo K scoring: f16 K with Turbo V is close to f16, while Turbo K with f16 V is nearly as bad as full Turbo.

Layer-selective K precision runs measured on 2026-04-30 used the same one-sequence Go-engine setup with `-reference-kv-cache-type f16`:

| candidate | mean NLL | perplexity | mean KL |
|---|---:|---:|---:|
| all Turbo4 K/V | 6.10887882 | 449.83408783 | 4.46774773 |
| layer 0 K=f16, otherwise Turbo4 K/V | 1.86998669 | 6.48821003 | 0.12911092 |
| layer 27 K=f16, otherwise Turbo4 K/V | 6.05931296 | 428.08122659 | 4.41347988 |
| layers 0,27 K=f16, otherwise Turbo4 K/V | 1.82140214 | 6.18051836 | 0.08046964 |
| layer 0 K=q8_0, otherwise Turbo4 K/V | 1.85752395 | 6.40785095 | 0.12340394 |
| layers 0,27 K=q8_0, otherwise Turbo4 K/V | 1.80890804 | 6.10377869 | 0.05754024 |
| all K=q8_0, V=Turbo4 | 1.77819596 | 5.91916839 | 0.01312757 |
| all K=f16, V=Turbo4 | 1.78206660 | 5.94212374 | 0.01206953 |

Layer 0 carries most of the quality failure on this prompt. Raising only layers 0 and 27 to q8_0 recovers most of the gap from all-Turbo4 toward the f16-K/Turbo4-V upper bound, while all-q8_0 K is nearly indistinguishable from f16 K at this scale.

The same adaptive policy was re-run on the first 16 Phase 0 sequences on 2026-04-30:

| candidate | sequences | tokens | mean NLL | perplexity | mean KL | duration |
|---|---:|---:|---:|---:|---:|---:|
| all Turbo4 K/V | 16 | 16368 | 5.76649882 | 319.41743575 | 3.99087118 | 71.5s |
| layers 0,27 K=q8_0, otherwise Turbo4 K/V | 16 | 16368 | 2.02367253 | 7.56606057 | 0.08171422 | 71.9s |
| layers 0,1,3,27 K=q8_0, otherwise Turbo4 K/V | 16 | 16368 | 1.99192706 | 7.32964486 | 0.05225737 | 76.4s |
| all K=q8_0, V=Turbo4 | 16 | 16368 | 1.96968769 | 7.16843736 | 0.02197516 | 70.4s |

The two-layer q8_0 key policy generalizes beyond the first sequence and removes almost all of the catastrophic Turbo4 K error. Adding layers 1 and 3 cuts the remaining KL gap by about a third, but all-q8_0 K is still materially cleaner. The four-layer adaptive policy also completed the full 256-sequence Phase 0 gate with `mean_nll=2.16333712`, `perplexity=8.70012261`, and `mean_kl=0.04249074`; this is close in perplexity but still higher-KL than the all-q8_0 K safe fallback.

`tools/turboquant/layer_sweep.sh` runs reproducible single-layer sweeps and writes CSV output after every layer:

```bash
LIMIT=4 \
OUTPUT=/tmp/turboquant-layer-sweep-limit4-q8_0.csv \
LOG_DIR=/tmp/turboquant-layer-sweep-logs-limit4-q8_0 \
tools/turboquant/layer_sweep.sh
```

The 2026-04-30 run completed all 28 layers with no failures. Top single-layer q8_0 key overrides on the first 4 Phase 0 sequences were:

| layer | mean NLL | perplexity | mean KL |
|---:|---:|---:|---:|
| all Turbo4 K/V baseline | 5.94366594 | 381.33030463 | 4.16417714 |
| 0 | 2.07640661 | 7.97575733 | 0.15234685 |
| 1 | 5.74972980 | 314.10577687 | 3.95995938 |
| 27 | 5.80746057 | 332.77299651 | 4.01762444 |
| 3 | 5.86085046 | 351.02254797 | 4.07728846 |
| 15 | 5.90891402 | 368.30596396 | 4.12998841 |

Combined q8_0 key candidates on the same 4-sequence slice:

| layers | mean NLL | perplexity | mean KL |
|---|---:|---:|---:|
| 0,27 | 2.01736285 | 7.51847146 | 0.08794280 |
| 0,1 | 2.05679336 | 7.82085087 | 0.13404165 |
| 0,1,27 | 1.99842163 | 7.37740262 | 0.06613430 |
| 0,1,3,27 | 1.99482829 | 7.35094072 | 0.05676946 |
| 0,1,3,12,15,27 | 1.98915988 | 7.30939042 | 0.05356584 |
| all K=q8_0, V=Turbo4 | 1.96418408 | 7.12909343 | 0.02479755 |

Layer 0 is the only single-layer override that escapes the catastrophic regime. Layer 27 has a weak standalone effect but helps with layer 0, and layers 1 and 3 add smaller interaction gains. The six-layer set barely improves over `0,1,3,27`, so the current practical choices are either the `qwen2.5-7b-q4km-adaptive` preset or all-q8_0 K. The latter remains the best quality fallback while Turbo4 K scoring is under investigation.

To generate a model-specific adaptive layer spec from a sweep CSV:

```bash
go run ./cmd/turboquant-select-layers \
  -input /tmp/turboquant-layer-sweep-limit4-q8_0.csv \
  -top 4 \
  -dtype q8_0
```

For the 2026-04-30 sweep, this emits:

```text
0:q8_0,1:q8_0,3:q8_0,27:q8_0
```

This selector is deliberately simple: it ranks successful single-layer sweep rows by output-logit KL, chooses the top `N`, and emits a deterministic layer spec. It is a calibration scaffold, not a replacement for final validation; run the resulting spec on a larger Phase 0 slice before treating it as a preset.

### Automatic Calibration

`cmd/turboquant-calibrate` runs the calibration scaffold end to end: generate or reuse a single-layer sweep CSV, select the lowest-KL layers, validate the selected layer spec on a larger slice, and write a JSON artifact with the selected spec, commands, and validation metrics. When `-sweep-csv` is omitted, it invokes `tools/turboquant/layer_sweep.sh` and writes `layer-sweep.csv` plus per-layer logs under `-artifact-dir`.

```bash
go run ./cmd/turboquant-calibrate \
  -artifact-dir /tmp/turboquant-calibration-qwen25-7b-q4km \
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

To reuse an already completed sweep and only run selection plus validation:

```bash
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

The 2026-04-30 calibration artifact at `/tmp/turboquant-calibration-qwen25-7b-q4km/calibration.json` selected `0:q8_0,1:q8_0,3:q8_0,27:q8_0` and validated it on 16 sequences: `mean_nll=1.99192706`, `perplexity=7.32964486`, `mean_kl=0.05225737`, `duration=75.4s`.

The full-gate artifact at `/tmp/turboquant-calibration-qwen25-7b-q4km-full/calibration.json` reused the same sweep and selected spec, then validated all 256 sequences: `mean_nll=2.16333712`, `perplexity=8.70012261`, `mean_kl=0.04249074`, `duration=1159.0s`.

### Residual-Window Key Protection (Experimental)

`cmd/turboquant-eval` exposes `-key-cache-residual-window N` plus
`-key-cache-residual-dtype DTYPE` to test whether keeping the most recent N
key-cache positions at a higher dtype (e.g. `f16` or `q8_0`) and degrading
older positions to a Turbo* representation recovers most of the
`kq8-vturbo4` quality at lower memory. The flags are eval-only; the
production runtime is unchanged.

The round trip is implemented as a graph composition inside
`kvcache.Causal.Get`. After the standard cell-range view is built, the
cache splits the K view at the window boundary, casts the leading "old"
slice to the configured Turbo dtype and back to the recent dtype, and
concatenates the round-tripped old slice with the untouched recent slice.
This avoids in-place mutation of cache memory and runs entirely as
graph nodes per layer per forward pass.

Caveats:

- The split point is computed from `c.curPositions` and the cell-range
  positions and only applies when those positions are strictly monotonic
  in the active range. SWA caches and other allocators that do not store
  positions in monotonic order skip the split (`oldCount == 0`).
- The "recent N" window is per-batch in eval mode: across-batch state is
  whatever was already in the cache from earlier `Put` calls. This is
  acceptable for prefill-style Phase 0 evaluation where each sequence is
  a fresh forward pass.

Constraints enforced today:

- `-engine go` only; `-flash-attention=true` required.
- `-key-cache-residual-dtype` must be a Turbo dtype (`turbo2`, `turbo3`,
  `turbo4`).
- `-key-cache-type` must be a higher-precision dtype (`f16` or `q8_0`); the
  cache stores recent keys at this dtype.
- Cannot be combined with `-key-cache-layer-types` or
  `-key-cache-layer-preset`; the two policies target different axes
  (per-position vs per-layer) and we want to interpret either result on
  its own first.

### Residual-Window Phase 0 Results

Measured on 2026-04-30 with `qwen2.5:7b` Q4_K_M, CUDA on an RTX 4090,
`-num-ctx 1024`, `-batch-size 512`, full GPU offload, `-engine go`,
`-flash-attention=true`, `-reference-kv-cache-type f16`, snapshot
`tools/turboquant/testdata/qwen25_7b_phase0_tokens.json`. The smoke grid
was halted at `-limit 1` because every non-trivial window produces
worse-than-all-Turbo4-K quality; see the "Result" paragraph below.

| engine | cache | reference | sequences | tokens | mean NLL | perplexity | mean KL | duration |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| go | f16 K, turbo4 V (baseline, no window) | f16 | 1 | 1023 | 1.78206660 | 5.94212374 | 0.01206953 | 5.7s |
| go | f16 K, turbo4 V, window=2048, residual=turbo4 (no-op gate) | f16 | 1 | 1023 | 1.78206660 | 5.94212374 | 0.01206953 | 5.7s |
| go | f16 K, turbo4 V, window=64, residual=turbo4 | f16 | 1 | 1023 | 10.41724398 | 33431.17016 | 1.10987407 | 7.2s |
| go | f16 K, turbo4 V, window=1, residual=turbo4 | f16 | 1 | 1023 | 11.01511932 | 60786.27616 | 1.00385583 | 7.6s |
| go | all turbo4 K/V (reference for "all old" target) | f16 | 1 | 1023 | 6.10887882 | 449.83408783 | 4.46774773 | 5.7s |

The window=2048 row confirms the split-point gate is correct: when no
cells are older than the window, `applyKeyResidualWindow` is a no-op
and the result matches the baseline exactly.

The window=1 row is the closest the experiment can get to "round-trip
every old cell"; if the round trip were quality-preserving its result
would approach the all-turbo4 K row (`mean_nll≈6.11`). It does not —
it is materially **worse** than direct turbo4 K storage.

Result: **not promoting any window preset.** The
`Cast(F32→Turbo4).Cast(Turbo4→F32)` round trip used by
`kvcache.Causal.applyKeyResidualWindow` is not basis-invertible in the
attention path. `quantize_row_turbo4_0_ref`
(`ml/backend/ggml/ggml/src/ggml-turbo-quant.c:482-485`) applies a
forward Walsh-Hadamard rotation before quantization, and
`dequantize_row_turbo4_0` (lines 591-594) deliberately leaves the
output in the rotated domain because the production all-turbo4 K
cache compensates by applying `GGML_OP_TURBO_WHT` to Q before FA and
the inverse WHT to the FA output. The residual-window cast chain
runs against an `f16` (or `q8_0`) cache, which means the model graph
emits FA without those compensating WHT nodes, so the round-tripped
old-cell slice ends up in the rotated basis while Q stays in the
original basis. Concatenating the rotated old slice with the
unrotated recent slice and casting back to the recent dtype produces
mixed-basis K and catastrophic attention. See
`TURBOQUANT-DEBUG-LOG.md` (entry `2026-04-30 - Residual-window
Phase 0 grid: blocked by WHT basis mismatch`) for the full sequence
of aborts, F32-hop fixes, bisection, and root-cause evidence.

Two minimal F32-hop changes were committed to
`kvcache/causal.go applyKeyResidualWindow`. They were necessary to
escape the prior `ops.cpp:571` and `concat.cu:165` aborts and they
are justified by the supported quantize/dequantize traits and CUDA
concat type constraints — but they do **not** make the round trip
basis-correct. The next session should either implement a
`GGML_OP_TURBO_ROUNDTRIP` graph op that wraps quantize/dequantize in
compensating WHT for use on old cells only, or switch the experiment
to a non-WHT residual representation. Until one of those lands, the
`-key-cache-residual-window`/`-key-cache-residual-dtype` flags should
be treated as a not-yet-implemented research scaffold.

### Community Implementation Notes

The `tonbistudio/turboquant-pytorch` implementation is useful context for the remaining research path: <https://github.com/tonbistudio/turboquant-pytorch>. Its V3 direction matches the pattern seen here: QJL can improve raw inner-product theory while hurting softmax attention in practice; keys need materially more precision than values; keeping a recent fp16 window helps generation; and protecting sensitive layers improves attention-score match. In this codebase, the analogous production-safe baseline is `kq8-vturbo4`, while the adaptive preset is a lower-memory experiment that protects empirically sensitive key layers.

The local Qwen2.5 7B Q4_K_M dump at `/tmp/tqdump-layers-kv` also shows strong but layer-dependent K/V norm asymmetry. Per-layer stats were written to `/tmp/turboquant-kv-norms.csv`: mean K/V norm ratio averaged `8.40x`, ranging from `0.44x` to `106.89x`; the highest-ratio layers include 0, 1, 3, and 27, overlapping the adaptive key-layer set. Because some late layers invert the asymmetry, measured calibration is preferable to blindly protecting the first and last N layers. The next research step should therefore test residual-window key protection and K/V asymmetric policies before investing heavily in the paper's full JL residual path.

The snapshot format is:

```json
{
  "version": 1,
  "model": "qwen2.5:7b",
  "context_length": 2048,
  "sequences": [
    {"id": "example", "tokens": [1, 2, 3]}
  ]
}
```

## Per-Block Error Rig

Build the block rig against the local ggml base library:

```bash
g++ -std=c++17 \
  -Iml/backend/ggml/ggml/include \
  -Iml/backend/ggml/ggml/src \
  tools/turboquant/block_error.cpp \
  -Lbuild/lib/ollama -lggml-base -ldl -lpthread -lm \
  -Wl,-rpath=/home/yannik/Work/ollama-build/build/lib/ollama \
  -o /tmp/turboquant-block-error
```

Run it:

```bash
LD_LIBRARY_PATH=/home/yannik/Work/ollama-build/build/lib/ollama \
  /tmp/turboquant-block-error --rows 4096 --queries 32
```

It emits CSV for `turbo2`, `turbo3`, and `turbo4` with rotated-domain MSE and K.Q dot-product error. Use `--input-f32 path` to score a raw f32 K-cache dump with rows of width 128.

To dump real `SET_ROWS` source rows from the Go evaluator:

```bash
OLLAMA_TURBOQUANT_DUMP_DIR=/tmp/tqdump-setrows-src \
OLLAMA_TURBOQUANT_DUMP_LIMIT=12 \
OLLAMA_TURBOQUANT_DUMP_PACKED=1 \
OLLAMA_TURBOQUANT_DUMP_SET_ROWS_SRC=1 \
go run ./cmd/turboquant-eval \
  -engine go \
  -model qwen2.5:7b \
  -snapshot tools/turboquant/testdata/qwen25_7b_phase0_tokens.json \
  -kv-cache-type turbo4 \
  -num-ctx 1024 \
  -batch-size 512 \
  -num-gpu-layers 999 \
  -flash-attention=true \
  -limit 1 \
  -format json
```

The first two dumped f32 source tensors from that run were layer-0 K and V respectively. Scored as 2048 rows of width 128:

| source | variant | rows | mean cosine | MSE | dot MSE | dot MAE | dot max abs |
|---|---:|---:|---:|---:|---:|---:|---:|
| layer-0 K | turbo4 | 2048 | 0.99133785 | 11.82237582 | 13.47356062 | 2.59954361 | 17.01257514 |
| layer-0 V | turbo4 | 2048 | 0.99191456 | 0.00104463 | 0.00107373 | 0.02312298 | 0.25787297 |

The K and V cosines are similar, but K has much larger absolute dot-product error because the real K rows have much larger scale. This matches the split-cache perplexity result and points the next fix at the K inner-product estimator.

## KQ Estimator Diagnostic

`kq_error.cu` compares the exact CUDA `vec_dot_fattn_vec_KQ_turbo4` helper against f16/f32 KQ references on dumped real K rows and dumped rotated Q rows. It also compares CUDA Turbo KQ against CPU Turbo dequant KQ, which separates CUDA estimator/indexing errors from Turbo4 representation error.

Build it against the local CUDA ggml libraries:

```bash
nvcc -std=c++17 \
  -Iml/backend/ggml/ggml/include \
  -Iml/backend/ggml/ggml/src \
  -Iml/backend/ggml/ggml/src/ggml-cuda \
  tools/turboquant/kq_error.cu \
  -Lbuild/lib/ollama -lggml-cuda -lggml-base -lcublas -lcuda -ldl -lpthread -lm \
  -Xlinker -rpath -Xlinker /home/yannik/Work/ollama-build/build/lib/ollama \
  -o /tmp/turboquant-kq-error
```

Run it on the layer-0 K source dump and layer-0 rotated Q dump:

```bash
/tmp/turboquant-kq-error \
  --k-f32 /tmp/tqdump-setrows-src/000000_node_unknown_src0_RESHAPE_f32.bin \
  --q-rot-f32 /tmp/tqdump-setrows-src/000004_node_unknown_out_TURBO_WHT_f32.bin \
  --k-rows 2048 \
  --q-heads 28 \
  --q-tokens 512 \
  --kv-heads 4 \
  --q-head-mode all \
  --scale 0.08838834764831845 \
  --csv
```

Layer-0 result from the 2026-04-29 dump:

| comparison | pairs | mean abs error | RMS error | max abs error | mean relative error | sign mismatch |
|---|---:|---:|---:|---:|---:|---:|
| CUDA Turbo vs f16 reference | 14336 | 5.80298943 | 10.90493814 | 86.20838584 | 0.08412686 | 0.00258092 |
| CUDA Turbo vs f32 reference | 14336 | 5.80280525 | 10.90472032 | 86.16620204 | 0.08413211 | 0.00258092 |
| CUDA Turbo vs CPU Turbo | 14336 | 0.00000975 | 0.00002964 | 0.00039618 | 0.00000006 | 0 |
| CPU Turbo vs f16 reference | 14336 | 5.80298990 | 10.90493973 | 86.20850706 | 0.08412686 | 0.00258092 |

The CUDA KQ helper matches CPU Turbo dequant KQ, so the layer-0 failure is not caused by CUDA Turbo KQ indexing or Q-register layout. The large f16/f32 reference error is already present in the Turbo4 K representation.

## K Scale Calibration Diagnostic

`k_calibration.cpp` tests whether the Turbo4 K error is primarily a bad per-row scale fit. It compares current Turbo4 KQ error with two hypothetical rescalings:

- `vector_optimal_scale`: least-squares scalar that best fits the dequantized Turbo row to the f16 rotated row.
- `kq_optimal_scale`: scalar that best fits that row's observed KQ logits over the selected Q samples.

Build it:

```bash
g++ -std=c++17 \
  -Iml/backend/ggml/ggml/include \
  -Iml/backend/ggml/ggml/src \
  tools/turboquant/k_calibration.cpp \
  -Lbuild/lib/ollama -lggml-base -ldl -lpthread -lm \
  -Wl,-rpath=/home/yannik/Work/ollama-build/build/lib/ollama \
  -o /tmp/turboquant-k-calibration
```

Run it on the same layer-0 dumps:

```bash
/tmp/turboquant-k-calibration \
  --k-f32 /tmp/tqdump-setrows-src/000000_node_unknown_src0_RESHAPE_f32.bin \
  --q-rot-f32 /tmp/tqdump-setrows-src/000004_node_unknown_out_TURBO_WHT_f32.bin \
  --k-rows 2048 \
  --q-heads 28 \
  --q-tokens 512 \
  --kv-heads 4 \
  --q-head-mode all \
  --query-token-mode causal \
  --query-stride 16 \
  --scale 0.08838834764831845 \
  --top-rows 12
```

Layer-0 causal-query sample result:

| candidate | pairs | mean abs error | RMS error | max abs error | MAE reduction |
|---|---:|---:|---:|---:|---:|
| current Turbo4 | 236544 | 6.03566201 | 11.33435533 | 95.20256973 | - |
| vector-optimal scale | 236544 | 7.44733790 | 15.74678145 | 148.63571870 | -23.39% |
| KQ-optimal scale | 236544 | 4.35083507 | 7.18178773 | 82.62110747 | 27.91% |

Scale and distribution notes from the same run:

| metric | mean | p50 | p95 | min | max |
|---|---:|---:|---:|---:|---:|
| vector-optimal scale | 0.99134046 | 0.99410327 | 0.99529076 | 0.97982411 | 0.99682273 |
| KQ-optimal scale | 1.00934002 | 1.00553096 | 1.03465791 | 0.90061405 | 1.11748468 |
| ref row norm | 273.97882850 | 259.81039560 | 466.59495930 | 107.21186430 | 467.33040650 |
| Turbo row norm | 273.97724170 | 259.84729940 | 466.61460090 | 107.19186520 | 467.41422430 |
| saturation rate | 0.07919693 | 0.07812500 | 0.09375000 | 0.05468750 | 0.11718750 |

The row norms already match, and the vector-optimal rescale makes KQ error worse. A KQ-only rescale can partially overfit the observed Q samples, but it still leaves large error. That makes a simple Turbo4 norm calibration bug unlikely; the missing information is directional residual, which is what the paper's JL term is supposed to recover.

## JL Estimator Diagnostic

`jl_estimator.cpp` brings the paper's residual estimator forward into a CPU-only probe. It first runs a synthetic Gaussian sanity check for the estimator:

```text
estimate(<r, q>) = sqrt(pi/2) * ||r|| / m * sum_j sign(<g_j, r>) * <g_j, q>
```

Then, when real dumps are provided, it compares current 4-bit Turbo K, 3-bit centroid-only, and 3-bit+JL KQ error on the same real K/Q rows. The diagnostic also supports larger `m` values so estimator variance can be separated from implementation mistakes.

Build it:

```bash
g++ -std=c++17 \
  -Iml/backend/ggml/ggml/include \
  -Iml/backend/ggml/ggml/src \
  tools/turboquant/jl_estimator.cpp \
  -Lbuild/lib/ollama -lggml-base -ldl -lpthread -lm \
  -Wl,-rpath=/home/yannik/Work/ollama-build/build/lib/ollama \
  -o /tmp/turboquant-jl-estimator
```

Synthetic sanity only:

```bash
/tmp/turboquant-jl-estimator --synthetic-pairs 4096
```

Layer-0 synthetic result at `m=128`:

| metric | value |
|---|---:|
| mean error | 0.00049086 |
| mean abs error | 0.08902913 |
| RMS error | 0.11188156 |
| empirical variance | 0.01251748 |
| variance bound | 0.01558581 |

Run on the real layer-0 K/Q dumps:

```bash
/tmp/turboquant-jl-estimator \
  --k-f32 /tmp/tqdump-setrows-src/000000_node_unknown_src0_RESHAPE_f32.bin \
  --q-rot-f32 /tmp/tqdump-setrows-src/000004_node_unknown_out_TURBO_WHT_f32.bin \
  --k-rows 2048 \
  --q-heads 28 \
  --q-tokens 512 \
  --kv-heads 4 \
  --q-head-mode all \
  --query-token-mode causal \
  --query-stride 16 \
  --scale 0.08838834764831845 \
  --synthetic-pairs 4096 \
  --m 128
```

Additional diagnostic knobs:

- `--rotation-mode rademacher` reconstructs raw Q from the plain-WHT dump with `turbo_cpu_fwht_inverse`, then applies deterministic `D * WHT * D` to both K and Q for the 3-bit path.
- `--centroid-mode calibrated` runs Lloyd-Max iterations over the selected real K rows after the selected rotation.
- `rotation_oracle` reports the D-rotated f16 reference against the plain-WHT f16 reference to validate the Q reconstruction path.

Layer-0 causal-query sample results:

| m | candidate | mean abs error | RMS error | max abs error | vs current 4-bit MAE |
|---:|---|---:|---:|---:|---:|
| 128 | current 4-bit Turbo | 6.03566201 | 11.33435533 | 95.20256973 | 1.0000 |
| 128 | 3-bit centroid raw | 19.59945144 | 46.06117038 | 309.94949040 | 3.2473 |
| 128 | 3-bit+JL raw | 8.20214240 | 16.77033341 | 219.35475360 | 1.3589 |
| 128 | 3-bit centroid fit | 12.00185512 | 25.53061977 | 210.83858850 | 1.9885 |
| 128 | 3-bit+JL fit | 8.32803067 | 16.92898838 | 232.39961930 | 1.3798 |
| 256 | 3-bit+JL raw | 5.47819413 | 11.14487191 | 136.46838180 | 0.9076 |
| 512 | 3-bit+JL fit | 3.78422994 | 7.72563802 | 110.60164100 | 0.6270 |
| 1024 | 3-bit+JL fit | 2.73419320 | 5.56674177 | 78.95135753 | 0.4530 |

At the paper-sized `m=128`, JL recovers a large part of the 3-bit centroid error, but it is still worse than the current 4-bit centroid-only K path on these real rows. Increasing `m` improves error with the expected variance curve and beats current 4-bit at `m>=256`, so the estimator math is not obviously broken.

`D` rotation and centroid-calibration results at `m=128`:

| rotation | centroids | candidate | mean abs error | RMS error | vs current 4-bit MAE |
|---|---|---|---:|---:|---:|
| plain | paper | 3-bit+JL fit | 8.32803067 | 16.92898838 | 1.3798 |
| plain | calibrated | 3-bit+JL fit | 8.66629919 | 18.96905692 | 1.4358 |
| rademacher | paper | 3-bit+JL fit | 7.74360037 | 17.17663389 | 1.2830 |
| rademacher | calibrated | 3-bit+JL fit | 10.17055156 | 23.86975823 | 1.6851 |

For the Rademacher runs, `rotation_oracle` was `mean_abs_error=0.01452560`, so reconstructing raw Q from the plain-WHT dump and re-rotating with `D * WHT * D` is not the cause of the remaining error. Rademacher rotation helps slightly with paper centroids, but not enough for `m=128` to beat current 4-bit. Centroid calibration over this single layer/sample makes the JL path worse.

## Dump Indexing And Per-Layer Sweep

`cmd/turboquant-dump-index` parses TurboQuant dump metadata sidecars and infers layer-level K/V/Q groupings:

```bash
go run ./cmd/turboquant-dump-index \
  -dir /tmp/tqdump-layers-kv \
  -format csv > /tmp/turboquant-layer-index.csv
```

The CSV includes `k_src`, `v_src`, `q_rot`, packed SET_ROWS outputs, and inferred dimensions (`k_rows`, `kv_heads`, `q_heads`, `q_tokens`) per layer.

To capture the first 28-layer prompt batch with K/V source rows:

```bash
OLLAMA_TURBOQUANT_DUMP_DIR=/tmp/tqdump-layers-kv \
OLLAMA_TURBOQUANT_DUMP_LIMIT=196 \
OLLAMA_TURBOQUANT_DUMP_SET_ROWS_SRC=1 \
OLLAMA_TURBOQUANT_DUMP_PACKED=1 \
go run ./cmd/turboquant-eval \
  -engine go \
  -model qwen2.5:7b \
  -snapshot tools/turboquant/testdata/qwen25_7b_phase0_tokens.json \
  -kv-cache-type turbo4 \
  -num-ctx 1024 \
  -batch-size 512 \
  -num-gpu-layers 999 \
  -flash-attention=true \
  -limit 1 \
  -format json
```

That run writes 196 dumps for the first batch, 28 complete layer groups, and may still fail later at token 513 with the known Turbo4 invalid-NLL failure. The captured dump is still usable.

Per-layer JL sweep artifacts from the 2026-04-30 run:

```text
/tmp/turboquant-layer-index.csv
/tmp/turboquant-layer-jl-summary.csv
/tmp/turboquant-layer-jl-pivot.csv
```

The sweep used causal query sampling with `--query-stride 64` and compared current 4-bit against `3-bit+JL fit`:

| candidate | layers | average current MAE | average JL MAE | average JL/current | layers beating current |
|---|---:|---:|---:|---:|---:|
| plain paper, m=128 | 28 | 0.923319 | 1.495933 | 1.867506 | 0 |
| plain paper, m=256 | 28 | 0.923319 | 1.037808 | 1.312817 | 1 |
| Rademacher paper, m=128 | 28 | 0.923319 | 1.349065 | 1.850303 | 0 |

The only `m=256` win was layer 0 (`current MAE=6.0133`, `m=256 JL fit MAE=5.5962`). Layer 27 is also a large-current-error outlier (`current MAE=11.2406`), but `m=256` still did not beat current 4-bit there (`JL fit MAE=11.9529`). Middle layers already have low current 4-bit KQ error, and JL residual variance dominates.
