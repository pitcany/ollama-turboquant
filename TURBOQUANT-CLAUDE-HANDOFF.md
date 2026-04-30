# TurboQuant Claude Code Handoff

Date: 2026-04-30
Repo: `/home/yannik/Work/ollama-build`

## Current Status

The previous CUDA garbling/debug phase is closed. The current work is no longer
"find the first GPU mismatch"; it is TurboQuant quality/calibration research on
top of the corrected Go-engine eval path.

What is now implemented and documented:

- Phase 0 token snapshot and eval tools:
  - `cmd/turboquant-tokenize`
  - `cmd/turboquant-eval`
  - `cmd/turboquant-dump-index`
  - `tools/turboquant/eval`
- Split K/V cache dtype support:
  - `kvcache.Causal.InitSplit`
  - per-layer key dtype overrides
  - production preset `OLLAMA_KV_CACHE_TYPE=kq8-vturbo4`
- Automatic calibration:
  - `tools/turboquant/layer_sweep.sh`
  - `cmd/turboquant-select-layers`
  - `cmd/turboquant-calibrate`
  - reusable selector package `tools/turboquant/layerselect`
- Documentation:
  - `tools/turboquant/README.md` is the canonical user/operator doc for Phase 0
    tools, commands, presets, and measured results.
  - `TURBOQUANT-DEBUG-LOG.md` is the chronological evidence log.
  - This file is the active handoff for Claude Code.

## Key Measurements

All results below are for `qwen2.5:7b` Q4_K_M with
`tools/turboquant/testdata/qwen25_7b_phase0_tokens.json`, `-num-ctx 1024`,
`-batch-size 512`, full GPU offload, and f16 reference where KL is reported.

Full 256-sequence Phase 0 results:

| cache | sequences | tokens | mean NLL | perplexity | mean KL | duration |
|---|---:|---:|---:|---:|---:|---:|
| llama f16 | 256 | 261888 | 2.14183994 | 8.51509047 | - | 309.0s |
| `kq8-vturbo4` | 256 | 261888 | 2.15085919 | 8.59223759 | 0.01705642 | 1123.3s |
| adaptive K q8_0 layers `0,1,3,27` + Turbo4 V | 256 | 261888 | 2.16333712 | 8.70012261 | 0.04249074 | 1159.0s |

Interpretation:

- All-Turbo4 K/V still fails catastrophically.
- Split runs isolate the failure to Turbo4 key scoring.
- Turbo4 V is fine when K is f16/q8_0.
- `kq8-vturbo4` is the practical safe preset.
- The adaptive preset is full-gate validated but noisier than all-q8 K.

Artifacts:

- 16-sequence calibration:
  `/tmp/turboquant-calibration-qwen25-7b-q4km/calibration.json`
- Full 256-sequence adaptive gate:
  `/tmp/turboquant-calibration-qwen25-7b-q4km-full/calibration.json`
- Single-layer sweep CSV:
  `/tmp/turboquant-layer-sweep-limit4-q8_0.csv`
- Per-layer K/V norm stats:
  `/tmp/turboquant-kv-norms.csv`

## Community Context

The user pointed to <https://github.com/tonbistudio/turboquant-pytorch>.
Relevant takeaways from that implementation:

- QJL can be theoretically attractive for raw inner products but harmful under
  softmax attention because it adds variance.
- Keys need more precision than values.
- Recent-token fp16 residual windows help generation.
- Layer-adaptive protection is useful, but our local K/V norm data is
  layer-dependent enough that measured calibration is preferable to blindly
  protecting the first and last N layers.

Local norm check from `/tmp/tqdump-layers-kv`:

- Mean K/V norm ratio average: `8.40x`
- Min: `0.44x`
- Max: `106.89x`
- Highest-ratio layers include `0`, `1`, `3`, and `27`, overlapping the
  selected adaptive key layers.

## Recommended Next Step

Do a residual-window key-protection experiment before investing further in the
paper's full JL residual path.

Practical shape:

1. Add a Phase 0 evaluator mode that keeps the most recent N key-cache tokens at
   a higher key dtype, probably `q8_0` or `f16`, while older key rows remain
   Turbo4 and values remain Turbo4.
2. Start in the eval path, not production runtime, so the experiment can be
   validated without committing to cache-lifetime complexity.
3. Test a small grid:
   - no window
   - recent 64 tokens
   - recent 128 tokens
   - recent 256 tokens
   - compare against `kq8-vturbo4`
4. Use `-reference-kv-cache-type f16` and the Phase 0 snapshot.
5. Only promote to production runtime if KL/perplexity justify the added cache
   machinery.

## Verification Commands From Latest Session

```bash
git diff --check
bash -n tools/turboquant/layer_sweep.sh

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

GOCACHE=/tmp/ollama-build-gocache \
OLLAMA_LIBRARY_PATH=/home/yannik/Work/ollama-build/build/lib/ollama \
go test -count=1 ./ml/backend/ggml
```

## Prompt For Claude Code

```text
We are in /home/yannik/Work/ollama-build working on TurboQuant. Read these files first:

- AGENTS.md
- TURBOQUANT-CLAUDE-HANDOFF.md
- tools/turboquant/README.md
- TURBOQUANT-DEBUG-LOG.md
- TURBOQUANT-PAPER-IMPLEMENTATION-PLAN.md

Current state:
- The old CUDA garbling phase is closed.
- Phase 0 eval/calibration tooling exists and is documented.
- `kq8-vturbo4` is the practical safe preset: full 256-sequence result is mean_nll=2.15085919, perplexity=8.59223759, mean_kl=0.01705642.
- Adaptive key-layer preset `0:q8_0,1:q8_0,3:q8_0,27:q8_0` is full-gate validated but noisier: mean_nll=2.16333712, perplexity=8.70012261, mean_kl=0.04249074.
- Artifacts are in `/tmp/turboquant-calibration-qwen25-7b-q4km-full/calibration.json`, `/tmp/turboquant-layer-sweep-limit4-q8_0.csv`, and `/tmp/turboquant-kv-norms.csv`.
- The user pointed to https://github.com/tonbistudio/turboquant-pytorch; the useful takeaway is to test key precision, residual windows, and measured layer adaptation before spending more effort on full QJL/JL.

Task:
Continue with the next research step: design and implement an eval-only residual-window key-protection experiment. Keep values Turbo4, keep old keys Turbo4, but let the most recent N key-cache tokens use a higher dtype such as q8_0 or f16. Start with tests and a narrow implementation path in the eval harness. Do not make production runtime changes until the eval result justifies them.

Important constraints:
- Do not revert unrelated dirty work.
- Document every command/result in `TURBOQUANT-DEBUG-LOG.md`.
- Keep operator-facing instructions and final measured results in `tools/turboquant/README.md`.
- Use the existing Phase 0 snapshot and f16 reference KL.
- Verify with focused Go tests and `git diff --check`.
```

---

# Legacy 2026-04-29 CUDA Debug Handoff

Date: 2026-04-29
Repo: `/home/yannik/Work/ollama-build`

## Current Status

This is not fixed end-to-end yet.

What is fixed/verified:

- The turbo3 KQ bulk-load sign-bit mapping bug is fixed in `ml/backend/ggml/ggml/src/ggml-cuda/fattn-common.cuh`.
- CUDA backend builds successfully with the current patch.
- Direct deterministic CUDA FA diagnostics match CPU for:
  - turbo2 direct FA
  - turbo3 direct FA
  - turbo4-no-QJL direct FA
  - turbo2/turbo3/turbo4-no-QJL cache/GQA synthetic paths

What is still broken:

- Real model `qwen2.5:7b` with `OLLAMA_KV_CACHE_TYPE=turbo4` on GPU still produces bad one-token output.
- The remaining issue is not proven yet. Do not start speculative patching.

## Debugging Discipline

Do not blindly guess at fixes.

The priority is to rigorously find the first concrete mismatch, prove why it happens, and then make the smallest targeted fix. Previous attempts went in circles when patches were made before the bug was isolated. Avoid repeating that.

Required workflow:

1. Reproduce the failure with a deterministic command.
2. Identify the first divergent tensor/op boundary with trustworthy instrumentation.
3. Prove whether the input to that boundary matches.
4. If input matches and output diverges, inspect that exact kernel/op.
5. Create a minimal synthetic reproduction for that boundary where possible.
6. Patch only the confirmed root cause.
7. Re-run the focused repro and the real model smoke test.

Rules:

- Do not patch FA, WHT, or `set_rows` based on suspicion alone.
- Do not rely on post-compute intermediate dumps; allocator reuse can invalidate them.
- Do not treat CPU-vs-GPU ordinary matmul drift as proof of a TurboQuant bug.
- Do not keep multiple speculative edits in the tree. If an edit is only diagnostic, label it clearly or remove it before testing a fix.
- Every proposed code change should state the exact observed mismatch it explains.
- If a fix does not explain the first proven mismatch, do not make it.

## Logging Requirement

The next debugging session should keep a running log as it works. The log can be appended to this file or written to a new clearly named markdown file such as `TURBOQUANT-DEBUG-LOG.md`.

Each log entry should include:

- timestamp
- command or test run
- exact environment variables that affect CUDA/GGML/Ollama behavior
- observed result
- current hypothesis
- whether the result strengthens, weakens, or disproves that hypothesis
- files changed, if any
- reason each code change is justified by evidence

Use this format:

```md
## 2026-04-29 HH:MM - Short Title

Command:
```bash
...
```

Observation:
...

Interpretation:
...

Next:
...
```

Do not replace evidence with summaries only. Keep the key numbers: first mismatching op, shape, max error, byte offset, token output, and relevant file/line references.

## Important Findings

The original KQ bulk-load bug:

- In the turbo3 KQ dot product bulk path, `cpy_ne=2` lane 1 owns elements `4..7`, but the old code read upper sign bits `0..3`.
- Correct mapping uses the actual element offset:
  - `elem = j0 + 2*k1`
  - sign bit uses `elem & 7`
- This is documented in-code near the fixed turbo3 KQ path.

The first activation dump approach was invalid:

- Dumping graph intermediates after `ggml_backend_sched_graph_compute_async` completes is not reliable.
- ggml allocator memory reuse can make early intermediate tensors contain later values.
- The new dump path uses `ggml_backend_sched_set_eval_callback`, so selected tensors are dumped immediately after each node is computed.

CPU-vs-GPU activation comparison is noisy:

- Ordinary CUDA `MUL_MAT` differs numerically from CPU early in the graph.
- Do not treat the first CPU/GPU `MUL_MAT` difference as a TurboQuant bug.
- Better comparisons are same-backend comparisons, especially GPU `f16` KV vs GPU `turbo4` KV, or focused synthetic graphs.

## Current Debug Instrumentation

Added callback-based dump support in:

- `ml/backend/ggml/ggml.go`

Environment variables:

- `OLLAMA_TURBOQUANT_DUMP_DIR=/tmp/some-dir`
- `OLLAMA_TURBOQUANT_DUMP_LIMIT=256`
- `OLLAMA_TURBOQUANT_DUMP_ALL_F32=1`
- `OLLAMA_TURBOQUANT_DUMP_PACKED=1`

Default dump mode captures TurboQuant-relevant nodes:

- `TURBO_WHT`
- `FLASH_ATTN_EXT`
- packed `SET_ROWS` only when `OLLAMA_TURBOQUANT_DUMP_PACKED=1`

`OLLAMA_TURBOQUANT_DUMP_ALL_F32=1` captures all F32 graph nodes. This is intrusive and slow.

Comparator added:

```bash
scripts/turboquant-compare-dumps.py /tmp/reference-dump /tmp/candidate-dump --top 40
```

It compares matching `.bin` files in sorted order and reports:

- `f32` max absolute and relative error
- byte mismatch count for packed/raw dumps

## Build Commands Used

CUDA backend:

```bash
cmake --build build -j2 --target ggml-cuda
cp build/lib/ollama/libggml-cuda.so build/lib/ollama/cuda_v12/libggml-cuda.so
cp build/lib/ollama/libggml-base.so.0.0.0 build/lib/ollama/cuda_v12/libggml-base.so
```

Go compile check:

```bash
GOCACHE=/tmp/ollama-build-gocache go test ./ml/backend/ggml -run TestNonExistent
```

Binary rebuild:

```bash
GOCACHE=/tmp/ollama-build-gocache \
CGO_LDFLAGS="-L/home/yannik/Work/ollama-build/build/lib/ollama" \
go build -trimpath -buildmode=pie \
  -ldflags='-extldflags "-Wl,--version-script=hide-ggml.ver"' \
  -o ollama-tq .
```

## Runtime Library Gotcha

Do not keep both of these active:

- `build/lib/ollama/libggml-cuda.so`
- `build/lib/ollama/cuda_v12/libggml-cuda.so`

Keeping both causes the known double-load crash.

Current state after debugging:

- flat `build/lib/ollama/libggml-cuda.so` was moved to:
  - `/tmp/ollama-build-libggml-cuda-flat-0429a.so`
- `build/lib/ollama/cuda_v12/libggml-cuda.so` remains active.

## Diagnostic Commands

Deterministic CUDA harness:

```bash
g++ -std=c++17 \
  -Iml/backend/ggml/ggml/include \
  -Iml/backend/ggml/ggml/src \
  /tmp/turbo_set_rows_diag.cpp \
  -Lbuild/lib/ollama -lggml-base -ldl -lpthread -lm \
  -Wl,-rpath=/home/yannik/Work/ollama-build/build/lib/ollama \
  -o /tmp/turbo_set_rows_diag

LD_LIBRARY_PATH=/home/yannik/Work/ollama-build/build/lib/ollama:/home/yannik/Work/ollama-build/build/lib/ollama/cuda_v12 \
CUDA_VISIBLE_DEVICES=0 \
/tmp/turbo_set_rows_diag
```

Observed result:

- Direct FA paths match CPU.
- `set_rows` still reports a few packed-byte differences for 512-wide rows. These appear threshold/packing-adjacent and are not yet proven to explain model garbling.

GPU turbo4 smoke with dump:

```bash
OLLAMA_LIBRARY_PATH=/home/yannik/Work/ollama-build/build/lib/ollama \
LD_LIBRARY_PATH=/home/yannik/Work/ollama-build/build/lib/ollama/cuda_v12 \
OLLAMA_TURBOQUANT_DUMP_DIR=/tmp/tq-dump-gpu \
OLLAMA_TURBOQUANT_DUMP_LIMIT=256 \
CUDA_VISIBLE_DEVICES=0 \
OLLAMA_HOST=127.0.0.1:9999 \
OLLAMA_KV_CACHE_TYPE=turbo4 \
OLLAMA_FLASH_ATTENTION=1 \
OLLAMA_NEW_ENGINE=1 \
OLLAMA_CONTEXT_LENGTH=512 \
./ollama-tq serve
```

Request:

```bash
curl -sS http://127.0.0.1:9999/api/generate \
  -d '{"model":"qwen2.5:7b","prompt":"What comes after three? Answer with one word.","stream":false,"options":{"temperature":0,"num_predict":1,"num_ctx":128}}'
```

Observed bad turbo4 GPU outputs included:

- `" words"`
- `" DevComponents"`

CPU-only reference command shape:

```bash
OLLAMA_LIBRARY_PATH=/home/yannik/Work/ollama-build/build/lib/ollama \
LD_LIBRARY_PATH=/home/yannik/Work/ollama-build/build/lib/ollama \
OLLAMA_TURBOQUANT_DUMP_DIR=/tmp/tq-dump-cpu \
OLLAMA_TURBOQUANT_DUMP_LIMIT=256 \
OLLAMA_HOST=127.0.0.1:9998 \
OLLAMA_KV_CACHE_TYPE=turbo4 \
OLLAMA_FLASH_ATTENTION=1 \
OLLAMA_NEW_ENGINE=1 \
OLLAMA_CONTEXT_LENGTH=512 \
OLLAMA_LLM_LIBRARY=cpu_avx2 \
./ollama-tq serve
```

CPU-only request should include:

```json
"options": {"temperature": 0, "num_predict": 1, "num_ctx": 128, "num_gpu": 0}
```

CPU-only produced `"one"` for the same prompt in one run.

## Files Touched

Intentional changes from this session:

- `ml/backend/ggml/ggml/src/ggml-cuda/fattn-common.cuh`
  - corrected turbo KQ bulk indexing
  - updated turbo4 KQ bulk path for current 3-bit+QJL layout without QJL correction
- `ml/backend/ggml/ggml.go`
  - callback-based activation dump support
- `scripts/turboquant-compare-dumps.py`
  - dump comparator

Pre-existing/ongoing turbo4 QJL work was already dirty before this handoff:

- `ml/backend/ggml/ggml/src/ggml-common.h`
- `ml/backend/ggml/ggml/src/ggml-cuda/set-rows.cu`
- `ml/backend/ggml/ggml/src/ggml-cuda/turbo-quant.cuh`
- `ml/backend/ggml/ggml/src/ggml-turbo-quant.c`

## Next Steps

1. Keep the confirmed KQ bulk-load fix.

2. Do not use post-compute tensor dumps for intermediate evidence.

3. Improve dump comparator alignment.
   - Sorted sequence works only when both graphs emit the same selected ops.
   - GPU `f16` vs GPU `turbo4` differs structurally because turbo inserts `TURBO_WHT`.
   - Align by `(op, name, shape)` or write a small analyzer that maps expected corresponding nodes.

4. Build a focused synthetic graph that matches the real model turbo path:

```text
ROPE -> TURBO_WHT -> SET_ROWS turbo4 -> FLASH_ATTN_EXT -> inverse TURBO_WHT
```

5. Compare at the first concrete turbo boundary:
   - packed `SET_ROWS` output bytes
   - dequantized K/V as consumed by FA
   - FA output before inverse WHT
   - inverse WHT output

6. Only patch after the first concrete mismatch is proven.

Likely remaining suspects:

- GPU turbo4 quantize/packing flow in `set-rows.cu`
- FA turbo4 unpack/dequant path in `fattn-common.cuh`
- WHT contract mismatch only if input is proven matching and output diverges

Do not assume the remaining bug is the same as the original KQ bulk-load issue until the callback dumps or synthetic graph prove it.
