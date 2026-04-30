# TurboQuant Full-Paper Implementation Plan

This plan brings the codebase from "PolarQuant + plain WHT" (current state, commit `60a7e262`) to the full TurboQuant design from arXiv 2504.19874: **Hadamard–Rademacher rotation + 3-bit PolarQuant + JL residual sign encoding with random Gaussian R + JL-aware FA inner-product estimator.**

The plan is structured around a **fail-fast staircase**: each phase has a synthetic test that *must pass before the next phase is attempted*, plus a quality-tracking metric (perplexity / KL divergence vs fp16) that must improve monotonically.

---

## Phase 0: Eval infrastructure (prerequisite — do this FIRST)

**Why first:** Without a numerical quality metric, we have no way to detect partial regressions. Smoke tests like "say four" are too coarse — quantization bugs can leave smoke tests passing while perplexity blows up by 5 PP.

**Deliverables:**

1. **Perplexity harness** as a Go test (or a small CLI) that:
   - Loads `qwen2.5:7b` once.
   - Runs forward passes over a fixed held-out corpus (e.g. 256 sequences of 1024 tokens from Wiki-103 + a code subset). Use a tokenized snapshot checked into the repo so runs are reproducible.
   - Reports: mean NLL, perplexity, *and per-layer KL divergence* of the output logits vs the f16 reference (cached once).
   - Takes `OLLAMA_KV_CACHE_TYPE` as input so the same code runs on f16, turbo4, etc.
   - Total runtime target: <2 min on a 4090 so we can run it after every phase.

2. **Per-block error rig** (extension to `/tmp/turbo_set_rows_diag.cpp`):
   - Generate K of shape `(d=128, n=4096)` with realistic distribution (samples from a real model's K cache, dumped to disk).
   - Quantize, dequantize, measure: mean cosine similarity, mean angular error, max element error, MSE.
   - Repeat for K·Q dot products (most relevant metric for FA).
   - Emit a CSV row per phase so we can track regressions.

3. **Determinism check**: same seed, same input → bit-identical KV cache bytes on CPU and GPU. This is non-negotiable; without it, encode/decode contract bugs are silent.

**Gate to Phase 1:** f16 perplexity baseline + turbo4 (current) perplexity recorded. If turbo4 isn't within ~2 PP of f16 on this corpus, we found a hidden bug *before* adding complexity. Stop and investigate.

---

## Phase 1: Hadamard–Rademacher rotation (D · WHT · D scaffold)

This is the smallest standalone change and is required for any JL argument to be valid. Do it before touching the bit layout.

**Design choices to nail down before coding:**

- **Granularity of D.** Per-block (one Rademacher vector per 128-element block), per-head, or per-tensor (one vector for the entire K cache of a layer)? The paper uses per-tensor — fewer random bits to plumb, lower noise floor. Choose **per-tensor**, regenerated when the tensor is allocated.
- **Storage.** D is 128 sign bits = 16 bytes per quantized tensor. Store it as a `uint8_t[16]` field on the *tensor*, not the block. New ggml metadata or an external sidecar keyed by tensor pointer. The cleanest place is `struct ggml_tensor::extra` or a parallel map in the cache.
- **Determinism.** D must be generated on whichever side touches it first (CPU encode during prompt eval, or GPU encode if the runtime starts on GPU). It's then *frozen* for the lifetime of the cache. Seed from `tensor_pointer ^ a_well_known_constant` so a rebuild produces a different D but a session is consistent.
- **Inverse rotation.** Q (the query) must also be rotated by `D · WHT · D` before the K·Q inner product. Output of attention must be inverse-rotated by `D · WHT^{-1} · D = D · WHT · D / d` (since WHT is self-inverse up to scale). The existing `TURBO_WHT` op needs a sibling `TURBO_WHT_RADEMACHER` op that takes D as a second input.

**Implementation steps:**

1. **CPU forward path first** (`ggml-turbo-quant.c`):
   - Add `turbo_cpu_fwht_rademacher(float *x, const uint8_t *D, int dim, int direction)` that applies `D` (sign flip), then `WHT`, then `D` again.
   - Update `quantize_row_turbo4_0_ref` to accept a `D` pointer; if NULL, fall back to plain WHT (compat).
   - Update `dequantize_row_turbo4_0` similarly.
   - **Sanity test (synthetic harness):** generate random K, encode with D, dequantize with D — check round-trip max error. Should match the no-D case (rotation is orthogonal, so MSE doesn't change).

2. **CPU FA path** (`ggml-cpu/ops.cpp`): wire D into the rotation step. CPU FA tests must still pass.

3. **GPU encode** (`set-rows.cu` `k_set_rows_turbo4`):
   - Pass `D` as a `__constant__ uint8_t[16]` per cache or as a pointer in the kernel args.
   - In the encode kernel, after loading the source row, apply `x[i] *= D[i] ? 1 : -1` (one sign per dim), then do the WHT, then apply D again.
   - Use `__ldg` for D to land it in read-only cache.
   - **Synthetic test:** quantize the same input on CPU with D and on GPU with D; assert byte-equal block output. *This catches every encode disagreement before they compound through FA.*

4. **GPU dequant + FA** (`turbo-quant.cuh`, `fattn-common.cuh`):
   - Q rotation: add a step in the FA kernel prologue that loads D and rotates Q before the inner product. (Or do this rotation in a wrapper op above FA, which is cleaner — `TURBO_WHT_RADEMACHER` applied to Q.)
   - Note: the dequant path doesn't need to change yet, because the bits stored represent the *rotated* vector. The K·Q dot stays the same; only Q's rotation changes.

5. **Inverse rotation of attention output**: the existing inverse-WHT op needs to become inverse-Hadamard–Rademacher. Same kernel, accepts D.

**Phase 1 checkpoints:**

- ✅ Encode determinism: CPU(D) and GPU(D) produce byte-equal blocks for the same input + D.
- ✅ Round-trip MSE unchanged from no-D case (orthogonal rotation preserves L2).
- ✅ Perplexity from Phase 0 rig: should be **identical** to current (within numerical noise — say <0.05 PP) because we haven't changed the bit layout, only the rotation. *If perplexity changes here, there's a bug.*
- ✅ FA cosine-similarity test: K·Q dot on D-rotated vectors agrees with K·Q on un-rotated vectors to <1e-5.

If any of these fail, **stop**. Don't proceed to Phase 2.

---

## Phase 2: Restore 3-bit PolarQuant layout (bit budget reshuffle)

Now that rotation works, switch the bit layout from "16-centroid 4-bit" to "8-centroid 3-bit + 1-bit JL placeholder." The JL bit will *initially* still be the simplified per-element sign (R = I); we'll upgrade R in Phase 3.

**Key design decision:** keep the `TURBO4_USE_4BIT` switch — flipping it to 0 should give us the 3-bit + simplified-QJL state that exists today (post-`5e2a3175` minus our recent revert). That branch is already audited in the debug log; this phase just re-activates it on top of the new D-rotation.

**Steps:**

1. **Flip the build flag for the 3-bit branch in a feature build:**
   ```c
   #define TURBO4_USE_4BIT 0
   ```
   Initially do this in a *separate config* (say `OLLAMA_KV_CACHE_TYPE=turbo4_3bit`) so we can A/B against the 4-bit baseline without disturbing it. This requires:
   - A new `GGML_TYPE_TURBO4_3BIT_0` ggml type slot, or
   - A runtime-selectable variant via tensor metadata.

   **Recommended:** add the new type slot. It's mechanical and gives clean A/B without `#ifdef` branches.

2. **Re-derive the 8-centroid Lloyd-Max table** for the post-Hadamard-Rademacher distribution (which is now provably closer to N(0, 1/d)). The paper's Table 1 has the canonical centroids; verify them with a numerical Lloyd-Max iteration on actual K-cache samples.

3. **Encode**: use existing 3-bit + simplified-QJL CPU/GPU code paths, but now operating on D-rotated vectors.

4. **Decode**: use existing per-element-sign QJL term `sign(r_i) * rnorm / sqrt(d)`.

**Phase 2 checkpoints:**

- ✅ Synthetic harness: GPU 3-bit FA matches CPU 3-bit FA to <1e-5 (we already proved this in the prior session post-`5e2a3175`).
- ✅ Per-block MSE: 3-bit centroid + sign-residual should give *similar* MSE to 4-bit-no-residual on D-rotated vectors. If 3-bit MSE is dramatically worse, the centroid table is wrong — go back and re-derive.
- ✅ Perplexity from Phase 0: 3-bit + simplified QJL should be **within 1-2 PP** of the 4-bit baseline. We expect a small regression here because simplified QJL with R=I is a known-weak estimator; the gap closes in Phase 3.
- ✅ Smoke test: `What comes after three?` → still says "four" (not "three? three? three?" as it did before D-rotation existed).

If perplexity is *worse* than 4-bit by more than 3 PP at this stage, the centroid table or the encode is wrong — investigate before proceeding.

---

## Phase 3: Random Gaussian R and the JL inner-product estimator

This is the highest-risk phase because it changes the FA dot-product semantics, not just the bit layout.

**Design choices:**

- **Granularity of R.** Per-tensor (paper's choice) — `R ∈ R^{d×m}` shared across all blocks of one quantized tensor. For `d=128, m=128`, that's 128×128×4 = 64KB of random floats per quantized tensor, easily fits in `__constant__` memory or a small device global.
- **m (number of JL projections).** Paper's headline is m = d. For our 1-bit-per-projection budget that's fixed at 128 — same as the existing `signs[16]` (128 bits) field. No layout change vs Phase 2.
- **Generation.** Box-Muller from a deterministic seed, normalized so each column has unit norm (this isn't strictly required by the JL bound but reduces variance). Generate once at tensor allocation; freeze for the cache's lifetime.
- **Numerical care.** R is `float`, not fp16. The JL estimator is a sum of 128 small contributions; if you accumulate in fp16 you lose 1-2 bits of mantissa per add. Accumulate in fp32 inside the kernel.

**Implementation steps:**

1. **CPU encode**: replace `signs[i] = sign(r[i])` with `signs[j] = sign((R^T r)_j)` for j=0..m-1. This is a 128×128 mat-vec inside the encode loop — cheap on CPU (~16K mults per block). Use BLAS if available.

2. **CPU dequant + FA**:
   - The JL estimator changes the *form* of the dot product, not the dequant result. There is no scalar `r̂[i]` to write back. So `dequantize_row_turbo4_0` becomes "dequantize centroids only" (`y[i] = C[idx]*norm`), and the JL contribution is added inside the FA loop.
   - In CPU FA: after computing the centroid dot `<C[idx]*norm, Q>`, add `sqrt(2/π) * rnorm * (1/m) * sum_j signs[j] * (R^T Q)_j`.
   - **Precompute `R^T Q` once per query** outside the K loop. This is a 128×128 mat-vec per query head — `O(d^2)` instead of `O(d^2 * KV)`. Critical for FA performance.

3. **GPU encode** (`set-rows.cu`):
   - Add a 128×128 mat-vec inside the encode kernel. Each thread block does one block (= 128 elements). Each thread takes a stripe of m. Use shared memory to broadcast `r` to all threads computing the projections.
   - Sanity: cycle count per block goes from ~200 (current) to ~3000 (mat-vec dominates). Acceptable; encode is one-shot per token, not the bottleneck.

4. **GPU FA — the trickiest piece** (`fattn-common.cuh`):
   - **Pre-rotate Q**: add a kernel pass before FA that computes `Q_jl = R^T Q` for each query head. Output buffer is the same shape as Q. This is `O(NQ * H * d^2)` per generation step — a ~200μs overhead at d=128, dwarfed by FA itself. Or fold it into the existing Q rotation pass.
   - **Inside FA**: when computing K·Q for a turbo4 block, you now need *both* `Q_rotated_by_D_and_WHT` (for the centroid term) and `Q_jl = R^T Q_rotated` (for the JL term). Either pass both into the kernel as separate tensors, or store them stacked.
   - **The dot product changes from**:
     ```cpp
     sum += C[idx[i]] * norm * Q[i];
     ```
     **to**:
     ```cpp
     // centroid term (existing)
     sum_centroid += C[idx[i]] * norm * Q[i];
     // JL term (new, accumulates over j, not i)
     // signs[j] is the j-th JL projection sign for THIS K block
     // Q_jl[j] is the j-th component of R^T Q (precomputed)
     sum_jl += signs[j] * Q_jl[j];   // sign * float, runs over j=0..m-1
     // final
     KQ_dot = sum_centroid + sqrt_2_over_pi * rnorm * (1.0f/m) * sum_jl;
     ```
   - **Memory access pattern:** `signs[16]` is 16 bytes per block, the same as today. `Q_jl[m]` is `m*4 = 512 bytes` per query, loaded once per FA call from registers/shared memory. No new global-memory pressure on K loads.
   - **Constant memory pressure:** R is 64KB per tensor; CUDA `__constant__` memory is 64KB total. This means R *cannot* live in `__constant__` if we have multiple turbo tensors active. Put R in global memory and let it cache through `__ldg` / texture cache. Verify with `nvprof` that R reads aren't a bottleneck.

5. **Sign of `(R^T Q)`** is what gets multiplied into the sum, so the final magnitude estimate uses sign(signs)*float — single-instruction inner products via `__byte_perm` or `__vsignedu` are not directly applicable here because we have 1-bit signs vs float values, but a 128-element loop unrolls cleanly and the scheduler can hide latency.

**Phase 3 checkpoints (CRITICAL — do all):**

- ✅ **Mathematical sanity test** (CPU only, in isolation, no FA):
  - Generate 1000 random pairs of vectors `(x, y)` from N(0, 1/d).
  - Compute true `<x, y>`.
  - Quantize x with full TurboQuant, then estimate `<x, y>` using the JL estimator with the *un-quantized* y.
  - Assert: estimator is unbiased (mean error ~0) and variance matches the paper's bound `<= 2 * |x|^2 * |y|^2 / m`.
  - If this test fails, the JL estimator math is wrong; do not proceed.

- ✅ **CPU vs GPU determinism**: same x, same R, same y → CPU and GPU JL estimators agree to <1e-4. *This is the single highest-yield test in the whole plan*; almost every implementation bug shows up here first.

- ✅ **Per-block MSE on real K cache**: 3-bit + full QJL should beat 3-bit + simplified QJL by 30-50% on dot-product MSE (this is the paper's central claim and what justifies the whole exercise).

- ✅ **Perplexity from Phase 0**: 3-bit + full QJL should be **within 0.5 PP** of f16 on the held-out corpus. If it's not, either:
  - R is not properly normalized, or
  - The estimator scaling (`sqrt(2/π) * rnorm / m`) is off by a constant, or
  - Q's rotation by D and WHT is inconsistent between encode and decode.

  Go back to the mathematical sanity test and add intermediate prints to find which factor is wrong.

- ✅ **Smoke tests**: `Paris.` for "The capital of France is", `four` for "What comes after three?".

---

## Phase 4: Performance and integration

Once correctness is locked in, optimize.

1. **Profile the FA kernel** with `ncu` (NSight Compute). Likely findings:
   - `R^T Q` precomputation takes <2% of step time → leave alone.
   - JL accumulation inside FA takes 10-15% extra over baseline → acceptable; this *is* the paper's design.
   - K-block load is fine (no layout change).

2. **Bank conflicts on `signs[16]` reads.** Verify shared-memory layout for the JL inner sum.

3. **Use Tensor Cores for `R^T Q`**? If `m` is large enough, `R^T Q` is a tiny GEMM. For m=128, d=128, NQ=1, it's 128 FMA per query — too small for tensor cores to amortize the setup. Stick with hand-rolled mat-vec.

4. **End-to-end benchmark vs Phase 0 baseline**: gen tokens/sec should drop by no more than 15% vs current 4-bit nibble. If it's worse, the JL term dominates more than expected — check that Q's R-rotation isn't being re-run per K block.

5. **Long-context regression test**: re-run the 65k-context benchmark from `benchmark-context-20260429-023500.md`. The full TurboQuant should match f16 at long contexts and *outperform* it where f16 drifts (because TurboQuant works in fp32 inside the JL accumulator).

---

## Phase 5: Cleanup, documentation, and shipping

1. **Remove the `TURBO4_USE_4BIT` switch** if Phase 3 perplexity validates. Default everything to the full TurboQuant path. Or keep the switch one release cycle for safety.

2. **Document R generation, D generation, and seeding** in `TURBOQUANT-DEBUG.md`. Future engineers must be able to reproduce a cache deterministically.

3. **Final commit message must cite the perplexity numbers** from Phase 0 and the paper's reported results, side-by-side.

4. **Update the existing `benchmark-context-*.md`** with full-TurboQuant numbers across context lengths.

---

## Risk register and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Encode/decode contract drift between CPU and GPU | **High** | Phase 1.3 byte-equal test runs after every commit. CI should fail if it ever regresses. |
| R not propagated correctly to FA kernel | **High** | Phase 3 CPU-vs-GPU determinism test catches this. Plus print R's checksum in both encode and FA logs at debug level. |
| Off-by-constant in JL estimator scaling (`sqrt(2/π)`, `1/m`, `rnorm`) | **High** | Phase 3 mathematical sanity test (unbiased estimator on Gaussian data) is exactly designed to catch this. |
| `R` lives in `__constant__` and runs out at 64KB → silent performance cliff | Medium | Put R in global memory from the start, `__ldg` for cache. |
| Q's D-rotation done twice (once for centroid term, once for JL term) | Medium | Single Q rotation pass; pass both forms into FA as separate tensors. Add an assertion in the kernel checking `Q_rotated.dim == Q_jl.dim` matches expectation. |
| Perplexity rig too slow, gets skipped | Low | Phase 0 gates Phase 1; non-negotiable. Budget 2 days for it. |
| 3-bit Lloyd-Max centroids different from paper's table → quality regression | Medium | Compute them numerically from real K-cache distribution; print and check against paper's Table 1. |
| Long-context regression: WHT/Rademacher rotation drifts because of fp16 accumulator on GPU | Medium | Use fp32 inside WHT kernel (existing code already does); add a long-context test as a CI gate after Phase 3. |
| Storage/lifetime of D and R via `tensor::extra` is fragile | Medium | Wrap in a small `TurboQuantContext` struct keyed by tensor pointer in a global map. Cache eviction is owned by ggml's tensor lifetime. |

---

## Estimated timeline (1 senior engineer)

| Phase | Days | Notes |
|---|---|---|
| 0. Eval infrastructure | 2 | One-time investment. Saves days later. |
| 1. Hadamard–Rademacher rotation | 2 | Mostly mechanical; D propagation is the gotcha. |
| 2. 3-bit layout restore | 1 | Reactivating audited code with new D-rotation. |
| 3. Random R + JL estimator | 4 | The hard one. Budget extra for math debugging. |
| 4. Performance | 2 | Profile, tune, retest. |
| 5. Cleanup and docs | 1 | |
| **Total** | **12 days** | |

Add 30-50% buffer for surprises: realistic estimate **3 weeks of focused work**.

---

## What to commit at each phase

Each phase produces **one commit**, with the commit message explicitly citing:
1. The synthetic test results (cosine, MSE).
2. The perplexity number from Phase 0.
3. The paper's claimed result for comparison.

This makes regression bisection trivial. If Phase 4 introduces a bug, `git bisect` on perplexity gets you to the offending change in a few iterations.

---

## Stop conditions

If at the end of Phase 3 the perplexity gap to f16 is *not* under 1 PP, the implementation has a subtle bug; **do not ship**. The most likely culprits are (in order):
1. R generation seed differs between encode and FA (deterministic-but-different).
2. JL estimator scaling constant is off.
3. D applied with wrong sign convention on one side.
4. Centroid table is for the wrong distribution.

Each of these is catchable by the Phase 3 mathematical sanity test. If all four are clean and perplexity is still bad, the issue is upstream — probably in how Q is rotated through the attention layer norm or in how the inverse rotation interacts with attention bias. At that point, treat it as a research problem and bring in a second engineer.
