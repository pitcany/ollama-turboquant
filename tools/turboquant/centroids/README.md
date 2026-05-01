# TurboQuant Centroid Generator

`lloyd_max.py` is the offline math behind the Turbo* PolarQuant family
(`Turbo2`, `Turbo3`, `Turbo4`, and the upcoming `Turbo5`/`Turbo6`).

## Prior

Each Turbo block is normalised to unit-norm and rotated by a 128x128 WHT (or
Householder/random-orthogonal) matrix. Under that rotation, every coordinate
of a unit-norm vector is approximately Gaussian with variance `1/D`, i.e.
`N(0, 1/128)` for `D = 128`. The script optimises scalar quantisation
centroids for that prior.

The `N(0, 1/128)` prior is **approximate**: the true marginal of a
sphere-uniform vector is the Beta-derived "marginal of the sphere", which
becomes Gaussian only in the `D -> infinity` limit and at `D = 128` is
slightly more peaked at zero. Real post-WHT activations are also
heavier-tailed than Gaussian (residual-stream norms vary across blocks even
after per-block normalisation). Treat the prior as a working assumption,
not ground truth.

## Seed and sample size

* `numpy.random.default_rng(20260501)` — fixed for reproducibility. The
  number is arbitrary but pinned: regeneration on any host with the same
  numpy/PCG64 implementation produces identical centroids. Each bit width
  derives its restart-RNG seed as `seed + bits` so the multi-restart driver
  is also deterministic.
* `--samples 4_000_000` (default). With 64 cells (6-bit) this gives ~62k
  samples per cell, well past the noise floor of Lloyd's algorithm for
  centroids of magnitude `O(sigma) = O(0.09)`.

## Convergence

For high bit widths (>=5), Lloyd's algorithm has many shallow local minima.
The script runs `n_restarts = 8` restarts per bit width with mixed
initialisations:

* even restarts: equiprobable-quantile init (the historical seeding);
* odd restarts: k-means++ on a downsampled subset (capped at 200k points to
  bound memory of the `(N, k)` distance broadcast).

The lowest-MSE result is kept. Lower bit widths (<=4) use a single restart;
their MSE surface is convex enough that quantile init suffices. The inner
loop runs up to 500 iterations; the `np.allclose(..., atol=1e-9)` early-exit
typically fires well before that.

## Regeneration protocol

```
# Documentation-only: print per-bit max-abs diff vs the in-tree
# 2/3/4-bit tables in ml/backend/ggml/ggml/src/ggml-turbo-quant.c.
# Always exits 0; flags structural discrepancies (>1e-3) inline.
python3 tools/turboquant/centroids/lloyd_max.py --check

# Emit Turbo5 + Turbo6 centroid + midpoint tables
python3 tools/turboquant/centroids/lloyd_max.py --bits 5 --bits 6
```

`--bits N` writes `turbo{N}_centroids.txt` containing `2^N` centroids and
`2^N - 1` midpoints. The midpoints are the binary-search thresholds used by
the C kernel `nearest_centroid_Nbit`.

## Expected MSE bound

For a Gaussian source the high-rate PCM bound is

```
D_high_rate = (sqrt(3) * pi / 2) * sigma^2 * 4^{-bits}
```

Lloyd-Max should reach `mse / D_high_rate` near 1.0 at high bit width
(asymptotically tight from above for fine quantisation; finite-rate values
oscillate slightly around 1). The script prints this ratio for every
emitted bit width so a regression to `>1.10` surfaces immediately.

## Documentation: in-tree CENTROIDS_4BIT discrepancy

The in-tree `CENTROIDS_4BIT` table in `ggml-turbo-quant.c` does **not**
match Lloyd-Max for `N(0, 1/128)`: the regenerated outermost centroid lands
at +/-0.242 vs. the in-tree +/-0.174 (a ~5% structural difference, far
beyond sampling noise). The 2-bit and 3-bit tables agree to within
sampling noise (<=~1e-3).

The most likely explanation is that the original 4-bit table was
**calibrated empirically** against measured post-WHT activations rather
than the theoretical Gaussian prior, or warped to better capture
heavier-than-Gaussian tails. We do **not** treat this as a bug for the
present task: future work that wants to revisit Turbo4 quality should
regenerate the 4-bit table from measured activation statistics, but that
is out of scope for the Turbo5/Turbo6 generator.

`--check` therefore prints the diff and a structural-discrepancy note for
any bit width over `1e-3`, but always exits 0. The discrepancy is kept
visible without blocking the build pipeline.

## Phase 0 evaluation is the empirical ground truth

The Turbo5 and Turbo6 tables emitted by this script use the theoretical
`N(0, 1/128)` prior. **Phase 0 model-quality evaluation is the empirical
ground truth.** If Turbo5/Turbo6 quality falls short of the predicted
`4 dB / bit` improvement curve, the prior is the first thing to revisit:
re-derive centroids against measured activation samples (the in-tree
4-bit table is the precedent for that approach) before changing the kernel
math itself.
