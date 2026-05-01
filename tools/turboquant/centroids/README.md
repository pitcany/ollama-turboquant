# TurboQuant Centroid Generator

`lloyd_max.py` is the offline math behind the Turbo* PolarQuant family
(`Turbo2`, `Turbo3`, `Turbo4`, and the upcoming `Turbo5`/`Turbo6`).

## Prior

Each Turbo block is normalised to unit-norm and rotated by a 128x128 WHT (or
Householder/random-orthogonal) matrix. Under that rotation, every coordinate
of a unit-norm vector is approximately Gaussian with variance `1/D`, i.e.
`N(0, 1/128)` for `D = 128`. The script optimises scalar quantisation
centroids for that prior.

## Seed and sample size

* `numpy.random.default_rng(20260501)` — fixed for reproducibility. The
  number is arbitrary but pinned: regeneration on any host with the same
  numpy/PCG64 implementation produces identical centroids.
* `--samples 4_000_000` (default). With 16 cells (4-bit) this gives
  ~250k samples per cell, well past the noise floor of Lloyd's algorithm
  for centroids of magnitude `O(sigma) = O(0.09)`.

## Regeneration protocol

```
# Verify the in-tree 2/3/4-bit tables in
# ml/backend/ggml/ggml/src/ggml-turbo-quant.c reproduce to <=1e-4
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
emitted bit width so a regression to `>1.20` surfaces immediately.

## Known anomaly (2026-05-01)

The in-tree `CENTROIDS_4BIT` table in `ggml-turbo-quant.c` is **not** a
Lloyd-Max fixed point for `N(0, 1/128)`. The conditional mean of the outer
cell disagrees with the table by ~5%, and our regenerated 16-level Lloyd
table places the outermost centroid at +/-0.242 vs. the in-tree +/-0.174.
The 2-bit and 3-bit tables agree to within sampling noise (<=1e-3).

This is the trigger Phase A was designed to find: before Phase B emits new
Turbo5/Turbo6 kernels, the upstream 4-bit centroid math needs to be
reconciled (regenerate from this script, or document the bespoke scheme
the original 4-bit table came from).
