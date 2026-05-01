#!/usr/bin/env python3
"""Lloyd-Max centroid optimiser for the Turbo* PolarQuant family.

The Turbo* CPU/CUDA kernels operate on per-block normalised, WHT-rotated
vectors of length 128. Under the WHT-rotation prior, each rotated coordinate is
approximately N(0, 1/128). We optimise n_centroids quantisation levels for
that prior using Lloyd's algorithm (PCM scalar quantiser), then emit the
centroid list and the n_centroids-1 midpoints used by the C
`nearest_centroid_Nbit` binary search.

`--check` is documentation-only: it always exits 0 but prints the per-bit
max-abs diff vs the in-tree tables and a structural-discrepancy note for any
bit width where the diff exceeds 1e-3. The in-tree CENTROIDS_4BIT table is
known to differ structurally from Lloyd-Max(N(0, 1/128)) (see README); the
script keeps the diff visible without blocking on it.

For higher bit widths Lloyd's algorithm is run with multiple random restarts
(quantile + k-means++ seedings) and the best-MSE result is kept; this avoids
the local minima the single-shot version hit at 6 bits.
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import Tuple

import numpy as np

SIGMA = 1.0 / math.sqrt(128.0)
DEFAULT_N_ITERS = 500
RESTART_BIT_THRESHOLD = 5  # bits >= this get multiple restarts
DEFAULT_RESTARTS_HIGH = 8
DEFAULT_RESTARTS_LOW = 1
KMEANS_PP_SAMPLE_CAP = 50_000
STRUCTURAL_DIFF_THRESHOLD = 1e-3


def _kmeans_pp_init(samples: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Probabilistic farthest-point seeding for k centers from `samples`.

    Standard k-means++: pick the first center uniformly at random, then each
    subsequent center is sampled with probability proportional to squared
    distance from the nearest already-chosen center. Returned centers are
    sorted ascending so they can drop into Lloyd's binary-search update.
    """
    n = samples.size
    first = int(rng.integers(0, n))
    centers = [float(samples[first])]
    for _ in range(k - 1):
        centers_arr = np.asarray(centers)
        # samples shape (n,), centers shape (m,) -> broadcast to (n, m)
        d2 = np.min((samples[:, None] - centers_arr[None, :]) ** 2, axis=1)
        total = d2.sum()
        if total <= 0.0:
            # Degenerate (all samples coincide with a center); fall back to uniform.
            idx = int(rng.integers(0, n))
        else:
            probs = d2 / total
            idx = int(rng.choice(n, p=probs))
        centers.append(float(samples[idx]))
    return np.sort(np.asarray(centers, dtype=np.float64))


def _lloyd_max_inner(
    n_centroids: int,
    samples: np.ndarray,
    init: np.ndarray,
    n_iters: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Run Lloyd's algorithm to convergence (or `n_iters`) from a given init.

    Uses np.bincount for the cell-conditional means: a single O(N) reduction
    per iteration instead of k mask passes, which matters at k=64.
    """
    centroids = init.astype(np.float64).copy()
    for _ in range(n_iters):
        mids = (centroids[:-1] + centroids[1:]) / 2.0
        idx = np.searchsorted(mids, samples)
        sums = np.bincount(idx, weights=samples, minlength=n_centroids)
        counts = np.bincount(idx, minlength=n_centroids)
        new = centroids.copy()
        nonempty = counts > 0
        new[nonempty] = sums[nonempty] / counts[nonempty]
        if np.allclose(new, centroids, atol=1e-9):
            centroids = new
            break
        centroids = new
    mids = (centroids[:-1] + centroids[1:]) / 2.0
    mse = float(((samples - centroids[np.searchsorted(mids, samples)]) ** 2).mean())
    return centroids, mids, mse


def lloyd_max_restart(
    n_centroids: int,
    samples: np.ndarray,
    n_iters: int = DEFAULT_N_ITERS,
    n_restarts: int = 1,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Best-of-`n_restarts` Lloyd-Max from mixed quantile / k-means++ seedings.

    Even restarts use deterministic equiprobable-quantile init (the historical
    seeding); odd restarts use k-means++ on a downsampled subset (capped at
    200k points for memory) to inject diversity. Returns the centroid set
    with lowest empirical MSE on `samples`.
    """
    rng = np.random.default_rng(seed)
    best: Tuple[np.ndarray, np.ndarray, float] | None = None
    pp_pool = samples if samples.size <= KMEANS_PP_SAMPLE_CAP else samples[:KMEANS_PP_SAMPLE_CAP]
    for r in range(n_restarts):
        if r % 2 == 0:
            qs = (np.arange(n_centroids) + 0.5) / n_centroids
            init = np.quantile(samples, qs)
        else:
            init = _kmeans_pp_init(pp_pool, n_centroids, rng)
        c, m, mse = _lloyd_max_inner(n_centroids, samples, init, n_iters)
        if best is None or mse < best[2]:
            best = (c, m, mse)
    assert best is not None
    return best


def _restart_count(bits: int) -> int:
    return DEFAULT_RESTARTS_HIGH if bits >= RESTART_BIT_THRESHOLD else DEFAULT_RESTARTS_LOW


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--bits",
        type=int,
        action="append",
        default=[],
        help="bit widths to emit (e.g. --bits 5 --bits 6)",
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help=(
            "documentation-only: prints per-bit max-abs diff vs the in-tree "
            "tables; flags structural discrepancies but always exits 0"
        ),
    )
    ap.add_argument("--samples", type=int, default=4_000_000)
    ap.add_argument("--seed", type=int, default=20260501)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
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
        for bits, exp in EXPECTED.items():
            got, _, _ = lloyd_max_restart(
                2 ** bits,
                samples,
                n_restarts=_restart_count(bits),
                seed=args.seed + bits,
            )
            err = float(np.max(np.abs(got - np.array(exp))))
            print(f"bits={bits}: max abs diff vs in-tree table = {err:.2e}")
            if err > STRUCTURAL_DIFF_THRESHOLD:
                print(
                    f"  NOTE: bits={bits} diff > {STRUCTURAL_DIFF_THRESHOLD:.0e} indicates a "
                    f"structural discrepancy; the in-tree table is not a Lloyd-Max "
                    f"fixed point for N(0, 1/128). See "
                    f"tools/turboquant/centroids/README.md for context."
                )
        # Documentation-only: never block on diff.
        sys.exit(0)

    for b in args.bits:
        c, m, mse = lloyd_max_restart(
            2 ** b,
            samples,
            n_restarts=_restart_count(b),
            seed=args.seed + b,
        )
        bound = (math.sqrt(3) * math.pi / 2.0) * SIGMA * SIGMA * (4.0 ** -b)
        print(
            f"bits={b}: mse={mse:.3e}, theoretical_bound={bound:.3e}, "
            f"ratio={mse / bound:.3f}"
        )
        with open(f"tools/turboquant/centroids/turbo{b}_centroids.txt", "w") as f:
            f.write(
                f"# {2 ** b} Lloyd-Max centroids for N(0, 1/128), seed={args.seed}\n"
            )
            f.write("# centroids\n")
            for v in c:
                f.write(f"{v:+.6f}\n")
            f.write("# midpoints\n")
            for v in m:
                f.write(f"{v:+.6f}\n")


if __name__ == "__main__":
    main()
