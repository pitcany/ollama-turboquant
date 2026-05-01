#!/usr/bin/env python3
"""Lloyd-Max centroid optimiser for the Turbo* PolarQuant family.

The Turbo* CPU/CUDA kernels operate on per-block normalised, WHT-rotated
vectors of length 128. Under the WHT-rotation prior, each rotated coordinate is
approximately N(0, 1/128). We optimise n_centroids quantisation levels for that
prior using Lloyd's algorithm (PCM scalar quantiser), then emit the centroid
list and the n_centroids-1 midpoints used by the C `nearest_centroid_Nbit`
binary search. Run with `--check` to verify the existing 2/3/4-bit tables are
reproduced bit-for-bit (within 1e-4)."""

import argparse, math, sys
import numpy as np

SIGMA = 1.0 / math.sqrt(128.0)

def lloyd_max(n_centroids, samples, n_iters=200):
    qs = (np.arange(n_centroids) + 0.5) / n_centroids
    centroids = np.quantile(samples, qs)
    for _ in range(n_iters):
        mids = (centroids[:-1] + centroids[1:]) / 2.0
        idx = np.searchsorted(mids, samples)
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
