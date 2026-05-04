package ggml

import (
	"math"
	"math/rand"
	"testing"
)

// TestTurbo4_0_64RoundTrip exercises the PR-1 CPU reference path. It quantizes
// a 1024-element vector through both turbo4_0 (head_dim=128) and turbo4_0_64
// (head_dim=64), then compares the per-element MSE in the rotated domain.
//
// Both paths leave dequant in the rotated domain (the graph applies
// GGML_OP_TURBO_WHT to invert), so we compare against the WHT-rotated input.
func TestTurbo4_0_64RoundTrip(t *testing.T) {
	const k = 1024
	if k%128 != 0 || k%64 != 0 {
		t.Fatalf("k=%d must be divisible by both 64 and 128", k)
	}

	rng := rand.New(rand.NewSource(0x54225542))
	x := make([]float32, k)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}

	xRot128 := normalizedRotated(x, 128)
	xRot64 := normalizedRotated(x, 64)

	out128 := quantizeTurbo4_0RoundTrip(x)
	out64 := quantizeTurbo4_0_64RoundTrip(x)

	mse128 := mseRotated(xRot128, out128)
	mse64 := mseRotated(xRot64, out64)

	t.Logf("turbo4_0 MSE (rotated domain) = %.6e", mse128)
	t.Logf("turbo4_0_64 MSE (rotated domain) = %.6e", mse64)
	t.Logf("turbo4_0_64 / turbo4_0 ratio   = %.3f", mse64/mse128)

	// Block size halves but bits/value stays at 4.25, so the ideal
	// rotated-domain MSE is the same. The reference reuses centroids tuned
	// for N(0, 1/128) against d=64-rotated values whose components are
	// ~N(0, 1/64) — the outer centroid covers only ~1.4σ instead of 2σ, so
	// outer-bin clipping inflates MSE empirically by ~2.6x. Centroids
	// retuned for N(0, 1/64) are deferred to a follow-up PR; the cap here
	// just guards against catastrophic regressions while leaving headroom
	// for the known suboptimality.
	const ratioCeiling = 4.0
	if mse64 > ratioCeiling*mse128 {
		t.Fatalf("turbo4_0_64 MSE %.6e exceeds %.1fx of turbo4_0 MSE %.6e",
			mse64, ratioCeiling, mse128)
	}

	maxAbs := float32(0)
	for _, v := range out64 {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("turbo4_0_64 produced non-finite output: %v", v)
		}
		if av := absF32(v); av > maxAbs {
			maxAbs = av
		}
	}
	if maxAbs == 0 {
		t.Fatalf("turbo4_0_64 produced all-zero output")
	}
}

// TestTurbo4_0_64BlockSize verifies the on-disk layout: 2-byte norm + 32-byte
// nibble-packed indices = 34 bytes per 64 values (4.25 bpv).
func TestTurbo4_0_64BlockSize(t *testing.T) {
	const want uintptr = 34
	if got := turbo4_0_64BlockSize(); got != want {
		t.Fatalf("sizeof(block_turbo4_0_64) = %d, want %d", got, want)
	}
}

// TestTurbo4_0_64TypeTraits verifies the type registers with blck_size=64 and
// type_size=34 so PR-5's runtime resolver can rely on it.
func TestTurbo4_0_64TypeTraits(t *testing.T) {
	blk, sz := turbo4_0_64TypeTraits()
	if blk != 64 {
		t.Fatalf("ggml_blck_size(turbo4_0_64) = %d, want 64", blk)
	}
	if sz != 34 {
		t.Fatalf("ggml_type_size(turbo4_0_64) = %d, want 34", sz)
	}
}

// normalizedRotated mirrors the per-block normalize+WHT step that the
// quantize path applies internally, then rescales by the original norm so
// the result is comparable with dequant output (which scales by the
// corrected norm).
func normalizedRotated(x []float32, groupSize int) []float32 {
	out := make([]float32, len(x))
	copy(out, x)
	for off := 0; off < len(out); off += groupSize {
		var ns float64
		for j := 0; j < groupSize; j++ {
			ns += float64(out[off+j]) * float64(out[off+j])
		}
		norm := math.Sqrt(ns)
		if norm > 1e-10 {
			inv := 1.0 / norm
			for j := 0; j < groupSize; j++ {
				out[off+j] = float32(float64(out[off+j]) * inv)
			}
		}
		turboCPUFWHT(out[off:off+groupSize], groupSize)
		for j := 0; j < groupSize; j++ {
			out[off+j] = float32(float64(out[off+j]) * norm)
		}
	}
	return out
}

func mseRotated(ref, got []float32) float64 {
	if len(ref) != len(got) {
		return math.NaN()
	}
	var s float64
	for i := range ref {
		d := float64(ref[i]) - float64(got[i])
		s += d * d
	}
	return s / float64(len(ref))
}

func absF32(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}
