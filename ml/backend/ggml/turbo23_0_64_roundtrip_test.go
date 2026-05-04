package ggml

import (
	"math"
	"math/rand"
	"testing"
)

// TestTurbo2_0_64RoundTrip exercises the PR-4 CPU reference path for the
// 2-bit head_dim=64 variant. Quantizes a 1024-element vector through both
// turbo2_0 (head_dim=128) and turbo2_0_64 (head_dim=64), then compares
// the per-element MSE in the rotated domain. Cap is a 4x ratio against
// the head_dim=128 baseline -- the centroids reused here are tuned for
// N(0, 1/128) and clip outer bins more aggressively on d=64-rotated
// values whose components are ~N(0, 1/64). Retuned 2-bit centroids can
// land later if calibration shows a real win.
func TestTurbo2_0_64RoundTrip(t *testing.T) {
	const k = 1024
	rng := rand.New(rand.NewSource(0x54225542))
	x := make([]float32, k)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}

	xRot128 := normalizedRotated(x, 128)
	xRot64 := normalizedRotated(x, 64)
	out128 := quantizeTurbo2_0RoundTrip(x)
	out64 := quantizeTurbo2_0_64RoundTrip(x)

	mse128 := mseRotated(xRot128, out128)
	mse64 := mseRotated(xRot64, out64)
	t.Logf("turbo2_0    MSE (rotated domain) = %.6e", mse128)
	t.Logf("turbo2_0_64 MSE (rotated domain) = %.6e", mse64)
	t.Logf("turbo2_0_64 / turbo2_0 ratio    = %.3f", mse64/mse128)

	const ratioCeiling = 4.0
	if mse64 > ratioCeiling*mse128 {
		t.Fatalf("turbo2_0_64 MSE %.6e exceeds %.1fx of turbo2_0 MSE %.6e",
			mse64, ratioCeiling, mse128)
	}

	maxAbs := float32(0)
	for _, v := range out64 {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("turbo2_0_64 produced non-finite output: %v", v)
		}
		if av := absF32(v); av > maxAbs {
			maxAbs = av
		}
	}
	if maxAbs == 0 {
		t.Fatalf("turbo2_0_64 produced all-zero output")
	}
}

// TestTurbo3_0_64RoundTrip mirrors the 2-bit case for the 3-bit variant.
// Same rationale for the 4x ratio cap (centroids tuned for N(0, 1/128)).
func TestTurbo3_0_64RoundTrip(t *testing.T) {
	const k = 1024
	rng := rand.New(rand.NewSource(0x54225542))
	x := make([]float32, k)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}

	xRot128 := normalizedRotated(x, 128)
	xRot64 := normalizedRotated(x, 64)
	out128 := quantizeTurbo3_0RoundTrip(x)
	out64 := quantizeTurbo3_0_64RoundTrip(x)

	mse128 := mseRotated(xRot128, out128)
	mse64 := mseRotated(xRot64, out64)
	t.Logf("turbo3_0    MSE (rotated domain) = %.6e", mse128)
	t.Logf("turbo3_0_64 MSE (rotated domain) = %.6e", mse64)
	t.Logf("turbo3_0_64 / turbo3_0 ratio    = %.3f", mse64/mse128)

	const ratioCeiling = 4.0
	if mse64 > ratioCeiling*mse128 {
		t.Fatalf("turbo3_0_64 MSE %.6e exceeds %.1fx of turbo3_0 MSE %.6e",
			mse64, ratioCeiling, mse128)
	}

	maxAbs := float32(0)
	for _, v := range out64 {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("turbo3_0_64 produced non-finite output: %v", v)
		}
		if av := absF32(v); av > maxAbs {
			maxAbs = av
		}
	}
	if maxAbs == 0 {
		t.Fatalf("turbo3_0_64 produced all-zero output")
	}
}

// TestTurbo2_0_64BlockSize verifies layout: 2-byte norm + 16-byte qs = 18 bytes per 64 vals.
func TestTurbo2_0_64BlockSize(t *testing.T) {
	const want uintptr = 18
	if got := turbo2_0_64BlockSize(); got != want {
		t.Fatalf("sizeof(block_turbo2_0_64) = %d, want %d", got, want)
	}
}

// TestTurbo3_0_64BlockSize verifies layout: 2-byte norm + 16-byte qs + 8-byte signs = 26 bytes per 64 vals.
func TestTurbo3_0_64BlockSize(t *testing.T) {
	const want uintptr = 26
	if got := turbo3_0_64BlockSize(); got != want {
		t.Fatalf("sizeof(block_turbo3_0_64) = %d, want %d", got, want)
	}
}

func TestTurbo2_0_64TypeTraits(t *testing.T) {
	blk, sz := turbo2_0_64TypeTraits()
	if blk != 64 {
		t.Fatalf("ggml_blck_size(turbo2_0_64) = %d, want 64", blk)
	}
	if sz != 18 {
		t.Fatalf("ggml_type_size(turbo2_0_64) = %d, want 18", sz)
	}
}

func TestTurbo3_0_64TypeTraits(t *testing.T) {
	blk, sz := turbo3_0_64TypeTraits()
	if blk != 64 {
		t.Fatalf("ggml_blck_size(turbo3_0_64) = %d, want 64", blk)
	}
	if sz != 26 {
		t.Fatalf("ggml_type_size(turbo3_0_64) = %d, want 26", sz)
	}
}
