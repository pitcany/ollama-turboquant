package eval

import (
	"math"
	"testing"
)

func TestNegativeLogLikelihoodUsesStableSoftmax(t *testing.T) {
	nll, err := NegativeLogLikelihood([]float32{1000, 1000}, 1)
	if err != nil {
		t.Fatalf("NegativeLogLikelihood() error = %v", err)
	}

	want := math.Log(2)
	if math.Abs(nll-want) > 1e-6 {
		t.Fatalf("NLL = %.12f, want %.12f", nll, want)
	}
}

func TestNegativeLogLikelihoodRejectsBadTarget(t *testing.T) {
	_, err := NegativeLogLikelihood([]float32{0, 1}, 2)
	if err == nil {
		t.Fatal("NegativeLogLikelihood() error = nil, want target bounds error")
	}
}

func TestKLDivergenceIsZeroForIdenticalLogits(t *testing.T) {
	kl, err := KLDivergence([]float32{-1, 0, 2}, []float32{-1, 0, 2})
	if err != nil {
		t.Fatalf("KLDivergence() error = %v", err)
	}
	if math.Abs(kl) > 1e-12 {
		t.Fatalf("KL = %.12f, want 0", kl)
	}
}

func TestAccumulatorComputesPerplexity(t *testing.T) {
	var acc Accumulator
	if err := acc.AddNLL(math.Log(2)); err != nil {
		t.Fatalf("AddNLL() error = %v", err)
	}
	if err := acc.AddNLLSum(math.Log(8), 1); err != nil {
		t.Fatalf("AddNLLSum() error = %v", err)
	}
	if err := acc.AddKLSum(0.125, 2); err != nil {
		t.Fatalf("AddKLSum() error = %v", err)
	}

	metrics, err := acc.Metrics()
	if err != nil {
		t.Fatalf("Metrics() error = %v", err)
	}
	if metrics.TokenCount != 2 {
		t.Fatalf("TokenCount = %d, want 2", metrics.TokenCount)
	}
	if math.Abs(metrics.Perplexity-4) > 1e-9 {
		t.Fatalf("Perplexity = %.12f, want 4", metrics.Perplexity)
	}
	if metrics.KLTokenCount != 2 || math.Abs(metrics.MeanKL-0.0625) > 1e-12 {
		t.Fatalf("KL metrics = (%d, %.12f), want (2, 0.0625)", metrics.KLTokenCount, metrics.MeanKL)
	}
}
