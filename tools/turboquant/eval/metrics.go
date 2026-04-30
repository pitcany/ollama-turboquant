package eval

import (
	"fmt"
	"math"
)

type Metrics struct {
	TokenCount   int     `json:"token_count"`
	MeanNLL      float64 `json:"mean_nll"`
	Perplexity   float64 `json:"perplexity"`
	KLTokenCount int     `json:"kl_token_count,omitempty"`
	MeanKL       float64 `json:"mean_kl,omitempty"`
}

type Accumulator struct {
	nllSum     float64
	tokenCount int
	klSum      float64
	klCount    int
}

func (a *Accumulator) AddNLL(nll float64) error {
	if math.IsNaN(nll) || math.IsInf(nll, 0) {
		return fmt.Errorf("invalid nll %v", nll)
	}
	a.nllSum += nll
	a.tokenCount++
	return nil
}

func (a *Accumulator) AddNLLSum(nllSum float64, tokenCount int) error {
	if tokenCount <= 0 {
		return fmt.Errorf("token count must be positive")
	}
	if math.IsNaN(nllSum) || math.IsInf(nllSum, 0) {
		return fmt.Errorf("invalid nll sum %v", nllSum)
	}
	a.nllSum += nllSum
	a.tokenCount += tokenCount
	return nil
}

func (a *Accumulator) AddKLSum(klSum float64, tokenCount int) error {
	if tokenCount <= 0 {
		return fmt.Errorf("token count must be positive")
	}
	if math.IsNaN(klSum) || math.IsInf(klSum, 0) {
		return fmt.Errorf("invalid kl sum %v", klSum)
	}
	a.klSum += klSum
	a.klCount += tokenCount
	return nil
}

func (a Accumulator) Metrics() (Metrics, error) {
	if a.tokenCount == 0 {
		return Metrics{}, fmt.Errorf("no tokens evaluated")
	}
	mean := a.nllSum / float64(a.tokenCount)
	metrics := Metrics{
		TokenCount: a.tokenCount,
		MeanNLL:    mean,
		Perplexity: math.Exp(mean),
	}
	if a.klCount > 0 {
		metrics.KLTokenCount = a.klCount
		metrics.MeanKL = a.klSum / float64(a.klCount)
	}
	return metrics, nil
}

func NegativeLogLikelihood(logits []float32, target int) (float64, error) {
	if len(logits) == 0 {
		return 0, fmt.Errorf("logits are empty")
	}
	if target < 0 || target >= len(logits) {
		return 0, fmt.Errorf("target token %d outside vocabulary size %d", target, len(logits))
	}

	maxLogit := float64(logits[0])
	for _, logit := range logits[1:] {
		if v := float64(logit); v > maxLogit {
			maxLogit = v
		}
	}

	var sum float64
	for _, logit := range logits {
		sum += math.Exp(float64(logit) - maxLogit)
	}

	return maxLogit + math.Log(sum) - float64(logits[target]), nil
}

func KLDivergence(referenceLogits, candidateLogits []float32) (float64, error) {
	if len(referenceLogits) == 0 {
		return 0, fmt.Errorf("reference logits are empty")
	}
	if len(referenceLogits) != len(candidateLogits) {
		return 0, fmt.Errorf("logit length mismatch: reference=%d candidate=%d", len(referenceLogits), len(candidateLogits))
	}

	refLogZ := logSumExp(referenceLogits)
	candLogZ := logSumExp(candidateLogits)

	var kl float64
	for i := range referenceLogits {
		refLogProb := float64(referenceLogits[i]) - refLogZ
		candLogProb := float64(candidateLogits[i]) - candLogZ
		kl += math.Exp(refLogProb) * (refLogProb - candLogProb)
	}
	if kl < 0 && kl > -1e-12 {
		return 0, nil
	}
	return kl, nil
}

func logSumExp(logits []float32) float64 {
	maxLogit := float64(logits[0])
	for _, logit := range logits[1:] {
		if v := float64(logit); v > maxLogit {
			maxLogit = v
		}
	}

	var sum float64
	for _, logit := range logits {
		sum += math.Exp(float64(logit) - maxLogit)
	}
	return maxLogit + math.Log(sum)
}
