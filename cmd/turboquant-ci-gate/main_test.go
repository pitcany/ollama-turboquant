package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const baselineEvalJSON = `{
  "model": "qwen2.5:7b",
  "kv_cache_type": "f16",
  "num_sequences": 4,
  "metrics": {
    "token_count": 4092,
    "mean_nll": 2.142997,
    "perplexity": 8.524,
    "kl_token_count": 4092,
    "mean_kl": 0
  }
}`

const candidateEvalJSON = `{
  "model": "qwen2.5:7b",
  "kv_cache_preset": "kq8-vturbo4",
  "num_sequences": 4,
  "metrics": {
    "token_count": 4092,
    "mean_nll": 2.150859,
    "perplexity": 8.592,
    "kl_token_count": 4092,
    "mean_kl": 0.017
  }
}`

func TestGatePassesWithinThresholds(t *testing.T) {
	baseline := writeEvalJSON(t, "baseline.json", baselineEvalJSON)
	candidate := writeEvalJSON(t, "candidate.json", candidateEvalJSON)

	report, err := runGate(gateOptions{
		BaselinePath:          baseline,
		CandidatePath:         candidate,
		MaxMeanKL:             0.05,
		MaxPerplexityDriftRel: 0.10,
	})
	if err != nil {
		t.Fatalf("runGate() error = %v", err)
	}
	if report.PerplexityDriftRel <= 0 || report.PerplexityDriftRel >= 0.10 {
		t.Fatalf("perplexity drift = %v, want positive drift below threshold", report.PerplexityDriftRel)
	}
}

func TestGateRejectsMeanKLAboveThreshold(t *testing.T) {
	baseline := writeEvalJSON(t, "baseline.json", baselineEvalJSON)
	candidate := writeEvalJSON(t, "candidate.json", strings.Replace(candidateEvalJSON, `"mean_kl": 0.017`, `"mean_kl": 0.051`, 1))

	_, err := runGate(gateOptions{
		BaselinePath:          baseline,
		CandidatePath:         candidate,
		MaxMeanKL:             0.05,
		MaxPerplexityDriftRel: 0.10,
	})
	if err == nil || !strings.Contains(err.Error(), "mean_kl") {
		t.Fatalf("runGate() error = %v, want mean_kl failure", err)
	}
}

func TestGateRejectsRelativePerplexityDriftAboveThreshold(t *testing.T) {
	baseline := writeEvalJSON(t, "baseline.json", baselineEvalJSON)
	candidate := writeEvalJSON(t, "candidate.json", strings.Replace(candidateEvalJSON, `"perplexity": 8.592`, `"perplexity": 9.500`, 1))

	_, err := runGate(gateOptions{
		BaselinePath:          baseline,
		CandidatePath:         candidate,
		MaxMeanKL:             0.05,
		MaxPerplexityDriftRel: 0.10,
	})
	if err == nil || !strings.Contains(err.Error(), "perplexity drift") {
		t.Fatalf("runGate() error = %v, want perplexity drift failure", err)
	}
}

func TestGateRejectsCandidateWithoutMeanKL(t *testing.T) {
	baseline := writeEvalJSON(t, "baseline.json", baselineEvalJSON)
	candidate := writeEvalJSON(t, "candidate.json", strings.Replace(candidateEvalJSON, `    "kl_token_count": 4092,
    "mean_kl": 0.017`, `    "kl_token_count": 0`, 1))

	_, err := runGate(gateOptions{
		BaselinePath:          baseline,
		CandidatePath:         candidate,
		MaxMeanKL:             0.05,
		MaxPerplexityDriftRel: 0.10,
	})
	if err == nil || !strings.Contains(err.Error(), "mean_kl") {
		t.Fatalf("runGate() error = %v, want missing mean_kl failure", err)
	}
}

func TestGateRejectsInvalidThresholds(t *testing.T) {
	err := validateGateOptions(gateOptions{
		BaselinePath:          "baseline.json",
		CandidatePath:         "candidate.json",
		MaxMeanKL:             -0.01,
		MaxPerplexityDriftRel: 0.10,
	})
	if err == nil {
		t.Fatal("expected negative mean KL threshold to be rejected")
	}

	err = validateGateOptions(gateOptions{
		BaselinePath:          "baseline.json",
		CandidatePath:         "candidate.json",
		MaxMeanKL:             0.05,
		MaxPerplexityDriftRel: -0.01,
	})
	if err == nil {
		t.Fatal("expected negative perplexity drift threshold to be rejected")
	}
}

// TestDefaultThresholdsArePinned guards the documented Phase 0 quality
// budget. Bumping these values requires a matching update to the budget
// table in tools/turboquant/README.md ("CI Quality Gate") and to the
// -max-* flags in .github/workflows/turboquant-phase0.yml. See the
// README section for the rationale and signoff process.
func TestDefaultThresholdsArePinned(t *testing.T) {
	if defaultMaxMeanKL != 0.05 {
		t.Fatalf("defaultMaxMeanKL = %v, want 0.05 (see tools/turboquant/README.md CI Quality Gate)", defaultMaxMeanKL)
	}
	if defaultMaxPerplexityDriftRel != 0.05 {
		t.Fatalf("defaultMaxPerplexityDriftRel = %v, want 0.05 (see tools/turboquant/README.md CI Quality Gate)", defaultMaxPerplexityDriftRel)
	}

	opts, err := parseGateFlags([]string{"-baseline", "b.json", "-candidate", "c.json"})
	if err != nil {
		t.Fatalf("parseGateFlags() error = %v", err)
	}
	if opts.MaxMeanKL != defaultMaxMeanKL {
		t.Fatalf("parseGateFlags MaxMeanKL = %v, want %v", opts.MaxMeanKL, defaultMaxMeanKL)
	}
	if opts.MaxPerplexityDriftRel != defaultMaxPerplexityDriftRel {
		t.Fatalf("parseGateFlags MaxPerplexityDriftRel = %v, want %v", opts.MaxPerplexityDriftRel, defaultMaxPerplexityDriftRel)
	}
}

func writeEvalJSON(t *testing.T, name string, data string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(path, []byte(data), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}
