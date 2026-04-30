package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"math"
	"os"
	"strings"
)

// Phase 0 quality budget. See tools/turboquant/README.md "CI Quality Gate"
// for the rationale, headroom calculations, and signoff process. These
// constants are pinned by TestDefaultThresholdsArePinned.
const (
	defaultMaxMeanKL             = 0.05
	defaultMaxPerplexityDriftRel = 0.05
)

type gateOptions struct {
	BaselinePath          string
	CandidatePath         string
	MaxMeanKL             float64
	MaxPerplexityDriftRel float64
}

type gateReport struct {
	BaselinePerplexity    float64
	CandidatePerplexity   float64
	CandidateMeanKL       float64
	PerplexityDriftRel    float64
	MaxMeanKL             float64
	MaxPerplexityDriftRel float64
}

type evalResult struct {
	Metrics evalMetrics `json:"metrics"`
}

type evalMetrics struct {
	Perplexity *float64 `json:"perplexity"`
	MeanKL     *float64 `json:"mean_kl"`
}

func main() {
	opts, err := parseGateFlags(os.Args[1:])
	if err != nil {
		fmt.Fprintln(os.Stderr, "turboquant-ci-gate:", err)
		os.Exit(2)
	}

	report, err := runGate(opts)
	if err != nil {
		fmt.Fprintln(os.Stderr, "turboquant-ci-gate:", err)
		os.Exit(1)
	}

	fmt.Printf(
		"turboquant-ci-gate: passed mean_kl=%.8f <= %.8f perplexity_drift=%.6f <= %.6f baseline_perplexity=%.8f candidate_perplexity=%.8f\n",
		report.CandidateMeanKL,
		report.MaxMeanKL,
		report.PerplexityDriftRel,
		report.MaxPerplexityDriftRel,
		report.BaselinePerplexity,
		report.CandidatePerplexity,
	)
}

func parseGateFlags(args []string) (gateOptions, error) {
	opts := gateOptions{
		MaxMeanKL:             defaultMaxMeanKL,
		MaxPerplexityDriftRel: defaultMaxPerplexityDriftRel,
	}
	fs := flag.NewFlagSet("turboquant-ci-gate", flag.ContinueOnError)
	fs.StringVar(&opts.BaselinePath, "baseline", "", "baseline f16 turboquant-eval JSON")
	fs.StringVar(&opts.CandidatePath, "candidate", "", "candidate turboquant-eval JSON")
	fs.Float64Var(&opts.MaxMeanKL, "max-mean-kl", opts.MaxMeanKL, "maximum allowed candidate mean_kl")
	fs.Float64Var(&opts.MaxPerplexityDriftRel, "max-perplexity-drift", opts.MaxPerplexityDriftRel, "maximum allowed relative perplexity drift")
	if err := fs.Parse(args); err != nil {
		return gateOptions{}, err
	}
	if fs.NArg() != 0 {
		return gateOptions{}, fmt.Errorf("unexpected positional arguments: %s", strings.Join(fs.Args(), " "))
	}
	return opts, validateGateOptions(opts)
}

func validateGateOptions(opts gateOptions) error {
	if opts.BaselinePath == "" {
		return errors.New("-baseline is required")
	}
	if opts.CandidatePath == "" {
		return errors.New("-candidate is required")
	}
	if invalidThreshold(opts.MaxMeanKL) {
		return fmt.Errorf("-max-mean-kl must be finite and non-negative, got %v", opts.MaxMeanKL)
	}
	if invalidThreshold(opts.MaxPerplexityDriftRel) {
		return fmt.Errorf("-max-perplexity-drift must be finite and non-negative, got %v", opts.MaxPerplexityDriftRel)
	}
	return nil
}

func invalidThreshold(v float64) bool {
	return math.IsNaN(v) || math.IsInf(v, 0) || v < 0
}

func runGate(opts gateOptions) (gateReport, error) {
	if err := validateGateOptions(opts); err != nil {
		return gateReport{}, err
	}

	baseline, err := loadEvalResult(opts.BaselinePath, "baseline", false)
	if err != nil {
		return gateReport{}, err
	}
	candidate, err := loadEvalResult(opts.CandidatePath, "candidate", true)
	if err != nil {
		return gateReport{}, err
	}

	drift := (*candidate.Metrics.Perplexity - *baseline.Metrics.Perplexity) / *baseline.Metrics.Perplexity
	report := gateReport{
		BaselinePerplexity:    *baseline.Metrics.Perplexity,
		CandidatePerplexity:   *candidate.Metrics.Perplexity,
		CandidateMeanKL:       *candidate.Metrics.MeanKL,
		PerplexityDriftRel:    drift,
		MaxMeanKL:             opts.MaxMeanKL,
		MaxPerplexityDriftRel: opts.MaxPerplexityDriftRel,
	}

	failures := gateFailures(report)
	if len(failures) > 0 {
		return report, fmt.Errorf("quality gate failed: %s", strings.Join(failures, "; "))
	}
	return report, nil
}

func loadEvalResult(path string, label string, requireMeanKL bool) (evalResult, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return evalResult{}, fmt.Errorf("read %s JSON: %w", label, err)
	}

	var result evalResult
	if err := json.Unmarshal(data, &result); err != nil {
		return evalResult{}, fmt.Errorf("parse %s JSON: %w", label, err)
	}
	if result.Metrics.Perplexity == nil {
		return evalResult{}, fmt.Errorf("%s JSON missing metrics.perplexity", label)
	}
	if !finitePositive(*result.Metrics.Perplexity) {
		return evalResult{}, fmt.Errorf("%s metrics.perplexity must be finite and positive, got %v", label, *result.Metrics.Perplexity)
	}
	if requireMeanKL {
		if result.Metrics.MeanKL == nil {
			return evalResult{}, fmt.Errorf("%s JSON missing metrics.mean_kl; run cmd/turboquant-eval with -reference-kv-cache-type f16", label)
		}
		if !finiteNonNegative(*result.Metrics.MeanKL) {
			return evalResult{}, fmt.Errorf("%s metrics.mean_kl must be finite and non-negative, got %v", label, *result.Metrics.MeanKL)
		}
	}
	return result, nil
}

func finitePositive(v float64) bool {
	return finiteNonNegative(v) && v > 0
}

func finiteNonNegative(v float64) bool {
	return !math.IsNaN(v) && !math.IsInf(v, 0) && v >= 0
}

func gateFailures(report gateReport) []string {
	failures := make([]string, 0, 2)
	if report.CandidateMeanKL > report.MaxMeanKL {
		failures = append(failures, fmt.Sprintf("mean_kl %.8f > %.8f", report.CandidateMeanKL, report.MaxMeanKL))
	}
	if report.PerplexityDriftRel > report.MaxPerplexityDriftRel {
		failures = append(failures, fmt.Sprintf("perplexity drift %.6f > %.6f", report.PerplexityDriftRel, report.MaxPerplexityDriftRel))
	}
	return failures
}
