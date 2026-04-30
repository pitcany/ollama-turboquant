package main

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"
)

const calibrationSampleCSV = `layer,key_layer_types,sequences,tokens,mean_nll,perplexity,mean_kl,duration_ms,status
0,0:q8_0,4,4092,2.07640661,7.97575733,0.15234685,18973,ok
1,1:q8_0,4,4092,5.74972980,314.10577687,3.95995938,19091,ok
27,27:q8_0,4,4092,5.80746057,332.77299651,4.01762444,19526,ok
3,3:q8_0,4,4092,5.86085046,351.02254797,4.07728846,19048,ok
`

const validationJSON = `{
  "model": "qwen2.5:7b",
  "kv_cache_type": "turbo4",
  "key_cache_layer_types": "0:q8_0,1:q8_0,3:q8_0,27:q8_0",
  "num_sequences": 16,
  "duration_ms": 76416,
  "metrics": {
    "token_count": 16368,
    "mean_nll": 1.9919270644321017,
    "perplexity": 7.329644859354471,
    "kl_token_count": 16368,
    "mean_kl": 0.05225737156306552
  }
}`

type fakeRunner struct {
	calls []commandInvocation
}

func (r *fakeRunner) Run(ctx context.Context, cmd commandInvocation) (string, error) {
	r.calls = append(r.calls, cmd)
	if cmd.Name == "bash" {
		if err := os.WriteFile(cmd.Env["OUTPUT"], []byte(calibrationSampleCSV), 0o644); err != nil {
			return "", err
		}
		return "", nil
	}
	return validationJSON, nil
}

func TestRunCalibrationRunsSweepSelectsValidatesAndWritesArtifact(t *testing.T) {
	dir := t.TempDir()
	output := filepath.Join(dir, "calibration.json")
	runner := &fakeRunner{}

	artifact, err := runCalibration(context.Background(), calibrationOptions{
		WorkDir:       ".",
		ArtifactDir:   dir,
		Output:        output,
		Model:         "qwen2.5:7b",
		Snapshot:      "tools/turboquant/testdata/qwen25_7b_phase0_tokens.json",
		BaseKV:        "turbo4",
		ReferenceKV:   "f16",
		LayerDType:    "q8_0",
		Top:           4,
		SweepLimit:    4,
		ValidateLimit: 16,
		NumCtx:        1024,
		BatchSize:     512,
		NumGPULayers:  999,
	}, runner, func() time.Time { return time.Unix(42, 0).UTC() })
	if err != nil {
		t.Fatalf("runCalibration() error = %v", err)
	}

	if len(runner.calls) != 2 {
		t.Fatalf("runner call count = %d, want 2", len(runner.calls))
	}
	if runner.calls[0].Name != "bash" {
		t.Fatalf("first command = %q, want bash", runner.calls[0].Name)
	}
	if runner.calls[0].Env["LIMIT"] != "4" || runner.calls[0].Env["LAYER_DTYPE"] != "q8_0" {
		t.Fatalf("unexpected sweep env: %#v", runner.calls[0].Env)
	}
	if runner.calls[1].Name != "go" {
		t.Fatalf("second command = %q, want go", runner.calls[1].Name)
	}
	if !slices.Contains(runner.calls[1].Args, "0:q8_0,1:q8_0,3:q8_0,27:q8_0") {
		t.Fatalf("validation args missing selected spec: %#v", runner.calls[1].Args)
	}

	if artifact.KeyCacheLayerTypes != "0:q8_0,1:q8_0,3:q8_0,27:q8_0" {
		t.Fatalf("selected spec = %q", artifact.KeyCacheLayerTypes)
	}
	if artifact.Validation.Metrics.MeanKL != 0.05225737156306552 {
		t.Fatalf("mean KL = %v", artifact.Validation.Metrics.MeanKL)
	}

	data, err := os.ReadFile(output)
	if err != nil {
		t.Fatalf("read artifact: %v", err)
	}
	var fromDisk calibrationArtifact
	if err := json.Unmarshal(data, &fromDisk); err != nil {
		t.Fatalf("unmarshal artifact: %v", err)
	}
	if fromDisk.CreatedAt != "1970-01-01T00:00:42Z" {
		t.Fatalf("created_at = %q", fromDisk.CreatedAt)
	}
}

func TestRunCalibrationReusesExistingSweepCSV(t *testing.T) {
	dir := t.TempDir()
	sweepCSV := filepath.Join(dir, "sweep.csv")
	if err := os.WriteFile(sweepCSV, []byte(calibrationSampleCSV), 0o644); err != nil {
		t.Fatal(err)
	}

	runner := &fakeRunner{}
	_, err := runCalibration(context.Background(), calibrationOptions{
		WorkDir:       ".",
		ArtifactDir:   dir,
		Output:        filepath.Join(dir, "calibration.json"),
		SweepCSV:      sweepCSV,
		Model:         "qwen2.5:7b",
		Snapshot:      "snapshot.json",
		BaseKV:        "turbo4",
		ReferenceKV:   "f16",
		LayerDType:    "q8_0",
		Top:           4,
		SweepLimit:    4,
		ValidateLimit: 16,
		NumCtx:        1024,
		BatchSize:     512,
		NumGPULayers:  999,
	}, runner, func() time.Time { return time.Unix(42, 0).UTC() })
	if err != nil {
		t.Fatalf("runCalibration() error = %v", err)
	}
	if len(runner.calls) != 1 {
		t.Fatalf("runner call count = %d, want 1", len(runner.calls))
	}
}

func TestParseCalibrationOptionsRebasesDerivedPaths(t *testing.T) {
	dir := t.TempDir()

	opts, err := parseCalibrationOptions([]string{"-artifact-dir", dir}, func() time.Time {
		return time.Unix(42, 0).UTC()
	})
	if err != nil {
		t.Fatalf("parseCalibrationOptions() error = %v", err)
	}

	if opts.ArtifactDir != dir {
		t.Fatalf("artifact dir = %q, want %q", opts.ArtifactDir, dir)
	}
	if opts.Output != filepath.Join(dir, "calibration.json") {
		t.Fatalf("output = %q, want artifact-dir default", opts.Output)
	}
	if opts.LogDir != filepath.Join(dir, "logs") {
		t.Fatalf("log dir = %q, want artifact-dir default", opts.LogDir)
	}
}

func TestParseCalibrationOptionsPreservesExplicitDerivedPaths(t *testing.T) {
	dir := t.TempDir()
	output := filepath.Join(t.TempDir(), "out.json")
	logDir := filepath.Join(t.TempDir(), "logs")

	opts, err := parseCalibrationOptions([]string{
		"-artifact-dir", dir,
		"-output", output,
		"-log-dir", logDir,
	}, func() time.Time {
		return time.Unix(42, 0).UTC()
	})
	if err != nil {
		t.Fatalf("parseCalibrationOptions() error = %v", err)
	}

	if opts.Output != output {
		t.Fatalf("output = %q, want explicit output", opts.Output)
	}
	if opts.LogDir != logDir {
		t.Fatalf("log dir = %q, want explicit log dir", opts.LogDir)
	}
}
