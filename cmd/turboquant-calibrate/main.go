package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	tqeval "github.com/ollama/ollama/tools/turboquant/eval"
	"github.com/ollama/ollama/tools/turboquant/layerselect"
)

type calibrationOptions struct {
	WorkDir       string
	ArtifactDir   string
	Output        string
	SweepCSV      string
	LogDir        string
	Model         string
	Snapshot      string
	BaseKV        string
	ReferenceKV   string
	LayerDType    string
	Top           int
	SweepLimit    int
	ValidateLimit int
	NumCtx        int
	BatchSize     int
	NumGPULayers  int
}

type commandInvocation struct {
	Name string            `json:"name"`
	Args []string          `json:"args"`
	Dir  string            `json:"dir,omitempty"`
	Env  map[string]string `json:"env,omitempty"`
}

type commandRunner interface {
	Run(context.Context, commandInvocation) (string, error)
}

type execCommandRunner struct {
	stderr io.Writer
}

type calibrationArtifact struct {
	Version            int               `json:"version"`
	CreatedAt          string            `json:"created_at"`
	Model              string            `json:"model"`
	Snapshot           string            `json:"snapshot"`
	BaseKV             string            `json:"base_kv_cache_type"`
	ReferenceKV        string            `json:"reference_kv_cache_type"`
	LayerDType         string            `json:"layer_dtype"`
	Top                int               `json:"top"`
	SweepLimit         int               `json:"sweep_limit"`
	ValidateLimit      int               `json:"validate_limit"`
	NumCtx             int               `json:"num_ctx"`
	BatchSize          int               `json:"batch_size"`
	NumGPULayers       int               `json:"num_gpu_layers"`
	SweepCSV           string            `json:"sweep_csv"`
	LogDir             string            `json:"log_dir,omitempty"`
	KeyCacheLayerTypes string            `json:"key_cache_layer_types"`
	SweepCommand       commandInvocation `json:"sweep_command,omitempty"`
	ValidationCommand  commandInvocation `json:"validation_command"`
	Validation         evalResult        `json:"validation"`
}

type evalResult struct {
	Model               string         `json:"model"`
	ModelPath           string         `json:"model_path,omitempty"`
	KVCacheType         string         `json:"kv_cache_type,omitempty"`
	KVCachePreset       string         `json:"kv_cache_preset,omitempty"`
	KeyCacheType        string         `json:"key_cache_type,omitempty"`
	ValueCacheType      string         `json:"value_cache_type,omitempty"`
	KeyCacheLayerPreset string         `json:"key_cache_layer_preset,omitempty"`
	KeyCacheLayerTypes  string         `json:"key_cache_layer_types,omitempty"`
	ReferenceKV         string         `json:"reference_kv_cache_type,omitempty"`
	NumSequences        int            `json:"num_sequences"`
	DurationMS          int64          `json:"duration_ms"`
	Metrics             tqeval.Metrics `json:"metrics"`
}

func main() {
	opts, err := parseCalibrationOptions(os.Args[1:], time.Now)
	if err != nil {
		fmt.Fprintln(os.Stderr, "turboquant-calibrate:", err)
		os.Exit(2)
	}

	if _, err := runCalibration(context.Background(), opts, execCommandRunner{stderr: os.Stderr}, time.Now); err != nil {
		fmt.Fprintln(os.Stderr, "turboquant-calibrate:", err)
		os.Exit(1)
	}
}

func parseCalibrationOptions(args []string, now func() time.Time) (calibrationOptions, error) {
	opts := defaultCalibrationOptions(now)
	fs := flag.NewFlagSet("turboquant-calibrate", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	fs.StringVar(&opts.WorkDir, "work-dir", opts.WorkDir, "repository working directory")
	fs.StringVar(&opts.ArtifactDir, "artifact-dir", opts.ArtifactDir, "directory for generated sweep CSV, logs, and artifact")
	fs.StringVar(&opts.Output, "output", opts.Output, "calibration artifact JSON path")
	fs.StringVar(&opts.SweepCSV, "sweep-csv", opts.SweepCSV, "existing sweep CSV to reuse; skips running the sweep when set")
	fs.StringVar(&opts.LogDir, "log-dir", opts.LogDir, "layer sweep log directory")
	fs.StringVar(&opts.Model, "model", opts.Model, "model name or GGUF path")
	fs.StringVar(&opts.Snapshot, "snapshot", opts.Snapshot, "token snapshot JSON path")
	fs.StringVar(&opts.BaseKV, "base-kv-cache-type", opts.BaseKV, "base KV cache type for layer sweep and validation")
	fs.StringVar(&opts.ReferenceKV, "reference-kv-cache-type", opts.ReferenceKV, "reference KV cache type for validation")
	fs.StringVar(&opts.LayerDType, "layer-dtype", opts.LayerDType, "key cache dtype to use for selected layers")
	fs.IntVar(&opts.Top, "top", opts.Top, "number of lowest-KL layers to select from sweep")
	fs.IntVar(&opts.SweepLimit, "sweep-limit", opts.SweepLimit, "number of sequences for each single-layer sweep run")
	fs.IntVar(&opts.ValidateLimit, "validate-limit", opts.ValidateLimit, "number of sequences for selected-layer validation; 0 means all")
	fs.IntVar(&opts.NumCtx, "num-ctx", opts.NumCtx, "context length")
	fs.IntVar(&opts.BatchSize, "batch-size", opts.BatchSize, "decode batch size")
	fs.IntVar(&opts.NumGPULayers, "num-gpu-layers", opts.NumGPULayers, "number of GPU layers to offload")
	if err := fs.Parse(args); err != nil {
		return calibrationOptions{}, err
	}

	changed := map[string]bool{}
	fs.Visit(func(f *flag.Flag) {
		changed[f.Name] = true
	})
	if changed["artifact-dir"] {
		if !changed["output"] {
			opts.Output = filepath.Join(opts.ArtifactDir, "calibration.json")
		}
		if !changed["log-dir"] {
			opts.LogDir = filepath.Join(opts.ArtifactDir, "logs")
		}
	}
	return opts, nil
}

func defaultCalibrationOptions(now func() time.Time) calibrationOptions {
	artifactDir := filepath.Join("/tmp", "turboquant-calibration-"+now().UTC().Format("20060102-150405"))
	return calibrationOptions{
		WorkDir:       ".",
		ArtifactDir:   artifactDir,
		Output:        filepath.Join(artifactDir, "calibration.json"),
		LogDir:        filepath.Join(artifactDir, "logs"),
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
	}
}

func runCalibration(ctx context.Context, opts calibrationOptions, runner commandRunner, now func() time.Time) (calibrationArtifact, error) {
	if opts.WorkDir == "" {
		opts.WorkDir = "."
	}
	if opts.ArtifactDir == "" {
		opts.ArtifactDir = defaultCalibrationOptions(now).ArtifactDir
	}
	if opts.Output == "" {
		opts.Output = filepath.Join(opts.ArtifactDir, "calibration.json")
	}
	if opts.LogDir == "" {
		opts.LogDir = filepath.Join(opts.ArtifactDir, "logs")
	}
	if err := os.MkdirAll(opts.ArtifactDir, 0o755); err != nil {
		return calibrationArtifact{}, fmt.Errorf("create artifact dir: %w", err)
	}

	var sweepCommand commandInvocation
	sweepCSV := opts.SweepCSV
	if sweepCSV == "" {
		sweepCSV = filepath.Join(opts.ArtifactDir, "layer-sweep.csv")
		sweepCommand = layerSweepCommand(opts, sweepCSV)
		if _, err := runner.Run(ctx, sweepCommand); err != nil {
			return calibrationArtifact{}, fmt.Errorf("run layer sweep: %w", err)
		}
	}

	f, err := os.Open(sweepCSV)
	if err != nil {
		return calibrationArtifact{}, fmt.Errorf("open sweep CSV: %w", err)
	}
	spec, err := layerselect.SelectLayerSpec(f, opts.Top, opts.LayerDType)
	closeErr := f.Close()
	if err != nil {
		return calibrationArtifact{}, err
	}
	if closeErr != nil {
		return calibrationArtifact{}, fmt.Errorf("close sweep CSV: %w", closeErr)
	}

	validationCommand := validationEvalCommand(opts, spec)
	validationJSON, err := runner.Run(ctx, validationCommand)
	if err != nil {
		return calibrationArtifact{}, fmt.Errorf("run validation: %w", err)
	}
	var validation evalResult
	if err := json.Unmarshal([]byte(validationJSON), &validation); err != nil {
		return calibrationArtifact{}, fmt.Errorf("parse validation JSON: %w", err)
	}

	artifact := calibrationArtifact{
		Version:            1,
		CreatedAt:          now().UTC().Format(time.RFC3339),
		Model:              opts.Model,
		Snapshot:           opts.Snapshot,
		BaseKV:             opts.BaseKV,
		ReferenceKV:        opts.ReferenceKV,
		LayerDType:         opts.LayerDType,
		Top:                opts.Top,
		SweepLimit:         opts.SweepLimit,
		ValidateLimit:      opts.ValidateLimit,
		NumCtx:             opts.NumCtx,
		BatchSize:          opts.BatchSize,
		NumGPULayers:       opts.NumGPULayers,
		SweepCSV:           sweepCSV,
		LogDir:             opts.LogDir,
		KeyCacheLayerTypes: spec,
		SweepCommand:       sweepCommand,
		ValidationCommand:  validationCommand,
		Validation:         validation,
	}

	if err := writeArtifact(opts.Output, artifact); err != nil {
		return calibrationArtifact{}, err
	}
	fmt.Fprintf(os.Stderr, "wrote calibration artifact %s\n", opts.Output)
	return artifact, nil
}

func layerSweepCommand(opts calibrationOptions, output string) commandInvocation {
	return commandInvocation{
		Name: "bash",
		Args: []string{"tools/turboquant/layer_sweep.sh"},
		Dir:  opts.WorkDir,
		Env: map[string]string{
			"MODEL":          opts.Model,
			"SNAPSHOT":       opts.Snapshot,
			"BASE_KV":        opts.BaseKV,
			"REFERENCE_KV":   opts.ReferenceKV,
			"LAYER_DTYPE":    opts.LayerDType,
			"LIMIT":          strconv.Itoa(opts.SweepLimit),
			"NUM_CTX":        strconv.Itoa(opts.NumCtx),
			"BATCH_SIZE":     strconv.Itoa(opts.BatchSize),
			"NUM_GPU_LAYERS": strconv.Itoa(opts.NumGPULayers),
			"OUTPUT":         output,
			"LOG_DIR":        opts.LogDir,
		},
	}
}

func validationEvalCommand(opts calibrationOptions, spec string) commandInvocation {
	args := []string{
		"run", "./cmd/turboquant-eval",
		"-engine", "go",
		"-model", opts.Model,
		"-snapshot", opts.Snapshot,
		"-kv-cache-type", opts.BaseKV,
		"-key-cache-layer-types", spec,
		"-reference-kv-cache-type", opts.ReferenceKV,
		"-num-ctx", strconv.Itoa(opts.NumCtx),
		"-batch-size", strconv.Itoa(opts.BatchSize),
		"-num-gpu-layers", strconv.Itoa(opts.NumGPULayers),
		"-flash-attention=true",
		"-limit", strconv.Itoa(opts.ValidateLimit),
		"-format", "json",
	}
	return commandInvocation{
		Name: "go",
		Args: args,
		Dir:  opts.WorkDir,
	}
}

func writeArtifact(path string, artifact calibrationArtifact) error {
	if path == "" {
		return errors.New("artifact output path is required")
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create output dir: %w", err)
	}
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create artifact: %w", err)
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	if err := enc.Encode(artifact); err != nil {
		return fmt.Errorf("write artifact: %w", err)
	}
	return nil
}

func (r execCommandRunner) Run(ctx context.Context, inv commandInvocation) (string, error) {
	cmd := exec.CommandContext(ctx, inv.Name, inv.Args...)
	cmd.Dir = inv.Dir
	cmd.Stderr = r.stderr
	env := os.Environ()
	for key, value := range inv.Env {
		env = append(env, key+"="+value)
	}
	if _, ok := inv.Env["GOCACHE"]; !ok {
		env = append(env, "GOCACHE=/tmp/ollama-build-gocache")
	}
	if _, ok := inv.Env["OLLAMA_LIBRARY_PATH"]; !ok {
		env = append(env, "OLLAMA_LIBRARY_PATH="+filepath.Join(absOrDot(inv.Dir), "build/lib/ollama"))
	}
	cmd.Env = env
	out, err := cmd.Output()
	if err != nil {
		return "", fmt.Errorf("%s %s: %w", inv.Name, strings.Join(inv.Args, " "), err)
	}
	return string(out), nil
}

func absOrDot(dir string) string {
	if dir == "" {
		return "."
	}
	abs, err := filepath.Abs(dir)
	if err != nil {
		return dir
	}
	return abs
}
