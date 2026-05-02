package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"math"
	"os"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/ollama/ollama/llama"
	"github.com/ollama/ollama/ml"
	ollamamodel "github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
	_ "github.com/ollama/ollama/model/models"
	tqeval "github.com/ollama/ollama/tools/turboquant/eval"
)

type options struct {
	engine                  string
	model                   string
	snapshot                string
	kvCacheType             string
	kvCachePreset           string
	keyCacheType            string
	valueCacheType          string
	keyCacheLayerPreset     string
	keyCacheLayerTypes      string
	keyCacheResidualWindow  int
	keyCacheResidualDType   string
	referenceKV             string
	numCtx                  int
	batchSize               int
	threads                 int
	numGPULayers            int
	flashAttn               bool
	limit                   int
	format                  string
}

type result struct {
	Model                  string         `json:"model"`
	ModelPath              string         `json:"model_path"`
	KVCacheType            string         `json:"kv_cache_type,omitempty"`
	KVCachePreset          string         `json:"kv_cache_preset,omitempty"`
	KeyCacheType           string         `json:"key_cache_type,omitempty"`
	ValueCacheType         string         `json:"value_cache_type,omitempty"`
	KeyCacheLayerPreset    string         `json:"key_cache_layer_preset,omitempty"`
	KeyCacheLayerTypes     string         `json:"key_cache_layer_types,omitempty"`
	KeyCacheResidualWindow int            `json:"key_cache_residual_window,omitempty"`
	KeyCacheResidualDType  string         `json:"key_cache_residual_dtype,omitempty"`
	ReferenceKV            string         `json:"reference_kv_cache_type,omitempty"`
	NumSequences           int            `json:"num_sequences"`
	DurationMS             int64          `json:"duration_ms"`
	Metrics                tqeval.Metrics `json:"metrics"`
}

func main() {
	opts := parseFlags()
	if err := run(opts); err != nil {
		fmt.Fprintln(os.Stderr, "turboquant-eval:", err)
		os.Exit(1)
	}
}

func parseFlags() options {
	defaultKV := os.Getenv("OLLAMA_KV_CACHE_TYPE")
	if defaultKV == "" {
		defaultKV = "f16"
	}

	var opts options
	flag.StringVar(&opts.engine, "engine", "go", "evaluation engine: go or llama")
	flag.StringVar(&opts.model, "model", "qwen2.5:7b", "GGUF file path or local Ollama model name")
	flag.StringVar(&opts.snapshot, "snapshot", "", "token snapshot JSON path")
	flag.StringVar(&opts.kvCacheType, "kv-cache-type", defaultKV, "KV cache type passed to llama context, for example f16 or turbo4")
	flag.StringVar(&opts.kvCachePreset, "kv-cache-preset", "", "optional Go-engine named split K/V cache preset, for example kq8-vturbo4")
	flag.StringVar(&opts.keyCacheType, "key-cache-type", "", "optional Go-engine key cache type override for K/V isolation")
	flag.StringVar(&opts.valueCacheType, "value-cache-type", "", "optional Go-engine value cache type override for K/V isolation")
	flag.StringVar(&opts.keyCacheLayerPreset, "key-cache-layer-preset", "", "optional Go-engine named per-layer key cache override preset, for example qwen2.5-7b-q4km-adaptive")
	flag.StringVar(&opts.keyCacheLayerTypes, "key-cache-layer-types", "", "optional Go-engine per-layer key cache overrides, for example 0:f16,27:f16")
	flag.IntVar(&opts.keyCacheResidualWindow, "key-cache-residual-window", 0, "experimental Go-engine residual-window key protection: most recent N key positions stay at -key-cache-type while older positions round-trip through -key-cache-residual-dtype")
	flag.StringVar(&opts.keyCacheResidualDType, "key-cache-residual-dtype", "", "experimental residual-window base dtype (turbo2, turbo3, turbo4); positions older than -key-cache-residual-window are degraded to this dtype")
	flag.StringVar(&opts.referenceKV, "reference-kv-cache-type", "", "optional reference KV cache type for output-logit KL, usually f16")
	flag.IntVar(&opts.numCtx, "num-ctx", 2048, "context length for each evaluated sequence")
	flag.IntVar(&opts.batchSize, "batch-size", 512, "decode batch size")
	flag.IntVar(&opts.threads, "threads", runtime.NumCPU(), "CPU threads")
	flag.IntVar(&opts.numGPULayers, "num-gpu-layers", -1, "number of layers to offload to GPU; use 0 for CPU-only")
	flag.BoolVar(&opts.flashAttn, "flash-attention", true, "enable flash attention")
	flag.IntVar(&opts.limit, "limit", 0, "maximum number of sequences to evaluate; 0 means all")
	flag.StringVar(&opts.format, "format", "text", "output format: text or json")
	flag.Parse()
	return opts
}

func run(opts options) error {
	if err := validateOptions(opts); err != nil {
		return err
	}

	snapshotFile, err := os.Open(opts.snapshot)
	if err != nil {
		return fmt.Errorf("open snapshot: %w", err)
	}
	defer snapshotFile.Close()

	snapshot, err := tqeval.LoadSnapshot(snapshotFile)
	if err != nil {
		return err
	}
	if opts.limit > 0 && opts.limit < len(snapshot.Sequences) {
		snapshot.Sequences = snapshot.Sequences[:opts.limit]
	}
	for i, seq := range snapshot.Sequences {
		if len(seq.Tokens) > opts.numCtx {
			return fmt.Errorf("sequence %d has %d tokens, exceeds -num-ctx=%d", i, len(seq.Tokens), opts.numCtx)
		}
	}

	modelPath, err := tqeval.ResolveModelPath(opts.model)
	if err != nil {
		return err
	}

	start := time.Now()
	metrics, err := evaluate(modelPath, snapshot, opts)
	if err != nil {
		return err
	}

	out := result{
		Model:        opts.model,
		ModelPath:    modelPath,
		ReferenceKV:  opts.referenceKV,
		NumSequences: len(snapshot.Sequences),
		DurationMS:   time.Since(start).Milliseconds(),
		Metrics:      metrics,
	}
	keyCacheType, valueCacheType, err := goKVCacheTypes(opts)
	if err != nil {
		return err
	}
	if opts.kvCachePreset != "" {
		out.KVCachePreset = opts.kvCachePreset
	} else {
		out.KVCacheType = opts.kvCacheType
	}
	if keyCacheType != opts.kvCacheType {
		out.KeyCacheType = keyCacheType
	}
	if valueCacheType != opts.kvCacheType {
		out.ValueCacheType = valueCacheType
	}
	layerSpec, err := keyCacheLayerSpec(opts)
	if err != nil {
		return err
	}
	if opts.keyCacheLayerPreset != "" {
		out.KeyCacheLayerPreset = opts.keyCacheLayerPreset
	}
	if layerSpec != "" {
		out.KeyCacheLayerTypes = layerSpec
	}
	if opts.keyCacheResidualWindow > 0 {
		out.KeyCacheResidualWindow = opts.keyCacheResidualWindow
		out.KeyCacheResidualDType = opts.keyCacheResidualDType
	}

	switch opts.format {
	case "json":
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(out)
	case "text":
		fmt.Printf("model=%s\n", out.Model)
		fmt.Printf("model_path=%s\n", out.ModelPath)
		if out.KVCacheType != "" {
			fmt.Printf("kv_cache_type=%s\n", out.KVCacheType)
		}
		if out.KVCachePreset != "" {
			fmt.Printf("kv_cache_preset=%s\n", out.KVCachePreset)
		}
		if out.KeyCacheType != "" || out.ValueCacheType != "" {
			fmt.Printf("key_cache_type=%s value_cache_type=%s\n", keyCacheType, valueCacheType)
		}
		if out.KeyCacheLayerPreset != "" {
			fmt.Printf("key_cache_layer_preset=%s\n", out.KeyCacheLayerPreset)
		}
		if out.KeyCacheLayerTypes != "" {
			fmt.Printf("key_cache_layer_types=%s\n", out.KeyCacheLayerTypes)
		}
		if out.KeyCacheResidualWindow > 0 {
			fmt.Printf("key_cache_residual_window=%d key_cache_residual_dtype=%s\n", out.KeyCacheResidualWindow, out.KeyCacheResidualDType)
		}
		if out.ReferenceKV != "" {
			fmt.Printf("reference_kv_cache_type=%s\n", out.ReferenceKV)
		}
		fmt.Printf("sequences=%d tokens=%d\n", out.NumSequences, out.Metrics.TokenCount)
		fmt.Printf("mean_nll=%.8f perplexity=%.8f\n", out.Metrics.MeanNLL, out.Metrics.Perplexity)
		if out.Metrics.KLTokenCount > 0 {
			fmt.Printf("mean_kl=%.8f kl_tokens=%d\n", out.Metrics.MeanKL, out.Metrics.KLTokenCount)
		}
		fmt.Printf("duration_ms=%d\n", out.DurationMS)
		return nil
	default:
		return fmt.Errorf("unsupported -format %q", opts.format)
	}
}

func validateOptions(opts options) error {
	if opts.snapshot == "" {
		return errors.New("-snapshot is required")
	}
	if opts.numCtx <= 1 {
		return errors.New("-num-ctx must be greater than one")
	}
	if opts.batchSize <= 0 {
		return errors.New("-batch-size must be positive")
	}
	if opts.threads <= 0 {
		return errors.New("-threads must be positive")
	}
	keyLayerDTypes, err := keyCacheLayerDTypes(opts)
	if err != nil {
		return err
	}
	if hasKeyCacheLayerConfig(opts) {
		if !strings.EqualFold(opts.engine, "go") {
			return errors.New("key cache layer overrides are only supported with -engine go")
		}
	}
	if opts.kvCachePreset != "" && !strings.EqualFold(opts.engine, "go") {
		return errors.New("-kv-cache-preset is only supported with -engine go")
	}
	keyCacheType, valueCacheType, err := goKVCacheTypes(opts)
	if err != nil {
		return err
	}
	if strings.EqualFold(opts.engine, "go") && !opts.flashAttn && (isTurboKVCacheName(keyCacheType) || isTurboKVCacheName(valueCacheType)) {
		return errors.New("Go-engine Turbo KV cache types require -flash-attention=true")
	}
	if strings.EqualFold(opts.engine, "go") && !opts.flashAttn {
		for _, dtype := range keyLayerDTypes {
			if isTurboKVCacheDType(dtype) {
				return errors.New("Go-engine Turbo key layer overrides require -flash-attention=true")
			}
		}
	}
	if err := validateKeyCacheResidualOptions(opts, keyCacheType); err != nil {
		return err
	}
	return nil
}

// validateKeyCacheResidualOptions validates the eval-only residual-window
// flags. Both flags must be set together, the residual dtype must be a Turbo*
// dtype, the recent-window dtype (the cache's key dtype) must be higher
// precision than the residual dtype, and the residual window cannot be
// combined with per-layer key dtype overrides because the two policies target
// orthogonal axes (per-position vs per-layer).
func validateKeyCacheResidualOptions(opts options, keyCacheType string) error {
	hasWindow := opts.keyCacheResidualWindow > 0
	hasDType := opts.keyCacheResidualDType != ""
	if !hasWindow && !hasDType {
		return nil
	}
	if !hasWindow || !hasDType {
		return errors.New("-key-cache-residual-window and -key-cache-residual-dtype must be set together")
	}
	if !strings.EqualFold(opts.engine, "go") {
		return errors.New("-key-cache-residual-window is only supported with -engine go")
	}
	if !opts.flashAttn {
		return errors.New("-key-cache-residual-window requires -flash-attention=true")
	}
	if !isTurboKVCacheName(opts.keyCacheResidualDType) {
		return fmt.Errorf("-key-cache-residual-dtype must be a Turbo dtype (turbo2, turbo3, turbo4), got %q", opts.keyCacheResidualDType)
	}
	if !isResidualWindowKeyCacheName(keyCacheType) {
		return fmt.Errorf("-key-cache-type must be a higher-precision dtype (f16 or q8_0) when using -key-cache-residual-window, got %q", keyCacheType)
	}
	if hasKeyCacheLayerConfig(opts) {
		return errors.New("-key-cache-residual-window cannot be combined with -key-cache-layer-types or -key-cache-layer-preset")
	}
	return nil
}

func isResidualWindowKeyCacheName(s string) bool {
	switch strings.ToLower(s) {
	case "f16", "q8_0":
		return true
	default:
		return false
	}
}

func evaluate(modelPath string, snapshot tqeval.Snapshot, opts options) (tqeval.Metrics, error) {
	switch strings.ToLower(opts.engine) {
	case "go":
		return evaluateGo(modelPath, snapshot, opts)
	case "llama":
		return evaluateLlama(modelPath, snapshot, opts)
	default:
		return tqeval.Metrics{}, fmt.Errorf("unsupported -engine %q", opts.engine)
	}
}

func evaluateLlama(modelPath string, snapshot tqeval.Snapshot, opts options) (tqeval.Metrics, error) {
	llama.BackendInit()

	model, err := llama.LoadModelFromFile(modelPath, llama.ModelParams{
		NumGpuLayers: opts.numGPULayers,
		UseMmap:      true,
	})
	if err != nil {
		return tqeval.Metrics{}, err
	}
	defer llama.FreeModel(model)

	ctx, err := newEvalContext(model, opts, opts.kvCacheType)
	if err != nil {
		return tqeval.Metrics{}, err
	}
	var referenceCtx *llama.Context
	if opts.referenceKV != "" {
		referenceCtx, err = newEvalContext(model, opts, opts.referenceKV)
		if err != nil {
			return tqeval.Metrics{}, fmt.Errorf("reference context: %w", err)
		}
	}

	var acc tqeval.Accumulator
	for i, seq := range snapshot.Sequences {
		var nll float64
		var count int
		var kl float64
		var klCount int
		var err error
		if referenceCtx == nil {
			nll, count, err = evaluateSequence(ctx, seq.Tokens, opts.batchSize)
		} else {
			nll, count, kl, klCount, err = evaluateSequenceWithReference(ctx, referenceCtx, seq.Tokens, opts.batchSize)
		}
		if err != nil {
			return tqeval.Metrics{}, fmt.Errorf("sequence %d: %w", i, err)
		}
		if err := acc.AddNLLSum(nll, count); err != nil {
			return tqeval.Metrics{}, err
		}
		if klCount > 0 {
			if err := acc.AddKLSum(kl, klCount); err != nil {
				return tqeval.Metrics{}, err
			}
		}
	}

	return acc.Metrics()
}

type goEvalModel struct {
	model ollamamodel.Model
}

func (m goEvalModel) Close() {
	if m.model != nil {
		m.model.Backend().Close()
	}
}

func evaluateGo(modelPath string, snapshot tqeval.Snapshot, opts options) (tqeval.Metrics, error) {
	keyCacheType, valueCacheType, err := goKVCacheTypes(opts)
	if err != nil {
		return tqeval.Metrics{}, err
	}
	candidate, err := loadGoEvalModel(modelPath, opts, keyCacheType, valueCacheType)
	if err != nil {
		return tqeval.Metrics{}, err
	}
	defer candidate.Close()

	var reference *goEvalModel
	if opts.referenceKV != "" {
		ref, err := loadGoEvalModel(modelPath, opts, opts.referenceKV, opts.referenceKV)
		if err != nil {
			return tqeval.Metrics{}, fmt.Errorf("reference model: %w", err)
		}
		reference = &ref
		defer reference.Close()
	}

	var acc tqeval.Accumulator
	for i, seq := range snapshot.Sequences {
		var nll float64
		var count int
		var kl float64
		var klCount int
		var err error
		if reference == nil {
			nll, count, err = evaluateGoSequence(candidate.model, seq.Tokens, opts.batchSize)
		} else {
			nll, count, kl, klCount, err = evaluateGoSequenceWithReference(candidate.model, reference.model, seq.Tokens, opts.batchSize)
		}
		if err != nil {
			return tqeval.Metrics{}, fmt.Errorf("sequence %d: %w", i, err)
		}
		if err := acc.AddNLLSum(nll, count); err != nil {
			return tqeval.Metrics{}, err
		}
		if klCount > 0 {
			if err := acc.AddKLSum(kl, klCount); err != nil {
				return tqeval.Metrics{}, err
			}
		}
	}

	return acc.Metrics()
}

func loadGoEvalModel(modelPath string, opts options, keyCacheType string, valueCacheType string) (goEvalModel, error) {
	params, err := goBackendParams(modelPath, opts)
	if err != nil {
		return goEvalModel{}, err
	}

	m, err := ollamamodel.New(modelPath, params)
	if err != nil {
		return goEvalModel{}, err
	}

	if err := m.Backend().Load(context.Background(), func(float32) {}); err != nil {
		m.Backend().Close()
		return goEvalModel{}, err
	}

	if postLoader, ok := m.(ollamamodel.PostLoader); ok {
		if err := postLoader.PostLoad(); err != nil {
			m.Backend().Close()
			return goEvalModel{}, err
		}
	}

	if cache := m.Config().Cache; cache != nil {
		keyDType := goKVCacheDType(keyCacheType)
		valueDType := goKVCacheDType(valueCacheType)
		if keyDType == valueDType {
			cache.Init(m.Backend(), keyDType, 1, opts.numCtx, opts.batchSize)
		} else {
			split, ok := cache.(interface {
				InitSplit(ml.Backend, ml.DType, ml.DType, int, int, int)
			})
			if !ok {
				m.Backend().Close()
				return goEvalModel{}, fmt.Errorf("cache does not support split key/value dtypes: key=%s value=%s", keyCacheType, valueCacheType)
			}
			split.InitSplit(m.Backend(), keyDType, valueDType, 1, opts.numCtx, opts.batchSize)
		}
		if hasKeyCacheLayerConfig(opts) {
			overrides, err := keyCacheLayerDTypes(opts)
			if err != nil {
				m.Backend().Close()
				return goEvalModel{}, err
			}
			layerDTypes, ok := cache.(interface {
				SetKeyLayerDTypes(map[int]ml.DType)
			})
			if !ok {
				m.Backend().Close()
				return goEvalModel{}, errors.New("cache does not support per-layer key dtypes")
			}
			layerDTypes.SetKeyLayerDTypes(overrides)
		}
		if opts.keyCacheResidualWindow > 0 {
			residualSetter, ok := cache.(interface {
				SetKeyResidualWindow(int, ml.DType)
			})
			if !ok {
				m.Backend().Close()
				return goEvalModel{}, errors.New("cache does not support residual-window key protection")
			}
			residualSetter.SetKeyResidualWindow(opts.keyCacheResidualWindow, goKVCacheDType(opts.keyCacheResidualDType))
		}
	}

	return goEvalModel{model: m}, nil
}

func goBackendParams(modelPath string, opts options) (ml.BackendParams, error) {
	flashAttention := ml.FlashAttentionDisabled
	if opts.flashAttn {
		flashAttention = ml.FlashAttentionEnabled
	}

	params := ml.BackendParams{
		AllocMemory:    true,
		NumThreads:     opts.threads,
		FlashAttention: flashAttention,
	}

	if opts.numGPULayers == 0 {
		return params, nil
	}

	probe, err := ollamamodel.New(modelPath, ml.BackendParams{
		AllocMemory:    false,
		NumThreads:     opts.threads,
		FlashAttention: flashAttention,
	})
	if err != nil {
		return ml.BackendParams{}, err
	}
	defer probe.Backend().Close()

	memory := probe.Backend().BackendMemory()
	if len(memory.GPUs) == 0 {
		return params, nil
	}

	layerCount := len(memory.CPU.Weights)
	gpuLayerCount := opts.numGPULayers
	if gpuLayerCount < 0 || gpuLayerCount > layerCount {
		gpuLayerCount = layerCount
	}

	// Distribute layers contiguously across all visible GPUs. The previous
	// behavior — putting all layers on memory.GPUs[0] — fails for big models
	// (e.g. qwen3-coder:30b ~18 GiB Q4_K_M, qwen3.6:27b ~30 GiB Q8_0) when
	// any single device cannot satisfy the resulting contiguous weight buffer
	// allocation. KL eval is insensitive to which GPU owns each layer, so a
	// straightforward equal split across all devices is sufficient and
	// matches what the production runner does.
	gpuCount := len(memory.GPUs)
	splits := make(ml.GPULayersList, 0, gpuCount)
	base, extra := gpuLayerCount/gpuCount, gpuLayerCount%gpuCount
	next := 0
	for i := 0; i < gpuCount; i++ {
		count := base
		if i < extra {
			count++
		}
		if count == 0 {
			continue
		}
		layers := make([]int, count)
		for j := 0; j < count; j++ {
			layers[j] = next + j
		}
		next += count
		splits = append(splits, ml.GPULayers{
			DeviceID: memory.GPUs[i].DeviceID,
			Layers:   layers,
		})
	}
	params.GPULayers = splits

	return params, nil
}

const kvCachePresetKQ8VTurbo4 = "kq8-vturbo4"

var kvCachePresetTypes = map[string][2]string{
	kvCachePresetKQ8VTurbo4: {"q8_0", "turbo4"},
}

func goKVCacheTypes(opts options) (string, string, error) {
	if opts.kvCachePreset != "" {
		if opts.keyCacheType != "" || opts.valueCacheType != "" {
			return "", "", errors.New("-kv-cache-preset cannot be combined with -key-cache-type or -value-cache-type")
		}
		types, ok := kvCachePresetTypes[strings.ToLower(opts.kvCachePreset)]
		if !ok {
			return "", "", fmt.Errorf("unsupported -kv-cache-preset %q", opts.kvCachePreset)
		}
		return types[0], types[1], nil
	}

	keyCacheType := opts.kvCacheType
	valueCacheType := opts.kvCacheType
	if opts.keyCacheType != "" {
		keyCacheType = opts.keyCacheType
	}
	if opts.valueCacheType != "" {
		valueCacheType = opts.valueCacheType
	}
	return keyCacheType, valueCacheType, nil
}

const keyCacheLayerPresetQwen25_7BQ4KMAdaptive = "qwen2.5-7b-q4km-adaptive"

var keyCacheLayerPresetSpecs = map[string]string{
	keyCacheLayerPresetQwen25_7BQ4KMAdaptive: "0:q8_0,1:q8_0,3:q8_0,27:q8_0",
}

func hasKeyCacheLayerConfig(opts options) bool {
	return opts.keyCacheLayerPreset != "" || opts.keyCacheLayerTypes != ""
}

func keyCacheLayerSpec(opts options) (string, error) {
	if opts.keyCacheLayerPreset == "" {
		return opts.keyCacheLayerTypes, nil
	}
	if opts.keyCacheLayerTypes != "" {
		return "", errors.New("-key-cache-layer-preset cannot be combined with -key-cache-layer-types")
	}
	spec, ok := keyCacheLayerPresetSpecs[strings.ToLower(opts.keyCacheLayerPreset)]
	if !ok {
		return "", fmt.Errorf("unsupported -key-cache-layer-preset %q", opts.keyCacheLayerPreset)
	}
	return spec, nil
}

func keyCacheLayerDTypes(opts options) (map[int]ml.DType, error) {
	spec, err := keyCacheLayerSpec(opts)
	if err != nil {
		return nil, err
	}
	if spec == "" {
		return nil, nil
	}
	return parseKeyCacheLayerTypes(spec)
}

func isTurboKVCacheName(s string) bool {
	switch strings.ToLower(s) {
	case "turbo2", "turbo3", "turbo4":
		return true
	default:
		return false
	}
}

func isGoKVCacheName(s string) bool {
	switch strings.ToLower(s) {
	case "", "f16", "q8_0", "q4_0", "turbo2", "turbo3", "turbo4":
		return true
	default:
		return false
	}
}

func isTurboKVCacheDType(dtype ml.DType) bool {
	switch dtype {
	case ml.DTypeTurbo2, ml.DTypeTurbo3, ml.DTypeTurbo4:
		return true
	default:
		return false
	}
}

func goKVCacheDType(s string) ml.DType {
	switch strings.ToLower(s) {
	case "q8_0":
		return ml.DTypeQ80
	case "q4_0":
		return ml.DTypeQ40
	case "turbo2":
		return ml.DTypeTurbo2
	case "turbo3":
		return ml.DTypeTurbo3
	case "turbo4":
		return ml.DTypeTurbo4
	default:
		return ml.DTypeF16
	}
}

func parseKeyCacheLayerTypes(spec string) (map[int]ml.DType, error) {
	out := map[int]ml.DType{}
	for _, item := range strings.Split(spec, ",") {
		item = strings.TrimSpace(item)
		if item == "" {
			continue
		}
		layerText, dtypeText, ok := strings.Cut(item, ":")
		if !ok {
			return nil, fmt.Errorf("invalid -key-cache-layer-types item %q, want layer:type", item)
		}
		layerText = strings.TrimSpace(layerText)
		dtypeText = strings.TrimSpace(dtypeText)
		if layerText == "" || dtypeText == "" {
			return nil, fmt.Errorf("invalid -key-cache-layer-types item %q, want layer:type", item)
		}
		layer, err := strconv.Atoi(layerText)
		if err != nil || layer < 0 {
			return nil, fmt.Errorf("invalid key cache layer %q", layerText)
		}
		if !isGoKVCacheName(dtypeText) {
			return nil, fmt.Errorf("unsupported key cache layer dtype %q", dtypeText)
		}
		out[layer] = goKVCacheDType(dtypeText)
	}
	if len(out) == 0 {
		return nil, errors.New("-key-cache-layer-types did not contain any overrides")
	}
	return out, nil
}

func evaluateGoSequence(m ollamamodel.Model, tokens []int, batchSize int) (float64, int, error) {
	if err := clearGoCache(m); err != nil {
		return 0, 0, err
	}

	var nllSum float64
	var tokenCount int
	for offset := 0; offset < len(tokens); {
		end := min(offset+batchSize, len(tokens))
		logits, outputPositions, vocabSize, err := evaluateGoBatch(m, tokens, offset, end)
		if err != nil {
			return 0, 0, err
		}

		for i, pos := range outputPositions {
			absolutePos := offset + pos
			if absolutePos >= len(tokens)-1 {
				continue
			}
			tokenLogits := logits[i*vocabSize : (i+1)*vocabSize]
			nll, err := tqeval.NegativeLogLikelihood(tokenLogits, tokens[absolutePos+1])
			if err != nil {
				return 0, 0, err
			}
			if math.IsNaN(nll) || math.IsInf(nll, 0) {
				return 0, 0, fmt.Errorf("invalid nll at token position %d: %v", absolutePos, nll)
			}
			nllSum += nll
			tokenCount++
		}
		offset = end
	}

	if tokenCount == 0 {
		return 0, 0, errors.New("sequence produced no scored tokens")
	}
	return nllSum, tokenCount, nil
}

func evaluateGoSequenceWithReference(candidate ollamamodel.Model, reference ollamamodel.Model, tokens []int, batchSize int) (float64, int, float64, int, error) {
	if err := clearGoCache(candidate); err != nil {
		return 0, 0, 0, 0, err
	}
	if err := clearGoCache(reference); err != nil {
		return 0, 0, 0, 0, err
	}

	var nllSum float64
	var klSum float64
	var tokenCount int
	for offset := 0; offset < len(tokens); {
		end := min(offset+batchSize, len(tokens))
		candidateLogits, outputPositions, candidateVocabSize, err := evaluateGoBatch(candidate, tokens, offset, end)
		if err != nil {
			return 0, 0, 0, 0, err
		}
		referenceLogits, referenceOutputPositions, referenceVocabSize, err := evaluateGoBatch(reference, tokens, offset, end)
		if err != nil {
			return 0, 0, 0, 0, err
		}
		if candidateVocabSize != referenceVocabSize {
			return 0, 0, 0, 0, fmt.Errorf("vocabulary size mismatch: candidate=%d reference=%d", candidateVocabSize, referenceVocabSize)
		}
		if len(outputPositions) != len(referenceOutputPositions) {
			return 0, 0, 0, 0, fmt.Errorf("output count mismatch: candidate=%d reference=%d", len(outputPositions), len(referenceOutputPositions))
		}

		for i, pos := range outputPositions {
			if referenceOutputPositions[i] != pos {
				return 0, 0, 0, 0, fmt.Errorf("output position mismatch at %d: candidate=%d reference=%d", i, pos, referenceOutputPositions[i])
			}
			absolutePos := offset + pos
			if absolutePos >= len(tokens)-1 {
				continue
			}
			candidateTokenLogits := candidateLogits[i*candidateVocabSize : (i+1)*candidateVocabSize]
			referenceTokenLogits := referenceLogits[i*referenceVocabSize : (i+1)*referenceVocabSize]
			nll, err := tqeval.NegativeLogLikelihood(candidateTokenLogits, tokens[absolutePos+1])
			if err != nil {
				return 0, 0, 0, 0, err
			}
			kl, err := tqeval.KLDivergence(referenceTokenLogits, candidateTokenLogits)
			if err != nil {
				return 0, 0, 0, 0, err
			}
			nllSum += nll
			klSum += kl
			tokenCount++
		}
		offset = end
	}

	if tokenCount == 0 {
		return 0, 0, 0, 0, errors.New("sequence produced no scored tokens")
	}
	return nllSum, tokenCount, klSum, tokenCount, nil
}

func evaluateGoBatch(m ollamamodel.Model, tokens []int, offset int, end int) ([]float32, []int, int, error) {
	ctx := m.Backend().NewContext()
	defer ctx.Close()

	inputs := make([]int32, end-offset)
	positions := make([]int32, end-offset)
	sequences := make([]int, end-offset)
	outputPositions := make([]int, 0, end-offset)
	outputs := make([]int32, 0, end-offset)
	for i := range inputs {
		absolutePos := offset + i
		inputs[i] = int32(tokens[absolutePos])
		positions[i] = int32(absolutePos)
		if absolutePos < len(tokens)-1 {
			outputPositions = append(outputPositions, i)
			outputs = append(outputs, int32(i))
		}
	}

	batch := input.Batch{
		Inputs:    ctx.Input().FromInts(inputs, len(inputs)),
		Outputs:   ctx.Input().FromInts(outputs, len(outputs)),
		Positions: positions,
		Sequences: sequences,
	}
	ctx.SetBatchSize(len(inputs))

	out, err := ollamamodel.Forward(ctx, m, batch)
	if err != nil {
		return nil, nil, 0, err
	}
	ctx.Compute(out)

	logits := out.Floats()
	if len(outputPositions) == 0 {
		return logits, outputPositions, 0, nil
	}
	if len(logits)%len(outputPositions) != 0 {
		return nil, nil, 0, fmt.Errorf("logits length %d is not divisible by outputs %d", len(logits), len(outputPositions))
	}
	return logits, outputPositions, len(logits) / len(outputPositions), nil
}

func clearGoCache(m ollamamodel.Model) error {
	if cache := m.Config().Cache; cache != nil {
		return cache.Remove(0, 0, math.MaxInt32)
	}
	return nil
}

func newEvalContext(model *llama.Model, opts options, kvCacheType string) (*llama.Context, error) {
	flashAttention := ml.FlashAttentionDisabled
	if opts.flashAttn {
		flashAttention = ml.FlashAttentionEnabled
	}
	ctxParams := llama.NewContextParams(opts.numCtx, opts.batchSize, 1, opts.threads, flashAttention, kvCacheType)
	return llama.NewContextWithModel(model, ctxParams)
}

func evaluateSequence(ctx *llama.Context, tokens []int, batchSize int) (float64, int, error) {
	ctx.KvCacheClear()

	batch, err := llama.NewBatch(batchSize, 1, 0)
	if err != nil {
		return 0, 0, err
	}
	defer batch.Free()

	var nllSum float64
	var tokenCount int
	for offset := 0; offset < len(tokens); {
		batch.Clear()
		end := min(offset+batchSize, len(tokens))
		for pos := offset; pos < end; pos++ {
			batch.Add(tokens[pos], nil, pos, pos < len(tokens)-1, 0)
		}

		if err := ctx.Decode(batch); err != nil {
			return 0, 0, err
		}
		ctx.Synchronize()

		for i := 0; i < batch.NumTokens(); i++ {
			pos := offset + i
			if pos >= len(tokens)-1 {
				continue
			}
			logits := ctx.GetLogitsIth(i)
			nll, err := tqeval.NegativeLogLikelihood(logits, tokens[pos+1])
			if err != nil {
				return 0, 0, err
			}
			if math.IsNaN(nll) || math.IsInf(nll, 0) {
				return 0, 0, fmt.Errorf("invalid nll at token position %d: %v", pos, nll)
			}
			nllSum += nll
			tokenCount++
		}
		offset = end
	}

	if tokenCount == 0 {
		return 0, 0, errors.New("sequence produced no scored tokens")
	}
	return nllSum, tokenCount, nil
}

func evaluateSequenceWithReference(candidateCtx *llama.Context, referenceCtx *llama.Context, tokens []int, batchSize int) (float64, int, float64, int, error) {
	candidateCtx.KvCacheClear()
	referenceCtx.KvCacheClear()

	candidateBatch, err := llama.NewBatch(batchSize, 1, 0)
	if err != nil {
		return 0, 0, 0, 0, err
	}
	defer candidateBatch.Free()

	referenceBatch, err := llama.NewBatch(batchSize, 1, 0)
	if err != nil {
		return 0, 0, 0, 0, err
	}
	defer referenceBatch.Free()

	var nllSum float64
	var klSum float64
	var tokenCount int
	for offset := 0; offset < len(tokens); {
		end := min(offset+batchSize, len(tokens))
		fillBatch(candidateBatch, tokens, offset, end)
		fillBatch(referenceBatch, tokens, offset, end)

		if err := candidateCtx.Decode(candidateBatch); err != nil {
			return 0, 0, 0, 0, err
		}
		candidateCtx.Synchronize()
		if err := referenceCtx.Decode(referenceBatch); err != nil {
			return 0, 0, 0, 0, err
		}
		referenceCtx.Synchronize()

		for i := 0; i < candidateBatch.NumTokens(); i++ {
			pos := offset + i
			if pos >= len(tokens)-1 {
				continue
			}
			candidateLogits := candidateCtx.GetLogitsIth(i)
			referenceLogits := referenceCtx.GetLogitsIth(i)
			nll, err := tqeval.NegativeLogLikelihood(candidateLogits, tokens[pos+1])
			if err != nil {
				return 0, 0, 0, 0, err
			}
			kl, err := tqeval.KLDivergence(referenceLogits, candidateLogits)
			if err != nil {
				return 0, 0, 0, 0, err
			}
			nllSum += nll
			klSum += kl
			tokenCount++
		}
		offset = end
	}

	if tokenCount == 0 {
		return 0, 0, 0, 0, errors.New("sequence produced no scored tokens")
	}
	return nllSum, tokenCount, klSum, tokenCount, nil
}

func fillBatch(batch *llama.Batch, tokens []int, offset int, end int) {
	batch.Clear()
	for pos := offset; pos < end; pos++ {
		batch.Add(tokens[pos], nil, pos, pos < len(tokens)-1, 0)
	}
}
