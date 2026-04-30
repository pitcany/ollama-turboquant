// turboquant-bench measures prefill and decode throughput for the Ollama Go
// engine across KV cache configurations. It is intentionally a small,
// reproducible scaffold: pick a model, pick a KV cache type (including the
// turboquant-adaptive sentinel and OLLAMA_TURBOQUANT_CALIBRATION-style JSON
// artifacts), then report tokens/sec for prefill and decode plus peak
// per-device memory as observed by the backend.
//
// This command does not load weights end-to-end on a CI machine; it is
// designed to be run manually on the target GPU and the results recorded
// alongside `cmd/turboquant-eval` quality numbers.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"strings"
	"time"

	"github.com/ollama/ollama/ml"
	ollamamodel "github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
	_ "github.com/ollama/ollama/model/models"
	tqeval "github.com/ollama/ollama/tools/turboquant/eval"
	"github.com/ollama/ollama/tools/turboquant/calibration"
)

type options struct {
	model           string
	kvCacheType     string
	calibrationPath string
	promptTokens    int
	decodeTokens    int
	warmup          int
	repeats         int
	numCtx          int
	batchSize       int
	numGPULayers    int
	threads         int
	flashAttn       bool
	format          string
	seed            int64
}

type benchResult struct {
	Model                   string             `json:"model"`
	ModelPath               string             `json:"model_path"`
	KVCacheType             string             `json:"kv_cache_type"`
	EffectiveKVCacheType    string             `json:"effective_kv_cache_type"`
	KeyCacheLayerTypes      string             `json:"key_cache_layer_types,omitempty"`
	CalibrationSource       string             `json:"calibration_source,omitempty"`
	PromptTokens            int                `json:"prompt_tokens"`
	DecodeTokens            int                `json:"decode_tokens"`
	Warmup                  int                `json:"warmup"`
	Repeats                 int                `json:"repeats"`
	PrefillMSPerToken       float64            `json:"prefill_ms_per_token"`
	PrefillTokensPerSec     float64            `json:"prefill_tokens_per_sec"`
	DecodeMSPerToken        float64            `json:"decode_ms_per_token"`
	DecodeTokensPerSec      float64            `json:"decode_tokens_per_sec"`
	BackendDevicePeakBytes  []deviceMemUsage   `json:"backend_device_peak_bytes,omitempty"`
}

type deviceMemUsage struct {
	Name      string `json:"name"`
	Library   string `json:"library"`
	UsedBytes uint64 `json:"used_bytes"`
}

func main() {
	opts := parseOptions(os.Args[1:])
	if err := run(opts); err != nil {
		fmt.Fprintln(os.Stderr, "turboquant-bench:", err)
		os.Exit(1)
	}
}

func parseOptions(args []string) options {
	opts := options{
		promptTokens: 1024,
		decodeTokens: 128,
		warmup:       1,
		repeats:      3,
		numCtx:       2048,
		batchSize:    512,
		numGPULayers: 999,
		threads:      4,
		flashAttn:    true,
		format:       "json",
		seed:         1,
	}
	fs := flag.NewFlagSet("turboquant-bench", flag.ExitOnError)
	fs.StringVar(&opts.model, "model", opts.model, "model name or GGUF path")
	fs.StringVar(&opts.kvCacheType, "kv-cache-type", opts.kvCacheType, "KV cache type (f16, q8_0, q4_0, turbo*, kq8-vturbo4, turboquant-adaptive)")
	fs.StringVar(&opts.calibrationPath, "calibration", opts.calibrationPath, "optional calibration JSON path; overrides kv-cache-type to the artifact's base")
	fs.IntVar(&opts.promptTokens, "prompt-tokens", opts.promptTokens, "number of prompt tokens to time as prefill")
	fs.IntVar(&opts.decodeTokens, "decode-tokens", opts.decodeTokens, "number of decode (single-token) steps to time")
	fs.IntVar(&opts.warmup, "warmup", opts.warmup, "warmup repeats not included in timing")
	fs.IntVar(&opts.repeats, "repeats", opts.repeats, "timing repeats; medians are reported")
	fs.IntVar(&opts.numCtx, "num-ctx", opts.numCtx, "context length")
	fs.IntVar(&opts.batchSize, "batch-size", opts.batchSize, "decode batch size for prefill")
	fs.IntVar(&opts.numGPULayers, "num-gpu-layers", opts.numGPULayers, "number of GPU layers")
	fs.IntVar(&opts.threads, "threads", opts.threads, "CPU threads")
	fs.BoolVar(&opts.flashAttn, "flash-attention", opts.flashAttn, "enable flash attention")
	fs.StringVar(&opts.format, "format", opts.format, "output format: json or text")
	fs.Int64Var(&opts.seed, "seed", opts.seed, "RNG seed for synthetic prompt tokens")
	_ = fs.Parse(args)
	return opts
}

func run(opts options) error {
	if opts.model == "" {
		return errors.New("-model is required")
	}
	if opts.promptTokens <= 0 {
		return errors.New("-prompt-tokens must be positive")
	}
	if opts.decodeTokens < 0 {
		return errors.New("-decode-tokens must be non-negative")
	}
	if opts.numCtx < opts.promptTokens+opts.decodeTokens {
		return fmt.Errorf("-num-ctx (%d) must accommodate prompt+decode tokens (%d)", opts.numCtx, opts.promptTokens+opts.decodeTokens)
	}

	modelPath, err := tqeval.ResolveModelPath(opts.model)
	if err != nil {
		return err
	}

	effectiveKV := opts.kvCacheType
	keyOverrideSpec := ""
	calibSource := ""
	if opts.calibrationPath != "" {
		a, err := calibration.Load(opts.calibrationPath)
		if err != nil {
			return err
		}
		effectiveKV = a.BaseKVCacheType
		keyOverrideSpec = a.KeyCacheLayerTypes
		calibSource = "file:" + opts.calibrationPath
	} else if strings.EqualFold(opts.kvCacheType, "turboquant-adaptive") {
		manifest, err := calibration.LoadEmbeddedManifest()
		if err != nil {
			return err
		}
		// Resolution requires reading model metadata, which we don't have until
		// after backend Load. For bench purposes, default to the manifest
		// fallback unless the user provides --calibration explicitly.
		effectiveKV = manifest.DefaultFallback
		calibSource = "manifest:default-fallback"
	}
	if effectiveKV == "" {
		effectiveKV = "f16"
	}

	flashAttention := ml.FlashAttentionDisabled
	if opts.flashAttn {
		flashAttention = ml.FlashAttentionEnabled
	}
	params := ml.BackendParams{
		AllocMemory:    true,
		NumThreads:     opts.threads,
		FlashAttention: flashAttention,
	}

	m, err := ollamamodel.New(modelPath, params)
	if err != nil {
		return err
	}
	defer m.Backend().Close()
	if err := m.Backend().Load(context.Background(), func(float32) {}); err != nil {
		return err
	}
	if pl, ok := m.(ollamamodel.PostLoader); ok {
		if err := pl.PostLoad(); err != nil {
			return err
		}
	}

	if cache := m.Config().Cache; cache != nil {
		keyDType := dtypeFromString(effectiveKV, true)
		valueDType := dtypeFromString(effectiveKV, false)
		if keyDType == valueDType {
			cache.Init(m.Backend(), keyDType, 1, opts.numCtx, opts.batchSize)
		} else {
			split, ok := cache.(interface {
				InitSplit(ml.Backend, ml.DType, ml.DType, int, int, int)
			})
			if !ok {
				return errors.New("model cache does not support split key/value dtypes")
			}
			split.InitSplit(m.Backend(), keyDType, valueDType, 1, opts.numCtx, opts.batchSize)
		}
		if keyOverrideSpec != "" {
			overrides, err := calibration.ParseKeyLayerOverrides(keyOverrideSpec)
			if err != nil {
				return err
			}
			setter, ok := cache.(interface {
				SetKeyLayerDTypes(map[int]ml.DType)
			})
			if !ok {
				return errors.New("model cache does not support per-layer key dtypes")
			}
			dtypes := make(map[int]ml.DType, len(overrides))
			for layer, name := range overrides {
				dtypes[layer] = dtypeFromString(name, true)
			}
			setter.SetKeyLayerDTypes(dtypes)
		}
	}

	rng := rand.New(rand.NewSource(opts.seed))
	tokens := make([]int32, opts.promptTokens)
	for i := range tokens {
		tokens[i] = int32(rng.Intn(100))
	}

	prefillMS, err := timePrefill(m, tokens, opts.warmup, opts.repeats)
	if err != nil {
		return err
	}
	decodeMS := 0.0
	if opts.decodeTokens > 0 {
		decodeMS, err = timeDecode(m, tokens, opts.decodeTokens, opts.warmup, opts.repeats)
		if err != nil {
			return err
		}
	}

	result := benchResult{
		Model:                opts.model,
		ModelPath:            modelPath,
		KVCacheType:          opts.kvCacheType,
		EffectiveKVCacheType: effectiveKV,
		KeyCacheLayerTypes:   keyOverrideSpec,
		CalibrationSource:    calibSource,
		PromptTokens:         opts.promptTokens,
		DecodeTokens:         opts.decodeTokens,
		Warmup:               opts.warmup,
		Repeats:              opts.repeats,
		PrefillMSPerToken:    prefillMS,
		PrefillTokensPerSec:  msPerTokenToTPS(prefillMS),
		DecodeMSPerToken:     decodeMS,
		DecodeTokensPerSec:   msPerTokenToTPS(decodeMS),
	}
	for _, dev := range m.Backend().BackendMemory().GPUs {
		result.BackendDevicePeakBytes = append(result.BackendDevicePeakBytes, deviceMemUsage{
			Name:      dev.Name,
			Library:   dev.DeviceID.Library,
			UsedBytes: sumDeviceUsage(dev),
		})
	}

	if opts.format == "json" {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(result)
	}
	fmt.Printf("model=%s effective_kv=%s prefill_tps=%.2f decode_tps=%.2f\n",
		opts.model, effectiveKV, result.PrefillTokensPerSec, result.DecodeTokensPerSec)
	return nil
}

func msPerTokenToTPS(msPerToken float64) float64 {
	if msPerToken <= 0 {
		return 0
	}
	return 1000.0 / msPerToken
}

func sumDeviceUsage(dev ml.DeviceMemory) uint64 {
	var total uint64
	for _, w := range dev.Weights {
		total += w
	}
	for _, c := range dev.Cache {
		total += c
	}
	return total
}

func timePrefill(m ollamamodel.Model, tokens []int32, warmup, repeats int) (float64, error) {
	for i := 0; i < warmup; i++ {
		if err := runForward(m, tokens, 0); err != nil {
			return 0, err
		}
	}
	durations := make([]time.Duration, 0, repeats)
	for i := 0; i < repeats; i++ {
		start := time.Now()
		if err := runForward(m, tokens, 0); err != nil {
			return 0, err
		}
		durations = append(durations, time.Since(start))
	}
	median := medianDuration(durations)
	return float64(median.Microseconds()) / float64(len(tokens)) / 1000.0, nil
}

func timeDecode(m ollamamodel.Model, prompt []int32, decodeTokens, warmup, repeats int) (float64, error) {
	durations := make([]time.Duration, 0, repeats)
	for run := 0; run < warmup+repeats; run++ {
		// Reset cache implicitly by rebuilding sequence each iteration. Decode
		// from end of prompt to end+decodeTokens with single-token steps.
		if err := runForward(m, prompt, 0); err != nil {
			return 0, err
		}
		start := time.Now()
		for i := 0; i < decodeTokens; i++ {
			tok := []int32{prompt[len(prompt)-1]}
			if err := runForward(m, tok, len(prompt)+i); err != nil {
				return 0, err
			}
		}
		if run >= warmup {
			durations = append(durations, time.Since(start))
		}
	}
	median := medianDuration(durations)
	return float64(median.Microseconds()) / float64(decodeTokens) / 1000.0, nil
}

func runForward(m ollamamodel.Model, tokens []int32, offset int) error {
	ctx := m.Backend().NewContext()
	defer ctx.Close()
	positions := make([]int32, len(tokens))
	sequences := make([]int, len(tokens))
	outputs := []int32{int32(len(tokens) - 1)}
	for i := range tokens {
		positions[i] = int32(offset + i)
	}
	batch := input.Batch{
		Inputs:    ctx.Input().FromInts(tokens, len(tokens)),
		Outputs:   ctx.Input().FromInts(outputs, len(outputs)),
		Positions: positions,
		Sequences: sequences,
	}
	ctx.SetBatchSize(len(tokens))
	out, err := ollamamodel.Forward(ctx, m, batch)
	if err != nil {
		return err
	}
	ctx.Compute(out)
	return nil
}

func medianDuration(ds []time.Duration) time.Duration {
	if len(ds) == 0 {
		return 0
	}
	cp := make([]time.Duration, len(ds))
	copy(cp, ds)
	for i := 1; i < len(cp); i++ {
		j := i
		for j > 0 && cp[j] < cp[j-1] {
			cp[j], cp[j-1] = cp[j-1], cp[j]
			j--
		}
	}
	return cp[len(cp)/2]
}

func dtypeFromString(s string, isKey bool) ml.DType {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "kq8-vturbo4":
		if isKey {
			return ml.DTypeQ80
		}
		return ml.DTypeTurbo4
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
