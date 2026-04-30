package main

import (
	"reflect"
	"testing"

	"github.com/ollama/ollama/ml"
	tqeval "github.com/ollama/ollama/tools/turboquant/eval"
)

func TestGoKVCacheDType(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want ml.DType
	}{
		{name: "default", in: "", want: ml.DTypeF16},
		{name: "f16", in: "f16", want: ml.DTypeF16},
		{name: "q8", in: "q8_0", want: ml.DTypeQ80},
		{name: "q4", in: "q4_0", want: ml.DTypeQ40},
		{name: "turbo2", in: "turbo2", want: ml.DTypeTurbo2},
		{name: "turbo3", in: "turbo3", want: ml.DTypeTurbo3},
		{name: "turbo4", in: "turbo4", want: ml.DTypeTurbo4},
		{name: "case insensitive", in: "TURBO4", want: ml.DTypeTurbo4},
		{name: "unknown fallback", in: "not-a-cache", want: ml.DTypeF16},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := goKVCacheDType(tt.in); got != tt.want {
				t.Fatalf("goKVCacheDType(%q) = %v, want %v", tt.in, got, tt.want)
			}
		})
	}
}

func TestGoKVCacheTypes(t *testing.T) {
	tests := []struct {
		name  string
		opts  options
		wantK string
		wantV string
	}{
		{
			name:  "uses shared kv cache type by default",
			opts:  options{kvCacheType: "turbo4"},
			wantK: "turbo4",
			wantV: "turbo4",
		},
		{
			name:  "overrides key and value independently",
			opts:  options{kvCacheType: "turbo4", keyCacheType: "f16", valueCacheType: "q8_0"},
			wantK: "f16",
			wantV: "q8_0",
		},
		{
			name:  "overrides only key",
			opts:  options{kvCacheType: "turbo4", keyCacheType: "f16"},
			wantK: "f16",
			wantV: "turbo4",
		},
		{
			name:  "uses named safe preset",
			opts:  options{kvCachePreset: "kq8-vturbo4"},
			wantK: "q8_0",
			wantV: "turbo4",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotK, gotV, err := goKVCacheTypes(tt.opts)
			if err != nil {
				t.Fatalf("goKVCacheTypes() error = %v", err)
			}
			if gotK != tt.wantK || gotV != tt.wantV {
				t.Fatalf("goKVCacheTypes() = (%q, %q), want (%q, %q)", gotK, gotV, tt.wantK, tt.wantV)
			}
		})
	}
}

func TestKVCachePresetRejectsExplicitKeyValueOverrides(t *testing.T) {
	_, _, err := goKVCacheTypes(options{
		kvCachePreset: "kq8-vturbo4",
		keyCacheType:  "f16",
	})
	if err == nil {
		t.Fatal("expected preset and explicit key cache override to be rejected")
	}

	_, _, err = goKVCacheTypes(options{
		kvCachePreset:  "kq8-vturbo4",
		valueCacheType: "f16",
	})
	if err == nil {
		t.Fatal("expected preset and explicit value cache override to be rejected")
	}
}

func TestKVCachePresetRejectsUnknownPreset(t *testing.T) {
	_, _, err := goKVCacheTypes(options{kvCachePreset: "not-a-preset"})
	if err == nil {
		t.Fatal("expected unknown preset to be rejected")
	}
}

func TestValidateOptionsRejectsGoTurboWithoutFlashAttention(t *testing.T) {
	err := validateOptions(options{
		engine:      "go",
		snapshot:    "tokens.json",
		kvCacheType: "turbo4",
		numCtx:      16,
		batchSize:   1,
		threads:     1,
		flashAttn:   false,
	})
	if err == nil {
		t.Fatal("expected Turbo cache without flash attention to be rejected")
	}
}

func TestValidateOptionsRejectsKVCachePresetForLlamaEngine(t *testing.T) {
	err := validateOptions(options{
		engine:        "llama",
		snapshot:      "tokens.json",
		kvCachePreset: "kq8-vturbo4",
		numCtx:        16,
		batchSize:     1,
		threads:       1,
		flashAttn:     true,
	})
	if err == nil {
		t.Fatal("expected kv cache preset on llama engine to be rejected")
	}
}

func TestParseKeyCacheLayerTypes(t *testing.T) {
	got, err := parseKeyCacheLayerTypes("0:f16,27:turbo4,5:q8_0")
	if err != nil {
		t.Fatalf("parseKeyCacheLayerTypes() error = %v", err)
	}
	want := map[int]ml.DType{
		0:  ml.DTypeF16,
		27: ml.DTypeTurbo4,
		5:  ml.DTypeQ80,
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("parseKeyCacheLayerTypes() = %#v, want %#v", got, want)
	}
}

func TestParseKeyCacheLayerTypesRejectsInvalidSpec(t *testing.T) {
	tests := []string{"bad", "-1:f16", "0", "0:", ":f16", "0:not-a-cache"}
	for _, spec := range tests {
		t.Run(spec, func(t *testing.T) {
			if _, err := parseKeyCacheLayerTypes(spec); err == nil {
				t.Fatal("expected error")
			}
		})
	}
}

func TestKeyCacheLayerPresetMapsAdaptivePolicy(t *testing.T) {
	got, err := keyCacheLayerDTypes(options{keyCacheLayerPreset: "qwen2.5-7b-q4km-adaptive"})
	if err != nil {
		t.Fatalf("keyCacheLayerDTypes() error = %v", err)
	}
	want := map[int]ml.DType{
		0:  ml.DTypeQ80,
		1:  ml.DTypeQ80,
		3:  ml.DTypeQ80,
		27: ml.DTypeQ80,
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("keyCacheLayerDTypes() = %#v, want %#v", got, want)
	}
}

func TestKeyCacheLayerPresetRejectsExplicitLayerTypes(t *testing.T) {
	_, err := keyCacheLayerDTypes(options{
		keyCacheLayerPreset: "qwen2.5-7b-q4km-adaptive",
		keyCacheLayerTypes:  "0:f16",
	})
	if err == nil {
		t.Fatal("expected preset and explicit layer types to be rejected")
	}
}

func TestKeyCacheLayerPresetRejectsUnknownPreset(t *testing.T) {
	_, err := keyCacheLayerDTypes(options{keyCacheLayerPreset: "not-a-preset"})
	if err == nil {
		t.Fatal("expected unknown preset to be rejected")
	}
}

func TestValidateOptionsRejectsTurboLayerOverridesWithoutFlashAttention(t *testing.T) {
	err := validateOptions(options{
		engine:             "go",
		snapshot:           "tokens.json",
		kvCacheType:        "f16",
		keyCacheLayerTypes: "0:turbo4",
		numCtx:             16,
		batchSize:          1,
		threads:            1,
		flashAttn:          false,
	})
	if err == nil {
		t.Fatal("expected Turbo layer override without flash attention to be rejected")
	}
}

func TestValidateOptionsRejectsLayerOverridesForLlamaEngine(t *testing.T) {
	err := validateOptions(options{
		engine:             "llama",
		snapshot:           "tokens.json",
		kvCacheType:        "turbo4",
		keyCacheLayerTypes: "0:f16",
		numCtx:             16,
		batchSize:          1,
		threads:            1,
		flashAttn:          true,
	})
	if err == nil {
		t.Fatal("expected layer overrides on llama engine to be rejected")
	}
}

func TestValidateOptionsRejectsLayerPresetForLlamaEngine(t *testing.T) {
	err := validateOptions(options{
		engine:              "llama",
		snapshot:            "tokens.json",
		kvCacheType:         "turbo4",
		keyCacheLayerPreset: "qwen2.5-7b-q4km-adaptive",
		numCtx:              16,
		batchSize:           1,
		threads:             1,
		flashAttn:           true,
	})
	if err == nil {
		t.Fatal("expected layer preset on llama engine to be rejected")
	}
}

func TestEvaluateRejectsUnsupportedEngine(t *testing.T) {
	_, err := evaluate("unused.gguf", tqeval.Snapshot{}, options{engine: "bogus"})
	if err == nil {
		t.Fatal("expected unsupported engine error")
	}
}

func TestValidateOptionsAcceptsResidualWindow(t *testing.T) {
	err := validateOptions(options{
		engine:                 "go",
		snapshot:               "tokens.json",
		kvCacheType:            "f16",
		keyCacheType:           "f16",
		valueCacheType:         "turbo4",
		keyCacheResidualWindow: 64,
		keyCacheResidualDType:  "turbo4",
		numCtx:                 16,
		batchSize:              1,
		threads:                1,
		flashAttn:              true,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestValidateOptionsRejectsResidualWindowMissingDType(t *testing.T) {
	err := validateOptions(options{
		engine:                 "go",
		snapshot:               "tokens.json",
		kvCacheType:            "f16",
		keyCacheResidualWindow: 64,
		numCtx:                 16,
		batchSize:              1,
		threads:                1,
		flashAttn:              true,
	})
	if err == nil {
		t.Fatal("expected residual window without residual dtype to be rejected")
	}
}

func TestValidateOptionsRejectsResidualDTypeMissingWindow(t *testing.T) {
	err := validateOptions(options{
		engine:                "go",
		snapshot:              "tokens.json",
		kvCacheType:           "f16",
		keyCacheResidualDType: "turbo4",
		numCtx:                16,
		batchSize:             1,
		threads:               1,
		flashAttn:             true,
	})
	if err == nil {
		t.Fatal("expected residual dtype without residual window to be rejected")
	}
}

func TestValidateOptionsRejectsResidualWindowForLlamaEngine(t *testing.T) {
	err := validateOptions(options{
		engine:                 "llama",
		snapshot:               "tokens.json",
		kvCacheType:            "f16",
		keyCacheResidualWindow: 64,
		keyCacheResidualDType:  "turbo4",
		numCtx:                 16,
		batchSize:              1,
		threads:                1,
		flashAttn:              true,
	})
	if err == nil {
		t.Fatal("expected residual window on llama engine to be rejected")
	}
}

func TestValidateOptionsRejectsResidualWindowWithoutFlashAttention(t *testing.T) {
	err := validateOptions(options{
		engine:                 "go",
		snapshot:               "tokens.json",
		kvCacheType:            "f16",
		keyCacheResidualWindow: 64,
		keyCacheResidualDType:  "turbo4",
		numCtx:                 16,
		batchSize:              1,
		threads:                1,
		flashAttn:              false,
	})
	if err == nil {
		t.Fatal("expected residual window without flash attention to be rejected")
	}
}

func TestValidateOptionsRejectsResidualWindowWithNonTurboDType(t *testing.T) {
	err := validateOptions(options{
		engine:                 "go",
		snapshot:               "tokens.json",
		kvCacheType:            "f16",
		keyCacheResidualWindow: 64,
		keyCacheResidualDType:  "q8_0",
		numCtx:                 16,
		batchSize:              1,
		threads:                1,
		flashAttn:              true,
	})
	if err == nil {
		t.Fatal("expected residual dtype that is not Turbo to be rejected")
	}
}

func TestValidateOptionsRejectsResidualWindowWithLowPrecisionKey(t *testing.T) {
	err := validateOptions(options{
		engine:                 "go",
		snapshot:               "tokens.json",
		kvCacheType:            "turbo4",
		keyCacheResidualWindow: 64,
		keyCacheResidualDType:  "turbo4",
		numCtx:                 16,
		batchSize:              1,
		threads:                1,
		flashAttn:              true,
	})
	if err == nil {
		t.Fatal("expected residual window with turbo key cache type to be rejected")
	}
}

func TestValidateOptionsRejectsResidualWindowWithLayerOverrides(t *testing.T) {
	err := validateOptions(options{
		engine:                 "go",
		snapshot:               "tokens.json",
		kvCacheType:            "f16",
		keyCacheLayerTypes:     "0:turbo4",
		keyCacheResidualWindow: 64,
		keyCacheResidualDType:  "turbo4",
		numCtx:                 16,
		batchSize:              1,
		threads:                1,
		flashAttn:              true,
	})
	if err == nil {
		t.Fatal("expected residual window with per-layer overrides to be rejected")
	}
}

func TestIsResidualWindowKeyCacheName(t *testing.T) {
	tests := map[string]bool{
		"f16":    true,
		"q8_0":   true,
		"q4_0":   false,
		"turbo4": false,
		"":       false,
	}
	for in, want := range tests {
		if got := isResidualWindowKeyCacheName(in); got != want {
			t.Fatalf("isResidualWindowKeyCacheName(%q) = %v, want %v", in, got, want)
		}
	}
}
