package lfm2

import (
	"testing"

	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
)

func TestHybridCache_New(t *testing.T) {
	cache := NewHybridCache(nil, 512, 2)
	if cache == nil {
		t.Fatal("expected cache to be created")
	}

	if cache.Recurrent == nil {
		t.Fatal("expected embedded recurrent cache to be created")
	}
}

func TestHybridCache_ImplementsCheckpointCache(t *testing.T) {
	cache := NewHybridCache(nil, 512, 2)

	if _, ok := any(cache).(kvcache.CheckpointCache); !ok {
		t.Fatal("expected HybridCache to implement CheckpointCache")
	}
}

func TestHybridCache_DefaultBatchState(t *testing.T) {
	cache := NewHybridCache(nil, 512, 2)

	if got := cache.numSeqs(); got != 0 {
		t.Fatalf("expected 0 sequences before StartForward, got %d", got)
	}

	if got := cache.seqTokens(); got != 0 {
		t.Fatalf("expected 0 sequence tokens before StartForward, got %d", got)
	}

	if cache.IsSupportedForBatch() {
		t.Fatal("expected unsupported batch layout before StartForward")
	}
}

// TestHybridCache_SatisfiesSplitInitInterface guards method promotion via
// embedding so the runner's split-init interface assertion keeps working
// for LFM2 (otherwise the runner falls back to uniform q8_0 and silently
// drops tier-1/2 KV savings on hybrid models).
func TestHybridCache_SatisfiesSplitInitInterface(t *testing.T) {
	var c kvcache.Cache = NewHybridCache(nil, 512, 2)
	defer c.Close()

	if _, ok := c.(interface {
		InitSplit(ml.Backend, ml.DType, ml.DType, int, int, int)
	}); !ok {
		t.Fatal("*HybridCache does not satisfy the InitSplit interface used by initKVCache")
	}
	if _, ok := c.(interface {
		SetKeyLayerDTypes(map[int]ml.DType)
	}); !ok {
		t.Fatal("*HybridCache does not satisfy the SetKeyLayerDTypes interface used by initKVCache")
	}
}
