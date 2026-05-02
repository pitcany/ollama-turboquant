package nemotronh

import (
	"testing"

	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
)

// TestHybridCacheSatisfiesSplitInitInterface guards method promotion via
// embedding so the runner's split-init interface assertion keeps working
// for Nemotron-H. If embedding stops promoting *kvcache.Recurrent's
// InitSplit / SetKeyLayerDTypes onto *HybridCache, the runner's
// initKVCache falls back to uniform q8_0 with a warning and the model
// silently drops out of tier-1/2 KV savings.
func TestHybridCacheSatisfiesSplitInitInterface(t *testing.T) {
	var c kvcache.Cache = NewHybridCache(1, 1, 1)
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
