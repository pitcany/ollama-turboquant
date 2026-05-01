package qwen3next

import (
	"testing"

	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
)

// TestHybridCacheSatisfiesSplitInitInterface guards method promotion via
// embedding: *HybridCache wraps *kvcache.Recurrent (which provides
// InitSplit + SetKeyLayerDTypes). The runner's initKVCache type-asserts
// against these interfaces; if a future refactor breaks promotion, the
// fallback warning at runner/ollamarunner/cache.go fires again and qwen3.5
// / qwen3.6 silently drop from tier-1/2 KV savings to uniform q8_0.
func TestHybridCacheSatisfiesSplitInitInterface(t *testing.T) {
	var c kvcache.Cache = NewHybridCache(nil, 1, 1, 1)
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
