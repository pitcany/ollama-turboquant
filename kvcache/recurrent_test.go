package kvcache

import (
	"testing"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model/input"
)

// newSplitTestRecurrent constructs a Recurrent cache with conv/recurrent
// dimensions large enough to allocate real buffers and invoke a forward
// pass with a single sequence and a single token.
func newSplitTestRecurrent() *Recurrent {
	return NewRecurrentCache(RecurrentConfig{
		ConvDim:            2,
		ConvChannels:       2,
		RecurrentStateSize: 4,
	})
}

// putOnce drives a single-token, single-sequence forward pass at the
// given layer so the embedded Causal sub-cache materialises K/V tensors
// for that layer. The resulting tensors expose their dtype through
// `cache.kv.keys[layer]` and `cache.kv.values[layer]`.
func putOnce(t *testing.T, ctx ml.Context, cache *Recurrent, layer int) {
	t.Helper()
	cache.SetLayer(layer)
	batch := input.Batch{Positions: []int32{0}, Sequences: []int{0}}
	if err := cache.StartForward(ctx, batch, false); err != nil {
		t.Fatalf("StartForward(layer=%d) error = %v", layer, err)
	}
	key := ctx.FromFloats([]float32{1, 2}, 2, 1, 1)
	value := ctx.FromFloats([]float32{3, 4}, 2, 1, 1)
	cache.Put(ctx, key, value)
}

func TestRecurrentInitSplitForwardsKeyValueDTypesToCausal(t *testing.T) {
	backend := &testBackend{}
	ctx := backend.NewContext()
	cache := newSplitTestRecurrent()
	defer cache.Close()

	cache.InitSplit(backend, ml.DTypeQ80, ml.DTypeTurbo4, 1, 4, 2)

	if cache.kv.KeyDType != ml.DTypeQ80 {
		t.Fatalf("inner Causal KeyDType = %v, want %v", cache.kv.KeyDType, ml.DTypeQ80)
	}
	if cache.kv.ValueDType != ml.DTypeTurbo4 {
		t.Fatalf("inner Causal ValueDType = %v, want %v", cache.kv.ValueDType, ml.DTypeTurbo4)
	}

	putOnce(t, ctx, cache, 0)

	if got := cache.kv.keys[0].DType(); got != ml.DTypeQ80 {
		t.Fatalf("layer 0 key dtype = %v, want %v", got, ml.DTypeQ80)
	}
	if got := cache.kv.values[0].DType(); got != ml.DTypeTurbo4 {
		t.Fatalf("layer 0 value dtype = %v, want %v", got, ml.DTypeTurbo4)
	}
}

func TestRecurrentInitSplitLeavesConvAndRecurrentBuffersF32(t *testing.T) {
	backend := &testBackend{}
	cache := newSplitTestRecurrent()
	defer cache.Close()

	// Even when callers request Turbo* K/V dtypes, the SSM-style state
	// (conv1d + recurrent) must stay at f32 — those buffers are not
	// keyed/valued in the attention sense and must keep numerical
	// stability for the SSM update.
	cache.InitSplit(backend, ml.DTypeTurbo4, ml.DTypeTurbo4, 1, 4, 2)

	if got := cache.convBuffer(0).DType(); got != ml.DTypeF32 {
		t.Fatalf("conv buffer dtype = %v, want %v", got, ml.DTypeF32)
	}
	if got := cache.recurrentBuffer(0).DType(); got != ml.DTypeF32 {
		t.Fatalf("recurrent buffer dtype = %v, want %v", got, ml.DTypeF32)
	}
}

func TestRecurrentSetKeyLayerDTypesForwardsToCausal(t *testing.T) {
	backend := &testBackend{}
	ctx := backend.NewContext()
	cache := newSplitTestRecurrent()
	defer cache.Close()

	cache.InitSplit(backend, ml.DTypeTurbo4, ml.DTypeTurbo4, 1, 4, 2)
	cache.SetKeyLayerDTypes(map[int]ml.DType{0: ml.DTypeQ80})

	putOnce(t, ctx, cache, 0)
	putOnce(t, ctx, cache, 1)

	if got := cache.kv.keys[0].DType(); got != ml.DTypeQ80 {
		t.Fatalf("layer 0 key dtype = %v, want %v (override)", got, ml.DTypeQ80)
	}
	if got := cache.kv.values[0].DType(); got != ml.DTypeTurbo4 {
		t.Fatalf("layer 0 value dtype = %v, want %v (V is uniform)", got, ml.DTypeTurbo4)
	}
	if got := cache.kv.keys[1].DType(); got != ml.DTypeTurbo4 {
		t.Fatalf("layer 1 key dtype = %v, want %v (no override)", got, ml.DTypeTurbo4)
	}
}

func TestRecurrentSetKeyLayerDTypesEmptyClearsOverrides(t *testing.T) {
	backend := &testBackend{}
	cache := newSplitTestRecurrent()
	defer cache.Close()

	cache.InitSplit(backend, ml.DTypeTurbo4, ml.DTypeTurbo4, 1, 4, 2)
	cache.SetKeyLayerDTypes(map[int]ml.DType{0: ml.DTypeQ80})
	cache.SetKeyLayerDTypes(nil)

	if cache.kv.KeyLayerDTypes != nil {
		t.Fatalf("KeyLayerDTypes = %v, want nil after clear", cache.kv.KeyLayerDTypes)
	}
}

func TestRecurrentInitDelegatesToInitSplitForUniformDtype(t *testing.T) {
	backend := &testBackend{}
	ctx := backend.NewContext()
	cache := newSplitTestRecurrent()
	defer cache.Close()

	cache.Init(backend, ml.DTypeF16, 1, 4, 2)
	if cache.kv.KeyDType != ml.DTypeF16 || cache.kv.ValueDType != ml.DTypeF16 {
		t.Fatalf("Init() did not seed K/V dtypes uniformly: K=%v V=%v",
			cache.kv.KeyDType, cache.kv.ValueDType)
	}

	putOnce(t, ctx, cache, 0)
	if got := cache.kv.keys[0].DType(); got != ml.DTypeF16 {
		t.Fatalf("Init() key dtype = %v, want %v", got, ml.DTypeF16)
	}
}

// TestRecurrentSatisfiesSplitInitInterface protects the runner's interface
// assertion (initKVCache: cache.(interface{ InitSplit(...) })) from
// regressing when methods are renamed or moved between Recurrent and its
// embedders.
func TestRecurrentSatisfiesSplitInitInterface(t *testing.T) {
	var c Cache = NewRecurrentCache(RecurrentConfig{ConvDim: 1, ConvChannels: 1, RecurrentStateSize: 1})
	defer c.Close()

	if _, ok := c.(interface {
		InitSplit(ml.Backend, ml.DType, ml.DType, int, int, int)
	}); !ok {
		t.Fatal("*Recurrent does not satisfy the InitSplit interface used by initKVCache")
	}
	if _, ok := c.(interface {
		SetKeyLayerDTypes(map[int]ml.DType)
	}); !ok {
		t.Fatal("*Recurrent does not satisfy the SetKeyLayerDTypes interface used by initKVCache")
	}
}
