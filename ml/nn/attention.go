package nn

import (
	"fmt"

	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
)

// turboWHT is the interface for TurboQuant Walsh-Hadamard Transform rotation.
type turboWHT interface {
	TurboWHT(ctx ml.Context, direction int, groupSize int) ml.Tensor
}

// isTurboDType checks if a DType is a TurboQuant KV cache type
func isTurboDType(dt ml.DType) bool {
	switch dt {
	case ml.DTypeTurbo2, ml.DTypeTurbo3, ml.DTypeTurbo4:
		return true
	}
	return false
}

// Attention implements scaled dot-product attention for transformer models:
// Attention(Q, K, V) = softmax(QK^T/√d_k)V
//
// Parameters:
//   - ctx: Context for tensor operations
//   - query: Query tensor (Q) with shape [d_k, heads, seq_len_q]
//   - key: Key tensor (K) with shape [d_k, kv_heads, seq_len_k], can be nil to read from cache only
//   - value: Value tensor (V) with shape [d_v, kv_heads, seq_len_k], can be nil to read from cache only
//   - scale: Scaling factor, typically 1/√d_k where d_k is the key dimension
//   - cache: KV cache to store key/value and get past history, can be nil to only use provided key/value
//
// Returns:
//
//	Attention output with shape [d_v, heads, seq_len_q]
func Attention(ctx ml.Context, query, key, value ml.Tensor, scale float64, cache kvcache.Cache) ml.Tensor {
	return AttentionWithVMLA(ctx, query, key, value, nil, nil, scale, cache)
}

func AttentionWithSinks(ctx ml.Context, query, key, value, sinks ml.Tensor, scale float64, cache kvcache.Cache) ml.Tensor {
	return AttentionWithVMLA(ctx, query, key, value, sinks, nil, scale, cache)
}

func AttentionWithVMLA(ctx ml.Context, query, key, value, sinks ml.Tensor, vmla ml.Tensor, scale float64, cache kvcache.Cache) ml.Tensor {
	ctx.Forward(query)
	if key != nil && value != nil {
		if query.Dim(0) != key.Dim(0) {
			panic(fmt.Errorf("d_k in attention operation does not match between query(%v) and key(%v)", query.Dim(0), key.Dim(0)))
		}

		if key.Dim(1) != value.Dim(1) {
			panic(fmt.Errorf("kv_heads in attention operation does not match between key(%v) and value(%v)", key.Dim(1), value.Dim(1)))
		}

		if key.Dim(2) != value.Dim(2) {
			panic(fmt.Errorf("seq_len_k in attention operation does not match between key(%v) and value(%v)", key.Dim(2), value.Dim(2)))
		}

		ctx.Forward(key, value)
		if cache != nil {
			cache.Put(ctx, key, value)
		}
	} else if cache == nil {
		panic("key & value tensors must be provided if cache is nil")
	}

	var mask ml.Tensor
	if cache != nil {
		key, value, mask = cache.Get(ctx)
	}

	// TurboQuant: detect turbo KV cache by checking the key tensor's DType.
	// This is cache-wrapper-agnostic — works with Causal, HybridCache, WrapperCache, etc.
	// K/V are already WHT-rotated by set_rows during Put(). Apply forward WHT to Q
	// so the dot product Q_rot · K_rot^T preserves correct attention scores.
	turbo := key != nil && isTurboDType(key.DType())
	if turbo {
		if t, ok := query.(turboWHT); ok {
			query = t.TurboWHT(ctx, 0, 0) // direction=0 (forward), groupSize=0 (auto)
		}
	}

	if sdpa, ok := query.(ml.ScaledDotProductAttention); ok {
		cacheConfigApplied := cache != nil
		kqv := sdpa.ScaledDotProductAttention(ctx, key, value, mask, sinks, vmla, scale, cacheConfigApplied)
		// TurboQuant: apply inverse WHT to attention output
		if turbo {
			if t, ok := kqv.(turboWHT); ok {
				kqv = t.TurboWHT(ctx, 1, 0) // direction=1 (inverse)
			}
		}
		return kqv
	} else {
		query = query.Permute(ctx, 0, 2, 1, 3)
		key = key.Permute(ctx, 0, 2, 1, 3)
		value = value.Permute(ctx, 1, 2, 0, 3).Contiguous(ctx)

		kq := key.MulmatFullPrec(ctx, query)

		kq = kq.Scale(ctx, scale)
		if mask != nil {
			kq = kq.Add(ctx, mask)
		}
		kq = kq.Softmax(ctx)

		kqv := value.Mulmat(ctx, kq)

		if vmla != nil {
			kqv = vmla.Mulmat(ctx, kqv)
		}

		return kqv.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
	}
}
