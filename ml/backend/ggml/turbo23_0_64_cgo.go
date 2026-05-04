package ggml

// #cgo CPPFLAGS: -I${SRCDIR}/ggml/src
// #include <stdint.h>
// #include "ggml.h"
// #include "ggml-quants.h"
import "C"

import "unsafe"

// turbo2_0_64BlockSize / turbo3_0_64BlockSize return sizeof(block_*) for
// the round-trip layout assertions in turbo23_0_64_roundtrip_test.go.
func turbo2_0_64BlockSize() uintptr {
	return unsafe.Sizeof(C.block_turbo2_0_64{})
}

func turbo3_0_64BlockSize() uintptr {
	return unsafe.Sizeof(C.block_turbo3_0_64{})
}

// turbo2_0_64TypeTraits / turbo3_0_64TypeTraits return (blck_size, type_size)
// reported by ggml's type-traits table.
func turbo2_0_64TypeTraits() (int, int) {
	return int(C.ggml_blck_size(C.GGML_TYPE_TURBO2_0_64)),
		int(C.ggml_type_size(C.GGML_TYPE_TURBO2_0_64))
}

func turbo3_0_64TypeTraits() (int, int) {
	return int(C.ggml_blck_size(C.GGML_TYPE_TURBO3_0_64)),
		int(C.ggml_type_size(C.GGML_TYPE_TURBO3_0_64))
}

// quantizeTurbo2_0_64RoundTrip / quantizeTurbo3_0_64RoundTrip pack a flat
// float32 row through quant→dequant. k must be a multiple of 64.
func quantizeTurbo2_0_64RoundTrip(x []float32) []float32 {
	const blk = 64
	if len(x)%blk != 0 {
		panic("turbo2_0_64 round trip requires len(x) % 64 == 0")
	}
	q := make([]C.block_turbo2_0_64, len(x)/blk)
	C.quantize_row_turbo2_0_64_ref((*C.float)(unsafe.Pointer(&x[0])), &q[0], C.int64_t(len(x)))
	out := make([]float32, len(x))
	C.dequantize_row_turbo2_0_64(&q[0], (*C.float)(unsafe.Pointer(&out[0])), C.int64_t(len(x)))
	return out
}

func quantizeTurbo3_0_64RoundTrip(x []float32) []float32 {
	const blk = 64
	if len(x)%blk != 0 {
		panic("turbo3_0_64 round trip requires len(x) % 64 == 0")
	}
	q := make([]C.block_turbo3_0_64, len(x)/blk)
	C.quantize_row_turbo3_0_64_ref((*C.float)(unsafe.Pointer(&x[0])), &q[0], C.int64_t(len(x)))
	out := make([]float32, len(x))
	C.dequantize_row_turbo3_0_64(&q[0], (*C.float)(unsafe.Pointer(&out[0])), C.int64_t(len(x)))
	return out
}

// turbo2_0 / turbo3_0 baseline round trips at head_dim=128, used by the
// PR-4 round-trip tests to set ratio caps relative to the existing variant.
// Note: turbo2_0_ref and turbo3_0_ref use a CPU-side group_size global; we
// force group_size = 128 by passing a 128-aligned k.

func quantizeTurbo2_0RoundTrip(x []float32) []float32 {
	const blk = 128
	if len(x)%blk != 0 {
		panic("turbo2_0 round trip requires len(x) % 128 == 0")
	}
	q := make([]C.block_turbo2_0, len(x)/blk)
	C.quantize_row_turbo2_0_ref((*C.float)(unsafe.Pointer(&x[0])), &q[0], C.int64_t(len(x)))
	out := make([]float32, len(x))
	C.dequantize_row_turbo2_0(&q[0], (*C.float)(unsafe.Pointer(&out[0])), C.int64_t(len(x)))
	return out
}

func quantizeTurbo3_0RoundTrip(x []float32) []float32 {
	const blk = 128
	if len(x)%blk != 0 {
		panic("turbo3_0 round trip requires len(x) % 128 == 0")
	}
	q := make([]C.block_turbo3_0, len(x)/blk)
	C.quantize_row_turbo3_0_ref((*C.float)(unsafe.Pointer(&x[0])), &q[0], C.int64_t(len(x)))
	out := make([]float32, len(x))
	C.dequantize_row_turbo3_0(&q[0], (*C.float)(unsafe.Pointer(&out[0])), C.int64_t(len(x)))
	return out
}
