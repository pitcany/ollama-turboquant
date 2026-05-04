package ggml

// #cgo CPPFLAGS: -I${SRCDIR}/ggml/src
// #include <stdint.h>
// #include "ggml.h"
// #include "ggml-quants.h"
//
// // turbo_cpu_fwht is implemented in ggml-turbo-quant.c but has no public header.
// // Declare it locally so the round-trip test can call it without leaking the
// // symbol into ggml-quants.h.
// extern void turbo_cpu_fwht(float * x, int group_size);
import "C"

import "unsafe"

// turbo4_0_64BlockSize returns sizeof(block_turbo4_0_64) for the round-trip
// layout assertion in turbo4_0_64_roundtrip_test.go.
func turbo4_0_64BlockSize() uintptr {
	return unsafe.Sizeof(C.block_turbo4_0_64{})
}

// turbo4_0_64TypeTraits returns (blck_size, type_size) reported by ggml's
// type-traits table for GGML_TYPE_TURBO4_0_64.
func turbo4_0_64TypeTraits() (int, int) {
	return int(C.ggml_blck_size(C.GGML_TYPE_TURBO4_0_64)),
		int(C.ggml_type_size(C.GGML_TYPE_TURBO4_0_64))
}

// quantizeTurbo4_0 packs a flat float32 row into block_turbo4_0 storage and
// returns the round-tripped (dequantized) output. k must be a multiple of 128.
func quantizeTurbo4_0RoundTrip(x []float32) []float32 {
	const blk = 128
	if len(x)%blk != 0 {
		panic("turbo4_0 round trip requires len(x) % 128 == 0")
	}
	q := make([]C.block_turbo4_0, len(x)/blk)
	C.quantize_row_turbo4_0_ref((*C.float)(unsafe.Pointer(&x[0])), &q[0], C.int64_t(len(x)))
	out := make([]float32, len(x))
	C.dequantize_row_turbo4_0(&q[0], (*C.float)(unsafe.Pointer(&out[0])), C.int64_t(len(x)))
	return out
}

// quantizeTurbo4_0_64 packs a flat float32 row into block_turbo4_0_64 storage
// and returns the round-tripped (dequantized) output. k must be a multiple of 64.
func quantizeTurbo4_0_64RoundTrip(x []float32) []float32 {
	const blk = 64
	if len(x)%blk != 0 {
		panic("turbo4_0_64 round trip requires len(x) % 64 == 0")
	}
	q := make([]C.block_turbo4_0_64, len(x)/blk)
	C.quantize_row_turbo4_0_64_ref((*C.float)(unsafe.Pointer(&x[0])), &q[0], C.int64_t(len(x)))
	out := make([]float32, len(x))
	C.dequantize_row_turbo4_0_64(&q[0], (*C.float)(unsafe.Pointer(&out[0])), C.int64_t(len(x)))
	return out
}

// turboCPUFWHT applies turbo's forward Walsh-Hadamard transform in place.
// groupSize must be 64 or 128.
func turboCPUFWHT(x []float32, groupSize int) {
	if len(x)%groupSize != 0 {
		panic("turboCPUFWHT requires len(x) % groupSize == 0")
	}
	for off := 0; off < len(x); off += groupSize {
		C.turbo_cpu_fwht((*C.float)(unsafe.Pointer(&x[off])), C.int(groupSize))
	}
}
