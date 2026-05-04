package ggml

import (
	"testing"
)

// stubModel implements the unexported model interface so tests can build a
// GGML value with hand-crafted KV metadata. Tensors() returns the zero value
// because SupportsKVCacheType only consults KV.
type stubModel struct {
	kv KV
}

func (s stubModel) KV() KV          { return s.kv }
func (s stubModel) Tensors() Tensors { return Tensors{} }

// makeGGML constructs a GGML value backed by stubModel. Architecture and
// block_count default to a single-layer attention model so head-count arrays
// resolve cleanly. Override headDim by passing both attention.key_length and
// attention.value_length, or set them via the second map.
func makeGGML(t *testing.T, kv KV) GGML {
	t.Helper()
	if kv == nil {
		kv = KV{}
	}
	if _, ok := kv["general.architecture"]; !ok {
		kv["general.architecture"] = "test"
	}
	if _, ok := kv["block_count"]; !ok {
		kv["block_count"] = uint32(1)
	}
	return GGML{model: stubModel{kv: kv}}
}

func TestSupportsKVCacheType_TurboHeadDim(t *testing.T) {
	// Base turbo dtypes have blck_size=128 (head_dim>=128, head_dim%128==0).
	// Head-packed turbo*_64 dtypes have blck_size=64 (head_dim>=64, head_dim%64==0).
	cases := []struct {
		name      string
		cacheType string
		kHeadDim  uint32
		vHeadDim  uint32
		want      bool
	}{
		{name: "turbo4 head_dim=128", cacheType: "turbo4", kHeadDim: 128, vHeadDim: 128, want: true},
		{name: "turbo4 head_dim=256", cacheType: "turbo4", kHeadDim: 256, vHeadDim: 256, want: true},
		{name: "turbo4 head_dim=64 rejected", cacheType: "turbo4", kHeadDim: 64, vHeadDim: 64, want: false},
		{name: "turbo4 head_dim=96 rejected (not multiple of 128)", cacheType: "turbo4", kHeadDim: 96, vHeadDim: 96, want: false},
		{name: "turbo2 head_dim=64 rejected", cacheType: "turbo2", kHeadDim: 64, vHeadDim: 64, want: false},
		{name: "turbo3 head_dim=128", cacheType: "turbo3", kHeadDim: 128, vHeadDim: 128, want: true},
		{name: "turbo5 head_dim=128", cacheType: "turbo5", kHeadDim: 128, vHeadDim: 128, want: true},
		{name: "turbo6 head_dim=128", cacheType: "turbo6", kHeadDim: 128, vHeadDim: 128, want: true},
		{name: "turbo4 asymmetric (vHead=64)", cacheType: "turbo4", kHeadDim: 128, vHeadDim: 64, want: false},
		{name: "turbo4 asymmetric (kHead=64)", cacheType: "turbo4", kHeadDim: 64, vHeadDim: 128, want: false},
		{name: "kq8-vturbo4 head_dim=128", cacheType: "kq8-vturbo4", kHeadDim: 128, vHeadDim: 128, want: true},
		{name: "kq8-vturbo4 head_dim=64 rejected", cacheType: "kq8-vturbo4", kHeadDim: 64, vHeadDim: 64, want: false},
		{name: "kturbo6-vturbo4 head_dim=64 rejected", cacheType: "kturbo6-vturbo4", kHeadDim: 64, vHeadDim: 64, want: false},
		// turbo*_64: must accept head_dim=64 and any multiple of 64 on both sides.
		{name: "turbo4_64 head_dim=64", cacheType: "turbo4_64", kHeadDim: 64, vHeadDim: 64, want: true},
		{name: "turbo4_64 head_dim=128", cacheType: "turbo4_64", kHeadDim: 128, vHeadDim: 128, want: true},
		{name: "turbo4_64 head_dim=192", cacheType: "turbo4_64", kHeadDim: 192, vHeadDim: 192, want: true},
		{name: "turbo4_64 head_dim=32 rejected", cacheType: "turbo4_64", kHeadDim: 32, vHeadDim: 32, want: false},
		{name: "turbo4_64 head_dim=96 rejected (not multiple of 64)", cacheType: "turbo4_64", kHeadDim: 96, vHeadDim: 96, want: false},
		{name: "turbo2_64 head_dim=64", cacheType: "turbo2_64", kHeadDim: 64, vHeadDim: 64, want: true},
		{name: "turbo3_64 head_dim=64", cacheType: "turbo3_64", kHeadDim: 64, vHeadDim: 64, want: true},
		{name: "turbo4_64 asymmetric (kHead=128 vHead=64)", cacheType: "turbo4_64", kHeadDim: 128, vHeadDim: 64, want: true},
		{name: "turbo4_64 asymmetric (kHead=32 vHead=64)", cacheType: "turbo4_64", kHeadDim: 32, vHeadDim: 64, want: false},
		{name: "kq8-vturbo4_64 head_dim=64", cacheType: "kq8-vturbo4_64", kHeadDim: 64, vHeadDim: 64, want: true},
		{name: "kq8-vturbo4_64 head_dim=128", cacheType: "kq8-vturbo4_64", kHeadDim: 128, vHeadDim: 128, want: true},
		{name: "q8_0 always supported", cacheType: "q8_0", kHeadDim: 64, vHeadDim: 64, want: true},
		{name: "q4_0 always supported", cacheType: "q4_0", kHeadDim: 64, vHeadDim: 64, want: true},
		{name: "f16 always supported", cacheType: "f16", kHeadDim: 64, vHeadDim: 64, want: true},
		{name: "empty always supported", cacheType: "", kHeadDim: 64, vHeadDim: 64, want: true},
		{name: "unknown rejected", cacheType: "bogus", kHeadDim: 128, vHeadDim: 128, want: false},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// keyValue prefixes the architecture name onto non-general,
			// non-tokenizer keys, so build the KV with that prefix already
			// applied. Use a fixed architecture for the fixture.
			arch := "test"
			f := makeGGML(t, KV{
				"general.architecture":          arch,
				arch + ".attention.key_length":   uint32(tc.kHeadDim),
				arch + ".attention.value_length": uint32(tc.vHeadDim),
				arch + ".embedding_length":       uint32(tc.kHeadDim),
				arch + ".attention.head_count":   uint32(1),
				arch + ".block_count":            uint32(1),
			})
			if got := f.SupportsKVCacheType(tc.cacheType); got != tc.want {
				t.Fatalf("SupportsKVCacheType(%q) with kHead=%d vHead=%d = %v, want %v",
					tc.cacheType, tc.kHeadDim, tc.vHeadDim, got, tc.want)
			}
		})
	}
}

func TestKVCacheBytesPerElementKV(t *testing.T) {
	tests := []struct {
		name  string
		in    string
		wantK float64
		wantV float64
	}{
		{name: "shared f16 default", in: "f16", wantK: 2, wantV: 2},
		{name: "shared turbo4", in: "turbo4", wantK: 68.0 / 128.0, wantV: 68.0 / 128.0},
		{name: "safe split preset", in: "kq8-vturbo4", wantK: 1, wantV: 68.0 / 128.0},
		{name: "k6 split preset", in: "kturbo6-vturbo4", wantK: 100.0 / 128.0, wantV: 68.0 / 128.0},
		{name: "shared turbo2_64", in: "turbo2_64", wantK: 18.0 / 64.0, wantV: 18.0 / 64.0},
		{name: "shared turbo3_64", in: "turbo3_64", wantK: 26.0 / 64.0, wantV: 26.0 / 64.0},
		{name: "shared turbo4_64", in: "turbo4_64", wantK: 34.0 / 64.0, wantV: 34.0 / 64.0},
		{name: "head_dim=64 split preset", in: "kq8-vturbo4_64", wantK: 1, wantV: 34.0 / 64.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotK, gotV := kvCacheBytesPerElementKV(tt.in)
			if gotK != tt.wantK || gotV != tt.wantV {
				t.Fatalf("kvCacheBytesPerElementKV(%q) = (%v, %v), want (%v, %v)", tt.in, gotK, gotV, tt.wantK, tt.wantV)
			}
		})
	}
}

func TestKVCacheBytesPerElement(t *testing.T) {
	if got, want := kvCacheBytesPerElement("q8_0"), 1.0; got != want {
		t.Fatalf("q8_0 bytes/element = %v, want %v", got, want)
	}
	if got, want := kvCacheBytesPerElement("turbo4"), 68.0/128.0; got != want {
		t.Fatalf("turbo4 bytes/element = %v, want %v", got, want)
	}
	if got, want := kvCacheBytesPerElement("turbo5"), 84.0/128.0; got != want {
		t.Fatalf("turbo5 bytes/element = %v, want %v", got, want)
	}
	if got, want := kvCacheBytesPerElement("turbo6"), 100.0/128.0; got != want {
		t.Fatalf("turbo6 bytes/element = %v, want %v", got, want)
	}
	// f16 default fallback
	if got, want := kvCacheBytesPerElement("f16"), 2.0; got != want {
		t.Fatalf("f16 bytes/element = %v, want %v", got, want)
	}
}
