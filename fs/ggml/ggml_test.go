package ggml

import (
	"testing"
)

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
	// f16 default fallback
	if got, want := kvCacheBytesPerElement("f16"), 2.0; got != want {
		t.Fatalf("f16 bytes/element = %v, want %v", got, want)
	}
}
