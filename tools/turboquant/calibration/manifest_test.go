package calibration

import (
	"strings"
	"testing"
)

func TestEmbeddedManifestLoads(t *testing.T) {
	m, err := LoadEmbeddedManifest()
	if err != nil {
		t.Fatalf("LoadEmbeddedManifest() error = %v", err)
	}
	if m.Version != 1 {
		t.Fatalf("Version = %d, want 1", m.Version)
	}
	if m.DefaultFallback != "kq8-vturbo4" {
		t.Fatalf("DefaultFallback = %q, want kq8-vturbo4", m.DefaultFallback)
	}
	if len(m.Entries) == 0 {
		t.Fatal("expected at least one bundled calibration entry")
	}
}

func TestResolveQwen25Q4KM(t *testing.T) {
	m, err := LoadEmbeddedManifest()
	if err != nil {
		t.Fatalf("LoadEmbeddedManifest() error = %v", err)
	}
	a, source, ok := m.Resolve("qwen2", "Q4_K_M", 128)
	if !ok {
		t.Fatal("expected qwen2/Q4_K_M/128 to resolve")
	}
	if a == nil || a.BaseKVCacheType != "turbo4" {
		t.Fatalf("unexpected artifact = %#v", a)
	}
	if a.KeyCacheLayerTypes != "0:q8_0,1:q8_0,3:q8_0,27:q8_0" {
		t.Fatalf("KeyCacheLayerTypes = %q", a.KeyCacheLayerTypes)
	}
	if source == "" {
		t.Fatal("expected non-empty source")
	}
}

func TestResolveCaseInsensitive(t *testing.T) {
	m, err := LoadEmbeddedManifest()
	if err != nil {
		t.Fatalf("LoadEmbeddedManifest() error = %v", err)
	}
	if _, _, ok := m.Resolve("QWEN2", "q4_k_m", 128); !ok {
		t.Fatal("expected case-insensitive match")
	}
}

func TestPreviewForModelMatch(t *testing.T) {
	m, err := LoadEmbeddedManifest()
	if err != nil {
		t.Fatalf("LoadEmbeddedManifest() error = %v", err)
	}
	preview, matched := m.PreviewForModel("qwen2", "Q4_K_M", 128)
	if !matched {
		t.Fatal("expected manifest match for qwen2 Q4_K_M head_dim=128")
	}
	if preview.BaseKVCacheType != "turbo4" {
		t.Fatalf("BaseKVCacheType = %q, want turbo4", preview.BaseKVCacheType)
	}
	if preview.SavedPctVsF16 < 50 {
		t.Fatalf("SavedPctVsF16 = %.2f, want >50%%", preview.SavedPctVsF16)
	}
}

func TestPreviewForModelFallback(t *testing.T) {
	m, err := LoadEmbeddedManifest()
	if err != nil {
		t.Fatalf("LoadEmbeddedManifest() error = %v", err)
	}
	preview, matched := m.PreviewForModel("nonexistent", "Q4_K_M", 128)
	if matched {
		t.Fatal("expected fallback for unknown architecture")
	}
	if preview.BaseKVCacheType != "kq8-vturbo4" {
		t.Fatalf("fallback BaseKVCacheType = %q, want kq8-vturbo4", preview.BaseKVCacheType)
	}
}

func TestResolveMisses(t *testing.T) {
	m, err := LoadEmbeddedManifest()
	if err != nil {
		t.Fatalf("LoadEmbeddedManifest() error = %v", err)
	}
	if _, _, ok := m.Resolve("nonexistent-arch", "Q4_K_M", 128); ok {
		t.Fatal("expected miss for unknown architecture")
	}
	if _, _, ok := m.Resolve("qwen2", "Q4_K_M", 64); ok {
		t.Fatal("expected miss for wrong head_dim")
	}
}

// TestResolveInlineBaseKVCacheType covers the head_dim=64 selection path: an
// entry without a bundled artifact but with an inline BaseKVCacheType still
// resolves, so the manifest can recommend turbo*_64 presets before a real
// calibration JSON exists. Pairs with the head_dim=128 path covered by
// TestResolveQwen25Q4KM (which does have an artifact).
func TestResolveInlineBaseKVCacheType(t *testing.T) {
	m := &Manifest{
		Version: 1,
		Entries: []ManifestEntry{
			{
				Architecture:    "gptoss",
				FileType:        "MXFP4",
				HeadDim:         64,
				BaseKVCacheType: "turbo4_64",
			},
			{
				Architecture:    "gptoss",
				FileType:        "MXFP4",
				HeadDim:         128,
				BaseKVCacheType: "turbo4",
			},
		},
	}

	tests := []struct {
		name        string
		headDim     int
		wantBase    string
		wantSrcHint string
	}{
		{name: "head_dim=64 picks turbo4_64", headDim: 64, wantBase: "turbo4_64", wantSrcHint: "head_dim=64"},
		{name: "head_dim=128 picks turbo4", headDim: 128, wantBase: "turbo4", wantSrcHint: "head_dim=128"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a, source, ok := m.Resolve("gptoss", "MXFP4", tt.headDim)
			if !ok || a == nil {
				t.Fatalf("Resolve(head_dim=%d) failed", tt.headDim)
			}
			if a.BaseKVCacheType != tt.wantBase {
				t.Fatalf("BaseKVCacheType = %q, want %q", a.BaseKVCacheType, tt.wantBase)
			}
			if a.KeyCacheLayerTypes != "" {
				t.Fatalf("inline entry should leave KeyCacheLayerTypes empty, got %q", a.KeyCacheLayerTypes)
			}
			if !strings.Contains(source, tt.wantSrcHint) {
				t.Fatalf("source = %q, want it to contain %q", source, tt.wantSrcHint)
			}
		})
	}
}

func TestResolveBundledArtifactPreferredOverInline(t *testing.T) {
	// When both Artifact and BaseKVCacheType are present on an entry, the
	// bundled artifact wins so calibration data never gets shadowed by the
	// inline placeholder.
	m, err := LoadEmbeddedManifest()
	if err != nil {
		t.Fatalf("LoadEmbeddedManifest() error = %v", err)
	}
	for i := range m.Entries {
		if m.Entries[i].Architecture == "qwen2" && m.Entries[i].FileType == "Q4_K_M" {
			m.Entries[i].BaseKVCacheType = "turbo2_64" // would be wrong if it took precedence
		}
	}
	a, _, ok := m.Resolve("qwen2", "Q4_K_M", 128)
	if !ok || a == nil {
		t.Fatal("expected resolve to succeed")
	}
	if a.BaseKVCacheType != "turbo4" {
		t.Fatalf("BaseKVCacheType = %q, want turbo4 (bundled artifact must win)", a.BaseKVCacheType)
	}
}
