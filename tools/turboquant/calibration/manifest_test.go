package calibration

import "testing"

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
