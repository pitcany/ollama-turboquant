package calibration

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestParseKeyLayerOverrides(t *testing.T) {
	got, err := ParseKeyLayerOverrides(" 0:q8_0, 1:Q8_0 ,3:f16, 27:turbo4 ")
	if err != nil {
		t.Fatalf("ParseKeyLayerOverrides() error = %v", err)
	}
	want := map[int]string{0: "q8_0", 1: "q8_0", 3: "f16", 27: "turbo4"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("ParseKeyLayerOverrides() = %#v, want %#v", got, want)
	}
}

func TestParseKeyLayerOverridesEmpty(t *testing.T) {
	got, err := ParseKeyLayerOverrides("")
	if err != nil {
		t.Fatalf("ParseKeyLayerOverrides() error = %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("ParseKeyLayerOverrides(\"\") = %#v, want empty map", got)
	}
}

func TestParseKeyLayerOverridesRejectsBadInput(t *testing.T) {
	cases := []string{
		"missingcolon",
		"abc:q8_0",
		"-1:q8_0",
		"0:",
		"0:q8_0,0:f16", // duplicate conflicting
	}
	for _, spec := range cases {
		t.Run(spec, func(t *testing.T) {
			if _, err := ParseKeyLayerOverrides(spec); err == nil {
				t.Fatalf("expected error for spec %q", spec)
			}
		})
	}
}

func TestCanonicalKeyLayerSpec(t *testing.T) {
	got := CanonicalKeyLayerSpec(map[int]string{27: "q8_0", 0: "q8_0", 3: "f16"})
	want := "0:q8_0,3:f16,27:q8_0"
	if got != want {
		t.Fatalf("CanonicalKeyLayerSpec() = %q, want %q", got, want)
	}
	if got := CanonicalKeyLayerSpec(nil); got != "" {
		t.Fatalf("CanonicalKeyLayerSpec(nil) = %q, want empty", got)
	}
}

func writeTempArtifact(t *testing.T, payload any) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "calibration.json")
	data, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	return path
}

func TestLoadValidArtifact(t *testing.T) {
	path := writeTempArtifact(t, map[string]any{
		"version":               1,
		"model":                 "qwen2.5:7b",
		"base_kv_cache_type":    "turbo4",
		"layer_dtype":           "q8_0",
		"key_cache_layer_types": "0:q8_0,1:q8_0,3:q8_0,27:q8_0",
		"sweep_csv":             "ignored",
	})
	got, err := Load(path)
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	if got.Version != 1 || got.Model != "qwen2.5:7b" || got.BaseKVCacheType != "turbo4" {
		t.Fatalf("Load() unexpected = %#v", got)
	}
	if got.KeyCacheLayerTypes != "0:q8_0,1:q8_0,3:q8_0,27:q8_0" {
		t.Fatalf("Load() KeyCacheLayerTypes = %q", got.KeyCacheLayerTypes)
	}
	overrides, err := ParseKeyLayerOverrides(got.KeyCacheLayerTypes)
	if err != nil {
		t.Fatalf("ParseKeyLayerOverrides() error = %v", err)
	}
	if len(overrides) != 4 {
		t.Fatalf("expected 4 overrides, got %d", len(overrides))
	}
}

func TestLoadRejectsBadVersion(t *testing.T) {
	path := writeTempArtifact(t, map[string]any{
		"version":               2,
		"base_kv_cache_type":    "turbo4",
		"key_cache_layer_types": "0:q8_0",
	})
	if _, err := Load(path); err == nil {
		t.Fatal("expected error for unsupported version")
	}
}

func TestLoadRejectsMissingBaseKV(t *testing.T) {
	path := writeTempArtifact(t, map[string]any{
		"version":               1,
		"key_cache_layer_types": "0:q8_0",
	})
	if _, err := Load(path); err == nil {
		t.Fatal("expected error for missing base_kv_cache_type")
	}
}

func TestLoadRejectsMissingLayerSpec(t *testing.T) {
	path := writeTempArtifact(t, map[string]any{
		"version":            1,
		"base_kv_cache_type": "turbo4",
	})
	if _, err := Load(path); err == nil {
		t.Fatal("expected error for missing key_cache_layer_types")
	}
}

func TestLoadFromCheckedInFixture(t *testing.T) {
	path := "/tmp/turboquant-calibration-qwen25-7b-q4km-full/calibration.json"
	if _, err := os.Stat(path); err != nil {
		t.Skipf("fixture not present: %v", err)
	}
	a, err := Load(path)
	if err != nil {
		t.Fatalf("Load(%q) error = %v", path, err)
	}
	if a.Model != "qwen2.5:7b" || a.BaseKVCacheType != "turbo4" {
		t.Fatalf("unexpected artifact = %#v", a)
	}
	overrides, err := ParseKeyLayerOverrides(a.KeyCacheLayerTypes)
	if err != nil {
		t.Fatalf("ParseKeyLayerOverrides() error = %v", err)
	}
	want := map[int]string{0: "q8_0", 1: "q8_0", 3: "q8_0", 27: "q8_0"}
	if !reflect.DeepEqual(overrides, want) {
		t.Fatalf("overrides = %#v, want %#v", overrides, want)
	}
}
