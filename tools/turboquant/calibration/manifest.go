package calibration

import (
	"embed"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

//go:embed manifest_data/*
var manifestFS embed.FS

// ManifestEntry maps a (architecture, file_type, head_dim) tuple to a
// bundled calibration artifact. file_type is matched case-insensitively
// against the GGUF file type string (e.g. "Q4_K_M").
type ManifestEntry struct {
	Architecture     string  `json:"architecture"`
	FileType         string  `json:"file_type"`
	HeadDim          int     `json:"head_dim"`
	Artifact         string  `json:"artifact"`
	ModelHint        string  `json:"model_hint,omitempty"`
	Validated        string  `json:"validated,omitempty"`
	Phase0MeanKL     float64 `json:"phase0_mean_kl,omitempty"`
	Phase0Perplexity float64 `json:"phase0_perplexity,omitempty"`
}

// Manifest is the bundled list of calibrations shipped with the binary.
type Manifest struct {
	Version         int             `json:"version"`
	Description     string          `json:"description,omitempty"`
	DefaultFallback string          `json:"default_fallback"`
	Entries         []ManifestEntry `json:"entries"`
}

// LoadEmbeddedManifest reads the bundled manifest. Returns an empty
// manifest with the default fallback set if the embedded file is missing
// (this is allowed so downstream forks can ship without bundled entries).
func LoadEmbeddedManifest() (*Manifest, error) {
	data, err := manifestFS.ReadFile("manifest_data/manifest.json")
	if err != nil {
		return &Manifest{Version: 1, DefaultFallback: "kq8-vturbo4"}, nil
	}
	var m Manifest
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, fmt.Errorf("parse embedded manifest: %w", err)
	}
	if m.Version != 1 {
		return nil, fmt.Errorf("unsupported manifest version %d", m.Version)
	}
	if strings.TrimSpace(m.DefaultFallback) == "" {
		m.DefaultFallback = "kq8-vturbo4"
	}
	return &m, nil
}

// Resolve picks the manifest entry that matches the given architecture,
// file type, and head dimension. Returns (artifact, source, true) on a
// match, where source is a human-readable identifier suitable for logs.
// Returns (nil, "", false) when no entry matches; the caller should fall
// back to manifest.DefaultFallback or another safe preset.
func (m *Manifest) Resolve(architecture, fileType string, headDim int) (*Artifact, string, bool) {
	if m == nil {
		return nil, "", false
	}
	archKey := strings.ToLower(strings.TrimSpace(architecture))
	ftKey := strings.ToLower(strings.TrimSpace(fileType))
	for _, entry := range m.Entries {
		if strings.ToLower(entry.Architecture) != archKey {
			continue
		}
		if strings.ToLower(entry.FileType) != ftKey {
			continue
		}
		if entry.HeadDim != 0 && entry.HeadDim != headDim {
			continue
		}
		artifact, err := loadEmbeddedArtifact(entry.Artifact)
		if err != nil {
			continue
		}
		return artifact, entry.Artifact, true
	}
	return nil, "", false
}

// Preview describes what the runtime would do for a given (arch, file_type,
// head_dim) tuple under the bundled manifest, without actually loading the
// model. Useful for building user-facing summaries (e.g. /api/show).
type Preview struct {
	Source             string  `json:"source"`
	BaseKVCacheType    string  `json:"base_kv_cache_type"`
	KeyCacheLayerTypes string  `json:"key_cache_layer_types,omitempty"`
	BytesPerKVPairF16  float64 `json:"bytes_per_kv_pair_f16"`
	BytesPerKVPair     float64 `json:"bytes_per_kv_pair"`
	SavedPctVsF16      float64 `json:"saved_pct_vs_f16"`
}

// PreviewForModel returns the manifest preview for a model. Returns
// (preview, true) on a manifest match; (preview-with-fallback-info, false)
// otherwise so callers can still report the fallback choice.
func (m *Manifest) PreviewForModel(architecture, fileType string, headDim int) (Preview, bool) {
	a, source, ok := m.Resolve(architecture, fileType, headDim)
	if ok && a != nil {
		bytesPair, bytesPerKV := previewBytes(a.BaseKVCacheType, a.KeyCacheLayerTypes)
		return Preview{
			Source:             "manifest:" + source,
			BaseKVCacheType:    a.BaseKVCacheType,
			KeyCacheLayerTypes: a.KeyCacheLayerTypes,
			BytesPerKVPairF16:  4.0,
			BytesPerKVPair:     bytesPair,
			SavedPctVsF16:      bytesPerKV,
		}, true
	}
	fallback := m.DefaultFallback
	if fallback == "" {
		fallback = "kq8-vturbo4"
	}
	bytesPair, bytesPerKV := previewBytes(fallback, "")
	return Preview{
		Source:            "fallback",
		BaseKVCacheType:   fallback,
		BytesPerKVPairF16: 4.0,
		BytesPerKVPair:    bytesPair,
		SavedPctVsF16:     bytesPerKV,
	}, false
}

// previewBytes returns (totalBytesPerPair, savedPctVsF16) for a base KV
// cache type and an optional per-layer override spec. The per-layer
// estimate assumes the spec dominates only the K side and treats each
// override layer as if it occupied 1 of every N layers, which is fine for
// /api/show summaries but not for accurate per-layer accounting.
func previewBytes(baseKV, layerSpec string) (float64, float64) {
	bK, bV := bytesPerElementForCacheType(baseKV)
	if layerSpec != "" {
		// approximate effect on the K side using the override count and
		// dtype mix; this keeps /api/show informative without recomputing
		// per-layer arrays.
		overrides, err := ParseKeyLayerOverrides(layerSpec)
		if err == nil && len(overrides) > 0 {
			var sum float64
			for _, dtype := range overrides {
				kbytes, _ := bytesPerElementForCacheType(dtype)
				sum += kbytes - bK
			}
			// Spread the delta across an assumed 32-layer model. This is
			// only used for the % savings preview; the live load log
			// reports the exact per-layer total.
			bK += sum / 32.0
		}
	}
	total := bK + bV
	saved := 0.0
	if total < 4.0 {
		saved = (1.0 - total/4.0) * 100
	}
	return total, saved
}

// bytesPerElementForCacheType is a string-only mirror of
// fs/ggml.kvCacheBytesPerElementKV that returns (kBytes, vBytes). It is
// duplicated here to avoid an import cycle with fs/ggml.
func bytesPerElementForCacheType(cacheType string) (float64, float64) {
	switch strings.ToLower(strings.TrimSpace(cacheType)) {
	case "kq8-vturbo4":
		return 1.0, 68.0 / 128.0
	case "q8_0":
		return 1.0, 1.0
	case "q4_0":
		return 0.5, 0.5
	case "turbo2":
		return 34.0 / 128.0, 34.0 / 128.0
	case "turbo3":
		return 50.0 / 128.0, 50.0 / 128.0
	case "turbo4":
		return 68.0 / 128.0, 68.0 / 128.0
	case "f32":
		return 4.0, 4.0
	default:
		return 2.0, 2.0
	}
}

func loadEmbeddedArtifact(name string) (*Artifact, error) {
	if strings.ContainsAny(name, "/\\") {
		return nil, errors.New("artifact name must not contain path separators")
	}
	data, err := manifestFS.ReadFile("manifest_data/" + name)
	if err != nil {
		return nil, fmt.Errorf("read embedded artifact %s: %w", name, err)
	}
	var a Artifact
	if err := json.Unmarshal(data, &a); err != nil {
		return nil, fmt.Errorf("parse embedded artifact %s: %w", name, err)
	}
	if err := a.Validate(); err != nil {
		return nil, fmt.Errorf("invalid embedded artifact %s: %w", name, err)
	}
	return &a, nil
}
