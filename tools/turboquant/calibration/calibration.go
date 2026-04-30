// Package calibration loads TurboQuant calibration artifacts produced by
// cmd/turboquant-calibrate so that the production runtime and memory
// estimator can apply per-layer key-cache dtype overrides without invoking
// the eval harness.
package calibration

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
)

// Artifact is the runtime-relevant subset of the calibration JSON written by
// cmd/turboquant-calibrate. Unknown fields are ignored so writers can keep
// adding metadata without breaking the loader.
type Artifact struct {
	Version            int    `json:"version"`
	Model              string `json:"model"`
	BaseKVCacheType    string `json:"base_kv_cache_type"`
	LayerDType         string `json:"layer_dtype"`
	KeyCacheLayerTypes string `json:"key_cache_layer_types"`
}

// Load reads and validates a calibration artifact from path.
func Load(path string) (*Artifact, error) {
	if path == "" {
		return nil, errors.New("calibration path is empty")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read calibration %s: %w", path, err)
	}
	var a Artifact
	if err := json.Unmarshal(data, &a); err != nil {
		return nil, fmt.Errorf("parse calibration %s: %w", path, err)
	}
	if err := a.Validate(); err != nil {
		return nil, fmt.Errorf("invalid calibration %s: %w", path, err)
	}
	return &a, nil
}

// Validate checks that an artifact has the runtime-required fields.
func (a *Artifact) Validate() error {
	if a == nil {
		return errors.New("artifact is nil")
	}
	if a.Version != 1 {
		return fmt.Errorf("unsupported version %d (expected 1)", a.Version)
	}
	if strings.TrimSpace(a.BaseKVCacheType) == "" {
		return errors.New("base_kv_cache_type is required")
	}
	if strings.TrimSpace(a.KeyCacheLayerTypes) == "" {
		return errors.New("key_cache_layer_types is required")
	}
	if _, err := ParseKeyLayerOverrides(a.KeyCacheLayerTypes); err != nil {
		return err
	}
	return nil
}

// ParseKeyLayerOverrides converts a "0:q8_0,3:f16" spec into a map.
// Whitespace is tolerated; layer indices must be non-negative.
func ParseKeyLayerOverrides(spec string) (map[int]string, error) {
	out := map[int]string{}
	spec = strings.TrimSpace(spec)
	if spec == "" {
		return out, nil
	}
	for _, raw := range strings.Split(spec, ",") {
		item := strings.TrimSpace(raw)
		if item == "" {
			continue
		}
		layerStr, dtypeStr, ok := strings.Cut(item, ":")
		if !ok {
			return nil, fmt.Errorf("invalid key_cache_layer_types item %q, want layer:dtype", item)
		}
		layer, err := strconv.Atoi(strings.TrimSpace(layerStr))
		if err != nil {
			return nil, fmt.Errorf("invalid layer index %q: %w", layerStr, err)
		}
		if layer < 0 {
			return nil, fmt.Errorf("layer index must be non-negative, got %d", layer)
		}
		dtype := strings.ToLower(strings.TrimSpace(dtypeStr))
		if dtype == "" {
			return nil, fmt.Errorf("invalid key_cache_layer_types item %q, dtype is empty", item)
		}
		if existing, ok := out[layer]; ok && existing != dtype {
			return nil, fmt.Errorf("duplicate layer %d with conflicting dtypes %q and %q", layer, existing, dtype)
		}
		out[layer] = dtype
	}
	return out, nil
}

// CanonicalKeyLayerSpec returns the spec sorted by ascending layer index, so
// the runtime always logs/normalizes overrides the same way.
func CanonicalKeyLayerSpec(overrides map[int]string) string {
	if len(overrides) == 0 {
		return ""
	}
	layers := make([]int, 0, len(overrides))
	for layer := range overrides {
		layers = append(layers, layer)
	}
	sort.Ints(layers)
	parts := make([]string, 0, len(layers))
	for _, layer := range layers {
		parts = append(parts, fmt.Sprintf("%d:%s", layer, overrides[layer]))
	}
	return strings.Join(parts, ",")
}
