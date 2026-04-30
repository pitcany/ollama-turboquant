package layerselect

import (
	"encoding/csv"
	"errors"
	"fmt"
	"io"
	"sort"
	"strconv"
	"strings"
)

type layerScore struct {
	layer  int
	meanKL float64
}

func SelectLayerSpec(r io.Reader, top int, dtype string) (string, error) {
	if top <= 0 {
		return "", errors.New("-top must be positive")
	}
	dtype = strings.TrimSpace(dtype)
	if dtype == "" {
		return "", errors.New("-dtype is required")
	}

	reader := csv.NewReader(r)
	header, err := reader.Read()
	if err != nil {
		return "", fmt.Errorf("read header: %w", err)
	}
	cols := map[string]int{}
	for i, name := range header {
		cols[name] = i
	}
	for _, name := range []string{"layer", "mean_kl", "status"} {
		if _, ok := cols[name]; !ok {
			return "", fmt.Errorf("missing required column %q", name)
		}
	}

	var scores []layerScore
	for {
		row, err := reader.Read()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return "", fmt.Errorf("read row: %w", err)
		}
		if row[cols["status"]] != "ok" {
			continue
		}
		layer, err := strconv.Atoi(row[cols["layer"]])
		if err != nil {
			return "", fmt.Errorf("invalid layer %q: %w", row[cols["layer"]], err)
		}
		meanKL, err := strconv.ParseFloat(row[cols["mean_kl"]], 64)
		if err != nil {
			return "", fmt.Errorf("invalid mean_kl for layer %d: %w", layer, err)
		}
		scores = append(scores, layerScore{layer: layer, meanKL: meanKL})
	}
	if len(scores) == 0 {
		return "", errors.New("input contains no successful layer rows")
	}

	sort.Slice(scores, func(i, j int) bool {
		if scores[i].meanKL == scores[j].meanKL {
			return scores[i].layer < scores[j].layer
		}
		return scores[i].meanKL < scores[j].meanKL
	})
	if top > len(scores) {
		top = len(scores)
	}

	layers := make([]int, 0, top)
	for _, score := range scores[:top] {
		layers = append(layers, score.layer)
	}
	sort.Ints(layers)

	parts := make([]string, 0, len(layers))
	for _, layer := range layers {
		parts = append(parts, fmt.Sprintf("%d:%s", layer, dtype))
	}
	return strings.Join(parts, ","), nil
}
