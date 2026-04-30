package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/ollama/ollama/tools/turboquant/layerselect"
)

type selectedLayers struct {
	LayerCount         int    `json:"layer_count"`
	DType              string `json:"dtype"`
	KeyCacheLayerTypes string `json:"key_cache_layer_types"`
}

func main() {
	var input string
	var top int
	var dtype string
	var format string
	flag.StringVar(&input, "input", "", "layer sweep CSV path")
	flag.IntVar(&top, "top", 4, "number of lowest-KL layers to select")
	flag.StringVar(&dtype, "dtype", "q8_0", "key cache dtype to emit for selected layers")
	flag.StringVar(&format, "format", "spec", "output format: spec or json")
	flag.Parse()

	if err := runSelectLayers(input, top, dtype, format, os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, "turboquant-select-layers:", err)
		os.Exit(1)
	}
}

func runSelectLayers(input string, top int, dtype string, format string, out io.Writer) error {
	if input == "" {
		return errors.New("-input is required")
	}
	f, err := os.Open(input)
	if err != nil {
		return fmt.Errorf("open input: %w", err)
	}
	defer f.Close()

	spec, err := layerselect.SelectLayerSpec(f, top, dtype)
	if err != nil {
		return err
	}

	switch format {
	case "spec":
		_, err = fmt.Fprintln(out, spec)
		return err
	case "json":
		layers := 0
		if spec != "" {
			layers = strings.Count(spec, ",") + 1
		}
		enc := json.NewEncoder(out)
		enc.SetIndent("", "  ")
		return enc.Encode(selectedLayers{
			LayerCount:         layers,
			DType:              dtype,
			KeyCacheLayerTypes: spec,
		})
	default:
		return fmt.Errorf("unsupported -format %q", format)
	}
}
