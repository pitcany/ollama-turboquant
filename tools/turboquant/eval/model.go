package eval

import (
	"fmt"
	"os"

	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

func ResolveModelPath(nameOrPath string) (string, error) {
	if info, err := os.Stat(nameOrPath); err == nil {
		if info.IsDir() {
			return "", fmt.Errorf("model path %q is a directory", nameOrPath)
		}
		return nameOrPath, nil
	}

	n := model.ParseName(nameOrPath)
	mf, err := manifest.ParseNamedManifest(n)
	if err != nil {
		return "", fmt.Errorf("resolve model %q: %w", nameOrPath, err)
	}
	for _, layer := range mf.Layers {
		if layer.MediaType == "application/vnd.ollama.image.model" {
			path, err := manifest.BlobsPath(layer.Digest)
			if err != nil {
				return "", err
			}
			return path, nil
		}
	}
	return "", fmt.Errorf("model %q has no GGUF model layer", nameOrPath)
}
