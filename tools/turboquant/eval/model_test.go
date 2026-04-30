package eval

import (
	"os"
	"testing"
)

func TestResolveModelPathAcceptsFilePath(t *testing.T) {
	f, err := os.CreateTemp(t.TempDir(), "model-*.gguf")
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}

	got, err := ResolveModelPath(f.Name())
	if err != nil {
		t.Fatalf("ResolveModelPath() error = %v", err)
	}
	if got != f.Name() {
		t.Fatalf("ResolveModelPath() = %q, want %q", got, f.Name())
	}
}
