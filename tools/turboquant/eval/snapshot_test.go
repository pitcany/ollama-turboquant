package eval

import (
	"strings"
	"testing"
)

func TestLoadSnapshotValidatesTokenSequences(t *testing.T) {
	snapshot, err := LoadSnapshot(strings.NewReader(`{
		"version": 1,
		"model": "qwen2.5:7b",
		"context_length": 4,
		"sequences": [
			{"id": "a", "tokens": [1, 2, 3]},
			{"id": "b", "tokens": [4, 5]}
		]
	}`))
	if err != nil {
		t.Fatalf("LoadSnapshot() error = %v", err)
	}

	if snapshot.Model != "qwen2.5:7b" {
		t.Fatalf("Model = %q, want qwen2.5:7b", snapshot.Model)
	}
	if len(snapshot.Sequences) != 2 {
		t.Fatalf("Sequences = %d, want 2", len(snapshot.Sequences))
	}
}

func TestLoadSnapshotRejectsShortSequence(t *testing.T) {
	_, err := LoadSnapshot(strings.NewReader(`{
		"version": 1,
		"sequences": [{"tokens": [7]}]
	}`))
	if err == nil {
		t.Fatal("LoadSnapshot() error = nil, want validation error")
	}
}

func TestLoadSnapshotRejectsWrongVersion(t *testing.T) {
	_, err := LoadSnapshot(strings.NewReader(`{
		"version": 2,
		"sequences": [{"tokens": [1, 2]}]
	}`))
	if err == nil {
		t.Fatal("LoadSnapshot() error = nil, want version error")
	}
}
