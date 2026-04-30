package eval

import (
	"encoding/json"
	"fmt"
	"io"
)

const CurrentSnapshotVersion = 1

type Snapshot struct {
	Version       int        `json:"version"`
	Model         string     `json:"model,omitempty"`
	Source        string     `json:"source,omitempty"`
	ContextLength int        `json:"context_length,omitempty"`
	Sequences     []Sequence `json:"sequences"`
}

type Sequence struct {
	ID     string `json:"id,omitempty"`
	Tokens []int  `json:"tokens"`
}

func LoadSnapshot(r io.Reader) (Snapshot, error) {
	var snapshot Snapshot
	if err := json.NewDecoder(r).Decode(&snapshot); err != nil {
		return Snapshot{}, fmt.Errorf("decode token snapshot: %w", err)
	}
	if err := ValidateSnapshot(snapshot); err != nil {
		return Snapshot{}, err
	}
	return snapshot, nil
}

func ValidateSnapshot(snapshot Snapshot) error {
	if snapshot.Version != CurrentSnapshotVersion {
		return fmt.Errorf("unsupported snapshot version %d", snapshot.Version)
	}
	if len(snapshot.Sequences) == 0 {
		return fmt.Errorf("snapshot must contain at least one sequence")
	}
	for i, seq := range snapshot.Sequences {
		if len(seq.Tokens) < 2 {
			return fmt.Errorf("sequence %d must contain at least two tokens", i)
		}
		for j, token := range seq.Tokens {
			if token < 0 {
				return fmt.Errorf("sequence %d token %d is negative", i, j)
			}
		}
	}
	return nil
}
