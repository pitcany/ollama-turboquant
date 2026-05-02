package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"

	"github.com/ollama/ollama/llama"
	ollamamodel "github.com/ollama/ollama/model"
	_ "github.com/ollama/ollama/model/models"
	tqeval "github.com/ollama/ollama/tools/turboquant/eval"
)

type options struct {
	model        string
	corpus       string
	sourceLabel  string
	output       string
	sequenceLen  int
	maxSequences int
	addSpecial   bool
}

func main() {
	opts := parseFlags()
	if err := run(opts); err != nil {
		fmt.Fprintln(os.Stderr, "turboquant-tokenize:", err)
		os.Exit(1)
	}
}

func parseFlags() options {
	var opts options
	flag.StringVar(&opts.model, "model", "qwen2.5:7b", "GGUF file path or local Ollama model name")
	flag.StringVar(&opts.corpus, "corpus", "", "UTF-8 text corpus path")
	flag.StringVar(&opts.sourceLabel, "source-label", "", "source description stored in the token snapshot; defaults to -corpus")
	flag.StringVar(&opts.output, "output", "", "output token snapshot path")
	flag.IntVar(&opts.sequenceLen, "sequence-len", 1024, "tokens per sequence")
	flag.IntVar(&opts.maxSequences, "max-sequences", 256, "maximum sequences to emit")
	flag.BoolVar(&opts.addSpecial, "add-special", true, "allow the tokenizer to add model special tokens")
	flag.Parse()
	return opts
}

func run(opts options) error {
	if opts.corpus == "" {
		return fmt.Errorf("-corpus is required")
	}
	if opts.output == "" {
		return fmt.Errorf("-output is required")
	}
	if opts.sequenceLen < 2 {
		return fmt.Errorf("-sequence-len must be at least 2")
	}
	if opts.maxSequences <= 0 {
		return fmt.Errorf("-max-sequences must be positive")
	}

	corpus, err := os.ReadFile(opts.corpus)
	if err != nil {
		return fmt.Errorf("read corpus: %w", err)
	}
	modelPath, err := tqeval.ResolveModelPath(opts.model)
	if err != nil {
		return err
	}

	tokens, err := tokenize(modelPath, string(corpus), opts.addSpecial)
	if err != nil {
		return err
	}

	source := opts.sourceLabel
	if source == "" {
		source = opts.corpus
	}
	snapshot := tqeval.Snapshot{
		Version:       tqeval.CurrentSnapshotVersion,
		Model:         opts.model,
		Source:        source,
		ContextLength: opts.sequenceLen,
		Sequences:     splitTokens(tokens, opts.sequenceLen, opts.maxSequences),
	}
	if err := tqeval.ValidateSnapshot(snapshot); err != nil {
		return err
	}

	out, err := os.Create(opts.output)
	if err != nil {
		return fmt.Errorf("create output: %w", err)
	}
	defer out.Close()

	enc := json.NewEncoder(out)
	enc.SetIndent("", "  ")
	if err := enc.Encode(snapshot); err != nil {
		return fmt.Errorf("write output: %w", err)
	}

	fmt.Fprintf(os.Stderr, "wrote %d sequences to %s\n", len(snapshot.Sequences), opts.output)
	return nil
}

// tokenize encodes corpus through the active engine's tokenizer. It
// prefers the Go runtime (model.NewTextProcessor) so it can handle every
// architecture the runner supports — including new ones like qwen35
// (qwen3.5/3.6 hybrid Causal+SSM) that the legacy llama.cpp loader does
// not register. If the Go runtime cannot construct a tokenizer for this
// GGUF (e.g. the architecture is registered but the model type does not
// implement tokenizer.Tokenizer), it falls back to the llama.cpp path
// the previous version of this command used. That preserves the prior
// success path for older snapshots while unblocking new architectures.
func tokenize(modelPath, corpus string, addSpecial bool) ([]int, error) {
	tp, err := ollamamodel.NewTextProcessor(modelPath)
	if err == nil {
		ids, err := tp.Encode(corpus, addSpecial)
		if err != nil {
			return nil, fmt.Errorf("go-engine tokenize: %w", err)
		}
		out := make([]int, len(ids))
		for i, id := range ids {
			out[i] = int(id)
		}
		return out, nil
	}
	if !errors.Is(err, ollamamodel.ErrUnsupportedTokenizer) && !errors.Is(err, ollamamodel.ErrUnsupportedModel) {
		// A real error from the Go-engine path (corrupt GGUF, missing key,
		// etc.) — fall back rather than failing, so behaviour matches the
		// old llama.cpp-only command on the broadest possible set of
		// inputs.
		fmt.Fprintf(os.Stderr, "turboquant-tokenize: go-engine tokenizer unavailable (%v); falling back to llama.cpp\n", err)
	}

	llama.BackendInit()
	m, err := llama.LoadModelFromFile(modelPath, llama.ModelParams{
		UseMmap:   true,
		VocabOnly: true,
	})
	if err != nil {
		return nil, fmt.Errorf("llama.cpp tokenize: %w", err)
	}
	defer llama.FreeModel(m)
	return m.Tokenize(corpus, addSpecial, true)
}

func splitTokens(tokens []int, sequenceLen int, maxSequences int) []tqeval.Sequence {
	sequences := make([]tqeval.Sequence, 0, maxSequences)
	for offset := 0; offset+2 <= len(tokens) && len(sequences) < maxSequences; offset += sequenceLen {
		end := min(offset+sequenceLen, len(tokens))
		if end-offset < 2 {
			break
		}
		seqTokens := append([]int(nil), tokens[offset:end]...)
		sequences = append(sequences, tqeval.Sequence{
			ID:     fmt.Sprintf("seq-%05d", len(sequences)),
			Tokens: seqTokens,
		})
	}
	return sequences
}
