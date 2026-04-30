package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/ollama/ollama/llama"
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

	llama.BackendInit()
	model, err := llama.LoadModelFromFile(modelPath, llama.ModelParams{
		UseMmap:   true,
		VocabOnly: true,
	})
	if err != nil {
		return err
	}
	defer llama.FreeModel(model)

	tokens, err := model.Tokenize(string(corpus), opts.addSpecial, true)
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
