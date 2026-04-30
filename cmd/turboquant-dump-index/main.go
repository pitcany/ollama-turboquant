package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/ollama/ollama/tools/turboquant/dumpindex"
)

type options struct {
	dir    string
	format string
}

func main() {
	opts := parseFlags()
	if err := run(opts); err != nil {
		fmt.Fprintln(os.Stderr, "turboquant-dump-index:", err)
		os.Exit(1)
	}
}

func parseFlags() options {
	var opts options
	flag.StringVar(&opts.dir, "dir", "", "TurboQuant dump directory containing .txt metadata sidecars")
	flag.StringVar(&opts.format, "format", "csv", "output format: csv or json")
	flag.Parse()
	return opts
}

func run(opts options) error {
	if opts.dir == "" {
		return fmt.Errorf("-dir is required")
	}
	records, err := dumpindex.LoadDir(opts.dir)
	if err != nil {
		return fmt.Errorf("load dump dir: %w", err)
	}
	layers := dumpindex.Summaries(dumpindex.InferLayers(records))
	switch opts.format {
	case "csv":
		return dumpindex.WriteCSV(os.Stdout, layers)
	case "json":
		return dumpindex.WriteJSON(os.Stdout, layers)
	default:
		return fmt.Errorf("unsupported -format %q", opts.format)
	}
}
