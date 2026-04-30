package main

import (
	"testing"
	"time"
)

func TestParseOptionsDefaults(t *testing.T) {
	opts := parseOptions(nil)
	if opts.promptTokens != 1024 {
		t.Fatalf("promptTokens = %d, want 1024", opts.promptTokens)
	}
	if opts.decodeTokens != 128 {
		t.Fatalf("decodeTokens = %d, want 128", opts.decodeTokens)
	}
	if !opts.flashAttn {
		t.Fatal("flashAttn should default true")
	}
	if opts.format != "json" {
		t.Fatalf("format = %q, want json", opts.format)
	}
}

func TestMSPerTokenToTPS(t *testing.T) {
	if got := msPerTokenToTPS(10); got != 100 {
		t.Fatalf("10ms/token -> %.2f tps, want 100", got)
	}
	if got := msPerTokenToTPS(0); got != 0 {
		t.Fatalf("0ms/token -> %.2f tps, want 0", got)
	}
}

func TestMedianDuration(t *testing.T) {
	got := medianDuration([]time.Duration{
		3 * time.Millisecond,
		1 * time.Millisecond,
		2 * time.Millisecond,
	})
	if got != 2*time.Millisecond {
		t.Fatalf("median = %v, want 2ms", got)
	}
}
