package main

import "testing"

func TestRunRequiresDir(t *testing.T) {
	err := run(options{format: "csv"})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestRunRejectsUnsupportedFormat(t *testing.T) {
	err := run(options{dir: t.TempDir(), format: "xml"})
	if err == nil {
		t.Fatal("expected error")
	}
}
