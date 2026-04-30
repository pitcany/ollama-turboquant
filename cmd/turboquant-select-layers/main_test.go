package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/tools/turboquant/layerselect"
)

const sampleCSV = `layer,key_layer_types,sequences,tokens,mean_nll,perplexity,mean_kl,duration_ms,status
0,0:q8_0,4,4092,2.07640661,7.97575733,0.15234685,18973,ok
1,1:q8_0,4,4092,5.74972980,314.10577687,3.95995938,19091,ok
27,27:q8_0,4,4092,5.80746057,332.77299651,4.01762444,19526,ok
3,3:q8_0,4,4092,5.86085046,351.02254797,4.07728846,19048,ok
2,2:q8_0,,,,,,,failed
15,15:q8_0,4,4092,5.90891402,368.30596396,4.12998841,18857,ok
`

func TestSelectLayerSpecChoosesLowestKLLayers(t *testing.T) {
	spec, err := selectLayerSpecForTest(strings.NewReader(sampleCSV), 4, "q8_0")
	if err != nil {
		t.Fatalf("selectLayerSpec() error = %v", err)
	}
	const want = "0:q8_0,1:q8_0,3:q8_0,27:q8_0"
	if spec != want {
		t.Fatalf("selectLayerSpec() = %q, want %q", spec, want)
	}
}

func TestSelectLayerSpecLimitsToAvailableOKRows(t *testing.T) {
	spec, err := selectLayerSpecForTest(strings.NewReader(sampleCSV), 99, "q8_0")
	if err != nil {
		t.Fatalf("selectLayerSpec() error = %v", err)
	}
	const want = "0:q8_0,1:q8_0,3:q8_0,15:q8_0,27:q8_0"
	if spec != want {
		t.Fatalf("selectLayerSpec() = %q, want %q", spec, want)
	}
}

func TestSelectLayerSpecRejectsInvalidTopN(t *testing.T) {
	if _, err := selectLayerSpecForTest(strings.NewReader(sampleCSV), 0, "q8_0"); err == nil {
		t.Fatal("expected error")
	}
}

func TestSelectLayerSpecRejectsNoOKRows(t *testing.T) {
	csv := `layer,key_layer_types,sequences,tokens,mean_nll,perplexity,mean_kl,duration_ms,status
0,0:q8_0,,,,,,,failed
`
	if _, err := selectLayerSpecForTest(strings.NewReader(csv), 1, "q8_0"); err == nil {
		t.Fatal("expected error")
	}
}

func TestRunSelectLayersWritesSpec(t *testing.T) {
	dir := t.TempDir()
	input := filepath.Join(dir, "sweep.csv")
	if err := os.WriteFile(input, []byte(sampleCSV), 0o644); err != nil {
		t.Fatal(err)
	}

	var out strings.Builder
	if err := runSelectLayers(input, 4, "q8_0", "spec", &out); err != nil {
		t.Fatalf("runSelectLayers() error = %v", err)
	}
	const want = "0:q8_0,1:q8_0,3:q8_0,27:q8_0\n"
	if out.String() != want {
		t.Fatalf("output = %q, want %q", out.String(), want)
	}
}

func selectLayerSpecForTest(r *strings.Reader, top int, dtype string) (string, error) {
	return layerselect.SelectLayerSpec(r, top, dtype)
}
