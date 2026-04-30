package layerselect

import (
	"strings"
	"testing"
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
	spec, err := SelectLayerSpec(strings.NewReader(sampleCSV), 4, "q8_0")
	if err != nil {
		t.Fatalf("SelectLayerSpec() error = %v", err)
	}
	const want = "0:q8_0,1:q8_0,3:q8_0,27:q8_0"
	if spec != want {
		t.Fatalf("SelectLayerSpec() = %q, want %q", spec, want)
	}
}

func TestSelectLayerSpecLimitsToAvailableOKRows(t *testing.T) {
	spec, err := SelectLayerSpec(strings.NewReader(sampleCSV), 99, "q8_0")
	if err != nil {
		t.Fatalf("SelectLayerSpec() error = %v", err)
	}
	const want = "0:q8_0,1:q8_0,3:q8_0,15:q8_0,27:q8_0"
	if spec != want {
		t.Fatalf("SelectLayerSpec() = %q, want %q", spec, want)
	}
}

func TestSelectLayerSpecRejectsInvalidTopN(t *testing.T) {
	if _, err := SelectLayerSpec(strings.NewReader(sampleCSV), 0, "q8_0"); err == nil {
		t.Fatal("expected error")
	}
}

func TestSelectLayerSpecRejectsNoOKRows(t *testing.T) {
	csv := `layer,key_layer_types,sequences,tokens,mean_nll,perplexity,mean_kl,duration_ms,status
0,0:q8_0,,,,,,,failed
`
	if _, err := SelectLayerSpec(strings.NewReader(csv), 1, "q8_0"); err == nil {
		t.Fatal("expected error")
	}
}
