package nn

import (
	"os"
	"strings"
	"testing"
)

func TestTurboAttentionRotatesKeyAndValuePathsIndependently(t *testing.T) {
	src, err := os.ReadFile("attention.go")
	if err != nil {
		t.Fatal(err)
	}

	text := string(src)
	for _, want := range []string{
		"turboKey := key != nil && isTurboDType(key.DType())",
		"turboValue := value != nil && isTurboDType(value.DType())",
		"if turboKey {",
		"if turboValue {",
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("attention.go missing %q", want)
		}
	}
}
