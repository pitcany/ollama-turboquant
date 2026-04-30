package turboquanttests

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestKQErrorDiagnosticUsesFattnTurbo4Helper(t *testing.T) {
	src, err := os.ReadFile(filepath.Join("..", "kq_error.cu"))
	if err != nil {
		t.Fatalf("read kq_error.cu: %v", err)
	}
	body := string(src)

	required := []string{
		"--k-f32",
		"--q-rot-f32",
		"--kv-heads",
		"--q-head-mode",
		"--scale",
		"--max-k-rows",
		"--csv",
		"quantize_row_turbo4_0_ref",
		"turbo_cpu_fwht",
		"vec_dot_fattn_vec_KQ_turbo4<kDim, kNThreadsKQ>",
		"mean_abs_error",
		"rms_error",
	}
	for _, needle := range required {
		if !strings.Contains(body, needle) {
			t.Fatalf("kq_error.cu missing %q", needle)
		}
	}
}
