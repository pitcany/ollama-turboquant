package turboquanttests

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestKCalibrationDiagnosticReportsScaleAndHistogram(t *testing.T) {
	src, err := os.ReadFile(filepath.Join("..", "k_calibration.cpp"))
	if err != nil {
		t.Fatalf("read k_calibration.cpp: %v", err)
	}
	body := string(src)

	required := []string{
		"--k-f32",
		"--q-rot-f32",
		"--q-head-mode",
		"--query-token-mode",
		"--query-stride",
		"--scale",
		"--top-rows",
		"quantize_row_turbo4_0_ref",
		"dequantize_row_turbo4_0",
		"turbo_cpu_fwht",
		"vector_optimal_scale",
		"kq_optimal_scale",
		"centroid_histogram",
		"worst_rows",
		"saturation_rate",
	}
	for _, needle := range required {
		if !strings.Contains(body, needle) {
			t.Fatalf("k_calibration.cpp missing %q", needle)
		}
	}
}
