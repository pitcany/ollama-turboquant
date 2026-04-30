package turboquanttests

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestJLEstimatorDiagnosticCoversSyntheticAndRealKQ(t *testing.T) {
	src, err := os.ReadFile(filepath.Join("..", "jl_estimator.cpp"))
	if err != nil {
		t.Fatalf("read jl_estimator.cpp: %v", err)
	}
	body := string(src)

	required := []string{
		"--synthetic-pairs",
		"--k-f32",
		"--q-rot-f32",
		"--query-token-mode",
		"--query-stride",
		"--rotation-mode",
		"--centroid-mode",
		"--centroid-iters",
		"--skip-synthetic",
		"--csv",
		"jl_sanity",
		"threebit_centroid",
		"threebit_jl",
		"current_turbo4",
		"sqrt_pi_over_2",
		"jl_error_reduction",
		"variance_bound",
		"generate_projection_matrix",
		"generate_rademacher_signs",
		"calibrate_centroids",
		"turbo_cpu_fwht_inverse",
	}
	for _, needle := range required {
		if !strings.Contains(body, needle) {
			t.Fatalf("jl_estimator.cpp missing %q", needle)
		}
	}
}
