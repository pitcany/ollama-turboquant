package ggml

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestTurboFlashAttentionRequiresVectorKernelStride(t *testing.T) {
	supported, err := turboFlashAttnSupport(128, 1, 128)
	if err != nil {
		if err == errTurboFlashAttnSkipped {
			t.Skip(err)
		}
		t.Fatal(err)
	}

	if supported {
		t.Fatal("expected GPU backend to reject TurboQuant flash attention when KV length is not vector-kernel aligned")
	}
}

func TestTurboFlashAttentionSupportsVectorKernelStride(t *testing.T) {
	supported, err := turboFlashAttnSupport(128, 1, 256)
	if err != nil {
		if err == errTurboFlashAttnSkipped {
			t.Skip(err)
		}
		t.Fatal(err)
	}

	if !supported {
		t.Fatal("expected GPU backend to accept TurboQuant flash attention when the vector kernel is available")
	}
}

func TestTurboFlashAttentionSupportsTurboKeyF16Value(t *testing.T) {
	supported, err := turboFlashAttnSupportTurboKeyF16Value(128, 1, 256)
	if err != nil {
		if err == errTurboFlashAttnSkipped {
			t.Skip(err)
		}
		t.Fatal(err)
	}

	if !supported {
		t.Fatal("expected GPU backend to accept TurboQuant K with f16 V for isolation diagnostics")
	}
}

func TestTurboFlashAttentionDispatchesTurboKeyF16Value(t *testing.T) {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("failed to resolve test path")
	}

	cudaDir := filepath.Join(filepath.Dir(file), "ggml", "src", "ggml-cuda")
	dispatch, err := os.ReadFile(filepath.Join(cudaDir, "fattn.cu"))
	if err != nil {
		t.Fatal(err)
	}

	if !strings.Contains(string(dispatch), "FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO4_0, GGML_TYPE_F16)") {
		t.Fatal("fattn.cu does not dispatch TurboQuant K with f16 V")
	}

	instancePath := filepath.Join(cudaDir, "template-instances", "fattn-vec-instance-turbo4_0-f16.cu")
	instance, err := os.ReadFile(instancePath)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(instance), "DECL_FATTN_VEC_CASE(128, GGML_TYPE_TURBO4_0, GGML_TYPE_F16)") {
		t.Fatal("missing 128-wide TurboQuant K + f16 V vector instance")
	}
}

func TestTurboFlashAttentionKQReadsHalf2QPath(t *testing.T) {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("failed to resolve test path")
	}

	sourcePath := filepath.Join(filepath.Dir(file), "ggml", "src", "ggml-cuda", "fattn-common.cuh")
	source, err := os.ReadFile(sourcePath)
	if err != nil {
		t.Fatal(err)
	}

	for _, name := range []string{"turbo2", "turbo3", "turbo4"} {
		body, ok := functionBody(string(source), "vec_dot_fattn_vec_KQ_"+name)
		if !ok {
			t.Fatalf("missing %s KQ dot function", name)
		}
		if !strings.Contains(body, "V_DOT2_F32_F16_AVAILABLE") {
			t.Fatalf("%s KQ dot does not branch on V_DOT2_F32_F16_AVAILABLE", name)
		}
		if !strings.Contains(body, "__half22float2(((const half2 *) Q_v)") {
			t.Fatalf("%s KQ dot does not read Q_v as half2 on the f16 dot2 path", name)
		}
	}
}

func functionBody(source string, name string) (string, bool) {
	start := strings.Index(source, name)
	if start < 0 {
		return "", false
	}
	open := strings.Index(source[start:], "{")
	if open < 0 {
		return "", false
	}
	open += start

	depth := 0
	for i := open; i < len(source); i++ {
		switch source[i] {
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				return source[open : i+1], true
			}
		}
	}
	return "", false
}
