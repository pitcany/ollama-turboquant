package ggml

import "testing"

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
