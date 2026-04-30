package ggml

/*
#cgo linux LDFLAGS: -lrt -lpthread -ldl -lstdc++ -lm
#cgo windows LDFLAGS: -lpthread
#cgo CPPFLAGS: -I${SRCDIR}/ggml/include
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include "ggml.h"
#include "ggml-backend.h"

static struct ggml_context * turbo_test_context(size_t mem_size) {
	struct ggml_init_params params = {
		.mem_size = mem_size,
		.mem_buffer = malloc(mem_size),
		.no_alloc = true,
	};
	return ggml_init(params);
}

static void turbo_test_context_free(struct ggml_context * ctx) {
	void * mem_buffer = ggml_get_mem_buffer(ctx);
	ggml_free(ctx);
	free(mem_buffer);
}

static struct ggml_tensor * turbo_test_flash_attn_op(
		struct ggml_context * ctx,
		enum ggml_type k_type,
		enum ggml_type v_type,
		int64_t head_dim,
		int64_t n_tokens,
		int64_t kv_len) {
	struct ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, n_tokens, 1, 1);
	struct ggml_tensor * k = ggml_new_tensor_4d(ctx, k_type,          head_dim, kv_len,   1, 1);
	struct ggml_tensor * v = ggml_new_tensor_4d(ctx, v_type,          head_dim, kv_len,   1, 1);
	struct ggml_tensor * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kv_len, n_tokens, 1, 1);
	return ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f, 0.0f, 0.0f);
}

static bool turbo_test_device_supports_op(ggml_backend_dev_t dev, struct ggml_tensor * op) {
	return ggml_backend_dev_supports_op(dev, op);
}
*/
import "C"

import (
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"unsafe"
)

var errTurboFlashAttnSkipped = errors.New("CUDA backend library or GPU device unavailable")

func turboFlashAttnSupport(headDim, nTokens, kvLen int64) (bool, error) {
	return turboFlashAttnSupportForTypes(C.GGML_TYPE_TURBO4_0, C.GGML_TYPE_TURBO4_0, headDim, nTokens, kvLen)
}

func turboFlashAttnSupportTurboKeyF16Value(headDim, nTokens, kvLen int64) (bool, error) {
	return turboFlashAttnSupportForTypes(C.GGML_TYPE_TURBO4_0, C.GGML_TYPE_F16, headDim, nTokens, kvLen)
}

func turboFlashAttnSupportForTypes(kType, vType C.enum_ggml_type, headDim, nTokens, kvLen int64) (bool, error) {
	if err := loadTurboTestBackends(); err != nil {
		return false, err
	}

	device := firstGPUDevice()
	if device == nil {
		return false, errTurboFlashAttnSkipped
	}

	ctx := C.turbo_test_context(1 << 20)
	if ctx == nil {
		return false, errors.New("failed to allocate ggml test context")
	}
	defer C.turbo_test_context_free(ctx)

	op := C.turbo_test_flash_attn_op(
		ctx,
		kType,
		vType,
		C.int64_t(headDim),
		C.int64_t(nTokens),
		C.int64_t(kvLen),
	)
	if op == nil {
		return false, errors.New("failed to construct flash attention op")
	}

	return bool(C.turbo_test_device_supports_op(device, op)), nil
}

func loadTurboTestBackends() error {
	candidates := []string{}
	if env := os.Getenv("OLLAMA_LIBRARY_PATH"); env != "" {
		candidates = append(candidates, filepath.SplitList(env)...)
	}

	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		return errors.New("failed to resolve test file path")
	}

	repoRoot := filepath.Clean(filepath.Join(filepath.Dir(thisFile), "..", "..", ".."))
	candidates = append(candidates,
		filepath.Join(repoRoot, "build", "lib", "ollama", "cuda_v12"),
		filepath.Join(repoRoot, "cuda_v12"),
		filepath.Join(repoRoot, "build", "lib", "ollama"),
		repoRoot,
	)

	for _, dir := range candidates {
		if dir == "" {
			continue
		}

		info, err := os.Stat(filepath.Join(dir, "libggml-cuda.so"))
		if err != nil || info.IsDir() {
			continue
		}

		cdir := C.CString(dir)
		C.ggml_backend_load_all_from_path(cdir)
		C.free(unsafe.Pointer(cdir))
		return nil
	}

	return errTurboFlashAttnSkipped
}

func firstGPUDevice() C.ggml_backend_dev_t {
	for i := C.size_t(0); i < C.ggml_backend_dev_count(); i++ {
		device := C.ggml_backend_dev_get(i)
		switch C.ggml_backend_dev_type(device) {
		case C.GGML_BACKEND_DEVICE_TYPE_GPU, C.GGML_BACKEND_DEVICE_TYPE_IGPU:
			return device
		}
	}

	return nil
}
