```markdown
# ollama-turboquant Development Patterns

> Auto-generated skill from repository analysis

## Overview

This skill teaches you how to contribute to the `ollama-turboquant` codebase, which implements TurboQuant quantization support for machine learning models in Go. You'll learn the project's coding conventions, how to update quantization tables, add new TurboQuant dtypes (including CUDA support), wire new presets through the stack, and update documentation. The guide also covers how to write and structure tests, and provides handy `/commands` for common workflows.

## Coding Conventions

- **File Naming:**  
  Use `camelCase` for file names.  
  _Example:_  
  ```
  turboQuant.go
  cacheTest.go
  ```

- **Import Style:**  
  Use **relative imports** within the Go project.  
  _Example:_  
  ```go
  import "../utils"
  ```

- **Export Style:**  
  Use **named exports** for functions and types.  
  _Example:_  
  ```go
  // Exported function
  func QuantizeBlock() {}

  // Exported type
  type TurboQuantPreset struct {}
  ```

- **Commit Messages:**  
  Follow **conventional commit** style with these prefixes: `feat`, `docs`, `fix`, `test`, `style`, `perf`.  
  _Example:_  
  ```
  feat: add Turbo6 quantization support to GGML backend
  fix: correct centroid generation for 5-bit TurboQuant
  ```

## Workflows

### Update Centroid Tables and Documentation
**Trigger:** When adding or updating TurboQuant centroid tables for new bit widths or improving centroid generation logic  
**Command:** `/update-centroids`

1. Edit or regenerate centroid tables using `lloyd_max.py` for the relevant bit widths.
2. Update `tools/turboquant/centroids/README.md` to document changes, discrepancies, or methodology.
3. Modify `tools/turboquant/centroids/lloyd_max.py` to improve algorithms, add features (e.g., multi-restart), or fix bugs.
4. Re-emit the relevant centroid text files (e.g., `turbo5_centroids.txt`, `turbo6_centroids.txt`).

_Example:_  
```bash
python tools/turboquant/centroids/lloyd_max.py --bits 6 > tools/turboquant/centroids/turbo6_centroids.txt
```

---

### Add New TurboQuant Dtype Support in GGML
**Trigger:** When implementing a new TurboQuant bit width in the GGML backend  
**Command:** `/add-turboquant-dtype`

1. Add new type enums and block layouts in `ggml.h` and `ggml-common.h`.
2. Add centroid and midpoint tables to `ggml-turbo-quant.c`.
3. Implement CPU quantize/dequantize logic for the new dtype in `ggml-turbo-quant.c`.
4. Register the new dtype in CPU type traits in `ggml-quants.h`, `ggml.c`, and `ggml-cpu/ggml-cpu.c`.

_Example:_  
```c
// ggml.h
typedef enum {
    GGML_TYPE_TURBO5,
    GGML_TYPE_TURBO6, // new
    // ...
} ggml_type;

// ggml-turbo-quant.c
static const float turbo6_centroids[] = { ... };
```

---

### Add CUDA Support for New TurboQuant Dtype
**Trigger:** When enabling CUDA support for a new TurboQuant dtype  
**Command:** `/add-cuda-turboquant`

1. Implement `dequantize_block_turbo{n}_0` and `quantize_f32_turbo{n}_0_block` in `convert.cu` and `turbo-quant.cuh`.
2. Add set-rows quantize kernel for the dtype in `set-rows.cu`.
3. Add or update flash attention vector inner product logic for the dtype in `fattn-common.cuh`, `fattn-vec.cuh`, `fattn.cu`, and register template instances.
4. Optionally, optimize (vectorize) inner product logic for performance.

_Example:_  
```cpp
// turbo-quant.cuh
__device__ void dequantize_block_turbo6_0(const uint8_t *src, float *dst, int n);
// convert.cu
dequantize_block_turbo6_0<<<...>>>(...);
```

---

### Add or Promote TurboQuant Preset and Wire Through Stack
**Trigger:** When introducing a new preset or promoting a previewed preset to general availability  
**Command:** `/add-turboquant-preset`

1. Add or update preset in `runner/ollamarunner/cache.go` and `cache_test.go`.
2. Wire new dtypes and preset through `ml.DType`, `backend.go`, `fs/ggml/ggml.go`, and related test files.
3. Add or update preset in `cmd/turboquant-eval/main.go` for the eval harness.
4. Add, update, or remove feature gates (e.g., `OLLAMA_TURBOQUANT_K6_PREVIEW`) in `envconfig/config.go` and `llm/server.go`.
5. Update or remove related tests (e.g., `llm/server_test.go`).
6. Document or log Phase 0 results in `TURBOQUANT-DEBUG-LOG.md`.

_Example:_  
```go
// envconfig/config.go
const OLLAMA_TURBOQUANT_K6_PREVIEW = "OLLAMA_TURBOQUANT_K6_PREVIEW"
```

---

### Update or Improve Documentation and Plans
**Trigger:** When planning new work, updating implementation plans, or clarifying technical discrepancies  
**Command:** `/update-docs`

1. Edit or add markdown files in `docs/superpowers/plans/` to outline new kernels, presets, or implementation phases.
2. Update `README.md` files in `tools/turboquant/` to clarify methodology, document discrepancies, or provide build instructions.
3. Document results or smoke test outcomes in `TURBOQUANT-DEBUG-LOG.md`.

_Example:_  
```markdown
# TurboQuant 6-bit Plan
- Implement kernel in CUDA
- Update centroid tables
- Wire through backend
```

## Testing Patterns

- **Test File Naming:**  
  Test files follow the pattern `*.test.*` (e.g., `cache_test.go`).

- **Testing Framework:**  
  No specific framework detected; use Go's standard `testing` package.

- **Example Test:**  
  ```go
  import "testing"

  func TestQuantizeBlock(t *testing.T) {
      // Arrange
      // Act
      // Assert
  }
  ```

## Commands

| Command                | Purpose                                                                                      |
|------------------------|----------------------------------------------------------------------------------------------|
| /update-centroids      | Regenerate and update centroid tables and documentation for new TurboQuant bit widths        |
| /add-turboquant-dtype  | Add support for a new TurboQuant dtype in the GGML backend                                   |
| /add-cuda-turboquant   | Implement CUDA support for a new TurboQuant dtype                                            |
| /add-turboquant-preset | Add or promote a TurboQuant preset and wire it through the stack                             |
| /update-docs           | Update documentation, plans, or debug logs                                                   |
```
