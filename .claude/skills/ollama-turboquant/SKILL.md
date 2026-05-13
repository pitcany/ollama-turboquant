```markdown
# ollama-turboquant Development Patterns

> Auto-generated skill from repository analysis

## Overview

This skill teaches the core development patterns, coding conventions, and workflows used in the `ollama-turboquant` Go codebase. It covers how to implement features, manage calibration artifacts, document calibration sessions, write and promote interface tests, and update operator documentation. The guide is based on observed repository practices and is intended to help new contributors quickly align with established standards.

## Coding Conventions

**File Naming**
- Use `snake_case` for all file names.
  - Example: `cache_manager.go`, `model_utils_test.go`

**Imports**
- Use **relative import paths** within the module.
  - Example:
    ```go
    import (
        "kvcache"
        "tools/turboquant/calibration"
    )
    ```

**Exports**
- Use **named exports** for all exported functions, types, and variables.
  - Example:
    ```go
    // Exported function
    func NewCacheManager() *CacheManager {
        // ...
    }
    ```

**Commit Messages**
- Follow **conventional commit** style.
- Prefixes: `feat`, `docs`, `test`, `tools`, `fix`
- Example: `feat(kvcache): add LRU eviction policy to cache manager`

## Workflows

### Feature Implementation with Tests and Docs
**Trigger:** When adding a new feature, refactoring a core subsystem, or changing a critical workflow  
**Command:** `/feature-with-tests-docs`

1. Implement or refactor the main logic in one or more source files (e.g., `kvcache/cache_manager.go`).
2. Add or update corresponding unit/integration tests in files matching `*_test.go`.
   - Example:
     ```go
     func TestCacheEviction(t *testing.T) {
         // test logic
     }
     ```
3. Update documentation or debug logs (e.g., `TURBOQUANT-DEBUG-LOG.md`) to describe the change, rationale, and verification steps.

---

### Calibration Artifact Generation and Bundling
**Trigger:** When adding support for a new model or quantization scheme to TurboQuant's adaptive calibration  
**Command:** `/calibrate-and-bundle`

1. Run the calibration pipeline:
   ```sh
   scripts/turboquant-calibrate.sh
   ```
2. Add or update a calibration artifact JSON file in `tools/turboquant/calibration/manifest_data/`.
3. Patch `manifest.json` to include or update the entry for the new artifact.

---

### Calibration Session Documentation
**Trigger:** After running a calibration or making a significant change to the calibration pipeline  
**Command:** `/document-calibration-session`

1. Edit `TURBOQUANT-DEBUG-LOG.md` to add a new dated entry.
2. Describe what was run, what artifacts were produced, and the results.
3. Note any issues, blocks, or next steps for future operators.

---

### Test Promotion or Interface Assertion
**Trigger:** When adding/modifying methods expected to be promoted via embedding, or when interface assertions are critical for runtime behavior  
**Command:** `/test-interface-promotion`

1. Add or update package-local `*_test.go` files in `model/models/*/` to assert interface satisfaction.
   - Example:
     ```go
     var _ MyInterface = (*MyCache)(nil)
     ```
2. Add or update runner-level tests (e.g., `runner/ollamarunner/cache_test.go`) to verify interface usage at runtime.

---

### Documentation and Operator Guide Update
**Trigger:** When new features, workflows, or gotchas need to be communicated to operators or end-users  
**Command:** `/update-operator-guide`

1. Create or update a markdown guide (e.g., `tools/turboquant/USAGE.md`) with step-by-step instructions.
2. Update existing docs (e.g., `tools/turboquant/README.md`) to reference the new or updated guide.

---

## Testing Patterns

- Test files are named with the pattern `*_test.go`.
- Tests are colocated with the code they test (e.g., `kvcache/cache_manager_test.go`).
- Use Go's standard `testing` package.
- Interface assertions are often included to ensure correct embedding and interface satisfaction.
  - Example:
    ```go
    var _ Cache = (*LRUCache)(nil)
    ```

## Commands

| Command                     | Purpose                                                                                 |
|-----------------------------|-----------------------------------------------------------------------------------------|
| /feature-with-tests-docs     | Implement a new feature or refactor, with tests and documentation updates              |
| /calibrate-and-bundle        | Run calibration pipeline, bundle artifact, and update manifest                         |
| /document-calibration-session| Document calibration session results and issues in TURBOQUANT-DEBUG-LOG.md             |
| /test-interface-promotion    | Add or update tests for interface promotion or assertion                               |
| /update-operator-guide       | Update or add operator/end-user documentation and guides                               |
```