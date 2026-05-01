---
name: add-new-turboquant-dtype-support-in-ggml
description: Workflow command scaffold for add-new-turboquant-dtype-support-in-ggml in ollama-turboquant.
allowed_tools: ["Bash", "Read", "Write", "Grep", "Glob"]
---

# /add-new-turboquant-dtype-support-in-ggml

Use this workflow when working on **add-new-turboquant-dtype-support-in-ggml** in `ollama-turboquant`.

## Goal

Introduce support for a new TurboQuant dtype (e.g., Turbo5, Turbo6) in the GGML backend, including type registration, block layouts, and quantize/dequantize logic.

## Common Files

- `ml/backend/ggml/ggml/include/ggml.h`
- `ml/backend/ggml/ggml/src/ggml-common.h`
- `ml/backend/ggml/ggml/src/ggml-turbo-quant.c`
- `ml/backend/ggml/ggml/src/ggml-quants.h`
- `ml/backend/ggml/ggml/src/ggml.c`
- `ml/backend/ggml/ggml/src/ggml-cpu/ggml-cpu.c`

## Suggested Sequence

1. Understand the current state and failure mode before editing.
2. Make the smallest coherent change that satisfies the workflow goal.
3. Run the most relevant verification for touched files.
4. Summarize what changed and what still needs review.

## Typical Commit Signals

- Add new type enums and block layouts in ggml.h and ggml-common.h.
- Add centroid and midpoint tables to ggml-turbo-quant.c.
- Implement CPU quantize/dequantize logic for the new dtype in ggml-turbo-quant.c.
- Register the new dtype in CPU type traits in ggml-quants.h, ggml.c, and ggml-cpu/ggml-cpu.c.

## Notes

- Treat this as a scaffold, not a hard-coded script.
- Update the command if the workflow evolves materially.