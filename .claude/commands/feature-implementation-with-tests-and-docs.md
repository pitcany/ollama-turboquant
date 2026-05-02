---
name: feature-implementation-with-tests-and-docs
description: Workflow command scaffold for feature-implementation-with-tests-and-docs in ollama-turboquant.
allowed_tools: ["Bash", "Read", "Write", "Grep", "Glob"]
---

# /feature-implementation-with-tests-and-docs

Use this workflow when working on **feature-implementation-with-tests-and-docs** in `ollama-turboquant`.

## Goal

Implements a new feature or significant refactor, accompanied by targeted tests and documentation/log updates.

## Common Files

- `kvcache/*.go`
- `kvcache/*_test.go`
- `TURBOQUANT-DEBUG-LOG.md`

## Suggested Sequence

1. Understand the current state and failure mode before editing.
2. Make the smallest coherent change that satisfies the workflow goal.
3. Run the most relevant verification for touched files.
4. Summarize what changed and what still needs review.

## Typical Commit Signals

- Implement or refactor main logic in one or more source files.
- Add or update corresponding unit/integration tests in *_test.go files.
- Update documentation or debug logs (e.g., TURBOQUANT-DEBUG-LOG.md) to describe the change, rationale, and verification steps.

## Notes

- Treat this as a scaffold, not a hard-coded script.
- Update the command if the workflow evolves materially.