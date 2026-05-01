---
name: update-centroid-tables-and-documentation
description: Workflow command scaffold for update-centroid-tables-and-documentation in ollama-turboquant.
allowed_tools: ["Bash", "Read", "Write", "Grep", "Glob"]
---

# /update-centroid-tables-and-documentation

Use this workflow when working on **update-centroid-tables-and-documentation** in `ollama-turboquant`.

## Goal

Regenerate and update centroid tables for new TurboQuant bit widths, update related documentation, and improve centroid generation scripts.

## Common Files

- `tools/turboquant/centroids/README.md`
- `tools/turboquant/centroids/lloyd_max.py`
- `tools/turboquant/centroids/turbo5_centroids.txt`
- `tools/turboquant/centroids/turbo6_centroids.txt`

## Suggested Sequence

1. Understand the current state and failure mode before editing.
2. Make the smallest coherent change that satisfies the workflow goal.
3. Run the most relevant verification for touched files.
4. Summarize what changed and what still needs review.

## Typical Commit Signals

- Edit or regenerate centroid tables using lloyd_max.py for the relevant bit widths.
- Update tools/turboquant/centroids/README.md to document changes, discrepancies, or methodology.
- Modify tools/turboquant/centroids/lloyd_max.py to improve algorithms, add features (e.g., multi-restart), or fix bugs.
- Re-emit the relevant centroid text files (e.g., turbo5_centroids.txt, turbo6_centroids.txt).

## Notes

- Treat this as a scaffold, not a hard-coded script.
- Update the command if the workflow evolves materially.