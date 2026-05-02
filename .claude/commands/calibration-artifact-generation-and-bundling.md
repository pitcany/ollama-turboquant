---
name: calibration-artifact-generation-and-bundling
description: Workflow command scaffold for calibration-artifact-generation-and-bundling in ollama-turboquant.
allowed_tools: ["Bash", "Read", "Write", "Grep", "Glob"]
---

# /calibration-artifact-generation-and-bundling

Use this workflow when working on **calibration-artifact-generation-and-bundling** in `ollama-turboquant`.

## Goal

Runs the calibration pipeline for a new model, bundles the resulting artifact, and updates the manifest for runtime resolution.

## Common Files

- `scripts/turboquant-calibrate.sh`
- `tools/turboquant/calibration/manifest_data/*.json`

## Suggested Sequence

1. Understand the current state and failure mode before editing.
2. Make the smallest coherent change that satisfies the workflow goal.
3. Run the most relevant verification for touched files.
4. Summarize what changed and what still needs review.

## Typical Commit Signals

- Run scripts/turboquant-calibrate.sh to generate calibration data.
- Add or update a calibration artifact JSON file under tools/turboquant/calibration/manifest_data/.
- Patch manifest.json to include or update the entry for the new artifact.

## Notes

- Treat this as a scaffold, not a hard-coded script.
- Update the command if the workflow evolves materially.