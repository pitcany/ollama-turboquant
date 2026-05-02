# Follow-up: bundle qwen3-coder:30b at Q8_0 weights

**Status:** not started — needs ≥36 GB VRAM (e.g. 2× 24 GB) on the calibration host.

**Why:** the shipped `qwen3moe-q4_k_m-adaptive.json` lands at mean_kl ≈ 0.052,
mostly because Q4_K_M weights already cost ~0.04 KL before TurboQuant touches the
KV cache. The qwen3.6:27b artifact (Q8_0 weights) lands at mean_kl ≈ 0.016 — the
pure KV-cache cost. Repeating the recipe at Q8_0 should put qwen3-coder in the
0.02 range — visually indistinguishable from f16 for users who can spare the
weight-side VRAM.

The multi-GPU calibration unblock landed in commit `103cac73` (`fix(turboquant-eval): split layers across all visible GPUs`) — that is the only
reason this is feasible now.

## Pre-flight

- `ollama pull qwen3-coder:30b-q8_0` (or pin a local GGUF — see `-m`).
- Confirm both GPUs are visible to the eval binary: `nvidia-smi -L`.
- Reuse an existing snapshot if available — calibration tokenization is the
  cheap step but a stable corpus is more important than a fresh one for KL
  comparability with the Q4_K_M run.

## Calibration command

```bash
scripts/turboquant-calibrate.sh \
  -m qwen3-coder:30b-q8_0 \
  -s /tmp/turboquant-calibration-qwen3moe-q4_k_m/tokens.json \
  -a qwen3moe \
  -f Q8_0 \
  -d 128 \
  -t 4 \
  -k 0.05 \
  -V 16
```

Flag rationale:

- `-s` reuses the qwen3-coder Q4_K_M snapshot. KL is corpus-stable across
  weight quantizations, so reusing the snapshot makes the new artifact
  directly comparable to the Q4_K_M number.
- `-t 4` matches the existing entries — protect 4 lowest-KL layers at q8_0.
- `-k 0.05` tightens the validation gate. The Q4_K_M shipped at 0.052;
  with Q8_0 weights we expect well under 0.05, so failing this gate would
  signal something off (bad sweep, GPU partition issue) rather than an
  acceptable result.
- `-V 16` follows the established 16-sequence convergence default. Bump
  to `-V 0` only if this becomes an upstream-bound artifact.

The script will:

1. Reuse the Phase 0 snapshot at `-s`.
2. Sweep K-dtype overrides and pick the top-4 lowest-KL layers.
3. Validate against `mean_kl <= 0.05`.
4. Write `tools/turboquant/calibration/manifest_data/qwen3moe-q8_0-adaptive.json`.
5. Patch `manifest.json` with the new entry.
6. Run `go test ./tools/turboquant/calibration/...` to confirm the loader
   accepts it.

## Manifest entry skeleton (what step 5 should produce)

The `manifest.json` should gain an entry shaped like:

```json
{
  "architecture": "qwen3moe",
  "file_type": "Q8_0",
  "head_dim": 128,
  "artifact": "qwen3moe-q8_0-adaptive.json",
  "model_hint": "qwen3-coder:30b-q8_0",
  "validated": "<YYYY-MM-DD>",
  "phase0_mean_kl": <fill from script output>,
  "phase0_perplexity": <fill from script output>
}
```

The existing Q4_K_M entry stays untouched — runtime resolves on the
`(architecture, file_type, head_dim)` triple, so both presets coexist and
the user's choice of weight quantization picks the right calibration
automatically.

## Expected outcome

| Metric | Q4_K_M (shipped) | Q8_0 (target) |
|---|---|---|
| Mean KL vs f16 | 0.052 | ~0.02 |
| Phase-0 perplexity | 9.27 | ~9.0 |
| Weight VRAM | ~18 GB | ~32 GB |
| Net "longer context" headroom on 24 GB GPU | ~2.7× | none — won't fit |
| Net headroom on 48 GB GPU (or 2×24) | ~2.7× | ~2.5× |

So the Q8_0 entry is for users who **already had to multi-GPU** the model
and want the crispest possible KV-compressed inference. It is not a
replacement for the Q4_K_M entry on consumer hardware.

## Verification before committing the artifact

- [ ] `go test ./tools/turboquant/calibration/...` passes.
- [ ] `ollama-serve-tq` loads the model without falling back to the
      `kq8-vturbo4` default (look for `manifest match` log line).
- [ ] Spot-check a few coding prompts side-by-side against vanilla
      `ollama run qwen3-coder:30b-q8_0` — answers should be effectively
      identical.

## Rollback

If the gate fails or the runtime regresses: delete the new artifact and
revert the manifest entry. The Q4_K_M entry is unaffected.
