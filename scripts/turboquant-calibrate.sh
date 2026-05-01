#!/usr/bin/env bash
# turboquant-calibrate.sh — generate a turboquant-adaptive calibration for one
# (model, file_type, head_dim) tuple and bundle it into the runtime manifest.
#
# Pipeline:
#   1. Tokenize a corpus into a Phase 0 snapshot (skipped if -s reuses one).
#   2. Run cmd/turboquant-calibrate to sweep K-dtype overrides per layer,
#      pick the -top lowest-KL layers, and validate the resulting policy.
#   3. Gate on mean_kl <= -max-mean-kl. Default 0.13 matches the
#      "adaptive" Phase 0 budget in tools/turboquant/README.md ("CI gate").
#   4. Copy the artifact under a deterministic name into the bundled manifest
#      directory and patch tools/turboquant/calibration/manifest_data/manifest.json
#      with the new (architecture, file_type, head_dim) entry.
#   5. Run go test ./tools/turboquant/calibration/... so the loader proves
#      it can read the new entry before anyone ships it.
#
# Re-running the script with the same -arch/-file-type/-head-dim is idempotent:
# it overwrites the artifact and replaces the matching manifest entry. Use this
# to iterate on -top or to refresh a calibration after a corpus change.
#
# Prereqs: go, jq, a working CUDA backend (calibrate flash-attention requires
# CUDA today), and the target model already pulled by ollama.
#
# Usage:
#   scripts/turboquant-calibrate.sh \
#     -m qwen3.6:27b-q8_0 \
#     -c /tmp/long-context-corpus.txt \
#     -a qwen35 \
#     -f Q8_0 \
#     -d 256 \
#     -t 4
#
# All flags:
#   -m  ollama model name or GGUF path                                (required)
#   -c  UTF-8 corpus path used to tokenize a fresh snapshot           (required unless -s)
#   -s  reuse an existing snapshot at this path                       (alternative to -c)
#   -a  architecture key for the manifest entry, e.g. qwen2, qwen35   (required)
#   -f  file_type key for the manifest entry, e.g. Q4_K_M, Q8_0       (required)
#   -d  head_dim key for the manifest entry, 0 = wildcard             (required)
#   -t  number of lowest-KL layers to protect at q8_0                 (default 4)
#   -k  max allowed mean_kl for the validation gate                   (default 0.13)
#   -n  num-ctx for tokenize and calibrate                            (default 1024)
#   -L  sweep-limit (sequences per single-layer sweep)                (default 4)
#   -V  validate-limit (0 = all sequences)                            (default 0)
#   -B  batch-size for calibrate                                      (default 512)
#   -G  num-gpu-layers for calibrate                                  (default 999)
#   -o  artifact basename under manifest_data/, default
#       <arch>-<lower-file-type>-adaptive.json
#   -h  validated date stamp (default today, ISO YYYY-MM-DD)
#   -W  artifact work dir for sweep CSV + logs
#       (default /tmp/turboquant-calibration-<arch>-<file-type>)
#   -P  validated phase ("validated" stamp model-hint), defaults to -m
#   --no-bundle: stop after calibrate+gate (do not write to manifest)
#   --no-loader-test: skip the go test step at the end
#
# All paths are interpreted relative to the repository root, which the script
# locates by climbing from its own directory.

set -euo pipefail

repo_root() {
  local dir
  dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cd "$dir/.."
  pwd
}

ROOT="$(repo_root)"
cd "$ROOT"

if ! command -v go >/dev/null 2>&1; then
  echo "error: go not found on PATH" >&2; exit 2
fi
if ! command -v jq >/dev/null 2>&1; then
  echo "error: jq not found on PATH (apt install jq)" >&2; exit 2
fi

MODEL=""
CORPUS=""
SNAPSHOT=""
ARCH=""
FTYPE=""
HEAD_DIM=""
TOP=4
MAX_MEAN_KL=0.13
NUM_CTX=1024
SWEEP_LIMIT=4
VALIDATE_LIMIT=0
BATCH_SIZE=512
NUM_GPU_LAYERS=999
ARTIFACT_BASENAME=""
VALIDATED_DATE="$(date -u +%Y-%m-%d)"
WORK_DIR=""
MODEL_HINT=""
SKIP_BUNDLE=0
SKIP_LOADER_TEST=0

usage() { sed -n '1,/^set -euo pipefail/p' "$0" | sed '$d' >&2; exit 2; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    -m) MODEL="$2"; shift 2 ;;
    -c) CORPUS="$2"; shift 2 ;;
    -s) SNAPSHOT="$2"; shift 2 ;;
    -a) ARCH="$2"; shift 2 ;;
    -f) FTYPE="$2"; shift 2 ;;
    -d) HEAD_DIM="$2"; shift 2 ;;
    -t) TOP="$2"; shift 2 ;;
    -k) MAX_MEAN_KL="$2"; shift 2 ;;
    -n) NUM_CTX="$2"; shift 2 ;;
    -L) SWEEP_LIMIT="$2"; shift 2 ;;
    -V) VALIDATE_LIMIT="$2"; shift 2 ;;
    -B) BATCH_SIZE="$2"; shift 2 ;;
    -G) NUM_GPU_LAYERS="$2"; shift 2 ;;
    -o) ARTIFACT_BASENAME="$2"; shift 2 ;;
    -h) VALIDATED_DATE="$2"; shift 2 ;;
    -W) WORK_DIR="$2"; shift 2 ;;
    -P) MODEL_HINT="$2"; shift 2 ;;
    --no-bundle) SKIP_BUNDLE=1; shift ;;
    --no-loader-test) SKIP_LOADER_TEST=1; shift ;;
    --help|-?) usage ;;
    *) echo "error: unknown flag: $1" >&2; usage ;;
  esac
done

[[ -z "$MODEL" ]] && { echo "error: -m model is required" >&2; exit 2; }
[[ -z "$ARCH"  ]] && { echo "error: -a architecture is required" >&2; exit 2; }
[[ -z "$FTYPE" ]] && { echo "error: -f file_type is required" >&2; exit 2; }
[[ -z "$HEAD_DIM" ]] && { echo "error: -d head_dim is required (0 = wildcard)" >&2; exit 2; }

if [[ -z "$SNAPSHOT" && -z "$CORPUS" ]]; then
  echo "error: must pass either -c <corpus> or -s <snapshot>" >&2; exit 2
fi
if [[ -n "$SNAPSHOT" && -n "$CORPUS" ]]; then
  echo "error: pass -c OR -s, not both" >&2; exit 2
fi

[[ -z "$MODEL_HINT" ]] && MODEL_HINT="$MODEL"

ftype_lower="$(echo "$FTYPE" | tr '[:upper:]' '[:lower:]')"
[[ -z "$ARTIFACT_BASENAME" ]] && ARTIFACT_BASENAME="${ARCH}-${ftype_lower}-adaptive.json"
[[ -z "$WORK_DIR" ]] && WORK_DIR="/tmp/turboquant-calibration-${ARCH}-${ftype_lower}"

MANIFEST_DIR="$ROOT/tools/turboquant/calibration/manifest_data"
MANIFEST="$MANIFEST_DIR/manifest.json"
ARTIFACT_DEST="$MANIFEST_DIR/$ARTIFACT_BASENAME"

[[ -f "$MANIFEST" ]] || { echo "error: manifest not found at $MANIFEST" >&2; exit 2; }

mkdir -p "$WORK_DIR"
WORK_ARTIFACT="$WORK_DIR/calibration.json"

export GOCACHE="${GOCACHE:-/tmp/ollama-build-gocache}"

echo "=== turboquant-calibrate.sh ==="
echo "  model:         $MODEL"
echo "  arch:          $ARCH"
echo "  file_type:     $FTYPE"
echo "  head_dim:      $HEAD_DIM"
echo "  top:           $TOP"
echo "  max_mean_kl:   $MAX_MEAN_KL"
echo "  num_ctx:       $NUM_CTX"
echo "  artifact:      $ARTIFACT_DEST"
echo "  work_dir:      $WORK_DIR"
echo "  validated:     $VALIDATED_DATE"
echo

# 1. Tokenize (or reuse).
if [[ -z "$SNAPSHOT" ]]; then
  [[ -f "$CORPUS" ]] || { echo "error: corpus not found: $CORPUS" >&2; exit 2; }
  SNAPSHOT="$WORK_DIR/tokens.json"
  echo "[1/5] Tokenizing $CORPUS -> $SNAPSHOT"
  go run ./cmd/turboquant-tokenize \
    -model "$MODEL" \
    -corpus "$CORPUS" \
    -sequence-len "$NUM_CTX" \
    -max-sequences 256 \
    -output "$SNAPSHOT"
else
  [[ -f "$SNAPSHOT" ]] || { echo "error: snapshot not found: $SNAPSHOT" >&2; exit 2; }
  echo "[1/5] Reusing snapshot $SNAPSHOT"
fi
echo

# 2. Calibrate (sweep + selected-layer validation).
echo "[2/5] Running cmd/turboquant-calibrate ($MODEL, top=$TOP)"
go run ./cmd/turboquant-calibrate \
  -model "$MODEL" \
  -snapshot "$SNAPSHOT" \
  -base-kv-cache-type turbo4 \
  -reference-kv-cache-type f16 \
  -layer-dtype q8_0 \
  -top "$TOP" \
  -sweep-limit "$SWEEP_LIMIT" \
  -validate-limit "$VALIDATE_LIMIT" \
  -num-ctx "$NUM_CTX" \
  -batch-size "$BATCH_SIZE" \
  -num-gpu-layers "$NUM_GPU_LAYERS" \
  -artifact-dir "$WORK_DIR" \
  -output "$WORK_ARTIFACT"
echo

# 3. Gate on mean_kl. The artifact's validation block is the candidate result.
echo "[3/5] Gate: mean_kl <= $MAX_MEAN_KL"
mean_kl="$(jq -r '.validation.metrics.mean_kl' "$WORK_ARTIFACT")"
perplexity="$(jq -r '.validation.metrics.perplexity' "$WORK_ARTIFACT")"
mean_nll="$(jq -r '.validation.metrics.mean_nll' "$WORK_ARTIFACT")"
key_layers="$(jq -r '.key_cache_layer_types' "$WORK_ARTIFACT")"

echo "  mean_nll=$mean_nll  perplexity=$perplexity  mean_kl=$mean_kl"
echo "  key_cache_layer_types=$key_layers"

# Use awk to compare floats; bash arithmetic is integer-only.
gate_pass="$(awk -v a="$mean_kl" -v b="$MAX_MEAN_KL" 'BEGIN { print (a+0 <= b+0) ? 1 : 0 }')"
if [[ "$gate_pass" != "1" ]]; then
  echo "FAIL: mean_kl=$mean_kl exceeds budget $MAX_MEAN_KL." >&2
  echo "      Try increasing -t (more q8_0-protected layers) and re-running." >&2
  echo "      Artifact (NOT bundled) is at $WORK_ARTIFACT" >&2
  exit 1
fi
echo "  gate: PASS"
echo

if [[ "$SKIP_BUNDLE" == "1" ]]; then
  echo "[4/5] --no-bundle: stopping before manifest update"
  echo "      Artifact at $WORK_ARTIFACT"
  exit 0
fi

# 4. Bundle: copy artifact under stable name, patch the manifest atomically.
echo "[4/5] Bundling: $ARTIFACT_DEST + manifest.json entry ($ARCH, $FTYPE, $HEAD_DIM)"
cp "$WORK_ARTIFACT" "$ARTIFACT_DEST"

manifest_tmp="$(mktemp "$MANIFEST.XXXXXX")"
trap 'rm -f "$manifest_tmp"' EXIT

# Build the new entry. Replace any existing entry that matches the same
# (architecture, file_type, head_dim) tuple — runtime resolution uses
# case-insensitive compare, so we match on lowercase here too.
jq \
  --arg arch        "$ARCH" \
  --arg ftype       "$FTYPE" \
  --argjson headdim "$HEAD_DIM" \
  --arg artifact    "$ARTIFACT_BASENAME" \
  --arg hint        "$MODEL_HINT" \
  --arg validated   "$VALIDATED_DATE" \
  --argjson kl      "$mean_kl" \
  --argjson ppl     "$perplexity" \
  '
    .entries |=
      ([
        .[]
        | select(
            (.architecture | ascii_downcase) != ($arch | ascii_downcase)
            or (.file_type | ascii_downcase) != ($ftype | ascii_downcase)
            or (.head_dim | tonumber) != $headdim
          )
      ] + [{
        "architecture":      $arch,
        "file_type":         $ftype,
        "head_dim":          $headdim,
        "artifact":          $artifact,
        "model_hint":        $hint,
        "validated":         $validated,
        "phase0_mean_kl":    $kl,
        "phase0_perplexity": $ppl
      }])
  ' "$MANIFEST" > "$manifest_tmp"

# jq sanity-check on the result before swapping in.
jq -e '.entries | length > 0' "$manifest_tmp" >/dev/null
mv "$manifest_tmp" "$MANIFEST"
trap - EXIT
echo "  manifest.json updated"
echo

# 5. Loader test — proves the new entry is parseable and the artifact path resolves.
if [[ "$SKIP_LOADER_TEST" == "1" ]]; then
  echo "[5/5] --no-loader-test: skipping go test"
  exit 0
fi
echo "[5/5] go test ./tools/turboquant/calibration/..."
go test -count=1 ./tools/turboquant/calibration/...
echo
echo "OK: turboquant-adaptive ready for $MODEL ($ARCH, $FTYPE, head_dim=$HEAD_DIM)"
echo "    OLLAMA_KV_CACHE_TYPE=turboquant-adaptive ollama run $MODEL"
