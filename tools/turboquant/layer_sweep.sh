#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-qwen2.5:7b}"
SNAPSHOT="${SNAPSHOT:-tools/turboquant/testdata/qwen25_7b_phase0_tokens.json}"
BASE_KV="${BASE_KV:-turbo4}"
REFERENCE_KV="${REFERENCE_KV:-f16}"
LAYER_DTYPE="${LAYER_DTYPE:-q8_0}"
LIMIT="${LIMIT:-4}"
NUM_CTX="${NUM_CTX:-1024}"
BATCH_SIZE="${BATCH_SIZE:-512}"
NUM_GPU_LAYERS="${NUM_GPU_LAYERS:-999}"
OUTPUT="${OUTPUT:-/tmp/turboquant-layer-sweep-limit${LIMIT}-${LAYER_DTYPE}.csv}"
LOG_DIR="${LOG_DIR:-/tmp/turboquant-layer-sweep-logs-limit${LIMIT}-${LAYER_DTYPE}}"

if [[ -z "${LAYERS:-}" ]]; then
  LAYERS="$(seq 0 27)"
fi

export GOCACHE="${GOCACHE:-/tmp/ollama-build-gocache}"
export OLLAMA_LIBRARY_PATH="${OLLAMA_LIBRARY_PATH:-${PWD}/build/lib/ollama}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "${LOG_DIR}"
printf 'layer,key_layer_types,sequences,tokens,mean_nll,perplexity,mean_kl,duration_ms,status\n' > "${OUTPUT}"

for layer in ${LAYERS}; do
  spec="${layer}:${LAYER_DTYPE}"
  printf 'running layer %s with %s, limit=%s\n' "${layer}" "${spec}" "${LIMIT}" >&2

  set +e
  json="$(
    go run ./cmd/turboquant-eval \
      -engine go \
      -model "${MODEL}" \
      -snapshot "${SNAPSHOT}" \
      -kv-cache-type "${BASE_KV}" \
      -key-cache-layer-types "${spec}" \
      -reference-kv-cache-type "${REFERENCE_KV}" \
      -num-ctx "${NUM_CTX}" \
      -batch-size "${BATCH_SIZE}" \
      -num-gpu-layers "${NUM_GPU_LAYERS}" \
      -flash-attention=true \
      -limit "${LIMIT}" \
      -format json 2> "${LOG_DIR}/layer-${layer}.log"
  )"
  status=$?
  set -e

  if [[ "${status}" -ne 0 ]]; then
    printf 'layer %s failed; see %s\n' "${layer}" "${LOG_DIR}/layer-${layer}.log" >&2
    printf '%s,%s,,,,,,,failed\n' "${layer}" "${spec}" >> "${OUTPUT}"
    continue
  fi

  python3 -c '
import csv
import json
import sys

layer = sys.argv[1]
spec = sys.argv[2]
result = json.load(sys.stdin)
metrics = result["metrics"]
csv.writer(sys.stdout).writerow([
    layer,
    spec,
    result["num_sequences"],
    metrics["token_count"],
    "{:.8f}".format(metrics["mean_nll"]),
    "{:.8f}".format(metrics["perplexity"]),
    "{:.8f}".format(metrics["mean_kl"]),
    result["duration_ms"],
    "ok",
])
' "${layer}" "${spec}" <<< "${json}" >> "${OUTPUT}"
done

printf 'wrote %s\n' "${OUTPUT}" >&2
