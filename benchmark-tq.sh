#!/bin/bash
# TurboQuant Benchmark: turbo4 vs f16
# Usage: ./benchmark-tq.sh
set -euo pipefail

OLLAMA_BIN="./ollama-tq"
PORT=9997
HOST="localhost:$PORT"
RESULTS_FILE="benchmark-results-$(date +%Y%m%d-%H%M%S).md"

# Prompts: short (speed test) and long (quality test)
PROMPT_SHORT="What is 2+2? Answer in one word."
PROMPT_MEDIUM="Explain the Pythagorean theorem in exactly 3 sentences."
PROMPT_LONG="Write a step-by-step guide for making scrambled eggs. Include 5 steps with details."

# Models to test
MODELS=("qwen2.5:7b")
MODEL_STORES=("") # empty = default store

# Check if qwen3.6:27b fits (needs ~17GB + KV cache)
FREE_VRAM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -rn | head -1)
if [ "$FREE_VRAM" -gt 20000 ]; then
    MODELS+=("qwen3.6:27b")
    MODEL_STORES+=("/home/yannik/.ollama/models-d256")
fi

KV_TYPES=("" "turbo4")  # empty = f16 default
KV_LABELS=("f16" "turbo4")

log() { echo "[$(date +%H:%M:%S)] $*"; }

wait_for_server() {
    for i in $(seq 1 60); do
        ss -tlnp 2>/dev/null | grep -q ":$PORT " && return 0
        sleep 1
    done
    echo "Server failed to start" >&2
    return 1
}

stop_server() {
    pkill -f "ollama-tq.*serve" 2>/dev/null || true
    sleep 2
}

get_vram() {
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | tr '\n' '|'
}

run_benchmark() {
    local model="$1"
    local kv_type="$2"
    local kv_label="$3"
    local model_store="$4"
    local prompt="$5"
    local prompt_label="$6"

    # Build env vars
    local env_args="OLLAMA_HOST=$HOST OLLAMA_FLASH_ATTENTION=1 OLLAMA_NEW_ENGINE=1 OLLAMA_CONTEXT_LENGTH=4096"
    if [ -n "$kv_type" ]; then
        env_args="$env_args OLLAMA_KV_CACHE_TYPE=$kv_type"
    fi
    if [ -n "$model_store" ]; then
        env_args="$env_args OLLAMA_MODELS=$model_store"
    fi

    log "Starting server: $model / $kv_label"
    stop_server

    # Start server
    eval "$env_args $OLLAMA_BIN serve" &>/tmp/bench-server.log &
    wait_for_server || return 1

    # Warm up: load model
    log "Loading model..."
    local vram_before=$(get_vram)
    OLLAMA_HOST=$HOST timeout 120 $OLLAMA_BIN run "$model" "hi" &>/dev/null || true
    sleep 2
    local vram_after=$(get_vram)

    # Get KV cache size from logs
    local kv_size=$(grep "kv cache" /tmp/bench-server.log | tail -1 | grep -oP 'size="\K[^"]+' || echo "N/A")

    # Run timed generation
    log "Running: $prompt_label"
    local start_time=$(date +%s%N)
    local output=$(OLLAMA_HOST=$HOST timeout 60 $OLLAMA_BIN run "$model" "$prompt" 2>/dev/null)
    local end_time=$(date +%s%N)
    local elapsed_ms=$(( (end_time - start_time) / 1000000 ))

    # Count tokens (rough: words * 1.3)
    local word_count=$(echo "$output" | wc -w)
    local est_tokens=$(( word_count * 13 / 10 ))
    local tokens_per_sec=0
    if [ "$elapsed_ms" -gt 0 ]; then
        tokens_per_sec=$(( est_tokens * 1000 / elapsed_ms ))
    fi

    # Get eval stats from API
    local eval_stats=$(OLLAMA_HOST=$HOST curl -s http://$HOST/api/generate -d "{\"model\":\"$model\",\"prompt\":\"ping\",\"stream\":false}" 2>/dev/null)
    local eval_rate=$(echo "$eval_stats" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"{d.get('eval_count',0) / (d.get('eval_duration',1)/1e9):.1f}\")" 2>/dev/null || echo "N/A")

    # Print result row
    echo "| $model | $kv_label | $prompt_label | ${elapsed_ms}ms | ~${tokens_per_sec} t/s | $eval_rate t/s | $kv_size | $vram_after |"

    # Save output for quality comparison
    echo -e "\n### $model / $kv_label / $prompt_label\n\`\`\`\n$output\n\`\`\`" >> /tmp/bench-outputs.txt

    stop_server
}

# === Main ===

echo "# TurboQuant Benchmark Results" > "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"
echo "Date: $(date)" >> "$RESULTS_FILE"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ', ')" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

echo "| Model | KV Cache | Prompt | Wall Time | Est. tok/s | Eval tok/s | KV Size | VRAM (GPU0\|GPU1) |" >> "$RESULTS_FILE"
echo "|-------|----------|--------|-----------|-----------|------------|---------|-------------------|" >> "$RESULTS_FILE"

> /tmp/bench-outputs.txt

for idx in "${!MODELS[@]}"; do
    model="${MODELS[$idx]}"
    store="${MODEL_STORES[$idx]}"

    for kv_idx in "${!KV_TYPES[@]}"; do
        kv="${KV_TYPES[$kv_idx]}"
        label="${KV_LABELS[$kv_idx]}"

        for prompt_pair in "short:$PROMPT_SHORT" "medium:$PROMPT_MEDIUM" "long:$PROMPT_LONG"; do
            plabel="${prompt_pair%%:*}"
            ptext="${prompt_pair#*:}"
            result=$(run_benchmark "$model" "$kv" "$label" "$store" "$ptext" "$plabel" 2>&1 | grep "^|" || echo "| $model | $label | $plabel | ERROR | - | - | - | - |")
            echo "$result" | tee -a "$RESULTS_FILE"
        done
    done
done

echo "" >> "$RESULTS_FILE"
echo "## Output Quality Comparison" >> "$RESULTS_FILE"
cat /tmp/bench-outputs.txt >> "$RESULTS_FILE"

echo ""
echo "Results saved to: $RESULTS_FILE"
log "Benchmark complete"
