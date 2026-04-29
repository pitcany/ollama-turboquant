#!/bin/bash
# TurboQuant Context Length Benchmark
# Tests at context sizes where f16 OOMs but turbo4 fits
set -euo pipefail

OLLAMA_BIN="$HOME/.local/bin/ollama-tq"
PORT=9997
HOST="localhost:$PORT"
RESULTS_FILE="benchmark-context-$(date +%Y%m%d-%H%M%S).md"

log() { echo "[$(date +%H:%M:%S)] $*"; }

wait_for_server() {
    for i in $(seq 1 60); do
        curl -sf http://$HOST/ &>/dev/null && return 0
        sleep 1
    done
    echo "Server failed to start" >&2
    return 1
}

stop_server() {
    pkill -f "ollama-tq.*:$PORT" 2>/dev/null || true
    sleep 2
}

# Generate a long prompt of approximately N tokens (1 token ≈ 4 chars)
make_long_prompt() {
    local target_tokens=$1
    local chars=$((target_tokens * 4))
    local preamble="Read the following text carefully, then answer the question at the end.\n\n"
    local question="\n\nQuestion: What was the very first word in this text? Answer in one word."
    # Fill with repeated sentences
    local filler="The quick brown fox jumps over the lazy dog near the river bank. "
    python3 -c "
preamble = '''Read the following text carefully, then answer the question at the end.\n\n'''
question = '''\n\nQuestion: What was the very first word of the filler text? Answer in one word.'''
filler = 'The quick brown fox jumps over the lazy dog near the river bank. '
target = $chars - len(preamble) - len(question)
repeats = target // len(filler) + 1
print(preamble + (filler * repeats)[:target] + question)
"
}

run_context_test() {
    local model="$1"
    local kv_type="$2"
    local kv_label="$3"
    local ctx_len="$4"
    local prompt_tokens="$5"

    log "Testing $model / $kv_label / ctx=${ctx_len} / prompt~${prompt_tokens} tokens"
    stop_server

    local env="OLLAMA_HOST=$HOST OLLAMA_FLASH_ATTENTION=1 OLLAMA_NEW_ENGINE=1"
    env="$env OLLAMA_CONTEXT_LENGTH=$ctx_len OLLAMA_NUM_GPU=999"
    [ -n "$kv_type" ] && env="$env OLLAMA_KV_CACHE_TYPE=$kv_type"

    eval "CUDA_VISIBLE_DEVICES=0 $env $OLLAMA_BIN serve" &>/tmp/bench-ctx-server.log &
    local srv_pid=$!

    if ! wait_for_server; then
        echo "| $model | $kv_label | $ctx_len | ${prompt_tokens} | FAIL (server) | - | - |"
        stop_server
        return
    fi

    # Generate prompt and write request body to temp file (avoids ARG_MAX for large prompts)
    log "  Generating ${prompt_tokens}-token prompt..."
    make_long_prompt "$prompt_tokens" > /tmp/bench-ctx-prompt.txt
    python3 -c "
import json
with open('/tmp/bench-ctx-prompt.txt') as f:
    prompt = f.read()
print(json.dumps({'model':'$model','prompt':prompt,'stream':False,'options':{'num_predict':16,'temperature':0}}))
" > /tmp/bench-ctx-request.json

    log "  Sending request..."
    local response
    response=$(curl -sf --max-time 600 http://$HOST/api/generate \
        -d @/tmp/bench-ctx-request.json 2>/tmp/bench-ctx-curl-err.txt)
    local rc=$?

    if [ $rc -ne 0 ] || [ -z "$response" ]; then
        # Check if it was an OOM
        local err_msg="FAIL"
        if grep -qi "out of memory\|CUDA error\|OOM\|alloc" /tmp/bench-ctx-server.log 2>/dev/null; then
            err_msg="OOM"
        elif grep -qi "SIGABRT\|SIGSEGV\|fatal" /tmp/bench-ctx-server.log 2>/dev/null; then
            err_msg="CRASH"
        elif [ $rc -eq 28 ]; then
            err_msg="TIMEOUT"
        fi
        echo "| $model | $kv_label | $ctx_len | ${prompt_tokens} | **$err_msg** | - | - |"
        stop_server
        return
    fi

    # Parse response
    local answer eval_rate prompt_tps kv_size
    answer=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('response','').strip()[:80])" 2>/dev/null || echo "?")
    eval_rate=$(echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"{d['eval_count']/(d['eval_duration']/1e9):.1f}\")" 2>/dev/null || echo "?")
    prompt_tps=$(echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"{d['prompt_eval_count']/(d['prompt_eval_duration']/1e9):.1f}\")" 2>/dev/null || echo "?")
    kv_size=$(grep "kv cache" /tmp/bench-ctx-server.log | tail -1 | grep -oP 'size="\K[^"]+' || echo "?")

    echo "| $model | $kv_label | $ctx_len | ${prompt_tokens} | $prompt_tps t/s | $eval_rate t/s | $kv_size | \`$answer\` |"
    stop_server
}

# === Main ===
{
echo "# TurboQuant Context Length Benchmark"
echo ""
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader -i 0)"
echo ""
echo "| Model | KV Cache | Context | Prompt Tokens | Prompt Speed | Gen Speed | KV Size | Answer |"
echo "|-------|----------|---------|---------------|-------------|-----------|---------|--------|"
} > "$RESULTS_FILE"

# qwen2.5:7b — test at 4K, 32K, 64K, 128K
for ctx_prompt in "4096:2000" "32768:16000" "65536:32000" "131072:64000"; do
    ctx="${ctx_prompt%%:*}"
    ptok="${ctx_prompt#*:}"
    for kv_pair in ":f16" "turbo4:turbo4"; do
        kv="${kv_pair%%:*}"
        label="${kv_pair#*:}"
        result=$(run_context_test "qwen2.5:7b" "$kv" "$label" "$ctx" "$ptok" 2>&1 | grep "^|")
        echo "$result" | tee -a "$RESULTS_FILE"
    done
done

# qwen3.6:27b — test at 4K, 32K, 64K, 98K
for ctx_prompt in "4096:2000" "32768:16000" "65536:32000" "98304:48000"; do
    ctx="${ctx_prompt%%:*}"
    ptok="${ctx_prompt#*:}"
    for kv_pair in ":f16" "turbo4:turbo4"; do
        kv="${kv_pair%%:*}"
        label="${kv_pair#*:}"
        result=$(run_context_test "qwen3.6:27b" "$kv" "$label" "$ctx" "$ptok" 2>&1 | grep "^|")
        echo "$result" | tee -a "$RESULTS_FILE"
    done
done

echo "" >> "$RESULTS_FILE"
echo "Results saved to: $RESULTS_FILE"
log "Benchmark complete"
