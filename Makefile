.DEFAULT_GOAL := help

GO ?= go

TURBOQUANT_PHASE0_MODEL ?= qwen2.5:7b
TURBOQUANT_PHASE0_SNAPSHOT ?= tools/turboquant/testdata/qwen25_7b_phase0_tokens.json
TURBOQUANT_PHASE0_LIMIT ?= 4
TURBOQUANT_PHASE0_NUM_CTX ?= 1024
TURBOQUANT_PHASE0_BATCH_SIZE ?= 512
TURBOQUANT_PHASE0_NUM_GPU_LAYERS ?= 999

.PHONY: help
help:
	@printf 'Targets:\n'
	@printf '  turboquant-phase0  Run TurboQuant Phase 0 f16 and kq8-vturbo4 quality evals\n'

.PHONY: turboquant-phase0
turboquant-phase0:
	$(GO) run ./cmd/turboquant-eval \
		-engine go \
		-model $(TURBOQUANT_PHASE0_MODEL) \
		-snapshot $(TURBOQUANT_PHASE0_SNAPSHOT) \
		-kv-cache-type f16 \
		-reference-kv-cache-type f16 \
		-limit $(TURBOQUANT_PHASE0_LIMIT) \
		-num-ctx $(TURBOQUANT_PHASE0_NUM_CTX) \
		-batch-size $(TURBOQUANT_PHASE0_BATCH_SIZE) \
		-num-gpu-layers $(TURBOQUANT_PHASE0_NUM_GPU_LAYERS) \
		-flash-attention=true \
		-format json > /tmp/turboquant-phase0-f16.json
	$(GO) run ./cmd/turboquant-eval \
		-engine go \
		-model $(TURBOQUANT_PHASE0_MODEL) \
		-snapshot $(TURBOQUANT_PHASE0_SNAPSHOT) \
		-kv-cache-preset kq8-vturbo4 \
		-reference-kv-cache-type f16 \
		-limit $(TURBOQUANT_PHASE0_LIMIT) \
		-num-ctx $(TURBOQUANT_PHASE0_NUM_CTX) \
		-batch-size $(TURBOQUANT_PHASE0_BATCH_SIZE) \
		-num-gpu-layers $(TURBOQUANT_PHASE0_NUM_GPU_LAYERS) \
		-flash-attention=true \
		-format json > /tmp/turboquant-phase0-kq8-vturbo4.json
