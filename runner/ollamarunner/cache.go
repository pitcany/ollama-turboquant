package ollamarunner

import (
	"errors"
	"fmt"
	"log/slog"
	"math"
	"strings"
	"time"

	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
	"github.com/ollama/ollama/tools/turboquant/calibration"
)

type InputCache struct {
	// context window size (per slot)
	numCtx int32

	// does the cache store data or do we need to always send the full input?
	// note that when enabled is false the underlying cache may either be nil
	// or a non-nil dummy that doesn't actually store anything
	enabled bool

	// individual KV caches
	slots []InputCacheSlot

	// optimize cache eviction for multiple users
	multiUserCache bool

	cache kvcache.Cache
}

func NewInputCache(model model.Model, kvCacheType string, keyCacheLayerTypes string, kvSize int32, numSlots int, batchSize int, multiUserCache bool) (*InputCache, error) {
	numCtx := kvSize / int32(numSlots)

	if int(numCtx) < batchSize {
		return nil, fmt.Errorf("kv size must be at least as large as batch size * parallel (kv: %v batch: %v parallel: %v)", kvSize, batchSize, numSlots)
	}

	slots := make([]InputCacheSlot, numSlots)

	for i := range slots {
		slots[i] = InputCacheSlot{Id: i}
	}

	cache := model.Config().Cache
	if cache != nil {
		backend := model.Backend()
		kvCacheType, keyCacheLayerTypes = downgradeTurboForBackend(backend, kvCacheType, keyCacheLayerTypes)
		if err := initKVCache(cache, backend, kvCacheType, keyCacheLayerTypes, numSlots, int(numCtx), batchSize); err != nil {
			return nil, err
		}
	}

	return &InputCache{
		numCtx:         numCtx,
		enabled:        cache != nil,
		slots:          slots,
		multiUserCache: multiUserCache,
		cache:          cache,
	}, nil
}

func initKVCache(cache kvcache.Cache, backend ml.Backend, kvCacheType, keyCacheLayerTypes string, maxSequences, capacity, maxBatch int) error {
	keyDType, valueDType := kvCacheTypesFromStr(kvCacheType)
	if keyDType == valueDType {
		cache.Init(backend, keyDType, maxSequences, capacity, maxBatch)
	} else {
		split, ok := cache.(interface {
			InitSplit(ml.Backend, ml.DType, ml.DType, int, int, int)
		})
		if !ok {
			// Hybrid attention+SSM caches (e.g. qwen3.5/3.6 *HybridCache) do
			// not implement split K/V dtypes. Fall back to the higher-precision
			// (key) dtype for both K and V so the model still loads with a
			// quantized cache; warn so the operator knows the value dtype was
			// upgraded. This keeps the runtime usable for users who set
			// OLLAMA_KV_CACHE_TYPE=kq8-vturbo4 or turboquant-adaptive on
			// hybrid models.
			slog.Warn("cache does not support split key/value dtypes; using key dtype for both",
				"requested_key_dtype", keyDType,
				"requested_value_dtype", valueDType,
				"effective_dtype", keyDType,
			)
			cache.Init(backend, keyDType, maxSequences, capacity, maxBatch)
		} else {
			split.InitSplit(backend, keyDType, valueDType, maxSequences, capacity, maxBatch)
		}
	}

	if strings.TrimSpace(keyCacheLayerTypes) == "" {
		return nil
	}
	overrides, err := calibration.ParseKeyLayerOverrides(keyCacheLayerTypes)
	if err != nil {
		return fmt.Errorf("parse key_cache_layer_types: %w", err)
	}
	if len(overrides) == 0 {
		return nil
	}
	setter, ok := cache.(interface {
		SetKeyLayerDTypes(map[int]ml.DType)
	})
	if !ok {
		// Same fallback rationale as the split-init path: if the cache cannot
		// honor per-layer key dtype overrides (hybrid caches today), drop the
		// overrides with a warning instead of failing the model load. The
		// uniform key dtype already chosen above is the conservative choice.
		slog.Warn("cache does not support per-layer key dtype overrides; dropping overrides",
			"spec", calibration.CanonicalKeyLayerSpec(overrides),
		)
		return nil
	}
	dtypes := make(map[int]ml.DType, len(overrides))
	for layer, name := range overrides {
		dtypes[layer] = kvCacheTypeFromStr(name)
	}
	setter.SetKeyLayerDTypes(dtypes)
	slog.Info("applied per-layer key cache dtype overrides", "spec", calibration.CanonicalKeyLayerSpec(overrides))
	return nil
}

func kvCacheTypesFromStr(s string) (ml.DType, ml.DType) {
	if strings.EqualFold(s, "kq8-vturbo4") {
		return ml.DTypeQ80, ml.DTypeTurbo4
	}
	dtype := kvCacheTypeFromStr(s)
	return dtype, dtype
}

// downgradeTurboForBackend rewrites Turbo* cache dtypes to q8_0 if the
// backend devices do not include a CUDA library, since the production Turbo
// flash-attention kernels are CUDA-only today. Returns the (possibly
// rewritten) kvCacheType and per-layer key spec, with a warning logged when
// any rewriting is performed.
func downgradeTurboForBackend(backend ml.Backend, kvCacheType, keyCacheLayerTypes string) (string, string) {
	if !hasTurboDtype(kvCacheType, keyCacheLayerTypes) {
		return kvCacheType, keyCacheLayerTypes
	}
	if backend == nil || backendSupportsTurbo(backend) {
		return kvCacheType, keyCacheLayerTypes
	}
	newKVType := rewriteTurboCacheType(kvCacheType)
	newSpec, err := rewriteTurboLayerSpec(keyCacheLayerTypes)
	if err != nil {
		slog.Warn("failed to downgrade per-layer Turbo dtypes; clearing overrides", "error", err)
		newSpec = ""
	}
	slog.Warn("Turbo* KV cache requires a CUDA backend; downgrading to q8_0",
		"requested_kv_cache_type", kvCacheType,
		"requested_key_cache_layer_types", keyCacheLayerTypes,
		"effective_kv_cache_type", newKVType,
		"effective_key_cache_layer_types", newSpec,
	)
	return newKVType, newSpec
}

func backendSupportsTurbo(backend ml.Backend) bool {
	for _, dev := range backend.BackendDevices() {
		if strings.EqualFold(dev.Library, "CUDA") {
			return true
		}
	}
	return false
}

func hasTurboDtype(kvCacheType, layerSpec string) bool {
	if isTurboDtypeName(kvCacheType) {
		return true
	}
	if strings.EqualFold(kvCacheType, "kq8-vturbo4") {
		return true
	}
	overrides, err := calibration.ParseKeyLayerOverrides(layerSpec)
	if err != nil {
		return false
	}
	for _, dtype := range overrides {
		if isTurboDtypeName(dtype) {
			return true
		}
	}
	return false
}

func isTurboDtypeName(s string) bool {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "turbo2", "turbo3", "turbo4":
		return true
	default:
		return false
	}
}

func rewriteTurboCacheType(s string) string {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "turbo2", "turbo3", "turbo4":
		return "q8_0"
	case "kq8-vturbo4":
		// Both halves of the split lose Turbo, collapse to uniform q8_0.
		return "q8_0"
	default:
		return s
	}
}

func rewriteTurboLayerSpec(spec string) (string, error) {
	if strings.TrimSpace(spec) == "" {
		return "", nil
	}
	overrides, err := calibration.ParseKeyLayerOverrides(spec)
	if err != nil {
		return "", err
	}
	for layer, dtype := range overrides {
		if isTurboDtypeName(dtype) {
			overrides[layer] = "q8_0"
		}
	}
	return calibration.CanonicalKeyLayerSpec(overrides), nil
}

func kvCacheTypeFromStr(s string) ml.DType {
	switch strings.ToLower(s) {
	case "q8_0":
		return ml.DTypeQ80
	case "q4_0":
		return ml.DTypeQ40
	case "turbo2":
		return ml.DTypeTurbo2
	case "turbo3":
		return ml.DTypeTurbo3
	case "turbo4":
		return ml.DTypeTurbo4
	default:
		return ml.DTypeF16
	}
}

func (c *InputCache) Close() {
	if c != nil && c.cache != nil {
		c.cache.Close()
	}
}

// Locking: Operations on InputCacheSlot (including finding one
// through LoadCacheSlot) require a lock to be held that serializes
// these operations with each other and processBatch

type InputCacheSlot struct {
	// Index in the KV cache
	Id int

	// Inputs that are stored in the KV cache
	Inputs []*input.Input

	// is this cache actively being processed as part of a sequence?
	InUse bool

	// last time this cache was used (as of start of processing)
	lastUsed time.Time
}

func (c *InputCache) LoadCacheSlot(prompt []*input.Input, cachePrompt bool) (*InputCacheSlot, []*input.Input, error) {
	var slot *InputCacheSlot
	var numPast int32
	var err error

	// In single-user scenarios, the longest cache slot works fine for getting good input
	// cache hit rates and it keeps the footprint of the cache small, which improves throughput.
	// For multiple users, the "best" cache slot produces better input cache hit rates
	// at the cost of worse performance when we miss the input cache.
	if !c.multiUserCache {
		slot, numPast, err = c.findLongestCacheSlot(prompt)
	} else {
		slot, numPast, err = c.findBestCacheSlot(prompt)
	}
	if err != nil {
		return nil, nil, err
	}

	if !cachePrompt {
		numPast = 0
	}

	slot.InUse = true
	slot.lastUsed = time.Now()

	if numPast == int32(len(prompt)) {
		// Leave one input to sample so we can get a response
		numPast--
	}

	if c.cache != nil {
		if numPast > 0 {
			// Recurrent caches use checkpoints to pick a safe resume position.
			if cc, ok := c.cache.(kvcache.CheckpointCache); ok {
				if restored, ok := cc.PrepareRestore(slot.Id, numPast); ok {
					numPast = restored
				} else {
					numPast = 0
				}
			} else if !c.cache.CanResume(slot.Id, numPast) {
				numPast = 0
			}
		}

		err = c.cache.Remove(slot.Id, numPast, math.MaxInt32)
		if err != nil {
			// Some models don't support partial erasure
			err = c.cache.Remove(slot.Id, 0, math.MaxInt32)
			if err != nil {
				return nil, nil, err
			}
			numPast = 0
		}
	}

	slog.Debug("loading cache slot", "id", slot.Id, "cache", len(slot.Inputs), "prompt", len(prompt),
		"used", numPast, "remaining", int32(len(prompt))-numPast)

	slot.Inputs = prompt[:numPast]
	prompt = prompt[numPast:]

	return slot, prompt, nil
}

func (c *InputCache) findLongestCacheSlot(prompt []*input.Input) (*InputCacheSlot, int32, error) {
	longest := int32(-1)
	var longestSlot *InputCacheSlot

	for i, s := range c.slots {
		if s.InUse {
			continue
		}

		count := countCommonPrefix(s.Inputs, prompt)
		if count > longest {
			longest = count
			longestSlot = &c.slots[i]
		}
	}

	if longestSlot == nil {
		return nil, 0, errors.New("no available cache slots")
	}

	return longestSlot, longest, nil
}

func (c *InputCache) findBestCacheSlot(prompt []*input.Input) (*InputCacheSlot, int32, error) {
	oldest := time.Now()
	var oldestSlot *InputCacheSlot

	longest := int32(-1)
	var longestSlot *InputCacheSlot

	for i, s := range c.slots {
		count := countCommonPrefix(s.Inputs, prompt)
		if count > longest {
			longest = count
			longestSlot = &c.slots[i]
		}

		if s.lastUsed.Compare(oldest) < 0 && !s.InUse {
			oldest = s.lastUsed
			oldestSlot = &c.slots[i]
		}
	}

	if longest == int32(len(longestSlot.Inputs)) && !longestSlot.InUse {
		return longestSlot, longest, nil
	}

	if oldestSlot.InUse {
		return nil, 0, errors.New("no available cache slots")
	}

	if len(oldestSlot.Inputs) != 0 {
		slog.Debug("evicting cache slot", "id", oldestSlot.Id, "inputs", len(oldestSlot.Inputs),
			"used", oldestSlot.lastUsed)
	}

	if longest > 0 && longestSlot != oldestSlot {
		slog.Debug("forking cache slot", "src", longestSlot.Id, "dst", oldestSlot.Id, "inputs", longest, "total",
			len(longestSlot.Inputs))
		oldestSlot.Inputs = make([]*input.Input, longest)
		copy(oldestSlot.Inputs, longestSlot.Inputs[:longest])
		if c.cache != nil {
			c.cache.CopyPrefix(longestSlot.Id, oldestSlot.Id, longest)
		}
	}

	return oldestSlot, longest, nil
}

func countCommonPrefix(a []*input.Input, b []*input.Input) int32 {
	var count int32

	for i := range a {
		if i >= len(b) {
			break
		}

		if a[i].Token != b[i].Token || a[i].MultimodalHash != b[i].MultimodalHash {
			break
		}

		count++
	}

	return count
}

// ShiftDiscard computes how many inputs can be discarded from the cache. Inputs in the same batch
// are discarded together.
func (c *InputCache) ShiftDiscard(inputs []*input.Input, numKeep int32) int32 {
	targetFree := max((c.numCtx-numKeep)/2, 1)
	currentFree := c.numCtx - int32(len(inputs))

	var discard, sameBatch int32
	for _, input := range inputs[numKeep:] {
		if sameBatch <= 0 && currentFree >= targetFree {
			break
		}

		sameBatch--
		currentFree++
		discard++

		if input.SameBatch > 0 {
			sameBatch = int32(input.SameBatch)
		}
	}

	return discard
}

type ErrReprocessInputs struct {
	Inputs []*input.Input
}

func (e *ErrReprocessInputs) Error() string {
	return fmt.Sprintf("kv cache shift not supported, inputs need reprocessing (input count: %v)", len(e.Inputs))
}

// Frees up space in the KV cache by deleting the oldest half of history and shifting
// the newest half into that space (saving numKeep inputs at the beginning).
//
// Assumes that at least 1 entry can be freed up by shifting (i.e. numKeep < numCtx)
func (c *InputCache) ShiftCacheSlot(slot *InputCacheSlot, numKeep int32) error {
	if numKeep >= c.numCtx {
		return fmt.Errorf("unable to shift context - keep exceeds context (keep: %v context: %v)", numKeep, c.numCtx)
	}

	inputLen := int32(len(slot.Inputs))
	discard := c.ShiftDiscard(slot.Inputs, numKeep)

	if discard <= 0 {
		return nil
	}

	slog.Debug("context limit hit - shifting", "id", slot.Id, "limit", c.numCtx, "input", len(slot.Inputs),
		"keep", numKeep, "discard", discard)

	if c.cache != nil {
		err := c.cache.Remove(slot.Id, numKeep, numKeep+discard)
		if err != nil {
			slog.Debug("kv cache removal unsupported, clearing cache and returning inputs for reprocessing",
				"id", slot.Id, "error", err)

			// Create new input slice with preserved tokens (numKeep + remaining tokens after discard)
			newInputs := make([]*input.Input, numKeep+inputLen-(numKeep+discard))
			copy(newInputs[:numKeep], slot.Inputs[:numKeep])
			copy(newInputs[numKeep:], slot.Inputs[numKeep+discard:])

			// Reset the cache
			_ = c.cache.Remove(slot.Id, 0, math.MaxInt32)
			slot.Inputs = []*input.Input{}

			// Return error with inputs that need to be reprocessed
			return &ErrReprocessInputs{Inputs: newInputs}
		}
	}

	for i := numKeep + discard; i < inputLen; i++ {
		slot.Inputs[i-discard] = slot.Inputs[i]
	}
	slot.Inputs = slot.Inputs[:inputLen-discard]

	return nil
}
