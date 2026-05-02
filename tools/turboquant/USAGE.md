# TurboQuant: end-user runbook

A focused runbook for running models on this fork of Ollama with
TurboQuant KV-cache compression. For the design and calibration side of
the same system, see [`README.md`](README.md). For the rigorous
provenance ledger of every fix on this fork, see
[`../../TURBOQUANT-DEBUG-LOG.md`](../../TURBOQUANT-DEBUG-LOG.md).

This document assumes you are running on a CUDA host (the Turbo* flash
attention kernels are CUDA-only today). On Metal/ROCm/Vulkan/CPU the
runtime gracefully downgrades to `q8_0` with a loud warning — see
"Non-CUDA backends" below.

---

## 1. Install the binary

### Option A — prebuilt release (recommended for end users)

```bash
curl -fsSL https://raw.githubusercontent.com/pitcany/ollama-turboquant/turboquant/runtime/scripts/install-tq.sh \
  | bash -s -- --systemd --symlink-ollama
```

This pulls the latest release tarball from GitHub, drops files under
`~/.local/{bin,lib,share}`, installs the systemd unit (with sudo
prompt), symlinks `ollama -> ollama-tq` so the standard `ollama` CLI
renders the `KV Cache` section, and appends `OLLAMA_HOST=http://localhost:8001`
to `~/.bashrc`. Drop any of the flags to skip parts:

- `--systemd` — install + enable the systemd unit
- `--symlink-ollama` — alias `ollama` to the fork's binary
- `--no-bashrc` — skip the OLLAMA_HOST export
- `--version vX.Y.Z` — pin a specific release tag (default: latest)
- `--local /path/to/tarball` — offline install from a downloaded tarball
- `--force` — overwrite existing files instead of backing up
- `--prefix DIR` — install elsewhere than `$HOME/.local`

After install, open a new shell (so `OLLAMA_HOST` takes effect) and
verify:

```bash
ollama show qwen2.5:7b   # should include 'KV Cache' section
```

### Option B — build from source (maintainers / new architectures)

```bash
git clone https://github.com/pitcany/ollama-turboquant.git
cd ollama-turboquant
git checkout turboquant/runtime
git pull origin turboquant/runtime

GOCACHE=/tmp/ollama-build-gocache go build -o ~/.local/bin/ollama-tq .

# Avoid the double-load SIGSEGV: only cuda_v12/libggml-cuda.so should
# be discoverable at runtime (see TURBOQUANT-DEBUG-LOG.md 2026-04-30
# "Bug 3"). The flat .so is required for in-process eval (the
# calibration pipeline) but not for the serve path.
test -f build/lib/ollama/libggml-cuda.so && \
  mv build/lib/ollama/libggml-cuda.so /tmp/ollama-build-libggml-cuda-flat-$(date +%s).so
```

Layout the binary expects:

```text
~/.local/bin/ollama-tq           # the binary built from this fork
~/.local/bin/ollama-serve-tq     # wrapper script (sets env, runs serve)
~/.local/lib/ollama -> <build dir>/build/lib/ollama
```

The binary discovers the runtime backend via `../lib/ollama` relative
to itself. Make sure `~/.local/lib/ollama` symlinks to your fork's
`build/lib/ollama` directory.

### Uninstalling

To remove what `install-tq.sh` installed and switch back to upstream Ollama:

```bash
curl -fsSL https://raw.githubusercontent.com/pitcany/ollama-turboquant/turboquant/runtime/scripts/uninstall-tq.sh \
  | bash -s -- --systemd --enable-vanilla
```

What it does:

- Stops + disables the `ollama-tq.service` systemd unit (with `--systemd`)
- Removes `~/.local/bin/{ollama-tq,ollama-serve-tq}`, the `~/.local/bin/ollama` symlink (only if it points at our binary), `~/.local/lib/ollama/`, and `~/.local/share/ollama-tq/`
- Strips the `OLLAMA_HOST` export block from `~/.bashrc` (only the lines we added with our marker comment)
- With `--enable-vanilla`: enables + starts the upstream `ollama.service` if it's already installed at `/etc/systemd/system/ollama.service`

What it does **not** touch (intentionally):

- `~/.ollama/models/` — your downloaded model blobs and metadata stay put
- Any source-tree clones of this fork (`~/Work/ollama-build` or wherever)
- Upstream Ollama at `/usr/local/bin/ollama` if you installed it separately

Useful flags:

- `--dry-run` — print what would be removed, change nothing
- `--restore-backups` — after each removal, restore the most recent
  `<file>.bak.<timestamp>` (created by `install-tq.sh` if it found an
  existing file at that path during install)
- `--purge` — also delete those backup files
- `--keep-bashrc` — leave `~/.bashrc` untouched
- `--prefix DIR` — uninstall from a non-default prefix
- `-y` / `--yes` — skip the interactive confirmation prompt

After the uninstall, open a new shell so `OLLAMA_HOST` is unset, and `ollama show` falls back to the upstream client + upstream server on the default port `11434`.

## 2. Start the server

The wrapper sets the canonical env:

```bash
~/.local/bin/ollama-serve-tq    # binds to 0.0.0.0:8001 by default
```

Or, equivalently, raw:

```bash
OLLAMA_HOST=0.0.0.0:8001 \
OLLAMA_MODELS=$HOME/.ollama/models \
OLLAMA_KV_CACHE_TYPE=turboquant-adaptive \
OLLAMA_FLASH_ATTENTION=1 \
OLLAMA_NEW_ENGINE=1 \
OLLAMA_KEEP_ALIVE=-1 \
OLLAMA_CONTEXT_LENGTH=65536 \
OLLAMA_MAX_LOADED_MODELS=1 \
OLLAMA_NUM_GPU=999 \
  ~/.local/bin/ollama-tq serve
```

Required env vars and what they mean:

| Variable | Why |
|---|---|
| `OLLAMA_NEW_ENGINE=1` | TurboQuant only runs on the Go engine; the legacy llama.cpp runner ignores both modes and warns at load. |
| `OLLAMA_FLASH_ATTENTION=1` | Required for the Turbo* kernels. Adaptive calibration is rejected at load time if FA is off. |
| `OLLAMA_KV_CACHE_TYPE=turboquant-adaptive` | The recommended default. Per-model auto-resolution; falls back to `kq8-vturbo4` for any model not in the manifest. |
| `OLLAMA_KEEP_ALIVE=-1` | Optional. Keeps the model resident; otherwise the runtime unloads it after idle timeout. |

## 3. Run inference

The standard `ollama` CLI works against your TurboQuant server with
`OLLAMA_HOST=http://localhost:8001`:

```bash
export OLLAMA_HOST=http://localhost:8001

# Models with a bundled adaptive calibration get the optimal preset:
ollama run qwen2.5:7b "..."           # 4 of 28 layers protected
ollama run qwen3-coder:30b "def f("   # layers 3,8,12,24 protected
ollama run qwen3.6:27b-q8_0 "..."     # layers 0,1,2,4,7,27 protected

# Models without a manifest entry (everything else) automatically
# fall back to kq8-vturbo4 — still ~62% savings on Causal models,
# ~57% on hybrids:
ollama run llama3.3 "..."
ollama run gemma3:1b "..."

# curl works too:
curl -sS http://localhost:8001/api/generate -d '{
  "model":"qwen3.6:27b-q8_0",
  "prompt":"Capital of France?",
  "options":{"num_ctx":98304,"num_predict":12}
}'
```

## 4. Inspect what'll happen *before* loading the model

`ollama show` includes a `KV Cache` section computed from the static
manifest — no model load required:

```bash
ollama show qwen3.6:27b-q8_0
# ...
# KV Cache:
#   manifest_source     manifest:qwen35-q8_0-adaptive.json
#   base_kv_cache_type  turbo4
#   per_layer_overrides 0:q8_0,1:q8_0,2:q8_0,4:q8_0,7:q8_0,27:q8_0
#   projected_bpe       0.62  (~61% vs f16)
```

Same data via the API:

```bash
curl -sS http://localhost:8001/api/show -d '{"model":"qwen3.6:27b-q8_0"}' | jq .kv_cache
```

This is the cheapest way to verify a calibration is bundled and will
apply, without paying for a model load.

## 5. Verify it's actually using TurboQuant (in production logs)

Tail the server's log; on every model load you should see:

```text
INFO  turboquant-adaptive resolved cache_type=<turbo4|kq8-vturbo4>
        manifest_source=manifest:<entry>.json   ← or manifest_source=fallback
INFO  applied turboquant-adaptive calibration
        base_kv_cache_type=... key_cache_layer_types=...
INFO  applied per-layer key cache dtype overrides
        spec=...
INFO  kv cache device=CUDA0 size="<X> GiB"
INFO  kv cache device=CUDA1 size="<Y> GiB"
[GIN]  POST /api/generate -> 200 OK in <wall-time>
```

You should **not** see:

```text
WARN  cache does not support split key/value dtypes; using key dtype for both
WARN  cache does not support per-layer key dtype overrides; dropping overrides
```

Both warnings indicate a regression to the older fallback path. The
first fired for hybrid models on builds before
`turboquant/hybridcache-split-init`; the second fires when an adaptive
entry exists but the cache cannot accept per-layer overrides.

## 6. Choosing your tier explicitly

`turboquant-adaptive` (the recommended default) is strictly best where
a manifest entry exists, equivalent to `kq8-vturbo4` where one
doesn't, and gracefully falls back to `q8_0` on non-CUDA backends.
Override only when you have a reason:

```bash
# Strict tier-1 floor (mean_kl <= 0.05 across all Phase 0 models):
OLLAMA_KV_CACHE_TYPE=kq8-vturbo4 ollama-tq serve

# Vanilla q8_0 (~50% savings, no Turbo* needed; safe-anywhere fallback):
OLLAMA_KV_CACHE_TYPE=q8_0 ollama-tq serve

# Original f16 baseline (no savings; for A/B comparisons only):
OLLAMA_KV_CACHE_TYPE=f16 ollama-tq serve
```

Phase 0 quality budgets (pinned in `cmd/turboquant-ci-gate`):

| preset | budget (`mean_kl` vs f16) | typical KV savings vs f16 (Causal) |
|---|---:|---:|
| `kq8-vturbo4` | 0.05 | ~62% |
| `turboquant-adaptive` | 0.13 | ~72% (per-model dependent) |

## 7. Non-CUDA backends

If the server is running on Metal/ROCm/Vulkan/CPU, the Turbo* kernels
aren't available. The runner emits:

```text
WARN  Turbo* KV cache requires a CUDA backend; downgrading to q8_0
        requested_kv_cache_type=kq8-vturbo4 effective_kv_cache_type=q8_0
```

and you get the q8_0 tier (~50% savings). No silent garbage — the
warning is loud and the dtype substitution is conservative.

## 8. Quick gotcha checklist

- **Both `build/lib/ollama/libggml-cuda.so` AND `cuda_v12/libggml-cuda.so` present at runtime → SIGSEGV on model load.** Move the flat one aside (step 1).
- **`OLLAMA_FLASH_ATTENTION=0` or omitted → adaptive calibration is rejected at load time.** The kernels run on FA; required.
- **Embedding models (e.g. `qwen3-embedding-8b`) bypass FA entirely** and quietly use f16 KV. Expected. The runner logs `embedding model; falling back to f16 KV cache`.
- **Calibration is per (architecture, file_type, head_dim).** A `qwen2:Q4_K_M:128` calibration does *not* apply to `qwen2:Q8_0:128` of the same model — layer sensitivity shifts with weight quantization. Different quants of a covered model fall back to `kq8-vturbo4`.
- **Calibrations only apply on Ollama's Go runtime (`OLLAMA_NEW_ENGINE=1`).** The legacy llama.cpp runner ignores both modes.

## 9. Adding a calibration for a new model

`scripts/turboquant-calibrate.sh` is the single-command end-to-end
driver:

```bash
scripts/turboquant-calibrate.sh \
  -m <model:tag> \
  -c /tmp/long-context-corpus.txt \
  -a <arch> -f <ftype> -d <head_dim> \
  -t 4 \
  -G 40        # add this for big-model OOM avoidance (multi-GPU layer split)
```

Args:

| Flag | Meaning |
|---|---|
| `-m` | Ollama model tag (or GGUF path) |
| `-c` | UTF-8 corpus file used to tokenize the calibration snapshot |
| `-a` | architecture (matches GGUF's `general.architecture`) |
| `-f` | file_type (`Q4_K_M`, `Q8_0`, ...) |
| `-d` | per-head dimension (`128`, `256`, ...) |
| `-t` | number of lowest-KL layers to protect at q8_0 (default 4) |
| `-G` | number of layers to keep on GPU (default = all). Useful for big models that don't fit; the rest spill to CPU. |

A successful run produces a new
`tools/turboquant/calibration/manifest_data/<arch>-<ftype>-adaptive.json`
and updates `manifest.json` automatically. Restart the server to pick
it up. See [`README.md`](README.md) "Adding a calibration to the
bundled manifest" for the design rationale.

## 10. Bundled calibrations

| architecture | file_type | head_dim | Phase 0 `mean_kl` | example model |
|---|---|---:|---:|---|
| `qwen2`    | Q4_K_M | 128 | 0.0425 | `qwen2.5:7b` |
| `qwen3moe` | Q4_K_M | 128 | 0.0524 | `qwen3-coder:30b` |
| `qwen35`   | Q8_0   | 256 | 0.0155 | `qwen3.6:27b-q8_0` |

All three fall under the adaptive Phase 0 budget (0.13). qwen2.5 and
qwen3.6 fall under the strict `kq8-vturbo4` budget (0.05) too;
qwen3-coder is just over (operators wanting the strict floor on
qwen3-coder should override to `OLLAMA_KV_CACHE_TYPE=kq8-vturbo4`).
