#!/usr/bin/env bash
# install-tq.sh — one-shot installer for ollama-tq (TurboQuant Ollama fork).
#
# Downloads the prebuilt tarball from a GitHub release, drops files under
# ~/.local/bin, ~/.local/lib, and ~/.local/share, and optionally installs the
# systemd unit and symlinks `ollama` -> `ollama-tq`.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/pitcany/ollama-turboquant/turboquant/runtime/scripts/install-tq.sh | bash
#   curl -fsSL .../install-tq.sh | bash -s -- --version v0.1.0 --systemd --symlink-ollama
#   ./scripts/install-tq.sh --local /path/to/ollama-tq-XXX.tar.gz   (offline)
#
# Flags:
#   --version <tag>       release tag to install (default: latest)
#   --local <tarball>     install from a local tarball instead of fetching
#   --systemd             install + enable the systemd unit
#   --symlink-ollama      symlink ~/.local/bin/ollama -> ollama-tq (so that
#                         `ollama show` renders the KV Cache section)
#   --no-bashrc           don't append OLLAMA_HOST export to ~/.bashrc
#   --prefix <dir>        install prefix (default: $HOME/.local)
#   --port <n>            server port (default: 8001)
#   --force               overwrite existing files

set -euo pipefail

REPO="pitcany/ollama-turboquant"
PLATFORM="linux-x86_64-cuda12"
VERSION="latest"
LOCAL_TARBALL=""
DO_SYSTEMD=0
DO_SYMLINK=0
DO_BASHRC=1
PREFIX="$HOME/.local"
PORT=8001
FORCE=0

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarn:\033[0m %s\n' "$*" >&2; }
err()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --version)         VERSION="$2"; shift 2 ;;
    --local)           LOCAL_TARBALL="$2"; shift 2 ;;
    --systemd)         DO_SYSTEMD=1; shift ;;
    --symlink-ollama)  DO_SYMLINK=1; shift ;;
    --no-bashrc)       DO_BASHRC=0; shift ;;
    --prefix)          PREFIX="$2"; shift 2 ;;
    --port)            PORT="$2"; shift 2 ;;
    --force)           FORCE=1; shift ;;
    -h|--help)         sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d' >&2; exit 0 ;;
    *) err "unknown flag: $1" ;;
  esac
done

# Sanity checks.
[[ "$(uname -s)" == "Linux" ]] || err "this installer supports Linux only (you're on $(uname -s))"
[[ "$(uname -m)" == "x86_64" ]] || err "this installer supports x86_64 only (you're on $(uname -m))"
command -v curl >/dev/null 2>&1 || err "curl is required"
command -v tar  >/dev/null 2>&1 || err "tar is required"
command -v sha256sum >/dev/null 2>&1 || err "sha256sum is required"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  warn "nvidia-smi not found — TurboQuant kernels are CUDA-only. Server will downgrade to q8_0 with a warning at runtime."
fi

mkdir -p "$PREFIX/bin" "$PREFIX/lib" "$PREFIX/share"

# 1. Acquire tarball.
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

if [[ -n "$LOCAL_TARBALL" ]]; then
  log "Using local tarball: $LOCAL_TARBALL"
  [[ -f "$LOCAL_TARBALL" ]] || err "local tarball not found: $LOCAL_TARBALL"
  cp "$LOCAL_TARBALL" "$TMP/release.tar.gz"
  if [[ -f "${LOCAL_TARBALL}.sha256" ]]; then
    cp "${LOCAL_TARBALL}.sha256" "$TMP/release.tar.gz.sha256"
  fi
else
  if [[ "$VERSION" == "latest" ]]; then
    log "Resolving latest release of $REPO"
    VERSION="$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" \
      | grep -E '"tag_name"\s*:' | head -1 | sed -E 's/.*"tag_name"\s*:\s*"([^"]+)".*/\1/')"
    [[ -n "$VERSION" ]] || err "could not resolve latest release tag (do any releases exist on $REPO?)"
    log "Latest is $VERSION"
  fi
  TARBALL_NAME="ollama-tq-${VERSION}-${PLATFORM}.tar.gz"
  TARBALL_URL="https://github.com/$REPO/releases/download/${VERSION}/${TARBALL_NAME}"
  SHA_URL="${TARBALL_URL}.sha256"

  log "Downloading $TARBALL_URL"
  curl -fLo "$TMP/release.tar.gz" "$TARBALL_URL" || err "download failed"
  curl -fLo "$TMP/release.tar.gz.sha256" "$SHA_URL" 2>/dev/null \
    || warn "no .sha256 alongside the tarball; skipping integrity check"
fi

# 2. Verify integrity if we have a checksum.
if [[ -f "$TMP/release.tar.gz.sha256" ]]; then
  log "Verifying sha256"
  ( cd "$TMP" && \
    expected="$(awk '{print $1}' release.tar.gz.sha256)" && \
    actual="$(sha256sum release.tar.gz | awk '{print $1}')" && \
    [[ "$expected" == "$actual" ]] || { echo "sha256 mismatch: $actual != $expected" >&2; exit 1; }
  )
fi

# 3. Extract.
log "Extracting tarball"
tar -C "$TMP" -xzf "$TMP/release.tar.gz"
SRC="$(find "$TMP" -maxdepth 1 -type d -name 'ollama-tq-*' | head -1)"
[[ -d "$SRC" ]] || err "extracted tarball does not contain a ollama-tq-* directory"

# 4. Install files.
log "Installing into $PREFIX"

install_file() {
  local src="$1" dst="$2"
  if [[ -e "$dst" && "$FORCE" != "1" ]]; then
    if cmp -s "$src" "$dst" 2>/dev/null; then
      return 0
    fi
    warn "$dst exists; backing up to ${dst}.bak.$(date +%s) (use --force to overwrite without backup)"
    mv "$dst" "${dst}.bak.$(date +%s)"
  fi
  install -D -m "$(stat -c '%a' "$src")" "$src" "$dst"
}

install_file "$SRC/bin/ollama-tq" "$PREFIX/bin/ollama-tq"

# Lib dir: clear out any prior ~/.local/lib/ollama install (or symlink),
# replace with a real directory copy.
if [[ -L "$PREFIX/lib/ollama" ]]; then
  log "Replacing existing $PREFIX/lib/ollama symlink"
  rm "$PREFIX/lib/ollama"
elif [[ -d "$PREFIX/lib/ollama" && "$FORCE" != "1" ]]; then
  warn "$PREFIX/lib/ollama exists; moving to ${PREFIX}/lib/ollama.bak.$(date +%s)"
  mv "$PREFIX/lib/ollama" "${PREFIX}/lib/ollama.bak.$(date +%s)"
fi
mkdir -p "$PREFIX/lib/ollama"
cp -a "$SRC/lib/ollama/." "$PREFIX/lib/ollama/"

# Wrapper + bundled artifacts under share/.
mkdir -p "$PREFIX/share/ollama-tq/calibration"
install_file "$SRC/share/ollama-tq/ollama-serve-tq" "$PREFIX/bin/ollama-serve-tq"
chmod +x "$PREFIX/bin/ollama-serve-tq"
install_file "$SRC/share/ollama-tq/ollama-tq.service" "$PREFIX/share/ollama-tq/ollama-tq.service.template"
cp -a "$SRC/share/ollama-tq/calibration/." "$PREFIX/share/ollama-tq/calibration/"

# 5. Env-var convenience: tell the local shell where the server lives.
if [[ "$DO_BASHRC" == "1" ]]; then
  bashrc="$HOME/.bashrc"
  marker="# ollama-tq install: OLLAMA_HOST default"
  if [[ -f "$bashrc" ]] && grep -q "$marker" "$bashrc"; then
    log "OLLAMA_HOST line already in $bashrc; skipping"
  else
    log "Appending OLLAMA_HOST=http://localhost:$PORT to $bashrc"
    {
      printf '\n%s\n' "$marker"
      printf 'export OLLAMA_HOST="http://localhost:%s"\n' "$PORT"
    } >> "$bashrc"
  fi
fi

# 6. Optional: symlink `ollama` -> `ollama-tq` so the standard CLI command
#    invokes the fork (which renders the KV Cache section in `ollama show`).
if [[ "$DO_SYMLINK" == "1" ]]; then
  if [[ -e "$PREFIX/bin/ollama" && ! -L "$PREFIX/bin/ollama" && "$FORCE" != "1" ]]; then
    warn "$PREFIX/bin/ollama exists and isn't a symlink; not overwriting (use --force)"
  else
    ln -sf "$PREFIX/bin/ollama-tq" "$PREFIX/bin/ollama"
    log "Symlinked $PREFIX/bin/ollama -> ollama-tq"
    if command -v ollama >/dev/null 2>&1 && [[ "$(command -v ollama)" != "$PREFIX/bin/ollama" ]]; then
      warn "PATH precedence: '$(command -v ollama)' is found before '$PREFIX/bin/ollama'."
      warn "Either remove the upstream binary or add $PREFIX/bin to the front of PATH."
    fi
  fi
fi

# 7. Optional: install + enable the systemd unit.
if [[ "$DO_SYSTEMD" == "1" ]]; then
  if ! command -v systemctl >/dev/null 2>&1; then
    warn "systemctl not found; skipping --systemd"
  else
    UNIT_DST="/etc/systemd/system/ollama-tq.service"
    UNIT_SRC="$TMP/ollama-tq.service.rendered"
    sed "s/__USER__/$USER/g" "$PREFIX/share/ollama-tq/ollama-tq.service.template" > "$UNIT_SRC"

    if [[ -f "$UNIT_DST" && "$FORCE" != "1" ]] && ! cmp -s "$UNIT_SRC" "$UNIT_DST"; then
      warn "$UNIT_DST exists and differs from the template; not overwriting (use --force)"
    else
      log "Installing $UNIT_DST (will prompt for sudo)"
      sudo cp "$UNIT_SRC" "$UNIT_DST"
      sudo systemctl daemon-reload
      sudo systemctl enable --now ollama-tq.service
      log "Service status:"
      systemctl status ollama-tq.service --no-pager | head -5 || true
    fi
  fi
fi

# 8. Wrap up.
echo
log "Done. Quick verification:"
echo "  $PREFIX/bin/ollama-tq --version"
"$PREFIX/bin/ollama-tq" --version 2>&1 | head -2 || true
echo
echo "Next steps:"
echo "  1) Start the server (skip if --systemd installed it):"
echo "       $PREFIX/bin/ollama-serve-tq &"
echo "  2) Open a new shell so OLLAMA_HOST takes effect (or 'source ~/.bashrc')."
echo "  3) Confirm a known-bundled model resolves to a calibration:"
if [[ "$DO_SYMLINK" == "1" ]]; then
  echo "       ollama show qwen2.5:7b"
else
  echo "       $PREFIX/bin/ollama-tq show qwen2.5:7b"
fi
echo "     The output should include a 'KV Cache' section pointing at"
echo "     manifest:qwen2.5-7b-q4_k_m-adaptive.json with ~72% saved vs f16."
echo
echo "Full operator guide: tools/turboquant/USAGE.md in the source tree."
