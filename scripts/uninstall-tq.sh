#!/usr/bin/env bash
# uninstall-tq.sh — undo what install-tq.sh installed.
#
# Removes (when present):
#   $PREFIX/bin/ollama-tq
#   $PREFIX/bin/ollama-serve-tq
#   $PREFIX/bin/ollama                       (only if it's a symlink we created)
#   $PREFIX/lib/ollama/                       (the directory we copied in)
#   $PREFIX/share/ollama-tq/                  (wrapper/template/calibrations)
#   /etc/systemd/system/ollama-tq.service     (only with --systemd)
#   the OLLAMA_HOST export block in ~/.bashrc (only if it has our marker)
#
# Does NOT touch (intentionally):
#   ~/.ollama/                                (your model blobs and metadata)
#   any source-tree clones                    (e.g. ~/Work/ollama-build)
#   any *.bak.<timestamp> backup files we created during install
#     (use --restore-backups to restore the most recent backup of each file
#      after uninstall, useful when you had an upstream binary at the same
#      path before installing ollama-tq)
#   the upstream ollama service /etc/systemd/system/ollama.service
#     (use --enable-vanilla to re-enable it after uninstall)
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/pitcany/ollama-turboquant/turboquant/runtime/scripts/uninstall-tq.sh | bash
#   curl -fsSL .../uninstall-tq.sh | bash -s -- --systemd --enable-vanilla
#   ./scripts/uninstall-tq.sh --systemd --restore-backups
#
# Flags:
#   --systemd            stop + disable + remove the systemd unit (sudo prompt)
#   --enable-vanilla     after uninstall, enable + start upstream ollama.service
#                        if /etc/systemd/system/ollama.service exists (sudo)
#   --restore-backups    after removing each file, mv its most recent
#                        <file>.bak.<timestamp> back into place
#   --purge              also remove ~/.local/share/ollama-tq backups and any
#                        ~/.local/{bin,lib}/*.bak.* files install-tq.sh created
#   --prefix <dir>       install prefix to undo (default: $HOME/.local)
#   --keep-bashrc        don't touch ~/.bashrc
#   --dry-run            print what would be removed without removing anything
#   --yes / -y           don't ask for interactive confirmation

set -euo pipefail

PREFIX="$HOME/.local"
DO_SYSTEMD=0
ENABLE_VANILLA=0
RESTORE_BACKUPS=0
PURGE=0
KEEP_BASHRC=0
DRY_RUN=0
ASSUME_YES=0

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarn:\033[0m %s\n' "$*" >&2; }
err()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --systemd)         DO_SYSTEMD=1; shift ;;
    --enable-vanilla)  ENABLE_VANILLA=1; shift ;;
    --restore-backups) RESTORE_BACKUPS=1; shift ;;
    --purge)           PURGE=1; shift ;;
    --prefix)          PREFIX="$2"; shift 2 ;;
    --keep-bashrc)     KEEP_BASHRC=1; shift ;;
    --dry-run)         DRY_RUN=1; shift ;;
    -y|--yes)          ASSUME_YES=1; shift ;;
    -h|--help)         sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d' >&2; exit 0 ;;
    *) err "unknown flag: $1" ;;
  esac
done

# do_action runs the command unless --dry-run is in effect; either way it
# echoes what's happening so the operator sees a removal log.
do_action() {
  printf '  rm: %s\n' "$*"
  [[ "$DRY_RUN" == "1" ]] || eval "$@"
}

# do_sudo wraps a sudo command the same way.
do_sudo() {
  printf '  sudo: %s\n' "$*"
  [[ "$DRY_RUN" == "1" ]] || sudo "$@"
}

# restore_latest_backup looks for <path>.bak.<timestamp> files, picks the
# newest by timestamp suffix, and restores it to <path>.
restore_latest_backup() {
  local target="$1"
  local backup
  backup="$(ls -1d "${target}.bak."* 2>/dev/null | sort -r | head -1 || true)"
  if [[ -n "$backup" && -e "$backup" ]]; then
    log "Restoring backup: $backup -> $target"
    [[ "$DRY_RUN" == "1" ]] || mv "$backup" "$target"
  fi
}

cat <<EOF
=== uninstall-tq.sh ===
  prefix:           $PREFIX
  systemd:          $DO_SYSTEMD
  enable-vanilla:   $ENABLE_VANILLA
  restore-backups:  $RESTORE_BACKUPS
  purge:            $PURGE
  keep-bashrc:      $KEEP_BASHRC
  dry-run:          $DRY_RUN

This will remove ollama-tq files placed by install-tq.sh.
Your ~/.ollama/models/ blobs and metadata are NOT touched.
EOF

if [[ "$ASSUME_YES" != "1" && "$DRY_RUN" != "1" ]]; then
  read -p "Proceed? [y/N] " ans
  [[ "$ans" =~ ^[Yy]$ ]] || { echo "aborted"; exit 1; }
fi

# 1. Stop + remove the systemd unit first (so the binary isn't held).
if [[ "$DO_SYSTEMD" == "1" ]]; then
  if ! command -v systemctl >/dev/null 2>&1; then
    warn "systemctl not found; skipping --systemd"
  elif systemctl list-unit-files 2>/dev/null | grep -q '^ollama-tq.service'; then
    log "Stopping + disabling ollama-tq.service"
    do_sudo systemctl disable --now ollama-tq.service || true
    if [[ -f /etc/systemd/system/ollama-tq.service ]]; then
      do_sudo rm -f /etc/systemd/system/ollama-tq.service
      do_sudo systemctl daemon-reload
    fi
  else
    log "ollama-tq.service is not registered; nothing to disable"
  fi
fi

# 2. Remove binaries.
log "Removing binaries from $PREFIX/bin"
for f in "$PREFIX/bin/ollama-tq" "$PREFIX/bin/ollama-serve-tq"; do
  if [[ -e "$f" ]]; then
    do_action rm -f "$f"
    [[ "$RESTORE_BACKUPS" == "1" ]] && restore_latest_backup "$f"
  fi
done

# 3. Remove the ollama symlink only if it points at our binary.
ollama_link="$PREFIX/bin/ollama"
if [[ -L "$ollama_link" ]]; then
  target="$(readlink -f "$ollama_link" || readlink "$ollama_link")"
  case "$target" in
    *ollama-tq|*ollama-tq.bak.*)
      log "Removing $ollama_link symlink (was pointing at $target)"
      do_action rm -f "$ollama_link"
      [[ "$RESTORE_BACKUPS" == "1" ]] && restore_latest_backup "$ollama_link"
      ;;
    *)
      warn "$ollama_link is a symlink to $target — not ours, leaving it alone"
      ;;
  esac
elif [[ -e "$ollama_link" ]]; then
  warn "$ollama_link exists and is not a symlink — leaving it alone"
fi

# 4. Remove the lib dir we installed.
if [[ -L "$PREFIX/lib/ollama" ]]; then
  log "Removing $PREFIX/lib/ollama (symlink)"
  do_action rm -f "$PREFIX/lib/ollama"
elif [[ -d "$PREFIX/lib/ollama" ]]; then
  log "Removing $PREFIX/lib/ollama (directory)"
  do_action rm -rf "$PREFIX/lib/ollama"
fi
[[ "$RESTORE_BACKUPS" == "1" ]] && restore_latest_backup "$PREFIX/lib/ollama"

# 5. Remove share/ollama-tq.
if [[ -d "$PREFIX/share/ollama-tq" ]]; then
  log "Removing $PREFIX/share/ollama-tq"
  do_action rm -rf "$PREFIX/share/ollama-tq"
fi

# 6. Strip the OLLAMA_HOST export block from ~/.bashrc if our marker is there.
if [[ "$KEEP_BASHRC" != "1" ]]; then
  bashrc="$HOME/.bashrc"
  marker="# ollama-tq install: OLLAMA_HOST default"
  if [[ -f "$bashrc" ]] && grep -qF "$marker" "$bashrc"; then
    log "Removing OLLAMA_HOST export block from $bashrc"
    if [[ "$DRY_RUN" == "1" ]]; then
      grep -n -A1 -F "$marker" "$bashrc" || true
    else
      # Remove the marker line and the very next line (the export). This
      # matches what install-tq.sh writes; if you've edited the block by
      # hand, --keep-bashrc is the safer flag.
      tmp="$(mktemp)"
      awk -v m="$marker" '
        BEGIN { skip=0 }
        skip { skip=0; next }
        $0==m { skip=1; next }
        { print }
      ' "$bashrc" > "$tmp"
      mv "$tmp" "$bashrc"
    fi
  fi
fi

# 7. --purge: clean up backup files install-tq.sh dropped during prior runs.
if [[ "$PURGE" == "1" ]]; then
  log "Purging install-tq.sh backups under $PREFIX"
  for d in "$PREFIX/bin" "$PREFIX/lib" "$PREFIX/share/ollama-tq"; do
    [[ -d "$d" ]] || continue
    while IFS= read -r -d '' f; do
      do_action rm -rf "$f"
    done < <(find "$d" -maxdepth 2 -name '*.bak.*' -print0 2>/dev/null)
  done
fi

# 8. Optional: re-enable upstream ollama.service if it exists.
if [[ "$ENABLE_VANILLA" == "1" ]]; then
  if ! command -v systemctl >/dev/null 2>&1; then
    warn "systemctl not found; skipping --enable-vanilla"
  elif [[ ! -f /etc/systemd/system/ollama.service ]] \
    && [[ ! -f /usr/lib/systemd/system/ollama.service ]] \
    && [[ ! -f /lib/systemd/system/ollama.service ]]; then
    warn "ollama.service unit not found anywhere on disk; cannot --enable-vanilla"
    warn "(install upstream Ollama first: curl -fsSL https://ollama.com/install.sh | sh)"
  else
    log "Enabling + starting upstream ollama.service"
    do_sudo systemctl enable --now ollama.service || true
    systemctl status ollama.service --no-pager 2>&1 | head -5 || true
  fi
fi

cat <<EOF

=== done ===
What's still on disk (intentionally):
  - ~/.ollama/models/                  (your downloaded model blobs)
  - any source clones (e.g. ~/Work/ollama-build)
  - upstream ollama at /usr/local/bin/ollama (if you installed it)

To switch back to vanilla Ollama as your default CLI:
  - If upstream ollama is installed at /usr/local/bin/ollama, plain
    \`ollama show ...\` already works once \$OLLAMA_HOST points at its server.
  - Run \`unset OLLAMA_HOST\` (or open a new shell) so the CLI defaults to
    127.0.0.1:11434 again, where upstream ollama listens by default.
  - If you used --enable-vanilla above, the upstream service is now running.

To reinstall ollama-tq later:
  curl -fsSL https://raw.githubusercontent.com/pitcany/ollama-turboquant/turboquant/runtime/scripts/install-tq.sh \\
    | bash -s -- --systemd --symlink-ollama
EOF
