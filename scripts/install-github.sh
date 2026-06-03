#!/usr/bin/env bash
set -euo pipefail

repo_url="${CODEX_TELEGRAM_REPO_URL:-https://github.com/nima20002000/codex-telegram.git}"
install_dir="${CODEX_TELEGRAM_INSTALL_DIR:-"$HOME/.local/share/codex-telegram"}"

log() {
  printf '%s\n' "$*"
}

if ! command -v git >/dev/null 2>&1; then
  log "git is required to install Codex Telegram from GitHub."
  exit 1
fi

if [ -d "$install_dir/.git" ]; then
  log "Updating existing Codex Telegram checkout at $install_dir."
  git -C "$install_dir" pull --ff-only
elif [ -e "$install_dir" ]; then
  log "Install directory exists but is not a git checkout: $install_dir"
  log "Set CODEX_TELEGRAM_INSTALL_DIR to another path, or move that directory first."
  exit 1
else
  log "Cloning Codex Telegram into $install_dir."
  mkdir -p "$(dirname "$install_dir")"
  git clone "$repo_url" "$install_dir"
fi

exec bash "$install_dir/scripts/install.sh"
