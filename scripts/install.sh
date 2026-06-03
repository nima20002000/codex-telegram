#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_file="${CODEX_TELEGRAM_ENV_FILE:-"$repo_root/.env"}"
service_name="${CODEX_TELEGRAM_SERVICE_NAME:-codex-telegram.service}"
skip_systemd="${CODEX_TELEGRAM_SKIP_SYSTEMD:-0}"
service_python="/usr/bin/python3"
created_env="0"

log() {
  printf '%s\n' "$*"
}

env_value() {
  local key="$1"
  if [ ! -f "$env_file" ]; then
    return 0
  fi
  awk -F= -v key="$key" '
    $0 !~ /^[[:space:]]*#/ && index($0, "=") {
      raw_key = $1
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", raw_key)
      if (raw_key == key) {
        sub(/^[^=]*=/, "")
        print
        exit
      }
    }
  ' "$env_file"
}

set_env_value() {
  local key="$1"
  local value="$2"
  python3 - "$env_file" "$key" "$value" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

path = Path(sys.argv[1])
key = sys.argv[2]
value = sys.argv[3]
line = f"{key}={value}"

lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
out: list[str] = []
updated = False
for existing in lines:
    existing_key = existing.split("=", 1)[0].strip()
    if not updated and not existing.lstrip().startswith("#") and existing_key == key:
        out.append(line)
        updated = True
    else:
        out.append(existing)
if not updated:
    if out and out[-1] != "":
        out.append("")
    out.append(line)
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text("\n".join(out) + "\n", encoding="utf-8")
PY
}

ensure_env_value() {
  local key="$1"
  local default_value="$2"
  local current
  current="$(env_value "$key" || true)"
  if [ -n "$current" ]; then
    log "Keeping existing $key."
    return
  fi
  set_env_value "$key" "$default_value"
  log "Set default $key."
}

migrate_env_value() {
  local old_key="$1"
  local new_key="$2"
  local current_new
  local current_old
  current_new="$(env_value "$new_key" || true)"
  if [ -n "$current_new" ]; then
    return
  fi
  current_old="$(env_value "$old_key" || true)"
  if [ -z "$current_old" ]; then
    return
  fi
  set_env_value "$new_key" "$current_old"
  log "Migrated legacy $new_key."
}

resolve_codex_command() {
  local resolved
  resolved="$(command -v codex || true)"
  if [ -z "$resolved" ]; then
    log "Codex CLI was not found on PATH. Set CODEX_COMMAND in $env_file to an absolute Codex path, then rerun."
    exit 1
  fi
  printf '%s\n' "$resolved"
}

ensure_codex_command() {
  local current
  current="$(env_value "CODEX_COMMAND" || true)"
  if [ "$created_env" = "1" ] && { [ -z "$current" ] || [ "$current" = "codex" ]; }; then
    set_env_value "CODEX_COMMAND" "$(resolve_codex_command)"
    log "Set default CODEX_COMMAND."
    return
  fi
  if [ -n "$current" ]; then
    log "Keeping existing CODEX_COMMAND."
    return
  fi
  set_env_value "CODEX_COMMAND" "$(resolve_codex_command)"
  log "Set default CODEX_COMMAND."
}

install_with_venv() {
  log "Using ignored repo-local virtual environment."
  python3 -m venv "$repo_root/.venv"
  service_python="$repo_root/.venv/bin/python"
  "$service_python" -m pip install -e "$repo_root"
}

prompt_secret_if_missing() {
  local key="$1"
  local prompt="$2"
  local current
  current="$(env_value "$key" || true)"
  if [ -n "$current" ]; then
    log "Keeping existing $key."
    return
  fi

  local value=""
  while [ -z "$value" ]; do
    printf '%s' "$prompt"
    IFS= read -r -s value
    printf '\n'
    if [ -z "$value" ]; then
      log "$key cannot be empty."
    fi
  done
  set_env_value "$key" "$value"
  log "Saved $key."
}

prompt_user_id_if_missing() {
  local key="TELEGRAM_ALLOWED_USERS"
  local current
  current="$(env_value "$key" || true)"
  if [ -n "$current" ]; then
    log "Keeping existing $key."
    return
  fi

  local value=""
  while true; do
    printf 'Telegram numeric user ID from userinfobot (https://t.me/userinfobot): '
    IFS= read -r -s value
    printf '\n'
    if [[ "$value" =~ ^-?[0-9]+([[:space:]]*,[[:space:]]*-?[0-9]+)*$ ]]; then
      break
    fi
    log "Enter one numeric Telegram user ID, or a comma-separated list."
  done
  set_env_value "$key" "$value"
  log "Saved $key."
}

if [ ! -f "$env_file" ]; then
  if [ ! -f "$repo_root/.env.example" ]; then
    log "Missing .env.example in $repo_root"
    exit 1
  fi
  mkdir -p "$(dirname "$env_file")"
  cp "$repo_root/.env.example" "$env_file"
  chmod 600 "$env_file"
  created_env="1"
  log "Created local env file."
else
  chmod 600 "$env_file"
  log "Using existing local env file."
fi

prompt_secret_if_missing "TELEGRAM_BOT_TOKEN" "Telegram bot API token from BotFather (https://t.me/BotFather): "
prompt_user_id_if_missing
migrate_env_value "HERMES_TELEGRAM_STATE_DIR" "CODEX_TELEGRAM_STATE_DIR"

if [ "$created_env" = "1" ]; then
  set_env_value "CODEX_WORKDIR" "$HOME/Desktop"
  log "Set default CODEX_WORKDIR."
  set_env_value "CODEX_TELEGRAM_STATE_DIR" "$repo_root/.codex-telegram"
  log "Set default CODEX_TELEGRAM_STATE_DIR."
else
  ensure_env_value "CODEX_WORKDIR" "$HOME/Desktop"
  ensure_env_value "CODEX_TELEGRAM_STATE_DIR" "$repo_root/.codex-telegram"
fi
ensure_codex_command
ensure_env_value "CODEX_SANDBOX" "workspace-write"
ensure_env_value "CODEX_TIMEOUT_SECONDS" "1800"
ensure_env_value "TELEGRAM_POLL_TIMEOUT_SECONDS" "30"
ensure_env_value "TELEGRAM_REQUEST_TIMEOUT_SECONDS" "45"
ensure_env_value "MAX_TELEGRAM_RESPONSE_CHARS" "12000"
ensure_env_value "SESSION_HISTORY_TURNS" "8"

log "Installing package in editable mode."
if python3 -m pip --version >/dev/null 2>&1; then
  if ! python3 -m pip install -e "$repo_root"; then
    log "System pip install failed; falling back to local virtual environment."
    install_with_venv
  fi
else
  log "python3 pip is unavailable; falling back to local virtual environment."
  install_with_venv
fi

if [ "$skip_systemd" = "1" ]; then
  log "Skipping systemd setup because CODEX_TELEGRAM_SKIP_SYSTEMD=1."
  exit 0
fi

service_dir="$HOME/.config/systemd/user"
service_file="$service_dir/$service_name"
mkdir -p "$service_dir"
for existing_service_file in "$service_dir"/*-telegram.service; do
  [ -e "$existing_service_file" ] || continue
  existing_service="$(basename "$existing_service_file")"
  if [ "$existing_service" = "$service_name" ]; then
    continue
  fi
  if grep -Eq 'Telegram Codex bridge|_telegram\.cli' "$existing_service_file"; then
    systemctl --user disable --now "$existing_service" >/dev/null 2>&1 || true
    rm -f "$existing_service_file"
    log "Removed previous Telegram Codex bridge service: $existing_service."
  fi
done
cat >"$service_file" <<EOF
[Unit]
Description=Codex Telegram bridge
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$repo_root
Environment=PYTHONPATH=$repo_root/src
ExecStart=$service_python -m codex_telegram.cli --env-file $env_file
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF

log "Installed user service."
systemctl --user daemon-reload
systemctl --user enable --now "$service_name"
loginctl enable-linger "$USER"
systemctl --user restart "$service_name"
systemctl --user status "$service_name" --no-pager
