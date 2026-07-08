#!/usr/bin/env bash
#
# Restore a laptop-held backup back into the VPS PostgreSQL database.
#
# Runs ON THE LAPTOP. It decompresses the chosen dump and streams it into psql
# inside the postgres container on the VPS. Intended for disaster recovery:
# after rebuilding the VPS, bring the database back up first (so the empty
# `chatbot` database exists), then run this.
#
# Usage:
#     ./scripts/restore_to_vps.sh ~/backups/games-chatbot/chatbot-2026-06-29.sql.gz
#
set -euo pipefail

SSH_TARGET="${SSH_TARGET:-chatbot-vps}"
REMOTE_PROJECT_DIR="${REMOTE_PROJECT_DIR:-/opt/games-chatbot}"
COMPOSE_SERVICE="${COMPOSE_SERVICE:-postgres}"

log() { printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1"; }
die() { log "ERROR: $1" >&2; exit 1; }

main() {
  local backup_file="${1:-}"
  [ -n "$backup_file" ] || die "usage: $0 <backup-file.sql.gz>"
  [ -f "$backup_file" ] || die "no such file: $backup_file"
  gzip -t "$backup_file" 2>/dev/null || die "file failed gzip integrity check"

  log "About to restore '$backup_file' into $SSH_TARGET ($COMPOSE_SERVICE)."
  log "This writes into the live database and may overwrite existing data."
  printf 'Type "restore" to continue: '
  read -r answer
  [ "$answer" = "restore" ] || die "aborted"

  local remote_cmd
  remote_cmd="$(printf 'cd %q && docker compose exec -T %q sh -c %q' \
    "$REMOTE_PROJECT_DIR" \
    "$COMPOSE_SERVICE" \
    'PGPASSWORD="$POSTGRES_PASSWORD" psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB"')"

  log "restoring..."
  gzip -dc "$backup_file" | ssh "$SSH_TARGET" "$remote_cmd"
  log "restore complete"
}

main "$@"
