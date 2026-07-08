#!/usr/bin/env bash
#
# Pull a PostgreSQL backup from the VPS down to this laptop.
#
# Runs ON THE LAPTOP. It SSHes into the VPS, runs pg_dump inside the postgres
# container (reading the container's own credentials), streams the gzipped dump
# straight to local disk, verifies it, and prunes old copies. The database
# password never touches this script or the laptop's command history.
#
# Usage:
#     ./scripts/backup_to_laptop.sh
#
# Configure via the block below or environment variables, e.g.:
#     SSH_TARGET=chatbot-vps ./scripts/backup_to_laptop.sh
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Config — edit these two, or pass them as environment variables.
# ---------------------------------------------------------------------------
SSH_TARGET="${SSH_TARGET:-chatbot-vps}"                       # ssh host/alias of the VPS
REMOTE_PROJECT_DIR="${REMOTE_PROJECT_DIR:-/opt/games-chatbot}" # where docker-compose.yml lives on the VPS
BACKUP_DIR="${BACKUP_DIR:-$HOME/backups/games-chatbot}"        # where to store dumps on this laptop
KEEP_WEEKS="${KEEP_WEEKS:-12}"                                 # how many recent dumps to keep
COMPOSE_SERVICE="${COMPOSE_SERVICE:-postgres}"                 # service name in docker-compose.yml
# ---------------------------------------------------------------------------

log() { printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1"; }
die() { log "ERROR: $1" >&2; exit 1; }

# Build the command that runs on the VPS. pg_dump reads POSTGRES_* from the
# container's own environment, so no secret is ever interpolated here.
remote_dump_command() {
  printf 'cd %q && docker compose exec -T %q sh -c %q | gzip -9' \
    "$REMOTE_PROJECT_DIR" \
    "$COMPOSE_SERVICE" \
    'PGPASSWORD="$POSTGRES_PASSWORD" pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB"'
}

# Fail loudly if the dump is missing, corrupt, or clearly not a pg_dump.
verify_dump() {
  local file="$1"
  [ -s "$file" ] || die "dump is empty"
  gzip -t "$file" 2>/dev/null || die "dump failed gzip integrity check"
  gzip -dc "$file" | head -c 4096 | grep -q "PostgreSQL database dump" \
    || die "dump does not look like pg_dump output (remote command likely failed)"
}

prune_old() {
  local kept
  kept=$(ls -1t "$BACKUP_DIR"/chatbot-*.sql.gz 2>/dev/null | tail -n +"$((KEEP_WEEKS + 1))" || true)
  [ -n "$kept" ] || return 0
  printf '%s\n' "$kept" | while IFS= read -r old; do
    rm -- "$old"
    log "pruned old backup: $(basename "$old")"
  done
}

main() {
  mkdir -p "$BACKUP_DIR"
  local stamp final tmp
  stamp="$(date '+%Y-%m-%d')"
  final="$BACKUP_DIR/chatbot-$stamp.sql.gz"
  tmp="$final.partial"

  log "pulling backup from $SSH_TARGET -> $final"
  if ! ssh "$SSH_TARGET" "$(remote_dump_command)" > "$tmp"; then
    rm -f "$tmp"
    die "ssh/pg_dump failed"
  fi

  verify_dump "$tmp"
  mv -f "$tmp" "$final"
  log "backup OK: $final ($(du -h "$final" | cut -f1))"

  prune_old
  log "done — keeping up to $KEEP_WEEKS backups in $BACKUP_DIR"
}

main "$@"
