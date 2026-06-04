#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ ! -f ".env" ]; then
  echo ".env was not found. Run make db-up first." >&2
  exit 1
fi

app_env_override="${APP_ENV-}"
backup_root_file_override="${BACKUP_ROOT_FILE-}"

set -a
# shellcheck disable=SC1091
source ".env"
set +a

POSTGRES_CONTAINER_NAME="${POSTGRES_CONTAINER_NAME:-topicos-av2-postgres}"
POSTGRES_USER="${POSTGRES_USER:-postgres}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-postgres}"
POSTGRES_DB="${POSTGRES_DB:-app_dev}"
APP_ENV="${app_env_override:-${APP_ENV:-dev}}"
timestamp="$(date +%Y%m%d_%H%M%S)"
backup_dir="outputs/backup"
backup_file="${backup_dir}/atividade_2_${timestamp}.sql"
root_backup_file="${backup_root_file_override:-${BACKUP_ROOT_FILE:-backup_atividade_2.sql}}"

mkdir -p "$backup_dir"

docker exec \
  -e PGPASSWORD="$POSTGRES_PASSWORD" \
  "$POSTGRES_CONTAINER_NAME" \
  pg_dump \
    -U "$POSTGRES_USER" \
    -d "$POSTGRES_DB" \
    --no-owner \
    --no-privileges > "$backup_file"

echo "Backup written to $backup_file"
if [ "$APP_ENV" = "prod" ]; then
  cp "$backup_file" "$root_backup_file"
  echo "Latest root backup written to $root_backup_file"
else
  echo "Latest root backup skipped for APP_ENV=$APP_ENV"
fi
