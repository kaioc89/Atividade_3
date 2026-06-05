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
promote_backup_override="${PROMOTE_BACKUP-}"

set -a
# shellcheck disable=SC1091
source ".env"
set +a

POSTGRES_CONTAINER_NAME="${POSTGRES_CONTAINER_NAME:-topicos-av2-postgres}"
POSTGRES_USER="${POSTGRES_USER:-postgres}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-postgres}"
POSTGRES_DB="${POSTGRES_DB:-app_dev}"
APP_ENV="${app_env_override:-${APP_ENV:-dev}}"
PROMOTE_BACKUP="${promote_backup_override:-0}"
timestamp="$(date +%Y%m%d_%H%M%S)"
backup_dir="outputs/backup"
backup_file="${backup_dir}/atividade_2_${timestamp}.sql"
root_backup_file="${backup_root_file_override:-${BACKUP_ROOT_FILE:-backup_atividade_2.sql}}"
validation_sql_file="scripts/sql/backup_promotion_validation.sql"

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

if [ "$PROMOTE_BACKUP" != "1" ]; then
  echo "Canonical root backup not updated. Use PROMOTE_BACKUP=1 or make db-backup-promote."
  exit 0
fi

if [ ! -f "$validation_sql_file" ]; then
  echo "Promotion validation SQL not found: $validation_sql_file" >&2
  exit 1
fi

validation_output="$(
  docker exec -i \
    -e PGPASSWORD="$POSTGRES_PASSWORD" \
    "$POSTGRES_CONTAINER_NAME" \
    psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB" -P pager=off -F $'\t' -At -f - \
    < "$validation_sql_file"
)"

while IFS=$'\t' read -r table_name row_count; do
  if [ -z "$table_name" ]; then
    continue
  fi
  echo "Promotion check ${table_name}: ${row_count} row(s)"
  if [ "${row_count}" = "0" ]; then
    echo "Promotion aborted: critical table is empty (${table_name})." >&2
    exit 1
  fi
done <<< "$validation_output"

cp "$backup_file" "$root_backup_file"
echo "Canonical root backup promoted to $root_backup_file"
