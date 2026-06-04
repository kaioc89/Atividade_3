#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required but was not found in PATH." >&2
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "Docker is installed, but the Docker daemon is not running." >&2
  exit 1
fi

if [ ! -f ".env" ]; then
  cp ".env.example" ".env"
  echo "Created .env from .env.example."
fi

set -a
# shellcheck disable=SC1091
source ".env"
set +a

POSTGRES_IMAGE="${POSTGRES_IMAGE:-pgvector/pgvector:0.8.2-pg18}"
POSTGRES_CONTAINER_NAME="${POSTGRES_CONTAINER_NAME:-topicos-av2-postgres}"
POSTGRES_USER="${POSTGRES_USER:-postgres}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-postgres}"
POSTGRES_DB="${POSTGRES_DB:-app_dev}"
POSTGRES_TEST_DB="${POSTGRES_TEST_DB:-app_test}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"

if [[ ! "$POSTGRES_TEST_DB" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
  echo "POSTGRES_TEST_DB must be a valid PostgreSQL identifier: $POSTGRES_TEST_DB" >&2
  exit 1
fi

if ! docker image inspect "$POSTGRES_IMAGE" >/dev/null 2>&1; then
  docker pull "$POSTGRES_IMAGE"
fi

docker compose --env-file ".env" up -d postgres

ready_attempts=0
max_ready_attempts=60
until docker exec "$POSTGRES_CONTAINER_NAME" pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB" >/dev/null 2>&1; do
  ready_attempts=$((ready_attempts + 1))
  if [ "$ready_attempts" -ge "$max_ready_attempts" ]; then
    echo "PostgreSQL did not become ready within ${max_ready_attempts} seconds." >&2
    docker compose --env-file ".env" ps >&2
    exit 1
  fi
  sleep 1
done

docker exec \
  -e PGPASSWORD="$POSTGRES_PASSWORD" \
  "$POSTGRES_CONTAINER_NAME" \
  psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT 1;" >/dev/null

printf "SELECT format('CREATE DATABASE %%I', '%s') WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = '%s')\\gexec\n" \
  "$POSTGRES_TEST_DB" \
  "$POSTGRES_TEST_DB" | docker exec -i \
  -e PGPASSWORD="$POSTGRES_PASSWORD" \
  "$POSTGRES_CONTAINER_NAME" \
  psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB" >/dev/null

echo "PostgreSQL is ready."
echo "DATABASE_URL=postgresql://${POSTGRES_USER}:<password>@localhost:${POSTGRES_PORT}/${POSTGRES_DB}"
