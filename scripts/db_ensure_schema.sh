#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ ! -x ".venv/bin/python" ]; then
  echo "Missing .venv/bin/python. Create the virtualenv and install dependencies first." >&2
  exit 1
fi

.venv/bin/python - <<'PY'
from atividade_2.config import load_settings
from atividade_2.db import connect
from atividade_2.repositories import JudgeRepository

settings = load_settings()
print("Ensuring database schema on active DATABASE_URL...")

connection = connect(settings.database_url)

try:
    repository = JudgeRepository(connection)
    repository.ensure_schema()
    print("Schema ensured.")

    print("Upserting AV3 candidate model assignments...")
    assignments = repository.upsert_default_candidate_model_assignments()
    print(f"Seeded {len(assignments)} candidate model assignments.")

    print("Database schema is up to date.")
finally:
    connection.close()
PY
