VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest

.PHONY: venv install test db-up db-down db-logs db-psql db-status db-reset db-migrate-or-create db-ensure-schema db-restore-validate db-backup db-backup-promote db-dump-structure db-dump-questions db-dump-responses db-dump-root-backup db-dump-all web-up web-down web-logs clean

venv:
	@if [ ! -d "$(VENV)" ]; then \
		if command -v python3.11 >/dev/null 2>&1; then \
			python3.11 -m venv $(VENV); \
		else \
			python3 -m venv $(VENV); \
		fi; \
	fi

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

test:
	$(PYTEST)

db-up:
	./scripts/db_up.sh

db-down:
	docker compose --env-file .env down

db-logs:
	docker compose --env-file .env logs -f postgres

db-psql:
	docker exec -it $$(grep '^POSTGRES_CONTAINER_NAME=' .env | cut -d '=' -f2-) psql -U $$(grep '^POSTGRES_USER=' .env | cut -d '=' -f2-) -d $$(grep '^POSTGRES_DB=' .env | cut -d '=' -f2-)

db-status:
	docker compose --env-file .env ps

db-reset:
	docker compose --env-file .env down -v

db-migrate-or-create:
	./scripts/db_migrate_or_create.sh $(if $(FORCE),--force)

db-ensure-schema:
	@./scripts/db_ensure_schema.sh

db-restore-validate:
	./scripts/db_restore_validate.sh

db-backup:
	./scripts/db_backup.sh

db-backup-promote:
	PROMOTE_BACKUP=1 ./scripts/db_backup.sh

web-up: db-up
	./scripts/web_up.sh

web-down:
	docker compose --env-file .env stop judge-web

web-logs:
	docker compose --env-file .env logs -f judge-web

db-dump-structure:
	./scripts/db_dump_structure.sh

db-dump-questions:
	./scripts/db_dump_questions.sh

db-dump-responses:
	./scripts/db_dump_responses.sh

db-dump-root-backup:
	./scripts/db_dump_root_backup.sh

db-dump-all: db-dump-structure db-dump-questions db-dump-responses db-backup

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .pytest_cache -prune -exec rm -rf {} +
	find . -type d -name "*.egg-info" -prune -exec rm -rf {} +
