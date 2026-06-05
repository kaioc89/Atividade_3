from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_db_ensure_schema_runs_repository_schema_and_seed_steps(tmp_path: Path) -> None:
    workspace = _prepare_workspace(tmp_path)

    result = subprocess.run(
        ["make", "db-ensure-schema"],
        cwd=workspace,
        env=dict(os.environ),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert result.stdout == (
        "Ensuring database schema on active DATABASE_URL...\n"
        "Schema ensured.\n"
        "Upserting AV3 candidate model assignments...\n"
        "Seeded 20 candidate model assignments.\n"
        "Database schema is up to date.\n"
    )


def test_db_ensure_schema_fails_when_project_venv_is_missing(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    for relative_path in ["Makefile", "scripts/db_ensure_schema.sh"]:
        source_path = REPO_ROOT / relative_path
        target_path = workspace / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        target_path.chmod(target_path.stat().st_mode | stat.S_IXUSR)

    result = subprocess.run(
        ["make", "db-ensure-schema"],
        cwd=workspace,
        env=dict(os.environ),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Missing .venv/bin/python" in result.stderr


def _prepare_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    for relative_path in ["Makefile", "scripts/db_ensure_schema.sh"]:
        source_path = REPO_ROOT / relative_path
        target_path = workspace / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        target_path.chmod(target_path.stat().st_mode | stat.S_IXUSR)

    venv_python = workspace / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True, exist_ok=True)
    venv_python.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
cat >/dev/null
printf '%s\\n' \
  'Ensuring database schema on active DATABASE_URL...' \
  'Schema ensured.' \
  'Upserting AV3 candidate model assignments...' \
  'Seeded 20 candidate model assignments.' \
  'Database schema is up to date.'
""",
        encoding="utf-8",
    )
    venv_python.chmod(venv_python.stat().st_mode | stat.S_IXUSR)

    return workspace
