from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_db_backup_writes_only_timestamped_dump(tmp_path: Path) -> None:
    workspace = _prepare_backup_workspace(tmp_path)
    root_backup = workspace / "backup_atividade_2.sql"
    root_backup.write_text("canonical backup\n", encoding="utf-8")

    result = subprocess.run(
        ["bash", "./scripts/db_backup.sh"],
        cwd=workspace,
        env=_build_test_env(tmp_path),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert root_backup.read_text(encoding="utf-8") == "canonical backup\n"

    backup_files = list((workspace / "outputs" / "backup").glob("atividade_2_*.sql"))
    assert len(backup_files) == 1
    assert backup_files[0].read_text(encoding="utf-8") == "SELECT 1;\n"
    assert "Canonical root backup not updated" in result.stdout


def test_db_backup_promote_copies_dump_to_canonical_backup(tmp_path: Path) -> None:
    workspace = _prepare_backup_workspace(tmp_path)
    root_backup = workspace / "backup_atividade_2.sql"
    root_backup.write_text("canonical backup\n", encoding="utf-8")

    result = subprocess.run(
        ["make", "db-backup-promote"],
        cwd=workspace,
        env=_build_test_env(tmp_path),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert root_backup.read_text(encoding="utf-8") == "SELECT 1;\n"
    assert "Canonical root backup promoted" in result.stdout


def test_db_backup_promote_aborts_when_critical_table_is_empty(tmp_path: Path) -> None:
    workspace = _prepare_backup_workspace(tmp_path)
    root_backup = workspace / "backup_atividade_2.sql"
    root_backup.write_text("canonical backup\n", encoding="utf-8")

    result = subprocess.run(
        ["make", "db-backup-promote"],
        cwd=workspace,
        env=_build_test_env(
            tmp_path,
            fake_psql_output=(
                "public.respostas_atividade_1\t5\n"
                "public.avaliacoes_juiz\t5\n"
                "av3.rag_chunks\t0\n"
                "av3.rag_embeddings\t5\n"
                "av3.retrieval_runs.ativo=true\t1\n"
            ),
        ),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert root_backup.read_text(encoding="utf-8") == "canonical backup\n"

    backup_files = list((workspace / "outputs" / "backup").glob("atividade_2_*.sql"))
    assert len(backup_files) == 1
    assert "Promotion aborted: critical table is empty (av3.rag_chunks)." in (result.stdout + result.stderr)


def _prepare_backup_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    for relative_path in [
        "Makefile",
        "scripts/db_backup.sh",
        "scripts/db_dump_root_backup.sh",
        "scripts/sql/backup_promotion_validation.sql",
    ]:
        source_path = REPO_ROOT / relative_path
        target_path = workspace / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        target_path.chmod(target_path.stat().st_mode | stat.S_IXUSR)

    (workspace / ".env").write_text(
        "\n".join(
            [
                "POSTGRES_CONTAINER_NAME=topicos-av2-postgres",
                "POSTGRES_USER=postgres",
                "POSTGRES_PASSWORD=postgres",
                "POSTGRES_DB=app_dev",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return workspace


def _build_test_env(tmp_path: Path, *, fake_psql_output: str | None = None) -> dict[str, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    docker_path = bin_dir / "docker"
    docker_path.write_text(
        """#!/bin/sh
set -eu

if [ "$1" = "ps" ]; then
  printf '%s\\n' "${FAKE_DOCKER_PS_NAMES:-topicos-av2-postgres}"
  exit 0
fi

if [ "$1" != "exec" ]; then
  echo "unexpected docker command: $*" >&2
  exit 1
fi

shift
if [ "${1:-}" = "-i" ]; then
  shift
fi
while [ "${1:-}" = "-e" ]; do
  shift 2
done

container="$1"
shift
tool="$1"
shift

if [ "$container" != "topicos-av2-postgres" ]; then
  echo "unexpected container: $container" >&2
  exit 1
fi

case "$tool" in
  pg_dump)
    printf 'SELECT 1;\\n'
    ;;
  psql)
    cat >/dev/null
    printf '%b' "${FAKE_PSQL_OUTPUT:-public.respostas_atividade_1\\t5\\npublic.avaliacoes_juiz\\t5\\nav3.rag_chunks\\t5\\nav3.rag_embeddings\\t5\\nav3.retrieval_runs.ativo=true\\t1\\n}"
    ;;
  *)
    echo "unexpected docker exec tool: $tool" >&2
    exit 1
    ;;
esac
""",
        encoding="utf-8",
    )
    docker_path.chmod(docker_path.stat().st_mode | stat.S_IXUSR)

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    if fake_psql_output is not None:
        env["FAKE_PSQL_OUTPUT"] = fake_psql_output
    return env
