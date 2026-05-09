"""PostgreSQL dump generation for Web UI audit exports."""

from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import load_settings

DUMP_FILENAME_PATTERN = re.compile(r"^atividade_2_\d{8}_\d{6}\.sql$")


@dataclass(frozen=True)
class DatabaseDumpResult:
    """Metadata for a generated database dump."""

    filename: str
    path: str
    size_bytes: int
    created_at: str
    download_url: str
    delivery: str


class DatabaseDumpService:
    """Generate complete plain SQL PostgreSQL dumps."""

    def __init__(
        self,
        *,
        output_dir: Path | str = Path("outputs") / "backup",
        root_backup_file: Path | str = Path("backup_atividade_2.sql"),
        settings_loader: Callable[[], Any] = load_settings,
        now: Callable[[], datetime] = datetime.now,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.root_backup_file = Path(root_backup_file)
        self._settings_loader = settings_loader
        self._now = now

    def create_dump(self) -> DatabaseDumpResult:
        """Run pg_dump and return the generated artifact metadata."""
        pg_dump = shutil.which("pg_dump")
        if pg_dump is None:
            raise RuntimeError("pg_dump não encontrado no ambiente da Web UI.")

        settings = self._settings_loader()
        created_at = self._now()
        filename = f"atividade_2_{created_at:%Y%m%d_%H%M%S}.sql"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = (self.output_dir / filename).resolve()
        command = [pg_dump, settings.database_url, "--file", str(output_path)]
        try:
            completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired as error:
            output_path.unlink(missing_ok=True)
            raise RuntimeError("pg_dump excedeu o tempo limite de 300 segundos.") from error

        if completed.returncode != 0:
            output_path.unlink(missing_ok=True)
            detail = _redact(completed.stderr or completed.stdout or "pg_dump falhou.", settings.database_url)
            raise RuntimeError(f"Falha ao gerar dump do banco: {detail}") from None

        configured_root_backup_file = Path(getattr(settings, "backup_root_file", self.root_backup_file))
        root_backup_path = configured_root_backup_file.resolve()
        delivery = "browser_download" if settings.app_env == "prod" else "local"
        if delivery == "local" and output_path != root_backup_path:
            shutil.copy2(output_path, root_backup_path)

        return DatabaseDumpResult(
            filename=filename,
            path=str(output_path),
            size_bytes=output_path.stat().st_size,
            created_at=created_at.isoformat(),
            download_url=f"/api/database-dumps/{filename}",
            delivery=delivery,
        )


class DatabaseResetService:
    """Restore the local database to the repository initial state."""

    def __init__(
        self,
        *,
        backup_file: Path | str = Path("backup_atividade_2_reset.sql"),
        settings_loader: Callable[[], Any] = load_settings,
        timeout_seconds: int = 600,
    ) -> None:
        self.backup_file = Path(backup_file)
        self._settings_loader = settings_loader
        self.timeout_seconds = timeout_seconds

    def reset_to_initial_state(self) -> dict:
        """Run the force restore flow and validate the restored database."""
        if Path("Makefile").exists() and shutil.which("make") is not None:
            _run_command(["make", "db-migrate-or-create", "FORCE=1"], self.timeout_seconds)
            _run_command(["make", "db-restore-validate"], self.timeout_seconds)
        else:
            self._restore_with_psql(self.backup_file)
        return {"status": "ok", "message": "Database restored to initial state."}

    def restore_backup(self, backup_file: Path | str) -> dict:
        """Restore a selected plain SQL backup and validate the restored database."""
        backup_path = Path(backup_file)
        self._restore_with_psql(backup_path)
        return {
            "status": "ok",
            "message": "Backup restored.",
            "filename": backup_path.name,
            "path": str(backup_path),
        }

    def _restore_with_psql(self, backup_file: Path) -> None:
        psql = shutil.which("psql")
        if psql is None:
            raise RuntimeError("psql não encontrado no ambiente da Web UI.")
        if not backup_file.exists():
            raise RuntimeError(f"Backup file not found: {backup_file}")

        settings = self._settings_loader()
        database_url = settings.database_url
        _run_command(
            [
                psql,
                "-v",
                "ON_ERROR_STOP=1",
                database_url,
                "-c",
                "DROP SCHEMA IF EXISTS public CASCADE; CREATE SCHEMA public;",
            ],
            self.timeout_seconds,
            secret=database_url,
        )
        _run_command(
            [psql, "-v", "ON_ERROR_STOP=1", database_url, "-f", str(backup_file)],
            self.timeout_seconds,
            secret=database_url,
        )
        _run_command(
            [
                psql,
                "-v",
                "ON_ERROR_STOP=1",
                database_url,
                "-c",
                (
                    "ALTER TABLE avaliacoes_juiz ADD COLUMN IF NOT EXISTS papel_juiz VARCHAR(20); "
                    "ALTER TABLE avaliacoes_juiz ADD COLUMN IF NOT EXISTS rodada_julgamento VARCHAR(30); "
                    "ALTER TABLE avaliacoes_juiz ADD COLUMN IF NOT EXISTS motivo_acionamento TEXT; "
                    "ALTER TABLE avaliacoes_juiz ADD COLUMN IF NOT EXISTS status_avaliacao VARCHAR(20) DEFAULT 'success';"
                ),
            ],
            self.timeout_seconds,
            secret=database_url,
        )
        for table_name in ["datasets", "modelos", "perguntas", "respostas_atividade_1", "avaliacoes_juiz"]:
            _run_command(
                [
                    psql,
                    "-v",
                    "ON_ERROR_STOP=1",
                    database_url,
                    "-tAc",
                    f"SELECT to_regclass('public.{table_name}') IS NOT NULL;",
                ],
                self.timeout_seconds,
                secret=database_url,
                expected_stdout="t",
            )


def resolve_dump_path(output_dir: Path | str, filename: str) -> Path:
    """Resolve a dump filename inside the configured output directory."""
    if not DUMP_FILENAME_PATTERN.fullmatch(filename):
        raise ValueError("Nome de dump inválido.")
    root = Path(output_dir).resolve()
    path = (root / filename).resolve()
    if path.parent != root:
        raise ValueError("Nome de dump inválido.")
    return path


def _redact(message: str, secret: str) -> str:
    redacted = message.replace(secret, "<redacted>")
    redacted = re.sub(r"postgresql://([^:\s]+):([^@\s]+)@", r"postgresql://\1:<redacted>@", redacted)
    return redacted.strip()


def _run_command(
    command: list[str],
    timeout_seconds: int,
    *,
    secret: str | None = None,
    expected_stdout: str | None = None,
) -> None:
    display_command = " ".join(command)
    if secret is not None:
        display_command = _redact(display_command, secret)
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout_seconds)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"Comando excedeu o tempo limite: {display_command}") from error

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "comando falhou").strip()
        if secret is not None:
            detail = _redact(detail, secret)
        raise RuntimeError(f"Falha ao executar {display_command}: {detail}") from None
    if expected_stdout is not None and completed.stdout.strip() != expected_stdout:
        raise RuntimeError(f"Falha ao validar {display_command}: resultado inesperado {completed.stdout.strip()!r}")
