from pathlib import Path
from types import SimpleNamespace

from atividade_2 import database_dump
from atividade_2.database_dump import DatabaseDumpService, DatabaseResetService


def test_create_dump_updates_root_backup_outside_prod(monkeypatch, tmp_path) -> None:
    output_dir = tmp_path / "outputs" / "backup"
    root_backup = tmp_path / "backup_atividade_2.sql"

    monkeypatch.setattr(database_dump.shutil, "which", lambda name: f"/usr/bin/{name}")

    def fake_run(command, **kwargs):
        output_path = Path(command[command.index("--file") + 1])
        output_path.write_text("SELECT 1;\n", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(database_dump.subprocess, "run", fake_run)

    service = DatabaseDumpService(
        output_dir=output_dir,
        root_backup_file=root_backup,
        settings_loader=lambda: SimpleNamespace(
            app_env="dev",
            database_url="postgresql://postgres:postgres@localhost:5432/app_dev",
            backup_root_file=str(root_backup),
        ),
        now=lambda: database_dump.datetime(2026, 5, 2, 9, 30, 0),
    )

    result = service.create_dump()

    history_backup = output_dir / "atividade_2_20260502_093000.sql"
    assert history_backup.read_text(encoding="utf-8") == "SELECT 1;\n"
    assert root_backup.read_text(encoding="utf-8") == "SELECT 1;\n"
    assert result.path == str(history_backup.resolve())
    assert result.filename == "atividade_2_20260502_093000.sql"
    assert result.delivery == "local"


def test_create_dump_omits_environment_specific_ownership_and_privileges(monkeypatch, tmp_path) -> None:
    output_dir = tmp_path / "outputs" / "backup"
    captured_commands = []

    monkeypatch.setattr(database_dump.shutil, "which", lambda name: f"/usr/bin/{name}")

    def fake_run(command, **kwargs):
        captured_commands.append(command)
        output_path = Path(command[command.index("--file") + 1])
        output_path.write_text("SELECT 1;\n", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(database_dump.subprocess, "run", fake_run)

    service = DatabaseDumpService(
        output_dir=output_dir,
        settings_loader=lambda: SimpleNamespace(
            app_env="prod",
            database_url="postgresql://postgres:postgres@localhost:5432/app_dev",
            backup_root_file=str(tmp_path / "backup_atividade_2.sql"),
        ),
        now=lambda: database_dump.datetime(2026, 5, 2, 9, 30, 0),
    )

    service.create_dump()

    assert captured_commands == [
        [
            "/usr/bin/pg_dump",
            "postgresql://postgres:postgres@localhost:5432/app_dev",
            "--no-owner",
            "--no-privileges",
            "--file",
            str((output_dir / "atividade_2_20260502_093000.sql").resolve()),
        ]
    ]


def test_create_dump_uses_browser_download_in_prod(monkeypatch, tmp_path) -> None:
    output_dir = tmp_path / "outputs" / "backup"
    root_backup = tmp_path / "backup_atividade_2.sql"

    monkeypatch.setattr(database_dump.shutil, "which", lambda name: f"/usr/bin/{name}")

    def fake_run(command, **kwargs):
        output_path = Path(command[command.index("--file") + 1])
        output_path.write_text("SELECT 1;\n", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(database_dump.subprocess, "run", fake_run)

    service = DatabaseDumpService(
        output_dir=output_dir,
        settings_loader=lambda: SimpleNamespace(
            app_env="prod",
            database_url="postgresql://postgres:postgres@localhost:5432/app_dev",
            backup_root_file=str(root_backup),
        ),
        now=lambda: database_dump.datetime(2026, 5, 2, 9, 30, 0),
    )

    result = service.create_dump()

    assert not root_backup.exists()
    assert result.delivery == "browser_download"
    assert result.download_url == "/api/database-dumps/atividade_2_20260502_093000.sql"


def test_restore_backup_adds_dashboard_compatibility_columns(monkeypatch, tmp_path) -> None:
    backup_file = tmp_path / "atividade_2_20260430_120000.sql"
    backup_file.write_text("SELECT 1;", encoding="utf-8")
    commands = []

    monkeypatch.setattr(database_dump.shutil, "which", lambda name: f"/usr/bin/{name}")

    def fake_run_command(command, timeout_seconds, *, secret=None, expected_stdout=None):
        commands.append(command)

    monkeypatch.setattr(database_dump, "_run_command", fake_run_command)
    service = DatabaseResetService(
        settings_loader=lambda: SimpleNamespace(database_url="postgresql://postgres:postgres@localhost:5432/app_dev")
    )

    service.restore_backup(backup_file)

    alter_commands = [command for command in commands if "ALTER TABLE avaliacoes_juiz" in " ".join(command)]
    assert len(alter_commands) == 1
    alter_sql = " ".join(alter_commands[0])
    assert "ADD COLUMN IF NOT EXISTS papel_juiz" in alter_sql
    assert "ADD COLUMN IF NOT EXISTS rodada_julgamento" in alter_sql
    assert "ADD COLUMN IF NOT EXISTS motivo_acionamento" in alter_sql
    assert "ADD COLUMN IF NOT EXISTS status_avaliacao" in alter_sql


def test_reset_service_defaults_to_fixed_reset_backup() -> None:
    service = DatabaseResetService()

    assert service.backup_file == Path("backup_atividade_2_reset.sql")
