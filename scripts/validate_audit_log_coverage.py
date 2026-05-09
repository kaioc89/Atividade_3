"""Validate final judge evaluations against versioned production audit logs.

This script is intentionally offline and read-only: it parses the root database
backup and the whitelisted audit logs without connecting to PostgreSQL.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
BACKUP_PATH = ROOT / "backup_atividade_2.sql"
AUDIT_DIR = ROOT / "outputs" / "audit"
EXPECTED_EVALUATIONS = 4853
PROD_LOG_NAMES = (
    "judge_run_20260502_011939.log",
    "judge_run_20260502_012032.log",
    "judge_run_20260502_012849.log",
    "judge_run_20260502_013551.log",
    "judge_run_20260502_114739.log",
    "judge_run_20260502_114752.log",
    "judge_run_20260502_115439.log",
    "judge_run_20260502_120033.log",
    "judge_run_20260502_120134.log",
    "judge_run_20260502_120852.log",
    "judge_run_20260502_121525.log",
    "judge_run_20260502_121547.log",
    "judge_run_20260502_123859.log",
    "judge_run_20260502_125316.log",
    "judge_run_20260502_125637.log",
    "judge_run_20260502_131434.log",
    "judge_run_20260502_131817.log",
    "judge_run_20260502_132809.log",
    "judge_run_20260502_133033.log",
    "judge_run_20260502_135929.log",
    "judge_run_20260502_140307.log",
    "judge_run_20260502_142245.log",
    "judge_run_20260502_143210.log",
    "judge_run_20260502_145559.log",
    "judge_run_20260502_152133.log",
    "judge_run_20260502_162154.log",
    "judge_run_20260502_184648.log",
    "judge_run_20260502_190325.log",
    "judge_run_20260502_191905.log",
    "judge_run_20260503_183245.log",
    "judge_run_20260503_183559.log",
    "judge_run_20260503_183955.log",
    "judge_run_20260503_185548.log",
    "judge_run_20260503_190135.log",
    "judge_run_20260503_190547.log",
    "judge_run_20260503_191029.log",
    "judge_run_20260503_200513.log",
    "judge_run_20260503_202358.log",
    "judge_run_20260503_203947.log",
    "judge_run_20260503_204303.log",
    "judge_run_20260503_220823.log",
    "judge_run_20260503_221218.log",
    "judge_run_20260503_222241.log",
    "judge_run_20260503_222934.log",
    "judge_run_20260503_223751.log",
    "judge_run_20260503_224237.log",
)

AUDIT_LINE_RE = re.compile(r"^(?P<ts>[^|]+) \| (?P<event>[^|]+)(?: \| (?P<detail>.*))?$")
KEY_VALUE_RE = re.compile(r"(\w+)=([^=]+?)(?= \w+=|$)")
SECRET_PATTERNS = (
    re.compile(r"authorization\s*:", re.IGNORECASE),
    re.compile(r"\bbearer\s+[A-Za-z0-9._~+/=-]{12,}", re.IGNORECASE),
    re.compile(r"\b(api[_-]?key|token|secret|password)\s*[:=]\s*(?!<missing>|<redacted>)[A-Za-z0-9._~+/=-]{12,}", re.IGNORECASE),
    re.compile(r"\bsk-[A-Za-z0-9]{20,}"),
)


@dataclass(frozen=True)
class BackupEvaluation:
    evaluation_id: int
    answer_id: int
    judge_model: str
    judge_name: str
    role: str
    score: int
    evaluated_at: datetime


@dataclass(frozen=True)
class LogEvaluation:
    answer_id: int
    judge_model: str
    role: str
    score: int


def main() -> int:
    errors: list[str] = []
    log_paths = [AUDIT_DIR / name for name in PROD_LOG_NAMES]
    missing_files = [path.name for path in log_paths if not path.is_file()]
    if missing_files:
        errors.append(f"Missing production audit logs: {', '.join(missing_files)}")

    existing_log_paths = [path for path in log_paths if path.is_file()]
    secret_hits = _scan_for_secrets(existing_log_paths)
    evaluations = _load_backup_evaluations()
    log_events = _load_log_evaluations(existing_log_paths)
    missing_evaluations, extra_events = _match_evaluations(evaluations, log_events)

    if len(evaluations) != EXPECTED_EVALUATIONS:
        errors.append(f"Expected {EXPECTED_EVALUATIONS} backup evaluations, found {len(evaluations)}.")
    if missing_evaluations:
        errors.append(f"{len(missing_evaluations)} final evaluations were not found in production logs.")
        errors.extend(_format_missing(row) for row in missing_evaluations[:20])
    if secret_hits:
        errors.append("Potential secret material found in versioned audit logs:")
        errors.extend(f"  {hit}" for hit in secret_hits[:20])

    print("Audit log coverage validation")
    print(f"- backup evaluations: {len(evaluations)}")
    print(f"- production logs checked: {len(PROD_LOG_NAMES)}")
    print(f"- evaluation events in logs: {len(log_events)}")
    print(f"- matched evaluations: {len(evaluations) - len(missing_evaluations)}/{len(evaluations)}")
    print(f"- final evaluations without log: {len(missing_evaluations)}")
    print(f"- extra log events not in final backup: {extra_events}")
    print(f"- potential secret findings: {len(secret_hits)}")

    if errors:
        print("\nValidation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("\nValidation passed.")
    return 0


def _copy_rows(table_name: str) -> list[list[str]]:
    marker = f"COPY public.{table_name} "
    rows: list[list[str]] = []
    in_copy = False
    for line in BACKUP_PATH.read_text(encoding="utf-8").splitlines():
        if not in_copy:
            if line.startswith(marker):
                in_copy = True
            continue
        if line == r"\.":
            break
        rows.append(line.split("\t"))
    return rows


def _load_backup_evaluations() -> list[BackupEvaluation]:
    models: dict[int, tuple[str, str]] = {}
    for row in _copy_rows("modelos"):
        name = row[1]
        version = row[2] if row[2] != r"\N" else name
        models[int(row[0])] = (name, version)

    evaluations: list[BackupEvaluation] = []
    for row in _copy_rows("avaliacoes_juiz"):
        model_name, model_version = models[int(row[2])]
        evaluations.append(
            BackupEvaluation(
                evaluation_id=int(row[0]),
                answer_id=int(row[1]),
                judge_model=model_version,
                judge_name=model_name,
                role=row[6],
                score=int(row[3]),
                evaluated_at=datetime.fromisoformat(row[5]),
            )
        )
    return evaluations


def _load_log_evaluations(paths: Iterable[Path]) -> list[LogEvaluation]:
    events: list[LogEvaluation] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = AUDIT_LINE_RE.match(line)
            if not match or match.group("event").strip() != "evaluation_parsed":
                continue
            values = dict(KEY_VALUE_RE.findall(match.group("detail") or ""))
            if {"answer_id", "model", "role", "score"} <= values.keys():
                events.append(
                    LogEvaluation(
                        answer_id=int(values["answer_id"]),
                        judge_model=values["model"].strip(),
                        role=values["role"].strip(),
                        score=int(values["score"]),
                    )
                )
    return events


def _match_evaluations(
    evaluations: list[BackupEvaluation],
    log_events: list[LogEvaluation],
) -> tuple[list[BackupEvaluation], int]:
    event_counts = Counter((event.answer_id, event.judge_model, event.role, event.score) for event in log_events)
    missing: list[BackupEvaluation] = []
    for evaluation in evaluations:
        keys = (
            (evaluation.answer_id, evaluation.judge_model, evaluation.role, evaluation.score),
            (evaluation.answer_id, evaluation.judge_name, evaluation.role, evaluation.score),
        )
        matched_key = next((key for key in keys if event_counts[key] > 0), None)
        if matched_key is None:
            missing.append(evaluation)
            continue
        event_counts[matched_key] -= 1
    return missing, sum(event_counts.values())


def _scan_for_secrets(paths: Iterable[Path]) -> list[str]:
    findings: list[str] = []
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
            if any(pattern.search(line) for pattern in SECRET_PATTERNS):
                findings.append(f"{path.relative_to(ROOT)}:{line_number}")
    return findings


def _format_missing(row: BackupEvaluation) -> str:
    return (
        f"  id_avaliacao={row.evaluation_id} answer_id={row.answer_id} "
        f"judge_model={row.judge_model} role={row.role} score={row.score} "
        f"evaluated_at={row.evaluated_at.isoformat(sep=' ')}"
    )


if __name__ == "__main__":
    sys.exit(main())
