#!/usr/bin/env python
"""Incremental smoke runner for AV3 candidate RAG execution.

Examples:

  # 1 call per runnable assignment/model range
  .venv/bin/python scripts/smoke_candidates_rag.py \
    --dataset J1 \
    --calls-per-model 1 \
    --sequence-offset 0

  # next 2 calls per assignment, avoiding the first question already tested
  .venv/bin/python scripts/smoke_candidates_rag.py \
    --dataset J1 \
    --calls-per-model 2 \
    --sequence-offset 1

  # next 5 calls per assignment
  .venv/bin/python scripts/smoke_candidates_rag.py \
    --dataset J1 \
    --calls-per-model 5 \
    --sequence-offset 3

  # dry-run only
  .venv/bin/python scripts/smoke_candidates_rag.py \
    --dataset J1 \
    --calls-per-model 1 \
    --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SmokeTarget:
    owner: str
    model: str
    av3_provider: str
    dataset: str
    range_start: int
    range_end: int
    selected_start: int
    selected_end: int
    calls: int


@dataclass
class SmokeResult:
    target: SmokeTarget
    return_code: int
    selected_questions: int | None = None
    processed_questions: int | None = None
    successful_answers: int | None = None
    failed_answers: int | None = None
    skipped_questions: int | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""

    @property
    def ok(self) -> bool:
        return (
            self.return_code == 0
            and (self.failed_answers in (0, None))
            and (self.successful_answers is None or self.successful_answers > 0 or self.skipped_questions > 0)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run incremental AV3 candidate RAG smoke tests.")
    parser.add_argument("--dataset", default="J1", choices=["J1", "J2"], help="Dataset to smoke test.")
    parser.add_argument("--calls-per-model", type=int, default=1, help="Number of questions to execute per target.")
    parser.add_argument(
        "--sequence-offset",
        type=int,
        default=0,
        help="Offset from the assignment range start. Use 0 for first smoke, 1 or more for later stages.",
    )
    parser.add_argument(
        "--provider",
        default="remote_http",
        choices=["remote_http"],
        help="Technical candidate provider adapter.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Resolve and print commands without remote calls.")
    parser.add_argument(
        "--candidate-execution-strategy",
        choices=["sequential", "parallel", "adaptive"],
        help="Pass through candidate execution strategy to run-candidates-rag.",
    )
    parser.add_argument(
        "--unique-models",
        action="store_true",
        help="Run only once per provider model id, using the first runnable assignment/range.",
    )
    parser.add_argument(
        "--include-model",
        action="append",
        default=[],
        help="Restrict to one provider model id. May be repeated.",
    )
    parser.add_argument(
        "--exclude-model",
        action="append",
        default=[],
        help="Exclude one provider model id. May be repeated.",
    )
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Continue executing remaining targets after a failed smoke.",
    )
    parser.add_argument(
        "--summary-json",
        default="outputs/av3/candidate_rag_smoke_summary.json",
        help="Path to write JSON summary.",
    )
    parser.add_argument(
        "--no-audit-animation",
        action="store_true",
        default=True,
        help="Disable CLI audit animation.",
    )
    return parser.parse_args()


def load_targets(args: argparse.Namespace) -> list[SmokeTarget]:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    from atividade_2.config import load_settings
    from atividade_2.db import connect
    from atividade_2.repositories import JudgeRepository

    settings = load_settings()
    connection = connect(settings.database_url)

    try:
        repository = JudgeRepository(connection)
        repository.ensure_schema()

        vector_base = repository.get_rag_vector_base_summary(dataset=args.dataset)
        if vector_base is None:
            raise SystemExit(f"No active RAG vector base found for {args.dataset}.")

        chunk_count = int(getattr(vector_base, "chunk_count", 0) or 0)
        embedding_count = int(getattr(vector_base, "embedding_count", 0) or 0)
        if chunk_count <= 0 or embedding_count <= 0:
            raise SystemExit(
                f"RAG vector base for {args.dataset} is not ready: "
                f"chunks={chunk_count}, embeddings={embedding_count}."
            )

        include_models = {model.strip() for model in args.include_model if model.strip()}
        exclude_models = {model.strip() for model in args.exclude_model if model.strip()}

        targets: list[SmokeTarget] = []
        seen_models: set[str] = set()

        for assignment in repository.list_candidate_model_assignments():
            if not assignment.active or not assignment.is_runnable():
                continue

            model = (assignment.av3_provider_model_id or "").strip()
            if not model:
                continue

            if include_models and model not in include_models:
                continue

            if model in exclude_models:
                continue

            if args.unique_models and model in seen_models:
                continue

            for assignment_range in assignment.ranges:
                if assignment_range.dataset_code != args.dataset:
                    continue

                range_start = int(assignment_range.question_sequence_start)
                range_end = int(assignment_range.question_sequence_end)

                selected_start = range_start + int(args.sequence_offset)
                selected_end = min(range_end, selected_start + int(args.calls_per_model) - 1)

                if selected_start > range_end:
                    continue

                calls = selected_end - selected_start + 1
                targets.append(
                    SmokeTarget(
                        owner=assignment.owner,
                        model=model,
                        av3_provider=assignment.av3_provider,
                        dataset=args.dataset,
                        range_start=range_start,
                        range_end=range_end,
                        selected_start=selected_start,
                        selected_end=selected_end,
                        calls=calls,
                    )
                )

                seen_models.add(model)
                break

        return targets
    finally:
        connection.close()


def build_command(args: argparse.Namespace, target: SmokeTarget) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "atividade_2.cli",
        "run-candidates-rag",
        "--dataset",
        target.dataset,
        "--candidate-model",
        target.model,
        "--provider",
        args.provider,
        "--batch-size",
        str(target.calls),
        "--question-sequence-start",
        str(target.selected_start),
        "--question-sequence-end",
        str(target.selected_end),
    ]

    if args.dry_run:
        command.append("--dry-run")

    if args.no_audit_animation:
        command.append("--no-audit-animation")

    if args.candidate_execution_strategy:
        command.extend(["--candidate-execution-strategy", args.candidate_execution_strategy])

    return command


def parse_execution_output(stdout: str) -> dict[str, int | None]:
    fields = {
        "selected_questions": r"Selected questions:\s+(\d+)",
        "processed_questions": r"Processed questions:\s+(\d+)",
        "successful_answers": r"Successful answers:\s+(\d+)",
        "failed_answers": r"Failed answers:\s+(\d+)",
        "skipped_questions": r"Skipped questions:\s+(\d+)",
    }

    parsed: dict[str, int | None] = {}
    for key, pattern in fields.items():
        match = re.search(pattern, stdout)
        parsed[key] = int(match.group(1)) if match else None

    return parsed


def run_target(args: argparse.Namespace, target: SmokeTarget) -> SmokeResult:
    command = build_command(args, target)

    print()
    print("=" * 100)
    print(f"Owner: {target.owner}")
    print(f"Model: {target.model}")
    print(f"AV3 provider: {target.av3_provider}")
    print(f"Dataset/range: {target.dataset} {target.selected_start}-{target.selected_end}")
    print("Command:")
    print(" ".join(command))
    print("=" * 100)

    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        env=os.environ.copy(),
        check=False,
    )

    print(completed.stdout)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr)

    parsed = parse_execution_output(completed.stdout)

    return SmokeResult(
        target=target,
        return_code=completed.returncode,
        selected_questions=parsed["selected_questions"],
        processed_questions=parsed["processed_questions"],
        successful_answers=parsed["successful_answers"],
        failed_answers=parsed["failed_answers"],
        skipped_questions=parsed["skipped_questions"],
        stdout_tail="\n".join(completed.stdout.splitlines()[-30:]),
        stderr_tail="\n".join(completed.stderr.splitlines()[-30:]),
    )


def write_summary(path: str, results: list[SmokeResult]) -> None:
    output_path = PROJECT_ROOT / path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "total": len(results),
        "ok": sum(1 for result in results if result.ok),
        "failed": sum(1 for result in results if not result.ok),
        "results": [
            {
                **asdict(result),
                "ok": result.ok,
            }
            for result in results
        ],
    }

    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(f"Summary JSON written to: {output_path}")


def main() -> int:
    args = parse_args()

    if args.calls_per_model < 1:
        raise SystemExit("--calls-per-model must be >= 1.")

    if args.sequence_offset < 0:
        raise SystemExit("--sequence-offset must be >= 0.")

    targets = load_targets(args)

    if not targets:
        print("No runnable smoke targets found.")
        return 1

    print(f"Smoke targets: {len(targets)}")
    for target in targets:
        print(
            f"- {target.owner} | {target.model} | {target.av3_provider} | "
            f"{target.dataset} {target.selected_start}-{target.selected_end}"
        )

    results: list[SmokeResult] = []
    for target in targets:
        result = run_target(args, target)
        results.append(result)

        if not result.ok and not args.continue_on_failure:
            print()
            print("Stopping after first failed smoke. Use --continue-on-failure to run all targets.")
            break

    write_summary(args.summary_json, results)

    ok = sum(1 for result in results if result.ok)
    failed = sum(1 for result in results if not result.ok)

    print()
    print("Smoke summary")
    print(f"- total executed: {len(results)}")
    print(f"- ok: {ok}")
    print(f"- failed: {failed}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
