"""Command-line entry point for the Atividade 2 package."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from .audit_log_parser import DEFAULT_PROD_LOGS_MANIFEST, format_audit_parse_report, parse_prod_logs_manifest
from .config import ConfigurationError
from .judge_clients.remote_http import RemoteJudgeError
from .parser import JudgeParseError
from .config import load_settings
from .db import connect
from .run_judge_service import ResolvedRun, RunJudgeRequest, RunJudgeService, format_execution_summary


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser without executing application logic."""
    parser = argparse.ArgumentParser(
        prog="atividade-2",
        description="Reusable command-line entry point for Atividade 2.",
    )
    subparsers = parser.add_subparsers(dest="command")
    run_judge = subparsers.add_parser(
        "run-judge",
        help="Run the local LLM-as-a-Judge pipeline with a remote HTTP model endpoint.",
    )
    run_judge.add_argument("--judge-provider", choices=["remote_http"])
    run_judge.add_argument("--panel-mode", choices=["single", "primary_only", "2plus1"])
    run_judge.add_argument("--judge-model", help="Judge 1 alias or provider model id.")
    run_judge.add_argument("--secondary-judge-model", help="Judge 2 alias or provider model id.")
    run_judge.add_argument("--arbiter-judge-model", help="Arbiter alias or provider model id.")
    run_judge.add_argument(
        "--always-run-arbiter",
        action="store_true",
        help="Run the arbiter for every answer in 2plus1 mode.",
    )
    run_judge.add_argument(
        "--judge-execution-strategy",
        choices=["sequential", "parallel", "adaptive"],
        help="Run judge API calls sequentially or in parallel within each answer.",
    )
    run_judge.add_argument(
        "--dataset",
        choices=["J1", "J2", "OAB_Bench", "OAB_Exames"],
        default="J2",
        help="Dataset to evaluate. J2 maps to OAB_Exames; J1 maps to OAB_Bench.",
    )
    run_judge.add_argument(
        "--limit",
        type=_positive_int,
        default=10,
        help="Maximum candidate answers to evaluate.",
    )
    run_judge.add_argument(
        "--batch-size",
        type=_positive_int,
        help="Maximum pending candidate answers to evaluate. Defaults to JUDGE_BATCH_SIZE or 10.",
    )
    run_judge.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve configuration and print the execution summary without DB or HTTP calls.",
    )
    run_judge.add_argument(
        "--preflight-report",
        action="store_true",
        help="Print the adaptive concurrency plan without DB selection or judge evaluation.",
    )
    run_judge.add_argument(
        "--audit-log",
        help="Path for detailed audit log. Defaults to outputs/audit/judge_run_<timestamp>.log.",
    )
    run_judge.add_argument(
        "--no-audit-animation",
        action="store_true",
        help="Disable animated terminal dots for long-running audit steps.",
    )
    run_judge.set_defaults(handler=run_judge_command)

    save_prompt = subparsers.add_parser(
        "save-default-prompt",
        help="Create and activate a new prompt_juizes version from the repository defaults.",
    )
    save_prompt.add_argument(
        "--dataset",
        choices=["J1", "J2", "OAB_Bench", "OAB_Exames"],
        default="J1",
        help="Prompt dataset to version. J1 maps to OAB_Bench; J2 maps to OAB_Exames.",
    )
    save_prompt.add_argument(
        "--changed-by",
        required=True,
        help="Value stored as created_by for the new prompt version.",
    )
    save_prompt.set_defaults(handler=save_default_prompt_command)

    parse_logs = subparsers.add_parser(
        "parse-prod-logs",
        help="Read-only parse of production audit logs listed in the manifest.",
    )
    parse_logs.add_argument(
        "--manifest",
        default=str(DEFAULT_PROD_LOGS_MANIFEST),
        help="Manifest file listing production audit logs to parse.",
    )
    parse_logs.set_defaults(handler=parse_prod_logs_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse command-line arguments and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        return 0
    try:
        return handler(args)
    except (ConfigurationError, RemoteJudgeError, JudgeParseError, RuntimeError) as error:
        parser.exit(2, f"error: {error}\n")


def run_judge_command(args: argparse.Namespace) -> int:
    """Run or dry-run the judge pipeline."""
    request = RunJudgeRequest(
        judge_provider=args.judge_provider,
        panel_mode=args.panel_mode,
        judge_model=args.judge_model,
        secondary_judge_model=args.secondary_judge_model,
        arbiter_judge_model=args.arbiter_judge_model,
        always_run_arbiter=True if args.always_run_arbiter else None,
        judge_execution_strategy=args.judge_execution_strategy,
        dataset=args.dataset,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        preflight_report=args.preflight_report,
        audit_log=args.audit_log,
        no_audit_animation=args.no_audit_animation,
    )
    result = RunJudgeService().run(request, on_resolved=_print_resolved_run)
    if result.summary is not None:
        print()
        print("Execution result:")
        print(f"Selected answers: {result.summary.selected_answers}")
        print(f"Executed evaluations: {result.summary.executed_evaluations}")
        print(f"Skipped existing evaluations: {result.summary.skipped_evaluations}")
        print(f"Arbiter evaluations: {result.summary.arbiter_evaluations}")
    return 0


def save_default_prompt_command(args: argparse.Namespace) -> int:
    """Persist a new prompt_juizes version using repository defaults."""
    from .judge_prompt_configs import resolve_prompt_dataset_name
    from .repositories import JudgeRepository, _default_prompt_config

    dataset_name = resolve_prompt_dataset_name(args.dataset)
    defaults = _default_prompt_config(dataset_name)

    settings = load_settings()
    connection = connect(settings.database_url)
    try:
        repository = JudgeRepository(connection)
        repository.ensure_schema()
        record = repository.create_prompt_config_version(
            dataset=dataset_name,
            prompt=defaults["prompt"],
            persona=defaults["persona"],
            context=defaults["context"],
            rubric=defaults["rubric"],
            output=defaults["output"],
            changed_by=str(args.changed_by).strip(),
        )
    finally:
        connection.close()

    print("Prompt version saved and activated:")
    print(f"- dataset: {record.dataset}")
    print(f"- prompt_id: {record.prompt_id}")
    print(f"- version: {record.version}")
    print(f"- created_by: {record.created_by}")
    return 0


def parse_prod_logs_command(args: argparse.Namespace) -> int:
    """Parse whitelisted production audit logs without side effects."""
    report = parse_prod_logs_manifest(args.manifest)
    print(format_audit_parse_report(report))
    return 1 if report.problems or report.missing_logs else 0


def _print_resolved_run(resolved: ResolvedRun) -> None:
    print(resolved.execution_summary)
    print(f"Batch size: {resolved.batch_size}")
    print(f"Audit log: {resolved.audit_path}")


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be an integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
