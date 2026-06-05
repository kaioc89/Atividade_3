"""Command-line entry point for the Atividade 2 package."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from collections.abc import Sequence

from .audit_log_parser import DEFAULT_PROD_LOGS_MANIFEST, format_audit_parse_report, parse_prod_logs_manifest
from .config import ConfigurationError
from .judge_clients.remote_http import RemoteJudgeError
from .parser import JudgeParseError
from .config import load_env, load_settings
from .db import connect
from .provider_catalogs import (
    DEFAULT_FEATHERLESS_CATALOG_MAX_RESPONSE_BYTES,
    DEFAULT_OPENROUTER_CATALOG_MAX_RESPONSE_BYTES,
    FeatherlessCatalogClient,
    OpenRouterCatalogClient,
    ProviderCatalogClient,
)
from .provider_model_validation import (
    SUPPORTED_PROVIDER_VALIDATION_PROVIDERS,
    format_provider_model_validation_report,
    provider_model_validation_exit_code,
)
from .run_candidates_rag_service import RunCandidatesRagRequest, RunCandidatesRagService
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

    import_details = subparsers.add_parser(
        "import-evaluation-details",
        help="Import auxiliary judge output metadata from explicit historical sources.",
    )
    import_details.add_argument(
        "--manifest",
        default=str(DEFAULT_PROD_LOGS_MANIFEST),
        help="Manifest file listing production audit logs to inspect for raw judge outputs.",
    )
    import_details.add_argument(
        "--raw-output-dir",
        action="append",
        default=[],
        help="Directory or JSON/JSONL file with historical parsed/raw judge outputs. May be repeated.",
    )
    import_details.set_defaults(handler=import_evaluation_details_command)

    materialize_rag = subparsers.add_parser(
        "materialize-rag-base",
        help="Materialize active AV3 curation into rag_documents, rag_chunks, and a retrieval_run.",
    )
    materialize_rag.add_argument(
        "--dataset",
        choices=["J1", "J2", "OAB_Bench", "OAB_Exames"],
        required=True,
        help="Curated dataset to materialize. J1 maps to OAB_Bench; J2 maps to OAB_Exames.",
    )
    materialize_rag.add_argument(
        "--retrieval-name",
        help="Optional retrieval run name. Defaults to <dataset>_source_urls_v1.",
    )
    materialize_rag.add_argument(
        "--top-k",
        type=_positive_int,
        default=5,
        help="Top-k value stored on the generated retrieval run.",
    )
    materialize_rag.add_argument(
        "--chunking-strategy",
        default="source_url_only_v1",
        help="Chunking strategy label recorded in rag_chunks and retrieval_runs.",
    )
    materialize_rag.set_defaults(handler=materialize_rag_base_command)

    generate_rag_embeddings = subparsers.add_parser(
        "generate-rag-embeddings",
        help="Generate embeddings for the active materialized AV3 RAG base.",
    )
    generate_rag_embeddings.add_argument(
        "--dataset",
        choices=["J1", "J2", "OAB_Bench", "OAB_Exames"],
        required=True,
        help="Materialized dataset to embed. J1 maps to OAB_Bench; J2 maps to OAB_Exames.",
    )
    generate_rag_embeddings.add_argument(
        "--batch-size",
        type=_positive_int,
        default=32,
        help="Maximum chunk texts per embedding request.",
    )
    generate_rag_embeddings.set_defaults(handler=generate_rag_embeddings_command)

    run_candidates_rag = subparsers.add_parser(
        "run-candidates-rag",
        help="Run the AV3 candidate RAG generation pipeline without subprocesses.",
    )
    run_candidates_rag.add_argument(
        "--dataset",
        choices=["J1", "J2", "OAB_Bench", "OAB_Exames"],
        required=True,
        help="Dataset to execute. J1 maps to OAB_Bench; J2 maps to OAB_Exames.",
    )
    run_candidates_rag.add_argument(
        "--candidate-model",
        required=True,
        help="Candidate model alias or provider model id.",
    )
    run_candidates_rag.add_argument(
        "--provider",
        choices=["remote_http"],
        required=True,
        help="Candidate provider adapter.",
    )
    run_candidates_rag.add_argument(
        "--batch-size",
        type=_positive_int,
        required=True,
        help="Maximum pending questions to execute in this batch.",
    )
    run_candidates_rag.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve configuration and print the execution summary without DB or remote candidate calls.",
    )
    run_candidates_rag.add_argument(
        "--audit-log",
        help="Path for detailed audit log. Defaults to logs/candidate-rag-<timestamp>.log.",
    )
    run_candidates_rag.add_argument(
        "--question-id",
        type=_positive_int,
        help="Restrict execution to one explicit question id.",
    )
    run_candidates_rag.add_argument(
        "--question-sequence-start",
        type=_positive_int,
        help="Lower inclusive sequence bound for question selection.",
    )
    run_candidates_rag.add_argument(
        "--question-sequence-end",
        type=_positive_int,
        help="Upper inclusive sequence bound for question selection.",
    )
    run_candidates_rag.add_argument(
        "--prompt-id",
        type=_positive_int,
        help="Use one explicit candidate prompt version instead of the active default.",
    )
    run_candidates_rag.add_argument(
        "--retrieval-run-id",
        type=_positive_int,
        help="Validate against one explicit retrieval run id.",
    )
    run_candidates_rag.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip questions that already have a successful Com_RAG answer for the same dataset/model.",
    )
    run_candidates_rag.add_argument(
        "--no-audit-animation",
        action="store_true",
        help="Disable animated terminal dots for long-running audit steps.",
    )
    run_candidates_rag.add_argument(
        "--retry-on-context-window",
        action="store_true",
        help="Retry one candidate call after learning an observed provider context window.",
    )
    run_candidates_rag.add_argument(
        "--candidate-execution-strategy",
        choices=["sequential", "parallel"],
        help="Run candidate questions sequentially or with bounded parallel workers.",
    )
    run_candidates_rag.add_argument(
        "--candidate-parallel-max-workers",
        type=_positive_int,
        help="Maximum candidate questions to execute concurrently when strategy is parallel.",
    )
    run_candidates_rag.set_defaults(handler=run_candidates_rag_command)

    validate_provider_models = subparsers.add_parser(
        "validate-provider-models",
        help="Read-only validation of AV3 provider model ids against provider catalogs.",
    )
    validate_provider_models.add_argument(
        "--provider",
        action="append",
        choices=list(SUPPORTED_PROVIDER_VALIDATION_PROVIDERS),
        help="Restrict validation to one provider. May be repeated.",
    )
    validate_provider_models.add_argument(
        "--include-pending-confirmation",
        action="store_true",
        help="Include pending-confirmation assignments such as Gemini subtype follow-up rows.",
    )
    validate_provider_models.add_argument(
        "--include-excluded",
        action="store_true",
        help="Include excluded assignments in the read-only report.",
    )
    validate_provider_models.add_argument(
        "--json",
        action="store_true",
        help="Print the validation report as JSON.",
    )
    validate_provider_models.set_defaults(handler=validate_provider_models_command)
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
    except (ConfigurationError, RemoteJudgeError, JudgeParseError, RuntimeError, ValueError) as error:
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


def import_evaluation_details_command(args: argparse.Namespace) -> int:
    """Import auxiliary judge metadata without reprocessing evaluations."""
    from .evaluation_details_import import EvaluationDetailsImporter
    from .repositories import JudgeRepository

    settings = load_settings()
    connection = connect(settings.database_url)
    try:
        repository = JudgeRepository(connection)
        repository.ensure_schema()
        report = EvaluationDetailsImporter(repository).import_sources(
            manifest_path=args.manifest,
            raw_output_dirs=tuple(args.raw_output_dir),
        )
    finally:
        connection.close()

    print("Evaluation details import report")
    print(f"Processed: {report.processed}")
    print(f"Imported: {report.imported}")
    print(f"Skipped: {report.skipped}")
    print(f"Problems: {len(report.problems)}")
    for problem in report.problems:
        print(f"- {problem}")
    return 1 if report.problems else 0


def materialize_rag_base_command(args: argparse.Namespace) -> int:
    """Build the initial AV3 RAG base from the active imported curation."""
    from .rag_curation import resolve_rag_curation_dataset
    from .repositories import JudgeRepository

    dataset_code = resolve_rag_curation_dataset(args.dataset)
    settings = load_settings()
    connection = connect(settings.database_url)
    try:
        repository = JudgeRepository(connection)
        repository.ensure_schema()
        summary = repository.materialize_rag_base_from_active_curation(
            dataset=dataset_code,
            retrieval_name=args.retrieval_name,
            top_k=args.top_k,
            chunking_strategy=args.chunking_strategy,
        )
    finally:
        connection.close()

    print("RAG base materialized:")
    print(f"- dataset: {summary.dataset} ({summary.dataset_name})")
    print(f"- import_run_id: {summary.import_run_id}")
    print(f"- retrieval_run_id: {summary.retrieval_run_id}")
    print(f"- retrieval_name: {summary.retrieval_name}")
    print(f"- chunking_strategy: {summary.chunking_strategy}")
    print(f"- top_k: {summary.top_k}")
    print(f"- document_count: {summary.document_count}")
    print(f"- chunk_count: {summary.chunk_count}")
    print(f"- embedding_count: {summary.embedding_count}")
    print(f"- vector_extension_enabled: {summary.vector_extension_enabled}")
    return 0


def generate_rag_embeddings_command(args: argparse.Namespace) -> int:
    """Populate av3.rag_embeddings for one materialized dataset."""
    from .rag_embeddings import RagEmbeddingGenerationService

    result = RagEmbeddingGenerationService().run(
        dataset=args.dataset,
        batch_size=args.batch_size,
    )
    summary = result["summary"]

    print("RAG embeddings generated:")
    print(f"- dataset: {summary.dataset} ({summary.dataset_name})")
    print(f"- import_run_id: {summary.import_run_id}")
    print(f"- retrieval_run_id: {summary.retrieval_run_id}")
    print(f"- retrieval_name: {summary.retrieval_name}")
    print(f"- embedding_model: {summary.embedding_model}")
    print(f"- provider: {summary.provider}")
    print(f"- requested_dimensions: {summary.requested_dimensions}")
    print(f"- generated_embeddings: {summary.generated_embeddings}")
    print(f"- total_chunks: {summary.total_chunks}")
    print(f"- latency_ms: {summary.latency_ms}")
    source_summary = result.get("source_url_summary") or {}
    if source_summary:
        print(f"- source_url_references: {source_summary.get('references', 0)}")
        print(f"- source_urls_attempted: {source_summary.get('attempted', 0)}")
        print(f"- source_urls_deduplicated: {source_summary.get('deduplicated', 0)}")
        print(f"- source_urls_succeeded: {source_summary.get('succeeded', 0)}")
        print(f"- source_urls_failed: {source_summary.get('failed', 0)}")
        print(f"- source_url_chunks: {source_summary.get('inserted_chunks', 0)}")
        for failure in source_summary.get("failures", []):
            print(f"- source_url_failure: {failure.get('url')} | {failure.get('reason')}")
    return 0


def run_candidates_rag_command(args: argparse.Namespace) -> int:
    """Run or dry-run the AV3 candidate RAG generation pipeline."""
    settings = load_settings()
    request = RunCandidatesRagRequest(
        model_name=args.candidate_model,
        provider=args.provider,
        dataset=args.dataset,
        batch_size=args.batch_size,
        question_sequence_start=args.question_sequence_start,
        question_sequence_end=args.question_sequence_end,
        question_id=args.question_id,
        prompt_id=args.prompt_id,
        retrieval_run_id=args.retrieval_run_id,
        skip_existing_successful=args.skip_existing,
        dry_run=args.dry_run,
        audit_log=args.audit_log,
        no_audit_animation=args.no_audit_animation,
        remote_candidate_temperature=settings.remote_candidate_temperature,
        remote_candidate_max_tokens=settings.remote_candidate_max_tokens,
        remote_candidate_top_p=settings.remote_candidate_top_p,
        remote_candidate_context_safety_margin_tokens=settings.remote_candidate_context_safety_margin_tokens,
        remote_candidate_context_window_tokens=settings.remote_candidate_context_window_tokens,
        remote_candidate_retry_on_context_window=(
            True if args.retry_on_context_window else settings.remote_candidate_retry_on_context_window
        ),
        candidate_execution_strategy=args.candidate_execution_strategy
        or getattr(settings, "candidate_execution_strategy", "sequential"),
        candidate_parallel_max_workers=args.candidate_parallel_max_workers
        or getattr(settings, "candidate_parallel_max_workers", 2),
    )
    result = RunCandidatesRagService().run(request)
    if result.runtime_config_summary is not None:
        print(result.runtime_config_summary)
        print()
    print(result.execution_summary)
    print(f"Batch size: {result.batch_size}")
    print(f"Audit log: {result.audit_log}")
    if result.summary is not None:
        print()
        print("Execution result:")
        print(f"Candidate run id: {result.candidate_run_id}")
        print(f"Retrieval run id: {result.retrieval_run_id}")
        print(f"Prompt id: {result.prompt_id}")
        print(f"Selected questions: {result.summary.selected_questions}")
        print(f"Processed questions: {result.summary.processed_questions}")
        print(f"Successful answers: {result.summary.successful_answers}")
        print(f"Failed answers: {result.summary.failed_answers}")
        print(f"Skipped questions: {result.summary.skipped_questions}")
    return 0


def validate_provider_models_command(args: argparse.Namespace) -> int:
    """Run read-only provider-model validation for AV3 assignments."""
    from .provider_model_validation import ProviderModelValidationService
    from .repositories import JudgeRepository

    settings = load_settings()
    env_values = load_env()
    connection = connect(settings.database_url)
    try:
        repository = JudgeRepository(connection)
        report = ProviderModelValidationService(
            assignment_repository=repository,
            catalog_clients=_build_provider_catalog_clients(env_values),
        ).validate(
            providers=args.provider,
            include_pending_confirmation=args.include_pending_confirmation,
            include_excluded=args.include_excluded,
        )
    finally:
        connection.close()

    print(format_provider_model_validation_report(report, as_json=args.json))
    return provider_model_validation_exit_code(report)


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


def _build_provider_catalog_clients(env_values: Mapping[str, str]) -> dict[str, ProviderCatalogClient]:
    return {
        "openrouter": OpenRouterCatalogClient(
            base_url=env_values.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            api_key=_resolve_provider_api_key(env_values, explicit_key_name="OPENROUTER_API_KEY"),
            max_response_bytes=_parse_catalog_max_response_bytes(
                env_values,
                key="OPENROUTER_CATALOG_MAX_RESPONSE_BYTES",
                default=DEFAULT_OPENROUTER_CATALOG_MAX_RESPONSE_BYTES,
            ),
        ),
        "featherless": FeatherlessCatalogClient(
            base_url=env_values.get("FEATHERLESS_BASE_URL", "https://api.featherless.ai/v1"),
            api_key=_resolve_provider_api_key(env_values, explicit_key_name="FEATHERLESS_API_KEY"),
            max_response_bytes=_parse_catalog_max_response_bytes(
                env_values,
                key="FEATHERLESS_CATALOG_MAX_RESPONSE_BYTES",
                default=DEFAULT_FEATHERLESS_CATALOG_MAX_RESPONSE_BYTES,
            ),
        ),
    }


def _resolve_provider_api_key(
    env_values: Mapping[str, str],
    *,
    explicit_key_name: str,
) -> str | None:
    explicit_value = (env_values.get(explicit_key_name) or "").strip()
    return explicit_value or None


def _parse_catalog_max_response_bytes(
    env_values: Mapping[str, str],
    *,
    key: str,
    default: int,
) -> int:
    raw_value = (env_values.get(key) or "").strip()
    if not raw_value:
        return default
    try:
        parsed = int(raw_value)
    except ValueError as error:
        raise ConfigurationError(f"{key} must be an integer.") from error
    if parsed < 1:
        raise ConfigurationError(f"{key} must be >= 1.")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
