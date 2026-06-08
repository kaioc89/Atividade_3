"""Dashboard metrics for AV2 PostgreSQL audit data."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Any

from .config import load_settings
from .db import connect
from .repositories import DATASET_ALIASES

DATASET_LABELS = {
    "OAB_Bench": "J1",
    "OAB_Exames": "J2",
}
DEFAULT_SPEARMAN_UNAVAILABLE = "Referência humana/gabarito/rubrica indisponível para o filtro selecionado."
COMPARATIVE_EXCLUDED_JUDGES = frozenset({"openai/gpt-oss-120b"})


@dataclass(frozen=True)
class DashboardFilters:
    """Filter values accepted by the audit dashboard."""

    dataset: str = "J1"
    candidate_models: tuple[str, ...] = ()
    judge_models: tuple[str, ...] = ()
    status: str = "all"
    date_from: date | None = None
    date_to: date | None = None
    group_by: str = "modelo"


class DashboardService:
    """Read PostgreSQL evaluation data and expose dashboard-ready aggregates."""

    def __init__(
        self,
        *,
        settings_loader: Callable[[], Any] = load_settings,
        connect_func: Callable[[str], Any] = connect,
    ) -> None:
        self._settings_loader = settings_loader
        self._connect = connect_func

    def load(self, filters: DashboardFilters) -> dict[str, Any]:
        """Return filtered dashboard metrics."""
        settings = self._settings_loader()
        connection = self._connect(settings.database_url)
        try:
            with connection.cursor() as cursor:
                rows = _fetch_evaluation_rows(cursor, filters)
                expected_answers = _fetch_expected_answers(cursor, filters)
                options = _fetch_filter_options(cursor)
        finally:
            connection.close()
        return build_dashboard_payload(rows, expected_answers=expected_answers, filters=filters, options=options)


class AV3DashboardService:
    """Read PostgreSQL AV3 evaluation data and expose dashboard-ready aggregates."""

    def __init__(
        self,
        *,
        settings_loader: Callable[[], Any] = load_settings,
        connect_func: Callable[[str], Any] = connect,
    ) -> None:
        self._settings_loader = settings_loader
        self._connect = connect_func

    def load(self, filters: DashboardFilters) -> dict[str, Any]:
        """Return filtered AV3 dashboard metrics."""
        settings = self._settings_loader()
        connection = self._connect(settings.database_url)
        try:
            with connection.cursor() as cursor:
                rows = _fetch_av3_evaluation_rows(cursor, filters)
                expected_answers = _fetch_av3_expected_answers(cursor, filters)
                options = _fetch_av3_filter_options(cursor)
        finally:
            connection.close()
        return build_dashboard_payload(rows, expected_answers=expected_answers, filters=filters, options=options)


class ComparativeDashboardService:
    """Read PostgreSQL AV2/AV3 comparison data and expose paired indicators."""

    def __init__(
        self,
        *,
        settings_loader: Callable[[], Any] = load_settings,
        connect_func: Callable[[str], Any] = connect,
    ) -> None:
        self._settings_loader = settings_loader
        self._connect = connect_func

    def load(self, filters: DashboardFilters) -> dict[str, Any]:
        """Return filtered AV2 Sem_RAG vs AV3 Com_RAG indicators."""
        settings = self._settings_loader()
        connection = self._connect(settings.database_url)
        try:
            with connection.cursor() as cursor:
                rows = _fetch_comparative_rows(cursor, filters)
                agreement_rows = _fetch_comparative_agreement_rows(cursor, filters)
                options = _fetch_comparative_filter_options(cursor)
        finally:
            connection.close()
        return build_comparative_dashboard_payload(
            rows,
            filters=filters,
            options=options,
            agreement_rows=agreement_rows,
        )


def parse_dashboard_filters(values: dict[str, str | None]) -> DashboardFilters:
    """Parse query parameters into validated dashboard filters."""
    dataset = (values.get("dataset") or "J1").strip() or "J1"
    status = (values.get("status") or "all").strip() or "all"
    group_by = (values.get("group_by") or "modelo").strip() or "modelo"
    return DashboardFilters(
        dataset=dataset,
        candidate_models=_split_csv(values.get("candidate_model")),
        judge_models=_split_csv(values.get("judge_model")),
        status=status,
        date_from=_parse_date(values.get("date_from")),
        date_to=_parse_date(values.get("date_to")),
        group_by=group_by,
    )


def build_dashboard_payload(
    rows: list[dict[str, Any]],
    *,
    expected_answers: int,
    filters: DashboardFilters,
    options: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Build deterministic dashboard aggregates from SQL rows."""
    successful_rows = [row for row in rows if _is_success(row)]
    scored_rows = [row for row in successful_rows if row.get("score") is not None]
    evaluated_answers = len({row["answer_id"] for row in successful_rows})
    total_evaluations = len(rows)
    success_count = len(successful_rows)
    average_score = _average(row["score"] for row in scored_rows)
    primary_spearman = _primary_spearman(scored_rows, filters.dataset)
    consistency_spearman = _judge_arbiter_spearman(scored_rows)
    critical_cases = _critical_cases(rows)
    minor_disagreement_cases = _minor_disagreement_cases(successful_rows)
    divergence_cases = _divergence_cases(successful_rows)
    judge_agreement = _judge_agreement(successful_rows)
    ordinal_confusion = _ordinal_confusion_matrix(scored_rows, filters.dataset)
    critical_error_analysis = _critical_error_analysis(rows, divergence_cases, filters.dataset)

    cards = {
        "evaluations": total_evaluations,
        "coverage": {
            "evaluated": evaluated_answers,
            "expected": expected_answers,
            "percent": _percent(evaluated_answers, expected_answers),
        },
        "success_rate": _percent(success_count, total_evaluations),
        "average_score": average_score,
        "spearman_reference": primary_spearman,
        "judge_arbiter_consistency": consistency_spearman,
        "critical_failures": len(critical_cases),
        "minor_disagreements": len(minor_disagreement_cases),
        "audit_divergences": len(divergence_cases),
        "judge_agreement": judge_agreement["cards"],
    }
    return {
        "filters": _serialize_filters(filters),
        "options": options or {"candidate_models": [], "judge_models": []},
        "cards": cards,
        "charts": {
            "candidate_ranking": _candidate_ranking(scored_rows),
            "score_distribution": _score_distribution(scored_rows),
            "score_distribution_by_model": _score_distribution_by_model(scored_rows),
            "judge_average": _average_by(scored_rows, "judge_model"),
            "reference_alignment": _reference_alignment_points(scored_rows, filters.dataset),
            "ordinal_confusion": ordinal_confusion,
            "divergences": _divergence_chart(divergence_cases),
            "critical_cases": _critical_chart(critical_cases),
            "critical_error_categories": critical_error_analysis["categories"],
            "rubric_heatmap": _rubric_heatmap(scored_rows),
            "judge_candidate_heatmap": _judge_candidate_heatmap(scored_rows),
            "judge_disagreement_boxplot": _judge_disagreement_boxplot(successful_rows),
            "legal_specialty_performance": _legal_specialty_performance(scored_rows),
            "difficulty_performance": _difficulty_performance(scored_rows),
        },
        "tables": {
            "critical_cases": critical_cases[:25],
            "minor_disagreement_cases": minor_disagreement_cases[:25],
            "divergence_cases": divergence_cases[:25],
            "judge_agreement_arbitrations": judge_agreement["arbitrations"][:40],
            "critical_error_analysis": critical_error_analysis["cases"][:40],
        },
        "methodology": {
            "primary_spearman": (
                "Spearman principal mede nota do Juiz-IA contra referência humana/gabarito/rubrica "
                "da mesma resposta candidata. Para J2, acerto do gabarito oficial vale 5 e erro vale 1. "
                "Para J1, o cálculo só é exibido quando há referência ordinal persistida."
            ),
            "judge_arbiter": (
                "Juiz x árbitro é meta-avaliação complementar de consistência entre avaliadores, "
                "não substitui Spearman contra gabarito humano."
            ),
        },
    }


def build_comparative_dashboard_payload(
    rows: list[dict[str, Any]],
    *,
    filters: DashboardFilters,
    options: dict[str, list[str]] | None = None,
    agreement_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build deterministic AV2 Sem_RAG vs AV3 Com_RAG indicators from paired rows."""
    rows = [row for row in rows if row.get("normalized_judge_model") not in COMPARATIVE_EXCLUDED_JUDGES]
    complete_pairs = [row for row in rows if row.get("av2_score") is not None and row.get("av3_score") is not None]
    av2_only_pairs = [row for row in rows if row.get("av2_score") is not None and row.get("av3_score") is None]
    av3_only_pairs = [row for row in rows if row.get("av2_score") is None and row.get("av3_score") is not None]
    deltas = [float(row["av3_score"]) - float(row["av2_score"]) for row in complete_pairs]
    normalized_judges = sorted({str(row["normalized_judge_model"]) for row in rows if row.get("normalized_judge_model")})
    av2_rows = [
        {
            **row,
            "candidate_answer": row.get("av2_candidate_answer"),
            "reference_answer": row.get("av2_reference_answer"),
            "score": row.get("av2_score"),
        }
        for row in complete_pairs
    ]
    av3_rows = [
        {
            **row,
            "candidate_answer": row.get("av3_candidate_answer"),
            "reference_answer": row.get("av3_reference_answer"),
            "score": row.get("av3_score"),
        }
        for row in complete_pairs
    ]
    spearman_av2 = _primary_spearman(av2_rows, filters.dataset)
    spearman_av3 = _primary_spearman(av3_rows, filters.dataset)
    specialty_breakdown = _comparative_specialty_breakdown(complete_pairs)
    spearman_breakdowns = _comparative_spearman_breakdowns(complete_pairs, filters.dataset)
    agreement = _comparative_judge_agreement(agreement_rows or [])
    diagnostics = {
        "complete_pairs": len(complete_pairs),
        "av2_only_tuples": len(av2_only_pairs),
        "av3_only_tuples": len(av3_only_pairs),
        "filtered_dataset": filters.dataset,
        "filtered_candidate_models": list(filters.candidate_models),
        "filtered_judge_models": list(filters.judge_models),
        "normalized_judge_models": normalized_judges,
    }
    diagnostics_summary = (
        f"Pares completos: {diagnostics['complete_pairs']} · "
        f"Apenas AV2: {diagnostics['av2_only_tuples']} · "
        f"Apenas AV3: {diagnostics['av3_only_tuples']} · "
        f"Dataset: {diagnostics['filtered_dataset']} · "
        f"Modelos candidatos: {', '.join(diagnostics['filtered_candidate_models']) or 'todos'} · "
        f"Juizes filtrados: {', '.join(diagnostics['filtered_judge_models']) or 'todos'} · "
        f"Juizes normalizados: {', '.join(diagnostics['normalized_judge_models']) or 'nenhum'}"
    )
    return {
        "filters": _serialize_filters(filters),
        "options": options or {"datasets": [], "candidate_models": [], "judge_models": [], "statuses": ["all"]},
        "cards": {
            "comparable_pairs": len(complete_pairs),
            "comparative_coverage": {
                "evaluated": len(complete_pairs),
                "expected": len(rows),
                "percent": _percent(len(complete_pairs), len(rows)),
            },
            "av2_average_score": _average(row["av2_score"] for row in complete_pairs),
            "av3_average_score": _average(row["av3_score"] for row in complete_pairs),
            "delta_average": _average(deltas),
            "improvement_rate": _percent(sum(1 for delta in deltas if delta > 0), len(deltas)),
            "regression_rate": _percent(sum(1 for delta in deltas if delta < 0), len(deltas)),
            "unchanged_rate": _percent(sum(1 for delta in deltas if delta == 0), len(deltas)),
            "spearman_av2": spearman_av2,
            "spearman_av3": spearman_av3,
            "spearman_delta": _delta_spearman(spearman_av2, spearman_av3),
            "pairing_diagnostics": diagnostics,
            "judge_agreement_comparison": agreement["cards"],
        },
        "charts": {
            "ranking_by_model": _comparative_model_ranking(rows),
            "delta_distribution": _comparative_delta_distribution(complete_pairs),
            "delta_outcomes": _comparative_delta_outcomes(complete_pairs),
        },
        "tables": {
            "ranking_by_model": _comparative_model_ranking(rows),
            "specialties": specialty_breakdown,
            "spearman_breakdowns": spearman_breakdowns,
            "largest_gains": _comparative_delta_cases(complete_pairs, minimum_delta=2, descending=True),
            "largest_regressions": _comparative_delta_cases(complete_pairs, minimum_delta=-2, descending=False),
            "judge_agreement_by_pair": agreement["by_pair"],
            "judge_agreement_by_candidate_model": agreement["by_candidate_model"],
        },
        "methodology": {
            "pairing": (
                "Os indicadores usam pares AV2 Sem_RAG vs AV3 Com_RAG por dataset/pergunta, "
                "modelo candidato comparavel e juiz normalizado."
            ),
            "judge_normalization": (
                "Unbabel/M-Prometheus-14B e Unbabel/M-Prometheus-7B sao agrupados como M-Prometheus apenas no dashboard."
            ),
            "legal_specialty_source": (
                "Especialidades juridicas reutilizam os metadados de perguntas ja usados pelos dashboards existentes."
            ),
            "spearman_reference": (
                "Spearman comparativo usa a mesma referencia ja adotada pelo dashboard principal: gabarito oficial em J2 "
                "e nota humana/rubrica ordinal persistida em J1 quando disponivel."
            ),
            "diagnostics_summary": diagnostics_summary,
            "judge_agreement_note": agreement["note"],
            "judge_agreement_empty_state": agreement["empty_state_note"],
            "empty_state_note": (
                "Nenhum par completo restou apos os filtros."
                if not complete_pairs
                else None
            ),
        },
    }


def spearman(xs: list[float], ys: list[float]) -> dict[str, Any]:
    """Calculate Spearman rho with average ranks for ties."""
    if len(xs) != len(ys):
        raise ValueError("Spearman inputs must have the same length.")
    sample_size = len(xs)
    if sample_size < 2:
        return _spearman_unavailable(sample_size, "Amostra insuficiente para Spearman.")
    ranked_x = _rank(xs)
    ranked_y = _rank(ys)
    rho = _pearson(ranked_x, ranked_y)
    if rho is None:
        return _spearman_unavailable(sample_size, "Variância insuficiente para Spearman.")
    return {
        "value": round(rho, 4),
        "p_value": _spearman_p_value(rho, sample_size),
        "sample_size": sample_size,
        "available": True,
        "note": "Calculado com ranks médios para empates; p-value aproximado.",
    }


def _fetch_evaluation_rows(cursor: Any, filters: DashboardFilters) -> list[dict[str, Any]]:
    clauses, params = _filter_clauses(filters, include_judge=True, include_dates=True)
    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    cursor.execute(
        f"""
        SELECT
            a.id_avaliacao,
            a.id_resposta_ativa1,
            p.id_pergunta,
            d.nome_dataset,
            mc.nome_modelo AS candidate_model,
            mj.nome_modelo AS judge_model,
            COALESCE(a.papel_juiz, '') AS role,
            COALESCE(a.status_avaliacao, 'success') AS status,
            a.nota_atribuida,
            a.data_avaliacao,
            a.chain_of_thought,
            r.texto_resposta,
            p.resposta_ouro,
            COALESCE(p.metadados, '{{}}'::jsonb),
            COALESCE(a.motivo_acionamento, '')
        FROM avaliacoes_juiz a
        JOIN respostas_atividade_1 r ON r.id_resposta = a.id_resposta_ativa1
        JOIN modelos mc ON mc.id_modelo = r.id_modelo
        JOIN modelos mj ON mj.id_modelo = a.id_modelo_juiz
        JOIN perguntas p ON p.id_pergunta = r.id_pergunta
        JOIN datasets d ON d.id_dataset = p.id_dataset
        {where_sql}
        ORDER BY a.data_avaliacao DESC, a.id_avaliacao DESC;
        """,
        params,
    )
    return [
        {
            "evaluation_id": row[0],
            "answer_id": row[1],
            "question_id": row[2],
            "dataset": DATASET_LABELS.get(row[3], row[3]),
            "dataset_name": row[3],
            "candidate_model": row[4],
            "judge_model": row[5],
            "role": row[6],
            "status": row[7],
            "score": int(row[8]) if row[8] is not None else None,
            "evaluated_at": row[9].isoformat() if row[9] is not None else None,
            "rationale": row[10],
            "candidate_answer": row[11],
            "reference_answer": row[12],
            "metadata": row[13] if isinstance(row[13], dict) else {},
            "trigger_reason": row[14],
        }
        for row in cursor.fetchall()
    ]


def _fetch_expected_answers(cursor: Any, filters: DashboardFilters) -> int:
    clauses, params = _filter_clauses(filters, include_judge=False, include_dates=False)
    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    cursor.execute(
        f"""
        SELECT COUNT(DISTINCT r.id_resposta)
        FROM respostas_atividade_1 r
        JOIN modelos mc ON mc.id_modelo = r.id_modelo
        JOIN perguntas p ON p.id_pergunta = r.id_pergunta
        JOIN datasets d ON d.id_dataset = p.id_dataset
        {where_sql};
        """,
        params,
    )
    row = cursor.fetchone()
    return int(row[0] or 0)


def _fetch_filter_options(cursor: Any) -> dict[str, list[str]]:
    cursor.execute(
        """
        SELECT DISTINCT m.nome_modelo
        FROM respostas_atividade_1 r
        JOIN modelos m ON m.id_modelo = r.id_modelo
        ORDER BY m.nome_modelo;
        """
    )
    candidate_models = [row[0] for row in cursor.fetchall()]
    cursor.execute(
        """
        SELECT DISTINCT m.nome_modelo
        FROM avaliacoes_juiz a
        JOIN modelos m ON m.id_modelo = a.id_modelo_juiz
        ORDER BY m.nome_modelo;
        """
    )
    judge_models = [row[0] for row in cursor.fetchall()]
    return {"candidate_models": candidate_models, "judge_models": judge_models}


def _fetch_av3_evaluation_rows(cursor: Any, filters: DashboardFilters) -> list[dict[str, Any]]:
    clauses, params = _av3_filter_clauses(filters, include_judge=True, include_dates=True)
    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    cursor.execute(
        f"""
        SELECT
            aj.id_avaliacao,
            ca.id_candidate_answer,
            ca.id_pergunta,
            cr.dataset_code,
            COALESCE(
                av2_model.nome_modelo,
                NULLIF(ca.model_name, ''),
                NULLIF(cr.model_name, ''),
                NULLIF(cma.av3_provider_model_id, ''),
                NULLIF(cma.owner, ''),
                'sem modelo'
            ) AS candidate_model,
            mj.nome_modelo AS judge_model,
            COALESCE(aj.papel_juiz, '') AS role,
            COALESCE(aj.status_avaliacao, 'success') AS status,
            aj.nota_atribuida,
            aj.data_avaliacao,
            aj.chain_of_thought,
            COALESCE(NULLIF(ca.final_choice, ''), NULLIF(ca.answer_text, '')) AS candidate_answer,
            p.resposta_ouro,
            COALESCE(p.metadados, '{{}}'::jsonb) AS question_metadata,
            COALESCE(aj.motivo_acionamento, '') AS trigger_reason,
            ca.status AS candidate_status,
            cr.run_status,
            COALESCE(cma.owner, '') AS candidate_owner,
            COALESCE(cma.av3_provider_model_id, '') AS candidate_provider_model_id,
            cma.id_modelo_av2,
            EXISTS (
                SELECT 1
                FROM av3.candidate_answer_context_chunks cacc
                WHERE cacc.id_candidate_answer = ca.id_candidate_answer
            ) AS has_context
        FROM av3.candidate_answers ca
        JOIN av3.candidate_runs cr
          ON cr.id_candidate_run = ca.id_candidate_run
        LEFT JOIN av3.candidate_model_assignments cma
          ON cma.id_assignment = cr.id_assignment
        LEFT JOIN public.modelos av2_model
          ON av2_model.id_modelo = cma.id_modelo_av2
        JOIN public.perguntas p
          ON p.id_pergunta = ca.id_pergunta
        JOIN public.avaliacoes_juiz aj
          ON aj.id_candidate_answer = ca.id_candidate_answer
        LEFT JOIN public.modelos mj
          ON mj.id_modelo = aj.id_modelo_juiz
        {where_sql}
        ORDER BY aj.data_avaliacao DESC, aj.id_avaliacao DESC;
        """,
        params,
    )
    records: list[dict[str, Any]] = []
    for row in cursor.fetchall():
        metadata = row[13] if isinstance(row[13], dict) else {}
        metadata = {
            **metadata,
            "dataset_code": row[3],
            "candidate_status": row[15],
            "run_status": row[16],
            "candidate_owner": row[17],
            "candidate_provider_model_id": row[18],
            "candidate_av2_model_id": row[19],
            "has_rag_context": bool(row[20]),
        }
        records.append(
            {
                "evaluation_id": row[0],
                "answer_id": row[1],
                "question_id": row[2],
                "dataset": row[3],
                "dataset_name": DATASET_ALIASES.get(row[3], row[3]),
                "candidate_model": row[4],
                "judge_model": row[5],
                "role": row[6],
                "status": row[7],
                "score": int(row[8]) if row[8] is not None else None,
                "evaluated_at": row[9].isoformat() if row[9] is not None else None,
                "rationale": row[10],
                "candidate_answer": row[11],
                "reference_answer": row[12],
                "metadata": metadata,
                "trigger_reason": row[14],
            }
        )
    return records


def _fetch_av3_expected_answers(cursor: Any, filters: DashboardFilters) -> int:
    clauses, params = _av3_filter_clauses(filters, include_judge=False, include_dates=False)
    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    cursor.execute(
        f"""
        SELECT COUNT(DISTINCT ca.id_candidate_answer)
        FROM av3.candidate_answers ca
        JOIN av3.candidate_runs cr
          ON cr.id_candidate_run = ca.id_candidate_run
        LEFT JOIN av3.candidate_model_assignments cma
          ON cma.id_assignment = cr.id_assignment
        LEFT JOIN public.modelos av2_model
          ON av2_model.id_modelo = cma.id_modelo_av2
        {where_sql};
        """,
        params,
    )
    row = cursor.fetchone()
    return int(row[0] or 0)


def _fetch_av3_filter_options(cursor: Any) -> dict[str, list[str]]:
    cursor.execute(
        """
        SELECT DISTINCT
            COALESCE(
                av2_model.nome_modelo,
                NULLIF(ca.model_name, ''),
                NULLIF(cr.model_name, ''),
                NULLIF(cma.av3_provider_model_id, ''),
                NULLIF(cma.owner, ''),
                'sem modelo'
            ) AS candidate_model
        FROM av3.candidate_answers ca
        JOIN av3.candidate_runs cr
          ON cr.id_candidate_run = ca.id_candidate_run
        LEFT JOIN av3.candidate_model_assignments cma
          ON cma.id_assignment = cr.id_assignment
        LEFT JOIN public.modelos av2_model
          ON av2_model.id_modelo = cma.id_modelo_av2
        WHERE ca.status = 'success'
          AND EXISTS (
              SELECT 1
              FROM av3.candidate_answer_context_chunks cacc
              WHERE cacc.id_candidate_answer = ca.id_candidate_answer
          )
        ORDER BY candidate_model;
        """
    )
    candidate_models = [row[0] for row in cursor.fetchall()]
    cursor.execute(
        """
        SELECT DISTINCT mj.nome_modelo
        FROM public.avaliacoes_juiz aj
        JOIN av3.candidate_answers ca
          ON ca.id_candidate_answer = aj.id_candidate_answer
        JOIN av3.candidate_runs cr
          ON cr.id_candidate_run = ca.id_candidate_run
        JOIN public.modelos mj
          ON mj.id_modelo = aj.id_modelo_juiz
        WHERE EXISTS (
            SELECT 1
            FROM av3.candidate_answer_context_chunks cacc
            WHERE cacc.id_candidate_answer = ca.id_candidate_answer
        )
        ORDER BY mj.nome_modelo;
        """
    )
    judge_models = [row[0] for row in cursor.fetchall()]
    cursor.execute(
        """
        SELECT DISTINCT cr.dataset_code
        FROM av3.candidate_runs cr
        JOIN av3.candidate_answers ca
          ON ca.id_candidate_run = cr.id_candidate_run
        WHERE ca.status = 'success'
          AND EXISTS (
              SELECT 1
              FROM av3.candidate_answer_context_chunks cacc
              WHERE cacc.id_candidate_answer = ca.id_candidate_answer
          )
        ORDER BY cr.dataset_code;
        """
    )
    datasets = [row[0] for row in cursor.fetchall()]
    cursor.execute(
        """
        SELECT DISTINCT COALESCE(aj.status_avaliacao, 'success') AS status
        FROM public.avaliacoes_juiz aj
        JOIN av3.candidate_answers ca
          ON ca.id_candidate_answer = aj.id_candidate_answer
        JOIN av3.candidate_runs cr
          ON cr.id_candidate_run = ca.id_candidate_run
        WHERE EXISTS (
            SELECT 1
            FROM av3.candidate_answer_context_chunks cacc
            WHERE cacc.id_candidate_answer = ca.id_candidate_answer
        )
        ORDER BY status;
        """
    )
    raw_statuses = [str(row[0] or "success") for row in cursor.fetchall()]
    statuses = ["all"]
    if any(status == "success" for status in raw_statuses):
        statuses.append("sucesso")
    if any(status != "success" for status in raw_statuses):
        statuses.append("erro")
    return {
        "datasets": datasets,
        "candidate_models": candidate_models,
        "judge_models": judge_models,
        "statuses": statuses,
    }


def _fetch_comparative_rows(cursor: Any, filters: DashboardFilters) -> list[dict[str, Any]]:
    av2_clauses, av2_params = _comparative_av2_filter_clauses(filters)
    av3_clauses, av3_params = _comparative_av3_filter_clauses(filters)
    av2_where_sql = f"WHERE {' AND '.join(av2_clauses)}" if av2_clauses else ""
    av3_where_sql = f"WHERE {' AND '.join(av3_clauses)}" if av3_clauses else ""
    cursor.execute(
        f"""
        WITH av2_ranked AS (
            SELECT
                {_comparative_av2_dataset_sql('d.nome_dataset')} AS dataset_code,
                p.id_pergunta AS question_id,
                r.id_modelo AS comparable_candidate_model_id,
                mc.nome_modelo AS av2_model_name,
                mc.nome_modelo AS comparable_candidate_model,
                mj.nome_modelo AS raw_judge_model,
                {_normalized_judge_sql('mj.nome_modelo')} AS normalized_judge_model,
                COALESCE(a.status_avaliacao, 'success') AS status,
                a.nota_atribuida AS score,
                r.texto_resposta AS candidate_answer,
                p.resposta_ouro AS reference_answer,
                COALESCE(p.metadados, '{{}}'::jsonb) AS question_metadata,
                NULLIF(p.metadados ->> 'question_number', '') AS question_sequence,
                ROW_NUMBER() OVER (
                        PARTITION BY
                        {_comparative_av2_dataset_sql('d.nome_dataset')},
                        p.id_pergunta,
                        r.id_modelo,
                        {_normalized_judge_sql('mj.nome_modelo')}
                    ORDER BY a.data_avaliacao DESC NULLS LAST, a.id_avaliacao DESC
                ) AS row_number
            FROM public.avaliacoes_juiz a
            JOIN public.respostas_atividade_1 r
              ON r.id_resposta = a.id_resposta_ativa1
            JOIN public.modelos mc
              ON mc.id_modelo = r.id_modelo
            JOIN public.modelos mj
              ON mj.id_modelo = a.id_modelo_juiz
            JOIN public.perguntas p
              ON p.id_pergunta = r.id_pergunta
            JOIN public.datasets d
              ON d.id_dataset = p.id_dataset
            {av2_where_sql}
        ),
        av2_scores AS (
            SELECT *
            FROM av2_ranked
            WHERE row_number = 1
        ),
        av3_ranked AS (
            SELECT
                cr.dataset_code AS dataset_code,
                ca.id_pergunta AS question_id,
                {_comparative_av3_candidate_model_id_sql()} AS comparable_candidate_model_id,
                COALESCE(av2_model_direct.nome_modelo, assignment_legacy.av2_model_name) AS av2_model_name,
                {_comparative_av3_candidate_model_name_sql()} AS comparable_candidate_model,
                COALESCE(
                    NULLIF(assignment_direct.owner, ''),
                    NULLIF(assignment_legacy.owner, '')
                ) AS candidate_owner,
                COALESCE(
                    NULLIF(assignment_direct.av3_provider_model_id, ''),
                    NULLIF(assignment_legacy.av3_provider_model_id, ''),
                    NULLIF(ca.model_name, ''),
                    NULLIF(cr.model_name, '')
                ) AS av3_provider_model_id,
                mj.nome_modelo AS raw_judge_model,
                {_normalized_judge_sql('mj.nome_modelo')} AS normalized_judge_model,
                COALESCE(aj.status_avaliacao, 'success') AS status,
                aj.nota_atribuida AS score,
                COALESCE(NULLIF(ca.final_choice, ''), NULLIF(ca.answer_text, '')) AS candidate_answer,
                p.resposta_ouro AS reference_answer,
                COALESCE(p.metadados, '{{}}'::jsonb) AS question_metadata,
                COALESCE(cq.question_sequence::text, NULLIF(p.metadados ->> 'question_number', '')) AS question_sequence,
                ROW_NUMBER() OVER (
                    PARTITION BY
                        cr.dataset_code,
                        ca.id_pergunta,
                        {_comparative_av3_candidate_model_id_sql()},
                        {_normalized_judge_sql('mj.nome_modelo')}
                    ORDER BY aj.data_avaliacao DESC NULLS LAST, aj.id_avaliacao DESC
                ) AS row_number
            FROM av3.candidate_answers ca
            JOIN av3.candidate_runs cr
              ON cr.id_candidate_run = ca.id_candidate_run
            JOIN public.perguntas p
              ON p.id_pergunta = ca.id_pergunta
            LEFT JOIN av3.retrieval_runs rr
              ON rr.id_retrieval_run = cr.id_retrieval_run
            LEFT JOIN av3.curadoria_questoes cq
              ON cq.id_import_run = rr.id_import_run
             AND cq.dataset_code = cr.dataset_code
             AND cq.id_pergunta = ca.id_pergunta
            LEFT JOIN av3.candidate_model_assignments assignment_direct
              ON assignment_direct.id_assignment = cr.id_assignment
            LEFT JOIN public.modelos av2_model_direct
              ON av2_model_direct.id_modelo = assignment_direct.id_modelo_av2
            LEFT JOIN LATERAL (
                SELECT
                    assignment.id_assignment,
                    assignment.id_modelo_av2,
                    assignment.owner,
                    av2_model.nome_modelo AS av2_model_name,
                    assignment.original_provider_model_id,
                    assignment.av3_provider_model_id
                FROM av3.candidate_model_assignments assignment
                JOIN av3.candidate_model_assignment_ranges assignment_range
                  ON assignment_range.id_assignment = assignment.id_assignment
                LEFT JOIN public.modelos av2_model
                  ON av2_model.id_modelo = assignment.id_modelo_av2
                WHERE assignment.active
                  AND cr.id_assignment IS NULL
                  AND assignment.av3_provider_model_id = ca.model_name
                  AND assignment_range.dataset_code = cr.dataset_code
                  AND cq.question_sequence BETWEEN
                      assignment_range.question_sequence_start
                      AND assignment_range.question_sequence_end
                ORDER BY assignment.updated_at DESC, assignment.id_assignment DESC
                LIMIT 1
            ) assignment_legacy ON TRUE
            JOIN public.avaliacoes_juiz aj
              ON aj.id_candidate_answer = ca.id_candidate_answer
            JOIN public.modelos mj
              ON mj.id_modelo = aj.id_modelo_juiz
            {av3_where_sql}
        ),
        av3_scores AS (
            SELECT *
            FROM av3_ranked
            WHERE row_number = 1
        )
        SELECT
            COALESCE(av2.dataset_code, av3.dataset_code) AS dataset_code,
            COALESCE(av2.question_id, av3.question_id) AS question_id,
            COALESCE(av2.comparable_candidate_model_id, av3.comparable_candidate_model_id) AS comparable_candidate_model_id,
            COALESCE(av2.av2_model_name, av3.av2_model_name, av2.comparable_candidate_model, av3.comparable_candidate_model) AS av2_model_name,
            COALESCE(av2.comparable_candidate_model, av3.comparable_candidate_model) AS comparable_candidate_model,
            av3.candidate_owner AS candidate_owner,
            av3.av3_provider_model_id AS av3_provider_model_id,
            COALESCE(av2.normalized_judge_model, av3.normalized_judge_model) AS normalized_judge_model,
            av2.raw_judge_model AS av2_raw_judge_model,
            av3.raw_judge_model AS av3_raw_judge_model,
            av2.status AS av2_status,
            av3.status AS av3_status,
            av2.score AS av2_score,
            av3.score AS av3_score,
            av2.candidate_answer AS av2_candidate_answer,
            av3.candidate_answer AS av3_candidate_answer,
            av2.reference_answer AS av2_reference_answer,
            av3.reference_answer AS av3_reference_answer,
            COALESCE(av2.question_metadata, av3.question_metadata, '{{}}'::jsonb) AS question_metadata,
            COALESCE(av2.question_sequence, av3.question_sequence) AS question_sequence
        FROM av2_scores av2
        FULL OUTER JOIN av3_scores av3
          ON av3.dataset_code = av2.dataset_code
         AND av3.question_id = av2.question_id
         AND av3.comparable_candidate_model_id = av2.comparable_candidate_model_id
         AND av3.normalized_judge_model = av2.normalized_judge_model
        ORDER BY 1, 2, 4, 5;
        """,
        [*av2_params, *av3_params],
    )
    return [
        {
            "dataset": row[0],
            "question_id": row[1],
            "comparable_candidate_model_id": row[2],
            "av2_model_name": row[3],
            "comparable_candidate_model": row[4],
            "candidate_owner": row[5],
            "av3_provider_model_id": row[6],
            "normalized_judge_model": row[7],
            "av2_raw_judge_model": row[8],
            "av3_raw_judge_model": row[9],
            "av2_status": row[10],
            "av3_status": row[11],
            "av2_score": int(row[12]) if row[12] is not None else None,
            "av3_score": int(row[13]) if row[13] is not None else None,
            "av2_candidate_answer": row[14],
            "av3_candidate_answer": row[15],
            "av2_reference_answer": row[16],
            "av3_reference_answer": row[17],
            "metadata": row[18] if isinstance(row[18], dict) else {},
            "question_sequence": _safe_int(row[19]),
        }
        for row in cursor.fetchall()
    ]


def _fetch_comparative_agreement_rows(cursor: Any, filters: DashboardFilters) -> list[dict[str, Any]]:
    av2_clauses, av2_params = _comparative_av2_filter_clauses(filters, include_arbiter=True)
    av3_clauses, av3_params = _comparative_av3_filter_clauses(filters, include_arbiter=True)
    av2_where_sql = f"WHERE {' AND '.join(av2_clauses)}" if av2_clauses else ""
    av3_where_sql = f"WHERE {' AND '.join(av3_clauses)}" if av3_clauses else ""
    cursor.execute(
        f"""
        WITH av2_ranked AS (
            SELECT
                {_comparative_av2_dataset_sql('d.nome_dataset')} AS dataset_code,
                p.id_pergunta AS question_id,
                r.id_modelo AS comparable_candidate_model_id,
                mc.nome_modelo AS comparable_candidate_model,
                COALESCE(a.papel_juiz, '') AS role,
                CASE WHEN COALESCE(a.papel_juiz, '') = 'arbitro' THEN 'arbiter' ELSE 'primary' END AS judge_bucket,
                {_normalized_judge_sql('mj.nome_modelo')} AS normalized_judge_model,
                COALESCE(a.status_avaliacao, 'success') AS status,
                a.nota_atribuida AS score,
                NULLIF(p.metadados ->> 'question_number', '') AS question_sequence,
                ROW_NUMBER() OVER (
                    PARTITION BY
                        {_comparative_av2_dataset_sql('d.nome_dataset')},
                        p.id_pergunta,
                        r.id_modelo,
                        {_normalized_judge_sql('mj.nome_modelo')},
                        CASE WHEN COALESCE(a.papel_juiz, '') = 'arbitro' THEN 'arbiter' ELSE 'primary' END
                    ORDER BY a.data_avaliacao DESC NULLS LAST, a.id_avaliacao DESC
                ) AS row_number
            FROM public.avaliacoes_juiz a
            JOIN public.respostas_atividade_1 r
              ON r.id_resposta = a.id_resposta_ativa1
            JOIN public.modelos mc
              ON mc.id_modelo = r.id_modelo
            JOIN public.modelos mj
              ON mj.id_modelo = a.id_modelo_juiz
            JOIN public.perguntas p
              ON p.id_pergunta = r.id_pergunta
            JOIN public.datasets d
              ON d.id_dataset = p.id_dataset
            {av2_where_sql}
        ),
        av2_scores AS (
            SELECT *
            FROM av2_ranked
            WHERE row_number = 1
        ),
        av3_ranked AS (
            SELECT
                cr.dataset_code AS dataset_code,
                ca.id_pergunta AS question_id,
                {_comparative_av3_candidate_model_id_sql()} AS comparable_candidate_model_id,
                {_comparative_av3_candidate_model_name_sql()} AS comparable_candidate_model,
                COALESCE(aj.papel_juiz, '') AS role,
                CASE WHEN COALESCE(aj.papel_juiz, '') = 'arbitro' THEN 'arbiter' ELSE 'primary' END AS judge_bucket,
                {_normalized_judge_sql('mj.nome_modelo')} AS normalized_judge_model,
                COALESCE(aj.status_avaliacao, 'success') AS status,
                aj.nota_atribuida AS score,
                COALESCE(cq.question_sequence::text, NULLIF(p.metadados ->> 'question_number', '')) AS question_sequence,
                ROW_NUMBER() OVER (
                    PARTITION BY
                        cr.dataset_code,
                        ca.id_pergunta,
                        {_comparative_av3_candidate_model_id_sql()},
                        {_normalized_judge_sql('mj.nome_modelo')},
                        CASE WHEN COALESCE(aj.papel_juiz, '') = 'arbitro' THEN 'arbiter' ELSE 'primary' END
                    ORDER BY aj.data_avaliacao DESC NULLS LAST, aj.id_avaliacao DESC
                ) AS row_number
            FROM av3.candidate_answers ca
            JOIN av3.candidate_runs cr
              ON cr.id_candidate_run = ca.id_candidate_run
            JOIN public.perguntas p
              ON p.id_pergunta = ca.id_pergunta
            LEFT JOIN av3.retrieval_runs rr
              ON rr.id_retrieval_run = cr.id_retrieval_run
            LEFT JOIN av3.curadoria_questoes cq
              ON cq.id_import_run = rr.id_import_run
             AND cq.dataset_code = cr.dataset_code
             AND cq.id_pergunta = ca.id_pergunta
            LEFT JOIN av3.candidate_model_assignments assignment_direct
              ON assignment_direct.id_assignment = cr.id_assignment
            LEFT JOIN public.modelos av2_model_direct
              ON av2_model_direct.id_modelo = assignment_direct.id_modelo_av2
            LEFT JOIN LATERAL (
                SELECT
                    assignment.id_assignment,
                    assignment.id_modelo_av2,
                    av2_model.nome_modelo AS av2_model_name,
                    assignment.av3_provider_model_id
                FROM av3.candidate_model_assignments assignment
                JOIN av3.candidate_model_assignment_ranges assignment_range
                  ON assignment_range.id_assignment = assignment.id_assignment
                LEFT JOIN public.modelos av2_model
                  ON av2_model.id_modelo = assignment.id_modelo_av2
                WHERE assignment.active
                  AND cr.id_assignment IS NULL
                  AND assignment.av3_provider_model_id = ca.model_name
                  AND assignment_range.dataset_code = cr.dataset_code
                  AND cq.question_sequence BETWEEN
                      assignment_range.question_sequence_start
                      AND assignment_range.question_sequence_end
                ORDER BY assignment.updated_at DESC, assignment.id_assignment DESC
                LIMIT 1
            ) assignment_legacy ON TRUE
            JOIN public.avaliacoes_juiz aj
              ON aj.id_candidate_answer = ca.id_candidate_answer
            JOIN public.modelos mj
              ON mj.id_modelo = aj.id_modelo_juiz
            {av3_where_sql}
        ),
        av3_scores AS (
            SELECT *
            FROM av3_ranked
            WHERE row_number = 1
        ),
        paired_answer_keys AS (
            SELECT DISTINCT
                av2.dataset_code,
                av2.question_id,
                av2.comparable_candidate_model_id,
                av2.comparable_candidate_model
            FROM av2_scores av2
            INNER JOIN av3_scores av3
              ON av3.dataset_code = av2.dataset_code
             AND av3.question_id = av2.question_id
             AND av3.comparable_candidate_model_id = av2.comparable_candidate_model_id
        )
        SELECT
            'AV2' AS source,
            av2.dataset_code,
            av2.question_id,
            av2.comparable_candidate_model_id,
            av2.comparable_candidate_model,
            av2.role,
            av2.judge_bucket,
            av2.normalized_judge_model,
            av2.status,
            av2.score,
            av2.question_sequence
        FROM av2_scores av2
        JOIN paired_answer_keys paired
          ON paired.dataset_code = av2.dataset_code
         AND paired.question_id = av2.question_id
         AND paired.comparable_candidate_model_id = av2.comparable_candidate_model_id
        UNION ALL
        SELECT
            'AV3' AS source,
            av3.dataset_code,
            av3.question_id,
            av3.comparable_candidate_model_id,
            av3.comparable_candidate_model,
            av3.role,
            av3.judge_bucket,
            av3.normalized_judge_model,
            av3.status,
            av3.score,
            av3.question_sequence
        FROM av3_scores av3
        JOIN paired_answer_keys paired
          ON paired.dataset_code = av3.dataset_code
         AND paired.question_id = av3.question_id
         AND paired.comparable_candidate_model_id = av3.comparable_candidate_model_id
        ORDER BY 2, 3, 5, 1, 7, 8;
        """,
        [*av2_params, *av3_params],
    )
    return [
        {
            "source": row[0],
            "dataset": row[1],
            "question_id": row[2],
            "comparable_candidate_model_id": row[3],
            "comparable_candidate_model": row[4],
            "role": row[5],
            "judge_bucket": row[6],
            "normalized_judge_model": row[7],
            "status": row[8],
            "score": int(row[9]) if row[9] is not None else None,
            "question_sequence": _safe_int(row[10]),
        }
        for row in cursor.fetchall()
    ]


def _fetch_comparative_filter_options(cursor: Any) -> dict[str, list[str]]:
    cursor.execute(
        f"""
        WITH av2_source_datasets AS (
            SELECT DISTINCT
                {_comparative_av2_dataset_sql('d.nome_dataset')} AS dataset_code
            FROM public.respostas_atividade_1 r
            JOIN public.perguntas p
              ON p.id_pergunta = r.id_pergunta
            JOIN public.datasets d
              ON d.id_dataset = p.id_dataset
        ),
        av2_evaluation_datasets AS (
            SELECT DISTINCT
                {_comparative_av2_dataset_sql('d.nome_dataset')} AS dataset_code
            FROM public.avaliacoes_juiz a
            JOIN public.respostas_atividade_1 r
              ON r.id_resposta = a.id_resposta_ativa1
            JOIN public.perguntas p
              ON p.id_pergunta = r.id_pergunta
            JOIN public.datasets d
              ON d.id_dataset = p.id_dataset
            WHERE a.id_resposta_ativa1 IS NOT NULL
        ),
        av3_run_datasets AS (
            SELECT DISTINCT cr.dataset_code
            FROM av3.candidate_runs cr
            WHERE COALESCE(cr.dataset_code, '') <> ''
        ),
        av3_evaluation_datasets AS (
            SELECT DISTINCT cr.dataset_code
            FROM public.avaliacoes_juiz aj
            JOIN av3.candidate_answers ca
              ON ca.id_candidate_answer = aj.id_candidate_answer
            JOIN av3.candidate_runs cr
              ON cr.id_candidate_run = ca.id_candidate_run
            WHERE aj.id_candidate_answer IS NOT NULL
              AND COALESCE(cr.dataset_code, '') <> ''
        ),
        comparative_datasets AS (
            SELECT dataset_code FROM av2_source_datasets
            UNION
            SELECT dataset_code FROM av2_evaluation_datasets
            UNION
            SELECT dataset_code FROM av3_run_datasets
            UNION
            SELECT dataset_code FROM av3_evaluation_datasets
        ),
        av2_candidates AS (
            SELECT DISTINCT
                {_comparative_av2_dataset_sql('d.nome_dataset')} AS dataset_code,
                p.id_pergunta AS question_id,
                r.id_modelo AS comparable_candidate_model_id,
                mc.nome_modelo AS comparable_candidate_model
            FROM public.respostas_atividade_1 r
            JOIN public.modelos mc
              ON mc.id_modelo = r.id_modelo
            JOIN public.perguntas p
              ON p.id_pergunta = r.id_pergunta
            JOIN public.datasets d
              ON d.id_dataset = p.id_dataset
        ),
        av3_candidates AS (
            SELECT DISTINCT
                cr.dataset_code AS dataset_code,
                ca.id_pergunta AS question_id,
                {_comparative_av3_candidate_model_id_sql()} AS comparable_candidate_model_id,
                {_comparative_av3_candidate_model_name_sql()} AS comparable_candidate_model
            FROM av3.candidate_answers ca
            JOIN av3.candidate_runs cr
              ON cr.id_candidate_run = ca.id_candidate_run
            LEFT JOIN av3.retrieval_runs rr
              ON rr.id_retrieval_run = cr.id_retrieval_run
            LEFT JOIN av3.curadoria_questoes cq
              ON cq.id_import_run = rr.id_import_run
             AND cq.dataset_code = cr.dataset_code
             AND cq.id_pergunta = ca.id_pergunta
            LEFT JOIN av3.candidate_model_assignments assignment_direct
              ON assignment_direct.id_assignment = cr.id_assignment
            LEFT JOIN public.modelos av2_model_direct
              ON av2_model_direct.id_modelo = assignment_direct.id_modelo_av2
            LEFT JOIN LATERAL (
                SELECT
                    assignment.id_assignment,
                    assignment.id_modelo_av2,
                    assignment.owner,
                    av2_model.nome_modelo AS av2_model_name,
                    assignment.original_provider_model_id,
                    assignment.av3_provider_model_id
                FROM av3.candidate_model_assignments assignment
                JOIN av3.candidate_model_assignment_ranges assignment_range
                  ON assignment_range.id_assignment = assignment.id_assignment
                LEFT JOIN public.modelos av2_model
                  ON av2_model.id_modelo = assignment.id_modelo_av2
                WHERE assignment.active
                  AND cr.id_assignment IS NULL
                  AND assignment.av3_provider_model_id = ca.model_name
                  AND assignment_range.dataset_code = cr.dataset_code
                  AND cq.question_sequence BETWEEN
                      assignment_range.question_sequence_start
                      AND assignment_range.question_sequence_end
                ORDER BY assignment.updated_at DESC, assignment.id_assignment DESC
                LIMIT 1
            ) assignment_legacy ON TRUE
            WHERE {_comparative_av3_candidate_model_id_sql()} IS NOT NULL
              AND ca.status = 'success'
              AND EXISTS (
                  SELECT 1
                  FROM av3.candidate_answer_context_chunks cacc
                  WHERE cacc.id_candidate_answer = ca.id_candidate_answer
              )
        ),
        paired_candidates AS (
            SELECT DISTINCT
                av2.dataset_code,
                av2.comparable_candidate_model
            FROM av2_candidates av2
            JOIN av3_candidates av3
              ON av3.dataset_code = av2.dataset_code
             AND av3.question_id = av2.question_id
             AND av3.comparable_candidate_model_id = av2.comparable_candidate_model_id
        ),
        av2_judges AS (
            SELECT DISTINCT {_normalized_judge_sql('mj.nome_modelo')} AS normalized_judge_model
            FROM public.avaliacoes_juiz a
            JOIN public.modelos mj
              ON mj.id_modelo = a.id_modelo_juiz
            WHERE a.id_resposta_ativa1 IS NOT NULL
              AND COALESCE(a.papel_juiz, '') <> 'arbitro'
        ),
        av3_judges AS (
            SELECT DISTINCT {_normalized_judge_sql('mj.nome_modelo')} AS normalized_judge_model
            FROM public.avaliacoes_juiz aj
            JOIN av3.candidate_answers ca
              ON ca.id_candidate_answer = aj.id_candidate_answer
            JOIN public.modelos mj
              ON mj.id_modelo = aj.id_modelo_juiz
            WHERE ca.status = 'success'
              AND COALESCE(aj.papel_juiz, '') <> 'arbitro'
              AND EXISTS (
                  SELECT 1
                  FROM av3.candidate_answer_context_chunks cacc
                  WHERE cacc.id_candidate_answer = ca.id_candidate_answer
              )
        )
        SELECT
            ARRAY(SELECT dataset_code FROM comparative_datasets WHERE dataset_code IS NOT NULL ORDER BY dataset_code),
            ARRAY(SELECT DISTINCT comparable_candidate_model FROM paired_candidates ORDER BY comparable_candidate_model),
            ARRAY(
                SELECT av2_judges.normalized_judge_model
                FROM av2_judges
                INNER JOIN av3_judges
                  ON av3_judges.normalized_judge_model = av2_judges.normalized_judge_model
                ORDER BY av2_judges.normalized_judge_model
            );
        """
    )
    row = cursor.fetchone() or ([], [], [])
    return {
        "datasets": list(row[0] or []),
        "candidate_models": list(row[1] or []),
        "judge_models": list(row[2] or []),
        "statuses": ["all", "sucesso", "erro"],
    }


def _filter_clauses(
    filters: DashboardFilters,
    *,
    include_judge: bool,
    include_dates: bool,
) -> tuple[list[str], list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    dataset = filters.dataset.strip()
    if dataset.lower() != "all":
        clauses.append("d.nome_dataset = %s")
        params.append(DATASET_ALIASES.get(dataset.upper(), dataset))
    if filters.candidate_models:
        clauses.append("mc.nome_modelo = ANY(%s)")
        params.append(list(filters.candidate_models))
    if include_judge and filters.judge_models:
        clauses.append("mj.nome_modelo = ANY(%s)")
        params.append(list(filters.judge_models))
    if include_judge and filters.status != "all":
        if filters.status == "erro":
            clauses.append("COALESCE(a.status_avaliacao, 'success') <> 'success'")
        elif filters.status == "sucesso":
            clauses.append("COALESCE(a.status_avaliacao, 'success') = 'success'")
        else:
            clauses.append("COALESCE(a.status_avaliacao, 'success') = %s")
            params.append(filters.status)
    if include_dates and filters.date_from is not None:
        clauses.append("a.data_avaliacao::date >= %s")
        params.append(filters.date_from)
    if include_dates and filters.date_to is not None:
        clauses.append("a.data_avaliacao::date <= %s")
        params.append(filters.date_to)
    return clauses, params


def _av3_filter_clauses(
    filters: DashboardFilters,
    *,
    include_judge: bool,
    include_dates: bool,
) -> tuple[list[str], list[Any]]:
    clauses = [
        "ca.status = 'success'",
        "EXISTS (SELECT 1 FROM av3.candidate_answer_context_chunks cacc WHERE cacc.id_candidate_answer = ca.id_candidate_answer)",
    ]
    params: list[Any] = []
    dataset = filters.dataset.strip()
    if dataset.lower() != "all":
        clauses.append("cr.dataset_code = %s")
        params.append(dataset.upper())
    if filters.candidate_models:
        clauses.append(
            """
            COALESCE(
                av2_model.nome_modelo,
                NULLIF(ca.model_name, ''),
                NULLIF(cr.model_name, ''),
                NULLIF(cma.av3_provider_model_id, ''),
                NULLIF(cma.owner, ''),
                'sem modelo'
            ) = ANY(%s)
            """.strip()
        )
        params.append(list(filters.candidate_models))
    if include_judge and filters.judge_models:
        clauses.append("mj.nome_modelo = ANY(%s)")
        params.append(list(filters.judge_models))
    if include_judge and filters.status != "all":
        if filters.status == "erro":
            clauses.append("COALESCE(aj.status_avaliacao, 'success') <> 'success'")
        elif filters.status == "sucesso":
            clauses.append("COALESCE(aj.status_avaliacao, 'success') = 'success'")
        else:
            clauses.append("COALESCE(aj.status_avaliacao, 'success') = %s")
            params.append(filters.status)
    if include_dates and filters.date_from is not None:
        clauses.append("aj.data_avaliacao::date >= %s")
        params.append(filters.date_from)
    if include_dates and filters.date_to is not None:
        clauses.append("aj.data_avaliacao::date <= %s")
        params.append(filters.date_to)
    return clauses, params


def _comparative_av2_filter_clauses(
    filters: DashboardFilters,
    *,
    include_arbiter: bool = False,
) -> tuple[list[str], list[Any]]:
    clauses: list[str] = []
    if not include_arbiter:
        clauses.append("COALESCE(a.papel_juiz, '') <> 'arbitro'")
    params: list[Any] = []
    dataset = filters.dataset.strip()
    if dataset.lower() != "all":
        clauses.append(f"{_comparative_av2_dataset_sql('d.nome_dataset')} = %s")
        params.append(dataset.upper())
    if filters.candidate_models:
        clauses.append("mc.nome_modelo = ANY(%s)")
        params.append(list(filters.candidate_models))
    if filters.judge_models:
        clauses.append(f"{_normalized_judge_sql('mj.nome_modelo')} = ANY(%s)")
        params.append(list(filters.judge_models))
    if filters.status != "all":
        if filters.status == "erro":
            clauses.append("COALESCE(a.status_avaliacao, 'success') <> 'success'")
        elif filters.status == "sucesso":
            clauses.append("COALESCE(a.status_avaliacao, 'success') = 'success'")
        else:
            clauses.append("COALESCE(a.status_avaliacao, 'success') = %s")
            params.append(filters.status)
    if filters.date_from is not None:
        clauses.append("a.data_avaliacao::date >= %s")
        params.append(filters.date_from)
    if filters.date_to is not None:
        clauses.append("a.data_avaliacao::date <= %s")
        params.append(filters.date_to)
    return clauses, params


def _comparative_av3_filter_clauses(
    filters: DashboardFilters,
    *,
    include_arbiter: bool = False,
) -> tuple[list[str], list[Any]]:
    clauses = [
        f"{_comparative_av3_candidate_model_id_sql()} IS NOT NULL",
        "ca.status = 'success'",
        "EXISTS (SELECT 1 FROM av3.candidate_answer_context_chunks cacc WHERE cacc.id_candidate_answer = ca.id_candidate_answer)",
    ]
    if not include_arbiter:
        clauses.insert(0, "COALESCE(aj.papel_juiz, '') <> 'arbitro'")
    params: list[Any] = []
    dataset = filters.dataset.strip()
    if dataset.lower() != "all":
        clauses.append("cr.dataset_code = %s")
        params.append(dataset.upper())
    if filters.candidate_models:
        clauses.append(f"{_comparative_av3_candidate_model_name_sql()} = ANY(%s)")
        params.append(list(filters.candidate_models))
    if filters.judge_models:
        clauses.append(f"{_normalized_judge_sql('mj.nome_modelo')} = ANY(%s)")
        params.append(list(filters.judge_models))
    if filters.status != "all":
        if filters.status == "erro":
            clauses.append("COALESCE(aj.status_avaliacao, 'success') <> 'success'")
        elif filters.status == "sucesso":
            clauses.append("COALESCE(aj.status_avaliacao, 'success') = 'success'")
        else:
            clauses.append("COALESCE(aj.status_avaliacao, 'success') = %s")
            params.append(filters.status)
    if filters.date_from is not None:
        clauses.append("aj.data_avaliacao::date >= %s")
        params.append(filters.date_from)
    if filters.date_to is not None:
        clauses.append("aj.data_avaliacao::date <= %s")
        params.append(filters.date_to)
    return clauses, params


def _primary_spearman(rows: list[dict[str, Any]], selected_dataset: str) -> dict[str, Any]:
    datasets = {row["dataset"] for row in rows}
    if selected_dataset.upper() == "J2" or datasets == {"J2"}:
        pairs = [
            (_j2_reference_score(row), row["score"])
            for row in rows
            if row["dataset"] == "J2" and _j2_reference_score(row) is not None
        ]
        if not pairs:
            return _spearman_unavailable(0, "Sem pares J2 com gabarito oficial e nota do juiz.")
        return spearman([float(pair[0]) for pair in pairs], [float(pair[1]) for pair in pairs])
    if selected_dataset.upper() == "J1" or datasets == {"J1"}:
        pairs = [
            (_j1_reference_score(row), row["score"])
            for row in rows
            if row["dataset"] == "J1" and _j1_reference_score(row) is not None
        ]
        if pairs:
            return spearman([float(pair[0]) for pair in pairs], [float(pair[1]) for pair in pairs])
        return _spearman_unavailable(
            0,
            "J1 não possui nota humana/rubrica ordinal persistida para calcular Spearman principal.",
        )
    return _spearman_unavailable(0, "Selecione J1 ou J2 para Spearman principal sem misturar tarefas.")


def _reference_alignment_points(rows: list[dict[str, Any]], selected_dataset: str) -> dict[str, Any]:
    points = []
    for row in rows:
        reference_score = _reference_score(row, selected_dataset)
        judge_score = row.get("score")
        if reference_score is None or judge_score is None:
            continue
        points.append(
            {
                "evaluation_id": row["evaluation_id"],
                "answer_id": row["answer_id"],
                "question_id": row["question_id"],
                "dataset": row["dataset"],
                "candidate_model": row["candidate_model"],
                "judge_model": row["judge_model"],
                "reference_score": round(float(reference_score), 4),
                "judge_score": int(judge_score),
            }
        )
    return {
        "points": points,
        "x_label": "nota humana / score derivado do gabarito",
        "y_label": "nota do juiz",
    }


def _ordinal_confusion_matrix(rows: list[dict[str, Any]], selected_dataset: str) -> dict[str, Any]:
    labels = [1, 2, 3, 4, 5]
    matrix = [[0 for _ in labels] for _ in labels]
    total = 0
    severe_false_positives = 0
    false_negatives = 0
    judge_score_counts = {score: 0 for score in labels}
    important_cases: list[dict[str, Any]] = []

    for row in rows:
        reference_score = _ordinal_score(_reference_score(row, selected_dataset))
        judge_score = _ordinal_score(row.get("score"))
        if reference_score is None or judge_score is None:
            continue
        matrix[reference_score - 1][judge_score - 1] += 1
        judge_score_counts[judge_score] += 1
        total += 1
        delta = judge_score - reference_score
        if reference_score <= 2 and judge_score >= 4:
            severe_false_positives += 1
            important_cases.append(
                _confusion_case(row, reference_score, judge_score, "falso positivo grave", delta)
            )
        elif reference_score >= 4 and judge_score <= 2:
            false_negatives += 1
            important_cases.append(_confusion_case(row, reference_score, judge_score, "falso negativo", delta))

    lenient_total = judge_score_counts[4] + judge_score_counts[5]
    conservative_total = judge_score_counts[2] + judge_score_counts[3]
    lenient_share = _percent(lenient_total, total)
    conservative_share = _percent(conservative_total, total)
    highlights = [
        {
            "label": "Humano baixo, juiz alto",
            "interpretation": "falso positivo grave",
            "count": severe_false_positives,
            "share": _percent(severe_false_positives, total),
        },
        {
            "label": "Humano alto, juiz baixo",
            "interpretation": "falso negativo",
            "count": false_negatives,
            "share": _percent(false_negatives, total),
        },
        {
            "label": "Juiz nota 4/5",
            "interpretation": "juiz leniente" if lenient_share is not None and lenient_share >= 60 else "tendencia a notas altas",
            "count": lenient_total,
            "share": lenient_share,
        },
        {
            "label": "Juiz nota 2/3",
            "interpretation": (
                "juiz conservador demais"
                if conservative_share is not None and conservative_share >= 60
                else "tendencia a notas intermediarias/baixas"
            ),
            "count": conservative_total,
            "share": conservative_share,
        },
    ]
    return {
        "rows": [f"Humano {score}" for score in labels],
        "columns": [f"Juiz {score}" for score in labels],
        "matrix": matrix,
        "total": total,
        "highlights": highlights,
        "important_cases": sorted(important_cases, key=lambda case: (-abs(case["delta"]), case["answer_id"]))[:25],
    }


def _ordinal_score(value: Any) -> int | None:
    try:
        score = round(float(value))
    except (TypeError, ValueError):
        return None
    if 1 <= score <= 5:
        return int(score)
    return None


def _confusion_case(
    row: dict[str, Any],
    reference_score: int,
    judge_score: int,
    interpretation: str,
    delta: int,
) -> dict[str, Any]:
    case = _case_row(row, reason=interpretation)
    case["reference_score"] = reference_score
    case["judge_score"] = judge_score
    case["delta"] = delta
    return case


def _reference_score(row: dict[str, Any], selected_dataset: str) -> float | None:
    dataset = row.get("dataset")
    if selected_dataset.upper() == "J2" or dataset == "J2":
        return _j2_reference_score(row)
    if selected_dataset.upper() == "J1" or dataset == "J1":
        return _j1_reference_score(row)
    return None


def _judge_arbiter_spearman(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[int, dict[str, list[int]]] = defaultdict(lambda: {"judge": [], "arbiter": []})
    for row in rows:
        if row["role"] == "arbitro":
            grouped[row["answer_id"]]["arbiter"].append(row["score"])
        else:
            grouped[row["answer_id"]]["judge"].append(row["score"])
    pairs: list[tuple[float, float]] = []
    for values in grouped.values():
        if values["judge"] and values["arbiter"]:
            pairs.append((statistics.mean(values["judge"]), statistics.mean(values["arbiter"])))
    if not pairs:
        return _spearman_unavailable(0, "Sem pares juiz x árbitro persistidos.")
    result = spearman([pair[0] for pair in pairs], [pair[1] for pair in pairs])
    if result["available"]:
        result["note"] = "Meta-avaliação complementar: média dos juízes por resposta comparada ao árbitro."
    return result


def _j2_reference_score(row: dict[str, Any]) -> int | None:
    expected = _normalize_choice(row.get("reference_answer"))
    actual = _normalize_choice(row.get("candidate_answer"))
    if not expected or not actual:
        return None
    return 5 if actual == expected else 1


def _j1_reference_score(row: dict[str, Any]) -> float | None:
    metadata = row.get("metadata") or {}
    for key in ("nota_humana", "human_score", "reference_score", "rubric_score"):
        value = metadata.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _candidate_ranking(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = _group_scores(rows, "candidate_model")
    result = []
    for label, scores in grouped.items():
        result.append(
            {
                "label": label,
                "value": round(statistics.mean(scores), 2),
                "count": len(scores),
                "stddev": round(statistics.pstdev(scores), 2) if len(scores) > 1 else 0,
            }
        )
    return sorted(result, key=lambda row: (-row["value"], row["label"]))


def _score_distribution(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"label": str(score), "value": sum(1 for row in rows if row["score"] == score)} for score in range(1, 6)]


def _score_distribution_by_model(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = _group_scores(rows, "candidate_model")
    result = []
    for label, scores in grouped.items():
        result.append(
            {
                "label": label,
                "total": len(scores),
                "average": round(statistics.mean(scores), 2),
                "scores": {str(score): scores.count(score) for score in range(1, 6)},
            }
        )
    return sorted(result, key=lambda row: (-row["average"], row["label"]))


def _rubric_heatmap(rows: list[dict[str, Any]]) -> dict[str, Any]:
    dimensions = (
        ("Argumentação", "argumentacao_score"),
        ("Precisão", "precisao_score"),
        ("Coesão legal", "coesao_legal_score"),
        ("Total", "total_score"),
    )
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: {key: [] for _, key in dimensions})
    for row in rows:
        label = str(row.get("candidate_model") or "sem valor")
        for _, key in dimensions:
            value = _dimension_score(row, key)
            if value is not None:
                grouped[label][key].append(value)

    heatmap_rows = []
    for label, scores_by_dimension in grouped.items():
        values = [
            round(statistics.mean(values), 2) if values else None
            for _, key in dimensions
            for values in [scores_by_dimension[key]]
        ]
        heatmap_rows.append(
            {
                "label": label,
                "values": values,
                "count": max((len(values) for values in scores_by_dimension.values()), default=0),
            }
        )
    return {
        "columns": [label for label, _ in dimensions],
        "rows": sorted(heatmap_rows, key=lambda row: (-(row["values"][-1] or 0), row["label"])),
    }


def _judge_candidate_heatmap(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    candidates: set[str] = set()
    for row in rows:
        judge = str(row.get("judge_model") or "sem juiz")
        candidate = str(row.get("candidate_model") or "sem modelo")
        score = row.get("score")
        if score is None:
            continue
        grouped[judge][candidate].append(int(score))
        candidates.add(candidate)

    sorted_candidates = sorted(candidates)
    heatmap_rows = []
    for judge, scores_by_candidate in grouped.items():
        values = [
            round(statistics.mean(scores_by_candidate[candidate]), 2) if scores_by_candidate.get(candidate) else None
            for candidate in sorted_candidates
        ]
        heatmap_rows.append(
            {
                "label": judge,
                "values": values,
                "count": sum(len(scores) for scores in scores_by_candidate.values()),
            }
        )
    return {
        "columns": sorted_candidates,
        "rows": sorted(heatmap_rows, key=lambda row: row["label"]),
    }


def _judge_disagreement_boxplot(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for answer_rows in _grouped_answer_rows(rows).values():
        scored = [
            int(row["score"])
            for row in answer_rows
            if row.get("status") == "success" and row.get("score") is not None
        ]
        if len(scored) < 2:
            continue
        candidate_model = str(answer_rows[0].get("candidate_model") or "sem modelo")
        grouped[candidate_model].append(max(scored) - min(scored))

    boxplot_rows = []
    for candidate_model, deltas in grouped.items():
        sorted_deltas = sorted(deltas)
        boxplot_rows.append(
            {
                "label": candidate_model,
                "count": len(sorted_deltas),
                "audit_count": sum(1 for delta in sorted_deltas if delta >= 2),
                "min": sorted_deltas[0],
                "q1": round(_percentile(sorted_deltas, 25), 2),
                "median": round(_percentile(sorted_deltas, 50), 2),
                "q3": round(_percentile(sorted_deltas, 75), 2),
                "max": sorted_deltas[-1],
            }
        )
    return {
        "metric": "judge_disagreement",
        "audit_threshold": 2,
        "rows": sorted(boxplot_rows, key=lambda row: row["label"]),
    }


def _legal_specialty_performance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    models: set[str] = set()
    for row in rows:
        specialty = _legal_specialty(row)
        model = str(row.get("candidate_model") or "sem modelo")
        score = row.get("score")
        if score is None:
            continue
        grouped[specialty][model].append(score)
        models.add(model)

    sorted_models = sorted(models)
    specialty_rows = []
    for specialty, scores_by_model in grouped.items():
        values = [
            round(statistics.mean(scores_by_model[model]), 2) if scores_by_model.get(model) else None
            for model in sorted_models
        ]
        total_count = sum(len(scores) for scores in scores_by_model.values())
        comparable_values = [value for value in values if value is not None]
        specialty_rows.append(
            {
                "label": specialty,
                "values": values,
                "count": total_count,
                "average": round(statistics.mean(comparable_values), 2) if comparable_values else None,
            }
        )
    return {
        "columns": sorted_models,
        "rows": sorted(specialty_rows, key=lambda row: (-(row["average"] or 0), row["label"])),
    }


def _difficulty_performance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    difficulties = ("Fácil", "Médio", "Difícil", "Muito difícil")
    grouped: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    models: set[str] = set()
    for row in rows:
        difficulty = _difficulty(row)
        if difficulty is None:
            continue
        model = str(row.get("candidate_model") or "sem modelo")
        score = row.get("score")
        if score is None:
            continue
        grouped[difficulty][model].append(score)
        models.add(model)

    sorted_models = sorted(models)
    return {
        "x_label": "dificuldade",
        "y_label": "média da nota",
        "difficulties": list(difficulties),
        "series": [
            {
                "label": model,
                "values": [
                    round(statistics.mean(grouped[difficulty][model]), 2) if grouped[difficulty].get(model) else None
                    for difficulty in difficulties
                ],
            }
            for model in sorted_models
        ],
    }


def _difficulty(row: dict[str, Any]) -> str | None:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    for key in ("difficulty", "dificuldade", "nivel_dificuldade", "complexidade"):
        value = metadata.get(key) or row.get(key)
        label = _normalize_difficulty(value)
        if label is not None:
            return label
    return None


def _normalize_difficulty(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    normalized = (
        text.replace("á", "a")
        .replace("é", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ú", "u")
        .replace("-", " ")
        .replace("_", " ")
    )
    normalized = " ".join(normalized.split())
    if normalized in {"facil", "easy"}:
        return "Fácil"
    if normalized in {"medio", "media", "moderado", "moderada", "medium"}:
        return "Médio"
    if normalized in {"dificil", "hard"}:
        return "Difícil"
    if normalized in {"muito dificil", "very hard", "muitodificil"}:
        return "Muito difícil"
    return None


def _legal_specialty(row: dict[str, Any]) -> str:
    if row.get("dataset") == "J2" or row.get("dataset_name") == "OAB_Exames":
        return _j2_legal_specialty(row)

    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    for key in ("legal_specialty", "especialidade", "disciplina", "area", "subject"):
        value = metadata.get(key)
        if value:
            return _format_specialty(value)
    category = metadata.get("category")
    if category:
        return _format_specialty(_strip_exam_prefix(str(category)))
    return "Sem especialidade"


def _j2_legal_specialty(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    question_type = metadata.get("tipo_questao") or metadata.get("category")
    specialty = _j2_specialty_from_question_type(question_type)
    if specialty is not None:
        return specialty

    specialty = _j2_specialty_from_question_number(metadata.get("question_number"))
    return specialty or "Sem especialidade"


def _j2_specialty_from_question_type(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().upper().replace("_", "-")
    if not normalized or normalized == "QUESTAO OBJETIVA":
        return None
    return {
        "ADMINISTRATIVE": "Direito Administrativo",
        "ENVIRONMENTAL": "Direito Administrativo",
        "BUSINESS": "Direito Empresarial",
        "CHILDREN": "Direito Civil",
        "CIVIL": "Direito Civil",
        "CIVIL-PROCEDURE": "Direito Civil",
        "CONSUMER": "Direito Civil",
        "CONSTITUTIONAL": "Direito Constitucional",
        "ETHICS": "Direito Constitucional",
        "HUMAN-RIGHTS": "Direito Constitucional",
        "INTERNATIONAL": "Direito Constitucional",
        "PHILOSOPHY": "Direito Constitucional",
        "CRIMINAL": "Direito Penal",
        "CRIMINAL-PROCEDURE": "Direito Penal",
        "LABOUR": "Direito Do Trabalho",
        "LABOUR-PROCEDURE": "Direito Do Trabalho",
        "TAXES": "Direito Tributario",
    }.get(normalized)


def _j2_specialty_from_question_number(value: Any) -> str | None:
    try:
        question_number = int(value)
    except (TypeError, ValueError):
        return None

    ranges = (
        (1, 12, "Direito Constitucional"),
        (13, 24, "Direito Constitucional"),
        (25, 29, "Direito Tributario"),
        (30, 36, "Direito Administrativo"),
        (37, 47, "Direito Civil"),
        (48, 52, "Direito Empresarial"),
        (53, 59, "Direito Civil"),
        (60, 70, "Direito Penal"),
        (71, 80, "Direito Do Trabalho"),
    )
    for start, end, specialty in ranges:
        if start <= question_number <= end:
            return specialty
    return None


def _strip_exam_prefix(value: str) -> str:
    parts = value.split("_", 1)
    return parts[1] if len(parts) == 2 and parts[0].isdigit() else value


def _format_specialty(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return "Sem especialidade"
    normalized = text.replace("-", "_").replace(" ", "_")
    words = [word for word in normalized.split("_") if word]
    return " ".join(word.capitalize() for word in words) if words else "Sem especialidade"


def _dimension_score(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None and key == "total_score":
        value = row.get("score")
    if value is None:
        criteria = row.get("criteria") if isinstance(row.get("criteria"), dict) else {}
        value = criteria.get(key)
    if value is None:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        value = metadata.get(key)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if 1 <= number <= 5 else None


def _average_by(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped = _group_scores(rows, key)
    return sorted(
        [{"label": label, "value": round(statistics.mean(scores), 2), "count": len(scores)} for label, scores in grouped.items()],
        key=lambda row: (-row["value"], row["label"]),
    )


def _comparative_model_ranking(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = (
            row.get("candidate_owner") or None,
            row.get("av2_model_name") or row.get("comparable_candidate_model") or "sem modelo",
            row.get("av3_provider_model_id") or None,
            row.get("comparable_candidate_model") or "sem modelo",
        )
        group = grouped.setdefault(
            key,
            {
                "owner": key[0],
                "av2_model_name": key[1],
                "av3_provider_model_id": key[2],
                "comparable_candidate_model": key[3],
                "all_pairs": 0,
                "complete_pairs": 0,
                "av2_scores": [],
                "av3_scores": [],
                "deltas": [],
            },
        )
        group["all_pairs"] += 1
        av2_score = row.get("av2_score")
        av3_score = row.get("av3_score")
        if av2_score is None or av3_score is None:
            continue
        delta = float(av3_score) - float(av2_score)
        group["complete_pairs"] += 1
        group["av2_scores"].append(float(av2_score))
        group["av3_scores"].append(float(av3_score))
        group["deltas"].append(delta)

    ranking = []
    for group in grouped.values():
        deltas = group["deltas"]
        complete_pairs = group["complete_pairs"]
        ranking.append(
            {
                "owner": group["owner"],
                "av2_model_name": group["av2_model_name"],
                "av3_provider_model_id": group["av3_provider_model_id"],
                "comparable_candidate_model": group["comparable_candidate_model"],
                "paired_evaluations": complete_pairs,
                "av2_average_score": _average(group["av2_scores"]),
                "av3_average_score": _average(group["av3_scores"]),
                "delta_average": _average(deltas),
                "improvement_rate": _percent(sum(1 for delta in deltas if delta > 0), len(deltas)),
                "regression_rate": _percent(sum(1 for delta in deltas if delta < 0), len(deltas)),
                "unchanged_rate": _percent(sum(1 for delta in deltas if delta == 0), len(deltas)),
                "comparative_coverage": {
                    "evaluated": complete_pairs,
                    "expected": group["all_pairs"],
                    "percent": _percent(complete_pairs, group["all_pairs"]),
                },
            }
        )
    return sorted(
        ranking,
        key=lambda row: (
            -(row["delta_average"] if row["delta_average"] is not None else float("-inf")),
            -row["paired_evaluations"],
            str(row["comparable_candidate_model"]),
            str(row["av3_provider_model_id"] or ""),
        ),
    )


def _comparative_delta_distribution(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = {delta: 0 for delta in range(-4, 5)}
    for row in rows:
        delta = int(float(row["av3_score"]) - float(row["av2_score"]))
        if delta in counts:
            counts[delta] += 1
    return [
        {
            "label": f"{delta:+d}" if delta > 0 else str(delta),
            "value": counts[delta],
        }
        for delta in range(-4, 5)
    ]


def _comparative_delta_outcomes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    outcomes = {"Piorou": 0, "Igual": 0, "Melhorou": 0}
    for row in rows:
        delta = float(row["av3_score"]) - float(row["av2_score"])
        if delta > 0:
            outcomes["Melhorou"] += 1
        elif delta < 0:
            outcomes["Piorou"] += 1
        else:
            outcomes["Igual"] += 1
    return [{"label": label, "value": value} for label, value in outcomes.items()]


def _comparative_delta_cases(
    rows: list[dict[str, Any]],
    *,
    minimum_delta: int,
    descending: bool,
    limit: int = 25,
) -> list[dict[str, Any]]:
    cases = []
    for row in rows:
        delta = int(float(row["av3_score"]) - float(row["av2_score"]))
        if descending and delta < minimum_delta:
            continue
        if not descending and delta > minimum_delta:
            continue
        case = {
            "dataset": row.get("dataset"),
            "question_id": row.get("question_id"),
            "question_sequence": row.get("question_sequence"),
            "candidate_model": row.get("comparable_candidate_model"),
            "normalized_judge_model": row.get("normalized_judge_model"),
            "av2_score": row.get("av2_score"),
            "av3_score": row.get("av3_score"),
            "delta": delta,
        }
        specialty = _legal_specialty(row)
        if specialty != "Sem especialidade":
            case["legal_specialty"] = specialty
        cases.append(case)
    cases.sort(
        key=lambda row: (
            -row["delta"] if descending else row["delta"],
            row["dataset"] or "",
            row["question_sequence"] if row["question_sequence"] is not None else row["question_id"] or 0,
            row["question_id"] or 0,
            row["candidate_model"] or "",
            row["normalized_judge_model"] or "",
        )
    )
    return cases[:limit]


def _comparative_specialty_breakdown(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        specialty = _legal_specialty(row)
        group = grouped.setdefault(
            specialty,
            {
                "specialty": specialty,
                "pairs": 0,
                "av2_scores": [],
                "av3_scores": [],
                "deltas": [],
                "model_deltas": defaultdict(list),
            },
        )
        av2_score = float(row["av2_score"])
        av3_score = float(row["av3_score"])
        delta = av3_score - av2_score
        candidate_model = str(row.get("comparable_candidate_model") or "sem modelo")
        group["pairs"] += 1
        group["av2_scores"].append(av2_score)
        group["av3_scores"].append(av3_score)
        group["deltas"].append(delta)
        group["model_deltas"][candidate_model].append(delta)

    summary = []
    for group in grouped.values():
        deltas = group["deltas"]
        model_averages = [
            {"candidate_model": candidate_model, "delta_average": _average(values)}
            for candidate_model, values in group["model_deltas"].items()
        ]
        model_averages = [
            item
            for item in model_averages
            if item["delta_average"] is not None
        ]
        model_averages.sort(key=lambda item: (-float(item["delta_average"]), item["candidate_model"]))
        best_model = model_averages[0] if model_averages else None
        worst_model = sorted(
            model_averages,
            key=lambda item: (float(item["delta_average"]), item["candidate_model"]),
        )[0] if model_averages else None
        delta_average = _average(deltas)
        summary.append(
            {
                "legal_specialty": group["specialty"],
                "paired_evaluations": group["pairs"],
                "av2_average_score": _average(group["av2_scores"]),
                "av3_average_score": _average(group["av3_scores"]),
                "delta_average": delta_average,
                "improvement_rate": _percent(sum(1 for delta in deltas if delta > 0), len(deltas)),
                "regression_rate": _percent(sum(1 for delta in deltas if delta < 0), len(deltas)),
                "unchanged_rate": _percent(sum(1 for delta in deltas if delta == 0), len(deltas)),
                "best_model_by_delta": best_model,
                "worst_model_by_delta": worst_model,
            }
        )
    return sorted(
        summary,
        key=lambda item: (
            -(abs(float(item["delta_average"])) if item["delta_average"] is not None else -1.0),
            -item["paired_evaluations"],
            item["legal_specialty"],
        ),
    )


def _comparative_spearman_breakdowns(rows: list[dict[str, Any]], selected_dataset: str) -> dict[str, list[dict[str, Any]]]:
    decorated = [_comparative_spearman_row(row, selected_dataset) for row in rows]
    return {
        "overall": [_comparative_spearman_summary("Geral", decorated)],
        "by_dataset": _comparative_spearman_grouped(decorated, lambda row: str(row.get("dataset") or "Sem dataset")),
        "by_candidate_model": _comparative_spearman_grouped(
            decorated,
            lambda row: str(row.get("comparable_candidate_model") or "sem modelo"),
        ),
        "by_judge_model": _comparative_spearman_grouped(
            decorated,
            lambda row: str(row.get("normalized_judge_model") or "sem juiz"),
        ),
    }


def _comparative_spearman_row(row: dict[str, Any], selected_dataset: str) -> dict[str, Any]:
    base = {
        "dataset": row.get("dataset"),
        "comparable_candidate_model": row.get("comparable_candidate_model"),
        "normalized_judge_model": row.get("normalized_judge_model"),
    }
    av2_row = {
        **base,
        "dataset": row.get("dataset"),
        "candidate_answer": row.get("av2_candidate_answer"),
        "reference_answer": row.get("av2_reference_answer"),
        "score": row.get("av2_score"),
        "metadata": row.get("metadata"),
    }
    av3_row = {
        **base,
        "dataset": row.get("dataset"),
        "candidate_answer": row.get("av3_candidate_answer"),
        "reference_answer": row.get("av3_reference_answer"),
        "score": row.get("av3_score"),
        "metadata": row.get("metadata"),
    }
    return {
        **base,
        "av2_score": row.get("av2_score"),
        "av3_score": row.get("av3_score"),
        "av2_reference_score": _reference_score(av2_row, selected_dataset),
        "av3_reference_score": _reference_score(av3_row, selected_dataset),
    }


def _comparative_spearman_grouped(
    rows: list[dict[str, Any]],
    label_func: Callable[[dict[str, Any]], str],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[label_func(row)].append(row)
    return sorted(
        (_comparative_spearman_summary(label, grouped_rows) for label, grouped_rows in grouped.items()),
        key=lambda item: (-item["paired_evaluations"], item["label"]),
    )


def _comparative_spearman_summary(label: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    av2_pairs = [
        (float(row["av2_reference_score"]), float(row["av2_score"]))
        for row in rows
        if row.get("av2_reference_score") is not None and row.get("av2_score") is not None
    ]
    av3_pairs = [
        (float(row["av3_reference_score"]), float(row["av3_score"]))
        for row in rows
        if row.get("av3_reference_score") is not None and row.get("av3_score") is not None
    ]
    spearman_av2 = (
        spearman([pair[0] for pair in av2_pairs], [pair[1] for pair in av2_pairs])
        if av2_pairs
        else _spearman_unavailable(0, DEFAULT_SPEARMAN_UNAVAILABLE)
    )
    spearman_av3 = (
        spearman([pair[0] for pair in av3_pairs], [pair[1] for pair in av3_pairs])
        if av3_pairs
        else _spearman_unavailable(0, DEFAULT_SPEARMAN_UNAVAILABLE)
    )
    return {
        "label": label,
        "paired_evaluations": len(rows),
        "reference_pairs_av2": len(av2_pairs),
        "reference_pairs_av3": len(av3_pairs),
        "spearman_av2": spearman_av2,
        "spearman_av3": spearman_av3,
        "spearman_delta": _delta_spearman(spearman_av2, spearman_av3),
    }


def _comparative_judge_agreement(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sides = _comparative_agreement_answer_sides(rows)
    av2_answers = sides["AV2"]
    av3_answers = sides["AV3"]
    observations: list[dict[str, Any]] = []
    comparable_answer_keys: set[tuple[Any, ...]] = set()

    for answer_key in sorted(set(av2_answers).intersection(av3_answers)):
        av2_primary = av2_answers[answer_key]["primary_scores"]
        av3_primary = av3_answers[answer_key]["primary_scores"]
        shared_judges = sorted(set(av2_primary).intersection(av3_primary))
        if len(shared_judges) < 2:
            continue
        comparable_answer_keys.add(answer_key)
        for index, judge_a in enumerate(shared_judges[:-1]):
            for judge_b in shared_judges[index + 1 :]:
                observations.append(
                    {
                        "answer_key": answer_key,
                        "dataset": answer_key[0],
                        "question_id": answer_key[1],
                        "candidate_model": av2_answers[answer_key]["candidate_model"],
                        "question_sequence": av2_answers[answer_key]["question_sequence"],
                        "judge_pair": _format_judge_pair(judge_a, judge_b),
                        "av2_delta": abs(av2_primary[judge_a] - av2_primary[judge_b]),
                        "av3_delta": abs(av3_primary[judge_a] - av3_primary[judge_b]),
                    }
                )

    return {
        "cards": _comparative_judge_agreement_cards(av2_answers, av3_answers, comparable_answer_keys, observations),
        "by_pair": _comparative_judge_agreement_by_pair(observations),
        "by_candidate_model": _comparative_judge_agreement_by_candidate_model(observations),
        "note": (
            "A concordancia compara, por resposta AV2/AV3 pareada, apenas pares de juizes normalizados "
            "presentes nos dois lados. Juizes principais incluem papeis principal/controle e registros sem "
            "metadado de papel; arbitro e medido separadamente."
        ),
        "empty_state_note": _comparative_agreement_empty_state(sides, observations),
    }


def _comparative_agreement_answer_sides(rows: list[dict[str, Any]]) -> dict[str, dict[tuple[Any, ...], dict[str, Any]]]:
    sides: dict[str, dict[tuple[Any, ...], dict[str, Any]]] = {"AV2": {}, "AV3": {}}
    for row in rows:
        source = str(row.get("source") or "")
        if source not in sides or not _is_success(row) or row.get("score") is None:
            continue
        answer_key = (
            row.get("dataset"),
            row.get("question_id"),
            row.get("comparable_candidate_model_id"),
            row.get("comparable_candidate_model"),
        )
        answer = sides[source].setdefault(
            answer_key,
            {
                "candidate_model": row.get("comparable_candidate_model") or "sem modelo",
                "question_sequence": row.get("question_sequence"),
                "primary_scores": {},
                "arbiter_scores": [],
            },
        )
        if row.get("judge_bucket") == "arbiter" or row.get("role") == "arbitro":
            answer["arbiter_scores"].append(int(row["score"]))
            continue
        answer["primary_scores"][str(row.get("normalized_judge_model") or "sem juiz")] = int(row["score"])
    return sides


def _comparative_judge_agreement_cards(
    av2_answers: dict[tuple[Any, ...], dict[str, Any]],
    av3_answers: dict[tuple[Any, ...], dict[str, Any]],
    comparable_answer_keys: set[tuple[Any, ...]],
    observations: list[dict[str, Any]],
) -> dict[str, Any]:
    total_observations = len(observations)
    av2_exact = _percent(sum(1 for row in observations if row["av2_delta"] == 0), total_observations)
    av3_exact = _percent(sum(1 for row in observations if row["av3_delta"] == 0), total_observations)
    av2_light = _percent(sum(1 for row in observations if row["av2_delta"] == 1), total_observations)
    av3_light = _percent(sum(1 for row in observations if row["av3_delta"] == 1), total_observations)
    av2_strong = _percent(sum(1 for row in observations if row["av2_delta"] >= 2), total_observations)
    av3_strong = _percent(sum(1 for row in observations if row["av3_delta"] >= 2), total_observations)
    av2_arbiter = _comparative_arbiter_metrics(av2_answers, comparable_answer_keys)
    av3_arbiter = _comparative_arbiter_metrics(av3_answers, comparable_answer_keys)
    return {
        "comparable_answers_with_multiple_judges": len(comparable_answer_keys),
        "comparable_pair_observations": total_observations,
        "av2_exact_agreement_rate": av2_exact,
        "av3_exact_agreement_rate": av3_exact,
        "delta_exact_agreement_rate": _subtract_nullable(av3_exact, av2_exact),
        "av2_light_divergence_rate": av2_light,
        "av3_light_divergence_rate": av3_light,
        "av2_strong_divergence_rate": av2_strong,
        "av3_strong_divergence_rate": av3_strong,
        "delta_strong_divergence_rate": _subtract_nullable(av3_strong, av2_strong),
        "av2_mean_absolute_delta": _average(row["av2_delta"] for row in observations),
        "av3_mean_absolute_delta": _average(row["av3_delta"] for row in observations),
        "av2_arbiter_rate": av2_arbiter["rate"],
        "av3_arbiter_rate": av3_arbiter["rate"],
        "av2_arbiter_available": av2_arbiter["available"],
        "av3_arbiter_available": av3_arbiter["available"],
        "av2_arbiter_consistency_rate": av2_arbiter["consistency_rate"],
        "av3_arbiter_consistency_rate": av3_arbiter["consistency_rate"],
    }


def _comparative_arbiter_metrics(
    answers: dict[tuple[Any, ...], dict[str, Any]],
    comparable_answer_keys: set[tuple[Any, ...]],
) -> dict[str, Any]:
    relevant_answers = [answers[key] for key in comparable_answer_keys if key in answers]
    if not relevant_answers:
        return {"available": False, "rate": None, "consistency_rate": None}
    answers_with_arbiter = [answer for answer in relevant_answers if answer["arbiter_scores"]]
    if not answers_with_arbiter:
        return {"available": False, "rate": None, "consistency_rate": None}
    consistent = 0
    for answer in answers_with_arbiter:
        primary_scores = list(answer["primary_scores"].values())
        arbiter_score = answer["arbiter_scores"][0]
        if primary_scores and min(primary_scores) <= arbiter_score <= max(primary_scores):
            consistent += 1
    return {
        "available": True,
        "rate": _percent(len(answers_with_arbiter), len(relevant_answers)),
        "consistency_rate": _percent(consistent, len(answers_with_arbiter)),
    }


def _comparative_judge_agreement_by_pair(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        grouped[row["judge_pair"]].append(row)
    result = []
    for judge_pair, pair_rows in grouped.items():
        av2_exact = _percent(sum(1 for row in pair_rows if row["av2_delta"] == 0), len(pair_rows))
        av3_exact = _percent(sum(1 for row in pair_rows if row["av3_delta"] == 0), len(pair_rows))
        av2_strong = _percent(sum(1 for row in pair_rows if row["av2_delta"] >= 2), len(pair_rows))
        av3_strong = _percent(sum(1 for row in pair_rows if row["av3_delta"] >= 2), len(pair_rows))
        result.append(
            {
                "judge_pair": judge_pair,
                "av2_comparable_evaluations": len(pair_rows),
                "av2_exact_agreement_rate": av2_exact,
                "av2_strong_divergence_rate": av2_strong,
                "av3_comparable_evaluations": len(pair_rows),
                "av3_exact_agreement_rate": av3_exact,
                "av3_strong_divergence_rate": av3_strong,
                "delta_exact_agreement_rate": _subtract_nullable(av3_exact, av2_exact),
                "delta_strong_divergence_rate": _subtract_nullable(av3_strong, av2_strong),
            }
        )
    return sorted(result, key=lambda row: (-row["av2_comparable_evaluations"], row["judge_pair"]))


def _comparative_judge_agreement_by_candidate_model(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        grouped[str(row["candidate_model"] or "sem modelo")].append(row)
    result = []
    for candidate_model, candidate_rows in grouped.items():
        av2_exact = _percent(sum(1 for row in candidate_rows if row["av2_delta"] == 0), len(candidate_rows))
        av3_exact = _percent(sum(1 for row in candidate_rows if row["av3_delta"] == 0), len(candidate_rows))
        av2_strong = _percent(sum(1 for row in candidate_rows if row["av2_delta"] >= 2), len(candidate_rows))
        av3_strong = _percent(sum(1 for row in candidate_rows if row["av3_delta"] >= 2), len(candidate_rows))
        result.append(
            {
                "candidate_model": candidate_model,
                "av2_exact_agreement_rate": av2_exact,
                "av3_exact_agreement_rate": av3_exact,
                "delta_agreement_rate": _subtract_nullable(av3_exact, av2_exact),
                "av2_strong_divergence_rate": av2_strong,
                "av3_strong_divergence_rate": av3_strong,
                "delta_strong_divergence_rate": _subtract_nullable(av3_strong, av2_strong),
                "comparable_pairs": len(candidate_rows),
            }
        )
    return sorted(result, key=lambda row: (-row["comparable_pairs"], row["candidate_model"]))


def _comparative_agreement_empty_state(
    sides: dict[str, dict[tuple[Any, ...], dict[str, Any]]],
    observations: list[dict[str, Any]],
) -> str | None:
    if observations:
        return None
    av2_answers = sides["AV2"]
    av3_answers = sides["AV3"]
    if not av2_answers or not av3_answers:
        return "Sem respostas AV2/AV3 pareadas para comparar a concordancia entre juizes com o filtro atual."
    overlap = set(av2_answers).intersection(av3_answers)
    if not overlap:
        return "As respostas restantes apos o filtro nao possuem par AV2/AV3 comparavel."
    if not any(len(av2_answers[key]["primary_scores"]) >= 2 for key in overlap if key in av2_answers):
        return "As respostas comparaveis em AV2 nao possuem pelo menos dois juizes primarios com nota."
    if not any(len(av3_answers[key]["primary_scores"]) >= 2 for key in overlap if key in av3_answers):
        return "As respostas comparaveis em AV3 nao possuem pelo menos dois juizes primarios com nota."
    return "Os pares de juizes disponiveis nao coincidem entre AV2 e AV3 para o filtro atual."


def _format_judge_pair(judge_a: str, judge_b: str) -> str:
    return " x ".join(sorted((judge_a, judge_b)))


def _subtract_nullable(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return round(left - right, 1)


def _critical_cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases = []
    for row in rows:
        if row["score"] == 1 or not _is_success(row):
            cases.append(_case_row(row, reason="nota 1" if row["score"] == 1 else f"status {row['status']}"))
    return cases


def _divergence_cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for answer_rows in _grouped_answer_rows(rows).values():
        disagreement = _primary_disagreement(answer_rows)
        if disagreement is None or disagreement["primary_delta"] < 2:
            continue
        base = disagreement["base_row"]
        case = _case_row(base, reason=f"delta {disagreement['primary_delta']}")
        case["primary_delta"] = disagreement["primary_delta"]
        case["scores"] = disagreement["scores"]
        case["arbitration_triggered"] = disagreement["arbitration_triggered"]
        case["arbitration_reason"] = disagreement["arbitration_reason"]
        cases.append(case)
    return sorted(cases, key=lambda row: (-row["primary_delta"], row["answer_id"]))


def _minor_disagreement_cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for answer_rows in _grouped_answer_rows(rows).values():
        disagreement = _primary_disagreement(answer_rows)
        if disagreement is None or disagreement["primary_delta"] != 1:
            continue
        base = disagreement["base_row"]
        case = _case_row(base, reason="delta 1 (leve)")
        case["primary_delta"] = disagreement["primary_delta"]
        case["scores"] = disagreement["scores"]
        case["arbitration_triggered"] = disagreement["arbitration_triggered"]
        case["arbitration_reason"] = disagreement["arbitration_reason"]
        cases.append(case)
    return sorted(cases, key=lambda row: (row["answer_id"], row["evaluation_id"]))


def _judge_agreement(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cards = {
        "total_compared": 0,
        "delta_0": 0,
        "delta_1": 0,
        "delta_2": 0,
        "delta_3": 0,
        "delta_4": 0,
        "arbiter_triggered": 0,
    }
    arbitrations: list[dict[str, Any]] = []
    for answer_rows in _grouped_answer_rows(rows).values():
        disagreement = _primary_disagreement(answer_rows)
        if disagreement is None:
            continue
        delta = int(disagreement["primary_delta"])
        if not 0 <= delta <= 4:
            continue
        cards["total_compared"] += 1
        cards[f"delta_{delta}"] += 1
        if not disagreement["arbitration_triggered"]:
            continue
        cards["arbiter_triggered"] += 1
        base = disagreement["base_row"]
        arbitrations.append(
            {
                "answer_id": base.get("answer_id"),
                "question_id": base.get("question_id"),
                "candidate_model": base.get("candidate_model"),
                "judge_1_score": disagreement["judge_1_score"],
                "judge_2_score": disagreement["judge_2_score"],
                "delta": delta,
                "arbiter_score": disagreement["arbiter_score"],
                "arbitration_reason": disagreement["arbitration_reason"],
            }
        )
    return {
        "cards": cards,
        "arbitrations": sorted(arbitrations, key=lambda row: (-row["delta"], row["answer_id"] or 0)),
    }


def _grouped_answer_rows(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["answer_id"])].append(row)
    return grouped


def _primary_disagreement(answer_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    principal_row: dict[str, Any] | None = None
    controle_row: dict[str, Any] | None = None
    for row in answer_rows:
        if row.get("status") != "success":
            continue
        score = row.get("score")
        if score is None:
            continue
        role = row.get("role")
        if role == "principal" and principal_row is None:
            principal_row = row
        elif role == "controle" and controle_row is None:
            controle_row = row
        if principal_row is not None and controle_row is not None:
            break
    if principal_row is None or controle_row is None:
        return None

    primary_delta = abs(int(principal_row["score"]) - int(controle_row["score"]))
    arbitration_triggered = False
    arbitration_reason = None
    arbiter_score = None
    for row in answer_rows:
        if row.get("status") != "success":
            continue
        if row.get("role") != "arbitro" or row.get("score") is None:
            continue
        arbitration_triggered = True
        arbitration_reason = _trigger_suffix(row.get("trigger_reason"))
        arbiter_score = int(row["score"])
        break

    base_row = principal_row
    scores = []
    scores.append(f"{principal_row['judge_model']}(principal)={principal_row['score']}")
    scores.append(f"{controle_row['judge_model']}(controle)={controle_row['score']}")
    if arbitration_triggered:
        arbiter = next(
            (
                row
                for row in answer_rows
                if row.get("status") == "success" and row.get("role") == "arbitro" and row.get("score") is not None
            ),
            None,
        )
        if arbiter is not None:
            scores.append(f"{arbiter['judge_model']}(arbitro)={arbiter['score']}")

    return {
        "base_row": base_row,
        "primary_delta": primary_delta,
        "judge_1_score": int(principal_row["score"]),
        "judge_2_score": int(controle_row["score"]),
        "arbiter_score": arbiter_score,
        "scores": ", ".join(scores),
        "arbitration_triggered": arbitration_triggered,
        "arbitration_reason": arbitration_reason,
    }


def _trigger_suffix(trigger_reason: str | None) -> str | None:
    if not trigger_reason:
        return None
    text = str(trigger_reason)
    if ":" in text:
        return text.split(":", 1)[1] or None
    return text


def _divergence_chart(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[str, int] = defaultdict(int)
    for case in cases:
        counts[case["candidate_model"]] += 1
    return sorted([{"label": label, "value": value} for label, value in counts.items()], key=lambda row: (-row["value"], row["label"]))


def _critical_chart(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[str, int] = defaultdict(int)
    for case in cases:
        counts[case["reason"]] += 1
    return sorted([{"label": label, "value": value} for label, value in counts.items()], key=lambda row: (-row["value"], row["label"]))


def _critical_error_analysis(
    rows: list[dict[str, Any]],
    divergence_cases: list[dict[str, Any]],
    selected_dataset: str,
) -> dict[str, list[dict[str, Any]]]:
    category_order = [
        "Nota alta para resposta errada",
        "Nota baixa para resposta correta",
        "Alucinacao normativa",
        "Resposta sem fundamentacao",
        "Divergencia entre juizes",
        "Erro de parsing",
        "Timeout/HTTP error",
    ]
    cases: list[dict[str, Any]] = []
    seen: set[tuple[int | None, str, str]] = set()

    def add_case(row: dict[str, Any], error_type: str, justification: str) -> None:
        key = (row.get("evaluation_id"), error_type, row.get("judge_model") or "")
        if key in seen:
            return
        seen.add(key)
        cases.append(_critical_error_case(row, error_type, justification))

    for row in rows:
        reference_score = _ordinal_score(_reference_score(row, selected_dataset))
        judge_score = _ordinal_score(row.get("score"))
        if judge_score is not None and reference_score is not None:
            if judge_score >= 4 and reference_score <= 2:
                add_case(row, "Nota alta para resposta errada", f"referencia {reference_score}, juiz {judge_score}")
            if judge_score <= 2 and reference_score >= 4:
                add_case(row, "Nota baixa para resposta correta", f"referencia {reference_score}, juiz {judge_score}")
        if _has_normative_hallucination(row):
            add_case(row, "Alucinacao normativa", _short_justification(row, "indicio de norma inexistente ou fabricada"))
        if _is_success(row) and row.get("score") is not None and not _has_legal_grounding(row):
            add_case(row, "Resposta sem fundamentacao", _short_justification(row, "sem citacao legal identificavel"))
        if row.get("score") is None:
            add_case(row, "Erro de parsing", _short_justification(row, "nota nao extraivel"))
        if _is_timeout_or_http_error(row):
            add_case(row, "Timeout/HTTP error", _short_justification(row, str(row.get("status") or "falha operacional")))

    rows_by_answer = {row.get("answer_id"): row for row in rows}
    for divergence in divergence_cases:
        base = rows_by_answer.get(divergence.get("answer_id"), divergence)
        add_case(base, "Divergencia entre juizes", str(divergence.get("scores") or divergence.get("reason") or "delta >= 2"))

    counts = {label: 0 for label in category_order}
    for case in cases:
        counts[case["error_type"]] = counts.get(case["error_type"], 0) + 1
    categories = [{"label": label, "value": counts[label]} for label in category_order]
    return {
        "categories": categories,
        "cases": sorted(cases, key=lambda row: (-counts.get(row["error_type"], 0), row["error_type"], row["question_id"], row["candidate_model"])),
    }


def _critical_error_case(row: dict[str, Any], error_type: str, justification: str) -> dict[str, Any]:
    return {
        "evaluation_id": row.get("evaluation_id"),
        "question_id": row.get("question_id"),
        "candidate_model": row.get("candidate_model"),
        "judge_model": row.get("judge_model"),
        "score": row.get("score"),
        "error_type": error_type,
        "short_justification": justification,
        "log_url": _log_url(row),
    }


def _has_normative_hallucination(row: dict[str, Any]) -> bool:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    for key in ("normative_hallucination", "hallucinated_norm", "invalid_legal_citation", "lei_inexistente"):
        if metadata.get(key) is True:
            return True
    text = f"{row.get('rationale') or ''} {row.get('trigger_reason') or ''}".lower()
    markers = ("lei inexistente", "artigo inexistente", "norma inexistente", "fundamento inexistente", "citação inexistente", "citacao inexistente", "alucina", "fabricad")
    return any(marker in text for marker in markers)


def _has_legal_grounding(row: dict[str, Any]) -> bool:
    text = str(row.get("rationale") or "")
    if not text.strip():
        return False
    legal_markers = ("art.", "artigo", "lei", "codigo", "código", "constituição", "constituicao", "cf", "cpp", "cpc", "cp", "clt", "sumula", "súmula")
    return any(marker in text.lower() for marker in legal_markers)


def _is_timeout_or_http_error(row: dict[str, Any]) -> bool:
    text = f"{row.get('status') or ''} {row.get('rationale') or ''} {row.get('trigger_reason') or ''}".lower()
    return "timeout" in text or "http" in text or "connection" in text


def _short_justification(row: dict[str, Any], fallback: str) -> str:
    text = str(row.get("rationale") or row.get("trigger_reason") or "").strip()
    if not text:
        return fallback
    collapsed = " ".join(text.split())
    return collapsed[:117] + "..." if len(collapsed) > 120 else collapsed


def _log_url(row: dict[str, Any]) -> str | None:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    for key in ("log_url", "audit_log_url"):
        value = metadata.get(key) or row.get(key)
        if value:
            return str(value)
    return None


def _case_row(row: dict[str, Any], *, reason: str) -> dict[str, Any]:
    return {
        "evaluation_id": row["evaluation_id"],
        "answer_id": row["answer_id"],
        "question_id": row["question_id"],
        "dataset": row["dataset"],
        "candidate_model": row["candidate_model"],
        "judge_model": row["judge_model"],
        "role": row["role"],
        "score": row["score"],
        "status": row["status"],
        "evaluated_at": row["evaluated_at"],
        "reason": reason,
    }


def _group_scores(rows: list[dict[str, Any]], key: str) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key) or "sem valor")].append(row["score"])
    return grouped


def _rank(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        end = index
        while end + 1 < len(indexed) and indexed[end + 1][1] == indexed[index][1]:
            end += 1
        average_rank = (index + 1 + end + 1) / 2
        for position in range(index, end + 1):
            ranks[indexed[position][0]] = average_rank
        index = end + 1
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    x_term = sum((x - mean_x) ** 2 for x in xs)
    y_term = sum((y - mean_y) ** 2 for y in ys)
    denominator = math.sqrt(x_term * y_term)
    if denominator == 0:
        return None
    return numerator / denominator


def _spearman_p_value(rho: float, sample_size: int) -> float | None:
    if sample_size <= 2:
        return None
    if abs(rho) >= 1:
        return 0.0
    t_value = abs(rho) * math.sqrt((sample_size - 2) / (1 - rho**2))
    p_value = math.erfc(t_value / math.sqrt(2))
    return round(max(0.0, min(1.0, p_value)), 6)


def _spearman_unavailable(sample_size: int, note: str) -> dict[str, Any]:
    return {"value": None, "p_value": None, "sample_size": sample_size, "available": False, "note": note}


def _delta_spearman(av2_value: dict[str, Any], av3_value: dict[str, Any]) -> dict[str, Any]:
    if not av2_value.get("available") or not av3_value.get("available"):
        return _spearman_unavailable(0, "Sem dados suficientes para calcular delta de Spearman.")
    return {
        "value": round(float(av3_value["value"]) - float(av2_value["value"]), 4),
        "p_value": None,
        "sample_size": min(int(av2_value.get("sample_size") or 0), int(av3_value.get("sample_size") or 0)),
        "available": True,
        "note": "Delta Spearman calculado como AV3 Com_RAG menos AV2 Sem_RAG.",
    }


def _serialize_filters(filters: DashboardFilters) -> dict[str, Any]:
    return {
        "dataset": filters.dataset,
        "candidate_models": list(filters.candidate_models),
        "judge_models": list(filters.judge_models),
        "status": filters.status,
        "date_from": filters.date_from.isoformat() if filters.date_from is not None else None,
        "date_to": filters.date_to.isoformat() if filters.date_to is not None else None,
        "group_by": filters.group_by,
    }


def _split_csv(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    return date.fromisoformat(value)


def _is_success(row: dict[str, Any]) -> bool:
    return (row.get("status") or "success") == "success"


def _average(values: Any) -> float | None:
    collected = [value for value in values if value is not None]
    return round(statistics.mean(collected), 2) if collected else None


def _percentile(values: list[int], percentile: int) -> float:
    if not values:
        raise ValueError("Percentile requires at least one value.")
    if len(values) == 1:
        return float(values[0])
    position = (len(values) - 1) * percentile / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(values[lower])
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def _percent(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round((numerator / denominator) * 100, 1)


def _safe_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _normalized_judge_sql(model_sql: str) -> str:
    return (
        f"CASE WHEN {model_sql} IN ('Unbabel/M-Prometheus-14B', 'Unbabel/M-Prometheus-7B') "
        f"THEN 'M-Prometheus' ELSE {model_sql} END"
    )


def _dataset_labels_json_literal() -> str:
    return '\'{"OAB_Bench":"J1","OAB_Exames":"J2"}\''


def _comparative_av2_dataset_sql(column_sql: str) -> str:
    return (
        f"COALESCE({_dataset_labels_json_literal()}::jsonb ->> {column_sql}, "
        f"CASE WHEN UPPER({column_sql}) IN ('J1', 'J2') THEN UPPER({column_sql}) ELSE {column_sql} END)"
    )


def _comparative_av3_candidate_model_id_sql() -> str:
    return "COALESCE(assignment_direct.id_modelo_av2, assignment_legacy.id_modelo_av2)"


def _comparative_av3_candidate_model_name_sql() -> str:
    return "COALESCE(av2_model_direct.nome_modelo, assignment_legacy.av2_model_name)"


def _normalize_choice(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    if not text:
        return None
    for char in text:
        if char in {"A", "B", "C", "D", "E"}:
            return char
    return None
