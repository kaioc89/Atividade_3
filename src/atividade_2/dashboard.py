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
