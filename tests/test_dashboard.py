from __future__ import annotations

from types import SimpleNamespace

from atividade_2.dashboard import (
    AV3DashboardService,
    ComparativeDashboardService,
    DashboardFilters,
    build_comparative_dashboard_payload,
    build_dashboard_payload,
    spearman,
)


class _DashboardCursor:
    def __init__(self) -> None:
        self.queries: list[str] = []
        self.params: list[list[object]] = []
        self._fetchall_responses = [
            [
                (
                    11,
                    301,
                    77,
                    "J1",
                    "av3-modelo",
                    "juiz-a",
                    "principal",
                    "success",
                    4,
                    None,
                    "Justificativa objetiva.",
                    "Resposta candidata AV3",
                    "Resposta ouro",
                    {"category": "39_direito_administrativo"},
                    "2plus1:primary_panel",
                    "success",
                    "completed",
                    "Diego",
                    "openai/gpt-5",
                    17,
                    True,
                )
            ],
            [("av3-modelo",)],
            [("juiz-a",)],
            [("J1",), ("J2",)],
            [("success",), ("timeout",)],
        ]
        self._fetchone_responses = [(1,)]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def execute(self, query, params=None) -> None:
        self.queries.append(query)
        self.params.append(list(params or []))

    def fetchall(self):
        return self._fetchall_responses.pop(0)

    def fetchone(self):
        return self._fetchone_responses.pop(0)


class _DashboardConnection:
    def __init__(self) -> None:
        self.cursor_instance = _DashboardCursor()
        self.closed = False

    def cursor(self):
        return self.cursor_instance

    def close(self) -> None:
        self.closed = True


class _ComparativeDashboardCursor:
    def __init__(self) -> None:
        self.queries: list[str] = []
        self.params: list[list[object]] = []
        self._fetchall_responses = [
            [
                ("J2", 77, 8, "modelo-a", "modelo-a", "Equipe A", "provider/modelo-a", "M-Prometheus", "Unbabel/M-Prometheus-14B", "Unbabel/M-Prometheus-7B", "success", "success", 3, 4, "B", "B", "B", "B", {"question_number": 77}, 77),
                ("J2", 78, 8, "modelo-a", "modelo-a", "Equipe A", "provider/modelo-a", "juiz-b", "juiz-b", "juiz-b", "success", "success", 4, 4, "C", "C", "C", "C", {"question_number": 78}, 78),
                ("J2", 79, 8, "modelo-a", "modelo-a", "Equipe A", "provider/modelo-a", "juiz-c", "juiz-c", None, "success", None, 5, None, "D", None, "D", None, {"question_number": 79}, 79),
            ],
            [
                ("AV2", "J2", 77, 8, "modelo-a", "principal", "primary", "M-Prometheus", "success", 3, 77),
                ("AV2", "J2", 77, 8, "modelo-a", "controle", "primary", "juiz-b", "success", 4, 77),
                ("AV2", "J2", 77, 8, "modelo-a", "arbitro", "arbiter", "juiz-arbitro", "success", 4, 77),
                ("AV3", "J2", 77, 8, "modelo-a", "principal", "primary", "M-Prometheus", "success", 4, 77),
                ("AV3", "J2", 77, 8, "modelo-a", "controle", "primary", "juiz-b", "success", 4, 77),
            ],
        ]
        self._fetchone_responses = [
            (["J1", "J2"], ["modelo-a"], ["M-Prometheus", "juiz-b"]),
        ]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def execute(self, query, params=None) -> None:
        self.queries.append(query)
        self.params.append(list(params or []))

    def fetchall(self):
        return self._fetchall_responses.pop(0)

    def fetchone(self):
        return self._fetchone_responses.pop(0)


class _ComparativeDashboardConnection:
    def __init__(self) -> None:
        self.cursor_instance = _ComparativeDashboardCursor()
        self.closed = False

    def cursor(self):
        return self.cursor_instance

    def close(self) -> None:
        self.closed = True


def _row(
    *,
    evaluation_id: int,
    answer_id: int,
    dataset: str,
    candidate_answer: str,
    reference_answer: str,
    score: int | None,
    role: str = "principal",
    candidate_model: str = "modelo-a",
    judge_model: str = "juiz-a",
    status: str = "success",
    rationale: str = "Art. 1 da lei aplicavel.",
) -> dict:
    return {
        "evaluation_id": evaluation_id,
        "answer_id": answer_id,
        "question_id": answer_id + 100,
        "dataset": dataset,
        "dataset_name": "OAB_Exames" if dataset == "J2" else "OAB_Bench",
        "candidate_model": candidate_model,
        "judge_model": judge_model,
        "role": role,
        "status": status,
        "score": score,
        "evaluated_at": "2026-04-30T10:00:00",
        "rationale": rationale,
        "candidate_answer": candidate_answer,
        "reference_answer": reference_answer,
        "metadata": {},
        "trigger_reason": "2plus1:primary_panel",
    }


def test_spearman_uses_average_ranks_for_ties() -> None:
    result = spearman([5, 1, 5, 1], [5, 1, 5, 1])

    assert result["available"] is True
    assert result["value"] == 1.0
    assert result["sample_size"] == 4


def test_av3_dashboard_service_uses_candidate_answer_join_path() -> None:
    connection = _DashboardConnection()
    service = AV3DashboardService(
        settings_loader=lambda: SimpleNamespace(database_url="postgresql://example.invalid/db"),
        connect_func=lambda _dsn: connection,
    )

    payload = service.load(
        DashboardFilters(
            dataset="J1",
            candidate_models=("av3-modelo",),
            judge_models=("juiz-a",),
            status="sucesso",
            group_by="juiz",
        )
    )

    assert payload["cards"]["evaluations"] == 1
    assert payload["cards"]["coverage"] == {"evaluated": 1, "expected": 1, "percent": 100.0}
    assert connection.closed is True
    selection_sql = connection.cursor_instance.queries[0]
    assert "FROM av3.candidate_answers ca" in selection_sql
    assert "JOIN av3.candidate_runs cr" in selection_sql
    assert "LEFT JOIN av3.candidate_model_assignments cma" in selection_sql
    assert "JOIN public.avaliacoes_juiz aj" in selection_sql
    assert "aj.id_candidate_answer = ca.id_candidate_answer" in selection_sql
    assert "cr.dataset_code = %s" in selection_sql
    assert "mj.nome_modelo = ANY(%s)" in selection_sql
    assert "COALESCE(aj.status_avaliacao, 'success') = 'success'" in selection_sql
    assert "id_resposta_ativa1" not in selection_sql
    assert "JOIN respostas_atividade_1" not in selection_sql
    expected_sql = connection.cursor_instance.queries[1]
    assert "COUNT(DISTINCT ca.id_candidate_answer)" in expected_sql
    assert "EXISTS (SELECT 1 FROM av3.candidate_answer_context_chunks cacc" in expected_sql
    assert connection.cursor_instance.params[0] == ["J1", ["av3-modelo"], ["juiz-a"]]


def test_comparative_dashboard_service_uses_distinct_av2_and_av3_join_paths() -> None:
    connection = _ComparativeDashboardConnection()
    service = ComparativeDashboardService(
        settings_loader=lambda: SimpleNamespace(database_url="postgresql://example.invalid/db"),
        connect_func=lambda _dsn: connection,
    )

    payload = service.load(
        DashboardFilters(
            dataset="J2",
            candidate_models=("modelo-a",),
            judge_models=("M-Prometheus",),
            status="sucesso",
            group_by="juiz",
        )
    )

    assert payload["cards"]["comparable_pairs"] == 2
    assert payload["cards"]["delta_average"] == 0.5
    assert payload["cards"]["judge_agreement_comparison"]["comparable_answers_with_multiple_judges"] == 1
    assert payload["options"]["datasets"] == ["J1", "J2"]
    assert connection.closed is True
    selection_sql = connection.cursor_instance.queries[0]
    assert "JOIN public.respostas_atividade_1 r" in selection_sql
    assert "a.id_resposta_ativa1" in selection_sql
    assert "JOIN public.avaliacoes_juiz aj" in selection_sql
    assert "aj.id_candidate_answer = ca.id_candidate_answer" in selection_sql
    assert "LEFT JOIN av3.candidate_model_assignments assignment_direct" in selection_sql
    assert "LEFT JOIN LATERAL (" in selection_sql
    assert "assignment.av3_provider_model_id = ca.model_name" in selection_sql
    assert "FULL OUTER JOIN av3_scores av3" in selection_sql
    assert "CASE WHEN UPPER(d.nome_dataset) IN ('J1', 'J2') THEN UPPER(d.nome_dataset) ELSE d.nome_dataset END" in selection_sql
    assert "M-Prometheus" in selection_sql
    assert connection.cursor_instance.params[0] == ["J2", ["modelo-a"], ["M-Prometheus"], "J2", ["modelo-a"], ["M-Prometheus"]]
    agreement_sql = connection.cursor_instance.queries[1]
    assert "SELECT\n            'AV2' AS source" in agreement_sql
    assert "judge_bucket" in agreement_sql
    assert "paired_answer_keys AS (" in agreement_sql
    assert "COALESCE(a.papel_juiz, '') AS role" in agreement_sql
    assert "COALESCE(aj.papel_juiz, '') AS role" in agreement_sql
    assert connection.cursor_instance.params[1] == ["J2", ["modelo-a"], ["M-Prometheus"], "J2", ["modelo-a"], ["M-Prometheus"]]
    options_sql = connection.cursor_instance.queries[2]
    assert "comparative_datasets AS (" in options_sql
    assert "av2_source_datasets AS (" in options_sql
    assert "av2_evaluation_datasets AS (" in options_sql
    assert "av3_run_datasets AS (" in options_sql
    assert "av3_evaluation_datasets AS (" in options_sql
    assert "ARRAY(SELECT dataset_code FROM comparative_datasets WHERE dataset_code IS NOT NULL ORDER BY dataset_code)" in options_sql
    assert "ARRAY(SELECT DISTINCT dataset_code FROM paired_candidates ORDER BY dataset_code)" not in options_sql


def test_comparative_dashboard_payload_calculates_pair_metrics_and_safely_handles_partial_rows() -> None:
    rows = [
        {
            "dataset": "J2",
            "question_id": 77,
            "av2_model_name": "modelo-a",
            "comparable_candidate_model": "modelo-a",
            "candidate_owner": "Equipe A",
            "av3_provider_model_id": "provider/modelo-a",
            "normalized_judge_model": "M-Prometheus",
            "av2_score": 3,
            "av3_score": 4,
            "av2_candidate_answer": "B",
            "av3_candidate_answer": "B",
            "av2_reference_answer": "B",
            "av3_reference_answer": "B",
            "metadata": {"question_number": 77},
            "question_sequence": 77,
        },
        {
            "dataset": "J2",
            "question_id": 78,
            "av2_model_name": "modelo-a",
            "comparable_candidate_model": "modelo-a",
            "candidate_owner": "Equipe A",
            "av3_provider_model_id": "provider/modelo-a",
            "normalized_judge_model": "juiz-b",
            "av2_score": 1,
            "av3_score": 1,
            "av2_candidate_answer": "A",
            "av3_candidate_answer": "A",
            "av2_reference_answer": "C",
            "av3_reference_answer": "C",
            "metadata": {"question_number": 78},
            "question_sequence": 78,
        },
        {
            "dataset": "J2",
            "question_id": 79,
            "av2_model_name": "modelo-a",
            "comparable_candidate_model": "modelo-a",
            "candidate_owner": "Equipe A",
            "av3_provider_model_id": "provider/modelo-a",
            "normalized_judge_model": "juiz-c",
            "av2_score": 5,
            "av3_score": None,
            "av2_candidate_answer": "D",
            "av3_candidate_answer": None,
            "av2_reference_answer": "D",
            "av3_reference_answer": None,
            "metadata": {"question_number": 79},
            "question_sequence": 79,
        },
    ]

    payload = build_comparative_dashboard_payload(rows, filters=DashboardFilters(dataset="J2"))

    assert payload["cards"]["comparable_pairs"] == 2
    assert payload["cards"]["comparative_coverage"] == {"evaluated": 2, "expected": 3, "percent": 66.7}
    assert payload["cards"]["av2_average_score"] == 2.0
    assert payload["cards"]["av3_average_score"] == 2.5
    assert payload["cards"]["delta_average"] == 0.5
    assert payload["cards"]["improvement_rate"] == 50.0
    assert payload["cards"]["regression_rate"] == 0.0
    assert payload["cards"]["unchanged_rate"] == 50.0
    assert payload["cards"]["spearman_av2"]["available"] is True
    assert payload["cards"]["spearman_av3"]["available"] is True
    assert payload["cards"]["spearman_delta"]["available"] is True
    assert payload["charts"]["delta_distribution"] == [
        {"label": "-4", "value": 0},
        {"label": "-3", "value": 0},
        {"label": "-2", "value": 0},
        {"label": "-1", "value": 0},
        {"label": "0", "value": 1},
        {"label": "+1", "value": 1},
        {"label": "+2", "value": 0},
        {"label": "+3", "value": 0},
        {"label": "+4", "value": 0},
    ]
    assert payload["charts"]["delta_outcomes"] == [
        {"label": "Piorou", "value": 0},
        {"label": "Igual", "value": 1},
        {"label": "Melhorou", "value": 1},
    ]
    assert payload["tables"]["ranking_by_model"] == [
        {
            "owner": "Equipe A",
            "av2_model_name": "modelo-a",
            "av3_provider_model_id": "provider/modelo-a",
            "comparable_candidate_model": "modelo-a",
            "paired_evaluations": 2,
            "av2_average_score": 2.0,
            "av3_average_score": 2.5,
            "delta_average": 0.5,
            "improvement_rate": 50.0,
            "regression_rate": 0.0,
            "unchanged_rate": 50.0,
            "comparative_coverage": {"evaluated": 2, "expected": 3, "percent": 66.7},
        }
    ]
    assert payload["tables"]["specialties"] == [
        {
            "legal_specialty": "Direito Do Trabalho",
            "paired_evaluations": 2,
            "av2_average_score": 2.0,
            "av3_average_score": 2.5,
            "delta_average": 0.5,
            "improvement_rate": 50.0,
            "regression_rate": 0.0,
            "unchanged_rate": 50.0,
            "best_model_by_delta": {"candidate_model": "modelo-a", "delta_average": 0.5},
            "worst_model_by_delta": {"candidate_model": "modelo-a", "delta_average": 0.5},
        }
    ]
    assert payload["tables"]["spearman_breakdowns"]["overall"] == [
        {
            "label": "Geral",
            "paired_evaluations": 2,
            "reference_pairs_av2": 2,
            "reference_pairs_av3": 2,
            "spearman_av2": {
                "value": 1.0,
                "p_value": None,
                "sample_size": 2,
                "available": True,
                "note": "Calculado com ranks médios para empates; p-value aproximado.",
            },
            "spearman_av3": {
                "value": 1.0,
                "p_value": None,
                "sample_size": 2,
                "available": True,
                "note": "Calculado com ranks médios para empates; p-value aproximado.",
            },
            "spearman_delta": {
                "value": 0.0,
                "p_value": None,
                "sample_size": 2,
                "available": True,
                "note": "Delta Spearman calculado como AV3 Com_RAG menos AV2 Sem_RAG.",
            },
        }
    ]
    assert payload["cards"]["pairing_diagnostics"] == {
        "complete_pairs": 2,
        "av2_only_tuples": 1,
        "av3_only_tuples": 0,
        "filtered_dataset": "J2",
        "filtered_candidate_models": [],
        "filtered_judge_models": [],
        "normalized_judge_models": ["M-Prometheus", "juiz-b", "juiz-c"],
    }
    assert "Pares completos: 2" in payload["methodology"]["diagnostics_summary"]


def test_comparative_dashboard_payload_builds_judge_agreement_metrics_from_shared_pairs() -> None:
    rows = [
        {
            "dataset": "J2",
            "question_id": 77,
            "av2_model_name": "modelo-a",
            "comparable_candidate_model": "modelo-a",
            "candidate_owner": "Equipe A",
            "av3_provider_model_id": "provider/modelo-a",
            "normalized_judge_model": "M-Prometheus",
            "av2_score": 3,
            "av3_score": 4,
            "av2_candidate_answer": "B",
            "av3_candidate_answer": "B",
            "av2_reference_answer": "B",
            "av3_reference_answer": "B",
            "metadata": {"question_number": 77},
            "question_sequence": 77,
        }
    ]
    agreement_rows = [
        {"source": "AV2", "dataset": "J2", "question_id": 77, "comparable_candidate_model_id": 8, "comparable_candidate_model": "modelo-a", "role": "principal", "judge_bucket": "primary", "normalized_judge_model": "M-Prometheus", "status": "success", "score": 3, "question_sequence": 77},
        {"source": "AV2", "dataset": "J2", "question_id": 77, "comparable_candidate_model_id": 8, "comparable_candidate_model": "modelo-a", "role": "controle", "judge_bucket": "primary", "normalized_judge_model": "juiz-b", "status": "success", "score": 4, "question_sequence": 77},
        {"source": "AV2", "dataset": "J2", "question_id": 77, "comparable_candidate_model_id": 8, "comparable_candidate_model": "modelo-a", "role": "arbitro", "judge_bucket": "arbiter", "normalized_judge_model": "juiz-arbitro", "status": "success", "score": 4, "question_sequence": 77},
        {"source": "AV3", "dataset": "J2", "question_id": 77, "comparable_candidate_model_id": 8, "comparable_candidate_model": "modelo-a", "role": "principal", "judge_bucket": "primary", "normalized_judge_model": "M-Prometheus", "status": "success", "score": 4, "question_sequence": 77},
        {"source": "AV3", "dataset": "J2", "question_id": 77, "comparable_candidate_model_id": 8, "comparable_candidate_model": "modelo-a", "role": "controle", "judge_bucket": "primary", "normalized_judge_model": "juiz-b", "status": "success", "score": 4, "question_sequence": 77},
    ]

    payload = build_comparative_dashboard_payload(
        rows,
        filters=DashboardFilters(dataset="J2"),
        agreement_rows=agreement_rows,
    )

    assert payload["cards"]["judge_agreement_comparison"] == {
        "comparable_answers_with_multiple_judges": 1,
        "comparable_pair_observations": 1,
        "av2_exact_agreement_rate": 0.0,
        "av3_exact_agreement_rate": 100.0,
        "delta_exact_agreement_rate": 100.0,
        "av2_light_divergence_rate": 100.0,
        "av3_light_divergence_rate": 0.0,
        "av2_strong_divergence_rate": 0.0,
        "av3_strong_divergence_rate": 0.0,
        "delta_strong_divergence_rate": 0.0,
        "av2_mean_absolute_delta": 1.0,
        "av3_mean_absolute_delta": 0.0,
        "av2_arbiter_rate": 100.0,
        "av3_arbiter_rate": None,
        "av2_arbiter_available": True,
        "av3_arbiter_available": False,
        "av2_arbiter_consistency_rate": 100.0,
        "av3_arbiter_consistency_rate": None,
    }
    assert payload["tables"]["judge_agreement_by_pair"] == [
        {
            "judge_pair": "M-Prometheus x juiz-b",
            "av2_comparable_evaluations": 1,
            "av2_exact_agreement_rate": 0.0,
            "av2_strong_divergence_rate": 0.0,
            "av3_comparable_evaluations": 1,
            "av3_exact_agreement_rate": 100.0,
            "av3_strong_divergence_rate": 0.0,
            "delta_exact_agreement_rate": 100.0,
            "delta_strong_divergence_rate": 0.0,
        }
    ]
    assert payload["tables"]["judge_agreement_by_candidate_model"] == [
        {
            "candidate_model": "modelo-a",
            "av2_exact_agreement_rate": 0.0,
            "av3_exact_agreement_rate": 100.0,
            "delta_agreement_rate": 100.0,
            "av2_strong_divergence_rate": 0.0,
            "av3_strong_divergence_rate": 0.0,
            "delta_strong_divergence_rate": 0.0,
            "comparable_pairs": 1,
        }
    ]


def test_comparative_dashboard_payload_reports_single_judge_and_missing_arbiter_empty_state() -> None:
    rows = [
        {
            "dataset": "J1",
            "question_id": 71,
            "av2_model_name": "modelo-a",
            "comparable_candidate_model": "modelo-a",
            "candidate_owner": "Equipe A",
            "av3_provider_model_id": "provider/modelo-a",
            "normalized_judge_model": "M-Prometheus",
            "av2_score": 4,
            "av3_score": 4,
            "av2_candidate_answer": "texto",
            "av3_candidate_answer": "texto",
            "av2_reference_answer": "rubrica",
            "av3_reference_answer": "rubrica",
            "metadata": {},
            "question_sequence": 71,
        }
    ]
    agreement_rows = [
        {"source": "AV2", "dataset": "J1", "question_id": 71, "comparable_candidate_model_id": 8, "comparable_candidate_model": "modelo-a", "role": "principal", "judge_bucket": "primary", "normalized_judge_model": "M-Prometheus", "status": "success", "score": 4, "question_sequence": 71},
        {"source": "AV3", "dataset": "J1", "question_id": 71, "comparable_candidate_model_id": 8, "comparable_candidate_model": "modelo-a", "role": "", "judge_bucket": "primary", "normalized_judge_model": "M-Prometheus", "status": "success", "score": 4, "question_sequence": 71},
    ]

    payload = build_comparative_dashboard_payload(
        rows,
        filters=DashboardFilters(dataset="J1"),
        agreement_rows=agreement_rows,
    )

    assert payload["cards"]["judge_agreement_comparison"]["comparable_answers_with_multiple_judges"] == 0
    assert payload["cards"]["judge_agreement_comparison"]["comparable_pair_observations"] == 0
    assert payload["cards"]["judge_agreement_comparison"]["av2_arbiter_available"] is False
    assert payload["cards"]["judge_agreement_comparison"]["av3_arbiter_available"] is False
    assert (
        payload["methodology"]["judge_agreement_empty_state"]
        == "As respostas comparaveis em AV2 nao possuem pelo menos dois juizes primarios com nota."
    )


def test_comparative_dashboard_payload_excludes_arbiter_rows_from_metrics_and_diagnostics() -> None:
    rows = [
        {
            "dataset": "J1",
            "question_id": 71,
            "comparable_candidate_model": "modelo-a",
            "normalized_judge_model": "M-Prometheus",
            "av2_score": 4,
            "av3_score": 5,
            "av2_candidate_answer": "a2",
            "av3_candidate_answer": "a3",
            "av2_reference_answer": "rubrica",
            "av3_reference_answer": "rubrica",
        },
        {
            "dataset": "J1",
            "question_id": 71,
            "comparable_candidate_model": "modelo-a",
            "normalized_judge_model": "meta-llama/Llama-3.3-70B-Instruct",
            "av2_score": 4,
            "av3_score": None,
            "av2_candidate_answer": "a2",
            "av3_candidate_answer": None,
            "av2_reference_answer": "rubrica",
            "av3_reference_answer": None,
        },
        {
            "dataset": "J1",
            "question_id": 71,
            "comparable_candidate_model": "modelo-a",
            "normalized_judge_model": "openai/gpt-oss-120b",
            "av2_score": None,
            "av3_score": 3,
            "av2_candidate_answer": None,
            "av3_candidate_answer": "arb",
            "av2_reference_answer": None,
            "av3_reference_answer": "rubrica",
        },
    ]

    payload = build_comparative_dashboard_payload(rows, filters=DashboardFilters(dataset="J1"))

    assert payload["cards"]["comparable_pairs"] == 1
    assert payload["cards"]["comparative_coverage"] == {"evaluated": 1, "expected": 2, "percent": 50.0}
    assert payload["cards"]["pairing_diagnostics"] == {
        "complete_pairs": 1,
        "av2_only_tuples": 1,
        "av3_only_tuples": 0,
        "filtered_dataset": "J1",
        "filtered_candidate_models": [],
        "filtered_judge_models": [],
        "normalized_judge_models": ["M-Prometheus", "meta-llama/Llama-3.3-70B-Instruct"],
    }


def test_comparative_dashboard_payload_keeps_complete_pairs_when_other_models_questions_or_judges_are_missing() -> None:
    rows = [
        {
            "dataset": "J1",
            "question_id": 71,
            "comparable_candidate_model": "modelo-a",
            "normalized_judge_model": "Llama",
            "av2_score": 4,
            "av3_score": 5,
            "av2_candidate_answer": "resposta a2",
            "av3_candidate_answer": "resposta a3",
            "av2_reference_answer": "rubrica",
            "av3_reference_answer": "rubrica",
        },
        {
            "dataset": "J1",
            "question_id": 72,
            "comparable_candidate_model": "modelo-a",
            "normalized_judge_model": "Llama",
            "av2_score": 3,
            "av3_score": 4,
            "av2_candidate_answer": "resposta a2",
            "av3_candidate_answer": "resposta a3",
            "av2_reference_answer": "rubrica",
            "av3_reference_answer": "rubrica",
        },
        {
            "dataset": "J1",
            "question_id": 71,
            "comparable_candidate_model": "modelo-b",
            "normalized_judge_model": "Llama",
            "av2_score": 2,
            "av3_score": None,
            "av2_candidate_answer": "somente av2",
            "av3_candidate_answer": None,
            "av2_reference_answer": "rubrica",
            "av3_reference_answer": None,
        },
        {
            "dataset": "J1",
            "question_id": 745,
            "comparable_candidate_model": "modelo-a",
            "normalized_judge_model": "Llama",
            "av2_score": 1,
            "av3_score": None,
            "av2_candidate_answer": "falha av3",
            "av3_candidate_answer": None,
            "av2_reference_answer": "rubrica",
            "av3_reference_answer": None,
        },
        {
            "dataset": "J1",
            "question_id": 71,
            "comparable_candidate_model": "modelo-a",
            "normalized_judge_model": "M-Prometheus",
            "av2_score": 5,
            "av3_score": None,
            "av2_candidate_answer": "prometheus av2",
            "av3_candidate_answer": None,
            "av2_reference_answer": "rubrica",
            "av3_reference_answer": None,
        },
        {
            "dataset": "J1",
            "question_id": 999,
            "comparable_candidate_model": "modelo-c",
            "normalized_judge_model": "Llama",
            "av2_score": None,
            "av3_score": 2,
            "av2_candidate_answer": None,
            "av3_candidate_answer": "somente av3",
            "av2_reference_answer": None,
            "av3_reference_answer": "rubrica",
        },
    ]

    payload = build_comparative_dashboard_payload(
        rows,
        filters=DashboardFilters(dataset="J1", candidate_models=("modelo-a",), judge_models=("Llama",)),
    )

    assert payload["cards"]["comparable_pairs"] == 2
    assert payload["cards"]["comparative_coverage"] == {"evaluated": 2, "expected": 6, "percent": 33.3}
    assert payload["cards"]["av2_average_score"] == 3.5
    assert payload["cards"]["av3_average_score"] == 4.5
    assert payload["cards"]["pairing_diagnostics"] == {
        "complete_pairs": 2,
        "av2_only_tuples": 3,
        "av3_only_tuples": 1,
        "filtered_dataset": "J1",
        "filtered_candidate_models": ["modelo-a"],
        "filtered_judge_models": ["Llama"],
        "normalized_judge_models": ["Llama", "M-Prometheus"],
    }


def test_comparative_dashboard_payload_reports_empty_state_reason_with_diagnostics() -> None:
    rows = [
        {
            "dataset": "J1",
            "question_id": 71,
            "comparable_candidate_model": "modelo-a",
            "normalized_judge_model": "Llama",
            "av2_score": 4,
            "av3_score": None,
            "av2_candidate_answer": "resposta a2",
            "av3_candidate_answer": None,
            "av2_reference_answer": "rubrica",
            "av3_reference_answer": None,
        },
        {
            "dataset": "J1",
            "question_id": 72,
            "comparable_candidate_model": "modelo-b",
            "normalized_judge_model": "M-Prometheus",
            "av2_score": None,
            "av3_score": 5,
            "av2_candidate_answer": None,
            "av3_candidate_answer": "resposta a3",
            "av2_reference_answer": None,
            "av3_reference_answer": "rubrica",
        },
    ]

    payload = build_comparative_dashboard_payload(rows, filters=DashboardFilters(dataset="J1"))

    assert payload["cards"]["comparable_pairs"] == 0
    assert payload["cards"]["pairing_diagnostics"]["av2_only_tuples"] == 1
    assert payload["cards"]["pairing_diagnostics"]["av3_only_tuples"] == 1
    assert payload["methodology"]["empty_state_note"] == "Nenhum par completo restou apos os filtros."


def test_comparative_dashboard_specialties_fall_back_when_metadata_is_missing() -> None:
    rows = [
        {
            "dataset": "J1",
            "question_id": 71,
            "comparable_candidate_model": "modelo-a",
            "normalized_judge_model": "Llama",
            "av2_score": 3,
            "av3_score": 4,
            "av2_candidate_answer": "resposta a2",
            "av3_candidate_answer": "resposta a3",
            "av2_reference_answer": "rubrica",
            "av3_reference_answer": "rubrica",
            "metadata": {},
            "question_sequence": 71,
        }
    ]

    payload = build_comparative_dashboard_payload(rows, filters=DashboardFilters(dataset="J1"))

    assert payload["tables"]["specialties"] == [
        {
            "legal_specialty": "Sem especialidade",
            "paired_evaluations": 1,
            "av2_average_score": 3.0,
            "av3_average_score": 4.0,
            "delta_average": 1.0,
            "improvement_rate": 100.0,
            "regression_rate": 0.0,
            "unchanged_rate": 0.0,
            "best_model_by_delta": {"candidate_model": "modelo-a", "delta_average": 1.0},
            "worst_model_by_delta": {"candidate_model": "modelo-a", "delta_average": 1.0},
        }
    ]


def test_comparative_dashboard_spearman_returns_values_when_reference_exists() -> None:
    rows = [
        {
            "dataset": "J2",
            "question_id": 77,
            "comparable_candidate_model": "modelo-a",
            "normalized_judge_model": "M-Prometheus",
            "av2_score": 1,
            "av3_score": 1,
            "av2_candidate_answer": "A",
            "av3_candidate_answer": "A",
            "av2_reference_answer": "B",
            "av3_reference_answer": "B",
            "metadata": {"question_number": 77},
        },
        {
            "dataset": "J2",
            "question_id": 78,
            "comparable_candidate_model": "modelo-a",
            "normalized_judge_model": "M-Prometheus",
            "av2_score": 5,
            "av3_score": 5,
            "av2_candidate_answer": "C",
            "av3_candidate_answer": "C",
            "av2_reference_answer": "C",
            "av3_reference_answer": "C",
            "metadata": {"question_number": 78},
        },
    ]

    payload = build_comparative_dashboard_payload(rows, filters=DashboardFilters(dataset="J2"))

    overall = payload["tables"]["spearman_breakdowns"]["overall"][0]
    assert overall["reference_pairs_av2"] == 2
    assert overall["reference_pairs_av3"] == 2
    assert overall["spearman_av2"]["available"] is True
    assert overall["spearman_av2"]["value"] == 1.0
    assert overall["spearman_av3"]["available"] is True
    assert overall["spearman_delta"]["available"] is True
    assert payload["tables"]["spearman_breakdowns"]["by_judge_model"][0]["label"] == "M-Prometheus"


def test_comparative_dashboard_spearman_renders_na_when_reference_is_absent() -> None:
    rows = [
        {
            "dataset": "J1",
            "question_id": 71,
            "comparable_candidate_model": "modelo-a",
            "normalized_judge_model": "Llama",
            "av2_score": 2,
            "av3_score": 4,
            "av2_candidate_answer": "texto",
            "av3_candidate_answer": "texto",
            "av2_reference_answer": "rubrica",
            "av3_reference_answer": "rubrica",
            "metadata": {},
        }
    ]

    payload = build_comparative_dashboard_payload(rows, filters=DashboardFilters(dataset="J1"))

    overall = payload["tables"]["spearman_breakdowns"]["overall"][0]
    assert overall["spearman_av2"]["available"] is False
    assert overall["spearman_av3"]["available"] is False
    assert overall["spearman_delta"]["available"] is False
    assert overall["spearman_av2"]["note"] == "Referência humana/gabarito/rubrica indisponível para o filtro selecionado."


def test_comparative_dashboard_prometheus_normalization_is_grouping_only() -> None:
    rows = [
        {
            "dataset": "J1",
            "question_id": 10,
            "av2_model_name": "modelo-a",
            "comparable_candidate_model": "modelo-a",
            "candidate_owner": "Equipe A",
            "av3_provider_model_id": "provider/modelo-a",
            "normalized_judge_model": "M-Prometheus",
            "av2_raw_judge_model": "Unbabel/M-Prometheus-14B",
            "av3_raw_judge_model": "Unbabel/M-Prometheus-7B",
            "av2_score": 4,
            "av3_score": 5,
            "av2_candidate_answer": "texto",
            "av3_candidate_answer": "texto",
            "av2_reference_answer": "rubrica",
            "av3_reference_answer": "rubrica",
            "metadata": {"especialidade": "direito constitucional"},
            "question_sequence": 10,
        }
    ]

    payload = build_comparative_dashboard_payload(rows, filters=DashboardFilters(dataset="J1"))

    assert payload["cards"]["comparable_pairs"] == 1
    assert payload["methodology"]["judge_normalization"].endswith("apenas no dashboard.")
    assert rows[0]["av2_raw_judge_model"] == "Unbabel/M-Prometheus-14B"
    assert rows[0]["av3_raw_judge_model"] == "Unbabel/M-Prometheus-7B"
    assert rows[0]["normalized_judge_model"] == "M-Prometheus"


def test_comparative_dashboard_ranking_sorts_by_delta_desc_and_delta_cases_apply_thresholds() -> None:
    rows = [
        {
            "dataset": "J1",
            "question_id": 11,
            "question_sequence": 11,
            "av2_model_name": "modelo-a",
            "comparable_candidate_model": "modelo-a",
            "candidate_owner": "Equipe A",
            "av3_provider_model_id": "provider/modelo-a",
            "normalized_judge_model": "M-Prometheus",
            "av2_score": 2,
            "av3_score": 5,
            "av2_candidate_answer": "texto",
            "av3_candidate_answer": "texto",
            "av2_reference_answer": "rubrica",
            "av3_reference_answer": "rubrica",
            "metadata": {"especialidade": "direito tributario"},
        },
        {
            "dataset": "J1",
            "question_id": 12,
            "question_sequence": 12,
            "av2_model_name": "modelo-a",
            "comparable_candidate_model": "modelo-a",
            "candidate_owner": "Equipe A",
            "av3_provider_model_id": "provider/modelo-a",
            "normalized_judge_model": "Llama",
            "av2_score": 4,
            "av3_score": 1,
            "av2_candidate_answer": "texto",
            "av3_candidate_answer": "texto",
            "av2_reference_answer": "rubrica",
            "av3_reference_answer": "rubrica",
            "metadata": {"especialidade": "direito penal"},
        },
        {
            "dataset": "J1",
            "question_id": 13,
            "question_sequence": 13,
            "av2_model_name": "modelo-b",
            "comparable_candidate_model": "modelo-b",
            "candidate_owner": "Equipe B",
            "av3_provider_model_id": "provider/modelo-b",
            "normalized_judge_model": "Llama",
            "av2_score": 3,
            "av3_score": 4,
            "av2_candidate_answer": "texto",
            "av3_candidate_answer": "texto",
            "av2_reference_answer": "rubrica",
            "av3_reference_answer": "rubrica",
            "metadata": {"especialidade": "direito civil"},
        },
    ]

    payload = build_comparative_dashboard_payload(rows, filters=DashboardFilters(dataset="J1", candidate_models=("modelo-a",)))

    ranking = payload["tables"]["ranking_by_model"]
    assert [row["comparable_candidate_model"] for row in ranking] == ["modelo-b", "modelo-a"]
    assert ranking[0]["delta_average"] == 1.0
    assert ranking[1]["delta_average"] == 0.0
    assert payload["tables"]["largest_gains"] == [
        {
            "dataset": "J1",
            "question_id": 11,
            "question_sequence": 11,
            "candidate_model": "modelo-a",
            "normalized_judge_model": "M-Prometheus",
            "av2_score": 2,
            "av3_score": 5,
            "delta": 3,
            "legal_specialty": "Direito Tributario",
        }
    ]
    assert payload["tables"]["largest_regressions"] == [
        {
            "dataset": "J1",
            "question_id": 12,
            "question_sequence": 12,
            "candidate_model": "modelo-a",
            "normalized_judge_model": "Llama",
            "av2_score": 4,
            "av3_score": 1,
            "delta": -3,
            "legal_specialty": "Direito Penal",
        }
    ]


def test_dashboard_calculates_j2_primary_spearman_from_answer_key() -> None:
    rows = [
        _row(evaluation_id=1, answer_id=1, dataset="J2", candidate_answer="B", reference_answer="B", score=5),
        _row(evaluation_id=2, answer_id=2, dataset="J2", candidate_answer="C", reference_answer="B", score=1),
        _row(evaluation_id=3, answer_id=3, dataset="J2", candidate_answer="D", reference_answer="D", score=5),
        _row(evaluation_id=4, answer_id=4, dataset="J2", candidate_answer="A", reference_answer="D", score=1),
    ]

    payload = build_dashboard_payload(rows, expected_answers=4, filters=DashboardFilters(dataset="J2"))

    spearman_card = payload["cards"]["spearman_reference"]
    assert spearman_card["available"] is True
    assert spearman_card["value"] == 1.0
    assert spearman_card["p_value"] == 0.0
    assert payload["cards"]["coverage"] == {"evaluated": 4, "expected": 4, "percent": 100.0}
    assert payload["charts"]["reference_alignment"]["points"] == [
        {
            "evaluation_id": 1,
            "answer_id": 1,
            "question_id": 101,
            "dataset": "J2",
            "candidate_model": "modelo-a",
            "judge_model": "juiz-a",
            "reference_score": 5.0,
            "judge_score": 5,
        },
        {
            "evaluation_id": 2,
            "answer_id": 2,
            "question_id": 102,
            "dataset": "J2",
            "candidate_model": "modelo-a",
            "judge_model": "juiz-a",
            "reference_score": 1.0,
            "judge_score": 1,
        },
        {
            "evaluation_id": 3,
            "answer_id": 3,
            "question_id": 103,
            "dataset": "J2",
            "candidate_model": "modelo-a",
            "judge_model": "juiz-a",
            "reference_score": 5.0,
            "judge_score": 5,
        },
        {
            "evaluation_id": 4,
            "answer_id": 4,
            "question_id": 104,
            "dataset": "J2",
            "candidate_model": "modelo-a",
            "judge_model": "juiz-a",
            "reference_score": 1.0,
            "judge_score": 1,
        },
    ]


def test_dashboard_marks_j1_primary_spearman_unavailable_without_reference_score() -> None:
    rows = [
        _row(evaluation_id=1, answer_id=1, dataset="J1", candidate_answer="texto", reference_answer="rubrica", score=4),
        _row(evaluation_id=2, answer_id=2, dataset="J1", candidate_answer="texto", reference_answer="rubrica", score=2),
    ]

    payload = build_dashboard_payload(rows, expected_answers=2, filters=DashboardFilters(dataset="J1"))

    spearman_card = payload["cards"]["spearman_reference"]
    assert spearman_card["available"] is False
    assert spearman_card["value"] is None
    assert spearman_card["p_value"] is None
    assert "J1" in spearman_card["note"]


def test_dashboard_reports_judge_arbiter_as_complementary_consistency() -> None:
    rows = [
        _row(evaluation_id=1, answer_id=1, dataset="J1", candidate_answer="texto", reference_answer="rubrica", score=2),
        _row(
            evaluation_id=2,
            answer_id=1,
            dataset="J1",
            candidate_answer="texto",
            reference_answer="rubrica",
            score=2,
            role="arbitro",
            judge_model="arbitro",
        ),
        _row(evaluation_id=3, answer_id=2, dataset="J1", candidate_answer="texto", reference_answer="rubrica", score=5),
        _row(
            evaluation_id=4,
            answer_id=2,
            dataset="J1",
            candidate_answer="texto",
            reference_answer="rubrica",
            score=5,
            role="arbitro",
            judge_model="arbitro",
        ),
    ]

    payload = build_dashboard_payload(rows, expected_answers=2, filters=DashboardFilters(dataset="J1"))

    assert payload["cards"]["spearman_reference"]["available"] is False
    consistency = payload["cards"]["judge_arbiter_consistency"]
    assert consistency["available"] is True
    assert consistency["value"] == 1.0
    assert "Meta-avaliação" in consistency["note"]


def test_dashboard_reports_score_distribution_by_candidate_model() -> None:
    rows = [
        _row(evaluation_id=1, answer_id=1, dataset="J1", candidate_answer="texto", reference_answer="rubrica", score=1, candidate_model="modelo-a"),
        _row(evaluation_id=2, answer_id=2, dataset="J1", candidate_answer="texto", reference_answer="rubrica", score=5, candidate_model="modelo-a"),
        _row(evaluation_id=3, answer_id=3, dataset="J1", candidate_answer="texto", reference_answer="rubrica", score=4, candidate_model="modelo-a"),
        _row(evaluation_id=4, answer_id=4, dataset="J1", candidate_answer="texto", reference_answer="rubrica", score=3, candidate_model="modelo-b"),
        _row(evaluation_id=5, answer_id=5, dataset="J1", candidate_answer="texto", reference_answer="rubrica", score=3, candidate_model="modelo-b"),
    ]

    payload = build_dashboard_payload(rows, expected_answers=5, filters=DashboardFilters(dataset="J1"))

    assert payload["charts"]["score_distribution_by_model"] == [
        {"label": "modelo-a", "total": 3, "average": 3.33, "scores": {"1": 1, "2": 0, "3": 0, "4": 1, "5": 1}},
        {"label": "modelo-b", "total": 2, "average": 3, "scores": {"1": 0, "2": 0, "3": 2, "4": 0, "5": 0}},
    ]


def test_dashboard_reports_rubric_dimension_heatmap_by_candidate_model() -> None:
    rows = [
        {
            **_row(evaluation_id=1, answer_id=1, dataset="J1", candidate_answer="texto", reference_answer="rubrica", score=5, candidate_model="modelo-a"),
            "argumentacao_score": 4.5,
            "precisao_score": 4,
            "coesao_legal_score": 5,
            "total_score": 4.5,
        },
        {
            **_row(evaluation_id=2, answer_id=2, dataset="J1", candidate_answer="texto", reference_answer="rubrica", score=4, candidate_model="modelo-a"),
            "argumentacao_score": 3.5,
            "precisao_score": 4,
            "coesao_legal_score": 4,
            "total_score": 4,
        },
        {
            **_row(evaluation_id=3, answer_id=3, dataset="J1", candidate_answer="texto", reference_answer="rubrica", score=2, candidate_model="modelo-b"),
            "argumentacao_score": 2,
            "precisao_score": 2.5,
            "coesao_legal_score": 3,
            "total_score": 2.5,
        },
    ]

    payload = build_dashboard_payload(rows, expected_answers=3, filters=DashboardFilters(dataset="J1"))

    assert payload["charts"]["rubric_heatmap"] == {
        "columns": ["Argumentação", "Precisão", "Coesão legal", "Total"],
        "rows": [
            {"label": "modelo-a", "values": [4.0, 4.0, 4.5, 4.25], "count": 2},
            {"label": "modelo-b", "values": [2.0, 2.5, 3.0, 2.5], "count": 1},
        ],
    }


def test_dashboard_reports_candidate_average_by_legal_specialty() -> None:
    rows = [
        {
            **_row(evaluation_id=1, answer_id=1, dataset="J1", candidate_answer="texto", reference_answer="rubrica", score=5, candidate_model="modelo-a"),
            "metadata": {"category": "39_direito_administrativo"},
        },
        {
            **_row(evaluation_id=2, answer_id=2, dataset="J1", candidate_answer="texto", reference_answer="rubrica", score=3, candidate_model="modelo-a"),
            "metadata": {"category": "39_direito_administrativo"},
        },
        {
            **_row(evaluation_id=3, answer_id=3, dataset="J1", candidate_answer="texto", reference_answer="rubrica", score=4, candidate_model="modelo-b"),
            "metadata": {"category": "39_direito_administrativo"},
        },
        {
            **_row(evaluation_id=4, answer_id=4, dataset="J1", candidate_answer="texto", reference_answer="rubrica", score=2, candidate_model="modelo-a"),
            "metadata": {"especialidade": "direito tributario"},
        },
    ]

    payload = build_dashboard_payload(rows, expected_answers=4, filters=DashboardFilters(dataset="J1"))

    assert payload["charts"]["legal_specialty_performance"] == {
        "columns": ["modelo-a", "modelo-b"],
        "rows": [
            {"label": "Direito Administrativo", "values": [4.0, 4.0], "count": 3, "average": 4.0},
            {"label": "Direito Tributario", "values": [2.0, None], "count": 1, "average": 2.0},
        ],
    }


def test_dashboard_keeps_j1_legal_specialty_from_existing_metadata() -> None:
    rows = [
        {
            **_row(evaluation_id=1, answer_id=1, dataset="J1", candidate_answer="texto", reference_answer="rubrica", score=5),
            "metadata": {"category": "39_direito_administrativo", "tipo_questao": "CIVIL"},
        },
    ]

    payload = build_dashboard_payload(rows, expected_answers=1, filters=DashboardFilters(dataset="J1"))

    assert payload["charts"]["legal_specialty_performance"]["rows"] == [
        {"label": "Direito Administrativo", "values": [5.0], "count": 1, "average": 5.0},
    ]


def test_dashboard_maps_j2_question_type_to_existing_legal_specialties() -> None:
    rows = [
        {
            **_row(evaluation_id=1, answer_id=1, dataset="J2", candidate_answer="A", reference_answer="A", score=5, candidate_model="modelo-a"),
            "metadata": {"tipo_questao": "CIVIL-PROCEDURE"},
        },
        {
            **_row(evaluation_id=2, answer_id=2, dataset="J2", candidate_answer="B", reference_answer="B", score=3, candidate_model="modelo-a"),
            "metadata": {"tipo_questao": "LABOUR"},
        },
        {
            **_row(evaluation_id=3, answer_id=3, dataset="J2", candidate_answer="C", reference_answer="C", score=4, candidate_model="modelo-a"),
            "metadata": {"tipo_questao": "BUSINESS"},
        },
        {
            **_row(evaluation_id=4, answer_id=4, dataset="J2", candidate_answer="D", reference_answer="D", score=2, candidate_model="modelo-a"),
            "metadata": {"tipo_questao": "CRIMINAL-PROCEDURE"},
        },
        {
            **_row(evaluation_id=5, answer_id=5, dataset="J2", candidate_answer="A", reference_answer="A", score=1, candidate_model="modelo-a"),
            "metadata": {"tipo_questao": "TAXES"},
        },
        {
            **_row(evaluation_id=6, answer_id=6, dataset="J2", candidate_answer="B", reference_answer="B", score=5, candidate_model="modelo-a"),
            "metadata": {"tipo_questao": "CONSTITUTIONAL"},
        },
        {
            **_row(evaluation_id=7, answer_id=7, dataset="J2", candidate_answer="C", reference_answer="C", score=3, candidate_model="modelo-a"),
            "metadata": {"tipo_questao": "ENVIRONMENTAL"},
        },
    ]

    payload = build_dashboard_payload(rows, expected_answers=7, filters=DashboardFilters(dataset="J2"))

    assert payload["charts"]["legal_specialty_performance"]["rows"] == [
        {"label": "Direito Civil", "values": [5.0], "count": 1, "average": 5.0},
        {"label": "Direito Constitucional", "values": [5.0], "count": 1, "average": 5.0},
        {"label": "Direito Empresarial", "values": [4.0], "count": 1, "average": 4.0},
        {"label": "Direito Administrativo", "values": [3.0], "count": 1, "average": 3.0},
        {"label": "Direito Do Trabalho", "values": [3.0], "count": 1, "average": 3.0},
        {"label": "Direito Penal", "values": [2.0], "count": 1, "average": 2.0},
        {"label": "Direito Tributario", "values": [1.0], "count": 1, "average": 1.0},
    ]


def test_dashboard_maps_j2_missing_question_type_from_question_number() -> None:
    rows = [
        {
            **_row(evaluation_id=1, answer_id=1, dataset="J2", candidate_answer="A", reference_answer="A", score=5),
            "metadata": {"tipo_questao": None, "question_number": 79},
        },
    ]

    payload = build_dashboard_payload(rows, expected_answers=1, filters=DashboardFilters(dataset="J2"))

    assert payload["charts"]["legal_specialty_performance"]["rows"] == [
        {"label": "Direito Do Trabalho", "values": [5.0], "count": 1, "average": 5.0},
    ]


def test_dashboard_reports_candidate_average_by_difficulty_ordered_by_complexity() -> None:
    rows = [
        {
            **_row(evaluation_id=1, answer_id=1, dataset="J1", candidate_answer="texto", reference_answer="rubrica", score=5, candidate_model="modelo-a"),
            "metadata": {"difficulty": "easy"},
        },
        {
            **_row(evaluation_id=2, answer_id=2, dataset="J1", candidate_answer="texto", reference_answer="rubrica", score=4, candidate_model="modelo-a"),
            "metadata": {"dificuldade": "Médio"},
        },
        {
            **_row(evaluation_id=3, answer_id=3, dataset="J1", candidate_answer="texto", reference_answer="rubrica", score=2, candidate_model="modelo-a"),
            "metadata": {"complexidade": "muito_dificil"},
        },
        {
            **_row(evaluation_id=4, answer_id=4, dataset="J1", candidate_answer="texto", reference_answer="rubrica", score=3, candidate_model="modelo-b"),
            "metadata": {"difficulty": "hard"},
        },
        {
            **_row(evaluation_id=5, answer_id=5, dataset="J1", candidate_answer="texto", reference_answer="rubrica", score=1, candidate_model="modelo-b"),
            "metadata": {},
        },
    ]

    payload = build_dashboard_payload(rows, expected_answers=5, filters=DashboardFilters(dataset="J1"))

    assert payload["charts"]["difficulty_performance"] == {
        "x_label": "dificuldade",
        "y_label": "média da nota",
        "difficulties": ["Fácil", "Médio", "Difícil", "Muito difícil"],
        "series": [
            {"label": "modelo-a", "values": [5.0, 4.0, None, 2.0]},
            {"label": "modelo-b", "values": [None, None, 3.0, None]},
        ],
    }


def test_dashboard_reports_ordinal_confusion_matrix_and_error_highlights() -> None:
    rows = [
        _row(evaluation_id=1, answer_id=1, dataset="J2", candidate_answer="B", reference_answer="B", score=5),
        _row(evaluation_id=2, answer_id=2, dataset="J2", candidate_answer="C", reference_answer="B", score=5),
        _row(evaluation_id=3, answer_id=3, dataset="J2", candidate_answer="D", reference_answer="D", score=1),
        _row(evaluation_id=4, answer_id=4, dataset="J2", candidate_answer="A", reference_answer="D", score=2),
    ]

    payload = build_dashboard_payload(rows, expected_answers=4, filters=DashboardFilters(dataset="J2"))

    confusion = payload["charts"]["ordinal_confusion"]
    assert confusion["rows"] == ["Humano 1", "Humano 2", "Humano 3", "Humano 4", "Humano 5"]
    assert confusion["columns"] == ["Juiz 1", "Juiz 2", "Juiz 3", "Juiz 4", "Juiz 5"]
    assert confusion["matrix"] == [
        [0, 1, 0, 0, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 0, 0, 0, 1],
    ]
    assert confusion["total"] == 4
    assert confusion["highlights"][0] == {
        "label": "Humano baixo, juiz alto",
        "interpretation": "falso positivo grave",
        "count": 1,
        "share": 25.0,
    }
    assert confusion["highlights"][1]["interpretation"] == "falso negativo"
    assert confusion["highlights"][1]["count"] == 1
    assert confusion["important_cases"][0]["reason"] == "falso positivo grave"
    assert confusion["important_cases"][0]["reference_score"] == 1
    assert confusion["important_cases"][0]["judge_score"] == 5


def test_dashboard_reports_critical_error_analysis_categories_and_table() -> None:
    rows = [
        _row(evaluation_id=1, answer_id=1, dataset="J2", candidate_answer="C", reference_answer="B", score=5),
        _row(evaluation_id=2, answer_id=2, dataset="J2", candidate_answer="D", reference_answer="D", score=1),
        _row(
            evaluation_id=3,
            answer_id=3,
            dataset="J2",
            candidate_answer="B",
            reference_answer="B",
            score=4,
            rationale="A justificativa menciona lei inexistente para sustentar a conclusao.",
        ),
        _row(evaluation_id=4, answer_id=4, dataset="J2", candidate_answer="B", reference_answer="B", score=3, rationale="Correto."),
        _row(evaluation_id=5, answer_id=5, dataset="J2", candidate_answer="B", reference_answer="B", score=None, status="failed", rationale="invalid JSON"),
        _row(evaluation_id=6, answer_id=6, dataset="J2", candidate_answer="B", reference_answer="B", score=None, status="HTTP 504 timeout"),
        _row(evaluation_id=7, answer_id=7, dataset="J2", candidate_answer="B", reference_answer="B", score=5, judge_model="juiz-a", role="principal"),
        _row(evaluation_id=8, answer_id=7, dataset="J2", candidate_answer="B", reference_answer="B", score=2, judge_model="juiz-b", role="controle"),
    ]

    payload = build_dashboard_payload(rows, expected_answers=7, filters=DashboardFilters(dataset="J2"))

    chart = payload["charts"]["critical_error_categories"]
    assert chart == [
        {"label": "Nota alta para resposta errada", "value": 1},
        {"label": "Nota baixa para resposta correta", "value": 2},
        {"label": "Alucinacao normativa", "value": 1},
        {"label": "Resposta sem fundamentacao", "value": 1},
        {"label": "Divergencia entre juizes", "value": 1},
        {"label": "Erro de parsing", "value": 2},
        {"label": "Timeout/HTTP error", "value": 1},
    ]
    table = payload["tables"]["critical_error_analysis"]
    assert {
        "evaluation_id": 1,
        "question_id": 101,
        "candidate_model": "modelo-a",
        "judge_model": "juiz-a",
        "score": 5,
        "error_type": "Nota alta para resposta errada",
        "short_justification": "referencia 1, juiz 5",
        "log_url": None,
    } in table
    assert any(
        row["error_type"] == "Divergencia entre juizes" and "juiz-a(principal)=5" in row["short_justification"]
        for row in table
    )


def test_dashboard_reports_minor_disagreements_as_separate_telemetry() -> None:
    rows = [
        _row(
            evaluation_id=1,
            answer_id=1,
            dataset="J1",
            candidate_answer="texto",
            reference_answer="rubrica",
            score=4,
            judge_model="juiz-a",
            role="principal",
        ),
        _row(
            evaluation_id=2,
            answer_id=1,
            dataset="J1",
            candidate_answer="texto",
            reference_answer="rubrica",
            score=3,
            judge_model="juiz-b",
            role="controle",
        ),
        _row(
            evaluation_id=3,
            answer_id=2,
            dataset="J1",
            candidate_answer="texto",
            reference_answer="rubrica",
            score=4,
            judge_model="juiz-a",
            role="principal",
        ),
        _row(
            evaluation_id=4,
            answer_id=2,
            dataset="J1",
            candidate_answer="texto",
            reference_answer="rubrica",
            score=4,
            judge_model="juiz-b",
            role="controle",
        ),
    ]

    payload = build_dashboard_payload(rows, expected_answers=2, filters=DashboardFilters(dataset="J1"))

    assert payload["cards"]["minor_disagreements"] == 1
    assert payload["cards"]["audit_divergences"] == 0
    assert payload["tables"]["minor_disagreement_cases"][0]["answer_id"] == 1
    assert payload["tables"]["minor_disagreement_cases"][0]["reason"] == "delta 1 (leve)"


def test_dashboard_reports_judge_agreement_by_answer_and_arbitration_table() -> None:
    rows = [
        _row(evaluation_id=1, answer_id=1, dataset="J1", candidate_answer="texto", reference_answer="rubrica", score=4, role="principal", judge_model="juiz-1"),
        _row(evaluation_id=2, answer_id=1, dataset="J1", candidate_answer="texto", reference_answer="rubrica", score=4, role="controle", judge_model="juiz-2"),
        _row(evaluation_id=3, answer_id=2, dataset="J1", candidate_answer="texto", reference_answer="rubrica", score=5, role="principal", judge_model="juiz-1"),
        _row(evaluation_id=4, answer_id=2, dataset="J1", candidate_answer="texto", reference_answer="rubrica", score=4, role="controle", judge_model="juiz-2"),
        _row(evaluation_id=5, answer_id=3, dataset="J1", candidate_answer="texto", reference_answer="rubrica", score=5, role="principal", judge_model="juiz-1"),
        _row(evaluation_id=6, answer_id=3, dataset="J1", candidate_answer="texto", reference_answer="rubrica", score=2, role="controle", judge_model="juiz-2"),
        _row(
            evaluation_id=7,
            answer_id=3,
            dataset="J1",
            candidate_answer="texto",
            reference_answer="rubrica",
            score=4,
            role="arbitro",
            judge_model="arbitro",
        ),
        _row(evaluation_id=8, answer_id=4, dataset="J1", candidate_answer="texto", reference_answer="rubrica", score=1, role="principal", judge_model="juiz-1"),
        _row(evaluation_id=9, answer_id=4, dataset="J1", candidate_answer="texto", reference_answer="rubrica", score=5, role="controle", judge_model="juiz-2", status="failed"),
    ]

    payload = build_dashboard_payload(rows, expected_answers=4, filters=DashboardFilters(dataset="J1"))

    assert payload["cards"]["judge_agreement"] == {
        "total_compared": 3,
        "delta_0": 1,
        "delta_1": 1,
        "delta_2": 0,
        "delta_3": 1,
        "delta_4": 0,
        "arbiter_triggered": 1,
    }
    assert payload["tables"]["judge_agreement_arbitrations"] == [
        {
            "answer_id": 3,
            "question_id": 103,
            "candidate_model": "modelo-a",
            "judge_1_score": 5,
            "judge_2_score": 2,
            "delta": 3,
            "arbiter_score": 4,
            "arbitration_reason": "primary_panel",
        }
    ]


def test_dashboard_reports_judge_candidate_heatmap_and_disagreement_boxplot() -> None:
    rows = [
        _row(evaluation_id=1, answer_id=1, dataset="J1", candidate_answer="texto", reference_answer="rubrica", score=5, role="principal", judge_model="juiz-1", candidate_model="modelo-a"),
        _row(evaluation_id=2, answer_id=1, dataset="J1", candidate_answer="texto", reference_answer="rubrica", score=3, role="controle", judge_model="juiz-2", candidate_model="modelo-a"),
        _row(evaluation_id=3, answer_id=1, dataset="J1", candidate_answer="texto", reference_answer="rubrica", score=4, role="arbitro", judge_model="arbitro", candidate_model="modelo-a"),
        _row(evaluation_id=4, answer_id=2, dataset="J1", candidate_answer="texto", reference_answer="rubrica", score=4, role="principal", judge_model="juiz-1", candidate_model="modelo-a"),
        _row(evaluation_id=5, answer_id=2, dataset="J1", candidate_answer="texto", reference_answer="rubrica", score=4, role="controle", judge_model="juiz-2", candidate_model="modelo-a"),
        _row(evaluation_id=6, answer_id=3, dataset="J1", candidate_answer="texto", reference_answer="rubrica", score=2, role="principal", judge_model="juiz-1", candidate_model="modelo-b"),
        _row(evaluation_id=7, answer_id=3, dataset="J1", candidate_answer="texto", reference_answer="rubrica", score=5, role="controle", judge_model="juiz-2", candidate_model="modelo-b"),
        _row(evaluation_id=8, answer_id=4, dataset="J1", candidate_answer="texto", reference_answer="rubrica", score=None, role="controle", judge_model="juiz-2", candidate_model="modelo-b", status="failed"),
    ]

    payload = build_dashboard_payload(rows, expected_answers=4, filters=DashboardFilters(dataset="J1"))

    assert payload["charts"]["judge_candidate_heatmap"] == {
        "columns": ["modelo-a", "modelo-b"],
        "rows": [
            {"label": "arbitro", "values": [4.0, None], "count": 1},
            {"label": "juiz-1", "values": [4.5, 2.0], "count": 3},
            {"label": "juiz-2", "values": [3.5, 5.0], "count": 3},
        ],
    }
    assert payload["charts"]["judge_disagreement_boxplot"] == {
        "metric": "judge_disagreement",
        "audit_threshold": 2,
        "rows": [
            {"label": "modelo-a", "count": 2, "audit_count": 1, "min": 0, "q1": 0.5, "median": 1.0, "q3": 1.5, "max": 2},
            {"label": "modelo-b", "count": 1, "audit_count": 1, "min": 3, "q1": 3.0, "median": 3.0, "q3": 3.0, "max": 3},
        ],
    }
