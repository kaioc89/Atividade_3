from __future__ import annotations

from atividade_2.dashboard import DashboardFilters, build_dashboard_payload, spearman


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
