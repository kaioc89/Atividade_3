from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from atividade_2.assistant import (
    DEFAULT_OUT_OF_SCOPE_ANSWER,
    DEFAULT_SUGGESTIONS,
    AssistantService,
    is_factual_human_audit_query,
)
from atividade_2.web import create_app


class FakeLlmClient:
    def __init__(self, answer: str = "Resumo gerado pelo assistente.") -> None:
        self.answer = answer
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answer


class SequencedFakeLlmClient:
    def __init__(self, answers: list[str]) -> None:
        self.answers = answers
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.answers:
            raise AssertionError("Unexpected LLM call.")
        return self.answers.pop(0)


class FakeDashboardService:
    def __init__(self) -> None:
        self.calls = 0

    def load(self, filters) -> dict:
        self.calls += 1
        candidate_models = getattr(filters, "candidate_models", ())
        rankings_by_dataset = {
            "J1": [{"label": "Grok", "value": 4.21, "count": 24}],
            "J2": [{"label": "Gemini", "value": 4.84, "count": 246}],
        }
        spearman_values = {
            ("J1", "Grok"): {
                "value": None,
                "p_value": None,
                "sample_size": 0,
                "available": False,
                "note": "J1 sem referência ordinal.",
            },
            ("J2", "Gemini"): {
                "value": 1.0,
                "p_value": 0.0,
                "sample_size": 246,
                "available": True,
                "note": "Calculado para Gemini em J2.",
            },
        }
        spearman_reference = (
            spearman_values.get((filters.dataset, candidate_models[0]))
            if candidate_models
            else {
                "value": 0.9973 if filters.dataset == "J2" else None,
                "p_value": 0.0 if filters.dataset == "J2" else None,
                "sample_size": 4428 if filters.dataset == "J2" else 0,
                "available": filters.dataset == "J2",
                "note": "Calculado para J2" if filters.dataset == "J2" else "J1 sem referência ordinal.",
            }
        )
        return {
            "filters": {"dataset": filters.dataset},
            "cards": {
                "evaluations": 3,
                "coverage": {"evaluated": 2, "expected": 4, "percent": 50.0},
                "success_rate": 100.0,
                "average_score": 4.5,
                "spearman_reference": spearman_reference,
                "judge_agreement": {
                    "total_compared": 8 if filters.dataset == "J1" else 12,
                    "delta_0": 2,
                    "delta_1": 1,
                    "delta_2": 0,
                    "delta_3": 0,
                    "delta_4": 0,
                    "arbiter_triggered": 5 if filters.dataset == "J1" else 0,
                },
            },
            "charts": {"candidate_ranking": rankings_by_dataset.get(filters.dataset, [])},
            "tables": {
                "critical_cases": [],
                "judge_agreement_arbitrations": [
                    {
                        "answer_id": 101,
                        "question_id": 201,
                        "candidate_model": "Grok",
                        "judge_1_score": 2,
                        "judge_2_score": 4,
                        "delta": 2,
                        "arbiter_score": 3,
                        "arbitration_reason": "score_delta_2",
                    }
                ]
                if filters.dataset == "J1"
                else [],
            },
            "methodology": {
                "primary_spearman": (
                    "Para J2, acerto do gabarito oficial vale 5 e erro vale 1. "
                    "Para J1, o cálculo só é exibido quando há referência ordinal persistida."
                )
            },
        }


class FakeAuditLogSummaryService:
    def __init__(self) -> None:
        self.calls = 0

    def load(self) -> dict:
        self.calls += 1
        return {
            "available": True,
            "totals": {"logs": 1, "events": 2, "failures": 0},
            "logs": [{"run_id": "run-1", "total_events": 2, "failures": 0}],
        }


class FakeRunJudgeService:
    def describe_config(self) -> dict:
        return {"defaults": {}, "endpoints": {}, "presets": []}


class FakeCursor:
    def __init__(self, rows_by_query: list[list[tuple]]) -> None:
        self.rows_by_query = rows_by_query
        self.queries: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def execute(self, query: str, params=None) -> None:
        self.queries.append(query)
        if not query.lstrip().upper().startswith("SELECT"):
            raise AssertionError(f"Non-read-only query executed: {query}")

    def fetchall(self) -> list[tuple]:
        return self.rows_by_query.pop(0)


class FakeConnection:
    def __init__(self, rows_by_query: list[list[tuple]] | None = None) -> None:
        self.cursor_obj = FakeCursor(
            rows_by_query
            or [
                [("datasets",), ("avaliacoes_juiz",), ("meta_avaliacoes",)],
                [("datasets", 2)],
                [("avaliacoes_juiz", 5)],
                [("meta_avaliacoes", 1)],
                [("Diego", 2, 4.5, 1)],
            ]
        )
        self.readonly = False
        self.closed = False

    def set_session(self, *, readonly: bool, autocommit: bool) -> None:
        self.readonly = readonly

    def cursor(self) -> FakeCursor:
        return self.cursor_obj

    def close(self) -> None:
        self.closed = True


class FakeConnectionFactory:
    def __init__(self, rows_by_connection: list[list[list[tuple]]]) -> None:
        self.rows_by_connection = rows_by_connection
        self.connections: list[FakeConnection] = []

    def __call__(self, database_url: str) -> FakeConnection:
        rows = self.rows_by_connection.pop(0)
        connection = FakeConnection(rows)
        self.connections.append(connection)
        return connection


def _rows_for_known_models() -> list[tuple]:
    return [
        ("Jurema:7b", "7B", "candidato"),
        ("Qwen", None, "candidato"),
        ("openai/gpt-oss-120b", None, "juiz"),
    ]


def _rows_for_known_models_and_auditors() -> list[tuple]:
    return [
        ("Jurema:7b", "7B", "candidato"),
        ("Qwen", None, "candidato"),
        ("openai/gpt-oss-120b", None, "juiz"),
        ("Diego", None, "auditor"),
    ]


def _rows_for_database_context() -> list[list[tuple]]:
    return [
        [("datasets",), ("avaliacoes_juiz",), ("meta_avaliacoes",)],
        [("datasets", 2)],
        [("avaliacoes_juiz", 5)],
        [("meta_avaliacoes", 1)],
        [("Diego", 2, 4.5, 1)],
    ]


def _assistant_service(
    *,
    llm_client: FakeLlmClient | SequencedFakeLlmClient | None = None,
    dashboard_service: FakeDashboardService | None = None,
    audit_log_summary_service: FakeAuditLogSummaryService | None = None,
    connect_func=None,
) -> AssistantService:
    readme = """
# Atividade 2

Use `make web-up` para abrir a aplicacao.

## Web UI local para execução auditável

A Web UI permite configurar, validar e acompanhar execuções do `run-judge`.
Ela mostra progresso, comando CLI equivalente e caminho do audit log.

### Modos de execução

| Modo | O que faz | Quando usar |
|---|---|---|
| `single` | Roda um juiz. | Smoke test, debug ou endpoint com um modelo só. |
| `primary_only` | Roda o painel primário. | Comparar dois juízes sem árbitro. |
| `2plus1` | Roda dois primários e chama árbitro se houver divergência. | Execução metodológica principal. |
| `2plus1 --always-run-arbiter` | Roda os três juízes sempre. | Amostra de auditoria ou apresentação. |

Para rodar a avaliação:

```bash
.venv/bin/python -m atividade_2.cli run-judge --panel-mode 2plus1 --dataset J2 --batch-size 10
```
"""
    return AssistantService(
        llm_client=llm_client or FakeLlmClient(),
        dashboard_service=dashboard_service or FakeDashboardService(),
        audit_log_summary_service=audit_log_summary_service or FakeAuditLogSummaryService(),
        settings_loader=lambda: SimpleNamespace(database_url="postgresql://example.invalid/app"),
        connect_func=connect_func or (lambda database_url: FakeConnection()),
        readme_loader=lambda: readme,
    )


def test_assistant_endpoint_answers_in_scope_question_with_llm() -> None:
    llm = FakeLlmClient("Ha 3 avaliacoes carregadas.")
    client = TestClient(
        create_app(
            FakeRunJudgeService(),
            assistant_service=_assistant_service(llm_client=llm),
        )
    )
    token = client.get("/api/config").json()["csrf_token"]

    response = client.post(
        "/api/assistant/chat",
        headers={"x-csrf-token": token},
        json={"message": "Mostre um resumo dos resultados carregados."},
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Ha 3 avaliacoes carregadas.",
        "in_scope": True,
        "suggestions": DEFAULT_SUGGESTIONS,
    }
    assert len(llm.prompts) >= 1
    assert "modo estritamente somente leitura" in llm.prompts[-1]


def test_assistant_endpoint_blocks_out_of_scope_question_before_llm() -> None:
    llm = FakeLlmClient()
    client = TestClient(
        create_app(
            FakeRunJudgeService(),
            assistant_service=_assistant_service(llm_client=llm),
        )
    )
    token = client.get("/api/config").json()["csrf_token"]

    response = client.post(
        "/api/assistant/chat",
        headers={"x-csrf-token": token},
        json={"message": "Quem ganhou a copa do mundo?"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": DEFAULT_OUT_OF_SCOPE_ANSWER,
        "in_scope": False,
        "suggestions": DEFAULT_SUGGESTIONS,
    }
    assert llm.prompts == []


def test_assistant_endpoint_blocks_write_attempt_before_llm_and_database() -> None:
    llm = FakeLlmClient()
    dashboard = FakeDashboardService()
    audit = FakeAuditLogSummaryService()
    connection_calls = 0

    def connect_func(database_url: str):
        nonlocal connection_calls
        connection_calls += 1
        return FakeConnection()

    client = TestClient(
        create_app(
            FakeRunJudgeService(),
            assistant_service=_assistant_service(
                llm_client=llm,
                dashboard_service=dashboard,
                audit_log_summary_service=audit,
                connect_func=connect_func,
            ),
        )
    )
    token = client.get("/api/config").json()["csrf_token"]

    response = client.post(
        "/api/assistant/chat",
        headers={"x-csrf-token": token},
        json={"message": "Execute DELETE FROM avaliacoes_juiz."},
    )

    assert response.status_code == 200
    assert response.json()["answer"] == DEFAULT_OUT_OF_SCOPE_ANSWER
    assert response.json()["in_scope"] is False
    assert llm.prompts == []
    assert dashboard.calls == 0
    assert audit.calls == 0
    assert connection_calls == 0


def test_assistant_blocks_request_to_evaluate_member() -> None:
    llm = FakeLlmClient()
    service = _assistant_service(llm_client=llm)

    response = service.answer("Avalie o desempenho do integrante João no projeto.")

    assert response == {
        "answer": DEFAULT_OUT_OF_SCOPE_ANSWER,
        "in_scope": False,
        "suggestions": DEFAULT_SUGGESTIONS,
    }
    assert llm.prompts == []


def test_assistant_blocks_request_to_evaluate_team() -> None:
    llm = FakeLlmClient()
    service = _assistant_service(llm_client=llm)

    response = service.answer("Dê uma nota para a equipe pelo trabalho entregue.")

    assert response["answer"] == DEFAULT_OUT_OF_SCOPE_ANSWER
    assert response["in_scope"] is False
    assert llm.prompts == []


def test_assistant_blocks_subjective_performance_request_by_known_auditor_name() -> None:
    llm = FakeLlmClient()
    connection_factory = FakeConnectionFactory([[_rows_for_known_models_and_auditors()]])
    service = _assistant_service(llm_client=llm, connect_func=connection_factory)

    response = service.answer("Avalie o desempenho de Diego.")

    assert response["answer"] == DEFAULT_OUT_OF_SCOPE_ANSWER
    assert response["in_scope"] is False
    assert llm.prompts == []


def test_assistant_blocks_subjective_good_work_request_by_known_auditor_name() -> None:
    llm = FakeLlmClient()
    connection_factory = FakeConnectionFactory([[_rows_for_known_models_and_auditors()]])
    service = _assistant_service(llm_client=llm, connect_func=connection_factory)

    response = service.answer("Diego fez um bom trabalho?")

    assert response["answer"] == DEFAULT_OUT_OF_SCOPE_ANSWER
    assert response["in_scope"] is False
    assert llm.prompts == []


def test_assistant_blocks_subjective_best_team_auditor_request() -> None:
    llm = FakeLlmClient()
    service = _assistant_service(llm_client=llm)

    response = service.answer("Quem da equipe auditou melhor?")

    assert response["answer"] == DEFAULT_OUT_OF_SCOPE_ANSWER
    assert response["in_scope"] is False
    assert llm.prompts == []


def test_assistant_blocks_request_to_list_group_failures() -> None:
    llm = FakeLlmClient()
    service = _assistant_service(llm_client=llm)

    response = service.answer("Liste as falhas do grupo na metodologia.")

    assert response["answer"] == DEFAULT_OUT_OF_SCOPE_ANSWER
    assert response["in_scope"] is False
    assert llm.prompts == []


def test_assistant_allows_request_about_loaded_result_failures_with_default_answer() -> None:
    llm = FakeLlmClient()
    service = _assistant_service(llm_client=llm)

    response = service.answer("Quais falhas aparecem nos resultados carregados?")

    assert response["answer"] == "Resumo gerado pelo assistente."
    assert response["in_scope"] is True
    assert len(llm.prompts) >= 1


def test_assistant_blocks_request_to_point_project_limitations() -> None:
    llm = FakeLlmClient()
    service = _assistant_service(llm_client=llm)

    response = service.answer("Aponte as limitações do projeto como avaliação acadêmica.")

    assert response["answer"] == DEFAULT_OUT_OF_SCOPE_ANSWER
    assert response["in_scope"] is False
    assert llm.prompts == []


def test_assistant_allows_request_about_loaded_evaluation_limitations_with_default_answer() -> None:
    llm = FakeLlmClient()
    service = _assistant_service(llm_client=llm)

    response = service.answer("Quais são as limitações das avaliações carregadas?")

    assert response["answer"] == "Resumo gerado pelo assistente."
    assert response["in_scope"] is True
    assert len(llm.prompts) >= 1


def test_assistant_blocks_request_to_assign_blame_for_missing_data() -> None:
    llm = FakeLlmClient()
    service = _assistant_service(llm_client=llm)

    response = service.answer("Quem é culpado pela falta de dados carregados no banco?")

    assert response["answer"] == DEFAULT_OUT_OF_SCOPE_ANSWER
    assert response["in_scope"] is False
    assert llm.prompts == []


def test_assistant_allows_request_about_missing_loaded_data_with_default_answer() -> None:
    llm = FakeLlmClient()
    service = _assistant_service(llm_client=llm)

    response = service.answer("Existe falta de dados nos datasets carregados?")

    assert response["answer"] == "Resumo gerado pelo assistente."
    assert response["in_scope"] is True
    assert len(llm.prompts) >= 1


def test_assistant_blocks_generated_forbidden_content_with_default_answer() -> None:
    llm = FakeLlmClient("Há limitações nos dados carregados.")
    service = _assistant_service(llm_client=llm)

    response = service.answer("Mostre um resumo dos resultados carregados.")

    assert response["answer"] == DEFAULT_OUT_OF_SCOPE_ANSWER
    assert response["in_scope"] is False
    assert len(llm.prompts) >= 1


def test_assistant_general_summary_uses_neutral_operational_content() -> None:
    llm = FakeLlmClient("Totais carregados: 3 avaliações, 1 auditoria e 3 tabelas disponíveis.")
    service = _assistant_service(llm_client=llm)

    response = service.answer("Mostre um resumo geral dos resultados carregados.")

    assert response["answer"] == "Totais carregados: 3 avaliações, 1 auditoria e 3 tabelas disponíveis."
    assert response["in_scope"] is True
    assert "failures" not in llm.prompts[-1]
    assert "error" not in llm.prompts[-1]


def test_assistant_allows_factual_loaded_data_count_question() -> None:
    llm = FakeLlmClient("Existem 3 avaliações carregadas.")
    service = _assistant_service(llm_client=llm)

    response = service.answer("Qual é a quantidade de dados carregados?")

    assert response["answer"] == "Existem 3 avaliações carregadas."
    assert response["in_scope"] is True
    assert len(llm.prompts) >= 1


def test_assistant_context_includes_candidate_rankings_by_dataset() -> None:
    llm = FakeLlmClient("J1: Grok. J2: Gemini.")
    dashboard = FakeDashboardService()
    service = _assistant_service(llm_client=llm, dashboard_service=dashboard)

    response = service.answer("Quais os top 5 ia avaliadas, classifique por dataset")

    assert response["answer"] == "J1: Grok. J2: Gemini."
    assert response["in_scope"] is True
    assert dashboard.calls == 2
    assert '"rankings_by_dataset"' in llm.prompts[-1]
    assert '"J1": [{"label": "Grok", "value": 4.21, "count": 24}]' in llm.prompts[-1]
    assert '"J2": [{"label": "Gemini", "value": 4.84, "count": 246}]' in llm.prompts[-1]


def test_assistant_context_includes_spearman_cards_by_dataset() -> None:
    llm = FakeLlmClient("J2: Spearman 0.9973.")
    dashboard = FakeDashboardService()
    service = _assistant_service(llm_client=llm, dashboard_service=dashboard)

    response = service.answer("Calcule a correlação de Spearman entre o juiz e o gabarito humano para J2.")

    assert response["answer"] == "J2: Spearman 0.9973."
    assert response["in_scope"] is True
    assert dashboard.calls == 2
    assert '"cards_by_dataset"' in llm.prompts[-1]
    assert '"J2": {"evaluations": 3' in llm.prompts[-1]
    assert '"spearman_reference": {"value": 0.9973, "p_value": 0.0, "sample_size": 4428, "available": true' in llm.prompts[-1]
    assert "Para J2, acerto do gabarito oficial vale 5 e erro vale 1" in llm.prompts[-1]


def test_assistant_context_includes_spearman_by_candidate_by_dataset() -> None:
    llm = FakeLlmClient("J1 sem Spearman. J2 Gemini: Spearman 1.0.")
    dashboard = FakeDashboardService()
    service = _assistant_service(llm_client=llm, dashboard_service=dashboard)

    response = service.answer("Calcule Spearman por modelo candidato e classifique por dataset")

    assert response["answer"] == "J1 sem Spearman. J2 Gemini: Spearman 1.0."
    assert response["in_scope"] is True
    assert dashboard.calls == 4
    assert '"spearman_by_candidate_by_dataset"' in llm.prompts[-1]
    assert '"J1": [{"label": "Grok", "spearman": {"value": null, "p_value": null, "sample_size": 0, "available": false' in llm.prompts[-1]
    assert '"J2": [{"label": "Gemini", "spearman": {"value": 1.0, "p_value": 0.0, "sample_size": 246, "available": true' in llm.prompts[-1]
    assert "não substitua Spearman por média de nota" in llm.prompts[-1]


def test_assistant_context_includes_arbiter_triggers_by_dataset() -> None:
    llm = FakeLlmClient("O árbitro foi acionado 5 vezes em J1 e 0 vezes em J2.")
    dashboard = FakeDashboardService()
    connection_factory = FakeConnectionFactory([[[("J1", 301, 101, 201, "Grok", "Judge C", 3, "score_delta_2", "Justificativa")]]])
    service = _assistant_service(llm_client=llm, dashboard_service=dashboard, connect_func=connection_factory)

    response = service.answer("Quantas vezes o árbitro foi acionado e em quais datasets isso ocorreu?")

    assert response["answer"] == "O árbitro foi acionado 5 vezes em J1 e 0 vezes em J2."
    assert response["in_scope"] is True
    assert "score_delta_2" in llm.prompts[-1]


def test_assistant_context_includes_highest_judge_disagreements_by_dataset() -> None:
    llm = FakeLlmClient("J1 tem divergência máxima no caso 101.")
    dashboard = FakeDashboardService()
    connection_factory = FakeConnectionFactory([[[("J1", 101, 201, "Grok", "Judge A", "Judge B", 2, 4, 2)]]])
    service = _assistant_service(llm_client=llm, dashboard_service=dashboard, connect_func=connection_factory)

    response = service.answer("Mostre as avaliações com maior divergência entre juiz principal e juiz controle.")

    assert response["answer"] == "J1 tem divergência máxima no caso 101."
    assert response["in_scope"] is True
    assert "delta" in llm.prompts[-1]
    assert "101" in llm.prompts[-1]


def test_assistant_evidence_loop_uses_deterministic_intent_tools() -> None:
    llm = FakeLlmClient("A maior divergência está no caso 101.")
    dashboard = FakeDashboardService()
    connection_factory = FakeConnectionFactory([[[("J1", 101, 201, "Grok", "Judge A", "Judge B", 2, 4, 2)]]])
    service = _assistant_service(llm_client=llm, dashboard_service=dashboard, connect_func=connection_factory)

    response = service.answer("Mostre as avaliações com maior divergência entre juiz principal e juiz controle.")

    assert response["answer"] == "A maior divergência está no caso 101."
    assert response["in_scope"] is True
    assert "id_resposta" in llm.prompts[-1]


def test_assistant_converts_tool_name_leak_into_final_context_answer() -> None:
    llm = SequencedFakeLlmClient(
        [
            "Para responder, use a ferramenta 'judge_disagreements'.",
        ]
    )
    dashboard = FakeDashboardService()
    connection_factory = FakeConnectionFactory([[[("J1", 101, 201, "Grok", "Judge A", "Judge B", 2, 4, 2)]]])
    service = _assistant_service(llm_client=llm, dashboard_service=dashboard, connect_func=connection_factory)

    response = service.answer("Mostre as avaliações com maior divergência entre juiz principal e juiz controle.")

    assert response["answer"] == "Para responder, use a ferramenta 'judge_disagreements'."
    assert response["in_scope"] is True
    assert len(llm.prompts) == 1
    assert "101" in llm.prompts[-1]


def test_assistant_converts_answer_tool_name_leak_to_deterministic_data_answer() -> None:
    llm = SequencedFakeLlmClient(
        [
            '{"sufficient": false, "required_context": ["judge_disagreements"]}',
            "Para identificar as avaliações com maior divergência, use a ferramenta 'judge_disagreements'.",
        ]
    )
    dashboard = FakeDashboardService()
    connection_factory = FakeConnectionFactory([[[("J1", 101, 201, "Grok", "Judge A", "Judge B", 2, 4, 2)]]])
    service = _assistant_service(llm_client=llm, dashboard_service=dashboard, connect_func=connection_factory)

    response = service.answer("Mostre as avaliações com maior divergência entre juiz principal e juiz controle.")

    assert response["answer"] == '{"sufficient": false, "required_context": ["judge_disagreements"]}'
    assert response["in_scope"] is True
    assert len(llm.prompts) == 1
    assert "101" in llm.prompts[-1]


def test_assistant_converts_sufficient_planner_tool_leak_to_data_answer() -> None:
    llm = SequencedFakeLlmClient(
        [
            (
                '{"sufficient": true, "required_context": [], '
                '"answer": "Para identificar as avaliações, use a ferramenta judge_disagreements."}'
            ),
            "Para identificar as avaliações, use a ferramenta judge_disagreements.",
        ]
    )
    dashboard = FakeDashboardService()
    connection_factory = FakeConnectionFactory([[[("J1", 101, 201, "Grok", "Judge A", "Judge B", 2, 4, 2)]]])
    service = _assistant_service(llm_client=llm, dashboard_service=dashboard, connect_func=connection_factory)

    response = service.answer("Mostre as avaliações com maior divergência entre juiz principal e juiz controle.")

    assert response["answer"].startswith('{"sufficient": true')
    assert response["in_scope"] is True
    assert len(llm.prompts) == 1
    assert "101" in llm.prompts[-1]


def test_assistant_general_app_question_uses_backend_selected_context() -> None:
    llm = SequencedFakeLlmClient(
        [
            '{"sufficient": true, "required_context": [], "answer": "Posso responder apenas com contexto local read-only."}',
        ]
    )
    dashboard = FakeDashboardService()
    audit = FakeAuditLogSummaryService()
    connection_calls = 0

    def connect_func(database_url: str):
        nonlocal connection_calls
        connection_calls += 1
        return FakeConnection()

    service = _assistant_service(
        llm_client=llm,
        dashboard_service=dashboard,
        audit_log_summary_service=audit,
        connect_func=connect_func,
    )

    response = service.answer("O app pode operar em modo somente leitura?")

    assert response["answer"] == "Posso responder apenas com contexto local read-only."
    assert response["in_scope"] is True
    assert len(llm.prompts) == 1
    assert dashboard.calls == 2
    assert audit.calls == 0
    assert connection_calls == 2
    assert "planejador read-only" not in llm.prompts[0]


def test_assistant_ignores_additional_tool_request_after_context_is_assembled() -> None:
    llm = SequencedFakeLlmClient(
        [
            '{"sufficient": false, "required_context": ["dashboard_summary"]}',
        ]
    )
    dashboard = FakeDashboardService()
    connection_factory = FakeConnectionFactory([[[("J1", 101, 201, "Grok", "Judge A", "Judge B", 2, 4, 2)]]])
    service = _assistant_service(llm_client=llm, dashboard_service=dashboard, connect_func=connection_factory)

    response = service.answer("Mostre as avaliações com maior divergência entre juiz principal e juiz controle.")

    assert response["answer"] == '{"sufficient": false, "required_context": ["dashboard_summary"]}'
    assert response["in_scope"] is True
    assert len(llm.prompts) == 1


def test_assistant_allows_factual_existing_audits_question() -> None:
    llm = FakeLlmClient("Existe 1 auditoria registrada.")
    service = _assistant_service(llm_client=llm)

    response = service.answer("Quais auditorias existentes foram carregadas?")

    assert response["answer"] == "Existe 1 auditoria registrada."
    assert response["in_scope"] is True
    assert len(llm.prompts) >= 1


def test_assistant_allows_factual_meta_evaluation_question() -> None:
    llm = FakeLlmClient("Existe 1 meta-avaliação registrada.")
    service = _assistant_service(llm_client=llm)

    response = service.answer("Quais meta-avaliações existem nas auditorias?")

    assert response["answer"] == "Existe 1 meta-avaliação registrada."
    assert response["in_scope"] is True
    assert len(llm.prompts) >= 1


def test_assistant_resolves_meta_analysis_aliases_as_human_audit_query() -> None:
    messages = [
        "Mostre um resumo das meta-análises já realizadas.",
        "Mostre um resumo das meta analises já realizadas.",
        "Mostre um resumo das meta-analises já realizadas.",
        "Quais metaanálises foram feitas?",
        "Quais meta-avaliações foram feitas?",
        "Mostre um resumo das auditorias já realizadas.",
        "Mostre um resumo da revisão humana.",
    ]

    assert all(is_factual_human_audit_query(message) for message in messages)


def test_assistant_meta_analysis_summary_uses_human_audit_database_context_before_readme() -> None:
    llm = FakeLlmClient("Existe 1 meta-análise humana registrada por Diego.")
    service = _assistant_service(llm_client=llm)

    response = service.answer("Mostre um resumo das meta-análise já realizadas.")

    assert response["answer"] == "Existe 1 meta-análise humana registrada por Diego."
    assert response["in_scope"] is True
    assert "sources_used" not in response
    assert '"auditorias_humanas": [{"avaliador": "Diego", "total_auditorias": 2' in llm.prompts[-1]
    assert "logs_execucao_juiz" in llm.prompts[-1]
    assert "README omitido" in llm.prompts[-1]
    assert "Trecho de documentação interna permitido" not in llm.prompts[-1]


def test_assistant_audit_summary_uses_human_audit_database_context_before_readme() -> None:
    llm = FakeLlmClient("Existe 1 auditoria humana registrada por Diego.")
    service = _assistant_service(llm_client=llm)

    response = service.answer("Mostre um resumo das auditorias já realizadas.")

    assert response["answer"] == "Existe 1 auditoria humana registrada por Diego."
    assert response["in_scope"] is True
    assert "sources_used" not in response
    assert '"auditorias_humanas": [{"avaliador": "Diego", "total_auditorias": 2' in llm.prompts[-1]


def test_assistant_allows_factual_audit_summary_by_known_auditor_name() -> None:
    llm = FakeLlmClient("Diego realizou 2 auditorias, com média 4,5.")
    connection_factory = FakeConnectionFactory([[_rows_for_known_models_and_auditors()], _rows_for_database_context()])
    service = _assistant_service(llm_client=llm, connect_func=connection_factory)

    response = service.answer("Mostre um resumo da auditoria feita por Diego.")

    assert response["answer"] == "Diego realizou 2 auditorias, com média 4,5."
    assert response["in_scope"] is True
    assert len(llm.prompts) >= 1
    assert "Diego" in llm.prompts[-1]


def test_assistant_allows_factual_audits_by_known_auditor_name() -> None:
    llm = FakeLlmClient("Diego fez auditorias nos casos 101 e 102.")
    connection_factory = FakeConnectionFactory([[_rows_for_known_models_and_auditors()], _rows_for_database_context()])
    service = _assistant_service(llm_client=llm, connect_func=connection_factory)

    response = service.answer("Quais auditorias foram feitas por Diego?")

    assert response["answer"] == "Diego fez auditorias nos casos 101 e 102."
    assert response["in_scope"] is True
    assert len(llm.prompts) >= 1


def test_assistant_allows_factual_audit_count_by_known_auditor_name() -> None:
    llm = FakeLlmClient("Diego realizou 2 auditorias.")
    connection_factory = FakeConnectionFactory([[_rows_for_known_models_and_auditors()], _rows_for_database_context()])
    service = _assistant_service(llm_client=llm, connect_func=connection_factory)

    response = service.answer("Quantas auditorias Diego realizou?")

    assert response["answer"] == "Diego realizou 2 auditorias."
    assert response["in_scope"] is True
    assert len(llm.prompts) >= 1


def test_assistant_allows_factual_meta_analysis_count_by_known_auditor_name() -> None:
    llm = FakeLlmClient("Diego realizou 2 meta-análises.")
    connection_factory = FakeConnectionFactory([[_rows_for_known_models_and_auditors()], _rows_for_database_context()])
    service = _assistant_service(llm_client=llm, connect_func=connection_factory)

    response = service.answer("Quantas meta-análises Diego fez?")

    assert response["answer"] == "Diego realizou 2 meta-análises."
    assert response["in_scope"] is True
    assert len(llm.prompts) >= 1
    assert "Diego" in llm.prompts[-1]


def test_assistant_allows_audits_grouped_by_evaluator() -> None:
    llm = FakeLlmClient("Diego: 2 auditorias.")
    service = _assistant_service(llm_client=llm)

    response = service.answer("Mostre auditorias por avaliador.")

    assert response["answer"] == "Diego: 2 auditorias."
    assert response["in_scope"] is True
    assert len(llm.prompts) >= 1


def test_assistant_allows_factual_audits_that_disagreed_with_judge() -> None:
    llm = FakeLlmClient("Existe 1 auditoria com divergência factual em relação ao juiz.")
    service = _assistant_service(llm_client=llm)

    response = service.answer("Quais auditorias discordaram do juiz?")

    assert response["answer"] == "Existe 1 auditoria com divergência factual em relação ao juiz."
    assert response["in_scope"] is True
    assert len(llm.prompts) >= 1


def test_assistant_allows_meta_evaluation_summary() -> None:
    llm = FakeLlmClient("Resumo factual da meta-avaliação.")
    service = _assistant_service(llm_client=llm)

    response = service.answer("Resumo da meta-avaliação.")

    assert response["answer"] == "Resumo factual da meta-avaliação."
    assert response["in_scope"] is True
    assert len(llm.prompts) >= 1


def test_assistant_allows_candidate_model_by_known_name() -> None:
    llm = FakeLlmClient("Resumo factual do modelo Jurema.")
    connection_factory = FakeConnectionFactory([[_rows_for_known_models()], _rows_for_database_context()])
    service = _assistant_service(llm_client=llm, connect_func=connection_factory)

    response = service.answer("Mostre um resumo do candidato Jurema.")

    assert response["answer"] == "Resumo factual do modelo Jurema."
    assert response["in_scope"] is True
    assert len(llm.prompts) >= 1
    assert all(connection.readonly for connection in connection_factory.connections)


def test_assistant_resolves_named_candidate_before_generic_candidate_term() -> None:
    connection_factory = FakeConnectionFactory([[_rows_for_known_models()]])
    service = _assistant_service(connect_func=connection_factory)

    resolution = service.resolve_assistant_entities("Mostre um resumo do candidato Jurema.")

    assert resolution.status == "single"
    assert [match.kind for match in resolution.matches] == ["model_candidate"]
    assert [match.value for match in resolution.matches] == ["Jurema:7b"]


def test_assistant_resolves_jurema_candidate_with_hyphenated_name() -> None:
    connection_factory = FakeConnectionFactory([[[("jurema-7b", None, "candidato")]]])
    service = _assistant_service(connect_func=connection_factory)

    entity = service.resolve_assistant_entity("Mostre um resumo do candidato Jurema.")

    assert entity is not None
    assert entity.kind == "model_candidate"


def test_assistant_resolves_jurema_candidate_with_provider_path_name() -> None:
    connection_factory = FakeConnectionFactory([[[("mauroneto/Jurema-7B-Q4_K_M-GGUF", None, "candidato")]]])
    service = _assistant_service(connect_func=connection_factory)

    entity = service.resolve_assistant_entity("Mostre um resumo do candidato Jurema.")

    assert entity is not None
    assert entity.kind == "model_candidate"


def test_assistant_resolves_jurema_to_multiple_candidate_entities() -> None:
    connection_factory = FakeConnectionFactory(
        [
            [
                [
                    ("Jurema:7b", "7B", "candidato"),
                    ("jurema-7b", None, "candidato"),
                    ("mauroneto/Jurema-7B-Q4_K_M-GGUF", None, "candidato"),
                    ("Qwen", None, "candidato"),
                ]
            ]
        ]
    )
    service = _assistant_service(connect_func=connection_factory)

    resolution = service.resolve_assistant_entities("Mostre resultados do Jurema.")

    assert resolution.status == "multiple"
    assert [match.value for match in resolution.matches] == [
        "Jurema:7b",
        "jurema-7b",
        "mauroneto/Jurema-7B-Q4_K_M-GGUF",
    ]


def test_assistant_average_scores_returns_all_candidates_without_top_n() -> None:
    llm = FakeLlmClient()
    connection_factory = FakeConnectionFactory(
        [
            [[("Jurema:7b", "7B", "candidato"), ("Qwen", None, "candidato")]],
            [
                [
                    ("J1", "Jurema:7b", "Judge A", "principal", 2, 4.5),
                    ("J1", "Qwen", "Judge A", "principal", 3, 4.0),
                    ("J2", "Jurema:7b", "Judge B", "controle", 1, 5.0),
                ]
            ],
        ]
    )
    service = _assistant_service(llm_client=llm, connect_func=connection_factory)

    response = service.answer("Mostre a média de notas por modelo candidato e por juiz.")

    assert response["in_scope"] is True
    assert "Jurema:7b" in response["answer"]
    assert "Qwen" in response["answer"]
    assert "modelo_candidato" in response["answer"]
    assert len(llm.prompts) == 2


def test_assistant_average_scores_filters_plural_jurema_matches() -> None:
    connection_factory = FakeConnectionFactory(
        [
            [
                [
                    ("Jurema:7b", "7B", "candidato"),
                    ("jurema-7b", None, "candidato"),
                    ("Qwen", None, "candidato"),
                ]
            ],
            [[("J1", "Jurema:7b", "Judge A", "principal", 2, 4.5), ("J1", "jurema-7b", "Judge A", "principal", 1, 4.0)]],
        ]
    )
    service = _assistant_service(connect_func=connection_factory)

    response = service.answer("Mostre a média de notas do Jurema por modelo candidato e por juiz.")

    assert response["in_scope"] is True
    assert "Jurema:7b" in response["answer"]
    assert "jurema-7b" in response["answer"]
    assert "Qwen" not in response["answer"]
    assert len(service._llm_client.prompts) == 2


def test_assistant_average_scores_applies_top_n_limit_in_catalog_query() -> None:
    connection_factory = FakeConnectionFactory(
        [
            [[("Jurema:7b", "7B", "candidato"), ("Qwen", None, "candidato")]],
            [[("J1", "Jurema:7b", "Judge A", "principal", 2, 4.5)]],
        ]
    )
    service = _assistant_service(connect_func=connection_factory)

    response = service.answer("Mostre o top 1 de média de notas por modelo candidato e por juiz.")

    assert response["in_scope"] is True
    assert len(service._llm_client.prompts) == 2
    assert connection_factory.connections[-1].cursor_obj.queries[-1].lstrip().upper().startswith("SELECT")


def test_assistant_active_prompts_and_rubrics_use_prompt_juizes() -> None:
    llm = FakeLlmClient()
    connection_factory = FakeConnectionFactory(
        [
            [_rows_for_known_models()],
            [[("J1", 2, True, "system", 1000, 500, "Prompt ativo", "Rubrica ativa")]],
        ]
    )
    service = _assistant_service(llm_client=llm, connect_func=connection_factory)

    response = service.answer("Quais prompts e rubricas ativos existem por dataset?")

    assert response["in_scope"] is True
    assert "prompt_juizes" in response["answer"]
    assert "Rubrica ativa" in response["answer"]
    assert len(llm.prompts) == 2


def test_assistant_model_counts_are_deterministic_sql() -> None:
    llm = FakeLlmClient()
    connection_factory = FakeConnectionFactory(
        [
            [_rows_for_known_models()],
            [[("J1", "Jurema:7b", "7B", 12, 24), ("J2", "Qwen", "", 8, 16)]],
        ]
    )
    service = _assistant_service(llm_client=llm, connect_func=connection_factory)

    response = service.answer("Quais modelos candidatos avaliados e quantidade de respostas por modelo?")

    assert response["in_scope"] is True
    assert "Jurema:7b" in response["answer"]
    assert "qtd_respostas" in response["answer"]
    assert len(llm.prompts) == 2
    assert connection_factory.connections[-1].cursor_obj.queries[-1].lstrip().upper().startswith("SELECT")


def test_assistant_judge_divergence_is_deterministic_sql() -> None:
    llm = FakeLlmClient()
    connection_factory = FakeConnectionFactory(
        [
            [[("J1", 101, 201, "Jurema:7b", "Judge A", "Judge B", 5, 2, 3)]],
        ]
    )
    service = _assistant_service(llm_client=llm, connect_func=connection_factory)

    response = service.answer("Mostre divergência entre juiz principal e juiz controle.")

    assert response["in_scope"] is True
    assert "101" in response["answer"]
    assert "delta" in response["answer"]
    assert len(llm.prompts) == 2


def test_assistant_arbiter_cases_are_deterministic_sql() -> None:
    llm = FakeLlmClient()
    connection_factory = FakeConnectionFactory(
        [
            [[("J1", 301, 101, 201, "Jurema:7b", "Judge C", 4, "score_delta_3", "Justificativa")]],
        ]
    )
    service = _assistant_service(llm_client=llm, connect_func=connection_factory)

    response = service.answer("Quais casos em que o árbitro foi acionado?")

    assert response["in_scope"] is True
    assert "score_delta_3" in response["answer"]
    assert len(llm.prompts) == 2


def test_assistant_trace_evaluation_returns_minimum_fields() -> None:
    llm = FakeLlmClient()
    connection_factory = FakeConnectionFactory(
        [
            [_rows_for_known_models()],
            [[("J1", 301, 201, "Pergunta completa", 101, "Resposta candidata", "Jurema:7b", "Judge A", "principal", 5, "Justificativa", 2, "Prompt usado", "Rubrica usada")]],
        ]
    )
    service = _assistant_service(llm_client=llm, connect_func=connection_factory)

    response = service.answer("Mostre rastreabilidade completa da avaliação 301.")

    assert response["in_scope"] is True
    assert "Pergunta completa" in response["answer"]
    assert "Resposta candidata" in response["answer"]
    assert "Jurema:7b" in response["answer"]
    assert "Justificativa" in response["answer"]
    assert "Prompt usado" in response["answer"]
    assert len(llm.prompts) == 2


def test_assistant_j2_performance_is_deterministic_sql() -> None:
    llm = FakeLlmClient()
    connection_factory = FakeConnectionFactory(
        [
            [_rows_for_known_models()],
            [[("OAB_Exames", "Jurema:7b", "7B", 10, 12, 8, 2, 4.25)]],
        ]
    )
    service = _assistant_service(llm_client=llm, connect_func=connection_factory)

    response = service.answer("Mostre desempenho no J2 por acertos, erros e notas.")

    assert response["in_scope"] is True
    assert "notas_5" in response["answer"]
    assert "notas_1" in response["answer"]
    assert "não acertos/erros únicos por pergunta" in response["answer"]
    assert "4.25" in response["answer"]
    assert len(llm.prompts) == 2


def test_assistant_extreme_disagreements_are_deterministic_sql() -> None:
    llm = FakeLlmClient()
    connection_factory = FakeConnectionFactory(
        [
            [_rows_for_known_models()],
            [[("J1", 101, 201, "Jurema:7b", "Judge A", "Judge B", 5, 1, "bom", "ruim")]],
        ]
    )
    service = _assistant_service(llm_client=llm, connect_func=connection_factory)

    response = service.answer("Mostre avaliações com nota 5 de um juiz e 1 de outro.")

    assert response["in_scope"] is True
    assert "Judge A" in response["answer"]
    assert "Judge B" in response["answer"]
    assert len(llm.prompts) == 2


def test_assistant_model_name_search_returns_variations() -> None:
    llm = FakeLlmClient()
    connection_factory = FakeConnectionFactory(
        [
            [[("Jurema:7b", "7B", "candidato"), ("jurema-7b", None, "candidato")]],
            [[("Jurema:7b", "7B", "candidato", 2, 4), ("jurema-7b", "", "candidato", 1, 2)]],
        ]
    )
    service = _assistant_service(llm_client=llm, connect_func=connection_factory)

    response = service.answer("Busca flexível por modelo Jurema.")

    assert response["in_scope"] is True
    assert "Jurema:7b" in response["answer"]
    assert "jurema-7b" in response["answer"]
    assert len(llm.prompts) == 2


def test_assistant_audit_case_recommendation_is_deterministic_sql() -> None:
    llm = FakeLlmClient()
    connection_factory = FakeConnectionFactory(
        [
            [_rows_for_known_models()],
            [[(1, "árbitro acionado", "J1", 101, 201, "Jurema:7b", 4855, 5, "arbitragem por delta")]],
        ]
    )
    service = _assistant_service(llm_client=llm, connect_func=connection_factory)

    response = service.answer("Recomende casos para auditoria manual ou meta-avaliação.")

    assert response["in_scope"] is True
    assert "4855" in response["answer"]
    assert "árbitro acionado" in response["answer"]
    assert "arbitragem por delta" in response["answer"]
    assert len(llm.prompts) == 2


def test_assistant_audit_case_recommendation_handles_full_criteria_question() -> None:
    llm = FakeLlmClient()
    connection_factory = FakeConnectionFactory(
        [
            [_rows_for_known_models()],
            [[(1, "árbitro acionado", "J2", 101, 201, "Jurema:7b", 4855, 5, "score_delta")]],
        ]
    )
    service = _assistant_service(llm_client=llm, connect_func=connection_factory)

    response = service.answer(
        "Quais são os principais casos recomendados para auditoria manual/meta-avaliação, "
        "considerando divergência entre juízes, erro evidente, nota extrema ou justificativa suspeita?"
    )

    assert response["in_scope"] is True
    assert "4855" in response["answer"]
    assert "árbitro acionado" in response["answer"]
    assert "não foram encontrados registros" not in response["answer"].lower()
    assert len(llm.prompts) == 2


def test_assistant_repairs_stale_audit_recommendation_absence() -> None:
    llm = SequencedFakeLlmClient(
        [
            "Não foram encontrados casos recomendados para auditoria manual/meta-avaliação.",
            "Há caso recomendado: J1 resposta 101, Jurema:7b, por delta 4 e árbitro acionado.",
        ]
    )
    connection_factory = FakeConnectionFactory(
        [
            [_rows_for_known_models()],
            [[(1, "árbitro acionado", "J1", 101, 201, "Jurema:7b", 4855, 5, "score_delta")]],
        ]
    )
    service = _assistant_service(llm_client=llm, connect_func=connection_factory)

    response = service.answer("Recomende casos para auditoria manual ou meta-avaliação.")

    assert response["in_scope"] is True
    assert response["answer"] == "Há caso recomendado: J1 resposta 101, Jurema:7b, por delta 4 e árbitro acionado."
    assert len(llm.prompts) == 2
    assert "Não diga que não há registros quando a evidência tem linhas" in llm.prompts[-1]


def test_assistant_deterministic_sql_no_rows_uses_factual_absence_message() -> None:
    llm = FakeLlmClient()
    connection_factory = FakeConnectionFactory([[_rows_for_known_models()], [[]]])
    service = _assistant_service(llm_client=llm, connect_func=connection_factory)

    response = service.answer("Mostre desempenho no J2 por acertos, erros e notas.")

    assert response["answer"] == "Resumo gerado pelo assistente."
    assert response["in_scope"] is True
    assert "Não encontrei registros para esse critério nas tabelas consultadas." in llm.prompts[-1]


def test_assistant_repairs_stale_llm_sql_analysis_fallback() -> None:
    llm = SequencedFakeLlmClient(
        [
            "Com base no contexto local, não é possível calcular a média por modelo candidato e juiz.",
            "A evidência SQL mostra Jurema:7b com média 4,5 pelo Judge A.",
        ]
    )
    connection_factory = FakeConnectionFactory(
        [
            [[("Jurema:7b", "7B", "candidato")]],
            [[("J1", "Jurema:7b", "Judge A", "principal", 2, 4.5)]],
        ]
    )
    service = _assistant_service(llm_client=llm, connect_func=connection_factory)

    response = service.answer("Mostre a média de notas por modelo candidato e por juiz.")

    assert response["in_scope"] is True
    assert "Jurema:7b" in response["answer"]
    assert "Judge A" in response["answer"]
    assert "4.5" in response["answer"]
    assert len(llm.prompts) == 2
    assert "Resposta anterior inválida" in llm.prompts[-1]


def test_assistant_average_scores_does_not_let_llm_deny_existing_sql_rows() -> None:
    llm = FakeLlmClient(
        "Com base nos dados disponíveis no contexto local, não é possível calcular a média de notas por modelo candidato separada por juiz avaliador."
    )
    connection_factory = FakeConnectionFactory(
        [
            [[("Jurema:7b", "7B", "candidato")]],
            [[("J1", "Jurema:7b", "Judge A", "principal", 2, 4.5)]],
        ]
    )
    service = _assistant_service(llm_client=llm, connect_func=connection_factory)

    response = service.answer("Mostre a média de notas por modelo candidato, separando por juiz avaliador.")

    assert response["in_scope"] is True
    assert "modelo_candidato" in response["answer"]
    assert "Jurema:7b" in response["answer"]
    assert "Judge A" in response["answer"]
    assert "4.5" in response["answer"]
    assert len(llm.prompts) == 2


def test_assistant_average_scores_accepts_llm_intro_when_table_is_preserved() -> None:
    table = (
        "dataset | modelo_candidato | juiz | papel_juiz | qtd_avaliações | média_nota\n"
        "--- | --- | --- | --- | ---: | ---:\n"
        "J1 | Jurema:7b | Judge A | principal | 2 | 4.5"
    )
    llm = FakeLlmClient(f"Segue a média de notas por modelo candidato e juiz avaliador:\n\n{table}")
    connection_factory = FakeConnectionFactory(
        [
            [[("Jurema:7b", "7B", "candidato")]],
            [[("J1", "Jurema:7b", "Judge A", "principal", 2, 4.5)]],
        ]
    )
    service = _assistant_service(llm_client=llm, connect_func=connection_factory)

    response = service.answer("Mostre a média de notas por modelo candidato, separando por juiz avaliador.")

    assert response["in_scope"] is True
    assert response["answer"].startswith("Segue a média")
    assert table in response["answer"]
    assert len(llm.prompts) == 1


def test_assistant_average_scores_rejects_llm_table_data_changes() -> None:
    llm = SequencedFakeLlmClient(
        [
            (
                "Segue a tabela:\n\n"
                "dataset | modelo_candidato | juiz | papel_juiz | qtd_avaliações | média_nota\n"
                "--- | --- | --- | --- | ---: | ---:\n"
                "J1 | Jurema:7b | Judge A | principal | 2 | 5.0"
            ),
            (
                "Segue a tabela correta:\n\n"
                "dataset | modelo_candidato | juiz | papel_juiz | qtd_avaliações | média_nota\n"
                "--- | --- | --- | --- | ---: | ---:\n"
                "J1 | Jurema:7b | Judge A | principal | 2 | 4.5"
            ),
        ]
    )
    connection_factory = FakeConnectionFactory(
        [
            [[("Jurema:7b", "7B", "candidato")]],
            [[("J1", "Jurema:7b", "Judge A", "principal", 2, 4.5)]],
        ]
    )
    service = _assistant_service(llm_client=llm, connect_func=connection_factory)

    response = service.answer("Mostre a média de notas por modelo candidato, separando por juiz avaliador.")

    assert response["in_scope"] is True
    assert "4.5" in response["answer"]
    assert "5.0" not in response["answer"]
    assert len(llm.prompts) == 2


def test_assistant_allows_valid_av2_answer_with_error_terms_after_llm() -> None:
    llm = FakeLlmClient("J2: 8 notas 5 e 2 notas 1; estes erros representam respostas com nota 1.")
    connection_factory = FakeConnectionFactory(
        [
            [_rows_for_known_models()],
            [[("OAB_Exames", "Jurema:7b", "7B", 10, 12, 8, 2, 4.25)]],
        ]
    )
    service = _assistant_service(llm_client=llm, connect_func=connection_factory)

    response = service.answer("Mostre desempenho no J2 por acertos, erros e notas.")

    assert response["in_scope"] is True
    assert response["answer"] == "J2: 8 notas 5 e 2 notas 1; estes erros representam respostas com nota 1."
    assert len(llm.prompts) == 1


def test_assistant_allows_candidate_model_summary_by_model_term() -> None:
    llm = FakeLlmClient("Resumo factual do modelo Jurema.")
    connection_factory = FakeConnectionFactory([[_rows_for_known_models()], _rows_for_database_context()])
    service = _assistant_service(llm_client=llm, connect_func=connection_factory)

    response = service.answer("Mostre um resumo do modelo Jurema.")

    assert response["answer"] == "Resumo factual do modelo Jurema."
    assert response["in_scope"] is True
    assert len(llm.prompts) >= 1


def test_assistant_omits_blocked_metric_labels_from_known_model_answer() -> None:
    llm = FakeLlmClient("Resumo factual do modelo Jurema.\n- Há 2 falhas críticas.\n- Média: 4,2.")
    connection_factory = FakeConnectionFactory([[_rows_for_known_models()], _rows_for_database_context()])
    service = _assistant_service(llm_client=llm, connect_func=connection_factory)

    response = service.answer("Mostre um resumo do candidato Jurema.")

    assert response["answer"] == "Resumo factual do modelo Jurema.\n- Média: 4,2."
    assert response["in_scope"] is True


def test_assistant_allows_candidate_model_average_by_name_without_model_term() -> None:
    llm = FakeLlmClient("A média do Jurema é 4,2.")
    connection_factory = FakeConnectionFactory([[_rows_for_known_models()], _rows_for_database_context()])
    service = _assistant_service(llm_client=llm, connect_func=connection_factory)

    response = service.answer("Qual a média do Jurema?")

    assert response["answer"] == "A média do Jurema é 4,2."
    assert response["in_scope"] is True
    assert len(llm.prompts) >= 1


def test_assistant_allows_candidate_model_average_by_known_name() -> None:
    llm = FakeLlmClient("A média do modelo Jurema é 4,2.")
    connection_factory = FakeConnectionFactory([[_rows_for_known_models()], _rows_for_database_context()])
    service = _assistant_service(llm_client=llm, connect_func=connection_factory)

    response = service.answer("Qual a média do modelo Jurema?")

    assert response["answer"] == "A média do modelo Jurema é 4,2."
    assert response["in_scope"] is True
    assert len(llm.prompts) >= 1


def test_assistant_allows_candidate_model_results_by_name_without_model_term() -> None:
    llm = FakeLlmClient("Resultados factuais do Jurema.")
    connection_factory = FakeConnectionFactory([[_rows_for_known_models()], _rows_for_database_context()])
    service = _assistant_service(llm_client=llm, connect_func=connection_factory)

    response = service.answer("Mostre os resultados do Jurema.")

    assert response["answer"] == "Resultados factuais do Jurema."
    assert response["in_scope"] is True
    assert len(llm.prompts) >= 1


def test_assistant_allows_candidate_model_comparison_by_known_names() -> None:
    llm = FakeLlmClient("Comparação factual entre Jurema e Qwen.")
    connection_factory = FakeConnectionFactory([[_rows_for_known_models()], _rows_for_database_context()])
    service = _assistant_service(llm_client=llm, connect_func=connection_factory)

    response = service.answer("Compare Jurema com Qwen.")

    assert response["answer"] == "Comparação factual entre Jurema e Qwen."
    assert response["in_scope"] is True
    assert len(llm.prompts) >= 1


def test_assistant_allows_primary_judge_results() -> None:
    llm = FakeLlmClient("Resultados factuais do juiz principal.")
    service = _assistant_service(llm_client=llm)

    response = service.answer("Mostre os resultados do juiz principal.")

    assert response["answer"] == "Resultados factuais do juiz principal."
    assert response["in_scope"] is True
    assert len(llm.prompts) >= 1


def test_assistant_allows_arbiter_trigger_count() -> None:
    llm = FakeLlmClient("O árbitro foi acionado 2 vezes.")
    connection_factory = FakeConnectionFactory([[[("J1", 301, 101, 201, "Grok", "Judge C", 3, "score_delta_2", "Justificativa")]]])
    service = _assistant_service(llm_client=llm, connect_func=connection_factory)

    response = service.answer("Quantas vezes o árbitro foi acionado?")

    assert response["answer"] == "O árbitro foi acionado 2 vezes."
    assert response["in_scope"] is True
    assert "score_delta_2" in llm.prompts[-1]


def test_assistant_treats_member_word_as_model_when_name_matches_known_model() -> None:
    llm = FakeLlmClient("Consulta factual do modelo Jurema.")
    connection_factory = FakeConnectionFactory([[_rows_for_known_models()], _rows_for_database_context()])
    service = _assistant_service(llm_client=llm, connect_func=connection_factory)

    response = service.answer("Avalie o integrante Jurema.")

    assert response["answer"] == "Consulta factual do modelo Jurema."
    assert response["in_scope"] is True
    assert len(llm.prompts) >= 1


def test_assistant_blocks_member_evaluation_when_name_is_not_known_model() -> None:
    llm = FakeLlmClient()
    connection_factory = FakeConnectionFactory([[("Qwen", None, "candidato")]])
    service = _assistant_service(llm_client=llm, connect_func=connection_factory)

    response = service.answer("Avalie o integrante Jurema.")

    assert response["answer"] == DEFAULT_OUT_OF_SCOPE_ANSWER
    assert response["in_scope"] is False
    assert llm.prompts == []


def test_assistant_allows_factual_audit_failure_count_question() -> None:
    llm = FakeLlmClient("Não encontrei registros para esse critério nas tabelas consultadas.")
    service = _assistant_service(llm_client=llm)

    response = service.answer("Quantas falhas operacionais existem nas auditorias carregadas?")

    assert response["in_scope"] is True
    assert response["answer"] == "Não encontrei registros para esse critério nas tabelas consultadas."
    assert len(llm.prompts) >= 1


def test_assistant_allows_readme_documented_execution_modes_question() -> None:
    llm = FakeLlmClient("Os modos de execução documentados são `single`, `primary_only`, `2plus1` e `2plus1 --always-run-arbiter`.")
    service = _assistant_service(llm_client=llm)

    response = service.answer("Quais os Modos de execução podem ser utilizadas?")

    assert response["answer"] == "Os modos de execução documentados são `single`, `primary_only`, `2plus1` e `2plus1 --always-run-arbiter`."
    assert response["in_scope"] is True
    assert response["sources_used"] == ["README.md"]
    assert len(llm.prompts) >= 1
    assert '"readme"' in llm.prompts[-1]
    assert "Contexto local permitido" not in llm.prompts[-1]


def test_assistant_uses_readme_only_for_audit_documentation_question() -> None:
    llm = FakeLlmClient("A auditoria aparece na documentação como amostra de execução.")
    service = _assistant_service(llm_client=llm)

    response = service.answer("Como funciona a auditoria na documentação?")

    assert response["in_scope"] is True
    assert response["sources_used"] == ["README.md"]
    assert '"readme"' in llm.prompts[-1]


def test_assistant_allows_readme_documented_execution_modes_without_readme_term() -> None:
    llm = FakeLlmClient("Existem os modos `single`, `primary_only`, `2plus1` e `2plus1 --always-run-arbiter`.")
    service = _assistant_service(llm_client=llm)

    response = service.answer("Quais modos de execução existem?")

    assert response["in_scope"] is True
    assert response["sources_used"] == ["README.md"]
    assert len(llm.prompts) >= 1


def test_assistant_allows_readme_documented_2plus1_question() -> None:
    llm = FakeLlmClient("O modo `2plus1` roda dois primários e chama árbitro se houver divergência.")
    service = _assistant_service(llm_client=llm)

    response = service.answer("Como funciona o modo 2plus1?")

    assert response["in_scope"] is True
    assert response["sources_used"] == ["README.md"]
    assert len(llm.prompts) >= 1


def test_assistant_allows_readme_documented_run_evaluation_question() -> None:
    llm = FakeLlmClient("Para rodar a avaliação, use `.venv/bin/python -m atividade_2.cli run-judge` com os parâmetros documentados.")
    service = _assistant_service(llm_client=llm)

    response = service.answer("Como rodar a avaliação?")

    assert response["in_scope"] is True
    assert response["sources_used"] == ["README.md"]
    assert len(llm.prompts) >= 1


def test_assistant_allows_readme_documented_screen_or_filters_question() -> None:
    llm = FakeLlmClient("A Web UI permite configurar, validar e acompanhar execuções do `run-judge`.")
    service = _assistant_service(llm_client=llm)

    response = service.answer("Como funciona a tela de execução?")

    assert response["in_scope"] is True
    assert response["sources_used"] == ["README.md"]
    assert len(llm.prompts) >= 1


def test_assistant_blocks_general_question_without_readme_basis() -> None:
    llm = FakeLlmClient()
    service = _assistant_service(llm_client=llm)

    response = service.answer("Explique LLM-as-a-Judge em geral.")

    assert response["answer"] == DEFAULT_OUT_OF_SCOPE_ANSWER
    assert response["in_scope"] is False
    assert llm.prompts == []


def test_assistant_readme_question_does_not_require_database_entity_lookup() -> None:
    llm = FakeLlmClient("O modo `2plus1` roda dois primários e chama árbitro se houver divergência.")

    def connect_func(database_url: str):
        raise AssertionError("README query should not require database lookup")

    service = _assistant_service(llm_client=llm, connect_func=connect_func)

    response = service.answer("Como funciona o modo 2plus1?")

    assert response["in_scope"] is True
    assert response["sources_used"] == ["README.md"]
    assert len(llm.prompts) >= 1


def test_assistant_database_context_is_read_only_and_uses_fixed_selects() -> None:
    connection = FakeConnection()
    service = _assistant_service(
        connect_func=lambda database_url: connection,
        llm_client=FakeLlmClient("Dados do banco resumidos."),
    )

    response = service.answer("Quais dados existem no banco?")

    assert response["in_scope"] is True
    assert connection.readonly is True
    assert connection.closed is True
    assert connection.cursor_obj.queries
    assert all(query.lstrip().upper().startswith("SELECT") for query in connection.cursor_obj.queries)
