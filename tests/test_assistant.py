from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from atividade_2.assistant import (
    DEFAULT_OUT_OF_SCOPE_ANSWER,
    DEFAULT_SUGGESTIONS,
    AssistantService,
)
from atividade_2.web import create_app


class FakeLlmClient:
    def __init__(self, answer: str = "Resumo gerado pelo assistente.") -> None:
        self.answer = answer
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answer


class FakeDashboardService:
    def __init__(self) -> None:
        self.calls = 0

    def load(self, filters) -> dict:
        self.calls += 1
        return {
            "cards": {
                "evaluations": 3,
                "coverage": {"evaluated": 2, "expected": 4, "percent": 50.0},
                "success_rate": 100.0,
                "average_score": 4.5,
            },
            "tables": {"critical_cases": []},
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
    llm_client: FakeLlmClient | None = None,
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
    assert len(llm.prompts) == 1
    assert "modo estritamente somente leitura" in llm.prompts[0]


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


def test_assistant_blocks_request_about_failures_with_default_answer() -> None:
    llm = FakeLlmClient()
    service = _assistant_service(llm_client=llm)

    response = service.answer("Quais falhas aparecem nos resultados carregados?")

    assert response["answer"] == DEFAULT_OUT_OF_SCOPE_ANSWER
    assert response["in_scope"] is False
    assert llm.prompts == []


def test_assistant_blocks_request_to_point_project_limitations() -> None:
    llm = FakeLlmClient()
    service = _assistant_service(llm_client=llm)

    response = service.answer("Aponte as limitações do projeto como avaliação acadêmica.")

    assert response["answer"] == DEFAULT_OUT_OF_SCOPE_ANSWER
    assert response["in_scope"] is False
    assert llm.prompts == []


def test_assistant_blocks_request_about_limitations_with_default_answer() -> None:
    llm = FakeLlmClient()
    service = _assistant_service(llm_client=llm)

    response = service.answer("Quais são as limitações das avaliações carregadas?")

    assert response["answer"] == DEFAULT_OUT_OF_SCOPE_ANSWER
    assert response["in_scope"] is False
    assert llm.prompts == []


def test_assistant_blocks_request_to_assign_blame_for_missing_data() -> None:
    llm = FakeLlmClient()
    service = _assistant_service(llm_client=llm)

    response = service.answer("Quem é culpado pela falta de dados carregados no banco?")

    assert response["answer"] == DEFAULT_OUT_OF_SCOPE_ANSWER
    assert response["in_scope"] is False
    assert llm.prompts == []


def test_assistant_blocks_request_about_missing_data_with_default_answer() -> None:
    llm = FakeLlmClient()
    service = _assistant_service(llm_client=llm)

    response = service.answer("Existe falta de dados nos datasets carregados?")

    assert response["answer"] == DEFAULT_OUT_OF_SCOPE_ANSWER
    assert response["in_scope"] is False
    assert llm.prompts == []


def test_assistant_blocks_generated_forbidden_content_with_default_answer() -> None:
    llm = FakeLlmClient("Há limitações nos dados carregados.")
    service = _assistant_service(llm_client=llm)

    response = service.answer("Mostre um resumo dos resultados carregados.")

    assert response["answer"] == DEFAULT_OUT_OF_SCOPE_ANSWER
    assert response["in_scope"] is False
    assert len(llm.prompts) == 1


def test_assistant_general_summary_uses_neutral_operational_content() -> None:
    llm = FakeLlmClient("Totais carregados: 3 avaliações, 1 auditoria e 3 tabelas disponíveis.")
    service = _assistant_service(llm_client=llm)

    response = service.answer("Mostre um resumo geral dos resultados carregados.")

    assert response["answer"] == "Totais carregados: 3 avaliações, 1 auditoria e 3 tabelas disponíveis."
    assert response["in_scope"] is True
    assert "failures" not in llm.prompts[0]
    assert "error" not in llm.prompts[0]


def test_assistant_allows_factual_loaded_data_count_question() -> None:
    llm = FakeLlmClient("Existem 3 avaliações carregadas.")
    service = _assistant_service(llm_client=llm)

    response = service.answer("Qual é a quantidade de dados carregados?")

    assert response["answer"] == "Existem 3 avaliações carregadas."
    assert response["in_scope"] is True
    assert len(llm.prompts) == 1


def test_assistant_allows_factual_existing_audits_question() -> None:
    llm = FakeLlmClient("Existe 1 auditoria registrada.")
    service = _assistant_service(llm_client=llm)

    response = service.answer("Quais auditorias existentes foram carregadas?")

    assert response["answer"] == "Existe 1 auditoria registrada."
    assert response["in_scope"] is True
    assert len(llm.prompts) == 1


def test_assistant_allows_factual_meta_evaluation_question() -> None:
    llm = FakeLlmClient("Existe 1 meta-avaliação registrada.")
    service = _assistant_service(llm_client=llm)

    response = service.answer("Quais meta-avaliações existem nas auditorias?")

    assert response["answer"] == "Existe 1 meta-avaliação registrada."
    assert response["in_scope"] is True
    assert len(llm.prompts) == 1


def test_assistant_allows_factual_audit_summary_by_known_auditor_name() -> None:
    llm = FakeLlmClient("Diego realizou 2 auditorias, com média 4,5.")
    connection_factory = FakeConnectionFactory([[_rows_for_known_models_and_auditors()], _rows_for_database_context()])
    service = _assistant_service(llm_client=llm, connect_func=connection_factory)

    response = service.answer("Mostre um resumo da auditoria feita por Diego.")

    assert response["answer"] == "Diego realizou 2 auditorias, com média 4,5."
    assert response["in_scope"] is True
    assert len(llm.prompts) == 1
    assert "Diego" in llm.prompts[0]


def test_assistant_allows_factual_audits_by_known_auditor_name() -> None:
    llm = FakeLlmClient("Diego fez auditorias nos casos 101 e 102.")
    connection_factory = FakeConnectionFactory([[_rows_for_known_models_and_auditors()], _rows_for_database_context()])
    service = _assistant_service(llm_client=llm, connect_func=connection_factory)

    response = service.answer("Quais auditorias foram feitas por Diego?")

    assert response["answer"] == "Diego fez auditorias nos casos 101 e 102."
    assert response["in_scope"] is True
    assert len(llm.prompts) == 1


def test_assistant_allows_factual_audit_count_by_known_auditor_name() -> None:
    llm = FakeLlmClient("Diego realizou 2 auditorias.")
    connection_factory = FakeConnectionFactory([[_rows_for_known_models_and_auditors()], _rows_for_database_context()])
    service = _assistant_service(llm_client=llm, connect_func=connection_factory)

    response = service.answer("Quantas auditorias Diego realizou?")

    assert response["answer"] == "Diego realizou 2 auditorias."
    assert response["in_scope"] is True
    assert len(llm.prompts) == 1


def test_assistant_allows_audits_grouped_by_evaluator() -> None:
    llm = FakeLlmClient("Diego: 2 auditorias.")
    service = _assistant_service(llm_client=llm)

    response = service.answer("Mostre auditorias por avaliador.")

    assert response["answer"] == "Diego: 2 auditorias."
    assert response["in_scope"] is True
    assert len(llm.prompts) == 1


def test_assistant_allows_factual_audits_that_disagreed_with_judge() -> None:
    llm = FakeLlmClient("Existe 1 auditoria com divergência factual em relação ao juiz.")
    service = _assistant_service(llm_client=llm)

    response = service.answer("Quais auditorias discordaram do juiz?")

    assert response["answer"] == "Existe 1 auditoria com divergência factual em relação ao juiz."
    assert response["in_scope"] is True
    assert len(llm.prompts) == 1


def test_assistant_allows_meta_evaluation_summary() -> None:
    llm = FakeLlmClient("Resumo factual da meta-avaliação.")
    service = _assistant_service(llm_client=llm)

    response = service.answer("Resumo da meta-avaliação.")

    assert response["answer"] == "Resumo factual da meta-avaliação."
    assert response["in_scope"] is True
    assert len(llm.prompts) == 1


def test_assistant_allows_candidate_model_by_known_name() -> None:
    llm = FakeLlmClient("Resumo factual do modelo Jurema.")
    connection_factory = FakeConnectionFactory([[_rows_for_known_models()], _rows_for_database_context()])
    service = _assistant_service(llm_client=llm, connect_func=connection_factory)

    response = service.answer("Mostre um resumo do candidato Jurema.")

    assert response["answer"] == "Resumo factual do modelo Jurema."
    assert response["in_scope"] is True
    assert len(llm.prompts) == 1
    assert all(connection.readonly for connection in connection_factory.connections)


def test_assistant_resolves_named_candidate_before_generic_candidate_term() -> None:
    connection_factory = FakeConnectionFactory([[_rows_for_known_models()]])
    service = _assistant_service(connect_func=connection_factory)

    entity = service.resolve_assistant_entity("Mostre um resumo do candidato Jurema.")

    assert entity is not None
    assert entity.kind == "model_candidate"
    assert entity.value == "jurema"


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


def test_assistant_allows_candidate_model_summary_by_model_term() -> None:
    llm = FakeLlmClient("Resumo factual do modelo Jurema.")
    connection_factory = FakeConnectionFactory([[_rows_for_known_models()], _rows_for_database_context()])
    service = _assistant_service(llm_client=llm, connect_func=connection_factory)

    response = service.answer("Mostre um resumo do modelo Jurema.")

    assert response["answer"] == "Resumo factual do modelo Jurema."
    assert response["in_scope"] is True
    assert len(llm.prompts) == 1


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
    assert len(llm.prompts) == 1


def test_assistant_allows_candidate_model_average_by_known_name() -> None:
    llm = FakeLlmClient("A média do modelo Jurema é 4,2.")
    connection_factory = FakeConnectionFactory([[_rows_for_known_models()], _rows_for_database_context()])
    service = _assistant_service(llm_client=llm, connect_func=connection_factory)

    response = service.answer("Qual a média do modelo Jurema?")

    assert response["answer"] == "A média do modelo Jurema é 4,2."
    assert response["in_scope"] is True
    assert len(llm.prompts) == 1


def test_assistant_allows_candidate_model_results_by_name_without_model_term() -> None:
    llm = FakeLlmClient("Resultados factuais do Jurema.")
    connection_factory = FakeConnectionFactory([[_rows_for_known_models()], _rows_for_database_context()])
    service = _assistant_service(llm_client=llm, connect_func=connection_factory)

    response = service.answer("Mostre os resultados do Jurema.")

    assert response["answer"] == "Resultados factuais do Jurema."
    assert response["in_scope"] is True
    assert len(llm.prompts) == 1


def test_assistant_allows_candidate_model_comparison_by_known_names() -> None:
    llm = FakeLlmClient("Comparação factual entre Jurema e Qwen.")
    connection_factory = FakeConnectionFactory([[_rows_for_known_models()], _rows_for_database_context()])
    service = _assistant_service(llm_client=llm, connect_func=connection_factory)

    response = service.answer("Compare Jurema com Qwen.")

    assert response["answer"] == "Comparação factual entre Jurema e Qwen."
    assert response["in_scope"] is True
    assert len(llm.prompts) == 1


def test_assistant_allows_primary_judge_results() -> None:
    llm = FakeLlmClient("Resultados factuais do juiz principal.")
    service = _assistant_service(llm_client=llm)

    response = service.answer("Mostre os resultados do juiz principal.")

    assert response["answer"] == "Resultados factuais do juiz principal."
    assert response["in_scope"] is True
    assert len(llm.prompts) == 1


def test_assistant_allows_arbiter_trigger_count() -> None:
    llm = FakeLlmClient("O árbitro foi acionado 2 vezes.")
    service = _assistant_service(llm_client=llm)

    response = service.answer("Quantas vezes o árbitro foi acionado?")

    assert response["answer"] == "O árbitro foi acionado 2 vezes."
    assert response["in_scope"] is True
    assert len(llm.prompts) == 1


def test_assistant_treats_member_word_as_model_when_name_matches_known_model() -> None:
    llm = FakeLlmClient("Consulta factual do modelo Jurema.")
    connection_factory = FakeConnectionFactory([[_rows_for_known_models()], _rows_for_database_context()])
    service = _assistant_service(llm_client=llm, connect_func=connection_factory)

    response = service.answer("Avalie o integrante Jurema.")

    assert response["answer"] == "Consulta factual do modelo Jurema."
    assert response["in_scope"] is True
    assert len(llm.prompts) == 1


def test_assistant_blocks_member_evaluation_when_name_is_not_known_model() -> None:
    llm = FakeLlmClient()
    connection_factory = FakeConnectionFactory([[("Qwen", None, "candidato")]])
    service = _assistant_service(llm_client=llm, connect_func=connection_factory)

    response = service.answer("Avalie o integrante Jurema.")

    assert response["answer"] == DEFAULT_OUT_OF_SCOPE_ANSWER
    assert response["in_scope"] is False
    assert llm.prompts == []


def test_assistant_blocks_factual_audit_failure_count_question() -> None:
    llm = FakeLlmClient()
    service = _assistant_service(llm_client=llm)

    response = service.answer("Quantas falhas operacionais existem nas auditorias carregadas?")

    assert response["answer"] == DEFAULT_OUT_OF_SCOPE_ANSWER
    assert response["in_scope"] is False
    assert llm.prompts == []


def test_assistant_allows_readme_documented_execution_modes_question() -> None:
    llm = FakeLlmClient("Os modos de execução documentados são `single`, `primary_only`, `2plus1` e `2plus1 --always-run-arbiter`.")
    service = _assistant_service(llm_client=llm)

    response = service.answer("Quais os Modos de execução podem ser utilizadas?")

    assert response["answer"] == "Os modos de execução documentados são `single`, `primary_only`, `2plus1` e `2plus1 --always-run-arbiter`."
    assert response["in_scope"] is True
    assert response["sources_used"] == ["README.md"]
    assert len(llm.prompts) == 1
    assert '"readme"' in llm.prompts[0]
    assert "Contexto local permitido" not in llm.prompts[0]


def test_assistant_allows_readme_documented_execution_modes_without_readme_term() -> None:
    llm = FakeLlmClient("Existem os modos `single`, `primary_only`, `2plus1` e `2plus1 --always-run-arbiter`.")
    service = _assistant_service(llm_client=llm)

    response = service.answer("Quais modos de execução existem?")

    assert response["in_scope"] is True
    assert response["sources_used"] == ["README.md"]
    assert len(llm.prompts) == 1


def test_assistant_allows_readme_documented_2plus1_question() -> None:
    llm = FakeLlmClient("O modo `2plus1` roda dois primários e chama árbitro se houver divergência.")
    service = _assistant_service(llm_client=llm)

    response = service.answer("Como funciona o modo 2plus1?")

    assert response["in_scope"] is True
    assert response["sources_used"] == ["README.md"]
    assert len(llm.prompts) == 1


def test_assistant_allows_readme_documented_run_evaluation_question() -> None:
    llm = FakeLlmClient("Para rodar a avaliação, use `.venv/bin/python -m atividade_2.cli run-judge` com os parâmetros documentados.")
    service = _assistant_service(llm_client=llm)

    response = service.answer("Como rodar a avaliação?")

    assert response["in_scope"] is True
    assert response["sources_used"] == ["README.md"]
    assert len(llm.prompts) == 1


def test_assistant_allows_readme_documented_screen_or_filters_question() -> None:
    llm = FakeLlmClient("A Web UI permite configurar, validar e acompanhar execuções do `run-judge`.")
    service = _assistant_service(llm_client=llm)

    response = service.answer("Como funciona a tela de execução?")

    assert response["in_scope"] is True
    assert response["sources_used"] == ["README.md"]
    assert len(llm.prompts) == 1


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
    assert len(llm.prompts) == 1


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
