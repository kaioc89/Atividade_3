"""Read-only scoped assistant for the local AV2 web app."""

from __future__ import annotations

import json
import re
import unicodedata
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .config import load_settings
from .dashboard import DashboardFilters, DashboardService
from .db import connect
from .model_aliases import JUDGE_MODEL_ALIASES


DEFAULT_OUT_OF_SCOPE_ANSWER = (
    "Não posso responder a essa pergunta porque ela está fora do escopo deste app. "
    "Posso ajudar com resultados carregados, auditorias, banco de dados, README ou dúvidas de uso da aplicação."
)
DEFAULT_SUGGESTIONS = [
    "Mostre um resumo dos resultados carregados.",
    "Quais auditorias foram realizadas?",
    "Quais dados existem no banco?",
    "Como uso os filtros da tela?",
    "O que está documentado no README?",
]

_WRITE_OPERATION_PATTERN = re.compile(
    r"\b(insert|update|delete|drop|alter|truncate|create)\b"
    r"|\b(crie|criar|apague|apagar|remova|remover|exclua|excluir|limpe|limpar)\b",
    re.IGNORECASE,
)
_IN_SCOPE_PATTERN = re.compile(
    r"\b("
    r"resultado|resultados|carregado|carregados|dashboard|avaliacao|avaliacoes|avaliação|avaliações|"
    r"meta-análise|meta-analise|metaanálise|metaanalise|meta análise|meta analise|"
    r"meta-avaliação|meta-avaliacao|meta avaliação|meta avaliacao|"
    r"auditoria|auditorias|audit|banco|database|dados|tabela|tabelas|readme|documentado|documentacao|"
    r"documentação|uso|usar|filtro|filtros|tela|aplicacao|aplicação|app|prompt|prompts|rubrica|rubricas|modelo|modelos|candidato|"
    r"candidatos|juiz|juizes|juízes|arbitro|árbitro|avaliador|avaliadores|dataset|datasets|ia|ias|top|ranking"
    r")\b",
    re.IGNORECASE,
)
_BLOCKED_ANALYSIS_PATTERN = re.compile(
    r"\b("
    r"falha|falhas|failure|failures|limitacao|limitacoes|limitação|limitações|problema|problemas|erro|erros|"
    r"inconsistencia|inconsistencias|inconsistência|inconsistências|lacuna|lacunas|"
    r"responsabilidade|responsavel|responsável|culpa|culpado|culpada|mérito|merito|"
    r"falta de dados|dados insuficientes|insuficiencia de dados|insuficiência de dados|"
    r"conceito"
    r")\b",
    re.IGNORECASE,
)
_STALE_SQL_ANALYSIS_PATTERN = re.compile(
    r"\b("
    r"não é possível calcular|nao e possivel calcular|não posso calcular|nao posso calcular|"
    r"contexto local insuficiente|contexto insuficiente|não há registros específicos que permitam|"
    r"nao ha registros especificos que permitam|não detalham as notas individuais|nao detalham as notas individuais"
    r")\b",
    re.IGNORECASE,
)
_SUBJECTIVE_BLOCK_PATTERN = re.compile(
    r"\b("
    r"avalie|avaliar|avalia|avaliacao|avaliação|julgue|julgar|critique|criticar|crítica|critica|"
    r"nota|conceito|desempenho|qualidade|mérito|merito|culpa|culpado|culpada|responsável|"
    r"responsavel|responsabilidade|limitação|limitações|limitacao|limitacoes|falha|falhas|"
    r"erro metodológico|erros metodológicos|falta de dados"
    r")\b.*\b("
    r"integrante|integrantes|membro|membros|equipe|grupo|pessoa|pessoas|projeto|trabalho|metodologia|dados"
    r")\b"
    r"|\b("
    r"integrante|integrantes|membro|membros|equipe|grupo|pessoa|pessoas|projeto|trabalho|metodologia|dados"
    r")\b.*\b("
    r"avalie|avaliar|avalia|avaliacao|avaliação|julgue|julgar|critique|criticar|crítica|critica|"
    r"nota|conceito|desempenho|qualidade|mérito|merito|culpa|culpado|culpada|responsável|"
    r"responsavel|responsabilidade|limitação|limitações|limitacao|limitacoes|falha|falhas|"
    r"erro metodológico|erros metodológicos|falta de dados"
    r")\b",
    re.IGNORECASE,
)
_PERSON_SUBJECTIVE_PATTERN = re.compile(
    r"\b("
    r"avalie|avaliar|avalia|julgue|julgar|critique|criticar|"
    r"desempenho|qualidade|mérito|merito|culpa|culpado|culpada|nota|conceito|"
    r"bom trabalho|boa trabalho|melhor|pior"
    r")\b",
    re.IGNORECASE,
)
_FACTUAL_AUDIT_PATTERN = re.compile(
    r"\b("
    r"auditoria|auditorias|audit|meta-avaliação|meta-avaliacao|meta-avaliações|meta-avaliacoes|"
    r"meta avaliação|meta avaliacao|meta avaliações|meta avaliacoes|"
    r"meta-análise|meta-analise|metaanálise|metaanalise|meta análise|meta analise|"
    r"avaliador|avaliadores|auditor|auditores|revisão humana|revisao humana|"
    r"casos auditados|notas auditadas|divergência|divergencias|divergência|divergências|discordaram|"
    r"realizou|realizadas|feitas|feitos|quantas|quantos|liste|lista|resumo|resultados"
    r")\b",
    re.IGNORECASE,
)
_README_OR_APP_DOCS_QUERY = "readme_or_app_docs_query"
_APP_DOCS_QUERY_PATTERN = re.compile(
    r"\b("
    r"readme|documentado|documentada|documentados|documentadas|documentacao|documentação|docs?|"
    r"modo de execucao|modos de execucao|modo de execução|modos de execução|execucao|execução|"
    r"configuracao|configuração|parametro|parametros|parâmetro|parâmetros|como usar|como funciona|"
    r"comando|rodar|executar|filtro|filtros|dashboard|tela|auditoria|auditorias|restore|exportar|"
    r"exportacao|exportação|avaliacao|avaliação"
    r")\b",
    re.IGNORECASE,
)
_DOC_TOKEN_STOPWORDS = {
    "como",
    "quais",
    "qual",
    "para",
    "pode",
    "podem",
    "ser",
    "sao",
    "são",
    "uma",
    "umas",
    "uso",
    "usar",
    "funciona",
    "existem",
    "utilizadas",
    "utilizados",
    "fazer",
    "sobre",
    "esta",
    "está",
    "documentado",
    "documentada",
}
_ALLOWED_TABLES = (
    "datasets",
    "modelos",
    "perguntas",
    "respostas_atividade_1",
    "avaliacoes_juiz",
    "avaliacao_juiz_detalhes",
    "prompt_juizes",
    "meta_avaliacoes",
)
_CONTEXT_TOOL_CATALOG = {
    "dashboard_summary": "Cards agregados por dataset, incluindo totais e concordância entre juízes.",
    "candidate_rankings": "Ranking dos modelos candidatos por dataset.",
    "candidate_spearman": "Spearman por modelo candidato e dataset quando disponível.",
    "arbiter_triggers": "Quantidade de acionamentos do árbitro por dataset.",
    "judge_disagreements": "Casos com maior divergência entre juiz principal e juiz controle.",
    "database_summary": "Tabelas permitidas e contagens de linhas.",
    "human_audits": "Resumo factual das auditorias/meta-avaliações humanas registradas.",
    "audit_logs": "Resumo neutro dos logs de execução do juiz.",
    "readme": "Trechos do README interno permitidos.",
    "usage": "Contexto de uso da Web UI e filtros.",
}
_DEFAULT_CONTEXT_TOOLS = ("dashboard_summary", "database_summary", "usage")
_NO_RECORDS_ANSWER = "Não encontrei registros para esse critério nas tabelas consultadas."
_INTENT_MODEL_COUNTS = "MODEL_COUNTS"
_INTENT_SCORE_MEAN_BY_CANDIDATE_AND_JUDGE = "SCORE_MEAN_BY_CANDIDATE_AND_JUDGE"
_INTENT_JUDGE_DIVERGENCE = "JUDGE_DIVERGENCE"
_INTENT_ARBITER_CASES = "ARBITER_CASES"
_INTENT_TRACE_EVALUATION = "TRACE_EVALUATION"
_INTENT_ACTIVE_PROMPTS_AND_RUBRICS = "ACTIVE_PROMPTS_AND_RUBRICS"
_INTENT_J2_PERFORMANCE = "J2_PERFORMANCE"
_INTENT_EXTREME_DISAGREEMENTS = "EXTREME_DISAGREEMENTS"
_INTENT_MODEL_NAME_SEARCH = "MODEL_NAME_SEARCH"
_INTENT_AUDIT_CASE_RECOMMENDATION = "AUDIT_CASE_RECOMMENDATION"
_INTENT_AVERAGE_SCORES_BY_CANDIDATE_AND_JUDGE = "average_scores_by_candidate_and_judge"
_INTENT_CANDIDATE_RANKING = "candidate_ranking"
_INTENT_CANDIDATE_SPEARMAN = "candidate_spearman"
_INTENT_JUDGE_DISAGREEMENTS = "judge_disagreements"
_INTENT_ARBITER_TRIGGERS = "arbiter_triggers"
_INTENT_ACTIVE_JUDGE_PROMPTS_AND_RUBRICS = "active_judge_prompts_and_rubrics"
_INTENT_HUMAN_AUDIT_SUMMARY = "human_audit_summary"
_INTENT_APP_USAGE_DOCUMENTATION = "app_usage_documentation"
_INTENT_DATABASE_SUMMARY = "database_summary"
_INTENT_GENERAL_RESULTS = "general_results"
_SQL_DETERMINISTIC_INTENTS = {
    _INTENT_MODEL_COUNTS,
    _INTENT_JUDGE_DIVERGENCE,
    _INTENT_ARBITER_CASES,
    _INTENT_TRACE_EVALUATION,
    _INTENT_J2_PERFORMANCE,
    _INTENT_EXTREME_DISAGREEMENTS,
    _INTENT_MODEL_NAME_SEARCH,
    _INTENT_AUDIT_CASE_RECOMMENDATION,
}
_SINGLE_ENTITY_INTENTS = {_INTENT_GENERAL_RESULTS}
_ASSISTANT_ENTITY_TERMS_PATTERN = re.compile(
    r"\b("
    r"candidato|candidatos|modelo candidato|modelo candidatos|modelo|modelos|"
    r"ia|ias|ranking|top|dataset|datasets|"
    r"juiz|juizes|juízes|modelo juiz|modelo avaliador|avaliador automático|avaliador automatico|"
    r"arbitro|árbitro"
    r")\b",
    re.IGNORECASE,
)
_JUDGE_ROLE_PATTERN = re.compile(r"\b(juiz principal|juiz controle|árbitro|arbitro)\b", re.IGNORECASE)
_ENTITY_SEPARATOR_PATTERN = re.compile(r"[_\-:/\.]+")
_DUPLICATE_SPACE_PATTERN = re.compile(r"\s+")
_GENERIC_ENTITY_TOKENS = {
    "7b",
    "q4",
    "k",
    "m",
    "gguf",
    "modelo",
    "model",
    "candidato",
    "juiz",
    "arbitro",
    "auditor",
    "avaliador",
    "principal",
    "controle",
}


@dataclass(frozen=True)
class AssistantEntity:
    kind: str
    value: str


@dataclass(frozen=True)
class ResolvedEntity:
    kind: str
    value: str
    matched_text: str
    score: int


@dataclass(frozen=True)
class EntityResolution:
    matches: tuple[ResolvedEntity, ...]
    status: str
    query_text: str
    normalized_query: str
    preferred_kinds: tuple[str, ...]

    @property
    def first(self) -> AssistantEntity | None:
        if not self.matches:
            return None
        match = self.matches[0]
        return AssistantEntity(kind=match.kind, value=match.value)


@dataclass(frozen=True)
class _KnownEntity:
    kind: str
    value: str
    variants: tuple[str, ...]


class AssistantLlmClient(Protocol):
    def complete(self, prompt: str) -> str:
        """Return an assistant answer for a fully assembled prompt."""


@dataclass(frozen=True)
class RemoteAssistantLlmClient:
    """OpenAI-compatible assistant client using existing remote judge settings."""

    settings_loader: Callable[[], Any] = load_settings
    timeout_seconds: int | None = None

    def complete(self, prompt: str) -> str:
        settings = self.settings_loader()
        if not settings.remote_judge_base_url:
            raise RuntimeError("REMOTE_JUDGE_BASE_URL is required for the assistant.")
        if not settings.remote_judge_api_key:
            raise RuntimeError("REMOTE_JUDGE_API_KEY is required for the assistant.")
        if not settings.remote_judge_default_model:
            raise RuntimeError("REMOTE_JUDGE_MODEL is required for the assistant.")

        payload = {
            "model": settings.remote_judge_default_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Você é o assistente read-only do app AV2. "
                        "Responda apenas com base no contexto local fornecido."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": min(settings.remote_judge_max_tokens, 1200),
            "top_p": settings.remote_judge_top_p,
        }
        url = _chat_completions_url(settings.remote_judge_base_url)
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {settings.remote_judge_api_key}",
                "User-Agent": "atividade-2-assistant/0.1",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout_seconds or settings.remote_judge_timeout_seconds,
            ) as response:
                raw_body = response.read(1_000_001)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"Assistant LLM returned HTTP {error.code}.") from error
        except (urllib.error.URLError, TimeoutError) as error:
            raise RuntimeError(f"Assistant LLM request failed: {error}") from error

        if len(raw_body) > 1_000_000:
            raise RuntimeError("Assistant LLM response exceeded the maximum allowed size.")
        try:
            parsed = json.loads(raw_body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise RuntimeError("Assistant LLM returned invalid JSON.") from error
        if not isinstance(parsed, dict):
            raise RuntimeError("Assistant LLM JSON response must be an object.")
        return _extract_llm_text(parsed)


class AssistantService:
    """Answer only scoped read-only questions about local app data."""

    def __init__(
        self,
        *,
        llm_client: AssistantLlmClient | None = None,
        dashboard_service: DashboardService | None = None,
        audit_log_summary_service: Any | None = None,
        settings_loader: Callable[[], Any] = load_settings,
        connect_func: Callable[[str], Any] = connect,
        readme_loader: Callable[[], str] | None = None,
    ) -> None:
        self._llm_client = llm_client or RemoteAssistantLlmClient(settings_loader=settings_loader)
        self._dashboard_service = dashboard_service or DashboardService()
        self._audit_log_summary_service = audit_log_summary_service
        self._settings_loader = settings_loader
        self._connect = connect_func
        self._readme_loader = readme_loader or _load_readme

    def answer(self, message: str) -> dict[str, Any]:
        question = message.strip()
        if _WRITE_OPERATION_PATTERN.search(question):
            return _blocked_response()

        factual_human_audit_query = is_factual_human_audit_query(question)
        docs_match = (
            None
            if factual_human_audit_query or _should_prefer_database_context(question)
            else _lookup_app_docs_context(question, self._readme_loader())
        )
        if docs_match is not None:
            prompt = _build_app_docs_prompt(question, docs_match)
            answer = self._llm_client.complete(prompt).strip()
            blocked_output = _validate_output(answer, question)
            if blocked_output is not None:
                return blocked_output
            return {
                "answer": answer,
                "in_scope": True,
                "suggestions": DEFAULT_SUGGESTIONS,
                "sources_used": docs_match["sources_used"],
            }

        resolution = resolveAssistantEntities(
            question,
            settings_loader=self._settings_loader,
            connect_func=self._connect,
        )
        entity = resolution.first
        scope = classify_scope(question, resolution)
        if scope != "allowed":
            scope = classify_scope(question, resolution)
        if scope != "allowed":
            return _blocked_response()
        if entity is None:
            entity = _generic_allowed_entity(question)

        answer = self._answer_with_evidence_loop(
            question,
            resolution=resolution,
            factual_human_audit_query=factual_human_audit_query,
        )
        answer = (
            answer
            if _is_sql_backed_intent_question(question, factual_human_audit_query=factual_human_audit_query)
            else _sanitize_entity_answer(answer, entity)
        ).strip()
        blocked_output = _validate_output(answer, question)
        if blocked_output is not None:
            return blocked_output
        return {
            "answer": answer,
            "in_scope": True,
            "suggestions": DEFAULT_SUGGESTIONS,
        }

    def _answer_with_evidence_loop(
        self,
        question: str,
        *,
        resolution: EntityResolution,
        factual_human_audit_query: bool,
    ) -> str:
        intent = classify_intent(question, factual_human_audit_query=factual_human_audit_query)
        if intent in _SINGLE_ENTITY_INTENTS and resolution.status == "ambiguous":
            return _format_entity_refinement_answer(resolution)
        if intent == _INTENT_SCORE_MEAN_BY_CANDIDATE_AND_JUDGE:
            payload = self._load_average_scores_by_candidate_and_judge(question, resolution)
            completeness = _validate_context_completeness(intent, payload)
            if completeness["complete"]:
                return self._answer_sql_payload_with_llm(
                    question,
                    intent,
                    payload,
                    _format_average_scores_answer(payload),
                )
            expanded = self._expand_context(intent, question, resolution, payload)
            return self._answer_sql_payload_with_llm(
                question,
                intent,
                expanded,
                _format_average_scores_answer(expanded),
            )
        if intent == _INTENT_ACTIVE_PROMPTS_AND_RUBRICS:
            payload = self._load_active_judge_prompts_and_rubrics(question)
            return self._answer_sql_payload_with_llm(
                question,
                intent,
                payload,
                _format_active_judge_prompts_answer(payload),
            )
        if intent in _SQL_DETERMINISTIC_INTENTS:
            return self._answer_deterministic_sql_intent(intent, question, resolution)

        tools = _context_tools_for_intent(intent, question)
        context = self._load_named_context_tools(
            tools,
            question=question,
            factual_human_audit_query=factual_human_audit_query,
        )
        context["assistant_trace"] = {
            "attempts": [
                {
                    "attempt": 1,
                    "stage": "initial_retrieval",
                    "intent": intent,
                    "sources_used": sorted(tools),
                    "resolved_entities": [match.value for match in resolution.matches],
                }
            ],
            "answer_mode": "llm_language_only",
        }
        answer = self._llm_client.complete(_build_context_answer_prompt(question, context, set(tools), final_attempt=True)).strip()
        structured_answer = _parse_context_plan(answer)
        if structured_answer is not None:
            if structured_answer["sufficient"] and structured_answer["answer"]:
                structured_text = str(structured_answer["answer"])
                if not _tool_names_mentioned(structured_text):
                    return structured_text
            return _deterministic_context_answer(question, context, set(tools))
        if _tool_names_mentioned(answer):
            return _deterministic_context_answer(question, context, set(tools))
        return answer

    def _load_named_context_tools(
        self,
        tool_names: list[str],
        *,
        question: str,
        factual_human_audit_query: bool,
    ) -> dict[str, Any]:
        context: dict[str, Any] = {}
        dashboard_payload: dict[str, Any] | None = None
        database_payload: dict[str, Any] | None = None

        def dashboard() -> dict[str, Any]:
            nonlocal dashboard_payload
            if dashboard_payload is None:
                dashboard_payload = self._load_dashboard_context(question)
            return dashboard_payload

        def database() -> dict[str, Any]:
            nonlocal database_payload
            if database_payload is None:
                database_payload = self._load_database_context()
            return database_payload

        for tool_name in tool_names:
            if tool_name not in _CONTEXT_TOOL_CATALOG:
                continue
            if tool_name == "dashboard_summary":
                payload = dashboard()
                context.setdefault("dashboard", {})
                context["dashboard"].update(
                    {
                        "available": payload.get("available", False),
                        "cards_by_dataset": payload.get("cards_by_dataset", {}),
                        "cards": payload.get("cards", {}),
                        "methodology_by_dataset": payload.get("methodology_by_dataset", {}),
                    }
                )
            elif tool_name == "candidate_rankings":
                context.setdefault("dashboard", {})
                context["dashboard"]["rankings_by_dataset"] = dashboard().get("rankings_by_dataset", {})
            elif tool_name == "candidate_spearman":
                context.setdefault("dashboard", {})
                context["dashboard"]["spearman_by_candidate_by_dataset"] = dashboard().get(
                    "spearman_by_candidate_by_dataset",
                    {},
                )
            elif tool_name == "arbiter_triggers":
                context.setdefault("dashboard", {})
                context["dashboard"]["arbiter_triggers_by_dataset"] = dashboard().get(
                    "arbiter_triggers_by_dataset",
                    {},
                )
            elif tool_name == "judge_disagreements":
                payload = dashboard()
                context.setdefault("dashboard", {})
                context["dashboard"]["judge_disagreements_by_dataset"] = payload.get(
                    "judge_disagreements_by_dataset",
                    {},
                )
                context["dashboard"].setdefault("cards_by_dataset", payload.get("cards_by_dataset", {}))
            elif tool_name == "database_summary":
                payload = database()
                context.setdefault("banco", {})
                context["banco"]["available"] = payload.get("available", False)
                context["banco"]["tables"] = payload.get("tables", [])
            elif tool_name == "human_audits":
                payload = database()
                context.setdefault("banco", {})
                context["banco"]["available"] = payload.get("available", False)
                context["banco"]["auditorias_humanas"] = payload.get("auditorias_humanas", [])
            elif tool_name == "audit_logs":
                context["logs_execucao_juiz"] = (
                    {"available": False, "skipped": "consulta factual de auditoria humana"}
                    if factual_human_audit_query
                    else self._load_audit_context()
                )
            elif tool_name == "readme":
                context["readme"] = (
                    "README omitido: consultas factuais sobre auditorias/meta-análises realizadas usam primeiro o banco/dados carregados."
                    if factual_human_audit_query
                    else self._readme_loader()[:8000]
                )
            elif tool_name == "usage":
                context["uso_app"] = _usage_context()
        return context

    def _load_dashboard_context(self, question: str = "") -> dict[str, Any]:
        rankings_by_dataset = {}
        spearman_by_candidate_by_dataset = {}
        cards_by_dataset = {}
        arbiter_triggers_by_dataset = {}
        judge_disagreements_by_dataset = {}
        methodology_by_dataset = {}
        base_payload: dict[str, Any] = {}
        include_candidate_spearman = _wants_candidate_spearman(question)
        top_n = _top_n_filter(question)
        try:
            for dataset in ("J1", "J2"):
                dataset_payload = self._dashboard_service.load(DashboardFilters(dataset=dataset))
                if not base_payload:
                    base_payload = dataset_payload
                candidate_ranking = dataset_payload.get("charts", {}).get("candidate_ranking", [])
                rankings_by_dataset[dataset] = candidate_ranking[:top_n] if top_n is not None else candidate_ranking
                if include_candidate_spearman:
                    spearman_by_candidate_by_dataset[dataset] = self._load_candidate_spearman(
                        dataset=dataset,
                        candidate_ranking=candidate_ranking,
                    )
                dataset_cards = dataset_payload.get("cards", {})
                cards_by_dataset[dataset] = dataset_cards
                arbiter_triggers_by_dataset[dataset] = (
                    dataset_cards.get("judge_agreement", {}).get("arbiter_triggered", 0)
                )
                judge_disagreements_by_dataset[dataset] = _top_judge_disagreements(dataset_payload)
                methodology_by_dataset[dataset] = dataset_payload.get("methodology", {})
        except (RuntimeError, ValueError) as error:
            return {"available": False, "error": str(error)}
        return {
            "available": True,
            "rankings_by_dataset": rankings_by_dataset,
            "spearman_by_candidate_by_dataset": spearman_by_candidate_by_dataset,
            "cards_by_dataset": cards_by_dataset,
            "arbiter_triggers_by_dataset": arbiter_triggers_by_dataset,
            "judge_disagreements_by_dataset": judge_disagreements_by_dataset,
            "cards": base_payload.get("cards", {}),
            "options": base_payload.get("options", {}),
            "methodology": base_payload.get("methodology", {}),
            "methodology_by_dataset": methodology_by_dataset,
        }

    def _load_candidate_spearman(
        self,
        *,
        dataset: str,
        candidate_ranking: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        rows = []
        for candidate in candidate_ranking:
            candidate_name = candidate.get("label")
            if not candidate_name:
                continue
            candidate_payload = self._dashboard_service.load(
                DashboardFilters(dataset=dataset, candidate_models=(str(candidate_name),))
            )
            spearman_card = candidate_payload.get("cards", {}).get("spearman_reference", {})
            rows.append(
                {
                    "label": candidate_name,
                    "spearman": spearman_card,
                    "average_score": candidate.get("value"),
                    "count": candidate.get("count"),
                }
            )
        return rows

    def _load_audit_context(self) -> dict[str, Any]:
        if self._audit_log_summary_service is None:
            return {"available": False}
        try:
            payload = self._audit_log_summary_service.load()
        except (RuntimeError, ValueError, OSError) as error:
            return {"available": False, "error": str(error)}
        logs = payload.get("logs", [])
        return {
            "available": payload.get("available", False),
            "totals": _neutral_audit_totals(payload.get("totals", {})),
            "logs": [
                {
                    "run_id": log.get("run_id"),
                    "dataset": log.get("dataset"),
                    "panel_mode": log.get("panel_mode"),
                    "total_events": log.get("total_events"),
                }
                for log in logs[:10]
                if isinstance(log, dict)
            ],
        }

    def _load_database_context(self) -> dict[str, Any]:
        settings = self._settings_loader()
        connection = self._connect(settings.database_url)
        try:
            if hasattr(connection, "set_session"):
                connection.set_session(readonly=True, autocommit=False)
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                      AND table_type = 'BASE TABLE'
                      AND table_name = ANY(%s)
                    ORDER BY table_name;
                    """,
                    (list(_ALLOWED_TABLES),),
                )
                tables = [row[0] for row in cursor.fetchall() if row and row[0] in _ALLOWED_TABLES]
                counts = []
                for table_name in tables:
                    cursor.execute(f"SELECT %s AS table_name, COUNT(*) FROM public.{table_name};", (table_name,))
                    counts.extend({"table": row[0], "rows": int(row[1] or 0)} for row in cursor.fetchall())
                meta_audit_summary = []
                if "meta_avaliacoes" in tables:
                    cursor.execute(
                        """
                        SELECT
                            nm_avaliador,
                            COUNT(*) AS total_auditorias,
                            AVG(vl_nota)::float AS media_nota_meta,
                            COUNT(*) FILTER (WHERE vl_nota IS DISTINCT FROM a.nota_atribuida) AS divergencias_juiz
                        FROM public.meta_avaliacoes ma
                        LEFT JOIN public.avaliacoes_juiz a ON a.id_avaliacao = ma.id_avaliacao
                        GROUP BY nm_avaliador
                        ORDER BY nm_avaliador
                        LIMIT 50;
                        """
                    )
                    meta_audit_summary = [
                        {
                            "avaliador": row[0],
                            "total_auditorias": int(row[1] or 0),
                            "media_nota_meta": row[2],
                            "divergencias_juiz": int(row[3] or 0),
                        }
                        for row in cursor.fetchall()
                    ]
            return {"available": True, "tables": counts, "auditorias_humanas": meta_audit_summary}
        except Exception as error:
            if isinstance(error, AssertionError):
                raise
            return {"available": False, "error": str(error)}
        finally:
            connection.close()

    def _load_average_scores_by_candidate_and_judge(
        self,
        question: str,
        resolution: EntityResolution,
    ) -> dict[str, Any]:
        settings = self._settings_loader()
        connection = self._connect(settings.database_url)
        candidate_names = [
            match.value
            for match in resolution.matches
            if match.kind == "model_candidate"
        ]
        dataset_names = _dataset_filter_names(question)
        top_n = _top_n_filter(question)
        try:
            if hasattr(connection, "set_session"):
                connection.set_session(readonly=True, autocommit=False)
            with connection.cursor() as cursor:
                clauses = ["a.nota_atribuida IS NOT NULL"]
                params: list[Any] = []
                if dataset_names is not None:
                    clauses.append("d.nome_dataset = ANY(%s)")
                    params.append(dataset_names)
                if candidate_names:
                    clauses.append("cm.nome_modelo = ANY(%s)")
                    params.append(candidate_names)
                where_sql = f"WHERE {' AND '.join(clauses)}"
                limit_sql = "LIMIT %s" if top_n is not None else ""
                if top_n is not None:
                    params.append(top_n)
                cursor.execute(
                    f"""
                    SELECT
                        d.nome_dataset,
                        cm.nome_modelo AS candidate_model,
                        jm.nome_modelo AS judge_model,
                        COALESCE(a.papel_juiz, '') AS judge_role,
                        COUNT(*) AS total_scores,
                        AVG(a.nota_atribuida)::float AS average_score
                    FROM public.avaliacoes_juiz a
                    JOIN public.respostas_atividade_1 r ON r.id_resposta = a.id_resposta_ativa1
                    JOIN public.modelos cm ON cm.id_modelo = r.id_modelo
                    JOIN public.modelos jm ON jm.id_modelo = a.id_modelo_juiz
                    JOIN public.perguntas p ON p.id_pergunta = r.id_pergunta
                    JOIN public.datasets d ON d.id_dataset = p.id_dataset
                    {where_sql}
                    GROUP BY d.nome_dataset, cm.nome_modelo, jm.nome_modelo, a.papel_juiz
                    ORDER BY d.nome_dataset, cm.nome_modelo, jm.nome_modelo, a.papel_juiz
                    {limit_sql};
                    """,
                    params,
                )
                rows = [
                    {
                        "dataset": row[0],
                        "candidate_model": row[1],
                        "judge_model": row[2],
                        "judge_role": row[3],
                        "count": int(row[4] or 0),
                        "average_score": round(float(row[5]), 2) if row[5] is not None else None,
                    }
                    for row in cursor.fetchall()
                ]
            return {
                "available": True,
                "rows": rows,
                "filters": {
                    "dataset": _dataset_filter(question),
                    "candidate_models": candidate_names,
                    "top_n": top_n,
                },
                "attempts": [
                    {
                        "attempt": 1,
                        "stage": "initial_retrieval",
                        "intent": _INTENT_AVERAGE_SCORES_BY_CANDIDATE_AND_JUDGE,
                        "sources_used": ["postgres"],
                        "queries_executed": ["average_scores_by_candidate_and_judge"],
                        "resolved_entities": candidate_names,
                    }
                ],
            }
        except Exception as error:
            if isinstance(error, AssertionError):
                raise
            return {"available": False, "error": str(error), "rows": [], "filters": {}, "attempts": []}
        finally:
            connection.close()

    def _load_active_judge_prompts_and_rubrics(self, question: str) -> dict[str, Any]:
        settings = self._settings_loader()
        connection = self._connect(settings.database_url)
        dataset_names = _dataset_filter_names(question)
        try:
            if hasattr(connection, "set_session"):
                connection.set_session(readonly=True, autocommit=False)
            with connection.cursor() as cursor:
                clauses = ["p.ativo = TRUE"]
                params: list[Any] = []
                if dataset_names is not None:
                    clauses.append("d.nome_dataset = ANY(%s)")
                    params.append(dataset_names)
                cursor.execute(
                    f"""
                    SELECT
                        d.nome_dataset,
                        p.versao,
                        p.ativo,
                        p.created_by,
                        LENGTH(p.ds_prompt) AS prompt_chars,
                        LENGTH(p.ds_rubrica) AS rubric_chars,
                        LEFT(p.ds_prompt, 300) AS prompt_preview,
                        LEFT(p.ds_rubrica, 500) AS rubric_preview
                    FROM public.prompt_juizes p
                    JOIN public.datasets d ON d.id_dataset = p.id_dataset
                    WHERE {' AND '.join(clauses)}
                    ORDER BY d.nome_dataset, p.versao DESC;
                    """,
                    params,
                )
                rows = [
                    {
                        "dataset": row[0],
                        "version": int(row[1]),
                        "active": bool(row[2]),
                        "created_by": row[3],
                        "prompt_chars": int(row[4] or 0),
                        "rubric_chars": int(row[5] or 0),
                        "prompt_preview": row[6],
                        "rubric_preview": row[7],
                    }
                    for row in cursor.fetchall()
                ]
            return {
                "available": True,
                "rows": rows,
                "filters": {"dataset": _dataset_filter(question)},
                "attempts": [
                    {
                        "attempt": 1,
                        "stage": "initial_retrieval",
                        "intent": _INTENT_ACTIVE_JUDGE_PROMPTS_AND_RUBRICS,
                        "sources_used": ["postgres"],
                        "queries_executed": ["active_judge_prompts_and_rubrics"],
                    }
                ],
            }
        except Exception as error:
            if isinstance(error, AssertionError):
                raise
            return {"available": False, "error": str(error), "rows": [], "filters": {}, "attempts": []}
        finally:
            connection.close()

    def _answer_deterministic_sql_intent(
        self,
        intent: str,
        question: str,
        resolution: EntityResolution,
    ) -> str:
        loaders = {
            _INTENT_MODEL_COUNTS: self._load_model_counts,
            _INTENT_JUDGE_DIVERGENCE: self._load_judge_divergence,
            _INTENT_ARBITER_CASES: self._load_arbiter_cases,
            _INTENT_TRACE_EVALUATION: self._load_trace_evaluation,
            _INTENT_J2_PERFORMANCE: self._load_j2_performance,
            _INTENT_EXTREME_DISAGREEMENTS: self._load_extreme_disagreements,
            _INTENT_MODEL_NAME_SEARCH: self._load_model_name_search,
            _INTENT_AUDIT_CASE_RECOMMENDATION: self._load_audit_case_recommendations,
        }
        payload = loaders[intent](question, resolution)
        return self._answer_sql_payload_with_llm(
            question,
            intent,
            payload,
            _format_deterministic_sql_answer(intent, payload),
        )

    def _answer_sql_payload_with_llm(
        self,
        question: str,
        intent: str,
        payload: dict[str, Any],
        evidence_answer: str,
    ) -> str:
        prompt = _build_sql_analysis_prompt(question, intent, payload, evidence_answer)
        answer = self._llm_client.complete(prompt).strip()
        if _is_stale_sql_analysis_answer(answer, evidence_answer):
            repair_prompt = _build_sql_analysis_repair_prompt(question, intent, evidence_answer, answer)
            repaired_answer = self._llm_client.complete(repair_prompt).strip()
            if not _is_stale_sql_analysis_answer(repaired_answer, evidence_answer):
                return repaired_answer
            return evidence_answer
        return answer

    def _load_model_counts(self, question: str, resolution: EntityResolution) -> dict[str, Any]:
        dataset_names = _dataset_filter_names(question)
        return self._fetch_sql_rows(
            """
            SELECT
                d.nome_dataset,
                cm.nome_modelo,
                COALESCE(cm.versao, ''),
                COUNT(DISTINCT r.id_resposta) AS total_respostas,
                COUNT(a.id_avaliacao) AS total_avaliacoes
            FROM public.respostas_atividade_1 r
            JOIN public.modelos cm ON cm.id_modelo = r.id_modelo
            JOIN public.perguntas p ON p.id_pergunta = r.id_pergunta
            JOIN public.datasets d ON d.id_dataset = p.id_dataset
            LEFT JOIN public.avaliacoes_juiz a ON a.id_resposta_ativa1 = r.id_resposta
            LEFT JOIN public.prompt_juizes pj ON pj.id_prompt_juiz = a.id_prompt_juiz
            WHERE (%s::text[] IS NULL OR d.nome_dataset = ANY(%s))
            GROUP BY d.nome_dataset, cm.nome_modelo, cm.versao
            ORDER BY d.nome_dataset, cm.nome_modelo
            LIMIT %s;
            """,
            [dataset_names, dataset_names, _top_n_filter(question) or 100],
            lambda row: {
                "dataset": row[0],
                "candidate_model": row[1],
                "version": row[2],
                "answer_count": int(row[3] or 0),
                "evaluation_count": int(row[4] or 0),
            },
        )

    def _load_judge_divergence(self, question: str, resolution: EntityResolution) -> dict[str, Any]:
        dataset_names = _dataset_filter_names(question)
        return self._fetch_sql_rows(
            """
            SELECT *
            FROM (
            WITH pairs AS (
            SELECT
                d.nome_dataset,
                r.id_resposta,
                p.id_pergunta,
                cm.nome_modelo,
                principal_judge.nome_modelo,
                controle_judge.nome_modelo,
                principal.nota_atribuida,
                controle.nota_atribuida,
                ABS(principal.nota_atribuida - controle.nota_atribuida) AS delta
            FROM public.avaliacoes_juiz principal
            JOIN public.avaliacoes_juiz controle
              ON controle.id_resposta_ativa1 = principal.id_resposta_ativa1
             AND controle.id_avaliacao <> principal.id_avaliacao
            JOIN public.respostas_atividade_1 r ON r.id_resposta = principal.id_resposta_ativa1
            JOIN public.perguntas p ON p.id_pergunta = r.id_pergunta
            JOIN public.datasets d ON d.id_dataset = p.id_dataset
            JOIN public.modelos cm ON cm.id_modelo = r.id_modelo
            JOIN public.modelos principal_judge ON principal_judge.id_modelo = principal.id_modelo_juiz
            JOIN public.modelos controle_judge ON controle_judge.id_modelo = controle.id_modelo_juiz
            LEFT JOIN public.prompt_juizes pj ON pj.id_prompt_juiz = principal.id_prompt_juiz
            WHERE translate(lower(COALESCE(principal.papel_juiz, '')), 'áàâãéêíóôõúç', 'aaaaeeiooouc') LIKE '%%principal%%'
              AND translate(lower(COALESCE(controle.papel_juiz, '')), 'áàâãéêíóôõúç', 'aaaaeeiooouc') LIKE '%%controle%%'
              AND (%s::text[] IS NULL OR d.nome_dataset = ANY(%s))
            )
            SELECT *
            FROM pairs
            WHERE delta = (SELECT MAX(delta) FROM pairs)
            ORDER BY delta DESC, id_resposta
            LIMIT %s
            ) AS highest_divergences;
            """,
            [dataset_names, dataset_names, _top_n_filter(question) or 25],
            lambda row: {
                "dataset": row[0],
                "answer_id": row[1],
                "question_id": row[2],
                "candidate_model": row[3],
                "principal_judge": row[4],
                "control_judge": row[5],
                "principal_score": int(row[6]),
                "control_score": int(row[7]),
                "delta": int(row[8]),
            },
        )

    def _load_arbiter_cases(self, question: str, resolution: EntityResolution) -> dict[str, Any]:
        dataset_names = _dataset_filter_names(question)
        return self._fetch_sql_rows(
            """
            SELECT
                d.nome_dataset,
                a.id_avaliacao,
                r.id_resposta,
                p.id_pergunta,
                cm.nome_modelo,
                jm.nome_modelo,
                a.nota_atribuida,
                COALESCE(a.motivo_acionamento, ''),
                COALESCE(a.chain_of_thought, '')
            FROM public.avaliacoes_juiz a
            JOIN public.respostas_atividade_1 r ON r.id_resposta = a.id_resposta_ativa1
            JOIN public.perguntas p ON p.id_pergunta = r.id_pergunta
            JOIN public.datasets d ON d.id_dataset = p.id_dataset
            JOIN public.modelos cm ON cm.id_modelo = r.id_modelo
            JOIN public.modelos jm ON jm.id_modelo = a.id_modelo_juiz
            LEFT JOIN public.prompt_juizes pj ON pj.id_prompt_juiz = a.id_prompt_juiz
            WHERE translate(lower(COALESCE(a.papel_juiz, '')), 'áàâãéêíóôõúç', 'aaaaeeiooouc') LIKE '%%arbitro%%'
              AND (%s::text[] IS NULL OR d.nome_dataset = ANY(%s))
            ORDER BY a.data_avaliacao DESC, a.id_avaliacao DESC
            LIMIT %s;
            """,
            [dataset_names, dataset_names, _top_n_filter(question) or 25],
            lambda row: {
                "dataset": row[0],
                "evaluation_id": row[1],
                "answer_id": row[2],
                "question_id": row[3],
                "candidate_model": row[4],
                "judge_model": row[5],
                "score": int(row[6]),
                "trigger_reason": row[7],
                "rationale": row[8],
            },
        )

    def _load_trace_evaluation(self, question: str, resolution: EntityResolution) -> dict[str, Any]:
        dataset_names = _dataset_filter_names(question)
        ids = _numeric_ids(question)
        target_id = ids[0] if ids else None
        payload = self._fetch_sql_rows(
            """
            SELECT
                d.nome_dataset,
                a.id_avaliacao,
                p.id_pergunta,
                p.enunciado,
                r.id_resposta,
                r.texto_resposta,
                cm.nome_modelo,
                jm.nome_modelo,
                COALESCE(a.papel_juiz, ''),
                a.nota_atribuida,
                a.chain_of_thought,
                COALESCE(pj.versao, 0),
                COALESCE(pj.ds_prompt, ''),
                COALESCE(pj.ds_rubrica, '')
            FROM public.avaliacoes_juiz a
            JOIN public.respostas_atividade_1 r ON r.id_resposta = a.id_resposta_ativa1
            JOIN public.perguntas p ON p.id_pergunta = r.id_pergunta
            JOIN public.datasets d ON d.id_dataset = p.id_dataset
            JOIN public.modelos cm ON cm.id_modelo = r.id_modelo
            JOIN public.modelos jm ON jm.id_modelo = a.id_modelo_juiz
            LEFT JOIN public.prompt_juizes pj ON pj.id_prompt_juiz = a.id_prompt_juiz
            WHERE (%s::text[] IS NULL OR d.nome_dataset = ANY(%s))
              AND (%s IS NULL OR a.id_avaliacao = %s OR r.id_resposta = %s OR p.id_pergunta = %s)
            ORDER BY a.data_avaliacao DESC, a.id_avaliacao DESC
            LIMIT 1;
            """,
            [dataset_names, dataset_names, target_id, target_id, target_id, target_id],
            lambda row: {
                "dataset": row[0],
                "evaluation_id": row[1],
                "question_id": row[2],
                "question": row[3],
                "answer_id": row[4],
                "candidate_answer": row[5],
                "candidate_model": row[6],
                "judge_model": row[7],
                "judge_role": row[8],
                "score": int(row[9]),
                "rationale": row[10],
                "prompt_version": row[11],
                "prompt": row[12],
                "rubric": row[13],
            },
        )
        if payload.get("rows") or target_id is None:
            return payload
        return self._fetch_sql_rows(
            """
            SELECT
                d.nome_dataset,
                a.id_avaliacao,
                p.id_pergunta,
                p.enunciado,
                r.id_resposta,
                r.texto_resposta,
                cm.nome_modelo,
                jm.nome_modelo,
                COALESCE(a.papel_juiz, ''),
                a.nota_atribuida,
                a.chain_of_thought,
                COALESCE(pj.versao, 0),
                COALESCE(pj.ds_prompt, ''),
                COALESCE(pj.ds_rubrica, '')
            FROM public.avaliacoes_juiz a
            JOIN public.respostas_atividade_1 r ON r.id_resposta = a.id_resposta_ativa1
            JOIN public.perguntas p ON p.id_pergunta = r.id_pergunta
            JOIN public.datasets d ON d.id_dataset = p.id_dataset
            JOIN public.modelos cm ON cm.id_modelo = r.id_modelo
            JOIN public.modelos jm ON jm.id_modelo = a.id_modelo_juiz
            LEFT JOIN public.prompt_juizes pj ON pj.id_prompt_juiz = a.id_prompt_juiz
            WHERE (%s::text[] IS NULL OR d.nome_dataset = ANY(%s))
            ORDER BY a.data_avaliacao DESC, a.id_avaliacao DESC
            LIMIT 1;
            """,
            [dataset_names, dataset_names],
            lambda row: {
                "dataset": row[0],
                "evaluation_id": row[1],
                "question_id": row[2],
                "question": row[3],
                "answer_id": row[4],
                "candidate_answer": row[5],
                "candidate_model": row[6],
                "judge_model": row[7],
                "judge_role": row[8],
                "score": int(row[9]),
                "rationale": row[10],
                "prompt_version": row[11],
                "prompt": row[12],
                "rubric": row[13],
            },
        )

    def _load_j2_performance(self, question: str, resolution: EntityResolution) -> dict[str, Any]:
        return self._fetch_sql_rows(
            """
            SELECT
                d.nome_dataset,
                cm.nome_modelo,
                COALESCE(cm.versao, ''),
                COUNT(DISTINCT r.id_resposta) AS qtd_respostas,
                COUNT(a.id_avaliacao) AS qtd_avaliacoes,
                COUNT(a.id_avaliacao) FILTER (WHERE a.nota_atribuida = 5) AS notas_5,
                COUNT(a.id_avaliacao) FILTER (WHERE a.nota_atribuida = 1) AS notas_1,
                AVG(a.nota_atribuida)::float AS media_nota
            FROM public.avaliacoes_juiz a
            JOIN public.respostas_atividade_1 r ON r.id_resposta = a.id_resposta_ativa1
            JOIN public.perguntas p ON p.id_pergunta = r.id_pergunta
            JOIN public.datasets d ON d.id_dataset = p.id_dataset
            JOIN public.modelos cm ON cm.id_modelo = r.id_modelo
            JOIN public.modelos jm ON jm.id_modelo = a.id_modelo_juiz
            LEFT JOIN public.prompt_juizes pj ON pj.id_prompt_juiz = a.id_prompt_juiz
            WHERE d.nome_dataset = ANY(%s)
            GROUP BY d.nome_dataset, cm.nome_modelo, cm.versao
            ORDER BY d.nome_dataset, cm.nome_modelo
            LIMIT %s;
            """,
            [_dataset_aliases("J2"), _top_n_filter(question) or 100],
            lambda row: {
                "dataset": row[0],
                "candidate_model": row[1],
                "version": row[2],
                "answer_count": int(row[3] or 0),
                "evaluation_count": int(row[4] or 0),
                "score_5_count": int(row[5] or 0),
                "score_1_count": int(row[6] or 0),
                "average_score": round(float(row[7]), 2) if row[7] is not None else None,
            },
        )

    def _load_extreme_disagreements(self, question: str, resolution: EntityResolution) -> dict[str, Any]:
        dataset_names = _dataset_filter_names(question)
        payload = self._fetch_sql_rows(
            """
            SELECT
                d.nome_dataset,
                r.id_resposta,
                p.id_pergunta,
                cm.nome_modelo,
                high_judge.nome_modelo,
                low_judge.nome_modelo,
                high_score.nota_atribuida,
                low_score.nota_atribuida,
                high_score.chain_of_thought,
                low_score.chain_of_thought
            FROM public.avaliacoes_juiz high_score
            JOIN public.avaliacoes_juiz low_score
              ON low_score.id_resposta_ativa1 = high_score.id_resposta_ativa1
             AND low_score.id_avaliacao <> high_score.id_avaliacao
            JOIN public.respostas_atividade_1 r ON r.id_resposta = high_score.id_resposta_ativa1
            JOIN public.perguntas p ON p.id_pergunta = r.id_pergunta
            JOIN public.datasets d ON d.id_dataset = p.id_dataset
            JOIN public.modelos cm ON cm.id_modelo = r.id_modelo
            JOIN public.modelos high_judge ON high_judge.id_modelo = high_score.id_modelo_juiz
            JOIN public.modelos low_judge ON low_judge.id_modelo = low_score.id_modelo_juiz
            LEFT JOIN public.prompt_juizes pj ON pj.id_prompt_juiz = high_score.id_prompt_juiz
            WHERE high_score.nota_atribuida = 5
              AND low_score.nota_atribuida = 1
              AND (%s::text[] IS NULL OR d.nome_dataset = ANY(%s))
            ORDER BY r.id_resposta
            LIMIT %s;
            """,
            [dataset_names, dataset_names, _top_n_filter(question) or 25],
            lambda row: {
                "dataset": row[0],
                "answer_id": row[1],
                "question_id": row[2],
                "candidate_model": row[3],
                "high_judge": row[4],
                "low_judge": row[5],
                "high_score": int(row[6]),
                "low_score": int(row[7]),
                "high_rationale": row[8],
                "low_rationale": row[9],
            },
        )
        if payload.get("rows"):
            return payload
        fallback = self._fetch_sql_rows(
            """
            SELECT COALESCE(MAX(ABS(a1.nota_atribuida - a2.nota_atribuida)), 0) AS max_delta
            FROM public.avaliacoes_juiz a1
            JOIN public.avaliacoes_juiz a2
              ON a2.id_resposta_ativa1 = a1.id_resposta_ativa1
             AND a2.id_avaliacao > a1.id_avaliacao
            JOIN public.respostas_atividade_1 r ON r.id_resposta = a1.id_resposta_ativa1
            JOIN public.perguntas p ON p.id_pergunta = r.id_pergunta
            JOIN public.datasets d ON d.id_dataset = p.id_dataset
            JOIN public.modelos cm ON cm.id_modelo = r.id_modelo
            LEFT JOIN public.prompt_juizes pj ON pj.id_prompt_juiz = a1.id_prompt_juiz
            WHERE %s::text[] IS NULL OR d.nome_dataset = ANY(%s);
            """,
            [dataset_names, dataset_names],
            lambda row: {"max_delta": int(row[0] or 0)},
        )
        return {
            "available": payload.get("available") or fallback.get("available"),
            "rows": [],
            "max_delta": (fallback.get("rows") or [{"max_delta": 0}])[0].get("max_delta", 0),
        }

    def _load_model_name_search(self, question: str, resolution: EntityResolution) -> dict[str, Any]:
        terms = _model_search_terms(question, resolution)
        pattern = f"%{' '.join(terms)}%" if terms else "%"
        token_pattern = f"%{terms[0]}%" if terms else "%"
        return self._fetch_sql_rows(
            """
            SELECT
                m.nome_modelo,
                COALESCE(m.versao, ''),
                m.tipo_modelo,
                COUNT(DISTINCT r.id_resposta) AS total_respostas,
                COUNT(a.id_avaliacao) AS total_avaliacoes
            FROM public.modelos m
            LEFT JOIN public.respostas_atividade_1 r ON r.id_modelo = m.id_modelo
            LEFT JOIN public.avaliacoes_juiz a ON a.id_resposta_ativa1 = r.id_resposta
            LEFT JOIN public.perguntas p ON p.id_pergunta = r.id_pergunta
            LEFT JOIN public.datasets d ON d.id_dataset = p.id_dataset
            LEFT JOIN public.prompt_juizes pj ON pj.id_prompt_juiz = a.id_prompt_juiz
            WHERE regexp_replace(translate(lower(COALESCE(m.nome_modelo, '')), 'áàâãéêíóôõúç/:._-', 'aaaaeeiooouc     '), '\\s+', ' ', 'g')
                  LIKE regexp_replace(translate(lower(%s), 'áàâãéêíóôõúç/:._-', 'aaaaeeiooouc     '), '\\s+', ' ', 'g')
               OR regexp_replace(translate(lower(COALESCE(m.versao, '')), 'áàâãéêíóôõúç/:._-', 'aaaaeeiooouc     '), '\\s+', ' ', 'g')
                  LIKE regexp_replace(translate(lower(%s), 'áàâãéêíóôõúç/:._-', 'aaaaeeiooouc     '), '\\s+', ' ', 'g')
            GROUP BY m.nome_modelo, m.versao, m.tipo_modelo
            ORDER BY m.nome_modelo, m.versao
            LIMIT %s;
            """,
            [pattern, token_pattern, _top_n_filter(question) or 50],
            lambda row: {
                "model": row[0],
                "version": row[1],
                "type": row[2],
                "answer_count": int(row[3] or 0),
                "evaluation_count": int(row[4] or 0),
            },
        )

    def _load_audit_case_recommendations(self, question: str, resolution: EntityResolution) -> dict[str, Any]:
        dataset_names = _dataset_filter_names(question)
        return self._fetch_sql_rows(
            """
            SELECT *
            FROM (
            WITH pair_deltas AS (
                SELECT
                    a1.id_resposta_ativa1,
                    MAX(ABS(a1.nota_atribuida - a2.nota_atribuida)) AS max_delta
                FROM public.avaliacoes_juiz a1
                JOIN public.avaliacoes_juiz a2
                  ON a2.id_resposta_ativa1 = a1.id_resposta_ativa1
                 AND a2.id_avaliacao > a1.id_avaliacao
                GROUP BY a1.id_resposta_ativa1
            ),
            answer_flags AS (
                SELECT
                    a.id_resposta_ativa1,
                    BOOL_OR(translate(lower(COALESCE(a.papel_juiz, '')), 'áàâãéêíóôõúç', 'aaaaeeiooouc') = 'arbitro') AS arbiter_triggered,
                    BOOL_OR(a.nota_atribuida IN (1, 5)) AS extreme_score,
                    BOOL_OR(LENGTH(BTRIM(COALESCE(a.chain_of_thought, ''))) < 40) AS short_rationale,
                    BOOL_OR(
                        (a.nota_atribuida >= 4 AND regexp_replace(translate(lower(COALESCE(a.chain_of_thought, '')), 'áàâãéêíóôõúç/:._-', 'aaaaeeiooouc     '), '\\s+', ' ', 'g') LIKE '%insuficient%')
                        OR (a.nota_atribuida <= 2 AND regexp_replace(translate(lower(COALESCE(a.chain_of_thought, '')), 'áàâãéêíóôõúç/:._-', 'aaaaeeiooouc     '), '\\s+', ' ', 'g') LIKE '%excelent%')
                    ) AS suspicious_rationale
                FROM public.avaliacoes_juiz a
                GROUP BY a.id_resposta_ativa1
            )
            SELECT
                d.nome_dataset,
                r.id_resposta,
                p.id_pergunta,
                cm.nome_modelo,
                COALESCE(pd.max_delta, 0) AS max_delta,
                COALESCE(af.arbiter_triggered, FALSE),
                COALESCE(af.extreme_score, FALSE),
                COALESCE(af.short_rationale, FALSE),
                COALESCE(af.suspicious_rationale, FALSE)
            FROM public.respostas_atividade_1 r
            JOIN public.perguntas p ON p.id_pergunta = r.id_pergunta
            JOIN public.datasets d ON d.id_dataset = p.id_dataset
            JOIN public.modelos cm ON cm.id_modelo = r.id_modelo
            LEFT JOIN public.avaliacoes_juiz a ON a.id_resposta_ativa1 = r.id_resposta
            LEFT JOIN public.prompt_juizes pj ON pj.id_prompt_juiz = a.id_prompt_juiz
            LEFT JOIN pair_deltas pd ON pd.id_resposta_ativa1 = r.id_resposta
            LEFT JOIN answer_flags af ON af.id_resposta_ativa1 = r.id_resposta
            WHERE (%s::text[] IS NULL OR d.nome_dataset = ANY(%s))
            GROUP BY d.nome_dataset, r.id_resposta, p.id_pergunta, cm.nome_modelo,
                     pd.max_delta, af.arbiter_triggered, af.extreme_score,
                     af.short_rationale, af.suspicious_rationale
            HAVING COALESCE(pd.max_delta, 0) > 0
                OR COALESCE(af.arbiter_triggered, FALSE)
                OR COALESCE(af.extreme_score, FALSE)
                OR COALESCE(af.short_rationale, FALSE)
                OR COALESCE(af.suspicious_rationale, FALSE)
            ORDER BY
                COALESCE(pd.max_delta, 0) DESC,
                COALESCE(af.arbiter_triggered, FALSE) DESC,
                COALESCE(af.extreme_score, FALSE) DESC,
                COALESCE(af.short_rationale, FALSE) DESC,
                r.id_resposta
            LIMIT %s
            ) AS audit_recommendations;
            """,
            [dataset_names, dataset_names, _top_n_filter(question) or 25],
            lambda row: {
                "dataset": row[0],
                "answer_id": row[1],
                "question_id": row[2],
                "candidate_model": row[3],
                "max_delta": int(row[4] or 0),
                "arbiter_triggered": bool(row[5]),
                "extreme_score": bool(row[6]),
                "short_rationale": bool(row[7]),
                "suspicious_rationale": bool(row[8]),
            },
        )

    def _fetch_sql_rows(
        self,
        query: str,
        params: list[Any],
        mapper: Callable[[Any], dict[str, Any]],
    ) -> dict[str, Any]:
        settings = self._settings_loader()
        connection = self._connect(settings.database_url)
        try:
            if hasattr(connection, "set_session"):
                connection.set_session(readonly=True, autocommit=False)
            with connection.cursor() as cursor:
                cursor.execute(query, params)
                rows = [mapper(row) for row in cursor.fetchall()]
            return {"available": True, "rows": rows}
        except Exception as error:
            if isinstance(error, AssertionError):
                raise
            return {"available": False, "error": str(error), "rows": []}
        finally:
            connection.close()

    def _expand_context(
        self,
        intent: str,
        question: str,
        resolution: EntityResolution,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if intent != _INTENT_SCORE_MEAN_BY_CANDIDATE_AND_JUDGE:
            return payload
        filters = payload.get("filters", {})
        if payload.get("rows") or not filters.get("candidate_models"):
            return payload
        expanded = self._load_average_scores_by_candidate_and_judge(
            question,
            EntityResolution(
                matches=(),
                status="none",
                query_text=resolution.query_text,
                normalized_query=resolution.normalized_query,
                preferred_kinds=resolution.preferred_kinds,
            ),
        )
        expanded_attempts = list(payload.get("attempts", []))
        expanded_attempts.append(
            {
                "attempt": 2,
                "stage": "context_expansion",
                "intent": intent,
                "sources_used": ["postgres"],
                "queries_executed": ["average_scores_by_candidate_and_judge"],
                "missing_facts": ["scores_for_resolved_candidate_set"],
                "expansion": "fetch_all_candidate_models_for_dataset",
            }
        )
        expanded["attempts"] = expanded_attempts
        expanded["initial_filters"] = filters
        return expanded

    def resolve_assistant_entity(self, message: str) -> AssistantEntity | None:
        return resolveAssistantEntity(
            message,
            settings_loader=self._settings_loader,
            connect_func=self._connect,
        )

    def resolve_assistant_entities(self, message: str) -> EntityResolution:
        return resolveAssistantEntities(
            message,
            settings_loader=self._settings_loader,
            connect_func=self._connect,
        )


def is_in_scope(message: str) -> bool:
    return classify_scope(message) == "allowed"


def classify_intent(message: str, *, factual_human_audit_query: bool = False) -> str:
    normalized = normalizeSearchText(message)
    if any(term in normalized for term in ("auditoria manual", "revisao manual", "casos para auditoria", "recomende casos", "recomendacao")):
        return _INTENT_AUDIT_CASE_RECOMMENDATION
    if factual_human_audit_query:
        return _INTENT_HUMAN_AUDIT_SUMMARY
    if "spearman" in normalized:
        return _INTENT_CANDIDATE_SPEARMAN
    if any(term in normalized for term in ("rastreabilidade", "rastrear", "trace", "tracar", "traca")):
        return _INTENT_TRACE_EVALUATION
    if "j2" in normalized and any(term in normalized for term in ("acerto", "acertos", "erro", "erros", "desempenho", "gabarito")):
        return _INTENT_J2_PERFORMANCE
    if any(term in normalized for term in ("nota 5", "5 de um", "nota cinco")) and any(term in normalized for term in ("nota 1", "1 de outro", "um de outro")):
        return _INTENT_EXTREME_DISAGREEMENTS
    if (
        any(term in normalized for term in ("busca", "buscar", "procure", "encontre", "liste", "listar", "variacoes", "alias"))
        and any(term in normalized for term in ("modelo", "modelos", "jurema"))
    ):
        return _INTENT_MODEL_NAME_SEARCH
    if any(term in normalized for term in ("quantidade de respostas", "quantas respostas", "modelos candidatos avaliados", "candidatos avaliados")):
        return _INTENT_MODEL_COUNTS
    if _is_app_docs_query(message):
        return _INTENT_APP_USAGE_DOCUMENTATION
    if "rubrica" in normalized or "rubricas" in normalized or "prompt" in normalized or "prompts" in normalized:
        if any(term in normalized for term in ("ativo", "ativos", "vigente", "vigentes", "juiz", "juizes", "dataset")):
            return _INTENT_ACTIVE_PROMPTS_AND_RUBRICS
    if "media" in normalized and any(term in normalized for term in ("juiz", "juizes", "avaliador", "avaliadores")):
        if any(term in normalized for term in ("candidato", "candidatos", "modelo", "modelos", "ia", "ias")):
            return _INTENT_SCORE_MEAN_BY_CANDIDATE_AND_JUDGE
    if any(term in normalized for term in ("divergencia", "divergencias", "discord")) or (
        "principal" in normalized and "controle" in normalized
    ):
        return _INTENT_JUDGE_DIVERGENCE
    if "arbitro" in normalized and any(term in normalized for term in ("acion", "caso", "casos", "vezes", "quantas", "quantos")):
        return _INTENT_ARBITER_CASES
    if any(term in normalized for term in ("ranking", "top", "classifique")):
        return _INTENT_CANDIDATE_RANKING
    if any(term in normalized for term in ("banco", "database", "tabela", "tabelas")):
        return _INTENT_DATABASE_SUMMARY
    return _INTENT_GENERAL_RESULTS


def classify_scope(message: str, entity: AssistantEntity | EntityResolution | None = None) -> str:
    if not message.strip():
        return "blocked"
    if _WRITE_OPERATION_PATTERN.search(message):
        return "blocked"
    factual_av2_data_query = _is_factual_av2_data_query(message, entity)
    if _is_subjective_person_or_team_request(message, entity) and not factual_av2_data_query:
        return "subjective_blocked"
    if _BLOCKED_ANALYSIS_PATTERN.search(message) and not factual_av2_data_query:
        return "blocked"
    if _entity_has_matches(entity):
        return "allowed"
    if factual_av2_data_query:
        return "allowed"
    if not _IN_SCOPE_PATTERN.search(message):
        return "blocked"
    return "allowed"


def classify_readme_or_app_docs_query(message: str) -> str:
    if _is_app_docs_query(message):
        return _README_OR_APP_DOCS_QUERY
    return "not_app_docs_query"


def _context_tools_for_intent(intent: str, question: str) -> list[str]:
    if intent == _INTENT_CANDIDATE_RANKING:
        return _dedupe_context_tools(["dashboard_summary", "candidate_rankings", "usage"])
    if intent == _INTENT_CANDIDATE_SPEARMAN:
        return _dedupe_context_tools(["dashboard_summary", "candidate_spearman", "candidate_rankings", "usage"])
    if intent == _INTENT_JUDGE_DISAGREEMENTS:
        return _dedupe_context_tools(["dashboard_summary", "judge_disagreements", "usage"])
    if intent == _INTENT_ARBITER_TRIGGERS:
        return _dedupe_context_tools(["dashboard_summary", "arbiter_triggers", "usage"])
    if intent == _INTENT_HUMAN_AUDIT_SUMMARY:
        return _dedupe_context_tools(["database_summary", "human_audits", "audit_logs", "readme"])
    if intent == _INTENT_DATABASE_SUMMARY:
        return _dedupe_context_tools(["database_summary", "usage"])
    return _heuristic_context_tools(question)


def _wants_candidate_spearman(message: str) -> bool:
    normalized = normalizeSearchText(message)
    if "spearman" not in normalized:
        return False
    return any(term in normalized for term in ("modelo candidato", "candidato", "candidatos", "modelo", "modelos"))


def is_factual_human_audit_query(message: str) -> bool:
    normalized = normalizeSearchText(message)
    if not normalized:
        return False
    audit_terms = {
        "auditoria",
        "auditorias",
        "audit",
        "auditor",
        "auditores",
        "avaliador",
        "avaliadores",
    }
    tokens = set(normalized.split())
    has_meta_alias = any(
        alias in normalized
        for alias in (
            "meta analise",
            "metaanalise",
            "meta avaliacao",
            "meta avaliacoes",
            "revisao humana",
        )
    )
    if not has_meta_alias and not (tokens & audit_terms):
        return False
    docs_terms = {"como", "funciona", "usar", "uso", "documentado", "documentacao", "readme", "tela", "filtro", "filtros"}
    factual_terms = {
        "resumo",
        "realizada",
        "realizadas",
        "feita",
        "feitas",
        "feito",
        "feitos",
        "quantas",
        "quantos",
        "quais",
        "total",
        "totais",
        "por",
        "avaliador",
        "auditor",
    }
    return bool(has_meta_alias or (tokens & factual_terms)) and not bool((tokens & docs_terms) - factual_terms)


def _is_subjective_person_or_team_request(message: str, entity: AssistantEntity | EntityResolution | None) -> bool:
    entity_kinds = _entity_kinds(entity)
    if entity_kinds & {"model_candidate", "judge_model", "arbiter"}:
        return False
    if _SUBJECTIVE_BLOCK_PATTERN.search(message):
        return True
    if not _PERSON_SUBJECTIVE_PATTERN.search(message):
        return False
    if "auditor" in entity_kinds:
        return not _FACTUAL_AUDIT_PATTERN.search(message)
    return True


def _is_factual_av2_data_query(message: str, entity: AssistantEntity | EntityResolution | None = None) -> bool:
    normalized = normalizeSearchText(message)
    if not normalized:
        return False
    av2_terms = {
        "av2",
        "resultado",
        "resultados",
        "carregado",
        "carregados",
        "avaliacao",
        "avaliacoes",
        "auditoria",
        "auditorias",
        "banco",
        "dados",
        "dataset",
        "datasets",
        "modelo",
        "modelos",
        "candidato",
        "candidatos",
        "juiz",
        "juizes",
        "arbitro",
        "nota",
        "notas",
        "j1",
        "j2",
    }
    factual_terms = {
        "quais",
        "quantas",
        "quantos",
        "liste",
        "listar",
        "mostre",
        "mostrar",
        "calcule",
        "resumo",
        "media",
        "divergencia",
        "desempenho",
        "acertos",
        "erros",
        "rastreabilidade",
        "busca",
        "buscar",
        "recomende",
        "recomendacao",
    }
    tokens = set(normalized.split())
    return bool(tokens & av2_terms) and (bool(tokens & factual_terms) or _entity_has_matches(entity))


def _is_sql_backed_intent_question(message: str, *, factual_human_audit_query: bool = False) -> bool:
    intent = classify_intent(message, factual_human_audit_query=factual_human_audit_query)
    return intent in (_SQL_DETERMINISTIC_INTENTS | {_INTENT_SCORE_MEAN_BY_CANDIDATE_AND_JUDGE, _INTENT_ACTIVE_PROMPTS_AND_RUBRICS})


def _entity_has_matches(entity: AssistantEntity | EntityResolution | None) -> bool:
    if entity is None:
        return False
    if isinstance(entity, EntityResolution):
        return bool(entity.matches)
    return True


def _entity_kinds(entity: AssistantEntity | EntityResolution | None) -> set[str]:
    if entity is None:
        return set()
    if isinstance(entity, EntityResolution):
        return {match.kind for match in entity.matches}
    return {entity.kind}


def resolveAssistantEntity(
    message: str,
    *,
    settings_loader: Callable[[], Any] = load_settings,
    connect_func: Callable[[str], Any] = connect,
) -> AssistantEntity | None:
    return resolveAssistantEntities(
        message,
        settings_loader=settings_loader,
        connect_func=connect_func,
    ).first


def resolveAssistantEntities(
    message: str,
    *,
    settings_loader: Callable[[], Any] = load_settings,
    connect_func: Callable[[str], Any] = connect,
) -> EntityResolution:
    normalized_message = normalizeSearchText(message)
    preferred_kinds = tuple(sorted(_preferred_entity_kinds(message)))
    if not message.strip():
        return EntityResolution(
            matches=(),
            status="none",
            query_text=message,
            normalized_query=normalized_message,
            preferred_kinds=preferred_kinds,
        )
    if _JUDGE_ROLE_PATTERN.search(message):
        role = _JUDGE_ROLE_PATTERN.search(message)
        assert role is not None
        kind = "arbiter" if "rbitro" in role.group(1).lower() else "judge_model"
        return EntityResolution(
            matches=(ResolvedEntity(kind=kind, value=role.group(1), matched_text=role.group(1), score=150),),
            status="single",
            query_text=message,
            normalized_query=normalized_message,
            preferred_kinds=preferred_kinds,
        )

    known_entities = _known_assistant_entities(settings_loader=settings_loader, connect_func=connect_func)
    message_tokens = _relevant_entity_tokens(normalized_message)
    matches_by_entity: dict[tuple[str, str], ResolvedEntity] = {}
    for entity in known_entities:
        if not entity.value:
            continue
        if preferred_kinds and entity.kind not in preferred_kinds:
            continue
        score = 0
        matched_text = ""
        for variant in entity.variants:
            value_tokens = _relevant_entity_tokens(variant)
            token_overlap = message_tokens & value_tokens
            if variant and variant in normalized_message:
                candidate_score = 100 + len(value_tokens)
            elif variant and normalized_message == variant:
                candidate_score = 100 + len(value_tokens)
            elif token_overlap:
                candidate_score = 60 + (10 * len(token_overlap)) + len(value_tokens)
            else:
                continue
            if candidate_score > score:
                score = candidate_score
                matched_text = variant
        if not score:
            continue
        if entity.kind in preferred_kinds:
            score += 25
        key = (entity.kind, entity.value)
        current = matches_by_entity.get(key)
        if current is None or score > current.score:
            matches_by_entity[key] = ResolvedEntity(
                kind=entity.kind,
                value=entity.value,
                matched_text=matched_text,
                score=score,
            )
    matches = tuple(sorted(matches_by_entity.values(), key=lambda match: (-match.score, match.kind, match.value)))
    if matches:
        unique_kinds = {match.kind for match in matches}
        status = "single" if len(matches) == 1 else ("multiple" if len(unique_kinds) == 1 else "ambiguous")
        return EntityResolution(
            matches=matches,
            status=status,
            query_text=message,
            normalized_query=normalized_message,
            preferred_kinds=preferred_kinds,
        )
    if _ASSISTANT_ENTITY_TERMS_PATTERN.search(message) and not _SUBJECTIVE_BLOCK_PATTERN.search(message):
        generic = _ASSISTANT_ENTITY_TERMS_PATTERN.search(message)
        assert generic is not None
        return EntityResolution(
            matches=(
                ResolvedEntity(
                    kind="assistant_entity_term",
                    value=generic.group(1),
                    matched_text=generic.group(1),
                    score=20,
                ),
            ),
            status="single",
            query_text=message,
            normalized_query=normalized_message,
            preferred_kinds=preferred_kinds,
        )
    return EntityResolution(
        matches=(),
        status="none",
        query_text=message,
        normalized_query=normalized_message,
        preferred_kinds=preferred_kinds,
    )


def _known_assistant_entities(
    *,
    settings_loader: Callable[[], Any],
    connect_func: Callable[[str], Any],
) -> tuple[_KnownEntity, ...]:
    entities: list[_KnownEntity] = []
    settings = settings_loader()
    for value in (
        getattr(settings, "remote_judge_default_model", None),
        getattr(settings, "remote_secondary_judge_model", None),
        getattr(settings, "remote_arbiter_judge_model", None),
    ):
        if value:
            entities.append(_known_entity("judge_model", str(value)))
    for alias, provider_model in JUDGE_MODEL_ALIASES.items():
        entities.append(
            _KnownEntity(
                kind="judge_model",
                value=provider_model,
                variants=tuple(sorted(_entity_name_variants(alias) | _entity_name_variants(provider_model))),
            )
        )

    connection = connect_func(settings.database_url)
    try:
        if hasattr(connection, "set_session"):
            connection.set_session(readonly=True, autocommit=False)
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT nome_modelo, versao, tipo_modelo
                FROM public.modelos
                WHERE tipo_modelo IN ('candidato', 'juiz', 'ambos')
                UNION ALL
                SELECT DISTINCT nm_avaliador, NULL::text, 'auditor'
                FROM public.meta_avaliacoes
                WHERE NULLIF(BTRIM(nm_avaliador), '') IS NOT NULL
                ORDER BY 1;
                """
            )
            for name, version, model_type in cursor.fetchall():
                if not name:
                    continue
                if model_type == "auditor":
                    kind = "auditor"
                elif model_type == "juiz":
                    kind = "judge_model"
                elif model_type == "ambos":
                    kind = "judge_model"
                else:
                    kind = "model_candidate"
                variants = set(_entity_name_variants(str(name)))
                if version:
                    variants.update(_entity_name_variants(str(version)))
                entities.append(_KnownEntity(kind=kind, value=str(name), variants=tuple(sorted(variants))))
    except Exception as error:
        if isinstance(error, AssertionError):
            raise
    finally:
        connection.close()
    return tuple(_dedupe_known_entities(entities))


def _known_entity(kind: str, value: str) -> _KnownEntity:
    return _KnownEntity(kind=kind, value=value, variants=tuple(sorted(_entity_name_variants(value))))


def _dedupe_known_entities(entities: list[_KnownEntity]) -> list[_KnownEntity]:
    deduped: dict[tuple[str, str], set[str]] = {}
    for entity in entities:
        deduped.setdefault((entity.kind, entity.value), set()).update(entity.variants)
    return [
        _KnownEntity(kind=kind, value=value, variants=tuple(sorted(variants)))
        for (kind, value), variants in deduped.items()
    ]


def _normalize_entity_text(value: str) -> str:
    return normalizeSearchText(value)


def normalizeSearchText(value: str) -> str:
    without_accents = "".join(
        char for char in unicodedata.normalize("NFKD", value.casefold()) if not unicodedata.combining(char)
    )
    separated = _ENTITY_SEPARATOR_PATTERN.sub(" ", without_accents)
    return _DUPLICATE_SPACE_PATTERN.sub(" ", separated).strip()


def _entity_name_variants(value: str) -> set[str]:
    normalized = normalizeSearchText(value)
    variants = {normalized}
    tokens = normalized.split()
    for token in _relevant_entity_tokens(normalized):
        variants.add(token)
    if tokens and tokens[0] in _relevant_entity_tokens(normalized):
        variants.add(tokens[0])
    for index, token in enumerate(tokens):
        if token in _GENERIC_ENTITY_TOKENS or len(token) < 3:
            continue
        aliases = [token]
        if index + 1 < len(tokens) and tokens[index + 1][0].isdigit():
            aliases.append(f"{token} {tokens[index + 1]}")
        variants.update(alias for alias in aliases if alias)
    return variants


def _relevant_entity_tokens(value: str) -> set[str]:
    return {
        token
        for token in normalizeSearchText(value).split()
        if len(token) >= 3 and token not in _GENERIC_ENTITY_TOKENS and not token.isdigit()
    }


def _preferred_entity_kinds(message: str) -> set[str]:
    normalized = normalizeSearchText(message)
    if "candidato" in normalized:
        return {"model_candidate"}
    if "juiz" in normalized:
        return {"judge_model"}
    if "arbitro" in normalized:
        return {"arbiter"}
    if is_factual_human_audit_query(message):
        return {"auditor"}
    if "modelo" in normalized:
        return {"model_candidate", "judge_model"}
    return set()


def _generic_allowed_entity(message: str) -> AssistantEntity | None:
    preferred_kinds = _preferred_entity_kinds(message)
    if "model_candidate" in preferred_kinds:
        return AssistantEntity(kind="model_candidate", value="modelo candidato")
    if "judge_model" in preferred_kinds:
        return AssistantEntity(kind="judge_model", value="modelo juiz")
    if "arbiter" in preferred_kinds:
        return AssistantEntity(kind="arbiter", value="arbitro")
    if "auditor" in preferred_kinds:
        return AssistantEntity(kind="auditor", value="auditor")
    return None


def _blocked_response() -> dict[str, Any]:
    return {
        "answer": DEFAULT_OUT_OF_SCOPE_ANSWER,
        "in_scope": False,
        "suggestions": DEFAULT_SUGGESTIONS,
    }


def _validate_output(answer: str, question: str = "") -> dict[str, Any] | None:
    if not answer.strip():
        return _blocked_response()
    if answer == _NO_RECORDS_ANSWER or answer.startswith("Com os dados encontrados nas tabelas consultadas"):
        return None
    if _is_sql_backed_intent_question(question):
        return None
    if _BLOCKED_ANALYSIS_PATTERN.search(answer) or _SUBJECTIVE_BLOCK_PATTERN.search(answer):
        return _blocked_response()
    return None


def _is_app_docs_query(message: str) -> bool:
    if not message.strip():
        return False
    if _WRITE_OPERATION_PATTERN.search(message):
        return False
    return bool(_APP_DOCS_QUERY_PATTERN.search(message))


def _should_prefer_database_context(message: str) -> bool:
    normalized = normalizeSearchText(message)
    return any(
        term in normalized
        for term in (
            "rastreabilidade",
            "rastrear",
            "trace",
            "acertos",
            "erros",
            "nota 5",
            "nota 1",
            "auditoria manual",
            "casos para auditoria",
            "recomende casos",
            "quantidade de respostas",
            "modelos candidatos avaliados",
        )
    )


def _lookup_app_docs_context(message: str, readme_text: str) -> dict[str, Any] | None:
    if classify_readme_or_app_docs_query(message) != _README_OR_APP_DOCS_QUERY:
        return None
    if _BLOCKED_ANALYSIS_PATTERN.search(message) or _SUBJECTIVE_BLOCK_PATTERN.search(message):
        return None
    snippets = _relevant_doc_snippets(message, readme_text)
    if not snippets:
        return None
    return {"readme": "\n\n".join(snippets), "sources_used": ["README.md"]}


def _relevant_doc_snippets(message: str, document: str) -> list[str]:
    tokens = _doc_query_tokens(message)
    if not tokens:
        return []
    sections = _markdown_sections(document)
    scored_sections: list[tuple[int, int, str]] = []
    for index, section in enumerate(sections):
        normalized = normalizeSearchText(section)
        score = sum(1 for token in tokens if token in normalized)
        if "2plus1" in message.casefold() and "2plus1" in section:
            score += 3
        if {"modo", "modos", "execucao"} & tokens and "modos de execucao" in normalized:
            score += 3
        if score:
            scored_sections.append((score, -index, section.strip()))
    scored_sections.sort(reverse=True)
    return [section for _, _, section in scored_sections[:3] if section]


def _doc_query_tokens(message: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", normalizeSearchText(message))
        if len(token) >= 3 and token not in _DOC_TOKEN_STOPWORDS
    }


def _markdown_sections(document: str) -> list[str]:
    sections: list[str] = []
    current: list[str] = []
    for line in document.splitlines():
        if line.startswith("#") and current:
            sections.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append("\n".join(current))
    return sections


def _sanitize_entity_answer(answer: str, entity: AssistantEntity | None) -> str:
    if entity is None or entity.kind not in {"model_candidate", "judge_model", "arbiter", "auditor"}:
        return answer
    allowed_lines = [
        line
        for line in answer.splitlines()
        if not _BLOCKED_ANALYSIS_PATTERN.search(line) and not _SUBJECTIVE_BLOCK_PATTERN.search(line)
    ]
    return "\n".join(allowed_lines)


def _neutral_audit_totals(totals: Any) -> dict[str, Any]:
    if not isinstance(totals, dict):
        return {}
    return {
        key: value
        for key, value in totals.items()
        if isinstance(key, str) and not _BLOCKED_ANALYSIS_PATTERN.search(key)
    }


def _top_judge_disagreements(payload: dict[str, Any], *, limit: int = 10) -> list[dict[str, Any]]:
    rows = payload.get("tables", {}).get("judge_agreement_arbitrations", [])
    if not isinstance(rows, list):
        return []
    clean_rows = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        clean_rows.append(
            {
                "answer_id": row.get("answer_id"),
                "question_id": row.get("question_id"),
                "candidate_model": row.get("candidate_model"),
                "judge_1_score": row.get("judge_1_score"),
                "judge_2_score": row.get("judge_2_score"),
                "delta": row.get("delta"),
                "arbiter_score": row.get("arbiter_score"),
                "arbitration_reason": row.get("arbitration_reason"),
            }
        )
    return sorted(clean_rows, key=lambda row: (-(row.get("delta") or 0), row.get("answer_id") or 0))[:limit]


def _heuristic_context_tools(question: str) -> list[str]:
    normalized = normalizeSearchText(question)
    tools = list(_DEFAULT_CONTEXT_TOOLS)
    if any(term in normalized for term in ("ranking", "top", "modelo candidato", "candidato", "candidatos", "ia", "ias")):
        tools.append("candidate_rankings")
    if "spearman" in normalized:
        tools.extend(["candidate_spearman", "candidate_rankings"])
    if "arbitro" in normalized and any(term in normalized for term in ("acion", "vezes", "quantas", "quantos")):
        tools.append("arbiter_triggers")
    if any(term in normalized for term in ("divergencia", "divergencias", "discord", "principal", "controle")):
        tools.append("judge_disagreements")
    if is_factual_human_audit_query(question):
        tools.extend(["human_audits", "audit_logs", "readme"])
    if _is_app_docs_query(question):
        tools.append("readme")
    return _dedupe_context_tools(tools)


def _dedupe_context_tools(tool_names: list[str] | tuple[str, ...]) -> list[str]:
    deduped: list[str] = []
    for tool_name in tool_names:
        if tool_name in _CONTEXT_TOOL_CATALOG and tool_name not in deduped:
            deduped.append(tool_name)
    return deduped


def _tool_names_mentioned(text: str) -> list[str]:
    normalized = normalizeSearchText(text)
    tool_names = [
        tool_name
        for tool_name in _CONTEXT_TOOL_CATALOG
        if normalizeSearchText(tool_name) in normalized
    ]
    return _dedupe_context_tools(tool_names)


def _deterministic_context_answer(question: str, context: dict[str, Any], used_tools: set[str]) -> str:
    normalized = normalizeSearchText(question)
    if "judge_disagreements" in used_tools or any(
        term in normalized for term in ("divergencia", "divergencias", "principal", "controle")
    ):
        return _format_judge_disagreements_answer(context)
    return "Não encontrei contexto local suficiente para responder de forma factual."


def _validate_context_completeness(intent: str, payload: dict[str, Any]) -> dict[str, Any]:
    if intent == _INTENT_SCORE_MEAN_BY_CANDIDATE_AND_JUDGE:
        rows = payload.get("rows", [])
        missing = []
        if not payload.get("available"):
            missing.append("postgres")
        if not rows:
            missing.append("scores")
        if rows and not {row.get("candidate_model") for row in rows if isinstance(row, dict)}:
            missing.append("candidate_models")
        if rows and not {row.get("judge_model") for row in rows if isinstance(row, dict)}:
            missing.append("judge_models")
        return {"complete": not missing, "missing_facts": missing}
    return {"complete": True, "missing_facts": []}


def _format_average_scores_answer(payload: dict[str, Any]) -> str:
    if not payload.get("available"):
        return _NO_RECORDS_ANSWER
    rows = payload.get("rows", [])
    filters = payload.get("filters", {})
    if not rows:
        return _NO_RECORDS_ANSWER

    lines = ["dataset | modelo_candidato | juiz | papel_juiz | qtd_avaliações | média_nota"]
    lines.append("--- | --- | --- | --- | ---: | ---:")
    for row in rows:
        lines.append(
            f"{row.get('dataset')} | {row.get('candidate_model')} | {row.get('judge_model')} | "
            f"{row.get('judge_role') or ''} | {row.get('count')} | {row.get('average_score')}"
        )
    return "\n".join(lines)


def _format_active_judge_prompts_answer(payload: dict[str, Any]) -> str:
    if not payload.get("available"):
        return _NO_RECORDS_ANSWER
    rows = payload.get("rows", [])
    if not rows:
        return _NO_RECORDS_ANSWER
    lines = ["Com os dados encontrados nas tabelas consultadas, prompts e rubricas ativos em prompt_juizes:"]
    for row in rows:
        lines.append(
            "- "
            f"{row.get('dataset')}: versão {row.get('version')}, ativo={row.get('active')}, "
            f"prompt {row.get('prompt_chars')} chars, rubrica {row.get('rubric_chars')} chars, "
            f"created_by={row.get('created_by')}."
        )
        rubric_preview = str(row.get("rubric_preview") or "").strip()
        if rubric_preview:
            lines.append(f"  Rubrica: {_compact_preview(rubric_preview)}")
    return "\n".join(lines)


def _format_deterministic_sql_answer(intent: str, payload: dict[str, Any]) -> str:
    if intent == _INTENT_EXTREME_DISAGREEMENTS and payload.get("available") and not payload.get("rows"):
        return f"Não há casos 5 vs 1. O maior delta encontrado foi {payload.get('max_delta', 0)}."
    if not payload.get("available") or not payload.get("rows"):
        return _NO_RECORDS_ANSWER
    rows = payload["rows"]
    lines = ["Com os dados encontrados nas tabelas consultadas:"]
    if intent == _INTENT_MODEL_COUNTS:
        lines = ["dataset | modelo | versão | qtd_respostas | qtd_avaliações"]
        lines.append("--- | --- | --- | ---: | ---:")
        for row in rows:
            lines.append(
                f"{row['dataset']} | {row['candidate_model']} | {row.get('version') or ''} | "
                f"{row['answer_count']} | {row['evaluation_count']}"
            )
    elif intent == _INTENT_JUDGE_DIVERGENCE:
        lines = ["dataset | id_resposta | id_pergunta | modelo | juiz_principal | nota_principal | juiz_controle | nota_controle | delta"]
        lines.append("--- | ---: | ---: | --- | --- | ---: | --- | ---: | ---:")
        for row in rows:
            lines.append(
                f"{row['dataset']} | {row['answer_id']} | {row['question_id']} | {row['candidate_model']} | "
                f"{row['principal_judge']} | {row['principal_score']} | {row['control_judge']} | "
                f"{row['control_score']} | {row['delta']}"
            )
    elif intent == _INTENT_ARBITER_CASES:
        lines = ["dataset | id_avaliacao | id_resposta | modelo_candidato | arbitro | nota | motivo_acionamento | justificativa_resumida"]
        lines.append("--- | ---: | ---: | --- | --- | ---: | --- | ---")
        for row in rows:
            reason = row.get("trigger_reason") or "motivo não registrado"
            lines.append(
                f"{row['dataset']} | {row['evaluation_id']} | {row['answer_id']} | {row['candidate_model']} | "
                f"{row['judge_model']} | {row['score']} | {reason} | "
                f"{_compact_preview(row.get('rationale', ''), limit=100)}"
            )
    elif intent == _INTENT_TRACE_EVALUATION:
        for row in rows:
            lines.append(
                f"- Avaliação {row['evaluation_id']} ({row['dataset']}): pergunta {row['question_id']} "
                f"{_compact_preview(row['question'], limit=120)} | resposta {row['answer_id']} "
                f"{_compact_preview(row['candidate_answer'], limit=120)} | candidato {row['candidate_model']} | "
                f"juiz {row['judge_model']} ({row['judge_role'] or 'papel não registrado'}) | "
                f"nota {row['score']} | justificativa {_compact_preview(row['rationale'], limit=160)} | "
                f"prompt/rubrica versão {row['prompt_version']}: "
                f"{_compact_preview(row['prompt'], limit=80)} / {_compact_preview(row['rubric'], limit=120)}."
            )
    elif intent == _INTENT_J2_PERFORMANCE:
        lines = ["dataset | modelo | qtd_respostas | qtd_avaliações | notas_5 | notas_1 | média_nota"]
        lines.append("--- | --- | ---: | ---: | ---: | ---: | ---:")
        for row in rows:
            lines.append(
                f"{row['dataset']} | {row['candidate_model']} | {row['answer_count']} | "
                f"{row['evaluation_count']} | {row['score_5_count']} | {row['score_1_count']} | "
                f"{row['average_score']}"
            )
    elif intent == _INTENT_EXTREME_DISAGREEMENTS:
        for row in rows:
            lines.append(
                f"- {row['dataset']} resposta {row['answer_id']} pergunta {row['question_id']}: "
                f"{row['candidate_model']}, {row['high_judge']} deu {row['high_score']} e "
                f"{row['low_judge']} deu {row['low_score']}."
            )
    elif intent == _INTENT_MODEL_NAME_SEARCH:
        for row in rows:
            version = f" versão {row['version']}" if row.get("version") else ""
            lines.append(
                f"- {row['model']}{version} ({row['type']}): "
                f"{row['answer_count']} respostas, {row['evaluation_count']} avaliações."
            )
    elif intent == _INTENT_AUDIT_CASE_RECOMMENDATION:
        for row in rows:
            reasons = []
            if row.get("max_delta"):
                reasons.append(f"delta {row['max_delta']}")
            if row.get("arbiter_triggered"):
                reasons.append("árbitro acionado")
            if row.get("extreme_score"):
                reasons.append("nota extrema")
            if row.get("short_rationale"):
                reasons.append("justificativa curta")
            if row.get("suspicious_rationale"):
                reasons.append("possível inconsistência nota/justificativa")
            lines.append(
                f"- {row['dataset']} resposta {row['answer_id']} pergunta {row['question_id']}: "
                f"{row['candidate_model']} -> {', '.join(reasons) or 'critério de auditoria'}."
            )
    return "\n".join(lines)


def _build_sql_analysis_prompt(question: str, intent: str, payload: dict[str, Any], evidence_answer: str) -> str:
    return (
        "Você é o assistente read-only do app AV2. A pergunta já foi classificada como válida.\n"
        "Analise e sintetize a resposta com base exclusivamente no resultado SQL determinístico abaixo. "
        "Não diga que está fora do escopo e não diga que não é possível calcular se há dados na evidência. "
        f"Se a evidência disser exatamente '{_NO_RECORDS_ANSWER}', responda essa ausência de forma factual e breve. "
        "Mantenha tabelas compactas quando a evidência vier em tabela. Não invente dados.\n\n"
        f"Intent: {intent}\n"
        f"Pergunta do usuário:\n{question}\n\n"
        "Resposta/evidência SQL compacta:\n"
        f"{evidence_answer}\n\n"
        "Payload estruturado resumido:\n"
        f"{json.dumps(_compact_sql_payload(payload), ensure_ascii=False, default=str)}"
    )


def _build_sql_analysis_repair_prompt(question: str, intent: str, evidence_answer: str, stale_answer: str) -> str:
    return (
        "A resposta anterior contradisse a evidência SQL ou caiu em fallback antigo. "
        "Reescreva a resposta final analisando somente a evidência abaixo. "
        "Não diga que não é possível calcular quando a evidência tem linhas. "
        "Não mencione contexto insuficiente. Mantenha tabelas compactas quando houver tabela.\n\n"
        f"Intent: {intent}\n"
        f"Pergunta do usuário:\n{question}\n\n"
        f"Resposta anterior inválida:\n{stale_answer}\n\n"
        f"Evidência SQL autorizada:\n{evidence_answer}"
    )


def _is_stale_sql_analysis_answer(answer: str, evidence_answer: str) -> bool:
    if not answer.strip():
        return True
    if not _STALE_SQL_ANALYSIS_PATTERN.search(answer):
        return False
    has_positive_evidence = bool(evidence_answer.strip()) and evidence_answer.strip() != _NO_RECORDS_ANSWER
    return has_positive_evidence


def _compact_sql_payload(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload.get("rows", [])
    if isinstance(rows, list):
        rows = rows[:30]
    return {
        key: value
        for key, value in {
            "available": payload.get("available"),
            "rows": rows,
            "max_delta": payload.get("max_delta"),
            "error": payload.get("error"),
        }.items()
        if value is not None
    }


def _format_entity_refinement_answer(resolution: EntityResolution) -> str:
    options = ", ".join(f"{match.value} ({match.kind})" for match in resolution.matches[:8])
    return f"Encontrei mais de uma entidade compatível com a pergunta. Refine usando uma destas opções: {options}."


def _compact_preview(value: str, *, limit: int = 240) -> str:
    compact = _DUPLICATE_SPACE_PATTERN.sub(" ", value).strip()
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit].rstrip()}..."


def _dataset_filter(message: str) -> str | None:
    normalized = normalizeSearchText(message)
    if re.search(r"\bj1\b", normalized):
        return "J1"
    if "oab bench" in normalized:
        return "J1"
    if re.search(r"\bj2\b", normalized):
        return "J2"
    if "oab exames" in normalized:
        return "J2"
    return None


def _dataset_aliases(dataset: str) -> list[str]:
    normalized = dataset.upper()
    if normalized == "J1":
        return ["J1", "OAB_Bench"]
    if normalized == "J2":
        return ["J2", "OAB_Exames"]
    return [dataset]


def _dataset_filter_names(message: str) -> list[str] | None:
    dataset = _dataset_filter(message)
    if dataset is None:
        return None
    return _dataset_aliases(dataset)


def _top_n_filter(message: str) -> int | None:
    normalized = normalizeSearchText(message)
    match = re.search(r"\btop\s+(\d{1,3})\b", normalized)
    if match is None:
        return None
    return max(1, min(int(match.group(1)), 100))


def _numeric_ids(message: str) -> list[int]:
    return [int(match) for match in re.findall(r"\b\d{1,10}\b", message)]


def _model_search_terms(message: str, resolution: EntityResolution) -> list[str]:
    named_matches = [match.matched_text for match in resolution.matches if match.kind == "model_candidate"]
    if named_matches:
        return [normalizeSearchText(named_matches[0]).split()[0]]
    normalized = normalizeSearchText(message)
    stopwords = {
        "busca",
        "buscar",
        "busque",
        "procure",
        "encontre",
        "modelo",
        "modelos",
        "por",
        "nome",
        "candidato",
        "candidatos",
        "flexivel",
        "flexivelmente",
    }
    terms = [token for token in normalized.split() if len(token) >= 3 and token not in stopwords]
    return terms[:1]


def _format_judge_disagreements_answer(context: dict[str, Any]) -> str:
    disagreements = context.get("dashboard", {}).get("judge_disagreements_by_dataset", {})
    if not isinstance(disagreements, dict):
        return "Não encontrei casos detalhados de divergência entre juiz principal e juiz controle no contexto local."

    rows_with_dataset: list[tuple[str, dict[str, Any]]] = []
    for dataset, rows in disagreements.items():
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, dict):
                rows_with_dataset.append((str(dataset), row))
    if not rows_with_dataset:
        return "Não encontrei casos detalhados de divergência entre juiz principal e juiz controle no contexto local."

    max_delta = max(int(row.get("delta") or 0) for _, row in rows_with_dataset)
    top_rows = [(dataset, row) for dataset, row in rows_with_dataset if int(row.get("delta") or 0) == max_delta]
    lines = [f"A maior divergência encontrada foi delta {max_delta}:"]
    for dataset, row in top_rows[:10]:
        arbiter_score = row.get("arbiter_score")
        arbiter_text = f", árbitro {arbiter_score}" if arbiter_score is not None else ""
        lines.append(
            "- "
            f"{dataset}: answer_id {row.get('answer_id')}, pergunta {row.get('question_id')}, "
            f"candidato {row.get('candidate_model')}, "
            f"principal {row.get('judge_1_score')}, controle {row.get('judge_2_score')}"
            f"{arbiter_text}."
        )
    return "\n".join(lines)


def _parse_context_plan(raw_response: str) -> dict[str, Any] | None:
    text = raw_response.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    raw_tools = parsed.get("required_context", parsed.get("required_tools", []))
    if isinstance(raw_tools, str):
        raw_tools = [raw_tools]
    if not isinstance(raw_tools, list):
        raw_tools = []
    tools = _dedupe_context_tools([str(tool) for tool in raw_tools])
    sufficient = bool(parsed.get("sufficient", False))
    answer = parsed.get("answer")
    return {"sufficient": sufficient, "tools": tools, "answer": answer if isinstance(answer, str) else ""}


def _build_context_answer_prompt(
    question: str,
    context: dict[str, Any],
    used_tools: set[str],
    *,
    final_attempt: bool = False,
) -> str:
    extra_instruction = (
        "Esta é a tentativa final do fluxo; responda com o que estiver disponível no contexto local."
        if final_attempt
        else (
            "Se o contexto ainda for insuficiente, responda apenas JSON no formato "
            '{"sufficient": false, "required_context": ["nome_da_ferramenta"]}. '
            "Caso contrário, responda ao usuário diretamente."
        )
    )
    return (
        "Você opera em modo estritamente somente leitura.\n"
        "Não altere arquivos, não gere SQL livre, não execute operações de escrita, "
        "não consulte internet, não use arquivos externos e não responda com conhecimento geral.\n"
        "Não faça avaliação subjetiva ou acadêmica sobre integrantes, equipe, grupo, qualidade do trabalho, "
        "responsabilidades, culpa, mérito, nota ou conceito. "
        "Não descreva categorias bloqueadas como falhas, limitações, falta de dados, problemas, erros, "
        "inconsistências ou lacunas. Consultas factuais sobre auditorias e meta-avaliações carregadas são permitidas.\n"
        "A pergunta abaixo já passou pelo classificador de escopo do app. Se ela for sobre modelo candidato, modelo juiz, "
        "árbitro, auditoria ou uso do app, responda factualmente e omita qualquer métrica ou campo cujo rótulo use termos "
        "bloqueados como falhas, problemas, erros, limitações, inconsistências ou lacunas.\n"
        "Neste app, meta-análise, meta-avaliação, auditoria e revisão humana significam a auditoria humana registrada "
        "em public.meta_avaliacoes por membros avaliadores sobre avaliações do Juiz-IA. Para perguntas sobre "
        "auditorias/meta-análises realizadas, use primeiro banco.auditorias_humanas e demais dados carregados; não use "
        "README para responder quais auditorias existem.\n"
        "Para resumo geral, exiba apenas fatos neutros e operacionais: totais carregados, datasets, modelos, "
        "avaliações, auditorias, tabelas disponíveis e consultas possíveis.\n"
        "Para perguntas sobre Spearman geral do dataset, use dashboard.cards_by_dataset[dataset].spearman_reference. "
        "Para perguntas sobre Spearman por modelo candidato, use "
        "dashboard.spearman_by_candidate_by_dataset; não substitua Spearman por média de nota. "
        "Não infira indisponibilidade de J2 a partir de J1: em J2 a referência ordinal é derivada do gabarito "
        "oficial, com acerto = 5 e erro = 1. Para ranking por modelo candidato, use "
        "dashboard.rankings_by_dataset. Para perguntas sobre acionamento do árbitro, use "
        "dashboard.arbiter_triggers_by_dataset; não use somente o dataset base nem troque J1 por J2.\n"
        "Para perguntas sobre maior divergência entre juiz principal e juiz controle, use "
        "dashboard.judge_disagreements_by_dataset para listar os casos e use "
        "dashboard.cards_by_dataset[dataset].judge_agreement apenas para os totais agregados.\n"
        "Não use nem mencione histórico de chat; esta chamada é stateless.\n"
        f"{extra_instruction}\n\n"
        f"Ferramentas de contexto usadas: {json.dumps(sorted(used_tools), ensure_ascii=False)}\n\n"
        f"Pergunta do usuário:\n{question}\n\n"
        "Contexto local permitido:\n"
        f"{json.dumps(context, ensure_ascii=False, default=str)}"
    )


def _build_prompt(question: str, context: dict[str, Any]) -> str:
    return (
        "Você opera em modo estritamente somente leitura.\n"
        "Não altere arquivos, não gere SQL livre, não execute operações de escrita, "
        "não consulte internet, não use arquivos externos e não responda com conhecimento geral.\n"
        "Não faça avaliação subjetiva ou acadêmica sobre integrantes, equipe, grupo, qualidade do trabalho, "
        "responsabilidades, culpa, mérito, nota ou conceito. "
        "Não descreva categorias bloqueadas como falhas, limitações, falta de dados, problemas, erros, "
        "inconsistências ou lacunas. Consultas factuais sobre auditorias e meta-avaliações carregadas são permitidas.\n"
        "A pergunta abaixo já passou pelo classificador de escopo do app. Se ela for sobre modelo candidato, modelo juiz, "
        "árbitro, auditoria ou uso do app, responda factualmente e omita qualquer métrica ou campo cujo rótulo use termos "
        "bloqueados como falhas, problemas, erros, limitações, inconsistências ou lacunas.\n"
        "Neste app, meta-análise, meta-avaliação, auditoria e revisão humana significam a auditoria humana registrada "
        "em public.meta_avaliacoes por membros avaliadores sobre avaliações do Juiz-IA. Para perguntas sobre "
        "auditorias/meta-análises realizadas, use primeiro banco.auditorias_humanas e demais dados carregados; não use "
        "README para responder quais auditorias existem.\n"
        "Para resumo geral, exiba apenas fatos neutros e operacionais: totais carregados, datasets, modelos, "
        "avaliações, auditorias, tabelas disponíveis e consultas possíveis.\n"
        "Para perguntas sobre Spearman geral do dataset, use dashboard.cards_by_dataset[dataset].spearman_reference. "
        "Para perguntas sobre Spearman por modelo candidato, use "
        "dashboard.spearman_by_candidate_by_dataset; não substitua Spearman por média de nota. "
        "Não infira indisponibilidade de J2 a partir de J1: em J2 a referência ordinal é derivada do gabarito "
        "oficial, com acerto = 5 e erro = 1. Para ranking por modelo candidato, use "
        "dashboard.rankings_by_dataset. Para perguntas sobre acionamento do árbitro, use "
        "dashboard.arbiter_triggers_by_dataset; não use somente o dataset base nem troque J1 por J2.\n"
        "Para perguntas sobre maior divergência entre juiz principal e juiz controle, use "
        "dashboard.judge_disagreements_by_dataset para listar os casos e use "
        "dashboard.cards_by_dataset[dataset].judge_agreement apenas para os totais agregados.\n"
        "Não use nem mencione histórico de chat; esta chamada é stateless.\n"
        "Responda somente com base no contexto local abaixo. Se a pergunta exigir conteúdo bloqueado, use o fluxo "
        "de bloqueio do app.\n\n"
        f"Pergunta do usuário:\n{question}\n\n"
        "Contexto local permitido:\n"
        f"{json.dumps(context, ensure_ascii=False, default=str)}"
    )


def _build_app_docs_prompt(question: str, context: dict[str, Any]) -> str:
    return (
        "Você opera em modo estritamente somente leitura.\n"
        "Responda apenas perguntas factuais de uso, configuração ou documentação do app.\n"
        "Use somente o trecho de README/documentação interna fornecido abaixo. "
        "Não use conhecimento geral, internet, banco de dados, SQL livre ou arquivos externos. "
        "Se o trecho não contiver a resposta, diga que a documentação interna fornecida não tem conteúdo suficiente.\n\n"
        f"Pergunta do usuário:\n{question}\n\n"
        "Trecho de documentação interna permitido:\n"
        f"{json.dumps(context, ensure_ascii=False, default=str)}"
    )


def _usage_context() -> dict[str, Any]:
    return {
        "web_ui": "A Web UI local roda em http://127.0.0.1:8000 após make web-up.",
        "tabs": ["Dashboard", "Execução", "Prompt Juizes", "Auditorias", "Histórico"],
        "filters": "Os filtros do dashboard permitem dataset, modelos candidatos, modelos juízes, status e agrupamento.",
        "safety": "Este assistente não executa ações de escrita nem persiste histórico.",
    }


def _load_readme() -> str:
    path = Path("README.md")
    if not path.exists():
        return "README.md indisponível."
    return path.read_text(encoding="utf-8", errors="replace")


def _chat_completions_url(base_url: str) -> str:
    stripped = base_url.rstrip("/")
    if stripped.endswith("/chat/completions"):
        return stripped
    return f"{stripped}/chat/completions"


def _extract_llm_text(raw_response: dict[str, Any]) -> str:
    choices = raw_response.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
            if isinstance(first.get("text"), str):
                return first["text"]
    for key in ("text", "output", "response"):
        value = raw_response.get(key)
        if isinstance(value, str):
            return value
    raise RuntimeError("Assistant LLM response did not contain text.")
