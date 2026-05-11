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
    r"documentação|uso|usar|filtro|filtros|tela|aplicacao|aplicação|app|modelo|modelos|candidato|"
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
_MAX_CONTEXT_INTERACTIONS = 3
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
        docs_match = None if factual_human_audit_query else _lookup_app_docs_context(question, self._readme_loader())
        if docs_match is not None:
            prompt = _build_app_docs_prompt(question, docs_match)
            answer = self._llm_client.complete(prompt).strip()
            blocked_output = _validate_output(answer)
            if blocked_output is not None:
                return blocked_output
            return {
                "answer": answer,
                "in_scope": True,
                "suggestions": DEFAULT_SUGGESTIONS,
                "sources_used": docs_match["sources_used"],
            }

        entity = None
        scope = classify_scope(question)
        if scope != "allowed":
            entity = self.resolve_assistant_entity(question)
            scope = classify_scope(question, entity)
        if scope != "allowed":
            return _blocked_response()
        if entity is None:
            entity = _generic_allowed_entity(question)

        answer = self._answer_with_iterative_context(
            question,
            entity=entity,
            factual_human_audit_query=factual_human_audit_query,
        )
        answer = _sanitize_entity_answer(answer, entity).strip()
        blocked_output = _validate_output(answer)
        if blocked_output is not None:
            return blocked_output
        return {
            "answer": answer,
            "in_scope": True,
            "suggestions": DEFAULT_SUGGESTIONS,
        }

    def _answer_with_iterative_context(
        self,
        question: str,
        *,
        entity: AssistantEntity | None,
        factual_human_audit_query: bool,
    ) -> str:
        used_tools: set[str] = set()
        context: dict[str, Any] = {}

        planning_prompt = _build_context_planning_prompt(question)
        planning_response = self._llm_client.complete(planning_prompt).strip()
        plan = _parse_context_plan(planning_response)
        planning_leaked_tools = _tool_names_mentioned(planning_response)
        if plan is not None and plan["sufficient"] and plan["answer"] and not planning_leaked_tools:
            return str(plan["answer"])
        requested_tools = planning_leaked_tools or (plan["tools"] if plan is not None else _heuristic_context_tools(question))
        if not requested_tools:
            requested_tools = list(_DEFAULT_CONTEXT_TOOLS)
        context.update(
            self._load_named_context_tools(
                requested_tools,
                question=question,
                factual_human_audit_query=factual_human_audit_query,
            )
        )
        used_tools.update(requested_tools)

        answer = self._llm_client.complete(_build_context_answer_prompt(question, context, used_tools)).strip()
        follow_up = _parse_context_plan(answer)
        if follow_up is not None and not follow_up["sufficient"] and follow_up["tools"]:
            next_tools = [tool for tool in follow_up["tools"] if tool not in used_tools]
            if next_tools:
                context.update(
                    self._load_named_context_tools(
                        next_tools,
                        question=question,
                        factual_human_audit_query=factual_human_audit_query,
                    )
                )
                used_tools.update(next_tools)
                answer = self._llm_client.complete(
                    _build_context_answer_prompt(
                        question,
                        context,
                        used_tools,
                        final_attempt=True,
                    )
                ).strip()
            else:
                return _deterministic_context_answer(question, context, used_tools)
        answer_leaked_tools = _tool_names_mentioned(answer)
        if answer_leaked_tools:
            next_tools = [tool for tool in answer_leaked_tools if tool not in used_tools]
            if next_tools:
                context.update(
                    self._load_named_context_tools(
                        next_tools,
                        question=question,
                        factual_human_audit_query=factual_human_audit_query,
                    )
                )
                used_tools.update(next_tools)
                answer = self._llm_client.complete(
                    _build_context_answer_prompt(
                        question,
                        context,
                        used_tools,
                        final_attempt=True,
                    )
                ).strip()
                if _tool_names_mentioned(answer):
                    return _deterministic_context_answer(question, context, used_tools)
            else:
                return _deterministic_context_answer(question, context, used_tools)
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
        try:
            for dataset in ("J1", "J2"):
                dataset_payload = self._dashboard_service.load(DashboardFilters(dataset=dataset))
                if not base_payload:
                    base_payload = dataset_payload
                candidate_ranking = dataset_payload.get("charts", {}).get("candidate_ranking", [])
                rankings_by_dataset[dataset] = candidate_ranking[:5]
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

    def resolve_assistant_entity(self, message: str) -> AssistantEntity | None:
        return resolveAssistantEntity(
            message,
            settings_loader=self._settings_loader,
            connect_func=self._connect,
        )


def is_in_scope(message: str) -> bool:
    return classify_scope(message) == "allowed"


def classify_scope(message: str, entity: AssistantEntity | None = None) -> str:
    if not message.strip():
        return "blocked"
    if _WRITE_OPERATION_PATTERN.search(message):
        return "blocked"
    if _is_subjective_person_or_team_request(message, entity):
        return "subjective_blocked"
    if _BLOCKED_ANALYSIS_PATTERN.search(message):
        return "blocked"
    if entity is not None:
        return "allowed"
    if not _IN_SCOPE_PATTERN.search(message):
        return "blocked"
    return "allowed"


def classify_readme_or_app_docs_query(message: str) -> str:
    if _is_app_docs_query(message):
        return _README_OR_APP_DOCS_QUERY
    return "not_app_docs_query"


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


def _is_subjective_person_or_team_request(message: str, entity: AssistantEntity | None) -> bool:
    if entity is not None and entity.kind in {"model_candidate", "judge_model", "arbiter"}:
        return False
    if _SUBJECTIVE_BLOCK_PATTERN.search(message):
        return True
    if not _PERSON_SUBJECTIVE_PATTERN.search(message):
        return False
    if entity is not None and entity.kind == "auditor":
        return not _FACTUAL_AUDIT_PATTERN.search(message)
    return True


def resolveAssistantEntity(
    message: str,
    *,
    settings_loader: Callable[[], Any] = load_settings,
    connect_func: Callable[[str], Any] = connect,
) -> AssistantEntity | None:
    if not message.strip():
        return None
    if _JUDGE_ROLE_PATTERN.search(message):
        role = _JUDGE_ROLE_PATTERN.search(message)
        assert role is not None
        return AssistantEntity(kind="arbiter" if "rbitro" in role.group(1).lower() else "judge_model", value=role.group(1))

    known_entities = _known_assistant_entities(settings_loader=settings_loader, connect_func=connect_func)
    normalized_message = normalizeSearchText(message)
    message_tokens = _relevant_entity_tokens(normalized_message)
    preferred_kinds = _preferred_entity_kinds(message)
    matches: list[tuple[int, int, str, str]] = []
    for value, kind in known_entities.items():
        if not value:
            continue
        value_tokens = _relevant_entity_tokens(value)
        token_overlap = message_tokens & value_tokens
        if value in normalized_message or normalized_message in value:
            score = 100 + len(value_tokens)
        elif token_overlap:
            score = 60 + (10 * len(token_overlap)) + len(value_tokens)
        else:
            continue
        if kind in preferred_kinds:
            score += 25
        matches.append((score, len(value), value, kind))
    if matches:
        _, _, value, kind = max(matches)
        return AssistantEntity(kind=kind, value=value)
    if _ASSISTANT_ENTITY_TERMS_PATTERN.search(message) and not _SUBJECTIVE_BLOCK_PATTERN.search(message):
        return AssistantEntity(kind="assistant_entity_term", value=_ASSISTANT_ENTITY_TERMS_PATTERN.search(message).group(1))
    return None


def _known_assistant_entities(
    *,
    settings_loader: Callable[[], Any],
    connect_func: Callable[[str], Any],
) -> dict[str, str]:
    entities: dict[str, str] = {}
    settings = settings_loader()
    for value in (
        getattr(settings, "remote_judge_default_model", None),
        getattr(settings, "remote_secondary_judge_model", None),
        getattr(settings, "remote_arbiter_judge_model", None),
    ):
        if value:
            entities[_normalize_entity_text(str(value))] = "judge_model"
    for alias, provider_model in JUDGE_MODEL_ALIASES.items():
        entities[_normalize_entity_text(alias)] = "judge_model"
        entities[_normalize_entity_text(provider_model)] = "judge_model"

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
                if model_type == "auditor":
                    kind = "auditor"
                elif model_type == "juiz":
                    kind = "judge_model"
                elif model_type == "ambos":
                    kind = "judge_model"
                else:
                    kind = "model_candidate"
                for value in (name, version):
                    if value:
                        for variant in _entity_name_variants(str(value)):
                            entities[variant] = kind
    except Exception as error:
        if isinstance(error, AssertionError):
            raise
    finally:
        connection.close()
    return entities


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
    if tokens:
        variants.add(tokens[0])
    for index, token in enumerate(tokens):
        if token in _GENERIC_ENTITY_TOKENS:
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


def _validate_output(answer: str) -> dict[str, Any] | None:
    if _BLOCKED_ANALYSIS_PATTERN.search(answer) or _SUBJECTIVE_BLOCK_PATTERN.search(answer):
        return _blocked_response()
    return None


def _is_app_docs_query(message: str) -> bool:
    if not message.strip():
        return False
    if _WRITE_OPERATION_PATTERN.search(message):
        return False
    return bool(_APP_DOCS_QUERY_PATTERN.search(message))


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


def _build_context_planning_prompt(question: str) -> str:
    return (
        "Você é o planejador read-only do assistente AV2.\n"
        "Sua tarefa é decidir se a pergunta já pode ser respondida com segurança ou quais ferramentas de contexto "
        "nomeadas são necessárias para respondê-la.\n"
        "Você não pode pedir SQL livre, internet, arquivos externos ou operações de escrita. "
        "Use somente os nomes de ferramentas listados.\n"
        "Se a pergunta exigir dados carregados, métricas, rankings, auditorias, banco, dashboard ou README, "
        "não invente: marque sufficient=false e peça as ferramentas necessárias. "
        "Só marque sufficient=true quando a pergunta puder ser respondida apenas com instruções estáticas deste prompt.\n"
        f"Limite total do fluxo: {_MAX_CONTEXT_INTERACTIONS} chamadas à LLM, incluindo esta.\n\n"
        "Ferramentas disponíveis:\n"
        f"{json.dumps(_CONTEXT_TOOL_CATALOG, ensure_ascii=False)}\n\n"
        "Responda apenas JSON em um destes formatos:\n"
        '{"sufficient": false, "required_context": ["dashboard_summary"], "answer": ""}\n'
        '{"sufficient": true, "required_context": [], "answer": "resposta final"}\n\n'
        f"Pergunta do usuário:\n{question}"
    )


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
