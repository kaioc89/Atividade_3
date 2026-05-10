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
    r"auditoria|auditorias|audit|banco|database|dados|tabela|tabelas|readme|documentado|documentacao|"
    r"documentação|uso|usar|filtro|filtros|tela|aplicacao|aplicação|app|modelo|modelos|candidato|"
    r"candidatos|juiz|juizes|juízes|arbitro|árbitro|avaliador|avaliadores"
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
    r"meta-análise|meta-analise|avaliador|avaliadores|auditor|auditores|revisão humana|revisao humana|"
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
_ASSISTANT_ENTITY_TERMS_PATTERN = re.compile(
    r"\b("
    r"candidato|candidatos|modelo candidato|modelo candidatos|modelo|modelos|"
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

        docs_match = _lookup_app_docs_context(question, self._readme_loader())
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

        entity = self.resolve_assistant_entity(question)
        scope = classify_scope(question, entity)
        if scope != "allowed":
            return _blocked_response()

        context = {
            "dashboard": self._load_dashboard_context(),
            "auditorias": self._load_audit_context(),
            "banco": self._load_database_context(),
            "readme": self._readme_loader()[:8000],
            "uso_app": _usage_context(),
        }
        prompt = _build_prompt(question, context)
        answer = self._llm_client.complete(prompt).strip()
        answer = _sanitize_entity_answer(answer, entity).strip()
        blocked_output = _validate_output(answer)
        if blocked_output is not None:
            return blocked_output
        return {
            "answer": answer,
            "in_scope": True,
            "suggestions": DEFAULT_SUGGESTIONS,
        }

    def _load_dashboard_context(self) -> dict[str, Any]:
        try:
            payload = self._dashboard_service.load(DashboardFilters())
        except (RuntimeError, ValueError) as error:
            return {"available": False, "error": str(error)}
        return {
            "available": True,
            "cards": payload.get("cards", {}),
            "options": payload.get("options", {}),
            "methodology": payload.get("methodology", {}),
        }

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
    if entity is not None:
        return "allowed"
    if _BLOCKED_ANALYSIS_PATTERN.search(message):
        return "blocked"
        if _SUBJECTIVE_BLOCK_PATTERN.search(message):
            return "subjective_blocked"
    if not _IN_SCOPE_PATTERN.search(message):
        return "blocked"
    return "allowed"


def classify_readme_or_app_docs_query(message: str) -> str:
    if _is_app_docs_query(message):
        return _README_OR_APP_DOCS_QUERY
    return "not_app_docs_query"


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
    if {"auditoria", "auditor", "auditores", "avaliador", "avaliadores"} & set(normalized.split()):
        return {"auditor"}
    if "modelo" in normalized:
        return {"model_candidate", "judge_model"}
    return set()


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
        "Para resumo geral, exiba apenas fatos neutros e operacionais: totais carregados, datasets, modelos, "
        "avaliações, auditorias, tabelas disponíveis e consultas possíveis.\n"
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
