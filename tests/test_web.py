from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from atividade_2.database_dump import DatabaseDumpResult
from atividade_2.contracts import BatchProgress, EligibilitySummary, EvaluationProgress, PipelineSummary
from atividade_2.dashboard import DashboardFilters
from atividade_2.run_judge_service import RunJudgeResult
from atividade_2.web import create_app


class FakeRunJudgeService:
    def __init__(self, audit_path: str = "outputs/audit/test.log") -> None:
        self.requests = []
        self.audit_path = audit_path

    def describe_config(self) -> dict:
        return {
            "defaults": {
                "panel_mode": "single",
                "dataset": "J2",
                "batch_size": 1,
                "judge_execution_strategy": "sequential",
                "judge_model": "gpt-oss-120b",
                "secondary_judge_model": "llama-3.3-70b-instruct",
                "arbiter_judge_model": "m-prometheus-14b",
                "always_run_arbiter": False,
                "judge_arbitration_min_delta": 2,
                "remote_judge_timeout_seconds": 180,
                "remote_judge_temperature": 0.0,
                "remote_judge_max_tokens": 4000,
                "remote_judge_top_p": 1.0,
                "remote_judge_openai_compatible": True,
                "judge_save_raw_response": True,
            },
            "endpoints": {"JUDGE": {"host": "example.invalid", "has_api_key": True}},
            "presets": [],
            "command_preview": ".venv/bin/python -m atividade_2.cli run-judge --dry-run",
        }

    def resolve(self, request):
        return SimpleNamespace(
            audit_path=Path(self.audit_path),
            command_preview=".venv/bin/python -m atividade_2.cli run-judge --dataset J2",
        )

    def run(
        self,
        request,
        *,
        progress_callback=None,
        on_resolved=None,
        eligibility_callback=None,
        evaluation_callback=None,
        should_stop=None,
    ):
        self.requests.append(request)
        eligibility = EligibilitySummary(missing=83, failed=7, successful=240, batch_size=1, will_process=1)
        if eligibility_callback is not None and not request.dry_run:
            eligibility_callback(eligibility)
        if evaluation_callback is not None and not request.dry_run:
            base_event = {
                "dataset": "J2",
                "question_id": 10,
                "answer_id": 20,
                "candidate_model": "modelo-candidato",
                "judge_model": "openai/gpt-oss-120b",
                "role": "principal",
                "panel_mode": "single",
                "trigger_reason": "single:single_mode",
            }
            evaluation_callback(
                EvaluationProgress(
                    status="running",
                    **base_event,
                    prompt="prompt usado",
                )
            )
            evaluation_callback(
                EvaluationProgress(
                    status="success",
                    **base_event,
                    score=5,
                    arbiter_triggered=None,
                    latency_ms=123,
                    prompt="prompt usado",
                    raw_response='{"score":5}',
                    rationale="justificativa curta",
                )
            )
        if progress_callback is not None:
            progress_callback(
                BatchProgress(
                    current=1,
                    total=1,
                    percent=100,
                    executed_evaluations=1,
                    skipped_evaluations=0,
                    arbiter_evaluations=0,
                )
            )
        return RunJudgeResult(
            dry_run=request.dry_run,
            audit_log=self.audit_path,
            execution_summary="Judge mode: single",
            command_preview=".venv/bin/python -m atividade_2.cli run-judge --dataset J2",
            batch_size=1,
            eligibility=None if request.dry_run else eligibility,
            summary=None
            if request.dry_run
            else PipelineSummary(
                selected_answers=1,
                executed_evaluations=1,
                skipped_evaluations=0,
                arbiter_evaluations=0,
            ),
        )


class BlockingRunJudgeService(FakeRunJudgeService):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def run(
        self,
        request,
        *,
        progress_callback=None,
        on_resolved=None,
        eligibility_callback=None,
        evaluation_callback=None,
        should_stop=None,
    ):
        self.started.set()
        self.release.wait(timeout=2)
        return super().run(
            request,
            progress_callback=progress_callback,
            on_resolved=on_resolved,
            eligibility_callback=eligibility_callback,
            evaluation_callback=evaluation_callback,
            should_stop=should_stop,
        )


class EvaluationBeforeProgressRunJudgeService(FakeRunJudgeService):
    event_status = "success"

    def __init__(self) -> None:
        super().__init__()
        self.evaluation_reported = threading.Event()
        self.release = threading.Event()

    def run(
        self,
        request,
        *,
        progress_callback=None,
        on_resolved=None,
        eligibility_callback=None,
        evaluation_callback=None,
        should_stop=None,
    ):
        self.requests.append(request)
        eligibility = EligibilitySummary(missing=2, failed=0, successful=240, batch_size=2, will_process=2)
        if eligibility_callback is not None:
            eligibility_callback(eligibility)
        if evaluation_callback is not None:
            evaluation_callback(
                EvaluationProgress(
                    status=self.event_status,
                    dataset="J2",
                    question_id=10,
                    answer_id=20,
                    candidate_model="modelo-candidato",
                    judge_model="openai/gpt-oss-120b",
                    role="principal",
                    panel_mode="single",
                    trigger_reason="single:single_mode",
                )
            )
        self.evaluation_reported.set()
        self.release.wait(timeout=2)
        if progress_callback is not None:
            progress_callback(
                BatchProgress(
                    current=1,
                    total=2,
                    percent=50,
                    executed_evaluations=1,
                    skipped_evaluations=0,
                    arbiter_evaluations=0,
                )
            )
        return RunJudgeResult(
            dry_run=request.dry_run,
            audit_log=self.audit_path,
            execution_summary="Judge mode: single",
            command_preview=".venv/bin/python -m atividade_2.cli run-judge --dataset J2",
            batch_size=2,
            eligibility=eligibility,
            summary=PipelineSummary(
                selected_answers=2,
                executed_evaluations=1,
                skipped_evaluations=0,
                arbiter_evaluations=0,
            ),
        )


class RunningEvaluationBeforeProgressRunJudgeService(EvaluationBeforeProgressRunJudgeService):
    event_status = "running"


class FailedRunJudgeService(FakeRunJudgeService):
    def run(
        self,
        request,
        *,
        progress_callback=None,
        on_resolved=None,
        eligibility_callback=None,
        evaluation_callback=None,
        should_stop=None,
    ):
        self.requests.append(request)
        eligibility = EligibilitySummary(missing=2, failed=0, successful=240, batch_size=2, will_process=2)
        if eligibility_callback is not None:
            eligibility_callback(eligibility)
        if evaluation_callback is not None:
            evaluation_callback(
                EvaluationProgress(
                    status="success",
                    dataset="J2",
                    question_id=10,
                    answer_id=20,
                    candidate_model="modelo-candidato",
                    judge_model="openai/gpt-oss-120b",
                    role="principal",
                    panel_mode="single",
                    trigger_reason="single:single_mode",
                )
            )
        if progress_callback is not None:
            progress_callback(
                BatchProgress(
                    current=1,
                    total=2,
                    percent=50,
                    executed_evaluations=1,
                    skipped_evaluations=0,
                    arbiter_evaluations=0,
                )
            )
        raise RuntimeError("provider unavailable")


class SkippedEvaluationRunJudgeService(FakeRunJudgeService):
    def run(
        self,
        request,
        *,
        progress_callback=None,
        on_resolved=None,
        eligibility_callback=None,
        evaluation_callback=None,
        should_stop=None,
    ):
        self.requests.append(request)
        eligibility = EligibilitySummary(missing=1, failed=0, successful=240, batch_size=1, will_process=1)
        if eligibility_callback is not None and not request.dry_run:
            eligibility_callback(eligibility)
        if evaluation_callback is not None and not request.dry_run:
            skipped_event = {
                "status": "skipped",
                "dataset": "J2",
                "question_id": 10,
                "answer_id": 20,
                "candidate_model": "modelo-candidato",
                "judge_model": "openai/gpt-oss-120b",
                "role": "principal",
                "panel_mode": "2plus1",
                "score": 5,
                "trigger_reason": "2plus1:existing_evaluation",
            }
            success_event = {
                **skipped_event,
                "status": "success",
                "judge_model": "meta-llama/Llama-3.3-70B-Instruct",
                "role": "controle",
                "trigger_reason": "2plus1:primary_panel",
                "latency_ms": 123,
            }
            evaluation_callback(EvaluationProgress(**skipped_event))
            evaluation_callback(EvaluationProgress(**success_event))
        if progress_callback is not None:
            progress_callback(
                BatchProgress(
                    current=1,
                    total=1,
                    percent=100,
                    executed_evaluations=1,
                    skipped_evaluations=1,
                    arbiter_evaluations=0,
                )
            )
        return RunJudgeResult(
            dry_run=request.dry_run,
            audit_log=self.audit_path,
            execution_summary="Judge mode: 2plus1",
            command_preview=".venv/bin/python -m atividade_2.cli run-judge --dataset J2",
            batch_size=1,
            eligibility=None if request.dry_run else eligibility,
            summary=None
            if request.dry_run
            else PipelineSummary(
                selected_answers=1,
                executed_evaluations=1,
                skipped_evaluations=1,
                arbiter_evaluations=0,
            ),
        )


class OutOfOrderEvaluationRunJudgeService(FakeRunJudgeService):
    def run(
        self,
        request,
        *,
        progress_callback=None,
        on_resolved=None,
        eligibility_callback=None,
        evaluation_callback=None,
        should_stop=None,
    ):
        self.requests.append(request)
        eligibility = EligibilitySummary(missing=3, failed=0, successful=240, batch_size=3, will_process=3)
        if eligibility_callback is not None and not request.dry_run:
            eligibility_callback(eligibility)
        if evaluation_callback is not None and not request.dry_run:
            base_event = {
                "dataset": "J2",
                "candidate_model": "modelo-candidato",
                "judge_model": "openai/gpt-oss-120b",
                "role": "principal",
                "panel_mode": "single",
            }
            evaluation_callback(
                EvaluationProgress(
                    status="success",
                    question_id=30,
                    answer_id=300,
                    trigger_reason="single:success",
                    score=5,
                    **base_event,
                )
            )
            evaluation_callback(
                EvaluationProgress(
                    status="running",
                    question_id=20,
                    answer_id=200,
                    trigger_reason="single:updated",
                    **base_event,
                )
            )
            evaluation_callback(
                EvaluationProgress(
                    status="success",
                    question_id=20,
                    answer_id=200,
                    trigger_reason="single:updated",
                    score=4,
                    **base_event,
                )
            )
            evaluation_callback(
                EvaluationProgress(
                    status="failed",
                    question_id=10,
                    answer_id=100,
                    trigger_reason="single:failed",
                    error="provider timeout",
                    **base_event,
                )
            )
            evaluation_callback(
                EvaluationProgress(
                    status="running",
                    question_id=40,
                    answer_id=400,
                    trigger_reason="single:running",
                    **base_event,
                )
            )
        if progress_callback is not None:
            progress_callback(
                BatchProgress(
                    current=3,
                    total=3,
                    percent=100,
                    executed_evaluations=3,
                    skipped_evaluations=0,
                    arbiter_evaluations=0,
                )
            )
        return RunJudgeResult(
            dry_run=request.dry_run,
            audit_log=self.audit_path,
            execution_summary="Judge mode: single",
            command_preview=".venv/bin/python -m atividade_2.cli run-judge --dataset J2",
            batch_size=3,
            eligibility=None if request.dry_run else eligibility,
            summary=None
            if request.dry_run
            else PipelineSummary(
                selected_answers=3,
                executed_evaluations=3,
                skipped_evaluations=0,
                arbiter_evaluations=0,
            ),
        )


class NoProgressCallbackRunJudgeService(FakeRunJudgeService):
    def run(
        self,
        request,
        *,
        progress_callback=None,
        on_resolved=None,
        eligibility_callback=None,
        evaluation_callback=None,
        should_stop=None,
    ):
        return super().run(
            request,
            progress_callback=None,
            on_resolved=on_resolved,
            eligibility_callback=eligibility_callback,
            evaluation_callback=evaluation_callback,
            should_stop=should_stop,
        )


class FakeDashboardService:
    def __init__(self) -> None:
        self.filters: list[DashboardFilters] = []

    def load(self, filters: DashboardFilters) -> dict:
        self.filters.append(filters)
        return {
            "filters": {"dataset": filters.dataset},
            "options": {
                "candidate_models": ["modelo-candidato"],
                "judge_models": ["openai/gpt-oss-120b"],
            },
            "cards": {
                "evaluations": 4,
                "coverage": {"evaluated": 2, "expected": 3, "percent": 66.7},
                "success_rate": 100.0,
                "average_score": 4.25,
                "spearman_reference": {
                    "value": None,
                    "p_value": None,
                    "sample_size": 0,
                    "available": False,
                    "note": "J1 não possui nota humana/rubrica ordinal persistida.",
                },
                "judge_arbiter_consistency": {
                    "value": 1.0,
                    "p_value": 0.0,
                    "sample_size": 2,
                    "available": True,
                    "note": "Meta-avaliação complementar.",
                },
                "critical_failures": 1,
                "minor_disagreements": 1,
                "audit_divergences": 1,
                "judge_agreement": {
                    "total_compared": 2,
                    "delta_0": 1,
                    "delta_1": 0,
                    "delta_2": 0,
                    "delta_3": 1,
                    "delta_4": 0,
                    "arbiter_triggered": 1,
                },
            },
            "charts": {
                "candidate_ranking": [{"label": "modelo-candidato", "value": 4.25}],
                "score_distribution": [{"label": str(score), "value": 1 if score == 5 else 0} for score in range(1, 6)],
                "score_distribution_by_model": [
                    {"label": "modelo-candidato", "total": 4, "average": 4.25, "scores": {"1": 1, "2": 0, "3": 0, "4": 0, "5": 3}}
                ],
                "judge_average": [{"label": "openai/gpt-oss-120b", "value": 4.25}],
                "reference_alignment": {
                    "x_label": "nota humana / score derivado do gabarito",
                    "y_label": "nota do juiz",
                    "points": [
                        {
                            "evaluation_id": 1,
                            "answer_id": 10,
                            "question_id": 20,
                            "dataset": "J2",
                            "candidate_model": "modelo-candidato",
                            "judge_model": "openai/gpt-oss-120b",
                            "reference_score": 5,
                            "judge_score": 5,
                        }
                    ],
                },
                "ordinal_confusion": {
                    "rows": ["Humano 1", "Humano 2", "Humano 3", "Humano 4", "Humano 5"],
                    "columns": ["Juiz 1", "Juiz 2", "Juiz 3", "Juiz 4", "Juiz 5"],
                    "matrix": [
                        [0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1],
                    ],
                    "total": 2,
                    "highlights": [
                        {
                            "label": "Humano baixo, juiz alto",
                            "interpretation": "falso positivo grave",
                            "count": 1,
                            "share": 50.0,
                        }
                    ],
                    "important_cases": [],
                },
                "divergences": [{"label": "modelo-candidato", "value": 1}],
                "critical_cases": [{"label": "nota 1", "value": 1}],
                "critical_error_categories": [
                    {"label": "Nota alta para resposta errada", "value": 1},
                    {"label": "Nota baixa para resposta correta", "value": 0},
                    {"label": "Alucinacao normativa", "value": 1},
                    {"label": "Resposta sem fundamentacao", "value": 0},
                    {"label": "Divergencia entre juizes", "value": 1},
                    {"label": "Erro de parsing", "value": 0},
                    {"label": "Timeout/HTTP error", "value": 0},
                ],
                "rubric_heatmap": {
                    "columns": ["Argumentação", "Precisão", "Coesão legal", "Total"],
                    "rows": [{"label": "modelo-candidato", "values": [4.4, 4.1, 4.2, 4.25], "count": 4}],
                },
                "legal_specialty_performance": {
                    "columns": ["modelo-candidato"],
                    "rows": [{"label": "Direito Administrativo", "values": [4.25], "count": 4, "average": 4.25}],
                },
                "difficulty_performance": {
                    "x_label": "dificuldade",
                    "y_label": "média da nota",
                    "difficulties": ["Fácil", "Médio", "Difícil", "Muito difícil"],
                    "series": [{"label": "modelo-candidato", "values": [5.0, 4.0, 3.0, 2.0]}],
                },
            },
            "tables": {
                "critical_cases": [
                    {
                        "reason": "nota 1",
                        "dataset": "J1",
                        "answer_id": 10,
                        "question_id": 20,
                        "candidate_model": "modelo-candidato",
                        "judge_model": "openai/gpt-oss-120b",
                        "role": "principal",
                        "score": 1,
                        "status": "success",
                    }
                ],
                "divergence_cases": [],
                "judge_agreement_arbitrations": [
                    {
                        "answer_id": 10,
                        "question_id": 20,
                        "candidate_model": "modelo-candidato",
                        "judge_1_score": 5,
                        "judge_2_score": 2,
                        "delta": 3,
                        "arbiter_score": 4,
                        "arbitration_reason": "primary_panel",
                    }
                ],
                "critical_error_analysis": [
                    {
                        "question_id": 20,
                        "candidate_model": "modelo-candidato",
                        "judge_model": "openai/gpt-oss-120b",
                        "score": 5,
                        "error_type": "Nota alta para resposta errada",
                        "short_justification": "referencia 1, juiz 5",
                        "log_url": "/api/run-history/test/audit-log",
                    }
                ],
            },
            "methodology": {"primary_spearman": "metodologia principal", "judge_arbiter": "consistencia"},
        }


class FakeDumpService:
    def __init__(self) -> None:
        self.calls = 0

    def create_dump(self) -> DatabaseDumpResult:
        self.calls += 1
        return DatabaseDumpResult(
            filename="atividade_2_20260430_120000.sql",
            path="outputs/backup/atividade_2_20260430_120000.sql",
            size_bytes=2048,
            created_at="2026-04-30T12:00:00",
            download_url="/api/database-dumps/atividade_2_20260430_120000.sql",
        )


class FakeDatabaseResetService:
    def __init__(self) -> None:
        self.calls = 0
        self.restored_paths = []

    def reset_to_initial_state(self) -> dict:
        self.calls += 1
        return {"status": "ok", "message": "Database restored to initial state."}

    def restore_backup(self, backup_file) -> dict:
        self.restored_paths.append(backup_file)
        return {
            "status": "ok",
            "message": "Backup restored.",
            "filename": backup_file.name,
            "path": str(backup_file),
        }


class FakeJudgePromptConfigService:
    def __init__(self) -> None:
        self.saved = []

    def options(self) -> dict:
        return {"datasets": [{"value": "J1", "label": "J1"}]}

    def get(self, *, dataset: str) -> dict:
        return {
            "record": {
                "prompt_id": 12,
                "dataset": dataset,
                "version": 3,
                "created_by": "Diego",
                "active": True,
                "prompt": "[PERSONA]\n\n[CONTEXTO]\n\n[RUBRICA]\n\n[SAIDA]",
                "persona": "persona",
                "context": "contexto",
                "rubric": "rubrica",
                "output": "saida",
                "created_at": "2026-05-02T10:00:00",
            },
            "versions": [
                {
                    "prompt_id": 12,
                    "version": 3,
                    "created_by": "Diego",
                    "created_at": "2026-05-02T10:00:00",
                    "active": True,
                    "prompt_chars": 44,
                    "persona_chars": 7,
                    "context_chars": 8,
                    "rubric_chars": 8,
                    "output_chars": 5,
                }
            ],
            "preview": {
                "dataset": dataset,
                "question_id": 71,
                "answer_id": 1,
                "candidate_model": "modelo-candidato",
                "rendered_prompt": "prompt montado",
                "version": 3,
            },
        }

    def save(
        self,
        *,
        dataset: str,
        prompt: str,
        persona: str,
        context: str,
        rubric: str,
        output: str,
        changed_by: str,
    ) -> dict:
        self.saved.append((dataset, prompt, persona, context, rubric, output, changed_by))
        return {
            "record": {
                "prompt_id": 13,
                "dataset": dataset,
                "version": 4,
                "created_by": changed_by,
                "active": True,
                "prompt": prompt,
                "persona": persona,
                "context": context,
                "rubric": rubric,
                "output": output,
                "created_at": "2026-05-02T10:05:00",
            },
            "versions": [],
            "preview": {
                "dataset": dataset,
                "question_id": 71,
                "answer_id": 1,
                "candidate_model": "modelo-candidato",
                "rendered_prompt": "prompt montado",
                "version": 4,
            },
        }


class FakeMetaEvaluationService:
    def __init__(self) -> None:
        self.saved = []
        self.deleted = []

    def options(self) -> dict:
        return {
            "evaluations": [
                {
                    "value": "101",
                    "label": "[pendente] Aval. 101 | Q71 | candidato x juiz | nota 4",
                    "meta_completed": False,
                    "meta_count": 0,
                }
            ]
        }

    def get(self, *, evaluation_id: int) -> dict:
        return {
            "subject": {
                "evaluation_id": evaluation_id,
                "dataset": "J1",
                "question_id": 71,
                "answer_id": 1,
                "candidate_model": "modelo-candidato",
                "judge_model": "modelo-juiz",
                "judge_score": 4,
                "judge_rationale": "Justificativa do juiz",
                "judge_chain_of_thought": "Chain of thoughts do juiz",
                "question_text": "Enunciado da questao",
                "reference_answer": "Gabarito oficial",
                "candidate_answer": "Resposta do candidato",
                "evaluated_at": "2026-05-04T10:00:00",
                "prompt_version": 6,
                "prompt_created_by": "Diego",
            },
            "records": [
                {
                    "meta_evaluation_id": 1,
                    "evaluation_id": evaluation_id,
                    "evaluator_name": "Ana",
                    "score": 5,
                    "rationale": "O juiz foi consistente.",
                    "created_at": "2026-05-04T11:00:00",
                }
            ],
        }

    def save(self, *, meta_evaluation_id: int | None, evaluation_id: int, evaluator_name: str, score: int, rationale: str) -> dict:
        self.saved.append((meta_evaluation_id, evaluation_id, evaluator_name, score, rationale))
        return {
            "action": "updated" if meta_evaluation_id else "created",
            "record": {
                "meta_evaluation_id": meta_evaluation_id or 2,
                "evaluation_id": evaluation_id,
                "evaluator_name": evaluator_name,
                "score": score,
                "rationale": rationale,
                "created_at": "2026-05-04T12:00:00",
            },
            "subject": self.get(evaluation_id=evaluation_id)["subject"],
            "records": self.get(evaluation_id=evaluation_id)["records"]
            + [
                {
                    "meta_evaluation_id": meta_evaluation_id or 2,
                    "evaluation_id": evaluation_id,
                    "evaluator_name": evaluator_name,
                    "score": score,
                    "rationale": rationale,
                    "created_at": "2026-05-04T12:00:00",
                }
            ],
        }

    def delete(self, *, meta_evaluation_id: int, evaluation_id: int) -> dict:
        self.deleted.append((meta_evaluation_id, evaluation_id))
        return {
            "action": "deleted",
            "subject": self.get(evaluation_id=evaluation_id)["subject"],
            "records": [],
        }


def test_web_index_contains_progress_element() -> None:
    client = TestClient(create_app(FakeRunJudgeService()))

    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert 'data-tab="dashboard-panel">Dashboard</button>' in response.text
    assert '<main id="dashboard-panel" class="dashboard-layout tab-panel">' in response.text
    assert '<main id="execution-panel" class="tab-panel" hidden>' in response.text
    assert "Resultados e Auditoria da Avaliacao" in response.text
    assert 'id="database-actions-toggle" class="database-actions-toggle"' in response.text
    assert "Acoes do Banco" in response.text
    assert 'id="database-clean" class="danger" type="button" role="menuitem">Clean DB (Initial State)</button>' in response.text
    assert 'id="database-restore" type="button" role="menuitem">Restaurar Backup</button>' in response.text
    assert 'id="database-restore-file" type="file" accept=".sql,application/sql,text/plain" hidden>' in response.text
    assert 'id="database-dump" type="button" role="menuitem">Exportar Dump do Banco</button>' in response.text
    assert 'id="database-clean-dialog" class="confirm-dialog"' in response.text
    assert "gera um backup automaticamente antes de limpar o schema public" in response.text
    assert 'id="database-clean-confirm" class="danger-button" type="button">Continuar</button>' in response.text
    assert "database-clean-backup-confirm" not in response.text
    assert "Fazer backup e limpar" not in response.text
    assert 'id="database-dump-dialog" class="confirm-dialog"' in response.text
    assert 'id="database-dump-status" class="status"></span>' in response.text
    assert "Dump completo em outputs/backup." not in response.text
    assert "confirm(" not in response.text
    assert '<progress id="batch-progress"' in response.text
    assert 'setText("selected", summary?.selected_answers ?? eligibility?.will_process ?? data.progress?.total);' in response.text
    assert 'id="eligible-missing"' in response.text
    assert 'id="execution-table-body"' in response.text
    assert "execution-row-enter" in response.text
    assert "execution-row-success" in response.text
    assert "execution-row-failed" in response.text
    assert "function animateExecutionRow" in response.text
    assert 'row.dataset.eventKey = eventKey;' in response.text
    assert 'if (!statusChanged || !previousRect || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;' in response.text
    assert 'prefers-reduced-motion: reduce' in response.text
    assert "/api/runs/" in response.text
    assert "Execucoes anteriores" in response.text
    assert 'id="history-table-body"' in response.text
    assert 'id="history-log-content"' in response.text
    assert 'id="post-run-panel" class="post-run-panel" hidden' in response.text
    assert 'id="post-run-cards"' in response.text
    assert 'id="score-distribution-chart"' in response.text
    assert 'id="judge-failures-chart"' in response.text
    assert 'id="candidate-average-chart"' in response.text
    assert 'id="judge-average-chart"' in response.text
    assert 'id="dashboard-model-distribution-carousel"' in response.text
    assert 'id="dashboard-model-distribution-chart"' in response.text
    assert "Indicadores gerais" in response.text
    assert response.text.index("Casos criticos e divergencias") < response.text.index('<h3>Distribuicao das notas por modelo</h3>')
    assert 'id="dashboard-cases-body"' in response.text
    assert 'data-carousel-index="0"' in response.text
    assert 'data-carousel-index="1"' in response.text
    assert 'data-carousel-index="2"' in response.text
    assert 'data-carousel-index="3"' in response.text
    assert 'data-carousel-index="4"' in response.text
    assert 'data-carousel-index="5"' in response.text
    assert 'data-carousel-index="6"' in response.text
    assert 'data-carousel-index="7"' in response.text
    assert "Concordancia entre Juizes" in response.text
    assert 'id="dashboard-judge-agreement-cards"' in response.text
    assert 'id="dashboard-judge-agreement-body"' in response.text
    assert "Correlacao juiz x referencia humana/gabarito" in response.text
    assert 'id="dashboard-reference-scatter"' in response.text
    assert "Matriz de concordancia / divergencia" in response.text
    assert 'id="dashboard-ordinal-confusion"' in response.text
    assert "Heatmap modelo x dimensao da rubrica" in response.text
    assert 'id="dashboard-rubric-heatmap"' in response.text
    assert "Desempenho por especialidade juridica" in response.text
    assert 'id="dashboard-legal-specialty-performance"' in response.text
    assert "Analise de erros criticos" in response.text
    assert "Categorias de erro" in response.text
    assert 'id="dashboard-critical-error-chart"' in response.text
    assert 'id="dashboard-critical-error-body"' in response.text
    assert "Link para log" in response.text
    assert "function renderModelDistributionChart" in response.text
    assert "function renderReferenceScatter" in response.text
    assert "function renderOrdinalConfusion" in response.text
    assert "rho Spearman" in response.text
    assert "p-value" in response.text
    assert "function renderRubricHeatmap" in response.text
    assert "function renderLegalSpecialtyPerformance" in response.text
    assert "function renderCriticalErrorAnalysis" in response.text
    assert "function moveCarousel" in response.text
    assert "function goToCarouselPage" in response.text
    assert "function scrollCarouselTabsIntoView" in response.text
    assert ".carousel-tabs { flex:1 1 auto;" in response.text
    assert ".carousel-controls { flex:0 0 auto;" in response.text
    assert "if (index <= 1)" in response.text
    assert "Math.max(0, index - 1)" in response.text
    assert "Math.min(tabs.length - 1, index + 1)" in response.text
    assert "const maxScrollLeft = Math.max(0, root.scrollWidth - root.clientWidth)" in response.text
    assert "scroll-padding-left:8px" in response.text
    assert 'activeTab.scrollIntoView({behavior: "smooth", block: "nearest", inline: "start"})' in response.text
    assert "Math.min(maxScrollLeft, Math.max(0, desiredLeft))" in response.text
    assert "function resetCarouselTabsScroll" in response.text
    assert "requestAnimationFrame(resetCarouselTabsScroll)" in response.text
    assert "(dashboardCarouselIndex + delta + cards.length) % cards.length" in response.text
    assert "track.style.transform" in response.text
    assert "score_distribution_by_model" in response.text
    assert "reference_alignment" in response.text
    assert "ordinal_confusion" in response.text
    assert "rubric_heatmap" in response.text
    assert "legal_specialty_performance" in response.text
    assert "difficulty_performance" in response.text
    assert "Desempenho por dificuldade" in response.text
    assert "critical_error_categories" in response.text
    assert "critical_error_analysis" in response.text
    assert "function buildPostRunStats" in response.text
    assert "function renderPostRunPanel" in response.text
    assert 'fetch(`/api/runs/${runId}`, {cache: "no-store"})' in response.text
    assert "event.error ? friendlyErrorMessage(event.error) : null" in response.text
    assert "showPercent: true" in response.text
    assert 'className = "bar-percent"' in response.text
    assert 'className = `bar-count ${valueTone(value, row, options)}`' in response.text
    assert "function applyBarTone" in response.text
    assert "function valueTone" in response.text


def test_dashboard_tab_selection_always_refreshes_dashboard_data() -> None:
    client = TestClient(create_app(FakeRunJudgeService()))

    response = client.get("/")

    assert response.status_code == 200
    assert 'if (targetId === "dashboard-panel") loadDashboard();' in response.text
    assert 'if (targetId === "dashboard-panel" && !dashboardLoaded) loadDashboard();' not in response.text


def test_web_index_contains_prompt_judges_tab() -> None:
    client = TestClient(create_app(FakeRunJudgeService(), judge_prompt_service=FakeJudgePromptConfigService()))

    response = client.get("/")

    assert response.status_code == 200
    assert 'data-tab="prompt-panel">Prompt Juizes</button>' in response.text
    assert 'id="prompt_dataset"' in response.text
    assert 'id="prompt_save"' in response.text
    assert 'id="prompt_logs_body"' in response.text
    assert 'id="prompt_preview_content"' in response.text


def test_judge_prompt_endpoints_return_options_and_allow_save() -> None:
    prompt_service = FakeJudgePromptConfigService()
    client = TestClient(create_app(FakeRunJudgeService(), judge_prompt_service=prompt_service))
    token = client.get("/api/config").json()["csrf_token"]

    options = client.get("/api/judge-prompts/options")
    assert options.status_code == 200
    assert options.json()["datasets"][0]["value"] == "J1"

    current = client.get("/api/judge-prompts", params={"dataset": "J1"})
    assert current.status_code == 200
    assert current.json()["record"]["persona"] == "persona"

    saved = client.put(
        "/api/judge-prompts",
        headers={"x-csrf-token": token},
        json={
            "dataset": "J1",
            "prompt": "[PERSONA]",
            "persona": "nova persona",
            "context": "novo contexto",
            "rubric": "nova rubrica",
            "output": "nova saida",
            "changed_by": "Diego",
        },
    )
    assert saved.status_code == 200
    assert prompt_service.saved[-1] == (
        "J1",
        "[PERSONA]",
        "nova persona",
        "novo contexto",
        "nova rubrica",
        "nova saida",
        "Diego",
    )


def test_web_index_contains_meta_evaluation_tab() -> None:
    client = TestClient(create_app(FakeRunJudgeService(), meta_evaluation_service=FakeMetaEvaluationService()))

    response = client.get("/")

    assert response.status_code == 200
    assert 'data-tab="meta-panel">Meta-Avaliacao</button>' in response.text
    assert 'id="meta_evaluation_select"' in response.text
    assert 'id="meta_save"' in response.text
    assert 'id="meta_cancel_edit"' in response.text
    assert 'id="meta_subject_chain_of_thought"' in response.text
    assert 'id="meta_records_body"' in response.text
    assert response.text.index("Meta-avaliacoes registradas") < response.text.index("Avaliacao selecionada")


def test_meta_evaluation_endpoints_return_options_and_allow_save() -> None:
    meta_service = FakeMetaEvaluationService()
    client = TestClient(create_app(FakeRunJudgeService(), meta_evaluation_service=meta_service))
    token = client.get("/api/config").json()["csrf_token"]

    options = client.get("/api/meta-evaluations/options")
    assert options.status_code == 200
    assert options.json()["evaluations"][0]["value"] == "101"
    assert options.json()["evaluations"][0]["label"].startswith("[pendente] ")

    current = client.get("/api/meta-evaluations", params={"evaluation_id": 101})
    assert current.status_code == 200
    assert current.json()["subject"]["dataset"] == "J1"

    saved = client.put(
        "/api/meta-evaluations",
        headers={"x-csrf-token": token},
        json={
            "meta_evaluation_id": 1,
            "evaluation_id": 101,
            "evaluator_name": "Diego",
            "score": 4,
            "rationale": "O juiz foi justo na avaliacao.",
        },
    )
    assert saved.status_code == 200
    assert meta_service.saved[-1] == (1, 101, "Diego", 4, "O juiz foi justo na avaliacao.")

    deleted = client.request(
        "DELETE",
        "/api/meta-evaluations/1?evaluation_id=101",
        headers={"x-csrf-token": token},
    )
    assert deleted.status_code == 200
    assert meta_service.deleted[-1] == (1, 101)


def test_tab_navigation_does_not_cancel_active_run_or_stop_polling() -> None:
    client = TestClient(create_app(FakeRunJudgeService()))

    response = client.get("/")

    assert response.status_code == 200
    switch_tab_source = response.text.split("function switchTab(targetId) {", maxsplit=1)[1].split(
        "async function loadConfig()",
        maxsplit=1,
    )[0]
    assert "/cancel" not in switch_tab_source
    assert "clearInterval(pollTimer)" not in switch_tab_source
    assert "activeRunId = null" not in switch_tab_source


def test_web_index_contains_endpoint_and_advanced_controls() -> None:
    client = TestClient(create_app(FakeRunJudgeService()))

    response = client.get("/")

    assert response.status_code == 200
    assert 'id="remote_judge_base_url"' in response.text
    assert 'id="remote_judge_api_key" type="password"' in response.text
    assert 'data-toggle-secret="remote_judge_api_key"' in response.text
    assert 'data-toggle-secret="remote_secondary_judge_api_key"' in response.text
    assert 'data-toggle-secret="remote_arbiter_judge_api_key"' in response.text
    assert 'id="endpoint_source_judge"' in response.text
    assert 'id="endpoint_source_secondary"' in response.text
    assert 'id="endpoint_source_arbiter"' in response.text
    assert 'remote_secondary_judge_base_url: secondaryEndpointSource === "custom"' in response.text
    assert 'remote_arbiter_judge_api_key: arbiterEndpointSource === "custom"' in response.text
    assert "function applyEndpointSources" not in response.text
    assert 'id="endpoint_fields_judge" class="endpoint-fields" hidden' in response.text
    assert 'id="endpoint_fields_secondary" class="endpoint-fields" hidden' in response.text
    assert 'id="endpoint_fields_arbiter" class="endpoint-fields" hidden' in response.text
    assert 'id="remote_judge_timeout_seconds"' in response.text
    assert 'id="remote_judge_openai_compatible"' in response.text
    assert "<summary>Campos avancados</summary>" in response.text
    assert 'id="always_run_arbiter"' in response.text
    assert response.text.index('id="always_run_arbiter"') < response.text.index("<details>")
    assert response.text.index('id="judge_save_raw_response"') > response.text.index("<details>")
    assert 'id="dry-run" disabled' in response.text
    assert 'id="run" disabled' in response.text
    assert 'id="stop-run" type="button" disabled' in response.text
    assert 'id="run-status-icon"' in response.text


def test_config_endpoint_is_secret_safe_and_returns_csrf_token() -> None:
    client = TestClient(create_app(FakeRunJudgeService()))

    response = client.get("/api/config")

    assert response.status_code == 200
    data = response.json()
    assert data["csrf_token"]
    assert data["endpoints"]["JUDGE"]["host"] == "example.invalid"
    assert "secret" not in response.text.lower()


def test_dashboard_endpoint_returns_filtered_audit_payload() -> None:
    dashboard = FakeDashboardService()
    client = TestClient(create_app(FakeRunJudgeService(), dashboard_service=dashboard))

    response = client.get("/api/dashboard?dataset=J2&candidate_model=modelo-candidato&status=sucesso")

    assert response.status_code == 200
    data = response.json()
    assert data["cards"]["evaluations"] == 4
    assert data["cards"]["spearman_reference"]["available"] is False
    assert dashboard.filters[0].dataset == "J2"
    assert dashboard.filters[0].candidate_models == ("modelo-candidato",)
    assert dashboard.filters[0].status == "sucesso"


def test_database_dump_endpoint_requires_csrf_token() -> None:
    dump_service = FakeDumpService()
    client = TestClient(create_app(FakeRunJudgeService(), dump_service=dump_service))

    response = client.post("/api/database-dumps", json={})

    assert response.status_code == 403
    assert dump_service.calls == 0


def test_database_dump_endpoint_returns_download_metadata() -> None:
    dump_service = FakeDumpService()
    client = TestClient(create_app(FakeRunJudgeService(), dump_service=dump_service))
    token = client.get("/api/config").json()["csrf_token"]

    response = client.post("/api/database-dumps", headers={"x-csrf-token": token}, json={})

    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "atividade_2_20260430_120000.sql"
    assert data["download_url"] == "/api/database-dumps/atividade_2_20260430_120000.sql"
    assert dump_service.calls == 1


def test_database_reset_endpoint_requires_csrf_token() -> None:
    reset_service = FakeDatabaseResetService()
    client = TestClient(create_app(FakeRunJudgeService(), database_reset_service=reset_service))

    response = client.post("/api/database-reset", json={})

    assert response.status_code == 403
    assert reset_service.calls == 0


def test_database_reset_endpoint_restores_initial_state() -> None:
    reset_service = FakeDatabaseResetService()
    client = TestClient(create_app(FakeRunJudgeService(), database_reset_service=reset_service))
    token = client.get("/api/config").json()["csrf_token"]

    response = client.post("/api/database-reset", headers={"x-csrf-token": token}, json={})

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": "Database restored to initial state."}
    assert reset_service.calls == 1


def test_database_restore_endpoint_requires_csrf_token(tmp_path) -> None:
    reset_service = FakeDatabaseResetService()
    client = TestClient(create_app(FakeRunJudgeService(), backup_dir=tmp_path, database_reset_service=reset_service))

    response = client.post(
        "/api/database-restore",
        headers={"x-backup-filename": "atividade_2_20260430_120000.sql"},
        content=b"SELECT 1;",
    )

    assert response.status_code == 403
    assert reset_service.restored_paths == []


def test_database_restore_endpoint_restores_uploaded_sql(tmp_path) -> None:
    reset_service = FakeDatabaseResetService()
    client = TestClient(create_app(FakeRunJudgeService(), backup_dir=tmp_path, database_reset_service=reset_service))
    token = client.get("/api/config").json()["csrf_token"]

    response = client.post(
        "/api/database-restore",
        headers={"x-csrf-token": token, "x-backup-filename": "atividade_2_20260430_120000.sql"},
        content=b"SELECT 1;",
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["filename"].endswith("_atividade_2_20260430_120000.sql")
    assert len(reset_service.restored_paths) == 1
    assert not reset_service.restored_paths[0].exists()


def test_database_dump_download_rejects_path_traversal(tmp_path) -> None:
    client = TestClient(create_app(FakeRunJudgeService(), backup_dir=tmp_path))

    response = client.get("/api/database-dumps/../secret.sql")

    assert response.status_code in {400, 404}


def test_mutating_endpoint_requires_csrf_token() -> None:
    client = TestClient(create_app(FakeRunJudgeService()))

    response = client.post("/api/runs/dry-run", json={"panel_mode": "single"})

    assert response.status_code == 403


def test_config_exposes_unique_judge_model_options() -> None:
    client = TestClient(create_app(FakeRunJudgeService()))

    response = client.get("/api/config")

    assert response.status_code == 200
    assert response.json()["judge_model_options"] == [
        "gpt-oss-120b",
        "llama-3.3-70b-instruct",
        "m-prometheus-14b",
    ]


def test_execution_form_uses_model_selects_and_mode_blocks() -> None:
    client = TestClient(create_app(FakeRunJudgeService()))

    response = client.get("/")

    assert response.status_code == 200
    assert '<select id="judge_model"></select>' in response.text
    assert '<select id="secondary_judge_model"></select>' in response.text
    assert '<select id="arbiter_judge_model"></select>' in response.text
    assert 'id="secondary_block" class="judge-block" data-judge-block="secondary"' in response.text
    assert 'id="arbiter_block" class="judge-block" data-judge-block="arbiter"' in response.text
    assert "function renderJudgeBlocks()" in response.text


def test_dry_run_returns_secret_safe_preview() -> None:
    client = TestClient(create_app(FakeRunJudgeService()))
    token = client.get("/api/config").json()["csrf_token"]

    response = client.post(
        "/api/runs/dry-run",
        headers={"x-csrf-token": token},
        json={"panel_mode": "single", "dataset": "J2", "batch_size": 1},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["dry_run"] is True
    assert data["summary"] is None
    assert "Judge mode: single" in data["execution_summary"]
    assert "secret" not in response.text.lower()


def test_dry_run_accepts_endpoint_and_advanced_overrides() -> None:
    service = FakeRunJudgeService()
    client = TestClient(create_app(service))
    token = client.get("/api/config").json()["csrf_token"]

    response = client.post(
        "/api/runs/dry-run",
        headers={"x-csrf-token": token},
        json={
            "panel_mode": "2plus1",
            "dataset": "J2",
            "batch_size": 3,
            "remote_judge_base_url": "https://judge1.example.invalid/v1",
            "remote_judge_api_key": "key-1",
            "remote_secondary_judge_base_url": "https://judge2.example.invalid/v1",
            "remote_secondary_judge_api_key": "key-2",
            "remote_arbiter_judge_base_url": "https://arbiter.example.invalid/v1",
            "remote_arbiter_judge_api_key": "key-3",
            "endpoint_source_secondary": "custom",
            "endpoint_source_arbiter": "custom",
            "judge_arbitration_min_delta": 1,
            "remote_judge_timeout_seconds": 240,
            "remote_judge_temperature": 0.0,
            "remote_judge_max_tokens": 4000,
            "remote_judge_top_p": 1.0,
            "remote_judge_openai_compatible": True,
            "judge_save_raw_response": False,
        },
    )

    assert response.status_code == 200
    request = service.requests[-1]
    assert request.remote_judge_base_url == "https://judge1.example.invalid/v1"
    assert request.remote_secondary_judge_api_key == "key-2"
    assert request.remote_arbiter_judge_base_url == "https://arbiter.example.invalid/v1"
    assert request.endpoint_source_secondary == "custom"
    assert request.endpoint_source_arbiter == "custom"
    assert request.judge_arbitration_min_delta == 1
    assert request.remote_judge_timeout_seconds == 240
    assert request.judge_save_raw_response is False


def test_dry_run_accepts_endpoint_copy_sources_without_target_secrets() -> None:
    service = FakeRunJudgeService()
    client = TestClient(create_app(service))
    token = client.get("/api/config").json()["csrf_token"]

    response = client.post(
        "/api/runs/dry-run",
        headers={"x-csrf-token": token},
        json={
            "panel_mode": "2plus1",
            "dataset": "J2",
            "batch_size": 1,
            "remote_judge_base_url": "https://judge1.example.invalid/v1",
            "remote_judge_api_key": "key-1",
            "remote_secondary_judge_base_url": "",
            "remote_secondary_judge_api_key": "",
            "remote_arbiter_judge_base_url": "",
            "remote_arbiter_judge_api_key": "",
            "endpoint_source_judge": "custom",
            "endpoint_source_secondary": "judge",
            "endpoint_source_arbiter": "secondary",
        },
    )

    assert response.status_code == 200
    request = service.requests[-1]
    assert request.remote_judge_base_url == "https://judge1.example.invalid/v1"
    assert request.remote_secondary_judge_base_url is None
    assert request.remote_arbiter_judge_api_key is None
    assert request.endpoint_source_judge == "custom"
    assert request.endpoint_source_secondary == "judge"
    assert request.endpoint_source_arbiter == "secondary"


def test_run_lifecycle_exposes_batch_progress() -> None:
    client = TestClient(create_app(FakeRunJudgeService()))
    token = client.get("/api/config").json()["csrf_token"]

    created = client.post(
        "/api/runs",
        headers={"x-csrf-token": token},
        json={"panel_mode": "single", "dataset": "J2", "batch_size": 1},
    )

    assert created.status_code == 200
    run_id = created.json()["run_id"]
    current = client.get(f"/api/runs/{run_id}")

    assert current.status_code == 200
    data = current.json()
    assert data["progress"]["percent"] == 100
    assert data["eligibility"]["missing"] == 83
    assert data["eligibility"]["failed"] == 7
    assert data["eligibility"]["successful"] == 240
    assert data["eligibility"]["will_process"] == 1
    assert data["result"]["summary"]["executed_evaluations"] == 1
    assert len(data["evaluation_events"]) == 1
    assert data["evaluation_events"][0]["status"] == "success"
    assert data["evaluation_events"][0]["question_id"] == 10
    assert data["evaluation_events"][0]["candidate_model"] == "modelo-candidato"
    assert data["evaluation_events"][0]["latency_ms"] == 123
    assert data["started_at"] is not None
    assert data["finished_at"] is not None
    assert data["duration"] is not None


def test_running_run_progress_updates_when_execution_table_receives_record() -> None:
    service = EvaluationBeforeProgressRunJudgeService()
    client = TestClient(create_app(service))
    token = client.get("/api/config").json()["csrf_token"]

    created = client.post(
        "/api/runs",
        headers={"x-csrf-token": token},
        json={"panel_mode": "single", "dataset": "J2", "batch_size": 2},
    )

    assert created.status_code == 200
    assert service.evaluation_reported.wait(timeout=2)
    data = client.get(f"/api/runs/{created.json()['run_id']}").json()
    service.release.set()

    assert data["status"] == "running"
    assert data["progress"]["current"] == 1
    assert data["progress"]["total"] == 2
    assert data["progress"]["percent"] == 50
    assert [event["answer_id"] for event in data["evaluation_events"]] == [20]


def test_running_run_progress_ignores_non_success_execution_table_records() -> None:
    service = RunningEvaluationBeforeProgressRunJudgeService()
    client = TestClient(create_app(service))
    token = client.get("/api/config").json()["csrf_token"]

    created = client.post(
        "/api/runs",
        headers={"x-csrf-token": token},
        json={"panel_mode": "single", "dataset": "J2", "batch_size": 2},
    )

    assert created.status_code == 200
    assert service.evaluation_reported.wait(timeout=2)
    data = client.get(f"/api/runs/{created.json()['run_id']}").json()
    service.release.set()

    assert data["status"] == "running"
    assert data["progress"]["current"] == 0
    assert data["progress"]["total"] == 2
    assert data["progress"]["percent"] == 0
    assert [event["status"] for event in data["evaluation_events"]] == ["running"]


def test_failed_run_reports_complete_process_progress() -> None:
    client = TestClient(create_app(FailedRunJudgeService()))
    token = client.get("/api/config").json()["csrf_token"]

    created = client.post(
        "/api/runs",
        headers={"x-csrf-token": token},
        json={"panel_mode": "single", "dataset": "J2", "batch_size": 2},
    )

    assert created.status_code == 200
    data = client.get(f"/api/runs/{created.json()['run_id']}").json()
    assert data["status"] == "failed"
    assert data["progress"]["current"] == 2
    assert data["progress"]["total"] == 2
    assert data["progress"]["percent"] == 100
    assert data["error"] == "provider unavailable"


def test_run_lifecycle_hides_skipped_evaluations_from_execution_table_payload() -> None:
    client = TestClient(create_app(SkippedEvaluationRunJudgeService()))
    token = client.get("/api/config").json()["csrf_token"]

    created = client.post(
        "/api/runs",
        headers={"x-csrf-token": token},
        json={"panel_mode": "2plus1", "dataset": "J2", "batch_size": 1},
    )

    assert created.status_code == 200
    data = client.get(f"/api/runs/{created.json()['run_id']}").json()
    assert data["progress"]["skipped_evaluations"] == 1
    assert [event["status"] for event in data["evaluation_events"]] == ["success"]
    assert data["evaluation_events"][0]["role"] == "controle"


def test_run_lifecycle_orders_execution_table_payload_by_status_priority() -> None:
    client = TestClient(create_app(OutOfOrderEvaluationRunJudgeService()))
    token = client.get("/api/config").json()["csrf_token"]

    created = client.post(
        "/api/runs",
        headers={"x-csrf-token": token},
        json={"panel_mode": "single", "dataset": "J2", "batch_size": 3},
    )

    assert created.status_code == 200
    data = client.get(f"/api/runs/{created.json()['run_id']}").json()
    assert [event["status"] for event in data["evaluation_events"]] == ["running", "failed", "success", "success"]
    assert [event["question_id"] for event in data["evaluation_events"]] == [40, 10, 30, 20]
    assert [event["answer_id"] for event in data["evaluation_events"]] == [400, 100, 300, 200]


def test_completed_run_progress_falls_back_to_summary_when_callback_is_missing() -> None:
    client = TestClient(create_app(NoProgressCallbackRunJudgeService()))
    token = client.get("/api/config").json()["csrf_token"]

    created = client.post(
        "/api/runs",
        headers={"x-csrf-token": token},
        json={"panel_mode": "single", "dataset": "J2", "batch_size": 1},
    )

    assert created.status_code == 200
    run_id = created.json()["run_id"]
    current = client.get(f"/api/runs/{run_id}")

    assert current.status_code == 200
    data = current.json()
    assert data["progress"] == {
        "current": 1,
        "total": 1,
        "percent": 100,
        "executed_evaluations": 1,
        "skipped_evaluations": 0,
        "arbiter_evaluations": 0,
    }


def test_run_exposes_audit_log_link_and_file_content(tmp_path) -> None:
    audit_path = tmp_path / "judge.log"
    audit_path.write_text("audit content\n", encoding="utf-8")
    client = TestClient(create_app(FakeRunJudgeService(str(audit_path))))
    token = client.get("/api/config").json()["csrf_token"]

    created = client.post(
        "/api/runs",
        headers={"x-csrf-token": token},
        json={"panel_mode": "single", "dataset": "J2", "batch_size": 1},
    )

    data = created.json()
    assert data["audit_log"] == str(audit_path)
    assert data["audit_log_url"] == f"/api/runs/{data['run_id']}/audit-log"
    log_response = client.get(data["audit_log_url"])
    assert log_response.status_code == 200
    assert log_response.text == "audit content\n"


def test_index_includes_modal_live_audit_log_viewer() -> None:
    client = TestClient(create_app(FakeRunJudgeService()))

    response = client.get("/")

    assert response.status_code == 200
    assert 'id="audit-log-dialog"' in response.text
    assert 'className = "audit-log-button"' in response.text
    assert 'className = "audit-log-button-icon"' in response.text
    assert "Live log" in response.text
    assert 'id="audit-log-content"' in response.text
    assert "overflow:auto" in response.text
    assert "openAuditLogDialog" in response.text
    assert "loadCurrentAuditLog" in response.text


def test_index_translates_common_runtime_errors() -> None:
    client = TestClient(create_app(FakeRunJudgeService()))

    response = client.get("/")

    assert response.status_code == 200
    assert "function friendlyErrorMessage" in response.text
    assert "Configure a URL do endpoint do juiz" in response.text
    assert "Configure a key local; não commitar" in response.text
    assert "O modelo não respeitou o contrato de saída" in response.text
    assert "Modelo sem acesso neste provedor" in response.text
    assert "Modelo inválido ou sem acesso nesse provedor" in response.text
    assert "aumentar timeout ou reduzir batch" in response.text
    assert "key inválida ou sem permissão" in response.text
    assert "acesso negado pelo provedor" in response.text
    assert "base URL/modelo incorreto" in response.text


def test_index_reloads_config_and_retries_stale_csrf_token() -> None:
    client = TestClient(create_app(FakeRunJudgeService()))

    response = client.get("/")

    assert response.status_code == 200
    assert "Invalid CSRF token." in response.text
    assert "return postJson(url, body, false)" in response.text
    assert 'document.getElementById("dry-run").disabled = false' in response.text
    assert 'document.getElementById("run").disabled = false' in response.text


def test_active_run_can_be_cancelled_without_deleting_progress() -> None:
    service = BlockingRunJudgeService()
    client = TestClient(create_app(service))
    token = client.get("/api/config").json()["csrf_token"]

    created = client.post(
        "/api/runs",
        headers={"x-csrf-token": token},
        json={"panel_mode": "single", "dataset": "J2", "batch_size": 1},
    )
    assert created.status_code == 200
    assert service.started.wait(timeout=1)

    run_id = created.json()["run_id"]
    cancelled = client.post(f"/api/runs/{run_id}/cancel", headers={"x-csrf-token": token}, json={})
    service.release.set()

    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelling"
    current = client.get(f"/api/runs/{run_id}")
    for _ in range(20):
        if current.json()["status"] == "cancelled":
            break
        time.sleep(0.05)
        current = client.get(f"/api/runs/{run_id}")
    assert current.status_code == 200
    assert current.json()["status"] == "cancelled"
    assert current.json()["result"]["summary"]["executed_evaluations"] == 1


def test_second_run_is_rejected_while_one_is_active() -> None:
    service = BlockingRunJudgeService()
    client = TestClient(create_app(service))
    token = client.get("/api/config").json()["csrf_token"]

    first = client.post(
        "/api/runs",
        headers={"x-csrf-token": token},
        json={"panel_mode": "single", "dataset": "J2", "batch_size": 1},
    )
    assert first.status_code == 200
    assert service.started.wait(timeout=1)

    second = client.post(
        "/api/runs",
        headers={"x-csrf-token": token},
        json={"panel_mode": "single", "dataset": "J2", "batch_size": 1},
    )
    service.release.set()

    assert second.status_code == 409


def test_run_history_lists_audit_logs_with_metadata(tmp_path) -> None:
    _write_audit_log(
        tmp_path / "judge_run_20260430_104512.log",
        """
2026-04-30T13:45:12+00:00 | audit_log_started | path=outputs/audit/judge_run_20260430_104512.log
2026-04-30T13:45:12+00:00 | execution_summary | Judge provider: remote_http | Judge mode: 2plus1 | Judge execution strategy: sequential
2026-04-30T13:45:12+00:00 | command_preview | .venv/bin/python -m atividade_2.cli run-judge --panel-mode 2plus1 --dataset J2 --batch-size 10 --judge-execution-strategy sequential
2026-04-30T13:45:20+00:00 | evaluation_parsed | answer_id=1 status=failed
2026-04-30T13:49:24+00:00 | execution_result | selected=10 executed=8 skipped=0 arbiters=1
2026-04-30T13:49:24+00:00 | audit_log_finished
""",
    )
    _write_audit_log(
        tmp_path / "judge_run_20260430_120000.log",
        """
2026-04-30T15:00:00+00:00 | audit_log_started | path=outputs/audit/judge_run_20260430_120000.log
2026-04-30T15:00:00+00:00 | execution_summary | Judge provider: remote_http | Judge mode: single | Judge execution strategy: parallel
2026-04-30T15:00:00+00:00 | command_preview | .venv/bin/python -m atividade_2.cli run-judge --panel-mode single --dataset J1 --batch-size 1 --judge-execution-strategy parallel
2026-04-30T15:00:07+00:00 | execution_result | selected=1 executed=1 skipped=0 arbiters=0
2026-04-30T15:00:07+00:00 | audit_log_finished
""",
    )
    client = TestClient(create_app(FakeRunJudgeService(), audit_dir=tmp_path))

    response = client.get("/api/run-history")

    assert response.status_code == 200
    data = response.json()
    assert [row["run_id"] for row in data] == ["judge_run_20260430_120000", "judge_run_20260430_104512"]
    older = data[1]
    assert older["timestamp"] == "2026-04-30T13:45:12+00:00"
    assert older["mode"] == "2plus1"
    assert older["dataset"] == "J2"
    assert older["batch_size"] == 10
    assert older["successes"] == 8
    assert older["failures"] == 1
    assert older["duration"] == "4min12s"
    assert older["log_url"] == "/api/run-history/judge_run_20260430_104512/audit-log"


def test_run_history_log_endpoint_returns_file_content(tmp_path) -> None:
    log_path = tmp_path / "judge_run_20260430_104512.log"
    _write_audit_log(log_path, "2026-04-30T13:45:12+00:00 | audit_log_started\n")
    client = TestClient(create_app(FakeRunJudgeService(), audit_dir=tmp_path))

    response = client.get("/api/run-history/judge_run_20260430_104512/audit-log")

    assert response.status_code == 200
    assert response.text == "2026-04-30T13:45:12+00:00 | audit_log_started\n"


def test_run_history_exports_csv_and_json(tmp_path) -> None:
    _write_audit_log(
        tmp_path / "judge_run_20260430_104512.log",
        """
2026-04-30T13:45:12+00:00 | audit_log_started
2026-04-30T13:45:12+00:00 | execution_summary | Judge mode: single
2026-04-30T13:45:12+00:00 | command_preview | .venv/bin/python -m atividade_2.cli run-judge --dataset J2 --batch-size 3
2026-04-30T13:45:13+00:00 | execution_result | selected=3 executed=3 skipped=0 arbiters=0
""",
    )
    client = TestClient(create_app(FakeRunJudgeService(), audit_dir=tmp_path))

    json_response = client.get("/api/run-history/export.json")
    csv_response = client.get("/api/run-history/export.csv")

    assert json_response.status_code == 200
    assert json_response.json()[0]["run_id"] == "judge_run_20260430_104512"
    assert csv_response.status_code == 200
    assert csv_response.text.splitlines()[0] == (
        "run_id,timestamp,mode,dataset,batch_size,successes,failures,duration,log_path"
    )
    assert "judge_run_20260430_104512,2026-04-30T13:45:12+00:00,single,J2,3,3,0,1s," in csv_response.text


def test_run_history_rejects_path_traversal(tmp_path) -> None:
    client = TestClient(create_app(FakeRunJudgeService(), audit_dir=tmp_path))

    response = client.get("/api/run-history/../secret/audit-log")

    assert response.status_code in {400, 404}


def _write_audit_log(path: Path, content: str) -> None:
    path.write_text(content.strip() + "\n", encoding="utf-8")
