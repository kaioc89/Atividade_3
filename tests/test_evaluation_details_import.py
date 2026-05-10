from __future__ import annotations

from pathlib import Path

from atividade_2.evaluation_details_import import EvaluationDetailsImporter


class FakeDetailsRepository:
    def __init__(self) -> None:
        self.details_by_id = {}
        self.match_id = 777
        self.match_calls = []

    def persist_evaluation_details(self, *, evaluation_id, details, cursor=None) -> None:
        current = self.details_by_id.get(evaluation_id)
        if current is None:
            self.details_by_id[evaluation_id] = details
            return
        merged_criteria = {**current.criteria, **details.criteria}
        self.details_by_id[evaluation_id] = details.__class__(
            legal_accuracy=details.legal_accuracy or current.legal_accuracy,
            hallucination_risk=details.hallucination_risk or current.hallucination_risk,
            rubric_alignment=details.rubric_alignment or current.rubric_alignment,
            requires_human_review=(
                details.requires_human_review
                if details.requires_human_review is not None
                else current.requires_human_review
            ),
            criteria=merged_criteria,
            raw_output_jsonb=details.raw_output_jsonb or current.raw_output_jsonb,
            source_log_path=details.source_log_path or current.source_log_path,
            run_id=details.run_id or current.run_id,
        )

    def find_evaluation_id_for_details(self, **kwargs):
        self.match_calls.append(kwargs)
        return self.match_id


def test_import_json_file_with_evaluation_id_is_idempotent_and_sanitized(tmp_path: Path) -> None:
    source = tmp_path / "details.jsonl"
    source.write_text(
        "\n".join(
            [
                '{"id_avaliacao": 10, "raw_output": {"score": 5, "rationale": "ok", '
                '"legal_accuracy": "alta", "hallucination_risk": "baixo", '
                '"rubric_alignment": "aderente", "requires_human_review": false, '
                '"criteria": {"citation_quality": "boa"}, '
                '"answer_completeness": "completa", "api_key": "sk-test-secret"}}'
            ]
        ),
        encoding="utf-8",
    )
    repository = FakeDetailsRepository()
    importer = EvaluationDetailsImporter(repository)  # type: ignore[arg-type]

    first = importer.import_sources(manifest_path=tmp_path / "missing_manifest.txt", raw_output_dirs=(source,))
    second = importer.import_sources(manifest_path=tmp_path / "missing_manifest.txt", raw_output_dirs=(source,))

    assert first.imported == 1
    assert second.imported == 1
    assert len(repository.details_by_id) == 1
    details = repository.details_by_id[10]
    assert details.legal_accuracy == "alta"
    assert details.hallucination_risk == "baixo"
    assert details.rubric_alignment == "aderente"
    assert details.requires_human_review is False
    assert details.criteria["citation_quality"] == "boa"
    assert details.criteria["answer_completeness"] == "completa"
    assert details.raw_output_jsonb["api_key"] == "<redacted>"


def test_import_json_file_can_match_unique_evaluation_from_metadata(tmp_path: Path) -> None:
    source = tmp_path / "details.json"
    source.write_text(
        '{"answer_id": 20, "model": "provider/judge", "role": "principal", '
        '"panel_mode": "single", "trigger": "single_mode", "score": 4, '
        '"parsed_output": {"score": 4, "rationale": "ok", "legal_accuracy": "media"}}',
        encoding="utf-8",
    )
    repository = FakeDetailsRepository()

    report = EvaluationDetailsImporter(repository).import_sources(  # type: ignore[arg-type]
        manifest_path=tmp_path / "missing_manifest.txt",
        raw_output_dirs=(source,),
    )

    assert report.imported == 1
    assert repository.details_by_id[777].legal_accuracy == "media"
    assert repository.match_calls == [
        {
            "answer_id": 20,
            "judge_model": "provider/judge",
            "role": "principal",
            "panel_mode": "single",
            "trigger_reason": "single_mode",
            "score": 4,
        }
    ]


def test_import_skips_record_when_match_is_ambiguous(tmp_path: Path) -> None:
    source = tmp_path / "details.json"
    source.write_text(
        '{"answer_id": 20, "model": "provider/judge", '
        '"raw_output": {"score": 4, "rationale": "ok", "legal_accuracy": "media"}}',
        encoding="utf-8",
    )
    repository = FakeDetailsRepository()
    repository.match_id = None

    report = EvaluationDetailsImporter(repository).import_sources(  # type: ignore[arg-type]
        manifest_path=tmp_path / "missing_manifest.txt",
        raw_output_dirs=(source,),
    )

    assert report.imported == 0
    assert report.skipped == 1
    assert "could not resolve a unique evaluation" in report.problems[0]


def test_import_reads_manifest_logs_only_when_raw_output_is_present(tmp_path: Path) -> None:
    manifest = tmp_path / "outputs" / "audit" / "prod_logs_manifest.txt"
    manifest.parent.mkdir(parents=True)
    log = manifest.parent / "judge_run_sample.log"
    log.write_text(
        "\n".join(
            [
                "2026-05-03T22:43:28+00:00 | evaluation_parsed | answer_id=1 model=ignored role=principal score=5",
                "2026-05-03T22:43:29+00:00 | evaluation_parsed | matched_evaluation_id=44 raw_output={\"score\":5,\"rationale\":\"ok\",\"legal_accuracy\":\"alta\"}",
            ]
        ),
        encoding="utf-8",
    )
    manifest.write_text("outputs/audit/judge_run_sample.log\n", encoding="utf-8")
    repository = FakeDetailsRepository()

    report = EvaluationDetailsImporter(repository).import_sources(manifest_path=manifest)  # type: ignore[arg-type]

    assert report.processed == 1
    assert report.imported == 1
    assert repository.details_by_id[44].legal_accuracy == "alta"
