from __future__ import annotations

from collections import Counter
from datetime import datetime

from atividade_2.repositories import JudgeRepository


DEFAULT_MODELS = {
    1: "Gemma3:12b",
    2: "Llama3.1:8b",
    3: "Qwen2.5-3B-Instruct",
    4: "qwen2.5-7b-instruct",
    5: "Llama-3.2-3B-Instruct",
    6: "gemma2",
    7: "llama321b",
    8: "jurema-7b",
    9: "llama323b",
    10: "gemma-2-2b-it",
    11: "Jurema:7b",
    12: "curio-edu-7b",
    13: "gemini-3-pro",
    14: "chatgpt-5.3",
    15: "grok-3",
    16: "qwen2-1.5b",
    17: "phi-3-mini",
    18: "tinyllama-1.1b",
    19: "qwen2.5-1.5b",
    20: "tinyllama-1.1b",
}


class AssignmentRegistryCursor:
    def __init__(self, *, models: dict[int, str]) -> None:
        self.models = dict(models)
        self.queries: list[str] = []
        self.params: list[list[object]] = []
        self.rowcount = 1
        self._fetchone_row = None
        self._fetchall_rows: list[tuple[object, ...]] = []
        self.state = {
            "assignments_by_key": {},
            "assignments_by_id": {},
            "ranges_by_assignment": {},
            "next_assignment_id": 1,
            "next_assignment_range_id": 1,
        }

    def __enter__(self) -> AssignmentRegistryCursor:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def execute(self, query, params=None) -> None:
        params_list = list(params or [])
        self.queries.append(query)
        self.params.append(params_list)
        compact_query = " ".join(query.split())
        self._fetchone_row = None
        self._fetchall_rows = []

        if "SELECT id_modelo, nome_modelo FROM public.modelos" in compact_query:
            model_id = int(params_list[0])
            model_name = self.models.get(model_id)
            self._fetchone_row = (model_id, model_name) if model_name is not None else None
            return

        if "INSERT INTO av3.candidate_model_assignments" in compact_query:
            (
                id_modelo_av2,
                owner,
                original_provider_model_id,
                original_runtime,
                av3_provider,
                av3_provider_model_id,
                hf_model_id,
                artifact_format,
                original_quantization,
                av3_quantization,
                match_type,
                validation_status,
                notes,
                active,
            ) = params_list
            assignment_key = (int(id_modelo_av2), str(owner), str(original_provider_model_id))
            assignment_id = self.state["assignments_by_key"].get(assignment_key)
            now = datetime(2026, 6, 4, 12, 0, 0)
            if assignment_id is None:
                assignment_id = self.state["next_assignment_id"]
                self.state["next_assignment_id"] += 1
                self.state["assignments_by_key"][assignment_key] = assignment_id
                created_at = now
            else:
                created_at = self.state["assignments_by_id"][assignment_id]["created_at"]
            self.state["assignments_by_id"][assignment_id] = {
                "id_assignment": assignment_id,
                "id_modelo_av2": int(id_modelo_av2),
                "owner": str(owner),
                "original_provider_model_id": str(original_provider_model_id),
                "original_runtime": str(original_runtime),
                "av3_provider": str(av3_provider),
                "av3_provider_model_id": av3_provider_model_id,
                "hf_model_id": hf_model_id,
                "artifact_format": str(artifact_format),
                "original_quantization": original_quantization,
                "av3_quantization": av3_quantization,
                "match_type": str(match_type),
                "validation_status": str(validation_status),
                "notes": notes,
                "active": bool(active),
                "created_at": created_at,
                "updated_at": now,
            }
            self.state["ranges_by_assignment"].setdefault(assignment_id, [])
            self._fetchone_row = (assignment_id,)
            return

        if "DELETE FROM av3.candidate_model_assignment_ranges" in compact_query:
            assignment_id = int(params_list[0])
            self.state["ranges_by_assignment"][assignment_id] = []
            return

        if "INSERT INTO av3.candidate_model_assignment_ranges" in compact_query:
            assignment_id, dataset_code, question_sequence_start, question_sequence_end = params_list
            assignment_range_id = self.state["next_assignment_range_id"]
            self.state["next_assignment_range_id"] += 1
            self.state["ranges_by_assignment"].setdefault(int(assignment_id), []).append(
                {
                    "id_assignment_range": assignment_range_id,
                    "id_assignment": int(assignment_id),
                    "dataset_code": str(dataset_code),
                    "question_sequence_start": int(question_sequence_start),
                    "question_sequence_end": int(question_sequence_end),
                }
            )
            self._fetchone_row = (assignment_range_id,)
            return

        if (
            "FROM av3.candidate_model_assignments a" in compact_query
            and "LEFT JOIN av3.candidate_model_assignment_ranges r" in compact_query
        ):
            rows = _select_assignment_rows(self.state, self.models)
            if "WHERE a.id_assignment = %s" in compact_query:
                requested_assignment_id = int(params_list[0])
                rows = [row for row in rows if int(row[0]) == requested_assignment_id]
            self._fetchall_rows = rows
            return

        raise AssertionError(f"Unexpected query in assignment registry test: {compact_query}")

    def fetchone(self):
        row = self._fetchone_row
        self._fetchone_row = None
        return row

    def fetchall(self):
        rows = list(self._fetchall_rows)
        self._fetchall_rows = []
        return rows


class AssignmentRegistryConnection:
    def __init__(self, cursor: AssignmentRegistryCursor) -> None:
        self.cursor_instance = cursor

    def __enter__(self) -> AssignmentRegistryConnection:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def cursor(self) -> AssignmentRegistryCursor:
        return self.cursor_instance


def _select_assignment_rows(state: dict[str, object], models: dict[int, str]) -> list[tuple[object, ...]]:
    rows: list[tuple[object, ...]] = []
    assignments = state["assignments_by_id"]
    ranges_by_assignment = state["ranges_by_assignment"]
    for assignment_id, assignment in sorted(
        assignments.items(),
        key=lambda item: (
            str(item[1]["owner"]),
            int(item[1]["id_modelo_av2"]),
            int(item[0]),
        ),
    ):
        ranges = list(ranges_by_assignment.get(assignment_id, []))
        if not ranges:
            rows.append(
                (
                    assignment_id,
                    int(assignment["id_modelo_av2"]),
                    models[int(assignment["id_modelo_av2"])],
                    assignment["owner"],
                    assignment["original_provider_model_id"],
                    assignment["original_runtime"],
                    assignment["av3_provider"],
                    assignment["av3_provider_model_id"],
                    assignment["hf_model_id"],
                    assignment["artifact_format"],
                    assignment["original_quantization"],
                    assignment["av3_quantization"],
                    assignment["match_type"],
                    assignment["validation_status"],
                    assignment["notes"],
                    assignment["active"],
                    assignment["created_at"],
                    assignment["updated_at"],
                    None,
                    None,
                    None,
                    None,
                )
            )
            continue
        for assignment_range in sorted(
            ranges,
            key=lambda item: (
                str(item["dataset_code"]),
                int(item["question_sequence_start"]),
                int(item["question_sequence_end"]),
                int(item["id_assignment_range"]),
            ),
        ):
            rows.append(
                (
                    assignment_id,
                    int(assignment["id_modelo_av2"]),
                    models[int(assignment["id_modelo_av2"])],
                    assignment["owner"],
                    assignment["original_provider_model_id"],
                    assignment["original_runtime"],
                    assignment["av3_provider"],
                    assignment["av3_provider_model_id"],
                    assignment["hf_model_id"],
                    assignment["artifact_format"],
                    assignment["original_quantization"],
                    assignment["av3_quantization"],
                    assignment["match_type"],
                    assignment["validation_status"],
                    assignment["notes"],
                    assignment["active"],
                    assignment["created_at"],
                    assignment["updated_at"],
                    int(assignment_range["id_assignment_range"]),
                    assignment_range["dataset_code"],
                    int(assignment_range["question_sequence_start"]),
                    int(assignment_range["question_sequence_end"]),
                )
            )
    return rows


def _seeded_repository(models: dict[int, str] | None = None) -> tuple[JudgeRepository, AssignmentRegistryCursor]:
    cursor = AssignmentRegistryCursor(models=models or DEFAULT_MODELS)
    repository = JudgeRepository(AssignmentRegistryConnection(cursor))
    return repository, cursor


def test_upsert_default_candidate_model_assignments_creates_full_mapping_and_owner_counts() -> None:
    repository, _cursor = _seeded_repository()

    assignments = repository.upsert_default_candidate_model_assignments()

    assert len(assignments) == 20
    assert {assignment.owner for assignment in assignments} == {
        "Diego",
        "José Bruno",
        "Kaio",
        "Paulo",
        "Victor",
        "Wagner",
    }
    assert Counter(assignment.owner for assignment in assignments) == {
        "Diego": 3,
        "Kaio": 3,
        "Wagner": 3,
        "Victor": 5,
        "José Bruno": 3,
        "Paulo": 3,
    }
    assert all(assignment.assignment_id is not None for assignment in assignments)
    assert all(assignment.av2_model_name for assignment in assignments)
    assert all(assignment.id_modelo_av2 >= 1 for assignment in assignments)
    assert all(
        assignment_range.dataset_code in {"J1", "J2"}
        and assignment_range.question_sequence_start <= assignment_range.question_sequence_end
        for assignment in assignments
        for assignment_range in assignment.ranges
    )


def test_upsert_default_candidate_model_assignments_is_idempotent_and_replaces_ranges() -> None:
    repository, cursor = _seeded_repository()

    repository.upsert_default_candidate_model_assignments()
    assignments = repository.upsert_default_candidate_model_assignments()

    assert len(assignments) == 20
    assert len(cursor.state["assignments_by_id"]) == 20
    assert sum(len(ranges) for ranges in cursor.state["ranges_by_assignment"].values()) == 36
    assert len(repository.find_candidate_model_assignments_for_model_id(17)[0].ranges) == 2


def test_upsert_default_candidate_model_assignments_fails_when_required_public_model_is_missing() -> None:
    models = dict(DEFAULT_MODELS)
    del models[20]
    repository, _cursor = _seeded_repository(models=models)

    error_message = None
    try:
        repository.upsert_default_candidate_model_assignments()
    except ValueError as exc:
        error_message = str(exc)

    assert error_message == "Missing public.modelos row for id_modelo_av2=20."


def test_assignment_registry_query_helpers_filter_by_owner_dataset_question_provider_and_model() -> None:
    repository, _cursor = _seeded_repository()
    repository.upsert_default_candidate_model_assignments()

    jose_assignments = repository.find_candidate_model_assignments_for_owner("Jose Bruno")
    j1_assignments = repository.find_candidate_model_assignments_for_dataset("J1")
    j2_assignments = repository.find_candidate_model_assignments_for_dataset("J2")
    j1_question_119 = repository.find_candidate_model_assignments_for_question("J1", 119)
    j2_question_1231 = repository.find_candidate_model_assignments_for_question("J2", 1231)
    j1_question_95 = repository.find_candidate_model_assignments_for_question("J1", 95)
    out_of_range = repository.find_candidate_model_assignments_for_question("J1", 1000)
    openrouter_assignments = repository.find_candidate_model_assignments_for_provider("openrouter")
    model_id_14 = repository.find_candidate_model_assignments_for_model_id(14)

    assert len(jose_assignments) == 3
    assert len(j1_assignments) == 18
    assert len(j2_assignments) == 18
    assert {assignment.owner for assignment in j1_question_119} == {"José Bruno"}
    assert len(j1_question_119) == 3
    assert {assignment.owner for assignment in j2_question_1231} == {"José Bruno"}
    assert len(j2_question_1231) == 3
    assert {assignment.owner for assignment in j1_question_95} == {"Wagner"}
    assert len(j1_question_95) == 3
    assert out_of_range == ()
    assert {assignment.id_modelo_av2 for assignment in openrouter_assignments} == {5, 9, 13, 14, 15}
    assert len(model_id_14) == 1
    assert model_id_14[0].owner == "José Bruno"
    assert model_id_14[0].av3_provider_model_id == "openai/gpt-5-chat"
    assert model_id_14[0].notes == (
        "AV3 runtime adjustment: using openai/gpt-5-chat because the original "
        "AV1 runtime was ChatGPT UI and openai/gpt-5 returned empty/unparsed "
        "model text through OpenRouter."
    )
    jose_grok = repository.find_candidate_model_assignments_for_model_id(15)[0]
    diego_llama = repository.find_candidate_model_assignments_for_model_id(9)[0]
    kaio_llama = repository.find_candidate_model_assignments_for_model_id(5)[0]

    assert diego_llama.av3_provider == "openrouter"
    assert diego_llama.av3_provider_model_id == "meta-llama/llama-3.2-3b-instruct"
    assert diego_llama.hf_model_id == "meta-llama/Llama-3.2-3B-Instruct"
    assert diego_llama.notes == (
        "AV3 runtime adjustment: Featherless returned a provider chat-template "
        "error for meta-llama/Llama-3.2-3B-Instruct. Using the same model "
        "identity through OpenRouter as meta-llama/llama-3.2-3b-instruct for "
        "execution compatibility."
    )
    assert kaio_llama.av3_provider == "openrouter"
    assert kaio_llama.av3_provider_model_id == "meta-llama/llama-3.2-3b-instruct"
    assert kaio_llama.hf_model_id == "meta-llama/Llama-3.2-3B-Instruct"
    assert kaio_llama.notes == diego_llama.notes

    assert jose_grok.owner == "José Bruno"
    assert jose_grok.id_modelo_av2 == 15
    assert jose_grok.av2_model_name == "grok-3"
    assert jose_grok.original_provider_model_id == "Grok 3"
    assert jose_grok.original_runtime == "Grok UI"
    assert jose_grok.av3_provider == "openrouter"
    assert jose_grok.av3_provider_model_id == "x-ai/grok-4.3"
    assert jose_grok.match_type == "same_family_newer_version_substitution"
    assert jose_grok.validation_status == "approved_version_substitution"
    assert jose_grok.notes == (
        "Grok 3 label preserved from AV1/AV2. AV3 execution uses x-ai/grok-4.3 via "
        "OpenRouter as a team-approved newer-version substitution because Grok 3 is "
        "not currently available in the provider catalog."
    )
    assert jose_grok.is_runnable() is True


def test_assignment_registry_runnable_pending_and_excluded_filters_follow_required_rules() -> None:
    repository, _cursor = _seeded_repository()
    repository.upsert_default_candidate_model_assignments()

    default_runnable = repository.list_runnable_candidate_model_assignments()
    pending_enabled = repository.list_runnable_candidate_model_assignments(
        include_pending_confirmation=True
    )
    pending = repository.list_pending_candidate_model_assignments()
    excluded = repository.list_excluded_candidate_model_assignments()
    gpt5_assignment = repository.find_candidate_model_assignments_for_model_id(14)[0]

    assert gpt5_assignment.is_runnable() is True
    assert len(default_runnable) == 16
    assert len(pending_enabled) == 17
    assert 14 in {assignment.id_modelo_av2 for assignment in default_runnable}
    assert 13 not in {assignment.id_modelo_av2 for assignment in default_runnable}
    assert 15 in {assignment.id_modelo_av2 for assignment in default_runnable}
    assert 13 in {assignment.id_modelo_av2 for assignment in pending_enabled}
    assert {assignment.id_modelo_av2 for assignment in excluded} == {8, 11, 12}
    assert {assignment.id_modelo_av2 for assignment in pending} == {13}
    assert all(assignment.av3_provider_model_id for assignment in default_runnable)
    assert all(assignment.av3_provider not in {"excluded", "unresolved"} for assignment in default_runnable)
    assert "openai/gpt-5" not in {assignment.av3_provider_model_id for assignment in default_runnable}
    assert not any(
        assignment.id_modelo_av2 in {5, 9} and assignment.av3_provider == "featherless"
        for assignment in default_runnable
    )


def test_assignment_registry_serialization_preserves_both_identities_and_helpers_do_not_mutate_records() -> None:
    repository, _cursor = _seeded_repository()
    repository.upsert_default_candidate_model_assignments()

    assignment = repository.find_candidate_model_assignments_for_model_id(13)[0]
    payload = assignment.to_dict(include_pending_confirmation=True)

    payload["ranges"][0]["dataset_code"] = "J9"

    assert isinstance(repository.list_candidate_model_assignments(), tuple)
    assert isinstance(assignment.ranges, tuple)
    assert payload["id_modelo_av2"] == 13
    assert payload["av2_model_name"] == "gemini-3-pro"
    assert payload["av3_provider"] == "openrouter"
    assert payload["av3_provider_model_id"] == "google/gemini-3.5-flash"
    assert payload["runnable"] is True
    assert payload["warning_message"] == (
        "Pending owner confirmation: Gemini subtype still needs exact confirmation."
    )
    assert assignment.ranges[0].dataset_code == "J1"


def test_assignment_registry_serialization_preserves_grok_visible_identity_and_av3_execution_details() -> None:
    repository, _cursor = _seeded_repository()
    repository.upsert_default_candidate_model_assignments()

    assignment = repository.find_candidate_model_assignments_for_model_id(15)[0]
    payload = assignment.to_dict()

    assert payload["id_modelo_av2"] == 15
    assert payload["av2_model_name"] == "grok-3"
    assert payload["original_provider_model_id"] == "Grok 3"
    assert payload["original_runtime"] == "Grok UI"
    assert payload["av3_provider"] == "openrouter"
    assert payload["av3_provider_model_id"] == "x-ai/grok-4.3"
    assert payload["match_type"] == "same_family_newer_version_substitution"
    assert payload["validation_status"] == "approved_version_substitution"
    assert payload["runnable"] is True
