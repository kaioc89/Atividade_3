# PRIMING.md

## Purpose

This file contains project/product/domain context. It should be loaded only when the task depends on AV2, AV3, legal datasets, LLM-as-a-Judge evaluation, RAG, PostgreSQL persistence, or experiment reproducibility.

Do not place task-specific workflows here. Put repeatable workflows in `.agents/skills/`.

## Project mission

This repository implements AV2 and is being evolved for AV3 in the graduate course "Tópicos Avançados em Engenharia de Software e Sistemas de Informação".

AV2 mission:

- build an auditable LLM-as-a-Judge evaluation framework backed by PostgreSQL;
- ingest original legal datasets;
- import AV1 model answers;
- execute judge models using explicit rubrics;
- persist scores and structured rationales;
- support SQL-based statistical analysis;
- calculate and interpret Spearman correlation when applicable;
- produce reproducible scripts, prompts, README instructions, backup, and restore evidence.

AV3 mission:

- evolve the AV2 foundation to evaluate candidate model answers generated with retrieved legal context;
- compare the existing `Sem_RAG` baseline against new `Com_RAG` answers;
- keep retrieval, candidate execution, judge evaluation, and analytics traceable in PostgreSQL;
- preserve the AV1/AV2 baseline while adding AV3-specific data under the `av3` schema.

## Domain

The project targets the legal domain for Equipe 4.

Datasets:

| Dataset | Source | Role |
|---|---|---|
| J1 | `maritaca-ai/oab-bench` | Open-ended OAB questions |
| J2 | `eduagarcia/oab_exams` | Multiple-choice OAB questions |

## Core AV2 experiment workflow

Preserve this pipeline:

```text
raw dataset
  -> normalized questions
  -> imported AV1 model answers
  -> judge prompt builder
  -> judge execution
  -> structured judge output
  -> PostgreSQL persistence
  -> SQL analysis
  -> Spearman correlation
  -> error analysis
  -> README/PDF/video evidence
```

The judge must not evaluate a candidate answer in isolation. The prompt must include:

- original question;
- candidate answer;
- reference, answer key, or rubric;
- judge instructions;
- expected output schema.

## Core AV3 experiment workflow

Preserve this AV3 comparison model:

```text
AV1/AV2 answers in public.respostas_atividade_1
  -> Sem_RAG baseline

AV3 curated legal source material
  -> av3.rag_documents
  -> av3.rag_chunks
  -> av3.rag_embeddings
  -> av3.retrieval_runs
  -> top-k retrieval per question
  -> candidate prompt with retrieved context only
  -> av3.candidate_runs
  -> av3.candidate_answers
  -> av3.candidate_answer_context_chunks
  -> AV2 judge pipeline adapted later for Com_RAG answers
  -> analytics comparing Sem_RAG vs Com_RAG
```

AV3 must answer not only whether RAG improved scores, but also:

- where RAG improved, harmed, or had no measurable effect;
- whether retrieved context was relevant;
- whether retrieved context introduced noise;
- whether hallucination risk decreased;
- whether gains vary by dataset, discipline, difficulty, or question type.

## Database role

The database is the audit source for the experiment, not temporary storage.

Minimum expected AV2 entities:

- datasets;
- models;
- questions;
- AV1 candidate answers;
- judge prompts;
- rubrics;
- judge evaluations;
- execution metadata.

Every judge evaluation must be traceable to:

- original question;
- candidate model;
- candidate answer;
- judge model;
- prompt version;
- rubric version;
- assigned score;
- structured rationale;
- execution timestamp.

Minimum expected AV3 entities:

- curated source import runs;
- RAG documents;
- RAG chunks;
- embeddings;
- retrieval runs;
- candidate prompt versions;
- candidate execution runs;
- candidate answers generated with RAG;
- retrieved context snapshots per candidate answer;
- post-RAG judge evaluations or adapters;
- analytics comparing `Sem_RAG` and `Com_RAG`.

## Recommended tables

The exact schema may evolve, but the model should preserve these responsibilities:

| Table | Responsibility |
|---|---|
| `datasets` | Register J1, J2, and domain metadata |
| `modelos` or `models` | Register candidate and judge models |
| `perguntas` or `questions` | Store prompt, answer key/reference, and metadata |
| `respostas_atividade_1` or `av1_answers` | Store AV1 candidate answers / `Sem_RAG` baseline |
| `rubricas` or `rubrics` | Store rubric versions |
| `prompts_juiz` or `judge_prompts` | Version judge prompts |
| `avaliacoes_juiz` or `judge_evaluations` | Store individual judge scores and rationales |
| `execucoes` or `executions` | Track execution metadata |
| `decisoes_finais` or `final_decisions` | Optional aggregation or 2+1 judge decisions |
| `av3.rag_documents` | Store traceable legal source documents for AV3 RAG |
| `av3.rag_chunks` | Store chunks used for retrieval |
| `av3.rag_embeddings` | Store vector embeddings for chunks |
| `av3.retrieval_runs` | Version retrieval configuration and active retrieval base |
| `av3.prompt_candidatos` | Version candidate prompts for Com_RAG generation |
| `av3.candidate_runs` | Track candidate model execution batches |
| `av3.candidate_answers` | Store Com_RAG answers by candidate model and question |
| `av3.candidate_answer_context_chunks` | Store exact retrieved context snapshots used per answer |

## AV3 candidate-safety rule

Candidate-side RAG must not receive answer material.

Do not include these in candidate retrieval context or candidate prompts:

- J1 guideline/rubric;
- J1 expected answer;
- J2 official answer key;
- correct alternative;
- judge prompt;
- judge rubric;
- human score;
- previous judge score;
- previous model ranking.

These materials belong to judge evaluation or analytics, not candidate generation.

## RAG traceability rule

Every Com_RAG candidate answer must be traceable to:

- original question;
- dataset code J1 or J2;
- candidate model;
- candidate prompt version;
- retrieval run;
- embedding model;
- top-k configuration;
- rendered prompt;
- generated answer;
- retrieved chunk ids;
- immutable chunk text snapshots;
- execution status and latency.

The system must preserve enough information to explain why a candidate saw a specific context even if the RAG base changes later.

## Chain-of-thought handling

The assignment may use the term "Chain-of-Thought". In implementation, store concise auditable rationale, not hidden reasoning.

If a database column is named `chain_of_thought` for assignment compatibility, treat it as `judge_rationale`.

## Judge output contract

Judge output should be machine-parseable.

Preferred JSON shape:

```json
{
  "score": 1,
  "rationale": "...",
  "legal_accuracy": "...",
  "hallucination_risk": "...",
  "rubric_alignment": "...",
  "requires_human_review": false
}
```

Scores must be integers from 1 to 5.

## Legal rubric priorities

The judge should prioritize:

1. correctness of the legal conclusion;
2. accuracy of legal basis;
3. absence of fabricated laws, articles, precedents, or doctrines;
4. reasoning quality;
5. alignment with expected answer/rubric;
6. concision and relevance.

The judge must not reward verbosity by itself.

## J2 rule

For J2, preserve the official answer key for evaluation only.

Candidate answers should be evaluated by:

- whether the selected option is correct;
- whether the explanation is legally coherent;
- whether hallucinated legal basis appears.

For correlation analysis, the human reference can be mapped to an ordinal score when needed:

- `5` when the candidate selected the correct option;
- `1` when the candidate selected the wrong option.

Do not expose the correct alternative to AV3 candidate retrieval or candidate prompt construction.

## J1 rule

For J1, evaluate each answer against the item-specific guideline/rubric.

Do not compare one model answer against another as if it were gold. All candidate answers must be judged independently against the same reference.

Do not expose the guideline/rubric/reference answer to AV3 candidate retrieval or candidate prompt construction.

## Recommended execution strategy

For AV2 judge work:

1. J2 first, because multiple-choice has an objective answer key and is easier to validate.
2. J1 second, because open-ended questions require richer rubrics.
3. Add 2+1 judge review only after the single-judge pipeline is stable.
4. Persist every prompt, rubric, judge output, score, rationale, and execution metadata.
5. Treat error analysis as a core deliverable, not a stretch goal.

For AV3 candidate-RAG work:

1. Preserve the AV1/AV2 `Sem_RAG` baseline.
2. Build or validate the AV3 RAG base.
3. Retrieve top-k chunks per question for J1 and J2.
4. Persist context snapshots for every Com_RAG answer.
5. Generate candidate answers through the centralized app/worker flow.
6. Reuse or adapt the AV2 judge pipeline for Com_RAG answers.
7. Add analytics only after persistence and judge adaptation are stable.
8. Report both improvements and regressions.

## Python infrastructure baseline

Python tooling should follow a `src/` package layout with `pyproject.toml` as the single packaging/config entry point.

Use:

```bash
make install
make test
```

Generated artifacts belong under `outputs/`.
Stable input files belong under `resources/`.
Source code belongs under `src/`.
Tests belong under `tests/`.

## Local PostgreSQL baseline

Local PostgreSQL must be reproducible through Make and Docker Compose.

Use PostgreSQL 18.3 or the version required by the checked-in backup artifacts and compose image.

Core commands:

```bash
make db-up
make db-migrate-or-create
make db-restore-validate
make db-backup
make db-down
```

The root `backup_atividade_2_reset.sql` is the fixed reset artifact. Backup generation writes timestamped history under `outputs/backup/` and may update the single shared latest backup at `backup_atividade_2.sql` according to the current application settings and documentation.

`atividade2.ipynb` is not part of the reproducible setup path. Prefer deterministic Make/script workflows for professor and teammate validation.

## Reproducibility expectations

The final project should include:

- PostgreSQL DDL;
- data import scripts;
- RAG base materialization or validation scripts;
- candidate prompt templates;
- judge prompt templates;
- SQL analysis queries;
- Spearman correlation calculation when applicable;
- `.sql` or `.dump` backup;
- restore instructions;
- validation query after restore;
- README explaining methodology, database, execution, rubric, RAG workflow, and analysis.
