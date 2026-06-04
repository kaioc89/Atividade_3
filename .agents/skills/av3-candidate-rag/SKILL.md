---
name: av3-candidate-rag
description: Use when implementing or reviewing AV3 candidate-RAG work, including retrieval, Com_RAG candidate answers, context snapshots, candidate prompt rendering, candidate execution, or Sem_RAG vs Com_RAG comparison. Do not use for AV2-only judge pipeline work unless adapting judges to AV3 candidate answers.
---

# AV3 Candidate RAG

## 1. Purpose

Implement and validate the AV3 candidate-RAG workflow while preserving the delivered AV1/AV2 baseline.

The core AV3 objective is to compare:

```text
Sem_RAG = existing AV1/AV2 candidate answers
Com_RAG = new candidate answers generated with retrieved legal context
```

The workflow must remain auditable, reproducible, and safe from answer leakage.

## 2. When to use

Use this skill when the task involves:

- AV3 RAG source curation;
- RAG document/chunk/embedding/retrieval-run behavior;
- retrieval for one question by `id_pergunta`;
- candidate-safe context construction;
- candidate prompt templates for Com_RAG;
- candidate model execution runs;
- `av3.candidate_runs`;
- `av3.candidate_answers`;
- `av3.candidate_answer_context_chunks`;
- adapting AV2 judge flow to evaluate AV3 Com_RAG answers;
- analytics comparing Sem_RAG and Com_RAG.

## 3. When not to use

Do not use this skill for:

- AV2-only judge pipeline changes with no AV3 impact;
- generic Python setup;
- generic PostgreSQL backup/restore;
- pure rubric authoring;
- standalone SQL analysis unrelated to AV3;
- repository instruction maintenance.

## 4. Required inputs

Identify:

- dataset: `J1` or `J2`;
- `id_pergunta` or question range;
- active retrieval run;
- embedding model;
- top-k configuration;
- candidate model identifier;
- candidate prompt version;
- whether the task is retrieval-only, snapshot persistence, prompt rendering, candidate execution, judge adaptation, or analytics;
- database connection method;
- expected validation command.

For J1:

- input is an open-ended OAB question;
- candidate output is free-form legal reasoning;
- rubric/guideline/reference answer is judge-only material.

For J2:

- input is a multiple-choice OAB question;
- candidate output must eventually include a selected alternative;
- official answer key is judge-only material.

## 5. Required commands

Use project-specific commands when available.

Baseline command pattern:

```bash
.venv/bin/python -m atividade_2.cli --help
.venv/bin/python -m pytest
```

For database validation:

```bash
make db-up
make db-migrate-or-create
make db-restore-validate
```

Inspect changed files:

```bash
git diff --stat
git diff -- src tests database README.md AGENTS.md PRIMING.md PLANS.md .agents
```

## 6. Workflow

### Step 1 — Read project context

Load `PRIMING.md`.

Confirm:

- J1 and J2 dataset semantics;
- AV2 baseline must remain stable;
- `Sem_RAG` uses existing AV1/AV2 answers;
- `Com_RAG` uses new AV3 candidate answers;
- candidate-side retrieval must not expose answer material;
- every Com_RAG answer must be traceable to retrieved context snapshots.

### Step 2 — Classify the AV3 slice

Classify the task as one slice:

1. schema/contracts;
2. retrieval;
3. context snapshot persistence;
4. candidate prompt rendering;
5. candidate execution;
6. Web UI orchestration;
7. judge adaptation;
8. analytics.

Do not combine slices unless explicitly requested.

### Step 3 — Map the data flow

For retrieval/candidate work, document the relevant flow:

```text
question id
  -> question text
  -> active retrieval run
  -> embedding model
  -> query embedding
  -> top-k chunks
  -> candidate-safe context
  -> rendered prompt
  -> candidate answer
  -> context snapshot rows
```

For judge adaptation, document:

```text
Com_RAG answer
  + original question
  + judge-only reference/rubric/answer key
  + judge prompt version
  -> judge request
  -> structured judge output
  -> persisted evaluation
```

### Step 4 — Enforce candidate-safety

Candidate-facing retrieval and prompts must not include:

- J1 guideline/rubric;
- J1 expected answer;
- J2 official answer key;
- correct alternative;
- judge prompt;
- judge rubric;
- human score;
- previous judge score;
- previous model ranking.

If any of those appear in candidate-facing context, stop and fix the design.

### Step 5 — Preserve traceability

Every Com_RAG candidate answer should be traceable to:

- original question;
- dataset;
- candidate model;
- candidate prompt version;
- retrieval run;
- embedding model;
- top-k value;
- rendered prompt;
- generated answer;
- retrieved chunk ids;
- immutable chunk text snapshots;
- status and latency.

### Step 6 — Test before broad execution

Prefer fake repositories/providers for unit tests.

Use live PostgreSQL tests only when verifying real constraints, restore behavior, or database-specific functionality.

Run a small fixture before full dataset execution.

## 7. Output format

```md
## Context & Goal

## AV3 slice

## Data flow

## Candidate-safety checks

## Implementation summary

## Persistence mapping

## Validation performed

## Risks and follow-ups
```

## 8. Validation checklist

- [ ] AV1/AV2 baseline tables remain unchanged unless explicitly scoped.
- [ ] J1 and J2 are handled explicitly where relevant.
- [ ] Candidate-facing retrieval excludes answer keys, rubrics, guidelines, and gold/reference answers.
- [ ] Retrieval uses active AV3 vector/retrieval metadata.
- [ ] Top-k behavior is deterministic or explicitly documented.
- [ ] No fake chunks are created for fallback states.
- [ ] Com_RAG answers have context snapshots before evaluation.
- [ ] Prompt versions are persisted or referenced.
- [ ] Status/error handling is explicit.
- [ ] Tests cover success and fallback states.
- [ ] Small execution sample passes before full run.

## 9. Guardrails

- Do not rewrite AV2 to implement AV3.
- Do not use notebooks as the official execution path.
- Do not use spreadsheets as the source of truth.
- Do not expose answer material to candidate RAG.
- Do not evaluate Com_RAG answers without question and judge-only reference/rubric context.
- Do not rank models without reporting regressions, retrieval noise, and failure cases.
- Do not persist hidden chain-of-thought.
- Do not silently accept missing retrieval context as success.
- Do not mix candidate execution, judge adaptation, and analytics in one PR unless explicitly requested.
