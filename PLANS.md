# PLANS.md

## Purpose

This file defines how to plan larger, risky, architectural, or multi-step work.

Do not use this file as a replacement for task-specific skills. Use it to structure execution plans and risk control.

## When to create a plan

Create a plan when the task:

- changes database schema;
- changes import or persistence behavior;
- changes retrieval or RAG context-selection behavior;
- changes candidate execution behavior;
- changes judge scoring semantics;
- touches multiple files or directories;
- affects reproducibility;
- has unclear blast radius;
- requires migration or backfill;
- involves external services or model execution;
- could invalidate previous experiment results.

For small localized edits, provide only a concise inline plan.

## Scope classification

Classify work before implementation:

| Scope | Definition |
|---|---|
| `[MINOR]` | Small localized change, usually 1 file or mechanical update |
| `[MODERATE]` | Multiple related files or behavior changes with limited blast radius |
| `[MAJOR]` | Schema, architecture, pipeline, model execution, retrieval, or reproducibility-impacting change |

## Required planning sections

For non-trivial work, produce:

1. Problem statement
2. Assumptions
3. Constraints
4. Scope classification
5. Impacted files
6. Data flow
7. Risks
8. Implementation slices
9. Validation gates
10. Rollback or recovery strategy

## Data flow expectations

When changing data ingestion, persistence, retrieval, evaluation, or analysis, map:

- input source;
- input shape;
- transformation logic;
- output shape;
- validation points;
- database write path;
- generated artifacts;
- failure modes.

## AV3 data flow expectations

When changing AV3 candidate-RAG behavior, map the relevant subset of this flow:

```text
question id
  -> question text
  -> active retrieval run
  -> embedding model
  -> query embedding
  -> top-k retrieved chunks
  -> candidate-safe context
  -> rendered candidate prompt
  -> candidate model answer
  -> context snapshot persistence
  -> post-RAG judge evaluation
  -> Sem_RAG vs Com_RAG analytics
```

Explicitly state which steps are in scope and which are deferred.

## Implementation slices

Prefer small, functional slices.

Example sequence:

1. Add or update typed contract.
2. Add failing tests or validation fixtures.
3. Implement minimal logic.
4. Add CLI command or script.
5. Add deterministic validation.
6. Update README only if behavior or usage changed.

Each slice should be independently reviewable when possible.

## AV3 candidate-RAG slice guidance

Prefer this sequence for AV3 candidate-RAG work:

1. Schema and contracts for candidate prompts, runs, answers, and context snapshots.
2. Retrieval service for one question by `id_pergunta`.
3. Snapshot persistence for retrieved chunks.
4. Candidate prompt rendering.
5. Candidate runner service/CLI.
6. Web UI controls for centralized execution.
7. Judge adaptation for Com_RAG answers.
8. Analytics comparing Sem_RAG and Com_RAG.

Do not combine candidate execution, judge adaptation, analytics, and UI in the same PR unless explicitly requested.

## Approval checkpoints

Ask for explicit approval before implementation when:

- schema changes are destructive;
- existing outputs would be invalidated;
- task requires broad refactoring;
- task changes scoring semantics;
- task changes dataset interpretation;
- task introduces new external service behavior;
- task changes candidate-facing retrieval context semantics;
- narrower implementation options are not viable.

For low-risk localized changes, proceed with explicit assumptions.

## Validation gates

Use deterministic validation whenever possible.

Common gates:

```bash
make test
```

For local database setup, backup, or restore changes, prefer project-specific commands such as:

```bash
make db-up
make db-migrate-or-create
make db-restore-validate
make db-backup
make db-status
make db-down
```

Use direct `psql` commands only when a task specifically requires ad hoc SQL validation beyond the existing scripts.

For repository instruction changes:

```bash
find . -maxdepth 4 -type f \( -name "AGENTS.md" -o -name "PRIMING.md" -o -name "PLANS.md" -o -name "SKILL.md" \) -print
git diff --stat
git diff -- AGENTS.md PRIMING.md PLANS.md .agents
```

For AV3 candidate-RAG work, use focused tests first. Run broader tests only after the narrow tests pass.

## Risk categories

Track risks using these categories:

- data loss;
- duplicate imports;
- broken foreign keys;
- invalid judge output;
- non-reproducible execution;
- hidden prompt changes;
- scoring drift;
- model/provider nondeterminism;
- slow or flaky tests;
- overfitted rubric;
- context pollution;
- retrieval drift;
- retrieval noise;
- stale embeddings;
- candidate access to answer material;
- missing context snapshots;
- Sem_RAG/Com_RAG comparison mismatch.

## Plan output format

Use this format:

```md
## Context & Goal

## Assumptions

## Constraints

## Scope

## Impacted files

## Proposed implementation slices

## Validation plan

## Risks

## Open questions
```

## Execution summary format

After implementation, respond with:

```md
## Summary

## Files created

## Files modified

## Validation performed

## Remaining risks or follow-ups

## Suggested next command
```
