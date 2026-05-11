# AGENTS.md

## Purpose

This file contains only global, always-applicable guidance for AI coding agents working in this repository.

Task-specific workflows live under `.agents/skills/`.
Project/domain context lives in `PRIMING.md`.
Planning guidance for larger or risky work lives in `PLANS.md`.

Repo-local skill auto-loading behavior requires validation in the local Codex environment. If unsupported, open the relevant `SKILL.md` manually before executing the task.

## Global engineering principles

- Prefer minimal, targeted, testable changes.
- Do not refactor unrelated code.
- Preserve existing behavior unless the task explicitly requires changing it.
- Keep implementations simple and readable.
- Prefer composition over mutation of stable working logic.
- Follow existing project structure, naming, and test patterns.
- Avoid placeholders, TODOs, incomplete code, and vague fallback behavior.
- Do not commit changes unless explicitly requested.

## AI assistant LLM invocation rule

The AI assistant must always call the LLM for valid questions. This rule must
not be changed unless explicitly requested by the user; when possible, confirm
with the user before changing it.

## Python command policy

Use the project virtual environment for Python commands:

```bash
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest
.venv/bin/python -m atividade_2.cli --help
```

Do not use `python`, `python3`, `pip`, or `pip3`.

If `.venv` does not exist, stop and report the required setup command. Do not silently fall back to system Python.

## Validation expectations

Before considering work complete:

- Run the narrowest relevant validation first.
- Run broader validation when the change affects shared behavior.
- Prefer deterministic scripts and tests over natural-language instructions.
- Document commands executed and results observed.
- Do not claim validation passed if it was not run.

## Output defaults

For implementation work, respond with:

1. Summary
2. Files created
3. Files modified
4. Validation performed
5. Remaining risks or follow-ups
6. Suggested next command

For review or debugging work, use the specific skill output format.

## Skill routing

Use these task-specific workflows:

| Task type | Skill |
|---|---|
| Python project setup/review | `.agents/skills/python-project-bootstrap/SKILL.md` |
| AV2 judge pipeline | `.agents/skills/av2-judge-pipeline/SKILL.md` |
| Database import and validation | `.agents/skills/database-import-validation/SKILL.md` |
| LLM judge rubric design | `.agents/skills/llm-judge-rubric-design/SKILL.md` |
| SQL analysis and Spearman correlation | `.agents/skills/sql-analysis-and-spearman/SKILL.md` |
| PostgreSQL backup/restore | `.agents/skills/backup-restore-validation/SKILL.md` |
| Senior code review | `.agents/skills/senior-code-review/SKILL.md` |
| Defect/root-cause analysis | `.agents/skills/defect-root-cause-analysis/SKILL.md` |
| Test strategy | `.agents/skills/test-strategy/SKILL.md` |
| Instruction/context maintenance | `.agents/skills/repository-instruction-maintenance/SKILL.md` |

## When to read PRIMING.md

Read `PRIMING.md` when the task depends on:

- project mission or scope;
- domain vocabulary;
- dataset meaning;
- database/pipeline invariants;
- existing architectural decisions;
- AV2/J1/J2/LLM-as-a-Judge context.

Do not load `PRIMING.md` for generic formatting, small mechanical edits, or unrelated tooling tasks.

## When to read PLANS.md

Read `PLANS.md` when the task is:

- multi-step;
- risky;
- architectural;
- data-model related;
- migration-related;
- likely to affect several files;
- unclear in blast radius.

Small localized changes can use a concise plan inline.

## Chain-of-thought handling

If the assignment or database mentions `chain_of_thought`, treat it as an auditable judge rationale field. Store concise structured justification, not hidden reasoning.

## Done definition

A task is done only when:

- the requested behavior is implemented or the limitation is clearly reported;
- relevant tests or validation commands were run;
- no unrelated source files were changed;
- generated artifacts are separated from source code;
- README or docs are updated only when behavior or usage changes.
