# Audit Log Contract

PostgreSQL remains the source of truth for persisted evaluation data, including questions, answers, judge models, scores, rationales, prompt links, and timestamps stored in `avaliacoes_juiz`.

Versioned audit logs are static operational evidence for production judge runs. They may complement, but must not replace, database-backed metrics or official evaluation records.

Future dashboard and meta-evaluation work may derive optional metadata from logs:

- run metadata: `run_id`, start/end time, dataset, panel mode, execution strategy, batch size, and model panel;
- operational events: selected answers, skipped evaluations, retry/requeue events, adaptive concurrency changes, and run results;
- evaluation links: `answer_id`, judge model, role, score, and trigger reason for matching log events to persisted evaluations;
- performance and reliability: `latency_ms`, `status_code`, retries, backoff, failures, timeout/rate-limit signals, and provider errors;
- arbitration: arbiter execution, arbiter skip reason, score delta, and trigger reason.

Rules:

- Do not use logs as the source for official scores, rankings, or coverage metrics when those values exist in PostgreSQL.
- Treat logs as read-only evidence and optional enrichment.
- Missing log metadata must not hide or invalidate a persisted evaluation.
- New logs remain ignored by default; only explicitly validated production logs should be versioned.
