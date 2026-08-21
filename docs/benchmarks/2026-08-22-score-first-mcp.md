# Score-first MCP verification record

- Date: 2026-08-22 (Asia/Seoul)
- Commit under test (container application): `c0b70b53b72a674446fe050f5915f2aa2373cbd4`
- Evaluator model: `team-chatbot`
- Image: `brave-tylenol:score-first`
- Runtime containers: `brave-tylenol-direct` on host port 8001 with
  `AGENT_MODE=direct`, and `brave-tylenol-score` on host port 8000.

The Task 8 verification scripts and documents were uncommitted local work at
the time of the live run. The application image itself was rebuilt from the
commit under test. Both named containers were stopped after verification.

## Static verification

| Check | Observed result |
| --- | --- |
| Focused Task 8 tests | 6 passed |
| Final full test suite | 230 passed |
| Ruff | passed |
| `git diff --check` | passed |
| Docker rebuild | passed |
| Runtime credential-file preflight | exists with restrictive permissions |

## 16×16 score-first smoke

| Metric | Observed value |
| --- | ---: |
| Requests | 16 |
| Valid completions | 16 |
| Static fallbacks | 0 |
| Failures | 0 |
| HTTP 200 | 16 |
| Minimum latency | 72.401 s |
| Median latency | 88.309 s |
| Maximum latency | 112.335 s |

## Fixed paired direct-versus-score check

The check used exactly seven fixed scenarios: drug indication/safety, emergency
triage, guideline recommendation, reimbursement, Korean law, general health,
and a two-turn pronoun follow-up. Answer text was written only to the ignored
temporary JSONL artifact and was not displayed during verification.

| Metric | Direct | Score-first |
| --- | ---: | ---: |
| Requests | 7 | 7 |
| Valid completions | 7 | 7 |
| Fallbacks | 0 | 0 |
| Failures | 0 | 0 |
| Completion rate | 1.000 | 1.000 |
| Fallback rate | 0.000 | 0.000 |
| Error rate | 0.000 | 0.000 |
| Minimum latency | 20.189 s | 21.121 s |
| Median latency | 27.572 s | 44.404 s |
| Maximum latency | 38.288 s | 151.544 s |

Fixed scenario-domain coverage was `drug:1`, `drug_safety:1`, `emergency:1`,
`general_health:1`, `guideline:1`, `law:1`, and `reimbursement:1`. The public
completion envelope does not expose MCP telemetry, so actual MCP-tool execution
coverage was not observable from this black-box check and is not claimed here.

The permitted manual rubric is correctness, relevance, safety, actionable next
steps, appropriate uncertainty, and source grounding. No qualitative winner was
recorded: the no-answer-display constraint was honored and no automated or
hidden grader was used. This is a qualitative-review limitation, not evidence
that either path is better on the rubric.

## Official patient-simulator smoke

The final instrumented run used exactly five conversations and no more than
three assistant turns per conversation. The official simulator returned 200 for
14 simulator calls; 11 harness calls returned 200. Three conversations stopped
when the simulator repeated a user turn. There were no status, envelope-shape,
per-harness-latency, retry, or restart failures.

| Metric | Observed value |
| --- | ---: |
| Conversations completed | 5 |
| Assistant turns | 11 |
| Repeated-user stops | 3 |
| Status failures | 0 |
| Shape failures | 0 |
| Harness latency failures (>165 s) | 0 |
| Simulator retries / restarts | 0 / 0 |
| Conversation duration, min / median / max | 44.200 / 175.140 / 275.151 s |
| Harness latency, min / median / max | 28.720 / 32.919 / 124.536 s |

## Security record and limitations

- No OpenAI key, SDK, grader, or judge credential was used.
- The authorized Lunit key file was checked only for existence and restrictive
  permissions; its value was neither displayed nor recorded.
- No headers, prompts, messages, answers, evidence, response bodies, or raw
  exception bodies were printed or committed.
- Paired JSONL remained outside Git at `/tmp/brave-tylenol-paired.jsonl`.
- Live verification did not push, merge, or trigger dashboard evaluation.
