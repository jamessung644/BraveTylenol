# Architecture

`team-chatbot` is an OpenAI-compatible medical chat service whose user-visible
medical answers are generated and, where appropriate, checked only by official
Lunit L2 (`Lunit/L2-preview`). The score-first path uses the official Lunit MCP
endpoint; it does not use an OpenAI SDK, OpenAI judge credential, hidden
HealthBench data, or an external answer grader.

## Request pipeline

```text
POST /v1/chat/completions
  -> request validation and one 165-second monotonic deadline
  -> resolve the internal Lunit credential (never the evaluator placeholder)
  -> deterministic multi-domain routing from the complete recent conversation
  -> routed MCP retrieval in at most two parallel waves (at most six calls)
  -> authority ranking, exact-duplicate removal, and 24,000-character bound
  -> Lunit L2 grounded generation
  -> selective Lunit L2 verification for high-risk routes when time remains
  -> valid team-chatbot completion envelope
```

The router combines all matching domains and exposes only their allowlisted MCP
tools. Drug, safety, reimbursement, coding, law, emergency, and vulnerable-
population signals can require verification. General health is the bounded
fallback retrieval domain rather than permission to expose every discovered MCP
tool. A self-contained query retains the latest user turn and up to four recent
user/assistant turns in chronological order.

## Time and failure control

Every request receives one absolute 165-second `RequestDeadline`. Retrieval has
a 75-second cap while preserving generation time; grounded generation has a
60-second cap; verification runs only with at least 25 seconds remaining.
Evidence is capped at 24,000 characters and MCP execution is capped at six calls
across two waves.

Expected upstream failures follow a typed, fail-soft path:

1. A typed retrieval failure triggers one direct L2 attempt on the original
   conversation.
2. A grounded-generation failure triggers one direct L2 attempt only when at
   least ten seconds remain.
3. A verifier failure retains the grounded candidate.
4. A failed or unavailable direct L2 call returns the static Korean medical
   safety fallback in a normal HTTP 200 completion envelope.

Malformed evaluator requests and `stream=true` remain validation failures.
Valid requests always return model `team-chatbot`, a nonblank assistant choice,
`finish_reason="stop"`, and nonnegative usage fields.

## Components

| Component | Responsibility |
| --- | --- |
| `app.py` | OpenAI-shaped HTTP contract, request-scoped construction, and envelope recovery |
| `credentials.py` | Shared Lunit credential validation and resolution without logging values |
| `deadline.py` | Monotonic request budget and stage reservations |
| `routing.py` | Deterministic domains, allowlists, and bounded multi-turn retrieval query |
| `retrieval.py` | MCP discovery, routed two-wave planning, typed errors, and evidence collection |
| `evidence.py` | Source authority ranking, exact deduplication, citation filtering, and size bound |
| `generation.py` | Direct and evidence-grounded L2 generation |
| `verification.py` | High-risk L2 answer repair under the remaining request deadline |
| `orchestrator.py` | Typed score-first routing and direct/static recovery order |
| `scripts/concurrent_smoke.py` | Aggregate-only 16×16 public-envelope gate |
| `scripts/paired_quality_check.py` | Fixed seven-scenario direct-versus-score answer capture outside Git |
| `scripts/patient_simulator_smoke.py` | Five-conversation bounded official patient-simulator smoke gate |

## Runtime and privacy boundaries

The container contains only the production application and dependencies, runs
as a non-root user, and serves `0.0.0.0:8000`. The default score-first mode is
`rag` with `https://mcp.hackathon.lunit.io/mcp`; `AGENT_MODE=direct` is retained
for the reference comparison path.

The Lunit credential is resolved internally and never placed in logs, smoke
output, Docker build context, or version control. Authorization values, user
messages, patient-simulator turns, generated answers, prompts, raw MCP evidence,
raw response bodies, and raw exception bodies are not logged. Operational output
is restricted to aggregate status, latency, fallback, and shape metrics. Paired
answers are written only to a caller-selected path outside the repository (or a
Git-ignored path), with restrictive file permissions.
