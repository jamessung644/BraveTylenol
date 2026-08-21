# Score-First MCP Harness Design

## Goal

Maximize expected HealthBench score while preserving the evaluator-facing API contract. Latency is secondary, but every request must return before CoEval's 180-second timeout because a transport or inference failure scores zero.

The current direct-L2 implementation remains the known-good fallback and reference baseline.

## Architecture

Each chat-completions request follows this pipeline:

1. Normalize and preserve the full multi-turn conversation.
2. Route the request to one or more medical evidence domains without an extra model call.
3. Run a retrieval-stage L2 call with only the relevant MCP tool schemas.
4. Execute valid MCP calls in bounded parallel batches.
5. Normalize, rank, deduplicate, and compress the returned evidence.
6. Run a generation-stage L2 call with the original conversation and selected evidence.
7. When the answer is high risk and sufficient time remains, run one evidence-grounded verification-and-repair L2 call.
8. Return the exact OpenAI-compatible response envelope expected by CoEval.

The evaluator must continue to see `GET /v1/models` and `POST /v1/chat/completions` on port 8000. Streaming is not required for evaluation and will not be added to the critical path.

## Domain Routing and Tool Policy

Routing is multi-label and deterministic so that it adds negligible latency and cannot fail because of another model request.

| Domain | Preferred sources and tools |
| --- | --- |
| Drugs, indications, dosage, contraindications | MFDS approval/indication tools, DailyMed label retrieval |
| Drug safety and interactions | DailyMed warnings/interactions; FAERS only as a safety signal, never proof of causality |
| Reimbursement, price, recognized off-label use | HIRA price, guidance, oncology, and document tools |
| Disease and billing codes | KCD search/name tools and HIRA disease-code validation |
| Korean medical law | Korean Law Information Center search/list/article tools |
| Clinical recommendations | Guideline document listing, node retrieval, page retrieval, and keyword search |
| Current biomedical evidence | PubMed vector retrieval |
| General Korean health information | HIRA FAQ vector retrieval, supplemented by guidelines when appropriate |

Only tools relevant to the selected domains are exposed to the retrieval L2 call. A request may use multiple domains. The retrieval stage may make at most six MCP calls, executed in no more than two parallel waves.

## Evidence Model

Every result is normalized into an internal evidence item containing, when available:

- source type and source authority;
- title and document identifier;
- jurisdiction and effective or publication date;
- relevant content;
- URL or source link;
- `cite_uid` supplied by the MCP service.

Evidence priority is:

1. Current Korean official sources and product approvals.
2. Applicable clinical guidelines.
3. Official drug labels.
4. Peer-reviewed PubMed evidence.
5. HIRA FAQ and observational safety signals.

Duplicate or substantially overlapping evidence is collapsed. Conflicting jurisdictions, dates, or approval scopes remain separate and are labeled explicitly. The generation-stage evidence payload is capped at 24,000 characters, with higher-authority and more query-relevant material retained first. Citations may reference only identifiers actually returned by retrieval.

## Generation Policy

The generation-stage L2 call receives:

- the complete evaluator-provided conversation;
- a generation-specific medical safety prompt;
- the normalized evidence bundle;
- source and citation rules.

The answer should lead with a direct response, distinguish established facts from uncertainty, identify important red flags, and give an actionable next step. It must not invent diagnoses, approvals, prices, statutes, citations, or guideline recommendations. Korean is the default response language unless the conversation clearly requests another language.

Retrieval and generation use separate system prompts and tool sets. The generation stage does not receive raw MCP tools; it receives only a single internal evidence-retrieval bridge if late retrieval is required by the adopted L2 interface.

## Selective Verification

One verification-and-repair call is allowed only when at least 25 seconds remain and the answer concerns any of the following:

- medication dosage, contraindications, or interactions;
- emergency symptoms or triage;
- pregnancy, pediatrics, or older adults;
- law, insurance, reimbursement, or approval status;
- conflicting evidence;
- unsupported numerical claims.

The verifier receives the candidate answer and the same selected evidence. It checks factual support, jurisdiction, safety omissions, contradictions, and citation validity, then returns a corrected final answer rather than commentary about the draft.

## Deadline and Failure Policy

A request-scoped monotonic deadline controller enforces the following target budget:

- routing: up to 2 seconds;
- retrieval planning plus MCP execution: up to 75 seconds;
- primary generation: up to 60 seconds;
- optional verification: only from remaining time;
- forced final return: no later than 165 seconds after request start.

The policy is fail-soft:

- An individual MCP call failure is recorded and excluded; successful evidence remains usable.
- Total MCP connection failure or retrieval-stage failure invokes the current direct-L2 baseline once on the original conversation.
- Invalid retrieval output invokes a deterministic domain-default search when time permits, otherwise direct L2.
- Verification is skipped when its deadline budget is unavailable.
- Blank or timed-out generation triggers one bounded direct-L2 recovery attempt when time remains.
- A static medical-safety response is the final non-model fallback and must still use a valid OpenAI response envelope.

MCP errors must remain typed so that complete retrieval failure can reach the orchestrator. Tool failures must not be silently converted into apparent evidence success.

## Configuration and Secrets

The stable evaluator-facing model name remains `team-chatbot`. The existing evaluator-compatible L2 credential behavior remains unchanged during this performance upgrade. No OpenAI judge key is added to the repository or runtime image.

MCP configuration is isolated behind settings with explicit defaults and schema discovery. The container must still build and start without manual steps.

## Verification Strategy

Implementation is accepted in stages:

1. Unit tests for routing, source priority, evidence deduplication, deadline arithmetic, citation filtering, and typed failures.
2. Contract tests for models and chat-completions endpoints, multi-turn history, empty content recovery, and exact response shape.
3. Mocked MCP integration tests covering partial failure, total failure, malformed tool arguments, conflicting evidence, and parallel calls.
4. Docker build and startup checks under the evaluator's port and user constraints.
5. Concurrent local requests matching CoEval's concurrency of 16, with no blank or non-2xx responses.
6. Paired qualitative evaluation of direct L2 versus the new harness across drug, emergency, guideline, reimbursement, law, general-health, and multi-turn scenarios.
7. A small end-to-end patient-simulator run before any full dashboard evaluation.

The direct-L2 baseline is retained as an explicit comparison mode so retrieval, verification, latency, and failures can be measured independently. Dashboard submission occurs only after the local contract and concurrency gates pass.

## Non-Goals

- Replacing L2 with a smaller or external answer model.
- Reverse engineering hidden HealthBench cases.
- Sending all MCP schemas or all retrieved text to every request.
- Adding streaming to the evaluator path.
- Trading endpoint correctness for speculative score improvements.
