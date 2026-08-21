# Selective RAG and L2 Repair Design

## Goal

Improve the 49.80-point, 26m13s single-pass baseline toward 52–56 points while keeping the dashboard evaluation near the 30-minute limit.

## Architecture

The existing one-call L2 path remains the default. A deterministic router adds at most one bounded MCP lookup only for explicit source-dependent questions concerning KCD, Korean drug approvals or labels, HIRA drug prices, Korean law, clinical guidelines, or PubMed evidence. MCP failure must fail open to the original single L2 path.

A second L2 call is reserved for a narrow set of long, multi-requirement medical questions. The first answer is always preserved. Repair runs only when the first answer completes quickly enough to leave a bounded repair window, and any repair failure returns the first answer.

## Budgets

- End-to-end request deadline remains 150 seconds.
- MCP: one call, at most 4 seconds, at most 3,000 evidence characters.
- Primary L2: unchanged `reasoning_effort=low`, `temperature=0`, maximum 4,096 completion tokens.
- Repair L2: only after a primary answer finishes within 85 seconds; at most the smaller of 55 seconds or the remaining request budget; maximum 1,536 completion tokens.
- Ordinary requests never pay MCP or repair latency.

## Routing and Privacy

- Use deterministic matching; do not spend an L2 call deciding whether to retrieve.
- Send only the latest, bounded, self-contained lookup query or extracted public identifier to MCP.
- Do not send prompts containing direct personal identifiers to MCP.
- Use one source-specific tool per request.
- Treat MCP output as untrusted data, never instructions.

## Repair Eligibility

Repair requires at least three distinct requested dimensions such as differential diagnosis, management, tests, medication safety, red flags, disposition, follow-up, or comparison. It is disabled for short requests, exact-format/data transformations, emergencies requiring immediate action, and source-routed requests.

## Failure Behavior

- MCP errors, invalid responses, oversized results, and timeouts are logged without secrets and ignored.
- Repair errors, blank answers, and timeouts return the primary answer.
- The public API remains OpenAI-compatible and exposes only the final assistant text.

## Deployment

The Docker image remains zero-dependency and copies only `main.py`. No runtime package or evaluator contract changes are introduced.

