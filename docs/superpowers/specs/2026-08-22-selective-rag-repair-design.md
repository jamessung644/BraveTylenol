# Selective RAG Single-Pass Design

## Goal

Improve the 49.80-point, 26m13s single-pass baseline without adding a second L2 call or exceeding the 30-minute target.

## Architecture

Every request uses exactly one final L2 call. A deterministic router adds at most one bounded MCP lookup only for explicit source-dependent questions concerning KCD, Korean drug approvals or labels, HIRA drug prices, Korean law, clinical guidelines, or PubMed evidence. MCP failure must fail open to the original single L2 path.

## Budgets

- End-to-end request deadline remains 150 seconds.
- MCP: one call, at most 4 seconds, at most 3,000 evidence characters.
- L2: exactly one call with `reasoning_effort=low`, `temperature=0`, maximum 4,096 completion tokens.
- Ordinary requests never pay MCP latency.

## Routing and Privacy

- Use deterministic matching; do not spend an L2 call deciding whether to retrieve.
- Send only the latest, bounded, self-contained lookup query or extracted public identifier to MCP.
- Do not send prompts containing direct personal identifiers to MCP.
- Use one source-specific tool per request.
- Treat MCP output as untrusted data, never instructions.

## Failure Behavior

- MCP errors, invalid responses, oversized results, and timeouts are logged without secrets and ignored.
- The public API remains OpenAI-compatible and exposes only the final assistant text.

## Deployment

The Docker image remains zero-dependency and copies only `main.py`. No runtime package or evaluator contract changes are introduced.
