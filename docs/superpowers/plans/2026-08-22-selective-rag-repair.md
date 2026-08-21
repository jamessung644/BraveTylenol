# Selective RAG Single-Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one bounded selective MCP lookup while keeping every request to one L2 generation.

**Architecture:** Extend the zero-dependency `main.py` gateway with deterministic routing and bounded JSON-RPC MCP access. Preserve the 150-second request deadline and current OpenAI response contract.

**Tech Stack:** Python 3.13 standard library, HTTP JSON/SSE, `ThreadingHTTPServer`, Docker.

**Spec:** `docs/superpowers/specs/2026-08-22-selective-rag-repair-design.md`

## Global Constraints

- Keep the default path to exactly one L2 call.
- Use no new runtime dependencies.
- Use at most one MCP call with a 4-second timeout and 3,000-character evidence cap.
- Never make a second L2 call.
- Preserve an end-to-end request deadline of 150 seconds.
- Fail open from MCP to the direct L2 path.
- Do not log credentials or raw conversations.
- Run fast local unit and static checks without live external API calls.

---

### Task 1: Deterministic selective MCP grounding

**Files:**
- Modify: `main.py`
- Test definition (execution skipped by user): `tests/test_selective_grounding.py`

**Interfaces:**
- Produces: `MCPRoute`, `_select_mcp_route(messages)`, `request_mcp_evidence(route, api_key)`.
- Consumes: normalized request messages and the resolved Lunit API key.

- [ ] Add bounded MCP constants and the immutable `MCPRoute` data object.
- [ ] Add strict source-specific route extraction for KCD, MFDS/DailyMed, HIRA, law, guideline, and PubMed requests.
- [ ] Add JSON-RPC request and JSON/SSE result parsing with size and time limits.
- [ ] Inject retrieved evidence as untrusted reference data before the single final L2 call.
- [ ] Confirm by static inspection that failures return to the direct path.
- [ ] Commit the grounding change.

### Task 2: Enforce one L2 generation

**Files:**
- Modify: `main.py`
- Test definition (execution skipped by user): `tests/test_selective_grounding.py`

**Interfaces:**
- Consumes: original messages, optional MCP evidence, API key, shared deadline, and HTTP opener.

- [ ] Remove all conditional answer-repair and second-generation paths.
- [ ] Confirm complex and source-routed requests make exactly one upstream L2 call.
- [ ] Keep MCP failures fail-open to the direct single-L2 path.
- [ ] Commit the single-pass change.

### Task 3: Documentation and urgent deployment

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: the final runtime constants and routing behavior.
- Produces: operator documentation and submission branch SHA.

- [ ] Document the direct-first, one-MCP, one-L2 behavior and hard budgets.
- [ ] Run the local unit suite, Ruff, and `git diff --check` without live API calls.
- [ ] Inspect the changed-file list, Docker copy target, and Git status.
- [ ] Commit all tracked changes.
- [ ] Push the same HEAD to `main`, `lunit/hackathon-submission`, and `candidate/fast-single-pass-50`.
- [ ] Report the exact 40-character SHA and model name `team-chatbot`.
