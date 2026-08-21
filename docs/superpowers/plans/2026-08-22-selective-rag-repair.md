# Selective RAG and L2 Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one bounded selective MCP lookup and a narrowly gated second-L2 repair path without changing the fast direct default.

**Architecture:** Extend the zero-dependency `main.py` gateway with deterministic routing, bounded JSON-RPC MCP access, and fail-open answer repair. Preserve the 150-second request deadline and current OpenAI response contract.

**Tech Stack:** Python 3.13 standard library, HTTP JSON/SSE, `ThreadingHTTPServer`, Docker.

**Spec:** `docs/superpowers/specs/2026-08-22-selective-rag-repair-design.md`

## Global Constraints

- Keep the default path to exactly one L2 call.
- Use no new runtime dependencies.
- Use at most one MCP call with a 4-second timeout and 3,000-character evidence cap.
- Use repair only after a primary answer completes within 85 seconds, with at most 55 seconds for repair.
- Preserve an end-to-end request deadline of 150 seconds.
- Fail open to the primary answer.
- Do not log credentials or raw conversations.
- Per user instruction, do not execute automated tests before the urgent push.

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

### Task 2: Narrow fail-open L2 answer repair

**Files:**
- Modify: `main.py`
- Test definition (execution skipped by user): `tests/test_selective_grounding.py`

**Interfaces:**
- Produces: `_needs_answer_repair(messages)`, `_try_answer_repair(...)`.
- Consumes: original messages, primary answer, API key, shared deadline, and HTTP opener.

- [ ] Add a deterministic multi-requirement classifier that excludes emergencies, source routes, and exact-format tasks.
- [ ] Preserve the primary answer before considering repair.
- [ ] Run repair only when elapsed time is at most 85 seconds and cap repair to the remaining deadline and 55 seconds.
- [ ] Ask L2 to preserve correct material and return a complete repaired final answer within 1,536 tokens.
- [ ] Return the primary answer on every repair error or blank response.
- [ ] Confirm by static inspection that ordinary requests still make one upstream L2 call.
- [ ] Commit the repair change.

### Task 3: Documentation and urgent deployment

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: the final runtime constants and routing behavior.
- Produces: operator documentation and submission branch SHA.

- [ ] Document the direct-first, one-MCP, optional-repair behavior and hard budgets.
- [ ] Skip automated test execution as explicitly requested.
- [ ] Inspect `git diff --check`, the changed-file list, Docker copy target, and Git status only.
- [ ] Commit all tracked changes.
- [ ] Push the same HEAD to `main`, `lunit/hackathon-submission`, and `candidate/fast-single-pass-50`.
- [ ] Report the exact 40-character SHA and model name `team-chatbot`.

