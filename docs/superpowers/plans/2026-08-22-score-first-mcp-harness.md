# Score-First MCP Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a score-first Lunit L2 harness that routes each medical conversation to authoritative MCP sources, generates an evidence-grounded answer, selectively verifies high-risk answers, and still returns a valid CoEval response before 165 seconds.

**Architecture:** Keep the current direct-L2 server as a reference and last-resort response source, but make the tested FastAPI `lunit_hackathon` stack the container entry point. A request-scoped deadline controls deterministic domain routing, bounded two-wave MCP retrieval, evidence ranking/compression, final L2 generation, and optional L2 verification. Every expected upstream failure is converted into a valid OpenAI-compatible HTTP 200 completion, with direct L2 attempted once before the static fallback.

**Tech Stack:** Python 3.13, FastAPI 0.141.1, Pydantic 2.13.4, httpx 0.28.1, MCP SDK 2.0.0, pytest, Docker

**Spec:** `docs/plans/2026-08-22-score-first-mcp-harness-design.md`

## Global Constraints

- Expose `GET /v1/models` and `POST /v1/chat/completions` on `0.0.0.0:8000` with evaluator model ID `team-chatbot`.
- Preserve the complete evaluator-provided multi-turn conversation in order.
- Use Lunit L2 (`Lunit/L2-preview`) as the sole author and verifier of user-facing medical answers.
- Default MCP endpoint is `https://mcp.hackathon.lunit.io/mcp` and authentication uses the resolved Lunit credential.
- Limit retrieval to six MCP calls in at most two parallel waves.
- Bound selected evidence to 24,000 characters.
- Force a final valid response no later than 165 seconds after request start.
- Never add an OpenAI judge credential to the repository or image.
- Never log Authorization headers, API keys, medical conversation text, or raw evidence.
- Keep `main.py` and `tests/test_baseline_server.py` as the known-good direct-L2 regression baseline until the new container passes all acceptance gates.

## File Map

- `lunit_hackathon/credentials.py`: one credential-resolution policy shared by L2 and MCP.
- `lunit_hackathon/deadline.py`: request-scoped monotonic deadline and stage budget calculations.
- `lunit_hackathon/routing.py`: deterministic multi-label medical-domain routing and MCP allowlists.
- `lunit_hackathon/evidence.py`: authority ranking, deduplication, citation filtering, and 24K compression.
- `lunit_hackathon/config.py`: exact score-first defaults and configurable budgets.
- `lunit_hackathon/schemas.py`: domain, route, evidence, and verification data contracts.
- `lunit_hackathon/retrieval.py`: routed tool exposure, bounded planner waves, typed partial/total failures.
- `lunit_hackathon/generation.py`: direct and evidence-grounded L2 generation only.
- `lunit_hackathon/verification.py`: high-risk answer verification and repair.
- `lunit_hackathon/orchestrator.py`: full request pipeline and direct-L2 recovery.
- `lunit_hackathon/l2_client.py`: per-call token and timeout overrides under one request deadline.
- `lunit_hackathon/prompts.py`: separate retrieval, generation, and verification prompts.
- `app.py`: exact evaluator contract, credential injection, fallback envelope, and request wiring.
- `Dockerfile`: production dependencies and FastAPI entry point.
- `tests/`: focused unit, integration, contract, and concurrency coverage.

---

### Task 1: Shared Credentials and Request Deadline

**Files:**
- Create: `lunit_hackathon/credentials.py`
- Create: `lunit_hackathon/deadline.py`
- Modify: `lunit_hackathon/config.py`
- Modify: `tests/test_config.py`
- Create: `tests/test_credentials.py`
- Create: `tests/test_deadline.py`

**Interfaces:**
- Produces: `resolve_lunit_api_key(authorization: str | None, environment_key: str | None) -> str`.
- Produces: `RequestDeadline.start(total_seconds: float = 165.0, clock: Callable[[], float] = perf_counter) -> RequestDeadline`.
- Produces: `RequestDeadline.remaining() -> float`, `RequestDeadline.stage_timeout(cap_seconds: float, reserve_seconds: float = 0.0) -> float`, and `RequestDeadline.can_spend(minimum_seconds: float) -> bool`.
- Produces: `Settings` defaults `agent_mode="rag"`, `mcp_url="https://mcp.hackathon.lunit.io/mcp"`, `request_timeout_seconds=165.0`, `retrieval_timeout_seconds=75.0`, `generation_timeout_seconds=60.0`, `verification_minimum_seconds=25.0`, `max_mcp_calls=6`, `max_evidence_chars=24_000`, `max_completion_tokens=4_096`, `retrieval_reasoning_effort="medium"`, `generation_reasoning_effort="high"`, and `verification_reasoning_effort="medium"`.

- [ ] **Step 1: Write failing credential and deadline tests**

```python
def test_resolver_rejects_evaluator_dummy_and_uses_embedded_key():
    value = resolve_lunit_api_key("Bearer evaluator-token", None)
    assert value.startswith("lunit_")


def test_environment_key_precedes_valid_bearer():
    assert resolve_lunit_api_key("Bearer lunit_request", "lunit_environment") == (
        "lunit_environment"
    )


def test_stage_timeout_preserves_reserve():
    now = iter([10.0, 30.0])
    deadline = RequestDeadline.start(total_seconds=100.0, clock=lambda: next(now))
    assert deadline.stage_timeout(75.0, reserve_seconds=25.0) == 55.0
```

- [ ] **Step 2: Run the focused tests and confirm missing modules/defaults fail**

Run: `pytest tests/test_credentials.py tests/test_deadline.py tests/test_config.py -q`

Expected: FAIL because `credentials.py`, `deadline.py`, and the score-first settings do not exist.

- [ ] **Step 3: Implement credential validation and deadline arithmetic**

```python
def resolve_lunit_api_key(
    authorization: str | None,
    environment_key: str | None,
) -> str:
    for candidate in (environment_key, _bearer_token(authorization), EMBEDDED_LUNIT_API_KEY):
        if _valid_lunit_key(candidate):
            return candidate.strip()
    raise ConfigurationError("No valid Lunit API credential is available")


@dataclass(frozen=True)
class RequestDeadline:
    expires_at: float
    clock: Callable[[], float]

    @classmethod
    def start(cls, total_seconds: float = 165.0, clock: Callable[[], float] = perf_counter):
        return cls(expires_at=clock() + total_seconds, clock=clock)

    def remaining(self) -> float:
        return max(0.0, self.expires_at - self.clock())

    def stage_timeout(self, cap_seconds: float, reserve_seconds: float = 0.0) -> float:
        return max(0.0, min(cap_seconds, self.remaining() - reserve_seconds))

    def can_spend(self, minimum_seconds: float) -> bool:
        return self.remaining() >= minimum_seconds
```

Move the already-existing embedded credential constant out of `main.py` into `credentials.py`; import it from both paths rather than duplicating or printing it.

- [ ] **Step 4: Add the exact validated settings fields and defaults**

Add positive, upper-bounded Pydantic fields for the three stage budgets, the three stage-specific reasoning-effort fields, and update the existing defaults listed in the Interfaces block. Keep environment aliases for every value.

- [ ] **Step 5: Run focused tests and lint**

Run: `pytest tests/test_credentials.py tests/test_deadline.py tests/test_config.py tests/test_baseline_server.py -q && ruff check lunit_hackathon/credentials.py lunit_hackathon/deadline.py lunit_hackathon/config.py main.py tests/test_credentials.py tests/test_deadline.py tests/test_config.py`

Expected: PASS, including the unchanged direct-L2 baseline tests.

- [ ] **Step 6: Commit**

```bash
git add main.py lunit_hackathon/credentials.py lunit_hackathon/deadline.py lunit_hackathon/config.py tests/test_credentials.py tests/test_deadline.py tests/test_config.py
git commit -m "feat: add score-first request budgets"
```

### Task 2: Deterministic Multi-Domain Router

**Files:**
- Create: `lunit_hackathon/routing.py`
- Modify: `lunit_hackathon/schemas.py`
- Create: `tests/test_routing.py`

**Interfaces:**
- Produces: `MedicalDomain` string enum values `drug`, `drug_safety`, `reimbursement`, `coding`, `law`, `guideline`, `research`, `general_health`, `emergency`, and `vulnerable_population`.
- Produces: `RouteDecision(domains: frozenset[MedicalDomain], tool_names: tuple[str, ...], retrieval_required: bool, verification_required: bool)`.
- Produces: `route_messages(messages: Sequence[ChatMessage]) -> RouteDecision`.
- Produces: `self_contained_query(messages: Sequence[ChatMessage], maximum_chars: int = 4_000) -> str`.

- [ ] **Step 1: Write table-driven routing tests**

```python
@pytest.mark.parametrize(
    ("question", "domains", "tools", "verify"),
    [
        ("임신 중 아세트아미노펜 용량과 금기는?", {"drug", "vulnerable_population"}, {"openapi_mfds_get_drug_indication", "adr_retrieve_drug_info"}, True),
        ("이 항암요법은 건강보험 급여인가요?", {"reimbursement"}, {"hira_updates_search"}, True),
        ("KCD-9 I10 코드가 청구에 유효한가요?", {"coding"}, {"kcd_get_name", "openapi_hira_disease_check_code"}, True),
        ("최신 고혈압 가이드라인과 논문 근거는?", {"guideline", "research"}, {"index_get_relevant_nodes", "rag_vector_query"}, False),
    ],
)
def test_route_messages(question, domains, tools, verify):
    route = route_messages([ChatMessage(role="user", content=question)])
    assert domains <= {domain.value for domain in route.domains}
    assert tools <= set(route.tool_names)
    assert route.verification_required is verify
```

Add a multi-turn test proving `self_contained_query` contains both the prior subject and the latest pronoun-based follow-up without changing message order.

- [ ] **Step 2: Run tests and confirm they fail**

Run: `pytest tests/test_routing.py -q`

Expected: FAIL because the routing interfaces are undefined.

- [ ] **Step 3: Implement domain keyword maps and exact tool allowlists**

Use immutable mappings. The router must union all matching domains, add `general_health` only when no specific evidence domain matches, and never return all discovered MCP tools as a fallback. Use these allowlists:

```python
DOMAIN_TOOLS = {
    MedicalDomain.DRUG: (
        "openapi_mfds_check_drug_permission",
        "openapi_mfds_find_drugs_by_ingredient",
        "openapi_mfds_get_drug_indication",
        "adr_retrieve_drug_info",
    ),
    MedicalDomain.DRUG_SAFETY: ("adr_retrieve_drug_info", "rag_sql_query"),
    MedicalDomain.REIMBURSEMENT: ("hira_updates_search", "openapi_hira_get_drug_price"),
    MedicalDomain.CODING: ("kcd_search_codes", "kcd_get_name", "openapi_hira_disease_check_code"),
    MedicalDomain.LAW: ("openapi_law_search", "openapi_law_list_articles", "openapi_law_get_article"),
    MedicalDomain.GUIDELINE: (
        "index_list_documents",
        "index_get_relevant_nodes",
        "index_get_page_content",
        "index_keyword_search",
    ),
    MedicalDomain.RESEARCH: ("rag_vector_query",),
    MedicalDomain.GENERAL_HEALTH: ("rag_vector_query", "index_get_relevant_nodes"),
}
```

Emergency and vulnerable-population domains affect verification and prompt policy but do not add tools by themselves.

- [ ] **Step 4: Implement bounded multi-turn query construction**

Include up to the last four user/assistant messages, prefix each with its role, preserve the latest user message in full when possible, and truncate from the oldest context first until the UTF-8-safe Python string is at most 4,000 characters.

- [ ] **Step 5: Run tests and lint**

Run: `pytest tests/test_routing.py -q && ruff check lunit_hackathon/routing.py lunit_hackathon/schemas.py tests/test_routing.py`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add lunit_hackathon/routing.py lunit_hackathon/schemas.py tests/test_routing.py
git commit -m "feat: route medical questions to MCP domains"
```

### Task 3: Authority-Aware Evidence Compression

**Files:**
- Create: `lunit_hackathon/evidence.py`
- Modify: `lunit_hackathon/schemas.py`
- Create: `tests/test_evidence.py`

**Interfaces:**
- Extends: `EvidenceItem` with `authority_rank: int = Field(ge=0, le=5)`, `title: str | None = None`, `url: str | None = None`, `jurisdiction: str | None = None`, and `effective_date: str | None = None` while keeping `cite_uid`, `source_tool`, `relevance_score`, and `content`.
- Produces: `authority_rank(source_tool: str) -> int`.
- Produces: `extract_evidence_metadata(content: str) -> dict[str, str | None]`.
- Produces: `rank_deduplicate_and_bound(items: Sequence[EvidenceItem], maximum_chars: int = 24_000) -> list[EvidenceItem]`.
- Produces: `valid_cite_uids(items: Sequence[EvidenceItem]) -> frozenset[str]`.

- [ ] **Step 1: Write failing authority, deduplication, and bound tests**

```python
def evidence(cite_uid: str, source_tool: str, relevance: float, content: str) -> EvidenceItem:
    return EvidenceItem(
        cite_uid=cite_uid,
        source_tool=source_tool,
        relevance_score=relevance,
        authority_rank=authority_rank(source_tool),
        content=content,
    )


def test_korean_official_source_outranks_pubmed_at_equal_relevance():
    official = evidence("mfds:1", "openapi_mfds_get_drug_indication", 0.8, "허가사항")
    paper = evidence("pmid:1", "rag_vector_query", 0.8, "논문")
    assert rank_deduplicate_and_bound([paper, official])[0].cite_uid == "mfds:1"


def test_duplicate_content_keeps_higher_authority_item():
    low = evidence("faq:1", "rag_vector_query", 0.9, "동일한 핵심 근거")
    high = evidence("hira:1", "hira_updates_search", 0.7, "동일한 핵심 근거")
    assert [item.cite_uid for item in rank_deduplicate_and_bound([low, high])] == ["hira:1"]


def test_bound_never_splits_an_item_or_exceeds_limit():
    result = rank_deduplicate_and_bound(
        [evidence("a", "index_get_page_content", 1.0, "가" * 700), evidence("b", "rag_vector_query", 0.9, "나" * 700)],
        maximum_chars=1_000,
    )
    assert sum(len(item.content) for item in result) <= 1_000
    assert [item.cite_uid for item in result] == ["a"]


def test_metadata_extraction_keeps_jurisdiction_date_and_source_link():
    metadata = extract_evidence_metadata(
        '{"title":"허가사항","jurisdiction":"KR","effective_date":"2026-01-01","source_link":"https://example.test/item"}'
    )
    assert metadata == {
        "title": "허가사항",
        "url": "https://example.test/item",
        "jurisdiction": "KR",
        "effective_date": "2026-01-01",
    }
```

- [ ] **Step 2: Run tests and confirm they fail**

Run: `pytest tests/test_evidence.py -q`

Expected: FAIL because evidence ranking is not implemented.

- [ ] **Step 3: Implement source authority tiers and stable sorting**

Assign rank 5 to Korean regulator, HIRA, law, and KCD tools; rank 4 to guideline index tools and DailyMed; rank 3 to PubMed vector results; rank 2 to HIRA FAQ; rank 1 to FAERS safety-signal rows; and rank 0 to unknown tools. Sort by `(authority_rank, relevance_score, -original_index)` descending.

- [ ] **Step 4: Implement conservative deduplication and bounding**

Normalize whitespace and punctuation, case-fold, and hash the first 1,000 normalized characters. Collapse exact normalized duplicates only; do not merge merely similar claims. Add whole items until the character budget is exhausted. Reject blank `cite_uid` values from `valid_cite_uids`. Parse JSON recursively to copy only string values from the keys `title`, `url`, `source_link`, `jurisdiction`, `effective_date`, `publication_date`, and `date`; do not infer missing metadata from free text.

- [ ] **Step 5: Run tests and lint**

Run: `pytest tests/test_evidence.py -q && ruff check lunit_hackathon/evidence.py lunit_hackathon/schemas.py tests/test_evidence.py`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add lunit_hackathon/evidence.py lunit_hackathon/schemas.py tests/test_evidence.py
git commit -m "feat: rank and compress medical evidence"
```

### Task 4: Routed Two-Wave MCP Retrieval

**Files:**
- Modify: `lunit_hackathon/retrieval.py`
- Modify: `lunit_hackathon/prompts.py`
- Modify: `tests/test_retrieval.py`

**Interfaces:**
- Changes: `RetrievalEngine.retrieve(query: str, route: RouteDecision, deadline: RequestDeadline) -> RetrievalResult`.
- Consumes: `route.tool_names`, `deadline.stage_timeout(...)`, and `rank_deduplicate_and_bound(...)`.
- Produces: a `RetrievalResult` containing only discovered, routed tools and actual MCP-returned citation IDs.
- Preserves: `RetrievalError.code` for total connection, discovery, schema, planner, and all-calls-failed errors.

- [ ] **Step 1: Add failing routed-tool and no-all-tools tests**

```python
async def test_retrieval_exposes_only_routed_discovered_tools(monkeypatch):
    route = RouteDecision(
        domains=frozenset({MedicalDomain.DRUG}),
        tool_names=("openapi_mfds_get_drug_indication", "adr_retrieve_drug_info"),
        retrieval_required=True,
        verification_required=True,
    )
    planner_completion = L2Completion(tool_calls=[call("c1", "openapi_mfds_get_drug_indication", {"query": "타이레놀"})])
    finalization_completion = L2Completion(tool_calls=[call("f1", "finalize_retrieval", {"status": "sufficient", "items": [{"cite_uid": "mfds:1", "relevance_score": 1.0}], "note": ""})])
    schema = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
    discovered_tools = [
        MCPTool(name="openapi_mfds_get_drug_indication", input_schema=schema),
        MCPTool(name="adr_retrieve_drug_info", input_schema=schema),
        MCPTool(name="openapi_law_search", input_schema=schema),
    ]
    tool_results = {"openapi_mfds_get_drug_indication": MCPCallResult(content='{"cite_uid":"mfds:1"}')}
    l2 = ScriptedL2([planner_completion, finalization_completion])
    mcp = FakeMCP(discovered_tools, tool_results)
    engine = RetrievalEngine(l2, mcp, settings(monkeypatch))
    deadline = RequestDeadline.start(total_seconds=165.0)
    result = await engine.retrieve("타이레놀 적응증", route, deadline)
    exposed = {tool["function"]["name"] for tool in l2.calls[0]["tools"]}
    assert exposed == {"openapi_mfds_get_drug_indication", "adr_retrieve_drug_info", "finalize_retrieval"}


async def test_unknown_route_never_exposes_every_discovered_tool(monkeypatch):
    route = route_messages([ChatMessage(role="user", content="두통이 있어요")])
    planner_completion = L2Completion(tool_calls=[call("c1", "rag_vector_query", {"query": "두통"})])
    finalization_completion = L2Completion(tool_calls=[call("f1", "finalize_retrieval", {"status": "no_evidence", "items": [], "note": ""})])
    schema = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
    discovered_tools = [
        MCPTool(name="rag_vector_query", input_schema=schema),
        MCPTool(name="index_get_relevant_nodes", input_schema=schema),
        MCPTool(name="openapi_law_search", input_schema=schema),
    ]
    tool_results = {"rag_vector_query": MCPCallResult(content="{}")}
    l2 = ScriptedL2([planner_completion, finalization_completion])
    mcp = FakeMCP(discovered_tools, tool_results)
    engine = RetrievalEngine(l2, mcp, settings(monkeypatch))
    await engine.retrieve("두통이 있어요", route, RequestDeadline.start())
    exposed = {tool["function"]["name"] for tool in l2.calls[0]["tools"]}
    assert "openapi_law_search" not in exposed
```

- [ ] **Step 2: Add failing partial-versus-total failure tests**

Test one successful and one failed parallel MCP call returns partial evidence. Test every call raising `RetrievalError` raises `RetrievalError(code="mcp_all_calls_failed")` to the orchestrator rather than returning tool-error text as successful retrieval.

- [ ] **Step 3: Add failing two-wave and six-call budget tests**

Script planner completions for three calls in wave one, two calls in wave two, and finalization on the third planner turn. Assert calls within each wave overlap using `asyncio.Event`, total calls equal five, planner turns equal three, and a seventh proposed call is rejected without execution.

- [ ] **Step 4: Run retrieval tests and confirm the new cases fail**

Run: `pytest tests/test_retrieval.py -q`

Expected: FAIL on the new signature, routing, failure propagation, and call budget.

- [ ] **Step 5: Replace `_select_tools` with route intersection**

Filter discovery by `route.tool_names`; preserve discovery order; add `FINALIZE_RETRIEVAL_TOOL`; raise `RetrievalError(code="mcp_no_routed_tools")` when no routed tool is available. Never use `selected or discovered`.

- [ ] **Step 6: Bound planner execution to two waves and one finalization turn**

Use `max_planner_turns = 3`. Count non-finalization calls against `min(settings.max_mcp_calls, 6)`. Execute valid calls in each turn with `asyncio.gather(..., return_exceptions=True)`, append structured errors for individual failures, and raise `mcp_all_calls_failed` only when the batch produced no usable result and no earlier candidates exist.

If a planner turn returns no calls or malformed calls, force the first available routed tool on one recovery planner call by passing only that discovered schema and `tool_choice={"type":"function","function":{"name": preferred_name}}`. The L2 planner still constructs schema-valid arguments; Python chooses the deterministic tool.

- [ ] **Step 7: Apply evidence ranking and the 75-second stage budget**

Wrap retrieval in `asyncio.timeout(deadline.stage_timeout(settings.retrieval_timeout_seconds, reserve_seconds=settings.generation_timeout_seconds))`. Convert a zero or expired timeout to `RetrievalError(code="retrieval_deadline_exhausted")`. Pass finalized items through `rank_deduplicate_and_bound(..., settings.max_evidence_chars)`.

- [ ] **Step 8: Run retrieval, MCP, and orchestrator regression tests**

Run: `pytest tests/test_retrieval.py tests/test_mcp_client.py tests/test_orchestrator.py -q && ruff check lunit_hackathon/retrieval.py lunit_hackathon/prompts.py tests/test_retrieval.py`

Expected: PASS. The old test that expected a swallowed tool exception must be replaced by the explicit partial/total semantics above.

- [ ] **Step 9: Commit**

```bash
git add lunit_hackathon/retrieval.py lunit_hackathon/prompts.py tests/test_retrieval.py
git commit -m "feat: add routed two-wave MCP retrieval"
```

### Task 5: Grounded Generation and Selective Verification

**Files:**
- Create: `lunit_hackathon/verification.py`
- Modify: `lunit_hackathon/generation.py`
- Modify: `lunit_hackathon/l2_client.py`
- Modify: `lunit_hackathon/prompts.py`
- Modify: `tests/test_generation.py`
- Modify: `tests/test_l2_client.py`
- Create: `tests/test_verification.py`

**Interfaces:**
- Produces: `GenerationEngine.grounded_answer(messages: Sequence[ChatMessage], retrieval: RetrievalResult, deadline: RequestDeadline) -> str`.
- Preserves: `GenerationEngine.direct_answer(messages: Sequence[ChatMessage], deadline: RequestDeadline) -> str`.
- Produces: `AnswerVerifier.verify(messages: Sequence[ChatMessage], candidate: str, retrieval: RetrievalResult, deadline: RequestDeadline) -> str`.
- Changes: `L2Client.complete(..., max_tokens: int | None = None, timeout_seconds: float | None = None, reasoning_effort: Literal["low", "medium", "high"] | None = None) -> L2Completion`.

- [ ] **Step 1: Write failing grounded-generation tests**

Assert `grounded_answer` makes one L2 call with the original conversation, serialized selected evidence, and no raw MCP tools. Assert it preserves at least one relevant actual `cite_uid`, rejects invented identifiers, and uses a maximum of 4,096 tokens with a 60-second stage timeout.

- [ ] **Step 2: Write failing verifier tests**

```python
async def test_verifier_returns_repaired_answer_for_high_risk_drug_claim():
    l2 = ScriptedL2([L2Completion(content="수정된 근거 기반 답변")])
    verifier = AnswerVerifier(l2, Settings(_env_file=None))
    messages = [ChatMessage(role="user", content="이 약을 두 배 먹어도 되나요?")]
    retrieval = RetrievalResult(status="partial", items=[], note="근거 부족")
    answer = await verifier.verify(
        messages,
        "근거 없는 2배 복용 권고",
        retrieval,
        RequestDeadline.start(),
    )
    assert answer == "수정된 근거 기반 답변"


async def test_verifier_is_not_called_without_25_seconds_remaining():
    l2 = ScriptedL2([])
    verifier = AnswerVerifier(l2, Settings(_env_file=None))
    messages = [ChatMessage(role="user", content="용량을 늘려도 되나요?")]
    retrieval = RetrievalResult(status="partial", items=[], note="근거 부족")
    candidate = "복용량을 임의로 늘리지 마세요."
    clock = iter([0.0, 140.1, 140.1])
    deadline = RequestDeadline.start(total_seconds=165.0, clock=lambda: next(clock))
    assert await verifier.verify(messages, candidate, retrieval, deadline) == candidate
    assert l2.calls == []
```

- [ ] **Step 3: Run generation, verifier, and client tests to confirm failure**

Run: `pytest tests/test_generation.py tests/test_verification.py tests/test_l2_client.py -q`

Expected: FAIL because the new signatures and verifier are absent.

- [ ] **Step 4: Add per-call caps without resetting the absolute L2 deadline**

In `L2Client.complete`, choose `payload["max_tokens"] = min(max_tokens or settings.max_completion_tokens, settings.max_completion_tokens)` and use the explicit stage reasoning effort when supplied. In `_post`, set the HTTP timeout to the smaller of the client's existing absolute request deadline and the supplied call timeout. Retries and blank-completion recovery must share that same smaller deadline.

- [ ] **Step 5: Simplify generation into direct and grounded entry points**

Remove the preliminary `retrieve_relevant_content` decision call from the score-first path. `grounded_answer` must add `MEDICAL_GENERATION_SYSTEM_PROMPT`, original messages, one system-delimited untrusted evidence block, and `_grounding_instruction(retrieval)`, then call L2 once without tools using `settings.generation_reasoning_effort`. The prompt must identify FAERS rows as observational safety signals that do not establish causality and require explicit jurisdiction/date distinctions when metadata differs. Keep protocol-leak and blank-completion correction bounded to one recovery call only when the deadline can spend at least 10 seconds.

- [ ] **Step 6: Implement verification prompt and repair**

Add `MEDICAL_VERIFICATION_SYSTEM_PROMPT` instructing L2 to return only a corrected user-facing answer, preserve supported advice, remove unsupported numbers, validate jurisdiction and citations, and put emergency action first. The verifier must use the candidate and selected evidence as untrusted quoted data, call L2 once with at most 2,048 tokens, `settings.verification_reasoning_effort`, and `timeout_seconds=min(25.0, deadline.remaining() - 2.0)`, and return the candidate on any expected upstream failure.

- [ ] **Step 7: Run focused tests and lint**

Run: `pytest tests/test_generation.py tests/test_verification.py tests/test_l2_client.py -q && ruff check lunit_hackathon/generation.py lunit_hackathon/verification.py lunit_hackathon/l2_client.py lunit_hackathon/prompts.py tests/test_generation.py tests/test_verification.py tests/test_l2_client.py`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add lunit_hackathon/generation.py lunit_hackathon/verification.py lunit_hackathon/l2_client.py lunit_hackathon/prompts.py tests/test_generation.py tests/test_verification.py tests/test_l2_client.py
git commit -m "feat: generate and verify grounded answers"
```

### Task 6: Request Orchestration and Always-Valid API Recovery

**Files:**
- Modify: `lunit_hackathon/orchestrator.py`
- Modify: `app.py`
- Modify: `tests/test_orchestrator.py`
- Modify: `tests/test_api.py`
- Create: `tests/test_score_pipeline.py`

**Interfaces:**
- Changes: `ChatOrchestrator.answer(messages: Sequence[ChatMessage], deadline: RequestDeadline) -> str`.
- Consumes: `route_messages`, `self_contained_query`, `RetrievalEngine`, `GenerationEngine`, and `AnswerVerifier`.
- Produces: exactly one final string plus accumulated `TokenUsage` and public `finish_reason="stop"`.
- Preserves: every valid evaluator request receives HTTP 200 with a nonblank `choices[0].message.content`.

- [ ] **Step 1: Write a failing end-to-end orchestrator success test**

Build fakes for routed retrieval, grounded generation, and verification. Assert a drug-safety request executes them in that order, passes the same full messages to generation and verification, and returns the verifier's repaired text.

- [ ] **Step 2: Write failing fail-soft path tests**

Cover these exact cases:

- retrieval raises `RetrievalError`: call direct L2 once on the original messages;
- one tool fails but retrieval has evidence: continue to grounded generation;
- grounded generation times out with at least 10 seconds left: call direct L2 once;
- verifier fails: return the grounded candidate;
- direct L2 fails: return the static Korean medical-safety content;
- deadline has fewer than 25 seconds after generation: skip verifier.

- [ ] **Step 3: Write failing API-envelope recovery tests**

Assert configuration, MCP, L2 HTTP, malformed response, and timeout exceptions never escape `POST /v1/chat/completions`. Every case must return status 200, model `team-chatbot`, `finish_reason="stop"`, nonblank content, and nonnegative integer usage fields. Keep invalid JSON and empty messages behavior compatible with the existing contract tests.

- [ ] **Step 4: Run the new tests and confirm failure**

Run: `pytest tests/test_orchestrator.py tests/test_api.py tests/test_score_pipeline.py -q`

Expected: FAIL because the pipeline and always-valid recovery are not wired.

- [ ] **Step 5: Implement the score-first orchestrator**

For retrieval-required routes, create the self-contained query, retrieve evidence, generate the grounded candidate, and verify only when `route.verification_required` and `deadline.can_spend(settings.verification_minimum_seconds)`. For non-retrieval routes, generate directly. Catch only typed retrieval failures for the normal direct fallback; catch expected L2 failures around generation and direct recovery. Do not treat programmer errors as successful evidence.

- [ ] **Step 6: Wire request-scoped objects in `app.py`**

Resolve a valid Lunit credential using `resolve_lunit_api_key`, create one `RequestDeadline` per request, and build request-scoped L2/retrieval/generation/verifier/orchestrator objects over the lifespan-owned `httpx.AsyncClient`. Remove the current branch that trusts any evaluator bearer token as the upstream Lunit key.

- [ ] **Step 7: Convert expected terminal failures to a completion payload**

Replace 502/503/504 responses for expected upstream failures with the existing Korean safety fallback in a normal `ChatCompletionResponse`. Continue returning validation errors for malformed evaluator requests. Sanitize logs to request ID, path, status, duration, route domain names, call counts, and error codes only.

- [ ] **Step 8: Run API, pipeline, and regression tests**

Run: `pytest tests/test_orchestrator.py tests/test_api.py tests/test_score_pipeline.py tests/test_baseline_server.py -q && ruff check app.py lunit_hackathon/orchestrator.py tests/test_orchestrator.py tests/test_api.py tests/test_score_pipeline.py`

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add app.py lunit_hackathon/orchestrator.py tests/test_orchestrator.py tests/test_api.py tests/test_score_pipeline.py
git commit -m "feat: orchestrate score-first medical answers"
```

### Task 7: Production Container and CoEval-Concurrency Gate

**Files:**
- Modify: `Dockerfile`
- Create: `.dockerignore`
- Modify: `requirements.txt`
- Create: `tests/test_container_contract.py`
- Create: `scripts/concurrent_smoke.py`
- Modify: `README.md`

**Interfaces:**
- Produces: a container that starts `uvicorn app:app --host 0.0.0.0 --port 8000` as non-root.
- Produces: `python scripts/concurrent_smoke.py --base-url http://127.0.0.1:8000 --requests 16 --concurrency 16` with nonzero exit on any non-200, blank content, malformed envelope, or request above 165 seconds.

- [ ] **Step 1: Write static container contract tests**

Assert Dockerfile uses Python 3.13 slim, installs only `requirements.txt`, copies `app.py` and `lunit_hackathon`, exposes 8000, runs as non-root, and starts Uvicorn on `0.0.0.0:8000`. Assert it does not copy `.env`, tests, Git metadata, dashboard credentials, or an OpenAI key.

- [ ] **Step 2: Run the static test and confirm it fails**

Run: `pytest tests/test_container_contract.py -q`

Expected: FAIL because the Dockerfile still copies only `main.py`.

- [ ] **Step 3: Update the production image**

Use this execution shape:

```dockerfile
FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY app.py /app/app.py
COPY lunit_hackathon /app/lunit_hackathon
USER 65532:65532
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).read()"]
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 4: Implement the 16-request smoke checker**

Use only the Python standard library plus `concurrent.futures.ThreadPoolExecutor`. Send the same valid OpenAI-shaped Korean medical request 16 times, measure wall time per response, validate the exact model/choices/message/usage shape, and print only aggregate success count, fallback count, minimum/median/maximum latency, and HTTP status counts.

- [ ] **Step 5: Build and run the container contract locally**

Run:

```bash
pytest tests/test_container_contract.py -q
docker build -t brave-tylenol:score-first .
docker run --rm -d --name brave-tylenol-score -p 8000:8000 brave-tylenol:score-first
python scripts/concurrent_smoke.py --base-url http://127.0.0.1:8000 --requests 16 --concurrency 16
docker stop brave-tylenol-score
```

Expected: image builds within five minutes; health and models endpoints return 200; all 16 completions are valid and nonblank; no request exceeds 165 seconds.

- [ ] **Step 6: Document exact runtime and fallback behavior**

Update README with model ID, endpoints, default MCP URL, stage budgets, local Docker commands, fallback order, and the rule that the Lunit credential is resolved internally while no OpenAI judge key is bundled.

- [ ] **Step 7: Commit**

```bash
git add Dockerfile .dockerignore requirements.txt tests/test_container_contract.py scripts/concurrent_smoke.py README.md
git commit -m "build: package score-first MCP submission"
```

### Task 8: Full Verification, Paired Quality Check, and Submission Handoff

**Files:**
- Create: `scripts/paired_quality_check.py`
- Create: `scripts/patient_simulator_smoke.py`
- Create: `docs/benchmarks/2026-08-22-score-first-mcp.md`
- Modify: `docs/architecture.md`

**Interfaces:**
- Produces: paired JSONL output kept outside Git, with one direct and one score-first answer per prompt.
- Produces: a bounded patient-simulator smoke run of five conversations and at most three turns per conversation.
- Produces: a benchmark report containing commit SHA, test counts, Docker results, completion/fallback/error rates, latency distribution, retrieval coverage by domain, and qualitative paired-review outcomes.

- [ ] **Step 1: Implement a fixed, non-hidden seven-domain prompt set**

Include one prompt each for drug indication/safety, emergency triage, guideline recommendation, reimbursement, Korean law, general health, and a two-turn pronoun follow-up. Accept separate `--direct-url` and `--score-url` arguments, redact Authorization values, and write answer text only to a user-specified path ignored by Git.

- [ ] **Step 2: Implement the bounded patient-simulator smoke script**

Read the Lunit credential from a required `--api-key-file` path, never print it, request the first simulator user turn with an empty message array, preserve each complete conversation verbatim, and stop after three assistant turns or when the simulator repeats the same user message. Retry a simulator 502 once and restart a simulator 404 with an empty history. Run exactly five conversations and report only counts, durations, and response-shape failures.

- [ ] **Step 3: Run the complete test and lint suite**

Run: `pytest -q && ruff check .`

Expected: all tests PASS and Ruff reports no errors.

- [ ] **Step 4: Run Docker contract and concurrency verification again**

Run:

```bash
docker build -t brave-tylenol:score-first .
docker run --rm -d --name brave-tylenol-direct -e AGENT_MODE=direct -p 8001:8000 brave-tylenol:score-first
docker run --rm -d --name brave-tylenol-score -p 8000:8000 brave-tylenol:score-first
python scripts/concurrent_smoke.py --base-url http://127.0.0.1:8000 --requests 16 --concurrency 16
python scripts/paired_quality_check.py --direct-url http://127.0.0.1:8001 --score-url http://127.0.0.1:8000 --output /tmp/brave-tylenol-paired.jsonl
python scripts/patient_simulator_smoke.py --harness-url http://127.0.0.1:8000 --api-key-file /tmp/bravetylenol-lunit-key --conversations 5 --max-turns 3
docker stop brave-tylenol-direct brave-tylenol-score
```

Expected: zero malformed or non-200 completions, zero blank answers, and all requests within 165 seconds.

- [ ] **Step 5: Review paired answers against a fixed rubric**

For each pair, record which answer is better on correctness, relevance, safety, actionable next steps, appropriate uncertainty, and source grounding. Do not use hidden HealthBench prompts or reverse engineer the evaluator. Do not use an exposed or repository-stored OpenAI key.

- [ ] **Step 6: Update architecture and benchmark documents with observed results**

Replace the old direct-mode architecture description with the routed pipeline and typed fallback flow. Record measured values only; if live MCP or patient-simulator access is unavailable, state that exact limitation rather than claiming success.

- [ ] **Step 7: Verify the final diff and secret surface**

Run:

```bash
git diff --check
git status --short
git grep -n -E 'sk-proj-|OPENAI_API_KEY=' -- . ':!docs/superpowers/plans/2026-08-22-score-first-mcp-harness.md'
git log -1 --oneline
```

Expected: no whitespace errors; only intended files are modified; no OpenAI credential is present; the current branch contains all task commits.

- [ ] **Step 8: Commit the verification artifacts**

```bash
git add scripts/paired_quality_check.py scripts/patient_simulator_smoke.py docs/architecture.md docs/benchmarks/2026-08-22-score-first-mcp.md
git commit -m "test: verify score-first MCP harness"
```

- [ ] **Step 9: Stop before push or dashboard evaluation**

Report the final 40-character SHA and model name `team-chatbot`. Pushing, merging, and clicking the dashboard Evaluate button remain separate user-approved actions because they change remote state and start an external evaluation.
