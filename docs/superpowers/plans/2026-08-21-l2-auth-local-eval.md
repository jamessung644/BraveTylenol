# L2 Auth and Local Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ensure the submission uses only a real `lunit_` credential, prefers the injected environment credential over evaluator placeholders, and passes a concurrency-16 local L2 gate before promotion.

**Architecture:** Keep the zero-dependency `ThreadingHTTPServer` and its always-HTTP-200 fallback contract. Add two pure credential helpers in `main.py`, raise the single upstream socket timeout to 30 seconds, and verify the behavior first with deterministic fake openers and then with a secret injected only into a local Docker container.

**Tech Stack:** Python 3.13 standard library, `unittest`/pytest compatibility, Ruff, Docker/Colima, Lunit OpenAI-compatible API.

**Spec:** `docs/superpowers/specs/2026-08-21-l2-auth-local-eval-design.md`

## Global Constraints

- Never add `/tmp/bravetylenol-lunit-key`, a `.env` file, or any API key to Git, Docker build context, logs, responses, test fixtures, or documentation.
- Preserve `GET /health`, `GET /healthz`, `GET /v1/models`, and `POST /v1/chat/completions`.
- Preserve a non-empty OpenAI-compatible HTTP 200 completion for every chat failure.
- Make at most one L2 request per chat request; do not add retries or MCP.
- Use model `Lunit/L2-preview`, maximum 4,096 completion tokens, and a 30-second upstream timeout.
- Prefer a valid `LUNIT_FM_API_KEY` over a valid request Bearer; accept only credentials beginning with `lunit_`.
- Promote only after at least 15 of 16 concurrent requests use L2, all 16 return HTTP 200 and non-empty assistant content, and every request finishes within 35 seconds.
- Never force-push; preserve the latest remote `lunit/hackathon-submission` history.

---

### Task 1: Harden credential resolution and timeout

**Files:**
- Modify: `tests/test_baseline_server.py:44-114`
- Modify: `main.py:17-166`
- Modify: `README.md:3-20`

**Interfaces:**
- Consumes: request `Authorization: Bearer ...` and `Mapping[str, str]` environment values.
- Produces: `_is_valid_lunit_key(value: str | None) -> bool` and `_resolve_lunit_api_key(authorization: str | None, environment: Mapping[str, str]) -> str | None`.
- Produces: `L2_TIMEOUT_SECONDS = 30.0` for the single `urlopen` call.

- [ ] **Step 1: Write failing credential-priority tests**

Add these tests to `BoundedL2FallbackTest`:

```python
def test_environment_lunit_key_wins_over_evaluator_placeholder(self):
    opener = RecordingOpener(
        FakeResponse({"choices": [{"message": {"content": "L2 답변"}}]})
    )
    result = main.request_l2_or_fallback(
        {"messages": [{"role": "user", "content": "질문"}]},
        "Bearer evaluator-placeholder",
        opener=opener,
        environ={"LUNIT_FM_API_KEY": "lunit_environment_test"},
    )
    self.assertEqual(result["choices"][0]["message"]["content"], "L2 답변")
    self.assertEqual(
        opener.requests[0].get_header("Authorization"),
        "Bearer lunit_environment_test",
    )

def test_valid_lunit_bearer_is_used_without_environment_key(self):
    opener = RecordingOpener(
        FakeResponse({"choices": [{"message": {"content": "L2 답변"}}]})
    )
    main.request_l2_or_fallback(
        {"messages": [{"role": "user", "content": "질문"}]},
        "Bearer lunit_request_test",
        opener=opener,
        environ={},
    )
    self.assertEqual(
        opener.requests[0].get_header("Authorization"),
        "Bearer lunit_request_test",
    )

def test_non_lunit_bearer_without_environment_falls_back_without_network(self):
    opener = RecordingOpener(AssertionError("network must not be called"))
    result = main.request_l2_or_fallback(
        {"messages": [{"role": "user", "content": "질문"}]},
        "Bearer evaluator-placeholder",
        opener=opener,
        environ={},
    )
    self.assertEqual(
        result["choices"][0]["message"]["content"],
        main.KOREAN_BASELINE_RESPONSE,
    )
    self.assertEqual(opener.requests, [])
```

- [ ] **Step 2: Update the timeout assertion and run focused tests to verify failure**

Change the existing success fixture Bearer and every `test_l2_failure_returns_baseline_without_retry`
fixture Bearer from `Bearer evaluator-secret` to `Bearer lunit_request_test`. This keeps those
tests on the network-attempt path after arbitrary Bearers are rejected. In the success test,
assert:

```python
self.assertEqual(opener.timeouts, [30.0])
```

Run:

```bash
/tmp/brave-testenv.Fycuur/bin/python -m pytest \
  tests/test_baseline_server.py::BoundedL2FallbackTest -q
```

Expected: FAIL because the placeholder Bearer still overrides the environment key, arbitrary Bearers are still accepted, and the timeout remains 18 seconds.

- [ ] **Step 3: Implement minimal credential helpers and 30-second timeout**

Add to `main.py`:

```python
L2_TIMEOUT_SECONDS = 30.0
MAX_API_KEY_LENGTH = 4_096


def _is_valid_lunit_key(value: str | None) -> bool:
    if not isinstance(value, str):
        return False
    key = value.strip()
    return (
        key.startswith("lunit_")
        and len(key) <= MAX_API_KEY_LENGTH
        and not any(character in key for character in "\r\n")
    )


def _resolve_lunit_api_key(
    authorization: str | None,
    environment: Mapping[str, str],
) -> str | None:
    environment_key = environment.get("LUNIT_FM_API_KEY")
    if _is_valid_lunit_key(environment_key):
        return environment_key.strip()
    bearer_key = _bearer_token(authorization)
    return bearer_key if _is_valid_lunit_key(bearer_key) else None
```

Replace credential selection in `request_l2_or_fallback` with:

```python
api_key = _resolve_lunit_api_key(authorization, environment)
```

- [ ] **Step 4: Run focused and full tests**

Run:

```bash
/tmp/brave-testenv.Fycuur/bin/python -m pytest \
  tests/test_baseline_server.py::BoundedL2FallbackTest -q
/tmp/brave-testenv.Fycuur/bin/python -m pytest -q
/tmp/brave-testenv.Fycuur/bin/python -m ruff check main.py tests/test_baseline_server.py
/tmp/brave-testenv.Fycuur/bin/python -m ruff format --check main.py tests/test_baseline_server.py
python3.13 -m compileall -q main.py tests
git diff --check
```

Expected: all focused tests and all 79 existing plus 3 new tests pass; Ruff, compileall, and whitespace checks pass.

- [ ] **Step 5: Update runtime documentation**

Change `README.md` to state:

```markdown
- A valid `LUNIT_FM_API_KEY` (`lunit_...`) takes precedence over request Authorization.
- A request Bearer is used only when it is itself a valid `lunit_...` key.
- L2 is called once with a 30-second timeout; all failures retain the HTTP 200 fallback.
```

Update the worst-case generation wait to `ceil(301 / 16) × 30 = 570 seconds`.

- [ ] **Step 6: Commit the implementation**

```bash
git add main.py tests/test_baseline_server.py README.md
git commit -m "fix: prefer valid Lunit credential for L2"
```

### Task 2: Run the Docker and concurrency-16 local gate

**Files:**
- Create: `docs/benchmarks/2026-08-21-l2-auth-local.md`

**Interfaces:**
- Consumes: `/tmp/bravetylenol-lunit-key`, image `bravetylenol:l2-auth-local-eval`, and local port 18084.
- Produces: a secret-free benchmark report with build result, L2 successes, fallbacks, HTTP errors, content validity, p50, p95, maximum latency, and pass/fail decision.

- [ ] **Step 1: Confirm secret isolation before building**

Run:

```bash
stat -f 'mode=%Lp bytes=%z' /tmp/bravetylenol-lunit-key
git status --short
git grep -n 'lunit_' -- ':!docs/superpowers/**' ':!tests/**'
```

Expected: key file mode 600, no key file in Git status, and no real credential in tracked files.

- [ ] **Step 2: Build and start the production image with runtime-only secret injection**

Run:

```bash
docker build -t bravetylenol:l2-auth-local-eval .
bt_key=$(</tmp/bravetylenol-lunit-key)
docker run --rm -d --name bravetylenol-l2-auth-local-eval \
  -p 127.0.0.1:18084:8000 \
  -e LUNIT_FM_API_KEY="$bt_key" \
  bravetylenol:l2-auth-local-eval
unset bt_key
```

Expected: image builds without package installation and the container starts without the secret appearing in command output.

- [ ] **Step 3: Verify runtime contract and non-root process**

Run:

```bash
curl --fail --silent http://127.0.0.1:18084/health
curl --fail --silent http://127.0.0.1:18084/v1/models
docker inspect -f \
  'health={{.State.Health.Status}} running={{.State.Running}} oom={{.State.OOMKilled}} user={{.Config.User}}' \
  bravetylenol-l2-auth-local-eval
```

Expected: health is healthy, running is true, OOM is false, and user is `65532:65532`.

- [ ] **Step 4: Run the concurrency-16 probe**

Use a Python `ThreadPoolExecutor(max_workers=16)` to send the same safe Korean medical prompt 16 times to `http://127.0.0.1:18084/v1/chat/completions`, including `Authorization: Bearer evaluator-placeholder` on every request. For each response, record only HTTP status, elapsed seconds, whether content equals `KOREAN_BASELINE_RESPONSE`, and content length. Print aggregate counts and p50/p95/max; never print response content or headers.

Expected: `l2_success >= 15`, `fallback <= 1`, `http_errors = 0`, `empty_content = 0`, and `max_latency < 35.0`.

- [ ] **Step 5: Stop the temporary container and write the benchmark report**

Run:

```bash
docker stop bravetylenol-l2-auth-local-eval
```

Create `docs/benchmarks/2026-08-21-l2-auth-local.md` with the tested commit, image tag, aggregate metrics from Step 4, checks from Step 3, and `PASS` only if every threshold is met. Do not include prompts, answers, credentials, hashes of credentials, or response bodies.

- [ ] **Step 6: Commit the benchmark evidence**

```bash
git add docs/benchmarks/2026-08-21-l2-auth-local.md
git commit -m "test: record L2 auth concurrency gate"
```

### Task 3: Promote the locally validated tree

**Files:**
- Verify only: all tracked files.

**Interfaces:**
- Consumes: passing Task 1 tests, passing Task 2 benchmark, latest `origin/lunit/hackathon-submission`.
- Produces: a fast-forwardable remote submission head containing both the latest remote history and the validated upgrade tree.

- [ ] **Step 1: Re-fetch and preserve the latest submission history**

```bash
git fetch origin lunit/hackathon-submission
git merge --no-ff -s ours origin/lunit/hackathon-submission \
  -m "merge: promote validated L2 auth upgrade"
```

Expected: merge succeeds without changing the validated working tree.

- [ ] **Step 2: Re-run the deterministic verification after the merge**

```bash
/tmp/brave-testenv.Fycuur/bin/python -m pytest -q
/tmp/brave-testenv.Fycuur/bin/python -m ruff check main.py tests/test_baseline_server.py
/tmp/brave-testenv.Fycuur/bin/python -m ruff format --check main.py tests/test_baseline_server.py
python3.13 -m compileall -q main.py tests
git diff --check
```

Expected: all checks pass and `git status --short` is empty.

- [ ] **Step 3: Push without rewriting history and verify the remote SHA**

```bash
git push origin HEAD:refs/heads/lunit/hackathon-submission
git ls-remote --heads origin lunit/hackathon-submission
git rev-parse HEAD
```

Expected: push is a fast-forward, and the remote and local 40-character SHAs match.
