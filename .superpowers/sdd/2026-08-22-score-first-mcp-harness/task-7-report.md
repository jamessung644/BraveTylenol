# Task 7 Report — Production Container and CoEval Concurrency Gate

Implementation commit: `c324e3d6619407d4375e0b173d0c7d625c91a010`

## Delivered files

- `Dockerfile`: Python 3.13 slim production runtime, production dependency install,
  only `app.py` and `lunit_hackathon` copied, non-root `65532:65532`, Uvicorn on
  `0.0.0.0:8000`, and a standard-library `/health` healthcheck.
- `.dockerignore`: strict root allowlist for Dockerfile, requirements, app, and package,
  with nested cache, bytecode, environment, Git, IDE, coverage, test/docs/scripts, key-file,
  and OpenAI-material exclusions.
- `requirements.txt`: explicitly identifies its contents as production runtime dependencies;
  development tooling remains outside the image.
- `scripts/concurrent_smoke.py`: standard-library-only, `ThreadPoolExecutor` 16x16 gate that
  performs health/models preflight, validates completion envelopes and the 165-second limit,
  and emits aggregate-only output.
- `tests/test_container_contract.py` and `tests/test_concurrent_smoke.py`: static container
  contract plus deterministic envelope, fallback, aggregation, and nonzero-exit coverage.
- `README.md`: score-first MCP model/API/runtime/defaults/fallback/key-hygiene documentation and
  observed Docker result.

## TDD evidence

- RED: `python3 -m pytest tests/test_container_contract.py tests/test_concurrent_smoke.py -q`
  produced 7 failures and 1 pass against the legacy `main.py` image and absent smoke checker.
- Additional RED: the strengthened nested build-context exclusion contract produced 1 failure
  before cache/bytecode and sensitive-artifact exclusions were added.
- GREEN: `python3 -m pytest tests/test_container_contract.py tests/test_concurrent_smoke.py -q`
  produced `8 passed`.

## Final verification

- `python3 -m pytest` — `211 passed in 1.58s`.
- `python3 -m ruff check .` — `All checks passed!`.
- `git diff --check` and staged `git diff --cached --check` — no whitespace errors.
- Changed-file secret-name scan emitted no matching changed file for an OpenAI judge credential;
  no credential values, request bodies, answers, headers, or exception details were printed.

## Docker verification

Docker was available (`29.5.2`). The final image was built with:

```bash
docker build -t brave-tylenol:score-first .
```

The explicitly named `brave-tylenol-score` container was started on port 8000. Both `/health`
and `/v1/models` returned HTTP 200, and `/v1/models` returned `team-chatbot`. The required smoke
run completed with exit code 0:

```text
requests=16 success=16 fallback=0 failure=0
http_status_counts=200:16
latency_seconds=min=51.324 median=60.661 max=74.665
```

`docker stop brave-tylenol-score` completed, and a final inspect reported `exited`. No other
container was stopped.

## Decisions and concerns

- The exact static Korean safety fallback is a structurally successful completion but remains a
  separate aggregate count, so expected recoveries are visible without exposing response text.
- Smoke transport and parsing failures are reduced to aggregate status `0`/HTTP counts; raw
  exception detail is intentionally not output.
- The observed live maximum (74.665 seconds) was below the 165-second per-request gate. Live MCP
  and L2 latency remains external and can vary, so this is recorded as an observed local result,
  not a guarantee.

## Fix Round 1 — Docker ignores and absolute smoke deadlines

Fix commit: `09ac4b08d8341e03f891c427c000c7dabbfbca0f`

- The package allowlist now excludes OpenAI-named material with fully case-insensitive character
  classes at every package depth. A Docker semantics regression creates a non-secret uppercase
  sentinel filename temporarily and proves a `COPY` of it fails from the Docker context; the test
  captures build output and never prints the sentinel contents.
- Each health/models/completion read now runs in a standard-library child process supervised by
  the required `ThreadPoolExecutor`. A monotonic parent deadline terminates and, if needed, kills
  that child, so a peer that drips bytes cannot extend a socket-idle timeout indefinitely. Batch
  waiting is bounded by preflight plus `ceil(requests / concurrency) * deadline` and a two-second
  scheduling/cleanup allowance; timed-out work becomes aggregate status `0` failures.
- URL construction and transport happen inside the guarded child process. Empty or malformed base
  URLs now return a nonzero aggregate-only result with no traceback or raw endpoint detail.
- A regression compares the standalone smoke fallback literal with the server fallback constant
  without importing application dependencies into the smoke script itself.

TDD and verification evidence:

- RED: focused container/smoke tests reported 4 failures and 8 passes: the uppercase artifact was
  copyable, and the new absolute-deadline CLI option was absent.
- GREEN: focused tests reported `12 passed in 1.40s`, including the local drip-server and malformed
  URL subprocess cases.
- Final: `python3 -m pytest` reported `215 passed in 2.90s`; `python3 -m ruff check .` reported
  `All checks passed!`; `git diff --check` was clean.
- Docker context/runtime re-verification: `docker build -t brave-tylenol:score-first .` succeeded.
  Image inspection confirmed `65532:65532` and `uvicorn app:app --host 0.0.0.0 --port 8000`.
  No live L2 smoke run was repeated because this fix changes only build-context filtering and
  smoke-client supervision, not the application runtime behavior.
