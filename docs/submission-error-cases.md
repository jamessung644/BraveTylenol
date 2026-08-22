# Submission error cases

Use this checklist before promoting a submission commit.

## Entrypoint and packaged-file mismatch

- Confirm that the Docker `CMD` executes the file being changed. When the image
  starts `app.py`, `main.py` cannot change routing, prompts, or medical output;
  the only explicit exception in this submission is its credential bridge.
- Confirm that `.dockerignore` admits every file copied by the Dockerfile and
  excludes credentials, local environment files, caches, tests, and unrelated
  sources.
- Never infer runtime behavior from a branch name; inspect the candidate
  commit's Dockerfile and packaged file list.

## Mode and readiness failures

- A direct-agent setting does not bypass checks that happen before
  orchestration. Credential or readiness failure can therefore break both
  direct and hybrid modes in the same way.
- `AGENT_MODE` has no effect when the packaged entrypoint does not import the
  modular application. Assert the packaged entrypoint and the active mode in
  the same release test.
- After rebasing or merging a submission fix, re-check configuration defaults;
  a valid runtime commit can silently change `hybrid` back to `direct`.
- Diagnose early failures in order: deployed SHA, image build/startup,
  readiness/credential delivery, request contract, then routing or MCP.
- If the submission credential is supplied through `main.py`, the modular
  image must copy that file and explicitly enable the credential bridge while
  continuing to start `uvicorn app:app`. Merely editing `main.py` does not make
  the key visible to the modular runtime.
- Credential precedence is environment, packaged `main.py`, then a valid
  request Bearer. Only 401/403 may advance to the next distinct credential;
  timeout, 429, and 5xx must not multiply calls across credentials.

## Live MCP schema drift

- HTTP success from MCP discovery does not prove that every discovered schema
  is usable as a strict L2 function. Compile every approved binding against the
  live schema before claiming MCP readiness.
- On 2026-08-22, `rag_vector_query.filters` was an optional arbitrary-object
  field. Closing that object was not lossless and caused
  `strict_projection_not_lossless`, even though discovery returned all 21
  tools. A safe wrapper may omit an unprojectable *optional* capability and
  revalidate projected arguments against the raw transport schema; it must
  never omit a required field or expose an open object to L2.
- Re-run the exact live binding canary after any projection change. Record only
  aggregate tool counts, hashes, status, and bounded result length—not medical
  evidence or credential values.

## Test-environment mismatches

- A sandbox `PermissionError` while binding loopback is not a product failure;
  rerun the same socket tests in the approved loopback-capable environment.
- Tests for a legacy entrypoint do not prove the active image. Keep the active
  runtime suite separate, and require the packaging test to show that excluded
  legacy files cannot affect the image.
- Run the `CANDIDATE_COMMIT` packaging test from the Git worktree, not from the
  extracted archive. The test deliberately invokes `git ls-files`, verifies the
  commit, and creates its own clean archive; an archive has no `.git` metadata
  and will otherwise report a false packaging failure.

## Model latency and final-answer starvation

- A successful short greeting does not establish a safe medical-answer timeout.
  The official L2 returned a compact direct medical answer with `stop` only after
  about 104 seconds; 45-second phase limits caused systematic 504 responses, and
  a 1,024-token attempt completed late with `length` rather than a usable final.
- Keep the release final/emergency attempt ceiling at 145 seconds. The current
  final candidate lowers the regular final/recovery ceiling to 2,048, matching
  the emergency output ceiling; re-check nonempty, `stop`, and quality
  before promotion rather than reducing the timeout to manufacture faster errors.
  Retrieval is optional and must preserve the configured final-answer reserve;
  successful MCP transport followed by a starved final is still a failed request.
- Validate direct, emergency, MCP-success, and MCP-failure paths separately.
  Record aggregate status, latency, finish reason, call count, and citation yield,
  but never credentials, prompts, answers, raw evidence, or citation identifiers.

## Final validation and bounded safety completion

- A valid HTTP response from L2 can still be unusable: blank output, tool-call
  syntax, an unsupported finish reason, invented citation syntax, or an
  unverified official claim previously produced evaluator-visible 502 errors.
- The active path rebuilds one clean final transcript from the frozen request.
  If that output is also invalid, malformed, or timed out and the absolute
  request deadline still has room, it makes exactly one 30-second/256-token
  `safe_completion_final` L2 call with only a fixed normal/emergency indicator.
  It never copies the medical question, rejected draft, tool transcript, or
  evidence into this last call.
- The safety notice is accepted only when it is Korean plain text in one or two
  sentences, explicitly says it is not a medical diagnosis, and gives a
  positive clinician-referral or emergency action. Explicitly negated actions
  such as “119에 연락하지 마세요” or “의료진에게 상담하지 마세요” must fail.
- If L2 itself remains unavailable or the fixed safety call is invalid, return
  the sanitized upstream error. Never manufacture a Python-authored medical
  answer merely to force HTTP 200; the competition requires L2-authored final
  output.
- Real-parser tests must cover all three requests. Updating only scripted model
  fixtures can miss a stale two-call assertion or a parser-only function-call
  failure.

## Latency optimizations that did not work

- Lowering the tool-decision token ceiling did not improve the official Model
  micro-gate: 1,536/1,024/768 tokens all produced valid calls, while 1,024 was
  about 1% slower and 768 about 14% slower on the fixed three-query sample.
  Keep 1,536 unless a new paired gate proves a benefit.
- Prefer lossless sparse model-facing envelopes over shorter answer budgets.
  Null, empty, and constant audit placeholders may be omitted while preserving
  raw dialogue, uncertainty state, evidence status, citation mapping, and
  material limitations.
- The current fast candidate changes the regular hard cap from 4,096 to 2,048
  and the default answer target from about 700 to 500 output tokens. Emergency
  remains 2,048 and decision/retrieval remains 1,536. Treat the percentage
  reductions as configuration changes, not a latency claim, until the same
  direct/RAG prompts are remeasured on the exact image.

## Approved official-network validation

- Runtime validation may use the team credential without an artificial question
  count limit, but outbound traffic remains restricted to the official Lunit
  Model and MCP endpoints. This does not authorize hidden-test access or arbitrary
  Internet calls.
- Treat synthetic fixture generations as append-only after observation. Broad
  evaluation may add new permanent cases, but must not rewrite an observed case
  to fit its result.

## Natural-language regression retention

- Do not equate a grammatical form with clinical urgency. A noun-like fragment
  such as `삼킴` is an exposure action only when a hazardous object and a
  current, non-negated assertion are both present. Definition, prevention,
  hypothetical, quoted, historical, and explicitly negated uses must remain
  non-emergency.
- Run the frozen synthetic medical gate before release. Once a fixture
  generation has been observed, never rewrite its prompts or invariants to make
  a result pass. Preserve it and append a new permanent case ID in the next
  versioned generation.
- Record newly observed error classes here and add a dataset-independent
  regression case. Do not add benchmark item IDs, reference answers, or
  question-specific routing rules.

## Branch synchronization

- Check whether the remote branch advanced before every push. Do not overwrite
  newer work merely to make commit IDs equal.
- If two branches must ship the same files, prefer a normal fast-forward commit
  whose tree matches the validated candidate. Preserve the prior remote history
  instead of force-pushing it away.
- Every agent-created remote commit must keep
  `main.py:EMBEDDED_LUNIT_API_KEY` equal to the literal placeholder
  `"fffffff"`. The user inserts the short-lived organizer key manually after
  the push. Never push a locally tested live key, even when the key is scheduled
  for same-day disposal.

## Release checklist

1. Verify the remote branch SHA immediately before pushing or submitting.
2. Build the exact candidate from `git archive`, targeting `linux/amd64`.
3. Start that clean image without host bind mounts and verify `/health` and
   `/v1/models`.
4. Run a nonempty chat-completion smoke through the same HTTP contract used by
   the evaluator.
5. The committed `main.py` contains only `"fffffff"`; the user replaces it in
   the submission branch. Never duplicate a live value into generated
   artifacts, environment metadata, logs, responses, or test fixtures.
