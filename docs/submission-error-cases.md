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
