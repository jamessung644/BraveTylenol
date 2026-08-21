# Submission error cases

Use this checklist before promoting a submission commit.

## Entrypoint and packaged-file mismatch

- Confirm that the Docker `CMD` executes the file being changed. An edit to
  `main.py` is ignored when the image starts `app.py`, and vice versa.
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

## Test-environment mismatches

- A sandbox `PermissionError` while binding loopback is not a product failure;
  rerun the same socket tests in the approved loopback-capable environment.
- Tests for a legacy entrypoint do not prove the active image. Keep the active
  runtime suite separate, and require the packaging test to show that excluded
  legacy files cannot affect the image.

## Branch synchronization

- Check whether the remote branch advanced before every push. Do not overwrite
  newer work merely to make commit IDs equal.
- If two branches must ship the same files, prefer a normal fast-forward commit
  whose tree matches the validated candidate. Preserve the prior remote history
  instead of force-pushing it away.

## Release checklist

1. Verify the remote branch SHA immediately before pushing or submitting.
2. Build the exact candidate from `git archive`, targeting `linux/amd64`.
3. Start that clean image without host bind mounts and verify `/health` and
   `/v1/models`.
4. Run a nonempty chat-completion smoke through the same HTTP contract used by
   the evaluator.
5. Keep credentials out of Git, Docker layers, logs, responses, and test
   artifacts.
