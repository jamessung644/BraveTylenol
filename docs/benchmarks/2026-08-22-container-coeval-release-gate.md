# Container / CoEval release gate (2026-08-22)

## Status and scope

This is a reproducible contract gate, not a benchmark score and not evidence of
live L2/MCP quality.  It uses only dataset-independent synthetic messages.  It
does not read HealthBench main/consensus, `conquer_test`, held-out IDs, rubrics,
or expected answers.

The official public reference is pinned to CoEval commit
`741263cfafba687f8baeb7422c747ef9557df1c4`.  The gate verifies the exact commit
and SHA-256 digests of only these public files before importing the official
client:

- `src/coeval/clients/passthrough.py`
- `src/coeval/conf/datasets/conquer_val.yaml`
- `uv.lock`

The observed public validation contract is 301 examples, concurrency 16,
inference attempts 2, retry delay 2 seconds, maximum 6,144 tokens, timeout 180
seconds, and inference failures scored as zero.  `PassthroughClient` forwards
the complete message list and treats `content is None` as an inference failure.

## Evidence emitted

The script emits one JSON object containing only:

- pin and contract metadata;
- case IDs and structural pass/fail reason codes;
- HTTP status counts, nonempty rate, and aggregate latency;
- declared call count and retry count.

It does not emit request messages, model response content, credentials,
Authorization headers, or upstream error bodies.  A live credential can only be
read from an environment variable.  The target URL is restricted to loopback so
the gate cannot send that credential to a remote host.

## Clean candidate and network-none boundary

Create a clean directory from the candidate commit, then run the image commands
there.  Substitute the frozen candidate SHA; do not archive the mutable index or
working tree.  Do not use a bind mount, `.env`, or untracked files.

```sh
mkdir /private/tmp/brave-tylenol-candidate
git archive --format=tar --output=/private/tmp/brave-tylenol-candidate.tar <candidate-sha>
tar -xf /private/tmp/brave-tylenol-candidate.tar -C /private/tmp/brave-tylenol-candidate
cd /private/tmp/brave-tylenol-candidate
```

```sh
docker buildx build --platform linux/amd64 --load --tag brave-tylenol:candidate .
docker run --detach --name brave-tylenol-boundary --network none --read-only --tmpfs /tmp:rw,noexec,nosuid,size=64m brave-tylenol:candidate
docker cp scripts/container_contract_gate.py brave-tylenol-boundary:/tmp/container_contract_gate.py
docker exec brave-tylenol-boundary python /tmp/container_contract_gate.py --phase boundary --base-url http://127.0.0.1:8000
docker stop --time 10 brave-tylenol-boundary
```

The gate runs inside the container because Docker port publishing cannot make a
`--network none` namespace reachable from the host.  `docker cp` writes the
tracked gate from the clean archive into the `/tmp` tmpfs; it is not a source
bind mount and does not alter the read-only image.

Expected evidence is `passed=true`, 14 boundary cases, zero expected external
calls, `/health` and `/healthz` HTTP 200, `/readyz` HTTP 503 without a valid key,
and chat HTTP 503 rather than a fabricated assistant answer.  Malformed JSON,
duplicate keys, unsupported compression/content type, oversize input, streaming,
empty messages, non-text content, and caller-supplied tools must be rejected.

This phase proves offline startup/liveness and safe request rejection only.  A
successful chat under `--network=none` would be a failure of the L2-only final
invariant, not a release success.

## Pinned official dependency preflight

Prepare the official repository separately at the public pin.  Installing its
locked dependencies is a development/release operation and must not occur in the
candidate image or at evaluation runtime.

```sh
git -C /private/tmp/coeval-741263 rev-parse HEAD
uv sync --project /private/tmp/coeval-741263 --locked
uv run --project /private/tmp/coeval-741263 python /absolute/path/to/candidate/scripts/container_contract_gate.py --phase coeval-preflight --coeval-root /private/tmp/coeval-741263
```

The preflight reports `passed=false` and an import error type when the official
locked dependency stack is unavailable.  That is an unmet release gate, not a
skip or a pass.

## Authorized Lunit-network live contract

Only run this after explicit authorization and only with the candidate container
able to reach the two official Lunit services.  Keep the team key out of command
arguments and logs.  The container may hold the valid environment key; in that
case the client-side gate variable can remain unset.  If bearer-only behavior is
being checked, place the same approved key in `COEVAL_GATE_API_KEY` without
printing it.

```sh
uv run --project /private/tmp/coeval-741263 python /absolute/path/to/candidate/scripts/container_contract_gate.py --phase release-live --authorized-live --base-url http://127.0.0.1:8000 --coeval-root /private/tmp/coeval-741263
```

The live phase has a hard, auditable shape:

- 16 concurrent single-attempt HTTP vectors, split 8 direct-shaped and 8
  RAG-shaped; the matrix includes single-turn, multi-turn, and text content
  parts;
- two single-attempt calls through the pinned official `PassthroughClient`;
- exactly 18 inference requests at most, zero gate-internal retries;
- nonempty OpenAI-style assistant content required for every HTTP 200;
- no answer text is judged or retained.

The official CoEval runner's two-attempt behavior is recorded in metadata but is
not activated by this release gate.  This prevents retry multiplication while
testing the container.  A scored public `conquer_val` run is a separate paired
A/B experiment and must preserve the predeclared dev/confirmation split.

## Release evidence record

Record the following alongside the JSON output, without secrets or message
content:

- candidate commit SHA and `git archive` tree SHA;
- image digest, platform, UID/GID, read-only/tmpfs settings;
- CoEval commit and lock hash;
- boundary, C16, and official-client pass/fail summaries;
- p50/p95/max latency, wall time, HTTP/error counts, and retry count;
- whether the run was local fake, live Lunit Model/MCP, or official dashboard.

Do not call the gate result a score improvement.  Promotion still requires a
paired public evaluation, no increase in inference failures, and no regression
in completion or latency relative to the frozen baseline.
