# CoEval public-val paired program v2

## Claim scope

This program supports a provisional competition claim about the pinned CoEval
`conquer_val` slice. It does not access or estimate `conquer_test`, and it is not
the same experiment as the paper's 5,000-example HealthBench Main result. The
paper uses GPT-4.1 as its default grader; pinned CoEval uses `glm-5.2-fp8` for
`conquer_val`. A result must therefore be named precisely, for example:

> CoEval `conquer_val` at `741263cf`, GLM judge, score 0.xxxx.

Do not describe that result as a paper-comparable HealthBench Main score. Only
the organizer's held-out evaluation can provide an independent competition
confirmation.

## Frozen public contract

- CoEval commit: `741263cfafba687f8baeb7422c747ef9557df1c4`
- Published `conquer_val_ids.json`: 301 unique public IDs, SHA-256
  `e4047a7adb607cda11270d3d76b80270e75bff8081bdb64f20f78f77f963c8b9`
- Generation concurrency: 16
- Runner inference attempts: 2, with 2-second exponential-backoff base
- Candidate max tokens / timeout: 6,144 / 180 seconds
- Candidate-client `max_retries`: 3 inside each runner attempt
- Inference failure: included as score zero
- GLM judge concurrency / attempts / retry-delay base: 32 / 4 / 2 seconds
- Overall metric: mean of per-example HealthBench rubric scores, then clipped to
  `[0, 1]`; negative criteria remain penalties before aggregation

The pinned YAML comment says approximately 3,337 judge calls. The published ID
manifest at the same commit records 3,410. Cost and capacity planning must use
the larger 3,410-call value and must finally reconcile against the actual sum of
rubric-grade counts in the run artifact. For a full direct/hybrid pair that is
6,820 initial judge verdicts and at most 27,280 judge requests if every criterion
uses all four metric attempts. The nested candidate retry settings allow up to
eight HTTP attempts per item in the transport-error worst case: four low-level
attempts inside each of two runner attempts.

## Contamination disclosure and v2 partition

The earlier v1 `confirm181` is not an untouched confirmation set. Local evidence
contains a full-301 baseline inference artifact, and one of the three public
items scored by the historical non-official Codex judge fell in v1 confirmation.
V2 therefore derives those known first-three IDs from the public manifest and
forces them into development. It then orders every remaining public ID by
`SHA-256("coeval-741263-dev-v2:" + prompt_id)`, fills development to 120, and
places the remaining 181 in prospective score holdout.

This is a prospective **score-holdout**, not a strict prompt-holdout. Neither the
generator nor runtime code reads prompt text, rubrics, answers, HealthBench data,
or held-out IDs. The output ID files are required only because CoEval's public
`ids_path` loader needs explicit membership. Keep them and all CoEval result
artifacts outside the repository.

From this repository, with the pinned CoEval checkout already present:

```bash
python scripts/prepare_conquer_val_split.py \
  --ids /private/tmp/coeval-741263/src/coeval/data/conquer_val_ids.json \
  --output-dir /private/tmp/conquer-val-v2
```

If more public-val IDs were inspected or scored, put only those IDs in a local
JSON `prompt_ids` list and regenerate before any confirmation run:

```bash
python scripts/prepare_conquer_val_split.py \
  --ids /private/tmp/coeval-741263/src/coeval/data/conquer_val_ids.json \
  --touched-ids /private/tmp/coeval-additional-touched.json \
  --output-dir /private/tmp/conquer-val-v2-next
```

The generator rejects touched IDs outside published `conquer_val`, refuses to
overwrite a generation by default, writes mode-0600 files, and prints only the
ID-free metadata record. With no additional touched input, expected hashes are:

| Artifact | SHA-256 |
| --- | --- |
| dev membership, newline-canonical | `4798f915f66c6b7ab77b7af93053ac887b797b174c0e6621a382789bc3c86792` |
| confirm membership, newline-canonical | `3c6af66a9efd321b3c291b8d7fe7b5d66d848ebdbfc0e3abf0ec5e369e581870` |
| `dev120.ids.json` | `5fa12b48a6832bb5386bdbe4b7eb0fc48227409f4c0f335b063f3207e491f65f` |
| `confirm181.ids.json` | `272ce61f3ad247f9b676e2d48dc0920d7fe809526f752bead75a272662739b89` |
| `split-metadata.json` | `94352cb9ae7c705229266bb1c7d495eaad4f617c147dddcaa0f6124cf813fcb2` |

Any additional touched input intentionally changes these hashes and starts a
new declared generation.

## Freeze one image and create the two arms

Archive a commit rather than the mutable working tree. Replace
`<candidate-sha>` with the frozen commit and retain the image ID and platform in
the evidence record.

```bash
mkdir /private/tmp/brave-hb60-candidate
git archive --format=tar \
  --output=/private/tmp/brave-hb60-candidate/candidate.tar \
  <candidate-sha>
mkdir /private/tmp/brave-hb60-candidate/src
tar -xf /private/tmp/brave-hb60-candidate/candidate.tar \
  -C /private/tmp/brave-hb60-candidate/src
docker buildx build --platform linux/amd64 --load \
  --tag brave-tylenol:hb60-candidate \
  /private/tmp/brave-hb60-candidate/src
docker image inspect brave-tylenol:hb60-candidate \
  --format '{{.Id}} {{.Architecture}}/{{.Os}}'
```

The approved Lunit key must already be present in the operator's environment.
`--env LUNIT_FM_API_KEY` passes it without putting its value in the command.
Never copy a `.env` into the archive or image.

```bash
docker run --detach --name hb60-direct \
  --read-only --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  --publish 127.0.0.1:18001:8000 \
  --env LUNIT_FM_API_KEY \
  --env AGENT_MODE=direct \
  --env MAX_MCP_CALLS=1 \
  brave-tylenol:hb60-candidate

docker run --detach --name hb60-hybrid \
  --read-only --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  --publish 127.0.0.1:18002:8000 \
  --env LUNIT_FM_API_KEY \
  --env AGENT_MODE=hybrid \
  --env MAX_MCP_CALLS=1 \
  brave-tylenol:hb60-candidate
```

Run the existing boundary, pinned-import, live-C16, and official-client gates
from [the container release-gate procedure](2026-08-22-container-coeval-release-gate.md)
before scoring. A contract gate is not a quality score.

## Exact paired dev commands

Set up the pinned CoEval checkout with its locked toolchain:

```bash
cd /private/tmp/coeval-741263
git rev-parse HEAD
mise trust
mise run sync
```

The resolved commit must equal the pin above. The GLM judge host in the pinned
configuration is required; if it is unavailable, stop and record the run as
blocked. A Codex, GPT, or other substitute judge is useful only as an explicitly
non-comparable diagnostic.

First display each fully composed Hydra config with the same overrides plus
`--cfg job`. Then run direct and hybrid sequentially so the two C16 jobs do not
double shared upstream concurrency. `local-no-auth` is a non-secret evaluator
placeholder; the candidate containers use their approved environment key.

```bash
mise run eval -- \
  datasets=conquer_val \
  datasets/metrics/judge@conquer_judge=glm-5.2-fp8 \
  '++datasets.conquer_val.ids_path=/private/tmp/conquer-val-v2/dev120.ids.json' \
  client.llm.config.api_base=http://127.0.0.1:18001/v1 \
  client.llm.config.model=team-chatbot \
  client.llm.config.api_key=local-no-auth \
  client.llm.config.max_retries=3 \
  client.llm.config.max_tokens=6144 \
  client.llm.config.timeout=180 \
  runner.concurrent_limit=16 \
  runner.inference_max_attempts=2 \
  runner.inference_retry_delay_s=2.0 \
  metrics.conquer_val.healthbench_rubric.concurrent_limit=32 \
  metrics.conquer_val.healthbench_rubric.max_attempts=4 \
  metrics.conquer_val.healthbench_rubric.retry_delay_s=2.0 \
  hydra.run.dir=/private/tmp/hb60-v2/dev/direct

mise run eval -- \
  datasets=conquer_val \
  datasets/metrics/judge@conquer_judge=glm-5.2-fp8 \
  '++datasets.conquer_val.ids_path=/private/tmp/conquer-val-v2/dev120.ids.json' \
  client.llm.config.api_base=http://127.0.0.1:18002/v1 \
  client.llm.config.model=team-chatbot \
  client.llm.config.api_key=local-no-auth \
  client.llm.config.max_retries=3 \
  client.llm.config.max_tokens=6144 \
  client.llm.config.timeout=180 \
  runner.concurrent_limit=16 \
  runner.inference_max_attempts=2 \
  runner.inference_retry_delay_s=2.0 \
  metrics.conquer_val.healthbench_rubric.concurrent_limit=32 \
  metrics.conquer_val.healthbench_rubric.max_attempts=4 \
  metrics.conquer_val.healthbench_rubric.retry_delay_s=2.0 \
  hydra.run.dir=/private/tmp/hb60-v2/dev/hybrid
```

Do not select a candidate from the overall mean alone. Inspect the primary
metric, pass rate separately, five axes, seven themes, inference/scoring
failures, response length, latency, and sanitized MCP counters. Any prompt,
answer, rubric, or per-item rule learned from dev must remain out of runtime
routing and prompt artifacts.

## One-shot confirmation

Freeze the candidate commit, image ID, environment names, and complete Hydra
config before confirmation. Replace only `dev120.ids.json` with
`confirm181.ids.json` and the two output directories with:

```text
/private/tmp/hb60-v2/confirm/direct
/private/tmp/hb60-v2/confirm/hybrid
```

Run each arm once with the exact dev commands above. Reading confirmation and
then changing the candidate invalidates this generation. Do not compensate by
opening HealthBench Main/Consensus outside published `conquer_val`, and never
access held-out competition IDs.

## Artifact audit and promotion rule

For every arm retain, outside Git:

- `.hydra/config.yaml` and `.hydra/overrides.yaml`;
- `results_conquer_val.json`, `summary_conquer_val.json`, and
  `summary_combined.json`;
- candidate commit/tree, image ID/platform, split metadata, CoEval commit and
  lock hash;
- sanitized container logs and contract-gate JSON.

Require result length and unique sample IDs to match 120 or 181, identical input
fingerprints between arms, and:

```text
num_evaluated = num_samples - num_inference_failed - num_scoring_failed
```

For a promotion claim, both inference and scoring failures must be zero. A
judge infrastructure failure must never be converted to a negative label or
silently dropped. Use the official paired leaderboard utility over the two
confirmation directories:

```bash
python scripts/build_leaderboard.py \
  --runs /private/tmp/hb60-v2/confirm \
  --dataset conquer_val \
  --stage official \
  --pair-items on \
  --iterations 10000 \
  --confidence 0.95 \
  --seed 42 \
  --judge glm-5.2-fp8 \
  --out /private/tmp/hb60-v2/confirm-paired-leaderboard.json
```

Two claim levels must remain distinct:

1. **Competition point claim:** hybrid `HealthBench Rubric` score is at least
   `0.6000` on one-shot `confirm181` under the pinned GLM judge.
2. **Strong “at least 60” claim:** the leaderboard's total-noise 95% lower bound
   for hybrid is also at least `0.6000`, and the paired hybrid-minus-direct
   interval excludes zero in hybrid's favor.

Predeclare which level is required before reading confirmation. Also reject
promotion for a material emergency, accuracy, or context-awareness regression
even if the overall threshold passes.

## MCP execution evidence and remaining limitation

The hybrid log must contain nonzero `retrieval_tool_result` and
`retrieval_complete` counts. Record `rag_admission_full`, retrieval error codes,
tool calls, planner turns, evidence-item counts, and citation counts. Direct
must contain no remote MCP tool results. The current logs are content-free, but
they provide aggregate rather than per-CoEval-item linkage. Therefore a score
delta can establish the effect of the complete hybrid runtime, while a causal
claim about which individual MCP call changed which item still requires a
privacy-safe request-ID/hash telemetry layer that does not persist prompt or
response content.
