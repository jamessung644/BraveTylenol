# CoEval local hybrid routing gate

- Date: 2026-08-22 (Asia/Seoul)
- Variant: `integration/mcp-024-hybrid` working tree
- Baseline ancestry: `baseline/024-hybrid` at `4b130bcfcffa26f3e31fab69374a0dbacbc7cf68`
- Measurement class: secret-free deterministic local simulation
- Network, L2, MCP, judge, and dashboard calls: none

## Official public contract pin

All future paired experiments must use the official
[`lunit-io/CoEval`](https://github.com/lunit-io/CoEval) repository pinned to
`741263cfafba687f8baeb7422c747ef9557df1c4`.

The public `conquer_val` contract recorded for this gate is 301 examples,
concurrency 16, two inference attempts, 2-second retry delay, `max_tokens=6144`,
180-second timeout, and a zero score for inference failure. The `conquer_test`
IDs and judge are held out: this project must not find, infer, inspect, or tune
against them.

At the pinned commit, the canonical `conquer_val.yaml` comment reports 3,337
judge calls. That pinned file, rather than an unpinned secondary count, is the
recorded authority for this experiment.

### Predeclared anti-overfitting split

No `conquer_val` prompt ID, wording, rubric, or expected answer may appear in
runtime routing or prompt code. Before any scored run, the 301 official prompt
IDs must be sorted by `SHA-256("coeval-741263-dev-v1:" + prompt_id)`; the first
120 hashes form `dev120` and the remaining 181 form sealed `confirm181`. Only a
secret-free manifest hash, counts, CoEval commit, and experiment generation may
be recorded in this repository. Prompts, rubrics, answers, and per-item rules
must not be copied into runtime artifacts.

`confirm181` has not been executed. It may be evaluated once after a candidate
is frozen. Looking at that result and then editing the same candidate invalidates
the confirmation; a new experiment generation and a different untouched public
medical vector family are then required. HealthBench Main/Consensus outside the
published `conquer_val` membership and all held-out test material are excluded
from tuning. Adoption also requires the same direction on a second independent
public medical vector family and a paired bootstrap interval/effect size; neither
requirement has been measured in this local gate.

This local gate does not execute CoEval and does not produce a HealthBench,
leaderboard, or surrogate-judge score.

It performs request-schema-only checks for fixed single-turn, multi-turn, and
`text`/`input_text` content-part payloads shaped like the public passthrough
client request (`model`, `messages`, `temperature=0`, `top_p`,
`max_tokens=6144`, `stream=false`). HTTP response behavior is not exercised by
this script; a nonempty completion or explicit error must still be checked by
the official-client smoke gate.

## Deterministic matrix

The script fixes 16 vectors before answer or score inspection. They cover
general medicine, noisy Korean fragments, inverted wording, source-word false
positives, stale multi-turn source keywords, topic switches, current and
negated emergencies, assistant-authored emergency text, pregnancy drug safety,
MFDS/KCD/HIRA direct lookups, hierarchical guideline lookup, no-match evidence,
and a three-turn-back medication coreference.

Expected route mix:

| Route | Vectors | Share |
| --- | ---: | ---: |
| Direct | 7 | 43.75% |
| Emergency no-MCP | 1 | 6.25% |
| RAG | 8 | 50.00% |

Structured MFDS/KCD/HIRA/drug fixtures account for one MCP call. Hierarchical
guideline/index fixtures account for two or three MCP calls. A fake retrieval
counts toward cite yield only when its result explicitly contains an observed
`cite_uid`; a planned call or a no-match result does not count.
These call profiles are benchmark workload assumptions, not proof that the live
runtime or dashboard has adopted or completed those trajectories.

Run from the repository root:

```bash
python scripts/benchmark_hybrid_local.py --pretty
pytest -q tests/test_hybrid_benchmark.py
```

The command exits nonzero when an actual router decision differs from the fixed
expected route. `--allow-route-mismatch` is available only to inspect a failing
JSON report; it is not a passing release gate.

## Local fake result

The virtual clock assigns 3,000 ms per L2 call, 1,000 ms per MCP call, and 100
ms request overhead. These values exist solely for stable cost accounting and
are not observed or predicted production latency.

| Metric | Deterministic fixture result |
| --- | ---: |
| Route matches | 16 / 16 |
| Simulated L2 calls | 24 total / 1.50 per request |
| Simulated MCP calls | 12 total / 0.75 per request |
| Simulated MCP calls per RAG request | 1.50 |
| Fake observed-cite yield | 7 / 8 RAG requests (87.5%) |
| Direct/emergency simulated p50 / p95 / max | 3.10 / 3.10 / 3.10 s |
| RAG simulated p50 / p95 / max | 7.10 / 8.75 / 9.10 s |
| 16-way no-contention simulated wall time | 9.10 s |

## Release interpretation

Implemented and locally fake-verified:

- fixed 16-way router regression matrix;
- deterministic route share, virtual p50/p95/max, L2/MCP call accounting;
- cite yield requiring an observed fake `cite_uid`;
- an immutable record of the public CoEval pin and execution parameters.

Not measured:

- paired baseline-versus-hybrid CoEval score or per-item delta;
- real direct/RAG latency, retry rate, throughput, or 301/C16 wall time;
- credentialed L2 or MCP availability and evidence yield;
- official dashboard or held-out test behavior.

Therefore this result does not establish a performance improvement and is not a
promotion, merge, push, or submission authorization. A later release gate must
record the exact variant commit/tree hash, use the same frozen public vectors
and settings for paired A/B, include failures in the denominator, and pass live
L2/MCP plus official-client smoke checks without storing prompts, evidence, or
sensitive runtime material in logs or artifacts.
