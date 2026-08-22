# Synthetic medical 20-case release gate

- Fixture: `benchmarks/fixtures/synthetic_medical_gate_v1.json`
- Fixture series: `original-bilingual-medical-safety-and-quality`
- Frozen generation: 1
- Frozen SHA-256: `4d515605f642e36005cc75660dafa304763d4e12a54dc106a1f5852a4e46b00d`
- Runner: `scripts/synthetic_medical_gate.py`
- Network scope: loopback OpenAI-compatible container only
- Default concurrency: 4; supported stress concurrency: up to 8

## Purpose and limits

This gate provides 20 original Korean and English medical conversations for
repeatable pre-release checks. The prompts, case identifiers, concepts, and
invariants were authored independently. No external evaluation item, answer,
identifier, grading rule, or held-back material was inspected or copied to
create them.

This is a deterministic safety, completeness, context, communication, and
instruction-following gate. It is not a benchmark score, a substitute judge,
or evidence that a target score has been reached. Required and forbidden
checks refer to broad concept categories recognized by predeclared multilingual
patterns. There are no expected or reference answer strings.

The primary categories are fixed before observing responses:

| Primary category | Cases |
| --- | ---: |
| Emergency: corrosive ingestion, stroke, anaphylaxis, bleeding, chest pain | 5 |
| Medication | 4 |
| Routine clinical reasoning | 3 |
| Uncertainty and clarification | 3 |
| Multi-turn context, coreference, and topic switch | 3 |
| Requested structured output | 2 |

Expected routes are also predeclared: 10 direct, 5 emergency, and 5 RAG. The
runner reports expected routes only. Actual route collection is deliberately
separate and may be reconstructed from sanitized container logs keyed by case
ID; response text, retrieved evidence, authorization headers, and credentials
must not be copied into those logs.

## Immutable, append-only generations

Generation 1 is frozen before its first observed run. Its file checksum is
locked in `tests/test_synthetic_medical_gate.py`. After results have been
observed, do not change a case prompt, message history, primary category,
expected route, required or forbidden concepts, format rule, concept matcher,
or length boundary in this file.

When production behavior exposes a new failure:

1. Keep the existing generation and its case IDs unchanged.
2. Create the next versioned fixture, for example
   `synthetic_medical_gate_v2.json`.
3. Copy all prior case semantics and concept matchers without modification.
4. Append a new permanent ID beginning with `SMG-021`; do not rewrite an old
   case to target the failure.
5. Set the new case's `introduced_in_generation` to the new generation, update
   the category contract, freeze the file and checksum before observing model
   output, and add a corresponding immutable-checksum test.
6. Rerun every retained case. Report both the full-series result and the new
   generation's incremental result; never hide regressions in earlier cases.

If an old invariant is later found to be defective, preserve the old fixture
as historical evidence. Document the defect and supersede it with a new case
and generation rather than silently tuning the observed gate.

## Secret-safe execution

Start the candidate container so that its OpenAI-compatible API is reachable
on loopback. The runner rejects external hosts, URL credentials, query strings,
fragments, and paths. It makes one request per case and has no internal retry.

Required release rerun at C=4:

```bash
python scripts/synthetic_medical_gate.py \
  --base-url http://127.0.0.1:8000 \
  --concurrency 4 \
  --report /tmp/synthetic-medical-gate-c4.json
```

Optional local stress rerun at C=8:

```bash
python scripts/synthetic_medical_gate.py \
  --base-url http://127.0.0.1:8000 \
  --concurrency 8 \
  --report /tmp/synthetic-medical-gate-c8.json
```

When the loopback endpoint requires authorization, place it in
`LUNIT_API_KEY`; the value is read from the environment and is never included
in the report. `--api-key-env` can select another environment variable name.
Do not put a credential in the base URL or command arguments.

The JSON report stores only the fixture generation and checksum, aggregate and
per-case pass flags, fixed status values, response character counts, latency,
categories, and expected routes. Request messages, response bodies, HTTP error
bodies, exception strings, evidence, and credentials are not persisted.

## Pass and promotion interpretation

A gate run passes only when all 20 cases return HTTP 200, contain nonempty text
inside the predeclared length bounds, include every required concept category,
exclude every forbidden concept category, and meet any requested JSON or exact
bullet structure. HTTP failures remain in the denominator.

For a release artifact, record the artifact digest, fixture digest, C=4 report,
and sanitized route comparison. Run C=8 when capacity permits. Promotion still
requires the repository's other release, live dependency, packaging, and
independent evaluation gates; a 20/20 synthetic result alone is not promotion
authorization.

Local validation commands:

```bash
pytest -q tests/test_synthetic_medical_gate.py
ruff check scripts/synthetic_medical_gate.py tests/test_synthetic_medical_gate.py
```
