# BraveTylenol L2 MCP baseline

OpenAI-compatible submission service for the Lunit Foundation Model hackathon.
`/v1/chat/completions` sends the final answer from the L2 model; this service
does not synthesize a substitute final answer. In `rag` mode it lets L2 use
the contest MCP service for retrieval before producing that final answer.

## Configuration

Set the contest credential only in the runtime environment. Do not commit it,
bake it into an image, or pass it as a Docker build argument.

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `LUNIT_FM_API_KEY` | No when caller sends Bearer auth | none | Contest L2 API credential. If unset, chat forwards the request's `Authorization: Bearer …` token. |
| `LUNIT_FM_API_URL` | No | `https://model.hackathon.lunit.io` | L2 API base URL. |
| `LUNIT_FM_MODEL` | No | `Lunit/L2-preview` | L2 model identifier. |
| `LUNIT_MCP_URL` | No | `https://mcp.hackathon.lunit.io/mcp` | Contest MCP endpoint used in `rag` mode. |
| `HARNESS_MODE` | No | `passthrough` | `passthrough` calls L2 directly; `rag` enables MCP retrieval. |
| `MAX_TOOL_CALLS` | No | `4` | Maximum MCP tool-call rounds in RAG mode. |
| `UPSTREAM_TIMEOUT_SECONDS` | No | `150` | Deadline for the complete chat request (including RAG), in seconds. |
| `MAX_COMPLETION_TOKENS` | No | `3072` | L2 output ceiling that prevents long-tail completions from exceeding the evaluation wall-clock budget. |
| `LUNIT_REASONING_EFFORT` | No | `low` | L2 reasoning effort used to reduce long-tail latency while preserving a complete answer. |
| `MAX_TOOL_RESULT_CHARS` | No | `12000` | Per-tool-result truncation limit. |
| `MAX_EVIDENCE_CHARS` | No | `32000` | Total retrieved-evidence truncation limit. |

## Local development

Python 3.13 is the target runtime.

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt -r requirements-dev.txt
.venv/bin/pytest -q
.venv/bin/ruff check app.py harness tests
.venv/bin/python -m compileall -q app.py harness
test -n "$LUNIT_FM_API_KEY" && .venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000
```

## Docker

Build the minimal Python 3.13 runtime image. For local use, inject the
credential only when starting the container:

```bash
docker build -t brave-tylenol:baseline .
docker run --rm -p 8000:8000 \
  -e LUNIT_FM_API_KEY \
  -e HARNESS_MODE=passthrough \
  brave-tylenol:baseline
```

The image listens on port 8000 and starts `uvicorn app:app --host 0.0.0.0 --port 8000`.

## API examples

Readiness and model discovery do not require the contest credential:

```bash
curl --fail --silent http://127.0.0.1:8000/v1/models
```

Call the OpenAI-compatible chat endpoint after starting with the runtime key.
If the container was started without it, include the same key as Bearer auth;
the evaluation client uses this OpenAI-compatible path:

```bash
curl --fail --silent http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $LUNIT_FM_API_KEY" \
  -d '{"model":"team-chatbot","messages":[{"role":"user","content":"안녕하세요"}]}'
```

### Mode comparison

- `HARNESS_MODE=passthrough` (default): no MCP retrieval client is constructed; the
  original chat messages go straight to L2, whose final answer is returned.
- `HARNESS_MODE=rag`: the orchestrator allows L2 to retrieve relevant contest
  data through MCP, then returns L2's final answer.

## Submission and live checks

Prepare the final submission on the exact branch `lunit/hackathon-submission`.
The following checks require a valid contest API credential and therefore are
not part of the credential-free deterministic suite:

1. Ensure `$LUNIT_FM_API_KEY` is set, then start the container as above.
2. Run the chat request above and confirm a successful L2-backed completion.
3. Run the hackathon's CoEval command or dashboard evaluation against
   `http://127.0.0.1:8000/v1/chat/completions` (or the deployed equivalent)
   using the contest-provided credential and evaluator settings.

For a credential-free runtime smoke check, start the container without the
key: `/v1/models` must return 200, while a chat request without Bearer auth
must return 503.

### One-command evaluator-style verifier

The verifier builds and starts the image **without** placing the credential in
the container. It then behaves like the evaluator: the key is sent only as the
request Bearer token. It checks model discovery, unauthenticated rejection, a
real single-turn L2 completion, and a real multi-turn completion. Response text
and the credential are never printed.

```bash
read -rs LUNIT_FM_API_KEY
export LUNIT_FM_API_KEY
.venv/bin/python scripts/verify_submission.py
unset LUNIT_FM_API_KEY
```

Use another local port if 8000 is occupied, or reuse an existing image:

```bash
.venv/bin/python scripts/verify_submission.py --port 8123
.venv/bin/python scripts/verify_submission.py --skip-build --image brave-tylenol:verify
```

This is a submission-runtime smoke verifier, not a replacement for the
dashboard's full CoEval/HealthBench score.
