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
| `LUNIT_FM_API_KEY` | Yes for chat | none | Contest L2 API credential. `/v1/models` works without it; chat returns 503. |
| `LUNIT_FM_API_URL` | No | `https://model.hackathon.lunit.io` | L2 API base URL. |
| `LUNIT_FM_MODEL` | No | `Lunit/L2-preview` | L2 model identifier. |
| `LUNIT_MCP_URL` | No | `https://mcp.hackathon.lunit.io/mcp` | Contest MCP endpoint used in `rag` mode. |
| `HARNESS_MODE` | No | `rag` | `rag` enables MCP retrieval; `passthrough` calls L2 directly. |
| `HARNESS_LOG_LEVEL` | No | `INFO` | Safe harness diagnostics without prompts, evidence, or credentials. |
| `MAX_TOOL_CALLS` | No | `4` | Maximum MCP tool-call rounds in RAG mode. |
| `L2_MAX_ATTEMPTS` | No | `3` | Maximum attempts for retryable L2 HTTP responses. |
| `UPSTREAM_TIMEOUT_SECONDS` | No | `150` | Deadline for the complete chat request (including RAG), in seconds. |
| `MAX_TOOL_RESULT_CHARS` | No | `12000` | Per-tool-result truncation limit. |
| `MAX_EVIDENCE_CHARS` | No | `32000` | Total retrieved-evidence truncation limit. |

## Local development

Python 3.13 is the target runtime. The repository includes an idempotent
bootstrap command: it creates `.venv` when needed and only reinstalls
dependencies when the selected requirements file changes.

```bash
make setup
make check
make run
```

`make run` starts successfully without a credential so readiness can be
checked; chat requests return 503 until `LUNIT_FM_API_KEY` is configured.
For local values, copy `.env.example` to the ignored `.env` file and fill in
the credential, or export variables in the current shell. The Makefile loads
simple `KEY=value` entries from `.env` when it exists.

The commands are configurable without editing repository files:

```bash
# Use a specific interpreter or an isolated environment name.
PYTHON=/opt/homebrew/bin/python3.13 VENV=.venv-l2 make setup

# Install runtime-only dependencies instead of developer tooling.
PROFILE=runtime make setup

# Change the listener or enable Uvicorn reload during development.
PORT=8080 make run UVICORN_ARGS=--reload

# Work interactively with the environment activated in a child shell.
make shell
```

Run `make help` for the complete command list. If Python 3.13 is not present
on macOS, install it with `brew install python@3.13`; `.python-version` also
advertises the required interpreter to compatible version managers.

## Docker

Build the minimal Python 3.13 runtime image, then inject the credential only
when starting the container:

```bash
docker build -t brave-tylenol:baseline .
docker run --rm -p 8000:8000 \
  -e LUNIT_FM_API_KEY \
  -e HARNESS_MODE=rag \
  brave-tylenol:baseline
```

The image listens on port 8000 and starts `uvicorn app:app --host 0.0.0.0 --port 8000`.

## API examples

Readiness and model discovery do not require the contest credential:

```bash
curl --fail --silent http://127.0.0.1:8000/v1/models
```

Call the OpenAI-compatible chat endpoint after starting with the runtime key:

```bash
curl --fail --silent http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"team-chatbot","messages":[{"role":"user","content":"안녕하세요"}]}'
```

### Mode comparison

- `HARNESS_MODE=rag` (default): the orchestrator allows L2 to retrieve
  relevant contest data through MCP, then returns L2's final answer.
- `HARNESS_MODE=passthrough`: no MCP retrieval client is constructed; the
  original chat messages go straight to L2, whose final answer is returned.

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
key: `/v1/models` must return 200, while a chat request must return 503.
