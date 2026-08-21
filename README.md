# BraveTylenol score-first MCP medical harness

`team-chatbot`은 CoEval용 OpenAI 호환 의료 채팅 서버입니다. 사용자에게 보이는 의료
답변은 공식 Lunit L2(`Lunit/L2-preview`)만 생성하거나 검증합니다. 서버는 요청 내용을,
근거 원문, Authorization 헤더, 또는 자격 증명을 로그나 smoke 출력에 남기지 않습니다.

## Public API

- `GET /health`와 `GET /healthz`: 상태 확인
- `GET /v1/models`: evaluator 모델 `team-chatbot` 반환
- `POST /v1/chat/completions`: OpenAI-shaped completion 반환

모든 유효한 completion은 `model="team-chatbot"`, 하나의 nonblank assistant choice,
`finish_reason="stop"`, 그리고 음수가 아닌 usage를 갖습니다. 예상된 L2/MCP 실패도
HTTP 오류로 전파하지 않고 같은 정상 completion envelope로 복구합니다. malformed 요청과
`stream=true`만 검증 오류입니다.

## Score-first pipeline

기본 모드는 `rag`이고 MCP endpoint는
`https://mcp.hackathon.lunit.io/mcp`입니다. 결정론적 다중 도메인 라우터가 약물, 안전성,
급여, KCD, 법령, 가이드라인, 연구, 일반 건강, 응급과 취약집단 신호를 분류합니다. 라우터가
선택한 도구만 첫 두 병렬 retrieval wave에서 호출하며, 요청당 MCP 호출은 최대 6회입니다.
공식성·관련성 순으로 중복을 제거한 근거는 최대 24,000 문자만 L2 생성에 전달합니다.

기본 절대 요청 제한은 165초입니다. retrieval은 75초, grounded generation은 60초이며,
최소 25초가 남을 때만 고위험 답변을 L2 verifier가 한 번 점검합니다. 응답 복구 순서는 다음과
같습니다.

1. retrieval 실패는 원 대화로 direct L2 한 번을 시도합니다.
2. grounded L2 실패는 10초 이상 남았을 때 direct L2 한 번을 시도합니다.
3. verifier 실패는 grounded 답변을 유지합니다.
4. direct L2도 실패하거나 시간이 부족하면 한국어 안전 fallback을 반환합니다.

따라서 expected upstream failure도 evaluator가 파싱 가능한 HTTP 200 completion이 됩니다.

## Run locally

Python 3.13 환경에서 production requirements를 설치한 뒤 실행합니다.

```bash
python3 -m pip install -r requirements.txt
python3 -m uvicorn app:app --host 0.0.0.0 --port 8000
```

다른 터미널에서 endpoint와 16x16 concurrency gate를 확인합니다.

```bash
curl --max-time 5 http://127.0.0.1:8000/health
curl --max-time 5 http://127.0.0.1:8000/v1/models
python3 scripts/concurrent_smoke.py \
  --base-url http://127.0.0.1:8000 --requests 16 --concurrency 16
```

smoke checker는 health와 models를 먼저 확인한 뒤 같은 한국어 multi-turn 의료 요청을 N회
보냅니다. 200/model/envelope/usage/165초 제한을 검증하고, 정확한 정적 안전 fallback은
구조적으로 성공으로 세되 별도 집계합니다. 출력은 요청·성공·fallback·실패 수, HTTP 상태 수,
최소/중앙/최대 지연 시간뿐이며 답변·요청·헤더·예외 상세를 출력하지 않습니다.
안전한 로컬 gate 범위는 `--requests` 1–256, `--concurrency` 1–32이며 실제 worker 수는 요청
수를 넘지 않습니다. 기본 16x16 gate는 이 제한 안에서 그대로 16개 worker를 사용합니다.

## Docker

이미지는 Python 3.13 slim에서 production dependencies만 설치하고 `app.py`와
`lunit_hackathon`만 복사합니다. Uvicorn은 UID/GID `65532:65532`로 `0.0.0.0:8000`에서
실행하며 표준 라이브러리 healthcheck가 `/health`를 검사합니다. `.dockerignore`는 이 runtime
파일만 허용하므로 Git metadata, `.env*`, tests, docs, scripts, caches, virtualenv, IDE/coverage
파일, 사용자 산출물 및 key material은 build context에 포함되지 않습니다.

```bash
docker build -t brave-tylenol:score-first .
docker run -d --name brave-tylenol-score -p 8000:8000 brave-tylenol:score-first
curl --max-time 5 http://127.0.0.1:8000/health
curl --max-time 5 http://127.0.0.1:8000/v1/models
python3 scripts/concurrent_smoke.py \
  --base-url http://127.0.0.1:8000 --requests 16 --concurrency 16
docker stop brave-tylenol-score
```

Observed Docker verification on 2026-08-22: `brave-tylenol:score-first` built successfully;
the explicitly named `brave-tylenol-score` container returned 200 from health and models; the
16x16 smoke run had 16 successes, 0 fallbacks, 0 failures, and 74.665 seconds maximum latency.
That named container was stopped after the check. The Task 7 report has the complete command
evidence; do not treat these local results as evidence that Docker is available on another machine.

## Configuration and key hygiene

The server resolves the Lunit credential internally; `LUNIT_FM_API_KEY` takes precedence over
a valid request Bearer credential, and no credential value should be put in this repository,
command output, logs, or image context. No OpenAI SDK or OpenAI judge key is used or bundled.

Useful environment overrides are:

```text
AGENT_MODE=rag|direct|passthrough
LUNIT_MCP_URL=https://mcp.hackathon.lunit.io/mcp
LUNIT_FM_API_URL=...
LUNIT_FM_API_KEY=...                 # inject at runtime only
REQUEST_TIMEOUT_SECONDS=165
RETRIEVAL_TIMEOUT_SECONDS=75
GENERATION_TIMEOUT_SECONDS=60
VERIFICATION_MINIMUM_SECONDS=25
MAX_MCP_CALLS=6
MAX_EVIDENCE_CHARS=24000
MAX_COMPLETION_TOKENS=4096
RETRIEVAL_REASONING_EFFORT=medium
GENERATION_REASONING_EFFORT=high
VERIFICATION_REASONING_EFFORT=medium
```

`LUNIT_FM_MODEL` defaults to `Lunit/L2-preview`. Keep local `.env` files, dashboard credentials,
temporary key files, raw patient data, and smoke answer captures outside the Docker build context
and out of version control.

## Verification

```bash
python3 -m pytest
python3 -m ruff check .
```
