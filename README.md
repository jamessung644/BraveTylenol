# Brave Tylenol — Lunit L2 Medical Chat

Conquer Health 제출용 OpenAI 호환 의료 대화 서비스다. 기본 경로는 의료 안전
프롬프트와 전체 대화 이력을 `Lunit/L2-preview`에 전달해 L2가 생성한 답변을
반환한다. 고정 답변이나 외부 모델은 사용하지 않는다.

## 평가 API

- `GET /health`, `GET /healthz`
- `GET /v1/models` (`team-chatbot`)
- `POST /v1/chat/completions`
- 요청 `model` 생략 가능
- 비스트리밍 OpenAI Chat Completions 응답
- 평가 요청의 `Authorization: Bearer ...`를 요청 범위 L2 인증으로 전달
- 멀티턴 `messages` 이력을 그대로 보존

## CoEval 기본 동작

공식 `conquer_val` 설정은 약 301개 문항을 동시성 16으로 실행하고, 추론 실패를
한 번 재시도하며, L2의 reasoning과 최종 답변을 위해 `max_tokens=6144`를 요청한다.
이에 맞춘 제출 기본값은 다음과 같다.

| 변수 | 기본값 | 설명 |
| --- | --- | --- |
| `LUNIT_FM_API_URL` | `https://model.hackathon.lunit.io` | L2 base URL |
| `LUNIT_FM_MODEL` | `Lunit/L2-preview` | 최종 답변 모델 |
| `AGENT_MODE` | `direct` | 평가 기본 경로: L2 1회 직접 생성 |
| `REQUEST_TIMEOUT_SECONDS` | `65` | 요청 전체 제한 |
| `MAX_COMPLETION_TOKENS` | `6144` | CoEval 요청을 자르지 않는 L2 토큰 상한 |
| `L2_RETRY_ATTEMPTS` | `0` | CoEval 재시도와 중첩되는 내부 HTTP 재시도 방지 |
| `LUNIT_REASONING_EFFORT` | `low` | 지연을 줄이는 추론 수준 |

`AGENT_MODE=rag`와 `LUNIT_MCP_URL`을 함께 설정하면 공식 MCP 기반 검색 경로를
사용할 수 있다. 평가 컨테이너에 MCP 주소가 없는 기본 제출에서는 direct 경로가
사용되어 요청당 정상 L2 호출은 한 번이다.

## 로컬 실행

Python 3.12 또는 3.13 환경에서:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
cp .env.example .env
.venv/bin/python main.py serve
```

L2 연결만 확인하려면:

```bash
.venv/bin/python main.py check-l2
```

## 검증

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
.venv/bin/python -m compileall -q app.py main.py lunit_hackathon
.venv/bin/python -m pip check
```

## Docker 제출

```bash
docker build -t brave-tylenol:lunit .
docker run --rm -p 8000:8000 brave-tylenol:lunit
```

이미지는 비루트 사용자로 실행되며 API 키를 이미지에 포함하지 않는다. CoEval은
각 요청의 Bearer 키를 전달하므로 평가 컨테이너에 `.env`가 없어도 `/health`,
`/v1/models` 및 채팅 경로가 정상 기동한다.

## 보안 및 격리

- 실제 API 키, 대시보드 계정, 의료 대화 내용을 저장하거나 로그에 남기지 않는다.
- `.env`, 테스트, 캐시 및 개발 문서는 Docker 이미지에 포함하지 않는다.
- 런타임 외부 호출은 Lunit L2와 명시적으로 설정된 공식 MCP로 제한한다.
- HealthBench 문항이나 답변을 하드코딩하지 않는다.
