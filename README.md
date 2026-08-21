# Brave Tylenol — L2 fast baseline

문제 풀이 시간을 우선한 Lunit Hackathon 베이스라인입니다. 각 conversation turn마다
`Lunit/L2-preview`를 **정확히 한 번** 호출하며, 반환된 L2 답변을 수정하지 않고 그대로
OpenAI-compatible 응답으로 전달합니다.

## 왜 빠른가

```text
Evaluator request -> L2 1회 -> final response
```

- MCP retrieval과 retrieval-planner 호출을 기본 경로에서 제거했습니다.
- tool schema를 보내지 않습니다.
- `reasoning_effort=low`, 최대 1,024 completion token을 사용합니다.
- 실패 시 느린 자동 재시도를 하지 않습니다.
- 하나의 async HTTP client와 keep-alive connection을 재사용합니다.
- 전체 multi-turn history는 순서와 내용을 유지합니다.

Dashboard의 팀 전체 관측값은 Model E2E p50 7.14초, p95 43.00초였습니다. 따라서 이
베이스라인의 turn당 현실적인 예상 범위는 대략 **7~45초**이며, 기존 다단계 구조처럼 이
시간이 L2 호출 수만큼 누적되지 않습니다. 실제 값은 Model queue와 출력 길이에 따라 달라집니다.

## 대회 조건 반영

- 최종 출력은 반드시 L2가 생성
- repository root의 `Dockerfile`
- `0.0.0.0:8000`, `EXPOSE 8000`
- `GET /v1/models`
- `POST /v1/chat/completions`
- multi-turn conversation history 전달
- 평가용 branch: `lunit/hackathon-submission`
- API key를 image나 Git에 포함하지 않음

L2 공식 가이드의 generation/retrieval 2단계는 권장 구조이며 필수는 아닙니다. 이
베이스라인은 속도 측정을 위해 retrieval을 의도적으로 제외했습니다. 최신 guideline,
허가사항, 법률처럼 외부 근거가 꼭 필요한 질문에서는 품질 손해가 있을 수 있습니다.

## 설정

`.env.example`을 참고해 runtime 환경 변수를 주입하세요.

| 변수 | 기본값 | 설명 |
|---|---|---|
| `LUNIT_FM_API_KEY` | 없음 | 필수 팀 API key |
| `LUNIT_FM_API_URL` | `https://model.hackathon.lunit.io` | L2 endpoint |
| `LUNIT_FM_MODEL` | `Lunit/L2-preview` | 사용할 Model |
| `REQUEST_TIMEOUT_SECONDS` | `65` | 단일 L2 호출 deadline |
| `MAX_COMPLETION_TOKENS` | `1024` | server-side 출력 상한 |
| `LUNIT_REASONING_EFFORT` | `low` | L2 reasoning 설정 |

Client가 더 큰 `max_tokens`를 보내도 server-side 상한을 넘지 않습니다.

## Docker 실행

```bash
docker build -t brave-tylenol:fast .
docker run --rm --env-file .env -p 8000:8000 brave-tylenol:fast
```

```bash
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Lunit/L2-preview","messages":[{"role":"user","content":"고혈압약을 먹고 어지러운데 어떻게 해야 하나요?"}]}'
```

## 검증

```bash
python -m pytest -q
ruff check app tests
python -m compileall -q app
```

현재 deterministic suite는 단일 upstream 호출, tool 미사용, low reasoning, 토큰 상한,
multi-turn 보존, streaming 거부, timeout mapping, 빈 L2 응답의 무재시도 실패를 검증합니다.
