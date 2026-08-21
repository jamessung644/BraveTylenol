# BraveTylenol minimal L2 baseline

평가 서버에서 안정적으로 기동하면서 기본 점수를 얻기 위한 최소 Lunit L2 driver다.
기존 정적 한국어 baseline의 무의존 HTTP/Docker 구조는 유지하되, 채팅 답변은 반드시
`Lunit/L2-preview`가 생성하도록 바꿨다.

## 선택한 범위

공식 규칙은 최종 출력이 반드시 L2가 생성한 결과여야 한다고 명시한다. L2 가이드는
retrieval과 generation의 2단계 구성을 권장하지만 필수로 규정하지 않으며, 일반 의료
질문은 L2 memory로 직접 답할 수 있다고 설명한다. 따라서 이 baseline은 기본 점수와
실행 안정성에 필요한 다음 경로만 구현한다.

1. 평가 요청의 Bearer token을 L2 인증에 사용한다. 없으면 `LUNIT_FM_API_KEY`를 사용한다.
2. 전체 multi-turn history 앞에 짧은 의료 안전 system prompt를 추가한다.
3. `https://model.hackathon.lunit.io/v1/chat/completions`의 L2를 직접 호출한다.
4. L2가 생성한 비어 있지 않은 text만 OpenAI-compatible completion으로 반환한다.

MCP, retrieval planner, citation orchestration, 외부 data source, tool call은 포함하지 않는다.
공식 L2 가이드상 이 기능들은 더 높은 품질을 위해 유용하지만 기본 동작에는 필수가 아니다.

## 원격 8분 실패 방지

정적 baseline은 원격 평가를 끝냈지만 L2를 호출한 제출들은 반복해서 약 8분에
`coeval_failed`로 종료됐다. 공식 CoEval 설정은 301개 요청을 16-way로 실행하면서
요청당 180초, runner 2회 시도, HTTP client 재시도 3회를 겹쳐 사용한다. 따라서 한 번의
느린 요청이나 5xx가 전체 inference wall-clock budget을 소진할 수 있다. 이 baseline은
다음처럼 처리량과 실패 범위를 제한한다.

- 평가 요청은 받아들이되 L2 생성 예산을 4,096 tokens로 제한한다. 공개 validation shape
  24개를 16-way로 실측했을 때 24/24가 성공했고 평균 23.3초, p95 32.4초였다.
- `MAX_COMPLETION_TOKENS=1024`, `REQUEST_TIMEOUT_SECONDS=3` 같은 이전 환경값을
  사용하지 않는다.
- 공식 L2 URL, model, low reasoning, 65초 전체 budget을 코드에 고정해 이전 배포 설정이
  평가 경로를 바꾸지 못하게 한다.
- 정상 요청은 tool 없이 L2 direct generation만 수행한다.
- L2 동시 호출을 16개로 제한하고 첫 시도는 35초로 제한한다.
- L2가 2xx를 반환했지만 blank/malformed인 경우에만 2,048-token cap, 25초 제한으로
  더 짧은 최종 답변을 한 번 다시 요청한다.
- timeout, HTTP 오류, transport 오류는 내부에서 증폭하지 않고 즉시 degraded
  completion으로 종료한다.
- 응답 body가 조금씩 도착해 socket timeout을 우회해도 monotonic deadline에서 중단한다.
- terminal L2 failure에는 Python 의료 문장을 대신 만들지 않고 HTTP 200의 빈
  completion을 반환한다. 공개 CoEval은 이 sample을 재시도하지 않고 빈 답변으로 계속
  평가하므로 중첩 재시도가 전체 제출을 다시 8분 동안 막지 않는다.
- 안전한 failure kind와 request ID만 기록하며 질문, API key, upstream 본문은 기록하지
  않는다.

전체 요청 제한은 65초, 첫 L2 시도 제한은 35초, blank recovery 제한은 25초다. 외부
CoEval의 180초보다 충분히 일찍 valid OpenAI response를 끝내 상위 HTTP 재시도를
방지한다.

## 평가 API 계약

- repository root의 `Dockerfile`
- 별도 수동 작업 없이 `0.0.0.0:8000`에서 시작
- `GET /health`, `GET /healthz`
- `GET /v1/models`
- `POST /v1/chat/completions`
- evaluator-facing model ID: `team-chatbot`
- `model` 생략 및 추가 field 허용
- `max_tokens`, `max_completion_tokens`, text content parts, `developer` role 지원
- `stream=true`는 지원하지 않으며 HTTP 400 반환
- L2 terminal failure는 Python-authored 답변 없이 HTTP 200 empty completion 반환
- 지원하는 API 응답에 `X-Request-ID`를 부여하고 SIGTERM에 즉시 정상 종료
- Python 표준 라이브러리만 사용하므로 Docker build 중 package 설치가 없음

## 설정

| 환경변수 | 기본값 | 설명 |
| --- | --- | --- |
| `LUNIT_FM_API_KEY` | 없음 | 로컬 실행용 key; 요청 Bearer가 우선 |

L2 URL은 `https://model.hackathon.lunit.io/v1/chat/completions`, model은
`Lunit/L2-preview`, reasoning effort는 `low`, 생성 cap은 4,096 tokens, 요청 budget은
65초로 고정되어 있다.

## 실행 및 검증

~~~bash
python main.py serve
~~~

~~~bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
curl \
  -H 'Authorization: Bearer <LUNIT_FM_API_KEY>' \
  -H 'Content-Type: application/json' \
  -d '{"model":"team-chatbot","messages":[{"role":"user","content":"고혈압의 위험 신호는 무엇인가요?"}],"max_tokens":6144}' \
  http://127.0.0.1:8000/v1/chat/completions
~~~

~~~bash
python -m unittest discover -s tests -p 'test_baseline_server.py' -v
docker build -t brave-tylenol-minimal .
~~~

단위 테스트는 실제 key나 network 없이 Bearer 전달, multi-turn 보존, 4,096-token cap,
blank-only recovery, transport/timeout 비증폭, empty-completion degradation, response deadline,
32-way 입력에서 L2 동시성 16 상한, request ID, SIGTERM 종료를 검증한다.

## 제출 전 주의

이 브랜치는 최소 구현을 검증하기 위한 staging branch다. 공식 제출은
`lunit/hackathon-submission` branch HEAD의 40자리 SHA를 dashboard에 입력해야 한다.
MCP retrieval이 없으므로 최신 guideline, 법률, 의약품 허가·급여처럼 근거 조회가 필요한
질문의 품질은 제한된다. 기본 안정성을 확인한 다음에만 별도 단계로 retrieval을 추가한다.
