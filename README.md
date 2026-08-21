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

## 3초 오류 방지

문제가 된 integration 결과는 CoEval의 `max_tokens=6144` 요청을 내부에서 1,024로 줄이고,
L2가 reasoning만 생성한 빈 응답을 같은 작은 예산으로 반복한 뒤 502를 반환할 수 있었다.
이 baseline은 다음처럼 단순화한다.

- 평가 요청의 6,144-token 예산을 그대로 유지한다.
- `MAX_COMPLETION_TOKENS=1024`, `REQUEST_TIMEOUT_SECONDS=3` 같은 이전 환경값을
  사용하지 않는다.
- 공식 L2 URL, model, low reasoning, 170초 deadline을 코드에 고정해 이전 배포 설정이
  평가 경로를 바꾸지 못하게 한다.
- 정상 요청은 tool 없이 L2 direct generation만 수행한다.
- 16-way 평가에서 요청 수가 증폭되지 않도록 HTTP/전송/timeout 실패를 내부 재시도하지
  않는다. 이 경우 evaluator의 상위 재시도에 맡긴다.
- 성공 응답이 blank/malformed이면 L2에 평문 최종 답변을 한 번만 다시 요청한다.
- Python이 의료 답변을 대신 만들지 않는다. 최종 L2 실패는 세부정보를 숨긴 502/504다.

전체 요청 제한 기본값은 170초다. 첫 L2 호출에 남은 deadline을 모두 허용하고, 빠르게
도착한 blank/malformed 성공 응답에만 남은 시간 안에서 한 번의 L2 recovery를 수행한다.

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
- Python 표준 라이브러리만 사용하므로 Docker build 중 package 설치가 없음

## 설정

| 환경변수 | 기본값 | 설명 |
| --- | --- | --- |
| `LUNIT_FM_API_KEY` | 없음 | 로컬 실행용 key; 요청 Bearer가 우선 |

L2 URL은 `https://model.hackathon.lunit.io/v1/chat/completions`, model은
`Lunit/L2-preview`, reasoning effort는 `low`, 요청 deadline은 170초로 고정되어 있다.

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

단위 테스트는 실제 key나 network 없이 Bearer 전달, multi-turn 보존, 6,144-token 회귀,
blank recovery, 내부 재시도 억제, 오류 정리, 16-way 동시 요청을 검증한다.

## 제출 전 주의

이 브랜치는 최소 구현을 검증하기 위한 staging branch다. 공식 제출은
`lunit/hackathon-submission` branch HEAD의 40자리 SHA를 dashboard에 입력해야 한다.
MCP retrieval이 없으므로 최신 guideline, 법률, 의약품 허가·급여처럼 근거 조회가 필요한
질문의 품질은 제한된다. 기본 안정성을 확인한 다음에만 별도 단계로 retrieval을 추가한다.
