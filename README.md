# BraveTylenol bounded-L2 Korean baseline

이 브랜치는 Trial 16에서 완주한 `9ead580` no-timeout Korean baseline을 바탕으로
Lunit L2 호출을 한 번만 추가한 안정성 우선 제출본입니다. L2가 18초 안에 정상
텍스트를 반환하면 그 답변을 사용하고, 인증·네트워크·HTTP·JSON·빈 응답·시간초과
문제가 생기면 HTTP 오류를 평가기에 전파하지 않고 검증된 한국어 기준 응답으로
복귀합니다. MCP와 재시도는 사용하지 않습니다.

## 선택한 범위

- 저장소 루트의 Dockerfile로 실행
- 0.0.0.0:8000에서 수신
- GET /health, GET /healthz
- GET /v1/models
- POST /v1/chat/completions
- 정상 CoEval 요청은 대화 전체와 요청 Bearer token을 L2로 전달
- L2 호출은 요청당 최대 1회, 최대 18초, 최대 4,096 token
- model 생략, 임의 추가 필드, stream=true, 빈 본문, 잘못된 JSON과 모든 L2
  실패도 채팅 엔드포인트에서는 HTTP 200의 일반 JSON completion으로 처리
- API 키와 Authorization 헤더가 없어도 실행
- Python 표준 라이브러리만 사용하며 빌드 중 pip install 없음

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

Docker가 있는 환경에서는 네트워크를 끊어 fallback 완주 경로를 확인할 수 있습니다.

## 제출 전 주의

## 중요한 제한

대회 문서의 공식 규칙에 따라 정상 경로의 최종 답변은 `Lunit/L2-preview`가
생성합니다. 단, 단일 L2 호출이 실패하거나 18초를 넘으면 전체 CoEval 실행을
실패시키지 않기 위해 정적 안전 응답을 반환합니다. 검증 세트 약 301개와 동시성
16을 기준으로 모든 L2 요청이 제한시간까지 지연되어도 생성 대기 상한은 대략
`ceil(301 / 16) × 18 = 342초`입니다.
