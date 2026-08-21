# BraveTylenol bounded-L2 Korean baseline

이 브랜치는 Trial 16에서 완주한 `9ead580` no-timeout Korean baseline을 바탕으로
Lunit L2 호출을 한 번만 추가한 안정성 우선 제출본입니다. L2가 30초 안에 정상
텍스트를 반환하면 그 답변을 사용하고, 인증·네트워크·HTTP·JSON·빈 응답·시간초과
문제가 생기면 HTTP 오류를 평가기에 전파하지 않고 검증된 한국어 기준 응답으로
복귀합니다. MCP와 재시도는 사용하지 않습니다.

## 평가 환경 계약

- 저장소 루트의 Dockerfile로 실행
- 0.0.0.0:8000에서 수신
- GET /health, GET /healthz
- GET /v1/models
- POST /v1/chat/completions
- `lunit_...` 형식의 `LUNIT_FM_API_KEY`가 요청 Bearer보다 우선
- 환경변수가 없을 때만 `lunit_...` 형식의 요청 Bearer를 L2에 전달
- 두 runtime credential이 모두 없거나 무효하면 `main.py`의 내장 placeholder를 사용
- 내장 placeholder `ffffffff`는 로컬 실행·Docker build 전에 실제 event key로 교체하고,
  실제 값은 commit 또는 push하지 않음
- L2 호출은 요청당 최대 1회, 최대 30초, 최대 4,096 token
- L2 outbound 동시 실행은 공식 CoEval 동시성과 같은 16개로 제한
- 응답 본문은 4MB 및 요청 전체 35초 deadline으로 제한해 slow body가 무한정
  worker를 점유하지 않도록 처리
- model 생략, 임의 추가 필드, stream=true, 빈 본문, 잘못된 JSON과 모든 L2
  실패도 채팅 엔드포인트에서는 HTTP 200의 일반 JSON completion으로 처리
- 모든 지원 응답에는 `X-Request-ID`를 부여하고 SIGTERM으로 즉시 정상 종료
- API 키와 Authorization 헤더가 없어도 실행
- Python 표준 라이브러리만 사용하며 빌드 중 pip install 없음

## 실행

~~~bash
python main.py serve
~~~

~~~bash
curl --max-time 2 http://127.0.0.1:8000/health
curl --max-time 2 http://127.0.0.1:8000/v1/models
curl --max-time 2 \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What should I do?"}]}' \
  http://127.0.0.1:8000/v1/chat/completions
~~~

## 테스트

~~~bash
python -m unittest discover -s tests -p 'test_baseline_server.py' -v
~~~

Docker가 있는 환경에서는 네트워크를 끊어 fallback 완주 경로를 확인할 수 있습니다.

~~~bash
docker build -t brave-tylenol-baseline .
docker run --rm --network=none -p 8000:8000 brave-tylenol-baseline
~~~

## 중요한 제한

대회 문서의 공식 규칙에 따라 정상 경로의 최종 답변은 `Lunit/L2-preview`가
생성합니다. 단, 단일 L2 호출이 실패하거나 30초를 넘으면 전체 CoEval 실행을
실패시키지 않기 위해 정적 안전 응답을 반환합니다. 검증 세트 약 301개와 동시성
16을 기준으로 모든 L2 요청이 제한시간까지 지연되어도 생성 대기 상한은 대략
`ceil(301 / 16) × 30 = 570초`입니다.
