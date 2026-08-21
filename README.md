# BraveTylenol bounded-L2 submission

CoEval의 OpenAI-compatible 요청을 받아 `Lunit/L2-preview`가 최종 답변을 생성하는
해커톤 제출 서버입니다. 일반 의료 질문은 L2가 바로 답하고, 공식 출처가
명시된 일부 질문만 Lunit MCP에서 최대 한 번 근거를 조회합니다. L2 생성이
실패하면 CoEval이 재시도할 수 있도록 명시적인 오류를 반환합니다.

## 평가 환경 계약

- 저장소 루트의 `Dockerfile`로 실행
- `0.0.0.0:8000`에서 수신
- `GET /health`, `GET /healthz`, `GET /v1/models`
- `POST /v1/chat/completions`
- 제출 model name: `team-chatbot`
- `lunit_...` 형식의 `LUNIT_FM_API_KEY`가 유효한 요청 Bearer보다 우선
- 일반 질문은 L2 1회, 복잡한 다중 요구 질문만 짧은 coverage plan과 최종 생성 2회
- KCD, MFDS, HIRA, PubMed의 명시적 출처 질문만 결정적 MCP 조회 최대 1회
- MCP timeout 5초, coverage plan 예산 35초, 요청 전체 deadline 150초, queue wait 최대 5초
- completion budget 최대 6,144 token, 일반 `reasoning_effort=low`, plan 반영 최종 생성은 `high`
- L2 outbound 동시 실행은 공식 CoEval 동시성과 같은 16개로 제한
- 응답 본문은 4MB로 제한하며 slow body도 전체 deadline을 넘길 수 없음
- L2 timeout, HTTP 오류, 빈 응답, 잘못된 JSON은 정적 답변으로 숨기지 않고
  OpenAI 형태의 HTTP 424 오류로 반환
- 잘못된 messages는 HTTP 400으로 반환
- 모든 지원 응답에는 `X-Request-ID`를 부여하고 SIGTERM으로 정상 종료
- Python 표준 라이브러리만 사용하며 빌드 중 `pip install` 없음

## 답변 품질 전략

- 전체 대화와 evaluator system context를 보존
- 사용자 언어를 동적으로 재확인하되 명시적 언어 요청을 우선
- 환자, 임상의, 데이터 계산, 의료 문서 작성 요청을 구분
- 응급/당일/외래/자가관리 단계와 특수집단 위험을 문맥에 맞게 적용
- 정확한 항목 수, heading, schema, 길이와 제공 사실만 사용하라는 지시를 재확인
- 약 이름·제형·현재 계획이 없을 때 새 용량이나 복용 시점을 임의로 생성하지 않음
- 최종 사용자 답변은 항상 L2가 작성
- 일반 증상·응급도·복약상담은 MCP 없이 직접 답변
- 공식 근거는 사용자가 요청한 사실에만 반영하며 내부 도구명·JSON·`cite_uid`를
  사용자 출력 포맷으로 강제하지 않음
- 복잡한 질문은 사전 coverage plan으로 요청 항목 누락을 줄이되, 계획 실패 시
  기존 직접 L2 경로로 즉시 계속

## 실행

```bash
python main.py serve
```

로컬에서는 실제 키를 소스에 추가하지 말고 프로세스 환경에 주입합니다.

```bash
export LUNIT_FM_API_KEY="lunit_..."
python main.py serve
```

```bash
curl --max-time 2 http://127.0.0.1:8000/health
curl --max-time 2 http://127.0.0.1:8000/v1/models
curl --max-time 160 \
  -H 'Content-Type: application/json' \
  -d '{"model":"team-chatbot","messages":[{"role":"user","content":"What should I do?"}]}' \
  http://127.0.0.1:8000/v1/chat/completions
```

## 테스트

```bash
python -m pytest -q
python -m ruff check .
```

Docker가 있는 환경에서는 실제 제출 이미지로 확인합니다.

```bash
docker build -t brave-tylenol-submission .
docker run --rm -p 8000:8000 brave-tylenol-submission
```

## 시간 목표와 제한

공식 전체 평가 시간에는 답변 생성뿐 아니라 rubric judging도 포함되므로 로컬 서버가
전체 시간을 단독으로 보장할 수는 없습니다. 서버 내부 재시도를 제거하고 동시성을
16으로 제한해 중복 호출과 과부하를 막았으며, 정상 생성 구간은 30분 목표에 맞춰
설계했습니다. 제출 전에는 실제 Docker 이미지로 health, models, chat completion 및
동시 요청을 확인해야 합니다.
