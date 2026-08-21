# BraveTylenol — CoEval-compatible Lunit L2 medical driver

CoEval의 다중 턴 의료 대화를 OpenAI 호환 API로 받아 전체 대화 문맥과 의료 안전
지침을 Lunit L2에 전달하고, L2가 생성한 최종 답변을 반환하는 제출물입니다.

```text
CoEval -> BraveTylenol Driver -> Lunit/L2-preview -> CoEval rubric judge
```

기본 모드는 MCP를 호출하지 않는 단일 L2 요청입니다. 실행 안정성과 응답시간을 먼저
확보하고, MCP는 평가 환경에서 실제 endpoint가 제공될 때만 선택적으로 사용합니다.

## 평가 환경 계약

- 저장소 루트의 `Dockerfile`로 실행
- `0.0.0.0:8000`에서 수신
- `GET /health`, `GET /healthz`
- `GET /v1/models`
- `POST /v1/chat/completions`
- model 생략 및 임의 추가 필드 허용
- evaluator의 Bearer 키를 L2 API 인증에 우선 사용
- system/user/assistant 다중 턴 메시지 순서 보존
- 최종 사용자 답변은 `Lunit/L2-preview`가 생성
- 외부 검색 API나 런타임 다운로드에 의존하지 않음
- 요청·오류 로그에 의료 질문이나 인증정보를 기록하지 않음

## 환경 변수

`.env.example`을 `.env`로 복사하고 실제 값은 로컬에서만 입력합니다. `.env`는 Git과
Docker 이미지에 포함되지 않습니다.

```env
LUNIT_FM_API_URL=https://model.hackathon.lunit.io
LUNIT_FM_API_KEY=여기에_직접_입력
LUNIT_FM_MODEL=Lunit/L2-preview

DASHBOARD_USERNAME=여기에_직접_입력
DASHBOARD_PASSWORD=여기에_직접_입력
```

## 로컬 실행

```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

```bash
curl --max-time 90 \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer YOUR_LUNIT_FM_API_KEY' \
  -d '{"model":"team-chatbot","messages":[{"role":"user","content":"What should I do?"}]}' \
  http://127.0.0.1:8000/v1/chat/completions
```

## 테스트

```bash
python -m pytest -q
python -m ruff check .
```

Docker가 설치된 환경에서는 다음을 추가로 확인합니다.

```bash
docker build -t brave-tylenol-l2 .
docker run --rm -p 8000:8000 brave-tylenol-l2
```

## 제출 값

- Driver model name: `team-chatbot`
- 제출 SHA: 성능 개선 브랜치를 commit/push한 뒤 `git rev-parse HEAD`

실제 API 키와 대시보드 비밀번호를 소스, Dockerfile, README, 로그에 기록하지 않습니다.
