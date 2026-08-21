# BraveTylenol minimal Lunit L2 proxy

처음 0점 정적 서버를 유지하면서, 최종 답변만 Lunit L2가 생성하도록 연결한
최소 실행 버전입니다. MCP, 검색, 의료 라우팅, 정적 의료 답변은 아직 사용하지
않습니다.

## 현재 요청 흐름

```text
CoEval 의료 대화
  -> POST /v1/chat/completions
  -> 전체 messages를 Lunit/L2-preview에 한 번 전달
  -> L2가 생성한 choices[0].message.content
  -> OpenAI 호환 응답으로 CoEval에 반환
```

## 평가 환경 계약

- 저장소 루트의 `Dockerfile`로 실행
- `0.0.0.0:8000`에서 수신
- `GET /health`, `GET /healthz`
- `GET /v1/models`에서 Driver 모델 `team-chatbot` 제공
- `POST /v1/chat/completions`에서 비스트리밍 JSON 응답 제공
- Python 표준 라이브러리만 사용하며 Docker 빌드 중 패키지 설치 없음
- `ThreadingHTTPServer`로 CoEval의 동시 요청을 처리

## 시간초과 방지 설정

- L2 호출은 요청당 정확히 한 번
- 내부 재시도와 빈 답변 복구 호출 없음
- L2 요청 제한시간 35초
- L2 최대 생성 토큰 1,536
- L2 실패 시 Python 고정 의료 답변으로 대체하지 않음

CoEval이 자체적으로 여러 요청과 재시도를 수행하므로 드라이버 내부 재시도를
추가하지 않습니다.

## 인증과 보안

평가 요청에 `Authorization: Bearer ...`가 있으면 해당 토큰을 L2 요청에
전달합니다. 로컬 실행에서는 `.env`의 `LUNIT_FM_API_KEY`를 보조 수단으로
사용합니다.

```env
LUNIT_FM_API_URL=https://model.hackathon.lunit.io
LUNIT_FM_API_KEY=여기에_직접_입력
LUNIT_FM_MODEL=Lunit/L2-preview
```

`.env`는 Git과 Docker 이미지에서 제외됩니다. 코드와 로그에는 API 키, 대시보드
계정, 의료 질문 본문을 기록하지 않습니다.

## 실행

```bash
python main.py serve
```

다른 터미널에서 확인합니다.

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
curl \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LUNIT_FM_API_KEY" \
  -d '{"model":"team-chatbot","messages":[{"role":"user","content":"What should I do for a mild fever?"}]}' \
  http://127.0.0.1:8000/v1/chat/completions
```

## 테스트

```bash
python -m unittest discover -s tests -p "test_baseline_server.py" -v
```

테스트는 실제 L2를 호출하지 않고 다음 항목을 검증합니다.

- health/models/chat 응답 형식
- 전체 다중 대화 보존
- Bearer 및 `.env` 인증 전달
- 최대 토큰과 35초 제한
- L2 답변이 최종 응답에 그대로 들어가는지
- 잘못된 요청과 L2 오류의 안전한 처리
- 16개 동시 요청 격리

실제 API 연결 시험은 `.env`에 본인의 키를 넣은 로컬 환경에서 한 요청만 별도로
실행합니다.
