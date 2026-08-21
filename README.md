# BraveTylenol no-timeout Korean baseline

이 브랜치는 점수보다 실행 안정성을 우선하는 비상용 기준선입니다. 영어로 된
HealthBench 입력을 포함해 어떤 채팅 요청이 들어와도 외부 API, Lunit L2, MCP,
검색, 재시도를 호출하지 않고 고정된 한국어 응답을 즉시 반환합니다.

## 평가 환경 계약

- 저장소 루트의 Dockerfile로 실행
- 0.0.0.0:8000에서 수신
- GET /health, GET /healthz
- GET /v1/models
- POST /v1/chat/completions
- model 생략, 임의 추가 필드, stream=true, 빈 본문, 잘못된 JSON도 채팅
  엔드포인트에서는 HTTP 200의 일반 JSON completion으로 처리
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

Docker가 있는 환경에서는 다음처럼 네트워크를 끊고 확인할 수 있습니다.

~~~bash
docker build -t brave-tylenol-baseline .
docker run --rm --network=none -p 8000:8000 brave-tylenol-baseline
~~~

## 중요한 제한

대회 문서의 공식 규칙은 최종 답변을 Lunit/L2-preview가 생성하도록 요구합니다.
이 브랜치는 그 규칙을 의도적으로 충족하지 않는 정적 0점 기준선이며, API 기동과
응답 형식만 보장하기 위한 비상용입니다. 해당 규칙을 지켜 점수를 얻으려면 기존
L2 기반 제출 브랜치를 사용해야 합니다.
