# Brave Tylenol — Lunit L2 Medical Chat

Lunit 해커톤 제출용 OpenAI 호환 의료 대화 서비스다. 기본 `fast` 모드는 로컬 위험도
라우터가 질문별 토큰 예산과 짧은 의료 지침만 고른 뒤 L2를 정확히 한 번 호출한다.
**최종 사용자 답변은 항상 Lunit L2가 생성한 텍스트를 그대로 반환한다.**

자세한 설계와 격리 환경 검토는 [아키텍처 문서](docs/architecture.md)를 참고한다.

## 1. 로컬 설치

Python 3.12 또는 3.13을 사용한다. 제출 Docker 이미지는 Python 3.13이다.

Windows PowerShell:

    py -3.12 -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
    Copy-Item .env.example .env

.env를 열어 LUNIT_FM_API_KEY를 실제 발급 키로 바꾼다. 대시보드 ID와 비밀번호는
대시보드 사용을 위해서만 보관하며 애플리케이션 런타임에는 사용하지 않는다. 실제 .env는
Git과 Docker build context에서 제외된다.

요청한 보안 규칙에 따라 .env.example에는 다음 다섯 항목만 들어 있다.

- LUNIT_FM_API_URL
- LUNIT_FM_API_KEY
- LUNIT_FM_MODEL
- DASHBOARD_USERNAME
- DASHBOARD_PASSWORD

## 2. 최소 L2 연결 확인

MCP를 전혀 사용하지 않고 L2 Chat Completions endpoint를 한 번 호출한다.

    .\.venv\Scripts\python.exe main.py check-l2

성공하면 L2 API connection OK와 L2가 만든 짧은 응답을 출력한다. 키가 없거나 잘못됐으면
Authorization 값이나 전체 응답 본문을 출력하지 않고 실패한다.

## 3. API 서버 실행

    .\.venv\Scripts\python.exe main.py serve

기본 주소는 http://127.0.0.1:8000 이다.

준비 상태:

    Invoke-RestMethod http://127.0.0.1:8000/health
    Invoke-RestMethod http://127.0.0.1:8000/v1/models

채팅 예시:

    $body = @{
      model = "team-chatbot"
      messages = @(@{ role = "user"; content = "고혈압 응급 증상은 무엇인가요?" })
    } | ConvertTo-Json -Depth 5
    Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/chat/completions -ContentType "application/json" -Body $body

지원하는 평가 endpoint:

- GET /health
- GET /v1/models
- POST /v1/chat/completions
- evaluator-facing model ID: team-chatbot
- 비스트리밍 요청만 지원

## 4. 실행 모드와 선택 설정

| 변수 | 기본값 | 설명 |
| --- | --- | --- |
| LUNIT_FM_API_URL | https://model.hackathon.lunit.io | L2 base URL |
| LUNIT_FM_MODEL | Lunit/L2-preview | L2 모델 |
| AGENT_MODE | fast | fast, rag 또는 passthrough |
| LUNIT_MCP_URL | 없음 | 주최 측이 공식 제공한 MCP URL |
| MAX_MCP_CALLS | 4 | 한 검색 실행의 최대 MCP 호출 수 |
| REQUEST_TIMEOUT_SECONDS | 110 | Driver 종료 전 제어된 응답을 위한 전체 요청 제한 |
| L2_RETRY_ATTEMPTS | 1 | 429/502/503/504 제한 재시도 |
| MAX_COMPLETION_TOKENS | 3072 | 평가 시간을 위한 L2 응답 토큰 상한 |
| LUNIT_REASONING_EFFORT | low | 긴 추론 지연을 줄이는 L2 reasoning effort |
| MAX_TOOL_RESULT_CHARS | 12000 | 개별 도구 결과 크기 제한 |
| MAX_EVIDENCE_CHARS | 32000 | L2에 전달할 전체 근거 제한 |
| LOG_LEVEL | INFO | 운영 로그 수준 |

.env.example 형식을 고정하기 위해 선택 설정은 예제 파일에 넣지 않았다. 공식 대회 문서에서
MCP URL을 확인한 경우에만 개인 .env 또는 배포 환경 변수로 LUNIT_MCP_URL을 추가한다.
`fast`는 MCP나 검색 판단 호출 없이 L2를 정확히 한 번만 호출한다. 이 ultrafast 브랜치는
일반 질문에 768토큰, 응급 표현 또는 긴 임상 문맥에 1,280토큰 상한을 사용한다. Python 라우터는 답변을 만들지
않고 실행 프로필만 선택한다. 빈 응답일 때에만 제한된 복구 호출을 한 번 허용한다.

`AGENT_MODE=rag`는 공식 MCP endpoint가 있을 때만 사용하는 비교 모드이며 더 느리다.
`AGENT_MODE=passthrough`는 의료 지침 없이 입력 대화를 L2에 바로 보내는 진단 모드다.

## 5. 검증

    .\.venv\Scripts\python.exe -m pytest -q
    .\.venv\Scripts\python.exe -m ruff check .
    .\.venv\Scripts\python.exe -m compileall -q app.py main.py lunit_hackathon
    .\.venv\Scripts\python.exe -m pip check

테스트는 실제 네트워크나 비밀값을 사용하지 않고 모의 L2/MCP 전송으로 다음을 확인한다.

- L2 Authorization, payload, 재시도, 오류 정리, 응답 검증
- .env 로딩, 자리표시자 거부, 비밀값 마스킹
- 위험도별 토큰 예산과 HealthBench 지향 단일 L2 호출
- L2 직접 답변과 검색 후 최종 L2 답변
- MCP 도구 발견, 인자 스키마 검증, 인용 선택, 호출/크기 제한
- MCP 장애 시 L2 폴백과 L2 오류의 올바른 전파
- OpenAI 호환 응답, HTTP 상태, 로그의 의료 텍스트/비밀정보 배제

## 6. Docker 제출

Docker가 설치된 환경에서:

    docker build -t brave-tylenol:lunit .
    docker run --rm -p 8000:8000 --env-file .env brave-tylenol:lunit

이미지는 비루트 사용자로 실행되고 .env, 테스트, 문서, 캐시를 포함하지 않는다. API Key를
Dockerfile의 ARG/ENV로 빌드하지 않는다.

이 브랜치는 120~180단어 답변을 목표로 한다. 실제 L2 지연은 모델 서버 부하에 따라 달라지므로 전체 3분 완료를 보장하지 않는다. 301개를
동시성 16으로 3분 안에 생성하려면 요청당 평균 약 9.6초가 필요하다. 제출 전 실제 L2로
대표 질문의 지연과 빈 응답률을 측정해야 한다.

## 7. 격리 환경 원칙

런타임은 일반 인터넷 API, 웹 검색, 원격 벡터 DB, 외부 인증 서비스에 의존하지 않는다.
네트워크 호출은 대회에서 허용되는 Lunit L2와 구성된 공식 MCP endpoint로 한정한다. MCP가
없어도 L2-only 경로로 동작한다. HealthBench validation set을 탐색하거나 문항별 규칙과
답변을 하드코딩하지 않는다.

패키지 설치는 이미지 빌드 단계에서만 필요하다. 빌드 단계도 패키지 저장소에 접근할 수 없는
정책이라면 주최 측의 사전 빌드 이미지 또는 내부 패키지 미러 요구사항을 공식 규칙에서
확인해야 한다.

CoEval처럼 컨테이너에 키를 주입하지 않는 평가기는 요청의 `Authorization: Bearer ...`
헤더로 L2 자격증명을 전달할 수 있다. 서비스는 이 값을 요청 범위에서만 사용하며 저장하거나
로그에 남기지 않는다.

## 8. 보안 체크리스트

- 실제 API Key, ID, 비밀번호를 소스·README·Dockerfile에 쓰지 않는다.
- .env를 Git에 추가하지 않는다.
- 의료 질문, 대화, MCP 근거, Authorization 헤더를 로그에 남기지 않는다.
- 제출 전 git diff와 비밀 패턴 검사를 수행한다.
- 채팅 데이터는 저장하거나 요청 간 캐시하지 않는다.
