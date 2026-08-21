# Brave Tylenol — Lunit L2 Medical Chat

Lunit 해커톤 제출용 OpenAI 호환 의료 대화 서비스다. 기본 RAG 모드에서 L2가 검색 필요성을
판단하고 공식 제공 MCP 도구를 사용할 수 있으며, **최종 사용자 답변은 항상 Lunit L2가
생성한 텍스트를 그대로 반환한다.**

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
| AGENT_MODE | rag | rag 또는 passthrough |
| LUNIT_MCP_URL | https://mcp.hackathon.lunit.io/mcp | 주최 측 공식 MCP URL(빈 값이면 직접 생성) |
| MAX_MCP_CALLS | 4 | 구성 가능한 상한(평가 지연 방지를 위해 실행당 최대 2회로 추가 제한) |
| REQUEST_TIMEOUT_SECONDS | 110 | Driver 종료 전 제어된 응답을 위한 전체 요청 제한 |
| L2_RETRY_ATTEMPTS | 2 | 429/502/503/504 제한 재시도(최대 총 3회 호출) |
| MAX_COMPLETION_TOKENS | 3072 | 평가 시간을 위한 L2 응답 토큰 상한 |
| LUNIT_REASONING_EFFORT | low | 긴 추론 지연을 줄이는 L2 reasoning effort |
| MAX_TOOL_RESULT_CHARS | 12000 | 개별 도구 결과 크기 제한 |
| MAX_EVIDENCE_CHARS | 32000 | L2에 전달할 전체 근거 제한 |
| LOG_LEVEL | INFO | 운영 로그 수준 |

기존 baseline 환경의 `HARNESS_MODE`, `MAX_TOOL_CALLS`, `UPSTREAM_TIMEOUT_SECONDS`,
`HARNESS_LOG_LEVEL`도 동일한 설정의 호환 별칭으로 인식한다.

.env.example 형식을 고정하기 위해 선택 설정은 예제 파일에 넣지 않았다. 제출 컨테이너는
공식 대회 MCP 주소를 기본으로 사용하고, 필요할 때 배포 환경 변수 LUNIT_MCP_URL로 덮어쓴다.
MCP URL을 빈 값으로 설정했거나 MCP가 실패하면 일반화된 의료 안전 프롬프트를 포함한 L2 직접 생성으로
자동 전환한다. MCP URL이 처음부터 없으면 검색 판단 호출도 생략하므로 L2를 정확히 한 번만
호출한다. 선택적 검색 단계는 전체 요청 제한의 45%, 최대 45초까지만 사용하여 MCP나 검색
플래너가 느릴 때도 직접 L2 폴백을 실행할 시간을 남긴다.

AGENT_MODE=passthrough는 입력 대화를 L2에 바로 보내는 비교/진단 모드다. 최종 제출 기본값은
rag다.

## 5. 검증

    .\.venv\Scripts\python.exe -m pytest -q
    .\.venv\Scripts\python.exe -m ruff check .
    .\.venv\Scripts\python.exe -m compileall -q app.py main.py lunit_hackathon
    .\.venv\Scripts\python.exe -m pip check

테스트는 실제 네트워크나 비밀값을 사용하지 않고 모의 L2/MCP 전송으로 다음을 확인한다.

- L2 Authorization, payload, 재시도, 오류 정리, 응답 검증
- .env 로딩, 자리표시자 거부, 비밀값 마스킹
- L2 직접 답변과 검색 후 최종 L2 답변
- MCP 도구 발견, 인자 스키마 검증, 인용 선택, 호출/크기 제한
- MCP 장애 시 L2 폴백과 L2 오류의 올바른 전파
- OpenAI 호환 응답, HTTP 상태, 로그의 의료 텍스트/비밀정보 배제

## 6. Docker 제출

Docker가 설치된 환경에서:

    docker build -t brave-tylenol:lunit .
    docker run --rm -p 8000:8000 --env-file .env brave-tylenol:lunit

이미지는 비루트 사용자로 실행되고 .env, 테스트, 문서, 캐시를 포함하지 않는다. API Key를
Dockerfile의 ARG/ENV로 빌드하지 않는다. 베이스 이미지는 Python 3.13.15 slim-trixie로,
주요 런타임 패키지는 로컬 검증 버전으로 고정해 평가 시 SDK 변경을 방지한다.

현재 개발 PC에는 Docker CLI가 없어 실제 이미지 빌드는 실행하지 못했다. 대신 고정한 공식
베이스 태그의 존재, Linux amd64/Python 3.13용 전체 의존성 wheel 해석, `.env`가 없는
컨테이너 유사 환경의 애플리케이션 import를 확인했다. 제출 파이프라인에서 실제 build와
`/health`, `/v1/models`, Bearer가 전달된 L2 연결을 마지막으로 재확인해야 한다.

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
