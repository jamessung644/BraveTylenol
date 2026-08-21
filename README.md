# BraveTylenol direct-default L2 submission with opt-in MCP

`baseline/024-hybrid`의 동시성·지연시간 대조군을 바탕으로, 환경 설정이 없는 제출
Docker에서는 Lunit L2를 한 번만 호출하는 direct 경로를 사용한다. 공식 MCP Retrieval을
사용하는 hybrid/RAG 경로는 `AGENT_MODE`로 명시적으로 활성화한다. 실제 Docker image는
`main.py`가 아니라 FastAPI `app.py`를 `uvicorn`으로
실행한다. `main.py`는 이전 direct-only 대조군을 재현하기 위한 저장소 파일이며 image에
포함되지 않는다.

핵심 목표는 두 가지다.

- 일반 의료 질문과 응급 질문은 한 번의 L2 생성으로 빠르게 답한다.
- KCD·MFDS·HIRA·법령·현행 지침·논문·명시적 출처 요청처럼 기억만으로 답하면 안 되는
  질문은 제한된 MCP/RAG 경로로 보내되, MCP 지연이 최종 답변 시간을 잠식하지 않게 한다.

어느 경로에서도 Python이 의료 답변을 합성하지 않는다. 사용자에게 반환되는 최종 의료
텍스트는 Lunit L2가 작성한다.

## 평가 환경 계약

Docker image는 다음 인터페이스를 제공한다.

- `0.0.0.0:8000`
- `GET /health`, `GET /healthz`, `GET /readyz`
- `GET /v1/models`
- `POST /v1/chat/completions`
- 형식이 유효한 환경 `LUNIT_FM_API_KEY`가 요청 Bearer보다 우선
- 환경 credential이 없거나 L2가 401/403을 반환한 경우에만 형식이 유효한 요청 Bearer를
  한 번의 보조 후보로 사용하며, 같은 request의 Model과 MCP에 같은 후보를 전달
- 환경 credential이 없는 evaluator bearer-only 실행도 `/readyz`에 형식상 유효한
  `Authorization: Bearer ...`를 공급하면 static readiness 200
- 유효한 credential이 없으면 process liveness는 유지하지만 `/readyz`와 chat은 503
- 평가용 model ID `team-chatbot`과 upstream model ID `Lunit/L2-preview`
- non-streaming OpenAI-compatible JSON completion
- 모든 응답의 `X-Request-ID`

요청 body 크기·JSON 깊이·중복 key·비유한 수·지원하지 않는 model/stream을 경계에서
검증한다. Docker context는 `app.py`, `lunit_hackathon/`, `requirements.txt`만 image에
복사하며 테스트·문서·canonical 원문·`main.py`는 실행 image에서 제외한다.

CoEval 확장 입력을 위해 대화 선두의 `system`/`developer` context, `text`·`input_text`
content part, `max_completion_tokens`와 무해한 metadata/sampling field를 수용한다. 호출자가
준 system-like text는 compiled medical system 권한을 얻지 않고 명시적인 untrusted user
context로 downgrade된다. Client가 보낸 tool/function protocol field와 tool message는 계속
400으로 거절한다.

## 기본 direct 라우팅과 opt-in hybrid

| 입력 유형 | direct 기본 경로 | hybrid opt-in 경로 |
| --- | --- | --- |
| 일반 증상, 생활습관, 안정적인 건강 상식 | direct L2 1회 | 동일 |
| 의식 저하·호흡 곤란·흉통·과량복용 등 응급 표현 | emergency L2 1회 | 검색 없이 동일 |
| KCD/상병코드, MFDS 허가, HIRA 급여·약가·수가, 대한민국 법령 | direct L2 1회 | MCP/RAG 후보 |
| 최신 지침·논문·연구 또는 명시적인 근거·출처·인용 요청 | direct L2 1회 | MCP/RAG 후보 |

`hybrid`는 direct가 기본인 고정밀 routing이다. “최신”, “현재”, “출처” 같은 넓은 단어
하나만으로 RAG를 강제하지 않으며 실제 근거 요청이나 승인된 행정·법률 영역과 결합된 경우에
선택한다. 최근 두 사용자 turn만 routing 판단에 사용해 짧은 대명사 연결은 보존하면서 오래된
대화의 키워드가 현재 질문을 불필요하게 RAG로 보내지 않게 한다.

기본값은 `AGENT_MODE=direct`이며 MCP endpoint를 호출하지 않는다. `AGENT_MODE=hybrid`는
위 source-dependent routing을 활성화한다. Hybrid와 forced RAG 모두 admission 4개가 이미
사용 중이면 대기열에서 최종 답변 시간을 소모하지 않는다. source-dependent 요청은 fresh
`mcp_failure_final`로, 그 밖의 요청은 direct L2로 전환한다. `AGENT_MODE=rag`는 모든 정상
질문에 Retrieval 도구를 노출하는 진단용, `AGENT_MODE=passthrough`는 검색 조정 없이 안전
system prompt를 붙이는 진단용이다.

## 시간·동시성 예산

| 항목 | 기본값 | 역할 |
| --- | ---: | --- |
| 전체 요청 | 165초 | API 요청의 절대 상한 |
| L2 generic/default attempt | 45초 | direct·passthrough 및 단계별 override가 없는 model 호출 상한 |
| L2 retry | 0회 | 재시도 꼬리 지연 제거 |
| RAG initial application-tool | 25초 | retrieval query 생성 단계 |
| forced-tool retry | 10초 | source-dependent 요청에서 tool call 누락 시 한 번만 강제 |
| emergency final | 45초, 최대 1,536 tokens | deterministic guard가 선택한 fresh no-tool 최종 단계 상한 |
| Retrieval hard slice | `min(50초, 165초 × 0.31)` = 50초 | MCP와 Retrieval planner 전체 격리 |
| Retrieval planner L2 | attempt당 25초 | MCP call 계획과 local finalization 단계; 전체 Retrieval 50초 상한 안에서 동작 |
| final Generation | 45초, 최대 3,072 tokens | evidence/no-evidence 이후 사용자 답변 reserve |
| clean final recovery | 30초, 최대 1,536 tokens | final 출력의 구조·citation 검증 실패 또는 최초 final timeout 시 fresh no-tool 재작성 상한 |
| RAG admission | 4개 | 느린 source-dependent 경로의 동시 진입 제한 |
| MCP remote call | 기본 1회, release ceiling 3회 | 도메인별 완결 단계만 허용하고 무제한 loop 방지 |
| L2 동시 호출 | 16개 | CoEval 동시성에 맞춘 upstream 보호 |
| MCP 동시 연결 | 16개 | connection/session 점유 제한 |
| 요청 completion 상한 | 최대 4,096 tokens | decision/planner는 1,536, 일반 final은 3,072, 응급·복구는 1,536으로 phase별 제한 |

각 phase timeout에는 model semaphore 대기시간도 포함된다. `tool_decision`의 출력은 사용자에게
반환하지 않는다. `direct_final`, `post_retrieval_final`, `mcp_failure_final`, `emergency_final`은
항상 frozen inbound와 해당 final context로 새 no-tool transcript를 만들고, 구조·finish reason·
tool protocol·citation 검증에 실패하면 invalid draft를 포함하지 않은 `clean_recovery_final`을
같은 absolute deadline 안에서 정확히 한 번 실행한다. Emergency final은 MCP를 호출하거나
normal Retrieval route를 새로 시작하지 않는다. 이 no-evidence 응급 phase는 실제로 조회하지 않은
최신·공식 출처·URL·학회·저널·법령을 확인했다고 단정하거나 새 경구약·구체 용량을 시작하라는
출력을 거부한다. 출혈 안내는 지속 직접압박과 현지 응급번호·안전한 위치·dispatcher 지시를
우선한다. Recovery도 invalid이면 Python 의료 fallback 없이
sanitized 502로 종료한다.

MCP client 자체를 무기한 기다리게 두는 구조가 아니다. Discovery·planning·remote call 전체를
50초 Retrieval hard slice 안에 격리하고, 초과·연결 실패·schema 오류는 검증된 실패 상태로
단조 하향한다. 이후 frozen inbound와 failure context에서 fresh `mcp_failure_final` L2 request를
만들어 출처를 찾았다고 가장하지 않는 최종 답변을 작성한다. Tool-decision·MCP transcript와 raw
protocol은 이 final request나 사용자 출력에 포함하지 않는다. 따라서 MCP 장애가 user-visible
end-to-end timeout으로 확대되는 경로를 줄인다.

## MCP binding과 근거 처리

이 프로그램은 MCP server가 아니라 대회 공식 MCP의 client/orchestrator다.

1. 첫 RAG 요청에서 live tool discovery의 각 logical alias를 독립적으로 strict JSON schema와
   대조한다. 누락·중복·drift·lossless projection 불가 alias만 격리하고 미검증 도구는 노출하지
   않는다. 한 도메인과 무관한 도구 하나의 변화가 전체 MCP 경로를 중단시키지 않는다.
2. 격리 후 검증된 binding registry는 process lifetime 동안 공유해 이후 요청의
   discovery/compile 비용을 없앤다. 질문에 필요한 도메인 도구가 하나도 남지 않으면 그
   route만 근거 없음으로 안전하게 종료한다. 같은 도메인의 일부 alias만 남으면 해당 alias만
   설명하는 안전한 prompt fragment를 만들고, 빠진 후속 열람 단계의 결과는 final 근거로
   과장하지 않는다.
3. 각 query에는 KCD, MFDS, HIRA, 법령, ADR 또는 일반 guideline/research 중 관련된 최소
   tool subset만 Retrieval L2에 노출한다.
4. remote MCP call은 기본 한 번이다. 실험·release artifact와 도메인 ceiling 안에서 최대
   세 번까지 설정할 수 있고, `finalize_retrieval`은 endpoint로 보내지 않는 local finalizer다.
   `AGENT_MODE=hybrid`만 설정하면 `MAX_MCP_CALLS=1`이 유지된다. MFDS 2-hop 또는
   guideline·research·법령 3-hop 실험은 `MAX_MCP_CALLS=2|3`을 별도로 명시해야 한다.
5. 실제 tool result에서 관찰한 `cite_uid`만 선택하고 중복·크기·citation 범위를 검증한다.
   근거가 없으면 UID나 출처를 합성하지 않는다.
6. 최종 답변은 허용된 숫자형 citation만 사용할 수 있으며 검증 실패 시에도 Python이 답변을
   고쳐 쓰지 않고 frozen inbound/context의 L2 clean recovery를 정확히 한 번 실행한다.

## 자연어 입력 내성

한국어 의료 질문은 완성된 문장일 필요가 없다. prompt artifact와 routing layer는 다음 입력을
유효한 요청으로 다룬다.

- 맞춤법·띄어쓰기·키보드/음성 전사 오류와 반복 문자
- “머리아픔 어제부터 약 뭐먹지” 같은 단어 나열과 조사 생략
- 목적어·시간·대상자가 앞뒤로 바뀐 문장 도치
- 구어체·축약어·방언·이모지·한국어/영어 혼용
- 여러 대상자, 시점 또는 질문이 섞인 입력

원문은 보존하고 NFKC·공백 정리는 routing의 표면형 비교에만 사용한다. 약물·성분·제품,
용량·농도·단위, 양성/음성, 신체 부위, 임신·연령·체중, 노출 물질처럼 위해를 바꾸는 오타는
조용히 하나로 확정하지 않는다. 모호성이 결론이나 긴급도를 바꾸면 안전한 조건부 답변과 핵심
확인 질문 1~3개를 우선한다. 응급 단어 나열·축약형은 별도 compact guard로 포착하되 과거
증상·부정·인용을 현재 응급으로 단정하지 않도록 L2가 다시 판단한다.

## KDCA ASP/KONAS 자료의 경계

[KDCA 항생제 사용관리 자료 목록](https://www.kdca.go.kr/kdca/2857/subview.do?enc=Zm5jdDF8QEB8JTJGYmJzJTJGa2RjYSUyRjQ5JTJGYXJ0Y2xMaXN0LmRvJTNGcmdzQmduZGVTdHIlM0QlMjZjc3JmVG9rZW4lM0QzMDAzODA1ZS03MjNkLTRiMDktOTdjMy0wNjI3YTU2ODBhYTklMjZmaW5kT3Bud3JkJTNEJTI2ZmluZFdvcmQlM0QlMjZyZ3NFbmRkZVN0ciUzRCUyNmZpbmRUeXBlJTNEJTI2ZmluZENsU2VxJTNEJTI2cGFnZSUzRDElMjY%3D)과
[KDCA 적정사용을 위한 각종 정책](https://www.kdca.go.kr/kdca/3515/subview.do)은 ASP 및
KONAS 관련 source role을 정하는 정책 참고자료다. 기관 운영자료·집계·교육·시범사업을 개인
환자의 항생제 필요성, 약제 선택, 용량 또는 기간 근거로 승격하지 않는 경계를 prompt에
반영했다.

다만 현재 release에는 KDCA 게시물·첨부를 runtime에서 검색하고 observed `cite_uid` ledger에
등록하는 승인된 evidence adapter 또는 local snapshot이 없다. 위 URL은 자동 수집 대상도,
MCP가 반환한 model-facing evidence도 아니다. 따라서 exact 게시물 내용이나 최신 운영 사실을
조회했다고 주장하지 않으며, KDCA coverage가 필요한 질문은 확인된 공식 MCP 근거만 사용하고
없으면 `coverage_gap`/`no_evidence`로 남긴다.

## 공식 대회 문서

구현·제출 시 다음 Dashboard 계약을 우선한다.

- [Lunit FM quick start](https://dashboard.hackathon.lunit.io/quick-start/lunit-fm)
- [MCP tools quick start](https://dashboard.hackathon.lunit.io/quick-start/mcp-tools)
- [Rules](https://dashboard.hackathon.lunit.io/rules)
- [Model quick start](https://dashboard.hackathon.lunit.io/quick-start/model)

Dashboard tool schema 또는 규칙이 바뀌면 canonical source, compiled runtime artifact, binding
manifest와 테스트를 같은 변경으로 갱신해야 한다.

## 실행과 검증

~~~bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --no-access-log
~~~

~~~bash
docker build -t brave-tylenol-hybrid .
docker run --rm -p 8000:8000 -e LUNIT_FM_API_KEY brave-tylenol-hybrid
~~~

~~~bash
python -m pytest -q
python scripts/compile_runtime_artifacts.py --check
~~~

`GET /health`는 process liveness만 확인한다. `GET /readyz`는 local artifact·공식 endpoint 설정과
형식상 유효한 환경 또는 request Bearer credential 존재를 확인하지만 외부 L2/MCP를 미리 호출하지
않는다. 따라서 환경 key가 없는 evaluator bearer-only 실행도 유효한 Bearer header를 함께 보내면
readiness 200이고, 어느 credential도 없는 격리 실행은 기동·liveness만 성공하며 readiness/chat은
503이어야 한다.
실제 Lunit network에서 제출 전 live canary를 별도로 실행해야 한다.

## 벤치마크 해석

[2026-08-21 L2 authentication local performance gate](docs/benchmarks/2026-08-21-l2-auth-local.md)는
`baseline/024-hybrid`가 계승한 direct-only 대조군의 16동시 요청 기록이다. 첫 측정은 16/16
실 L2 응답, 중앙 25.965초, 최대 30.317초였고 별도 evaluator simulation도 문서에 그대로
보존되어 있다. 이 기록은 L2 연결·동시성·지연시간 gate이지 현재 MCP 경로의 품질 또는 공식
HealthBench 점수가 아니다.

[2026-08-22 live MCP paired promotion decision](docs/benchmarks/2026-08-22-live-mcp-paired-no-go.md)은
고정 synthetic 의료 질문의 실제 L2/MCP 비교 결과다. 유효 retrieval 6쌍에서 hybrid가
32점, direct가 34점이었고 평균 지연은 각각 41.5초와 20.2초였다. 별도 응급·특이 13쌍은
총점이 102 대 101이었지만 assistant-history topic-switch 안전성 회귀가 있어 전체 gate를
통과하지 못했다. 따라서 이 commit은 fast direct를 기본으로 유지하고 hybrid/RAG를 명시적
진단·실험 opt-in으로만 보존한다.
