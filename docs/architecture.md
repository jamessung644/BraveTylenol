# Architecture

이 문서는 `baseline/024-hybrid` 위에 이식한 제출 runtime의 실행 구조를 설명한다. 구현 계약의
최종 authority는 [Lunit FM quick start](https://dashboard.hackathon.lunit.io/quick-start/lunit-fm),
[MCP tools quick start](https://dashboard.hackathon.lunit.io/quick-start/mcp-tools),
[Rules](https://dashboard.hackathon.lunit.io/rules),
[Model quick start](https://dashboard.hackathon.lunit.io/quick-start/model)이며 Dashboard와 저장소
문서가 다르면 Dashboard를 우선한다.

## 목표와 불변 조건

가장 중요한 불변 조건은 **사용자에게 반환되는 의료 답변의 모든 문장이 Lunit L2가 생성한
텍스트여야 한다**는 것이다.

- Python은 요청 검증, 경로 선택, 도구 protocol, strict schema binding, 근거 크기·인용 검증,
  오류 mapping만 담당한다.
- direct 경로는 L2의 최종 생성 결과를 그대로 반환한다.
- RAG 경로는 MCP result를 신뢰하지 않는 evidence data로 frozen inbound에서 새로 만든
  post-retrieval final transcript에 넣고, 마지막 L2의 사용자용 텍스트를 반환한다.
- MCP 장애·초과시간에는 Python이 의료 답변이나 출처를 만들지 않는다. 구조화된 제한 상태를
  독립된 evidence-failure final prompt에 넣고 남겨 둔 시간으로 L2가 최종 답변을 작성한다.
- HealthBench 문항, 예상 답변 또는 문항별 routing을 식별하거나 하드코딩하지 않는다.

## 실제 image와 entrypoint

Docker는 `app.py`, `main.py`, `lunit_hackathon/`을 복사하고 다음 command로 FastAPI를 실행한다.

~~~text
uvicorn app:app --host 0.0.0.0 --port 8000 --no-access-log
~~~

`main.py`는 direct-only baseline의 실행 코드이지만 제출 image에서는 entrypoint나 의료
fallback provider로 사용하지 않고, 주최 측이 허용한 임시 embedded credential만 modular
Settings에 제공한다. `tests/`, `docs/`, `canonical_sources/`, `runtime_sources/`는 image에
포함되지 않는다. Git에는 `"fffffff"`만 두고 실제 값은 사용자가 제출 branch에서 직접
교체한다. Runtime prompt와 binding manifest는 build 전에 compile되어
`lunit_hackathon/runtime_artifacts/` 안의 검증된 artifact로 image에 들어간다.

## 요청 흐름

~~~text
POST /v1/chat/completions
        |
        v
HTTP/JSON boundary 검증 -- 실패 --> 4xx
        |
        v
mode selection (default hybrid; direct opt-out)
        |
        +-- direct 또는 일반/응급/저위험·안정 정보 --> direct Generation L2 1회 --> L2 text
        |
        +-- source-dependent --> RAG admission (최대 4)
                                  |
                                  +-- full + source-dependent --> mcp_failure_final L2 1회
                                  +-- full + non-source --> direct L2 1회
                                  |
                                  +-- admitted
                                        |
                                        v
                              Generation L2가 retrieval 필요성 선택
                              harness가 검증된 사용자 원문으로 query 구성
                                        |
                                        v
                        process-cached strict MCP registry resolve
                                        |
                                        v
                           query별 최소 tool subset 선택
                                        |
                                        v
                       Retrieval L2 -- MCP configured 기본/ceiling 3회
                                        |
                                        v
                      local finalize + observed cite_uid 검증
                                        |
                                        v
                       frozen inbound + evidence final context
                                        |
                                        v
                    phase-specific final Generation L2 --> validator
                                        |
                                        +-- invalid --> clean fresh recovery 정확히 1회
                                                            |
                                                            +-- invalid/malformed/timeout + deadline 남음
                                                                   --> fixed-indicator safe_completion_final L2 1회
                                        |
                                        v
                                  plain L2 text
~~~

기본값은 `AGENT_MODE=hybrid`다. 일반 질문은 direct final fast path로 유지하고,
source-dependent 질문만 Retrieval로 보낸다. `direct`는 MCP를 완전히 끄는 비교·복구
variant이고, `rag`는 정상 질문 전체를 Generation tool trajectory에 넣는 진단 variant다.
`passthrough`도 검색 조정만
건너뛰며 direct final prompt와 동일한 plain-answer validator/recovery를 통과한다.

응급 표현은 application 기능이나 내부 route 출력 형식을 포함하지 않는 독립 emergency final
prompt를 사용해 검색 때문에 즉시 행동 안내가 늦어지지 않게 한다. Guard가 모호한 과거·부정·
인용을 현재 응급으로 오인해도 같은 prompt가 원문을 다시 판단해 자연어로 답한다. 이 phase는
근거를 조회하지 않으므로 조회하지 않은 공식 출처·URL·학회·저널·법령 단정과 새 경구약·구체
용량 지시를 validator가 거부한다. 현재 위험에서는 현지 응급번호(대한민국 위치 확인 시 119),
안전한 위치, dispatcher 지시를 우선하며 통제되지 않는 외부 출혈은 지속 직접압박을 중심으로 한다.

Generation prompt는 `tool_decision`, `direct_final`, `post_retrieval_final`,
`mcp_failure_final`, `emergency_final`, `clean_recovery_final`, `safe_completion_final` 일곱
artifact로 나뉘고 각각 SHA-256 검증된다. 근거 기능의 이름·schema·호출 지시는 tool-decision
phase에만 존재한다.
Final phase는 이전 decision system, application call, invalid assistant draft를 append하지 않고
항상 frozen inbound와 phase context에서 새 transcript를 만든다.

일반 Generation과 evidence의 model-facing v4 projection은 sparse하다. 선택적인 null·unknown·
빈 문자열·빈 객체 placeholder는 보내지 않고 의미가 있는 status·분류·근거만 보존한다. 이
생략은 임상 사실의 부재나 검증 완료를 뜻하지 않으며 최신 사용자 원문, history 순서,
evidence status를 변경하지 않는다.

## Hybrid routing 계약

Routing은 source가 필요한 가능성이 높은 영역에만 보수적으로 RAG를 허용한다.

- KCD·질병분류·상병코드
- MFDS·식약처 허가/품목/성분
- HIRA·심평원 급여/비급여/약가/수가
- 대한민국 법령·의료법·약사법·조문
- 최신/현행 확인이 붙은 진료지침·논문·연구
- “공식 근거”, “출처를 알려”, “인용해” 같은 명시적 source request

반대로 일반 증상·생활습관·안정적인 의학 상식, 불완전한 단어 나열, 넓은 의미의 “최신”이나
“출처” 단독 표현은 direct를 기본으로 한다. 최신 두 user turn만 NFKC와 공백 정리 후 비교해
도치·띄어쓰기 변화는 흡수하되, 오래된 대화 keyword가 현재 질문을 RAG로 끌고 가지 않게 한다.
이 routing은 진단기가 아니며 질문의 임상 의미를 수정하지 않는다.

## 자연어 입력 계약

Compiled Generation/Retrieval prompt는 오타, 띄어쓰기 오류, 음성 전사 오류, 단어 나열,
조사·주어 생략, 문장 도치, 구어체·방언·축약, 이모지, 한국어/영어 혼용을 유효한 자연어로
다룬다.

원문 사용자 진술과 내부 표면형 정리를 분리한다. 약·성분·제품·용량·단위·검사 극성·부위·
대상자·시간 관계는 인접 keyword나 어순만으로 결합하거나 자동 교정하지 않는다. 가능한 해석이
여러 개이고 결론·긴급도·금기가 달라지면 unknown과 후보를 보존하고, 안전한 조건부 안내 뒤
가장 중요한 확인 질문 1~3개만 묻는다. Retrieval query도 사용자에게 없던 대상 인구·관할·
기준일을 발명하지 않으며 불확실하면 `partial` 또는 `no_evidence`로 끝낸다.

## MCP discovery, binding, 최소 노출

Lifespan에서 하나의 `ToolBindingRegistry`를 만들고 모든 요청이 공유한다. 최초 RAG 연결에서
live discovery의 각 canonical logical alias를 raw schema와 strict wrapper에 독립적으로
compile한다. 누락·drift·lossless projection 불가 alias는 그 alias만 quarantine하고, 검증된
불변 tuple을 process lifetime 동안 재사용한다. 사용자 입력으로 tool 이름을 만들거나 미등록
tool을 L2에 노출하지 않는다. 질문 도메인의 도구가 전부 격리되었을 때만 해당 route를
근거 없음으로 안전하게 종료한다. 일부 alias만 남은 route는 살아 있는 alias만 exact하게
설명하는 generic fragment를 만들고, 필요한 후속 열람 도구가 없으면 discovery 결과를 최종
근거로 승격하지 않는다.

검증된 registry에서 query domain별 최소 subset만 Retrieval L2에 제공한다.

| Domain | 노출되는 logical tool 묶음 |
| --- | --- |
| KCD | code search + exact name |
| MFDS | ingredient/permission lookup + indication/source metadata (최대 2-hop) |
| HIRA | update + disease code + drug price |
| 법령 | law search + article list + article body |
| ADR | drug information |
| 일반 guideline | index keyword/relevant node discovery + page content (최대 3-hop) |
| research | data-source discovery + source detail + vector query (최대 3-hop) |

Retrieval L2의 MCP remote call configured budget과 release artifact ceiling은 모두 3이며,
실제 budget은 도메인별 1~3회 ceiling으로 다시 제한한다. Budget 뒤에는 strict local
`finalize_retrieval`을 강제한다. Finalizer는 MCP endpoint로 보내지 않으며 실제
result에서 관찰된 `cite_uid`, score 범위, 중복, item 수를 검증한다. Tool content는 8,000자,
전체 evidence는 12,000자로 제한한다.

기본 `MAX_MCP_CALLS=3`은 structured HIRA/ADR을 여전히 1-hop으로 유지하면서 MFDS의 필요한
2-hop, guideline의 discovery→page 2-hop, 법령·research의 3-hop 완결 경로를 허용한다.
Configured 값은 artifact ceiling과 domain ceiling을 넘지 못하며 `AGENT_MODE=direct`로 MCP를
즉시 비활성화할 수 있다.

## 시간 예산과 MCP stall 격리

| Budget | 기본값 | 설계 의도 |
| --- | ---: | --- |
| request deadline | 165초 | 모든 L2/MCP 단계의 절대 상한 |
| generic/default model attempt | 최대 145초 | 단계별 override가 없는 model 호출 상한; 항상 남은 전체 요청 deadline 이하 |
| model retry | 0 | tail latency 억제 |
| RAG initial application-tool | 25초 | retrieval query 생성 |
| forced-tool retry | 10초 | 명시적 source 요청에서 tool call 누락 시 한 번만 재강제 |
| emergency final | 명목 최대 145초, 최대 2,048 tokens | safe completion용 35초를 잠근 뒤 남은 network slice 사용 |
| retrieval hard slice | 최대 45초, 매 단계 동적 재계산 | `min(50초, request × 0.31, 남은 deadline − final reserve)`로 MCP와 planner 격리 |
| final reserve | 기본 120초 | Retrieval 전 반드시 보존; 짧은 비운영 deadline에서는 request의 75% |
| retrieval planner L2 | attempt당 25초 | remote call 계획 및 local finalization; 동적 Retrieval 상한 적용 |
| final Generation | 명목 최대 145초, 최대 2,048 tokens | safe completion 30초 + HTTP guard 5초를 먼저 잠가 최초 direct network slice는 최대 130초 |
| clean final recovery | 명목 최대 145초, 최대 2,048 tokens | 잠근 35초 위의 남은 시간에서만 최대 1회; reserve 소진 시 전송하지 않음 |
| safe completion final | 최대 30초, 최대 256 tokens | fixed indicator만으로 한국어 1~2문장 생성하고 outer 응답 정리 5초 보존; tool·retry 없음 |
| RAG admission | 4 (설정 가능 1~4) | slow trajectory 동시 진입 hard limit; C16 direct/final 용량 보존 |
| MCP call | configured 기본 3, effective ceiling 1~3 | artifact·설정·도메인 중 최솟값으로 tool loop 상한 |
| model semaphore | 16 | CoEval 동시성에 맞춘 보호 |
| MCP session semaphore | 16 | connect/discovery/call 전체 점유 제한 |

각 phase timeout에는 model semaphore acquire 대기시간도 포함된다. RAG의 initial,
forced-tool, planner와 모든 final/recovery 단계는 client 내부 blank-completion recovery를 끄고
명시된 phase budget을 넘기지 않는다. Final content가 비어 있거나 application protocol 흔적을
포함하거나 종료 사유가 `stop`/미지정 이외이거나 인용 계약을 위반하면 frozen inbound에서
clean final recovery를 정확히 한 번 실행한다. Recovery도 invalid·malformed이거나 timeout이고
Final과 recovery의 network boundary는 35초 locked reserve를 공통 적용한다. Recovery를 시작할
여유가 없으면 사용자 의료 원문·draft·tool/evidence가 없는 fixed indicator transcript로
`safe_completion_final`을 정확히 한 번 실행한다. 이 phase의 호출·출력까지 실패할 때만
Python 의료 fallback 없이 sanitized upstream error로 종료한다.

MCP transport의 connect/discovery/call이 정지해도 Retrieval 전체 `asyncio.timeout`이 먼저
취소하고 `retrieval_timeout` no-evidence로 전환한다. Hybrid와 forced RAG admission은 사용
가능한 permit을 타이머 없이 즉시 획득하고, 4개가 모두 점유된 경우에는 waiter를 만들지 않고
하향한다. 이 상한은 환경 설정으로 높일 수 없다.
이 둘은 MCP를 무한 대기시키는 것이 아니라
**MCP 때문에 최종
L2 응답 기회를 잃지 않도록** first-routing slice와 final-generation slice를 예약하는 방식이다.

RAG가 시작된 뒤 Retrieval 실패 시 일반 direct prompt로 몰래 전환하지 않는다. Frozen inbound와
구조화된 실패 상태만 사용해 전용 evidence-failure final transcript를 만들고 L2가 답한다.

## 평가 API 계약

- `POST /v1/chat/completions`의 `model`은 생략할 수 있다.
- `GET /v1/models`는 `team-chatbot`과 `Lunit/L2-preview`를 반환한다.
- 선택적인 `max_tokens`/`max_completion_tokens`는 서버 상한 2,048을 늘릴 수 없다.
- 대화 선두의 `system`/`developer`, `text`·`input_text` content part와 허용된
  metadata/sampling field는 CoEval 호환 입력으로 받는다.
- 호출자 제공 system-like text는 untrusted user context로 downgrade하며 compiled medical
  system prompt를 덮어쓸 수 없다.
- Client-supplied `tools`, `tool_choice`, function/tool protocol과 tool message는 400으로
  거절해 내부 application-tool authority와 분리한다.
- Retrieval 계획 단계는 1,536 token으로 제한해 final answer token reserve를 보존한다.
- 응답 `finish_reason`은 L2의 실제 종료 사유를 보존한다.
- streaming은 지원하지 않으며 `stream=true`는 400으로 거절한다.
- 전체 request body는 600,000 byte, JSON 깊이는 32로 제한하고 duplicate key와 non-finite
  number를 거절한다.

## 컴포넌트

| 파일 | 책임 |
| --- | --- |
| `app.py` | FastAPI lifespan, OpenAI-compatible API, 전체 deadline과 semaphore |
| `main.py` | legacy direct 대조군 코드와 제출용 embedded credential source; entrypoint 아님 |
| `config.py` | .env/환경/main credential 설정과 비밀값 마스킹 |
| `artifacts.py` | compiled prompt·model/MCP manifest load와 hash 검증 |
| `l2_client.py` | async L2 Chat Completions, attempt deadline, 응답 검증 |
| `generation.py` | hybrid source routing, direct/RAG/emergency L2 trajectory, citation 검증 |
| `mcp_client.py` | MCP SDK Streamable HTTP 전송과 result normalization |
| `tool_bindings.py` | live schema의 strict projection, fingerprint와 name-plane binding |
| `retrieval.py` | registry cache, 최소 tool 선택, one-call budget, evidence ledger/finalizer |
| `orchestrator.py` | mode 선택, non-blocking RAG admission, direct/RAG 실행 |
| `prompts.py` | compiled 의료 안전·자연어·Retrieval instruction 노출 |
| `logging_config.py` | prompt·근거·자격 증명을 제외한 운영 로그 |

## 오류와 degrade 동작

| 상황 | 동작 |
| --- | --- |
| 유효한 환경·embedded main·request Bearer credential 모두 없음 | health는 200, readiness/chat은 503 |
| 환경 key 없는 evaluator가 유효한 Bearer로 `/readyz` 호출 | static readiness 200; 외부 dependency는 preflight하지 않음 |
| Initial final과 clean recovery의 timeout·invalid·malformed | deadline이 남으면 fixed-indicator `safe_completion_final` L2 1회 |
| `safe_completion_final`도 timeout·invalid·malformed | Python 의료 fallback 없이 세부정보를 숨긴 502/504 |
| 그 밖의 L2 전송/인증 오류 | Python 의료 fallback 없이 세부정보를 숨긴 upstream error |
| RAG admission hard limit(4) full + source-dependent | waiter/MCP 없이 fresh `mcp_failure_final` L2 |
| RAG admission hard limit(4) full + non-source | waiter/MCP 없이 direct L2 |
| MCP timeout/연결 실패 | frozen inbound + failure context의 fresh `mcp_failure_final` L2 |
| live registry 누락·중복·schema drift | MCP call 없이 retrieval dependency error |
| 잘못된 tool 이름/인자 | 실행하지 않고 구조화된 protocol error |
| MCP 호출 예산 소진 | local finalizer만 허용하고 현재 evidence 상태로 종료 |
| citation 없음·불일치 | fresh `clean_recovery_final` L2 1회; 계속 invalid이고 deadline이 남으면 `safe_completion_final`; Python 답변 합성 금지 |
| streaming 요청 | 400 |

## KDCA ASP/KONAS source boundary

[KDCA 항생제 사용관리 자료 목록](https://www.kdca.go.kr/kdca/2857/subview.do?enc=Zm5jdDF8QEB8JTJGYmJzJTJGa2RjYSUyRjQ5JTJGYXJ0Y2xMaXN0LmRvJTNGcmdzQmduZGVTdHIlM0QlMjZjc3JmVG9rZW4lM0QzMDAzODA1ZS03MjNkLTRiMDktOTdjMy0wNjI3YTU2ODBhYTklMjZmaW5kT3Bud3JkJTNEJTI2ZmluZFdvcmQlM0QlMjZyZ3NFbmRkZVN0ciUzRCUyNmZpbmRUeXBlJTNEJTI2ZmluZENsU2VxJTNEJTI2cGFnZSUzRDElMjY%3D)과
[KDCA 적정사용을 위한 각종 정책](https://www.kdca.go.kr/kdca/3515/subview.do)은 현재
prompt에서 ASP/KONAS의 source role과 적용 경계를 정하는 정책 참고자료다. 기관 운영자료,
항생제 사용 집계, 교육·시범사업은 개인 환자의 항생제 필요성·선택·용량·기간을 직접 지지하지
않는다. 관리료 보상 연계의 존재와 현재 대상·금액·청구 조건도 같은 claim이 아니며 후자는
적절한 최신 HIRA 근거가 필요하다.

현재 release에는 KDCA live 게시물/첨부 또는 local snapshot을 검색해 observed `cite_uid`를
만드는 승인된 adapter가 없다. 그러므로 두 URL과 canonical source map은 개발·policy fixture일
뿐 model-facing evidence가 아니다. KDCA의 정확한 최신 사실을 검색 가능한 것처럼 표현하지
않고 공식 MCP corpus에서 실제 근거를 찾지 못하면 `coverage_gap` 또는 `no_evidence`를
Generation에 전달한다.

## 격리 평가 환경

Runtime의 외부 network 대상은 compiled manifest에 고정된 Lunit L2 endpoint와 공식 MCP
endpoint 두 종류뿐이다. 일반 웹 검색, 상용 검색 API, cloud vector DB, remote analytics에는
의존하지 않는다. MCP가 느리거나 실패해도 final L2 reserve를 사용하지만, hybrid/rag manifest가
요구한 endpoint나 registry가 잘못된 경우 조용히 다른 remote source로 우회하지 않는다.

Image에는 `.env`, test, docs, cache, canonical Markdown를 복사하지 않는다. 제출 호환성을 위해
`main.py`만 credential source로 포함한다. `GET /readyz`는
static manifest 상태와 “live canary required”를 보여 줄 뿐 외부 dependency를 preflight하지
않는다. 제출 직전 Lunit network에서 model, MCP discovery/schema, 실제 one-call Retrieval과
final answer를 포함한 live canary가 필요하다.

## 보안과 개인정보

- 일반 배포의 API Key와 대시보드 자격 증명은 .env 또는 런타임 환경에 둔다. 이번
  organizer-approved 임시 제출에서는 사용자가 관리하는 `main.py` embedded key도 허용한다.
- .env와 모든 변형은 Git 및 Docker context에서 제외하고 .env.example만 허용한다.
- SecretStr로 설정 객체 표현에서 비밀을 마스킹한다.
- 요청 로그에는 생성된 request ID, 메서드, 경로, 상태, 시간, 오류 클래스만 기록한다.
- 의료 질문, 대화, MCP 근거 본문, Authorization 헤더는 기록하거나 저장하지 않는다.
- MCP 결과와 사용자 입력은 시스템 지시가 아닌 신뢰하지 않는 데이터로 취급한다.

## 벤치마크의 위치

[`docs/benchmarks/2026-08-21-l2-auth-local.md`](benchmarks/2026-08-21-l2-auth-local.md)는
16 concurrent request direct-only 대조군의 실 L2 연결·지연시간 gate다. 원 기록은 변경하지
않으며 현재 hybrid의 MCP 품질 또는 공식 HealthBench score로 재해석하지 않는다. Hybrid는 그
대조군의 one-call fast path를 일반 질문에 유지하고 source-dependent subset에만 추가 비용을
쓰도록 설계했다.
