# Lunit L2·MCP 적용 청사진

> 상위 색인: [MODEL_INSTRUCTIONS.md](../../MODEL_INSTRUCTIONS.md)
> Dashboard 정합성 감사: [85_DASHBOARD_CONSISTENCY_AUDIT.md](85_DASHBOARD_CONSISTENCY_AUDIT.md)
> 확인 기준: 인증된 Dashboard의 [Lunit FM](https://dashboard.hackathon.lunit.io/quick-start/lunit-fm), [MCP tools](https://dashboard.hackathon.lunit.io/quick-start/mcp-tools), [규칙](https://dashboard.hackathon.lunit.io/rules), [Model API](https://dashboard.hackathon.lunit.io/quick-start/model), 2026-08-21 KST

이 문서는 분야별 임상·대화·RAG 지침을 실제 Lunit L2 harness에 적용하는 canonical 구현 청사진이다. Dashboard 계약과 이 문서가 다르면 Dashboard를 우선하고, prompt·adapter·fixture·이 문서를 같은 변경 단위로 갱신한다.

## 절대 불변 조건

1. L2는 Retrieval과 Generation을 **별도 Model 호출**로 실행한다. 두 system prompt나 tool set을 합치지 않는다.
2. Generation의 `tool_decision` phase에만 strict `retrieve_relevant_content` 하나를 등록한다. `direct_final`, `post_retrieval_final`, `mcp_failure_final`, `emergency_final`, `clean_recovery_final`, `safe_completion_final`에는 tool을 등록하지 않는다.
3. Retrieval L2에는 질문에 허용된 Lunit MCP tools와 harness 함수 `finalize_retrieval`만 등록한다. `finalize_retrieval`은 MCP tool이 아니다.
4. 사용자에게 보이는 최종 의료 답변은 반드시 L2가 생성한다. Harness·다른 생성 모델·template이 의료문을 대신 작성하지 않는다.
5. L2는 single-turn 성향이 강하므로 multi-turn 원문 보존, 주체·정정·상태 추적, query rewriting은 harness가 담당한다.
6. 평가 runtime은 외부 접근이 없는 격리 환경이다. Runtime internet 검색·package/model download·외부 vector DB에 의존하지 않는다.
7. Dashboard validation을 debug 경로로 사용하되 HealthBench holdout을 추정하거나 과도하게 reverse engineering하지 않는다.

## 실제 호출 토폴로지

```text
Evaluator / Patient Simulator의 OpenAI-compatible messages
  -> strict request validator
  -> 원문 history 보존 + clinical-input-v4/state 계산
  -> deterministic emergency guard
      -> emergency candidate: fresh emergency_final L2(no tool)
      -> normal: tool_decision L2(sole tool: retrieve_relevant_content)
            -> tool call이 없으면 fresh direct_final L2(no tool)
            -> valid tool call이면 사용자 원문 기반 context-complete query를 결정적으로 구성
                -> Retrieval L2(MCP tools + finalize_retrieval)
                -> bounded MCP loop
                -> finalize 결과 + 실제 tool-result provenance 검증
                -> 성공: fresh post_retrieval_final L2(no tool)
                -> 실패: fresh mcp_failure_final L2(no tool)
      -> final text 전체 검증
            -> invalid이면 frozen inbound + 같은 final context로
               fresh clean_recovery_final L2(no tool) 정확히 1회
            -> recovery도 invalid·malformed이거나 timeout이고 deadline이 남으면
               사용자 의료 원문·draft·tool/evidence가 없는 fixed indicator로
               fresh safe_completion_final L2(no tool/no retry) 정확히 1회
            -> valid L2 final text만 반환
```

`tool_decision`의 자연어 content와 application tool-call transcript는 사용자 답변이 아니며 어떤 final transcript에도 append하지 않는다. Emergency guard 오탐은 독립 `emergency_final` prompt가 원문의 부정·과거·인용·가정을 다시 판단해 자연어로 처리한다. Emergency request에 tool을 동적으로 추가하거나 내부 route 전환 envelope를 사용하지 않는다. 이 no-evidence phase에서는 조회하지 않은 최신·공식 출처나 URL·학회·저널·법령 단정, 새 경구약·구체 용량 시작 지시를 final validator가 거부하고 fresh clean recovery를 한 번만 허용한다. 현재 위험은 현지 응급번호, 안전한 위치, dispatcher 지시를 우선하며 통제되지 않는 외부 출혈은 지속적인 직접 압박을 중심으로 한다. Initial final과 clean recovery가 모두 실패한 뒤 deadline이 남을 때만 `safe_completion_final`을 30초·256 tokens, tool·retry 없음으로 한 번 실행한다. 이 request에는 normal/emergency 구분의 fixed indicator만 넣고, L2가 한국어 1~2문장으로 진단 대체 불가·전문가 확인 고지(응급이면 즉시 현지 응급서비스 안내)를 작성한다. 이 phase도 실패하면 sanitized upstream error를 반환하며 Python이 의료문을 만들지 않는다.

## Model API adapter

| 항목 | 고정값·규칙 |
| --- | --- |
| Base URL | `https://model.hackathon.lunit.io` |
| Endpoint | `POST /v1/chat/completions` |
| Model | `Lunit/L2-preview` |
| 인증 | `Authorization: Bearer <LUNIT_FM_API_KEY>` |
| Generation tools | `tool_decision`: `retrieve_relevant_content` 하나, 여섯 final phase: 없음 |
| Retrieval tools | allowlisted Lunit MCP tools + `finalize_retrieval` |

Dashboard의 advanced Model 예제는 `function.strict=true`를 사용한다. `retrieve_relevant_content`와 `finalize_retrieval`의 exact release schema는 각각 [20](20_GENERATION_PROMPT.md)·[30](30_RETRIEVAL_PROMPT.md), MCP별 raw-schema→strict-wrapper name/schema/projection 계약은 [10](10_MCP_CATALOG.md)을 단일 canonical source로 삼는다. Dashboard가 MCP별 wrapper를 제공한다고 가정하지 않으며, 실제 L2 endpoint에서 호환성을 trial한 manifest만 등록한다. Model이 출력한 tool name·JSON arguments를 wrapper와 raw MCP schema로 다시 검증한 뒤에만 실행한다. Client가 보낸 `system`, `tool`, `tools`, `functions`는 내부 L2 권한으로 전달하지 않는다.

MCP name plane, raw schema→Model wrapper, argument projection은 [10](10_MCP_CATALOG.md)의 canonical manifest 계약을 그대로 사용한다. Active prompt의 model-function 집합과 Retrieval `tools[]`가 exact-equal하지 않으면 해당 request를 실행하지 않는다. Live binding hash가 drift하거나 lossless compile이 불가능한 alias는 first-use discovery에서 독립 격리하며, 선택 route에 검증된 alias가 하나도 없을 때만 그 근거 경로를 실패시킨다. Static readiness는 package·manifest·endpoint 무결성을 뜻하며 live MCP 호환성을 미리 주장하지 않는다. Codex의 `mcp__<local-server>__` 표시 prefix를 manifest 입력이나 L2/MCP 호출 이름으로 사용하지 않는다.

OpenAI-compatible이라는 이유로 Responses API나 모든 Chat Completions parameter·role·stream 조합을 지원한다고 가정하지 않는다. Startup canary와 Dashboard trial에서 실제 지원 범위를 pin하고, 지원하지 않는 parameter는 조용히 무시하지 말고 결정적 4xx로 처리한다. Streaming은 [45](45_RUNTIME_RESILIENCE.md)의 검증 후 buffered replay가 없으면 명시적으로 미지원 처리한다.

API key는 Model·Patient Simulator·MCP에 같은 팀 key를 사용할 수 있지만 환경변수로만 주입한다. Prompt, source, Docker image, test fixture, log에 key를 기록하지 않는다.

## Generation 적용

`safe_completion_final`을 제외한 각 Generation phase request는 다음 순서를 사용한다.

1. [20](20_GENERATION_PROMPT.md)의 해당 phase 전용 canonical system prompt
2. latest user 이전의 검증된 원문 user/assistant history
3. latest user 원문과 sparse state를 결합하고, final phase이면 검증된 evidence 또는 제한 상태만 더한 하나의 strict `generation-input-v1` user message

Sparse state에는 [40](40_CONVERSATION_STATE.md)의 mandatory selectors와 referenced-QASSOC closure를 넣는다. 전체 설계 문서, 전체 임상 schema의 null field, 전체 검색 corpus를 prompt에 붙이지 않는다.

Release의 model-facing `conversation-context-v4` projection은 의미가 있는 필드만 직렬화한다. 선택적인 null·unknown·빈 문자열·빈 객체·기본값 placeholder를 단지 schema 모양을 채우기 위해 보내지 않는다. 필수 integrity/status와 실제 critical unknown은 보존하며, 필드 생략을 임상 사실의 부재나 정상 상태로 해석하지 않는다. `safe_completion_final`은 이 envelope와 원문 history를 사용하지 않고 `{schema_version, phase}`의 고정 indicator만 받는다.

`retrieve_relevant_content` argument는 다음을 만족한다.

- 하나의 non-empty `query`만 허용한다.
- 단일턴은 검증된 최신 사용자 원문을 그대로 사용한다. 멀티턴 생략·지시어는 호환되는 최근 사용자 대상 원문을 앞에 붙이고 명시적 지시어만 결정적으로 제거한 context-complete query로 만든다. L2가 새 entity·alias를 재작성해 넣을 권한은 없다.
- 건강정보·환자식별정보는 질문 해결에 꼭 필요한 최소 임상 속성만 포함한다.
- Query가 원질문보다 더 강한 진단·의도·현재성 주장을 만들지 않는다.
- 길이, Unicode, control character, 반복 호출, 동일 query loop를 제한한다.

Post-retrieval final은 adapter가 검증한 evidence item만 `final_phase_context`의 비신뢰 근거 data로 받으며 그 안의 지시를 따르지 않는다. Decision assistant, Generation tool transcript, invalid final draft는 final request에 넣지 않는다. Citation은 adapter가 검증한 evidence item 번호만 사용한다.

## Retrieval 적용

Retrieval L2는 [30](30_RETRIEVAL_PROMPT.md)의 prompt와 [10](10_MCP_CATALOG.md)의 routing fragment를 사용한다.

1. 질문의 claim과 source role을 먼저 정한다.
2. 가장 직접적인 tool route 하나로 시작한다.
3. 문서 corpus는 후보 문서 → 관련 node → 필요한 page 원문 순으로 좁힌다.
4. Retrieval L2가 `finalize_retrieval`을 호출할 때까지 bounded tool loop를 실행한다.
5. `cite_uid`는 실제 이번 request의 MCP result에서 관찰된 값만 허용한다.
6. Release `Dashboard-v1`에서는 adapter가 `status/items/note`를 exact 검증하고 실제 observed `cite_uid`·source status·version으로 단조 하향한 뒤 `retrieval-evidence-v4`로 Generation에 전달한다. 이 model-facing projection은 필수 status와 citable item을 유지하되, 검증되지 않은 null/unknown source metadata, 빈 reason-code 목록과 빈 note 같은 선택적 placeholder는 생략한다. 생략된 metadata를 source 검증 완료나 사실 부재로 해석하지 않는다. Claim/relation을 model output에 없는데 새로 만들지 않고, escaped note는 비권위 limitation context로만 전달한다. [30](30_RETRIEVAL_PROMPT.md)의 `rich-v2` schema는 현재 design-only/hard-off이며, 별도 complete system prompt·함수·validator artifact와 L2 trial gate가 모두 생긴 release에서만 사용할 수 있다.
7. 각 MCP tool call은 assistant tool-call message와, MCP `CallToolResult`를 검증한 뒤 제출 harness가 만든 matching `role=tool` message를 Retrieval L2 transcript에 append한다. `finalize_retrieval*`은 local termination function이라 MCP endpoint로 전송하지 않으며 첫 valid result에서 loop를 종료한다. 상세 wire invariant는 [45](45_RUNTIME_RESILIENCE.md)를 따른다.
8. 공식 Python default와 Model strict-required 계약의 차이는 endpoint trial로 결정한다. Default omission이 실제 관찰된 release만 `items=[]`, `note=""` 누락 보완 compat flag를 pin하고, zero-MCP `sufficient+[]`는 evidence sufficient가 아닌 `not_needed`로만 정규화한다.

Dashboard 한도를 token·latency 예산에 직접 반영한다.

- `index_get_document_structure`: 시작 node부터 최대 50 node
- `index_get_page_content`: 1-based page, 호출당 최대 20 page
- 문서 corpus: HIRA 249건, guideline 120건(확인 시점 기준)
- Codex MCP 연결 예시 timeout: 60초. 실제 request deadline 안에서 tool별 더 짧은 timeout과 전체 budget을 둔다.

`status=no_evidence` 하나로 “검색 불필요”와 “검색 실패/근거 없음”을 합치지 않는다. Model-facing finalize 결과와 harness execution status를 분리하고, Generation은 failure·partial·not-needed를 서로 다른 불확실성으로 표현한다.

## MCP를 현재 문서에 연결하는 방법

| 질문/claim | 우선 route | 적용 지침 |
| --- | --- | --- |
| 한국 임상 권고·목표 | guideline index tools | [12](12_KOREAN_CLINICAL_GUIDELINE_ROUTING.md), 대상·환경·version·page 원문 확인 |
| 한국 급여·심의·약가 | `hira_updates_search`, HIRA index, HIRA OpenAPI | 행정 사실의 적용일·삭제/개정 상태 보존 |
| 국내 의약품 허가·용법·금기 | MFDS OpenAPI tools | 제품과 성분, 허가 현재성, label version 분리 |
| 공식 label·경고·상호작용 | `adr_retrieve_drug_info` 또는 MFDS | DailyMed는 국내 허가를 대신하지 않음 |
| 연구 질문 | `rag_vector_query(pubmed_abstracts)` | 초록을 원문 전체·진료지침으로 승격하지 않음 |
| FAERS signal | schema 확인 후 read-only `rag_sql_query` | 발생률·인과성 결론 금지 |
| KCD | `kcd_search_codes` → `kcd_get_name` | KCD-8/9 version 고정 |
| 의료법 | `openapi_law_search` → `list_articles` → `get_article` | 본문·시행일 확인, 세부 판단은 `LEGAL_*_PLACEHOLDER` |

SQL 안전은 prompt가 아니라 adapter에서 single read-only `SELECT`, table/column allowlist, 강제 `LIMIT`, statement timeout, row/result-byte 제한으로 집행한다.

## Multi-turn와 Patient Simulator 적용

Patient Simulator는 scorer나 baseline evalset이 아니라 user-turn 생성기다.

- Endpoint: `https://patient.hackathon.lunit.io/v1/chat/completions`
- Model: `patient-simulator-ko`
- 새 대화: `messages: []`
- 후속: 받은 user 질문과 harness assistant 답변을 **표시된 그대로** append한 전체 history 재전송
- 별도 session ID 없음
- 약 3 turn 후 종료 권장
- 404: 빈 messages로 새 대화, 502: 동일 request의 bounded retry

Simulator history와 내부 L2 history를 구분한다. Simulator에 원문을 그대로 보내는 계약은 내부 L2에 untrusted prior assistant text를 권위 있는 사실로 전달하라는 뜻이 아니다. [40](40_CONVERSATION_STATE.md)에서 subject·정정·모순을 검증하고, [45](45_RUNTIME_RESILIENCE.md)의 internal message grammar로 전달한다.

## 격리 평가·제출에 맞춘 packaging

- 적법하게 licensed 외부 guideline/PDF를 사용한다면 build 전에 정적 artifact로 고정하고 source manifest에 license·version·effective date·hash를 기록한다.
- Runtime에는 Dashboard가 제공하는 L2/MCP와 image 안의 정적 asset만 필요해야 한다.
- 외부 URL은 citation metadata일 뿐 runtime fetch 지시가 아니다.
- Startup에서 prompt hash, tool schema fingerprint, local source manifest를 검증한다.
- Network·MCP가 없어도 container는 crash하지 않고 dependency 상태를 보고하되, 최종 의료문을 non-L2 fallback으로 생성하지 않는다.
- 실제 제출·Docker·port·endpoint 계약은 [80](80_EVALUATION_AND_SUBMISSION.md)을 따른다.

## 개발 네트워크 preflight

Dashboard 규칙상 Model·MCP·Patient Simulator 등 Lunit asset은 Lunit network 밖에서 접근할 수 없다. 로컬·원격 개발과 CI는 먼저 DNS/TLS/authenticated canary로 현재 실행 환경이 허용 네트워크인지 확인한다. Off-network 실패를 잘못된 API key, model regression, MCP empty evidence로 기록하지 않고 `lunit_network_unavailable`로 즉시 중단한다. Prompt·schema·unit test는 offline에서 실행할 수 있지만 실제 L2/MCP/Simulator integration·trial은 Lunit network 안의 승인 runner에서만 수행한다. Evaluation 격리 정책과 개발 네트워크 정책은 별개 manifest field로 고정한다.

## 제출 프로그램 접목 계약

이 문서 묶음은 실행 가능한 MCP package가 아니라 제출 프로그램을 만드는 semantic source다. 대회의 intended topology에서 역할은 다음처럼 고정한다.

| 구성요소 | 제출물에서의 역할 | 금지되는 오해 |
| --- | --- | --- |
| `l2-healthcare-instructions` | canonical prompt·routing·schema·검증 계약의 design/build input | MCP 서버, 자동으로 로드되는 model instruction, 통째로 붙이는 system prompt |
| `BraveTylenol` 같은 제출 service | Evaluator용 OpenAI-compatible API와 Model/MCP client orchestration | 제공 MCP tool의 자체 복제 서버 |
| `https://mcp.hackathon.lunit.io/mcp` | Bearer 인증 Streamable HTTP 원격 MCP server | 일반 인터넷 fallback 또는 local finalizer 실행 대상 |
| `finalize_retrieval` | Retrieval L2 request에만 등록하는 harness-local 종료 함수 | MCP endpoint로 보내는 remote tool |
| `retrieve_relevant_content` | Generation `tool_decision`의 유일한 application tool | MCP tool 또는 최종 답변 제출 함수 |

접목 build는 다음을 원자적으로 수행한다.

1. [20](20_GENERATION_PROMPT.md)의 일곱 phase block(tool decision 하나와 no-tool final 여섯 개), [30](30_RETRIEVAL_PROMPT.md)의 Dashboard-v1 Retrieval block, [10](10_MCP_CATALOG.md)의 routing fragment와 exact strict tool schema를 결정적으로 compile한다. 문서 전체나 design-only rich-v2 block을 prompt에 포함하지 않는다.
2. Compiled phase별 artifact와 서로 다른 `prompt_hash`, tool별 schema fingerprint, placeholder 치환 결과, Dashboard 확인 기준일, feature flag를 versioned manifest에 기록하고 submission image에 넣는다. Runtime은 독립적으로 복사한 짧은 prompt 상수를 사용하지 않고 이 artifact를 import하며 startup에서 manifest와 byte hash를 검증한다.
3. [10](10_MCP_CATALOG.md)의 canonical 계약으로 approved alias registry와 lossless projection을 고정한다. First-use live discovery에서 alias별로 compile하고 Missing·duplicate·drifted·lossless 변환 불가 tool만 quarantine한다. Active fragment와 실제 등록한 MCP-backed model function 집합은 request마다 exact-equal이어야 하며, 선택 route에 검증된 tool이 하나도 없을 때만 그 route를 근거 없음으로 종료한다.
4. Normal의 내부 `tool_decision`에만 strict `retrieve_relevant_content` 하나를 등록한다. `direct_final`, `post_retrieval_final`, `mcp_failure_final`, `emergency_final`, `clean_recovery_final`은 각각 frozen inbound와 trusted phase context로 새 transcript를 만들고 application tool을 하나도 등록하지 않는다. `safe_completion_final`도 tool 없이 fixed indicator만으로 독립 transcript를 만든다. `submit_final_answer` 같은 두 번째 application tool도 추가하지 않는다.
5. Retrieval에는 승인된 MCP-backed strict Model function entries와 strict Dashboard-v1 `finalize_retrieval`만 등록한다. Assistant tool call과, MCP result를 검증한 제출 harness가 만든 matching `role=tool` message는 이 별도 Retrieval transcript에만 보존하고 local finalizer는 원격 MCP로 전송하지 않는다.
6. MCP 사용 release의 static readiness는 RAG mode, 공식 endpoint, credential 존재, compiled artifact 무결성을 확인하되 외부 preflight를 하지 않는다. Live alias 호환성은 first-use discovery와 별도 release canary에서 검증한다. 질문에 필요한 route의 alias가 전부 격리되면 direct 성공으로 위장하지 않고 `mcp_failure_final`의 명시적 근거 없음 상태로 단조 하향한다. 의도적으로 direct-only인 비교 variant는 별도 manifest와 Dashboard trial 결과를 가진다.
7. Offline mock test와 별도로 Lunit network 안에서 Model strict-tool canary, MCP `list_tools`/대표 `call_tool`, Retrieval finalization, evidence 반환, fresh phase-specific Generation L2 final text까지 한 E2E trace로 검증한다. Mock 통과를 live MCP 호환성으로 표현하지 않는다.

### 현재 workspace의 접목 상태 — 2026-08-22 감사 snapshot

`BraveTylenol`은 `/mcp`를 제공하는 MCP 서버가 아니라 원격 Lunit MCP를 호출하는 client/orchestrator다. 현재 snapshot은 canonical source에서 컴파일한 runtime artifact를 submission runtime에 연결한 상태다.

- Runtime은 hash-verified bundle의 일곱 Generation phase artifact와 Retrieval artifact를 import하며 startup에서 manifest와 byte hash를 검증한다. 여섯 final artifact에는 local function 2개와 등록된 MCP alias 21개 및 function/tool protocol 표면형이 없어야 한다.
- 제출 runtime의 기본 mode는 hybrid다. 일반·비근거 의존 질문은 direct final fast path로 답하고, source-dependent 질문만 내부 `tool_decision`과 bounded Retrieval로 보낸다. 최종 답변은 어느 분기에서도 phase별 fresh no-tool L2 request가 생성하며 `AGENT_MODE=direct`는 MCP를 끄는 명시적 비교·복구 variant다.
- 승인 alias allowlist, per-tool raw/wrapper schema와 projection, observed-`cite_uid` ledger, `retrieval-evidence-v4` adapter, MCP-aware readiness를 코드와 offline contract test로 검증한다.
- 실제 L2/MCP endpoint 호환성과 품질은 별도 Lunit-network canary·E2E trial로 확인해야 하며, offline mock 통과만으로 live 검증을 주장하지 않는다.

따라서 아래 acceptance checkbox는 실제 code·image·Lunit-network trace가 위 계약을 만족한 commit에서만 체크한다. Runtime 접목, unit test 또는 문서 자체의 Dashboard 정합성만으로 live 항목을 체크하지 않는다.

## 최소 구현 순서

1. OpenAI request validator와 exact history 보존
2. 일곱 Generation phase prompt와 Retrieval prompt 및 서로 배타적인 tool registry
3. Dashboard-v1 `finalize_retrieval(status, items, note)`와 `retrieve_relevant_content` strict schema; rich-v2는 별도 trial flag
4. MCP adapter와 observed-`cite_uid` ledger
5. 임상 normalization·state의 sparse mandatory projection
6. bounded timeout/retry/concurrency/context budget
7. Patient Simulator 3-turn smoke test
8. locked clinical/adversarial suite
9. Dashboard validation trial
10. 마지막 제출 전 clean offline build·startup·holdout leakage audit

## Dashboard 적용 acceptance test

- [ ] Retrieval request에 Generation tool이 없고 Generation request에 MCP tool이 없음
- [ ] Retrieval은 final answer text가 아니라 valid `finalize_retrieval`로 종료
- [ ] Dashboard-v1과 rich-v2 tool 이름·prompt·validator가 같은 request에서 혼합되지 않음
- [ ] Complete rich-v2 prompt artifact/hash가 없는 manifest는 rich flag를 startup에서 거부
- [ ] Retrieval MCP assistant-tool/tool-result call ID가 exact match하고 local finalizer가 MCP로 전송되지 않음
- [ ] Dashboard-v1 default compat on/off가 trial manifest와 일치하고 empty sufficient가 근거 sufficient로 승격되지 않음
- [ ] `tool_decision`의 sole tool call이 strict `retrieve_relevant_content(query)`이고 여섯 final phase의 tool set이 비어 있음
- [ ] Tool-decision transcript가 user-visible text나 final transcript로 유입되지 않음
- [ ] Direct·근거 성공·근거 실패·응급 final이 각각 fresh no-tool transcript이며 최종 text의 model provenance가 L2
- [ ] Invalid final은 이전 draft 없이 fresh `clean_recovery_final`을 정확히 한 번만 실행하고, recovery도 invalid·malformed이거나 timeout이면 deadline이 남을 때 fixed-indicator `safe_completion_final`을 30초·256 tokens로 한 번만 실행하며 이 phase 실패만 sanitized upstream error
- [ ] Self-contained query가 subject·제품·성분·기준일을 잃지 않음
- [ ] Observed `cite_uid`가 아닌 값, duplicate UID, stale request UID를 거부
- [ ] Patient Simulator 첫 질문·전체 history·3-turn·404/502 계약 통과
- [ ] External DNS 차단과 read-only filesystem에서 startup·request 의미론 통과
- [ ] API key·전체 건강 대화가 build artifact/log에 없음
- [ ] Validation 개선을 holdout 성능으로 과장하지 않고 같은 최종 submission으로 두 평가를 수행
