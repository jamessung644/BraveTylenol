# Runtime 신뢰 경계, 복원력과 실패 계약

> 상위 색인: [MODEL_INSTRUCTIONS.md](../../MODEL_INSTRUCTIONS.md)
> 임상 입력: [25_CLINICAL_DATA_AND_INPUT_NORMALIZATION.md](25_CLINICAL_DATA_AND_INPUT_NORMALIZATION.md)
> Harness gate: [70_HARNESS_VALIDATION.md](70_HARNESS_VALIDATION.md)
> Adversarial·chaos test: [75_ADVERSARIAL_AND_CHAOS_TESTING.md](75_ADVERSARIAL_AND_CHAOS_TESTING.md)

이 문서는 정상적인 의료 질문뿐 아니라 malformed request, prompt injection, tool 장애, schema drift, context 초과, 동시 요청, resource exhaustion에서도 안전하고 결정적으로 실패하기 위한 canonical runtime 계약이다.

## 신뢰 경계

| 데이터 | 신뢰 수준 | 처리 |
| --- | --- | --- |
| Versioned canonical system prompt·build-time 법률 정책 | trusted_config | hash·version 검증 후 system instruction으로 사용 |
| Harness clock·고정 endpoint·승인 tool schema·검증된 phase artifact hash | trusted_runtime | 형식·범위·manifest binding 검증 후 phase별 request에만 사용 |
| Versioned deterministic red-flag guard output | trusted_routing_hint | 독립 `emergency_final` no-tool phase 선택에만 사용; 진단·응급도·의료문 생성 금지 |
| 사용자 message·사용자 파생 conversation state | untrusted_data | system prompt에 raw 보간 금지, 최종 내부 user JSON envelope로 결합 |
| MCP·문서·법령·논문 content | untrusted_evidence | instruction이 아닌 근거 데이터, schema·크기·provenance 검증 |
| Client-supplied role·tool·function·parameter | untrusted_control | allowlist 밖의 제어 입력 거부 |
| Model이 생성한 tool argument·cite UID·최종 text | untrusted_model_output | schema·budget·현재 request provenance 검증 |

사용자·근거 문자열을 XML tag, Markdown fence, 정규식 삭제만으로 “안전하게 정제”했다고 간주하지 않는다. 권한 채널 분리와 코드 수준 allowlist가 주 방어선이다.

## System prompt와 data payload 분리

- Canonical system prompt에는 정적 지침, 검증된 ISO date, 제한된 응답 locale code 같은 trusted 값만 넣는다. Locale code를 환자의 현재 물리적 위치로 사용하지 않는다.
- 사용자 원문이나 파생 `CONVERSATION_STATE`를 system string 안에 보간하지 않는다.
- 정규화 상태와 latest user turn은 같은 비신뢰 data 계층의 strict JSON `generation-input-v1` envelope로 결합해 내부 Generation 요청의 마지막 user content로 전달한다.
- 이 JSON의 문자열은 표준 JSON escaping, field별 길이 제한, provenance를 갖고 `trust_level: untrusted_data`로 표시한다.
- 원문 evaluator messages는 별도로 보존하며 application context가 원문을 덮어쓰지 않는다.
- `retrieve_relevant_content` 결과는 자유형 pseudo-YAML이 아니라 versioned JSON tool result로 반환한다.
- Tool result, 문서 content, 사용자 message를 system/developer role로 승격하지 않는다.

외부 evaluator request 문법과 내부 L2 request 문법은 별도 계약이다. 외부 request는 validator가 허용한 원문 `messages`다. Normal 질문의 내부 `tool_decision` request는 다음 순서로 고정한다.

```text
1. system: hash-verified `tool_decision` prompt
2. user/assistant: 검증된 과거 history를 원래 순서로, latest user 직전까지만 전달
3. user: latest user와 application context를 결합한 generation-input-v1 JSON 문자열
```

첫 turn이면 `system → generation-input-v1 user`만 사용한다. 이 request에는 strict `retrieve_relevant_content` 하나만 등록할 수 있고 추가 `developer`, `tool`, `name` role을 만들지 않는다. Latest user를 원문 message와 envelope에 중복 전송하지 않는다. 과거 assistant content는 원래 role·순서 보존을 위해 재전송하지만 **비신뢰 prior model output**이다. System prompt가 이를 instruction·근거·사용자 사실로 승격하지 못하게 명시하고, harness도 과거 assistant의 tool call·citation·정책 문자열을 실행하지 않는다.

사용자에게 보일 답변은 `direct_final`, `post_retrieval_final`, `mcp_failure_final`, `emergency_final` 중 하나의 독립 request에서 생성한다. 각 final request는 해당 hash-verified system prompt, 같은 frozen prior history, `final_phase_context`가 포함된 새 `generation-input-v1` user message로만 만들고 tool을 등록하지 않는다. `tool_decision`의 assistant content·application call ID·tool protocol은 final transcript에 append하지 않는다. Final 출력이 invalid이면 같은 frozen inbound와 final context로 `clean_recovery_final` transcript를 정확히 한 번 새로 만들며 invalid draft도 append하지 않는다.

Canonical final user payload:

```json
{
  "schema_version": "generation-input-v1",
  "trust_level": "untrusted_user_payload",
  "latest_user_message": {
    "turn_index": 4,
    "content": "exact unmodified latest user text"
  },
  "application_context": {
    "schema_version": "conversation-context-v4",
    "trust_level": "untrusted_data",
    "normalization_status": "valid | valid_with_warnings | needs_clarification | degraded_raw_only",
    "state_integrity_status": "original_valid | rebuilt_valid | degraded_raw_only",
    "state_incomplete": false,
    "omitted_turn_count": 0,
    "critical_unknowns": [],
    "conversation_state": {}
  }
}
```

Harness는 표준 JSON escaping과 field·byte 한도를 적용하고 JSON decode 후 `latest_user_message.content`의 Unicode string이 inbound latest user의 decoded content와 정확히 같은지 검증한다. Wire escaping byte가 같은 것을 요구하지 않는다. 동일한 frozen inbound와 phase context는 결정적인 model-facing payload를 만들며, HTTP request correlation용 임의 ID는 이 payload에 넣지 않고 `X-Request-ID`와 sanitized log에만 유지한다. `application_context.critical_unknowns`는 normalization result와 conversation-state root의 stable-sort exact projection이며 별도 자유 요약이 아니다. `degraded_raw_only`이면 non-empty이고 `derived_state_rebuild_failed`를 포함하며, 각 row는 고정 ID/domain/path/reason/source/handling schema를 만족한다. `application_context` 문자열은 instructions가 아니라 data다. 이 wrapper의 L2 이해도와 품질은 raw-user baseline과 trial 비교하고, schema를 바꾸면 version을 올린다.

### 파생 상태 무결성 회복

- External request validator의 4xx와 내부 normalization/state builder 오류를 섞지 않는다.
- Clinical-input 또는 conversation-state cross-field 검증이 실패하면 invalid snapshot을 폐기하고 같은 frozen raw messages·승인 sidecar·고정 parser version으로 결정적 rebuild를 최대 한 번 수행한다. L2나 MCP를 이 rebuild에 사용하지 않는다.
- 재검증 성공 시 `state_integrity_status=rebuilt_valid`로 진행한다. 실패했지만 관련 raw user history가 모두 예산 안에 있으면 derived graph·계산·verified identity를 빼고 `normalization_status=degraded_raw_only`, `state_integrity_status=degraded_raw_only`, explicit critical unknown을 넣는다. L2는 raw history를 안전 재평가하며 absence·수치·제품 identity를 추정하지 않는다.
- 알려진 고위험 turn 또는 mandatory fact를 무손실로 넣지 못하면 L2 미호출 413 `context_overflow`다. Frozen raw digest·message ordering 또는 builder가 손상되어 valid raw-only envelope도 만들 수 없으면 L2 미호출 500 `state_integrity_error`다. 내부 오류를 client 4xx, 빈 정상 state, 일부 graph drop으로 바꾸지 않는다.
- 이 rebuild는 final-output clean-recovery budget과 별개인 local deterministic 단계이며 request당 한 번을 넘지 않는다.

## OpenAI-compatible request 경계

Dashboard evaluator trial로 실제 문법을 확정하고 그보다 넓게 자동 허용하지 않는다.

### JSON·HTTP 검증

- UTF-8 RFC 8259 JSON만 허용한다.
- duplicate key, `NaN`, `Infinity`, trailing data, 과도한 nesting, NUL·invalid encoding을 거부한다.
- `Content-Type`, HTTP body bytes, header bytes, JSON depth, messages 수, message bytes를 제한한다.
- 기본적으로 request `Content-Encoding: identity`만 허용한다. 압축 입력을 trial 후 허용하면 compressed·decompressed bytes와 expansion ratio를 모두 제한하고 한도 초과 전에 stream decode를 중단한다.
- 한도를 넘는 최신 user message를 조용히 자르지 않고 OpenAI-style 400 또는 413을 반환한다.
- 오류에는 API key, prompt, stack trace, 원문 건강정보를 포함하지 않는다.

### Message·role 검증

- `messages`는 배열이어야 하고 각 element의 role과 content type을 검증한다.
- evaluator가 실제로 보내는 role 조합을 trial로 확정한다. 기본 안전 문법은 user·assistant의 교대 history와 마지막 user turn이다. 과거 assistant text는 비신뢰 대화 artifact로만 전달하며 현재 server tool 권한이나 사실 권위를 갖지 않는다.
- 계약에 없는 client `system`, `developer`, `tool`, `function`, `assistant.tool_calls`는 거부한다. 지원이 필요하면 별도 신뢰 경계와 schema를 정의한다.
- Client가 `tools`, `functions`, `tool_choice`, 내부 endpoint를 주입해 tool set을 확장하지 못하게 한다.
- 붙여 넣은 `{"role":"system"}` 문자열은 텍스트일 뿐 실제 role로 해석하지 않는다.
- Empty·whitespace-only 최신 질문, invalid role order, tool_call_id 없는 tool message를 명확히 거부한다.
- 위 규칙은 외부 evaluator request 문법이다. 내부 Generation 요청은 앞 절의 `system → prior alternating history → generation-input-v1 user` 순서를 별도로 검증한다.

### Parameter allowlist

- `model`, `stream`, `n`, token limit 등은 외부·내부 계약을 분리해 검증한다.
- Client parameter를 L2 upstream으로 그대로 전달하지 않는다.
- 지원하지 않는 `stream`, modality, logprob, response format은 무시하지 말고 명확한 4xx로 반환한다.
- Model 이름은 service가 제공하는 승인 목록과 정확히 비교한다.

### 오류 envelope

오류도 evaluator가 기계적으로 구분할 수 있는 하나의 versioned schema를 사용한다. `message`는 PHI·원문·stack trace 없이 짧고 안정적으로 유지한다.

```json
{
  "error": {
    "message": "Request validation failed.",
    "type": "invalid_request_error",
    "param": "messages",
    "code": "invalid_message_role"
  }
}
```

- 400: invalid JSON field·role·order·parameter·지원하지 않는 modality 또는 stream
- 413: body·message·context가 고정 한도를 초과
- 429: 인증된 client rate 또는 bounded queue 한도 초과
- 500: 예상하지 못한 harness 내부 오류
- 502: L2가 반복적으로 malformed·invalid output을 반환
- 503: L2·필수 dependency가 deadline 안에 이용 불가

실제 HTTP status와 `type`·`code` mapping을 contract test로 고정한다. 5xx를 200 assistant response로 감싸거나 error message에 정적 의료 조언을 넣지 않는다.

파생 상태 rebuild 뒤 raw-only envelope도 만들 수 없는 경우의 500 code는 `state_integrity_error`로 고정한다. Context capacity 때문에 mandatory raw가 들어가지 않는 경우는 기존 413 `context_overflow`를 사용한다.

## Unicode·제어문자

- 원문 임상 text는 불변으로 보존하고 검색·비교용 normalized view를 별도로 만든다.
- Role, tool name, schema key, `cite_uid`에는 ASCII exact allowlist를 적용하고 bidi override·zero-width·newline을 허용하지 않는다.
- 사용자 text의 bidi·zero-width를 조용히 삭제해 약명·용량 의미를 바꾸지 않는다. 감지 flag를 남기고 고위험 값이면 확인한다.
- NFC 등 정규화 방식과 version을 고정한다. NFKC로 의료 기호를 무조건 합치지 않는다.
- `µ`, `μ`, `u`, full-width 숫자, 조합형 한글, 혼합 한국어·영어를 임상 단위·제품명 parser에서 시험한다.

## Context budget과 truncation

입력 예산은 `model_context - reserved_output - safety_margin`으로 계산한다.

보존 우선순위:

1. Canonical system prompt와 latest user request
2. 현재 red flag, 약물·알레르기, 활성 고위험 노출의 intent·agent·constituent/medication-component·route/event/application/interval 참조와 양·농도·단위·시각·동시노출, 특별 집단, 모순
3. 사용자 정정과 최근 관련 turn
4. 선택된 직접 근거의 조건·예외
5. 오래된 assistant 설명과 중복 내용

- 원문을 중간 byte, JSON token, 숫자·단위, citation item 중간에서 자르지 않는다.
- Compaction은 provenance와 superseded 상태를 보존한다.
- 내용을 버렸으면 `omitted_turn_count`, `state_incomplete`, `evidence_truncated`를 표시한다.
- 근거 일부가 잘렸으면 `evidence_status`를 그대로 `sufficient`로 유지하지 않고 coverage를 재평가한다.
- 결정적 compaction 뒤에도 latest user와 mandatory clinical context(버전 고정 selector가 뽑은 현재·material 약물과 모든 active component, confirmed/uncertain actionable allergy, 현재 red flag·critical value·임신/산후/소아·최근 고위험 약물 변경, 활성 과량복용·독성노출의 intent·이미 발생한 노출/손상·primary/co-agent·모든 알려진 constituent/medication component·route/event/application/interval 참조와 양·조성/노출 농도·단위·시각, 이 자료가 참조하는 `quantitative-association-v1`의 value semantics·center/bounds·unit·target/value ambiguity와 모든 target candidate·revision/fingerprint transitive closure, event/symptom/observation/regimen/medication-usage/exposure-concentration 및 selected vulnerable diagnosis clinical-status time ambiguity의 raw candidate·normalized point/interval·precision·anchor closure, 그 relative timestamp가 의존하는 append-stable message-time anchor, 핵심 모순)가 함께 들어가지 않으면 L2를 호출하지 않고 413 `context_overflow`로 종료한다. Critical context를 제거한 채 L2가 제한만 설명하도록 맡기지 않는다.
- `state_incomplete=true`는 낮은 우선순위 history가 생략됐다는 뜻으로만 허용한다. Mandatory clinical context 손실을 표시하는 대체 수단이 아니다.

## Tool 실행 경계

### 공통

- Runtime에 승인된 exact tool 이름·argument schema만 노출한다.
- 모든 argument는 extra field 금지, 문자열 길이·숫자 범위·enum을 코드에서 검증한다.
- User가 endpoint, corpus identifier, citation URL을 임의로 바꾸지 못하게 한다.
- Tool result의 content type, schema, row·page·item·byte 수를 제한한다.
- 새 tool이나 drift된 parameter를 자동 활성화하지 않는다.
- 동일 argument의 반복 호출을 감지하고 예산을 소진시키는 loop를 중단한다.
- Generation의 `retrieve_relevant_content`는 [20](20_GENERATION_PROMPT.md)의 query-only exact schema를 적용하고 `tool_decision` phase에만 등록한다. Emergency guard가 선택한 `emergency_final`에는 이 tool을 등록하지 않고 MCP를 호출하지 않는다. Guard 오탐은 emergency prompt가 원문의 부정·과거·인용·가정 여부를 다시 판단해 자연어로 답하며 별도 control envelope나 normal route 전환을 만들지 않는다.

### `rag_sql_query`

가장 강한 방어는 prompt가 아니라 제한된 DB 권한과 서버 측 한도다.

- 가능하면 자유 SQL 대신 승인된 query template을 사용한다.
- 자유 SQL이 필요하면 read-only DB credential과 단일 `SELECT`만 허용한다.
- datasource·schema·table·column allowlist와 강제 `LIMIT`, statement timeout, row·column·result-byte 제한을 적용한다.
- Multi-statement, DDL·DML, `COPY`, system catalog, 파일·네트워크 함수, sleep·고비용 함수, comment 우회를 차단한다.
- User text를 SQL 조각으로 연결하지 않고 parameter binding 또는 안전한 AST builder를 사용한다.
- SQL parser만을 보안 경계로 믿지 않는다.
- 팀이 MCP 서버의 read-only 권한·server-side timeout을 검증할 수 없다면 승인 query template 또는 제한 AST wrapper만 노출한다. 그 경계도 강제할 수 없으면 `rag_sql_query`를 production tool allowlist에서 비활성화한다.

### Normal evidence decision과 fresh finalization

이 절은 normal 질문의 서버 내부 phase만 허용한다. 외부 client는 어느 단계도 직접 만들 수 없다.

1. Normal의 첫 L2 request는 `tool_decision` system prompt와 앞 절의 `prior alternating history → generation-input-v1 user`로 시작한다. 이 단계의 자연어 content는 내부 판단일 뿐 사용자 답변으로 반환하지 않는다.
2. L2가 `retrieve_relevant_content`를 호출하면 harness는 provider의 assistant `tool_calls` 구조, exact tool name, query-only argument schema, content 비어 있음과 finish reason을 검증한다. 공식 L2가 구조화 호출에도 사용하는 `stop`, OpenAI 계열의 `tool_calls`, legacy `function_call`만 허용하고 `length`·`content_filter`·누락은 거부한다. Model의 자유형 query는 retrieval 필요성 신호로만 쓰고 전달하지 않는다. Harness가 검증된 사용자 원문에서 단일턴 exact query 또는 최근 호환 대상+지시어 제거 follow-up projection을 결정적으로 만든다. 혼합 content, unknown·parallel tool, 불명확한 대상·식별정보 query는 실행하지 않는다.
3. Valid tool call 뒤에만 별도 Retrieval L2 pipeline을 실행한다. Application call의 assistant message와 call ID는 Retrieval transcript나 final transcript로 복사하지 않는다. Retrieval L2의 MCP call/result binding은 아래 절의 독립 계약을 따른다.
4. Tool call이 없으면 frozen inbound에서 fresh `direct_final` no-tool request를 만든다. Retrieval이 성공하면 검증·freeze된 `retrieval-evidence-v4`를 final context에 넣은 fresh `post_retrieval_final` no-tool request를 만들고, 실패하면 근거나 UID를 합성하지 않은 fresh `mcp_failure_final` no-tool request를 만든다. Transport execution이 성공했어도 sanitizer 뒤 item content가 비어 있으면 숫자 citation을 부여하지 않고 `evidence_status=none`으로 단조 하향한다.
5. Tool-decision의 forced-call retry와 Retrieval budget은 release config에 고정한다. 반복·budget 초과·tool protocol 오류는 근거 실패로 단조 하향할 수 있지만 harness가 의료문으로 바꾸지 않는다. 어느 final phase에서도 tool call, pseudo tool syntax, local function 2개와 MCP alias 21개를 합한 등록 function name의 노출은 invalid model output이다.
6. 최종 assistant text만 client에 반환한다. 허용 citation은 현재 final context의 frozen evidence에만 결합하며 첫 response byte 전에 finish reason·공백·tool protocol·citation을 모두 검증한다. `mcp_failure_final` 또는 인용 가능한 item이 없는 `post_retrieval_final`에서 공식·현행·제품 라벨상 임상 수치, 조문·시행일, 공식 코드·금기·급여 조건, 검사·모니터링·추적 일정의 단정적 출력을 감지하면 답변을 고쳐 쓰지 않고 invalid로 판정한다. Invalid이면 아래 clean recovery를 한 번만 실행한다.

합법적인 `tool_decision` call 자체는 `tool-only` fault가 아니다. 다만 그 raw assistant message나 tool JSON은 절대 user-visible text가 아니며, no-tool final phase의 application tool call은 즉시 invalid output이다.

### Retrieval L2 MCP round-trip

이 절은 위 normal evidence decision 뒤 제출 harness가 소유하는 별도 내부 transcript다. Generation의 application call ID/message를 Retrieval transcript에 재사용하지 않고 외부 client가 어느 message도 공급할 수 없다.

1. Harness는 [30](30_RETRIEVAL_PROMPT.md)의 pinned Dashboard-v1 Retrieval system prompt, 검증된 독립 query 한 개, startup discovery로 확정한 allowlisted MCP tool entries, local `finalize_retrieval` 하나만 넣어 첫 Retrieval L2 request를 만든다. 현재 rich-v2는 complete 별도 prompt artifact가 없어 hard-off다. 향후 rich-v2를 release하려면 v1을 부분 치환하지 않은 별도 prompt hash+tool+validator manifest가 모두 있어야 하며 그 request에는 `finalize_retrieval_rich_v2`만 등록한다. 어느 mode에서도 두 finalizer를 동시에 등록하지 않는다.
2. Assistant response가 tool call이면 provider의 원 assistant message를 byte-stable audit transcript에 보존하고, 허용 finish reason(`stop`·`tool_calls`·`function_call`), ASCII request-unique call ID, empty user-visible content, exact runtime tool name, strict argument schema를 검증한다. `length`·`content_filter`·누락 finish reason, mixed text+tool, parallel/multiple calls, unknown name, duplicate/stale call ID, 다음 message와 맞지 않는 call ID는 schema error다.
3. 호출 이름이 allowlisted `model_function_name`이면 budget·deadline·read-only policy와 strict wrapper argument를 검사한다. [10](10_MCP_CATALOG.md)의 승인 mapping을 반드시 resolve해 exact `transport_tool_name`과 raw argument를 구한 뒤 MCP Streamable HTTP `tools/call`로 실행한다. 두 이름의 문자열이 같아도 mapping 검증을 생략하지 않으며 Codex prefix는 절대 전송하지 않는다. Harness는 MCP `CallToolResult`의 크기·schema·provenance를 검증하고 여기서 관찰한 `cite_uid`를 request-local ledger에 기록한 뒤, 같은 call ID의 `role=tool` message 하나를 직접 append해 같은 Retrieval system/query와 누적 trajectory로 L2를 재개한다.
4. 호출 이름이 현재 local finalizer이면 MCP endpoint로 보내지 않는다. Harness가 strict Dashboard-v1 또는 rich-v2 schema, status/item cardinality, 실제 observed-UID subset, score 범위, 중복·budget을 검증하고 첫 유효 결과를 freeze해 Retrieval episode를 즉시 종료한다. Finalizer 뒤 자연어 답변, 다른 tool call, second finalizer는 실행하지 않는다.
5. MCP timeout·일시 5xx는 전체 deadline과 read-only retry policy 안에서만 재시도한다. Schema/권한/인증 오류, finalizer 미호출 자연어 종료, budget 초과는 결정적 retrieval failure로 바꾸며 harness가 의료 근거나 UID를 합성하지 않는다. 부분적으로 freeze된 valid items가 있으면 `partial`, 없으면 `none`의 execution status 계약을 Generation에 전달한다.
6. Observed UID ledger, tool result digest, call IDs, per-tool latency와 retry는 audit sidecar에 남기되 selected evidence만 Generation envelope에 넣는다. Transcript append 순서는 항상 `assistant(tool_call) → matching tool(result)`이며 MCP result를 user/assistant role로 바꾸거나 local finalizer를 remote MCP result처럼 echo하지 않는다.

### Tool deadline·retry

- 전체 request deadline 아래에 Generation, Retrieval, 개별 MCP deadline을 계층적으로 배분한다.
- Retry는 read-only·idempotent 호출의 timeout, 408, 429, 일시적 5xx에만 제한한다.
- 400, 401, 403, schema invalid는 재시도하지 않는다.
- `Retry-After`, exponential backoff와 jitter를 적용하되 전체 deadline을 넘지 않는다.
- Client disconnect 또는 request 취소 시 in-flight model·MCP task를 취소한다.
- 반복 인증 실패·schema 오류는 circuit breaker로 격리하고 secret 없는 오류만 기록한다.

### Emergency final

- Deterministic guard가 현재 응급 가능성을 표시하면 frozen inbound에서 독립 `emergency_final` L2 request를 만든다. Tool은 등록하지 않고 MCP call은 0이다.
- Emergency prompt는 현재 위험과 부정·과거 종료·인용·가상 상황을 원문에서 독립적으로 구분한다. Guard 오탐도 내부 control JSON이나 normal route 전환 없이 사용자에게 보일 자연어로 직접 답한다.
- 현재 위험이면 현지 응급번호(대한민국 위치 확인 시 119), 즉각 행동, 추가 위험을 피한 안전한 위치, 응급상담원·구급대원 지시를 앞세운다. 통제되지 않는 외부 출혈은 지속적인 직접 압박을 중심으로 하며 검증되지 않은 지혈대·사지 올리기·자가 약 세부를 만들지 않는다.
- `evidence_status=not_requested`이므로 최신·공식 출처나 URL, 학회·저널, 법령을 조회했다고 단정하거나 새 경구약·구체 용량을 시작하도록 지시한 출력은 invalid다. 이미 처방된 rescue plan 또는 현장 dispatcher의 명시적 지시를 따르는 표현만 보수적으로 예외로 한다.
- 첫 response byte 전에 finish reason·공백·tool protocol을 검증한다. Valid text만 반환하고 invalid이면 아래 공통 clean recovery를 정확히 한 번 실행한다.
- Harness는 응급 답변을 이어 붙이거나 보완하지 않고, final 또는 recovery L2가 생성한 한 개의 valid text만 반환한다.

### Clean final recovery

- 이 recovery는 `direct_final`, `post_retrieval_final`, `mcp_failure_final`, `emergency_final`의 결정적 invalid output 또는 최초 final L2 timeout에 공통으로 최대 한 번 적용한다. Recovery는 30초·1,536-token 상한이며 client 내부 retry를 사용하지 않는다.
- Empty content, 허용되지 않은 finish reason, tool call·pseudo tool syntax·등록된 function name 노출, 존재하지 않는 citation, no-evidence phase의 단정적 권위 주장 등 첫 response byte 전에 판정 가능한 위반만 대상으로 한다.
- Harness는 invalid draft와 tool-decision transcript를 폐기하고 Retrieval을 다시 실행하지 않는다. 같은 frozen evaluator history, latest user, application context와 trusted final-phase context로 fresh `clean_recovery_final` request를 만들며 tool을 등록하지 않는다.
- Recovery transcript에는 이전 assistant draft, application/MCP tool call·result, validator feedback용 자유형 의료문을 append하지 않는다. Frozen evidence가 있으면 final context의 검증된 숫자 citation만 다시 사용할 수 있다.
- Recovery도 finish reason·공백·tool protocol·잘못된 citation ID·내부 식별자 노출을 위반하면 502, recovery L2도 deadline 안에 응답하지 않으면 504다. 유효한 숫자 citation을 두 번 연속 생략한 것만 남은 경우에는 Python이 인용을 합성하지 않고 두 번째 L2 원문을 반환하되 safe telemetry에 omission을 기록한다. Client에 첫 response byte를 보낸 뒤에는 recovery하지 않는다.

## Citation 무결성

- 선택한 `cite_uid`는 현재 request에서 실제로 반환된 item 집합의 부분집합이어야 한다.
- UID를 opaque exact string으로 비교하고 모델이 만든 UID, 이전 request UID, Unicode-normalized UID를 허용하지 않는다.
- 같은 UID가 다른 content hash를 가리키면 item을 격리하고 sufficient로 사용하지 않는다.
- 선택 시 content hash와 payload를 freeze해 재조회에 의한 TOCTOU를 피한다.
- Title·publisher·URL의 control character와 newline을 제한하고 표시 URL은 검증된 `http`·`https`만 허용한다. Runtime이 URL을 외부 fetch하지 않는다.
- 존재하지 않는 citation 번호, claim과 불일치한 citation은 harness가 문장을 고치지 않고 fresh clean recovery를 최대 한 번 수행한다.

## 동시성·상태·cache

- 가능하면 evaluator의 전체 history만 사용하는 stateless request 구조를 유지한다.
- 모든 conversation state, citation map, Retrieval tool transcript는 request-scoped immutable snapshot이다.
- Process-global mutable clinical state를 금지한다.
- 같은 conversation의 동시 update가 필요하면 version/CAS 또는 좁은 lock을 사용한다.
- 동일 tenant에서 같은 idempotency key와 같은 canonical body인 경우만 결과 또는 in-flight work를 재사용한다. 같은 key와 다른 body는 409로 거부한다.
- Canonical digest가 필요하면 tenant/request scope의 secret-keyed HMAC을 사용하고 raw 의료 text나 unsalted digest를 log·global key로 남기지 않는다.
- `finalize_retrieval`의 첫 유효 결과를 freeze하고 이후 중복 호출을 실행하지 않는다.
- 개인 맥락과 민감한 건강정보가 제거됐다고 결정적으로 확인된 retrieval query만 model·prompt·corpus·schema version을 포함해 cross-user cache할 수 있다. 안전한 분류가 어렵다면 retrieval cache도 request-local로 제한한다. 개인화된 최종 답변은 cross-user cache하지 않는다.

## 최종 L2 출력과 failure 의미론

Harness는 의료 문장을 새로 쓰거나 L2 text를 post-edit하지 않는다.

| 실패 | 허용 동작 |
| --- | --- |
| Malformed client request | 모델 호출 없이 OpenAI-style 4xx |
| 일부 retrieval source 실패 | 다른 승인 source 또는 `partial`과 semantic·execution status |
| 전체 retrieval 실패 | `evidence_status="unavailable"`과 sanitized reason code를 fresh `mcp_failure_final` context에 전달 |
| Initial final phase의 timeout·empty·bad finish reason·tool protocol 노출 | frozen inbound와 같은 final context로 fresh clean recovery 1회 |
| Emergency final의 application tool call·pseudo tool syntax·invalid output | 같은 no-tool clean recovery 1회; 반복 시 OpenAI-style 502 |
| 잘못된 citation·필수 안전구조 누락 | 원문 post-edit 없이 fresh clean recovery 1회 |
| Recovery도 invalid | 비-L2 의료 답변을 만들지 않고 sanitized OpenAI-style 502 |
| L2 timeout·인증 실패 | 비-L2 의료 답변을 만들지 않고 OpenAI-style 504·503 |
| Harness 내부 예외 | PHI 없는 error ID를 가진 OpenAI-style 500 |

HTTP error object는 assistant 의료 응답이 아니므로 “최종 출력 L2” 규칙을 위반하지 않는다. Product UI가 정적 장애 안내를 표시할 수 있지만 `/v1/chat/completions`의 assistant content로 위장하지 않는다.

## Streaming

- 기본 권장은 upstream L2 출력을 제한된 buffer에 완성한 뒤 empty·truncation·citation·tool-loop를 검증하는 방식이다. SSE를 쓰면 모든 `delta.content`를 순서대로 연결한 Unicode text가 검증된 L2 content와 정확히 같아야 한다. Wire JSON escaping과 chunk 경계는 같을 필요가 없다.
- 검증 전 의료 문장을 client에 relay하지 않는다.
- Streaming을 지원하면 SSE framing, `[DONE]`, finish reason, disconnect cancellation, 중간 오류를 시험한다.
- 안전한 buffering·검증을 구현하지 않았다면 `stream=true`를 명시적으로 미지원 처리하고 evaluator 호환성을 trial로 확인한다.
- Harness가 streaming 도중 안전 문구를 삽입하거나 L2 문장을 수정하지 않는다.

## Resource exhaustion과 격리 환경

- Per-request tool budget 외에 process 전체 Model·MCP concurrency semaphore와 bounded queue를 둔다.
- 포화 시 무제한 대기 대신 429 또는 503을 반환한다.
- Evidence item·page·SQL row·총 bytes, output bytes, decompressed bytes, log bytes를 제한한다. `per_response_buffer × max_concurrency`가 process memory budget 안에 들도록 gate한다.
- Cache는 bounded LRU·TTL을 적용한다. Read-only container에서는 bounded structured log를 stdout·stderr로 보내고 로컬 rotation file에 의존하지 않는다. 외부 수집기의 보존·회전 정책을 별도로 둔다.
- Abandoned task, SSE connection, file descriptor를 정리한다.
- Liveness와 readiness를 분리한다. 필수 local corpus·prompt·pinned schema hash가 없거나 다르면 readiness를 실패시킨다. 환경 credential이 없어도 evaluator가 형식상 유효한 `Authorization: Bearer ...`를 `/readyz`에 공급하면 bearer-only 실행의 static readiness는 200이며, 환경·요청 credential이 모두 없으면 503이다. Readiness는 외부 dependency를 preflight하지 않으므로 일시적 MCP 장애는 startup 자체를 막지 않고 해당 retrieval 요청에서 구조화 실패로 처리한다.
- Read-only filesystem, 누락 asset, 잘못된 timezone·clock, DNS 실패, 외부 네트워크 차단에서 시험한다.
- Startup 중 package·model·corpus를 다운로드하지 않는다.

## 관찰 가능성

Trace에는 request ID, model·phase prompt hash·taxonomy·state·tool schema·corpus version, deadline, retry, selected citation hash, compaction, validator·fallback·clean-recovery event를 남긴다. API key, full prompt, invalid draft, raw tool protocol, 불필요한 원문 건강정보와 stack trace는 남기지 않는다.
