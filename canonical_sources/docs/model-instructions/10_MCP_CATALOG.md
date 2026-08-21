# MCP 도구 카탈로그와 공통 선택 규칙

> 상위 색인: [MODEL_INSTRUCTIONS.md](../../MODEL_INSTRUCTIONS.md)
> 국내 지침 routing: [12_KOREAN_CLINICAL_GUIDELINE_ROUTING.md](12_KOREAN_CLINICAL_GUIDELINE_ROUTING.md)
> Retrieval prompt: [30_RETRIEVAL_PROMPT.md](30_RETRIEVAL_PROMPT.md)
> 근거·불확실성: [35_EVIDENCE_RAG_AND_UNCERTAINTY.md](35_EVIDENCE_RAG_AND_UNCERTAINTY.md)
> Runtime tool 경계: [45_RUNTIME_RESILIENCE.md](45_RUNTIME_RESILIENCE.md)
> 법률 도구 상세 계약: [50_LEGAL_PLACEHOLDERS.md](50_LEGAL_PLACEHOLDERS.md)

## Canonical source

제공 도구의 canonical 목록과 최신 사용 설명은 Dashboard의 [Lunit MCP tools 살펴보기](https://dashboard.hackathon.lunit.io/quick-start/mcp-tools)를 따른다. 아래 표는 2026-08-21 확인 기준의 harness 및 prompt 설계용 요약이다. Dashboard와 이 문서가 다르면 Dashboard를 기준으로 문서와 테스트를 함께 갱신한다.

## 연결 계약

- Endpoint: `https://mcp.hackathon.lunit.io/mcp`
- Transport: Streamable HTTP
- 인증: `Authorization: Bearer <LUNIT_FM_API_KEY>`
- Codex 예시 timeout: `tool_timeout_sec = 60`
- Codex 개발표시 prefix: `mcp__lunit_mcp__`; `lunit_mcp`은 변경 가능한 local server 이름이며 이 prefix를 L2 `tools[].function.name`이나 MCP `tools/call` 이름으로 사용하지 않음

Prompt에는 client가 실제로 등록한 함수 이름과 schema를 제공한다. API key는 환경변수로 주입하고 Git, image, prompt 또는 log에 남기지 않는다.

## 기본 도구 목록

| 영역 | 기본 도구 | 용도 |
| --- | --- | --- |
| 문서 탐색 | `index_list_documents`, `index_get_relevant_nodes`, `index_get_document_structure`, `index_get_page_content`, `index_keyword_search` | HIRA 249개 및 임상 가이드라인 120개 문서 검색, section·ancestor·page range 탐색, 원문 page 확인 |
| RAG source 탐색 | `rag_get_all_data_sources`, `rag_get_data_source_detail` | `pubmed_abstracts`, `hira_faq`, `faers_12q4_25q4`, `dailymed_26_08`, `kcd`의 identifier·schema·table·column 확인 |
| PubMed·HIRA FAQ 검색 | `rag_vector_query` | `pubmed_abstracts`, `hira_faq`의 vector 또는 dense-plus-sparse hybrid 검색 |
| 구조화 데이터 | `rag_sql_query` | `faers_12q4_25q4`, `dailymed_26_08`, KCD의 허용된 PostgreSQL data 조회 |
| 의약품 label | `adr_retrieve_drug_info` | `dailymed_26_08`에서 영문 brand 또는 INN으로 warning, adverse reaction, interaction, source 조회 |
| 식약처 의약품 | `openapi_mfds_check_drug_permission`, `openapi_mfds_find_drugs_by_ingredient`, `openapi_mfds_get_drug_indication` | 국내 허가 상태, 성분 기반 제품, 적응증·용법·주의사항 조회 |
| HIRA | `hira_updates_search`, `openapi_hira_disease_check_code`, `openapi_hira_get_drug_price` | 급여기준·심의사례, 청구 질병코드, 약가 조회 |
| KCD | `kcd_search_codes`, `kcd_get_name` | KCD-8·KCD-9 질병명 기반 후보 검색과 정확한 공식 한글·영문 이름 확인; 기본 version은 KCD-9 |
| 의료법·법률 | `openapi_law_search`, `openapi_law_list_articles`, `openapi_law_get_article` | MST identifier 검색, 제목 filter가 가능한 조문 목록과 stable article key 조회, 조문 전문·시행일·law.go.kr link·citation 조회. 의료법은 이 경로를 우선 사용하고 제공 PDF는 교차검증·fallback에 사용 |

표와 아래 routing fragment의 unprefixed 이름은 canonical **logical tool alias**다. 이 절이 name-plane과 raw-schema 변환 규칙의 단일 canonical source다. 구현은 다음 세 name plane을 섞지 않는다.

1. `transport_tool_name`: MCP `tools/list`가 반환하고 `tools/call`에 보내는 실제 server name
2. `model_function_name`: Retrieval L2의 Chat Completions `tools[].function.name`에 등록하는 승인 wrapper name. 기본값은 Dashboard L2 trajectory와 같은 unprefixed canonical alias이며, startup manifest에서 `transport_tool_name`과 정확히 1:1 bind한다
3. `codex_exposed_name`: Codex 개발 client가 붙이는 `mcp__<local-server-name>__<tool>` 표시명. Local server 이름에 따라 바뀌므로 제출 harness의 prompt·Model request·MCP call에 사용하지 않는다

Prompt builder는 routing fragment의 alias를 AST/token 단위로 `model_function_name`에 compile하고 동일한 이름을 L2 `tools[]`에 등록한다. Runtime은 현재 질문의 source domain과 검증된 alias 집합으로 active prompt variant를 결정적으로 만든다. 모든 alias가 남은 canonical routing 문장은 그대로 유지하고, 일부 alias만 남은 문장은 제거한 뒤 살아 있는 각 alias에 대해 schema에 맞는 원격 단계만 허용하고 누락된 후속 열람을 근거로 가장하지 말라는 generic fragment를 넣는다. Active variant의 alias 집합은 해당 Model request의 MCP-backed `tools[]` 집합과 exact-equal이어야 하며 inactive alias를 설명용으로도 남기지 않는다. 호출 예산 소진 turn에도 같은 registry를 유지하고 `tool_choice=finalize_retrieval`로 추가 원격 호출을 차단한다. L2가 호출하면 adapter가 manifest의 exact mapping으로 `transport_tool_name`을 찾아 MCP에 보낸다. Codex prefix 연결, local server name 추측, user 입력 기반 치환을 금지한다. Missing·duplicate·malformed·drifted binding은 해당 alias만 quarantine하고 secret 없는 trace에 드러내며 미등록 alias를 L2 prompt에 남기지 않는다. 한 alias의 실패로 무관한 source domain까지 중단하지 않고, 질문에 필요한 route의 검증된 alias가 하나도 남지 않을 때만 그 route를 근거 없음으로 종료한다.

MCP raw `inputSchema`는 곧바로 Model strict schema라고 가정하지 않는다. Dashboard는 Model API의 strict function 예를 제공하지만 MCP별 model-ready wrapper를 제공하지 않는다. Tool binding manifest는 `transport_tool_name`, raw schema hash, `model_function_name`, strict wrapper schema hash, argument-projection version을 함께 고정한다. Adapter는 MCP optional/default/null 의미를 보존하는 `required`·nullable/empty 표현과 `additionalProperties=false`를 tool별로 명시하고 실제 L2 endpoint trial에서 검증한다. Model argument를 wrapper schema로 검증한 뒤 결정적으로 raw MCP argument로 투영하고 원 raw schema로 다시 검증한다. Lossless strict 표현이나 역투영이 불가능한 tool, raw schema나 변환 hash가 바뀐 tool은 자동 노출하지 않고 alias 단위로 quarantine한다. Static readiness는 package·manifest·endpoint만 검증하고 live discovery를 대신하지 않으며, first-use 검증 뒤 해당 질문 route에 안전한 도구가 없을 때만 근거 경로를 실패시킨다.

이 문서의 표는 manifest template의 semantic source이지 현재 MCP별 raw/wrapper schema hash를 채운 release artifact가 아니다. Submission image는 승인 alias와 lossless projection 코드를 고정하고, first-use Lunit-network discovery에서 각 live schema를 독립 검증한다. `/readyz`의 static-ready는 live MCP 호환성을 주장하지 않으며 release 전 별도 Model/MCP canary 결과를 남긴다.

## 공통 선택 규칙

- 먼저 질문에 맞는 source를 고르고, source 발견용 호출을 매 질문마다 관성적으로 반복하지 않는다.
- 지침 권고, 현재 행정 사실, 환자교육, 연구 근거의 source role을 분리한다. 환자교육·보도자료를 치료 권고 authority로 승격하지 않는다.
- 문서형 corpus는 검색 결과의 제목·summary에서 멈추지 말고 필요한 원문 page까지 확인한다. 따라서 guideline/index route는 `index_list_documents`, `index_get_relevant_nodes`, `index_get_document_structure`, `index_get_page_content`, `index_keyword_search`의 완결된 subset을 함께 등록한다.
- 진료지침은 보통 후보 1~3개를 먼저 찾고 대상·환경·버전이 맞는 문서의 관련 section/page만 읽는다. 현행판·대체판·철회판과 근거 확실성·권고 강도를 구분한다.
- `index_get_document_structure`는 시작 node부터 최대 50개 node를 반환한다. 큰 문서는 관련 node 또는 subtree로 범위를 좁힌다.
- `index_get_page_content`의 page는 1부터 시작하고 호출당 최대 20 page다. 넓은 범위를 한 번에 요구하지 말고 관련 section의 page range를 사용하며 반환된 flowchart path도 근거 일부로 보존한다.
- `index_keyword_search`는 대소문자를 구분하지 않는 정확 keyword 검색이며 pagination을 고려한다. 의미 검색과 같은 것으로 취급하지 않는다.
- SQL은 제공 schema를 먼저 확인하고 허용된 table·column만 사용한다. Prompt 지침과 별도로 read-only credential, 단일 `SELECT`, allowlist, 강제 `LIMIT`, timeout, row·byte 제한을 코드에서 적용한다.
- vector 검색 결과의 유사도는 진실성 또는 근거 수준을 의미하지 않는다.
- claim 유형에 맞는 source를 고른 뒤 직접성, 최신성, 대상 인구·지역 적용 가능성을 검토한다. 모든 질문에 하나의 고정 근거 서열을 적용하지 않는다.
- tool result에 `cite_uid`가 있을 때만 최종 인용 후보로 선택한다. 다만 `index_list_documents`, `index_get_relevant_nodes`, `index_get_document_structure`, `index_keyword_search`, `rag_get_all_data_sources`, `rag_get_data_source_detail`, `kcd_search_codes`, `openapi_law_search`, `openapi_law_list_articles`는 discovery-only다. 이 중간 결과에 UID가 포함되어도 observed evidence ledger에 넣지 않고 각각 원문 page, 실제 근거 query, 정확한 KCD 명칭, 법령 조문 본문을 조회한 뒤 그 결과의 UID만 선택한다.
- 법령은 검색 결과나 조문 제목에서 멈추지 않고 `openapi_law_get_article`로 실제 본문과 시행일을 확인한다.
- 제공 PDF의 현재·장래 시행 병렬 조문을 사용할 때는 질문 기준일에 적용되는 버전을 명시적으로 선택한다.
- Dashboard의 tool schema가 바뀌면 system prompt, adapter, fixture를 같은 변경으로 취급한다.
- PubMed 초록은 원문 전체를 대체하지 않으며, 한국 제품 허가 질문은 MFDS를 우선하고, FAERS는 발생률·인과성 근거로 사용하지 않는다.

Tool 이름·argument·result schema fingerprint는 startup·CI에서 검증한다. Drift된 tool은 자동 노출하지 않고 quarantine한다. 모든 argument의 extra field·길이·범위와 result의 item·page·row·byte를 제한한다. 상세 실패·retry·SQL 계약은 [45_RUNTIME_RESILIENCE.md](45_RUNTIME_RESILIENCE.md)를 따른다.

## 확인된 source coverage gap

2026-08-21 Dashboard 목록에는 다음 전용 source가 명시되어 있지 않다.

- KDCA 실시간 outbreak·격리·예방접종 공지
- KDCA 교육자료 discovery index, ASP policy landing page와 그 연결 첨부의 live 내용
- 국가·지역별 poison center와 정신건강 crisis resource registry
- 의료기기 safety communication·recall
- 실시간 여행보건 advisory

Guideline corpus에 관련 문서가 실제 포함됐는지 검색으로 확인한다. 없다면 [18](18_PUBLIC_HEALTH_TOXICOLOGY_AND_REMOTE_LIMITS.md)의 `coverage_gap` 규칙을 적용하고 현재 정보를 검색 가능한 것처럼 표현하지 않는다. [국내 공식 source map](12_KOREAN_CLINICAL_GUIDELINE_ROUTING.md)은 개발·검증용이며 격리 runtime의 인터넷 도구 목록이 아니다. 현재 release에는 local KDCA snapshot을 검색하고 observed `cite_uid` ledger에 넣는 승인된 tool/adapter가 없으므로 exact 게시물·첨부도 model-facing evidence가 아니라 개발·검증 fixture로만 둔다. 향후 Lunit 규칙이 local retrieval을 허용하고 별도 adapter를 구현한다면 공공누리 유형·이용 범위·귀속문구, source role, version·status, 정오표 관계, snapshot, content digest, 승인 기록을 component manifest에 넣고 strict tool schema·UID provenance·end-to-end trial을 통과해야 한다. Discovery index나 landing page URL만 등록해 하위 문서를 일괄 허가하지 않는다.

## Retrieval prompt용 routing fragment

아래 `text` code block이 `MCP_ROUTING_PROMPT_FRAGMENT`의 semantic source다. 이름 token은 위 logical alias manifest를 통해 `model_function_name`으로 compile한 뒤 [30_RETRIEVAL_PROMPT.md](30_RETRIEVAL_PROMPT.md)의 같은 이름 placeholder에 삽입한다. Compiled fragment의 tool-name 집합은 Retrieval request `tools[]` 중 **MCP-backed model function entries만** 추출한 집합과 exact-equal해야 한다. Harness-local `finalize_retrieval`은 이 비교에서 제외하되 request 전체 registry에서는 정확히 한 번 별도 검증한다. 둘 중 어느 집합이라도 누락·추가·중복이면 L2를 호출하지 않는다.

```text
- 임상 권고, 진단 기준, 치료 목표: 알려진 대상·진료환경·개입·결과·기준일로 index_list_documents 또는 index_get_relevant_nodes에서 guideline 후보 1~3개를 찾고, 큰 문서는 index_get_document_structure로 관련 subtree를 확인한 뒤 현행판의 관련 section을 index_get_page_content로 읽는다. 권고 대상, 제외 기준, 권고 강도, 근거 확실성을 원문에서 확인하며 환자교육 자료를 권고 authority로 사용하지 않는다. 정확한 용어 검색이 필요하면 index_keyword_search를 보조로 사용한다.
- 항생제 질문: 개인 치료 claim과 기관 ASP 운영 claim을 분리한다. 개인의 항생제 필요성·선택·용량·기간은 해당 감염질환·대상·진료환경의 지침 원문과 국내 허가정보로 routing하고, ASP·KONAS·교육·시범사업·운영기준은 기관 운영자료로만 사용한다. 기관 집계지표, 교육자료 목록, policy landing page를 개인 처방 근거로 선택하지 않는다. Landing page는 관리료 보상 연계의 존재만 지지하며 현재 대상·금액·지급·청구 조건은 최신 시범사업 지침과 필요한 HIRA source로 별도 확인한다.
- 국내 급여기준과 공개 심의사례: hira_updates_search 또는 HIRA corpus의 index 도구를 사용한다.
- 약가 및 급여 등재: openapi_hira_get_drug_price를 사용한다.
- HIRA 청구용 질병 코드 유효성: openapi_hira_disease_check_code를 사용한다.
- 국내 의약품 허가·적응증·용법·금기: 질문에 따라 openapi_mfds_check_drug_permission, openapi_mfds_find_drugs_by_ingredient, openapi_mfds_get_drug_indication을 사용한다.
- 공식 의약품 label, 경고, 상호작용, 이상반응: adr_retrieve_drug_info 또는 적절한 MFDS 도구를 사용한다. 성분명과 제품명을 먼저 구분한다.
- 논문 근거, 효과 비교, 드문 임상 질문: 필요할 때 rag_get_all_data_sources와 rag_get_data_source_detail로 승인 source identifier/schema를 확인하고 rag_vector_query로 pubmed_abstracts를 검색한다. 초록만 확인한 경우 원문 전체를 검토했다고 표현할 근거로 사용하지 않는다.
- 이상사례 신호: rag_get_all_data_sources와 rag_get_data_source_detail로 실제 FAERS source/schema/table/column을 먼저 확인한 뒤 rag_sql_query로 조회할 수 있으나 인과관계나 발생률의 증거로 취급하지 않는다.
- 질병 코드: kcd_search_codes로 후보를 찾고 kcd_get_name으로 정확한 code와 version을 확인한다.
- 의료법, 약사법, 감염병의 예방 및 관리에 관한 법률 등 관련 법령: openapi_law_search로 정확한 법령을 식별하고, openapi_law_list_articles로 관련 조문을 찾은 뒤, openapi_law_get_article로 조문 전문·시행일·출처를 확인한다. 제19조, 제21~23조, 제27조, 제56조 또는 병원 추천·예약이 관련되면 조문 번호와 제목을 query에 포함한다. 감염병 질문도 법률·법령·조문·시행 의도가 명시된 경우에만 법령 route로 보내며 일반 감염 치료 질문과 혼동하지 않는다. 검색 결과만으로 법적 결론을 만들지 않는다. 세부 관할·현행성·fallback 규칙은 [LEGAL_MCP_ROUTING_PLACEHOLDER]에 정의한다.

한국 제도·제품 질문에는 한국어 명칭과 국내 출처를 우선한다. 법률 질문의 지역·관할 처리 규칙은 [LEGAL_JURISDICTION_PLACEHOLDER]에 정의한다. PubMed나 DailyMed 검색에는 필요한 경우 질환명·성분명을 표준 영문 용어로 변환한다. 제품명과 성분명을 혼동하지 않는다.
```
