# Runtime artifact build inputs

`release_config_v1.json`, `legal_policy_ko_v1.json`,
`natural_language_policy_ko_v1.json`은 canonical
`l2-healthcare-instructions` 문서를 submission용 prompt bundle로 컴파일할 때 사용하는
검증 가능한 build-time 입력이다.

- 원문 Markdown 전체는 runtime image에 넣지 않는다.
- 법률 placeholder는 보수적인 일반 정보 경계와 현재 Lunit 법령 MCP 우선 원칙으로
  모두 치환한다.
- 자연어 policy는 오타·단어 나열·도치·축약을 유효한 입력으로 다루되 원문을
  보존하고, 임상 의미를 자동 교정하지 않는 Generation/ Retrieval 별도 계약으로
  컴파일한다.
- API key, 사용자 대화 또는 평가 데이터는 이 디렉터리에 넣지 않는다.
- `scripts/compile_runtime_artifacts.py --check`로 committed artifact와 원문·설정의
  hash 일치를 확인한다.
