# 2026-08-22 live MCP paired promotion decision

이 문서는 공식 Lunit Model/MCP endpoint를 사용한 고정 synthetic 의료 질문의 paired
promotion gate를 기록한다. 공식 leaderboard 또는 HealthBench 점수가 아니며, 질문·답변·근거
원문, `cite_uid`, credential, Authorization 값은 저장하지 않았다.

## 유효 결과

| Gate | 비교 범위 | Hybrid | Direct baseline | 지연 | 판정 |
| --- | --- | ---: | ---: | --- | --- |
| C2 run 2 | retrieval-sensitive 6쌍 | 32 | 34 | 평균 41.5초 | NO-GO |
| v4 | 응급·특이 13쌍 | 102 | 101 | MCP 호출 0 | NO-GO |

C2 run 2는 methodology, secret handling, 실행 경로 검사를 모두 통과했다. Hybrid는 direct보다
품질 점수가 낮고 평균 지연은 direct 20.2초의 약 두 배였다. 따라서 MCP를 제출 기본 경로로
승격할 근거가 없다.

v4도 methodology 검사를 통과했다. Routing, PII, tool-protocol containment 신호는 통과했고
총점은 hybrid가 1점 높았지만, assistant history 뒤 topic switch 한 건에서 안전성 회귀가
발생했다. 평균점수만으로 안전성 실패를 덮지 않는 사전 gate에 따라 전체 판정은 NO-GO다.

## 제외 결과

후속 run 3은 methodology 검증에 실패해 점수 근거로 사용하지 않는다. 해당 실행에서 hybrid
retrieval은 6건 중 4건만 HTTP 200이었고 두 건은 502였으며, direct comparator도 당시 잘못
적용된 no-evidence guard 때문에 6건 모두 502였다. 비교 조건이 깨졌으므로 품질·지연 우열을
해석하지 않는다.

## Release 결정

- 이 실험의 promotion 판정은 NO-GO로 유지한다. 다만 이후 명시적인 제출 지시에 따라
  `AGENT_MODE=hybrid`를 runtime 기본값으로 선택한다.
- `direct`는 MCP를 즉시 끄는 비교·복구 경로로 보존한다.
- `MAX_MCP_CALLS` 기본값은 1이다. 2~3 hop source graph는 별도 설정과 후속 검증 없이는
  기본 release로 승격하지 않는다.
- 문항별 문자열이나 예상답을 runtime routing에 넣지 않는다.
- 원격 push 또는 dashboard 제출은 이 문서의 결과만으로 승인하지 않는다.
