# L2 authentication local performance gate

- Date: 2026-08-21 (Asia/Seoul)
- Tested commit: `d89bbb98828001c14555bf582cb333651afa385d`
- Image: `bravetylenol:l2-auth-local-eval`
- Runtime: Docker 29.5.2, non-root user `65532:65532`
- Evaluator shape: 16 concurrent OpenAI-compatible chat completion requests
- Request authorization: non-Lunit evaluator placeholder
- L2 credential source: runtime-only `LUNIT_FM_API_KEY`
- Token budget: evaluator `max_tokens=6144`, bounded upstream to 4096

## Result

| Metric | Result |
| --- | ---: |
| Requests | 16 |
| HTTP 200 | 16 |
| Real L2 responses | 16 |
| Static fallbacks | 0 |
| Empty responses | 0 |
| Request errors | 0 |
| Wall time | 30.320 s |
| Minimum latency | 23.013 s |
| Median latency | 25.965 s |
| Maximum latency | 30.317 s |
| Content length range | 1,314–1,858 characters |
| Container OOM killed | false |
| Container restarts | 0 |

## Gate

PASS. The run met the local gate of at least 15 real L2 responses, at most one
fallback, no HTTP or empty-response errors, and maximum latency below 35 seconds.

This is an integration and latency gate, not an official HealthBench score. In
this first run, the valid Lunit credential was supplied at container runtime.

## Embedded-key evaluator simulation

After the organizers explicitly permitted embedding the team credential, the
image was rebuilt and started without an env file and without
`LUNIT_FM_API_KEY`. Every request carried only a non-Lunit evaluator placeholder
Bearer token.

| Metric | Result |
| --- | ---: |
| Requests | 16 |
| HTTP 200 | 16 |
| Real L2 responses | 15 |
| Static fallbacks | 1 |
| Empty responses | 0 |
| Request errors | 0 |
| Wall time | 31.168 s |
| Minimum latency | 21.473 s |
| Median latency | 24.307 s |
| Maximum latency | 31.164 s |
| `LUNIT_FM_API_KEY` present in container env | false |
| Container OOM killed | false |
| Container restarts | 0 |

PASS. This confirms the container can reach L2 when the dashboard does not
inject an API-key environment variable or forward a usable Lunit Bearer.
