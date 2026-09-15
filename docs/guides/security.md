---
title: "Security"
---

# Security

Memotrix is an embeddable library. Your host application owns authn/authz, multi-tenancy, and network exposure.

## Findings (documentation-level)

| Severity | Topic | Notes |
|---|---|---|
| High | Untrusted file parse | PDF/Office/HTML parsers can be attack surface. Sandbox or scan uploads. |
| High | Prompt injection via retrieved text | Search will return hostile instructions if they were ingested. Frame context in the LLM prompt. |
| Medium | Secrets in env | `DATABASE_URL`, `OPENAI_API_KEY` must not be committed. Library refuses silent defaults. |
| Medium | Postgres | Parameterized queries in the store path; still protect the DSN and DB roles. |
| Medium | No auth layer | Anyone who can call `Memory` in-process can read/write that store. |
| Low | Logging | Retrieval may log query previews (`utils.trace`). Do not log secrets. |
| Informational | CORS / rate limit | Not applicable inside the library. |
| Informational | Dependency risk | Optional extras pull native libs (hnswlib, PyMuPDF, Whisper). Keep them updated. |

Do not treat this page as a penetration-test report. No dedicated security test suite is claimed.

## Recommendations

- Isolate tenants with separate tables, databases, or payload filters **you** enforce in the application.
- Disable image description / external AI if you cannot send file bytes to a provider.
- Prefer `describe_images=False` when processing untrusted images without a vision policy.
