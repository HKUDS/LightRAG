# Security And Governance

## Implemented Baseline

- RBAC roles: `admin`, `manager`, `agent`, `viewer`, `service`.
- Tenant and workspace checks on resource scopes.
- Optional document ACL subject checks.
- PII detection and masking for email, phone-like values, and CPF-like values.
- Prompt-injection pattern detection.
- Destructive skill guardrails requiring human approval.
- Structured audit event contract.
- Model policy separation between visible models and permitted models.

## Production Gaps To Close

- Replace in-memory audit/metrics sinks with append-only durable storage.
- Bind FastAPI auth tokens to tenant/workspace roles.
- Add request rate limiting and per-tenant model cost caps at middleware level.
- Add document metadata and ACL enforcement inside enterprise query wrappers before
  calling LightRAG.
- Add human approval persistence for destructive actions.
- Add dashboards for latency, fallback rate, provider failures, citation coverage,
  cost by tenant, and retrieval quality regression.
