# TRAG Security Spec

## Controls

- Do not expose secrets, tokens, `.env`, API keys, passwords, or raw credentials.
- Store provider credentials only as references, hashes, encrypted handles, or external secret IDs.
- Require human approval for destructive rebuild, cleanup, external exports, legal-critical decisions, and sensitive-data exceptions.
- Preserve workspace isolation across Postgres, Neo4j, Qdrant, and LightRAG runtime.

## Local Development

- Local Neo4j without authentication is diagnostic-only.
- Qdrant client/server version mismatches are diagnostic-only.
- Pilot data must be inventoried before deletion.

## Acceptance Criteria

- Default feature flags are safe.
- Audit records capture governed actions.
- LGPD/export controls are designed before external sharing.
