<!--
=== SYNC IMPACT REPORT ===
Version change: N/A (initial) → 1.0.0
Modified principles: N/A (initial creation)
Added sections:
  - 4 core principles (API Backward Compatibility, Workspace/Tenant Isolation,
    Explicit Server Configuration, Multi-Workspace Test Coverage)
  - Additional Constraints (security + performance)
  - Development Workflow
  - Governance
Removed sections: N/A (initial creation)
Templates requiring updates:
  - .specify/templates/plan-template.md: ✅ No changes needed (Constitution Check references dynamic)
  - .specify/templates/spec-template.md: ✅ No changes needed (generic requirements structure)
  - .specify/templates/tasks-template.md: ✅ No changes needed (test-first pattern compatible)
Follow-up TODOs: None
========================
-->

# LightRAG-MT Constitution

## Core Principles

### I. API Backward Compatibility

All changes to the LightRAG public API MUST maintain full backward compatibility with existing client code.

**Non-Negotiable Rules:**
- The public Python API (`LightRAG` class, `QueryParam`, storage interfaces, embedding/LLM function signatures) MUST NOT introduce breaking changes
- Existing method signatures MUST be preserved; new parameters MUST have default values that maintain current behavior
- Deprecations MUST follow a two-release warning cycle before removal
- Any workspace-related parameters added to public methods MUST default to single-workspace behavior when not specified
- REST API endpoints MUST maintain version prefixes (e.g., `/api/v1/`) and existing routes MUST NOT change semantics

**Rationale:** LightRAG has a large user base. Breaking the public API would force costly migrations on downstream projects and erode user trust. Multi-tenancy features must be additive, not disruptive.

### II. Workspace and Tenant Isolation

Workspaces and tenants MUST be fully isolated to prevent data leakage, cross-contamination, and unauthorized access.

**Non-Negotiable Rules:**
- Each workspace MUST have completely separate storage namespaces (KV, vector, graph, doc status)
- Queries from one workspace MUST NEVER return data from another workspace
- Authentication tokens MUST be scoped to specific workspace(s); tokens lacking workspace scope MUST be rejected for workspace-specific operations
- Workspace identifiers MUST be validated and sanitized to prevent injection attacks (path traversal, SQL injection, collection name manipulation)
- Background tasks (indexing, cache cleanup) MUST be workspace-aware and MUST NOT process data across workspace boundaries
- Workspace deletion MUST cascade to all associated data without leaving orphaned records

**Rationale:** Multi-tenant systems handle sensitive data from multiple parties. Any cross-workspace data exposure would be a critical security and privacy breach.

### III. Explicit Server Configuration

Server configuration for multi-workspace operation MUST be explicit, documented, and validated at startup.

**Non-Negotiable Rules:**
- All multi-workspace settings MUST be configurable via environment variables or configuration files (no hidden defaults)
- The server MUST validate workspace configuration at startup and fail fast with clear error messages for invalid configurations
- Default behavior without workspace configuration MUST be single-workspace mode (backward compatible)
- Configuration schema MUST be documented in env.example and referenced in README/quickstart
- Runtime configuration changes (e.g., adding workspaces) MUST be logged and auditable
- Sensitive configuration (credentials, API keys) MUST support secret management patterns (environment variables, secret files)

**Rationale:** Implicit or undocumented configuration leads to deployment errors, security misconfigurations, and debugging nightmares. Operators must clearly understand what they are deploying.

### IV. Multi-Workspace Test Coverage

Every new multi-workspace behavior MUST have comprehensive automated test coverage before merge.

**Non-Negotiable Rules:**
- New workspace isolation logic MUST include tests verifying data cannot cross workspace boundaries
- API changes MUST include contract tests proving backward compatibility
- Configuration validation logic MUST include tests for both valid and invalid configurations
- Tests MUST cover both single-workspace (legacy) and multi-workspace operation modes
- Integration tests MUST verify workspace isolation across all storage backends (Postgres, Neo4j, Redis, MongoDB, etc.)
- Test coverage for new multi-workspace code paths MUST be documented in PR descriptions

**Rationale:** Multi-tenant bugs often manifest as subtle data leaks that are hard to detect in production. Comprehensive testing is the primary defense against shipping isolation failures.

## Additional Constraints

### Security Requirements

- All workspace identifiers MUST be treated as untrusted input and validated
- Cross-workspace operations (admin bulk actions) MUST require elevated permissions and explicit audit logging
- Storage backend credentials MUST NOT be logged or exposed in error messages
- API rate limiting MUST be workspace-aware to prevent noisy-neighbor problems

### Performance Standards

- Multi-workspace operation MUST NOT degrade single-workspace performance by more than 5%
- Workspace resolution (determining which workspace a request belongs to) MUST add less than 1ms latency
- Storage backend queries MUST use workspace-scoped indexes, not post-query filtering

## Development Workflow

### Change Process

1. **Specification**: Multi-workspace changes MUST reference the affected constitutional principle(s) in PR description
2. **Review Gate**: PRs affecting workspace isolation MUST have explicit sign-off on security implications
3. **Test Evidence**: PR description MUST include test coverage summary for new multi-workspace paths
4. **Documentation**: Configuration changes MUST update env.example and relevant documentation before merge

### Quality Gates

- All PRs MUST pass existing test suite (backward compatibility verification)
- New multi-workspace tests MUST be added and passing
- Configuration validation tests MUST cover error paths
- Linting and type checking MUST pass

## Governance

This constitution supersedes all other development practices for the LightRAG-MT project. Amendments require:

1. **Proposal**: Written description of change with rationale
2. **Review**: Discussion period with stakeholders
3. **Approval**: Explicit approval from project maintainers
4. **Migration**: If principles are removed or redefined, a migration plan for existing implementations

All pull requests and code reviews MUST verify compliance with these principles. Complexity exceeding these constraints MUST be explicitly justified in the PR description with reference to the relevant principle(s).

**Version**: 1.0.0 | **Ratified**: 2025-12-01 | **Last Amended**: 2025-12-01