# Feature Specification: Multi-Workspace Server Support

**Feature Branch**: `001-multi-workspace-server`
**Created**: 2025-12-01
**Status**: Draft
**Input**: Multi-workspace/multi-tenant support at the server level for LightRAG Server with instance pooling and header-based workspace routing

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Tenant-Isolated Document Ingestion (Priority: P1)

As a SaaS platform operator, I need each tenant's documents to be stored and indexed completely separately so that one tenant's data never appears in another tenant's queries, ensuring privacy and data isolation for multi-tenant deployments.

**Why this priority**: This is the core value proposition - without workspace isolation, the feature cannot support multi-tenant use cases. A SaaS operator cannot deploy without this guarantee.

**Independent Test**: Can be fully tested by ingesting a document for Tenant A, then querying from Tenant B and verifying the document is not accessible. Delivers the fundamental isolation guarantee.

**Acceptance Scenarios**:

1. **Given** a server with multi-workspace enabled, **When** Tenant A sends a document upload request with workspace header "tenant_a", **Then** the document is stored in Tenant A's isolated workspace only
2. **Given** Tenant A has ingested documents, **When** Tenant B queries the server with workspace header "tenant_b", **Then** Tenant B receives no results from Tenant A's documents
3. **Given** Tenant A has ingested documents, **When** Tenant A queries with workspace header "tenant_a", **Then** Tenant A receives results from their own documents

---

### User Story 2 - Header-Based Workspace Routing (Priority: P1)

As an API client developer, I need to specify which workspace my requests should target by including a header, so that my application can interact with the correct tenant's data without managing multiple server URLs.

**Why this priority**: This is the mechanism that enables isolation - equally critical as US1. Without header routing, clients cannot target specific workspaces.

**Independent Test**: Can be fully tested by sending requests with different workspace headers and verifying each targets the correct workspace.

**Acceptance Scenarios**:

1. **Given** a valid request, **When** the `LIGHTRAG-WORKSPACE` header is set to "workspace_x", **Then** the request operates on workspace "workspace_x"
2. **Given** a valid request without `LIGHTRAG-WORKSPACE` header, **When** the `X-Workspace-ID` header is set to "workspace_y", **Then** the request operates on workspace "workspace_y" (fallback)
3. **Given** a request with both headers set to different values, **When** the server receives the request, **Then** `LIGHTRAG-WORKSPACE` takes precedence

---

### User Story 3 - Backward Compatible Single-Workspace Mode (Priority: P2)

As an existing LightRAG user, I need my current deployment to continue working without changes, so that upgrading to the new version doesn't break my single-tenant setup or require configuration changes.

**Why this priority**: Critical for adoption - existing users must not be disrupted. However, new multi-tenant deployments are the primary goal.

**Independent Test**: Can be fully tested by deploying the new version with existing configuration and verifying all existing functionality works unchanged.

**Acceptance Scenarios**:

1. **Given** an existing deployment using `WORKSPACE` env var, **When** no workspace header is sent in requests, **Then** requests use the configured default workspace
2. **Given** an existing deployment, **When** upgraded to the new version without config changes, **Then** all existing functionality works identically
3. **Given** default workspace is configured, **When** requests arrive without workspace headers, **Then** the server serves requests from the default workspace without errors

---

### User Story 4 - Configurable Missing Header Behavior (Priority: P2)

As an operator of a strict multi-tenant deployment, I need to require workspace headers on all requests, so that I can prevent accidental data leakage from misconfigured clients defaulting to a shared workspace.

**Why this priority**: Important for security-conscious deployments but not required for basic functionality.

**Independent Test**: Can be fully tested by disabling default workspace and verifying requests without headers are rejected.

**Acceptance Scenarios**:

1. **Given** default workspace is disabled in configuration, **When** a request arrives without any workspace header, **Then** the server rejects the request with a clear error message
2. **Given** default workspace is enabled in configuration, **When** a request arrives without any workspace header, **Then** the request proceeds using the default workspace
3. **Given** a rejected request due to missing header, **When** the client receives the error, **Then** the error message clearly indicates a workspace header is required

---

### User Story 5 - Workspace Instance Management (Priority: P3)

As an operator of a high-traffic multi-tenant deployment, I need the server to efficiently manage workspace instances, so that the server can handle many tenants without excessive memory usage or startup delays.

**Why this priority**: Performance optimization - important for scale but basic functionality works without it.

**Independent Test**: Can be tested by monitoring memory usage as workspaces are created and verifying resource limits are respected.

**Acceptance Scenarios**:

1. **Given** a request for a new workspace, **When** the workspace has not been accessed before, **Then** the server initializes it on-demand without blocking other requests
2. **Given** the maximum workspace limit is configured, **When** the limit is reached and a new workspace is requested, **Then** the least recently used workspace is released to make room
3. **Given** multiple concurrent requests for the same new workspace, **When** processed simultaneously, **Then** only one initialization occurs and all requests share the same instance

---

### Edge Cases

- What happens when workspace identifier contains special characters (slashes, unicode, empty string)?
  - System validates identifiers and rejects invalid patterns with clear error messages
- How does the system handle concurrent initialization requests for the same workspace?
  - System ensures only one initialization occurs; concurrent requests wait for completion
- What happens when a workspace initialization fails (storage unavailable)?
  - System returns an error for that request without affecting other workspaces
- How does the system behave when the instance pool is full?
  - System evicts least-recently-used workspace and initializes the new one
- What happens if the default workspace is not configured but required?
  - System returns a 400 error clearly indicating the missing configuration

## Requirements *(mandatory)*

### Functional Requirements

**Workspace Routing:**
- **FR-001**: System MUST read workspace identifier from the `LIGHTRAG-WORKSPACE` request header
- **FR-002**: System MUST fall back to `X-Workspace-ID` header if `LIGHTRAG-WORKSPACE` is not present
- **FR-003**: System MUST support configuring a default workspace for requests without headers
- **FR-004**: System MUST support rejecting requests without workspace headers (configurable)
- **FR-005**: System MUST validate workspace identifiers (alphanumeric, hyphens, underscores, 1-64 characters)

**Instance Management:**
- **FR-006**: System MUST maintain separate isolated workspace instances per workspace identifier
- **FR-007**: System MUST initialize workspace instances on first access (lazy initialization)
- **FR-008**: System MUST support configuring a maximum number of concurrent workspace instances
- **FR-009**: System MUST evict least-recently-used instances when the limit is reached
- **FR-010**: System MUST ensure thread-safe workspace instance access under concurrent requests

**Data Isolation:**
- **FR-011**: System MUST ensure documents ingested in one workspace are not accessible from other workspaces
- **FR-012**: System MUST ensure queries in one workspace only return results from that workspace
- **FR-013**: System MUST ensure graph operations in one workspace do not affect other workspaces

**Backward Compatibility:**
- **FR-014**: System MUST work unchanged for existing deployments without workspace headers
- **FR-015**: System MUST respect existing `WORKSPACE` environment variable as default
- **FR-016**: System MUST not change existing request/response formats

**Security:**
- **FR-017**: System MUST enforce authentication before workspace routing (workspace header does not bypass auth)
- **FR-018**: System MUST log workspace identifiers in access logs for audit purposes
- **FR-019**: System MUST NOT log sensitive configuration values (credentials, API keys)

**Configuration:**
- **FR-020**: System MUST support `LIGHTRAG_DEFAULT_WORKSPACE` environment variable
- **FR-021**: System MUST support `LIGHTRAG_ALLOW_DEFAULT_WORKSPACE` environment variable (true/false)
- **FR-022**: System MUST support `LIGHTRAG_MAX_WORKSPACES_IN_POOL` environment variable (optional)

### Key Entities

- **Workspace**: A logical isolation boundary identified by a unique string. Contains all data (documents, embeddings, graphs) for one tenant. Key attributes: identifier (string), creation time, last access time
- **Workspace Instance**: A running instance serving requests for a specific workspace. Relationship: one-to-one with Workspace when active
- **Instance Pool**: Collection of active workspace instances. Key attributes: maximum size, current size, eviction policy (LRU)

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Existing single-workspace deployments continue working with zero configuration changes after upgrade
- **SC-002**: Data from Workspace A is never returned in queries from Workspace B (100% isolation)
- **SC-003**: First request to a new workspace completes initialization within 5 seconds under normal conditions
- **SC-004**: Workspace switching via header adds less than 10ms overhead per request
- **SC-005**: Server supports at least 50 concurrent workspace instances (configurable)
- **SC-006**: Memory usage per workspace instance remains proportional to single-workspace deployment
- **SC-007**: All multi-workspace functionality is covered by automated tests demonstrating isolation

## Assumptions

- Workspace identifiers are provided by trusted upstream systems (API gateway, SaaS platform) after authentication
- The underlying storage backends (databases, vector stores) support namespace isolation through the existing workspace parameter
- Operators will configure appropriate memory limits based on their workload
- LRU eviction is acceptable for workspace instance management (frequently accessed workspaces stay loaded)
