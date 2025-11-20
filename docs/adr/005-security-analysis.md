# ADR 005: Security Analysis and Mitigation Strategies

## Status: Proposed

## Overview
This document identifies security considerations, potential vulnerabilities, and mitigation strategies for the multi-tenant architecture.

## Security Principles

### Zero Trust Model
Every request is treated as potentially untrusted:
- All tenant/KB context must be explicitly verified
- No implicit assumptions about user access
- Cross-tenant data access denied by default

### Defense in Depth
Multiple layers of security:
1. Authentication (identity verification)
2. Authorization (permission checking)
3. Data isolation (storage layer filtering)
4. Audit logging (forensic capability)
5. Rate limiting (abuse prevention)

### Complete Mediation
All data access controlled through API layer, never direct storage access.

## Threat Model

### Attack Vectors & Mitigations

#### 1. Unauthorized Cross-Tenant Access

**Threat**: Attacker gains access to another tenant's data
```
Attacker (Tenant A) → Exploit → Access Tenant B data
```

**Likelihood**: HIGH (if not mitigated)
**Impact**: CRITICAL (data breach)

**Mitigation Strategies**:

```python
# 1. Strict tenant validation in dependency injection
async def get_tenant_context(
    tenant_id: str = Path(...),
    kb_id: str = Path(...),
    authorization: str = Header(...),
    token_service = Depends(get_token_service)
) -> TenantContext:
    # Decode and validate token
    token_data = token_service.validate_token(authorization)
    
    # CRITICAL: Verify tenant in token matches path parameter
    if token_data["tenant_id"] != tenant_id:
        logger.warning(
            f"Tenant mismatch: token claims {token_data['tenant_id']}, "
            f"but path requests {tenant_id}",
            extra={"user_id": token_data["sub"], "request_id": request_id}
        )
        raise HTTPException(status_code=403, detail="Tenant mismatch")
    
    # Verify KB accessibility
    if kb_id not in token_data["knowledge_base_ids"] and "*" not in token_data["knowledge_base_ids"]:
        raise HTTPException(status_code=403, detail="KB not accessible")
    
    return TenantContext(tenant_id=tenant_id, kb_id=kb_id, ...)

# 2. Storage layer filtering (defense in depth)
async def query_with_tenant_filter(
    sql: str,
    tenant_id: str,
    kb_id: str,
    params: List[Any]
):
    # Always add tenant/kb filter to WHERE clause
    if "WHERE" in sql:
        sql += " AND tenant_id = ? AND kb_id = ?"
    else:
        sql += " WHERE tenant_id = ? AND kb_id = ?"
    
    params.extend([tenant_id, kb_id])
    return await execute(sql, params)

# 3. Composite key validation
def validate_composite_key(entity_id: str, expected_tenant: str, expected_kb: str):
    parts = entity_id.split(":")
    if len(parts) != 3 or parts[0] != expected_tenant or parts[1] != expected_kb:
        raise ValueError(f"Invalid entity_id: {entity_id}")
```

#### 2. Authentication Bypass via Token Manipulation

**Threat**: Attacker forges or modifies JWT token to gain unauthorized access
```
Valid Token → Modify claims → Invalid signature but accepted
```

**Likelihood**: MEDIUM (if not mitigated)
**Impact**: CRITICAL

**Mitigation Strategies**:

```python
# 1. Strong signature verification
def validate_token(token: str) -> TokenPayload:
    try:
        # Use strong algorithm (HS256 minimum, RS256 preferred)
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,  # Keep secret secure
            algorithms=["HS256"],  # Only allow expected algorithms
            options={"verify_signature": True}
        )
        
        # Verify required claims
        required_claims = ["sub", "tenant_id", "exp", "iat"]
        for claim in required_claims:
            if claim not in payload:
                raise jwt.InvalidTokenError(f"Missing claim: {claim}")
        
        # Check expiration
        if payload["exp"] < time.time():
            raise jwt.ExpiredSignatureError("Token expired")
        
        # Check issued-at time (prevent tokens from future)
        if payload["iat"] > time.time() + 60:  # 60 second clock skew tolerance
            raise jwt.InvalidTokenError("Token issued in future")
        
        return TokenPayload(**payload)
    
    except jwt.DecodeError as e:
        logger.warning(f"Invalid token signature: {e}")
        raise HTTPException(status_code=401, detail="Invalid token")
```

#### 3. Parameter Injection / Path Traversal

**Threat**: Attacker passes malicious tenant_id to access unintended data
```
GET /api/v1/tenants/../../admin/data
POST /api/v1/tenants/"; DROP TABLE tenants; --
```

**Likelihood**: MEDIUM
**Impact**: HIGH

**Mitigation Strategies**:

```python
# 1. Strict input validation
from pydantic import constr, validator

class TenantPathParams(BaseModel):
    tenant_id: constr(regex="^[a-f0-9-]{36}$")  # UUID format only
    kb_id: constr(regex="^[a-f0-9-]{36}$")      # UUID format only

@router.get("/api/v1/tenants/{tenant_id}")
async def get_tenant(params: TenantPathParams = Depends()):
    # tenant_id is guaranteed to be valid UUID format
    pass

# 2. Parameterized queries (prevent SQL injection)
# VULNERABLE:
query = f"SELECT * FROM tenants WHERE tenant_id = '{tenant_id}'"

# SAFE:
query = "SELECT * FROM tenants WHERE tenant_id = ?"
result = await db.execute(query, [tenant_id])

# 3. API rate limiting per tenant
class RateLimitMiddleware:
    async def __call__(self, request: Request, call_next):
        tenant_id = request.path_params.get("tenant_id")
        rate_limit_key = f"tenant:{tenant_id}:rateimit"
        
        if await redis.incr(rate_limit_key) > RATE_LIMIT:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        redis.expire(rate_limit_key, 60)
        return await call_next(request)
```

#### 4. Information Disclosure via Error Messages

**Threat**: Detailed error messages leak information about system structure
```
Error: "User john@acme.com does not have access to tenant-id-xyz"
```

**Likelihood**: HIGH
**Impact**: MEDIUM (reconnaissance for further attacks)

**Mitigation Strategies**:

```python
# 1. Generic error messages
# VULNERABLE:
if tenant not found:
    return {"error": f"Tenant '{tenant_id}' not found in system"}

# SAFE:
if tenant not found or user cannot access tenant:
    return {
        "status": "error",
        "code": "ACCESS_DENIED",
        "message": "Access denied"
    }

# 2. Detailed logging (not exposed to client)
logger.warning(
    f"Unauthorized access attempt",
    extra={
        "user_id": user_id,
        "requested_tenant": tenant_id,
        "user_tenants": user_tenants,
        "ip_address": client_ip,
        "request_id": request_id
    }
)

# 3. Generic HTTP status codes
# 401: Authentication failed (invalid token)
# 403: Authorization failed (valid token, but no access)
# 404: Not found (could mean doesn't exist OR no access)
```

#### 5. Denial of Service (DoS) via Resource Exhaustion

**Threat**: Attacker uses API to exhaust resources
```
Attacker sends 100k queries/sec → Exhausts database connections → System unavailable
```

**Likelihood**: MEDIUM
**Impact**: HIGH

**Mitigation Strategies**:

```python
# 1. Per-tenant rate limiting
class TenantRateLimiter:
    async def check_limit(self, tenant_id: str, operation: str):
        key = f"limit:{tenant_id}:{operation}"
        current = await redis.get(key)
        
        limits = {
            "query": 100,      # 100 queries per minute
            "document_add": 10, # 10 documents per hour
            "api_call": 1000,   # 1000 API calls per hour
        }
        
        if int(current or 0) >= limits[operation]:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={"Retry-After": "60"}
            )
        
        pipe = redis.pipeline()
        pipe.incr(key)
        pipe.expire(key, 60)
        await pipe.execute()

# 2. Query complexity limits
async def validate_query_complexity(query_param: QueryParam):
    complexity_score = 0
    
    # Penalize expensive operations
    if query_param.mode == "global":
        complexity_score += 10
    if query_param.top_k > 50:
        complexity_score += query_param.top_k - 50
    
    # Check against quota
    tenant = await get_current_tenant()
    max_complexity = tenant.quota.max_monthly_api_calls
    
    if complexity_score > max_complexity:
        raise HTTPException(status_code=429, detail="Quota exceeded")

# 3. Connection pooling limits
# In storage implementation:
class DatabasePool:
    def __init__(self, max_connections: int = 50):
        self.pool = create_pool(max_size=max_connections)
    
    async def execute(self, query: str, params: List):
        async with self.pool.acquire() as conn:
            return await conn.execute(query, params)
```

#### 6. Data Leakage via Logs

**Threat**: Sensitive data logged and exposed via log access
```
Log: "Processing document for tenant-acme with content: [secret API key]"
```

**Likelihood**: MEDIUM
**Impact**: HIGH

**Mitigation Strategies**:

```python
# 1. Data sanitization in logs
def sanitize_for_logging(data: Any) -> Any:
    """Remove sensitive fields before logging"""
    sensitive_fields = {
        "password", "api_key", "secret", "token", "auth_header",
        "llm_binding_api_key", "embedding_binding_api_key"
    }
    
    if isinstance(data, dict):
        return {
            k: "***REDACTED***" if k in sensitive_fields else v
            for k, v in data.items()
        }
    return data

# 2. Structured logging with field control
logger.warning(
    "Authentication failed",
    extra={
        "user_id": user_id,
        "tenant_id": tenant_id,
        "reason": "Invalid token",
        # Sensitive fields not included
    }
)

# 3. Log retention and access control
# - Keep logs only as long as needed (e.g., 90 days)
# - Encrypt logs at rest
# - Restrict access to logs (RBAC)
# - Audit log access

# 4. PII handling
# Strip/hash PII in logs
def hash_email(email: str) -> str:
    import hashlib
    return hashlib.sha256(email.encode()).hexdigest()[:8]

logger.info(
    "Document added",
    extra={"created_by": hash_email(user_email)}
)
```

#### 7. Replay Attacks

**Threat**: Attacker replays captured API requests
```
Attacker captures: POST /query with response
Attacker replays: Same request multiple times
```

**Likelihood**: LOW-MEDIUM
**Impact**: MEDIUM

**Mitigation Strategies**:

```python
# 1. Nonce/JTI (JWT ID) tracking
class TokenBlacklist:
    def __init__(self):
        self.blacklist = set()
    
    async def revoke_token(self, jti: str):
        self.blacklist.add(jti)
        # Expire after token expiration time
        scheduler.schedule_removal(jti, expiration_time)
    
    async def is_revoked(self, jti: str) -> bool:
        return jti in self.blacklist

# 2. Request idempotency for mutation operations
class IdempotencyMiddleware:
    async def __call__(self, request: Request, call_next):
        if request.method in ["POST", "PUT", "DELETE"]:
            idempotency_key = request.headers.get("Idempotency-Key")
            
            if idempotency_key:
                # Check if already processed
                cached_response = await redis.get(f"idempotency:{idempotency_key}")
                if cached_response:
                    return JSONResponse(cached_response)
                
                # Process request
                response = await call_next(request)
                
                # Cache response
                await redis.setex(
                    f"idempotency:{idempotency_key}",
                    3600,  # 1 hour
                    response.body
                )
                return response
        
        return await call_next(request)

# 3. Timestamp validation
async def validate_request_timestamp(request: Request):
    timestamp = request.headers.get("X-Timestamp")
    if not timestamp:
        raise HTTPException(status_code=400, detail="Missing timestamp")
    
    request_time = datetime.fromisoformat(timestamp)
    current_time = datetime.utcnow()
    
    # Reject requests older than 5 minutes
    if abs((current_time - request_time).total_seconds()) > 300:
        raise HTTPException(status_code=400, detail="Request expired")
```

## Security Configuration

### 1. JWT Configuration

```python
# settings.py
class JWTSettings:
    # Use RS256 (asymmetric) in production instead of HS256
    ALGORITHM = "RS256"  # Production: asymmetric
    
    # Generate key pair:
    # openssl genrsa -out private_key.pem 2048
    # openssl rsa -in private_key.pem -pubout -out public_key.pem
    PRIVATE_KEY = load_private_key()
    PUBLIC_KEY = load_public_key()
    
    # Token expiration times (keep short)
    ACCESS_TOKEN_EXPIRE_MINUTES = 15
    REFRESH_TOKEN_EXPIRE_DAYS = 7
    
    # Token claims validation
    REQUIRED_CLAIMS = ["sub", "tenant_id", "exp", "iat", "jti"]
```

### 2. API Key Security

```python
class APIKeySettings:
    # Use bcrypt for hashing API keys
    HASH_ALGORITHM = "bcrypt"
    
    # Require minimum key length
    MIN_KEY_LENGTH = 32
    
    # Key rotation policy
    KEY_ROTATION_DAYS = 90
    
    # Revocation tracking
    TRACK_REVOKED_KEYS = True
    REVOKED_KEY_RETENTION_DAYS = 30
```

### 3. TLS/HTTPS Configuration

```python
# Enforce HTTPS in production
if settings.environment == "production":
    # Force HTTPS redirect
    app.add_middleware(HTTPSRedirectMiddleware)
    
    # HSTS header (1 year)
    app.add_middleware(
        BaseHTTPMiddleware,
        dispatch=lambda request, call_next: add_hsts_header(call_next(request))
    )
```

### 4. CORS Configuration

```python
# Restrict CORS origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://lightrag.example.com",
        "https://app.example.com"
    ],
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
    allow_credentials=True,
    max_age=3600
)
```

## Audit Logging

### Audit Trail

```python
class AuditLog(BaseModel):
    audit_id: str = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    user_id: str
    tenant_id: str
    kb_id: Optional[str]
    action: str  # create_document, query, delete_entity, etc.
    resource_type: str  # document, entity, relationship, etc.
    resource_id: str
    changes: Optional[Dict[str, Any]]  # What changed
    status: str  # success | failure
    status_code: int  # HTTP status
    ip_address: str
    user_agent: str
    error_message: Optional[str]

# Store audit logs (cannot be modified after creation)
async def log_audit_event(event: AuditLog):
    # Store in append-only log storage
    await audit_storage.insert(event.dict())
    
    # Also emit to audit stream for real-time monitoring
    await audit_event_stream.publish(event)

# Example events to audit
AUDIT_EVENTS = [
    "tenant_created",
    "tenant_modified",
    "kb_created",
    "kb_deleted",
    "document_added",
    "document_deleted",
    "entity_modified",
    "query_executed",
    "api_key_created",
    "api_key_revoked",
    "user_access_denied",
    "quota_exceeded",
]
```

## Vulnerability Scanning

### Regular Security Activities

1. **Dependencies Audit**
   ```bash
   # Monthly
   pip-audit
   safety check
   bandit -r lightrag/
   ```

2. **SAST (Static Application Security Testing)**
   ```bash
   # On every commit
   bandit -r lightrag/
   # Scan for hardcoded secrets
   git-secrets scan
   detect-secrets scan
   ```

3. **DAST (Dynamic Application Security Testing)**
   - Run against staging before deployment
   - Test common OWASP Top 10 vulnerabilities

4. **Penetration Testing**
   - Quarterly by external security firm
   - Focus on multi-tenant isolation

## Security Checklist

- [ ] All API endpoints require authentication
- [ ] All endpoints verify tenant context matches user token
- [ ] All queries include tenant/kb filters at storage layer
- [ ] Error messages don't leak system information
- [ ] Rate limiting enabled per tenant
- [ ] JWT tokens have short expiration (< 1 hour)
- [ ] API keys hashed with bcrypt, not plain text
- [ ] All sensitive data sanitized from logs
- [ ] HTTPS enforced in production
- [ ] CORS properly configured
- [ ] Audit logging for all sensitive operations
- [ ] Secret keys rotated regularly
- [ ] Dependencies audited for vulnerabilities
- [ ] SAST tools run on every commit
- [ ] Regular penetration testing scheduled

## Compliance Considerations

- **GDPR**: Data deletion, right to be forgotten
- **SOC 2 Type II**: Audit trails, access controls
- **ISO 27001**: Information security management
- **HIPAA** (if healthcare): Data encryption, audit trails

---

**Document Version**: 1.0  
**Last Updated**: 2025-11-20  
**Related Files**: 004-api-design.md, 002-implementation-strategy.md
