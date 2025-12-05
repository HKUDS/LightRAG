## Your Role
Your an expert in System Engineering and Cloud Architecture working for a large SaaS company that provides multi-tenant applications to various clients. You have been tasked with designing a solution to address the challenges of data isolation, security, and scalability in a multi-tenant environment.

## The state of the art solution

This the current best practice approach to designing a multi-tenant architecture that ensures data isolation, security, and scalability for FastAPI applications using PostgreSQL, Neo4j, MongoDB, and Redis as data stores.# Multi-Tenant REST API with FastAPI + PostgreSQL + Neo4j + MongoDB + Redis: The 2025 Battle-Tested Approach

Here is the **battle-tested, production-grade approach** used by top multi-tenant SaaS companies in 2025 (e.g., Vercel, Render, Supabase, PostHog, Clerk, Auth0-scale systems) when building a multi-tenant REST API with **FastAPI + PostgreSQL + Neo4j + MongoDB + Redis**.

### Core Principle: "Shared Everything + Strict Tenant Scoping"
The winning strategy is **single database/cluster per data store**, with **row-level / document-level tenant isolation** enforced automatically at the framework level — never trust application code alone.

### 1. Tenant Identification (The One True Way)
Use **subdomains only** (`app.tenant-slug.yourdomain.com`)
Combined with **JWT `tenant_id` claim** (validated on every request)

```python
# middleware/tenant.py
from fastapi import Request, HTTPException, Depends
from sqlalchemy import text
from redis.asyncio import Redis
import uuid

async def get_tenant_context(request: Request, redis: Redis = Depends(get_redis)):
    host = request.headers.get("host", "")
    subdomain = host.split(".")[0] if "." in host else None  # or use X-Tenant-ID fallback

    if not subdomain:
        raise HTTPException(400, "Tenant not found")

    # Resolve tenant_id from Redis (cached) or DB
    cache_key = f"tenant:slug:{subdomain}"
    tenant_id = await redis.get(cache_key)

    if not tenant_id:
        async with db_session() as session:
            result = await session.execute(
                text("SELECT id FROM tenants WHERE slug = :slug AND active = true"),
                {"slug": subdomain}
            )
            row = result.fetchone()
            if not row:
                raise HTTPException(404, "Tenant not found")
            tenant_id = str(row[0])
            await redis.setex(cache_key, 3600, tenant_id)  # cache 1h

    request.state.tenant_id = uuid.UUID(tenant_id)
    request.state.tenant_slug = subdomain
    return request.state
```

Use this dependency **globally**:

```python
app = FastAPI(dependencies=[Depends(get_tenant_context)])
```

### 2. PostgreSQL – Row-Level Security (RLS) + Tenant UUID PK
**This is the gold standard in 2025 SaaS**

```sql
-- Every table
CREATE TABLE projects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL DEFAULT current_setting('app.tenant_id')::UUID,
    name TEXT,
    -- all other fields
);

-- Enable RLS
ALTER TABLE projects ENABLE ROW LEVEL SECURITY;

-- Policy: tenants can only see their rows
CREATE POLICY tenant_isolation ON projects
    USING (tenant_id = current_setting('app.tenant_id')::UUID);
```

Set tenant context **per connection** using a Pydantic-aware SQLAlchemy engine:

```python
# db/postgres.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy import event
import uuid

engine = create_async_engine(DATABASE_URL)

@event.listens_for(engine.sync_engine, "connect")
def set_tenant_id(dbapi_connection, connection_record):
    tenant_id = request_state_tenant_id()  # thread-local from FastAPI state
    cursor = dbapi_connection.cursor()
    cursor.execute(f"SET app.tenant_id = '{tenant_id}'")

async def get_db() -> AsyncSession:
    async with AsyncSession(engine) as session:
        yield session
```

Now **even raw SQL leaks are impossible** — PostgreSQL blocks cross-tenant data.

### 3. MongoDB – Automatic Tenant Filtering with Beanie ODM (or Motor)
Use **Beanie** (Motor + Pydantic) with a base document:

```python
from beanie import Document, PydanticObjectId
from uuid import UUID

class TenantDocument(Document):
    tenant_id: UUID

    class Settings:
        name = "collection_name"
        use_state_management = True

    @classmethod
    def get_motor_collection(cls):
        # Auto-inject tenant filter
        collection = super().get_motor_collection()
        tenant_id = get_current_tenant_id()  # from FastAPI state
        return collection.with_options(
            # This doesn't exist natively → use a wrapper instead
        )
```

Better: **Create a tenant-scoped repository pattern**

```python
class MongoTenantRepo:
    def __init__(self, collection):
        self.collection = collection
        self.tenant_id = get_current_tenant_id()

    async def find(self, filter: dict | None = None, **kwargs):
        filter = filter or {}
        filter["tenant_id"] = self.tenant_id
        return await self.collection.find(filter, **kwargs).to_list()
```

Never call `collection.find()` directly — always go through tenant repo.

### 4. Neo4j – Tenant Prefix + APOC + Parameterized Queries
Neo4j has no RLS → enforce with **prefix labels** or **tenant_id property + strict query wrapper**

Best 2025 approach:

```python
class Neo4jTenantSession:
    def __init__(self, driver):
        self.driver = driver
        self.tenant_id = get_current_tenant_id()

    async def run(self, cypher: str, **params):
        # Force tenant scoping
        if "WHERE" not in cypher.upper():
            cypher = cypher.rstrip() + f" WHERE n.tenant_id = $tenant_id"
        params["tenant_id"] = str(self.tenant_id)
        async with self.driver.session() as session:
            return await session.run(cypher, params)
```

Or use **label prefixing** for extreme isolation:

```cypher
CREATE (n:Tenant_{tenant_id}:User {id: $id})
```

### 5. Redis – Key Prefixing + Lua Scripts for Atomicity
Always prefix keys:

```python
def redis_key(*parts):
    tenant_id = get_current_tenant_id()
    return f"tenant:{tenant_id}:" + ":".join(str(p) for p in parts)

# Usage
await redis.set(redis_key("session", user_id), data, ex=3600)
await redis.get(redis_key("rate_limit", ip))
```

Use **Redis modules** like RediSearch or RedisJSON with tenant prefix indexes.

### 6. FastAPI Dependency Injection – The Magic Glue

```python
# dependencies.py
def get_tenant_db() -> AsyncSession:
    # SQLAlchemy session with tenant context already set
    ...

def get_mongo_tenant_repo():
    return MongoTenantRepo(get_db_collection())

def get_neo4j_tenant():
    return Neo4jTenantSession(neo4j_driver)

def get_redis_tenant():
    tenant_id = get_current_tenant_id()
    # Optionally use Redis with namespace
    return redis  # keys are prefixed in app code
```

Now every route is automatically tenant-safe:

```python
@router.get("/projects")
async def list_projects(
    db: AsyncSession = Depends(get_tenant_db),  # RLS enforced
    mongo: MongoTenantRepo = Depends(get_mongo_tenant_repo),
    neo: Neo4jTenantSession = Depends(get_neo4j_tenant),
):
    projects = await db.execute(select(Project))  # auto-filtered by RLS
    ...
```

### 7. Additional Winning Practices (2025 SaaS Standard)

| Feature                    | Implementation                              |
|----------------------------|---------------------------------------------|
| Tenant Creation            | Async task + background provisioning        |
| Schema Migrations          | Alembic + `SEARCH_PATH` per tenant (or shared with RLS) |
| Background Jobs           | Celery/RQ with tenant context propagation  |
| Analytics                  | Separate ClickHouse per tenant or tenant_id column |
| Rate Limiting              | Redis Sliding Window per tenant             |
| Feature Flags              | LaunchDarkly / Unleash with tenant context  |
| Audit Logs                 | MongoDB (append-only) + tenant_id           |
| Testing                    | Pytest + tenant fixtures                    |

### Summary: The 2025 Winning Stack

- **FastAPI** with global tenant dependency
- **PostgreSQL + RLS** (hard isolation)
- **MongoDB** with tenant repository wrapper
- **Neo4j** with tenant session wrapper or prefixed labels
- **Redis** with strict key prefixing
- **Subdomain → tenant_id resolution** cached in Redis
- **Zero trust**: never allow raw queries without tenant context

This exact pattern powers many $100M+ SaaS companies today. It scales to 100k+ tenants with near-zero cross-tenant risk.

## Your Task

Make a full audit of the implementation of multi-tenancy in my LightRAG project. Identify any gaps or weaknesses in the current approach compared to the battle-tested approach outlined above. Provide specific recommendations for improvements to ensure robust data isolation, security, and scalability across all data stores used (PostgreSQL, Neo4j, MongoDB, Redis).

And provide a clear, concise, actionable plan to address these gaps, including code snippets where applicable and link to code that needs to be modified.

The plan must delivered in a markdown format with proper sections and subsections in multiple documents in docs/action_plan

YOU MUST navigate the codebase to identify the relevant files and code snippets that need to be changed. And conduct a thorough analysis of the current multi-tenant implementation.
