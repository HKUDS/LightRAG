from contextvars import ContextVar
from typing import Optional

# ContextVar to store the current tenant_id
# This is thread-safe and async-safe
tenant_id_var: ContextVar[Optional[str]] = ContextVar("tenant_id", default=None)

def get_current_tenant_id() -> Optional[str]:
    """Get the current tenant_id from the context."""
    return tenant_id_var.get()

def set_current_tenant_id(tenant_id: str):
    """Set the current tenant_id in the context."""
    return tenant_id_var.set(tenant_id)
