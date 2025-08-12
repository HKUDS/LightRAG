"""
LightRAG Authentication Module.

Enhanced authentication system with password security, rate limiting,
security headers, and comprehensive audit logging.
"""

from .password_manager import PasswordManager, PasswordPolicy, PasswordStrength
from .handler import get_auth_handler

__all__ = ["PasswordManager", "PasswordPolicy", "PasswordStrength", "auth_handler"]


# For backward compatibility
def __getattr__(name):
    if name == "auth_handler":
        return get_auth_handler()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
